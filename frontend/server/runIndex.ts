import { open, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { TAPE_MAGIC } from '@shared/types';
import type { RunDetail, RunIndexEntry, RunMeta, RunSummary } from '@shared/types';
import { statOrNull } from './fs';
import { readPoseIndex, toRecordingRefs, type PoseIndex, type RecordingEntry } from './recordings';

export type BaselineIds = Partial<Record<'ppo' | 'random', string>>;

export interface TapeShape {
  laneCount: number;
  fields: string[];
  rows: number;
  maxStep: number;
  rowIntervalMs: number;
  mtimeMs: number;
}

interface FileStamp { size: number; mtimeMs: number }

function statMemo<T>(compute: (path: string, st: FileStamp) => Promise<T>) {
  const memo = new Map<string, { mtimeMs: number; size: number; value: T }>();
  return async (path: string): Promise<T | null> => {
    const st = await statOrNull(path);
    if (st === null) return null;
    const hit = memo.get(path);
    if (hit !== undefined && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit.value;
    const value = await compute(path, st);
    if (memo.size >= 4096) memo.clear();
    memo.set(path, { mtimeMs: st.mtimeMs, size: st.size, value });
    return value;
  };
}

export const readTapeShape = statMemo(readTapeShapeUncached);

async function readTapeShapeUncached(path: string, st: FileStamp): Promise<TapeShape | null> {
  const handle = await open(path, 'r');
  try {
    const head = Buffer.allocUnsafe(Math.min(4096, st.size));
    const got = await handle.read(head, 0, head.byteLength, 0);
    if (got.bytesRead < 12) return null;
    if (head.readUInt32LE(0) !== TAPE_MAGIC) return null;
    const laneCount = head.readUInt32LE(4);
    const jsonLen = head.readUInt32LE(8);
    if (12 + jsonLen > got.bytesRead) return null;
    let fields: string[];
    try {
      const parsed = JSON.parse(head.subarray(12, 12 + jsonLen).toString('utf8')) as { fields?: unknown };
      if (!Array.isArray(parsed.fields)) return null;
      fields = parsed.fields.filter((f): f is string => typeof f === 'string');
    } catch {
      return null;
    }
    const headerBytes = 12 + jsonLen;
    let rows = 0;
    let maxStep = 0;
    let rowIntervalMs = 0;
    let at = headerBytes;
    let lastWall = 0;
    let prevWall = 0;
    const rowHead = Buffer.allocUnsafe(16);
    while (at + 16 <= st.size) {
      const read = await handle.read(rowHead, 0, 16, at);
      if (read.bytesRead < 16) break;
      const compLen = rowHead.readUInt32LE(12);
      if (at + 16 + compLen > st.size) break;
      rows += 1;
      maxStep = rowHead.readUInt32LE(0) + rowHead.readUInt32LE(4);
      prevWall = lastWall;
      lastWall = rowHead.readFloatLE(8);
      at += 16 + compLen;
    }
    if (rows > 1) rowIntervalMs = (lastWall - prevWall) * 1000;
    return {
      laneCount, fields, rows, maxStep,
      rowIntervalMs: Number.isFinite(rowIntervalMs) && rowIntervalMs > 0 ? rowIntervalMs : 0,
      mtimeMs: st.mtimeMs,
    };
  } finally {
    await handle.close().catch(() => undefined);
  }
}

export interface ResolvedRun {
  id: string;
  dir: string;
  meta: RunMeta;
  tape: TapeShape;
  entry: RunIndexEntry;
}

function readRunMeta(raw: string): RunMeta | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  const o = parsed as Record<string, unknown>;
  if (typeof o['batch'] !== 'number' || typeof o['steps'] !== 'number') return null;
  if (!Array.isArray(o['seeds']) || !Array.isArray(o['envSchedule'])) return null;
  if (typeof o['ablationLabel'] !== 'string') return null;
  const policy = o['policy'];
  if (policy !== 'oak' && policy !== 'ppo' && policy !== 'random') return null;
  if (!Array.isArray(o['envCatalog']) || o['envCatalog'].length === 0) return null;
  if (typeof o['flopsPerStep'] !== 'number') return null;
  const envFlops = o['envFlopsPerStep'];
  if (envFlops === null || typeof envFlops !== 'object' || Array.isArray(envFlops)) return null;
  return parsed as RunMeta;
}

function envIdsOf(meta: RunMeta): string[] {
  const seen: string[] = [];
  for (const s of meta.envSchedule) {
    if (typeof s.envId === 'string' && !seen.includes(s.envId)) seen.push(s.envId);
  }
  return seen;
}

export interface RunIndexOptions {
  benchRoot: string;
  baselinesRoot: string;
}

export class RunIndex {
  readonly benchRoot: string;
  readonly baselinesRoot: string;

  private runs = new Map<string, ResolvedRun>();
  private order: string[] = [];
  private recordingsCache = new Map<string, { index: PoseIndex | null }>();
  private baselinePointers: BaselineIds = {};
  private lastRejected = '';
  private lastDangling = '';
  private inFlightScan: Promise<void> | null = null;

  constructor(opts: RunIndexOptions) {
    this.benchRoot = opts.benchRoot;
    this.baselinesRoot = opts.baselinesRoot;
  }

  async list(): Promise<RunIndexEntry[]> {
    await this.ensureFresh();
    return this.order
      .map((id) => this.runs.get(id)?.entry)
      .filter((e): e is RunIndexEntry => e !== undefined);
  }

  baselines(): BaselineIds {
    return this.baselinePointers;
  }

  async resolve(id: string): Promise<ResolvedRun | undefined> {
    await this.ensureFresh();
    return this.runs.get(id);
  }

  async recordingsFor(id: string): Promise<RecordingEntry[]> {
    const run = await this.resolve(id);
    if (run === undefined) return [];
    return (await this.poseIndexCached(id, run.dir))?.entries ?? [];
  }

  async poseIndexFor(id: string): Promise<PoseIndex | null> {
    const run = await this.resolve(id);
    if (run === undefined) return null;
    return this.poseIndexCached(id, run.dir);
  }

  private async poseIndexCached(id: string, dir: string): Promise<PoseIndex | null> {
    const idxStat = await statOrNull(join(dir, 'pose.idx'));
    const hit = this.recordingsCache.get(id);
    if (hit !== undefined && idxStat !== null && hit.index !== null
      && hit.index.idxMtimeMs === idxStat.mtimeMs && hit.index.idxSize === idxStat.size) {
      return hit.index;
    }
    const index = await readPoseIndex(dir);
    this.recordingsCache.set(id, { index });
    return index;
  }

  private async ensureFresh(): Promise<void> {
    if (this.inFlightScan !== null) return this.inFlightScan;
    this.inFlightScan = this.scan().finally(() => {
      this.inFlightScan = null;
    });
    return this.inFlightScan;
  }

  private readPointer = statMemo(async (path: string) => {
    try {
      return (await Bun.file(path).text()).trim() || null;
    } catch {
      return null;
    }
  });

  private readMeta = statMemo(async (path: string) => {
    try {
      return readRunMeta(await Bun.file(path).text());
    } catch {
      return null;
    }
  });

  private async loadRun(id: string, root: string, keeper: string | null): Promise<ResolvedRun | null> {
    const dir = join(root, id);
    const metaStat = await statOrNull(join(dir, 'meta.json'));
    if (metaStat === null) return null;
    const meta = await this.readMeta(join(dir, 'meta.json'));
    if (meta === null) return null;
    const tape = await readTapeShape(join(dir, 'tape.bin')).catch(() => null);
    if (tape === null || tape.rows === 0) return null;
    const hasRecordings = ((await this.poseIndexCached(id, dir).catch(() => null))?.entries.length ?? 0) > 0;
    const summary = await readSummaryFile(join(dir, 'summary.json'), id);
    const complete = tape.maxStep >= meta.steps && meta.steps > 0;
    return {
      id,
      dir,
      meta,
      tape,
      entry: {
        id,
        envIds: envIdsOf(meta),
        batch: meta.batch,
        steps: meta.steps,
        ablationLabel: meta.ablationLabel,
        startedAt: typeof meta.startedAt === 'string' ? meta.startedAt : null,
        complete,
        maxStep: tape.maxStep,
        inFlight: !complete
          && (tape.rowIntervalMs === 0
            || Date.now() - tape.mtimeMs < 2 * tape.rowIntervalMs),
        hasSummary: summary !== null,
        hasRecordings,
        policy: meta.policy,
        role: id === keeper ? 'keeper' : null,
      },
    };
  }

  private async scan(): Promise<void> {
    let dirNames: string[];
    try {
      const ents = await readdir(this.benchRoot, { withFileTypes: true });
      dirNames = ents.filter((e) => e.isDirectory()).map((e) => e.name);
    } catch {
      dirNames = [];
    }

    const [keeper, baselinePpo, baselineRandom] = await Promise.all([
      this.readPointer(join(this.benchRoot, '..', 'keeper')),
      this.readPointer(join(this.baselinesRoot, 'ppo')),
      this.readPointer(join(this.baselinesRoot, 'random')),
    ]);
    const next = new Map<string, ResolvedRun>();
    const rejected: string[] = [];

    for (const id of dirNames) {
      const run = await this.loadRun(id, this.benchRoot, keeper);
      if (run === null) {
        if ((await statOrNull(join(this.benchRoot, id, 'meta.json'))) !== null) rejected.push(id);
        continue;
      }
      next.set(id, run);
    }

    const rejectedKey = rejected.join(',');
    if (rejected.length > 0 && rejectedKey !== this.lastRejected) {
      console.warn(
        `[oak-frontend] ${rejected.length} of ${dirNames.length} directories under ${this.benchRoot}`
        + ` carry a meta.json this build cannot read and are absent from the run list`
        + ` (first: ${rejected.slice(0, 3).join(', ')})`,
      );
    }
    this.lastRejected = rejectedKey;

    this.order = [...next.keys()].sort((a, b) => {
      const sa = next.get(a)?.entry.startedAt ?? '';
      const sb = next.get(b)?.entry.startedAt ?? '';
      if (sa !== sb) return sa < sb ? 1 : -1;
      return a < b ? 1 : a > b ? -1 : 0;
    });

    const baselines: BaselineIds = {};
    const dangling: string[] = [];
    if (keeper !== null && !next.has(keeper)) dangling.push(`keeper → ${keeper}`);
    for (const [name, id] of [['ppo', baselinePpo], ['random', baselineRandom]] as const) {
      if (id === null) continue;
      const run = next.get(id) ?? await this.loadRun(id, this.baselinesRoot, keeper);
      if (run !== null && run !== undefined) {
        next.set(id, run);
        baselines[name] = id;
      } else {
        dangling.push(`${name} → ${id}`);
      }
    }
    const danglingKey = dangling.join(',');
    if (dangling.length > 0 && danglingKey !== this.lastDangling) {
      console.warn(
        `[oak-frontend] ${dangling.length} pointer file(s) name run directories`
        + ` that could not be loaded and count as absent (${dangling.join('; ')})`,
      );
    }
    this.lastDangling = danglingKey;
    this.baselinePointers = baselines;
    this.runs = next;
  }
}

function readRunSummary(raw: string): RunSummary | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  const o = parsed as Record<string, unknown>;
  for (const key of ['envs', 'aggregate', 'compute', 'bootstrap'] as const) {
    const v = o[key];
    if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  }
  if (!Array.isArray(o['curveSteps']) || !Array.isArray(o['plasticitySteps'])) return null;
  if (typeof o['scale'] !== 'string' || typeof o['generatedAt'] !== 'string') return null;
  if (typeof o['horizon'] !== 'number' || typeof o['maxStep'] !== 'number') return null;
  if (typeof o['seedsPerEnv'] !== 'number' || typeof o['complete'] !== 'boolean') return null;
  return parsed as RunSummary;
}

const readSummaryMemo = statMemo(async (path: string) => {
  try {
    return readRunSummary(await Bun.file(path).text());
  } catch {
    return null;
  }
});

async function readSummaryFile(path: string, runId: string): Promise<RunSummary | null> {
  const summary = await readSummaryMemo(path);
  if (summary === null && (await statOrNull(path)) !== null) {
    console.warn(`[oak-frontend] ${runId}: summary.json is not a run summary — ignoring it`);
  }
  return summary;
}

export async function buildRunDetail(
  run: ResolvedRun,
  recs: RecordingEntry[],
  baselines: BaselineIds,
): Promise<RunDetail> {
  return {
    id: run.id,
    meta: run.meta,
    summary: await readSummaryFile(join(run.dir, 'summary.json'), run.id),
    recordings: toRecordingRefs(recs),
    baselines,
  };
}
