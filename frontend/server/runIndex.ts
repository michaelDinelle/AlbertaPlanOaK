import { open, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { TAPE_MAGIC } from '@shared/types';
import type { RunDetail, RunIndexEntry, RunMeta, RunSummary } from '@shared/types';
import { statOrNull } from './fs';
import { scanRecordings, toRecordingRefs, type RecordingEntry } from './recordings';

export type BaselineIds = Partial<Record<'ppo' | 'random', string>>;

export interface TapeShape {
  laneCount: number;
  fields: string[];
  headerBytes: number;
  rowBytes: number;
  rows: number;
  maxStep: number;
  rowIntervalMs: number;
  mtimeMs: number;
  size: number;
}

const tapeMemo = new Map<string, { mtimeMs: number; size: number; shape: TapeShape | null }>();

export async function readTapeShape(path: string): Promise<TapeShape | null> {
  const st = await statOrNull(path);
  if (st === null || st.size < 12) return null;
  const hit = tapeMemo.get(path);
  if (hit !== undefined && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit.shape;
  const shape = await readTapeShapeUncached(path, st);
  if (tapeMemo.size >= 4096) tapeMemo.clear();
  tapeMemo.set(path, { mtimeMs: st.mtimeMs, size: st.size, shape });
  return shape;
}

async function readTapeShapeUncached(path: string, st: { size: number; mtimeMs: number }): Promise<TapeShape | null> {
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
    const rowBytes = 12 + fields.length * laneCount * 4;
    const rows = Math.floor((st.size - headerBytes) / rowBytes);
    let maxStep = 0;
    let rowIntervalMs = 0;
    if (rows > 0) {
      const tail = Buffer.allocUnsafe(12);
      const at = headerBytes + (rows - 1) * rowBytes;
      const read = await handle.read(tail, 0, 12, at);
      if (read.bytesRead >= 8) maxStep = tail.readUInt32LE(0) + tail.readUInt32LE(4);
      if (read.bytesRead === 12 && rows > 1) {
        const prev = Buffer.allocUnsafe(12);
        const back = await handle.read(prev, 0, 12, at - rowBytes);
        if (back.bytesRead === 12) {
          rowIntervalMs = (tail.readFloatLE(8) - prev.readFloatLE(8)) * 1000;
        }
      }
    }
    return {
      laneCount, fields, headerBytes, rowBytes, rows, maxStep,
      rowIntervalMs: Number.isFinite(rowIntervalMs) && rowIntervalMs > 0 ? rowIntervalMs : 0,
      mtimeMs: st.mtimeMs, size: st.size,
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
}

export class RunIndex {
  readonly benchRoot: string;

  private runs = new Map<string, ResolvedRun>();
  private order: string[] = [];
  private recordingsCache = new Map<string, { mtimeMs: number; list: RecordingEntry[] }>();
  private baselinePointers: BaselineIds = {};
  private lastRejected = '';
  private lastDangling = '';
  private inFlightScan: Promise<void> | null = null;

  constructor(opts: RunIndexOptions) {
    this.benchRoot = opts.benchRoot;
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
    return this.recordingsCached(id, run.dir);
  }

  private async recordingsCached(id: string, dir: string): Promise<RecordingEntry[]> {
    const dirStat = await statOrNull(dir);
    const mtimeMs = dirStat?.mtimeMs ?? 0;
    const hit = this.recordingsCache.get(id);
    if (hit !== undefined && hit.mtimeMs === mtimeMs) return hit.list;
    const list = await scanRecordings(dir, { root: this.benchRoot });
    this.recordingsCache.set(id, { mtimeMs, list });
    return list;
  }

  private async ensureFresh(): Promise<void> {
    if (this.inFlightScan !== null) return this.inFlightScan;
    this.inFlightScan = this.scan().finally(() => {
      this.inFlightScan = null;
    });
    return this.inFlightScan;
  }

  private pointerMemo = new Map<string, { mtimeMs: number; size: number; id: string | null }>();

  private async readPointer(name: string): Promise<string | null> {
    const path = join(this.benchRoot, '..', name);
    const st = await statOrNull(path);
    if (st === null) return null;
    const hit = this.pointerMemo.get(name);
    if (hit !== undefined && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit.id;
    let id: string | null = null;
    try {
      id = (await Bun.file(path).text()).trim() || null;
    } catch {
    }
    this.pointerMemo.set(name, { mtimeMs: st.mtimeMs, size: st.size, id });
    return id;
  }

  private metaMemo = new Map<string, { mtimeMs: number; size: number; meta: RunMeta | null }>();

  private async readMeta(path: string): Promise<RunMeta | null> {
    const st = await statOrNull(path);
    if (st === null) return null;
    const hit = this.metaMemo.get(path);
    if (hit !== undefined && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit.meta;
    let meta: RunMeta | null = null;
    try {
      meta = readRunMeta(await Bun.file(path).text());
    } catch {
    }
    this.metaMemo.set(path, { mtimeMs: st.mtimeMs, size: st.size, meta });
    return meta;
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
      this.readPointer('keeper'),
      this.readPointer('baseline-ppo'),
      this.readPointer('baseline-random'),
    ]);
    const next = new Map<string, ResolvedRun>();
    const rejected: string[] = [];

    for (const id of dirNames) {
      const dir = join(this.benchRoot, id);
      const metaStat = await statOrNull(join(dir, 'meta.json'));
      if (metaStat === null) continue;
      const meta = await this.readMeta(join(dir, 'meta.json'));
      if (meta === null) {
        rejected.push(id);
        continue;
      }
      const tape = await readTapeShape(join(dir, 'tape.bin')).catch(() => null);
      if (tape === null || tape.rows === 0) {
        rejected.push(id);
        continue;
      }
      const hasRecordings = (await this.recordingsCached(id, dir).catch(() => [])).length > 0;
      const summaryStat = await statOrNull(join(dir, 'summary.json'));
      const complete = tape.maxStep >= meta.steps && meta.steps > 0;
      next.set(id, {
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
          hasSummary: summaryStat !== null,
          hasRecordings,
          policy: meta.policy,
          role: id === keeper ? 'keeper' : null,
        },
      });
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

    const baselines: BaselineIds = {};
    const dangling: string[] = [];
    if (keeper !== null && !next.has(keeper)) dangling.push(`keeper → ${keeper}`);
    for (const [name, id] of [['ppo', baselinePpo], ['random', baselineRandom]] as const) {
      if (id === null) continue;
      if (next.has(id)) baselines[name] = id;
      else dangling.push(`baseline-${name} → ${id}`);
    }
    const danglingKey = dangling.join(',');
    if (dangling.length > 0 && danglingKey !== this.lastDangling) {
      console.warn(
        `[oak-frontend] ${dangling.length} pointer file(s) beside ${this.benchRoot} name run directories`
        + ` that are not in the scanned run list and count as absent (${dangling.join('; ')})`,
      );
    }
    this.lastDangling = danglingKey;
    this.baselinePointers = baselines;
    this.runs = next;
    this.order = [...next.keys()].sort((a, b) => {
      const sa = next.get(a)?.entry.startedAt ?? '';
      const sb = next.get(b)?.entry.startedAt ?? '';
      if (sa !== sb) return sa < sb ? 1 : -1;
      return a < b ? 1 : a > b ? -1 : 0;
    });
  }
}

function readRunSummary(raw: string): RunSummary | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  const o = parsed as Record<string, unknown>;
  if (o['envSlices'] === null || typeof o['envSlices'] !== 'object') return null;
  return parsed as RunSummary;
}

const summaryMemo = new Map<string, { mtimeMs: number; size: number; summary: RunSummary | null }>();

export async function buildRunDetail(
  run: ResolvedRun,
  recs: RecordingEntry[],
  baselines: BaselineIds,
): Promise<RunDetail> {
  const path = join(run.dir, 'summary.json');
  const st = await statOrNull(path);
  let summary: RunSummary | null = null;
  if (st !== null) {
    const hit = summaryMemo.get(run.id);
    if (hit !== undefined && hit.mtimeMs === st.mtimeMs && hit.size === st.size) {
      summary = hit.summary;
    } else {
      try {
        summary = readRunSummary(await Bun.file(path).text());
        if (summary === null) {
          console.warn(`[oak-frontend] ${run.id}: summary.json is not a run summary — ignoring it`);
        }
      } catch {
      }
      summaryMemo.set(run.id, { mtimeMs: st.mtimeMs, size: st.size, summary });
    }
  }
  return {
    id: run.id,
    meta: run.meta,
    summary,
    recordings: toRecordingRefs(recs),
    baselines,
  };
}
