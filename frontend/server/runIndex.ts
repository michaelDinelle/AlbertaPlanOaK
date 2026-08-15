import { open, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { TAPE_MAGIC } from '@shared/types';
import type { RunDetail, RunIndexEntry, RunMeta, RunSummary } from '@shared/types';
import { INDEX_TTL_MS, IN_FLIGHT_MS } from './config';
import { statOrNull } from './fs';
import { readKeeper } from './keeper';
import { scanRecordings, toRecordingRefs, type RecordingEntry } from './recordings';

export interface TapeShape {
  laneCount: number;
  fields: string[];
  headerBytes: number;
  rowBytes: number;
  rows: number;
  maxStep: number;
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
    if (rows > 0) {
      const tail = Buffer.allocUnsafe(8);
      const at = headerBytes + (rows - 1) * rowBytes;
      const read = await handle.read(tail, 0, 8, at);
      if (read.bytesRead === 8) maxStep = tail.readUInt32LE(0) + tail.readUInt32LE(4);
    }
    return { laneCount, fields, headerBytes, rowBytes, rows, maxStep, mtimeMs: st.mtimeMs, size: st.size };
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
  ttlMs?: number;
}

export class RunIndex {
  readonly benchRoot: string;
  private readonly ttlMs: number;

  private runs = new Map<string, ResolvedRun>();
  private order: string[] = [];
  private recordingsCache = new Map<string, { at: number; list: RecordingEntry[] }>();
  private lastScanAt = 0;
  private inFlightScan: Promise<void> | null = null;

  constructor(opts: RunIndexOptions) {
    this.benchRoot = opts.benchRoot;
    this.ttlMs = opts.ttlMs ?? INDEX_TTL_MS;
  }

  async list(): Promise<RunIndexEntry[]> {
    await this.ensureFresh();
    return this.order
      .map((id) => this.runs.get(id)?.entry)
      .filter((e): e is RunIndexEntry => e !== undefined);
  }

  async resolve(id: string): Promise<ResolvedRun | undefined> {
    await this.ensureFresh();
    return this.runs.get(id);
  }

  async recordingsFor(id: string): Promise<RecordingEntry[]> {
    const run = await this.resolve(id);
    if (run === undefined) return [];
    const now = Date.now();
    const hit = this.recordingsCache.get(id);
    if (hit !== undefined && now - hit.at < this.ttlMs) return hit.list;
    const list = await scanRecordings(run.dir, { root: this.benchRoot });
    this.recordingsCache.set(id, { at: Date.now(), list });
    return list;
  }

  private async ensureFresh(): Promise<void> {
    if (this.runs.size === 0) {
      if (this.inFlightScan === null) {
        this.inFlightScan = this.scan().finally(() => {
          this.inFlightScan = null;
        });
      }
      return this.inFlightScan;
    }
    if (Date.now() - this.lastScanAt < this.ttlMs || this.inFlightScan !== null) return;
    this.inFlightScan = this.scan().finally(() => {
      this.inFlightScan = null;
    });
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

    const keeper = await readKeeper(this.benchRoot);
    const now = Date.now();
    const next = new Map<string, ResolvedRun>();

    for (const id of dirNames) {
      const dir = join(this.benchRoot, id);
      const meta = await this.readMeta(join(dir, 'meta.json'));
      if (meta === null) continue;
      const tape = await readTapeShape(join(dir, 'tape.bin')).catch(() => null);
      if (tape === null || tape.rows === 0) continue;
      const hasRecordings = (await envDirCount(dir)) > 0;
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
          inFlight: !complete && now - tape.mtimeMs < IN_FLIGHT_MS,
          hasSummary: summaryStat !== null,
          hasRecordings,
          role: id === keeper ? 'keeper' : null,
        },
      });
    }

    this.runs = next;
    this.order = [...next.keys()].sort((a, b) => {
      const sa = next.get(a)?.entry.startedAt ?? '';
      const sb = next.get(b)?.entry.startedAt ?? '';
      if (sa !== sb) return sa < sb ? 1 : -1;
      return a < b ? 1 : a > b ? -1 : 0;
    });
    this.lastScanAt = Date.now();
  }
}

async function envDirCount(runDir: string): Promise<number> {
  try {
    const ents = await readdir(runDir, { withFileTypes: true });
    return ents.filter((e) => e.isDirectory() && e.name.includes('-')).length;
  } catch {
    return 0;
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
  if (o['overall'] === null || typeof o['overall'] !== 'object') return null;
  if (o['envSlices'] === null || typeof o['envSlices'] !== 'object') return null;
  return parsed as RunSummary;
}

export async function buildRunDetail(run: ResolvedRun, recs: RecordingEntry[]): Promise<RunDetail> {
  let summary: RunSummary | null = null;
  try {
    const text = await Bun.file(join(run.dir, 'summary.json')).text();
    summary = readRunSummary(text);
    if (summary === null) {
      console.warn(`[oak-frontend] ${run.id}: summary.json is not a run summary — ignoring it`);
    }
  } catch {
  }
  return {
    id: run.id,
    meta: run.meta,
    summary,
    recordings: toRecordingRefs(recs),
  };
}
