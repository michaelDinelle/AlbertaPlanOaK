
import { readFileSync } from 'node:fs';
import { readdir, unlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { readTape, sampleWindows, tapeValue } from './metricsTape';
import { isEnvId } from '@shared/envs';
import type { FleetColumnar, FleetRow } from '@shared/types';
import {
  FLEET_CACHE_MAX_BYTES,
  FLEET_CACHE_MAX_FILES,
  FLEET_COMPUTE_CONCURRENCY,
  FLEET_LRU_MAX,
  MAX_FLEET_FIELDS,
  hash32,
} from './config';
import { parseFieldPath, pluckNumber, statOrNull, streamJsonl } from './jsonl';
import { ensureCacheDir } from './runIndex';

export const DEFAULT_FLEET_FIELDS = ['rewardSum', 'health.rewardRate'] as const;

const MAX_TAPE_WINDOWS = 256;

export interface FleetQuery {
  policy: string;
  fields: string[];
  envId: string | null;
}

export function parseFleetQuery(url: URL): { query: FleetQuery } | { error: string } {
  const policy = url.searchParams.get('policy') ?? 'oak';
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(policy)) return { error: 'bad policy' };

  const rawEnv = url.searchParams.get('envId');
  const envText = rawEnv === null ? '' : rawEnv.trim();
  if (envText.length > 0 && !isEnvId(envText)) return { error: `bad envId: ${envText}` };
  const envId = envText.length > 0 ? envText : null;

  const rawFields = url.searchParams.get('fields');
  const list =
    rawFields === null || rawFields.trim().length === 0
      ? [...DEFAULT_FLEET_FIELDS]
      : rawFields.split(',').map((f) => f.trim()).filter((f) => f.length > 0);
  if (list.length === 0) return { error: 'no fields' };
  if (list.length > MAX_FLEET_FIELDS) return { error: `too many fields (max ${MAX_FLEET_FIELDS})` };

  const seen: string[] = [];
  for (const f of list) {
    if (parseFieldPath(f) === null) return { error: `bad field: ${f}` };
    if (!seen.includes(f)) seen.push(f);
  }
  seen.sort();
  return { query: { policy, fields: seen, envId } };
}

function queryKey(q: FleetQuery): string {
  return `${q.policy}|${q.fields.join(',')}|${q.envId ?? '*'}`;
}


interface LruEntry {
  key: string;
  value: FleetColumnar;
}

const lru: LruEntry[] = [];

function lruGet(key: string): FleetColumnar | undefined {
  const i = lru.findIndex((e) => e.key === key);
  if (i < 0) return undefined;
  const [hit] = lru.splice(i, 1);
  if (hit === undefined) return undefined;
  lru.push(hit);
  return hit.value;
}

function lruPut(key: string, value: FleetColumnar): void {
  const i = lru.findIndex((e) => e.key === key);
  if (i >= 0) lru.splice(i, 1);
  lru.push({ key, value });
  while (lru.length > FLEET_LRU_MAX) lru.shift();
}

export function clearFleetCache(): void {
  lru.length = 0;
  inFlight.clear();
}


const inFlight = new Map<string, Promise<FleetColumnar>>();

let computeActive = 0;
const computeQueue: (() => void)[] = [];

async function withComputeSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (computeActive >= FLEET_COMPUTE_CONCURRENCY || computeQueue.length > 0) {
    await new Promise<void>((res) => computeQueue.push(res));
  }
  computeActive++;
  try {
    return await fn();
  } finally {
    computeActive--;
    const next = computeQueue.shift();
    if (next !== undefined) next();
  }
}


function safeName(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
}

interface DiskCacheFile {
  v: 3;
  key: string;
  policy: string;
  fields: string[];
  envId: string | null;
  data: FleetColumnar;
}

function diskCachePath(cacheDir: string, id: string, q: FleetQuery): string {
  return join(cacheDir, `${safeName(id)}--${hash32(queryKey(q))}.json`);
}

function sameQuery(f: Partial<DiskCacheFile>, q: FleetQuery): boolean {
  if (f.policy !== q.policy) return false;
  if ((f.envId ?? null) !== q.envId) return false;
  if (!Array.isArray(f.fields) || f.fields.length !== q.fields.length) return false;
  for (let i = 0; i < q.fields.length; i++) if (f.fields[i] !== q.fields[i]) return false;
  return true;
}

async function readDiskCache(path: string, key: string, q: FleetQuery): Promise<FleetColumnar | null> {
  try {
    const parsed: unknown = JSON.parse(await Bun.file(path).text());
    if (parsed === null || typeof parsed !== 'object') return null;
    const f = parsed as Partial<DiskCacheFile>;
    if (f.v !== 3 || f.key !== key || f.data === undefined) return null;
    if (!sameQuery(f, q)) return null;
    return f.data;
  } catch {
    return null;
  }
}

async function writeDiskCache(path: string, payload: DiskCacheFile): Promise<void> {
  try {
    await ensureCacheDir(path);
    await writeFile(path, JSON.stringify(payload));
  } catch {
  }
  await evictDiskCache(path);
}

async function evictDiskCache(writtenPath: string): Promise<void> {
  const dir = dirname(writtenPath);
  try {
    const names = (await readdir(dir)).filter((n) => n.endsWith('.json'));
    const entries: { path: string; mtimeMs: number; size: number }[] = [];
    let total = 0;
    for (const name of names) {
      const p = join(dir, name);
      const st = await statOrNull(p);
      if (st === null) continue;
      entries.push({ path: p, mtimeMs: st.mtimeMs, size: st.size });
      total += st.size;
    }
    if (entries.length <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) return;
    entries.sort((a, b) => a.mtimeMs - b.mtimeMs);
    let count = entries.length;
    for (const e of entries) {
      if (count <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) break;
      if (e.path === writtenPath) continue;
      await unlink(e.path).catch(() => undefined);
      count--;
      total -= e.size;
    }
  } catch {
  }
}


export interface DeriveOptions {
  id: string;
  dir: string;
  query: FleetQuery;
  inFlight: boolean;
  cacheDir?: string | null;
}

export async function deriveFleetColumnar(opts: DeriveOptions): Promise<FleetColumnar> {
  const { id, dir, query } = opts;
  const live = opts.inFlight;
  const cacheDir = opts.cacheDir ?? null;
  const fleetPath = join(dir, 'fleet.jsonl');
  const st = await statOrNull(fleetPath);
  if (st === null) return { steps: [], lanes: [], series: emptySeries(query.fields) };

  const fileKey = `${st.mtimeMs}:${st.size}`;
  const memKey = `${dir}|${queryKey(query)}|${fileKey}`;
  const memHit = lruGet(memKey);
  if (memHit !== undefined) return memHit;

  const pending = inFlight.get(memKey);
  if (pending !== undefined) return pending;

  const task = (async (): Promise<FleetColumnar> => {
    const diskPath = cacheDir === null ? null : diskCachePath(cacheDir, id, query);
    if (!live && diskPath !== null) {
      const cached = await readDiskCache(diskPath, fileKey, query);
      if (cached !== null) {
        lruPut(memKey, cached);
        return cached;
      }
    }

    const data = await withComputeSlot(() => computeColumnar(fleetPath, query));
    lruPut(memKey, data);
    if (!live && diskPath !== null) {
      void writeDiskCache(diskPath, {
        v: 3,
        key: fileKey,
        policy: query.policy,
        fields: [...query.fields],
        envId: query.envId,
        data,
      });
    }
    return data;
  })().finally(() => {
    inFlight.delete(memKey);
  });

  inFlight.set(memKey, task);
  return task;
}

function emptySeries(fields: readonly string[]): Record<string, (number | null)[][]> {
  const out: Record<string, (number | null)[][]> = {};
  for (const f of fields) out[f] = [];
  return out;
}

interface LaneKeyed {
  lane: number;
  seed: number;
  policy: string;
  envId: string;
}

async function computeColumnar(fleetPath: string, query: FleetQuery): Promise<FleetColumnar> {
  const paths = query.fields.map((f) => parseFieldPath(f) ?? []);

  const stepValues: number[] = [];
  const stepSlot = new Map<number, number>();
  const laneValues: LaneKeyed[] = [];
  const laneSlot = new Map<number, number>();

  const rowLane: number[] = [];
  const rowStep: number[] = [];
  const rowVals: (number | null)[][] = paths.map(() => []);
  await streamJsonl<FleetRow>(fleetPath, (row) => {
    if (row === null || typeof row !== 'object') return;
    if ((row as { kind?: unknown }).kind === 'hb') return;
    if (typeof row.policy === 'string' && row.policy !== query.policy) return;
    if (typeof row.policy !== 'string' && query.policy !== 'oak') return;
    if (query.envId !== null && row.envId !== query.envId) return;
    const lane = typeof row.lane === 'number' ? row.lane : -1;
    const step = typeof row.step === 'number' ? row.step : -1;
    if (lane < 0 || step < 0) return;

    let si = stepSlot.get(step);
    if (si === undefined) {
      si = stepValues.length;
      stepValues.push(step);
      stepSlot.set(step, si);
    }
    let li = laneSlot.get(lane);
    if (li === undefined) {
      li = laneValues.length;
      laneValues.push({
        lane,
        seed: typeof row.seed === 'number' ? row.seed : lane,
        policy: typeof row.policy === 'string' ? row.policy : query.policy,
        envId: typeof row.envId === 'string' ? row.envId : '',
      });
      laneSlot.set(lane, li);
    }

    rowLane.push(li);
    rowStep.push(si);
    for (let f = 0; f < paths.length; f++) {
      const p = paths[f];
      const dst = rowVals[f];
      if (p === undefined || dst === undefined) continue;
      dst.push(pluckNumber(row, p));
    }
  });

  {
    const dir = dirname(fleetPath);
    const tape = readTape(join(dir, `metrics.${query.policy}.oakt`));
    if (tape !== null) {
      let meta: { envSchedule?: { envId: string; seedStart: number; seedEnd: number }[]; envId?: string; seeds?: number[] } = {};
      try {
        meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8')) as typeof meta;
      } catch {
      }
      const envOf = (seed: number): string => {
        for (const e of meta.envSchedule ?? []) {
          if (seed >= e.seedStart && seed <= e.seedEnd) return e.envId;
        }
        return meta.envId ?? '';
      };
      const seeds = meta.seeds ?? Array.from({ length: tape.laneCount }, (_, i) => i);
      const windows = sampleWindows(tape.data.length, MAX_TAPE_WINDOWS);
      for (const w of windows) {
        const stepEnd = (tape.stepStart[w] ?? 0) + (tape.stepCount[w] ?? 0);
        if (stepSlot.has(stepEnd)) continue;
        let si: number | null = null;
        for (let lane = 0; lane < tape.laneCount; lane++) {
          const seed = seeds[lane] ?? lane;
          const envId = envOf(seed);
          if (query.envId !== null && envId !== query.envId) continue;
          if (si === null) {
            si = stepValues.length;
            stepValues.push(stepEnd);
            stepSlot.set(stepEnd, si);
          }
          let li = laneSlot.get(lane);
          if (li === undefined) {
            li = laneValues.length;
            laneValues.push({ lane, seed, policy: query.policy, envId });
            laneSlot.set(lane, li);
          }
          rowLane.push(li);
          rowStep.push(si);
          for (let f = 0; f < query.fields.length; f++) {
            const name = query.fields[f];
            const dst = rowVals[f];
            if (name === undefined || dst === undefined) continue;
            dst.push(tapeValue(tape, name, w, lane));
          }
        }
      }
    }
  }

  const stepOrder = stepValues.map((_, i) => i).sort((a, b) => (stepValues[a] ?? 0) - (stepValues[b] ?? 0));
  const stepRemap = new Array<number>(stepValues.length).fill(0);
  const steps: number[] = [];
  for (let newIdx = 0; newIdx < stepOrder.length; newIdx++) {
    const oldIdx = stepOrder[newIdx];
    if (oldIdx === undefined) continue;
    stepRemap[oldIdx] = newIdx;
    steps.push(stepValues[oldIdx] ?? 0);
  }

  const laneOrder = laneValues.map((_, i) => i).sort((a, b) => (laneValues[a]?.lane ?? 0) - (laneValues[b]?.lane ?? 0));
  const laneRemap = new Array<number>(laneValues.length).fill(0);
  const lanes: LaneKeyed[] = [];
  for (let newIdx = 0; newIdx < laneOrder.length; newIdx++) {
    const oldIdx = laneOrder[newIdx];
    if (oldIdx === undefined) continue;
    laneRemap[oldIdx] = newIdx;
    const lv = laneValues[oldIdx];
    if (lv !== undefined) lanes.push(lv);
  }

  const nL = lanes.length;
  const nS = steps.length;
  const series: Record<string, (number | null)[][]> = {};
  for (let f = 0; f < query.fields.length; f++) {
    const name = query.fields[f];
    if (name === undefined) continue;
    const grid: (number | null)[][] = new Array(nL);
    for (let l = 0; l < nL; l++) grid[l] = new Array<number | null>(nS).fill(null);
    const src = rowVals[f];
    if (src !== undefined) {
      for (let r = 0; r < src.length; r++) {
        const li = laneRemap[rowLane[r] ?? 0] ?? 0;
        const si = stepRemap[rowStep[r] ?? 0] ?? 0;
        const lane = grid[li];
        if (lane === undefined) continue;
        const v = src[r];
        lane[si] = v === undefined ? null : v;
      }
    }
    series[name] = grid;
  }

  return { steps, lanes, series };
}
