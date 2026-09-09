import { useMemo, useSyncExternalStore } from 'react';
import type { EnvScheduleEntry } from '../data/types';
import { fetchSeries, follow } from '../data/oak';
import {
  buildSeedRows,
  cumulativeReward,
  rhoBinned,
  survivalSeries,
  windowedRates,
  type BinnedRates,
  type CumulativeReward,
  type SeedRow,
  type SurvivalSeries,
  type Tape,
} from './derive';

export interface DeriveResults {
  rates: BinnedRates;
  windowedRates: BinnedRates;
  rho: BinnedRates;
  survival: SurvivalSeries;
  rows: SeedRow[];
  cumulativeReward: CumulativeReward;
}

export type DeriveMode = keyof DeriveResults;

export type DeriveSpec =
  | { mode: 'rho' | 'survival' | 'cumulativeReward'; lanes: number[] }
  | { mode: 'rates' | 'windowedRates'; lanes: number[]; win: number }
  | {
      mode: 'rows';
      envId: string;
      envSchedule: EnvScheduleEntry[];
      seeds: number[];
      steps: number;
      recorded: number[];
    };

export interface TapeProgress {
  rows: number;
  lastStep: number;
  windowSteps: number;
}

export const TAPE_COLUMNS = ['rewardSum', 'rho', 'deathStep', 'deathDetector'] as const;

interface SeriesCache {
  runId: string;
  tape: Tape | null;
  columns: Map<string, number[]>;
  ends: number[];
  wall: number[];
  loading: Promise<void> | null;
  error: string | null;
  columnOrder: string[] | null;
}

const caches = new Map<string, SeriesCache>();

function cacheFor(runId: string): SeriesCache {
  let cache = caches.get(runId);
  if (cache === undefined) {
    cache = { runId, tape: null, columns: new Map(), ends: [], wall: [], loading: null, error: null, columnOrder: null };
    caches.set(runId, cache);
  }
  return cache;
}

function rebuildTape(cache: SeriesCache): void {
  const rows = cache.ends.length;
  const first = cache.columns.get(TAPE_COLUMNS[0]);
  if (first === undefined || rows === 0) {
    cache.tape = null;
    return;
  }
  const laneCount = first.length / rows;
  const columns = new Map<string, Float32Array>();
  for (const [name, values] of cache.columns) columns.set(name, Float32Array.from(values));
  cache.tape = {
    rows,
    laneCount,
    stepEnd: Float64Array.from(cache.ends),
    wallSeconds: Float32Array.from(cache.wall),
    column(name: string) {
      const c = columns.get(name);
      if (c === undefined) throw new Error(`run has no column ${name}`);
      return c;
    },
  };
  recordProgress(cache.runId, {
    rows,
    lastStep: cache.ends[rows - 1] ?? 0,
    windowSteps: rows > 1 ? (cache.ends[rows - 1] ?? 0) - (cache.ends[rows - 2] ?? 0) : cache.ends[0] ?? 1,
  });
}

function load(cache: SeriesCache): Promise<void> {
  if (cache.loading !== null) return cache.loading;
  cache.loading = (async () => {
    try {
      const series = await Promise.all(TAPE_COLUMNS.map((column) => fetchSeries(cache.runId, column)));
      const head = series[0]!;
      cache.ends = Array.from(head.ends);
      cache.wall = Array.from(head.wall);
      cache.columns.clear();
      TAPE_COLUMNS.forEach((column, i) => cache.columns.set(column, Array.from(series[i]!.values)));
      cache.error = null;
      rebuildTape(cache);
    } catch (error) {
      cache.error = error instanceof Error ? error.message : String(error);
    } finally {
      cache.loading = null;
    }
    refreshRun(cache.runId);
  })();
  return cache.loading;
}

function appendWindow(cache: SeriesCache, end: number, wall: number, values: Float32Array): void {
  const order = cache.columnOrder;
  if (order === null || cache.tape === null) return;
  if (end <= (cache.ends[cache.ends.length - 1] ?? -1)) return;
  const laneCount = cache.tape.laneCount;
  if (values.length !== order.length * laneCount) return;
  if (TAPE_COLUMNS.some((column) => order.indexOf(column) < 0 || !cache.columns.has(column))) return;
  cache.ends.push(end);
  cache.wall.push(wall);
  for (const column of TAPE_COLUMNS) {
    const f = order.indexOf(column);
    const dst = cache.columns.get(column)!;
    for (let lane = 0; lane < laneCount; lane++) dst.push(values[f * laneCount + lane] ?? NaN);
  }
  rebuildTape(cache);
  refreshRun(cache.runId);
}

interface Slot {
  runId: string;
  spec: DeriveSpec;
  data: DeriveResults[DeriveMode] | null;
  error: string | null;
  listeners: Set<() => void>;
}

const slots = new Map<string, Slot>();

export function snapshotKey(runId: string, envId: string | null, mode: DeriveMode): string {
  return `${runId}|${envId ?? ''}|${mode}`;
}

function derive(tape: Tape, spec: DeriveSpec): DeriveResults[DeriveMode] {
  switch (spec.mode) {
    case 'rates':
    case 'windowedRates':
      return windowedRates(tape, spec.lanes, spec.win);
    case 'rho':
      return rhoBinned(tape, spec.lanes);
    case 'survival':
      return survivalSeries(tape, spec.lanes);
    case 'cumulativeReward':
      return cumulativeReward(tape, spec.lanes);
    case 'rows':
      return buildSeedRows(
        tape,
        { envSchedule: spec.envSchedule, seeds: spec.seeds, steps: spec.steps },
        spec.envId,
        new Set(spec.recorded),
      );
  }
}

function notify(slot: Slot): void {
  for (const listener of slot.listeners) listener();
}

function runDerive(key: string): void {
  const slot = slots.get(key);
  if (slot === undefined) return;
  const cache = cacheFor(slot.runId);
  if (cache.tape === null) {
    if (cache.error !== null && slot.error !== cache.error) {
      slot.error = cache.error;
      notify(slot);
    }
    void load(cache);
    return;
  }
  try {
    slot.data = derive(cache.tape, slot.spec);
    slot.error = null;
  } catch (error) {
    slot.error = error instanceof Error ? error.message : String(error);
  }
  notify(slot);
}

export function subscribeSnapshot(
  runId: string,
  envId: string | null,
  mode: DeriveMode,
  spec: DeriveSpec,
  listener: () => void,
): () => void {
  const key = snapshotKey(runId, envId, mode);
  let slot = slots.get(key);
  if (slot === undefined) {
    slot = { runId, spec, data: null, error: null, listeners: new Set() };
    slots.set(key, slot);
  } else {
    slot.spec = spec;
  }
  slot.listeners.add(listener);
  if (slot.data === null) runDerive(key);
  return () => {
    slot.listeners.delete(listener);
  };
}

export function snapshot<M extends DeriveMode>(
  runId: string,
  envId: string | null,
  mode: M,
): DeriveResults[M] | null {
  const slot = slots.get(snapshotKey(runId, envId, mode));
  return (slot?.data as DeriveResults[M] | null | undefined) ?? null;
}

export function refreshRun(runId: string): void {
  for (const [key, slot] of slots) {
    if (slot.runId === runId) runDerive(key);
  }
}

interface RunProgress {
  progress: TapeProgress | null;
  listeners: Set<() => void>;
}

const progressByRun = new Map<string, RunProgress>();

function runProgressSlot(runId: string): RunProgress {
  let slot = progressByRun.get(runId);
  if (slot === undefined) {
    slot = { progress: null, listeners: new Set() };
    progressByRun.set(runId, slot);
  }
  return slot;
}

function recordProgress(runId: string, progress: TapeProgress): void {
  const slot = runProgressSlot(runId);
  if (slot.progress !== null && slot.progress.rows === progress.rows) return;
  slot.progress = progress;
  for (const listener of slot.listeners) listener();
}

export function tapeProgress(runId: string): TapeProgress | null {
  return progressByRun.get(runId)?.progress ?? null;
}

export function subscribeProgress(runId: string, listener: () => void): () => void {
  const slot = runProgressSlot(runId);
  slot.listeners.add(listener);
  return () => {
    slot.listeners.delete(listener);
  };
}

const follows = new Map<string, () => void>();

export function setLive(runId: string, live: boolean): void {
  const current = follows.get(runId);
  if (live === (current !== undefined)) return;
  if (!live) {
    current?.();
    follows.delete(runId);
    return;
  }
  const cache = cacheFor(runId);
  follows.set(runId, follow(runId, -1, {
    columns(names) {
      cache.columnOrder = names;
    },
    window(_index, end, wall, values) {
      if (cache.tape === null) {
        void load(cache);
        return;
      }
      appendWindow(cache, end, wall, values);
    },
  }));
}

const noopSubscribe = (): (() => void) => () => {};

export function useTapeSnapshot<M extends DeriveMode>(
  runId: string | null,
  envId: string | null,
  mode: M,
  spec: (DeriveSpec & { mode: M }) | null,
): DeriveResults[M] | null {
  const subscribe = useMemo(() => {
    if (runId === null || spec === null) return noopSubscribe;
    return (listener: () => void) => subscribeSnapshot(runId, envId, mode, spec, listener);
  }, [runId, envId, mode, spec]);
  return useSyncExternalStore(subscribe, () => (runId === null ? null : snapshot(runId, envId, mode)));
}

export function useTapeProgress(runId: string | null): TapeProgress | null {
  const subscribe = useMemo(() => {
    if (runId === null) return noopSubscribe;
    return (listener: () => void) => subscribeProgress(runId, listener);
  }, [runId]);
  return useSyncExternalStore(subscribe, () => (runId === null ? null : tapeProgress(runId)));
}
