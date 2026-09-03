import { useMemo, useSyncExternalStore } from 'react';
import type { EnvScheduleEntry } from '@shared/types';
import type { BinnedRates, CumulativeReward, SeedRow, SurvivalSeries } from './derive';

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
  byteLength: number;
  lastStep: number;
  nextRowDueMs: number;
  windowSteps: number;
}

export type WorkerCall =
  | { op: 'tail'; runId: string }
  | { op: 'derive'; runId: string; spec: DeriveSpec };

export type WorkerRequest = WorkerCall & { id: number };

export type WorkerResponse =
  | { id: number; ok: true; progress: TapeProgress; result?: DeriveResults[DeriveMode] }
  | { id: number; ok: false; error: string };

interface Slot {
  runId: string;
  spec: DeriveSpec;
  data: DeriveResults[DeriveMode] | null;
  error: string | null;
  inflight: boolean;
  stale: boolean;
  listeners: Set<() => void>;
}

const slots = new Map<string, Slot>();

export function snapshotKey(runId: string, envId: string | null, mode: DeriveMode): string {
  return `${runId}|${envId ?? ''}|${mode}`;
}

let worker: Worker | null = null;
let nextRequestId = 1;
const pending = new Map<number, (response: WorkerResponse) => void>();

function ensureWorker(): Worker {
  if (worker !== null) return worker;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const resolve = pending.get(event.data.id);
    if (resolve === undefined) return;
    pending.delete(event.data.id);
    resolve(event.data);
  };
  return worker;
}

function rpc(request: WorkerCall): Promise<WorkerResponse> {
  const target = ensureWorker();
  const id = nextRequestId++;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    target.postMessage({ ...request, id });
  });
}

function notify(slot: Slot): void {
  for (const listener of slot.listeners) listener();
}

function runDerive(key: string): void {
  const slot = slots.get(key);
  if (slot === undefined) return;
  if (slot.inflight) {
    slot.stale = true;
    return;
  }
  slot.inflight = true;
  slot.stale = false;
  void rpc({ op: 'derive', runId: slot.runId, spec: slot.spec }).then((response) => {
    slot.inflight = false;
    if (response.ok) {
      slot.data = response.result ?? null;
      slot.error = null;
      recordProgress(slot.runId, response.progress);
    } else {
      slot.error = response.error;
    }
    notify(slot);
    if (slot.stale) runDerive(key);
  });
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
    slot = { runId, spec, data: null, error: null, inflight: false, stale: false, listeners: new Set() };
    slots.set(key, slot);
  } else {
    slot.spec = spec;
  }
  slot.listeners.add(listener);
  if (slot.data === null && !slot.inflight) runDerive(key);
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

interface TailState {
  live: boolean;
  timer: ReturnType<typeof setTimeout> | null;
  backoffMs: number;
  lastRows: number;
  ticking: boolean;
}

const tails = new Map<string, TailState>();

export function setLive(runId: string, live: boolean): void {
  let tail = tails.get(runId);
  if (tail === undefined) {
    tail = { live: false, timer: null, backoffMs: 0, lastRows: 0, ticking: false };
    tails.set(runId, tail);
  }
  if (tail.live === live) return;
  tail.live = live;
  if (live) {
    if (tail.timer === null && !tail.ticking) void tick(runId);
  } else if (tail.timer !== null) {
    clearTimeout(tail.timer);
    tail.timer = null;
  }
}

async function tick(runId: string): Promise<void> {
  const tail = tails.get(runId);
  if (tail === undefined || !tail.live) return;
  tail.ticking = true;
  tail.timer = null;
  const startedAt = performance.now();
  const response = await rpc({ op: 'tail', runId });
  tail.ticking = false;
  if (!tail.live) return;
  let dueMs = 0;
  if (response.ok) {
    dueMs = response.progress.nextRowDueMs;
    recordProgress(runId, response.progress);
    if (response.progress.rows > tail.lastRows) {
      tail.lastRows = response.progress.rows;
      tail.backoffMs = 0;
      refreshRun(runId);
    } else {
      tail.backoffMs = Math.max(performance.now() - startedAt, tail.backoffMs * 2);
    }
  } else {
    tail.backoffMs = Math.max(1000, tail.backoffMs * 2);
  }
  tail.timer = setTimeout(() => {
    void tick(runId);
  }, Math.max(dueMs, tail.backoffMs));
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
