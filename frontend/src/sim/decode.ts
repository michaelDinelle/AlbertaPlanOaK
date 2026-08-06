
import { ENVS, isEnvId } from '@shared/envs';
import { NO_EVENTS } from './events';
import { RECORD_FLOATS } from './record';
import type { DecodedRecording, EnvDecoder, EnvFrame, RawRecording, SimEvent } from './types';

export { NO_EVENTS };

export class DecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DecodeError';
  }
}

export interface DecodeOptions {
  maxSteps?: number;
}

function rowHasNonFinite(row: Float32Array): boolean {
  for (let j = 0; j < RECORD_FLOATS; j++) {
    if (!Number.isFinite(row[j])) return true;
  }
  return false;
}

function poseAgreesOf(state: unknown): boolean | undefined {
  if (typeof state !== 'object' || state === null) return undefined;
  const v = (state as { poseAgrees?: unknown }).poseAgrees;
  return typeof v === 'boolean' ? v : undefined;
}

function markDisagree<S>(state: S): S {
  if (poseAgreesOf(state) === undefined) return state;
  return { ...(state as object), poseAgrees: false } as S;
}

interface Drive<S, St> {
  decoder: EnvDecoder<S, St>;
  st: St;
  tracks: boolean;
  frozen: boolean;
  frozenState: S | undefined;
  nonFiniteRows: number;
  disagreeCount: number;
  firstDisagreeStep: number | null;
}

function startDrive<S, St>(
  decoder: EnvDecoder<S, St>,
  st: St,
  init: EnvFrame<S>,
): Drive<S, St> {
  return {
    decoder,
    st,
    tracks: decoder.needsReplay || poseAgreesOf(init.state) !== undefined,
    frozen: false,
    frozenState: undefined,
    nonFiniteRows: 0,
    disagreeCount: 0,
    firstDisagreeStep: null,
  };
}

function driveRow<S, St>(
  d: Drive<S, St>,
  i: number,
  row: Float32Array,
  prev: EnvFrame<S>,
  out: { frame: EnvFrame<S>; events: SimEvent[] },
): void {
  const damaged = rowHasNonFinite(row);
  if (damaged) d.nonFiniteRows++;

  if (d.frozen || (damaged && d.decoder.needsReplay)) {
    if (!d.frozen) {
      d.frozen = true;
      d.frozenState = markDisagree(prev.state);
    }
    out.frame = {
      step: i,
      ctrl: prev.ctrl,
      done: 0,
      pose: prev.pose,
      state: d.frozenState as S,
    };
    out.events = NO_EVENTS;
  } else {
    out.frame = d.decoder.decodeFrame(i, row, d.st, prev);
    out.events = d.decoder.deriveEvents(prev, out.frame, d.st);
  }

  if (d.tracks && (d.frozen || poseAgreesOf(out.frame.state) === false)) {
    d.disagreeCount++;
    if (d.firstDisagreeStep === null) d.firstDisagreeStep = i;
  }
}

export function decodeRecording<S, St>(
  raw: RawRecording,
  decoder: EnvDecoder<S, St>,
  opts: DecodeOptions = {},
): DecodedRecording<S> {
  if (!isEnvId(raw.envId)) {
    throw new DecodeError(`unknown recording envId ${JSON.stringify(raw.envId)}`);
  }
  if (decoder.envId !== raw.envId) {
    throw new DecodeError(
      `decoder envId ${decoder.envId} does not match recording envId ${raw.envId}`,
    );
  }
  const available = Math.floor(raw.data.length / RECORD_FLOATS);
  if (available < raw.steps) {
    throw new DecodeError(
      `recording claims ${raw.steps} steps but data holds only ${available}`,
    );
  }
  const limit = clampSteps(raw.steps, opts);

  const info = ENVS[raw.envId];
  const st = decoder.initStatic(raw.seed);
  const geometry = decoder.geometry(st, raw.seed);

  const frames: EnvFrame<S>[] = new Array<EnvFrame<S>>(limit + 1);
  const events: SimEvent[][] = new Array<SimEvent[]>(limit + 1);

  const init = decoder.initFrame(raw.seed, st);
  frames[0] = init;
  events[0] = NO_EVENTS;

  const d = startDrive(decoder, st, init);
  const out = { frame: init, events: NO_EVENTS };
  let prev = init;
  for (let i = 0; i < limit; i++) {
    const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
    driveRow(d, i, row, prev, out);
    frames[i + 1] = out.frame;
    events[i + 1] = out.events;
    prev = out.frame;
  }

  return {
    envId: raw.envId,
    seed: raw.seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry,
    frames,
    events,
    ...(d.tracks
      ? { replay: { firstDisagreeStep: d.firstDisagreeStep, disagreeCount: d.disagreeCount } }
      : {}),
    nonFiniteRows: d.nonFiniteRows,
  };
}

export function walkRecording<S, St>(
  raw: RawRecording,
  decoder: EnvDecoder<S, St>,
  visit: (frame: EnvFrame<S>, events: SimEvent[], index: number) => boolean | void,
  opts: DecodeOptions = {},
): void {
  if (!isEnvId(raw.envId)) {
    throw new DecodeError(`unknown recording envId ${JSON.stringify(raw.envId)}`);
  }
  if (decoder.envId !== raw.envId) {
    throw new DecodeError(
      `decoder envId ${decoder.envId} does not match recording envId ${raw.envId}`,
    );
  }
  const available = Math.floor(raw.data.length / RECORD_FLOATS);
  if (available < raw.steps) {
    throw new DecodeError(
      `recording claims ${raw.steps} steps but data holds only ${available}`,
    );
  }
  const limit = clampSteps(raw.steps, opts);
  const st = decoder.initStatic(raw.seed);
  const init = decoder.initFrame(raw.seed, st);
  if (visit(init, NO_EVENTS, 0) === false) return;

  const d = startDrive(decoder, st, init);
  const out = { frame: init, events: NO_EVENTS };
  let prev = init;
  for (let i = 0; i < limit; i++) {
    const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
    driveRow(d, i, row, prev, out);
    if (visit(out.frame, out.events, i + 1) === false) return;
    prev = out.frame;
  }
}

function clampSteps(steps: number, opts: DecodeOptions): number {
  return opts.maxSteps === undefined
    ? steps
    : Math.max(0, Math.min(steps, Math.floor(opts.maxSteps)));
}


export const LENS_BLOCK = 4096;

interface LensCheckpoint<S> {
  frame: EnvFrame<S>;
  frozen: boolean;
  frozenState: S | undefined;
}

export interface RecordingLens<S = unknown> {
  envId: RawRecording['envId'];
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: ReturnType<EnvDecoder<S, never>['geometry']>;
  frameCount: number;
  steps: number;
  frameAt(i: number): EnvFrame<S>;
  eventsAt(i: number): SimEvent[];
  eventEntries(): IterableIterator<[number, SimEvent[]]>;
  replay?: { firstDisagreeStep: number | null; disagreeCount: number };
  nonFiniteRows: number;
}

export function lensRecording<S, St>(
  raw: RawRecording,
  decoder: EnvDecoder<S, St>,
): RecordingLens<S> {
  if (!isEnvId(raw.envId)) {
    throw new DecodeError(`unknown recording envId ${JSON.stringify(raw.envId)}`);
  }
  if (decoder.envId !== raw.envId) {
    throw new DecodeError(
      `decoder envId ${decoder.envId} does not match recording envId ${raw.envId}`,
    );
  }
  const available = Math.floor(raw.data.length / RECORD_FLOATS);
  if (available < raw.steps) {
    throw new DecodeError(
      `recording claims ${raw.steps} steps but data holds only ${available}`,
    );
  }
  const steps = raw.steps;
  const info = ENVS[raw.envId];
  const st = decoder.initStatic(raw.seed);
  const geometry = decoder.geometry(st, raw.seed);
  const init = decoder.initFrame(raw.seed, st);

  const eventsMap = new Map<number, SimEvent[]>();
  const checkpoints: LensCheckpoint<S>[] = [];
  const d = startDrive(decoder, st, init);
  const out = { frame: init, events: NO_EVENTS as SimEvent[] };
  let prev = init;
  for (let i = 0; i < steps; i++) {
    if (i % LENS_BLOCK === 0) {
      checkpoints.push({ frame: prev, frozen: d.frozen, frozenState: d.frozenState });
    }
    const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
    driveRow(d, i, row, prev, out);
    if (out.events !== NO_EVENTS && out.events.length > 0) eventsMap.set(i + 1, out.events);
    prev = out.frame;
  }

  const cache = new Map<number, EnvFrame<S>[]>();
  const decodeBlock = (b: number): EnvFrame<S>[] => {
    const hit = cache.get(b);
    if (hit !== undefined) return hit;
    const start = b * LENS_BLOCK;
    const count = Math.min(LENS_BLOCK, steps - start);
    const ck = checkpoints[b];
    if (ck === undefined) throw new DecodeError(`lens block ${b} out of range`);
    const rd: Drive<S, St> = {
      decoder,
      st,
      tracks: false,
      frozen: ck.frozen,
      frozenState: ck.frozenState,
      nonFiniteRows: 0,
      disagreeCount: 0,
      firstDisagreeStep: null,
    };
    const block: EnvFrame<S>[] = new Array<EnvFrame<S>>(count);
    const o = { frame: ck.frame, events: NO_EVENTS as SimEvent[] };
    let p = ck.frame;
    for (let k = 0; k < count; k++) {
      const i = start + k;
      const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
      driveRow(rd, i, row, p, o);
      block[k] = o.frame;
      p = o.frame;
    }
    cache.set(b, block);
    if (cache.size > 3) {
      for (const key of cache.keys()) {
        if (Math.abs(key - b) > 1) cache.delete(key);
        if (cache.size <= 3) break;
      }
    }
    return block;
  };

  const frameAt = (i: number): EnvFrame<S> => {
    const clamped = Math.max(0, Math.min(steps, Math.floor(i)));
    if (clamped === 0) return init;
    const row = clamped - 1;
    const b = Math.floor(row / LENS_BLOCK);
    const block = decodeBlock(b);
    const frame = block[row - b * LENS_BLOCK];
    if (frame === undefined) throw new DecodeError(`lens frame ${clamped} missing`);
    return frame;
  };

  const sortedEvents = [...eventsMap.entries()].sort((a, b) => a[0] - b[0]);

  return {
    envId: raw.envId,
    seed: raw.seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry,
    frameCount: steps + 1,
    steps,
    frameAt,
    eventsAt: (i) => eventsMap.get(Math.floor(i)) ?? NO_EVENTS,
    eventEntries: () => sortedEvents.values() as IterableIterator<[number, SimEvent[]]>,
    ...(d.tracks
      ? { replay: { firstDisagreeStep: d.firstDisagreeStep, disagreeCount: d.disagreeCount } }
      : {}),
    nonFiniteRows: d.nonFiniteRows,
  };
}

export function eventTimeline<S>(
  dec: DecodedRecording<S>,
): { frame: number; step: number; event: SimEvent }[] {
  const out: { frame: number; step: number; event: SimEvent }[] = [];
  for (let i = 0; i < dec.events.length; i++) {
    const evs = dec.events[i];
    if (!evs || evs.length === 0) continue;
    const step = dec.frames[i]?.step ?? i - 1;
    for (const event of evs) out.push({ frame: i, step, event });
  }
  return out;
}
