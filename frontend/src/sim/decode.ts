// The decode driver: one sequential pass over a RawRecording, producing the
// DecodedRecording the theater plays back.
//
// Contract (src/sim/types.ts):
//   frames[0]   = decoder.initFrame(seed, static)   — the synthesized step -1
//   frames[i+1] = decoder.decodeFrame(i, row_i, static, frames[i])
//   events[i]   = events fired on the transition INTO frames[i]; events[0] = []
//                 (empty entries are the shared frozen NO_EVENTS singleton and
//                 must never be mutated — see DecodedRecording.events)
//
// The pass is strictly sequential and prev-threaded so replay-dependent envs
// (MultiKeyCorridor) can advance hidden state as they go. Row views are
// zero-copy subarrays of the recording buffer: rows are disjoint and the
// buffer outlives the decode, so a decoder may retain a row view safely.
//
// Damage handling: a record row carrying a non-finite float is counted
// (`nonFiniteRows`) and, for `needsReplay` decoders, FREEZES the replay —
// `Math.trunc(NaN) & 3 === 0` would otherwise feed the hidden-state replay a
// plausible-looking action and desync every later frame with no way back.

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
  /** decode only the first N record rows (frames = N + 1). Default: all. */
  maxSteps?: number;
}

/** true when any of the row's 16 floats is NaN/±Infinity (or the row is short). */
function rowHasNonFinite(row: Float32Array): boolean {
  for (let j = 0; j < RECORD_FLOATS; j++) {
    if (!Number.isFinite(row[j])) return true;
  }
  return false;
}

/** the state's `poseAgrees` flag if it has one (replay decoders), else undefined. */
function poseAgreesOf(state: unknown): boolean | undefined {
  if (typeof state !== 'object' || state === null) return undefined;
  const v = (state as { poseAgrees?: unknown }).poseAgrees;
  return typeof v === 'boolean' ? v : undefined;
}

/** shallow copy with `poseAgrees` forced false; unchanged if the state has no flag. */
function markDisagree<S>(state: S): S {
  if (poseAgreesOf(state) === undefined) return state;
  return { ...(state as object), poseAgrees: false } as S;
}

/** shared per-pass bookkeeping so `decodeRecording` and `walkRecording` agree. */
interface Drive<S, St> {
  decoder: EnvDecoder<S, St>;
  st: St;
  /** replay health is tracked for needsReplay decoders and any poseAgrees state */
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

/**
 * Decode one row. Once a `needsReplay` decoder hits a damaged row the frame is
 * a HOLD of the previous one (ctrl/pose/state carried over, done cleared, no
 * events) with `poseAgrees` forced false — the replay is stopped rather than
 * marched forward through garbage. `step` still advances so playback indices,
 * scrubbing and event alignment stay intact.
 */
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

/**
 * Decode a recording with its env decoder.
 *
 * Throws a DecodeError if the recording's envId is not a known env, if the
 * decoder's envId does not match it, or if the recording claims more steps than
 * its buffer holds. Reports two health fields alongside the frames:
 * `nonFiniteRows` (damaged record rows in the decoded window) and, for replay
 * decoders, `replay` (first disagreeing row + total).
 */
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

/**
 * Streaming variant: decode without materializing every frame. `visit` is
 * called for the init frame (step -1, no events) and then once per row.
 * Return `false` from `visit` to stop early. Same `maxSteps` clamp and same
 * damaged-row handling as `decodeRecording`, so both agree frame for frame.
 */
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

/** `opts.maxSteps` clamped into [0, steps]; `steps` when unset. */
function clampSteps(steps: number, opts: DecodeOptions): number {
  return opts.maxSteps === undefined
    ? steps
    : Math.max(0, Math.min(steps, Math.floor(opts.maxSteps)));
}

// ---------------------------------------------------------------------------
// the lens: full-horizon random access without full-horizon materialization
// ---------------------------------------------------------------------------

/** Frames materialized per block; checkpoints are block-aligned. */
export const LENS_BLOCK = 4096;

interface LensCheckpoint<S> {
  /** the frame BEFORE the block's first row (prev-chain resume point) */
  frame: EnvFrame<S>;
  frozen: boolean;
  frozenState: S | undefined;
}

/**
 * Random access over a full recording at bounded memory: the raw buffer stays
 * whole (64 MB for a 1M-step lane), frames materialize in LENS_BLOCK windows
 * around the playhead, and every mutable input of the decode rides the
 * prev-frame chain — so a block-aligned checkpoint (`{frame, frozen,
 * frozenState}`) resumes `driveRow` exactly, replay decoders included.
 *
 * One full streaming pass at construction captures the sparse event map,
 * damage/replay health, and the checkpoints (~245 retained frames for a 1M
 * lane). Sequential playback hits the block cache; a scrub re-decodes at most
 * one block (≤ LENS_BLOCK rows) from its checkpoint.
 */
export interface RecordingLens<S = unknown> {
  envId: RawRecording['envId'];
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: ReturnType<EnvDecoder<S, never>['geometry']>;
  /** steps + 1 (frames[0] is the synthesized init frame, step −1) */
  frameCount: number;
  /** record steps (frameCount − 1) */
  steps: number;
  frameAt(i: number): EnvFrame<S>;
  /** events fired on the transition INTO frame i (NO_EVENTS when none) */
  eventsAt(i: number): SimEvent[];
  /** sparse — only frames with events; ascending frame index */
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

  // ---- construction pass: events + health + block checkpoints ----
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

  // ---- the block cache: current + previous (interpolation reads i and i+1) ----
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
      tracks: false, // health was captured in the construction pass
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
    // keep the cache tiny: the current block plus its neighbours
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
    const row = clamped - 1; // record row index
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

/** Flatten a DecodedRecording's events into a time-indexed list. */
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
