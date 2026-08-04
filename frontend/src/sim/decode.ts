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
