// DeepSea-cont-v0 decoder (N = 8).
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_deep_step L205, gw_deep_obs L192
//   tools/gridworld_ref.py       deepsea_reset L181, deepsea_step L194
//   Sources/OakKernel.metal      pose packing L2682-2692, init L2258-2265
//
// The hidden per-cell action map M[r][c] is bit r*8+c of a 64-bit mask drawn
// ONCE per lane from the RESET stream (ctr = (0,0,0,0), words x and y) and
// fixed for the lane's lifetime. M == 1 -> action 1 moves right, action 0
// moves left; M == 0 inverts that. c is clamped to [0,7]; r += 1 every step.
// On r+1 >= 7 the agent RESURFACES to (0,0) (mask kept) — and if it arrived
// at column 7 it also collects the treasure (+1). The wrap makes the treasure
// invisible in the record, so success is recovered by re-simulating the last
// dive step against the regenerated mask.
//
// pose = [r, c, 7, 7, 0, 0, 0, 0, 0]; done is never emitted.

import { philoxDraw, STREAM_RESET } from '../philox';
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const DEEPSEA_N = 8;
/** the -0.01/N per-right-move cost */
export const DEEPSEA_RIGHT_COST = -0.00125;
/** the last row the agent can occupy: reaching r = N-1 wraps to (0,0) */
export const DEEPSEA_LAST_ROW = DEEPSEA_N - 2;

export interface DeepSeaState {
  /** row 0..N-2 (row N-1 is never observed: it wraps in the same step) */
  r: number;
  /** column 0..N-1 */
  c: number;
  /** action & 1 of the step that produced this frame */
  lastA: number;
}

export interface DeepSeaStatic {
  seed: number;
  /** the two mask words as drawn (bit r*8+c) */
  maskLo: number;
  maskHi: number;
  /** N*N row-major expansion of the mask, 1 = action 1 moves right */
  mask: Uint8Array;
}

/** RESET-stream mask draw (`deepsea_reset`). */
export function deepSeaMaskWords(seed: number): [number, number] {
  const raw = philoxDraw(seed, STREAM_RESET, 0, 0);
  return [raw[0], raw[1]];
}

/** N*N row-major expansion of a (lo, hi) mask pair. */
export function deepSeaMask(maskLo: number, maskHi: number): Uint8Array {
  const mask = new Uint8Array(DEEPSEA_N * DEEPSEA_N);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = i < 32 ? (maskLo >>> i) & 1 : (maskHi >>> (i - 32)) & 1;
  }
  return mask;
}

/** M[r][c] as the kernel reads it. */
export function deepSeaBit(st: DeepSeaStatic, r: number, c: number): number {
  return st.mask[r * DEEPSEA_N + c] ?? 0;
}

/**
 * Column the agent lands in when taking `action` from (r, c) — the pre-wrap
 * column, i.e. `nc` in `gw_deep_step`.
 */
export function deepSeaNextCol(st: DeepSeaStatic, r: number, c: number, action: number): number {
  const dc = (action & 1) === deepSeaBit(st, r, c) ? 1 : -1;
  return Math.min(Math.max(c + dc, 0), DEEPSEA_N - 1);
}

function state(r: number, c: number, lastA: number): DeepSeaState {
  return { r, c, lastA };
}

export const deepSeaDecoder: EnvDecoder<DeepSeaState, DeepSeaStatic> = {
  envId: 'DeepSea-cont-v0',
  needsReplay: false,

  initStatic(seed: number): DeepSeaStatic {
    const [maskLo, maskHi] = deepSeaMaskWords(seed);
    return { seed, maskLo, maskHi, mask: deepSeaMask(maskLo, maskHi) };
  },

  geometry(st: DeepSeaStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: DEEPSEA_N,
        h: DEEPSEA_N,
        walls: new Uint8Array(DEEPSEA_N * DEEPSEA_N),
        goal: { x: DEEPSEA_N - 1, y: DEEPSEA_N - 1 },
        actionMask: st.mask,
      },
    };
  },

  initFrame(): EnvFrame<DeepSeaState> {
    const pose = new Float32Array(POSE_LEN);
    pose[2] = DEEPSEA_N - 1;
    pose[3] = DEEPSEA_N - 1;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(0, 0, 0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<DeepSeaState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    const ctrl = raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN);
    return {
      step: i,
      ctrl,
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[1] ?? 0, Math.trunc(ctrl[0] ?? 0) & 1),
    };
  },

  deriveEvents(
    prev: EnvFrame<DeepSeaState>,
    next: EnvFrame<DeepSeaState>,
    st: DeepSeaStatic,
  ): SimEvent[] {
    // r increments every step, so leaving row N-2 ALWAYS wraps to (0, 0).
    if (prev.state.r !== DEEPSEA_LAST_ROW) return NO_EVENTS;
    const action = Math.trunc(next.ctrl[0] ?? 0);
    const landed = deepSeaNextCol(st, prev.state.r, prev.state.c, action);
    const out: SimEvent[] = [];
    if (landed === DEEPSEA_N - 1) out.push({ type: 'goal' });
    out.push({ type: 'resurface' });
    return out;
  },
};
