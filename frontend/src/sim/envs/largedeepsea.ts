// LargeDeepSea-cont-v0 decoder (N = 16).
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_ldeep_step L356, gw_ldeep_obs L343
//   tools/gridworld_ref.py       ldeepsea_reset L310, ldeepsea_step L324
//   Sources/OakKernel.metal      pose packing L2726-2737, init L2278-2290
//
// DeepSea scaled to 16x16: the hidden per-cell action map is bit r*16+c of a
// 256-bit mask drawn ONCE per lane from the RESET stream — words 0..3 from
// ctr = (0,0,0,0) and words 4..7 from ctr = (1,0,0,0) — then fixed for the
// lane's lifetime. Reaching r+1 >= 15 resurfaces to (0,0); arriving at
// column 15 on that step also collects the treasure (+1), which the wrap
// hides, so success is recovered by re-simulating the last dive step.
//
// pose = [r, c, 15, 15, 0, 0, 0, 0, 0]; done is never emitted.

import { philoxDraw, STREAM_RESET } from '../philox';
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const LARGE_DEEPSEA_N = 16;
/** the -0.01/N per-right-move cost */
export const LARGE_DEEPSEA_RIGHT_COST = -0.000625;
/** the last row the agent can occupy: reaching r = N-1 wraps to (0,0) */
export const LARGE_DEEPSEA_LAST_ROW = LARGE_DEEPSEA_N - 2;

export interface LargeDeepSeaState {
  /** row 0..N-2 (row N-1 is never observed: it wraps in the same step) */
  r: number;
  /** column 0..N-1 */
  c: number;
  /** action & 1 of the step that produced this frame */
  lastA: number;
}

export interface LargeDeepSeaStatic {
  seed: number;
  /** the 8 mask words as drawn (bit r*16+c, word = idx>>5) */
  words: Uint32Array;
  /** N*N row-major expansion of the mask, 1 = action 1 moves right */
  mask: Uint8Array;
}

/** RESET-stream mask draw (`ldeepsea_reset`): blocks 0 and 1, 4 words each. */
export function largeDeepSeaMaskWords(seed: number): Uint32Array {
  const a = philoxDraw(seed, STREAM_RESET, 0, 0);
  const b = philoxDraw(seed, STREAM_RESET, 1, 0);
  return Uint32Array.from([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]);
}

/** N*N row-major expansion of the 8 mask words. */
export function largeDeepSeaMask(words: ArrayLike<number>): Uint8Array {
  const mask = new Uint8Array(LARGE_DEEPSEA_N * LARGE_DEEPSEA_N);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = ((words[i >>> 5] ?? 0) >>> (i & 31)) & 1;
  }
  return mask;
}

/** M[r][c] as the kernel reads it. */
export function largeDeepSeaBit(st: LargeDeepSeaStatic, r: number, c: number): number {
  return st.mask[r * LARGE_DEEPSEA_N + c] ?? 0;
}

/** Pre-wrap column reached by `action` from (r, c) — `nc` in `gw_ldeep_step`. */
export function largeDeepSeaNextCol(
  st: LargeDeepSeaStatic,
  r: number,
  c: number,
  action: number,
): number {
  const dc = (action & 1) === largeDeepSeaBit(st, r, c) ? 1 : -1;
  return Math.min(Math.max(c + dc, 0), LARGE_DEEPSEA_N - 1);
}

function state(r: number, c: number, lastA: number): LargeDeepSeaState {
  return { r, c, lastA };
}

export const largeDeepSeaDecoder: EnvDecoder<LargeDeepSeaState, LargeDeepSeaStatic> = {
  envId: 'LargeDeepSea-cont-v0',
  needsReplay: false,

  initStatic(seed: number): LargeDeepSeaStatic {
    const words = largeDeepSeaMaskWords(seed);
    return { seed, words, mask: largeDeepSeaMask(words) };
  },

  geometry(st: LargeDeepSeaStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: LARGE_DEEPSEA_N,
        h: LARGE_DEEPSEA_N,
        walls: new Uint8Array(LARGE_DEEPSEA_N * LARGE_DEEPSEA_N),
        goal: { x: LARGE_DEEPSEA_N - 1, y: LARGE_DEEPSEA_N - 1 },
        actionMask: st.mask,
      },
    };
  },

  initFrame(): EnvFrame<LargeDeepSeaState> {
    const pose = new Float32Array(POSE_LEN);
    pose[2] = LARGE_DEEPSEA_N - 1;
    pose[3] = LARGE_DEEPSEA_N - 1;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(0, 0, 0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<LargeDeepSeaState> {
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
    prev: EnvFrame<LargeDeepSeaState>,
    next: EnvFrame<LargeDeepSeaState>,
    st: LargeDeepSeaStatic,
  ): SimEvent[] {
    if (prev.state.r !== LARGE_DEEPSEA_LAST_ROW) return NO_EVENTS;
    const action = Math.trunc(next.ctrl[0] ?? 0);
    const landed = largeDeepSeaNextCol(st, prev.state.r, prev.state.c, action);
    const out: SimEvent[] = [];
    if (landed === LARGE_DEEPSEA_N - 1) out.push({ type: 'goal' });
    out.push({ type: 'resurface' });
    return out;
  },
};
