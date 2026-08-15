
import { philoxDraw, STREAM_RESET } from '../philox';
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const LARGE_DEEPSEA_N = 16;
export const LARGE_DEEPSEA_RIGHT_COST = -0.000625;
export const LARGE_DEEPSEA_LAST_ROW = LARGE_DEEPSEA_N - 2;

export interface LargeDeepSeaState {
  r: number;
  c: number;
  lastA: number;
}

export interface LargeDeepSeaStatic {
  seed: number;
  words: Uint32Array;
  mask: Uint8Array;
}

export function largeDeepSeaMaskWords(seed: number): Uint32Array {
  const a = philoxDraw(seed, STREAM_RESET, 0, 0);
  const b = philoxDraw(seed, STREAM_RESET, 1, 0);
  return Uint32Array.from([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]);
}

export function largeDeepSeaMask(words: ArrayLike<number>): Uint8Array {
  const mask = new Uint8Array(LARGE_DEEPSEA_N * LARGE_DEEPSEA_N);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = ((words[i >>> 5] ?? 0) >>> (i & 31)) & 1;
  }
  return mask;
}

export function largeDeepSeaBit(st: LargeDeepSeaStatic, r: number, c: number): number {
  return st.mask[r * LARGE_DEEPSEA_N + c] ?? 0;
}

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
    const ctrl = effortView(raw16, 4);
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
