import { philoxDraw, STREAM_RESET } from '../philox';
import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const LARGE_DEEPSEA_N = 16;
export const LARGE_DEEPSEA_LAST_ROW = LARGE_DEEPSEA_N - 2;

export const LARGE_DEEPSEA_ROW = 0;
export const LARGE_DEEPSEA_COL = 1;
export const LARGE_DEEPSEA_GX = 2;
export const LARGE_DEEPSEA_GY = 3;
export const LARGE_DEEPSEA_ACTION = 4;

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

export const largeDeepSeaDecoder: EnvDecoder<LargeDeepSeaStatic> = {
  envId: 'LargeDeepSea-cont-v0',
  seedGeometry: true,

  initStatic(seed: number): LargeDeepSeaStatic {
    const words = largeDeepSeaMaskWords(seed);
    return { seed, words, mask: largeDeepSeaMask(words) };
  },

  geometry(st: LargeDeepSeaStatic): EnvGeometry {
    return {
      grid: {
        w: LARGE_DEEPSEA_N,
        h: LARGE_DEEPSEA_N,
        walls: new Uint8Array(LARGE_DEEPSEA_N * LARGE_DEEPSEA_N),
        goal: { x: LARGE_DEEPSEA_N - 1, y: LARGE_DEEPSEA_N - 1 },
        actionMask: st.mask,
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[2] = LARGE_DEEPSEA_N - 1;
    pose[3] = LARGE_DEEPSEA_N - 1;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },

  deriveEvents(
    prev: EnvFrame,
    next: EnvFrame,
    st: LargeDeepSeaStatic,
  ): SimEvent[] {
    const row = prev.pose[LARGE_DEEPSEA_ROW] ?? 0;
    if (row !== LARGE_DEEPSEA_LAST_ROW) return NO_EVENTS;
    const action = Math.trunc(next.pose[LARGE_DEEPSEA_ACTION] ?? 0);
    const landed = largeDeepSeaNextCol(st, row, prev.pose[LARGE_DEEPSEA_COL] ?? 0, action);
    const out: SimEvent[] = [];
    if (landed === LARGE_DEEPSEA_N - 1) out.push({ type: 'goal' });
    out.push({ type: 'resurface' });
    return out;
  },
};
