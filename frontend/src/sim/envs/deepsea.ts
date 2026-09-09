import { philoxDraw, STREAM_RESET } from '../philox';
import { POSE_LEN } from '../../data/oak';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const DEEPSEA_N = 8;
export const DEEPSEA_LAST_ROW = DEEPSEA_N - 2;

export const DEEPSEA_ROW = 0;
export const DEEPSEA_COL = 1;
export const DEEPSEA_ACTION = 4;

export interface DeepSeaStatic {
  seed: number;
  maskLo: number;
  maskHi: number;
  mask: Uint8Array;
}

export function deepSeaMaskWords(seed: number): [number, number] {
  const raw = philoxDraw(seed, STREAM_RESET, 0, 0);
  return [raw[0], raw[1]];
}

export function deepSeaMask(maskLo: number, maskHi: number): Uint8Array {
  const mask = new Uint8Array(DEEPSEA_N * DEEPSEA_N);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = i < 32 ? (maskLo >>> i) & 1 : (maskHi >>> (i - 32)) & 1;
  }
  return mask;
}

export function deepSeaBit(st: DeepSeaStatic, r: number, c: number): number {
  return st.mask[r * DEEPSEA_N + c] ?? 0;
}

export function deepSeaNextCol(st: DeepSeaStatic, r: number, c: number, action: number): number {
  const dc = (action & 1) === deepSeaBit(st, r, c) ? 1 : -1;
  return Math.min(Math.max(c + dc, 0), DEEPSEA_N - 1);
}

export const deepSeaDecoder: EnvDecoder<DeepSeaStatic> = {
  envId: 'DeepSea-cont-v0',
  seedGeometry: true,

  initStatic(seed: number): DeepSeaStatic {
    const [maskLo, maskHi] = deepSeaMaskWords(seed);
    return { seed, maskLo, maskHi, mask: deepSeaMask(maskLo, maskHi) };
  },

  geometry(st: DeepSeaStatic): EnvGeometry {
    return {
      grid: {
        w: DEEPSEA_N,
        h: DEEPSEA_N,
        walls: new Uint8Array(DEEPSEA_N * DEEPSEA_N),
        goal: { x: DEEPSEA_N - 1, y: DEEPSEA_N - 1 },
        actionMask: st.mask,
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[2] = DEEPSEA_N - 1;
    pose[3] = DEEPSEA_N - 1;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },

  deriveEvents(
    prev: EnvFrame,
    next: EnvFrame,
    st: DeepSeaStatic,
  ): SimEvent[] {
    const row = prev.pose[DEEPSEA_ROW] ?? 0;
    if (row !== DEEPSEA_LAST_ROW) return NO_EVENTS;
    const action = Math.trunc(next.pose[DEEPSEA_ACTION] ?? 0);
    const landed = deepSeaNextCol(st, row, prev.pose[DEEPSEA_COL] ?? 0, action);
    const out: SimEvent[] = [];
    if (landed === DEEPSEA_N - 1) out.push({ type: 'goal' });
    out.push({ type: 'resurface' });
    return out;
  },
};
