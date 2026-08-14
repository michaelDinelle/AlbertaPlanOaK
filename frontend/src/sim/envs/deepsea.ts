
import { philoxDraw, STREAM_RESET } from '../philox';
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const DEEPSEA_N = 8;
export const DEEPSEA_RIGHT_COST = -0.00125;
export const DEEPSEA_LAST_ROW = DEEPSEA_N - 2;

export interface DeepSeaState {
  r: number;
  c: number;
  lastA: number;
}

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

function state(r: number, c: number, lastA: number): DeepSeaState {
  return { r, c, lastA };
}

export const deepSeaDecoder: EnvDecoder<DeepSeaState, DeepSeaStatic> = {
  envId: 'DeepSea-cont-v0',

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
    if (prev.state.r !== DEEPSEA_LAST_ROW) return NO_EVENTS;
    const action = Math.trunc(next.ctrl[0] ?? 0);
    const landed = deepSeaNextCol(st, prev.state.r, prev.state.c, action);
    const out: SimEvent[] = [];
    if (landed === DEEPSEA_N - 1) out.push({ type: 'goal' });
    out.push({ type: 'resurface' });
    return out;
  },
};
