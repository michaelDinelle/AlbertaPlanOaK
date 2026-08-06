
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const AB_DT = 0.2;
export const AB_L1 = 1;
export const AB_L2 = 1;
export const AB_W1MAX = Math.fround(12.566371);
export const AB_W2MAX = Math.fround(28.274334);
export const AB_SUCCESS_HEIGHT = 1;
export const AB_ACTIONS = 3;


export interface AcrobotStatic {
  l1: number;
  l2: number;
  successHeight: number;
  w1Max: number;
  w2Max: number;
}

export interface AcrobotState {
  th1: number;
  th2: number;
  w1: number;
  w2: number;
  tipHeight: number;
  action: number;
  torque: number;
}


export const ACROBOT_POINT_FLOATS = 4;

export function acrobotPoints(th1: number, th2: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(ACROBOT_POINT_FLOATS);
  const ex = AB_L1 * Math.sin(th1);
  const ey = -AB_L1 * Math.cos(th1);
  o[0] = ex;
  o[1] = ey;
  o[2] = ex + AB_L2 * Math.sin(th1 + th2);
  o[3] = ey - AB_L2 * Math.cos(th1 + th2);
  return o;
}

export function acrobotTipHeight(th1: number, th2: number): number {
  return -Math.cos(th1) - Math.cos(th1 + th2);
}


const STATIC: AcrobotStatic = Object.freeze({
  l1: AB_L1,
  l2: AB_L2,
  successHeight: AB_SUCCESS_HEIGHT,
  w1Max: AB_W1MAX,
  w2Max: AB_W2MAX,
});

const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  restElbow: [0, -AB_L1, 0],
  restTip: [0, -(AB_L1 + AB_L2), 0],
  successLine: [0, AB_SUCCESS_HEIGHT, 0],
  boundsMin: [-2.2, -2.2, 0],
  boundsMax: [2.2, 2.2, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ anchors: ANCHORS }),
});

function torqueOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) % AB_ACTIONS) - 1;
}

function makeState(pose: Float32Array, ctrl0: number, applied: boolean): AcrobotState {
  const th1 = pose[0] ?? 0;
  const th2 = pose[1] ?? 0;
  return {
    th1,
    th2,
    w1: pose[2] ?? 0,
    w2: pose[3] ?? 0,
    tipHeight: acrobotTipHeight(th1, th2),
    action: ctrl0,
    torque: applied ? torqueOf(ctrl0) : 0,
  };
}

export const acrobotDecoder: EnvDecoder<AcrobotState, AcrobotStatic> = {
  envId: 'Acrobot-cont-v0',
  needsReplay: false,

  initStatic(): AcrobotStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<AcrobotState> {
    const pose = new Float32Array(9);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0, false),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<AcrobotState> {
    const pose = raw16.subarray(7, 16);
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, raw16[0] ?? 0, true),
    };
  },

  deriveEvents(_prev: EnvFrame<AcrobotState>, next: EnvFrame<AcrobotState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
