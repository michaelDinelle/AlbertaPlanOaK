
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';
import { OFF_POSE, effortView, POSE_LEN, OFF_DONE } from '../record';


export const CP_TAU = 0.02;
export const CP_X_LIMIT = Math.fround(2.4);
export const CP_TH_LIMIT = Math.fround(0.2094395);
export const CP_FORCE_MAG = 10;
export const CP_POLE_HALF_LENGTH = 0.5;
export const CP_POLE_LENGTH = 1;
export const CP_CART_WIDTH = 0.4;
export const CP_CART_HEIGHT = 0.24;
export const CP_POLE_WIDTH = 0.08;
export const CP_ACTIONS = 2;


export interface CartPoleStatic {
  xLimit: number;
  thLimit: number;
  poleLength: number;
  cartWidth: number;
  cartHeight: number;
}

export interface CartPoleState {
  x: number;
  xDot: number;
  th: number;
  thDot: number;
  action: number;
  force: number;
  xMargin: number;
  thMargin: number;
}


export const CARTPOLE_POINT_FLOATS = 2;

export function cartPoleTip(x: number, th: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(CARTPOLE_POINT_FLOATS);
  o[0] = x + CP_POLE_LENGTH * Math.sin(th);
  o[1] = CP_POLE_LENGTH * Math.cos(th);
  return o;
}


const STATIC: CartPoleStatic = Object.freeze({
  xLimit: CP_X_LIMIT,
  thLimit: CP_TH_LIMIT,
  poleLength: CP_POLE_LENGTH,
  cartWidth: CP_CART_WIDTH,
  cartHeight: CP_CART_HEIGHT,
});

const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  trackLeft: [-CP_X_LIMIT, 0, 0],
  trackRight: [CP_X_LIMIT, 0, 0],
  failLeft: [-CP_X_LIMIT, 0, 0],
  failRight: [CP_X_LIMIT, 0, 0],
  failArcLeft: [
    -CP_POLE_LENGTH * Math.sin(CP_TH_LIMIT),
    CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT),
    0,
  ],
  failArcRight: [
    CP_POLE_LENGTH * Math.sin(CP_TH_LIMIT),
    CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT),
    0,
  ],
  poleTipRest: [0, CP_POLE_LENGTH, 0],
  cartHalfExtent: [CP_CART_WIDTH / 2, CP_CART_HEIGHT / 2, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ anchors: ANCHORS }),
});

function forceOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) & 1) !== 0 ? CP_FORCE_MAG : -CP_FORCE_MAG;
}

function makeState(pose: Float32Array, ctrl0: number, applied: boolean): CartPoleState {
  const x = pose[0] ?? 0;
  const th = pose[2] ?? 0;
  return {
    x,
    xDot: pose[1] ?? 0,
    th,
    thDot: pose[3] ?? 0,
    action: ctrl0,
    force: applied ? forceOf(ctrl0) : 0,
    xMargin: Math.min(1, Math.abs(x) / CP_X_LIMIT),
    thMargin: Math.min(1, Math.abs(th) / CP_TH_LIMIT),
  };
}

export const cartPoleDecoder: EnvDecoder<CartPoleState, CartPoleStatic> = {
  envId: 'CartPole-cont-v0',

  initStatic(): CartPoleStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<CartPoleState> {
    const pose = new Float32Array(9);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0, false),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<CartPoleState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: effortView(raw16, 4),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: makeState(pose, raw16[OFF_POSE + 4] ?? 0, true),
    };
  },

  deriveEvents(_prev: EnvFrame<CartPoleState>, next: EnvFrame<CartPoleState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }, { type: 'teleport' }];
  },
};
