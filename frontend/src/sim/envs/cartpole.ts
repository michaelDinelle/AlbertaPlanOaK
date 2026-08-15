
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const CP_TAU = 0.02;
export const CP_X_LIMIT = Math.fround(2.4);
export const CP_TH_LIMIT = Math.fround(0.2094395);
export const CP_FORCE_MAG = 10;
export const CP_POLE_LENGTH = 1;
export const CP_CART_WIDTH = 0.4;
export const CP_CART_HEIGHT = 0.24;
export const CP_POLE_WIDTH = 0.08;
export const CARTPOLE_X = 0;
export const CARTPOLE_XDOT = 1;
export const CARTPOLE_TH = 2;
export const CARTPOLE_THDOT = 3;
export const CARTPOLE_ACTION = 4;

export function cartPoleForce(action: number): number {
  return (Math.trunc(action) & 1) === 1 ? CP_FORCE_MAG : -CP_FORCE_MAG;
}


export interface CartPoleStatic {
  xLimit: number;
  thLimit: number;
  poleLength: number;
  cartWidth: number;
  cartHeight: number;
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

export const cartPoleDecoder: EnvDecoder<CartPoleStatic> = {
  envId: 'CartPole-cont-v0',

  initStatic(): CartPoleStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(9);
    return {
      step: -1,
      done: 0,
      pose,
    };
  },


  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }, { type: 'teleport' }];
  },
};
