import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

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
export const CARTPOLE_ACTION = 4;

export function cartPoleForce(action: number): number {
  return (Math.trunc(action) & 1) === 1 ? CP_FORCE_MAG : -CP_FORCE_MAG;
}

export function cartPoleTip(x: number, th: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(2);
  o[0] = x + CP_POLE_LENGTH * Math.sin(th);
  o[1] = CP_POLE_LENGTH * Math.cos(th);
  return o;
}

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
  classic: Object.freeze({ anchors: ANCHORS }),
});

export const cartPoleDecoder: EnvDecoder<null> = {
  envId: 'CartPole-cont-v0',
  seedGeometry: false,

  initStatic(): null {
    return null;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    return {
      step: -1,
      done: 0,
      pose,
    };
  },

  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0 && prev.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'done' });
    if (prev.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
