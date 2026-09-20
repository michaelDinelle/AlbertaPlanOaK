import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const PD_MAX_SPEED = 8;
export const PD_MAX_TORQUE = 2;
export const PD_L = 1;

export const PENDULUM_TH = 0;
export const PENDULUM_THDOT = 1;
export const PENDULUM_TORQUE = 2;


export function wrapPi(x: number): number {
  return x - 2 * Math.PI * Math.floor((x + Math.PI) / (2 * Math.PI));
}


const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  restTip: [0, -PD_L, 0],
  invertedTip: [0, PD_L, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  classic: Object.freeze({ anchors: ANCHORS }),
});

export const pendulumDecoder: EnvDecoder = {
  geometry(): EnvGeometry {
    return GEOMETRY;
  },


  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }];
  },
};
