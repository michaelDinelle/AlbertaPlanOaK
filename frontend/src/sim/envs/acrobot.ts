import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const AB_L1 = 1;
export const AB_L2 = 1;
export const AB_SUCCESS_HEIGHT = 1;
export const AB_ACTIONS = 3;

export const ACROBOT_TH1 = 0;
export const ACROBOT_TH2 = 1;
export const ACROBOT_W1 = 2;
export const ACROBOT_W2 = 3;
export const ACROBOT_ACTION = 4;

export function acrobotTorque(action: number): number {
  return ((Math.trunc(action) >>> 0) % AB_ACTIONS) - 1;
}



const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  restElbow: [0, -AB_L1, 0],
  restTip: [0, -(AB_L1 + AB_L2), 0],
  successLine: [0, AB_SUCCESS_HEIGHT, 0],
  boundsMin: [-2.2, -2.2, 0],
  boundsMax: [2.2, 2.2, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  classic: Object.freeze({ anchors: ANCHORS }),
});

export const acrobotDecoder: EnvDecoder = {
  geometry(): EnvGeometry {
    return GEOMETRY;
  },


  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'goal' }, { type: 'done' });
    if (next.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
