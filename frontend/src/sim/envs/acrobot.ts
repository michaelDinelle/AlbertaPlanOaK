
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


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


export interface AcrobotStatic {
  l1: number;
  l2: number;
  successHeight: number;
}


const STATIC: AcrobotStatic = Object.freeze({
  l1: AB_L1,
  l2: AB_L2,
  successHeight: AB_SUCCESS_HEIGHT,
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
  classic: Object.freeze({ anchors: ANCHORS }),
});

export const acrobotDecoder: EnvDecoder<AcrobotStatic> = {
  envId: 'Acrobot-cont-v0',
  seedGeometry: false,

  initStatic(): AcrobotStatic {
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


  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0 && prev.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'goal' }, { type: 'done' });
    if (prev.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
