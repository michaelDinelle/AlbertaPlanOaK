import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const HP_ROOT_X = 0;
export const HP_ROOT_Z = 1;
export const HP_TORSO_TH = 2;
export const HP_THIGH = 3;
export const HP_LEG = 4;
export const HP_FOOT = 5;
export const HP_TORSO_HALF = 0.2;
export const HP_THIGH_LEN = 0.45;
export const HP_LEG_LEN = 0.5;
export const HP_FOOT_LEN = 0.39;
export const HOPPER_DT = 0.008;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const hopperDecoder: EnvDecoder<null> = {
  envId: 'Hopper-v5',
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
