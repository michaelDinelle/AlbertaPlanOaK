import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const HU_X = 0;
export const HU_Y = 1;
export const HU_Z = 2;
export const HU_QW = 3;
export const HU_QX = 4;
export const HU_QY = 5;
export const HU_QZ = 6;
export const HU_ABD_Z = 7;
export const HU_ABD_Y = 8;
export const HU_ABD_X = 9;
export const HU_R_HIP_X = 10;
export const HU_R_HIP_Z = 11;
export const HU_R_HIP_Y = 12;
export const HU_R_KNEE = 13;
export const HU_L_HIP_X = 14;
export const HU_L_HIP_Z = 15;
export const HU_L_HIP_Y = 16;
export const HU_L_KNEE = 17;
export const HU_R_SHOULDER1 = 18;
export const HU_R_SHOULDER2 = 19;
export const HU_R_ELBOW = 20;
export const HU_L_SHOULDER1 = 21;
export const HU_L_SHOULDER2 = 22;
export const HU_L_ELBOW = 23;
export const HUMANOID_DT = 0.015;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const humanoidDecoder: EnvDecoder<null> = {
  envId: 'Humanoid-v5',
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
