import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const ANT_X = 0;
export const ANT_Y = 1;
export const ANT_Z = 2;
export const ANT_QW = 3;
export const ANT_QX = 4;
export const ANT_QY = 5;
export const ANT_QZ = 6;
export const ANT_HIP0 = 7;
export const ANT_TORSO_R = 0.25;
export const ANT_UPPER_LEN = 0.28;
export const ANT_LOWER_LEN = 0.57;
export const ANT_DT = 0.05;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const antDecoder: EnvDecoder<null> = {
  envId: 'Ant-v5',
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
