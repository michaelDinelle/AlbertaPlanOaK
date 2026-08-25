import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const SW_ROOT_X = 0;
export const SW_ROOT_Y = 1;
export const SW_YAW = 2;
export const SW_JOINT1 = 3;
export const SW_JOINT2 = 4;
export const SW_LINK_LEN = 1;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const swimmerDecoder: EnvDecoder<null> = {
  envId: 'Swimmer-v5',
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
