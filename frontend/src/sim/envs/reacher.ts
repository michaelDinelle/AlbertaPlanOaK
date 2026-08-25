import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const RE_JOINT0 = 0;
export const RE_JOINT1 = 1;
export const RE_TARGET_X = 2;
export const RE_TARGET_Y = 3;
export const RE_LINK0_LEN = 0.1;
export const RE_LINK1_LEN = 0.11;
export const RE_ARENA_HALF = 0.27;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const reacherDecoder: EnvDecoder<null> = {
  envId: 'Reacher-v5',
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
