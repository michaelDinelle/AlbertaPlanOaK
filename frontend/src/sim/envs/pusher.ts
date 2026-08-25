import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const PU_SHOULDER_PAN = 0;
export const PU_SHOULDER_LIFT = 1;
export const PU_UPPER_ROLL = 2;
export const PU_ELBOW = 3;
export const PU_FOREARM_ROLL = 4;
export const PU_WRIST_FLEX = 5;
export const PU_WRIST_ROLL = 6;
export const PU_OBJ_X = 7;
export const PU_OBJ_Y = 8;
export const PU_GOAL_X = 9;
export const PU_GOAL_Y = 10;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const pusherDecoder: EnvDecoder<null> = {
  envId: 'Pusher-v5',
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
