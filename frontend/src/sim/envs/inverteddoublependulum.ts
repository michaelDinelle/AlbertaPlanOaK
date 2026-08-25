import { POSE_LEN } from '../record';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const IDP_CART_X = 0;
export const IDP_HINGE1 = 1;
export const IDP_HINGE2 = 2;
export const IDP_LINK_LEN = 0.6;
export const IDP_RAIL_HALF = 1;

const GEOMETRY: EnvGeometry = Object.freeze({});

export const invertedDoublePendulumDecoder: EnvDecoder<null> = {
  envId: 'InvertedDoublePendulum-v5',
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
