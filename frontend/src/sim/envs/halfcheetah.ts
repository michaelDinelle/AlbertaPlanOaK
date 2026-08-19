import { cheetahGeometry } from '../fk/cheetah';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const CHEETAH_DT = 0.05;

export const CHEETAH_ROOT_X = 0;

const GEOMETRY: EnvGeometry = Object.freeze({
  cheetah: cheetahGeometry,
});

export const halfCheetahDecoder: EnvDecoder<null> = {
  envId: 'HalfCheetah-v4',
  seedGeometry: false,

  initStatic(): null {
    return null;
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

  deriveEvents(
    prev: EnvFrame,
    next: EnvFrame,
  ): SimEvent[] {
    if (next.done === 0 && prev.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'done' });
    if (prev.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
