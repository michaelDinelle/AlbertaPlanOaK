import { POSE_LEN } from '../../data/oak';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const DELAYEDCHAIN_L = 32;
export const DELAYEDCHAIN_END = DELAYEDCHAIN_L - 1;

export const DELAYEDCHAIN_POS = 0;
export const DELAYEDCHAIN_ACTION = 2;

export interface DelayedChainStatic {
  seed: number;
  walls: Uint8Array;
}

export const delayedChainDecoder: EnvDecoder<DelayedChainStatic> = {
  envId: 'DelayedChain-cont-v0',
  seedGeometry: false,

  initStatic(seed: number): DelayedChainStatic {
    return { seed, walls: new Uint8Array(DELAYEDCHAIN_L) };
  },

  geometry(st: DelayedChainStatic): EnvGeometry {
    return {
      grid: { w: DELAYEDCHAIN_L, h: 1, walls: st.walls, goal: { x: DELAYEDCHAIN_END, y: 0 } },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = DELAYEDCHAIN_END;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },

  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
