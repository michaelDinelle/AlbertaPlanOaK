
import { POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const CHAIN_L = 16;
export const CHAIN_END = CHAIN_L - 1;

export const CHAIN_POS = 0;

export interface ChainStatic {
  seed: number;
  walls: Uint8Array;
}

export const chainDecoder: EnvDecoder<ChainStatic> = {
  envId: 'Chain-cont-v0',

  initStatic(seed: number): ChainStatic {
    return { seed, walls: new Uint8Array(CHAIN_L) };
  },

  geometry(st: ChainStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: { w: CHAIN_L, h: 1, walls: st.walls, goal: { x: CHAIN_END, y: 0 } },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = CHAIN_END;
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
