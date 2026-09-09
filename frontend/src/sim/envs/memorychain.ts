import { POSE_LEN } from '../../data/oak';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const MEMORYCHAIN_L = 32;
export const MEMORYCHAIN_END = MEMORYCHAIN_L - 1;

export const MEMORYCHAIN_POS = 0;
export const MEMORYCHAIN_CUE = 2;
export const MEMORYCHAIN_ACTION = 3;

export interface MemoryChainStatic {
  seed: number;
  walls: Uint8Array;
}

export const memoryChainDecoder: EnvDecoder<MemoryChainStatic> = {
  envId: 'MemoryChain-cont-v0',
  seedGeometry: false,

  initStatic(seed: number): MemoryChainStatic {
    return { seed, walls: new Uint8Array(MEMORYCHAIN_L) };
  },

  geometry(st: MemoryChainStatic): EnvGeometry {
    return {
      grid: { w: MEMORYCHAIN_L, h: 1, walls: st.walls, goal: { x: MEMORYCHAIN_END, y: 0 } },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = MEMORYCHAIN_END;
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
    if (next.done === 0) return NO_EVENTS;
    const choice = Math.trunc(next.pose[MEMORYCHAIN_ACTION] ?? 0) & 1;
    const out: SimEvent[] = [];
    if (choice === (prev.pose[MEMORYCHAIN_CUE] ?? 0)) out.push({ type: 'goal' });
    out.push({ type: 'cue', value: (next.pose[MEMORYCHAIN_CUE] ?? 0) === 1 ? 1 : 0 });
    out.push({ type: 'teleport' });
    out.push({ type: 'done' });
    return out;
  },
};
