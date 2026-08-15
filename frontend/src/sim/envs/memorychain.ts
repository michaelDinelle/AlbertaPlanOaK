
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const MEMORYCHAIN_L = 32;
export const MEMORYCHAIN_END = MEMORYCHAIN_L - 1;

export interface MemoryChainState {
  pos: number;
  cue: number;
  lastA: number;
  len: number;
}

export interface MemoryChainStatic {
  seed: number;
  walls: Uint8Array;
}

function state(pos: number, cue: number, lastA: number): MemoryChainState {
  return { pos, cue, lastA, len: MEMORYCHAIN_L };
}

export const memoryChainDecoder: EnvDecoder<MemoryChainState, MemoryChainStatic> = {
  envId: 'MemoryChain-cont-v0',

  initStatic(seed: number): MemoryChainStatic {
    return { seed, walls: new Uint8Array(MEMORYCHAIN_L) };
  },

  geometry(st: MemoryChainStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: { w: MEMORYCHAIN_L, h: 1, walls: st.walls, goal: { x: MEMORYCHAIN_END, y: 0 } },
    };
  },

  initFrame(): EnvFrame<MemoryChainState> {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = MEMORYCHAIN_END;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(0, 0, 0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<MemoryChainState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    const ctrl = effortView(raw16, 3);
    return {
      step: i,
      ctrl,
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[2] ?? 0, Math.trunc(ctrl[0] ?? 0) & 1),
    };
  },

  deriveEvents(
    prev: EnvFrame<MemoryChainState>,
    next: EnvFrame<MemoryChainState>,
  ): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    const choice = Math.trunc(next.ctrl[0] ?? 0) & 1;
    const out: SimEvent[] = [];
    if (choice === prev.state.cue) out.push({ type: 'goal' });
    out.push({ type: 'cue', value: next.state.cue === 1 ? 1 : 0 });
    out.push({ type: 'teleport' });
    out.push({ type: 'done' });
    return out;
  },
};
