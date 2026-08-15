
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const CHAIN_L = 16;
export const CHAIN_END = CHAIN_L - 1;

export interface ChainState {
  pos: number;
  len: number;
}

export interface ChainStatic {
  seed: number;
  walls: Uint8Array;
}

function state(pos: number): ChainState {
  return { pos, len: CHAIN_L };
}

export const chainDecoder: EnvDecoder<ChainState, ChainStatic> = {
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

  initFrame(): EnvFrame<ChainState> {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = CHAIN_END;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<ChainState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: effortView(raw16, -1),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<ChainState>, next: EnvFrame<ChainState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
