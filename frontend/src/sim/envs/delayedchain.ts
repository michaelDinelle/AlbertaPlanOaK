
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const DELAYEDCHAIN_L = 32;
export const DELAYEDCHAIN_END = DELAYEDCHAIN_L - 1;

export interface DelayedChainState {
  pos: number;
  lastA: number;
  len: number;
}

export interface DelayedChainStatic {
  seed: number;
  walls: Uint8Array;
}

function state(pos: number, lastA: number): DelayedChainState {
  return { pos, lastA, len: DELAYEDCHAIN_L };
}

export const delayedChainDecoder: EnvDecoder<DelayedChainState, DelayedChainStatic> = {
  envId: 'DelayedChain-cont-v0',
  needsReplay: false,

  initStatic(seed: number): DelayedChainStatic {
    return { seed, walls: new Uint8Array(DELAYEDCHAIN_L) };
  },

  geometry(st: DelayedChainStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: { w: DELAYEDCHAIN_L, h: 1, walls: st.walls, goal: { x: DELAYEDCHAIN_END, y: 0 } },
    };
  },

  initFrame(): EnvFrame<DelayedChainState> {
    const pose = new Float32Array(POSE_LEN);
    pose[1] = DELAYEDCHAIN_END;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(0, 0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<DelayedChainState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[2] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<DelayedChainState>, next: EnvFrame<DelayedChainState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
