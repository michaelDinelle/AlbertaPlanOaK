// DelayedChain-cont-v0 decoder (bare corridor, L = 32).
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_dchain_step L867
//   tools/gridworld_ref.py       dchain_step L745
//   Sources/OakKernel.metal      pose packing L2810-2822, init L2320-2325
//
// The cleanest long credit chain: action 1 walks pos+1, action 0 walks pos-1
// (clamped at 0), and the ONLY reward is +1 for reaching pos 31, which sets
// done=1 and wraps to pos 0. pos 31 is therefore never observed.
//
// pose = [pos, 31, lastA, 0, 0, 0, 0, 0, 0].

import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const DELAYEDCHAIN_L = 32;
/** the paying cell — reached and immediately wrapped away from */
export const DELAYEDCHAIN_END = DELAYEDCHAIN_L - 1;

export interface DelayedChainState {
  pos: number;
  /** action & 1 of the step that produced this frame */
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
