// Chain-cont-v0 decoder (bsuite chain, L = 16).
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_chain_step L314
//   tools/gridworld_ref.py       chain_step L288
//   Sources/OakKernel.metal      pose packing L2712-2724, init L2320-2325
//
// A 16-cell corridor: action 1 walks pos+1, action 0 walks pos-1 (clamped at
// 0). Reaching pos 15 pays the delayed +1, sets done=1 and WRAPS to pos 0 —
// so pos 15 is never observed in the record and every `done` frame is both
// the reward and a teleport back to the start.
//
// pose = [pos, 15, 0, 0, 0, 0, 0, 0, 0].

import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const CHAIN_L = 16;
/** the paying cell — reached and immediately wrapped away from */
export const CHAIN_END = CHAIN_L - 1;

export interface ChainState {
  pos: number;
  /** corridor length, so a scene never has to import the constant */
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
  needsReplay: false,

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
      ctrl: raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN),
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
