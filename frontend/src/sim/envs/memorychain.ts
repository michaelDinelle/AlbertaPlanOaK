// MemoryChain-cont-v0 decoder (bsuite Memory Chain, L = 32).
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_mchain_step L498, gw_mchain_obs L488
//   tools/gridworld_ref.py       mchain_step L445
//   Sources/OakKernel.metal      pose packing L2758-2772, init L2320-2325
//
// A 32-cell corridor. The cue (0 = left, 1 = right) is observable ONLY at
// pos 0; the agent must carry it to the far end, where its action IS the
// choice: +1 iff (action & 1) == cue, done=1 EITHER WAY, then respawn at
// pos 0 with a freshly drawn cue (Philox stream 3). Away from the end,
// action 1 walks pos+1 and action 0 walks pos-1 (clamped at 0).
//
// So `done` marks every choice — right or wrong — and `goal` fires only on a
// correct one. The cue that was in force for a choice is the PREVIOUS frame's
// cue: the recorded cue is the post-step (freshly drawn) one.
//
// pose = [pos, 31, cue, 0, 0, 0, 0, 0, 0].

import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const MEMORYCHAIN_L = 32;
/** the choice cell */
export const MEMORYCHAIN_END = MEMORYCHAIN_L - 1;

export interface MemoryChainState {
  pos: number;
  /** 0 = left, 1 = right; visible to the agent only at pos 0 */
  cue: number;
  /** action & 1 of the step that produced this frame */
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
  needsReplay: false,

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
    const ctrl = raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN);
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
    // the respawn redraws the cue every time, whether or not its value changed
    out.push({ type: 'cue', value: next.state.cue === 1 ? 1 : 0 });
    out.push({ type: 'teleport' });
    out.push({ type: 'done' });
    return out;
  },
};
