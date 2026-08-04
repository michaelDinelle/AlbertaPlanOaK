// The Long Night Road — canonical poses.
//
// Five states that between them exercise every accuracy claim this scene makes: the
// deterministic fleet init, the long dark middle of the walk, the last cell the agent
// ever observes, the wrap itself, and a step of the walk back out. Three of them are
// LIFTED FROM THE REAL RECORDING (run frontend-demo-16env-20260803, DelayedChain seed
// 120, steps 473–478) — the frames around the lane's first pay-off, so the fixtures the
// tests run on are literally the stream the harness plays.
//
// The wrap fixture is the one that matters: `dchain_step` pays +1 for ARRIVING at pos 31
// and wraps to pos 0 in the same step, so the frame carrying the goal event already says
// pos 0. Cell 31 is never observed — which is why the celebration is nailed to the goal
// cell rather than to the agent's own (the opposite of FourRooms' rule, for the opposite
// reason), and why this scene hard-cuts the wisp instead of gliding it home.

import type { SceneFixture } from '@kit/types';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L, type DelayedChainState } from '@sim/envs/delayedchain';
import { delayedChainPose } from './manifest';

/** ctrl[0] is the discrete action index (0 = step back, 1 = step on). */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(pos: number, lastA: number): DelayedChainState {
  return { pos, lastA, len: DELAYEDCHAIN_L };
}

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: pos 0, lastA 0 — the milestone, thirty-one cells of nothing ahead
    name: 'road-start',
    pose: delayedChainPose(0, 0),
    ctrl: ctrl(1),
    derived: derived(0, 0),
    step: 0,
  },
  {
    // the long dark middle: no reward here, no reward anywhere near here
    name: 'mid-moor',
    pose: delayedChainPose(15, 1),
    ctrl: ctrl(1),
    derived: derived(15, 1),
    step: 3140,
  },
  {
    // seed 120, step 474: a step BACKWARD one cell short of the end — the walk really
    // does dither on the doorstep, and the aurora is overhead for all of it
    name: 'aurora-doorstep',
    pose: delayedChainPose(29, 0),
    ctrl: ctrl(0),
    derived: derived(29, 0),
    step: 474,
  },
  {
    // seed 120, step 475: pos 30, the last cell the agent ever observes
    name: 'last-observed-cell',
    pose: delayedChainPose(30, 1),
    ctrl: ctrl(1),
    derived: derived(30, 1),
    step: 475,
  },
  {
    // seed 120, step 476: the pay-off. +1 at 31, done, and the pose ALREADY reads pos 0
    name: 'the-wrap',
    pose: delayedChainPose(0, 1),
    ctrl: ctrl(1),
    derived: derived(0, 1),
    step: 476,
  },
];

export const DELAYEDCHAIN_GOAL_CELL = DELAYEDCHAIN_END;

export default fixtures;
