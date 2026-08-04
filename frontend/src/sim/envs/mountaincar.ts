// MountainCar-cont-v0 decoder — record-v3 pose = [pos, vel, 0.5, 0, ...].
//
// Authoritative sources:
//   Sources/ClassicControl.metal L106-136   (gw_mcar_step)
//   Sources/OakKernel.metal      L2848-2862 (pose = [pos, vel, 0.5, 0...])
//   Sources/OakKernel.metal      L2269-2272 (fleet init: pos = -0.5, vel = 0)
//   tools/physics_parity.py      L348-361   (the fp64 oracle)
//
// Terrain: the gym curve `h(x) = sin(3x) * 0.45 + 0.55` over x in [-1.2, 0.6].
// Its crest sits at x = pi/6 ~ 0.5236 (h = 1.0), just past the goal line at
// x = 0.5; its trough is at x = -pi/6 ~ -0.5236, just left of the respawn at
// x = -0.5.
//
// Success is invisible in the pose: the kernel tests `pos >= 0.5` on the
// post-step state, then respawns to (-0.5, 0) in the SAME step, so a winning
// row reads [-0.5, 0, 0.5] with done=1. Verified on
// frontend-demo-16env-20260803 seed40: every done row is exactly the respawn.
// pose[2] is a constant goal marker the kernel writes every step, not state.

import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

// ---------------------------------------------------------------- constants

export const MC_MIN_POS = -1.2;
export const MC_MAX_POS = 0.6;
/** the goal line; reaching it scores +1 and respawns */
export const MC_GOAL_POS = 0.5;
export const MC_MAX_SPEED = 0.07;
/** respawn / fleet-init position (the kernel's "valley") */
export const MC_START_POS = -0.5;
/** per-action engine force */
export const MC_FORCE = 0.001;
/** gravity coefficient of the terrain slope term */
export const MC_GRAVITY = 0.0025;
/** discrete action count (force {-1, 0, +1}) */
export const MC_ACTIONS = 3;
/** true trough / crest of `mountainCarHeight` inside the position range */
export const MC_TROUGH_X = -Math.PI / 6;
export const MC_CREST_X = Math.PI / 6;

// ------------------------------------------------------------------- types

/** MountainCar has no hidden static state; the constants travel with it. */
export interface MountainCarStatic {
  minPos: number;
  maxPos: number;
  goalPos: number;
  maxSpeed: number;
  startPos: number;
}

export interface MountainCarState {
  /** car position along the track, in [-1.2, 0.6] */
  pos: number;
  /** car velocity, in [-0.07, 0.07] */
  vel: number;
  /** the goal line the kernel stamps into pose[2] (0.5) */
  goal: number;
  /** terrain height under the car, `mountainCarHeight(pos)` */
  height: number;
  /** recorded ctrl[0] (the discrete action index, 0..2); 0 on the init frame */
  action: number;
  /**
   * Applied force direction `(action % 3) - 1`, in {-1, 0, +1}. Exactly 0 on
   * the synthesized init frame, where no action has been taken yet (decoding
   * its all-zero ctrl as action 0 would read as a real leftward force).
   */
  force: number;
}

// ------------------------------------------------------------- kinematics

/** gym terrain: `sin(3x) * 0.45 + 0.55`. */
export function mountainCarHeight(x: number): number {
  return Math.sin(3 * x) * 0.45 + 0.55;
}

/** Terrain slope dh/dx = `3 * cos(3x) * 0.45` — the surface tangent angle. */
export function mountainCarSlope(x: number): number {
  return 1.35 * Math.cos(3 * x);
}

// ---------------------------------------------------------------- decoder

const STATIC: MountainCarStatic = Object.freeze({
  minPos: MC_MIN_POS,
  maxPos: MC_MAX_POS,
  goalPos: MC_GOAL_POS,
  maxSpeed: MC_MAX_SPEED,
  startPos: MC_START_POS,
});

/**
 * Anchors (world x-y, z always 0), all pinned to the terrain curve:
 *   trackLeft   x = -1.2, the wall that zeroes leftward velocity
 *   trackRight  x = 0.6, the right end of the track
 *   goalFlag    x = 0.5, the flag base
 *   successLine same point as goalFlag, for scenes that draw a rule
 *   start       x = -0.5, the respawn / init position
 *   trough      x = -pi/6, the true low point
 *   crest       x = +pi/6, the true high point (h = 1)
 */
const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  trackLeft: [MC_MIN_POS, mountainCarHeight(MC_MIN_POS), 0],
  trackRight: [MC_MAX_POS, mountainCarHeight(MC_MAX_POS), 0],
  goalFlag: [MC_GOAL_POS, mountainCarHeight(MC_GOAL_POS), 0],
  successLine: [MC_GOAL_POS, mountainCarHeight(MC_GOAL_POS), 0],
  start: [MC_START_POS, mountainCarHeight(MC_START_POS), 0],
  trough: [MC_TROUGH_X, mountainCarHeight(MC_TROUGH_X), 0],
  crest: [MC_CREST_X, mountainCarHeight(MC_CREST_X), 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ heightAt: mountainCarHeight, anchors: ANCHORS }),
});

/** `uint(ctrl[0]) % 3 - 1`, the kernel's force decode. */
function forceOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) % MC_ACTIONS) - 1;
}

/** `applied` is false only for the synthesized init frame (no action yet). */
function makeState(pose: Float32Array, ctrl0: number, applied: boolean): MountainCarState {
  const pos = pose[0] ?? 0;
  return {
    pos,
    vel: pose[1] ?? 0,
    goal: pose[2] ?? MC_GOAL_POS,
    height: mountainCarHeight(pos),
    action: ctrl0,
    force: applied ? forceOf(ctrl0) : 0,
  };
}

export const mountainCarDecoder: EnvDecoder<MountainCarState, MountainCarStatic> = {
  envId: 'MountainCar-cont-v0',
  needsReplay: false,

  initStatic(): MountainCarStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<MountainCarState> {
    // oak_fleet_init: gw[0] = -0.5, gw[1] = 0; pose[2] is the goal marker
    const pose = new Float32Array(9);
    pose[0] = MC_START_POS;
    pose[2] = MC_GOAL_POS;
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0, false),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<MountainCarState> {
    const pose = raw16.subarray(7, 16);
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, raw16[0] ?? 0, true),
    };
  },

  deriveEvents(
    _prev: EnvFrame<MountainCarState>,
    next: EnvFrame<MountainCarState>,
  ): SimEvent[] {
    // done <=> summit reached; the same step teleports the car to the respawn.
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
