// CartPole-cont-v0 decoder — record-v3 pose = [x, xdot, th, thdot, 0, ...].
//
// Authoritative sources:
//   Sources/ClassicControl.metal L138-176   (gw_pole_step)
//   Sources/OakKernel.metal      L2863-2877 (pose = [x, xd, th, thd, 0...])
//   Sources/OakKernel.metal      L2311-2315 (fleet init: deterministic zeros)
//   tools/physics_parity.py      L362-381   (the fp64 oracle)
//
// th is the pole's tilt from vertical, positive to +x; the pole tip sits at
// (x + L sin th, L cos th) with L = 2 * halfLength = 1. Reward is a dense +1
// per surviving step. A fall (|x| > 2.4 or |th| > 0.2094395 rad = 12 deg)
// scores the +1, sets done=1, and resets all four coordinates to zero IN THE
// SAME STEP, so a failing row reads [0,0,0,0] with done=1 — the failure state
// itself is never recorded. Verified on frontend-demo-16env-20260803 seed48.
//
// Note the threshold constants are the fp32 literals the kernel compares
// against (`2.4f`, `0.2094395f`), reproduced here with Math.fround so a
// scene's failure arc lands exactly where the env's does.

import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

// ---------------------------------------------------------------- constants

/** integrator step (gym CartPole Euler) */
export const CP_TAU = 0.02;
/** |x| failure bound */
export const CP_X_LIMIT = Math.fround(2.4);
/** |th| failure bound (12 degrees) */
export const CP_TH_LIMIT = Math.fround(0.2094395);
/** applied force magnitude, sign from the action bit */
export const CP_FORCE_MAG = 10;
/** gym's `length` — "actually half the pole's length" */
export const CP_POLE_HALF_LENGTH = 0.5;
/** drawn pole length (gym's renderer uses 2 * length) */
export const CP_POLE_LENGTH = 1;
/** cart body extents in world units (gym: 50x30 px at 125 px/unit) */
export const CP_CART_WIDTH = 0.4;
export const CP_CART_HEIGHT = 0.24;
/** pole thickness in world units (gym: 10 px) */
export const CP_POLE_WIDTH = 0.08;
/** discrete action count (force {-10, +10}) */
export const CP_ACTIONS = 2;

// ------------------------------------------------------------------- types

/** CartPole has no hidden static state; the constants travel with it. */
export interface CartPoleStatic {
  xLimit: number;
  thLimit: number;
  poleLength: number;
  cartWidth: number;
  cartHeight: number;
}

export interface CartPoleState {
  /** cart position; |x| > 2.4 fails */
  x: number;
  /** cart velocity */
  xDot: number;
  /** pole angle from vertical (rad, wrapped to [-pi, pi)); |th| > 0.2094 fails */
  th: number;
  /** pole angular velocity */
  thDot: number;
  /** recorded ctrl[0] (the discrete action index, 0 or 1); 0 on the init frame */
  action: number;
  /**
   * Applied force: +10 when the action bit is set, else -10. Exactly 0 on the
   * synthesized init frame, where no action has been taken yet (decoding its
   * all-zero ctrl as action 0 would read as a real leftward shove).
   */
  force: number;
  /** |x| / 2.4 and |th| / 0.2094 — how close this frame is to failing, in [0, 1] */
  xMargin: number;
  thMargin: number;
}

// ------------------------------------------------------------- kinematics

/** floats written by `cartPoleTip`: [tipX, tipY] */
export const CARTPOLE_POINT_FLOATS = 2;

/** Pole tip position for a cart at `x` with tilt `th`, axle at y = 0. */
export function cartPoleTip(x: number, th: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(CARTPOLE_POINT_FLOATS);
  o[0] = x + CP_POLE_LENGTH * Math.sin(th);
  o[1] = CP_POLE_LENGTH * Math.cos(th);
  return o;
}

// ---------------------------------------------------------------- decoder

const STATIC: CartPoleStatic = Object.freeze({
  xLimit: CP_X_LIMIT,
  thLimit: CP_TH_LIMIT,
  poleLength: CP_POLE_LENGTH,
  cartWidth: CP_CART_WIDTH,
  cartHeight: CP_CART_HEIGHT,
});

/**
 * Anchors (world x-y, z always 0):
 *   pivot           the cart axle at rest (also the pole hinge)
 *   trackLeft/Right the |x| = 2.4 failure bounds, which are the track ends
 *   failLeft/Right  aliases of the same bounds, for scenes that mark them red
 *   failArcLeft     pole tip at th = -0.2094395 (the tilt failure arc)
 *   failArcRight    pole tip at th = +0.2094395
 *   poleTipRest     pole tip when upright
 *   cartHalfExtent  half-width / half-height of the cart body (an offset, not
 *                   a position)
 */
const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  trackLeft: [-CP_X_LIMIT, 0, 0],
  trackRight: [CP_X_LIMIT, 0, 0],
  failLeft: [-CP_X_LIMIT, 0, 0],
  failRight: [CP_X_LIMIT, 0, 0],
  failArcLeft: [
    -CP_POLE_LENGTH * Math.sin(CP_TH_LIMIT),
    CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT),
    0,
  ],
  failArcRight: [
    CP_POLE_LENGTH * Math.sin(CP_TH_LIMIT),
    CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT),
    0,
  ],
  poleTipRest: [0, CP_POLE_LENGTH, 0],
  cartHalfExtent: [CP_CART_WIDTH / 2, CP_CART_HEIGHT / 2, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ anchors: ANCHORS }),
});

/** `((action & 1) != 0) ? +10 : -10`, the kernel's force decode. */
function forceOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) & 1) !== 0 ? CP_FORCE_MAG : -CP_FORCE_MAG;
}

/** `applied` is false only for the synthesized init frame (no action yet). */
function makeState(pose: Float32Array, ctrl0: number, applied: boolean): CartPoleState {
  const x = pose[0] ?? 0;
  const th = pose[2] ?? 0;
  return {
    x,
    xDot: pose[1] ?? 0,
    th,
    thDot: pose[3] ?? 0,
    action: ctrl0,
    force: applied ? forceOf(ctrl0) : 0,
    xMargin: Math.min(1, Math.abs(x) / CP_X_LIMIT),
    thMargin: Math.min(1, Math.abs(th) / CP_TH_LIMIT),
  };
}

export const cartPoleDecoder: EnvDecoder<CartPoleState, CartPoleStatic> = {
  envId: 'CartPole-cont-v0',
  needsReplay: false,

  initStatic(): CartPoleStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<CartPoleState> {
    // oak_fleet_init: hc_resetData -> the deterministic upright zero state
    const pose = new Float32Array(9);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0, false),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<CartPoleState> {
    const pose = raw16.subarray(7, 16);
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, raw16[0] ?? 0, true),
    };
  },

  deriveEvents(_prev: EnvFrame<CartPoleState>, next: EnvFrame<CartPoleState>): SimEvent[] {
    // done <=> the pole fell or the cart ran off the track. There is no goal
    // event: CartPole's reward is the dense per-step +1, not an achievement.
    // The same step snaps the state back to upright zero, hence the teleport.
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }, { type: 'teleport' }];
  },
};
