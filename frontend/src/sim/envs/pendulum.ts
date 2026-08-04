// Pendulum-cont-v0 decoder — record-v3 pose = [th, thdot, torque, 0, ...].
//
// Authoritative sources:
//   Sources/ClassicControl.metal L178-201   (gw_pend_step)
//   Sources/OakKernel.metal      L2878-2890 (pose = [th, thd, clamped torque])
//   Sources/OakKernel.metal      L2273-2276 (fleet init: th = pi, thdot = 0)
//   tools/physics_parity.py      L382-391   (the fp64 oracle)
//
// ANGLE CONVENTION — derived from the dynamics, not from the source comments.
// The kernel integrates
//     thdot' = thdot + (15 sin(th + pi) + 3u) * dt = thdot - 15 sin(th) * dt
// so th = 0 is the STABLE equilibrium (bob hanging straight down) and th = +-pi
// is the unstable inverted position. The fleet init is th = pi, i.e. the bob
// starts balanced at the TOP and falls; the shaped cost `th^2 + 0.1 thdot^2 +
// 0.001 u^2` is minimised at th = 0, i.e. at rest hanging down. (The Metal
// comments say "hang down" for the init and "angle from upright" for the cost;
// both are mislabelled with respect to this sign convention.) Confirmed
// empirically on frontend-demo-16env-20260803 seed56: the stream leaves pi
// within a few steps and then spends 99.6% of 100k steps with |th| < 0.5.
//
// Render the bob at (sin th, -cos th): th = 0 points down, th = pi points up.
//
// Pendulum is the one classic env that is truly continuing: it never sets
// done, never resets, and never teleports. th is wrapped into [-pi, pi) every
// step, which is continuous on the circle — a wrap is NOT a teleport, and
// scenes must interpolate th on the circle (shortest arc), not linearly.

import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

// ---------------------------------------------------------------- constants

/** integrator step */
export const PD_DT = 0.05;
/** |thdot| clip */
export const PD_MAX_SPEED = 8;
/** |torque| clip; ctrl[0] in [-1, 1] is scaled by 2 */
export const PD_MAX_TORQUE = 2;
/** gravity / mass / length (gym Pendulum) */
export const PD_G = 10;
export const PD_M = 1;
export const PD_L = 1;

// ------------------------------------------------------------------- types

/** Pendulum has no hidden static state; the constants travel with it. */
export interface PendulumStatic {
  length: number;
  maxSpeed: number;
  maxTorque: number;
  /** the angle the fleet init starts at (inverted) */
  initTheta: number;
}

export interface PendulumState {
  /** angle from the DOWNWARD vertical, wrapped to [-pi, pi) */
  th: number;
  /** angular velocity, clipped to +-8 */
  thDot: number;
  /** applied torque as the kernel clamped it, in [-2, 2] */
  torque: number;
  /** recorded ctrl[0], the raw Box action in [-1, 1] */
  action: number;
  /** |torque| / 2, in [0, 1] — how hard the actuator is pushing */
  effort: number;
}

// ------------------------------------------------------------- kinematics

/** floats written by `pendulumTip`: [tipX, tipY] */
export const PENDULUM_POINT_FLOATS = 2;

/** Bob position for angle `th`, pivot at the origin. th = 0 hangs down. */
export function pendulumTip(th: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(PENDULUM_POINT_FLOATS);
  o[0] = PD_L * Math.sin(th);
  o[1] = -PD_L * Math.cos(th);
  return o;
}

/** `gw_wrap_pi` — floor-mod into [-pi, pi), the kernel's own formula. */
export function wrapPi(x: number): number {
  return x - 2 * Math.PI * Math.floor((x + Math.PI) / (2 * Math.PI));
}

// ---------------------------------------------------------------- decoder

const STATIC: PendulumStatic = Object.freeze({
  length: PD_L,
  maxSpeed: PD_MAX_SPEED,
  maxTorque: PD_MAX_TORQUE,
  initTheta: Math.fround(Math.PI),
});

/**
 * Anchors (world x-y, z always 0):
 *   pivot        the hinge
 *   restTip      bob at th = 0 (hanging down; the cost minimum)
 *   invertedTip  bob at th = +-pi (the fleet-init position)
 */
const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  restTip: [0, -PD_L, 0],
  invertedTip: [0, PD_L, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ anchors: ANCHORS }),
});

function makeState(pose: Float32Array, ctrl0: number): PendulumState {
  const torque = pose[2] ?? 0;
  return {
    th: pose[0] ?? 0,
    thDot: pose[1] ?? 0,
    torque,
    action: ctrl0,
    effort: Math.min(1, Math.abs(torque) / PD_MAX_TORQUE),
  };
}

export const pendulumDecoder: EnvDecoder<PendulumState, PendulumStatic> = {
  envId: 'Pendulum-cont-v0',
  needsReplay: false,

  initStatic(): PendulumStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<PendulumState> {
    // oak_fleet_init: gw[0] = M_PI_F (inverted), gw[1] = 0; no torque yet.
    // Writing Math.PI into a Float32Array rounds to the same float32 M_PI_F is.
    const pose = new Float32Array(9);
    pose[0] = Math.PI;
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<PendulumState> {
    const pose = raw16.subarray(7, 16);
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, raw16[0] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<PendulumState>, next: EnvFrame<PendulumState>): SimEvent[] {
    // The env never sets done and never resets, so this is always empty in
    // practice; the branch is defensive against a future recorder change.
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }];
  },
};
