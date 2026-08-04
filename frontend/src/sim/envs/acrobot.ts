// Acrobot-cont-v0 decoder — record-v3 pose = [th1, th2, w1, w2, 0, 0, 0, 0, 0].
//
// Authoritative sources:
//   Sources/ClassicControl.metal L36-104   (gw_acro_step, gw_acro_dsdt)
//   Sources/OakKernel.metal      L2832-2847 (pose packing, info[3] = tip height)
//   Sources/OakKernel.metal      L2311-2315 (fleet init: deterministic zeros)
//   tools/physics_parity.py      L317-347  (the fp64 oracle)
//
// Angle convention (from the dynamics, not from a comment): th1 is the first
// link's angle from the DOWNWARD vertical, th2 the second link's angle
// relative to the first. The env's success quantity is
//     tipHeight = -cos(th1) - cos(th1 + th2)
// which is exactly the y of the free end with links of unit length placed at
//     elbow = (sin th1, -cos th1),  tip = elbow + (sin(th1+th2), -cos(th1+th2))
// so `acrobotPoints` and the reward condition agree by construction.
//
// Success is invisible in the pose: the kernel checks `tipHeight > 1` on the
// post-integration state, then resets all four coordinates to zero in the SAME
// step. The recorded row is therefore the hang state with done=1 — the only
// evidence of a success is the done flag. Verified on
// frontend-demo-16env-20260803 seed32: every done row is [0,0,0,0] and the row
// before it is still below the threshold.

import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

// ---------------------------------------------------------------- constants

/** integrator step (gym acrobot RK4) */
export const AB_DT = 0.2;
/** link lengths (l1 = l2 = 1) */
export const AB_L1 = 1;
export const AB_L2 = 1;
/** velocity clips: |w1| <= 4pi, |w2| <= 9pi */
export const AB_W1MAX = Math.fround(12.566371);
export const AB_W2MAX = Math.fround(28.274334);
/** success threshold on `tipHeight` (free end one link above the base) */
export const AB_SUCCESS_HEIGHT = 1;
/** discrete action count (torque {-1, 0, +1}) */
export const AB_ACTIONS = 3;

// ------------------------------------------------------------------- types

/** Acrobot has no hidden static state; the constants travel with it. */
export interface AcrobotStatic {
  l1: number;
  l2: number;
  successHeight: number;
  w1Max: number;
  w2Max: number;
}

export interface AcrobotState {
  /** first link angle from the downward vertical (rad, wrapped to [-pi, pi)) */
  th1: number;
  /** second link angle relative to the first (rad, wrapped) */
  th2: number;
  /** dth1/dt, clipped to +-4pi */
  w1: number;
  /** dth2/dt, clipped to +-9pi */
  w2: number;
  /** `-cos(th1) - cos(th1+th2)`: the free end's height, and the reward test */
  tipHeight: number;
  /** recorded ctrl[0] (the discrete action index, 0..2); 0 on the init frame */
  action: number;
  /**
   * Applied torque `(action % 3) - 1`, in {-1, 0, +1}. Exactly 0 on the
   * synthesized init frame, where no action has been taken yet (decoding its
   * all-zero ctrl as action 0 would read as a real -1 torque).
   */
  torque: number;
}

// ------------------------------------------------------------- kinematics

/** floats written by `acrobotPoints`: [elbowX, elbowY, tipX, tipY] */
export const ACROBOT_POINT_FLOATS = 4;

/**
 * Planar joint positions with the base pivot at the origin.
 * out = [elbowX, elbowY, tipX, tipY]; `out[3]` equals `tipHeight`.
 */
export function acrobotPoints(th1: number, th2: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(ACROBOT_POINT_FLOATS);
  const ex = AB_L1 * Math.sin(th1);
  const ey = -AB_L1 * Math.cos(th1);
  o[0] = ex;
  o[1] = ey;
  o[2] = ex + AB_L2 * Math.sin(th1 + th2);
  o[3] = ey - AB_L2 * Math.cos(th1 + th2);
  return o;
}

/** The env's own success quantity (kernel `info[3]`). */
export function acrobotTipHeight(th1: number, th2: number): number {
  return -Math.cos(th1) - Math.cos(th1 + th2);
}

// ---------------------------------------------------------------- decoder

const STATIC: AcrobotStatic = Object.freeze({
  l1: AB_L1,
  l2: AB_L2,
  successHeight: AB_SUCCESS_HEIGHT,
  w1Max: AB_W1MAX,
  w2Max: AB_W2MAX,
});

/**
 * Anchors (world x-y, z always 0):
 *   pivot        base hinge
 *   restElbow    elbow at the hang state (th1 = 0)
 *   restTip      free end at the hang state
 *   successLine  the `tipHeight > 1` threshold, drawn as a horizontal rule
 *   boundsMin/Max  a square view box that contains the full reach (2.2, gym's)
 */
const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  pivot: [0, 0, 0],
  restElbow: [0, -AB_L1, 0],
  restTip: [0, -(AB_L1 + AB_L2), 0],
  successLine: [0, AB_SUCCESS_HEIGHT, 0],
  boundsMin: [-2.2, -2.2, 0],
  boundsMax: [2.2, 2.2, 0],
});

const GEOMETRY: EnvGeometry = Object.freeze({
  kind: 'classic' as const,
  classic: Object.freeze({ anchors: ANCHORS }),
});

/** `uint(ctrl[0]) % 3 - 1`, the kernel's torque decode. */
function torqueOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) % AB_ACTIONS) - 1;
}

/** `applied` is false only for the synthesized init frame (no action yet). */
function makeState(pose: Float32Array, ctrl0: number, applied: boolean): AcrobotState {
  const th1 = pose[0] ?? 0;
  const th2 = pose[1] ?? 0;
  return {
    th1,
    th2,
    w1: pose[2] ?? 0,
    w2: pose[3] ?? 0,
    tipHeight: acrobotTipHeight(th1, th2),
    action: ctrl0,
    torque: applied ? torqueOf(ctrl0) : 0,
  };
}

export const acrobotDecoder: EnvDecoder<AcrobotState, AcrobotStatic> = {
  envId: 'Acrobot-cont-v0',
  needsReplay: false,

  initStatic(): AcrobotStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<AcrobotState> {
    // oak_fleet_init: hc_resetData -> the deterministic zero (hang) state
    const pose = new Float32Array(9);
    return {
      step: -1,
      ctrl: new Float32Array(6),
      done: 0,
      pose,
      state: makeState(pose, 0, false),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<AcrobotState> {
    const pose = raw16.subarray(7, 16);
    return {
      step: i,
      ctrl: raw16.subarray(0, 6),
      done: raw16[6] ?? 0,
      pose,
      state: makeState(pose, raw16[0] ?? 0, true),
    };
  },

  deriveEvents(_prev: EnvFrame<AcrobotState>, next: EnvFrame<AcrobotState>): SimEvent[] {
    // done <=> swing-up success; the same step snaps the state back to hang,
    // so the visual jump is part of the same transition.
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
