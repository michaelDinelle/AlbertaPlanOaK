
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';
import { OFF_POSE, effortView, POSE_LEN, OFF_DONE } from '../record';


export const PD_DT = 0.05;
export const PD_MAX_SPEED = 8;
export const PD_MAX_TORQUE = 2;
export const PD_G = 10;
export const PD_M = 1;
export const PD_L = 1;


export interface PendulumStatic {
  length: number;
  maxSpeed: number;
  maxTorque: number;
  initTheta: number;
}

export interface PendulumState {
  th: number;
  thDot: number;
  torque: number;
  action: number;
  effort: number;
}


export const PENDULUM_POINT_FLOATS = 2;

export function pendulumTip(th: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(PENDULUM_POINT_FLOATS);
  o[0] = PD_L * Math.sin(th);
  o[1] = -PD_L * Math.cos(th);
  return o;
}

export function wrapPi(x: number): number {
  return x - 2 * Math.PI * Math.floor((x + Math.PI) / (2 * Math.PI));
}


const STATIC: PendulumStatic = Object.freeze({
  length: PD_L,
  maxSpeed: PD_MAX_SPEED,
  maxTorque: PD_MAX_TORQUE,
  initTheta: Math.fround(Math.PI),
});

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

  initStatic(): PendulumStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame<PendulumState> {
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
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: effortView(raw16, 2),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: makeState(pose, raw16[OFF_POSE + 2] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<PendulumState>, next: EnvFrame<PendulumState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'done' }];
  },
};
