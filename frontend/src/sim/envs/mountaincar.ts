
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const MC_MIN_POS = -1.2;
export const MC_MAX_POS = 0.6;
export const MC_GOAL_POS = 0.5;
export const MC_MAX_SPEED = 0.07;
export const MC_START_POS = -0.5;
export const MC_FORCE = 0.001;
export const MC_GRAVITY = 0.0025;
export const MC_ACTIONS = 3;
export const MC_TROUGH_X = -Math.PI / 6;
export const MC_CREST_X = Math.PI / 6;


export interface MountainCarStatic {
  minPos: number;
  maxPos: number;
  goalPos: number;
  maxSpeed: number;
  startPos: number;
}

export interface MountainCarState {
  pos: number;
  vel: number;
  goal: number;
  height: number;
  action: number;
  force: number;
}


export function mountainCarHeight(x: number): number {
  return Math.sin(3 * x) * 0.45 + 0.55;
}

export function mountainCarSlope(x: number): number {
  return 1.35 * Math.cos(3 * x);
}


const STATIC: MountainCarStatic = Object.freeze({
  minPos: MC_MIN_POS,
  maxPos: MC_MAX_POS,
  goalPos: MC_GOAL_POS,
  maxSpeed: MC_MAX_SPEED,
  startPos: MC_START_POS,
});

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

function forceOf(ctrl0: number): number {
  return ((Math.trunc(ctrl0) >>> 0) % MC_ACTIONS) - 1;
}

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
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
