
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '../types';
import { NO_EVENTS } from '../events';


export const MC_MIN_POS = -1.2;
export const MC_MAX_POS = 0.6;
export const MC_GOAL_POS = 0.5;
export const MC_MAX_SPEED = 0.07;
export const MC_START_POS = -0.5;
export const MC_ACTIONS = 3;

export const MOUNTAINCAR_POS = 0;
export const MOUNTAINCAR_VEL = 1;
export const MOUNTAINCAR_GOAL = 2;
export const MOUNTAINCAR_ACTION = 3;

export function mountainCarForce(action: number): number {
  return ((Math.trunc(action) >>> 0) % MC_ACTIONS) - 1;
}
export const MC_TROUGH_X = -Math.PI / 6;
export const MC_CREST_X = Math.PI / 6;


export interface MountainCarStatic {
  minPos: number;
  maxPos: number;
  goalPos: number;
  maxSpeed: number;
  startPos: number;
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
  classic: Object.freeze({ heightAt: mountainCarHeight, anchors: ANCHORS }),
});

export const mountainCarDecoder: EnvDecoder<MountainCarStatic> = {
  envId: 'MountainCar-cont-v0',
  seedGeometry: false,

  initStatic(): MountainCarStatic {
    return STATIC;
  },

  geometry(): EnvGeometry {
    return GEOMETRY;
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(9);
    pose[0] = MC_START_POS;
    pose[2] = MC_GOAL_POS;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },


  deriveEvents(
    prev: EnvFrame,
    next: EnvFrame,
  ): SimEvent[] {
    if (next.done === 0 && prev.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'goal' }, { type: 'done' });
    if (prev.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
