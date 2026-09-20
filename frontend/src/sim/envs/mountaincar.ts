import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';

export const MC_MIN_POS = -1.2;
export const MC_MAX_POS = 0.6;
export const MC_GOAL_POS = 0.5;
export const MC_MAX_SPEED = 0.07;
export const MC_START_POS = -0.5;
export const MC_ACTIONS = 3;

export const MOUNTAINCAR_POS = 0;
export const MOUNTAINCAR_VEL = 1;
export const MOUNTAINCAR_ACTION = 3;

export function mountainCarForce(action: number): number {
  return ((Math.trunc(action) >>> 0) % MC_ACTIONS) - 1;
}
export const MC_TROUGH_X = -Math.PI / 6;
export const MC_CREST_X = Math.PI / 6;


export function mountainCarHeight(x: number): number {
  return Math.sin(3 * x) * 0.45 + 0.55;
}

export function mountainCarSlope(x: number): number {
  return 1.35 * Math.cos(3 * x);
}


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

export const mountainCarDecoder: EnvDecoder = {
  geometry(): EnvGeometry {
    return GEOMETRY;
  },


  deriveEvents(
    _prev: EnvFrame,
    next: EnvFrame,
  ): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (next.done !== 0) out.push({ type: 'goal' }, { type: 'done' });
    if (next.done !== 0) out.push({ type: 'teleport' });
    return out;
  },
};
