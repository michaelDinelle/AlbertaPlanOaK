import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const PUDDLE_STEP = 0.05;
export const PUDDLE_GOAL = 0.95;
export const PUDDLE_RADIUS = 0.1;
export const PUDDLE_PENALTY = 400;
export const PUDDLE_ARMS: readonly (readonly [number, number, number, number])[] = Object.freeze([
  [0.1, 0.75, 0.45, 0.75],
  [0.45, 0.4, 0.45, 0.8],
]);

export const PUDDLE_X = 0;
export const PUDDLE_Y = 1;
export const PUDDLE_ACTION = 2;

export function puddleDepth(x: number, y: number): number {
  let depth = 0;
  for (const [ax, ay, bx, by] of PUDDLE_ARMS) {
    const abx = bx - ax, aby = by - ay;
    const t = Math.min(1, Math.max(0, ((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby)));
    const d = Math.hypot(x - ax - abx * t, y - ay - aby * t);
    if (d < PUDDLE_RADIUS) depth += PUDDLE_RADIUS - d;
  }
  return depth;
}

const ANCHORS: Record<string, readonly [number, number, number]> = Object.freeze({
  goal: [PUDDLE_GOAL, 0, PUDDLE_GOAL],
  boundsMin: [0, 0, 0],
  boundsMax: [1, 0, 1],
});

const GEOMETRY: EnvGeometry = Object.freeze({ classic: Object.freeze({ anchors: ANCHORS }) });

export const puddleDecoder: EnvDecoder = {
  slots: 3,
  decode(frame, out) {
    out[PUDDLE_X] = wordFloat(frame.words[0]!);
    out[PUDDLE_Y] = wordFloat(frame.words[1]!);
    out[PUDDLE_ACTION] = frame.actions[0]!;
  },
  geometry(): EnvGeometry { return GEOMETRY; },
  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    return next.done === 0 ? NO_EVENTS : [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }];
  },
};
