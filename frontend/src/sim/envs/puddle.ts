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

export function puddlePlan(side: number, goal: number, penalty: number, sweeps: number): Float32Array {
  const plan = new Float32Array(side * side), scale = side - 1;
  const atGoal = (x: number, y: number): boolean => x >= goal && y >= goal;
  const cost = (x: number, y: number): number => 1 + penalty * puddleDepth(x, y);
  const node = (x: number, y: number): number => Math.floor(y * scale + 0.5) * side + Math.floor(x * scale + 0.5);
  const score = (x: number, y: number): number => (atGoal(x, y) ? 0 : plan[node(x, y)]! + cost(x, y));
  for (let i = 0; i < side * side; i++) plan[i] = atGoal((i % side) / scale, Math.floor(i / side) / scale) ? 0 : Infinity;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let changed = false;
    for (let i = 0; i < side * side; i++) {
      const x = (i % side) / scale, y = Math.floor(i / side) / scale;
      if (atGoal(x, y)) continue;
      let best = plan[i]!;
      for (let action = 0; action < 4; action++) {
        const [dx, dy] = PUDDLE_DELTA[action]!;
        best = Math.min(best, score(Math.min(1, Math.max(0, x + PUDDLE_STEP * dx)), Math.min(1, Math.max(0, y - PUDDLE_STEP * dy))));
      }
      if (best < plan[i]!) { plan[i] = best; changed = true; }
    }
    if (!changed) break;
  }
  return plan;
}

export function puddleBestAction(plan: Float32Array, side: number, goal: number, penalty: number, x: number, y: number): number {
  const scale = side - 1;
  let best = 0, bestScore = Infinity;
  for (let action = 0; action < 4; action++) {
    const [dx, dy] = PUDDLE_DELTA[action]!, nx = Math.min(1, Math.max(0, x + PUDDLE_STEP * dx)), ny = Math.min(1, Math.max(0, y - PUDDLE_STEP * dy));
    const score = nx >= goal && ny >= goal ? 0 : plan[Math.floor(ny * scale + 0.5) * side + Math.floor(nx * scale + 0.5)]! + 1 + penalty * puddleDepth(nx, ny);
    if (score < bestScore) { bestScore = score; best = action; }
  }
  return best;
}

export const PUDDLE_DELTA: readonly (readonly [number, number])[] = Object.freeze([[0, -1], [0, 1], [-1, 0], [1, 0]]);

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
