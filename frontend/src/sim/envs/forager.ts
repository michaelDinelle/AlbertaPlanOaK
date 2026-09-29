import { constant, constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const FW = { X: 0, Y: 1, COUNT: 2 } as const;
export const FORAGER_X = 0;
export const FORAGER_Y = 1;
export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;
const KEYS = 4;

export interface ForagerShape {
  side: number;
  fov: number;
  span: number;
  switching: boolean;
  period: number;
  values: readonly (readonly number[])[];
  kinds: readonly number[];
  target: number;
}

export function foragerShape(env: EnvDescriptor): ForagerShape {
  const side = parameter(env, 'side'), fov = parameter(env, 'fov'), span = 2 * fov + 1, flat = constantList(env, 'FORAGER_VALUES'), kinds = constantList(env, 'FORAGER_KINDS');
  if (side < 4 || side > 64 || side % 4 !== 0) throw new Error(`${env.envId} lays a ${side}-cell torus`);
  if (env.words !== 3 || env.view !== side * side / 16) throw new Error(`${env.envId} records ${env.words} words and ${env.view} view words where the forager layout reads 3 and ${side * side / 16}`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} actions where the forager reads ${KEYS}`);
  if (env.obs !== span * span + 1) throw new Error(`${env.envId} observes ${env.obs} values where a ${span}x${span} window and a bias show ${span * span + 1}`);
  if (flat.length !== 6) throw new Error(`${env.envId} lists ${flat.length} item values where two phases of three read 6`);
  if (kinds.length !== 3 || new Set(kinds).size !== 3) throw new Error(`${env.envId} lists ${kinds.length} kind codes where empty and two species read 3 distinct`);
  return { side, fov, span, switching: parameter(env, 'switching') !== 0, period: constant(env, 'FORAGER_SWITCH_PERIOD'), values: [flat.slice(0, 3), flat.slice(3, 6)], kinds, target: side * side / 16 };
}

export const itemAt = (view: ArrayLike<number>, cell: number): number => (view[cell >>> 4]! >>> (2 * (cell & 15))) & 3;
export const phaseOf = (shape: ForagerShape, step: number): number => (shape.switching && Math.floor(step / shape.period) % 2 === 1 ? 1 : 0);
export const valueOf = (shape: ForagerShape, step: number, item: number): number => shape.values[phaseOf(shape, step)]![item]!;
export const kindOf = (shape: ForagerShape, code: number): number => shape.kinds.indexOf(code);
export const wrapOf = (v: number, side: number): number => ((v % side) + side) % side;
export const deltaOf = (from: number, to: number, side: number): number => wrapOf(to - from + side / 2, side) - side / 2;

export function windowCell(shape: ForagerShape, frame: EnvFrame, i: number): number {
  const { side, fov, span } = shape, x = wrapOf(frame.words[FW.X]! + (i % span) - fov, side), y = wrapOf(frame.words[FW.Y]! + Math.floor(i / span) - fov, side);
  return y * side + x;
}

export function checkFrame(shape: ForagerShape, frame: EnvFrame): void {
  const { side, span } = shape, x = frame.words[FW.X]!, y = frame.words[FW.Y]!;
  if (x >= side || y >= side) throw new Error(`Forager step ${frame.step} stands at (${x}, ${y}) off a ${side}-cell torus`);
  let count = 0;
  for (let cell = 0; cell < side * side; cell++) {
    const item = itemAt(frame.view, cell);
    if (item === 3) throw new Error(`Forager step ${frame.step} holds item 3 at cell ${cell}`);
    count += item !== 0 ? 1 : 0;
  }
  if (count !== frame.words[FW.COUNT]) throw new Error(`Forager step ${frame.step} counts ${frame.words[FW.COUNT]} items where its map holds ${count}`);
  for (let i = 0; i < span * span; i++) {
    const seen = frame.observations[i]!, drawn = shape.kinds[itemAt(frame.view, windowCell(shape, frame, i))]!;
    if (seen !== drawn) throw new Error(`Forager step ${frame.step} observes ${seen} at window cell ${i} where its map shows ${drawn}`);
  }
}

const GOAL: SimEvent = { type: 'goal' }, HARM: SimEvent = { type: 'harm' };
const GATHERED = Object.freeze([GOAL]) as SimEvent[];
const FAILED = Object.freeze([HARM]) as SimEvent[];

export const foragerDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[FORAGER_X] = frame.words[FW.X]!;
    out[FORAGER_Y] = frame.words[FW.Y]!;
  },
  geometry({ environment }, initial) {
    const shape = foragerShape(environment);
    checkFrame(shape, initial);
    return { grid: { w: shape.side, h: shape.side, walls: new Uint8Array(shape.side * shape.side) } };
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 ? GATHERED : next.reward < 0 ? FAILED : NO_EVENTS;
  },
};
