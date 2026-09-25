import { parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const LW = { PLAYER: 0, FOOD: 3, CLOCK: 6, SPAWNED: 7, EPISODE: 8, KINDS: 9, LAST: 10, COUNT: 11 } as const;

export const NONE = 0;
export const LOAD = 5;
const KEYS = 6;
const MOST = 3;
export const STEP_ROW = [0, -1, 1, 0, 0, 0] as const;
export const STEP_COL = [0, 0, 0, -1, 1, 0] as const;

export const LBF_COL = 0;
export const LBF_ROW = 1;

export interface LbfShape {
  field: number;
  players: number;
  foods: number;
  sight: number;
  levels: number;
  steps: number;
  full: boolean;
}

export interface Seen {
  top: number;
  left: number;
  foods: boolean[];
  players: boolean[];
}

export const rowOf = (entity: number): number => (entity >>> 4) & 15;
export const colOf = (entity: number): number => entity & 15;
export const levelOf = (entity: number): number => entity >>> 8;
export const playerAt = (words: ArrayLike<number>, i: number): number => words[LW.PLAYER + i]!;
export const foodAt = (words: ArrayLike<number>, j: number): number => words[LW.FOOD + j]!;
export const lastOf = (words: ArrayLike<number>, i: number): number => (words[LW.LAST]! >>> (4 * i)) & 15;
export const kindOf = (words: ArrayLike<number>, i: number): number => (words[LW.KINDS]! >>> (4 * i)) & 15;
export const isMove = (action: number): boolean => action >= 1 && action <= 4;
const touching = (a: number, b: number): boolean => Math.abs(rowOf(a) - rowOf(b)) + Math.abs(colOf(a) - colOf(b)) === 1;

export function intentOf(frame: EnvFrame): number {
  const pressed = frame.actions[0]!;
  return pressed >= 0 && pressed < KEYS ? Math.trunc(pressed) : NONE;
}

export function besideOf(words: ArrayLike<number>, foods: number, entity: number): number {
  for (let j = 0; j < foods; j++) if (foodAt(words, j) !== 0 && touching(entity, foodAt(words, j))) return j;
  return -1;
}

export function lbfShape(env: EnvDescriptor): LbfShape {
  const field = parameter(env, 'field'), players = parameter(env, 'players'), foods = parameter(env, 'foods'), sight = parameter(env, 'sight'), levels = parameter(env, 'levels');
  if (env.words !== LW.COUNT || env.view !== 0) throw new Error(`${env.envId} records ${env.words} words and ${env.view} view words where the foraging layout reads ${LW.COUNT} and 0`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} actions where foraging reads ${KEYS}`);
  if (players < 1 || players > MOST || foods < 1 || foods > MOST) throw new Error(`${env.envId} fields ${players} players and ${foods} foods past the ${MOST} slots its words hold`);
  if (env.obs !== 3 * (players + foods)) throw new Error(`${env.envId} observes ${env.obs} values where ${players} players and ${foods} foods show ${3 * (players + foods)}`);
  if (field < 3 || field > 10 || levels < 1) throw new Error(`${env.envId} lays a ${field}-cell field with ${levels} levels`);
  return { field, players, foods, sight, levels, steps: parameter(env, 'steps'), full: sight + 1 >= field };
}

export function seenBy(frame: EnvFrame, shape: LbfShape): Seen {
  const { foods, players } = shape, o = frame.observations, self = playerAt(frame.words, 0), at = 3 * foods;
  if (o[at + 2] !== levelOf(self)) throw new Error(`LBF step ${frame.step} observes its own level as ${o[at + 2]} though it stands at level ${levelOf(self)}`);
  const top = rowOf(self) - o[at]!, left = colOf(self) - o[at + 1]!;
  const seen: Seen = { top, left, foods: new Array<boolean>(foods).fill(false), players: new Array<boolean>(players).fill(false) };
  seen.players[0] = true;
  for (let k = 0; k < foods; k++) {
    const level = o[3 * k + 2]!;
    if (level === 0) continue;
    const j = Array.from({ length: foods }, (_, n) => n).find((n) => { const f = foodAt(frame.words, n); return levelOf(f) === level && rowOf(f) === o[3 * k]! + top && colOf(f) === o[3 * k + 1]! + left; });
    if (j === undefined) throw new Error(`LBF step ${frame.step} observes a level ${level} food at (${o[3 * k]! + top}, ${o[3 * k + 1]! + left}) that no food word holds`);
    seen.foods[j] = true;
  }
  for (let k = 0; k < players - 1; k++) {
    const base = at + 3 + 3 * k, level = o[base + 2]!;
    if (level === 0) continue;
    const i = Array.from({ length: players - 1 }, (_, n) => n + 1).find((n) => { const p = playerAt(frame.words, n); return levelOf(p) === level && rowOf(p) === o[base]! + top && colOf(p) === o[base + 1]! + left; });
    if (i === undefined) throw new Error(`LBF step ${frame.step} observes a level ${level} partner at (${o[base]! + top}, ${o[base + 1]! + left}) that no player word holds`);
    seen.players[i] = true;
  }
  return seen;
}

const GOAL: SimEvent = { type: 'goal' }, HARM: SimEvent = { type: 'harm' }, TELEPORT: SimEvent = { type: 'teleport' }, DONE: SimEvent = { type: 'done' };
const GATHERED = Object.freeze([GOAL]) as SimEvent[];
const FAILED = Object.freeze([HARM]) as SimEvent[];
const ENDED = Object.freeze([TELEPORT, DONE]) as SimEvent[];
const ENDED_GATHERED = Object.freeze([GOAL, TELEPORT, DONE]) as SimEvent[];
const ENDED_FAILED = Object.freeze([HARM, TELEPORT, DONE]) as SimEvent[];

export const lbfDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    const self = playerAt(frame.words, 0);
    out[LBF_COL] = colOf(self);
    out[LBF_ROW] = rowOf(self);
  },
  geometry({ environment }, initial) {
    const shape = lbfShape(environment), { field, players, foods, levels } = shape;
    for (let i = 0; i < players; i++) {
      const p = playerAt(initial.words, i);
      if (rowOf(p) >= field || colOf(p) >= field || levelOf(p) < 1 || levelOf(p) > levels) throw new Error(`${environment.envId} opens with player ${i} at (${rowOf(p)}, ${colOf(p)}) level ${levelOf(p)}`);
    }
    for (let j = 0; j < foods; j++) {
      const f = foodAt(initial.words, j);
      if (f !== 0 && (rowOf(f) < 1 || colOf(f) < 1 || rowOf(f) > field - 2 || colOf(f) > field - 2 || levelOf(f) < 1)) throw new Error(`${environment.envId} opens with food ${j} at (${rowOf(f)}, ${colOf(f)}) level ${levelOf(f)}`);
    }
    seenBy(initial, shape);
    return { grid: { w: field, h: field, walls: new Uint8Array(field * field) } };
  },
  deriveEvents(_prev, next) {
    if (next.done) return next.reward > 0 ? ENDED_GATHERED : next.reward < 0 ? ENDED_FAILED : ENDED;
    return next.reward > 0 ? GATHERED : next.reward < 0 ? FAILED : NO_EVENTS;
  },
};
