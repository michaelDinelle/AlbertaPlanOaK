import { constant, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

const PW = { TICK: 0, GAME: 1, SEAT: 2, ALIVE: 3, RESULT: 4, CELL: 5 } as const;
const PER_SEAT = 5;
const SELF = 4;
const PER_OTHER = 3;
const KEYS = 6;
const CELL_BITS = 7;

export const UP = 1;
export const DOWN = 2;
export const LEFT = 3;
export const RIGHT = 4;

export const RIGID = 1;
export const WOOD = 2;

export const WON = 1;
export const LOST = 2;
export const TIED = 3;

export const STEP_X = [0, 0, 0, -1, 1, 0] as const;
export const STEP_Y = [0, -1, 1, 0, 0, 0] as const;

export const POM_CELL = 0;

export interface PomShape {
  side: number;
  cells: number;
  seats: number;
  slots: number;
  sight: number;
  others: number;
  team: boolean;
  fog: number;
  tickLimit: number;
  collapseStart: number;
  collapseEvery: number;
  collapseRings: number;
  maxBlast: number;
  bombLife: number;
  flameLife: number;
  maxAmmo: number;
  bombs: number;
  roster: number;
}

export interface PomBomb { cell: number; life: number; dir: number; owner: number; blast: number }
export interface PomSighting { seat: number; code: number; dx: number; dy: number }

export function pomShape(env: EnvDescriptor): PomShape {
  const side = parameter(env, 'side'), cells = side * side, seats = parameter(env, 'seats'), slots = (env.words - PW.CELL) / PER_SEAT, roster = (cells + 1) >> 1;
  if (!Number.isInteger(slots) || slots < seats) throw new Error(`${env.envId} records ${env.words} words, which do not hold ${seats} seats of ${PER_SEAT} words after ${PW.CELL}`);
  if (cells > 1 << CELL_BITS) throw new Error(`${env.envId} lays a ${side}-cell side past the ${CELL_BITS}-bit cells its bomb list packs`);
  if (env.view <= roster) throw new Error(`${env.envId} views ${env.view} words, too few for ${cells} cells and a bomb list`);
  const sight = constant(env, 'POMMERMAN_SIGHT'), span = 2 * sight + 1, others = span * span + SELF, observed = others + PER_OTHER * (seats - 1);
  if (env.obs !== observed) throw new Error(`${env.envId} observes ${env.obs} values where a bomber senses ${observed}`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} actions where a bomber reads ${KEYS}`);
  if (seats !== 2 && seats !== 4) throw new Error(`${env.envId} seats ${seats} bombers`);
  const team = parameter(env, 'team') !== 0;
  if (team && seats !== 4) throw new Error(`${env.envId} plays teams with ${seats} seats`);
  return {
    side, cells, seats, slots, team, roster, sight, others,
    fog: parameter(env, 'fog'),
    tickLimit: parameter(env, 'tickLimit'),
    collapseStart: parameter(env, 'collapseStart'),
    collapseEvery: parameter(env, 'collapseEvery'),
    collapseRings: parameter(env, 'collapseRings'),
    maxBlast: parameter(env, 'maxBlast'),
    bombLife: constant(env, 'POMMERMAN_BOMB_LIFE'),
    flameLife: constant(env, 'POMMERMAN_FLAME_LIFE'),
    maxAmmo: constant(env, 'POMMERMAN_MAX_AMMO'),
    bombs: 2 * (env.view - roster),
  };
}

export const tickOf = (words: ArrayLike<number>): number => words[PW.TICK]!;
export const gameOf = (words: ArrayLike<number>): number => words[PW.GAME]!;
export const seatOf = (words: ArrayLike<number>): number => words[PW.SEAT]!;
export const resultOf = (words: ArrayLike<number>): number => words[PW.RESULT]!;
export const aliveOf = (words: ArrayLike<number>, seat: number): boolean => ((words[PW.ALIVE]! >>> seat) & 1) === 1;
export const cellOf = (words: ArrayLike<number>, seat: number): number => words[PW.CELL + seat]!;
export const ammoOf = (words: ArrayLike<number>, shape: PomShape, seat: number): number => words[PW.CELL + shape.slots + seat]!;
export const blastOf = (words: ArrayLike<number>, shape: PomShape, seat: number): number => words[PW.CELL + 2 * shape.slots + seat]!;
export const kickOf = (words: ArrayLike<number>, shape: PomShape, seat: number): boolean => words[PW.CELL + 3 * shape.slots + seat]! !== 0;
export const actOf = (words: ArrayLike<number>, shape: PomShape, seat: number): number => words[PW.CELL + 4 * shape.slots + seat]!;
export const partnerOf = (shape: PomShape, seat: number): number => (shape.team ? (seat + 2) & 3 : -1);

export function sightingsOf(observations: ArrayLike<number>, shape: PomShape, me: number): PomSighting[] {
  const out: PomSighting[] = [];
  for (let j = 1; j < shape.seats; j++) {
    const at = shape.others + PER_OTHER * (j - 1), code = observations[at]!;
    if (code === 0) continue;
    out.push({ seat: (me + j) % shape.seats, code, dx: Math.round(observations[at + 1]! * shape.side), dy: Math.round(observations[at + 2]! * shape.side) });
  }
  return out;
}

export const fieldOf = (view: ArrayLike<number>, cell: number): number => (view[cell >> 1]! >>> ((cell & 1) << 4)) & 0xffff;
export const terrainOf = (field: number): number => field & 3;
export const itemOf = (field: number): number => (field >>> 2) & 3;
export const hiddenOf = (field: number): boolean => ((field >>> 4) & 1) === 1;
export const flameOf = (field: number): number => (field >>> 5) & 3;
const lifeOf = (field: number): number => (field >>> 7) & 15;
const dirOf = (field: number): number => (field >>> 11) & 7;
const ownerOf = (field: number): number => (field >>> 14) & 3;

export function bombsOf(view: ArrayLike<number>, shape: PomShape): PomBomb[] {
  const out: PomBomb[] = [];
  for (let b = 0; b < shape.bombs; b++) {
    const entry = (view[shape.roster + (b >> 1)]! >>> ((b & 1) << 4)) & 0xffff;
    if (!entry) continue;
    const cell = entry & ((1 << CELL_BITS) - 1), field = fieldOf(view, cell), life = lifeOf(field);
    if (cell >= shape.cells || life === 0) throw new Error(`Pommerman lists bomb ${b} at cell ${cell}, where the board holds no bomb`);
    out.push({ cell, life, dir: dirOf(field), owner: ownerOf(field), blast: entry >>> CELL_BITS });
  }
  return out;
}

export function collapsedRings(shape: PomShape, tick: number): number {
  if (!shape.collapseRings || tick <= shape.collapseStart) return 0;
  return Math.min(shape.collapseRings, Math.ceil((tick - shape.collapseStart) / shape.collapseEvery));
}

export const ringOf = (shape: PomShape, x: number, y: number): number => Math.min(x, y, shape.side - 1 - x, shape.side - 1 - y);

const WIN: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOSS: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const TIE: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const pommermanDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[POM_CELL] = cellOf(frame.words, seatOf(frame.words));
  },
  geometry({ environment }, initial) {
    const shape = pomShape(environment), words = initial.words, full = (1 << shape.seats) - 1;
    if (tickOf(words) !== 0 || words[PW.ALIVE] !== full) throw new Error(`${environment.envId} opens at tick ${tickOf(words)}, alive mask ${words[PW.ALIVE]}`);
    if (seatOf(words) >= shape.seats) throw new Error(`${environment.envId} seats the agent at ${seatOf(words)} of ${shape.seats}`);
    for (let s = 0; s < shape.seats; s++) if (cellOf(words, s) >= shape.cells) throw new Error(`${environment.envId} opens with seat ${s} off the board`);
    if (bombsOf(initial.view, shape).length) throw new Error(`${environment.envId} opens with bombs on the board`);
    for (let c = 0; c < shape.cells; c++) if (flameOf(fieldOf(initial.view, c))) throw new Error(`${environment.envId} opens with flames on the board`);
    return { grid: { w: shape.side, h: shape.side, walls: new Uint8Array(shape.cells) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const result = resultOf(next.words);
    if (result === WON) return WIN as SimEvent[];
    if (result === LOST) return LOSS as SimEvent[];
    if (result === TIED) return TIE as SimEvent[];
    throw new Error(`Pommerman step ${next.step} ends a game with result ${result}`);
  },
};
