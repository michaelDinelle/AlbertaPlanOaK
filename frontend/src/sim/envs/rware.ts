import { parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const NOOP = 0;
export const FORWARD = 1;
export const TOGGLE = 4;
export const NONE = 511;
export const CARRIED = 0x8000;
export const RWARE_X = 0;
export const RWARE_Y = 1;
export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;

const ACTIONS = 5;
const OBSERVED = 15;
const CLOCK = 0;
const FIRST = 1;
const LANES = 4;
const STRIDE = 16;

export interface RwareShape {
  w: number;
  h: number;
  agents: number;
  requests: number;
  steps: number;
  queue: number;
  board: number;
}

export const cellOf = (word: number): number => word & NONE;
export const colOf = (cell: number): number => cell & (STRIDE - 1);
export const rowOf = (cell: number): number => cell >>> 4;
export const cellAt = (x: number, y: number): number => y * STRIDE + x;
export const facingOf = (word: number): number => (word >>> 9) & 3;
export const loadedOf = (word: number): number => (word >>> 11) & 1;
export const aimOf = (word: number): number => (word >>> 13) & NONE;
export const stuckOf = (word: number): number => (word >>> 22) & 3;
export const scoreOf = (word: number): number => word >>> 24;
export const clockOf = (words: ArrayLike<number>): number => words[CLOCK]!;
export const agentAt = (words: ArrayLike<number>, i: number): number => words[FIRST + i]!;
export const entryAt = (words: ArrayLike<number>, shape: RwareShape, slot: number): number => (words[shape.queue + (slot >> 1)]! >>> ((slot & 1) * 16)) & 0xffff;
export const standingAt = (words: ArrayLike<number>, shape: RwareShape, cell: number): boolean => ((words[shape.board + (cell >> 5)]! >>> (cell & 31)) & 1) !== 0;
export const placeOf = (words: ArrayLike<number>, entry: number): number => (entry & CARRIED ? cellOf(agentAt(words, entry & 3)) : entry);

export function actionOf(frame: EnvFrame): number {
  const pressed = frame.actions[0]!;
  return pressed >= 0 && pressed < ACTIONS ? Math.trunc(pressed) : NOOP;
}

export function wanted(words: ArrayLike<number>, shape: RwareShape, entry: number): boolean {
  for (let s = 0; s < shape.requests; s++) if (entryAt(words, shape, s) === entry) return true;
  return false;
}

export function rwareShape(env: EnvDescriptor): RwareShape {
  const rows = parameter(env, 'shelfRows'), columns = parameter(env, 'shelfColumns'), height = parameter(env, 'columnHeight');
  const agents = parameter(env, 'agents'), requests = parameter(env, 'requests'), steps = parameter(env, 'maxSteps');
  const w = 3 * columns + 1, h = (height + 1) * rows + 2, queue = FIRST + agents, board = queue + ((requests + 1) >> 1);
  if (env.view !== 0 || env.obs !== OBSERVED || env.actions !== ACTIONS) throw new Error(`${env.envId} records ${env.view} view words, ${env.obs} observations and ${env.actions} actions where the warehouse reads 0, ${OBSERVED} and ${ACTIONS}`);
  if (agents < 1 || agents > LANES || requests < 1 || requests > LANES) throw new Error(`${env.envId} runs ${agents} robots and ${requests} requests past the ${LANES} lanes its words hold`);
  if (w > STRIDE || h * STRIDE > NONE) throw new Error(`${env.envId} lays a ${w} by ${h} floor past the ${STRIDE}-cell rows its cells address`);
  if (env.words !== board + ((h + 1) >> 1)) throw new Error(`${env.envId} records ${env.words} words where ${agents} robots, ${requests} requests and ${h} rows take ${board + ((h + 1) >> 1)}`);
  return { w, h, agents, requests, steps, queue, board };
}

export function homesOf(initial: EnvFrame, shape: RwareShape): Uint8Array {
  const homes = new Uint8Array(shape.w * shape.h);
  for (let y = 0; y < shape.h; y++) for (let x = 0; x < shape.w; x++) homes[y * shape.w + x] = standingAt(initial.words, shape, cellAt(x, y)) ? 1 : 0;
  for (let cell = 0; cell < ((shape.h + 1) >> 1) * 32; cell++) {
    if (standingAt(initial.words, shape, cell) && (colOf(cell) >= shape.w || rowOf(cell) >= shape.h)) throw new Error(`RWARE opens with a shelf at (${colOf(cell)}, ${rowOf(cell)}) off its ${shape.w} by ${shape.h} floor`);
  }
  return homes;
}

const DELIVERED: SimEvent = { type: 'goal' }, TELEPORT: SimEvent = { type: 'teleport' }, DONE: SimEvent = { type: 'done' };
const DELIVERY = Object.freeze([DELIVERED]) as SimEvent[];
const ENDED = Object.freeze([TELEPORT, DONE]) as SimEvent[];
const ENDED_DELIVERY = Object.freeze([DELIVERED, TELEPORT, DONE]) as SimEvent[];

export const rwareDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    const cell = cellOf(agentAt(frame.words, 0));
    out[RWARE_X] = colOf(cell);
    out[RWARE_Y] = rowOf(cell);
  },
  geometry({ environment }, initial) {
    const shape = rwareShape(environment), { w, h, agents, requests } = shape;
    homesOf(initial, shape);
    if (clockOf(initial.words) !== 0) throw new Error(`${environment.envId} opens at clock ${clockOf(initial.words)}`);
    for (let i = 0; i < agents; i++) {
      const word = agentAt(initial.words, i), cell = cellOf(word);
      if (colOf(cell) >= w || rowOf(cell) >= h || loadedOf(word) !== 0 || aimOf(word) !== NONE || scoreOf(word) !== 0) throw new Error(`${environment.envId} opens with robot ${i} at (${colOf(cell)}, ${rowOf(cell)}) loaded ${loadedOf(word)} aiming ${aimOf(word)} scoring ${scoreOf(word)}`);
    }
    for (let s = 0; s < requests; s++) {
      const entry = entryAt(initial.words, shape, s);
      if (entry & CARRIED || !standingAt(initial.words, shape, entry)) throw new Error(`${environment.envId} opens requesting ${entry}, which is no standing shelf`);
    }
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(prev, next) {
    if (next.reward !== 0 && next.reward !== 1) throw new Error(`RWARE step ${next.step} pays ${next.reward} where a delivery pays 1`);
    if (next.done) return next.reward > 0 ? ENDED_DELIVERY : ENDED;
    const delivered = (scoreOf(next.words[FIRST]!) - scoreOf(prev.words[FIRST]!)) & 255;
    if (delivered !== next.reward) throw new Error(`RWARE step ${next.step} pays ${next.reward} while its robot's tally moves by ${delivered}`);
    return next.reward > 0 ? DELIVERY : NO_EVENTS;
  },
};
