import { constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const OW = { AGENT: 0, PARTNER: 1, POTS: 2, COUNTERS: 3, CLOCK: 5, GAME: 6, LAST: 7, COUNT: 8 } as const;
export const TILE = { FLOOR: 0, COUNTER: 1, ONIONS: 2, DISHES: 3, POT: 4, WINDOW: 5 } as const;
export const ITEM = { EMPTY: 0, ONION: 1, DISH: 2, SOUP: 3 } as const;
export const STAY = 4;
export const INTERACT = 5;
export const FULL = 3;
export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;

export const OC_X = 0;
export const OC_Y = 1;
export const OC_PARTNER_X = 2;
export const OC_PARTNER_Y = 3;

const ACTIONS = 6;
const OBS = 17;
const SIDE = 16;
const COUNTER_SLOTS = 32;
const POT_SLOTS = 2;

export interface Kitchen {
  w: number;
  h: number;
  tiles: Uint8Array;
  counters: number[];
  pots: number[];
  starts: readonly [number, number];
  cookTime: number;
  horizon: number;
  patience: number;
  soupReward: number;
}

export const xOf = (chef: number): number => chef & 15;
export const yOf = (chef: number): number => (chef >>> 4) & 15;
export const cellOf = (chef: number): number => chef & 255;
export const facingOf = (chef: number): number => (chef >>> 8) & 3;
export const heldOf = (chef: number): number => (chef >>> 10) & 3;
export const stillOf = (chef: number): number => (chef >>> 12) & 3;
export const chefOf = (words: ArrayLike<number>, i: number): number => words[OW.AGENT + i]!;
export const potOf = (words: ArrayLike<number>, p: number): number => (words[OW.POTS]! >>> (8 * p)) & 255;
export const counterOf = (words: ArrayLike<number>, slot: number): number => (words[OW.COUNTERS + (slot >> 4)]! >>> ((slot & 15) * 2)) & 3;
export const clockOf = (words: ArrayLike<number>): number => words[OW.CLOCK]! & 0xffff;
export const gameOf = (words: ArrayLike<number>): number => words[OW.GAME]!;
export const actOf = (words: ArrayLike<number>, i: number): number => (words[OW.LAST]! >>> (3 * i)) & 7;
export const chanceOf = (words: ArrayLike<number>): number => (words[OW.LAST]! >>> 6) & 1;
export const aheadX = (chef: number): number => xOf(chef) + STEP_X[facingOf(chef)]!;
export const aheadY = (chef: number): number => yOf(chef) + STEP_Y[facingOf(chef)]!;
export const onionsIn = (pot: number): number => Math.min(pot, FULL);

export function tileAt(kitchen: Kitchen, x: number, y: number): number {
  return x < 0 || y < 0 || x >= kitchen.w || y >= kitchen.h ? -1 : kitchen.tiles[y * kitchen.w + x]!;
}

export function kitchenOf(env: EnvDescriptor): Kitchen {
  if (env.words !== OW.COUNT || env.view !== 0 || env.obs !== OBS || env.actions !== ACTIONS || env.box !== 0) throw new Error(`${env.envId} records ${env.words} words, ${env.view} view words, ${env.obs} observations and ${env.actions} actions where the kitchen reads ${OW.COUNT}, 0, ${OBS} and ${ACTIONS}`);
  const terrain = constantList(env, 'OVERCOOKED_TERRAIN'), at = parameter(env, 'terrain'), w = parameter(env, 'width'), h = parameter(env, 'height');
  if (w < 3 || h < 3 || w > SIDE || h > SIDE || at < 0 || at + w * h > terrain.length) throw new Error(`${env.envId} lays a ${w}x${h} kitchen at ${at} of ${terrain.length} terrain cells`);
  const tiles = Uint8Array.from(terrain.slice(at, at + w * h)), counters: number[] = [], pots: number[] = [];
  tiles.forEach((code, cell) => {
    if (code > TILE.WINDOW) throw new Error(`${env.envId} holds terrain code ${code} at cell ${cell}`);
    if (code === TILE.COUNTER) counters.push(cell);
    if (code === TILE.POT) pots.push(cell);
  });
  if (counters.length > COUNTER_SLOTS || pots.length < 1 || pots.length > POT_SLOTS) throw new Error(`${env.envId} fits ${counters.length} counters and ${pots.length} pots where its words hold ${COUNTER_SLOTS} and ${POT_SLOTS}`);
  const start = (key: string) => {
    const x = parameter(env, `x${key}`), y = parameter(env, `y${key}`);
    if (x >= w || y >= h || tiles[y * w + x] !== TILE.FLOOR) throw new Error(`${env.envId} starts a chef at (${x}, ${y}) off the floor`);
    return x | y << 4;
  };
  return {
    w, h, tiles, counters, pots,
    starts: [start('0'), start('1')],
    cookTime: parameter(env, 'cookTime'),
    horizon: parameter(env, 'horizon'),
    patience: parameter(env, 'commisPatience'),
    soupReward: parameter(env, 'soupReward'),
  };
}

const SERVED: SimEvent[] = Object.freeze([{ type: 'goal' }]) as SimEvent[];
const ENDED: SimEvent[] = Object.freeze([{ type: 'teleport' }, { type: 'done' }]) as SimEvent[];
const SERVED_ENDED: SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]) as SimEvent[];

export const overcookedDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) {
    const agent = chefOf(frame.words, 0), partner = chefOf(frame.words, 1);
    out[OC_X] = xOf(agent);
    out[OC_Y] = yOf(agent);
    out[OC_PARTNER_X] = xOf(partner);
    out[OC_PARTNER_Y] = yOf(partner);
  },
  geometry({ environment }, initial) {
    const kitchen = kitchenOf(environment), { w, h, starts } = kitchen, words = initial.words;
    const agent = chefOf(words, 0), partner = chefOf(words, 1);
    const seated = (cellOf(agent) === starts[0] && cellOf(partner) === starts[1]) || (cellOf(agent) === starts[1] && cellOf(partner) === starts[0]);
    if (!seated || clockOf(words) !== 0 || words[OW.POTS] !== 0 || heldOf(agent) !== ITEM.EMPTY || heldOf(partner) !== ITEM.EMPTY) throw new Error(`${environment.envId} opens with chefs at ${cellOf(agent)} and ${cellOf(partner)} on tick ${clockOf(words)} where it seats them at ${starts[0]} and ${starts[1]} on an empty kitchen`);
    return { grid: { w, h, walls: Uint8Array.from(kitchen.tiles, (code) => (code === TILE.FLOOR ? 0 : 1)) } };
  },
  deriveEvents(_prev, next) {
    if (next.done) return next.reward > 0 ? SERVED_ENDED : ENDED;
    return next.reward > 0 ? SERVED : NO_EVENTS;
  },
};
