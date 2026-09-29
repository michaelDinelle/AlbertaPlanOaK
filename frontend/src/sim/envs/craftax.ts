import { constantList, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const CXW = { PLAYER: 0, VITALS: 1, HUNGER: 2, THIRST: 3, FATIGUE: 4, RECOVER: 5, INVENTORY: 6, ACHIEVED: 9, TIME: 10, GAME: 11, MENU: 12, GEN: 13, DEATHS: 14, ZOMBIE: 15, COW: 18, SKELETON: 21, ARROW: 23, COUNT: 26 } as const;

export const OBS_LIGHT = 29;

export const BLOCK = { OUT: 1, GRASS: 2, WATER: 3, STONE: 4, TREE: 5, WOOD: 6, PATH: 7, COAL: 8, IRON: 9, DIAMOND: 10, TABLE: 11, FURNACE: 12, SAND: 13, LAVA: 14, PLANT: 15, RIPE: 16 } as const;

export const ACTION = { NOOP: 0, LEFT: 1, RIGHT: 2, UP: 3, DOWN: 4, DO: 5, SLEEP: 6, PLACE_STONE: 7, PLACE_TABLE: 8, PLACE_FURNACE: 9, PLACE_PLANT: 10, WOOD_PICKAXE: 11, STONE_PICKAXE: 12, IRON_PICKAXE: 13, WOOD_SWORD: 14, STONE_SWORD: 15, IRON_SWORD: 16 } as const;

export const ACHIEVEMENTS = ['collect wood', 'place table', 'eat cow', 'collect sapling', 'collect drink', 'make wood pickaxe', 'make wood sword', 'place plant', 'defeat zombie', 'collect stone', 'place stone', 'eat plant', 'defeat skeleton', 'make stone pickaxe', 'make stone sword', 'wake up', 'place furnace', 'collect coal', 'collect iron', 'collect diamond', 'make iron pickaxe', 'make iron sword'] as const;

export const ITEMS = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'wood pickaxe', 'stone pickaxe', 'iron pickaxe', 'wood sword', 'stone sword', 'iron sword'] as const;

export const MOB_SLOTS = CXW.COUNT - CXW.ZOMBIE;
export const PRESS = { NEXT: 5, CONFIRM: 4 } as const;
export const FACING = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;

export const CX_X = 0;
export const CX_Y = 1;
export const CX_FACING = 2;
export const CX_SLEEPING = 3;
export const CX_GAME = 4;

export const cellOf = (word: number): number => word & 4095;
export const facingOf = (words: Uint32Array): number => (words[CXW.PLAYER]! >>> 12) & 3;
export const sleepingOf = (words: Uint32Array): boolean => ((words[CXW.PLAYER]! >>> 14) & 1) === 1;
export const vital = (words: Uint32Array, k: number): number => (words[CXW.VITALS]! >>> (4 * k)) & 15;
export const itemCount = (words: Uint32Array, item: number): number => (words[CXW.INVENTORY + (item >> 2)]! >>> (8 * (item & 3))) & 255;
export const cursorOf = (words: Uint32Array): number => words[CXW.MENU]! & 15;
export const readoutOf = (words: Uint32Array): number => (words[CXW.MENU]! >>> 8) & 127;
export const alive = (word: number): boolean => word >>> 31 === 1;
export const healthOf = (word: number): number => (word >>> 12) & 15;
export const restOf = (word: number): number => (word >>> 16) & 15;
export const arrowDirection = (word: number): number => (word >>> 20) & 3;
export const blockAt = (view: Uint32Array, cell: number): number => ((view[cell >> 3]! >>> (4 * (cell & 7))) & 15) + 1;

export interface CraftaxShape {
  side: number;
  readout: (readonly [number, number])[];
  patch: number;
  menu: number[];
  moves: number[];
  presses: number;
}

export function craftaxShape(env: EnvDescriptor): CraftaxShape {
  const side = Math.sqrt(env.view * 8);
  if (!Number.isInteger(side)) throw new Error(`${env.envId} records ${env.view} view words, which do not pack a square map at four bits a cell`);
  if (env.words !== CXW.COUNT) throw new Error(`${env.envId} records ${env.words} state words where the decoder reads ${CXW.COUNT}`);
  const flat = constantList(env, 'CX_READOUT'), readout: (readonly [number, number])[] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) readout.push([flat[i]!, flat[i + 1]!] as const);
  const patch = 2 * Math.min(...readout.map(([dy, dx]) => Math.max(Math.abs(dy), Math.abs(dx)))) - 1;
  return { side, readout, patch, menu: constantList(env, 'CX_MENU_ACTION'), moves: constantList(env, 'CX_MOVE_ACTION'), presses: env.actions };
}

const health = (frame: EnvFrame): number => vital(frame.words, 0);

export const pressOf = (frame: EnvFrame, shape: CraftaxShape): number => {
  const control = frame.actions[0]!;
  return control >= 0 && control < shape.presses ? Math.trunc(control) : 0;
};

export const died = (prev: EnvFrame, next: EnvFrame): boolean => prev.words[CXW.DEATHS] !== next.words[CXW.DEATHS];

export const craftaxDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) {
    const cell = cellOf(frame.words[CXW.PLAYER]!);
    out[CX_X] = cell & 63;
    out[CX_Y] = cell >>> 6;
    out[CX_FACING] = facingOf(frame.words);
    out[CX_SLEEPING] = sleepingOf(frame.words) ? 1 : 0;
    out[CX_GAME] = frame.words[CXW.GAME]!;
  },
  geometry({ environment }, initial) {
    const { side } = craftaxShape(environment);
    if (initial.words.length !== CXW.COUNT || initial.view.length * 8 !== side * side) throw new Error(`${environment.envId} frames do not carry the decoded state and map`);
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(prev, next) {
    const same = prev.words[CXW.GAME] === next.words[CXW.GAME], fell = died(prev, next);
    if (same && !fell && !next.done && next.words[CXW.ACHIEVED] === prev.words[CXW.ACHIEVED] && health(next) >= health(prev)) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (same) for (let gained = next.words[CXW.ACHIEVED]! & ~prev.words[CXW.ACHIEVED]!; gained; gained &= gained - 1) events.push({ type: 'goal' });
    if (fell || (same && health(next) < health(prev))) events.push({ type: 'harm' });
    if (next.done) events.push({ type: 'done' });
    if (!same || fell) events.push({ type: 'teleport' });
    return events;
  },
};
