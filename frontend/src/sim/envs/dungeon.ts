import type { EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const DW_X = 0;
export const DW_Y = 1;
export const DW_HP = 2;
export const DW_LEVEL = 3;
export const DW_XP = 4;
export const DW_FOOD = 5;
export const DW_DEPTH = 6;
export const DW_TURN = 7;
export const DW_RUN = 8;
export const DW_SLOT = 9;
export const DW_RATIONS = 10;
const DW_PACK = 11;
export const DW_POISON = 13;
export const DW_CONFUSED = 14;
export const DW_BLIND = 15;
export const DW_FIRE = 16;
export const DW_COLD = 17;
export const DW_EVENT = 18;
export const DW_MAGNITUDE = 19;
export const DW_GOLD = 20;
const DW_PERM = 22;
const DW_MONSTERS = 24;
const DW_ITEMS = 25;
export const DW_STAIRS = 26;
export const DW_DEEPEST = 28;
const DW_WORDS = 29;

export const SIDE = 32;
const ITEM_SLOTS = 16;
const MONSTER_SLOTS = 24;
const CELL_WORDS = SIDE * SIDE / 8;
const VIEW_WORDS = CELL_WORDS + ITEM_SLOTS / 2 + MONSTER_SLOTS;

export const ROCK = 0;
export const WALL = 1;
export const FLOOR = 2;
export const CORRIDOR = 3;
export const DOOR = 4;
export const STAIRS = 5;

export const GOLD = 1;
export const RATION = 2;
export const POTION = 3;

export const EFFECTS = ['heal', 'extra heal', 'cure', 'fire ward', 'frost ward', 'poison', 'confusion', 'blindness'] as const;

export const EV_NONE = 0;
export const EV_HIT = 1;
export const EV_MISS = 2;
export const EV_KILL = 3;
export const EV_HURT = 4;
export const EV_GOLD = 5;
export const EV_RATION = 6;
export const EV_POTION = 7;
export const EV_EAT = 8;
export const EV_QUAFF = 9;
export const EV_DESCEND = EV_QUAFF + 8;
export const EV_REST = EV_DESCEND + 1;
export const EV_BUMP = EV_DESCEND + 2;
export const EV_STARVE = EV_DESCEND + 3;
export const EV_SICK = EV_DESCEND + 4;
export const EV_RUN = EV_DESCEND + 5;
export const EV_WIN = EV_DESCEND + 6;

export const ACT_USE = 4;
export const ACT_NEXT = 5;

export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;

export const AWAKE = 1;
export const GUARD = 2;

export interface Monster { x: number; y: number; type: number; flag: number; hp: number }
export interface Item { x: number; y: number; kind: number }

export const terrainAt = (view: Uint32Array, x: number, y: number): number => {
  const at = y * SIDE + x;
  return view[at >> 3]! >>> ((at & 7) << 2) & 7;
};

export const brightAt = (view: Uint32Array, x: number, y: number): boolean => {
  const at = y * SIDE + x;
  return (view[at >> 3]! >>> ((at & 7) << 2) & 8) !== 0;
};

export const open = (terrain: number): boolean => terrain >= FLOOR;

export function itemsOf(frame: EnvFrame): Item[] {
  const out: Item[] = [];
  for (let i = 0; i < frame.words[DW_ITEMS]!; i++) {
    const packed = frame.view[CELL_WORDS + (i >> 1)]! >>> ((i & 1) << 4) & 0xffff;
    out.push({ x: packed & 31, y: packed >>> 5 & 31, kind: packed >>> 10 & 15 });
  }
  return out;
}

export function monstersOf(frame: EnvFrame): Monster[] {
  const out: Monster[] = [];
  for (let m = 0; m < frame.words[DW_MONSTERS]!; m++) {
    const packed = frame.view[CELL_WORDS + ITEM_SLOTS / 2 + m]!;
    out.push({ x: packed & 31, y: packed >>> 5 & 31, type: packed >>> 10 & 15, flag: packed >>> 14 & 3, hp: packed >>> 16 & 255 });
  }
  return out;
}

export const potions = (words: Uint32Array, colour: number): number => words[DW_PACK + (colour >> 2)]! >>> ((colour & 3) << 3) & 255;

export const effectOf = (words: Uint32Array, colour: number): number => words[DW_PERM]! >>> (3 * colour) & 7;

export const most = (level: number): number => 8 + 4 * level;

export const turned = (prev: EnvFrame, next: EnvFrame): boolean => next.done !== 0 || next.words[DW_TURN] !== prev.words[DW_TURN];

export const floorKey = (frame: EnvFrame): number => frame.words[DW_RUN]! * 16 + frame.words[DW_DEPTH]!;

function checkLayout(env: EnvDescriptor): void {
  if (env.words !== DW_WORDS || env.view !== VIEW_WORDS) throw new Error(`${env.envId} records ${env.words} words and ${env.view} view words where the dungeon layout reads ${DW_WORDS} and ${VIEW_WORDS}`);
}

const CUED = new Set([EV_KILL, EV_GOLD, EV_POTION, EV_RATION, EV_DESCEND, ...Array.from({ length: 8 }, (_, e) => EV_QUAFF + e)]);

export const dungeonDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[0] = frame.words[DW_X]!;
    out[1] = frame.words[DW_Y]!;
  },
  geometry({ environment }, initial) {
    checkLayout(environment);
    if (initial.words.length !== DW_WORDS) throw new Error(`${environment.envId} state does not carry the dungeon words`);
    return { grid: { w: SIDE, h: SIDE, walls: new Uint8Array(SIDE * SIDE) } };
  },
  deriveEvents(prev, next): SimEvent[] {
    if (!turned(prev, next)) return NO_EVENTS;
    const event = next.words[DW_EVENT]!;
    if (next.done) return event === EV_WIN ? [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }] : [{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }];
    if (event === EV_DESCEND) return next.reward > 0 ? [{ type: 'goal' }, { type: 'cue', value: event }, { type: 'teleport' }] : [{ type: 'cue', value: event }, { type: 'teleport' }];
    return CUED.has(event) ? [{ type: 'cue', value: event }] : NO_EVENTS;
  },
};
