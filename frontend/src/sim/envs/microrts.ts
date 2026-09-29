import { constantList, constantRecords, parameter, type CatalogRecord, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const MW = { GAME: 0, SIDE: 1, PHASE: 2, LENGTH: 3, SELECTED: 4, MODE: 5, SWEEP: 6, CURSOR: 7, FIFO: 8, QUEUED: 9, EVENT: 10, FOE: 11, MEMBER: 12, COUNT: 13 } as const;

export const RESOURCE = 0;
export const BASE = 1;
export const BARRACKS = 2;
export const WORKER = 3;
export const LIGHT = 4;
export const HEAVY = 5;
export const RANGED = 6;
export const KINDS = 7;

export const NONE = 0;
export const MOVE = 1;
export const HARVEST = 2;
export const RETURN = 3;
export const PRODUCE = 4;
export const ATTACK = 5;
export const IDLE = 6;

export const LOST = 1;
export const KILLED = 2;
export const PRODUCED = 3;
export const REJECTED = 4;
export const RETURNED = 5;

export const DOCTRINE = 5;
export const NEUTRAL = 2;
export const NOBODY = 255;
export const KEYS = 6;
export const MODE_KEY = 5;
export const STEP_X = [0, 1, 0, -1] as const;
export const STEP_Y = [-1, 0, 1, 0] as const;

export const MRTS_CURSOR = 0;

export const REACH = 2;
export const SPAN = 2 * REACH + 1;
export const PATCH_AT = 5;
export const NEAREST_AT = PATCH_AT + SPAN * SPAN;
export const SWEEP_AT = NEAREST_AT + 2;
export const ORE_GLYPH = 1;
export const UNSEEN_GLYPH = 1.25;
export const OFF_MAP_GLYPH = 1.5;

const OBSERVED = SWEEP_AT + 9;
const HEADER = 2;
const DISC_ROWS = 11;
const MAPS: Record<number, string> = { 8: 'MICRORTS_MAP8', 16: 'MICRORTS_MAP16' };

export interface UnitType {
  cost: number;
  hp: number;
  damage: number;
  range: number;
  sight: number;
  move: number;
  attack: number;
  harvest: number;
  mobile: boolean;
  fighter: boolean;
  harvester: boolean;
  stockpile: boolean;
  makes: number;
  build: number;
}

export interface MicroShape {
  width: number;
  partial: boolean;
  opponent: number;
  substeps: number;
  maxCycles: number;
  capacity: number;
  types: UnitType[];
  disc: number[][];
  richest: number;
}

export interface Unit {
  cell: number;
  x: number;
  y: number;
  kind: number;
  owner: number;
  act: number;
  hp: number;
  carry: number;
  arg: number;
}

const field = (record: CatalogRecord, key: string, env: EnvDescriptor): number => {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${env.envId} unit type lacks ${key}`);
  return value;
};

export function microShape(env: EnvDescriptor): MicroShape {
  const width = parameter(env, 'width');
  if (width !== 8 && width !== 16) throw new Error(`${env.envId} lays a ${width}-cell board where the maps cover 8 and 16`);
  if (env.words !== MW.COUNT) throw new Error(`${env.envId} records ${env.words} state words where the decoder reads ${MW.COUNT}`);
  if (env.obs !== OBSERVED) throw new Error(`${env.envId} observes ${env.obs} values where the command card shows ${OBSERVED}`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} keys where the decoder reads ${KEYS}`);
  const capacity = env.view - HEADER;
  if (capacity < 1 || capacity > width * width) throw new Error(`${env.envId} views ${capacity} units on ${width * width} cells`);
  const types = constantRecords(env, 'MICRORTS_TYPES').map((record) => ({
    cost: field(record, 'cost', env), hp: field(record, 'hp', env), damage: field(record, 'damage', env), range: field(record, 'range', env),
    sight: field(record, 'sight', env), move: field(record, 'move', env), attack: field(record, 'attack', env), harvest: field(record, 'harvest', env),
    mobile: field(record, 'mobile', env) !== 0, fighter: field(record, 'fighter', env) !== 0, harvester: field(record, 'harvester', env) !== 0,
    stockpile: field(record, 'stockpile', env) !== 0, makes: field(record, 'makes', env), build: field(record, 'build', env),
  }));
  if (types.length !== KINDS) throw new Error(`${env.envId} lists ${types.length} unit types where the view packs ${KINDS}`);
  const flat = constantList(env, 'MICRORTS_DISC'), rows = flat.length / DISC_ROWS;
  if (!Number.isInteger(rows) || types.some((type) => type.sight >= rows)) throw new Error(`${env.envId} sight discs do not cover every unit's sight`);
  const disc = Array.from({ length: rows }, (_, s) => flat.slice(s * DISC_ROWS, s * DISC_ROWS + 2 * s + 1));
  const map = constantRecords(env, MAPS[width]!);
  return {
    width, partial: parameter(env, 'partial') !== 0, opponent: parameter(env, 'opponent'), substeps: parameter(env, 'substeps'), maxCycles: parameter(env, 'maxCycles'),
    capacity, types, disc, richest: Math.max(...map.map((placement) => field(placement, 'carry', env))),
  };
}

export const timeOf = (frame: EnvFrame): number => frame.view[0]!;
export const goldOf = (frame: EnvFrame, player: number): number => (frame.view[1]! >>> (16 * player)) & 0xffff;

export function unitOf(word: number, width: number): Unit {
  const cell = word & 255;
  return {
    cell, x: cell % width, y: Math.floor(cell / width), kind: (word >>> 8) & 7, owner: (word >>> 11) & 3, act: (word >>> 13) & 7,
    hp: (word >>> 16) & 15, carry: (word >>> 20) & 31, arg: (word >>> 25) & 63,
  };
}

export function unitsOf(frame: EnvFrame, shape: MicroShape): Unit[] {
  const out: Unit[] = [];
  for (let i = 0; i < shape.capacity; i++) {
    const word = frame.view[HEADER + i]!;
    if (word === 0) break;
    out.push(unitOf(word, shape.width));
  }
  return out;
}

export const directionOf = (unit: Unit): number => unit.arg & 3;
export const makeOf = (unit: Unit): number => unit.arg >>> 2;

export function aimOf(unit: Unit): { x: number; y: number } {
  if (unit.act === ATTACK) return { x: unit.x + (unit.arg & 7) - 3, y: unit.y + (unit.arg >>> 3) - 3 };
  const d = directionOf(unit);
  return { x: unit.x + STEP_X[d]!, y: unit.y + STEP_Y[d]! };
}

export function etaOf(unit: Unit, types: readonly UnitType[]): number {
  const type = types[unit.kind]!;
  switch (unit.act) {
    case MOVE: return type.move;
    case HARVEST: return type.harvest;
    case RETURN: return type.move;
    case PRODUCE: return types[makeOf(unit)]!.build;
    case ATTACK: return type.attack;
    default: return 0;
  }
}

export function keyOf(frame: EnvFrame): number {
  const pressed = frame.actions[0]!;
  return pressed >= 0 && pressed < KEYS ? Math.trunc(pressed) : 0;
}

export function sightOf(units: readonly Unit[], side: number, shape: MicroShape, out: Uint8Array): Uint8Array {
  const width = shape.width, span = 2 ** width;
  if (!shape.partial) return out.fill(1);
  out.fill(0);
  for (const unit of units) {
    if (unit.owner !== side) continue;
    const sight = shape.types[unit.kind]!.sight, rows = shape.disc[sight]!;
    for (let y = Math.max(0, unit.y - sight); y <= Math.min(width - 1, unit.y + sight); y++) {
      const bits = Math.floor(rows[y - unit.y + sight]! * 2 ** unit.x / 65536) % span;
      for (let x = 0; x < width; x++) if (Math.floor(bits / 2 ** x) % 2 === 1) out[y * width + x] = 1;
    }
  }
  return out;
}

function checkOpening(env: EnvDescriptor, shape: MicroShape, initial: EnvFrame): void {
  const units = unitsOf(initial, shape), map = constantRecords(env, MAPS[shape.width]!);
  if (timeOf(initial) !== 0 || initial.words[MW.PHASE] !== 0) throw new Error(`${env.envId} opens at cycle ${timeOf(initial)} phase ${initial.words[MW.PHASE]}`);
  if (units.length !== map.length) throw new Error(`${env.envId} opens with ${units.length} units where its map places ${map.length}`);
  map.forEach((placement, i) => {
    const unit = units[i]!, kind = field(placement, 'kind', env);
    if (unit.kind !== kind || unit.owner !== field(placement, 'owner', env) || unit.cell !== field(placement, 'cell', env) || unit.carry !== field(placement, 'carry', env) || unit.hp !== shape.types[kind]!.hp || unit.act !== IDLE)
      throw new Error(`${env.envId} opening unit ${i} reads kind ${unit.kind} owner ${unit.owner} cell ${unit.cell} carry ${unit.carry} hp ${unit.hp} though its map places kind ${kind} at cell ${field(placement, 'cell', env)}`);
  });
  const cursor = initial.words[MW.CURSOR]!, selected = initial.words[MW.SELECTED]!;
  if (selected === NOBODY || !units.some((unit) => unit.cell === cursor && unit.owner === initial.words[MW.SIDE])) throw new Error(`${env.envId} opens with its cursor on cell ${cursor}, away from its own units`);
}

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST_GAME: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const ENDED: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const micrortsDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[MRTS_CURSOR] = frame.words[MW.CURSOR]!;
  },
  geometry({ environment }, initial) {
    const shape = microShape(environment);
    checkOpening(environment, shape, initial);
    return { grid: { w: shape.width, h: shape.width, walls: new Uint8Array(shape.width * shape.width) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST_GAME : ENDED) as SimEvent[];
  },
};
