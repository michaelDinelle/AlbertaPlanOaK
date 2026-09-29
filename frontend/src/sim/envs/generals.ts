import { constant, constantList, parameter, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type GridCell, type SimEvent } from '../types';

export const GW = { CURSOR: 0, MODE: 1, PHASE: 2, TICK: 3, PENDING: 4, SEAT: 5, SWEEP: 6, GAME: 7, QUIET: 8, COUNT: 9 } as const;

export const PEAK = 1;
export const CITY = 2;
export const CAPITAL = 3;
export const NOBODY = 8;
const CODE = { FOG: 0, OBSTACLE: 1, PEAK: 2, CITY: 3, EDGE: 4, EMPTY_OWN: 5, EMPTY_FOE: 6 } as const;
export const KEYS = 6;
export const JUMP_KEY = 5;
export const STEP_X = [0, 0, -1, 1] as const;
export const STEP_Y = [-1, 1, 0, 0] as const;

const PACK = 32;
const MOVE_BIT = 1 << 13;
const TAIL = 10;

export const GEN_CURSOR_X = 0;
export const GEN_CURSOR_Y = 1;

export interface GeneralsShape {
  side: number;
  players: number;
  cities: number;
  substeps: number;
  turn: number;
  round: number;
  reach: number;
  codes: number[];
}

export interface Order { cell: number; dir: number; split: boolean }

export const cellX = (cell: number): number => cell & (PACK - 1);
export const cellY = (cell: number): number => cell >>> 5;
export const armyOf = (word: number): number => (word & 0xffff) + ((word >>> 16) & 0x3ff) * 0x10000;
export const terrainOf = (word: number): number => (word >>> 26) & 3;
export const ownerOf = (word: number): number => word >>> 28;
export const keyOf = (frame: EnvFrame): number => {
  const a = frame.actions[0]!;
  return a >= 0 && a < KEYS ? Math.trunc(a) : 0;
};

export function orderOf(words: ArrayLike<number>): Order | null {
  const code = words[GW.PENDING]!;
  if (!code) return null;
  if (!(code & MOVE_BIT)) throw new Error(`Generals pending word ${code} lacks its move bit`);
  const cell = code & 1023;
  return { cell, dir: (code >>> 10) & 3, split: ((code >>> 12) & 1) === 1 };
}

export function heeded(view: Uint32Array, side: number, seat: number, order: Order): boolean {
  const x = cellX(order.cell) + STEP_X[order.dir]!, y = cellY(order.cell) + STEP_Y[order.dir]!, from = view[cellY(order.cell) * side + cellX(order.cell)]!;
  return x >= 0 && y >= 0 && x < side && y < side && terrainOf(view[y * side + x]!) !== PEAK && ownerOf(from) === seat && armyOf(from) >= 2;
}

export function generalsShape(env: EnvDescriptor): GeneralsShape {
  const side = parameter(env, 'side'), players = parameter(env, 'players'), cities = parameter(env, 'cities');
  const reach = constant(env, 'GENERALS_REACH'), codes = constantList(env, 'GENERALS_CODES'), span = 2 * reach + 1;
  if (env.words !== GW.COUNT) throw new Error(`${env.envId} records ${env.words} state words where the decoder reads ${GW.COUNT}`);
  if (env.view !== side * side) throw new Error(`${env.envId} views ${env.view} cells on a ${side}-cell side`);
  if (env.obs !== 2 * span * span + TAIL) throw new Error(`${env.envId} observes ${env.obs} values where two ${span}×${span} cursor planes show ${2 * span * span + TAIL}`);
  if (codes.length !== CODE.EMPTY_FOE + 1) throw new Error(`${env.envId} names ${codes.length} patch codes where the decoder reads ${CODE.EMPTY_FOE + 1}`);
  if (env.actions !== KEYS) throw new Error(`${env.envId} offers ${env.actions} keys where the decoder reads ${KEYS}`);
  if (players < 2 || players > NOBODY) throw new Error(`${env.envId} seats ${players} players`);
  return { side, players, cities, substeps: constant(env, 'GENERALS_SUBSTEPS'), turn: constant(env, 'GENERALS_TURN'), round: constant(env, 'GENERALS_ROUND'), reach, codes };
}

export function sightOf(view: Uint32Array, side: number, seat: number, out: Uint8Array): Uint8Array {
  out.fill(0);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    if (ownerOf(view[y * side + x]!) !== seat) continue;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const u = x + dx, v = y + dy;
      if (u >= 0 && v >= 0 && u < side && v < side) out[v * side + u] = 1;
    }
  }
  return out;
}

export function patchCell(cursor: number, reach: number, index: number, out: GridCell): GridCell {
  const span = 2 * reach + 1;
  out.x = cellX(cursor) + (index % span) - reach;
  out.y = cellY(cursor) + Math.floor(index / span) - reach;
  return out;
}

function seenAt(view: Uint32Array, shape: GeneralsShape, seat: number, sight: Uint8Array, { x, y }: GridCell): number {
  const { side, codes } = shape;
  if (x < 0 || y < 0 || x >= side || y >= side) return codes[CODE.EDGE]!;
  const word = view[y * side + x]!, kind = terrainOf(word), owner = ownerOf(word);
  if (sight[y * side + x] !== 1) return codes[kind === PEAK || kind === CITY ? CODE.OBSTACLE : CODE.FOG]!;
  if (owner !== NOBODY) return armyOf(word) ? (owner === seat ? armyOf(word) : -armyOf(word)) : codes[owner === seat ? CODE.EMPTY_OWN : CODE.EMPTY_FOE]!;
  return kind === PEAK ? codes[CODE.PEAK]! : kind === CITY ? codes[CODE.CITY]! : 0;
}

function builtAt(view: Uint32Array, side: number, sight: Uint8Array, { x, y }: GridCell): number {
  if (x < 0 || y < 0 || x >= side || y >= side || sight[y * side + x] !== 1) return 0;
  const word = view[y * side + x]!, kind = terrainOf(word);
  if (kind === CITY && ownerOf(word) === NOBODY) return armyOf(word);
  return kind === CITY ? 1 : kind === CAPITAL ? 2 : 0;
}

export function checkPatch(frame: EnvFrame, shape: GeneralsShape, sight: Uint8Array): void {
  const cursor = frame.words[GW.CURSOR]!, seat = frame.words[GW.SEAT]!, span = 2 * shape.reach + 1, cell: GridCell = { x: 0, y: 0 };
  for (let i = 0; i < span * span; i++) {
    const shown = frame.observations[i]!, board = seenAt(frame.view, shape, seat, sight, patchCell(cursor, shape.reach, i, cell));
    if (shown !== board) throw new Error(`Generals step ${frame.step} patch cell (${cell.x}, ${cell.y}) reads ${shown} though the board shows ${board}`);
    const shownBuilt = frame.observations[span * span + i]!, boardBuilt = builtAt(frame.view, shape.side, sight, cell);
    if (shownBuilt !== boardBuilt) throw new Error(`Generals step ${frame.step} structure cell (${cell.x}, ${cell.y}) reads ${shownBuilt} though the board shows ${boardBuilt}`);
  }
}

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const ENDED: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const generalsDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    const cursor = frame.words[GW.CURSOR]!;
    out[GEN_CURSOR_X] = cellX(cursor);
    out[GEN_CURSOR_Y] = cellY(cursor);
  },
  geometry({ environment }, initial) {
    const shape = generalsShape(environment), { side, players } = shape, words = initial.words, seat = words[GW.SEAT]!, cursor = words[GW.CURSOR]!;
    if (seat >= players) throw new Error(`${environment.envId} seats the agent at ${seat} of ${players}`);
    if (cellX(cursor) >= side || cellY(cursor) >= side) throw new Error(`${environment.envId} opens with its cursor off the board`);
    const home = initial.view[cellY(cursor) * side + cellX(cursor)]!;
    if (terrainOf(home) !== CAPITAL || ownerOf(home) !== seat) throw new Error(`${environment.envId} opens with its cursor away from its general`);
    checkPatch(initial, shape, sightOf(initial.view, side, seat, new Uint8Array(side * side)));
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST : ENDED) as SimEvent[];
  },
};
