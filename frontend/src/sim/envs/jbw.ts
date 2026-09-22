import { philox4x32_10 } from '../philox';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordInt } from '../words';

export const JBW_EMPTY = 0;
export const JBW_BANANA = 1;
export const JBW_ONION = 2;
export const JBW_JELLY = 3;
export const JBW_WALL = 4;
export const JBW_TREE = 5;
export const JBW_TRUFFLE = 6;
export const JBW_X = 0;
export const JBW_Y = 1;
export const JBW_HEADING = 2;
export const JBW_BANANAS = 3;
export const JBW_ONIONS = 4;
export const JBW_JELLIES = 5;
export const JBW_TRUFFLES = 6;

export const JBW_BLOCK = 16;
export const JBW_BLOCK_ITEMS = 4;

export const JBW_COLOR: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [0.96, 0.88, 0.2],
  [0.68, 0.01, 0.99],
  [0.82, 0.27, 0.2],
  [0.2, 0.47, 0.67],
  [0, 0.47, 0.06],
  [0.42, 0.24, 0.13],
];

export function jbwBlocks(item: number): boolean {
  return item === JBW_WALL || item === JBW_TREE;
}

export function jbwCollectible(item: number): boolean {
  return item === JBW_BANANA || item === JBW_ONION || item === JBW_JELLY || item === JBW_TRUFFLE;
}

export function jbwWallLine(seed: number, bx: number, by: number): readonly [boolean, number] | null {
  const r = jbwField(seed, bx, by, 80);
  if (((r[2] ?? 0) & 7) !== 0) return null;
  return ((r[3] ?? 0) & 1) === 0
    ? [true, by * JBW_BLOCK + ((r[1] ?? 0) & 15)]
    : [false, bx * JBW_BLOCK + ((r[0] ?? 0) & 15)];
}

export function jbwScentIndex(dx: number, dy: number): number {
  let x = dx < 0 ? -dx : dx;
  let y = dy < 0 ? -dy : dy;
  if (y > x) {
    const t = x;
    x = y;
    y = t;
  }
  return (x * (x + 1)) / 2 + y;
}

const floorDiv = (x: number, d: number): number => Math.floor(x / d);

export function jbwField(seed: number, x: number, y: number, stream: number): readonly number[] {
  return philox4x32_10(seed, 3, x >>> 0, y >>> 0, stream, 0);
}

export function jbwBlockField(seed: number, bx: number, by: number): readonly number[] {
  return jbwField(seed, bx, by, 16);
}

export function jbwCandidate(
  block: readonly number[],
  bx: number,
  by: number,
  slot: number,
): readonly [number, number] {
  const bits = ((block[0] ?? 0) >>> (8 * slot)) & 255;
  return [bx * 16 + (bits & 15), by * 16 + ((bits >>> 4) & 15)];
}

export function jbwTypeOf(block: readonly number[], ecology: readonly number[], slot: number): number {
  const roll = ((block[1] ?? 0) >>> (8 * slot)) & 255;
  if (((ecology[1] ?? 0) & 3) === 0) return (roll & 31) === 0 ? JBW_TRUFFLE : JBW_TREE;
  const roll24 =
    roll |
    ((((block[2] ?? 0) >>> (8 * slot)) & 255) << 8) |
    ((((block[3] ?? 0) >>> (8 * slot)) & 255) << 16);
  const pick = roll24 % 3;
  return pick === 0 ? JBW_BANANA : pick === 1 ? JBW_JELLY : JBW_ONION;
}

export function jbwWallAt(seed: number, x: number, y: number): boolean {
  const bx = floorDiv(x, 16);
  const by = floorDiv(y, 16);
  const r = jbwField(seed, bx, by, 80);
  if (((r[2] ?? 0) & 7) !== 0) return false;
  return ((r[3] ?? 0) & 1) === 0
    ? y === by * 16 + ((r[1] ?? 0) & 15)
    : x === bx * 16 + ((r[0] ?? 0) & 15);
}

export function jbwItemAt(seed: number, x: number, y: number): number {
  if (jbwWallAt(seed, x, y)) return JBW_WALL;
  const bx = floorDiv(x, 16);
  const by = floorDiv(y, 16);
  const block = jbwBlockField(seed, bx, by);
  for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
    const p = jbwCandidate(block, bx, by, slot);
    if (p[0] !== x || p[1] !== y) continue;
    return jbwTypeOf(block, jbwField(seed, floorDiv(x, 64), floorDiv(y, 64), 64), slot);
  }
  return JBW_EMPTY;
}


const items = ['banana', 'onion', 'jellyBean', 'truffle'] as const;
const INVENTORY_WORD = 4;

export const jbwDecoder: EnvDecoder = {
  slots: 7,
  discrete: true,
  sequential: true,
  decode(frame, out) {
    out[JBW_X] = wordInt(frame.words[0]!);
    out[JBW_Y] = wordInt(frame.words[1]!);
    out[JBW_HEADING] = frame.words[2]!;
    for (let i = 0; i < 4; i++) out[JBW_BANANAS + i] = frame.words[INVENTORY_WORD + i]!;
  },
  geometry: () => ({}),
  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    for (let i = 0; i < items.length; i++) {
      if (next.words[INVENTORY_WORD + i]! > prev.words[INVENTORY_WORD + i]!) {
        return [{ type: 'collection', item: items[i]!, reward: next.reward as -1 | 0 | 1, x: wordInt(next.words[0]!), y: wordInt(next.words[1]!) }];
      }
    }
    return NO_EVENTS;
  },
};
