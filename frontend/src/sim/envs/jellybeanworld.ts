import { POSE_LEN } from '../record';
import { philox4x32_10 } from '../philox';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

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
export const JBW_DEPLETION_LO = 7;
export const JBW_DEPLETION_HI = 8;

const floorDiv = (x: number, d: number): number => Math.floor(x / d);

export function jbwField(seed: number, x: number, y: number, stream: number): readonly number[] {
  return philox4x32_10(seed, 3, x >>> 0, y >>> 0, stream, 0);
}

export function jbwCandidate(seed: number, bx: number, by: number): readonly [number, number] {
  const r = jbwField(seed, bx, by, 16);
  return [bx * 16 + ((r[0] ?? 0) & 15), by * 16 + ((r[1] ?? 0) & 15)];
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
  const p = jbwCandidate(seed, bx, by);
  if (p[0] !== x || p[1] !== y) return JBW_EMPTY;
  const r = jbwField(seed, bx, by, 16);
  const ecology = jbwField(seed, floorDiv(x, 64), floorDiv(y, 64), 64);
  if (((ecology[1] ?? 0) & 3) === 0) return ((r[3] ?? 0) & 31) === 0 ? JBW_TRUFFLE : JBW_TREE;
  if (((ecology[0] ?? 0) & 3) === 0) return ((r[2] ?? 0) & 1) === 0 ? JBW_BANANA : JBW_JELLY;
  return ((r[2] ?? 0) & 3) === 0 ? JBW_ONION : JBW_EMPTY;
}

export function jbwDepleted(pose: ArrayLike<number>, dx: number, dy: number): boolean {
  let bit = (dy + 3) * 7 + dx + 3;
  if (dy > 0 || (dy === 0 && dx > 0)) bit--;
  const mask = bit < 24 ? pose[JBW_DEPLETION_LO] : pose[JBW_DEPLETION_HI];
  return ((Math.trunc(mask ?? 0) >>> (bit % 24)) & 1) !== 0;
}

const inventories = [JBW_BANANAS, JBW_ONIONS, JBW_JELLIES, JBW_TRUFFLES] as const;
const items = ['banana', 'onion', 'jellyBean', 'truffle'] as const;

export const jellyBeanWorldDecoder: EnvDecoder<number> = {
  envId: 'JellyBeanWorld-cont-v0',
  seedGeometry: true,
  initStatic: (seed) => seed,
  geometry: () => ({}),
  initFrame(): EnvFrame {
    return { step: -1, done: 0, pose: new Float32Array(POSE_LEN) };
  },
  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    for (let i = 0; i < inventories.length; i++) {
      const slot = inventories[i] ?? 0;
      if ((next.pose[slot] ?? 0) > (prev.pose[slot] ?? 0)) {
        const item = items[i] ?? 'banana';
        const sign: 1 | -1 = ((next.step / 100000) & 1) === 0 ? 1 : -1;
        const reward: -1 | 0 | 1 = item === 'jellyBean' ? sign : item === 'onion' ? (sign === 1 ? -1 : 1) : 0;
        return [{ type: 'collection', item, reward }];
      }
    }
    return NO_EVENTS;
  },
};
