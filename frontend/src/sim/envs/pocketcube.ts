import { constantList, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const CUBE_SLOTS = 8;
export const CUBE_TWIST = CUBE_SLOTS;
export const CUBE_MOVES = 2 * CUBE_SLOTS;
const TURN_BYTES = 16;

export type Vec3 = readonly [number, number, number];
export type Axis = 0 | 1 | 2;
export interface Facelet { axis: Axis; side: number }
export interface CubeTurn { axis: Axis; side: number; quarter: number; label: string }
export interface CubeModel { turns: CubeTurn[]; fixed: number[] }

export const CORNERS: readonly Vec3[] = [[1, 1, 1], [-1, 1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, 1], [-1, -1, 1], [1, -1, -1], [-1, -1, -1]];
const FACE_NAMES = [['L', 'R'], ['D', 'U'], ['B', 'F']] as const;

const byteOf = (words: ArrayLike<number>, slot: number): number => (words[slot >> 2]! >>> ((slot & 3) << 3)) & 255;
export const cubieAt = (words: ArrayLike<number>, slot: number): number => byteOf(words, slot) & 15;
export const twistAt = (words: ArrayLike<number>, slot: number): number => byteOf(words, slot) >>> 4;
export const movesOf = (words: ArrayLike<number>): number => words[2]!;
export const cubeKey = (actions: ArrayLike<number>, keys: number): number => {
  const pressed = actions[0]!;
  return pressed >= 0 && pressed < keys ? Math.trunc(pressed) : 0;
};

export const faceName = (axis: Axis, side: number): string => FACE_NAMES[axis][side > 0 ? 1 : 0];

export const faceletsOf = (slot: number): [Facelet, Facelet, Facelet] => {
  const [x, y, z] = CORNERS[slot]!, up: Facelet = { axis: 1, side: y }, across: Facelet = { axis: 0, side: x }, deep: Facelet = { axis: 2, side: z };
  return x * y * z > 0 ? [up, across, deep] : [up, deep, across];
};

function rotated(v: Vec3, axis: Axis, quarter: number): [number, number, number] {
  const [x, y, z] = v;
  if (axis === 0) return quarter > 0 ? [x, -z, y] : [x, z, -y];
  if (axis === 1) return quarter > 0 ? [z, y, -x] : [-z, y, x];
  return quarter > 0 ? [-y, x, z] : [y, -x, z];
}

const slotOf = (p: Vec3): number => CORNERS.findIndex(([x, y, z]) => x === p[0] && y === p[1] && z === p[2]);

const faceletVector = (f: Facelet): [number, number, number] => [f.axis === 0 ? f.side : 0, f.axis === 1 ? f.side : 0, f.axis === 2 ? f.side : 0];

function matches(row: readonly number[], axis: Axis, side: number, quarter: number): boolean {
  for (let slot = 0; slot < CUBE_SLOTS; slot++) {
    const corner = CORNERS[slot]!;
    if (corner[axis] !== side) {
      if (row[slot] !== slot || row[CUBE_SLOTS + slot] !== 0) return false;
      continue;
    }
    const to = slotOf(rotated(corner, axis, quarter));
    if (row[to] !== slot) return false;
    const [rx, ry, rz] = rotated(faceletVector(faceletsOf(slot)[0]), axis, quarter);
    const twist = faceletsOf(to).findIndex((f) => f.axis === (rx !== 0 ? 0 : ry !== 0 ? 1 : 2) && f.side === rx + ry + rz);
    if (row[CUBE_SLOTS + to] !== twist << 4) return false;
  }
  return true;
}

export function pocketCubeModel(env: EnvDescriptor): CubeModel {
  const table = constantList(env, 'POCKETCUBE_TURNS');
  if (table.length !== env.actions * TURN_BYTES) throw new Error(`${env.envId} lists ${table.length} turn bytes for ${env.actions} keys`);
  const turns: CubeTurn[] = [];
  for (let key = 0; key < env.actions; key++) {
    const row = table.slice(key * TURN_BYTES, (key + 1) * TURN_BYTES);
    let found: CubeTurn | null = null;
    for (const axis of [0, 1, 2] as const) for (const side of [1, -1]) for (const quarter of [1, -1]) {
      if (found || !matches(row, axis, side, quarter)) continue;
      found = { axis, side, quarter, label: `${faceName(axis, side)}${quarter === side ? '′' : ''}` };
    }
    if (!found) throw new Error(`${env.envId} key ${key} is no quarter turn of a 2×2×2 cube`);
    turns.push(found);
  }
  const fixed = CORNERS.map((_, slot) => slot).filter((slot) => turns.every((turn) => CORNERS[slot]![turn.axis] !== turn.side));
  return { turns, fixed };
}

const SOLVED: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]);
const LAPSED: readonly SimEvent[] = Object.freeze([{ type: 'teleport' }, { type: 'done' }]);

export const pocketCubeDecoder: EnvDecoder = {
  slots: CUBE_MOVES + 1,
  discrete: true,
  decode(frame, out) {
    for (let slot = 0; slot < CUBE_SLOTS; slot++) {
      out[slot] = cubieAt(frame.words, slot);
      out[CUBE_TWIST + slot] = twistAt(frame.words, slot);
    }
    out[CUBE_MOVES] = movesOf(frame.words);
  },
  geometry({ environment: env }, initial) {
    if (env.words !== 3) throw new Error(`${env.envId} records ${env.words} words where a cube packs into 3`);
    if (env.obs !== 2 * (CUBE_SLOTS - 1) + 1) throw new Error(`${env.envId} observes ${env.obs} values where ${CUBE_SLOTS - 1} moving corners show ${2 * (CUBE_SLOTS - 1) + 1}`);
    const { fixed } = pocketCubeModel(env);
    let seen = 0, twist = 0;
    for (let slot = 0; slot < CUBE_SLOTS; slot++) {
      seen |= 1 << cubieAt(initial.words, slot);
      twist += twistAt(initial.words, slot);
      if (twistAt(initial.words, slot) > 2) throw new Error(`${env.envId} opens with corner ${slot} twisted ${twistAt(initial.words, slot)} times`);
    }
    if (seen !== (1 << CUBE_SLOTS) - 1 || twist % 3) throw new Error(`${env.envId} opens on no reachable cube`);
    for (const slot of fixed) if (cubieAt(initial.words, slot) !== slot || twistAt(initial.words, slot)) throw new Error(`${env.envId} opens with its unturned corner ${slot} out of place`);
    return {};
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? SOLVED : LAPSED) as SimEvent[];
  },
};
