import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const HANOI_PEGS = 3;
export const HANOI_TARGET = 3;
export const HANOI_CODES = 4;

export interface HanoiMove { disk: number; from: number; to: number }

export const hanoiPegOf = (words: ArrayLike<number>, disk: number): number => ((words[1]! >>> disk) & 1) | (((words[2]! >>> disk) & 1) << 1);

export const hanoiTarget = (words: ArrayLike<number>): number => words[HANOI_TARGET]!;

export const hanoiHeight = (words: ArrayLike<number>, peg: number): number => {
  let mask = words[peg]! >>> 0, count = 0;
  for (; mask; mask &= mask - 1) count++;
  return count;
};

export const hanoiTop = (words: ArrayLike<number>, peg: number): number => {
  const mask = words[peg]! >>> 0;
  return mask ? 31 - Math.clz32(mask & -mask) : -1;
};

export function hanoiMoved(before: ArrayLike<number>, after: ArrayLike<number>): HanoiMove | null {
  let from = -1, to = -1, bit = 0;
  for (let peg = 0; peg < HANOI_PEGS; peg++) {
    const changed = (before[peg]! ^ after[peg]!) >>> 0;
    if (!changed) continue;
    bit = changed;
    if (after[peg]! & changed) to = peg; else from = peg;
  }
  if (from < 0) return null;
  if (to < 0 || (bit & (bit - 1)) !== 0) throw new Error('A Hanoi step moved more than one disk');
  return { disk: 31 - Math.clz32(bit), from, to };
}

export const hanoiCode = (actions: ArrayLike<number>, keys: number): number => {
  const pressed = actions[0]!;
  return pressed >= 0 && pressed < keys ? Math.trunc(pressed) : 0;
};

export const hanoiRoute = (codes: number, code: number): { from: number; to: number } => {
  const move = (codes >>> (4 * code)) & 15, from = move >>> 1, side = move & 1;
  return { from, to: side + (side >= from ? 1 : 0) };
};

export function hanoiDistance(words: ArrayLike<number>, disks: number): number {
  let goal = hanoiTarget(words), distance = 0;
  for (let disk = disks - 1; disk >= 0; disk--) {
    const peg = hanoiPegOf(words, disk);
    if (peg === goal) continue;
    distance += 2 ** disk;
    goal = HANOI_PEGS - peg - goal;
  }
  return distance;
}

const SOLVED: readonly SimEvent[] = Object.freeze([{ type: 'goal' }]);

export const hanoiDecoder: EnvDecoder = {
  slots: HANOI_TARGET + 1,
  discrete: true,
  decode(frame, out) {
    for (let k = 0; k <= HANOI_TARGET; k++) out[k] = frame.words[k]!;
  },
  geometry({ environment: env }, initial) {
    const disks = parameter(env, 'disks'), remaps = parameter(env, 'remapRate') > 0, words = remaps ? HANOI_CODES + 1 : HANOI_CODES;
    if (env.words !== words) throw new Error(`${env.envId} records ${env.words} words where a ${remaps ? 'remapped' : 'fixed'} Hanoi keyboard packs into ${words}`);
    if (env.obs !== disks + 2) throw new Error(`${env.envId} observes ${env.obs} values where ${disks} disks show ${disks + 2}`);
    if (env.actions !== HANOI_PEGS * (HANOI_PEGS - 1)) throw new Error(`${env.envId} offers ${env.actions} keys where ${HANOI_PEGS} pegs pair into ${HANOI_PEGS * (HANOI_PEGS - 1)} moves`);
    const all = 2 ** disks - 1, [left, middle, right] = [initial.words[0]!, initial.words[1]!, initial.words[2]!];
    if ((left | middle | right) !== all || (left & middle) || (left & right) || (middle & right)) throw new Error(`${env.envId} opens without its ${disks} disks on exactly one peg each`);
    if (hanoiTarget(initial.words) >= HANOI_PEGS) throw new Error(`${env.envId} opens aiming at peg ${hanoiTarget(initial.words)}`);
    return {};
  },
  deriveEvents(prev, next) {
    const remapped = next.words[HANOI_CODES] !== prev.words[HANOI_CODES];
    if (!remapped) return next.done ? (SOLVED as SimEvent[]) : NO_EVENTS;
    const cue: SimEvent = { type: 'cue', value: next.words[HANOI_CODES]! };
    return next.done ? [{ type: 'goal' }, cue] : [cue];
  },
};
