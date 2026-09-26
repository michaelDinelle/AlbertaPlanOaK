import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const HOMEOSTASIS_KINDS = 3;
export const HOMEOSTASIS_GROUP = 8;
export const HOMEOSTASIS_SLOTS = HOMEOSTASIS_KINDS * HOMEOSTASIS_GROUP;
export const HOMEOSTASIS_EMPTY = 0xff;
export const HOMEOSTASIS_LEVELS = 8;
export const HOMEOSTASIS_AGENT = 12;
export const HOMEOSTASIS_WORDS = 13;

export const HOMEOSTASIS_X = 0;
export const HOMEOSTASIS_Y = 1;
export const HOMEOSTASIS_LIKED = 2;
export const HOMEOSTASIS_CRAVED = 3;
export const HOMEOSTASIS_ITEMS = 4;

export const homeostasisCode = (frame: EnvFrame, slot: number): number => (frame.words[slot >>> 2]! >>> ((slot & 3) * 8)) & 0xff;

export const homeostasisKind = (slot: number): number => Math.floor(slot / HOMEOSTASIS_GROUP) + 1;

export const homeostasisLevel = (frame: EnvFrame, nutrient: number): number =>
  (frame.words[HOMEOSTASIS_LEVELS + 2 * nutrient + 1]! | 0) * 4294967296 + frame.words[HOMEOSTASIS_LEVELS + 2 * nutrient]!;

export const homeostasisAgent = (frame: EnvFrame): number => frame.words[HOMEOSTASIS_AGENT]!;

export function homeostasisMeal(prev: EnvFrame, next: EnvFrame): number {
  const at = homeostasisAgent(next);
  for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) if (homeostasisCode(prev, slot) === at && homeostasisCode(next, slot) !== at) return slot;
  return -1;
}

const MEALS: readonly (readonly SimEvent[])[][] = [1, 2, 3].map((value) => [
  Object.freeze([{ type: 'cue', value }, { type: 'harm' }] as SimEvent[]),
  Object.freeze([{ type: 'cue', value }] as SimEvent[]),
  Object.freeze([{ type: 'cue', value }, { type: 'goal' }] as SimEvent[]),
]);

export const homeostasisDecoder: EnvDecoder = {
  slots: HOMEOSTASIS_ITEMS + HOMEOSTASIS_SLOTS,
  discrete: true,
  decode(frame, out) {
    const at = homeostasisAgent(frame);
    out[HOMEOSTASIS_X] = at & 15;
    out[HOMEOSTASIS_Y] = at >>> 4;
    out[HOMEOSTASIS_LIKED] = homeostasisLevel(frame, 0);
    out[HOMEOSTASIS_CRAVED] = homeostasisLevel(frame, 1);
    for (let slot = 0; slot < HOMEOSTASIS_SLOTS; slot++) {
      const code = homeostasisCode(frame, slot);
      out[HOMEOSTASIS_ITEMS + slot] = code === HOMEOSTASIS_EMPTY ? -1 : code;
    }
  },
  geometry({ environment }, initial) {
    const side = constant(environment, 'HOMEOSTASIS_SIDE'), perKind = constant(environment, 'HOMEOSTASIS_PER_TYPE');
    const start = constant(environment, 'HOMEOSTASIS_START') * constant(environment, 'HOMEOSTASIS_SCALE');
    if (initial.words.length !== HOMEOSTASIS_WORDS || environment.actions !== 4 || environment.box !== 0) throw new Error(`${environment.envId} is not a ${HOMEOSTASIS_WORDS}-word, four-move homeostasis state`);
    if (side > 16 || perKind > HOMEOSTASIS_GROUP) throw new Error(`${environment.envId} does not pack its cells and kinds into bytes`);
    const at = homeostasisAgent(initial);
    if (at > 0xff || (at & 15) >= side || at >>> 4 >= side) throw new Error(`${environment.envId} starts its forager off the ${side}×${side} torus`);
    for (let slot = 0; slot < 4 * HOMEOSTASIS_LEVELS; slot++) {
      const code = homeostasisCode(initial, slot), used = slot < HOMEOSTASIS_SLOTS && slot % HOMEOSTASIS_GROUP < perKind;
      if (used ? code === HOMEOSTASIS_EMPTY || (code & 15) >= side || code >>> 4 >= side || code === at : code !== HOMEOSTASIS_EMPTY) throw new Error(`${environment.envId} item slot ${slot} does not follow the ${HOMEOSTASIS_KINDS} kinds × ${perKind} layout`);
    }
    if (homeostasisLevel(initial, 0) !== start || homeostasisLevel(initial, 1) !== start) throw new Error(`${environment.envId} does not start both levels at ${start}`);
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(prev, next) {
    const slot = homeostasisMeal(prev, next);
    if (slot < 0) return NO_EVENTS;
    return MEALS[homeostasisKind(slot) - 1]![next.reward < 0 ? 0 : next.reward > 0 ? 2 : 1]! as SimEvent[];
  },
};
