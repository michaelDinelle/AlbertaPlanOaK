import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const ALCHEMY_STONES = 3;
export const ALCHEMY_POTIONS = 12;
export const ALCHEMY_CAULDRON = ALCHEMY_POTIONS;
export const ALCHEMY_CONTAINERS = ALCHEMY_POTIONS + 1;
export const ALCHEMY_VERTICES = 8;
export const ALCHEMY_WORDS = 12;
export const ALCHEMY_SPENT_STONE = 8;
export const ALCHEMY_SPENT_POTION = 64;

const POTION_AT = 1;
const LOOK_AT = 4;
export const ALCHEMY_EDGES = 6;
export const ALCHEMY_CURSOR = 8;
export const ALCHEMY_TICK = 9;
export const ALCHEMY_TRIAL = 10;
export const ALCHEMY_EPISODE = 11;

const byteOf = (words: ArrayLike<number>, base: number, i: number): number => (words[base + (i >>> 2)]! >>> ((i & 3) << 3)) & 255;

export const alchemyStone = (words: ArrayLike<number>, s: number): number => byteOf(words, 0, s);
export const alchemyPotion = (words: ArrayLike<number>, p: number): number => byteOf(words, POTION_AT, p);
export const alchemyFeature = (words: ArrayLike<number>, vertex: number, feature: number): number => ((byteOf(words, LOOK_AT, vertex) >>> (2 * feature)) & 3) - 1;
export const alchemyEdge = (words: ArrayLike<number>, axis: number, lower: number): boolean => ((words[ALCHEMY_EDGES]! >>> (8 * axis + lower)) & 1) === 1;
export const alchemyAction = (actions: ArrayLike<number>): number => {
  const c = actions[0]!;
  return c >= 0 && c < 6 ? Math.trunc(c) : 0;
};

const WON: SimEvent[] = [{ type: 'goal' }];
const LOST: SimEvent[] = [{ type: 'harm' }];
const ENDED: SimEvent[] = [{ type: 'done' }];

export const alchemyDecoder: EnvDecoder = {
  slots: 8,
  discrete: true,
  decode(frame, out) {
    const w = frame.words;
    out[0] = w[ALCHEMY_CURSOR]!;
    out[1] = w[ALCHEMY_TICK]!;
    out[2] = w[ALCHEMY_TRIAL]!;
    out[3] = w[ALCHEMY_EPISODE]!;
    for (let s = 0; s < ALCHEMY_STONES; s++) out[4 + s] = alchemyStone(w, s);
    let spent = 0;
    for (let p = 0; p < ALCHEMY_POTIONS; p++) if (alchemyPotion(w, p) & ALCHEMY_SPENT_POTION) spent |= 1 << p;
    out[7] = spent;
  },
  geometry({ environment: env }, initial) {
    if (env.words !== ALCHEMY_WORDS || initial.words.length !== ALCHEMY_WORDS) throw new Error(`${env.envId} records ${env.words} words where alchemy packs ${ALCHEMY_WORDS}`);
    if (env.obs !== 4 * ALCHEMY_STONES + 6) throw new Error(`${env.envId} observes ${env.obs} values where ${ALCHEMY_STONES} stones show ${4 * ALCHEMY_STONES + 6}`);
    if (env.actions !== 6) throw new Error(`${env.envId} offers ${env.actions} actions where alchemy has 6`);
    if (initial.words[ALCHEMY_CURSOR]! >= ALCHEMY_CONTAINERS) throw new Error(`${env.envId} opens with its cursor past the cauldron`);
    return {};
  },
  deriveEvents(_prev, next) {
    const events = next.reward > 0 ? WON : next.reward < 0 ? LOST : NO_EVENTS;
    if (!next.done) return events;
    return events === NO_EVENTS ? ENDED : [...events, ...ENDED];
  },
};
