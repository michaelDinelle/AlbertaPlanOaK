import { constant, constantList, parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const MINATAR_BREAKOUT = 0;
export const MINATAR_INVADERS = 1;
export const MINATAR_FREEWAY = 2;
export const MINATAR_ASTERIX = 3;
export const MINATAR_SEAQUEST = 4;
export const MINATAR_AGENT_X = 0;
export const MINATAR_AGENT_Y = 1;
export const MINATAR_TARGET_X = 2;
export const MINATAR_TARGET_Y = 3;
export const MINATAR_LAST = [10, 54, 26, 15, 18] as const;
export const FREEWAY_CHICKEN = 4;
export const SEA_FISH = 0;
export const SEA_SUBS = 1;
export const SEA_DIVERS = 2;
export const SEA_SHOTS = 3;
export const SEA_BOMBS = 4;
export const SEA_VIEW = [0, 48, 96, 100, 104] as const;
export const SEA_COUNT = [19, 20, 21, 22, 23] as const;

export const u8 = (w: Uint32Array, i: number): number => (w[i >> 2]! >>> ((i & 3) << 3)) & 255;
export const s8 = (w: Uint32Array, i: number): number => (u8(w, i) << 24) >> 24;
export const bit = (w: Uint32Array, base: number, k: number): number => (w[base + (k >> 5)]! >>> (k & 31)) & 1;

const LAYOUT = [4, 14, 8, 5, 6] as const;
let side = 10;

export const minatarGame = (words: Uint32Array): number => LAYOUT.indexOf(words.length as (typeof LAYOUT)[number]);
export const minatarSide = (): number => side;

export const minatarDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) {
    const w = frame.words, g = minatarGame(w), low = side - 1;
    out.fill(-1, 0, 4);
    if (g === MINATAR_BREAKOUT) {
      out[MINATAR_AGENT_X] = u8(w, 4);
      out[MINATAR_AGENT_Y] = low;
      out[MINATAR_TARGET_X] = u8(w, 5);
      out[MINATAR_TARGET_Y] = u8(w, 6);
    } else if (g === MINATAR_INVADERS) {
      out[MINATAR_AGENT_X] = u8(w, 48);
      out[MINATAR_AGENT_Y] = low;
    } else if (g === MINATAR_FREEWAY) {
      out[MINATAR_AGENT_X] = FREEWAY_CHICKEN;
      out[MINATAR_AGENT_Y] = u8(w, 24);
    } else if (g === MINATAR_ASTERIX) {
      out[MINATAR_AGENT_X] = u8(w, 8);
      out[MINATAR_AGENT_Y] = u8(w, 9);
    } else if (g === MINATAR_SEAQUEST) {
      out[MINATAR_AGENT_X] = u8(w, 8);
      out[MINATAR_AGENT_Y] = u8(w, 9);
    }
  },
  geometry({ environment }) {
    const game = parameter(environment, 'game');
    if (constantList(environment, 'MINATAR_WORDS')[game] !== LAYOUT[game]) throw new Error(`${environment.envId} words differ from the decoder layout`);
    side = Math.round(1 / constant(environment, 'MINATAR_NINTH')) + 1;
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side) } };
  },
  deriveEvents(prev, next) {
    const g = minatarGame(next.words), events: SimEvent[] = [];
    if (next.reward > 0) events.push({ type: 'goal' });
    if (g === MINATAR_FREEWAY) {
      if (!next.done && next.reward === 0 && u8(prev.words, 24) !== side - 1 && u8(next.words, 24) === side - 1) events.push({ type: 'harm' });
    } else if (next.done) events.push({ type: 'harm' });
    if (g === MINATAR_SEAQUEST && !next.done) {
      const was = prev.words[1]! | 0, now = next.words[1]! | 0;
      if (now > was) events.push({ type: 'key', index: now - 1 });
      if (u8(prev.words, 9) > 0 && u8(next.words, 9) === 0) events.push({ type: 'resurface' });
    }
    if (next.done) events.push({ type: 'done' });
    return events.length ? events : NO_EVENTS;
  },
};
