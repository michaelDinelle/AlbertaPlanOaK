import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

const LC_MINE = 0;
const LC_THEIRS = 8;
const LC_AGENT_WORD = 16;
const LC_RIVAL_WORD = 17;
const LC_GAME_WORD = 18;
const LC_TICK_WORD = 19;
const LC_LAST_WORD = 20;
const LC_WORDS = 21;
const LC_SENSES = 16;
const LC_BOARD = 16;

export const LC_EMPTY = 0;
export const LC_AGENT = 1;
export const LC_RIVAL = 2;

export const LC_LEFT = 0;
export const LC_STRAIGHT = 1;
export const LC_RIGHT = 2;

export const LC_WON = 1;
export const LC_LOST = 2;
export const LC_DRAWN = 3;

export const LC_AGENT_X = 0;
export const LC_AGENT_Y = 1;
export const LC_AGENT_HEADING = 2;
export const LC_RIVAL_X = 3;
export const LC_RIVAL_Y = 4;
export const LC_RIVAL_HEADING = 5;
const LC_GAME_SLOT = 6;
const LC_TICK_SLOT = 7;
const LC_RESULT_SLOT = 8;

export interface Head { x: number; y: number; heading: number }

const headOf = (word: number): Head => ({ x: word & 15, y: (word >>> 4) & 15, heading: (word >>> 8) & 3 });

export const lcHead = (words: ArrayLike<number>, who: number): Head => headOf(words[who === LC_AGENT ? LC_AGENT_WORD : LC_RIVAL_WORD]!);
export const lcGame = (words: ArrayLike<number>): number => words[LC_GAME_WORD]!;
export const lcTick = (words: ArrayLike<number>): number => words[LC_TICK_WORD]!;
export const lcResult = (words: ArrayLike<number>): number => words[LC_LAST_WORD]! & 3;

export function lcTarget(words: ArrayLike<number>, who: number): { x: number; y: number } {
  const last = words[LC_LAST_WORD]!, shift = who === LC_AGENT ? 2 : 12;
  return { x: ((last >>> shift) & 31) - 1, y: ((last >>> (shift + 5)) & 31) - 1 };
}

export function lcOwner(words: ArrayLike<number>, x: number, y: number): number {
  const word = y >>> 1, bit = 1 << (x + ((y & 1) << 4));
  if (words[LC_MINE + word]! & bit) return LC_AGENT;
  if (words[LC_THEIRS + word]! & bit) return LC_RIVAL;
  return LC_EMPTY;
}

export function lcFacing(observations: ArrayLike<number>, who: number): { x: number; y: number } {
  const hx = observations[7]!, hy = observations[8]!;
  if (who === LC_AGENT) return { x: hx, y: hy };
  const along = observations[11]!, across = observations[12]!;
  return { x: along * hx - across * hy, y: along * hy + across * hx };
}

export const lcAction = (actions: ArrayLike<number>): number => {
  const control = actions[0]!;
  return control >= 0 && control < 3 ? Math.trunc(control) : LC_STRAIGHT;
};

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const DRAWN: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const lightCyclesDecoder: EnvDecoder = {
  slots: LC_RESULT_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, agent = lcHead(words, LC_AGENT), rival = lcHead(words, LC_RIVAL);
    out[LC_AGENT_X] = agent.x;
    out[LC_AGENT_Y] = agent.y;
    out[LC_AGENT_HEADING] = agent.heading;
    out[LC_RIVAL_X] = rival.x;
    out[LC_RIVAL_Y] = rival.y;
    out[LC_RIVAL_HEADING] = rival.heading;
    out[LC_GAME_SLOT] = lcGame(words);
    out[LC_TICK_SLOT] = lcTick(words);
    out[LC_RESULT_SLOT] = lcResult(words);
  },
  geometry(descriptor, initial) {
    const env = descriptor.environment, w = parameter(env, 'width'), h = parameter(env, 'height');
    if (env.words !== LC_WORDS) throw new Error(`${env.envId} records ${env.words} words where a light-cycle arena packs into ${LC_WORDS}`);
    if (env.obs !== LC_SENSES) throw new Error(`${env.envId} observes ${env.obs} values where a light cycle senses ${LC_SENSES}`);
    if (w > LC_BOARD || h > LC_BOARD) throw new Error(`${env.envId} declares a ${w}×${h} arena, larger than the ${LC_BOARD}×${LC_BOARD} its walls pack into`);
    let claimed = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (lcOwner(initial.words, x, y) !== LC_EMPTY) claimed++;
    if (lcTick(initial.words) !== 0 || claimed !== 2) throw new Error(`${env.envId} opens with ${claimed} lit cells after ${lcTick(initial.words)} ticks`);
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const result = lcResult(next.words);
    return (result === LC_WON ? WON : result === LC_LOST ? LOST : DRAWN) as SimEvent[];
  },
};
