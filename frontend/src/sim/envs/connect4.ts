import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const C4_COLUMNS = 7;
export const C4_ROWS = 6;
export const C4_CELLS = C4_COLUMNS * C4_ROWS;
const C4_STRIDE = C4_ROWS + 1;

const C4_MINE = 0;
const C4_THEIRS = 2;
const C4_PROPOSALS = 4;
const C4_SHOWN = 5;
const C4_GAME = 6;
const C4_MEMBER = 7;
const C4_LAST = 8;
const C4_WORDS = 9;

const C4_EMPTY = 0;
export const C4_AGENT = 1;
export const C4_RIVAL = 2;

export const C4_PLAY = 0;
export const C4_NEXT = 1;

export const C4_WON = 1;
export const C4_LOST = 2;
export const C4_DRAWN = 3;

export const c4Slot = (column: number, row: number): number => column * C4_ROWS + row;
export const c4Observed = (column: number, row: number): number => C4_COLUMNS * row + column;
export const C4_PROPOSED_SLOT = C4_CELLS;
const C4_SHOWN_SLOT = C4_CELLS + 1;
const C4_OFFERED_SLOT = C4_CELLS + 2;
const C4_GAME_SLOT = C4_CELLS + 3;
const C4_MEMBER_SLOT = C4_CELLS + 4;
const C4_AGENT_COLUMN_SLOT = C4_CELLS + 5;
const C4_RIVAL_COLUMN_SLOT = C4_CELLS + 6;
const C4_RESULT_SLOT = C4_CELLS + 7;

export function c4Owner(words: ArrayLike<number>, column: number, row: number): number {
  const bit = C4_STRIDE * column + row, word = bit >>> 5, mask = 1 << (bit & 31);
  if (words[C4_MINE + word]! & mask) return C4_AGENT;
  if (words[C4_THEIRS + word]! & mask) return C4_RIVAL;
  return C4_EMPTY;
}

export function c4Height(words: ArrayLike<number>, column: number): number {
  let row = 0;
  while (row < C4_ROWS && c4Owner(words, column, row) !== C4_EMPTY) row++;
  return row;
}

export function c4Discs(words: ArrayLike<number>): number {
  let discs = 0;
  for (let column = 0; column < C4_COLUMNS; column++) discs += c4Height(words, column);
  return discs;
}

export const c4Offered = (words: ArrayLike<number>): number => words[C4_PROPOSALS]! >>> 28;
export const c4Shown = (words: ArrayLike<number>): number => words[C4_SHOWN]!;
const c4Proposal = (words: ArrayLike<number>, index: number): number => (words[C4_PROPOSALS]! >>> (4 * index)) & 15;
export const c4Proposed = (words: ArrayLike<number>): number => c4Proposal(words, c4Shown(words));
export const c4Game = (words: ArrayLike<number>): number => words[C4_GAME]!;
export const c4Member = (words: ArrayLike<number>): number => words[C4_MEMBER]!;
export const c4AgentColumn = (words: ArrayLike<number>): number => (words[C4_LAST]! & 7) - 1;
export const c4RivalColumn = (words: ArrayLike<number>): number => ((words[C4_LAST]! >>> 3) & 7) - 1;
export const c4Result = (words: ArrayLike<number>): number => (words[C4_LAST]! >>> 6) & 3;
export const c4Action = (actions: ArrayLike<number>): number => {
  const control = actions[0]!;
  return control >= 0 && control < 2 ? Math.trunc(control) : C4_NEXT;
};

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const DRAWN: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const connectFourDecoder: EnvDecoder = {
  slots: C4_RESULT_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words;
    for (let column = 0; column < C4_COLUMNS; column++) for (let row = 0; row < C4_ROWS; row++) out[c4Slot(column, row)] = c4Owner(words, column, row);
    out[C4_PROPOSED_SLOT] = c4Proposed(words);
    out[C4_SHOWN_SLOT] = c4Shown(words);
    out[C4_OFFERED_SLOT] = c4Offered(words);
    out[C4_GAME_SLOT] = c4Game(words);
    out[C4_MEMBER_SLOT] = c4Member(words);
    out[C4_AGENT_COLUMN_SLOT] = c4AgentColumn(words);
    out[C4_RIVAL_COLUMN_SLOT] = c4RivalColumn(words);
    out[C4_RESULT_SLOT] = c4Result(words);
  },
  geometry(descriptor, initial) {
    const env = descriptor.environment;
    if (env.words !== C4_WORDS) throw new Error(`${env.envId} records ${env.words} words where a Connect Four position packs into ${C4_WORDS}`);
    if (env.obs !== C4_CELLS) throw new Error(`${env.envId} observes ${env.obs} values where a ${C4_COLUMNS}-column board shows ${C4_CELLS}`);
    for (let column = 0; column < C4_COLUMNS; column++) if (c4Height(initial.words, column)) throw new Error(`${env.envId} opens with discs in column ${column + 1}`);
    if (c4Offered(initial.words) !== C4_COLUMNS || c4Shown(initial.words) !== 0) throw new Error(`${env.envId} opens without proposing every column once`);
    return { grid: { w: C4_COLUMNS, h: C4_ROWS, walls: new Uint8Array(C4_CELLS) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : next.reward < 0 ? LOST : DRAWN) as SimEvent[];
  },
};
