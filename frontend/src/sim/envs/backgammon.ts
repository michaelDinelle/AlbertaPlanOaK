import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const BG_MINE = 0;
export const BG_THEIRS = 4;
export const BG_DICE = 8;
export const BG_HIGH_LEGAL = 9;
export const BG_LOW_LEGAL = 10;
export const BG_CURSOR = 11;
export const BG_PHASE = 12;
export const BG_GAME = 13;
export const BG_TURN = 14;
export const BG_LAST = 15;
export const BG_HITS = 16;
export const BG_MEMBER = 17;
const BG_WORDS = 18;

export const BG_POINTS = 24;
export const BG_BAR = 25;
export const BG_OFF = 26;
export const BG_MEN = 15;
export const BG_NONE = 0xffff;
const BG_OPPONENT_DICE = 1 << 10;

export const BG_AGENT_THINK = 0;
export const BG_AGENT_MOVE = 1;
export const BG_OPPONENT_THINK = 2;
export const BG_OPPONENT_PLAY = 3;

export const BG_DOWN = 0;
export const BG_UP = 1;
export const BG_HIGH = 2;
export const BG_LOW = 3;
export const BG_JUMP = 4;

export const bgSlot = (side: number, at: number): number => side * BG_OFF + at - 1;
export const BG_CURSOR_SLOT = 2 * BG_OFF;
export const BG_HIGH_SLOT = BG_CURSOR_SLOT + 1;
export const BG_LOW_SLOT = BG_CURSOR_SLOT + 2;
export const BG_USED_SLOT = BG_CURSOR_SLOT + 3;
export const BG_OWNER_SLOT = BG_CURSOR_SLOT + 4;
export const BG_PHASE_SLOT = BG_CURSOR_SLOT + 5;
export const BG_GAME_SLOT = BG_CURSOR_SLOT + 6;
export const BG_TURN_SLOT = BG_CURSOR_SLOT + 7;
export const BG_FROM_SLOT = BG_CURSOR_SLOT + 8;
export const BG_TO_SLOT = BG_CURSOR_SLOT + 9;
export const BG_MEMBER_SLOT = BG_CURSOR_SLOT + 10;

const nibble = (words: ArrayLike<number>, base: number, point: number): number => (words[base + ((point - 1) >> 3)]! >>> (4 * ((point - 1) & 7))) & 15;

export function bgMen(words: ArrayLike<number>, side: number, at: number): number {
  if (at === BG_OFF) {
    let aboard = 0;
    for (let p = 1; p <= BG_BAR; p++) aboard += nibble(words, side ? BG_THEIRS : BG_MINE, p);
    return BG_MEN - aboard;
  }
  return side === 0 ? nibble(words, BG_MINE, at) : nibble(words, BG_THEIRS, at === BG_BAR ? BG_BAR : BG_BAR - at);
}

export const bgDoubles = (dice: number): boolean => (dice & 7) === ((dice >>> 3) & 7);
export const bgOpponentDice = (dice: number): boolean => (dice & BG_OPPONENT_DICE) !== 0;
export const bgRemaining = (dice: number): number => {
  const state = (dice >>> 6) & 15;
  return bgDoubles(dice) ? 4 - state : 2 - (state & 1) - ((state >>> 1) & 1);
};

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }]);

export const backgammonDecoder: EnvDecoder = {
  slots: BG_MEMBER_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, dice = words[BG_DICE]!, last = words[BG_LAST]!;
    for (let side = 0; side < 2; side++) for (let at = 1; at <= BG_OFF; at++) out[bgSlot(side, at)] = bgMen(words, side, at);
    out[BG_CURSOR_SLOT] = words[BG_CURSOR]!;
    out[BG_HIGH_SLOT] = dice & 7;
    out[BG_LOW_SLOT] = (dice >>> 3) & 7;
    out[BG_USED_SLOT] = (dice >>> 6) & 15;
    out[BG_OWNER_SLOT] = bgOpponentDice(dice) ? 1 : 0;
    out[BG_PHASE_SLOT] = words[BG_PHASE]!;
    out[BG_GAME_SLOT] = words[BG_GAME]!;
    out[BG_TURN_SLOT] = words[BG_TURN]!;
    out[BG_FROM_SLOT] = last === BG_NONE ? -1 : last & 255;
    out[BG_TO_SLOT] = last === BG_NONE ? -1 : last >>> 8;
    out[BG_MEMBER_SLOT] = words[BG_MEMBER]!;
  },
  geometry(descriptor, initial) {
    if (descriptor.environment.words !== BG_WORDS) throw new Error(`${descriptor.environment.envId} records ${descriptor.environment.words} words where a backgammon position packs into ${BG_WORDS}`);
    for (let side = 0; side < 2; side++) {
      let men = 0;
      for (let at = 1; at <= BG_BAR; at++) men += bgMen(initial.words, side, at);
      if (men !== BG_MEN) throw new Error(`${descriptor.environment.envId} opens with ${men} checkers on side ${side}, not ${BG_MEN}`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : LOST) as SimEvent[];
  },
};
