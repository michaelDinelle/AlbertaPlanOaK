import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const BG_MINE = 0;
export const BG_THEIRS = 4;
export const BG_DICE = 8;
export const BG_PATH = 9;
export const BG_PROPOSAL = 10;
export const BG_PROPOSALS = 11;
export const BG_GAME = 12;
export const BG_TURN = 13;
export const BG_REPLY = 14;
export const BG_REPLY_DICE = 15;
export const BG_MEMBER = 16;
const BG_WORDS = 17;

export const BG_POINTS = 24;
export const BG_BAR = 25;
export const BG_OFF = 26;
export const BG_MEN = 15;
export const BG_STEPS = 4;

export const BG_PLAY = 0;
export const BG_NEXT = 1;

export const bgSlot = (side: number, at: number): number => side * BG_OFF + at - 1;
export const BG_HIGH_SLOT = 2 * BG_OFF;
export const BG_LOW_SLOT = BG_HIGH_SLOT + 1;
export const BG_REPLY_HIGH_SLOT = BG_HIGH_SLOT + 2;
export const BG_REPLY_LOW_SLOT = BG_HIGH_SLOT + 3;
export const BG_PROPOSAL_SLOT = BG_HIGH_SLOT + 4;
export const BG_PROPOSALS_SLOT = BG_HIGH_SLOT + 5;
export const BG_GAME_SLOT = BG_HIGH_SLOT + 6;
export const BG_TURN_SLOT = BG_HIGH_SLOT + 7;
export const BG_MEMBER_SLOT = BG_HIGH_SLOT + 8;
export const bgMoveSlot = (side: number, index: number): number => BG_MEMBER_SLOT + 1 + 2 * (side * BG_STEPS + index);
const BG_SLOTS = bgMoveSlot(2, 0);

const nibble = (words: ArrayLike<number>, base: number, point: number): number => (words[base + ((point - 1) >> 3)]! >>> (4 * ((point - 1) & 7))) & 15;

const mine = new Int8Array(BG_BAR + 1);
const theirs = new Int8Array(BG_BAR + 1);

const hopSource = (path: number, index: number): number => (path >>> (6 * index)) & 31;

function hopTo(path: number, dice: number, index: number): number {
  const move = (path >>> (6 * index)) & 63, source = move & 31, die = (dice >>> (3 * (move >>> 5))) & 7;
  return source > die ? source - die : 0;
}

export const bgHops = (path: number): number => path >>> 24;

export function bgFrom(path: number, index: number, side: number): number {
  const source = hopSource(path, index);
  return source === BG_BAR || !side ? source : BG_BAR - source;
}

export function bgTo(path: number, dice: number, index: number, side: number): number {
  const to = hopTo(path, dice, index);
  return to === 0 ? BG_OFF : side ? BG_BAR - to : to;
}

export function bgCounts(words: ArrayLike<number>, out: { [index: number]: number }): void {
  for (let p = 1; p <= BG_BAR; p++) {
    mine[p] = nibble(words, BG_MINE, p);
    theirs[p] = nibble(words, BG_THEIRS, p);
  }
  const path = words[BG_PATH]!, dice = words[BG_DICE]!;
  for (let i = 0; i < bgHops(path); i++) {
    const to = hopTo(path, dice, i);
    mine[hopSource(path, i)]!--;
    if (!to) continue;
    mine[to]!++;
    if (theirs[BG_BAR - to] !== 1) continue;
    theirs[BG_BAR - to] = 0;
    theirs[BG_BAR]!++;
  }
  for (let side = 0; side < 2; side++) {
    let aboard = 0;
    for (let at = 1; at <= BG_BAR; at++) {
      const n = side ? theirs[at === BG_BAR ? BG_BAR : BG_BAR - at]! : mine[at]!;
      out[bgSlot(side, at)] = n;
      aboard += n;
    }
    out[bgSlot(side, BG_OFF)] = BG_MEN - aboard;
  }
}

const WON: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]);
const LOST: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }]);

export const backgammonDecoder: EnvDecoder = {
  slots: BG_SLOTS,
  discrete: true,
  decode(frame, out) {
    const words = frame.words, dice = words[BG_DICE]!, reply = words[BG_REPLY_DICE]!;
    bgCounts(words, out);
    out[BG_HIGH_SLOT] = dice & 7;
    out[BG_LOW_SLOT] = (dice >>> 3) & 7;
    out[BG_REPLY_HIGH_SLOT] = reply & 7;
    out[BG_REPLY_LOW_SLOT] = (reply >>> 3) & 7;
    out[BG_PROPOSAL_SLOT] = words[BG_PROPOSAL]!;
    out[BG_PROPOSALS_SLOT] = words[BG_PROPOSALS]!;
    out[BG_GAME_SLOT] = words[BG_GAME]!;
    out[BG_TURN_SLOT] = words[BG_TURN]!;
    out[BG_MEMBER_SLOT] = words[BG_MEMBER]!;
    for (let side = 0; side < 2; side++) for (let i = 0; i < BG_STEPS; i++) {
      const path = side ? words[BG_REPLY]! : words[BG_PATH]!, roll = side ? reply : dice, present = i < bgHops(path), at = bgMoveSlot(side, i);
      out[at] = present ? bgFrom(path, i, side) : -1;
      out[at + 1] = present ? bgTo(path, roll, i, side) : -1;
    }
  },
  geometry(descriptor, initial) {
    if (descriptor.environment.words !== BG_WORDS) throw new Error(`${descriptor.environment.envId} records ${descriptor.environment.words} words where a backgammon position packs into ${BG_WORDS}`);
    for (let side = 0; side < 2; side++) {
      let men = 0;
      for (let p = 1; p <= BG_BAR; p++) men += nibble(initial.words, side ? BG_THEIRS : BG_MINE, p);
      if (men !== BG_MEN) throw new Error(`${descriptor.environment.envId} opens with ${men} checkers on side ${side}, not ${BG_MEN}`);
    }
    if (initial.words[BG_PROPOSALS]! < 1) throw new Error(`${descriptor.environment.envId} opens without a proposal to show`);
    return {};
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return (next.reward > 0 ? WON : LOST) as SimEvent[];
  },
};
