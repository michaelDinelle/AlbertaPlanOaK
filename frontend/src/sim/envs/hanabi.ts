import { constant, constantList, parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const HANABI_COLOURS = 5;
export const HANABI_RANKS = 5;
export const HANABI_SEATS = 16;
export const HANABI_PLAYERS = 4;

export const HANABI_NEXT = 0;
export const HANABI_PLAY = 2;

export const HANABI_PLAYED = 1;
export const HANABI_MISFIRED = 2;
export const HANABI_DISCARDED = 3;
export const HANABI_TOLD_COLOUR = 4;
export const HANABI_TOLD_RANK = 5;

const CURSOR_SLOT = 0;
const CLOCK_SLOT = 1;
const INFO_SLOT = 2;
const LIFE_SLOT = 3;
const DECK_SLOT = 4;
const FINAL_SLOT = 5;
const TURN_SLOT = 6;
const GAME_SLOT = 7;
const HEIGHT_SLOT = 8;
const SCORE_SLOT = HEIGHT_SLOT + HANABI_COLOURS;

const FIREWORKS = HANABI_SEATS;
const INFO = FIREWORKS + 1;
const LIFE = INFO + 1;
const DECK = LIFE + 1;
const DISCARDS = DECK + 1;
const DISCARDS_HIGH = DISCARDS + 1;
const GONE = DISCARDS_HIGH + 1;
const LAST_COPY = GONE + 1;
const TOUCHED = LAST_COPY + 1;
const CLOCK = TOUCHED + 1;
const CURSOR = CLOCK + 1;
const FINAL = CURSOR + 1;
const TURN = FINAL + 1;
const GAME = TURN + 1;
const ENDED = GAME + 1;
const MOVES = ENDED + 1;
const WORDS = MOVES + HANABI_PLAYERS;
const SEEN = 11;
const RECALL = 5;
const ACTIONS = 6;
const INFORMATION = 8;
const PLAUSIBLE = 0x1ffffff;
const NONE = 0x3e000000;
const CLUED = 0x40000000;
const FORCED = 0x20000000;
const MARKED = 0x80000000;
const ONES = 0x108421;
const ENDED_LIFE = 15;

export const hanabiEmpty = (slot: number): boolean => slot === NONE;
export const hanabiId = (slot: number): number => (slot >>> 25) & 31;
export const hanabiColourOf = (id: number): number => Math.floor(id / HANABI_RANKS);
export const hanabiRankOf = (id: number): number => id % HANABI_RANKS;
export const hanabiClued = (slot: number): boolean => (slot & CLUED) !== 0;
export const hanabiMarked = (slot: number): boolean => slot >>> 31 === 1;

export function hanabiColours(slot: number): number {
  const p = slot & PLAUSIBLE;
  let mask = 0;
  for (let c = 0; c < HANABI_COLOURS; c++) if ((p >>> (HANABI_RANKS * c)) & 31) mask |= 1 << c;
  return mask;
}

export function hanabiRanks(slot: number): number {
  const p = slot & PLAUSIBLE;
  return (p | (p >>> 5) | (p >>> 10) | (p >>> 15) | (p >>> 20)) & 31;
}

export const hanabiHeight = (words: ArrayLike<number>, colour: number): number => (words[FIREWORKS]! >>> (3 * colour)) & 7;
export function hanabiScore(words: ArrayLike<number>): number {
  let score = 0;
  for (let c = 0; c < HANABI_COLOURS; c++) score += hanabiHeight(words, c);
  return score;
}
export const hanabiInfo = (words: ArrayLike<number>): number => words[INFO]!;
export const hanabiLife = (words: ArrayLike<number>): number => words[LIFE]!;
export const hanabiDeck = (words: ArrayLike<number>): number => words[DECK]!;
export const hanabiDiscarded = (words: ArrayLike<number>, id: number): number => (words[id < 16 ? DISCARDS : DISCARDS_HIGH]! >>> (2 * (id & 15))) & 3;
export const hanabiGone = (words: ArrayLike<number>, id: number): boolean => ((words[GONE]! >>> id) & 1) === 1;
export const hanabiLastCopy = (words: ArrayLike<number>, id: number): boolean => ((words[LAST_COPY]! >>> id) & 1) === 1;
export const hanabiTouched = (words: ArrayLike<number>, position: number): boolean => ((words[TOUCHED]! >>> position) & 1) === 1;
export const hanabiClock = (words: ArrayLike<number>): number => words[CLOCK]!;
export const hanabiCursor = (words: ArrayLike<number>): number => words[CURSOR]!;
export const hanabiFinal = (words: ArrayLike<number>): number => words[FINAL]!;
export const hanabiTurn = (words: ArrayLike<number>): number => words[TURN]!;
export const hanabiGame = (words: ArrayLike<number>): number => words[GAME]!;
export const hanabiMove = (words: ArrayLike<number>, seat: number): number => words[MOVES + seat]!;
export const hanabiMoveSeat = (record: number): number => record & 3;
export const hanabiMoveKind = (record: number): number => (record >>> 2) & 7;
export const hanabiMoveAt = (record: number): number => (record >>> 5) & 7;
export const hanabiMoveCard = (record: number): number => (record >>> 8) & 31;
export const hanabiMoveTouched = (record: number): number => (record >>> 13) & 0xffff;
export const hanabiMoveForced = (record: number): boolean => (record & FORCED) !== 0;

export function hanabiLatest(words: ArrayLike<number>): number {
  for (let seat = HANABI_PLAYERS - 1; seat >= 0; seat--) if (hanabiMove(words, seat)) return hanabiMove(words, seat);
  return 0;
}

export function hanabiEnding(before: ArrayLike<number>, after: ArrayLike<number>): Uint32Array {
  const out = Uint32Array.from(before);
  out[FIREWORKS] = after[ENDED]! & ((1 << ENDED_LIFE) - 1);
  out[LIFE] = after[ENDED]! >>> ENDED_LIFE;
  return out;
}

export function hanabiAction(actions: ArrayLike<number>): number {
  const control = actions[0]!;
  return control >= 0 && control < ACTIONS ? Math.trunc(control) : HANABI_NEXT;
}

export const hanabiMoved = (prev: EnvFrame, next: EnvFrame): boolean => next.done !== 0 || next.words[TURN] !== prev.words[TURN];

function playable(words: ArrayLike<number>): number {
  let mask = 0;
  for (let c = 0; c < HANABI_COLOURS; c++) {
    const height = hanabiHeight(words, c), lost = (words[GONE]! >>> (HANABI_RANKS * c)) & 31 & (31 << height);
    if (height < (lost ? 31 - Math.clz32(lost & -lost) : HANABI_RANKS)) mask |= 1 << (HANABI_RANKS * c + height);
  }
  return mask;
}

function bury(words: Uint32Array, id: number, copies: readonly number[]): void {
  const count = hanabiDiscarded(words, id) + 1, bit = 1 << id, field = id < 16 ? DISCARDS : DISCARDS_HIGH, total = copies[id % HANABI_RANKS]!;
  words[field] = words[field]! + (1 << (2 * (id & 15)));
  if (count + 1 === total) words[LAST_COPY] = words[LAST_COPY]! | bit;
  if (count === total) {
    words[LAST_COPY] = words[LAST_COPY]! & ~bit;
    words[GONE] = words[GONE]! | bit;
  }
}

function tell(words: Uint32Array, record: number, hand: number, focus: boolean, m: number): void {
  const kind = hanabiMoveKind(record), value = hanabiMoveCard(record), base = hanabiMoveAt(record) * hand, lit = playable(words);
  const group = kind === HANABI_TOLD_COLOUR ? 31 << (HANABI_RANKS * value) : ONES << value;
  let touched = 0, clued = 0, chop = -1;
  words[INFO] = words[INFO]! - 1;
  for (let s = 0; s < hand && words[base + s] !== NONE; s++) {
    const slot = words[base + s]!;
    if (slot & CLUED) clued |= 1 << s;
    else if (chop < 0) chop = s;
    if ((group >>> hanabiId(slot)) & 1) {
      touched |= 1 << s;
      words[base + s] = (slot & (~PLAUSIBLE | group)) | CLUED;
    } else words[base + s] = slot & ~group;
  }
  if (touched !== hanabiMoveTouched(record)) throw new Error(`Hanabi step ${m} replays a hint touching ${touched.toString(2)} where the recording touched ${hanabiMoveTouched(record).toString(2)}`);
  words[TOUCHED] = touched << base;
  if (!focus) return;
  const fresh = touched & ~clued, aim = chop >= 0 && (touched >>> chop) & 1 ? chop : 31 - Math.clz32(fresh ? fresh : touched);
  if (words[base + aim]! & lit) words[base + aim] = words[base + aim]! | MARKED;
}

function spend(words: Uint32Array, seat: number, record: number, after: ArrayLike<number>, players: number, hand: number, copies: readonly number[], m: number): void {
  const kind = hanabiMoveKind(record), at = hanabiMoveAt(record), id = hanabiMoveCard(record), base = seat * hand;
  if (hanabiId(words[base + at]!) !== id) throw new Error(`Hanabi step ${m} replays seat ${seat} spending slot ${at + 1}, which does not hold card ${id}`);
  if (kind === HANABI_PLAYED) {
    words[FIREWORKS] = words[FIREWORKS]! + (1 << (3 * hanabiColourOf(id)));
    if (hanabiRankOf(id) === HANABI_RANKS - 1 && words[INFO]! < INFORMATION) words[INFO] = words[INFO]! + 1;
  } else {
    bury(words, id, copies);
    if (kind === HANABI_MISFIRED) words[LIFE] = words[LIFE]! - 1;
    else words[INFO] = words[INFO]! + 1;
  }
  const touched = words[TOUCHED]!, field = touched >>> base, lower = (1 << at) - 1;
  for (let s = at; s + 1 < hand; s++) words[base + s] = words[base + s + 1]!;
  const left = words[DECK]!;
  if (left) {
    words[DECK] = left - 1;
    if (left === 1) words[FINAL] = players + 1;
  }
  words[base + hand - 1] = left ? (hanabiId(after[base + hand - 1]!) << 25) | PLAUSIBLE : NONE;
  words[TOUCHED] = (touched & ~(((1 << hand) - 1) << base)) | ((((field & lower) | ((field >>> 1) & ~lower)) & ((1 << (hand - 1)) - 1)) << base);
}

export function hanabiRound(before: ArrayLike<number>, after: ArrayLike<number>, m: number, players: number, hand: number, focus: boolean, copies: readonly number[]): Uint32Array[] {
  const states = [Uint32Array.from(before)];
  for (let seat = 0; seat < players; seat++) {
    const words = Uint32Array.from(states[seat]!), record = hanabiMove(after, seat), kind = hanabiMoveKind(record);
    if (hanabiMoveSeat(record) !== seat) throw new Error(`Hanabi step ${m} records seat ${hanabiMoveSeat(record)} in seat ${seat}'s move`);
    if (kind === HANABI_TOLD_COLOUR || kind === HANABI_TOLD_RANK) tell(words, record, hand, focus, m);
    else if (kind >= HANABI_PLAYED && kind <= HANABI_DISCARDED) spend(words, seat, record, after, players, hand, copies, m);
    else throw new Error(`Hanabi step ${m} records move kind ${kind} for seat ${seat}`);
    words[CLOCK] = 0;
    words[CURSOR] = after[CURSOR]!;
    words[TURN] = words[TURN]! + 1;
    if (words[FINAL]) words[FINAL] = words[FINAL]! - 1;
    words[MOVES + seat] = record;
    states.push(words);
  }
  const last = states[players]!;
  for (let i = 0; i < WORDS; i++) if (last[i] !== after[i]) throw new Error(`Hanabi step ${m} replays word ${i} to ${last[i]} where the recording holds ${after[i]}`);
  return states;
}

function tally(record: number, events: SimEvent[]): void {
  const kind = hanabiMoveKind(record);
  if (kind === HANABI_PLAYED) events.push({ type: 'goal' });
  else if (kind === HANABI_MISFIRED) events.push({ type: 'harm' });
}

export const hanabiDecoder: EnvDecoder = {
  slots: SCORE_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words;
    out[CURSOR_SLOT] = words[CURSOR]!;
    out[CLOCK_SLOT] = words[CLOCK]!;
    out[INFO_SLOT] = words[INFO]!;
    out[LIFE_SLOT] = words[LIFE]!;
    out[DECK_SLOT] = words[DECK]!;
    out[FINAL_SLOT] = words[FINAL]!;
    out[TURN_SLOT] = words[TURN]!;
    out[GAME_SLOT] = words[GAME]!;
    for (let c = 0; c < HANABI_COLOURS; c++) out[HEIGHT_SLOT + c] = hanabiHeight(words, c);
    out[SCORE_SLOT] = hanabiScore(words);
  },
  geometry({ environment: env }, initial) {
    if (env.words !== WORDS || env.view !== 0) throw new Error(`${env.envId} records ${env.words} words and ${env.view} view words where a Hanabi table packs into ${WORDS} words`);
    const players = parameter(env, 'players'), handSize = parameter(env, 'handSize'), copies = constantList(env, 'HANABI_COPIES'), deck = constantList(env, 'HANABI_ORDERED');
    const view = 2 * players * handSize + SEEN + RECALL * players;
    if (env.obs !== view || env.actions !== ACTIONS) throw new Error(`${env.envId} observes ${env.obs} values over ${env.actions} actions where Hanabi shows ${view} over ${ACTIONS}`);
    if (players < 2 || players > HANABI_PLAYERS || players * handSize > HANABI_SEATS) throw new Error(`${env.envId} seats ${players} hands of ${handSize} where the table holds ${HANABI_PLAYERS} hands and ${HANABI_SEATS} cards`);
    if (copies.length !== HANABI_RANKS || deck.length !== HANABI_COLOURS * copies.reduce((sum, n) => sum + n, 0)) throw new Error(`${env.envId} deals a deck that is not ${HANABI_COLOURS} colours of ${HANABI_RANKS} ranks`);
    if (initial.words[DECK] !== deck.length - players * handSize) throw new Error(`${env.envId} opens with ${initial.words[DECK]} cards in the deck after dealing ${players * handSize} of ${deck.length}`);
    if (constant(env, 'HANABI_PATIENCE') < 1) throw new Error(`${env.envId} gives the move clock no patience`);
    return {};
  },
  deriveEvents(prev, next) {
    if (!hanabiMoved(prev, next)) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (next.done) {
      const end = hanabiEnding(prev.words, next.words);
      for (let n = hanabiScore(end) - hanabiScore(prev.words); n > 0; n--) events.push({ type: 'goal' });
      for (let n = hanabiLife(prev.words) - hanabiLife(end); n > 0; n--) events.push({ type: 'harm' });
    }
    for (let seat = 0; seat < HANABI_PLAYERS; seat++) tally(hanabiMove(next.words, seat), events);
    if (next.done) events.push({ type: 'done' });
    return events;
  },
};
