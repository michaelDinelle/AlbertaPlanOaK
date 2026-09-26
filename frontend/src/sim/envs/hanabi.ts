import { constantList, parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const HANABI_COLOURS = 5;
export const HANABI_RANKS = 5;
export const HANABI_SEATS = 16;

export const HANABI_NEXT = 0;
export const HANABI_PLAY = 2;

export const HANABI_PLAYED = 1;
export const HANABI_MISFIRED = 2;
export const HANABI_DISCARDED = 3;
export const HANABI_TOLD_COLOUR = 4;
export const HANABI_TOLD_RANK = 5;

const CURSOR_SLOT = 0;
const ACTOR_SLOT = 1;
const THINK_SLOT = 2;
const INFO_SLOT = 3;
const LIFE_SLOT = 4;
const DECK_SLOT = 5;
const FINAL_SLOT = 6;
const TURN_SLOT = 7;
const GAME_SLOT = 8;
const HEIGHT_SLOT = 9;
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
const ACTOR = TOUCHED + 1;
const THINK = ACTOR + 1;
const CURSOR = THINK + 1;
const FINAL = CURSOR + 1;
const TURN = FINAL + 1;
const GAME = TURN + 1;
const LAST = GAME + 1;
const WORDS = LAST + 1;
const OBSERVED = 18;
const ACTIONS = 6;
const PLAUSIBLE = 0x1ffffff;
const NONE = 0x3e000000;
const CLUED = 0x40000000;

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
export const hanabiActor = (words: ArrayLike<number>): number => words[ACTOR]!;
export const hanabiThink = (words: ArrayLike<number>): number => words[THINK]!;
export const hanabiCursor = (words: ArrayLike<number>): number => words[CURSOR]!;
export const hanabiFinal = (words: ArrayLike<number>): number => words[FINAL]!;
export const hanabiTurn = (words: ArrayLike<number>): number => words[TURN]!;
export const hanabiGame = (words: ArrayLike<number>): number => words[GAME]!;
export const hanabiLastSeat = (words: ArrayLike<number>): number => words[LAST]! & 3;
export const hanabiLastKind = (words: ArrayLike<number>): number => (words[LAST]! >>> 2) & 7;
export const hanabiLastAt = (words: ArrayLike<number>): number => (words[LAST]! >>> 5) & 7;
export const hanabiLastCard = (words: ArrayLike<number>): number => (words[LAST]! >>> 8) & 31;
export const hanabiLastTouched = (words: ArrayLike<number>): number => words[LAST]! >>> 13;

export function hanabiAction(actions: ArrayLike<number>): number {
  const control = actions[0]!;
  return control >= 0 && control < ACTIONS ? Math.trunc(control) : HANABI_NEXT;
}

export const hanabiMoved = (prev: EnvFrame, next: EnvFrame): boolean => next.done !== 0 || next.words[TURN] !== prev.words[TURN];

export const hanabiAgentPlayedOwn = (prev: EnvFrame, next: EnvFrame): boolean =>
  prev.words[ACTOR] === 0 && hanabiAction(next.actions) === HANABI_PLAY && prev.observations[0] === 0;

export const hanabiMisfireEnding = (prev: EnvFrame, next: EnvFrame): boolean =>
  next.done !== 0 && next.reward <= 0 && (next.reward < 0 || prev.words[FINAL] !== 1 || hanabiAgentPlayedOwn(prev, next));

const GOAL: readonly SimEvent[] = Object.freeze([{ type: 'goal' }]);
const HARM: readonly SimEvent[] = Object.freeze([{ type: 'harm' }]);
const DONE: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);
const GOAL_DONE: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const HARM_DONE: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);

export const hanabiDecoder: EnvDecoder = {
  slots: SCORE_SLOT + 1,
  discrete: true,
  decode(frame, out) {
    const words = frame.words;
    out[CURSOR_SLOT] = words[CURSOR]!;
    out[ACTOR_SLOT] = words[ACTOR]!;
    out[THINK_SLOT] = words[THINK]!;
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
    if (env.obs !== OBSERVED || env.actions !== ACTIONS) throw new Error(`${env.envId} observes ${env.obs} values over ${env.actions} actions where Hanabi shows ${OBSERVED} over ${ACTIONS}`);
    const players = parameter(env, 'players'), handSize = parameter(env, 'handSize'), copies = constantList(env, 'HANABI_COPIES'), deck = constantList(env, 'HANABI_ORDERED');
    if (players < 2 || players * handSize > HANABI_SEATS) throw new Error(`${env.envId} seats ${players} hands of ${handSize} where the table holds ${HANABI_SEATS} cards`);
    if (copies.length !== HANABI_RANKS || deck.length !== HANABI_COLOURS * copies.reduce((sum, n) => sum + n, 0)) throw new Error(`${env.envId} deals a deck that is not ${HANABI_COLOURS} colours of ${HANABI_RANKS} ranks`);
    if (initial.words[DECK] !== deck.length - players * handSize) throw new Error(`${env.envId} opens with ${initial.words[DECK]} cards in the deck after dealing ${players * handSize} of ${deck.length}`);
    return {};
  },
  deriveEvents(prev, next) {
    if (next.done) return (next.reward > 0 ? GOAL_DONE : hanabiMisfireEnding(prev, next) ? HARM_DONE : DONE) as SimEvent[];
    if (next.words[TURN] === prev.words[TURN]) return NO_EVENTS;
    const kind = hanabiLastKind(next.words);
    return (kind === HANABI_PLAYED ? GOAL : kind === HANABI_MISFIRED ? HARM : NO_EVENTS) as SimEvent[];
  },
};
