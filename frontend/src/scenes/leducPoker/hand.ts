import { POKER_BOARD, POKER_MINE, POKER_MY_CHIPS, POKER_NODE, POKER_SEAT, POKER_THEIRS, POKER_THEIR_CHIPS, pokerDecoder } from '@sim/envs/poker';
import type { EnvFrame } from '@sim/types';
import { parameter, type EnvDescriptor } from '../../data/envs';

export const FOLD = 0;
export const CALL = 1;
export const RAISE = 2;
export const AGENT = 0;
export const RIVAL = 1;
export const RANK_NAMES = ['J', 'Q', 'K'] as const;
export const SUIT_NAMES = ['♠', '♥'] as const;
export const OPPONENTS = ['uniform', 'CFR', 'fictitious play'] as const;
export const FICTITIOUS = 2;

const TABLE = 1;
const LAST = 2;
const VERSION = 3;
const IDS = 6;
const LEVELS = 3;
const MOVE_SHIFT = 26;
const ENDED_SHIFT = 28;

export interface Rules {
  suits: number;
  rounds: number;
  ante: number;
  raise: readonly number[];
  cap: number;
  actions: number;
  unowed: number;
  capped: number;
  opponent: number;
  iterations: number;
  sweep: number;
}

export interface Move {
  round: number;
  mover: number;
  move: number;
  owed: boolean;
  units: number;
  chips: readonly [number, number];
}

export interface Hand {
  index: number;
  seat: number;
  mine: number;
  theirs: number;
  board: number;
  node: number;
  round: number;
  line: Move[];
  chips: readonly [number, number];
  ending: number;
}

export interface Story {
  from: Hand;
  hand: Hand;
  moves: Move[];
  ended: boolean;
  to: Hand;
  intent: number;
  reward: number;
  adopted: boolean;
}

export function rulesOf(env: EnvDescriptor): Rules {
  const rules: Rules = {
    suits: parameter(env, 'suits'), rounds: parameter(env, 'rounds'), ante: parameter(env, 'ante'), raise: [parameter(env, 'firstRaise'), parameter(env, 'secondRaise')],
    cap: parameter(env, 'raiseCap'), actions: parameter(env, 'actions'), unowed: parameter(env, 'foldUnowed'), capped: parameter(env, 'raiseCapped'),
    opponent: parameter(env, 'opponent'), iterations: parameter(env, 'iterations'), sweep: parameter(env, 'sweepSteps'),
  };
  if (rules.suits > SUIT_NAMES.length || OPPONENTS[rules.opponent] === undefined) throw new Error(`${env.envId} deals ${rules.suits} suits against opponent ${rules.opponent}, which the table does not name`);
  return rules;
}

export const rankOf = (card: number, rules: Rules): number => Math.floor(card / rules.suits);
export const cardName = (card: number, rules: Rules): string => `${RANK_NAMES[rankOf(card, rules)]}${SUIT_NAMES[card % rules.suits]}`;
export const deckLeft = (rules: Rules): number => RANK_NAMES.length * rules.suits - 2 - (rules.rounds > 1 ? 1 : 0);
export const versionOf = (frame: EnvFrame): number => frame.words[VERSION] ?? 0;
export const wordOf = (m: Move): string => m.move === FOLD ? 'FOLD' : m.move === CALL ? (m.owed ? 'CALL' : 'CHECK') : m.owed ? 'RAISE' : 'BET';

export function choice(value: number, rules: Rules): number {
  const pick = value >= 0 && value < rules.actions ? Math.trunc(value) : 0;
  return rules.actions === 2 ? RAISE * pick : pick;
}

export function legal(rules: Rules, node: number, move: number): number {
  const raises = node % IDS % LEVELS;
  if (move === FOLD && raises === 0) return rules.unowed;
  return move === RAISE && raises === rules.cap ? rules.capped : move;
}

const pose = new Float32Array(pokerDecoder.slots);
const probe: EnvFrame = { step: 0, done: 0, words: new Uint32Array(2), view: new Uint32Array(0), observations: new Float32Array(0), actions: new Float32Array(0), reward: 0, reset: false };

function read(index: number, table: number): Float32Array {
  probe.words[0] = index;
  probe.words[1] = table;
  pokerDecoder.decode(probe, pose);
  return pose;
}

const branch = (id: number): number[] => id < LEVELS ? Array<number>(id).fill(RAISE) : [CALL, ...Array<number>(id - LEVELS).fill(RAISE)];

function act(rules: Rules, chips: [number, number], round: number, mover: number, move: number): Move {
  const own = chips[mover]!, other = chips[1 - mover]!, after = move === RAISE ? other + rules.raise[round]! : move === CALL ? other : own;
  chips[mover] = after;
  return { round, mover, move, owed: other > own, units: (after - own) / rules.raise[round]!, chips: [chips[0], chips[1]] };
}

function build(index: number, p: Float32Array, rules: Rules, ending: number, step: number): Hand {
  const node = p[POKER_NODE]!, seat = p[POKER_SEAT]!, round = node < IDS ? 0 : 1, block = Math.floor(node / IDS) - 1;
  const streets = round === 0 ? [branch(node)] : [[...branch(Math.floor(block / RANK_NAMES.length) + 1), CALL], branch(node % IDS)];
  const chips: [number, number] = [rules.ante, rules.ante], line: Move[] = [];
  streets.forEach((moves, r) => moves.forEach((move, k) => line.push(act(rules, chips, r, k % 2 === 0 ? seat : 1 - seat, move))));
  const held = [p[POKER_MY_CHIPS]!, p[POKER_THEIR_CHIPS]!] as const, board = p[POKER_BOARD]!;
  if (chips[0] !== held[0] || chips[1] !== held[1] || (round === 1 && block % RANK_NAMES.length !== rankOf(board, rules))) {
    throw new Error(`LeducPoker step ${step}: betting node ${node} of hand ${index} stakes ${chips.join('/')} on board rank ${block % RANK_NAMES.length}, but the table holds ${held.join('/')} with ${board}`);
  }
  if (ending >= 0) line.push(act(rules, chips, round, (node % IDS) % 2 === 0 ? seat : 1 - seat, ending));
  return { index, seat, mine: p[POKER_MINE]!, theirs: p[POKER_THEIRS]!, board, node, round, line, chips: held, ending };
}

export const handOf = (frame: EnvFrame, rules: Rules): Hand => build(frame.words[0]!, read(frame.words[0]!, frame.words[TABLE]!), rules, -1, frame.step);

export function priorOf(frame: EnvFrame, rules: Rules): Hand | null {
  const last = frame.words[LAST]!, index = frame.words[0]! - 1;
  if ((last >>> ENDED_SHIFT & 1) === 0) return null;
  return build(index, read(index, last & ((1 << MOVE_SHIFT) - 1)), rules, last >>> MOVE_SHIFT & 3, frame.step);
}

export function storyOf(before: EnvFrame, after: EnvFrame, rules: Rules): Story {
  const from = handOf(before, rules), to = handOf(after, rules), hand = after.done ? priorOf(after, rules) : to;
  const intent = choice(after.actions[0]!, rules), moves = hand ? hand.line.slice(from.line.length) : [];
  const agreed = hand !== null && hand.index === from.index && from.line.every((m, k) => m.move === hand.line[k]!.move && m.mover === hand.line[k]!.mover);
  if (!hand || !agreed || moves[0]?.mover !== AGENT || moves[0].move !== legal(rules, from.node, intent)) {
    throw new Error(`LeducPoker step ${after.step}: hand ${from.index} at node ${from.node} does not continue into ${hand ? `hand ${hand.index} at node ${hand.node}` : 'a finished hand'} by the agent's action ${after.actions[0]}`);
  }
  return { from, hand, moves, ended: after.done !== 0, to, intent, reward: after.reward, adopted: rules.opponent === FICTITIOUS && versionOf(after) !== versionOf(before) };
}
