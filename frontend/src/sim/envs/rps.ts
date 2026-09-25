import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const RPS_MOVES = 3;
export const RPS_NONE = -1;
export const RPS_MINE = 0;
export const RPS_THEIRS = 1;
export const RPS_COUNTS = 2;
export const RPS_FILLED = 5;
export const RPS_CURSOR = 6;

const OPPONENT = 0;
const COUNTS = 1;
const HISTORY = 4;
const FILLED = 11;
const CURSOR = 12;
export const RPS_WORDS = CURSOR + 1;
export const RPS_CAPACITY = (FILLED - HISTORY) * 16;

export const rpsFilled = (frame: EnvFrame): number => frame.words[FILLED]!;
export const rpsCursor = (frame: EnvFrame): number => frame.words[CURSOR]!;
export const rpsCount = (frame: EnvFrame, move: number): number => frame.words[COUNTS + move]!;
export const rpsSlot = (frame: EnvFrame, index: number): number => (frame.words[HISTORY + (index >> 4)]! >>> (2 * (index & 15))) & 3;
export const rpsPlayed = (frame: EnvFrame): boolean => rpsFilled(frame) > 0;
export const rpsMine = (frame: EnvFrame): number => (rpsPlayed(frame) ? (frame.actions[0]! >>> 0) % RPS_MOVES : RPS_NONE);
export const rpsTheirs = (frame: EnvFrame): number => (rpsPlayed(frame) ? frame.words[OPPONENT]! : RPS_NONE);
export const rpsCounter = (move: number): number => (move + 1) % RPS_MOVES;
export const rpsOutcome = (mine: number, theirs: number): number => (mine === theirs ? 0 : mine === rpsCounter(theirs) ? 1 : -1);

const outcomeOf = (reward: number): number => (reward > 0 ? 0 : reward < 0 ? 1 : 2);
const EVENTS: readonly SimEvent[][][] = [0, 1, 2].map((outcome) =>
  [-1, 0, 1, 2].map((cue) => {
    const events: SimEvent[] = [];
    if (outcome === 0) events.push({ type: 'goal' });
    if (outcome === 1) events.push({ type: 'harm' });
    if (cue >= 0) events.push({ type: 'cue', value: cue });
    return Object.freeze(events) as SimEvent[];
  }),
);

export const rpsDecoder: EnvDecoder = {
  slots: 7,
  discrete: true,
  decode(frame, out) {
    out[RPS_MINE] = rpsMine(frame);
    out[RPS_THEIRS] = rpsTheirs(frame);
    for (let move = 0; move < RPS_MOVES; move++) out[RPS_COUNTS + move] = rpsCount(frame, move);
    out[RPS_FILLED] = rpsFilled(frame);
    out[RPS_CURSOR] = rpsCursor(frame);
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    if (!rpsPlayed(next)) return NO_EVENTS;
    const switched = rpsPlayed(prev) && rpsTheirs(next) !== rpsTheirs(prev);
    const events = EVENTS[outcomeOf(next.reward)]![switched ? rpsTheirs(next) + 1 : 0]!;
    return events.length === 0 ? NO_EVENTS : events;
  },
};
