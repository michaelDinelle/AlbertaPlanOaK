import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const POKER_SEAT = 0;
export const POKER_MINE = 1;
export const POKER_THEIRS = 2;
export const POKER_BOARD = 3;
export const POKER_NODE = 4;
export const POKER_SEEN = 5;
export const POKER_MY_CHIPS = 6;
export const POKER_THEIR_CHIPS = 7;
export const POKER_NONE = 7;

export const pokerDecoder: EnvDecoder = {
  slots: 8,
  discrete: true,
  decode(frame, out) {
    const table = frame.words[1]!;
    out[POKER_SEAT] = frame.words[0]! & 1;
    out[POKER_MINE] = table & 7;
    out[POKER_THEIRS] = table >>> 3 & 7;
    out[POKER_BOARD] = table >>> 6 & 7;
    out[POKER_NODE] = table >>> 9 & 127;
    out[POKER_SEEN] = table >>> 16 & 3;
    out[POKER_MY_CHIPS] = table >>> 18 & 15;
    out[POKER_THEIR_CHIPS] = table >>> 22 & 15;
  },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = [];
    if (next.reward > 0) events.push({ type: 'goal' });
    if (next.reward < 0) events.push({ type: 'harm' });
    events.push({ type: 'done' });
    return events;
  },
};
