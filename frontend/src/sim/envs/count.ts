import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const COUNT_TALLY = 0;
export const COUNT_PHASE = 1;
export const COUNT_EVENT = 2;
export const COUNT_QUERY = 3;
export const COUNT_KEY = 4;

export const countKey = (frame: EnvFrame): number => frame.actions[0]! >>> 0;

export const countDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) {
    for (let k = 0; k < 4; k++) out[k] = frame.words[k]!;
    out[COUNT_KEY] = countKey(frame);
  },
  geometry(_descriptor, initial) {
    if (initial.words.length !== 4) throw new Error('CountRecall state does not carry its count, phase, event and query words');
    return {};
  },
  deriveEvents(prev, next) {
    const events: SimEvent[] = [];
    if (prev.words[COUNT_QUERY] === 1) events.push(next.reward > 0 ? { type: 'goal' } : { type: 'harm' });
    if (next.words[COUNT_QUERY] === 1) events.push({ type: 'cue', value: next.words[COUNT_TALLY]! });
    return events.length ? events : NO_EVENTS;
  },
};
