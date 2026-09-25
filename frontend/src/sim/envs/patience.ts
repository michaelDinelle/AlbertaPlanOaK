import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const PATIENCE_POSITION = 0;
export const PATIENCE_WAITED = 1;

const PAYOUT = Object.freeze([{ type: 'goal' }, { type: 'done' }]) as unknown as SimEvent[];
const CASH_OUT = Object.freeze([{ type: 'done' }]) as unknown as SimEvent[];

export const patienceWaited = (frame: EnvFrame): boolean => ((frame.actions[0]! >>> 0) & 1) === 1;

export const patienceDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[PATIENCE_POSITION] = frame.words[0]!;
    out[PATIENCE_WAITED] = patienceWaited(frame) ? 1 : 0;
  },
  geometry(_descriptor, initial) {
    if (initial.words.length !== 1) throw new Error('Patience state does not carry its single position word');
    return {};
  },
  deriveEvents(_prev, next) {
    return next.done ? (patienceWaited(next) ? PAYOUT : CASH_OUT) : NO_EVENTS;
  },
};
