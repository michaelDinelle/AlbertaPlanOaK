import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const UMBRELLA_PHASE = 0;
export const UMBRELLA_NEED = 1;
export const UMBRELLA_HAVE = 2;

export const umbrellaDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    for (let k = 0; k < 3; k++) out[k] = frame.words[k]!;
  },
  geometry(_descriptor, initial) {
    if (initial.words.length !== 3) throw new Error('Umbrella state does not carry its phase, need and have words');
    return {};
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = [next.reward > 0 ? { type: 'goal' } : { type: 'harm' }];
    return events;
  },
};
