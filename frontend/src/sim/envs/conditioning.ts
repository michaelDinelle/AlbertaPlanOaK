import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const CONDITIONING_COUNTDOWN = 0;
export const CONDITIONING_CUES = 1;
export const CONDITIONING_STIMULUS = 2;
export const CONDITIONING_BLINK = 3;

export const conditioningDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) {
    for (let i = 0; i < 3; i++) out[i] = frame.words[i]!;
    out[CONDITIONING_BLINK] = Math.trunc(frame.actions[0]!) & 1;
  },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    const cues = next.words[CONDITIONING_CUES]!, events: SimEvent[] = [];
    if (cues !== 0) events.push({ type: 'cue', value: cues });
    if (next.words[CONDITIONING_STIMULUS] !== 0 && next.reward > 0) events.push({ type: 'goal' });
    return events.length ? events : NO_EVENTS;
  },
};
