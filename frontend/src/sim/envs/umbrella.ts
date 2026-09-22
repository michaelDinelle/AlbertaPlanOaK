import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const UMBRELLA_PHASE = 0;
export const UMBRELLA_NEED = 1;
export const UMBRELLA_HAVE = 2;
export const UMBRELLA_ACTION = 3;
export const UMBRELLA_DISTRACTOR = 4;
export const UMBRELLA_MAX_DISTRACTORS = 12;

export const umbrellaDecoder: EnvDecoder = {
  slots: UMBRELLA_DISTRACTOR + UMBRELLA_MAX_DISTRACTORS,
  discrete: true,
  decode(frame, out) {
    out[UMBRELLA_PHASE] = frame.words[0]!; out[UMBRELLA_NEED] = frame.words[1]!; out[UMBRELLA_HAVE] = frame.words[2]!; out[UMBRELLA_ACTION] = frame.actions[0]!;
    for (let d = 0; d < UMBRELLA_MAX_DISTRACTORS; d++) out[UMBRELLA_DISTRACTOR + d] = frame.observations[UMBRELLA_DISTRACTOR + d] ?? 0;
  },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
