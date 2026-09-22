import { NO_EVENTS, type EnvDecoder } from '../types';

export const ALIASED_POS = 0;
export const ALIASED_TARGET = 1;
export const ALIASED_ACTION = 2;
export const ALIASED_COLOUR = 3;

export const aliasedDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) { out[ALIASED_POS] = frame.words[0]!; out[ALIASED_TARGET] = frame.words[1]!; out[ALIASED_ACTION] = frame.actions[0]!; out[ALIASED_COLOUR] = frame.observations[0]!; },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.done !== 0 ? [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }] : NO_EVENTS,
};
