import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';

export const CONTEXTUAL_ACTION = 0;

export const contextualAction = (frame: EnvFrame): number => {
  const action = frame.actions[0]!;
  return action > 0 ? Math.min(Math.trunc(action), 0xffffffff) : 0;
};

export const contextualDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[CONTEXTUAL_ACTION] = contextualAction(frame);
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
