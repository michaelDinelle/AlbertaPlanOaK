import { NO_EVENTS, type EnvDecoder } from '../types';

export const ACCESS_FREE = 0;
export const ACCESS_PRIORITY = 1;
export const ACCESS_ACTION = 2;

export const accessDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[ACCESS_FREE] = frame.words[0]!; out[ACCESS_PRIORITY] = frame.words[1]!; out[ACCESS_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
