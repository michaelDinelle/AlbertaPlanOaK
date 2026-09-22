import { NO_EVENTS, type EnvDecoder } from '../types';

export const HIDDENMODE_MODE = 0;
export const HIDDENMODE_ACTION = 1;

export const hiddenmodeDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[HIDDENMODE_MODE] = frame.words[0]!; out[HIDDENMODE_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: (prev, next) => prev.words[0] === next.words[0] ? NO_EVENTS : [{ type: 'cue', value: next.words[0]! }],
};
