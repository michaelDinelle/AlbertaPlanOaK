import { NO_EVENTS, type EnvDecoder } from '../types';

export const COINSWAP_COIN = 0;
export const COINSWAP_ACTION = 1;

export const coinswapDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[COINSWAP_COIN] = frame.words[0]!; out[COINSWAP_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: (prev, next) => prev.words[0] === next.words[0] ? NO_EVENTS : [{ type: 'cue', value: next.words[0]! }],
};
