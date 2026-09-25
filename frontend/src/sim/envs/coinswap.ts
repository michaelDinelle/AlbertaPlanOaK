import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';

export const COINSWAP_LOADED = 0;
export const COINSWAP_TOSS = 1;

export const coinSwapLoaded = (frame: EnvFrame): number => frame.words[0]!;
export const coinSwapToss = (frame: EnvFrame): number => (frame.actions[0]! >>> 0) & 1;

export const coinSwapDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[COINSWAP_LOADED] = coinSwapLoaded(frame);
    out[COINSWAP_TOSS] = coinSwapToss(frame);
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    const loaded = coinSwapLoaded(next);
    return loaded === coinSwapLoaded(prev) ? NO_EVENTS : [{ type: 'cue', value: loaded }];
  },
};
