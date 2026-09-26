import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';

export const HIDDEN_MODE_STATION = 0;
export const HIDDEN_MODE_TUNED = 1;

export const hiddenModeStation = (frame: EnvFrame): number => frame.words[0]!;
export const hiddenModeTuned = (frame: EnvFrame): number => (frame.actions[0]! >>> 0) & 1;

export const hiddenModeDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[HIDDEN_MODE_STATION] = hiddenModeStation(frame);
    out[HIDDEN_MODE_TUNED] = hiddenModeTuned(frame);
  },
  geometry({ environment }, initial) {
    if (initial.words.length !== 1 || environment.actions !== 2 || environment.box !== 0) throw new Error(`${environment.envId} is not a one-bit hidden mode over two actions`);
    return {};
  },
  deriveEvents(prev, next) {
    const station = hiddenModeStation(next);
    return station === hiddenModeStation(prev) ? NO_EVENTS : [{ type: 'cue', value: station }];
  },
};
