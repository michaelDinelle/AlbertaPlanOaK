import { NO_EVENTS, type EnvDecoder } from '../types';

export const RPS_OPPONENT = 0;
export const RPS_COUNTS = 1;
export const RPS_ACTION = 4;
export const RPS_FILLED = 5;

export const rpsCounter = (move: number): number => (move + 1) % 3;

export const rpsDecoder: EnvDecoder = {
  slots: 6,
  discrete: true,
  decode(frame, out) {
    out[RPS_OPPONENT] = frame.words[0]!;
    for (let m = 0; m < 3; m++) out[RPS_COUNTS + m] = frame.words[1 + m]!;
    out[RPS_ACTION] = frame.actions[0]!;
    out[RPS_FILLED] = frame.words[11]!;
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
