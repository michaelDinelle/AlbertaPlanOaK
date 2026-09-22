import { NO_EVENTS, type EnvDecoder } from '../types';

export const FORAGER_X = 0;
export const FORAGER_Y = 1;

export const foragerDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  sequential: true,
  decode(frame, out) {
    out[FORAGER_X] = frame.words[0]!;
    out[FORAGER_Y] = frame.words[1]!;
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => (next.reward > 0 ? [{ type: 'goal' }] : next.reward < 0 ? [{ type: 'harm' }] : NO_EVENTS),
};
