import { NO_EVENTS, type EnvDecoder } from '../types';
import { wordFloat } from '../words';

export const CARTPOLE_X = 0;
export const CARTPOLE_X_DOT = 1;
export const CARTPOLE_THETA = 2;

export const cartPoleDecoder: EnvDecoder = {
  slots: 4,
  wrap: 1 << CARTPOLE_THETA,
  decode(frame, out) {
    for (let i = 0; i < 4; i++) out[i] = wordFloat(frame.words[i]!);
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => (next.done ? [{ type: 'harm' }, { type: 'done' }, { type: 'teleport' }] : NO_EVENTS),
};
