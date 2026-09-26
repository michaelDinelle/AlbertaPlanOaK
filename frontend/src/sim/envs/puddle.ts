import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const PUDDLE_X = 0;
export const PUDDLE_Y = 1;

const ARRIVED = Object.freeze([{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }]) as unknown as SimEvent[];
const WADED = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const puddleWet = (frame: EnvFrame): boolean => frame.done === 0 && frame.reward < -1;

export const puddleDecoder: EnvDecoder = {
  slots: 2,
  decode(frame, out) {
    out[PUDDLE_X] = wordFloat(frame.words[0]!);
    out[PUDDLE_Y] = wordFloat(frame.words[1]!);
  },
  geometry({ environment }) {
    if (environment.words !== 2 || environment.obs !== 3) throw new Error(`${environment.envId} does not carry its x and y words with an x, y and bias observation`);
    return {};
  },
  deriveEvents(prev, next) {
    if (next.done) return ARRIVED;
    return puddleWet(next) && !puddleWet(prev) ? WADED : NO_EVENTS;
  },
};
