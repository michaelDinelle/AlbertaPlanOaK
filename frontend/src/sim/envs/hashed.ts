import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const HASHED_STATE = 0;

const PAID = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const CHARGED = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const hashedDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[HASHED_STATE] = frame.words[0]!;
  },
  geometry({ environment }, initial) {
    const bits = parameter(environment, 'bits');
    if (bits % 2 !== 0 || environment.words !== 1 || environment.obs !== bits + 1 || initial.words[0]! >>> bits !== 0) throw new Error(`${environment.envId} does not carry one ${bits}-bit state word and ${bits} bit observations plus a bias`);
    return {};
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 ? PAID : next.reward < 0 ? CHARGED : NO_EVENTS;
  },
};
