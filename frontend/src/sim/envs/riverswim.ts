import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const RIVER_STATION = 0;

const SOURCE = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];

export const riverUpstream = (frame: EnvFrame): boolean => (Math.trunc(frame.actions[0]!) & 1) === 1;

export const riverSwimDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[RIVER_STATION] = frame.words[0]!;
  },
  geometry({ environment }, initial) {
    const states = parameter(environment, 'states'), bits = parameter(environment, 'bits');
    if (environment.words !== 1 || environment.view !== 0 || environment.actions !== 2 || environment.obs !== bits + 1 || states > 2 ** bits || initial.words[0]! >= states) throw new Error(`${environment.envId} does not carry one station word below ${states}, two strokes and ${bits} bit observations plus a bias`);
    return {};
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 && riverUpstream(next) ? SOURCE : NO_EVENTS;
  },
};
