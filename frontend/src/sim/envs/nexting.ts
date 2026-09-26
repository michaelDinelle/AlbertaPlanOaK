import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const NEXTING_PHASE = 0;
export const NEXTING_CALL = 1;

export const nextingPhase = (frame: EnvFrame): number => frame.words[0]!;
export const nextingCall = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;

const CAUGHT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const MISSED = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const nextingDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[NEXTING_PHASE] = nextingPhase(frame);
    out[NEXTING_CALL] = nextingCall(frame);
  },
  geometry({ environment }, initial) {
    const period = constant(environment, 'NEXTING_PERIOD');
    if (environment.words !== 1 || environment.view !== 0 || environment.obs !== 4 || environment.actions !== 2 || environment.box !== 0 || nextingPhase(initial) >= period) {
      throw new Error(`${environment.envId} does not carry one phase word below ${period}, four observations and a two-way call`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    if (nextingPhase(next) !== 0) return NO_EVENTS;
    return next.reward > 0 ? CAUGHT : MISSED;
  },
};
