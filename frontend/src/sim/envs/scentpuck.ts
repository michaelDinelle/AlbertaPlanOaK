import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const SCENTPUCK_X = 0;
export const SCENTPUCK_Y = 1;
export const SCENTPUCK_LURE_X = 2;
export const SCENTPUCK_LURE_Y = 3;
export const SCENTPUCK_RIVAL_X = 4;
export const SCENTPUCK_RIVAL_Y = 5;
export const SCENTPUCK_WORDS = 6;

export const SCENTPUCK_SCENT = 0;
export const SCENTPUCK_RIVAL_SCENT = 1;
export const SCENTPUCK_OBSERVATIONS = 5;
export const SCENTPUCK_ACTIONS = 5;

const CAUGHT = Object.freeze([{ type: 'goal' }, { type: 'done' }]) as unknown as SimEvent[];

export const scentPuckDecoder: EnvDecoder = {
  slots: SCENTPUCK_WORDS,
  decode(frame, out) {
    for (let i = 0; i < SCENTPUCK_WORDS; i++) out[i] = wordFloat(frame.words[i]!);
  },
  geometry({ environment }, initial) {
    const competing = parameter(environment, 'competing');
    if (environment.words !== SCENTPUCK_WORDS || initial.words.length !== SCENTPUCK_WORDS) throw new Error(`${environment.envId} does not carry its puck, lure and rival as ${SCENTPUCK_WORDS} words`);
    if (environment.obs !== SCENTPUCK_OBSERVATIONS || environment.actions !== SCENTPUCK_ACTIONS || environment.box !== 0) throw new Error(`${environment.envId} is not a ${SCENTPUCK_OBSERVATIONS}-reading, ${SCENTPUCK_ACTIONS}-move scent hunt`);
    if (competing !== 0 && competing !== 1) throw new Error(`${environment.envId} has competing ${competing}`);
    if (competing === 0 && (initial.words[SCENTPUCK_RIVAL_X] !== 0 || initial.words[SCENTPUCK_RIVAL_Y] !== 0)) throw new Error(`${environment.envId} places a rival it does not declare`);
    return {};
  },
  deriveEvents(_prev, next) {
    return next.done ? CAUGHT : NO_EVENTS;
  },
};
