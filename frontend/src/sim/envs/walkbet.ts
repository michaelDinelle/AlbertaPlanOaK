import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const walkPosition = (frame: EnvFrame): number => frame.words[0]!;
export const walkBet = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;
export const walkDest = (before: EnvFrame, after: EnvFrame, cells: number): number =>
  after.done ? (walkPosition(before) * 2 < cells ? 0 : cells) : walkPosition(after);

const MISSED = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];
const ENDED = Object.freeze([{ type: 'goal' }, { type: 'done' }]) as unknown as SimEvent[];
const LOST = Object.freeze([{ type: 'harm' }, { type: 'done' }]) as unknown as SimEvent[];

export const walkBetDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[0] = walkPosition(frame);
    out[1] = walkBet(frame);
  },
  geometry({ environment }, initial) {
    const cells = constant(environment, 'WALKBET_CELLS');
    if (environment.words !== 1 || environment.view !== 0 || environment.obs !== 2 || environment.actions !== 2 || environment.box !== 0 || walkPosition(initial) >= cells) {
      throw new Error(`${environment.envId} does not carry one position word below ${cells}, two observations and a two-way bet`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    if (next.done) return next.reward > 0 ? ENDED : LOST;
    return next.reward > 0 ? NO_EVENTS : MISSED;
  },
};
