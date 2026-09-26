import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const ALIASED_CELL = 0;
export const ALIASED_TARGET = 1;
export const ALIASED_CLAIM = 2;

export const aliasedCell = (frame: EnvFrame): number => frame.words[0]!;
export const aliasedTarget = (frame: EnvFrame): number => frame.words[1]!;
export const aliasedClaim = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;
export const aliasedColour = (frame: EnvFrame, cell: number): number => (frame.view[cell >>> 5]! >>> (cell & 31)) & 1;

const FOUND = Object.freeze([{ type: 'goal' }, { type: 'teleport' }]) as unknown as SimEvent[];
const WRONG = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const aliasedDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[ALIASED_CELL] = aliasedCell(frame);
    out[ALIASED_TARGET] = aliasedTarget(frame);
    out[ALIASED_CLAIM] = aliasedClaim(frame);
  },
  geometry({ environment }, initial) {
    const cells = parameter(environment, 'cells'), cell = aliasedCell(initial);
    if (environment.words !== 2 || environment.view !== Math.ceil(cells / 32) || environment.obs !== 2 || environment.actions !== 2 || environment.box !== 0
      || cell >= cells || aliasedTarget(initial) >= cells || initial.observations[0] !== aliasedColour(initial, cell) || initial.observations[1] !== 1) {
      throw new Error(`${environment.envId} does not carry a cell and a target word on a ${cells}-cell ring, its colour bits as view words, that cell's colour plus a bias, and a two-way claim`);
    }
    return {};
  },
  deriveEvents(_prev, next) {
    if (next.reward > 0) return FOUND;
    return next.reward < 0 ? WRONG : NO_EVENTS;
  },
};
