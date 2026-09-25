import { constant, parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const RING_CELL = 0;
export const RING_CALL = 1;

export const ringCell = (frame: EnvFrame): number => frame.words[0]!;
export const ringCall = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;
export const ringColour = (cell: number, arc: number): number => Math.floor(cell / arc) & 1;

const CAUGHT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const MISSED = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const ringDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[RING_CELL] = ringCell(frame);
    out[RING_CALL] = ringCall(frame);
  },
  geometry({ environment }, initial) {
    const cells = constant(environment, 'RING_CELLS'), arc = parameter(environment, 'arcLength'), cell = ringCell(initial);
    if (environment.words !== 1 || environment.view !== 0 || environment.obs !== 2 || environment.actions !== 2 || environment.box !== 0
      || !Number.isInteger(arc) || arc < 1 || cell >= cells || initial.observations[0] !== ringColour(cell, arc) || initial.observations[1] !== 1) {
      throw new Error(`${environment.envId} does not carry one cell word on a ${cells}-cell ring of ${arc}-cell arcs, that cell's colour plus a bias, and a two-way call`);
    }
    return {};
  },
  deriveEvents(prev, next) {
    if (next.reward <= 0) return MISSED;
    return next.observations[0] !== prev.observations[0] ? CAUGHT : NO_EVENTS;
  },
};
