import { constant } from '../../data/envs';
import type { EnvDecoder, EnvFrame, SimEvent } from '../types';

export const CELLULAR_SCORED = 0;
export const CELLULAR_FLIP = 1;

export const cellBit = (frame: EnvFrame, cell: number): number => (frame.words[cell >>> 5]! >>> (cell & 31)) & 1;
export const cellFlip = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;

const HIT = Object.freeze([{ type: 'goal' }]) as unknown as SimEvent[];
const MISS = Object.freeze([{ type: 'harm' }]) as unknown as SimEvent[];

export const cellularDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[CELLULAR_SCORED] = cellBit(frame, 1);
    out[CELLULAR_FLIP] = cellFlip(frame);
  },
  geometry({ environment }, initial) {
    const cells = constant(environment, 'CELLULAR_CELLS'), rule = constant(environment, 'CELLULAR_RULE');
    let ok = Number.isInteger(cells) && cells >= 3 && cells <= environment.words * 32 && Number.isInteger(rule) && rule >= 0 && rule < 256
      && environment.view === 0 && environment.obs === cells + 1 && environment.actions === 2 && environment.box === 0 && initial.observations[cells] === 1;
    for (let i = 0; ok && i < cells; i++) ok = initial.observations[i] === cellBit(initial, i);
    if (!ok) throw new Error(`${environment.envId} does not carry a ${cells}-cell ring in its words, observed cell by cell plus a bias, with a two-way flip of cell 0`);
    return {};
  },
  deriveEvents(_prev, next) {
    return next.reward > 0 ? HIT : MISS;
  },
};
