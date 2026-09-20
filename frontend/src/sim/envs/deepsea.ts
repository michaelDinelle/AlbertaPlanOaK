import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const DEEPSEA_ROW = 0;
export const DEEPSEA_COL = 1;
export const DEEPSEA_ACTION = 4;

export interface DeepSeaStatic { width: number; mask: Uint8Array }

export function deepSeaNextCol(st: DeepSeaStatic, r: number, c: number, action: number): number {
  const dc = (action & 1) === st.mask[r * st.width + c] ? 1 : -1;
  return Math.min(Math.max(c + dc, 0), st.width - 1);
}

export const deepSeaDecoder: EnvDecoder = {
  geometry({ environment, actionMap }) {
    const n = parameter(environment, 'extent') + 1;
    if (actionMap?.length !== n * n) throw new Error('DeepSea initialized action map is missing');
    return { grid: { w: n, h: n, walls: new Uint8Array(n * n), goal: { x: n - 1, y: n - 1 }, actionMask: Uint8Array.from(actionMap) } };
  },
  deriveEvents(prev, next) {
    if (next.pose[DEEPSEA_ROW] !== 0 || prev.pose[DEEPSEA_ROW] === 0) return NO_EVENTS;
    return next.reward > 0 ? [{ type: 'goal' }, { type: 'resurface' }] : [{ type: 'resurface' }];
  },
};
