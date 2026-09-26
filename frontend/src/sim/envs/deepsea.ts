import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const DEEPSEA_ROW = 0;
export const DEEPSEA_COL = 1;
export const DEEPSEA_ACTION = 2;

export interface DeepSeaStatic { width: number; mask: Uint8Array }

export function deepSeaNextCol(st: DeepSeaStatic, r: number, c: number, action: number): number {
  const dc = (action & 1) === st.mask[r * st.width + c] ? 1 : -1;
  return Math.min(Math.max(c + dc, 0), st.width - 1);
}

export const deepseaDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[DEEPSEA_ROW] = frame.words[0]!; out[DEEPSEA_COL] = frame.words[1]!; out[DEEPSEA_ACTION] = frame.actions[0]!; },
  geometry({ environment }, initial) {
    const n = parameter(environment, 'extent') + 1, mask = new Uint8Array(n * n);
    if (initial.words.length < 3 + Math.ceil(n * n / 32)) throw new Error('DeepSea state does not carry its action map');
    for (let i = 0; i < n * n; i++) mask[i] = (initial.words[3 + (i >>> 5)]! >>> (i & 31)) & 1;
    return { grid: { w: n, h: n, walls: new Uint8Array(n * n), goal: { x: n - 1, y: n - 1 }, actionMask: mask } };
  },
  deriveEvents(prev, next) {
    if (next.words[0] !== 0 || prev.words[0] === 0) return NO_EVENTS;
    return next.reward > 0 ? [{ type: 'goal' }, { type: 'resurface' }] : [{ type: 'resurface' }];
  },
};
