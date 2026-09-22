import { constant, constantList, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type GridCell } from '../types';

export const TAXI_X = 0;
export const TAXI_Y = 1;
export const TAXI_PASSENGER = 2;
export const TAXI_DESTINATION = 3;

export function taxiDepots(env: EnvDescriptor): GridCell[] {
  const xs = constantList(env, 'TAXI_DEPOT_X'), ys = constantList(env, 'TAXI_DEPOT_Y');
  if (xs.length !== constant(env, 'TAXI_DEPOTS') || ys.length !== xs.length) throw new Error(`${env.envId} depot table is incomplete`);
  return xs.map((x, i) => ({ x, y: ys[i]! }));
}

export function taxiBlocked(side: number, x: number, y: number, nx: number, ny: number): boolean {
  if (nx < 0 || ny < 0 || nx >= side || ny >= side) return true;
  if (ny !== y) return false;
  const lo = Math.min(x, nx);
  if (y <= 1 && lo === 1) return true;
  return y >= 3 && (lo === 0 || lo === 2);
}

export const taxiDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) { for (let i = 0; i < 4; i++) out[i] = frame.words[i]!; },
  geometry({ environment }) {
    const side = constant(environment, 'TAXI_SIDE');
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side), keys: taxiDepots(environment) } };
  },
  deriveEvents(prev, next) {
    if (next.done) return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
    if (next.words[2] !== prev.words[2]) return [{ type: 'key', index: prev.words[2]! }];
    return NO_EVENTS;
  },
};
