import { constant, type EnvDescriptor } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type GridGeometry } from '../types';

export const BLOCKING_X = 0;
export const BLOCKING_Y = 1;

export function blockingPhases(env: EnvDescriptor): [GridGeometry, GridGeometry] {
  const phases = env.parameters.phase;
  if (!Array.isArray(phases) || phases.length !== 2) throw new Error(`${env.envId} lacks its two phase layouts`);
  const grids = (phases as { w: number; h: number; walls: number[] }[]).map(phase => {
    if (!Array.isArray(phase.walls) || phase.walls.length !== phase.w * phase.h) throw new Error(`${env.envId} phase layout is incomplete`);
    return { w: phase.w, h: phase.h, walls: Uint8Array.from(phase.walls), goal: { x: phase.w - 1, y: 0 } };
  });
  const [first, second] = grids as [GridGeometry, GridGeometry];
  if (first.w !== second.w || first.h !== second.h) throw new Error(`${env.envId} phase layouts differ in size`);
  return [first, second];
}

export const blockingPhase = (env: EnvDescriptor, step: number): number => Math.floor(step / constant(env, 'BLOCKING_PERIOD')) & 1;

export const blockingDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[BLOCKING_X] = frame.words[0]!; out[BLOCKING_Y] = frame.words[1]!; },
  geometry({ environment }) { return { grid: blockingPhases(environment)[0] }; },
  deriveEvents(_prev, next) { return next.done ? [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }] : NO_EVENTS; },
};
