import { NO_EVENTS, type EnvDecoder } from '../types';

export const BLOCKING_X = 0;
export const BLOCKING_Y = 1;

export const blockingDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[BLOCKING_X] = frame.words[0]!;
    out[BLOCKING_Y] = frame.words[1]!;
  },
  geometry({ environment }) {
    const phases = environment.parameters['phase'] as { w: number; h: number; walls: number[] }[] | undefined, first = phases?.[0], goal = environment.goal;
    if (!first || first.walls.length !== first.w * first.h || !goal) throw new Error(`${environment.envId} lacks its walls or goal`);
    return { grid: { w: first.w, h: first.h, walls: Uint8Array.from(first.walls), goal: { x: goal[0], y: goal[1] } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    if (next.reward !== 1) throw new Error('A blocking-maze reset has no goal reward');
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
