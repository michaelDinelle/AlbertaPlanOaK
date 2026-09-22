import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const COUNT_COUNT = 0;
export const COUNT_PHASE = 1;
export const COUNT_EVENT = 2;
export const COUNT_QUERY = 3;
export const COUNT_ACTION = 4;

export const countDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) {
    out[COUNT_COUNT] = frame.words[0]!; out[COUNT_PHASE] = frame.words[1]!; out[COUNT_EVENT] = frame.words[2]!; out[COUNT_QUERY] = frame.words[3]!;
    out[COUNT_ACTION] = frame.actions[0]!;
  },
  geometry({ environment }) {
    const period = parameter(environment, 'period');
    return { grid: { w: period, h: 1, walls: new Uint8Array(period) } };
  },
  deriveEvents(prev, next) {
    if (!prev.words[3]) return NO_EVENTS;
    const events: SimEvent[] = [{ type: 'done' }];
    if (next.reward > 0) events.push({ type: 'goal' });
    return events;
  },
};
