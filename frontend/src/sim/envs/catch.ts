import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type SimEvent } from '../types';
import { wordInt } from '../words';

export const CATCH_COLUMN = 0;
export const CATCH_ROW = 1;
export const CATCH_PADDLE = 2;
export const CATCH_ACTION = 3;

export const catchWasCaught = (prev: EnvFrame, next: EnvFrame): boolean => wordInt(prev.words[0]!) === wordInt(next.words[2]!);

export const catchDecoder: EnvDecoder = {
  slots: 4,
  decode(frame, out) {
    for (let i = 0; i < 3; i++) out[i] = wordInt(frame.words[i]!);
    out[CATCH_ACTION] = frame.actions[0]!;
  },
  geometry(descriptor): EnvGeometry {
    const w = parameter(descriptor.environment, 'width'), h = parameter(descriptor.environment, 'height');
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    const out: SimEvent[] = [];
    if (catchWasCaught(prev, next)) out.push({ type: 'goal' });
    out.push({ type: 'done' }, { type: 'teleport' });
    return out;
  },
};
