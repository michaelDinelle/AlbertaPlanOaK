import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const SLIME_BALL = 0;
export const SLIME_AGENT = 4;
export const SLIME_RIVAL = 8;
export const SLIME_MEMORY = 12;
export const SLIME_X = 0;
export const SLIME_Y = 1;
export const SLIME_VX = 2;
export const SLIME_VY = 3;
export const SLIME_FRAME = 16;
export const SLIME_HOLD = 17;
export const SLIME_LIVES = 18;
export const SLIME_RIVAL_LIVES = 19;
export const SLIME_POINT = 20;
export const SLIME_GAME = 21;
export const SLIME_MEMBER = 22;
const SLIME_SLOTS = 16;

const SCORED: readonly SimEvent[] = Object.freeze([{ type: 'goal' }]);
const CONCEDED: readonly SimEvent[] = Object.freeze([{ type: 'harm' }]);
const SCORED_END: readonly SimEvent[] = Object.freeze([{ type: 'goal' }, { type: 'done' }]);
const CONCEDED_END: readonly SimEvent[] = Object.freeze([{ type: 'harm' }, { type: 'done' }]);
const END: readonly SimEvent[] = Object.freeze([{ type: 'done' }]);

export const slimeDecoder: EnvDecoder = {
  slots: SLIME_SLOTS,
  decode(frame, out) {
    for (let i = 0; i < SLIME_SLOTS; i++) out[i] = wordFloat(frame.words[i]!);
  },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (next.reward !== 0 && Math.abs(next.reward) !== 1) throw new Error(`SlimeVolley step ${next.step} pays ${next.reward}, but only points pay`);
    if (next.reward > 0) return (next.done ? SCORED_END : SCORED) as SimEvent[];
    if (next.reward < 0) return (next.done ? CONCEDED_END : CONCEDED) as SimEvent[];
    return next.done ? (END as SimEvent[]) : NO_EVENTS;
  },
};
