import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';

export const TIGER_LISTEN = 2;

export const tigerDoor = (frame: EnvFrame): number => frame.words[0]! & 1;
export const tigerListened = (frame: EnvFrame): boolean => frame.words[1] !== 0;
export const tigerHint = (frame: EnvFrame): number => frame.words[2]! & 1;
export const tigerTally = (frame: EnvFrame): number => frame.words[3]! | 0;
export const tigerAction = (frame: EnvFrame): number => (frame.actions[0]! >>> 0) % 3;
export const tigerOpened = (frame: EnvFrame): boolean => frame.done !== 0 && tigerAction(frame) !== TIGER_LISTEN;
export const tigerMauled = (frame: EnvFrame): boolean => tigerOpened(frame) && frame.reward < 0;

export const tigerDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) {
    out[0] = tigerDoor(frame);
    out[1] = tigerListened(frame) ? 1 : 0;
    out[2] = tigerHint(frame);
    out[3] = tigerTally(frame);
    out[4] = tigerAction(frame);
  },
  geometry({ environment }, initial) {
    if (initial.words.length !== 4 || environment.actions !== 3 || environment.box !== 0) throw new Error(`${environment.envId} is not a two-door tiger over three actions`);
    return {};
  },
  deriveEvents(_prev, next) {
    if (!tigerOpened(next)) return NO_EVENTS;
    return [{ type: next.reward > 0 ? 'goal' : 'harm' }, { type: 'done' }];
  },
};
