import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const PASSWORD_PHASE = 0;
export const PASSWORD_KEY = 1;

export const passwordKey = (frame: EnvFrame): number => Math.max(0, Math.floor(frame.actions[0]!));

export const passwordDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[PASSWORD_PHASE] = frame.words[PASSWORD_PHASE]!;
    out[PASSWORD_KEY] = passwordKey(frame);
  },
  geometry(descriptor, initial) {
    const length = descriptor.environment.parameters.length;
    if (initial.words.length !== 1 || initial.view.length !== length) throw new Error(`Password needs one phase word and ${String(length)} code view words`);
    return {};
  },
  deriveEvents(prev, next) {
    const events: SimEvent[] = [];
    if (next.reward > 0) events.push({ type: 'goal' });
    else if (prev.words[PASSWORD_PHASE]! > 0 && next.words[PASSWORD_PHASE] === 0) events.push({ type: 'harm' });
    return events.length ? events : NO_EVENTS;
  },
};
