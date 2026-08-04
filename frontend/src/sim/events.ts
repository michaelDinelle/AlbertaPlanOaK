// The shared empty-events singleton.
//
// `EnvDecoder.deriveEvents` returns `SimEvent[]` for the frozen contract's
// sake, but the driver and every consumer treat the returned array as
// READ-ONLY (see the doc comment on `DecodedRecording.events`). Most frames of
// most envs fire nothing, so all 16 decoders return this one frozen array
// instead of allocating a fresh `[]` per frame: identical `===`-comparable
// empties, zero per-frame garbage on the playback hot path, and a runtime
// tripwire (frozen) against anyone who tries to mutate a decoder's output.

import type { SimEvent } from './types';

/**
 * The canonical empty event list. Frozen: never mutate, never assume a fresh
 * array. Every decoder returns THIS object when a transition fires nothing,
 * so `events[i] === NO_EVENTS` is a valid "nothing happened" test.
 */
export const NO_EVENTS: SimEvent[] = Object.freeze([]) as unknown as SimEvent[];
