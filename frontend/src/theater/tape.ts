import { deriveStream, frameOf } from '@sim/events';
import { decoders, PLAYBACK } from '@sim/registry';
import { emptyStream, NO_EVENTS, type EnvFrame, type EnvGeometry, type EventStream, type SimEvent } from '@sim/types';
import type { EnvId } from '../data/envs';
import { FRAME_FLOATS, OFF_DONE, OFF_POSE } from '../data/oak';

export interface Tape {
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  wrapMask: number;
  geometry: EnvGeometry;
  readonly frameCount: number;
  readonly events: EventStream;
  readonly identical: boolean | null;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
  onChange: (error?: Error) => void;
}

type Slot = Tape & { drop(): void };

const cache = new Map<string, Slot>();

function open(runId: string, lane: number, envId: EnvId, seed: number, poseLen: number, obsDim: number, horizon: number): Slot {
  const decoder = decoders[envId];
  const st = decoder.initStatic(seed);
  const init = decoder.initFrame(seed, st);
  const playback = PLAYBACK[envId];
  const abort = new AbortController();
  const stream = emptyStream();
  let frames: Float32Array | null = null;
  let steps = 0;
  let identical: boolean | null = null;

  const frameAt = (i: number): EnvFrame => {
    const c = Math.max(0, Math.min(steps, Math.floor(i)));
    if (c === 0 || frames === null) return init;
    return frameOf(frames, c - 1, c - 1);
  };

  const load = async (): Promise<void> => {
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/replay/${lane}`, { signal: abort.signal });
      if (!response.ok) throw new Error(`${response.status} replay/${lane}: ${await response.text()}`);
      const rows = new Float32Array(await response.arrayBuffer());
      const width = poseLen + obsDim + 2;
      const count = Math.floor(rows.length / width);
      if (count > horizon) throw new Error(`lane ${lane} holds ${count} steps, past the horizon ${horizon}`);
      const next = new Float32Array(count * FRAME_FLOATS);
      for (let r = 0; r < count; r++) {
        const at = r * width;
        next[r * FRAME_FLOATS + OFF_DONE] = rows[at + poseLen + obsDim + 1] ?? 0;
        for (let k = 0; k < poseLen; k++) next[r * FRAME_FLOATS + OFF_POSE + k] = rows[at + k] ?? 0;
      }
      frames = next;
      steps = count;
      identical = response.headers.get('X-Oak-Identical') === '1';
      deriveStream(frames, 0, steps, decoder, init, st, stream);
      slot.onChange();
    } catch (error) {
      if (!abort.signal.aborted) slot.onChange(error instanceof Error ? error : new Error(String(error)));
    }
  };

  const slot: Slot = {
    envId,
    seed,
    controlHz: playback.controlHz,
    discretePose: playback.discretePose,
    wrapMask: playback.wrapMask,
    geometry: decoder.geometry(st, seed),
    get frameCount() { return steps + 1; },
    get events() { return stream; },
    get identical() { return identical; },
    frameAt,
    eventsAt(i) {
      const c = Math.floor(i);
      return c <= 0 || c > steps ? NO_EVENTS : decoder.deriveEvents(frameAt(c - 1), frameAt(c), st);
    },
    onChange: () => {},
    drop() { abort.abort(); },
  };
  void load();
  return slot;
}

export function tapeFor(runId: string, lane: number, envId: EnvId, seed: number, poseLen: number, obsDim: number, horizon: number): Tape {
  const key = `${runId}/${lane}`;
  if (!cache.has(key)) cache.set(key, open(runId, lane, envId, seed, poseLen, obsDim, horizon));
  return cache.get(key)!;
}

export function keepTapes(keep: Tape[]): void {
  for (const [key, slot] of cache) {
    if (keep.includes(slot)) continue;
    cache.delete(key);
    slot.drop();
  }
}
