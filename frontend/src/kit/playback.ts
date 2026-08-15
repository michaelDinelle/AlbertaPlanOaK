import type { EnvId } from '@shared/envs';
import type { EnvFrame, EnvGeometry, SimEvent } from '@sim/types';
import type { FrameState, QualityTier } from './types';
import { CTRL_LEN } from '@sim/record';

export const TIERS: readonly QualityTier[] = ['hero', 'medium', 'thumb'];

export interface PlaybackClock {
  time(): number;
  position(): number;
  playing: boolean;
  speed: number;
  advance(dtReal: number): number;
  seek(position: number): void;
  readonly duration: number;
}

export interface ClockSource {
  readonly frameCount: number;
  readonly loop: boolean;
}

export function makePlaybackClock(source: ClockSource, controlHz: number): PlaybackClock {
  const hz = controlHz > 0 ? controlHz : 8;
  const last = (): number => Math.max(0, source.frameCount - 1);
  let t = 0;
  const clock: PlaybackClock = {
    playing: true,
    speed: 1,
    get duration() {
      return last() / hz;
    },
    time: () => t,
    position: () => t * hz,
    advance(dtReal) {
      if (clock.playing) {
        t += dtReal * clock.speed;
        const duration = last() / hz;
        if (t > duration) {
          t = source.loop && duration > 0 ? t % duration : duration;
        }
        if (t < 0) t = 0;
      }
      return t * hz;
    },
    seek(position) {
      const l = last();
      const p = position < 0 ? 0 : position > l ? l : position;
      t = p / hz;
    },
  };
  return clock;
}

export interface PlaybackSource {
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: EnvGeometry;
  readonly frameCount: number;
  readonly complete: boolean;
  frameAt(i: number): EnvFrame<unknown>;
  eventsAt(i: number): SimEvent[];
  eventEntries(): IterableIterator<[number, SimEvent[]]>;
}

export function sampleFrame(source: PlaybackSource, position: number, out: FrameState): void {
  const n = source.frameCount;
  if (n === 0) return;
  const last = n - 1;
  const p = position < 0 ? 0 : position > last ? last : position;
  const i = Math.min(last, Math.floor(p));
  const frame = source.frameAt(i);
  const next = source.frameAt(Math.min(last, i + 1));
  const alpha = source.discretePose ? 0 : p - i;

  if (alpha > 0 && next !== frame) {
    for (let k = 0; k < 9; k++) {
      const a = frame.pose[k] ?? 0;
      const b = next.pose[k] ?? 0;
      out.pose[k] = a + (b - a) * alpha;
    }
  } else {
    out.pose.set(frame.pose.subarray(0, 9));
  }
  out.ctrl.set(frame.ctrl.subarray(0, CTRL_LEN));
  out.done = frame.done !== 0;
  out.step = frame.step;
  out.simTime = frame.step / (source.controlHz > 0 ? source.controlHz : 1);
  out.derived = frame.state;
}

export const MAX_CATCHUP_FRAMES = 12;
