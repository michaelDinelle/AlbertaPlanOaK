import type { EnvId } from '@shared/envs';
import type { EnvFrame, EnvGeometry, EventStream, SimEvent } from '@sim/types';
import type { FrameState } from '@kit/types';

export interface LiveEdge {
  framesPerSecond: number;
  chunkFrames: number;
}

export interface PlaybackClock {
  time(): number;
  position(): number;
  playing: boolean;
  speed: number;
  readonly rate: number;
  advance(dtReal: number): number;
  seek(position: number): void;
  readonly duration: number;
}

export interface ClockSource {
  readonly frameCount: number;
  readonly loop: boolean;
  readonly edge: LiveEdge | null;
}

const RATE_SMOOTHING_S = 0.25;
const EDGE_TARGET_CHUNKS = 1.5;
const EDGE_MIN_TARGET_S = 2;
const EDGE_GAIN = 0.5;
const EDGE_FLOOR = 0.25;
const EDGE_DECEL_WALL_S = 2;
const EDGE_NEAR_CHUNKS = 2;
const EDGE_NEAR_CATCHUP = 2;

export function liveEdgeBufferFrames(edge: LiveEdge): number {
  return Math.max(EDGE_MIN_TARGET_S * edge.framesPerSecond, edge.chunkFrames * EDGE_TARGET_CHUNKS);
}

export function governedSpeed(chosen: number, aheadS: number, edge: LiveEdge, hz: number): number {
  const ingest = edge.framesPerSecond / hz;
  if (chosen <= ingest) return chosen;
  const targetS = liveEdgeBufferFrames(edge) / hz;
  if (aheadS <= targetS) {
    const error = (aheadS - targetS) / targetS;
    return ingest * Math.max(EDGE_FLOOR, 1 + EDGE_GAIN * error);
  }
  const nearS = Math.max(EDGE_MIN_TARGET_S * ingest, (edge.chunkFrames / hz) * EDGE_NEAR_CHUNKS);
  const over = aheadS - targetS;
  const nearCeiling = Math.min(chosen, ingest * EDGE_NEAR_CATCHUP);
  if (over <= nearS) return ingest + (nearCeiling - ingest) * (over / nearS);
  const bandS = Math.max(nearS, EDGE_DECEL_WALL_S * chosen);
  const fraction = Math.min(1, (over - nearS) / bandS);
  return nearCeiling + (chosen - nearCeiling) * fraction;
}

export function makePlaybackClock(source: ClockSource, controlHz: number): PlaybackClock {
  const hz = controlHz > 0 ? controlHz : 8;
  const last = (): number => Math.max(0, source.frameCount - 1);
  let t = 0;
  let rate: number | null = null;
  let applied = 0;
  const clock: PlaybackClock = {
    playing: true,
    speed: 1,
    get duration() {
      return last() / hz;
    },
    get rate() {
      return applied;
    },
    time: () => t,
    position: () => t * hz,
    advance(dtReal) {
      if (!clock.playing) {
        applied = 0;
        return t * hz;
      }
      const duration = last() / hz;
      const edge = source.edge;
      const want = edge !== null && edge.framesPerSecond > 0 ? governedSpeed(clock.speed, duration - t, edge, hz) : clock.speed;
      if (rate === null) rate = want;
      else rate += (want - rate) * (1 - Math.exp(-dtReal / RATE_SMOOTHING_S));
      const before = t;
      t += dtReal * rate;
      if (t > duration) {
        t = source.loop && duration > 0 ? t % duration : duration;
      }
      if (t < 0) t = 0;
      applied = dtReal > 0 && t >= before ? (t - before) / dtReal : rate;
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
  wrapMask: number;
  geometry: EnvGeometry;
  readonly frameCount: number;
  readonly complete: boolean;
  readonly edge: LiveEdge | null;
  readonly events: EventStream;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
}

export function sampleFrame(source: PlaybackSource, position: number, out: FrameState): void {
  const n = source.frameCount;
  if (n === 0) return;
  const last = n - 1;
  const p = position < 0 ? 0 : position > last ? last : position;
  const i = Math.min(last, Math.floor(p));
  const frame = source.frameAt(i);
  const next = source.frameAt(Math.min(last, i + 1));
  const alpha = source.discretePose || frame.done !== 0 ? 0 : p - i;

  if (alpha > 0 && next !== frame) {
    for (let k = 0; k < 9; k++) {
      const a = frame.pose[k] ?? 0;
      const b = next.pose[k] ?? 0;
      let d = b - a;
      if (((source.wrapMask >>> k) & 1) !== 0) d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      out.pose[k] = a + d * alpha;
    }
  } else {
    out.pose.set(frame.pose.subarray(0, 9));
  }
  out.done = frame.done !== 0;
  out.step = frame.step;
  out.simTime = frame.step / (source.controlHz > 0 ? source.controlHz : 1);
}

export const MAX_CATCHUP_FRAMES = 12;
