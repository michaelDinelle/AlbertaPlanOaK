import { describe, expect, test } from 'bun:test';
import { governedSpeed, makePlaybackClock, type LiveEdge } from './playbackSource';

const HZ = 20;
const DT = 1 / 60;

function advanceFor(clock: ReturnType<typeof makePlaybackClock>, seconds: number, onTick?: () => void): void {
  for (let s = 0; s < seconds; s += DT) {
    onTick?.();
    clock.advance(DT);
  }
}

describe('live edge governor', () => {
  const edge: LiveEdge = { framesPerSecond: 32, chunkFrames: 1024 };

  test('far behind the head it plays at the chosen speed', () => {
    const source = { frameCount: 200000, loop: false, edge };
    const clock = makePlaybackClock(source, HZ);
    clock.speed = 64;
    advanceFor(clock, 3);
    expect(clock.rate).toBeCloseTo(64, 0);
  });

  test('at the head it never stops while frames remain and settles near the ingest pace', () => {
    let frames = 4000;
    const source = { loop: false, edge, get frameCount() { return frames; } };
    const clock = makePlaybackClock(source, HZ);
    clock.speed = 64;
    clock.seek(frames - 1);
    let pending = 0;
    let minRate = Infinity;
    let lastPosition = clock.position();
    advanceFor(clock, 90, () => {
      pending += edge.framesPerSecond * DT;
      const whole = Math.floor(pending);
      frames += whole;
      pending -= whole;
    });
    advanceFor(clock, 30, () => {
      pending += edge.framesPerSecond * DT;
      const whole = Math.floor(pending);
      frames += whole;
      pending -= whole;
      const position = clock.position();
      expect(position).toBeGreaterThanOrEqual(lastPosition);
      expect(position).toBeLessThanOrEqual(frames - 1);
      lastPosition = position;
      minRate = Math.min(minRate, clock.rate);
    });
    const ingest = edge.framesPerSecond / HZ;
    expect(minRate).toBeGreaterThan(0);
    expect(clock.rate).toBeGreaterThan(ingest * 0.7);
    expect(clock.rate).toBeLessThan(ingest * 1.3);
  });

  test('governed speed holds the chosen speed until the last seconds, then eases to the ingest pace', () => {
    const ingest = edge.framesPerSecond / HZ;
    const targetS = (edge.chunkFrames / HZ) * 1.5;
    const chunkS = edge.chunkFrames / HZ;
    const far = governedSpeed(64, 10000, edge, HZ);
    const oneArrivalOver = governedSpeed(64, targetS + chunkS, edge, HZ);
    const twoArrivalsOver = governedSpeed(64, targetS + 2 * chunkS, edge, HZ);
    const approaching = governedSpeed(64, targetS + 2 * chunkS + 64, edge, HZ);
    const atTarget = governedSpeed(64, targetS, edge, HZ);
    const empty = governedSpeed(64, 0, edge, HZ);
    expect(far).toBe(64);
    expect(oneArrivalOver).toBeCloseTo(ingest * 1.5, 6);
    expect(twoArrivalsOver).toBeCloseTo(ingest * 2, 6);
    expect(approaching).toBeGreaterThan(ingest * 2);
    expect(approaching).toBeLessThan(64);
    expect(atTarget).toBeCloseTo(ingest, 6);
    expect(empty).toBeGreaterThan(0);
    expect(empty).toBeLessThan(ingest);
    expect(governedSpeed(1, 10000, edge, HZ)).toBe(1);
    expect(governedSpeed(1, 0, edge, HZ)).toBe(1);
  });

  test('a finished recording clamps at its last frame', () => {
    const source = { frameCount: 600, loop: false, edge: null };
    const clock = makePlaybackClock(source, HZ);
    clock.speed = 64;
    advanceFor(clock, 5);
    expect(clock.position()).toBe(599);
    expect(clock.rate).toBe(0);
  });
});
