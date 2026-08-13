/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';

import { ENVS } from '@shared/envs';
import type { RecordingRef, RunIndexEntry } from '@shared/types';
import { MAX_CATCHUP_FRAMES } from '@kit/harness';
import type { SimEvent } from '@sim/types';
import {
  DEFAULT_THEATER_RUN,
  describeEvent,
  envChips,
  expectedBytes,
  formatBytes,
  makeProgress,
  parseTheaterParams,
  parseTheaterQuery,
  pickerRuns,
  progressFraction,
  recordingRef,
  replayHealth,
  seedsFor,
  seekTargetForTick,
  theaterHref,
  tickKindOf,
  timelineTicks,
  type LoadPhase,
  type LoadProgress,
  type TimelineEntry,
} from '../src/theater/loader';
import {
  REFRESH_CAP_MS,
  SPEEDS,
  advanceCursor,
  clampPosition,
  collectEvents,
  cycleTier,
  formatClock,
  keyAction,
  lowerTier,
  makeFrameCursor,
  makeTierGovernor,
  nearestSpeedIndex,
  nextSpeed,
  pushTicker,
  stepPosition,
  type TickerEntry,
} from '../src/theater/player';


function runEntry(overrides: Partial<RunIndexEntry> = {}): RunIndexEntry {
  return {
    id: 'run',
    envIds: ['HalfCheetah-v4'],
    batch: 128,
    steps: 100_000,
    policies: ['oak'],
    ablationLabel: 'none',
    startedAt: '2026-08-01T00:00:00Z',
    complete: true,
    maxStep: 100_000,
    inFlight: false,
    hasSummary: true,
    hasRecordings: true,
    role: null,
    ...overrides,
  };
}

function ref(envId: RecordingRef['envId'], seed: number, steps = 100_000): RecordingRef {
  return { envId, seed, steps };
}

function entry(frame: number, event: SimEvent, step = frame - 1): TimelineEntry {
  return { frame, step, event };
}


describe('deep link', () => {
  test('parses #/theater/:id/:envId/:seed', () => {
    expect(parseTheaterParams({ id: 'demo', envId: 'FourRooms-cont-v0', seed: '8' })).toEqual({
      runId: 'demo',
      envId: 'FourRooms-cont-v0',
      seed: 8,
    });
  });

  test('a run id with slashes-worth of punctuation survives the round trip', () => {
    const target = { runId: 'p93-tierL-20260721', envId: 'DeepSea-cont-v0', seed: 16 } as const;
    const href = theaterHref(target);
    expect(href).toBe('#/theater/p93-tierL-20260721/DeepSea-cont-v0/16');
    const segments = href.split('/');
    expect(
      parseTheaterParams({
        id: decodeURIComponent(segments[2] ?? ''),
        envId: decodeURIComponent(segments[3] ?? ''),
        seed: segments[4] ?? '',
      }),
    ).toEqual(target);
  });

  test('the picker is the answer for anything incomplete or invalid', () => {
    expect(parseTheaterParams({})).toBeNull();
    expect(parseTheaterParams({ id: 'demo', envId: 'FourRooms-cont-v0' })).toBeNull();
    expect(parseTheaterParams({ id: '', envId: 'FourRooms-cont-v0', seed: '8' })).toBeNull();
    expect(parseTheaterParams({ id: 'demo', envId: 'NotAnEnv', seed: '8' })).toBeNull();
    expect(parseTheaterParams({ id: 'demo', envId: 'FourRooms-cont-v0', seed: '-1' })).toBeNull();
    expect(parseTheaterParams({ id: 'demo', envId: 'FourRooms-cont-v0', seed: '3.5' })).toBeNull();
    expect(parseTheaterParams({ id: 'demo', envId: 'FourRooms-cont-v0', seed: 'eight' })).toBeNull();
  });

  test('seeds tolerate surrounding whitespace but not sign tricks', () => {
    expect(parseTheaterParams({ id: 'd', envId: 'Chain-cont-v0', seed: ' 64 ' })?.seed).toBe(64);
    expect(parseTheaterParams({ id: 'd', envId: 'Chain-cont-v0', seed: '+64' })).toBeNull();
  });
});

describe('parseTheaterQuery', () => {
  test('reads the deterministic-still switches', () => {
    const q = parseTheaterQuery('#/theater/a/Chain-cont-v0/64?tier=medium&at=1200&pause=1&debug=1&steps=4000&speed=2');
    expect(q).toEqual({ tier: 'medium', at: 1200, paused: true, debug: true, steps: 4000, speed: 2 });
  });

  test('junk is dropped, not coerced', () => {
    expect(parseTheaterQuery('#/theater/a/Chain-cont-v0/64')).toEqual({});
    expect(parseTheaterQuery('#/x?tier=ultra&steps=-5&at=nope&speed=0&pause=yes')).toEqual({});
  });
});


describe('pickerRuns', () => {
  test('keeps only runs that actually have recordings', () => {
    const runs = pickerRuns([
      runEntry({ id: 'a', hasRecordings: true }),
      runEntry({ id: 'b', hasRecordings: false }),
    ]);
    expect(runs.map((r) => r.id)).toEqual(['a']);
  });

  test('the 16-env demo run is pinned first, whatever the ranking says', () => {
    const runs = pickerRuns([
      runEntry({ id: 'best', role: 'keeper', startedAt: '2026-08-03T00:00:00Z' }),
      runEntry({ id: DEFAULT_THEATER_RUN, startedAt: '2026-07-01T00:00:00Z' }),
      runEntry({ id: 'other', startedAt: '2026-08-02T00:00:00Z' }),
    ]);
    expect(runs[0]?.id).toBe(DEFAULT_THEATER_RUN);
    expect(runs.slice(1).map((r) => r.id)).toEqual(['best', 'other']);
  });

  test('a missing featured run is simply absent — never invented', () => {
    const runs = pickerRuns([runEntry({ id: 'only' })]);
    expect(runs.map((r) => r.id)).toEqual(['only']);
  });
});

describe('envChips', () => {
  test('canonical env order, deduped seeds, real counts', () => {
    const chips = envChips([
      ref('DeepSea-cont-v0', 17),
      ref('HalfCheetah-v4', 3),
      ref('DeepSea-cont-v0', 16),
      ref('DeepSea-cont-v0', 16),
      ref('HalfCheetah-v4', 0),
    ]);
    expect(chips.map((c) => c.envId)).toEqual(['HalfCheetah-v4', 'DeepSea-cont-v0']);
    expect(chips[0]?.seeds).toEqual([0, 3]);
    expect(chips[1]?.seeds).toEqual([16, 17]);
    expect(chips[1]?.count).toBe(2);
    expect(chips[0]?.slug).toBe(ENVS['HalfCheetah-v4'].slug);
  });

  test('an unknown envId on disk cannot become a chip', () => {
    const chips = envChips([{ envId: 'Nope-v0' as RecordingRef['envId'], seed: 0, steps: 10 }]);
    expect(chips).toEqual([]);
  });

  test('seedsFor / recordingRef address one lane', () => {
    const recordings = [ref('Chain-cont-v0', 70), ref('Chain-cont-v0', 64), ref('POMaze-cont-v0', 112)];
    expect(seedsFor(recordings, 'Chain-cont-v0')).toEqual([64, 70]);
    expect(seedsFor(recordings, 'Acrobot-cont-v0')).toEqual([]);
    expect(recordingRef(recordings, 'Chain-cont-v0', 64)?.steps).toBe(100_000);
    expect(recordingRef(recordings, 'Chain-cont-v0', 99)).toBeNull();
  });
});


describe('load progress', () => {
  test('is monotone across the phase ladder', () => {
    const phases: LoadPhase[] = ['meta', 'download', 'decode', 'scene', 'ready'];
    let last = -1;
    for (const phase of phases) {
      const f = progressFraction(phase);
      expect(f).toBeGreaterThanOrEqual(last);
      last = f;
    }
    expect(progressFraction('ready')).toBe(1);
  });

  test('the download band tracks real bytes', () => {
    const lo = progressFraction('download', 0, 100);
    const mid = progressFraction('download', 50, 100);
    const hi = progressFraction('download', 100, 100);
    expect(lo).toBeCloseTo(0.04, 6);
    expect(mid).toBeCloseTo(0.04 + (0.78 - 0.04) / 2, 6);
    expect(hi).toBeCloseTo(0.78, 6);
    expect(progressFraction('decode')).toBeCloseTo(0.78, 6);
  });

  test('an unknown total pins the bar at the band start instead of guessing', () => {
    expect(progressFraction('download', 1024, 0)).toBeCloseTo(0.04, 6);
  });

  test('over- and under-shoot clamp inside the band', () => {
    expect(progressFraction('download', -5, 100)).toBeCloseTo(0.04, 6);
    expect(progressFraction('download', 500, 100)).toBeCloseTo(0.78, 6);
  });

  test('labels name the phase and the bytes', () => {
    const p: LoadProgress = makeProgress('download', 3.2 * 1024 * 1024, 6.4 * 1024 * 1024);
    expect(p.label).toBe('downloading 3.2 MB / 6.4 MB');
    expect(makeProgress('download', 100).label).toBe('downloading 100 B');
    expect(makeProgress('decode').label).toBe('decoding experience');
    expect(makeProgress('scene').label).toBe('building the scene');
  });

  test('formatBytes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(-1)).toBe('0 B');
    expect(formatBytes(900)).toBe('900 B');
    expect(formatBytes(2048)).toBe('2 kB');
    expect(formatBytes(6_400_000)).toBe('6.1 MB');
  });

  test('expectedBytes prefers content-range total, then content-length', () => {
    expect(expectedBytes(new Headers({ 'content-range': 'bytes 0-1023/6400000' }))).toBe(6_400_000);
    expect(expectedBytes(new Headers({ 'content-length': '1024' }))).toBe(1024);
    expect(expectedBytes(new Headers())).toBe(0);
  });
});


describe('replayHealth', () => {
  test('a clean recording raises nothing', () => {
    const health = replayHealth({ nonFiniteRows: 0, replay: { firstDisagreeStep: null, disagreeCount: 0 } });
    expect(health.damaged).toBe(false);
    expect(health.message).toBeNull();
  });

  test('a decoder with no replay tracking is clean by absence', () => {
    expect(replayHealth({}).damaged).toBe(false);
  });

  test('non-finite record rows are damage', () => {
    const health = replayHealth({ nonFiniteRows: 3 });
    expect(health.damaged).toBe(true);
    expect(health.message).toContain('3 record rows carry non-finite values');
  });

  test('a replay desync names the step it started at', () => {
    const health = replayHealth({ nonFiniteRows: 0, replay: { firstDisagreeStep: 4123, disagreeCount: 77 } });
    expect(health.damaged).toBe(true);
    expect(health.firstDisagreeStep).toBe(4123);
    expect(health.message).toContain('from step 4123');
    expect(health.message).toContain('77 frames');
  });

  test('both signals are reported together, and both agree with English', () => {
    const one = replayHealth({ nonFiniteRows: 1, replay: { firstDisagreeStep: null, disagreeCount: 2 } });
    expect(one.message).toContain('1 record row carries non-finite values');
    expect(one.message).toContain('desynced on 2 frames');
    expect(one.message).toContain(';');
    expect(replayHealth({ nonFiniteRows: 2 }).message).toContain('2 record rows carry');
  });
});


describe('event timeline ticks', () => {
  test('only landmark kinds get a tick', () => {
    expect(tickKindOf({ type: 'goal' })).toBe('goal');
    expect(tickKindOf({ type: 'done' })).toBe('done');
    expect(tickKindOf({ type: 'key', index: 0 })).toBe('key');
    expect(tickKindOf({ type: 'door', index: 1, open: true })).toBe('door');
    expect(tickKindOf({ type: 'cue', value: 1 })).toBe('cue');
    expect(tickKindOf({ type: 'teleport' })).toBeNull();
    expect(tickKindOf({ type: 'resurface' })).toBeNull();
  });

  test('maps frames onto the scrub range', () => {
    const ticks = timelineTicks(
      [entry(0, { type: 'goal' }), entry(50, { type: 'goal' }), entry(100, { type: 'goal' })],
      101,
      { maxTicks: 101 },
    );
    expect(ticks.map((t) => t.position)).toEqual([0, 0.5, 1]);
    expect(ticks.map((t) => t.frame)).toEqual([0, 50, 100]);
  });

  test('drops teleport/resurface and keeps the rest', () => {
    const ticks = timelineTicks(
      [entry(10, { type: 'teleport' }), entry(20, { type: 'goal' }), entry(30, { type: 'resurface' })],
      100,
      { maxTicks: 100 },
    );
    expect(ticks).toHaveLength(1);
    expect(ticks[0]?.kind).toBe('goal');
  });

  test('a dense lane is bucketed, never emitted one node per event', () => {
    const timeline: TimelineEntry[] = [];
    for (let i = 0; i < 4400; i++) {
      timeline.push(entry(i * 22, i % 3 === 0 ? { type: 'goal' } : { type: 'done' }));
    }
    const ticks = timelineTicks(timeline, 100_001, { maxTicks: 180 });
    expect(ticks.length).toBeLessThanOrEqual(180);
    expect(ticks.length).toBeGreaterThan(100);
    let total = 0;
    for (const tick of ticks) total += tick.count;
    expect(total).toBe(4400);
  });

  test('a merged bucket keeps its earliest frame and its highest-ranked kind', () => {
    const ticks = timelineTicks(
      [
        entry(41, { type: 'done' }, 40),
        entry(40, { type: 'goal' }, 39),
        entry(42, { type: 'done' }, 41),
      ],
      101,
      { maxTicks: 6 },
    );
    expect(ticks).toHaveLength(1);
    const tick = ticks[0];
    expect(tick?.kind).toBe('goal');
    expect(tick?.frame).toBe(40);
    expect(tick?.step).toBe(39);
    expect(tick?.count).toBe(3);
    expect(tick?.label).toContain('goal');
    expect(tick?.label).toContain('2× done');
    expect(tick?.label).toContain('step 39');
  });

  test('ticks come out in frame order', () => {
    const ticks = timelineTicks(
      [entry(90, { type: 'goal' }), entry(10, { type: 'key', index: 0 }), entry(50, { type: 'done' })],
      101,
      { maxTicks: 101 },
    );
    expect(ticks.map((t) => t.frame)).toEqual([10, 50, 90]);
  });

  test('an empty timeline yields no ticks', () => {
    expect(timelineTicks([], 100)).toEqual([]);
    expect(timelineTicks([entry(5, { type: 'teleport' })], 100)).toEqual([]);
  });

  test('clicking a tick lands BEFORE the event so the celebration plays', () => {
    expect(seekTargetForTick({ frame: 10 })).toBe(7);
    expect(seekTargetForTick({ frame: 10 }, 1)).toBe(9);
    expect(seekTargetForTick({ frame: 1 })).toBe(0);
    expect(seekTargetForTick({ frame: 0 })).toBe(0);
    expect(seekTargetForTick({ frame: 100 })).toBeGreaterThanOrEqual(100 - MAX_CATCHUP_FRAMES);
  });

  test('describeEvent labels the ticker rows', () => {
    expect(describeEvent({ type: 'goal' })).toBe('goal');
    expect(describeEvent({ type: 'key', index: 2 })).toBe('key 2');
    expect(describeEvent({ type: 'door', index: 1, open: true })).toBe('door 1 open');
    expect(describeEvent({ type: 'door', index: 1, open: false })).toBe('door 1 shut');
    expect(describeEvent({ type: 'cue', value: 0 })).toBe('cue 0');
    expect(describeEvent({ type: 'teleport' })).toBe('teleport');
  });
});


describe('transport rules', () => {
  test('speeds ladder from 0.25 to 8 and clamp at both ends', () => {
    expect(SPEEDS[0]).toBe(0.25);
    expect(SPEEDS[SPEEDS.length - 1]).toBe(8);
    expect(nextSpeed(1, 1)).toBe(2);
    expect(nextSpeed(1, -1)).toBe(0.5);
    expect(nextSpeed(8, 1)).toBe(8);
    expect(nextSpeed(0.25, -1)).toBe(0.25);
    expect(nearestSpeedIndex(1)).toBe(2);
    expect(nearestSpeedIndex(0)).toBe(2);
  });

  test('tier keys cycle down and the governor ladder ends at thumb', () => {
    expect(cycleTier('hero')).toBe('medium');
    expect(cycleTier('medium')).toBe('thumb');
    expect(cycleTier('thumb')).toBe('hero');
    expect(lowerTier('hero')).toBe('medium');
    expect(lowerTier('medium')).toBe('thumb');
    expect(lowerTier('thumb')).toBeNull();
  });

  test('the keymap is space / arrows / brackets / g / t', () => {
    expect(keyAction(' ')).toEqual({ kind: 'toggle' });
    expect(keyAction('ArrowRight')).toEqual({ kind: 'step', frames: 1 });
    expect(keyAction('ArrowLeft')).toEqual({ kind: 'step', frames: -1 });
    expect(keyAction('ArrowRight', true)).toEqual({ kind: 'step', frames: 10 });
    expect(keyAction('[')).toEqual({ kind: 'speed', direction: -1 });
    expect(keyAction(']')).toEqual({ kind: 'speed', direction: 1 });
    expect(keyAction('g')).toEqual({ kind: 'debug' });
    expect(keyAction('G')).toEqual({ kind: 'debug' });
    expect(keyAction('t')).toEqual({ kind: 'tier' });
    expect(keyAction('Home')).toEqual({ kind: 'seek', position: 'start' });
    expect(keyAction('End')).toEqual({ kind: 'seek', position: 'end' });
    expect(keyAction('q')).toBeNull();
    expect(keyAction('Enter')).toBeNull();
  });

  test('stepping is frame-accurate from a fractional position', () => {
    expect(stepPosition(4.7, 1, 100)).toBe(5);
    expect(stepPosition(4.7, -1, 100)).toBe(3);
    expect(stepPosition(0, -1, 100)).toBe(0);
    expect(stepPosition(99, 1, 100)).toBe(99);
    expect(stepPosition(50, 10, 100)).toBe(60);
  });

  test('clampPosition survives nonsense', () => {
    expect(clampPosition(NaN, 100)).toBe(0);
    expect(clampPosition(-3, 100)).toBe(0);
    expect(clampPosition(1e9, 100)).toBe(99);
    expect(clampPosition(12.5, 100)).toBe(12.5);
  });

  test('formatClock', () => {
    expect(formatClock(0)).toBe('0:00.0');
    expect(formatClock(-1)).toBe('0:00.0');
    expect(formatClock(65.44)).toBe('1:05.4');
    expect(formatClock(3725)).toBe('1:02:05.0');
  });
});

describe('frame cursor', () => {
  test('the first frame after a mount is a snap, never a sequential step', () => {
    const cursor = makeFrameCursor();
    const step = advanceCursor(cursor, 12.4, 100);
    expect(step.index).toBe(12);
    expect(step.sequential).toBe(false);
  });

  test('a walked step delivers exactly the frames it crossed', () => {
    const cursor = makeFrameCursor();
    advanceCursor(cursor, 5, 100);
    const step = advanceCursor(cursor, 8.9, 100);
    expect(step.sequential).toBe(true);
    expect(step.from).toBe(6);
    expect(step.to).toBe(8);
  });

  test('the catch-up window is the harness constant, not a new guess', () => {
    const near = makeFrameCursor();
    advanceCursor(near, 0, 1000);
    expect(advanceCursor(near, MAX_CATCHUP_FRAMES, 1000).sequential).toBe(true);

    const far = makeFrameCursor();
    advanceCursor(far, 0, 1000);
    expect(advanceCursor(far, MAX_CATCHUP_FRAMES + 1, 1000).sequential).toBe(false);
  });

  test('a backwards jump is a scrub', () => {
    const cursor = makeFrameCursor();
    advanceCursor(cursor, 40, 100);
    expect(advanceCursor(cursor, 39, 100).sequential).toBe(false);
  });

  test('a repeated frame is sequential and delivers nothing', () => {
    const cursor = makeFrameCursor();
    advanceCursor(cursor, 7, 100);
    const step = advanceCursor(cursor, 7.4, 100);
    expect(step.sequential).toBe(true);
    expect(step.from).toBe(8);
    expect(step.to).toBe(7);
    expect(collectEvents(() => [], step.from, step.to, [])).toEqual([]);
  });

  test('positions clamp to the frame range', () => {
    const cursor = makeFrameCursor();
    expect(advanceCursor(cursor, 1e9, 50).index).toBe(49);
    expect(advanceCursor(cursor, -20, 50).index).toBe(0);
  });

  test('collectEvents drains an inclusive frame range into one buffer', () => {
    const events: SimEvent[][] = [[], [{ type: 'goal' }], [], [{ type: 'done' }, { type: 'teleport' }]];
    const at = (i: number): readonly SimEvent[] => events[i] ?? [];
    const out: SimEvent[] = [{ type: 'cue', value: 0 }];
    collectEvents(at, 1, 3, out);
    expect(out).toEqual([{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }]);
    collectEvents(at, 2, 2, out);
    expect(out).toEqual([]);
  });

  test('collectEvents can report which frame each event came from', () => {
    const events: SimEvent[][] = [[], [{ type: 'goal' }], [], [{ type: 'done' }, { type: 'teleport' }]];
    const out: SimEvent[] = [];
    const frames: number[] = [99];
    collectEvents((i) => events[i] ?? [], 0, 3, out, frames);
    expect(frames).toEqual([1, 3, 3]);
    expect(frames).toHaveLength(out.length);
  });
});

describe('event ticker', () => {
  test('newest last, capped, with the step it fired on', () => {
    let ticker: TickerEntry[] = [];
    ticker = pushTicker(ticker, [{ type: 'goal' }], 10, 3);
    ticker = pushTicker(ticker, [{ type: 'key', index: 0 }, { type: 'door', index: 0, open: true }], 20, 3);
    ticker = pushTicker(ticker, [{ type: 'done' }], 30, 3);
    expect(ticker.map((e) => e.label)).toEqual(['key 0', 'door 0 open', 'done']);
    expect(ticker[2]?.step).toBe(30);
    expect(ticker[2]?.type).toBe('done');
  });

  test('a catch-up burst dates every event to ITS OWN step', () => {
    const ticker = pushTicker(
      [],
      [{ type: 'key', index: 0 }, { type: 'door', index: 0, open: true }, { type: 'goal' }],
      [3091, 3094, 3096],
    );
    expect(ticker.map((e) => e.step)).toEqual([3091, 3094, 3096]);
  });

  test('a short step list degrades instead of inventing numbers', () => {
    const ticker = pushTicker([], [{ type: 'goal' }, { type: 'done' }], [7]);
    expect(ticker.map((e) => e.step)).toEqual([7, -1]);
  });

  test('an empty delivery returns the same list (no re-render churn)', () => {
    const ticker: TickerEntry[] = [{ step: 1, label: 'goal', type: 'goal' }];
    expect(pushTicker(ticker, [], 2)).toBe(ticker);
  });
});


function runGovernor(
  governor: ReturnType<typeof makeTierGovernor>,
  frameMs: number,
  frames: number,
): (string | null)[] {
  const out: (string | null)[] = [];
  for (let i = 0; i < frames; i++) out.push(governor.sample(frameMs));
  return out;
}

describe('tier governor', () => {
  test('a comfortable frame time never touches the tier', () => {
    const governor = makeTierGovernor();
    const drops = runGovernor(governor, 8, 600);
    expect(drops.every((d) => d === null)).toBe(true);
    expect(governor.tier).toBe('hero');
    expect(governor.p95).toBeCloseTo(8, 6);
    expect(governor.overSeconds).toBe(0);
  });

  test('a HEALTHY vsync-locked 60 Hz page is never dropped', () => {
    const governor = makeTierGovernor();
    let dropped = false;
    for (let i = 0; i < 1200; i++) {
      if (governor.sample(i % 2 === 0 ? 16.7 : 18.6) !== null) dropped = true;
    }
    expect(dropped).toBe(false);
    expect(governor.p95).toBeCloseTo(18.6, 6);
    expect(governor.threshold).toBeCloseTo(1.5 * REFRESH_CAP_MS, 6);
    expect(governor.tier).toBe('hero');
  });

  test('at 120 Hz the absolute 20 ms budget still governs', () => {
    const governor = makeTierGovernor();
    let dropped = false;
    for (let i = 0; i < 400; i++) {
      if (governor.sample(i % 10 === 0 ? 8.3 : 22) !== null) dropped = true;
    }
    expect(governor.threshold).toBeCloseTo(20, 6);
    expect(governor.p95).toBeCloseTo(22, 6);
    expect(dropped).toBe(true);
  });

  test('vsyncFactor 0 falls back to the bare budget rule', () => {
    const governor = makeTierGovernor({ vsyncFactor: 0 });
    let dropped = false;
    for (let i = 0; i < 400; i++) {
      if (governor.sample(21) !== null) dropped = true;
    }
    expect(governor.threshold).toBe(20);
    expect(dropped).toBe(true);
  });

  test('p95 over budget for two sustained seconds drops one tier', () => {
    const governor = makeTierGovernor();
    const drops = runGovernor(governor, 30, 400);
    const first = drops.findIndex((d) => d !== null);
    expect(first).toBeGreaterThan(0);
    expect(drops[first]).toBe('medium');
    expect(first + 1).toBe(30 + Math.ceil(2 / 0.03) - 1);
    expect(governor.drops).toBeGreaterThanOrEqual(2);
    expect(governor.tier).toBe('thumb');
  });

  test('it never drops below thumb — it reports starvation instead', () => {
    const governor = makeTierGovernor({ tier: 'thumb' });
    const drops = runGovernor(governor, 40, 400);
    expect(drops.every((d) => d === null)).toBe(true);
    expect(governor.tier).toBe('thumb');
    expect(governor.starving).toBe(true);
    expect(governor.overSeconds).toBe(2);
  });

  test('a drop restarts the window, so the next one costs another full two seconds', () => {
    const governor = makeTierGovernor();
    const drops = runGovernor(governor, 30, 400);
    const indices: number[] = [];
    drops.forEach((d, i) => {
      if (d !== null) indices.push(i);
    });
    expect(indices).toHaveLength(2);
    const gap = (indices[1] ?? 0) - (indices[0] ?? 0);
    expect(gap).toBeGreaterThanOrEqual(30 + Math.floor(2000 / 30));
  });

  test('occasional spikes are spikes, not a budget failure', () => {
    const governor = makeTierGovernor();
    let dropped = false;
    for (let i = 0; i < 2000; i++) {
      if (governor.sample(i % 50 === 0 ? 60 : 8) !== null) dropped = true;
    }
    expect(dropped).toBe(false);
    expect(governor.tier).toBe('hero');
    expect(governor.p95).toBeCloseTo(8, 6);
  });

  test('a backgrounded tab (one four-second "frame") is discarded, not counted', () => {
    const governor = makeTierGovernor();
    runGovernor(governor, 5, 60);
    expect(governor.sample(4000)).toBeNull();
    expect(governor.sample(0)).toBeNull();
    expect(governor.sample(-3)).toBeNull();
    expect(governor.sample(Number.NaN)).toBeNull();
    expect(governor.p95).toBeCloseTo(5, 6);
    expect(governor.tier).toBe('hero');
  });

  test('a cold window is not judged: warmup frames cannot drop a tier', () => {
    const governor = makeTierGovernor();
    for (let i = 0; i < 29; i++) expect(governor.sample(100)).toBeNull();
    expect(governor.overSeconds).toBe(0);
    expect(governor.p95).toBe(0);
    expect(governor.threshold).toBe(0);
    expect(governor.tier).toBe('hero');
  });

  test('a scene that never once hits refresh still drops', () => {
    const governor = makeTierGovernor();
    const drops = runGovernor(governor, 30, 400);
    expect(governor.threshold).toBeCloseTo(1.5 * REFRESH_CAP_MS, 6);
    expect(drops.some((d) => d !== null)).toBe(true);
  });

  test('reset() forgets the window after a rebuild', () => {
    const governor = makeTierGovernor();
    runGovernor(governor, 30, 50);
    expect(governor.overSeconds).toBeGreaterThan(0);
    governor.reset();
    expect(governor.overSeconds).toBe(0);
    expect(governor.p95).toBe(0);
    expect(governor.tier).toBe('hero');
  });

  test('adopting a user-chosen tier restarts the window and clears starvation', () => {
    const governor = makeTierGovernor({ tier: 'thumb' });
    runGovernor(governor, 40, 200);
    expect(governor.starving).toBe(true);
    governor.adopt('hero');
    expect(governor.tier).toBe('hero');
    expect(governor.starving).toBe(false);
    expect(governor.overSeconds).toBe(0);
  });

  test('the budget and sustain window are configurable for tests and slow displays', () => {
    const governor = makeTierGovernor({ budgetMs: 33, sustainSeconds: 0.5, minSamples: 4, window: 8 });
    const drops = runGovernor(governor, 40, 100);
    expect(drops.findIndex((d) => d !== null)).toBeGreaterThanOrEqual(4);
    expect(governor.drops).toBeGreaterThan(0);
  });
});
