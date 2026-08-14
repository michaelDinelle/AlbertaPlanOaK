/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import {
  MC_ACTIONS,
  MC_CREST_X,
  MC_FORCE,
  MC_GOAL_POS,
  MC_GRAVITY,
  MC_MAX_POS,
  MC_MAX_SPEED,
  MC_MIN_POS,
  MC_START_POS,
  MC_TROUGH_X,
  mountainCarDecoder,
  mountainCarHeight,
  mountainCarSlope,
} from '@sim/envs/mountaincar';
import type { MountainCarState, MountainCarStatic } from '@sim/envs/mountaincar';
import { RECORD_FLOATS } from '@sim/record';
import type { EnvFrame } from '@sim/types';

interface InitDoc {
  envs: Record<string, { seeds: { seed: number; pose: number[] }[] }>;
}
const initDoc = JSON.parse(
  readFileSync(new URL('../fixtures/init_states.json', import.meta.url), 'utf8'),
) as InitDoc;

const STATIC: MountainCarStatic = mountainCarDecoder.initStatic(40);

function row(pose: number[], done = 0, ctrl0 = 0): Float32Array {
  const r = new Float32Array(RECORD_FLOATS);
  r[0] = ctrl0;
  r[6] = done;
  for (let i = 0; i < pose.length && i < 9; i++) r[7 + i] = pose[i] ?? 0;
  return r;
}

function frameOf(
  i: number,
  r: Float32Array,
  prev: EnvFrame<MountainCarState>,
): EnvFrame<MountainCarState> {
  return mountainCarDecoder.decodeFrame(i, r, STATIC, prev);
}

describe('constants mirror the kernel', () => {
  test('bounds, goal and dynamics coefficients', () => {
    expect(MC_MIN_POS).toBe(-1.2);
    expect(MC_MAX_POS).toBe(0.6);
    expect(MC_GOAL_POS).toBe(0.5);
    expect(MC_MAX_SPEED).toBe(0.07);
    expect(MC_START_POS).toBe(-0.5);
    expect(MC_FORCE).toBe(0.001);
    expect(MC_GRAVITY).toBe(0.0025);
    expect(MC_ACTIONS).toBe(3);
  });
});

describe('terrain', () => {
  test('h(x) = sin(3x) * 0.45 + 0.55', () => {
    expect(mountainCarHeight(0)).toBeCloseTo(0.55, 12);
    expect(mountainCarHeight(MC_CREST_X)).toBeCloseTo(1.0, 12);
    expect(mountainCarHeight(MC_TROUGH_X)).toBeCloseTo(0.1, 12);
  });

  test('the crest is the maximum and the trough the minimum over the track', () => {
    let hi = -Infinity;
    let lo = Infinity;
    for (let i = 0; i <= 2000; i++) {
      const x = MC_MIN_POS + ((MC_MAX_POS - MC_MIN_POS) * i) / 2000;
      const h = mountainCarHeight(x);
      hi = Math.max(hi, h);
      lo = Math.min(lo, h);
    }
    expect(mountainCarHeight(MC_CREST_X)).toBeGreaterThanOrEqual(hi - 1e-9);
    expect(mountainCarHeight(MC_TROUGH_X)).toBeLessThanOrEqual(lo + 1e-9);
  });

  test('slope is the derivative of the terrain', () => {
    const eps = 1e-6;
    for (const x of [-1.2, -0.9, -0.5, 0, 0.3, 0.6]) {
      const fd = (mountainCarHeight(x + eps) - mountainCarHeight(x - eps)) / (2 * eps);
      expect(mountainCarSlope(x)).toBeCloseTo(fd, 6);
    }
    expect(mountainCarSlope(MC_CREST_X)).toBeCloseTo(0, 12);
    expect(mountainCarSlope(MC_TROUGH_X)).toBeCloseTo(0, 12);
  });

  test('the goal sits below the crest — the car must be thrown past it', () => {
    expect(MC_GOAL_POS).toBeLessThan(MC_CREST_X);
    expect(mountainCarHeight(MC_GOAL_POS)).toBeLessThan(mountainCarHeight(MC_CREST_X));
  });
});

describe('decoder contract', () => {
  test('identity', () => {
    expect(mountainCarDecoder.envId).toBe('MountainCar-cont-v0');
  });

  test('init frame is the valley start for every lane', () => {
    const seeds = initDoc.envs['MountainCar-cont-v0']?.seeds ?? [];
    expect(seeds.length).toBe(8);
    for (const want of seeds) {
      const st = mountainCarDecoder.initStatic(want.seed);
      const f = mountainCarDecoder.initFrame(want.seed, st);
      expect(f.step).toBe(-1);
      expect(f.done).toBe(0);
      expect(Array.from(f.pose)).toEqual(want.pose);
      expect(f.state.pos).toBe(MC_START_POS);
      expect(f.state.vel).toBe(0);
      expect(f.state.goal).toBe(MC_GOAL_POS);
      expect(f.state.height).toBe(mountainCarHeight(MC_START_POS));
      expect(f.state.action).toBe(0);
      expect(f.state.force).toBe(0);
    }
  });

  test('geometry exposes heightAt and anchors that sit ON the terrain', () => {
    const g = mountainCarDecoder.geometry(STATIC, 40);
    expect(g.kind).toBe('classic');
    expect(g.classic?.heightAt).toBe(mountainCarHeight);
    const a = g.classic?.anchors ?? {};
    for (const name of ['trackLeft', 'trackRight', 'goalFlag', 'successLine', 'start',
      'trough', 'crest']) {
      const p = a[name];
      expect(p).toBeDefined();
      if (!p) continue;
      expect(p[1]).toBe(mountainCarHeight(p[0]));
      expect(p[2]).toBe(0);
    }
    expect(a.trackLeft?.[0]).toBe(MC_MIN_POS);
    expect(a.trackRight?.[0]).toBe(MC_MAX_POS);
    expect(a.goalFlag?.[0]).toBe(MC_GOAL_POS);
    expect(a.start?.[0]).toBe(MC_START_POS);
    expect(mountainCarDecoder.geometry(STATIC, 40)).toBe(g);
  });

  test('pose slots map to pos, vel and the goal marker', () => {
    const prev = mountainCarDecoder.initFrame(40, STATIC);
    const f = frameOf(11, row([-0.3, 0.02, MC_GOAL_POS], 0, 2), prev);
    expect(f.step).toBe(11);
    expect(f.state.pos).toBeCloseTo(-0.3, 6);
    expect(f.state.vel).toBeCloseTo(0.02, 6);
    expect(f.state.goal).toBe(MC_GOAL_POS);
    expect(f.state.height).toBe(mountainCarHeight(f.state.pos));
  });

  test('force decodes as (action % 3) - 1', () => {
    const prev = mountainCarDecoder.initFrame(40, STATIC);
    expect(frameOf(0, row([0, 0, 0.5], 0, 0), prev).state.force).toBe(-1);
    expect(frameOf(1, row([0, 0, 0.5], 0, 1), prev).state.force).toBe(0);
    expect(frameOf(2, row([0, 0, 0.5], 0, 2), prev).state.force).toBe(1);
  });
});

describe('events', () => {
  const prev = mountainCarDecoder.initFrame(40, STATIC);

  test('an ordinary transition fires nothing', () => {
    const a = frameOf(0, row([-0.5, 0.01, 0.5]), prev);
    const b = frameOf(1, row([-0.49, 0.01, 0.5]), a);
    expect(mountainCarDecoder.deriveEvents(a, b, STATIC)).toEqual([]);
  });

  test('hitting the left wall is not an event', () => {
    const a = frameOf(0, row([-1.19, -0.03, 0.5]), prev);
    const b = frameOf(1, row([MC_MIN_POS, 0, 0.5]), a);
    expect(mountainCarDecoder.deriveEvents(a, b, STATIC)).toEqual([]);
  });

  test('done is a summit, and the car is already back at the respawn', () => {
    const a = frameOf(0, row([0.487, 0.02, 0.5]), prev);
    const b = frameOf(1, row([MC_START_POS, 0, 0.5], 1), a);
    expect(mountainCarDecoder.deriveEvents(a, b, STATIC)).toEqual([
      { type: 'goal' },
      { type: 'done' },
      { type: 'teleport' },
    ]);
    expect(b.state.pos).toBe(MC_START_POS);
  });
});
