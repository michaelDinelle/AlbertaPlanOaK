/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import {
  AB_ACTIONS,
  AB_DT,
  AB_L1,
  AB_L2,
  AB_SUCCESS_HEIGHT,
  AB_W1MAX,
  AB_W2MAX,
  ACROBOT_POINT_FLOATS,
  acrobotDecoder,
  acrobotPoints,
  acrobotTipHeight,
} from '@sim/envs/acrobot';
import type { AcrobotState, AcrobotStatic } from '@sim/envs/acrobot';
import { RECORD_FLOATS } from '@sim/record';
import type { EnvFrame } from '@sim/types';

interface InitDoc {
  envs: Record<string, { seeds: { seed: number; pose: number[] }[] }>;
}
const initDoc = JSON.parse(
  readFileSync(new URL('../fixtures/init_states.json', import.meta.url), 'utf8'),
) as InitDoc;

const STATIC: AcrobotStatic = acrobotDecoder.initStatic(32);

function row(pose: number[], done = 0, ctrl0 = 0): Float32Array {
  const r = new Float32Array(RECORD_FLOATS);
  r[0] = ctrl0;
  r[6] = done;
  for (let i = 0; i < pose.length && i < 9; i++) r[7 + i] = pose[i] ?? 0;
  return r;
}

function frameOf(i: number, r: Float32Array, prev: EnvFrame<AcrobotState>): EnvFrame<AcrobotState> {
  return acrobotDecoder.decodeFrame(i, r, STATIC, prev);
}

describe('constants mirror the kernel', () => {
  test('velocity clips are the fp32 literals of 4pi and 9pi', () => {
    expect(AB_W1MAX).toBe(Math.fround(4 * Math.PI));
    expect(AB_W2MAX).toBe(Math.fround(9 * Math.PI));
  });

  test('integrator step and unit links', () => {
    expect(AB_DT).toBe(0.2);
    expect(AB_L1).toBe(1);
    expect(AB_L2).toBe(1);
    expect(AB_ACTIONS).toBe(3);
    expect(AB_SUCCESS_HEIGHT).toBe(1);
  });
});

describe('kinematics', () => {
  test('the hang state puts the elbow at -1 and the tip at -2', () => {
    const p = acrobotPoints(0, 0);
    expect(p.length).toBe(ACROBOT_POINT_FLOATS);
    expect(Array.from(p)).toEqual([0, -1, 0, -2]);
    expect(acrobotTipHeight(0, 0)).toBe(-2);
  });

  test('the fully inverted state puts the tip at +2 (the reward maximum)', () => {
    expect(acrobotTipHeight(Math.PI, 0)).toBeCloseTo(2, 12);
    const p = acrobotPoints(Math.PI, 0);
    expect(p[3]).toBeCloseTo(2, 6);
  });

  test('the tip y IS the env success quantity', () => {
    const out = new Float32Array(ACROBOT_POINT_FLOATS);
    for (let i = 0; i < 64; i++) {
      const th1 = -Math.PI + (2 * Math.PI * i) / 64;
      const th2 = Math.PI - (3 * Math.PI * i) / 64;
      acrobotPoints(th1, th2, out);
      expect(out[3] ?? NaN).toBeCloseTo(acrobotTipHeight(th1, th2), 6);
    }
  });

  test('link lengths are preserved for every configuration', () => {
    const out = new Float32Array(ACROBOT_POINT_FLOATS);
    for (let i = 0; i < 32; i++) {
      const th1 = (7 * i) % 6.28;
      const th2 = (11 * i) % 6.28;
      acrobotPoints(th1, th2, out);
      const [ex, ey, tx, ty] = [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0, out[3] ?? 0];
      expect(Math.hypot(ex, ey)).toBeCloseTo(AB_L1, 6);
      expect(Math.hypot(tx - ex, ty - ey)).toBeCloseTo(AB_L2, 6);
    }
  });

  test('writes into a supplied buffer without allocating', () => {
    const out = new Float32Array(ACROBOT_POINT_FLOATS);
    expect(acrobotPoints(0.3, -0.2, out)).toBe(out);
  });
});

describe('decoder contract', () => {
  test('identity', () => {
    expect(acrobotDecoder.envId).toBe('Acrobot-cont-v0');
  });

  test('init frame is the deterministic hang state for every lane', () => {
    const seeds = initDoc.envs['Acrobot-cont-v0']?.seeds ?? [];
    expect(seeds.length).toBe(8);
    for (const want of seeds) {
      const st = acrobotDecoder.initStatic(want.seed);
      const f = acrobotDecoder.initFrame(want.seed, st);
      expect(f.step).toBe(-1);
      expect(f.done).toBe(0);
      expect(Array.from(f.pose)).toEqual(want.pose);
      expect(f.state.th1).toBe(0);
      expect(f.state.th2).toBe(0);
      expect(f.state.w1).toBe(0);
      expect(f.state.w2).toBe(0);
      expect(f.state.tipHeight).toBe(-2);
      expect(f.state.action).toBe(0);
      expect(f.state.torque).toBe(0);
    }
  });

  test('geometry anchors match the kinematics', () => {
    const g = acrobotDecoder.geometry(STATIC, 32);
    expect(g.kind).toBe('classic');
    const a = g.classic?.anchors ?? {};
    expect(a.pivot).toEqual([0, 0, 0]);
    expect(a.restElbow).toEqual([0, -AB_L1, 0]);
    expect(a.restTip).toEqual([0, -(AB_L1 + AB_L2), 0]);
    expect(a.successLine).toEqual([0, AB_SUCCESS_HEIGHT, 0]);
    expect(g.classic?.heightAt).toBeUndefined();
    expect(acrobotDecoder.geometry(STATIC, 32)).toBe(g);
  });

  test('pose slots map to th1, th2, w1, w2 and the rest are unused', () => {
    const prev = acrobotDecoder.initFrame(32, STATIC);
    const f = frameOf(7, row([0.25, -0.5, 3, -4, 0, 0, 0, 0, 0], 0, 2), prev);
    expect(f.step).toBe(7);
    expect(f.state.th1).toBe(0.25);
    expect(f.state.th2).toBe(-0.5);
    expect(f.state.w1).toBe(3);
    expect(f.state.w2).toBe(-4);
    expect(f.state.tipHeight).toBe(acrobotTipHeight(0.25, -0.5));
  });

  test('torque decodes as (action % 3) - 1', () => {
    const prev = acrobotDecoder.initFrame(32, STATIC);
    expect(frameOf(0, row([0, 0, 0, 0], 0, 0), prev).state.torque).toBe(-1);
    expect(frameOf(1, row([0, 0, 0, 0], 0, 1), prev).state.torque).toBe(0);
    expect(frameOf(2, row([0, 0, 0, 0], 0, 2), prev).state.torque).toBe(1);
  });
});

describe('events', () => {
  const prev = acrobotDecoder.initFrame(32, STATIC);

  test('an ordinary transition fires nothing', () => {
    const a = frameOf(0, row([0.1, 0.1, 0, 0]), prev);
    const b = frameOf(1, row([0.2, 0.1, 0, 0]), a);
    expect(acrobotDecoder.deriveEvents(a, b, STATIC)).toEqual([]);
  });

  test('done is a swing-up success, and the state snaps back to hang', () => {
    const a = frameOf(0, row([1.5, 0.6, 2.6, -0.7]), prev);
    const b = frameOf(1, row([0, 0, 0, 0], 1), a);
    expect(acrobotDecoder.deriveEvents(a, b, STATIC)).toEqual([
      { type: 'goal' },
      { type: 'done' },
      { type: 'teleport' },
    ]);
    expect(b.state.tipHeight).toBe(-2);
  });
});
