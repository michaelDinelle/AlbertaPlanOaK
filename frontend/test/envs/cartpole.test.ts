/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import {
  CARTPOLE_POINT_FLOATS,
  CP_ACTIONS,
  CP_CART_HEIGHT,
  CP_CART_WIDTH,
  CP_FORCE_MAG,
  CP_POLE_HALF_LENGTH,
  CP_POLE_LENGTH,
  CP_TAU,
  CP_TH_LIMIT,
  CP_X_LIMIT,
  cartPoleDecoder,
  cartPoleTip,
} from '@sim/envs/cartpole';
import type { CartPoleState, CartPoleStatic } from '@sim/envs/cartpole';
import { RECORD_FLOATS } from '@sim/record';
import type { EnvFrame } from '@sim/types';

interface InitDoc {
  envs: Record<string, { seeds: { seed: number; pose: number[] }[] }>;
}
const initDoc = JSON.parse(
  readFileSync(new URL('../fixtures/init_states.json', import.meta.url), 'utf8'),
) as InitDoc;

const STATIC: CartPoleStatic = cartPoleDecoder.initStatic(48);

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
  prev: EnvFrame<CartPoleState>,
): EnvFrame<CartPoleState> {
  return cartPoleDecoder.decodeFrame(i, r, STATIC, prev);
}

describe('constants mirror the kernel', () => {
  test('the failure bounds are the fp32 literals the kernel compares against', () => {
    expect(CP_X_LIMIT).toBe(Math.fround(2.4));
    expect(CP_TH_LIMIT).toBe(Math.fround(0.2094395));
    expect(CP_TH_LIMIT).toBeCloseTo((12 * Math.PI) / 180, 6);
  });

  test('step, force and pole geometry', () => {
    expect(CP_TAU).toBe(0.02);
    expect(CP_FORCE_MAG).toBe(10);
    expect(CP_ACTIONS).toBe(2);
    expect(CP_POLE_LENGTH).toBe(2 * CP_POLE_HALF_LENGTH);
    expect(CP_CART_WIDTH).toBeGreaterThan(0);
    expect(CP_CART_HEIGHT).toBeGreaterThan(0);
  });
});

describe('kinematics', () => {
  test('an upright pole stands straight above the cart', () => {
    const p = cartPoleTip(0, 0);
    expect(p.length).toBe(CARTPOLE_POINT_FLOATS);
    expect(Array.from(p)).toEqual([0, CP_POLE_LENGTH]);
    expect(Array.from(cartPoleTip(1.5, 0))).toEqual([1.5, CP_POLE_LENGTH]);
  });

  test('positive theta leans the tip toward +x', () => {
    const p = cartPoleTip(0, 0.2);
    expect(p[0] ?? 0).toBeGreaterThan(0);
    expect(p[1] ?? 0).toBeGreaterThan(0);
    expect(cartPoleTip(0, -0.2)[0] ?? 0).toBeLessThan(0);
  });

  test('the pole keeps its length', () => {
    for (let i = -10; i <= 10; i++) {
      const th = i * 0.02;
      const p = cartPoleTip(0.7, th);
      expect(Math.hypot((p[0] ?? 0) - 0.7, p[1] ?? 0)).toBeCloseTo(CP_POLE_LENGTH, 6);
    }
  });

  test('writes into a supplied buffer without allocating', () => {
    const out = new Float32Array(CARTPOLE_POINT_FLOATS);
    expect(cartPoleTip(0.1, 0.05, out)).toBe(out);
  });
});

describe('decoder contract', () => {
  test('identity', () => {
    expect(cartPoleDecoder.envId).toBe('CartPole-cont-v0');
  });

  test('init frame is the upright zero state for every lane', () => {
    const seeds = initDoc.envs['CartPole-cont-v0']?.seeds ?? [];
    expect(seeds.length).toBe(8);
    for (const want of seeds) {
      const st = cartPoleDecoder.initStatic(want.seed);
      const f = cartPoleDecoder.initFrame(want.seed, st);
      expect(f.step).toBe(-1);
      expect(f.done).toBe(0);
      expect(Array.from(f.pose)).toEqual(want.pose);
      expect(f.state.x).toBe(0);
      expect(f.state.xDot).toBe(0);
      expect(f.state.th).toBe(0);
      expect(f.state.thDot).toBe(0);
      expect(f.state.xMargin).toBe(0);
      expect(f.state.thMargin).toBe(0);
      expect(f.state.action).toBe(0);
      expect(f.state.force).toBe(0);
    }
  });

  test('geometry anchors land on the failure bounds and the tilt arcs', () => {
    const g = cartPoleDecoder.geometry(STATIC, 48);
    expect(g.kind).toBe('classic');
    const a = g.classic?.anchors ?? {};
    expect(a.pivot).toEqual([0, 0, 0]);
    expect(a.trackLeft).toEqual([-CP_X_LIMIT, 0, 0]);
    expect(a.trackRight).toEqual([CP_X_LIMIT, 0, 0]);
    expect(a.failLeft).toEqual(a.trackLeft as readonly [number, number, number]);
    expect(a.failRight).toEqual(a.trackRight as readonly [number, number, number]);
    expect(a.poleTipRest).toEqual([0, CP_POLE_LENGTH, 0]);
    const right = cartPoleTip(0, CP_TH_LIMIT);
    expect(a.failArcRight?.[0]).toBeCloseTo(right[0] ?? NaN, 6);
    expect(a.failArcRight?.[1]).toBeCloseTo(right[1] ?? NaN, 6);
    expect(a.failArcLeft?.[0]).toBeCloseTo(-(right[0] ?? NaN), 6);
    expect(a.cartHalfExtent).toEqual([CP_CART_WIDTH / 2, CP_CART_HEIGHT / 2, 0]);
    expect(g.classic?.heightAt).toBeUndefined();
    expect(cartPoleDecoder.geometry(STATIC, 48)).toBe(g);
  });

  test('pose slots map to x, xdot, th, thdot', () => {
    const prev = cartPoleDecoder.initFrame(48, STATIC);
    const f = frameOf(5, row([1.2, -0.5, 0.1, 0.75], 0, 1), prev);
    expect(f.step).toBe(5);
    expect(f.state.x).toBeCloseTo(1.2, 6);
    expect(f.state.xDot).toBeCloseTo(-0.5, 6);
    expect(f.state.th).toBeCloseTo(0.1, 6);
    expect(f.state.thDot).toBeCloseTo(0.75, 6);
  });

  test('force is +-10 from the action bit', () => {
    const prev = cartPoleDecoder.initFrame(48, STATIC);
    expect(frameOf(0, row([0, 0, 0, 0], 0, 0), prev).state.force).toBe(-CP_FORCE_MAG);
    expect(frameOf(1, row([0, 0, 0, 0], 0, 1), prev).state.force).toBe(CP_FORCE_MAG);
  });

  test('margins report how close the frame is to failing', () => {
    const prev = cartPoleDecoder.initFrame(48, STATIC);
    const half = frameOf(0, row([CP_X_LIMIT / 2, 0, -CP_TH_LIMIT / 2, 0]), prev);
    expect(half.state.xMargin).toBeCloseTo(0.5, 6);
    expect(half.state.thMargin).toBeCloseTo(0.5, 6);
    const over = frameOf(1, row([CP_X_LIMIT * 3, 0, CP_TH_LIMIT * 3, 0]), prev);
    expect(over.state.xMargin).toBe(1);
    expect(over.state.thMargin).toBe(1);
  });
});

describe('events', () => {
  const prev = cartPoleDecoder.initFrame(48, STATIC);

  test('an ordinary surviving step fires nothing', () => {
    const a = frameOf(0, row([0, -0.19, 0, 0.29]), prev);
    const b = frameOf(1, row([-0.004, -0.39, 0.006, 0.58]), a);
    expect(cartPoleDecoder.deriveEvents(a, b, STATIC)).toEqual([]);
  });

  test('done is a failure reset — never a goal', () => {
    const a = frameOf(0, row([-0.18, -1.2, -0.2, -1.4]), prev);
    const b = frameOf(1, row([0, 0, 0, 0], 1), a);
    const evs = cartPoleDecoder.deriveEvents(a, b, STATIC);
    expect(evs).toEqual([{ type: 'done' }, { type: 'teleport' }]);
    expect(evs.some((e) => e.type === 'goal')).toBe(false);
  });
});
