/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import { decodeRecording } from '@sim/decode';
import {
  PD_DT,
  PD_G,
  PD_L,
  PD_MAX_SPEED,
  PD_MAX_TORQUE,
  PENDULUM_POINT_FLOATS,
  pendulumDecoder,
  pendulumTip,
  wrapPi,
} from '@sim/envs/pendulum';
import type { PendulumState, PendulumStatic } from '@sim/envs/pendulum';
import { RECORD_FLOATS } from '@sim/record';
import { parseRawTrace } from '../helpers/rawTrace';
import type { EnvFrame } from '@sim/types';

interface InitDoc {
  envs: Record<string, { seeds: { seed: number; pose: number[] }[] }>;
}
const initDoc = JSON.parse(
  readFileSync(new URL('../fixtures/init_states.json', import.meta.url), 'utf8'),
) as InitDoc;

const STATIC: PendulumStatic = pendulumDecoder.initStatic(56);

function loadTrace(): EnvFrame<PendulumState>[] {
  const buf = readFileSync(new URL('../fixtures/traces/Pendulum-cont-v0.bin', import.meta.url));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  const raw = parseRawTrace(ab, { envId: 'Pendulum-cont-v0', seed: 56 });
  return decodeRecording(raw, pendulumDecoder).frames;
}

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
  prev: EnvFrame<PendulumState>,
): EnvFrame<PendulumState> {
  return pendulumDecoder.decodeFrame(i, r, STATIC, prev);
}

function residualForGravitySign(frames: EnvFrame<PendulumState>[], sign: number): number {
  let worst = 0;
  for (let i = 1; i < frames.length; i++) {
    const pre = frames[i - 1]?.state;
    const post = frames[i]?.state;
    if (!pre || !post) continue;
    let nthd = pre.thDot + (sign * (PD_G * 1.5) * Math.sin(pre.th) + 3 * post.torque) * PD_DT;
    const nth = wrapPi(pre.th + nthd * PD_DT);
    nthd = Math.max(-PD_MAX_SPEED, Math.min(PD_MAX_SPEED, nthd));
    worst = Math.max(worst, Math.abs(nthd - post.thDot), Math.abs(wrapPi(nth - post.th)));
  }
  return worst;
}

describe('angle convention (falsified against recorded data)', () => {
  const frames = loadTrace();

  test('the trace decodes to 201 frames starting at the inverted init', () => {
    expect(frames.length).toBe(201);
    expect(frames[0]?.step).toBe(-1);
    expect(frames[0]?.state.th).toBe(Math.fround(Math.PI));
  });

  test('gravity pulls toward th = 0, so th = 0 is hanging DOWN', () => {
    const correct = residualForGravitySign(frames, -1);
    const flipped = residualForGravitySign(frames, +1);
    expect(correct).toBeLessThan(1e-5);
    expect(flipped).toBeGreaterThan(0.1);
    expect(flipped / Math.max(correct, 1e-12)).toBeGreaterThan(1e4);
  });

  test('the lane falls away from the inverted init within the trace', () => {
    expect(Math.abs(frames[0]?.state.th ?? 0)).toBeGreaterThan(3.1);
    expect(Math.abs(frames[200]?.state.th ?? 0)).toBeLessThan(2);
  });

  test('recorded state stays inside the kernel clips', () => {
    for (const f of frames) {
      expect(Math.abs(f.state.th)).toBeLessThanOrEqual(Math.PI + 1e-6);
      expect(Math.abs(f.state.thDot)).toBeLessThanOrEqual(PD_MAX_SPEED + 1e-6);
      expect(Math.abs(f.state.torque)).toBeLessThanOrEqual(PD_MAX_TORQUE + 1e-6);
    }
  });
});

describe('constants mirror the kernel', () => {
  test('dt, clips and unit link', () => {
    expect(PD_DT).toBe(0.05);
    expect(PD_MAX_SPEED).toBe(8);
    expect(PD_MAX_TORQUE).toBe(2);
    expect(PD_G).toBe(10);
    expect(PD_L).toBe(1);
    expect((3 * PD_G) / (2 * PD_L)).toBe(15);
  });
});

describe('kinematics', () => {
  test('th = 0 hangs down and th = pi points up', () => {
    const down = pendulumTip(0);
    expect(down.length).toBe(PENDULUM_POINT_FLOATS);
    expect(Array.from(down)).toEqual([0, -PD_L]);
    const up = pendulumTip(Math.PI);
    expect(up[0] ?? NaN).toBeCloseTo(0, 6);
    expect(up[1] ?? NaN).toBeCloseTo(PD_L, 6);
  });

  test('positive theta swings the bob toward +x', () => {
    expect(pendulumTip(0.5)[0] ?? 0).toBeGreaterThan(0);
    expect(pendulumTip(-0.5)[0] ?? 0).toBeLessThan(0);
  });

  test('the rod keeps its length', () => {
    for (let i = -8; i <= 8; i++) {
      const p = pendulumTip(i * 0.4);
      expect(Math.hypot(p[0] ?? 0, p[1] ?? 0)).toBeCloseTo(PD_L, 6);
    }
  });

  test('writes into a supplied buffer without allocating', () => {
    const out = new Float32Array(PENDULUM_POINT_FLOATS);
    expect(pendulumTip(0.3, out)).toBe(out);
  });
});

describe('wrapPi', () => {
  test('maps into [-pi, pi)', () => {
    for (let i = -20; i <= 20; i++) {
      const x = i * 0.7;
      const w = wrapPi(x);
      expect(w).toBeGreaterThanOrEqual(-Math.PI);
      expect(w).toBeLessThan(Math.PI);
      expect(Math.abs(Math.sin(w) - Math.sin(x))).toBeLessThan(1e-12);
      expect(Math.abs(Math.cos(w) - Math.cos(x))).toBeLessThan(1e-12);
    }
  });

  test('leaves an already-wrapped angle alone', () => {
    expect(wrapPi(0)).toBe(0);
    expect(wrapPi(1)).toBe(1);
    expect(wrapPi(-1)).toBe(-1);
    expect(wrapPi(Math.PI)).toBeCloseTo(-Math.PI, 12);
  });
});

describe('decoder contract', () => {
  test('identity', () => {
    expect(pendulumDecoder.envId).toBe('Pendulum-cont-v0');
    expect(pendulumDecoder.needsReplay).toBe(false);
  });

  test('init frame is the inverted start for every lane', () => {
    const seeds = initDoc.envs['Pendulum-cont-v0']?.seeds ?? [];
    expect(seeds.length).toBe(8);
    for (const want of seeds) {
      const st = pendulumDecoder.initStatic(want.seed);
      const f = pendulumDecoder.initFrame(want.seed, st);
      expect(f.step).toBe(-1);
      expect(f.done).toBe(0);
      expect(Array.from(f.pose)).toEqual(want.pose);
      expect(f.state.th).toBe(Math.fround(Math.PI));
      expect(f.state.thDot).toBe(0);
      expect(f.state.torque).toBe(0);
      expect(f.state.effort).toBe(0);
      expect(st.initTheta).toBe(Math.fround(Math.PI));
    }
  });

  test('geometry anchors follow the same convention as pendulumTip', () => {
    const g = pendulumDecoder.geometry(STATIC, 56);
    expect(g.kind).toBe('classic');
    const a = g.classic?.anchors ?? {};
    expect(a.pivot).toEqual([0, 0, 0]);
    expect(a.restTip).toEqual([0, -PD_L, 0]);
    expect(a.invertedTip).toEqual([0, PD_L, 0]);
    expect(a.restTip?.[1]).toBe(pendulumTip(0)[1] ?? NaN);
    expect(g.classic?.heightAt).toBeUndefined();
    expect(pendulumDecoder.geometry(STATIC, 56)).toBe(g);
  });

  test('pose slots map to th, thdot and the clamped torque', () => {
    const prev = pendulumDecoder.initFrame(56, STATIC);
    const f = frameOf(9, row([1.25, -2.5, 1.5], 0, 0.75), prev);
    expect(f.step).toBe(9);
    expect(f.state.th).toBe(1.25);
    expect(f.state.thDot).toBe(-2.5);
    expect(f.state.torque).toBe(1.5);
    expect(f.state.action).toBe(0.75);
    expect(f.state.effort).toBe(0.75);
  });

  test('effort saturates at the torque clamp', () => {
    const prev = pendulumDecoder.initFrame(56, STATIC);
    expect(frameOf(0, row([0, 0, -PD_MAX_TORQUE]), prev).state.effort).toBe(1);
    expect(frameOf(1, row([0, 0, 0]), prev).state.effort).toBe(0);
  });
});

describe('events', () => {
  const prev = pendulumDecoder.initFrame(56, STATIC);

  test('a continuing env fires nothing, not even across a wrap', () => {
    const a = frameOf(0, row([3.1409824, 0.1, 0]), prev);
    const b = frameOf(1, row([-3.1396763, 0.12, 0]), a);
    expect(pendulumDecoder.deriveEvents(a, b, STATIC)).toEqual([]);
  });

  test('the whole trace is event-free', () => {
    const buf = readFileSync(new URL('../fixtures/traces/Pendulum-cont-v0.bin', import.meta.url));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
    const raw = parseRawTrace(ab, { envId: 'Pendulum-cont-v0', seed: 56 });
    const dec = decodeRecording(raw, pendulumDecoder);
    expect(dec.events.every((e) => e.length === 0)).toBe(true);
    expect(dec.frames.every((f) => f.done === 0)).toBe(true);
  });
});
