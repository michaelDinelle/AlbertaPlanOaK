/// <reference types="bun" />
// Philox4x32-10 / u01 golden tests.
//
// Vectors come from tools/gridworld_ref.py (the bitwise certification oracle,
// itself cross-checked against tools/physics_parity.py and OakPhysics.metal).
// Regenerate with:
//   /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py --only core

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import {
  mulhi,
  philox4x32_10,
  philoxDraw,
  philoxWords,
  STREAM_ACT,
  STREAM_CTRL,
  STREAM_GRIDWORLD,
  STREAM_RESET,
  u01,
  u01open,
  uniform,
  uniformIndex,
} from '@sim/philox';

interface PhiloxCase {
  stream: number;
  streamName: string;
  k0: number;
  k1: number;
  ctr: [number, number, number, number];
  shape: string;
  out: [number, number, number, number];
  u01: [number, number, number, number];
}

interface U01Case {
  x: number;
  u01: number;
  u01open: number;
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8')) as T;
}

const vectors = load<{ version: number; cases: PhiloxCase[] }>('philox_vectors.json');
const edges = load<{ version: number; cases: U01Case[] }>('u01_edges.json');

describe('philox4x32_10', () => {
  test('fixture is the expected shape', () => {
    expect(vectors.cases.length).toBe(64);
    for (const s of [0, 1, 2, 3]) {
      expect(vectors.cases.filter((c) => c.stream === s).length).toBe(16);
    }
  });

  test('all 64 golden vectors match bitwise', () => {
    for (const c of vectors.cases) {
      const got = philox4x32_10(c.k0, c.k1, c.ctr[0], c.ctr[1], c.ctr[2], c.ctr[3]);
      expect(got).toEqual(c.out);
      // every word is a genuine uint32
      for (const w of got) {
        expect(Number.isInteger(w)).toBe(true);
        expect(w).toBeGreaterThanOrEqual(0);
        expect(w).toBeLessThanOrEqual(0xffffffff);
      }
    }
  });

  test('per-word u01 matches exactly (fround identity)', () => {
    for (const c of vectors.cases) {
      const got = philox4x32_10(c.k0, c.k1, c.ctr[0], c.ctr[1], c.ctr[2], c.ctr[3]);
      const want = c.u01;
      expect(Object.is(u01(got[0]), want[0])).toBe(true);
      expect(Object.is(u01(got[1]), want[1])).toBe(true);
      expect(Object.is(u01(got[2]), want[2])).toBe(true);
      expect(Object.is(u01(got[3]), want[3])).toBe(true);
    }
  });

  test('philoxDraw reproduces the OaK-shaped cases', () => {
    const oak = vectors.cases.filter((c) => c.shape === 'oak');
    expect(oak.length).toBe(32);
    for (const c of oak) {
      expect(philoxDraw(c.k0, c.k1, c.ctr[0], c.ctr[1])).toEqual(c.out);
    }
  });

  test('stream ids match the kernel constants', () => {
    expect([STREAM_RESET, STREAM_CTRL, STREAM_ACT, STREAM_GRIDWORLD]).toEqual([0, 1, 2, 3]);
    const byName = new Map(vectors.cases.map((c) => [c.streamName, c.stream]));
    expect(byName.get('RESET')).toBe(STREAM_RESET);
    expect(byName.get('CTRL')).toBe(STREAM_CTRL);
    expect(byName.get('ACT')).toBe(STREAM_ACT);
    expect(byName.get('GRIDWORLD')).toBe(STREAM_GRIDWORLD);
  });

  test('different streams and counters decorrelate', () => {
    const a = philoxDraw(7, STREAM_RESET, 0, 0);
    const b = philoxDraw(7, STREAM_GRIDWORLD, 0, 0);
    const c = philoxDraw(7, STREAM_RESET, 1, 0);
    const d = philoxDraw(7, STREAM_RESET, 0, 1);
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
    expect(a).not.toEqual(d);
  });

  test('inputs are coerced to uint32 (negative / oversized are wrapped)', () => {
    expect(philox4x32_10(-1, 0, 0, 0, 0, 0)).toEqual(philox4x32_10(0xffffffff, 0, 0, 0, 0, 0));
    expect(philox4x32_10(0, 0, 0x1_0000_0001, 0, 0, 0)).toEqual(
      philox4x32_10(0, 0, 1, 0, 0, 0),
    );
  });
});

describe('mulhi', () => {
  test('matches the exact 64-bit high word', () => {
    const cases: [number, number][] = [
      [0, 0],
      [1, 1],
      [0xffffffff, 0xffffffff],
      [0xffffffff, 1],
      [0xd2511f53, 0xcd9e8d57],
      [0x80000000, 0x80000000],
      [0x0000ffff, 0xffff0000],
      [0xdeadbeef, 0x12345678],
      [0x7fffffff, 0x7fffffff],
    ];
    for (const [a, b] of cases) {
      const want = Number((BigInt(a >>> 0) * BigInt(b >>> 0)) >> 32n);
      expect(mulhi(a, b)).toBe(want);
    }
    // fuzz against BigInt (BigInt is a test-only oracle; never in the hot path)
    let s = 0x12345678;
    for (let i = 0; i < 2000; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      const t = (Math.imul(s ^ (s >>> 15), 2246822519) >>> 0) >>> 0;
      const want = Number((BigInt(s) * BigInt(t)) >> 32n);
      expect(mulhi(s, t)).toBe(want);
    }
  });
});

describe('u01 / u01open', () => {
  test('boundary values match exactly', () => {
    expect(edges.cases.length).toBeGreaterThanOrEqual(24);
    for (const c of edges.cases) {
      expect(Object.is(u01(c.x), c.u01)).toBe(true);
      expect(Object.is(u01open(c.x), c.u01open)).toBe(true);
    }
  });

  test('structural properties', () => {
    expect(u01(0)).toBe(0);
    expect(u01(0xff)).toBe(0);
    expect(u01(0x100)).toBe(Math.fround(2 ** -24));
    expect(u01(0xffffffff)).toBeLessThan(1);
    expect(u01(0xffffffff)).toBe(Math.fround((0xffffff) * 2 ** -24));
    expect(u01open(0)).toBe(Math.fround(2 ** -24));
    expect(u01open(0xffffffff)).toBe(1);
    for (const c of edges.cases) {
      expect(c.u01).toBeGreaterThanOrEqual(0);
      expect(c.u01).toBeLessThan(1);
      expect(c.u01open).toBeGreaterThan(0);
      expect(c.u01open).toBeLessThanOrEqual(1);
      // every u01 value is exactly representable in float32
      expect(Object.is(Math.fround(c.u01), c.u01)).toBe(true);
    }
  });

  test('uniform maps the unit interval onto [lo, hi)', () => {
    expect(uniform(0, -1, 1)).toBe(-1);
    expect(uniform(0, -0.1, 0.1)).toBe(Math.fround(-0.1));
    expect(uniform(0xffffffff, -1, 1)).toBeLessThan(1);
    expect(uniform(0xffffffff, -1, 1)).toBeGreaterThan(0.99999);
    expect(Object.is(Math.fround(uniform(0x12345678, -1, 1)), uniform(0x12345678, -1, 1)))
      .toBe(true);
  });

  test('uniformIndex is the gridworld free-cell draw', () => {
    expect(uniformIndex(0, 81)).toBe(0);
    expect(uniformIndex(0xffffffff, 81)).toBe(80);
    expect(uniformIndex(0xffffffff, 68)).toBe(67);
    for (let i = 0; i < 500; i++) {
      const w = philoxDraw(i, STREAM_GRIDWORLD, i, 0)[0];
      const idx = uniformIndex(w, 81);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(81);
    }
  });
});

describe('philoxWords', () => {
  test('concatenates consecutive blocks in order', () => {
    const got = philoxWords(42, STREAM_RESET, 3, 12);
    const b0 = philoxDraw(42, STREAM_RESET, 3, 0);
    const b1 = philoxDraw(42, STREAM_RESET, 3, 1);
    const b2 = philoxDraw(42, STREAM_RESET, 3, 2);
    expect(Array.from(got)).toEqual([...b0, ...b1, ...b2]);
  });

  test('honours blockStart and partial counts', () => {
    const got = philoxWords(42, STREAM_RESET, 3, 6, 3);
    const b3 = philoxDraw(42, STREAM_RESET, 3, 3);
    const b4 = philoxDraw(42, STREAM_RESET, 3, 4);
    expect(Array.from(got)).toEqual([...b3, b4[0], b4[1]]);
    expect(got.length).toBe(6);
  });
});
