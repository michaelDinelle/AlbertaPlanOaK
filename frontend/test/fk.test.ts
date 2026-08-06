/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import { CHEETAH_FK_POINTS } from '@sim/types';
import {
  CHEETAH_BONES,
  CHEETAH_HEAD_RADIUS,
  CHEETAH_RADIUS,
  FK_FLOATS,
  FK_POINT_INDEX,
  cheetahGeometry,
  fk,
  fk64,
  fkInto,
} from '@sim/fk/cheetah';

const TOL = 1e-9;

interface FkCase {
  label: string;
  qpos: number[];
  pts: number[];
  hand: number[] | null;
}

interface FkFixture {
  version: number;
  points: string[];
  constants: Record<string, unknown>;
  cases: FkCase[];
}

const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/fk_cheetah.json', import.meta.url), 'utf8'),
) as FkFixture;

function caseByLabel(label: string): FkCase {
  const c = fixture.cases.find((x) => x.label === label);
  if (!c) throw new Error(`missing fixture case ${label}`);
  return c;
}

function maxAbsDiff(got: ArrayLike<number>, want: readonly number[], round: boolean): number {
  let m = 0;
  for (let i = 0; i < want.length; i++) {
    const w = round ? Math.fround(want[i] as number) : (want[i] as number);
    const d = Math.abs((got[i] as number) - w);
    if (d > m) m = d;
  }
  return m;
}

describe('fixture integrity', () => {
  test('point order matches the frozen CHEETAH_FK_POINTS contract', () => {
    expect(fixture.points).toEqual([...CHEETAH_FK_POINTS]);
    expect(FK_FLOATS).toBe(20);
    expect(CHEETAH_FK_POINTS.length).toBe(10);
  });

  test('has the two hand-verified poses plus 50 random rows', () => {
    expect(fixture.cases.length).toBe(52);
    expect(fixture.cases.filter((c) => c.hand !== null).length).toBe(2);
    for (const c of fixture.cases) {
      expect(c.qpos.length).toBe(9);
      expect(c.pts.length).toBe(20);
    }
  });

  test('capsule constants match the XML-derived values', () => {
    expect(CHEETAH_RADIUS).toBe(0.046);
    expect(CHEETAH_HEAD_RADIUS).toBe(0.15);
    expect(cheetahGeometry.radius).toBe(CHEETAH_RADIUS);
    expect(cheetahGeometry.headRadius).toBe(CHEETAH_HEAD_RADIUS);
    expect(fixture.constants.radius).toBe(CHEETAH_RADIUS);
    expect(fixture.constants.headRadius).toBe(CHEETAH_HEAD_RADIUS);
    expect(fixture.constants.rootZ0).toBe(0.7);
  });
});

describe('hand-verified poses', () => {
  for (const label of ['zero', 'bent']) {
    test(`${label}: 20 golden points within ${TOL}`, () => {
      const c = caseByLabel(label);
      const hand = c.hand;
      expect(hand).not.toBeNull();
      expect(maxAbsDiff(c.pts, hand as number[], false)).toBeLessThanOrEqual(TOL);
      expect(maxAbsDiff(fk64(c.qpos), hand as number[], false)).toBeLessThanOrEqual(TOL);
      expect(maxAbsDiff(fk(c.qpos), hand as number[], true)).toBeLessThanOrEqual(TOL);
    });
  }

  test('zero pose reads as the canonical skeleton', () => {
    const p = fk64([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const at = (name: keyof typeof FK_POINT_INDEX): [number, number] => {
      const i = FK_POINT_INDEX[name] * 2;
      return [p[i] as number, p[i + 1] as number];
    };
    const near = (got: [number, number], want: [number, number]) => {
      expect(Math.abs(got[0] - want[0])).toBeLessThan(1e-12);
      expect(Math.abs(got[1] - want[1])).toBeLessThan(1e-12);
    };
    near(at('o1'), [0, 0.7]);
    near(at('bt'), [-0.5, 0.7]);
    near(at('ft'), [0.5, 0.7]);
    near(at('head'), [0.6, 0.8]);
    near(at('bs'), [-0.34, 0.45]);
    near(at('bf'), [-0.62, 0.31]);
    near(at('toe_b'), [-0.565, 0.122]);
    near(at('fs'), [0.36, 0.46]);
    near(at('ff'), [0.49, 0.28]);
    near(at('toe_f'), [0.575, 0.152]);
  });
});

describe('random poses', () => {
  test('all 50 random rows match the oracle within 1e-9 (f64 path)', () => {
    const rows = fixture.cases.filter((c) => c.label.startsWith('rand'));
    expect(rows.length).toBe(50);
    let worst = 0;
    for (const c of rows) {
      const d = maxAbsDiff(fk64(c.qpos), c.pts, false);
      if (d > worst) worst = d;
    }
    expect(worst).toBeLessThanOrEqual(TOL);
  });

  test('all 50 random rows land on the same float32 (f32 path)', () => {
    const rows = fixture.cases.filter((c) => c.label.startsWith('rand'));
    let worst = 0;
    for (const c of rows) {
      const d = maxAbsDiff(fk(c.qpos), c.pts, true);
      if (d > worst) worst = d;
    }
    expect(worst).toBeLessThanOrEqual(TOL);
  });
});

describe('fk mechanics', () => {
  test('writes into a caller-supplied buffer without allocating', () => {
    const c = caseByLabel('bent');
    const out = new Float32Array(FK_FLOATS);
    const ret = fk(c.qpos, out);
    expect(ret).toBe(out);
    expect(maxAbsDiff(out, c.pts, true)).toBeLessThanOrEqual(TOL);
  });

  test('fkInto accepts Float64Array and Float32Array destinations', () => {
    const c = caseByLabel('bent');
    const a = fkInto(c.qpos, new Float64Array(FK_FLOATS));
    const b = fkInto(c.qpos, new Float32Array(FK_FLOATS));
    expect(a.length).toBe(FK_FLOATS);
    expect(b.length).toBe(FK_FLOATS);
    for (let i = 0; i < FK_FLOATS; i++) {
      expect(b[i]).toBe(Math.fround(a[i] as number));
    }
  });

  test('is a pure function of qpos (repeat calls agree exactly)', () => {
    const c = caseByLabel('rand07');
    expect(Array.from(fk64(c.qpos))).toEqual(Array.from(fk64(c.qpos)));
  });

  test('rootx translates the whole skeleton rigidly', () => {
    const c = caseByLabel('rand13');
    const shifted = [...c.qpos];
    shifted[0] = (shifted[0] as number) + 12.5;
    const a = fk64(c.qpos);
    const b = fk64(shifted);
    for (let i = 0; i < FK_FLOATS; i += 2) {
      expect(Math.abs((b[i] as number) - (a[i] as number) - 12.5)).toBeLessThan(1e-12);
      expect(Math.abs((b[i + 1] as number) - (a[i + 1] as number))).toBeLessThan(1e-12);
    }
  });

  test('rootz translates the whole skeleton vertically', () => {
    const c = caseByLabel('rand21');
    const shifted = [...c.qpos];
    shifted[1] = (shifted[1] as number) + 0.25;
    const a = fk64(c.qpos);
    const b = fk64(shifted);
    for (let i = 0; i < FK_FLOATS; i += 2) {
      expect(Math.abs((b[i] as number) - (a[i] as number))).toBeLessThan(1e-12);
      expect(Math.abs((b[i + 1] as number) - (a[i + 1] as number) - 0.25)).toBeLessThan(1e-12);
    }
  });

  test('joint angles are 2*pi periodic (FK sees only sin/cos)', () => {
    const c = caseByLabel('rand31');
    const wrapped = c.qpos.map((v, i) => (i >= 2 ? v + 2 * Math.PI : v));
    const a = fk64(c.qpos);
    const b = fk64(wrapped);
    for (let i = 0; i < FK_FLOATS; i++) {
      expect(Math.abs((b[i] as number) - (a[i] as number))).toBeLessThan(1e-9);
    }
  });

  test('bones index real points and form the drawn skeleton', () => {
    expect(CHEETAH_BONES.length).toBe(7);
    for (const [a, b] of CHEETAH_BONES) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(CHEETAH_FK_POINTS.length);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(CHEETAH_FK_POINTS.length);
      expect(a).not.toBe(b);
    }
    expect(CHEETAH_BONES[0]).toEqual([FK_POINT_INDEX.bt, FK_POINT_INDEX.ft]);
    expect(CHEETAH_BONES[6]).toEqual([FK_POINT_INDEX.ff, FK_POINT_INDEX.toe_f]);
  });

  test('torso half-length is 0.5 in the rotated frame', () => {
    const p = fk64([0, 0, 0.3, 0, 0, 0, 0, 0, 0]);
    const bt = [p[2] as number, p[3] as number];
    const ft = [p[4] as number, p[5] as number];
    const len = Math.hypot((ft[0] as number) - (bt[0] as number),
                           (ft[1] as number) - (bt[1] as number));
    expect(Math.abs(len - 1.0)).toBeLessThan(1e-12);
  });
});
