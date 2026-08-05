/// <reference types="bun" />

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  parseFieldPath,
  parseJsonl,
  parseJsonlDetailed,
  pluckNumber,
  readJsonlFrom,
  readTail,
  statOrNull,
  streamJsonl,
} from '../../server/jsonl';

const HERE = dirname(fileURLToPath(import.meta.url));
const TMP = join(HERE, '..', 'fixtures', 'server', 'jsonl-tmp');

beforeAll(async () => {
  await rm(TMP, { recursive: true, force: true });
  await mkdir(TMP, { recursive: true });
});

afterAll(async () => {
  await rm(TMP, { recursive: true, force: true });
});

describe('parseJsonlDetailed', () => {
  it('parses complete rows and ignores blank lines', () => {
    const text = '{"a":1}\n\n{"a":2}\n';
    const r = parseJsonlDetailed<{ a: number }>(text);
    expect(r.rows).toEqual([{ a: 1 }, { a: 2 }]);
    expect(r.skipped).toBe(0);
    expect(r.truncatedTail).toBe(false);
  });

  it('keeps earlier rows when the final line is truncated mid-write', () => {
    const text = '{"a":1}\n{"a":2}\n{"a":3,"health":{"rew';
    const r = parseJsonlDetailed<{ a: number }>(text);
    expect(r.rows).toEqual([{ a: 1 }, { a: 2 }]);
    expect(r.skipped).toBe(1);
    expect(r.truncatedTail).toBe(true);
  });

  it('does not call a mid-file bad line a truncated tail', () => {
    const text = '{"a":1}\n{oops\n{"a":3}\n';
    const r = parseJsonlDetailed<{ a: number }>(text);
    expect(r.rows).toEqual([{ a: 1 }, { a: 3 }]);
    expect(r.skipped).toBe(1);
    expect(r.truncatedTail).toBe(false);
  });

  it('handles empty input and a lone truncated row', () => {
    expect(parseJsonl('')).toEqual([]);
    const r = parseJsonlDetailed('{"partial":');
    expect(r.rows).toEqual([]);
    expect(r.truncatedTail).toBe(true);
  });
});

describe('readTail', () => {
  const path = () => join(TMP, 'tail.jsonl');

  beforeAll(async () => {
    const lines: string[] = [];
    for (let i = 0; i < 200; i++) lines.push(JSON.stringify({ i, pad: 'x'.repeat(64) }));
    await writeFile(path(), `${lines.join('\n')}\n`);
  });

  it('returns null for a missing file', async () => {
    expect(await readTail(join(TMP, 'nope.jsonl'), 1024)).toBeNull();
  });

  it('returns the whole file when the window covers it', async () => {
    const t = await readTail(path(), 1 << 20);
    expect(t).not.toBeNull();
    expect(t?.windowed).toBe(false);
    expect(parseJsonl(t?.text ?? '').length).toBe(200);
  });

  it('starts a windowed read on a row boundary (drops the leading partial line)', async () => {
    const t = await readTail(path(), 1000);
    expect(t?.windowed).toBe(true);
    const rows = parseJsonl<{ i: number }>(t?.text ?? '');
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.length).toBeLessThan(200);
    expect(rows[rows.length - 1]?.i).toBe(199);
    expect(t?.text.startsWith('{"i":')).toBe(true);
  });
});

describe('readJsonlFrom', () => {
  it('only advances past complete rows, then picks up the completed row', async () => {
    const p = join(TMP, 'grow.jsonl');
    await writeFile(p, '{"n":1}\n{"n":2}\n{"n":3');
    const first = await readJsonlFrom<{ n: number }>(p, 0);
    expect(first?.rows).toEqual([{ n: 1 }, { n: 2 }]);
    expect(first?.nextOffset).toBe(16);
    expect(first?.reset).toBe(false);

    await writeFile(p, '{"n":1}\n{"n":2}\n{"n":3}\n{"n":4}\n');
    const second = await readJsonlFrom<{ n: number }>(p, first?.nextOffset ?? 0);
    expect(second?.rows).toEqual([{ n: 3 }, { n: 4 }]);
  });

  it('resets when the file shrank (rewritten run root)', async () => {
    const p = join(TMP, 'shrink.jsonl');
    await writeFile(p, '{"n":1}\n{"n":2}\n');
    const r = await readJsonlFrom<{ n: number }>(p, 9999);
    expect(r?.reset).toBe(true);
    expect(r?.rows.length).toBe(2);
  });

  it('returns null for a missing file', async () => {
    expect(await readJsonlFrom(join(TMP, 'gone.jsonl'), 0)).toBeNull();
  });
});

describe('streamJsonl', () => {
  it('streams every row and tolerates a truncated tail', async () => {
    const p = join(TMP, 'stream.jsonl');
    const lines: string[] = [];
    for (let i = 0; i < 5000; i++) lines.push(JSON.stringify({ i, health: { rewardRate: i / 10 } }));
    await writeFile(p, `${lines.join('\n')}\n{"i":5000,"health":{"rewa`);

    const seen: number[] = [];
    const res = await streamJsonl<{ i: number }>(p, (row) => seen.push(row.i));
    expect(res.rows).toBe(5000);
    expect(res.skipped).toBe(1);
    expect(seen[0]).toBe(0);
    expect(seen[seen.length - 1]).toBe(4999);
  });

  it('is a no-op on a missing file', async () => {
    const res = await streamJsonl(join(TMP, 'absent.jsonl'), () => {
      throw new Error('should not be called');
    });
    expect(res).toEqual({ rows: 0, skipped: 0 });
  });
});

describe('parseFieldPath', () => {
  it('accepts plain and dotted health paths', () => {
    expect(parseFieldPath('rewardSum')).toEqual(['rewardSum']);
    expect(parseFieldPath('health.rewardRate')).toEqual(['health', 'rewardRate']);
    expect(parseFieldPath('health.nonFiniteByPhase.encode')).toEqual([
      'health',
      'nonFiniteByPhase',
      'encode',
    ]);
  });

  it('rejects prototype-pollution and path-shaped fields', () => {
    expect(parseFieldPath('__proto__')).toBeNull();
    expect(parseFieldPath('health.__proto__')).toBeNull();
    expect(parseFieldPath('health.constructor')).toBeNull();
    expect(parseFieldPath('../../etc')).toBeNull();
    expect(parseFieldPath('a-b')).toBeNull();
    expect(parseFieldPath('1abc')).toBeNull();
    expect(parseFieldPath('')).toBeNull();
    expect(parseFieldPath('a.b.c.d.e')).toBeNull();
  });
});

describe('pluckNumber', () => {
  const row = { rewardSum: -12.5, health: { rewardRate: 0.25, nested: null }, kind: 'hb', flag: true };

  it('reads top-level and nested numbers', () => {
    expect(pluckNumber(row, ['rewardSum'])).toBe(-12.5);
    expect(pluckNumber(row, ['health', 'rewardRate'])).toBe(0.25);
  });

  it('answers null for missing, non-numeric and non-finite leaves', () => {
    expect(pluckNumber(row, ['nope'])).toBeNull();
    expect(pluckNumber(row, ['health', 'nested', 'deep'])).toBeNull();
    expect(pluckNumber(row, ['kind'])).toBeNull();
    expect(pluckNumber({ x: Number.POSITIVE_INFINITY }, ['x'])).toBeNull();
    expect(pluckNumber({ x: Number.NaN }, ['x'])).toBeNull();
  });

  it('coerces booleans so flag-like health fields chart', () => {
    expect(pluckNumber(row, ['flag'])).toBe(1);
  });
});

describe('statOrNull', () => {
  it('answers null instead of throwing', async () => {
    expect(await statOrNull(join(TMP, 'definitely-absent'))).toBeNull();
    const p = join(TMP, 'present.txt');
    await writeFile(p, 'hi');
    expect((await statOrNull(p))?.size).toBe(2);
  });
});
