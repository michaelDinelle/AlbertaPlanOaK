// summary.json must be REAL JSON — the file, not just the object.
//
// The bug this locks down: v2 persists per-seed vectors, and `retentionBySeed`
// carries `nan` for any seed whose peak windowed reward rate was <= 0 (retention =
// final / peak is undefined there). Python's `json.dumps` emits that as a bare
// `NaN`, which is a Python extension — RFC 8259 / ECMA-404 have no NaN, so
// `JSON.parse` rejects the WHOLE document. `readJsonFile` in server/runIndex.ts
// catches the throw and hands the client `summary: null`, so one undefined seed
// silently blanked the entire population panel for the best-kernel run. Measured,
// not theorized: p93-tierL-20260721 carried 16 bare NaN and p7s12 carried 20.
//
// Two layers, because the object-level golden tests in summaryAdapter.test.ts
// passed straight through this (their fixture is synthetic and has no NaN):
//   1. the checked-in fixtures parse as STRICT JSON, from bytes;
//   2. fleet.py itself, fed a root where retention is genuinely undefined, still
//      writes a file that parses — the end-to-end regression test.

import { describe, expect, it } from 'bun:test';
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RunMeta } from '../../shared/types';
import { buildRunDetail, type ResolvedRun } from '../../server/runIndex';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', 'fixtures', 'server', 'summary-adapter');
const REPO = join(HERE, '..', '..', '..');
const PYTHON = join(REPO, '.venv', 'bin', 'python');
const FLEET_PY = join(REPO, 'tools', 'fleet.py');

/**
 * The tokens a strict parser rejects. `json.dumps` writes them BARE (unquoted), so
 * a word-boundary match outside quotes is what matters — but any occurrence at all
 * in these artifacts is a defect, since no legitimate string field contains them.
 */
const NON_FINITE_TOKENS = /\b(NaN|-?Infinity)\b/g;

function assertStrictJson(path: string): unknown {
  const text = readFileSync(path, 'utf8');
  const bare = text.match(NON_FINITE_TOKENS) ?? [];
  // Named explicitly: "unexpected token N" tells you nothing about which writer lied.
  expect(bare, `${path} contains bare non-finite literals: ${[...new Set(bare)].join(', ')}`).toEqual([]);
  return JSON.parse(text) as unknown;
}

describe('summary.json artifacts are strict JSON (file-level, from bytes)', () => {
  for (const name of ['summary-v1.json', 'summary-v2.json']) {
    it(`${name} parses with JSON.parse`, () => {
      const parsed = assertStrictJson(join(FIXTURES, name));
      expect(parsed).toBeTypeOf('object');
      expect(parsed).not.toBeNull();
    });
  }
});

// The end-to-end test: does fleet.py write a parseable file when retention is
// undefined? Needs the repo's .venv (the tools' documented interpreter); skipped
// rather than failed where it is absent, matching the demo-root guards elsewhere.
const toolsPresent = existsSync(PYTHON) && existsSync(FLEET_PY) && existsSync(join(FIXTURES, 'run', 'meta.json'));

describe.skipIf(!toolsPresent)('fleet.py report — a seed with no positive peak', () => {
  it('writes summary.json as valid JSON, encoding undefined retention as null', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'oak-summary-nan-'));
    try {
      // Seed 0's reward decreases monotonically BY CONSTRUCTION in the fixture
      // (make_fixture.py): every windowed rate negative, peak <= 0, so
      // `retention = end/peak` is undefined => nan. The series live on the
      // metrics tape (METRICS LAW 2026-08-05), so the fixture bakes the case
      // in rather than patching fleet.jsonl rows.
      cpSync(join(FIXTURES, 'run'), dir, { recursive: true });

      const proc = Bun.spawn([PYTHON, FLEET_PY, 'report', dir], {
        cwd: REPO,
        stdout: 'pipe',
        stderr: 'pipe',
      });
      const [code, stderr] = await Promise.all([proc.exited, new Response(proc.stderr).text()]);
      expect(stderr.length > 0 ? stderr : `exit ${code}`).toBe(`exit 0`);

      const summaryPath = join(dir, 'summary.json');
      const summary = assertStrictJson(summaryPath) as {
        results?: { policy?: string; retentionBySeed?: Record<string, number | null> }[];
      };

      // The bug is only reachable when the vector is actually written, so prove the
      // fixture reached that state rather than passing on an absent field.
      const oak = (summary.results ?? []).find((r) => r.policy === 'oak');
      expect(oak).toBeDefined();
      const retention = oak?.retentionBySeed;
      expect(retention, 'retentionBySeed must be persisted for this test to mean anything').toBeDefined();
      expect(Object.prototype.hasOwnProperty.call(retention ?? {}, '0')).toBe(true);
      // null, not NaN and not a fabricated 0: retention is UNDEFINED for this seed.
      expect(retention?.['0']).toBeNull();
      // …and the seeds that do have a positive peak still carry real numbers.
      const finite = Object.values(retention ?? {}).filter((v) => typeof v === 'number');
      expect(finite.length).toBeGreaterThan(0);

      assertStrictJson(join(dir, 'organ-report.json'));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 60_000);
});

// ---------------------------------------------------------------------------
// "no summary" and "broken summary" are different facts and must stay different.
//
// `readJsonFile` used to collapse both into `summary: null`, so the run page told
// the user "No summary.json on this root — run tools/fleet.py report" about a file
// that existed, whose contents it had just rejected. Re-running the tool rewrites
// the same broken file: a doom loop built out of a discarded error.

const META: RunMeta = {
  fleetSchemaVersion: 1,
  envId: 'HalfCheetah-v4',
  batch: 1,
  seeds: [0],
  steps: 1000,
  policies: ['oak'],
};

function resolved(dir: string): ResolvedRun {
  return {
    id: 'r',
    dir,
    meta: META,
    entry: {
      id: 'r',
      envIds: ['HalfCheetah-v4'],
      batch: 1,
      steps: 1000,
      policies: ['oak'],
      ablationLabel: 'none',
      startedAt: null,
      complete: true,
      maxStep: 1000,
      inFlight: false,
      hasSummary: false,
      hasRecordings: false,
      role: null,
    },
  };
}

describe('buildRunDetail — why the summary is missing', () => {
  function withDir<T>(fn: (dir: string) => T): T {
    const dir = mkdtempSync(join(tmpdir(), 'oak-summary-state-'));
    try {
      return fn(dir);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  it('no summary.json at all: summary null, summaryError null (a normal state)', async () => {
    const detail = await withDir((dir) => buildRunDetail(resolved(dir), []));
    expect(detail.summary).toBeNull();
    expect(detail.summaryError).toBeNull();
  });

  it('present but not JSON: summary null, summaryError "unreadable"', async () => {
    const detail = await withDir(async (dir) => {
      // Exactly the shape of the real failure: valid-looking v2 summary, one bare NaN.
      writeFileSync(
        join(dir, 'summary.json'),
        '{\n "summarySchemaVersion": 2,\n "results": [{"policy": "oak", "retentionBySeed": {"0": NaN}}]\n}\n',
      );
      return buildRunDetail(resolved(dir), []);
    });
    expect(detail.summary).toBeNull();
    expect(detail.summaryError).toBe('unreadable');
  });

  it('a truncated write is unreadable too, not missing', async () => {
    const detail = await withDir(async (dir) => {
      writeFileSync(join(dir, 'summary.json'), '{"summarySchemaVersion": 2, "results": [{"pol');
      return buildRunDetail(resolved(dir), []);
    });
    expect(detail.summary).toBeNull();
    expect(detail.summaryError).toBe('unreadable');
  });

  it('a good summary reports no error and still adapts to v2', async () => {
    const detail = await withDir(async (dir) => {
      writeFileSync(join(dir, 'summary.json'), JSON.stringify({ results: [{ policy: 'oak', lanes: 4 }] }));
      return buildRunDetail(resolved(dir), []);
    });
    expect(detail.summaryError).toBeNull();
    expect(detail.summary).not.toBeNull();
    // v1 on disk -> v2 vocabulary out (the adapter still runs on this path)
    expect((detail.summary?.results ?? [])[0]?.seeds).toBe(4);
  });
});
