
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

const NON_FINITE_TOKENS = /\b(NaN|-?Infinity)\b/g;

function assertStrictJson(path: string): unknown {
  const text = readFileSync(path, 'utf8');
  const bare = text.match(NON_FINITE_TOKENS) ?? [];
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

const toolsPresent = existsSync(PYTHON) && existsSync(FLEET_PY) && existsSync(join(FIXTURES, 'run', 'meta.json'));

describe.skipIf(!toolsPresent)('fleet.py report — a seed with no positive peak', () => {
  it('writes summary.json as valid JSON, encoding undefined retention as null', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'oak-summary-nan-'));
    try {
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

      const oak = (summary.results ?? []).find((r) => r.policy === 'oak');
      expect(oak).toBeDefined();
      const retention = oak?.retentionBySeed;
      expect(retention, 'retentionBySeed must be persisted for this test to mean anything').toBeDefined();
      expect(Object.prototype.hasOwnProperty.call(retention ?? {}, '0')).toBe(true);
      expect(retention?.['0']).toBeNull();
      const finite = Object.values(retention ?? {}).filter((v) => typeof v === 'number');
      expect(finite.length).toBeGreaterThan(0);

      assertStrictJson(join(dir, 'organ-report.json'));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 60_000);
});


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

  it('a legacy summary reads as missing — no adaptation, no error', async () => {
    const detail = await withDir(async (dir) => {
      writeFileSync(join(dir, 'summary.json'), JSON.stringify({ results: [{ policy: 'oak', lanes: 4 }] }));
      return buildRunDetail(resolved(dir), []);
    });
    expect(detail.summaryError).toBeNull();
    expect(detail.summary).toBeNull();
  });

  it('a current summary passes through whole', async () => {
    const detail = await withDir(async (dir) => {
      writeFileSync(
        join(dir, 'summary.json'),
        JSON.stringify({
          envSlices: { 'Chain-cont-v0': [{ policy: 'oak', recentRewardRateIQM: 0.05, seeds: 4 }] },
        }),
      );
      return buildRunDetail(resolved(dir), []);
    });
    expect(detail.summaryError).toBeNull();
    expect(detail.summary).not.toBeNull();
    expect(detail.summary?.envSlices?.['Chain-cont-v0']?.[0]?.seeds).toBe(4);
  });
});
