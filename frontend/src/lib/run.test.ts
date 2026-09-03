import { expect, test } from 'bun:test';
import { runDateLabel, runLabel, runShortId, type RunLabelSource } from './run';

const best: RunLabelSource = {
  id: 'c27eb7010fa644c770fdebf22f1527c82cfbce68',
  startedAt: '2026-08-06T18:30:00Z',
  role: 'best',
  ablationLabel: 'none',
  policy: 'oak',
};

test('runLabel names the best with a date and the SHA-7', () => {
  const label = runLabel(best);
  expect(label).toContain('Aug');
  expect(label).toContain('best');
  expect(label).toContain(best.id.slice(0, 7));
  expect(label).not.toContain(best.id.slice(0, 8));
});

test('runLabel distinguishes two runs on the same day', () => {
  const twin = { ...best, id: 'ee2d4233b1f0c9a58d6e7204c3ab91d05f62e871' };
  expect(runLabel(twin)).not.toBe(runLabel(best));
});

test('runLabel shows the ablation only when it is not none', () => {
  expect(runLabel(best)).not.toContain('none');
  expect(runLabel({ ...best, role: null, ablationLabel: 'capanchor' })).toContain('capanchor');
});

test('runLabel keeps a plain oak run to its date and SHA-7', () => {
  expect(runLabel({ ...best, role: null })).toBe(
    `${runDateLabel(best.startedAt ?? null)} · ${best.id.slice(0, 7)}`,
  );
});

test('runLabel names the policy when it is not oak', () => {
  expect(runLabel({ ...best, role: null, policy: 'ppo' })).toContain('ppo');
});

test('runShortId takes the SHA-7 of a commit id and the fingerprint of a baseline id', () => {
  expect(runShortId(best.id)).toBe('c27eb70');
  expect(runShortId('baseline-ppo-c1a93e857229')).toBe('c1a93e8');
});

test('runDateLabel writes month day, adding the year only when it differs', () => {
  const now = new Date('2026-12-01T00:00:00Z');
  expect(runDateLabel('2026-08-17T12:00:00Z', now)).toBe('Aug 17');
  expect(runDateLabel('2025-08-17T12:00:00Z', now)).toBe('Aug 17 2025');
  expect(runDateLabel(null)).toBe('undated');
  expect(runDateLabel('not a date')).toBe('undated');
});
