import { expect, test } from 'bun:test';
import { runDateLabel, runLabel, type RunLabelSource } from './run';

const keeper: RunLabelSource & { id: string } = {
  id: 'c27eb7010fa644c770fdebf22f1527c82cfbce68',
  startedAt: '2026-08-06T18:30:00Z',
  role: 'keeper',
  ablationLabel: 'none',
  policy: 'oak',
};

test('runLabel names the keeper with a date and no SHA', () => {
  const label = runLabel(keeper);
  expect(label).toContain('Aug');
  expect(label).toContain('keeper');
  expect(label).not.toContain(keeper.id);
  expect(label).not.toMatch(/[0-9a-f]{7,}/);
});

test('runLabel shows the ablation only when it is not none', () => {
  expect(runLabel(keeper)).not.toContain('none');
  expect(runLabel({ ...keeper, role: null, ablationLabel: 'capanchor' })).toContain('capanchor');
});

test('runLabel keeps a plain oak run to its date', () => {
  expect(runLabel({ ...keeper, role: null })).toBe(runDateLabel(keeper.startedAt ?? null));
});

test('runLabel names the policy when it is not oak', () => {
  expect(runLabel({ ...keeper, role: null, policy: 'ppo' })).toContain('ppo');
});

test('runDateLabel writes month day, adding the year only when it differs', () => {
  const now = new Date('2026-12-01T00:00:00Z');
  expect(runDateLabel('2026-08-17T12:00:00Z', now)).toBe('Aug 17');
  expect(runDateLabel('2025-08-17T12:00:00Z', now)).toBe('Aug 17 2025');
  expect(runDateLabel(null)).toBe('undated');
  expect(runDateLabel('not a date')).toBe('undated');
});
