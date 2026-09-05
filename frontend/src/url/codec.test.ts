import { expect, test } from 'bun:test';
import {
  DEFAULT_URL_STATE,
  mergeUrlState,
  parseUrlState,
  serializeUrlState,
  type UrlState,
} from './codec';

const states: UrlState[] = [
  DEFAULT_URL_STATE,
  { run: 'abc', env: 'Pendulum-cont-v0', seed: 12, at: 120000, speed: 64, sort: 'retention', policy: 'ppo' },
  { ...DEFAULT_URL_STATE, run: 'r1' },
  { ...DEFAULT_URL_STATE, env: 'HalfCheetah-v4', at: 5 },
  { ...DEFAULT_URL_STATE, policy: 'random', sort: 'seed', speed: 1024, seed: 0 },
];

test('parse(serialize(state)) round-trips', () => {
  for (const state of states) {
    expect(parseUrlState(serializeUrlState(state))).toEqual(state);
  }
});

test('defaults serialize to the empty string', () => {
  expect(serializeUrlState(DEFAULT_URL_STATE)).toBe('');
});

test('parse ignores junk values', () => {
  const state = parseUrlState('?env=NotAnEnv&seed=-3&at=nope&speed=7&sort=upward&policy=oak&run=');
  expect(state).toEqual(DEFAULT_URL_STATE);
});

const patches: Partial<UrlState>[] = [
  { policy: 'ppo' },
  { run: 'other' },
  { env: 'Pendulum-cont-v0' },
  { seed: 3, at: 42 },
  { speed: 256 },
  { sort: 'seed' },
  { seed: null },
  { at: 0 },
];

test('merge patches every named field and preserves the rest', () => {
  for (const state of states) {
    for (const patch of patches) {
      const merged = mergeUrlState(state, patch);
      const reparsed = parseUrlState(serializeUrlState(merged));
      for (const key of Object.keys(DEFAULT_URL_STATE) as (keyof UrlState)[]) {
        if (key in patch) {
          const wanted = key === 'at' ? Math.max(0, Math.floor(patch.at ?? 0)) : patch[key];
          expect(reparsed[key]).toEqual(wanted as UrlState[typeof key]);
        } else {
          expect(reparsed[key]).toEqual(state[key]);
        }
      }
    }
  }
});

test('a policy flip never loses env, seed, or step', () => {
  const deepLink = parseUrlState('?run=abc&env=Pendulum-cont-v0&seed=12&at=120000');
  const flipped = parseUrlState(serializeUrlState(mergeUrlState(deepLink, { policy: 'ppo' })));
  expect(flipped.env).toBe('Pendulum-cont-v0');
  expect(flipped.seed).toBe(12);
  expect(flipped.at).toBe(120000);
  expect(flipped.policy).toBe('ppo');
});
