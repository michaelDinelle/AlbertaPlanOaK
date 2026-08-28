import { isEnvId, type EnvId } from '@shared/envs';
import type { Policy } from '@shared/types';
import { isSortMode, type SortMode } from '../lib/selection';

export const SPEEDS = [1, 4, 16, 64, 256, 1024] as const;

export const DEFAULT_SPEED = 1;

export interface UrlState {
  run: string | null;
  env: EnvId | null;
  seed: number | null;
  at: number;
  speed: number;
  sort: SortMode;
  policy: Policy;
}

export const DEFAULT_URL_STATE: UrlState = {
  run: null,
  env: null,
  seed: null,
  at: 0,
  speed: DEFAULT_SPEED,
  sort: 'rate',
  policy: 'oak',
};

export function parseUrlState(search: string): UrlState {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const state: UrlState = { ...DEFAULT_URL_STATE };

  const run = params.get('run');
  if (run !== null && run.trim().length > 0) state.run = run.trim();

  const env = params.get('env');
  if (env !== null && isEnvId(env.trim())) state.env = env.trim() as EnvId;

  const seed = params.get('seed');
  if (seed !== null && /^\d+$/.test(seed.trim())) {
    const value = Number(seed.trim());
    if (Number.isSafeInteger(value) && value >= 0) state.seed = value;
  }

  const at = params.get('at');
  if (at !== null) {
    const value = Number(at);
    if (Number.isFinite(value) && value >= 0) state.at = Math.floor(value);
  }

  const speed = params.get('speed');
  if (speed !== null) {
    const value = Number(speed);
    if ((SPEEDS as readonly number[]).includes(value)) state.speed = value;
  }

  const sort = params.get('sort');
  if (sort !== null && isSortMode(sort.trim())) state.sort = sort.trim() as SortMode;

  const policy = params.get('policy');
  if (policy !== null && (policy.trim() === 'ppo' || policy.trim() === 'random')) {
    state.policy = policy.trim() as Policy;
  }

  return state;
}

export function serializeUrlState(state: UrlState): string {
  const parts: string[] = [];
  const push = (key: string, value: string): void => {
    parts.push(`${key}=${encodeURIComponent(value)}`);
  };
  if (state.run !== null && state.run !== '') push('run', state.run);
  if (state.env !== null) push('env', state.env);
  if (state.seed !== null && Number.isFinite(state.seed)) push('seed', String(Math.floor(state.seed)));
  if (Number.isFinite(state.at) && state.at > 0) push('at', String(Math.floor(state.at)));
  if (state.speed !== DEFAULT_SPEED) push('speed', String(state.speed));
  if (state.sort !== 'rate') push('sort', state.sort);
  if (state.policy !== 'oak') push('policy', state.policy);
  return parts.length > 0 ? `?${parts.join('&')}` : '';
}

export function mergeUrlState(state: UrlState, patch: Partial<UrlState>): UrlState {
  const next: UrlState = { ...state, ...patch };
  if (patch.at !== undefined) next.at = Number.isFinite(patch.at) && patch.at > 0 ? Math.floor(patch.at) : 0;
  return next;
}
