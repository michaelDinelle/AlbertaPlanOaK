import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

export const FRONTEND_DIR = resolve(HERE, '..');
export const REPO_ROOT = resolve(FRONTEND_DIR, '..');
export const DIST_DIR = resolve(FRONTEND_DIR, 'dist');

export const DEFAULT_BENCH_ROOT = resolve(REPO_ROOT, '.oak-benchmarks');
export const DEFAULT_BASELINES_ROOT = resolve(REPO_ROOT, '.oak-baselines');

export function benchRootFromEnv(env: Record<string, string | undefined> = process.env): string {
  const raw = env['OAK_BENCH_DIR'];
  return raw && raw.length > 0 ? resolve(raw) : DEFAULT_BENCH_ROOT;
}

export function baselinesRootFromEnv(env: Record<string, string | undefined> = process.env): string {
  const raw = env['OAK_BASELINES_DIR'];
  return raw && raw.length > 0 ? resolve(raw) : DEFAULT_BASELINES_ROOT;
}

export function portFromEnv(env: Record<string, string | undefined> = process.env): number {
  const raw = env['OAK_FRONTEND_PORT'];
  const n = raw === undefined ? NaN : Number(raw);
  return Number.isFinite(n) ? n : 5174;
}

export const HOSTNAME = '127.0.0.1';
