
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

export const FRONTEND_DIR = resolve(HERE, '..');
export const REPO_ROOT = resolve(FRONTEND_DIR, '..');
export const DIST_DIR = resolve(FRONTEND_DIR, 'dist');
export const CACHE_DIR = resolve(FRONTEND_DIR, '.cache');

export const DEFAULT_BENCH_ROOT = resolve(REPO_ROOT, '.oak-benchmarks');

export function benchRootFromEnv(env: Record<string, string | undefined> = process.env): string {
  const raw = env['OAK_BENCH_DIR'];
  return raw && raw.length > 0 ? resolve(raw) : DEFAULT_BENCH_ROOT;
}

export function hash32(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(36);
}

export function benchKey(benchRoot: string): string {
  return hash32(resolve(benchRoot));
}

export function runIndexCacheFile(benchRoot: string): string {
  return resolve(CACHE_DIR, `run-index-${benchKey(benchRoot)}.json`);
}

export function fleetCacheDir(benchRoot: string): string {
  return resolve(CACHE_DIR, 'fleet', benchKey(benchRoot));
}

export function portFromEnv(env: Record<string, string | undefined> = process.env): number {
  const raw = env['OAK_FRONTEND_PORT'];
  const n = raw === undefined ? NaN : Number(raw);
  return Number.isFinite(n) ? n : 5174;
}

export const HOSTNAME = '127.0.0.1';

export const FLEET_TAIL_BYTES = 256 * 1024;
export const IN_FLIGHT_MS = 120_000;
export const INDEX_TTL_MS = 1_500;
export const LIVE_POLL_MS = 2_000;
export const LIVE_KEEPALIVE_POLLS = 8;
export const FLEET_LRU_MAX = 4;
export const FLEET_COMPUTE_CONCURRENCY = 2;
export const FLEET_CACHE_MAX_FILES = 64;
export const FLEET_CACHE_MAX_BYTES = 128 * 1024 * 1024;
export const RECORD_V3_BYTES_PER_STEP = 64;
export const RECORD_VERSION_SERVABLE: ReadonlySet<number> = new Set([3, 4]);


export const RECORD_V4_FILE_HEADER_BYTES = 16;
export const RECORD_V4_CHUNK_HEADER_BYTES = 32;
export const RECORD_V4_FILE_MAGIC = 0x344b414f;
export const RECORD_V4_CHUNK_MAGIC = 0x344b4843;
export const RECORD_V4_MAX_CHUNKS = 1_000_000;
export const MAX_FLEET_FIELDS = 24;
