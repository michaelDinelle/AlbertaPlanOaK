// Server configuration: paths, ports, and the tuning constants the readers share.
// Everything is derived from the module location so the server works regardless of cwd.

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url)); // <repo>/frontend/server

/** <repo>/frontend */
export const FRONTEND_DIR = resolve(HERE, '..');
/** <repo> */
export const REPO_ROOT = resolve(FRONTEND_DIR, '..');
/** built client output served in production */
export const DIST_DIR = resolve(FRONTEND_DIR, 'dist');
/** derived-artifact cache (never checked in; a `.gitignore` with `*` is written on create) */
export const CACHE_DIR = resolve(FRONTEND_DIR, '.cache');

/** default fleet-run store; override with OAK_BENCH_DIR */
export const DEFAULT_BENCH_ROOT = resolve(REPO_ROOT, '.oak-benchmarks');

export function benchRootFromEnv(env: Record<string, string | undefined> = process.env): string {
  const raw = env['OAK_BENCH_DIR'];
  return raw && raw.length > 0 ? resolve(raw) : DEFAULT_BENCH_ROOT;
}

/** FNV-1a/32, base36. Cache-key material only — never a security boundary. */
export function hash32(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(36);
}

/**
 * Cache identity of a bench root. Derived artifacts are keyed by it so that pointing
 * OAK_BENCH_DIR at a second store cannot read (or clobber) the first store's cache.
 */
export function benchKey(benchRoot: string): string {
  return hash32(resolve(benchRoot));
}

/** per-bench-root run-index cache file */
export function runIndexCacheFile(benchRoot: string): string {
  return resolve(CACHE_DIR, `run-index-${benchKey(benchRoot)}.json`);
}

/** per-bench-root fleet columnar cache directory */
export function fleetCacheDir(benchRoot: string): string {
  return resolve(CACHE_DIR, 'fleet', benchKey(benchRoot));
}

export function portFromEnv(env: Record<string, string | undefined> = process.env): number {
  const raw = env['OAK_FRONTEND_PORT'];
  const n = raw === undefined ? NaN : Number(raw);
  return Number.isFinite(n) ? n : 5174;
}

export const HOSTNAME = '127.0.0.1';

/** completeness probe window: fleet.jsonl rows are step-ordered, finals cluster at the tail */
export const FLEET_TAIL_BYTES = 256 * 1024;
/** a run whose fleet.jsonl was touched more recently than this is still being written */
export const IN_FLIGHT_MS = 120_000;
/** run-index rescan throttle (a rescan is ~1200 stats; cheap but not free) */
export const INDEX_TTL_MS = 1_500;
/** SSE aggregate.jsonl poll period */
export const LIVE_POLL_MS = 2_000;
/** SSE keepalive comment cadence, in polls */
export const LIVE_KEEPALIVE_POLLS = 8;
/** in-memory columnar cache depth */
export const FLEET_LRU_MAX = 4;
/** simultaneous full fleet.jsonl parses (each one streams up to 175 MB) */
export const FLEET_COMPUTE_CONCURRENCY = 2;
/** fleet disk-cache ceiling: oldest-by-mtime entries are evicted on write */
export const FLEET_CACHE_MAX_FILES = 64;
export const FLEET_CACHE_MAX_BYTES = 128 * 1024 * 1024;
/** record v3: 16 float32 per step — ctrl[6], done, pose[9] */
export const RECORD_V3_BYTES_PER_STEP = 64;
/**
 * Record versions the API will serve. v3 is the flat float layout; v4 is the
 * chunked codec in `src/sim/RECORD_V4_SPEC.md`.
 */
export const RECORD_VERSION_SERVABLE: ReadonlySet<number> = new Set([3, 4]);

// record v4 file framing. DUPLICATED from `src/sim/record4.ts` on purpose: the
// server tsconfig includes only `server/` + `shared/`, so it cannot import from
// `src/` — the same reason RECORD_V3_BYTES_PER_STEP above restates `RECORD_BYTES`.
// The spec (`src/sim/RECORD_V4_SPEC.md`) is the shared source of truth for both.

/** u32 magic + formatVersion + envKind + codecVersion */
export const RECORD_V4_FILE_HEADER_BYTES = 16;
/** magic + stepStart + stepCount + rawBytes + compBytes + crc32 + doneCount + rewardDelta */
export const RECORD_V4_CHUNK_HEADER_BYTES = 32;
/** "OAK4" (0x344B414F LE = O,A,K,4) */
export const RECORD_V4_FILE_MAGIC = 0x344b414f;
/** "CHK4" (0x344B4843 LE = C,H,K,4) */
export const RECORD_V4_CHUNK_MAGIC = 0x344b4843;
/** belt-and-braces bound on a header walk of a hostile/garbage actions.bin */
export const RECORD_V4_MAX_CHUNKS = 1_000_000;
/** hard ceiling on requested fleet fields (DoS guard on a local server) */
export const MAX_FLEET_FIELDS = 24;
