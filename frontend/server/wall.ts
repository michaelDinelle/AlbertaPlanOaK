/// <reference lib="dom" />
//
// ^ WHY: this is the one server module that reaches into `src/sim` — the FROZEN record
// decoders, which are shared verbatim with the client and name `BufferSource` (a DOM
// global). The server tsconfig's lib set is ES2022 + bun types, where `BufferSource`
// lives under the `Bun` namespace and not in global scope, so pulling `@sim/record4`
// into this program without the directive fails to resolve a name inside a file neither
// this module nor this project owns. The alternative — a second, server-local
// implementation of a codec whose whole point is that exactly one implementation exists
// per language — is the thing RECORD_V4_SPEC.md exists to prevent. Bun implements every
// DOM name this pulls in (streams, Response, DecompressionStream), so the widened lib
// describes the runtime accurately rather than papering over it.
//
// The seed wall: every recorded seed of ONE env, in one compact response.
//
// The theater plays one lane at a time. The wall plays all of them at once, so its
// unit of work is the whole env — 64, 128, 1024 lanes — and the only way that is
// affordable is to (a) never send a full recording, (b) never decode a row nobody
// asked for, (c) never decode the same wall twice.
//
//   (a) PROJECTION. A record row is 16 floats; the wall draws 9 of them plus a done
//       bit, at one row in `stride`. That is 37 bytes per sample instead of 64 per
//       step — at the default stride 64, ~1/110th of the lane. The projection happens
//       here, server-side, because the alternative is shipping 64 lanes × the whole
//       actions.bin to a browser and projecting there.
//   (b) SELECTIVE DECODE. The projection is strided, so the decode is too. A v4 chunk
//       is independently decodable (its own DEFLATE stream, its own crc, its own dx32
//       base), so a chunk holding no sampled row is never crc'd, never inflated and
//       never decoded — which is what makes a coarse stride actually CHEAPER rather
//       than merely smaller on the wire. The cost is dominated by inflate (~66 ns/row)
//       and crc32 (~35 ns/row in JS, ~0.4 native), never by the row decode (~13), so
//       both are now spent only where a sample lands. Measured on v4ab-tierS-v4
//       HalfCheetah (64 seeds × 10k in 1000-step chunks): stride 64 121 ms → 63 ms
//       (native crc + a native inflate), stride 4096 114 ms → 27 ms (six chunks in ten
//       never touched). Before this, every stride paid the same ~114 ms floor.
//   (c) CACHING. Identical to server/fleetDerive.ts, and for the same three reasons:
//       SOUND (the disk file name is a hash — a hint; the record inside carries the
//       full identity AND a checksum of the body, so neither a collision nor a flipped
//       bit can be served), SINGLE-FLIGHT (a wall opening 16 env panels decodes each
//       env once, not once per panel), BOUNDED (LRU by entry count AND by bytes in
//       memory, oldest-by-mtime eviction on disk). In-flight runs skip the disk cache
//       entirely: their lane files are still growing.
//
// BUDGETS. Both are real, not advisory. The projection is streamed straight into one
// deflate stream, so the server never holds more than ONE lane's samples plus the
// compressed body — and `WALL_MAX_RAW_BYTES` / `WALL_MAX_DECODE_ROWS` first COARSEN the
// stride (the request is a floor) and then, if a lane turns out longer than run
// discovery said, cut that lane and say so (`truncated`). A wall can be big; it can
// never be unbounded, and it yields to the event loop while it works.
//
// Both record versions are walls-able. v4 lanes are read chunk by chunk through the
// FROZEN primitives of `@sim/record4` — `scanChunksV4` for the framing, the chunk's own
// crc32 for integrity, `decodeChunkPayload` for the rows (there is exactly one decoder,
// which is the point of RECORD_V4_SPEC.md; only the chunk SELECTION lives here). v3
// lanes are a flat Float32Array already, so they are strided in place.
//
// The framing this module emits is specified in `shared/types.ts` (`WallResponse`) and
// decoded by `src/api.ts`'s `fetchWall`. Those three places are the contract.

import { readdir, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createDeflateRaw, inflateRawSync } from 'node:zlib';
import { isEnvId } from '@shared/envs';
import type {
  RecordingMeta,
  WallChunkEvents,
  WallLaneHeader,
  WallResponse,
} from '@shared/types';
import {
  CACHE_DIR,
  FLEET_CACHE_MAX_BYTES,
  FLEET_CACHE_MAX_FILES,
  FLEET_COMPUTE_CONCURRENCY,
  benchKey,
  hash32,
} from './config';
import { statOrNull } from './jsonl';
import type { RecordingEntry } from './recordings';
import { ensureCacheDir } from './runIndex';
import {
  OFF_DONE,
  OFF_POSE,
  POSE_LEN,
  RECORD_BYTES,
  RECORD_FLOATS,
  parseActionsBin,
} from '@sim/record';
import {
  CODEC_VERSION as RECORD4_CODEC_VERSION,
  RECORD_V4_ENV_KINDS,
  RECORD_V4_VERSION,
  Record4ParseError,
  codecForEnvKind,
  codecMatches,
  crc32 as crc32Reference,
  decodeChunkPayload,
  scanChunksV4,
  type ChunkRecordV4,
  type CodecDescriptor,
} from '@sim/record4';

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

/** default rows-per-sample; ~110× smaller than the lane it summarizes */
export const WALL_DEFAULT_STRIDE = 64;
export const WALL_MIN_STRIDE = 1;
/**
 * A stride past this stops buying anything: at 4096 even a 10M-step lane is under
 * 2500 samples, and the clamp is what keeps a hostile `?stride=1e300` from becoming
 * a divide-by-Infinity.
 */
export const WALL_MAX_STRIDE = 4096;

/** pose[9] f32 + done u8, tightly packed */
export const WALL_SAMPLE_BYTES = POSE_LEN * 4 + 1;

/**
 * Ceiling on the UNCOMPRESSED projection of one wall — what the client inflates and
 * holds, and what the wire carries once deflate has had its way with it.
 *
 * The stride clamp alone does not bound this: a fully-recorded Tier-L is 1024 lanes ×
 * 1M steps, which at the DEFAULT stride 64 is 16M samples — 592 MB, and 2.4 GB at
 * stride 1. So the requested stride is a floor, not a promise: when it would exceed this
 * budget the wall is served at the finest stride that fits, and the header's `stride`
 * reports what was actually used (which is the field `fetchWall` maps samples back to
 * record rows with — a coarsened wall is correct, just less dense).
 *
 * ENFORCED, not predicted. The stride is chosen from discovery's lane lengths, so it is
 * only as good as they are; on top of it every lane carries a hard runtime cap at
 * `WALL_LANE_OVERSHOOT` × its own planned size (`laneCaps`), which is what actually stops
 * a lane that grew behind a stale index. The exposed total is therefore this ceiling in
 * the normal case and at most `WALL_LANE_OVERSHOOT` × it in the pathological one where
 * discovery was wrong about EVERY lane — never unbounded, and never held in one piece
 * (the projection is streamed into deflate a lane at a time).
 */
export const WALL_MAX_RAW_BYTES = 128 * 1024 * 1024;

/**
 * Ceiling on the record rows ONE derive may decode.
 *
 * The projection is cheap; the decode is what costs. Measured after selective
 * decoding: ~90 ns per decoded row (inflate ~66, decode ~13, non-finite scan ~7,
 * projection ~2 — crc32 is native and free at this scale). A fully-recorded Tier-L
 * (1024 seeds × 1M steps, 1000-step chunks) decodes ~2.5e8 rows at the stride ceiling,
 * so this budget clears the documented target with headroom and cuts anything past it.
 * Like the byte budget it COARSENS the stride first (see `effectiveStride`); truncating
 * a lane is the last resort, and it is reported (`WallLaneHeader.truncated`).
 */
export const WALL_MAX_DECODE_ROWS = 320_000_000;

/**
 * How far past its discovery-derived plan one lane may go before the wall cuts it.
 *
 * Both budgets are planned from `RecordingEntry.steps`, which is discovery's count — a
 * v4 lane's finalize-time hint or the header walk that built the run index. For an
 * in-flight run that count can be STALE and the file longer, which is exactly the case
 * a ceiling computed from it cannot catch. So each lane also carries a hard runtime cap
 * at this multiple of its plan: a lane that matches discovery sits at 1/2 of its cap and
 * is never touched; a lane that has doubled behind a stale index is cut and says so.
 */
export const WALL_LANE_OVERSHOOT = 2;

/**
 * Rows between event-loop yields. One derive can be seconds of CPU, and it holds one of
 * `FLEET_COMPUTE_CONCURRENCY` slots while it runs — so it breathes roughly every 6 ms
 * (65536 rows × ~90 ns) instead of freezing every other request behind it.
 */
const WALL_YIELD_ROWS = 1 << 16;

/** byte offset of pose[0] inside a record row */
const POSE_BYTE_OFF = OFF_POSE * 4;
/** byte length of pose[9] */
const POSE_BYTES = POSE_LEN * 4;

/**
 * Cache identity of the PROJECTION (not the framing — that is `WallResponse.v`).
 * Any change to what a sample contains, or to which rows are sampled, bumps this and
 * strands every entry already on disk. 2: selective chunk decoding (`scannedRows`),
 * budget truncation (`truncated`), streamed deflate.
 */
export const WALL_CODEC_VERSION = 2;

/**
 * In-memory depth: at most this many entries AND at most `WALL_LRU_MAX_BYTES` of bodies.
 * A wall is big (tens of MB at 1M steps), so counting entries alone bounds nothing —
 * both caps hold, except that the newest entry is always kept (a single body larger than
 * the byte cap would otherwise make the cache a no-op and defeat single-flighting).
 */
const WALL_LRU_MAX = 4;
export const WALL_LRU_MAX_BYTES = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// crc32
// ---------------------------------------------------------------------------

/**
 * zlib/IEEE crc32 over `bytes[start, end)` — the runtime's native one when it is
 * PROVEN to agree with `@sim/record4`'s frozen implementation, otherwise that one.
 *
 * The wall crc's two things a wrong answer would corrupt: v4 chunk payloads (where the
 * value must match what the Swift encoder wrote) and its own cached bodies. So the
 * native path is not assumed, it is checked once at load against the frozen function on
 * an empty buffer, a full buffer and an OFFSET WINDOW — the three ways a crc can differ
 * (initial/final xor, polynomial, and reading the wrong window). A runtime that fails
 * any of them simply gets the JS loop: ~35 ns/row instead of ~0.4, never a wrong hash.
 */
const crc32: (bytes: Uint8Array, start?: number, end?: number) => number = (() => {
  const probe = new Uint8Array(257);
  for (let i = 0; i < probe.length; i++) probe[i] = (i * 37 + 11) & 0xff;
  const native = (globalThis as { Bun?: { hash?: { crc32?: (b: Uint8Array) => number } } }).Bun
    ?.hash?.crc32;
  const agrees =
    typeof native === 'function' &&
    (native(new Uint8Array(0)) >>> 0) === crc32Reference(new Uint8Array(0)) &&
    (native(probe) >>> 0) === crc32Reference(probe) &&
    (native(probe.subarray(3, 200)) >>> 0) === crc32Reference(probe, 3, 200);
  if (!agrees) return crc32Reference;
  const fast = native;
  return (bytes, start = 0, end = bytes.length) =>
    fast(start === 0 && end === bytes.length ? bytes : bytes.subarray(start, end)) >>> 0;
})();

// ---------------------------------------------------------------------------
// request parsing
// ---------------------------------------------------------------------------

export interface WallRequest {
  envId: string;
  stride: number;
}

export interface WallRequestError {
  error: string;
  status: number;
}

/**
 * Validate `:envId` + `?stride=`.
 *
 * The env id is checked against the 16 known envs and answers 404 — a path segment
 * that is not an env is not a resource, and it is also the shape a traversal attempt
 * arrives in. A malformed stride is a 400: the client asked a real question badly.
 * An out-of-range stride is CLAMPED, not rejected, because there is no wrong picture
 * to be had from a coarser or finer sample — only a bigger or smaller one.
 */
export function parseWallRequest(
  envId: string | undefined,
  url: URL,
): WallRequest | WallRequestError {
  if (envId === undefined || !isEnvId(envId)) return { error: 'unknown env', status: 404 };
  const raw = url.searchParams.get('stride');
  if (raw === null || raw.trim().length === 0) return { envId, stride: WALL_DEFAULT_STRIDE };
  const n = Number(raw);
  if (!Number.isFinite(n)) return { error: 'bad stride', status: 400 };
  const stride = Math.trunc(n);
  return {
    envId,
    stride: stride < WALL_MIN_STRIDE ? WALL_MIN_STRIDE : stride > WALL_MAX_STRIDE ? WALL_MAX_STRIDE : stride,
  };
}

// ---------------------------------------------------------------------------
// cache location
// ---------------------------------------------------------------------------

/** per-bench-root wall cache directory */
export function wallCacheDir(benchRoot: string): string {
  return resolve(CACHE_DIR, 'wall', benchKey(benchRoot));
}

/**
 * Where this app's walls persist, given the app's fleet-cache setting: `undefined`
 * (production) takes the real per-bench-root directory, `null` disables persistence,
 * and an explicit directory (tests) gets a `wall/` subdirectory of it — so a test that
 * points its derived caches at a temp dir never writes into the repo's frontend/.cache.
 */
export function wallCacheFor(benchRoot: string, fleetCache: string | null | undefined): string | null {
  if (fleetCache === undefined) return wallCacheDir(benchRoot);
  if (fleetCache === null) return null;
  return join(fleetCache, 'wall');
}

// ---------------------------------------------------------------------------
// LRU
// ---------------------------------------------------------------------------

export interface WallPayload {
  /**
   * The exact response body: header JSON, 0x0A, deflate-raw binary. Pinned to an
   * `ArrayBuffer` (not `ArrayBufferLike`) because that is what `BodyInit` accepts — a
   * SharedArrayBuffer-backed view cannot become a Response.
   */
  body: Uint8Array<ArrayBuffer>;
  /** the header that was framed into `body` (already parsed, for callers and tests) */
  header: WallResponse;
}

interface LruEntry {
  key: string;
  value: WallPayload;
}

const lru: LruEntry[] = [];
/** running sum of `lru`'s body bytes — the cap the entry count cannot express */
let lruBytes = 0;

function lruGet(key: string): WallPayload | undefined {
  const i = lru.findIndex((e) => e.key === key);
  if (i < 0) return undefined;
  const [hit] = lru.splice(i, 1);
  if (hit === undefined) return undefined;
  lru.push(hit);
  return hit.value;
}

function lruDrop(i: number): void {
  const [gone] = lru.splice(i, 1);
  if (gone !== undefined) lruBytes -= gone.value.body.byteLength;
}

function lruPut(key: string, value: WallPayload): void {
  const i = lru.findIndex((e) => e.key === key);
  if (i >= 0) lruDrop(i);
  lru.push({ key, value });
  lruBytes += value.body.byteLength;
  while (lru.length > 1 && (lru.length > WALL_LRU_MAX || lruBytes > WALL_LRU_MAX_BYTES)) lruDrop(0);
}

/** live in-memory cache size, in body bytes (test hook / diagnostics) */
export function wallCacheBytes(): number {
  return lruBytes;
}

/** test hook */
export function clearWallCache(): void {
  lru.length = 0;
  lruBytes = 0;
  inFlight.clear();
}

// ---------------------------------------------------------------------------
// single-flight + compute semaphore
// ---------------------------------------------------------------------------

const inFlight = new Map<string, Promise<WallPayload>>();

let computeActive = 0;
const computeQueue: (() => void)[] = [];

async function withComputeSlot<T>(fn: () => Promise<T>): Promise<T> {
  // Queue behind existing waiters even when a slot looks free: a released slot is
  // claimed by the woken waiter one microtask later, so a "free" reading between the
  // release and that wake-up would let LIMIT+1 decodes run at once (and starve the
  // waiter). Same reasoning, same shape, as server/fleetDerive.ts.
  if (computeActive >= FLEET_COMPUTE_CONCURRENCY || computeQueue.length > 0) {
    await new Promise<void>((res) => computeQueue.push(res));
  }
  computeActive++;
  try {
    return await fn();
  } finally {
    computeActive--;
    const next = computeQueue.shift();
    if (next !== undefined) next();
  }
}

// ---------------------------------------------------------------------------
// disk cache
// ---------------------------------------------------------------------------

function safeName(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
}

/** identity record written ahead of the framed body, on its own line */
interface DiskCacheRecord {
  v: 1;
  kind: 'wall';
  /** `runId|envId|stride|codecVersion|recordCodecVersion|<lane mtime:size|metaMtime:metaSize>,…` */
  key: string;
  runId: string;
  envId: string;
  stride: number;
  codecVersion: number;
  /**
   * Body integrity, filled in at WRITE time (the reader cannot know them in advance —
   * unlike every other field here, which is identity). The header re-check in
   * `parseWallBody` catches truncation and a wrong split; these catch a flipped bit
   * INSIDE the deflate section, whose length a corruption does not change. Without them
   * a single bad byte is served forever, and the client's inflate — not the server's
   * cache — is what fails.
   */
  bodyBytes?: number;
  bodyCrc?: number;
}

function diskCachePath(cacheDir: string, id: string, envId: string, key: string): string {
  // The hash names the file; the record inside proves what it is.
  return join(cacheDir, `${safeName(id)}--${safeName(envId)}--${hash32(key)}.wall`);
}

function findNewline(bytes: Uint8Array): number {
  return bytes.indexOf(0x0a);
}

async function readDiskCache(path: string, rec: DiskCacheRecord): Promise<WallPayload | null> {
  try {
    const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
    const nl = findNewline(bytes);
    if (nl < 0) return null;
    const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes.subarray(0, nl)));
    if (parsed === null || typeof parsed !== 'object') return null;
    const f = parsed as Partial<DiskCacheRecord>;
    // A hash collision (or a hand-edited file) costs a re-derive, never a wrong wall.
    if (f.v !== 1 || f.kind !== 'wall' || f.key !== rec.key) return null;
    if (f.runId !== rec.runId || f.envId !== rec.envId) return null;
    if (f.stride !== rec.stride || f.codecVersion !== rec.codecVersion) return null;
    const body = bytes.slice(nl + 1);
    // Integrity before identity-of-content: an entry written by this module always
    // carries both, so a missing one is itself a mismatch (a pre-checksum file).
    if (typeof f.bodyBytes !== 'number' || f.bodyBytes !== body.byteLength) return null;
    if (typeof f.bodyCrc !== 'number' || f.bodyCrc !== crc32(body)) return null;
    const header = parseWallBody(body);
    if (header === null) return null;
    return { body, header };
  } catch {
    return null;
  }
}

/** Re-read a framed body's header. Used to validate a disk entry before serving it. */
function parseWallBody(body: Uint8Array): WallResponse | null {
  const nl = findNewline(body);
  if (nl < 0) return null;
  try {
    const parsed: unknown = JSON.parse(new TextDecoder().decode(body.subarray(0, nl)));
    if (parsed === null || typeof parsed !== 'object') return null;
    const h = parsed as Partial<WallResponse>;
    if (h.v !== 1 || !Array.isArray(h.lanes)) return null;
    if (h.headerBytes !== nl || h.binaryBytes !== body.byteLength - nl - 1) return null;
    return parsed as WallResponse;
  } catch {
    return null;
  }
}

async function writeDiskCache(path: string, rec: DiskCacheRecord, body: Uint8Array): Promise<void> {
  try {
    await ensureCacheDir(path);
    const stamped: DiskCacheRecord = { ...rec, bodyBytes: body.byteLength, bodyCrc: crc32(body) };
    const head = new TextEncoder().encode(`${JSON.stringify(stamped)}\n`);
    const out = new Uint8Array(head.byteLength + body.byteLength);
    out.set(head, 0);
    out.set(body, head.byteLength);
    await writeFile(path, out);
  } catch {
    /* cache is an optimization */
  }
  await evictDiskCache(path);
}

/**
 * Keep the cache directory under both ceilings by deleting oldest-by-mtime first. Only
 * files this module writes (`*.wall`) are candidates; the `.gitignore` marker and any
 * neighbouring cache are left be.
 */
async function evictDiskCache(writtenPath: string): Promise<void> {
  const dir = dirname(writtenPath);
  try {
    const names = (await readdir(dir)).filter((n) => n.endsWith('.wall'));
    const entries: { path: string; mtimeMs: number; size: number }[] = [];
    let total = 0;
    for (const name of names) {
      const p = join(dir, name);
      const st = await statOrNull(p);
      if (st === null) continue;
      entries.push({ path: p, mtimeMs: st.mtimeMs, size: st.size });
      total += st.size;
    }
    if (entries.length <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) return;
    entries.sort((a, b) => a.mtimeMs - b.mtimeMs); // oldest first
    let count = entries.length;
    for (const e of entries) {
      if (count <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) break;
      if (e.path === writtenPath) continue; // never evict what we just wrote
      await unlink(e.path).catch(() => undefined);
      count--;
      total -= e.size;
    }
  } catch {
    /* eviction is best-effort */
  }
}

// ---------------------------------------------------------------------------
// derivation
// ---------------------------------------------------------------------------

export interface WallOptions {
  id: string;
  /** run root; part of the in-memory key so two bench roots cannot share an entry */
  dir: string;
  envId: string;
  stride: number;
  /** the run's DISCOVERED recordings — the only source of lane paths */
  recordings: readonly RecordingEntry[];
  /** in-flight runs bypass the disk cache (lane files are still being appended to) */
  inFlight: boolean;
  /** disk-cache directory; null/absent disables persistence */
  cacheDir?: string | null;
}

/**
 * What a budget needs to know about one lane: how long it is, and — for a v4 lane whose
 * finalize-time chunk table survived — how long its chunks are, because a chunk is the
 * unit the decode is charged in.
 */
export interface WallLaneShape {
  steps: number;
  /** rows per v4 chunk; absent (or 0) means "assume every row must be decoded" */
  chunkSteps?: number;
}

/**
 * Rows a lane costs to DECODE at `stride`.
 *
 * A v4 chunk is decoded whole or not at all, so a lane costs one chunk per chunk that
 * holds a sampled row. Two regimes:
 *
 *   * `stride <= chunkSteps` — every chunk is at least a stride long, so every chunk
 *     holds a sample and the lane costs its whole length. (A final chunk shorter than
 *     the stride can be the one exception; it is charged anyway, since the sampling
 *     rule always takes the final row. So this is an upper bound by at most one short
 *     chunk, in the safe direction.)
 *   * `stride > chunkSteps` — consecutive samples land in DIFFERENT chunks, so the
 *     chunk count is the sample count, plus the chunk holding the final row when the
 *     stride missed it. Exact.
 *
 * Without a chunk table (v3, or a torn v4 lane whose table did not survive) nothing can
 * be skipped and the answer is `steps`.
 */
export function decodedRowsAt(lane: WallLaneShape, stride: number): number {
  const steps = lane.steps > 0 ? lane.steps : 0;
  const chunk = lane.chunkSteps !== undefined && lane.chunkSteps > 0 ? lane.chunkSteps : 0;
  if (steps === 0) return 0;
  if (chunk === 0 || stride <= chunk) return steps;
  const chunks = Math.ceil(steps / chunk);
  const lastChunkRows = steps - (chunks - 1) * chunk;
  const sampled = Math.floor((steps - 1) / stride) + 1;
  const lastSampleChunk = Math.floor((Math.floor((steps - 1) / stride) * stride) / chunk);
  // The final row's chunk is always decoded; it is only an EXTRA when the last strided
  // sample did not already land in it.
  const fullChunks = lastSampleChunk === chunks - 1 ? sampled - 1 : sampled;
  return Math.min(fullChunks, chunks - 1) * chunk + lastChunkRows;
}

/** Total decoded rows over every lane at `stride`. */
function totalDecodedRows(lanes: readonly WallLaneShape[], stride: number): number {
  let rows = 0;
  for (const lane of lanes) rows += decodedRowsAt(lane, stride);
  return rows;
}

/**
 * The stride the wall is actually built at: the requested one, unless it would exceed a
 * budget, in which case the finest stride that fits BOTH of them.
 *
 *   * PAYLOAD (`WALL_MAX_RAW_BYTES`) — what the response and the client's inflated copy
 *     cost. Solved directly: samples scale as `1/stride`.
 *   * DECODE (`WALL_MAX_DECODE_ROWS`) — what the server spends. This one is a step
 *     function of the chunk geometry (a stride finer than a chunk saves nothing), so it
 *     is solved by bisection on the monotone `totalDecodedRows`.
 *
 * `lane.steps` is discovery's count (a v4 lane's finalize-time hint, or the header walk
 * for a torn one), so it is an upper bound on what the decode will yield — exactly what
 * a budget wants. Every lane contributes one extra sample for its final row, so the
 * payload budget reserves one per lane before dividing. Coarsening is always the first
 * answer: it costs density, and density is what the request asked for. Truncating a lane
 * (the runtime cap, `WALL_LANE_OVERSHOOT`) costs SPAN, and only happens when discovery
 * turned out to be wrong.
 */
export function effectiveStride(lanes: readonly WallLaneShape[], requested: number): number {
  let totalSteps = 0;
  for (const lane of lanes) totalSteps += lane.steps > 0 ? lane.steps : 0;
  if (totalSteps <= 0) return requested;

  const budget = Math.floor(WALL_MAX_RAW_BYTES / WALL_SAMPLE_BYTES) - lanes.length;
  if (budget <= 0) return WALL_MAX_STRIDE;
  const stride = Math.min(WALL_MAX_STRIDE, Math.max(requested, Math.ceil(totalSteps / budget)));

  if (totalDecodedRows(lanes, stride) <= WALL_MAX_DECODE_ROWS) return stride;
  // Monotone in stride, so bisect for the FINEST stride that fits rather than doubling
  // past it. The ceiling is the answer when even it is not enough — the runtime cap
  // takes over from there.
  let lo = stride;
  let hi = WALL_MAX_STRIDE;
  if (totalDecodedRows(lanes, hi) > WALL_MAX_DECODE_ROWS) return hi;
  while (lo < hi) {
    const mid = lo + Math.floor((hi - lo) / 2);
    if (totalDecodedRows(lanes, mid) <= WALL_MAX_DECODE_ROWS) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** The two runtime caps one lane is held to. Both are `WALL_LANE_OVERSHOOT` × its plan. */
interface LaneCaps {
  /** sample bytes this lane may project */
  bytes: number;
  /** record rows this lane may decode */
  rows: number;
}

/**
 * The runtime caps for one lane at the chosen stride.
 *
 * Planned from discovery and then multiplied, so a lane that matches discovery is at
 * half its cap and can never be truncated by it — the cap exists for the lane that grew
 * behind a stale index, and for the run so large that even the stride ceiling could not
 * bring the decode inside `WALL_MAX_DECODE_ROWS`. The floor keeps a very short lane
 * (or a lane discovery reported as empty) from being cut to nothing.
 */
function laneCaps(lane: WallLaneShape, stride: number, laneCount: number): LaneCaps {
  const steps = lane.steps > 0 ? lane.steps : 0;
  const planSamples = steps > 0 ? Math.ceil(steps / stride) + 1 : 2;
  const shareRows = Math.floor(WALL_MAX_DECODE_ROWS / Math.max(1, laneCount));
  return {
    bytes: Math.max(64, planSamples * WALL_LANE_OVERSHOOT) * WALL_SAMPLE_BYTES,
    rows: Math.max(
      lane.chunkSteps !== undefined && lane.chunkSteps > 0 ? lane.chunkSteps : 1,
      Math.min(decodedRowsAt(lane, stride) * WALL_LANE_OVERSHOOT, shareRows),
    ),
  };
}

/**
 * Discovery's shape for one recorded lane: its length and its chunk geometry.
 *
 * The geometry is read off the FIRST chunk of the finalize-time table, because the
 * recorder drains on a fixed interval and the plan only needs the scale. Nothing depends
 * on it being right: an uneven or absent table costs a coarser (or finer) stride, and the
 * runtime caps — which count real rows — are what actually hold.
 */
function laneShape(lane: RecordingEntry): WallLaneShape {
  const first = lane.chunks?.[0];
  const chunkSteps = first === undefined ? 0 : first[1];
  return chunkSteps > 0 ? { steps: lane.steps, chunkSteps } : { steps: lane.steps };
}

export async function deriveWall(opts: WallOptions): Promise<WallPayload> {
  const { id, dir, envId } = opts;
  const lanes = opts.recordings.filter((r) => r.envId === envId);
  // Resolved BEFORE the cache key, so two requests that coarsen onto the same stride
  // share one entry instead of deriving the identical wall twice.
  const shapes = lanes.map(laneShape);
  const stride = effectiveStride(shapes, opts.stride);

  // Every lane file's mtime+size, so an appended or rewritten lane invalidates the
  // wall the same way fleet.jsonl's mtime+size invalidates a columnar reduction.
  //
  // BOTH files, because both are decode inputs: actions.bin carries the rows, and the
  // lane's meta.json carries `recordVersion`, `actDim`, `codec` and `codecVersion` —
  // change any of those and the same bytes decode to something else. A key over the
  // payload alone would keep serving the pre-edit projection.
  const laneKeys: string[] = [];
  for (const lane of lanes) {
    const [data, meta] = await Promise.all([
      statOrNull(lane.actionsPath),
      statOrNull(lane.metaPath),
    ]);
    const stamp = (st: { mtimeMs: number; size: number } | null): string =>
      st === null ? 'x' : `${st.mtimeMs}:${st.size}`;
    laneKeys.push(`${lane.seed}:${stamp(data)}:${stamp(meta)}`);
  }
  // RECORD4_CODEC_VERSION is in the key for the same reason WALL_CODEC_VERSION is: a
  // revision of the row decoder changes what these bytes MEAN, and no lane file's mtime
  // or size moves when it does.
  const key =
    `${id}|${envId}|${stride}|${WALL_CODEC_VERSION}|${RECORD4_CODEC_VERSION}|${laneKeys.join(',')}`;
  const memKey = `${dir}|${key}`;

  const memHit = lruGet(memKey);
  if (memHit !== undefined) return memHit;

  // Single-flight: 16 env panels opening together must decode each env once.
  const pending = inFlight.get(memKey);
  if (pending !== undefined) return pending;

  const cacheDir = opts.cacheDir ?? null;
  const live = opts.inFlight;
  const record: DiskCacheRecord = {
    v: 1,
    kind: 'wall',
    key,
    runId: id,
    envId,
    stride,
    codecVersion: WALL_CODEC_VERSION,
  };

  const task = (async (): Promise<WallPayload> => {
    const diskPath = cacheDir === null ? null : diskCachePath(cacheDir, id, envId, key);
    if (!live && diskPath !== null) {
      const cached = await readDiskCache(diskPath, record);
      if (cached !== null) {
        lruPut(memKey, cached);
        return cached;
      }
    }

    const payload = await withComputeSlot(() => computeWall(id, envId, stride, lanes));
    lruPut(memKey, payload);
    if (!live && diskPath !== null) void writeDiskCache(diskPath, record, payload.body);
    return payload;
  })().finally(() => {
    inFlight.delete(memKey);
  });

  inFlight.set(memKey, task);
  return task;
}

// ---------------------------------------------------------------------------
// projection
// ---------------------------------------------------------------------------

/**
 * One lane's samples: the raw 37-byte records, in row order, plus what the header
 * says about the lane.
 */
interface LaneProjection {
  bytes: Uint8Array;
  head: WallLaneHeader;
  events: WallChunkEvents | null;
}

/**
 * Copy row `row` of a decoded block into `out` at `outOff` as one wall sample.
 *
 * pose is copied as BYTES, not as nine float reads: the record's f32s go out exactly
 * as they came in, so a NaN keeps its payload and a diverged lane looks on the wire
 * precisely as it looks on disk. (Reading an f32 into a JS number and writing it back
 * is lossless for the VALUE but canonicalizes NaN.) `src` is the caller's hoisted byte
 * view of `data` — one view per block, not one per row. `@sim/record` already
 * normalizes a big-endian read into a native Float32Array, so this is correct on
 * either byte order.
 */
function projectRow(
  data: Float32Array,
  src: Uint8Array,
  row: number,
  out: Uint8Array,
  outOff: number,
): void {
  const at = row * RECORD_BYTES;
  out.set(src.subarray(at + POSE_BYTE_OFF, at + POSE_BYTE_OFF + POSE_BYTES), outOff);
  const done = data[row * RECORD_FLOATS + OFF_DONE] ?? 0;
  out[outOff + POSE_BYTES] = Number.isFinite(done) && done !== 0 ? 1 : 0;
}

/** True when any projected field of row `row` is non-finite. */
function rowNonFinite(data: Float32Array, row: number): boolean {
  const base = row * RECORD_FLOATS;
  if (!Number.isFinite(data[base + OFF_DONE] ?? 0)) return true;
  for (let j = 0; j < POSE_LEN; j++) {
    if (!Number.isFinite(data[base + OFF_POSE + j] ?? 0)) return true;
  }
  return false;
}

/** The RecordingMeta the decoders need, rebuilt from what discovery already read. */
function laneMeta(lane: RecordingEntry): RecordingMeta {
  const meta: RecordingMeta = {
    envId: lane.envId,
    seed: lane.seed,
    actDim: lane.actDim,
    recordVersion: lane.recordVersion,
  };
  if (lane.codec !== undefined) meta.codec = lane.codec;
  if (lane.codecVersion !== undefined) meta.codecVersion = lane.codecVersion;
  return meta;
}

/**
 * Accumulates a lane's samples without knowing how many there will be.
 *
 * A torn v4 tail means the decoded step count is only known once the walk ends, and
 * the FINAL row has to be sampled whatever the stride landed on — so the sampler
 * keeps the last row it saw in a one-record scratch and appends it at the end iff the
 * stride missed it.
 *
 * Blocks may arrive with GAPS (a stride coarser than a chunk skips whole chunks), which
 * the stride arithmetic already handles: `nextRow` is the next sample's GLOBAL row, so a
 * block that starts past it is sampled from the right offset and a block that ends before
 * it contributes nothing. What must hold is that the LAST block fed is the last row the
 * wall covers — the tail comes from it.
 *
 * `maxBytes` is the lane's hard runtime ceiling (`laneCaps`). Reaching it stops the lane
 * at its last emitted sample and sets `truncated`, which is what makes the wall's byte
 * budget real rather than a prediction from a possibly stale lane length.
 */
class LaneSampler {
  private readonly blocks: Uint8Array[] = [];
  private readonly tail = new Uint8Array(WALL_SAMPLE_BYTES);
  private samples = 0;
  /** global row index of the next stride sample */
  private nextRow = 0;
  /** global row index of the last sample emitted, −1 before the first */
  private lastEmitted = -1;
  private haveTail = false;
  private tailRow = -1;
  private readonly maxSamples: number;
  steps = 0;
  /** rows fed in — what the non-finite scan actually saw */
  scannedRows = 0;
  firstNonFiniteRow: number | null = null;
  /** the byte ceiling cut this lane short */
  truncated = false;

  constructor(
    private readonly stride: number,
    maxBytes: number,
  ) {
    // Two at the very least: one sample and the tail slot reserved for it.
    this.maxSamples = Math.max(2, Math.floor(maxBytes / WALL_SAMPLE_BYTES));
  }

  /**
   * The death certificate, restricted to the span the lane actually covers. A NaN in a
   * row the wall then truncated away is not this wall's news to report — the header
   * describes `[0, steps)` and nothing else.
   */
  get deathRow(): number | null {
    const row = this.firstNonFiniteRow;
    return row !== null && row < this.steps ? row : null;
  }

  /** global row of the next stride sample — which chunks must be decoded */
  get nextSampleRow(): number {
    return this.nextRow;
  }

  /**
   * Feed one decoded block whose first row is global row `stepStart`.
   * Answers false when the lane has hit its ceiling and must stop.
   */
  push(data: Float32Array, stepStart: number, stepCount: number): boolean {
    if (stepCount <= 0) return !this.truncated;
    const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

    if (this.firstNonFiniteRow === null) {
      for (let i = 0; i < stepCount; i++) {
        if (rowNonFinite(data, i)) {
          this.firstNonFiniteRow = stepStart + i;
          break;
        }
      }
    }
    this.scannedRows += stepCount;

    const end = stepStart + stepCount;
    // Sample count is arithmetic, so the block is allocated once, exactly. The ceiling
    // reserves one sample for the tail, so a truncated lane still ends on a real row.
    const room = Math.max(0, this.maxSamples - 1 - this.samples);
    const want = this.nextRow < end ? Math.ceil((end - this.nextRow) / this.stride) : 0;
    const hits = Math.min(want, room);
    if (hits > 0) {
      const block = new Uint8Array(hits * WALL_SAMPLE_BYTES);
      let off = 0;
      let r = this.nextRow;
      for (let k = 0; k < hits; k++, r += this.stride) {
        projectRow(data, src, r - stepStart, block, off);
        off += WALL_SAMPLE_BYTES;
        this.lastEmitted = r;
      }
      this.blocks.push(block);
      this.samples += hits;
      this.nextRow += hits * this.stride;
    }

    if (hits < want) {
      // Out of room. The lane ends on the last sample it managed, and `steps` says so:
      // the client's row mapping (`fetchWall`) reads the last sample as row `steps − 1`,
      // so a truncated lane must report the span it actually covers, not the one the
      // file holds.
      this.truncated = true;
      this.steps = this.lastEmitted + 1;
      this.haveTail = false;
      // Rows past the cut were read but are not covered, so they are not "scanned" as
      // far as the header is concerned. Clamped into [0, steps], the only range the
      // count can honestly take once blocks may also have arrived with gaps.
      const dropped = end - this.steps;
      if (dropped > 0) {
        const kept = this.scannedRows - dropped;
        this.scannedRows = kept < 0 ? 0 : kept > this.steps ? this.steps : kept;
      }
      return false;
    }

    projectRow(data, src, stepCount - 1, this.tail, 0);
    this.tailRow = end - 1;
    this.haveTail = true;
    this.steps = end;
    return true;
  }

  /** The lane's finished byte block. Appends the final row when the stride missed it. */
  finish(): Uint8Array {
    if (this.haveTail && this.lastEmitted !== this.tailRow) {
      this.blocks.push(this.tail.slice());
      this.samples++;
      this.lastEmitted = this.tailRow;
    }
    let total = 0;
    for (const b of this.blocks) total += b.byteLength;
    const out = new Uint8Array(total);
    let off = 0;
    for (const b of this.blocks) {
      out.set(b, off);
      off += b.byteLength;
    }
    this.blocks.length = 0;
    return out;
  }

  get sampleCount(): number {
    return this.samples;
  }
}

/**
 * `@sim/record4`'s own lane checks, restated over its exported primitives.
 *
 * `prepareV4` is private to that module and reachable only through the readers that
 * decode EVERY chunk, which is exactly what a strided wall must not do. So the checks
 * are restated here — and only the checks: the framing is `scanChunksV4`'s, the
 * integrity is the chunk's own crc32, and the rows are `decodeChunkPayload`'s. There is
 * still exactly one implementation of the codec, which is what RECORD_V4_SPEC.md
 * requires; what lives here is which chunks to hand it.
 */
function prepareLaneV4(
  bytes: Uint8Array,
  meta: RecordingMeta,
): { desc: CodecDescriptor; chunks: readonly ChunkRecordV4[] } {
  if (meta.recordVersion !== RECORD_V4_VERSION) {
    throw new Record4ParseError(
      `unsupported recordVersion ${meta.recordVersion} (this parser reads ${RECORD_V4_VERSION})`,
    );
  }
  if (!isEnvId(meta.envId)) {
    throw new Record4ParseError(`unknown envId ${JSON.stringify(meta.envId)}`);
  }
  if (!Number.isInteger(meta.seed) || meta.seed < 0) {
    throw new Record4ParseError(`invalid seed ${String(meta.seed)}`);
  }
  if (meta.codecVersion !== undefined && meta.codecVersion !== RECORD4_CODEC_VERSION) {
    throw new Record4ParseError(
      `meta codecVersion ${meta.codecVersion} (this build implements ${RECORD4_CODEC_VERSION})`,
    );
  }
  // The scan's own crc pass is skipped: it would crc EVERY chunk, including the ones
  // this lane is about to skip. Each decoded chunk is crc'd below instead — verify what
  // you read. Everything else the scan checks (magic, sizes, step contiguity, the torn
  // tail) is structural and still runs.
  const scan = scanChunksV4(bytes, { verifyCrc: false });
  const headerEnv = RECORD_V4_ENV_KINDS[scan.header.envKind];
  if (headerEnv === undefined) {
    throw new Record4ParseError(`file header envKind ${scan.header.envKind} is not a known env`);
  }
  if (headerEnv !== meta.envId) {
    throw new Record4ParseError(
      `file header envKind ${scan.header.envKind} (${headerEnv}) contradicts meta envId ${meta.envId}`,
    );
  }
  const desc = codecForEnvKind(scan.header.envKind);
  if (meta.codec !== undefined && meta.codec !== null && !codecMatches(desc, meta.codec)) {
    throw new Record4ParseError(
      `meta codec descriptor for ${meta.envId} does not match this build's codec table`,
    );
  }
  return { desc, chunks: scan.chunks };
}

/** Let pending I/O run; a multi-second derive must not freeze every other request. */
function yieldToLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
}

/**
 * Project one v4 lane, decoding only the chunks a sample falls in.
 *
 * Three rules make the skip safe:
 *   * the LAST covered chunk is always decoded — the sampling rule ends on the final
 *     row, and that row's bytes only exist once its chunk is inflated;
 *   * chunk events come from the chunk HEADERS, so a skipped chunk still reports its
 *     span, its done count and its reward delta;
 *   * `scannedRows` reports what the non-finite scan actually saw, so the header never
 *     implies a certainty the skip gave up.
 */
async function projectLaneV4(
  lane: RecordingEntry,
  bytes: Uint8Array,
  meta: RecordingMeta,
  sampler: LaneSampler,
  caps: LaneCaps,
): Promise<WallChunkEvents> {
  const { desc, chunks } = prepareLaneV4(bytes, meta);
  const ev: WallChunkEvents = {
    seed: lane.seed,
    stepStart: [],
    stepCount: [],
    doneCount: [],
    rewardDelta: [],
  };

  let scratch = new Float32Array(0);
  let rowsLeft = caps.rows;
  let sinceYield = 0;
  /** chunks whose rows the wall covers: everything up to the last one DECODED */
  let covered = 0;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (chunk === undefined) break;
    const end = chunk.stepStart + chunk.stepCount;
    const holdsSample = sampler.nextSampleRow < end;
    const isLast = i === chunks.length - 1;
    // The final row is always sampled, so the chunk holding it is always decoded.
    const needed = holdsSample || isLast;

    ev.stepStart.push(chunk.stepStart);
    ev.stepCount.push(chunk.stepCount);
    ev.doneCount.push(chunk.doneCount);
    // NaN is the recorder's "the host had no stats snapshot for this drain",
    // and JSON has no NaN — null says the same thing without inventing a 0.
    ev.rewardDelta.push(Number.isFinite(chunk.rewardDelta) ? chunk.rewardDelta : null);
    if (!needed) continue;

    // The decode ceiling stops BEFORE a chunk it cannot afford, so the wall never ends
    // on a half-read chunk: the last decoded chunk stays the last covered one. The FIRST
    // chunk is always taken — a lane whose real chunks are longer than the table
    // discovery read must still show something, and one chunk is a bounded overrun.
    if (covered > 0 && chunk.stepCount > rowsLeft) {
      sampler.truncated = true;
      break;
    }

    if (crc32(bytes, chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes) !== chunk.crc32) {
      // The scan's torn-tail rule, applied where the bytes are actually read.
      break;
    }
    let payload: Uint8Array;
    try {
      payload = new Uint8Array(
        inflateRawSync(bytes.subarray(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes)),
      );
      if (payload.byteLength !== chunk.rawBytes) break;
    } catch {
      // The spec's last torn-tail signal: the payload will not inflate, or it inflates
      // to the wrong size. Drop this chunk and everything after it.
      break;
    }

    const floats = chunk.stepCount * RECORD_FLOATS;
    if (scratch.length < floats) scratch = new Float32Array(floats);
    // NOT caught: a crc-valid chunk that inflated to exactly `rawBytes` and is still
    // structurally wrong is a CODEC bug, not a torn tail (`parseActionsBinV4` draws the
    // same line). Truncating there would quietly play half a lane and call it the truth.
    decodeChunkPayload(payload, desc, chunk.stepCount, scratch, 0);
    rowsLeft -= chunk.stepCount;
    covered = i + 1;
    const more = sampler.push(scratch.subarray(0, floats), chunk.stepStart, chunk.stepCount);

    sinceYield += chunk.stepCount;
    if (sinceYield >= WALL_YIELD_ROWS) {
      sinceYield = 0;
      await yieldToLoop();
    }
    if (!more) break;
  }

  // Events describe the rows the wall covers, so they end where the decode did.
  ev.stepStart.length = covered;
  ev.stepCount.length = covered;
  ev.doneCount.length = covered;
  ev.rewardDelta.length = covered;
  return ev;
}

async function projectLane(
  lane: RecordingEntry,
  stride: number,
  caps: LaneCaps,
): Promise<LaneProjection> {
  const sampler = new LaneSampler(stride, caps.bytes);
  const meta = laneMeta(lane);
  let events: WallChunkEvents | null = null;

  try {
    const bytes = new Uint8Array(await Bun.file(lane.actionsPath).arrayBuffer());
    if (lane.recordVersion === 4) {
      events = await projectLaneV4(lane, bytes, meta, sampler, caps);
    } else {
      // A v3 file cut mid-row (a killed run) is the same torn tail a v4 walk drops:
      // keep the whole rows and forget the fragment.
      const whole = bytes.byteLength - (bytes.byteLength % RECORD_BYTES);
      const raw = parseActionsBin(bytes.subarray(0, whole), meta);
      // v3 has no chunks to skip — the parse is a zero-copy view, so the ceiling is the
      // only thing that can stop it short of the whole lane.
      const rows = Math.min(raw.steps, caps.rows);
      if (rows > 0) {
        if (rows < raw.steps) sampler.truncated = true;
        sampler.push(raw.data, 0, rows);
      }
    }
  } catch (err) {
    // One unreadable lane must not take down an env's wall — the same error-isolation
    // rule server/recordings.ts already follows. The lane ships with zero samples and
    // says why.
    const head: WallLaneHeader = {
      seed: lane.seed,
      steps: 0,
      samples: 0,
      firstNonFiniteRow: null,
      error: err instanceof Error ? err.message : String(err),
    };
    return { bytes: new Uint8Array(0), head, events: null };
  }

  const bytes = sampler.finish();
  const head: WallLaneHeader = {
    seed: lane.seed,
    steps: sampler.steps,
    samples: sampler.sampleCount,
    firstNonFiniteRow: sampler.deathRow,
  };
  // Both are ABSENT unless they are true, so a wall of fully decoded lanes is framed
  // exactly as it always was.
  if (sampler.scannedRows < sampler.steps) head.scannedRows = sampler.scannedRows;
  if (sampler.truncated) head.truncated = true;
  return { bytes, head, events };
}

// ---------------------------------------------------------------------------
// framing
// ---------------------------------------------------------------------------

/**
 * Serialize the header with a truthful self-referential `headerBytes`.
 *
 * The field's own digits change the length it reports, so the value is found by
 * iteration: each pass re-serializes with the previous pass's length. The sequence is
 * monotone (one more digit adds one byte) and converges in at most a handful of steps
 * — the loop bound is a guard against a future field making it pathological, and the
 * fallback pads with spaces AFTER the JSON, which stays valid JSON of the stated size.
 */
function frameHeader(base: Omit<WallResponse, 'headerBytes'>): { text: string; bytes: number } {
  const encoder = new TextEncoder();
  let guess = 0;
  for (let i = 0; i < 8; i++) {
    const text = JSON.stringify({ ...base, headerBytes: guess });
    const len = encoder.encode(text).byteLength;
    if (len === guess) return { text, bytes: len };
    guess = len;
  }
  const text = JSON.stringify({ ...base, headerBytes: guess });
  const len = encoder.encode(text).byteLength;
  return { text: len < guess ? text + ' '.repeat(guess - len) : text, bytes: Math.max(len, guess) };
}

function frame(
  base: Omit<WallResponse, 'headerBytes'>,
  binary: readonly Uint8Array[],
  binaryBytes: number,
): WallPayload {
  const { text, bytes } = frameHeader(base);
  const head = new TextEncoder().encode(text);
  const body = new Uint8Array(head.byteLength + 1 + binaryBytes);
  body.set(head, 0);
  body[head.byteLength] = 0x0a;
  let off = head.byteLength + 1;
  for (const part of binary) {
    body.set(part, off);
    off += part.byteLength;
  }
  return { body, header: { ...base, headerBytes: bytes } };
}

/**
 * The deflate-raw section, compressed as the lanes arrive.
 *
 * The alternative — collect every lane's samples, concatenate into one flat buffer,
 * deflate that — holds the whole projection TWICE at its peak (measured: 29.6 MB of
 * samples cost 68 MB of RSS), which is 2× a ceiling that describes itself as 1×. Feeding
 * one deflate stream instead means the uncompressed projection never exists as a whole:
 * the peak is one lane's samples plus the compressed output.
 *
 * The stream is opened LAZILY, so an env whose every lane is empty still emits no
 * section at all (`binaryBytes: 0`) rather than deflate's two-byte empty block.
 */
class DeflateSink {
  private stream: ReturnType<typeof createDeflateRaw> | null = null;
  private readonly parts: Uint8Array[] = [];
  private bytes = 0;
  private ended: Promise<void> | null = null;
  private failure: Error | null = null;

  private open(): ReturnType<typeof createDeflateRaw> {
    if (this.stream !== null) return this.stream;
    const z = createDeflateRaw();
    z.on('data', (chunk: Buffer) => {
      this.parts.push(new Uint8Array(chunk));
      this.bytes += chunk.byteLength;
    });
    // RESOLVES on error rather than rejecting, and records the failure instead: a
    // rejection nobody is awaiting yet (the writer throws first) is an unhandled
    // rejection, which in a server is a crash and not a 500. `finish` raises it.
    this.ended = new Promise<void>((resolve) => {
      z.on('end', () => resolve());
      z.on('error', (err: Error) => {
        this.failure = err;
        resolve();
      });
    });
    this.stream = z;
    return z;
  }

  /** Wait for room — or for the stream to fail, so a dead stream can never hang a derive. */
  private drain(z: ReturnType<typeof createDeflateRaw>): Promise<void> {
    return new Promise<void>((resolve) => {
      const settle = (): void => {
        z.removeListener('drain', settle);
        z.removeListener('error', settle);
        z.removeListener('close', settle);
        resolve();
      };
      z.once('drain', settle);
      z.once('error', settle);
      z.once('close', settle);
    });
  }

  async write(block: Uint8Array): Promise<void> {
    if (block.byteLength === 0) return;
    if (this.failure !== null) throw this.failure;
    const z = this.open();
    // Back-pressure is real here: one lane can be megabytes, and ignoring `drain` is how
    // a stream's internal queue becomes the copy this class exists to avoid.
    if (!z.write(block)) await this.drain(z);
    if (this.failure !== null) throw this.failure;
  }

  /**
   * The compressed section, as the pieces zlib emitted it in.
   *
   * NOT merged into one buffer: the framed body is the next allocation either way, so
   * merging first would hold the compressed section twice for no gain. `frame` copies
   * the pieces straight in behind the header.
   */
  async finish(): Promise<{ parts: readonly Uint8Array[]; bytes: number }> {
    if (this.stream === null) return { parts: [], bytes: 0 };
    this.stream.end();
    await this.ended;
    if (this.failure !== null) throw this.failure;
    return { parts: this.parts, bytes: this.bytes };
  }
}

async function computeWall(
  runId: string,
  envId: string,
  stride: number,
  lanes: readonly RecordingEntry[],
): Promise<WallPayload> {
  const heads: WallLaneHeader[] = [];
  const events: WallChunkEvents[] = [];
  const sink = new DeflateSink();

  // Sequential on purpose: each lane's decode holds one lane's worth of rows, and the
  // point of the streaming walk is that the wall's memory ceiling is ONE lane, not N.
  for (const lane of lanes) {
    const caps = laneCaps(laneShape(lane), stride, lanes.length);
    const projected = await projectLane(lane, stride, caps);
    heads.push(projected.head);
    if (projected.events !== null) events.push(projected.events);
    // Straight into the compressor, then dropped: nothing accumulates but the header.
    await sink.write(projected.bytes);
  }

  const binary = await sink.finish();

  const base: Omit<WallResponse, 'headerBytes'> = {
    v: 1,
    runId,
    envId,
    stride,
    binaryBytes: binary.bytes,
    lanes: heads,
    ...(events.length > 0 ? { chunkEvents: events } : {}),
  };
  return frame(base, binary.parts, binary.bytes);
}
