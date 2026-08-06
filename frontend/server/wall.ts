/// <reference lib="dom" />

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


export const WALL_DEFAULT_STRIDE = 64;
export const WALL_MIN_STRIDE = 1;
export const WALL_MAX_STRIDE = 4096;

export const WALL_SAMPLE_BYTES = POSE_LEN * 4 + 1;

export const WALL_MAX_RAW_BYTES = 128 * 1024 * 1024;

export const WALL_MAX_DECODE_ROWS = 320_000_000;

export const WALL_LANE_OVERSHOOT = 2;

const WALL_YIELD_ROWS = 1 << 16;

const POSE_BYTE_OFF = OFF_POSE * 4;
const POSE_BYTES = POSE_LEN * 4;

export const WALL_CODEC_VERSION = 2;

const WALL_LRU_MAX = 4;
export const WALL_LRU_MAX_BYTES = 128 * 1024 * 1024;


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


export interface WallRequest {
  envId: string;
  stride: number;
}

export interface WallRequestError {
  error: string;
  status: number;
}

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


export function wallCacheDir(benchRoot: string): string {
  return resolve(CACHE_DIR, 'wall', benchKey(benchRoot));
}

export function wallCacheFor(benchRoot: string, fleetCache: string | null | undefined): string | null {
  if (fleetCache === undefined) return wallCacheDir(benchRoot);
  if (fleetCache === null) return null;
  return join(fleetCache, 'wall');
}


export interface WallPayload {
  body: Uint8Array<ArrayBuffer>;
  header: WallResponse;
}

interface LruEntry {
  key: string;
  value: WallPayload;
}

const lru: LruEntry[] = [];
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

export function wallCacheBytes(): number {
  return lruBytes;
}

export function clearWallCache(): void {
  lru.length = 0;
  lruBytes = 0;
  inFlight.clear();
}


const inFlight = new Map<string, Promise<WallPayload>>();

let computeActive = 0;
const computeQueue: (() => void)[] = [];

async function withComputeSlot<T>(fn: () => Promise<T>): Promise<T> {
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


function safeName(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
}

interface DiskCacheRecord {
  v: 1;
  kind: 'wall';
  key: string;
  runId: string;
  envId: string;
  stride: number;
  codecVersion: number;
  bodyBytes?: number;
  bodyCrc?: number;
}

function diskCachePath(cacheDir: string, id: string, envId: string, key: string): string {
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
    if (f.v !== 1 || f.kind !== 'wall' || f.key !== rec.key) return null;
    if (f.runId !== rec.runId || f.envId !== rec.envId) return null;
    if (f.stride !== rec.stride || f.codecVersion !== rec.codecVersion) return null;
    const body = bytes.slice(nl + 1);
    if (typeof f.bodyBytes !== 'number' || f.bodyBytes !== body.byteLength) return null;
    if (typeof f.bodyCrc !== 'number' || f.bodyCrc !== crc32(body)) return null;
    const header = parseWallBody(body);
    if (header === null) return null;
    return { body, header };
  } catch {
    return null;
  }
}

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
  }
  await evictDiskCache(path);
}

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
    entries.sort((a, b) => a.mtimeMs - b.mtimeMs);
    let count = entries.length;
    for (const e of entries) {
      if (count <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) break;
      if (e.path === writtenPath) continue;
      await unlink(e.path).catch(() => undefined);
      count--;
      total -= e.size;
    }
  } catch {
  }
}


export interface WallOptions {
  id: string;
  dir: string;
  envId: string;
  stride: number;
  recordings: readonly RecordingEntry[];
  inFlight: boolean;
  cacheDir?: string | null;
}

export interface WallLaneShape {
  steps: number;
  chunkSteps?: number;
}

export function decodedRowsAt(lane: WallLaneShape, stride: number): number {
  const steps = lane.steps > 0 ? lane.steps : 0;
  const chunk = lane.chunkSteps !== undefined && lane.chunkSteps > 0 ? lane.chunkSteps : 0;
  if (steps === 0) return 0;
  if (chunk === 0 || stride <= chunk) return steps;
  const chunks = Math.ceil(steps / chunk);
  const lastChunkRows = steps - (chunks - 1) * chunk;
  const sampled = Math.floor((steps - 1) / stride) + 1;
  const lastSampleChunk = Math.floor((Math.floor((steps - 1) / stride) * stride) / chunk);
  const fullChunks = lastSampleChunk === chunks - 1 ? sampled - 1 : sampled;
  return Math.min(fullChunks, chunks - 1) * chunk + lastChunkRows;
}

function totalDecodedRows(lanes: readonly WallLaneShape[], stride: number): number {
  let rows = 0;
  for (const lane of lanes) rows += decodedRowsAt(lane, stride);
  return rows;
}

export function effectiveStride(lanes: readonly WallLaneShape[], requested: number): number {
  let totalSteps = 0;
  for (const lane of lanes) totalSteps += lane.steps > 0 ? lane.steps : 0;
  if (totalSteps <= 0) return requested;

  const budget = Math.floor(WALL_MAX_RAW_BYTES / WALL_SAMPLE_BYTES) - lanes.length;
  if (budget <= 0) return WALL_MAX_STRIDE;
  const stride = Math.min(WALL_MAX_STRIDE, Math.max(requested, Math.ceil(totalSteps / budget)));

  if (totalDecodedRows(lanes, stride) <= WALL_MAX_DECODE_ROWS) return stride;
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

interface LaneCaps {
  bytes: number;
  rows: number;
}

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

function laneShape(lane: RecordingEntry): WallLaneShape {
  const first = lane.chunks?.[0];
  const chunkSteps = first === undefined ? 0 : first[1];
  return chunkSteps > 0 ? { steps: lane.steps, chunkSteps } : { steps: lane.steps };
}

export async function deriveWall(opts: WallOptions): Promise<WallPayload> {
  const { id, dir, envId } = opts;
  const lanes = opts.recordings.filter((r) => r.envId === envId);
  const shapes = lanes.map(laneShape);
  const stride = effectiveStride(shapes, opts.stride);

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
  const key =
    `${id}|${envId}|${stride}|${WALL_CODEC_VERSION}|${RECORD4_CODEC_VERSION}|${laneKeys.join(',')}`;
  const memKey = `${dir}|${key}`;

  const memHit = lruGet(memKey);
  if (memHit !== undefined) return memHit;

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


interface LaneProjection {
  bytes: Uint8Array;
  head: WallLaneHeader;
  events: WallChunkEvents | null;
}

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

function rowNonFinite(data: Float32Array, row: number): boolean {
  const base = row * RECORD_FLOATS;
  if (!Number.isFinite(data[base + OFF_DONE] ?? 0)) return true;
  for (let j = 0; j < POSE_LEN; j++) {
    if (!Number.isFinite(data[base + OFF_POSE + j] ?? 0)) return true;
  }
  return false;
}

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

class LaneSampler {
  private readonly blocks: Uint8Array[] = [];
  private readonly tail = new Uint8Array(WALL_SAMPLE_BYTES);
  private samples = 0;
  private nextRow = 0;
  private lastEmitted = -1;
  private haveTail = false;
  private tailRow = -1;
  private readonly maxSamples: number;
  steps = 0;
  scannedRows = 0;
  firstNonFiniteRow: number | null = null;
  truncated = false;

  constructor(
    private readonly stride: number,
    maxBytes: number,
  ) {
    this.maxSamples = Math.max(2, Math.floor(maxBytes / WALL_SAMPLE_BYTES));
  }

  get deathRow(): number | null {
    const row = this.firstNonFiniteRow;
    return row !== null && row < this.steps ? row : null;
  }

  get nextSampleRow(): number {
    return this.nextRow;
  }

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
      this.truncated = true;
      this.steps = this.lastEmitted + 1;
      this.haveTail = false;
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

function yieldToLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
}

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
  let covered = 0;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (chunk === undefined) break;
    const end = chunk.stepStart + chunk.stepCount;
    const holdsSample = sampler.nextSampleRow < end;
    const isLast = i === chunks.length - 1;
    const needed = holdsSample || isLast;

    ev.stepStart.push(chunk.stepStart);
    ev.stepCount.push(chunk.stepCount);
    ev.doneCount.push(chunk.doneCount);
    ev.rewardDelta.push(Number.isFinite(chunk.rewardDelta) ? chunk.rewardDelta : null);
    if (!needed) continue;

    if (covered > 0 && chunk.stepCount > rowsLeft) {
      sampler.truncated = true;
      break;
    }

    if (crc32(bytes, chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes) !== chunk.crc32) {
      break;
    }
    let payload: Uint8Array;
    try {
      payload = new Uint8Array(
        inflateRawSync(bytes.subarray(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes)),
      );
      if (payload.byteLength !== chunk.rawBytes) break;
    } catch {
      break;
    }

    const floats = chunk.stepCount * RECORD_FLOATS;
    if (scratch.length < floats) scratch = new Float32Array(floats);
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
      const whole = bytes.byteLength - (bytes.byteLength % RECORD_BYTES);
      const raw = parseActionsBin(bytes.subarray(0, whole), meta);
      const rows = Math.min(raw.steps, caps.rows);
      if (rows > 0) {
        if (rows < raw.steps) sampler.truncated = true;
        sampler.push(raw.data, 0, rows);
      }
    }
  } catch (err) {
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
  if (sampler.scannedRows < sampler.steps) head.scannedRows = sampler.scannedRows;
  if (sampler.truncated) head.truncated = true;
  return { bytes, head, events };
}


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
    if (!z.write(block)) await this.drain(z);
    if (this.failure !== null) throw this.failure;
  }

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

  for (const lane of lanes) {
    const caps = laneCaps(laneShape(lane), stride, lanes.length);
    const projected = await projectLane(lane, stride, caps);
    heads.push(projected.head);
    if (projected.events !== null) events.push(projected.events);
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
