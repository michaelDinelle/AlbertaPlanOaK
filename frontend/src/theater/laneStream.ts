import { entryFor, type EnvCatalog, type EnvEntry } from '@sim/catalog';
import { GEOMETRY_BYTES } from '@sim/geometry';
import {
  FILE_HEADER_BYTES,
  RECORD_FLOATS,
  codecFromEntry,
  decodeChunkPayload,
  inflateChunk,
  scanNextChunk,
} from '@sim/record';
import { apiUrl } from '../api';

export const LANES_MAGIC = 0x4c4b414f;
const INDEX_ROW_BYTES = 20;
const INDEX_HEAD_BYTES = 20;

export class LaneStreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LaneStreamError';
  }
}

export interface LaneWindow {
  from: number;
  count: number;
  seeds: Int32Array;
  geometries: Uint8Array[];
  poses: Float32Array;
  stride: number;
  laneCount: number;
  steps: number;
}

export interface LaneStream {
  readonly envId: string;
  readonly entry: EnvEntry;
  geometryBytes(lane: number): Uint8Array | null;
  readonly window: LaneWindow | null;
  readonly laneCount: number;
  seedAt(lane: number): number;
  pose(lane: number, step: number, out: Float32Array): boolean;
  ensure(step: number): Promise<void>;
  destroy(): void;
}

async function decodeBlock(
  bytes: Uint8Array,
  entry: EnvEntry,
  from: number,
  count: number,
): Promise<LaneWindow> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== LANES_MAGIC) {
    throw new LaneStreamError('lane block has the wrong magic');
  }
  const laneCount = dv.getUint32(4, true);
  const blockFrom = dv.getUint32(8, true);
  const desc = codecFromEntry(entry);
  const stride = RECORD_FLOATS;
  const poses = new Float32Array(laneCount * count * stride);
  const seeds = new Int32Array(laneCount);
  const geometries: Uint8Array[] = [];

  for (let lane = 0; lane < laneCount; lane++) {
    const row = INDEX_HEAD_BYTES + lane * INDEX_ROW_BYTES;
    seeds[lane] = dv.getInt32(row, true);
    const offset = dv.getUint32(row + 4, true);
    const length = dv.getUint32(row + 8, true);
    const stepStart = dv.getUint32(row + 12, true);
    geometries[lane] = bytes.slice(offset, offset + GEOMETRY_BYTES);
    if (length <= GEOMETRY_BYTES) continue;

    const laneBytes = bytes.subarray(offset + GEOMETRY_BYTES, offset + length);
    const scratch = new Float32Array(count * stride);
    let decoded = 0;
    let cursor = 0;
    let expect = stepStart;
    for (;;) {
      const chunk = scanNextChunk(laneBytes, cursor, expect, 0);
      if (chunk === null) break;
      const payload = await inflateChunk(laneBytes, chunk);
      const room = count - decoded;
      if (room <= 0) break;
      const take = Math.min(chunk.stepCount, room);
      decodeChunkPayload(payload, desc, chunk.stepCount, scratch, decoded * stride);
      decoded += take;
      expect = chunk.stepStart + chunk.stepCount;
      cursor = chunk.byteOffset + chunk.byteLength;
    }
    const skip = Math.max(0, from - stepStart);
    const usable = Math.max(0, decoded - skip);
    if (usable > 0) {
      poses.set(
        scratch.subarray(skip * stride, (skip + usable) * stride),
        lane * count * stride,
      );
    }
  }

  return { from: blockFrom, count, seeds, geometries, poses, stride, laneCount, steps: count };
}

export interface LaneStreamOptions {
  runId: string;
  envId: string;
  catalog: EnvCatalog;
  windowSteps?: number;
  onUpdate?(): void;
}

export function makeLaneStream(opts: LaneStreamOptions): LaneStream {
  const entry = entryFor(opts.catalog, opts.envId);
  const windowSteps = Math.max(1024, Math.floor(opts.windowSteps ?? 8192));
  let window: LaneWindow | null = null;
  let pending: Promise<void> | null = null;
  let pendingFrom = -1;
  let destroyed = false;

  const load = async (from: number): Promise<void> => {
    const url = apiUrl(
      `/runs/${encodeURIComponent(opts.runId)}/lanes/${encodeURIComponent(opts.envId)}`
        + `?from=${from}&count=${windowSteps}`,
    );
    const res = await fetch(url);
    if (!res.ok) throw new LaneStreamError(`lane block ${res.status} for ${opts.envId}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (destroyed) return;
    window = await decodeBlock(bytes, entry, from, windowSteps);
    opts.onUpdate?.();
  };

  const blockOf = (step: number): number =>
    Math.max(0, Math.floor(step / windowSteps) * windowSteps);

  return {
    envId: opts.envId,
    entry,
    get window() {
      return window;
    },
    get laneCount() {
      return window?.laneCount ?? 0;
    },
    seedAt(lane) {
      return window?.seeds[lane] ?? -1;
    },
    geometryBytes(lane) {
      return window?.geometries[lane] ?? null;
    },
    pose(lane, step, out) {
      const w = window;
      if (w === null || lane < 0 || lane >= w.laneCount) return false;
      const local = step - w.from;
      if (local < 0 || local >= w.count) return false;
      const base = (lane * w.count + local) * w.stride;
      for (let i = 0; i < 9; i++) out[i] = w.poses[base + 7 + i] ?? 0;
      return true;
    },
    async ensure(step) {
      if (destroyed) return;
      const want = blockOf(step);
      if (window !== null && window.from === want) return;
      if (pending !== null && pendingFrom === want) return pending;
      pendingFrom = want;
      pending = load(want).finally(() => {
        pending = null;
        pendingFrom = -1;
      });
      return pending;
    },
    destroy() {
      destroyed = true;
      window = null;
    },
  };
}

void FILE_HEADER_BYTES;
