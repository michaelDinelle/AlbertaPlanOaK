import { inflateRaw } from './parse';

export const REWARD_PACK_MAGIC = 0x4b505752;
export const REWARD_IDX_MAGIC = 0x5849_5752;
const IDX_HEADER_BYTES = 8;
const IDX_RECORD_BYTES = 20;

export interface RewardDetail {
  from: number;
  steps: number;
  lanes: number;
  prefix: Float64Array;
}

interface ChunkRecord {
  stepStart: number;
  stepCount: number;
  offset: number;
  length: number;
}

export class RewardStream {
  private lanes: number | null = null;
  private records: ChunkRecord[] = [];
  private idxBytes = 0;
  private idxFetchedAt = 0;
  private idxInflight = false;
  private readonly chunks = new Map<number, Float32Array>();
  private readonly inflight = new Set<number>();
  private readonly listeners = new Set<() => void>();
  private cached: { from: number; version: number; detail: RewardDetail | null } | null = null;
  version = 0;

  constructor(readonly packUrl: string, readonly idxUrl: string) {}

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private bump(): void {
    this.version += 1;
    for (const listener of this.listeners) listener();
  }

  ensure(from: number, to: number): void {
    const covered = this.records.length > 0 ? this.records[this.records.length - 1]! : null;
    const coveredTo = covered === null ? 0 : covered.stepStart + covered.stepCount;
    if ((this.lanes === null || coveredTo < to) && !this.idxInflight && performance.now() - this.idxFetchedAt > 1000) void this.fetchIndex();
    if (this.lanes === null) return;
    for (let c = 0; c < this.records.length; c++) {
      const record = this.records[c]!;
      if (record.stepStart + record.stepCount <= from || record.stepStart >= to) continue;
      if (this.chunks.has(c) || this.inflight.has(c)) continue;
      void this.fetchChunk(c, record);
    }
  }

  private async fetchIndex(): Promise<void> {
    this.idxInflight = true;
    try {
      const headers: Record<string, string> = this.idxBytes > 0 ? { range: `bytes=${this.idxBytes}-` } : {};
      const response = await fetch(this.idxUrl, { headers });
      this.idxFetchedAt = performance.now();
      if (response.status === 204 || response.status === 416) return;
      if (!response.ok && response.status !== 206) return;
      const bytes = new Uint8Array(await response.arrayBuffer());
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const fresh = response.status !== 206;
      let at = 0;
      if (fresh) {
        if (bytes.byteLength < IDX_HEADER_BYTES || view.getUint32(0, true) !== REWARD_IDX_MAGIC) return;
        this.lanes = view.getUint32(4, true);
        this.records = [];
        this.chunks.clear();
        this.idxBytes = 0;
        at = IDX_HEADER_BYTES;
      }
      let added = 0;
      while (at + IDX_RECORD_BYTES <= bytes.byteLength) {
        this.records.push({
          stepStart: view.getUint32(at, true),
          stepCount: view.getUint32(at + 4, true),
          offset: Number(view.getBigUint64(at + 8, true)),
          length: view.getUint32(at + 16, true),
        });
        at += IDX_RECORD_BYTES;
        added += 1;
      }
      this.idxBytes += at;
      if (added > 0 || fresh) this.bump();
    } catch {
      return;
    } finally {
      this.idxInflight = false;
    }
  }

  private async fetchChunk(index: number, record: ChunkRecord): Promise<void> {
    this.inflight.add(index);
    try {
      const response = await fetch(this.packUrl, { headers: { range: `bytes=${record.offset}-${record.offset + record.length - 1}` } });
      if (!response.ok && response.status !== 206) return;
      const compressed = new Uint8Array(await response.arrayBuffer());
      if (compressed.byteLength !== record.length) return;
      const raw = await inflateRaw(compressed);
      const values = new Float32Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 4));
      this.chunks.set(index, values.slice());
      this.bump();
    } catch {
      return;
    } finally {
      this.inflight.delete(index);
    }
  }

  detail(from: number): RewardDetail | null {
    const cached = this.cached;
    if (cached !== null && cached.from === from && cached.version === this.version) return cached.detail;
    const detail = this.buildDetail(from);
    this.cached = { from, version: this.version, detail };
    return detail;
  }

  private buildDetail(from: number): RewardDetail | null {
    const lanes = this.lanes;
    if (lanes === null) return null;
    let index = this.records.findIndex((record) => record.stepStart <= from && from < record.stepStart + record.stepCount);
    if (index < 0) return null;
    const parts: { chunk: Float32Array; startRow: number; rows: number }[] = [];
    let cursor = from;
    while (index < this.records.length) {
      const record = this.records[index]!;
      const chunk = this.chunks.get(index);
      if (chunk === undefined || record.stepStart > cursor) break;
      const rowsInChunk = Math.floor(chunk.length / lanes);
      const startRow = cursor - record.stepStart;
      if (startRow >= rowsInChunk) break;
      parts.push({ chunk, startRow, rows: rowsInChunk - startRow });
      cursor = record.stepStart + rowsInChunk;
      index += 1;
    }
    const steps = cursor - from;
    if (steps <= 0) return null;
    const prefix = new Float64Array(lanes * (steps + 1));
    const sums = new Float64Array(lanes);
    let k = 0;
    for (const part of parts) {
      for (let row = part.startRow; row < part.startRow + part.rows; row++) {
        const base = row * lanes;
        k += 1;
        for (let lane = 0; lane < lanes; lane++) {
          const value = part.chunk[base + lane] ?? 0;
          if (Number.isFinite(value)) sums[lane] = (sums[lane] ?? 0) + value;
          prefix[lane * (steps + 1) + k] = sums[lane] ?? 0;
        }
      }
    }
    return { from, steps, lanes, prefix };
  }
}
