import { fetchRewards } from './oak';

export interface RewardDetail {
  from: number;
  steps: number;
  lanes: number;
  prefix: Float64Array;
}

export class RewardsCache {
  private readonly chunks = new Map<number, Float32Array>();
  private readonly inflight = new Set<number>();
  private readonly listeners = new Set<() => void>();
  private cached: { from: number; version: number; detail: RewardDetail | null } | null = null;
  version = 0;

  constructor(
    readonly runId: string,
    readonly envId: string,
    readonly lanes: number,
    readonly windowSteps: number,
  ) {}

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
    const first = Math.floor(Math.max(0, from) / this.windowSteps);
    const last = Math.ceil(Math.max(0, to) / this.windowSteps);
    for (let w = first; w < last; w++) {
      if (this.chunks.has(w) || this.inflight.has(w)) continue;
      void this.fetchWindow(w);
    }
  }

  private async fetchWindow(w: number): Promise<void> {
    this.inflight.add(w);
    try {
      const values = await fetchRewards(this.runId, this.envId, w * this.windowSteps, this.windowSteps);
      if (values.length === 0) return;
      this.chunks.set(w, values);
      this.bump();
    } catch {
      return;
    } finally {
      this.inflight.delete(w);
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
    let w = Math.floor(from / this.windowSteps);
    const parts: { chunk: Float32Array; startRow: number; rows: number }[] = [];
    let cursor = from;
    for (;;) {
      const chunk = this.chunks.get(w);
      if (chunk === undefined) break;
      const rowsInChunk = Math.floor(chunk.length / lanes);
      const startRow = cursor - w * this.windowSteps;
      if (startRow >= rowsInChunk) break;
      parts.push({ chunk, startRow, rows: rowsInChunk - startRow });
      cursor = w * this.windowSteps + rowsInChunk;
      if (rowsInChunk < this.windowSteps) break;
      w += 1;
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
