import * as THREE from 'three';
import {
  bundleKernel,
  buildScentKernel,
  FIELD_TEXELS,
  FIELD_TILES,
  TILE_CELLS,
  type FieldLog,
  type KernelBundle,
  type TilePalette,
} from './fieldmath';
import type { FieldRequest, FieldTile, FieldWantTile } from './scentfield.worker';

const TILE_KEY_BIAS = 1 << 18;
const TILE_KEY_SPAN = 1 << 19;
const UPLOAD_BUDGET_MS = 2;
const AGE_BUCKETS = Math.ceil(Math.log2(512));
const RECENTER_MARGIN = 2 * TILE_CELLS;

function tileKey(tx: number, ty: number): number {
  return (tx + TILE_KEY_BIAS) * TILE_KEY_SPAN + (ty + TILE_KEY_BIAS);
}

function appliedBefore(steps: number[], playhead: number): number {
  let lo = 0;
  let hi = steps.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((steps[mid] ?? 0) <= playhead) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export class ScentField {
  readonly texture: THREE.DataTexture;
  readonly origin = new THREE.Vector2(-FIELD_TEXELS / 2, -FIELD_TEXELS / 2);
  readonly readiness = { value: 0 };
  readonly kernel: KernelBundle;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly worker: Worker;
  private readonly ageQuantum: number;
  private readonly cached = new Map<number, string>();
  private readonly inflight = new Map<number, string>();
  private readonly received: FieldTile[] = [];
  private readonly posting = new Map<number, number[]>();
  private readonly scratch: THREE.DataTexture;
  private readonly scratchData: Uint8Array;
  private readonly uploadAt = new THREE.Vector3();
  private log: FieldLog | null = null;
  private logVersion = 0;
  private sent = 0;
  private lastWants = '';

  constructor(renderer: THREE.WebGLRenderer, seed: number, palette: TilePalette) {
    this.renderer = renderer;
    this.kernel = bundleKernel(buildScentKernel());
    this.ageQuantum = Math.max(1, Math.ceil(this.kernel.settleAge / AGE_BUCKETS));
    const data = new Uint8Array(FIELD_TEXELS * FIELD_TEXELS * 4);
    this.texture = new THREE.DataTexture(data, FIELD_TEXELS, FIELD_TEXELS);
    this.texture.name = 'orchardScent';
    this.texture.wrapS = THREE.ClampToEdgeWrapping;
    this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.anisotropy = Math.max(1, renderer.capabilities.getMaxAnisotropy());
    this.texture.needsUpdate = true;
    this.scratchData = new Uint8Array(TILE_CELLS * TILE_CELLS * 4);
    this.scratch = new THREE.DataTexture(this.scratchData, TILE_CELLS, TILE_CELLS);
    this.scratch.name = 'orchardScentScratch';
    this.worker = new Worker(new URL('./scentfield.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent<FieldTile>) => {
      this.received.push(event.data);
    };
    this.post({ op: 'init', seed, palette });
  }

  private post(message: FieldRequest, transfer?: Transferable[]): void {
    if (transfer) this.worker.postMessage(message, transfer);
    else this.worker.postMessage(message);
  }

  sync(log: FieldLog, version: number): void {
    if (this.log === log && this.logVersion === version) {
      this.extend();
      return;
    }
    this.log = log;
    this.logVersion = version;
    this.sent = 0;
    this.cached.clear();
    this.inflight.clear();
    this.lastWants = '';
    this.posting.clear();
    this.extend();
  }

  private extend(): void {
    const log = this.log;
    if (!log || this.sent >= log.count) return;
    const from = this.sent;
    const count = log.count - from;
    const steps = new Int32Array(log.steps.subarray(from, from + count));
    const xs = new Int32Array(log.xs.subarray(from, from + count));
    const ys = new Int32Array(log.ys.subarray(from, from + count));
    for (let i = 0; i < count; i++) {
      this.index(steps[i] ?? 0, xs[i] ?? 0, ys[i] ?? 0);
    }
    this.sent = log.count;
    this.post({ op: 'log', from, steps, xs, ys }, [steps.buffer, xs.buffer, ys.buffer]);
  }

  private index(step: number, x: number, y: number): void {
    const reach = this.kernel.reach;
    const tx0 = Math.floor((x - reach) / TILE_CELLS);
    const tx1 = Math.floor((x + reach) / TILE_CELLS);
    const ty0 = Math.floor((y - reach) / TILE_CELLS);
    const ty1 = Math.floor((y + reach) / TILE_CELLS);
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        const key = tileKey(tx, ty);
        const list = this.posting.get(key);
        if (list) list.push(step);
        else this.posting.set(key, [step]);
      }
    }
  }

  private stampFor(tx: number, ty: number, playhead: number): string {
    const list = this.posting.get(tileKey(tx, ty));
    if (!list || list.length === 0) return `${this.logVersion}:0:-1`;
    const applied = appliedBefore(list, playhead);
    if (applied === 0) return `${this.logVersion}:0:-1`;
    const young = playhead - (list[applied - 1] ?? 0);
    const bucket = young < this.kernel.settleAge ? Math.floor(young / this.ageQuantum) : -1;
    return `${this.logVersion}:${applied}:${bucket}`;
  }

  private recenter(agentX: number, agentY: number): void {
    const outX =
      agentX < this.origin.x + RECENTER_MARGIN ||
      agentX > this.origin.x + FIELD_TEXELS - RECENTER_MARGIN;
    const outY =
      agentY < this.origin.y + RECENTER_MARGIN ||
      agentY > this.origin.y + FIELD_TEXELS - RECENTER_MARGIN;
    if (!outX && !outY) return;
    this.origin.set(
      (Math.round(agentX / TILE_CELLS) - FIELD_TILES / 2) * TILE_CELLS,
      (Math.round(agentY / TILE_CELLS) - FIELD_TILES / 2) * TILE_CELLS,
    );
    this.cached.clear();
    this.inflight.clear();
    this.lastWants = '';
  }

  frame(playhead: number, agentX: number, agentY: number): void {
    this.recenter(agentX, agentY);
    const firstTX = this.origin.x / TILE_CELLS;
    const firstTY = this.origin.y / TILE_CELLS;
    const wants: FieldWantTile[] = [];
    let fresh = 0;
    for (let ty = firstTY; ty < firstTY + FIELD_TILES; ty++) {
      for (let tx = firstTX; tx < firstTX + FIELD_TILES; tx++) {
        const key = tileKey(tx, ty);
        const stamp = this.stampFor(tx, ty, playhead);
        if (this.cached.get(key) === stamp) {
          fresh += 1;
          continue;
        }
        if (this.cached.has(key)) fresh += 1;
        if (this.inflight.get(key) === stamp) continue;
        wants.push({ key: `${tx}:${ty}`, tileX: tx, tileY: ty, stamp });
      }
    }
    this.readiness.value = fresh / (FIELD_TILES * FIELD_TILES);
    const signature = wants.length === 0 ? '' : `${wants.length}|${wants[0]?.key}${wants[0]?.stamp}|${wants[wants.length - 1]?.key}${wants[wants.length - 1]?.stamp}`;
    if (wants.length > 0 && signature !== this.lastWants) {
      this.lastWants = signature;
      for (const want of wants) this.inflight.set(tileKey(want.tileX, want.tileY), want.stamp);
      this.post({ op: 'want', playhead, tiles: wants });
    }
    this.upload();
  }

  private upload(): void {
    const started = performance.now();
    while (performance.now() - started < UPLOAD_BUDGET_MS) {
      const tile = this.received.shift();
      if (!tile) return;
      const slotX = tile.tileX - this.origin.x / TILE_CELLS;
      const slotY = tile.tileY - this.origin.y / TILE_CELLS;
      if (slotX < 0 || slotX >= FIELD_TILES || slotY < 0 || slotY >= FIELD_TILES) continue;
      this.scratchData.set(new Uint8Array(tile.rgba));
      this.scratch.needsUpdate = true;
      this.uploadAt.set(slotX * TILE_CELLS, slotY * TILE_CELLS, 0);
      this.renderer.copyTextureToTexture(this.scratch, this.texture, null, this.uploadAt);
      const key = tileKey(tile.tileX, tile.tileY);
      this.cached.set(key, tile.stamp);
      if (this.inflight.get(key) === tile.stamp) this.inflight.delete(key);
      this.lastWants = '';
    }
  }

  dispose(): void {
    this.worker.terminate();
    this.texture.dispose();
    this.scratch.dispose();
  }
}
