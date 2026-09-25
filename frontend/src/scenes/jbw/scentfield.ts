import type { EnvDescriptor } from '../../data/envs';
import * as THREE from 'three/webgpu';
import { scentConfig, cellKey, FIELD_TEXELS, FIELD_TILES, TILE_CELLS, TILE_VALUES, type FieldLog } from './fieldmath';
import type { FieldJob, FieldRequest, FieldTile } from './scentfield.worker';

export class ScentField {
  readonly origin = new THREE.Vector2(-FIELD_TEXELS / 2, -FIELD_TEXELS / 2);
  private readonly reach: number;
  private settleAge: number;
  readonly texture = new THREE.DataTexture(new Uint16Array(FIELD_TEXELS * FIELD_TEXELS * 4), FIELD_TEXELS, FIELD_TEXELS, THREE.RGBAFormat, THREE.HalfFloatType);
  private readonly scratch = new THREE.DataTexture(new Uint16Array(TILE_VALUES), TILE_CELLS, TILE_CELLS, THREE.RGBAFormat, THREE.HalfFloatType);
  private readonly worker = new Worker('/workers/scentfield.worker.js', { type: 'module' });
  private readonly cached = new Map<number, string>();
  private readonly pending = new Map<number, FieldJob>();
  private readonly posting = new Map<number, number[]>();
  private readonly received: FieldTile[] = [];
  private readonly uploadAt = new THREE.Vector3();
  private version = -1;
  private sent = 0;
  private sequence = 0;

  constructor(private readonly renderer: THREE.WebGPURenderer, seed: number, environment: EnvDescriptor) {
    const config = scentConfig(environment);
    this.reach = config.radius - 1; this.settleAge = config.lifetime;
    this.texture.name = 'threeChannelScent';
    this.texture.minFilter = this.texture.magFilter = THREE.LinearFilter;
    this.texture.needsUpdate = true;
    this.worker.onmessage = (event: MessageEvent<FieldTile | { op: 'ready'; settleAge: number }>) => { if ('op' in event.data) this.settleAge = event.data.settleAge; else this.received.push(event.data); };
    this.worker.onerror = event => { throw new Error(`Scent field worker failed: ${event.message}`); };
    this.post({ op: 'init', seed, environment });
  }

  private post(message: FieldRequest, transfer: Transferable[] = []): void { this.worker.postMessage(message, transfer); }

  sync(log: FieldLog, version: number): void {
    const reset = this.version !== version;
    if (reset) {
      this.version = version; this.sent = 0; this.posting.clear();
      this.clear();
    }
    if (!reset && this.sent === log.count) return;
    const from = this.sent, steps = log.steps.slice(from, log.count), xs = log.xs.slice(from, log.count), ys = log.ys.slice(from, log.count);
    const reach = this.reach;
    for (let i = 0; i < steps.length; i++) {
      for (let ty = Math.floor((ys[i]! - reach) / TILE_CELLS); ty <= Math.floor((ys[i]! + reach) / TILE_CELLS); ty++) {
        for (let tx = Math.floor((xs[i]! - reach) / TILE_CELLS); tx <= Math.floor((xs[i]! + reach) / TILE_CELLS); tx++) {
          const key = cellKey(tx, ty);
          if (!this.posting.has(key)) this.posting.set(key, []);
          this.posting.get(key)!.push(steps[i]!);
        }
      }
    }
    this.sent = log.count;
    this.post({ op: 'log', from, steps, xs, ys }, [steps.buffer, xs.buffer, ys.buffer]);
  }

  private clear(): void {
    this.cached.clear();
    this.texture.image.data!.fill(0);
    this.texture.needsUpdate = true;
  }

  private stamp(x: number, y: number, step: number): string {
    const steps = this.posting.get(cellKey(x, y)) ?? [];
    let lo = 0, hi = steps.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (steps[mid]! <= step) lo = mid + 1; else hi = mid;
    }
    const age = lo ? step - steps[lo - 1]! : this.settleAge;
    return `${this.version}:${lo}:${age < this.settleAge ? age : -1}`;
  }

  frame(step: number, x: number, y: number): void {
    const margin = TILE_CELLS * 2;
    if (x < this.origin.x + margin || y < this.origin.y + margin || x > this.origin.x + FIELD_TEXELS - margin || y > this.origin.y + FIELD_TEXELS - margin) {
      this.origin.set((Math.floor(x / TILE_CELLS) - FIELD_TILES / 2) * TILE_CELLS, (Math.floor(y / TILE_CELLS) - FIELD_TILES / 2) * TILE_CELLS);
      this.clear();
    }
    for (const tile of this.received.splice(0)) {
      this.pending.delete(tile.id);
      const sx = tile.x * TILE_CELLS - this.origin.x, sy = tile.y * TILE_CELLS - this.origin.y;
      if (sx < 0 || sy < 0 || sx >= FIELD_TEXELS || sy >= FIELD_TEXELS || tile.step > step || tile.stamp.split(':').slice(0, 2).join(':') !== this.stamp(tile.x, tile.y, step).split(':').slice(0, 2).join(':')) continue;
      this.scratch.image.data!.set(new Uint16Array(tile.rgba));
      this.scratch.needsUpdate = true;
      this.renderer.copyTextureToTexture(this.scratch, this.texture, null, this.uploadAt.set(sx, sy, 0));
      this.cached.set(cellKey(tile.x, tile.y), tile.stamp);
    }
    if (this.pending.size >= 4) return;
    const wanted: (FieldJob & { distance: number })[] = [];
    for (let ty = this.origin.y / TILE_CELLS; ty < this.origin.y / TILE_CELLS + FIELD_TILES; ty++) {
      for (let tx = this.origin.x / TILE_CELLS; tx < this.origin.x / TILE_CELLS + FIELD_TILES; tx++) {
        const stamp = this.stamp(tx, ty, step);
        if (this.cached.get(cellKey(tx, ty)) === stamp || [...this.pending.values()].some(job => job.x === tx && job.y === ty)) continue;
        wanted.push({ id: ++this.sequence, x: tx, y: ty, step, stamp, distance: ((tx + 0.5) * TILE_CELLS - x) ** 2 + ((ty + 0.5) * TILE_CELLS - y) ** 2 });
      }
    }
    wanted.sort((a, b) => a.distance - b.distance);
    for (const job of wanted.slice(0, 4 - this.pending.size)) {
      this.pending.set(job.id, job);
      this.post({ op: 'tile', job });
    }
  }

  dispose(): void { this.worker.terminate(); this.texture.dispose(); this.scratch.dispose(); }
}
