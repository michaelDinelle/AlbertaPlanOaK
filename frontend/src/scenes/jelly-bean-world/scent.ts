import { ClampToEdgeWrapping } from 'three';
import type * as THREE from 'three';
import { makeDataTexture } from '@kit/textures';
import {
  jbwBlockField,
  jbwCandidate,
  jbwItemAt,
  jbwScentIndex,
  JBW_BLOCK,
  JBW_BLOCK_ITEMS,
  JBW_DESTRUCTION_LIFETIME,
  JBW_EMPTY,
  JBW_SCENT,
  JBW_SCENT_CELLS,
  JBW_SCENT_DECAY,
  JBW_SCENT_DIFFUSION,
  JBW_SCENT_RADIUS,
} from '@sim/envs/jellybeanworld';

export const FIELD_SIZE = 256;
export const FIELD_MARGIN = 48;

const PRECISION_STEPS = 512;
const STALENESS_STEPS = JBW_DESTRUCTION_LIFETIME / 16;
const SCENT_PEAK = 8.4;
const KEY_BIAS = 1 << 22;
const KEY_SPAN = 1 << 23;

export function buildScentKernel(): Float32Array {
  const radius = JBW_SCENT_RADIUS;
  const cells = JBW_SCENT_CELLS;
  const lifetime = JBW_DESTRUCTION_LIFETIME;
  const response = new Float32Array(cells * lifetime);
  response[0] = 1;
  for (let t = 1; t < lifetime; t++) {
    const prev = (t - 1) * cells;
    const curr = t * cells;
    for (let i = 0; i < cells; i++) response[curr + i] = JBW_SCENT_DECAY * (response[prev + i] ?? 0);
    response[curr] = (response[curr] ?? 0) + 1;
    response[curr + cells - 1] =
      (response[curr + cells - 1] ?? 0) + 2 * JBW_SCENT_DIFFUSION * (response[prev + cells - 2] ?? 0);
    for (let y = 0; y < radius - 1; y++) {
      const at = curr + cells - radius + y;
      response[at] =
        (response[at] ?? 0) +
        JBW_SCENT_DIFFUSION *
          ((response[prev + jbwScentIndex(radius - 2, y)] ?? 0) +
            (response[prev + jbwScentIndex(radius - 1, y + 1)] ?? 0) +
            (response[prev + jbwScentIndex(radius - 1, y - 1)] ?? 0));
    }
    for (let x = 0; x < radius - 1; x++) {
      for (let y = 0; y <= x; y++) {
        const at = curr + jbwScentIndex(x, y);
        response[at] =
          (response[at] ?? 0) +
          JBW_SCENT_DIFFUSION *
            ((response[prev + jbwScentIndex(x + 1, y)] ?? 0) +
              (response[prev + jbwScentIndex(x - 1, y)] ?? 0) +
              (response[prev + jbwScentIndex(x, y + 1)] ?? 0) +
              (response[prev + jbwScentIndex(x, y - 1)] ?? 0));
      }
    }
  }
  return response;
}

function scarKey(x: number, y: number): number {
  return (x + KEY_BIAS) * KEY_SPAN + (y + KEY_BIAS);
}

export class ScentField {
  readonly texture: THREE.DataTexture;
  readonly knee: number;

  private readonly seed: number;
  private readonly kernel: Float32Array;
  private readonly steady: number;
  private readonly reach: number;
  private readonly data = new Uint8Array(FIELD_SIZE * FIELD_SIZE * 4);
  private readonly accum = new Float32Array(FIELD_SIZE * FIELD_SIZE * 3);
  private readonly scars = new Map<number, number>();

  private cornerX = 0;
  private cornerY = 0;
  private splatX = Number.NaN;
  private splatY = Number.NaN;
  private splatStep = 0;
  private dirty = true;

  constructor(seed: number, kernel: Float32Array, anisotropy: number) {
    this.seed = seed;
    this.kernel = kernel;
    this.steady = (JBW_DESTRUCTION_LIFETIME - 1) * JBW_SCENT_CELLS;
    this.knee = Math.max(1e-6, kernel[this.steady] ?? 1);
    let reach = JBW_SCENT_RADIUS;
    const floor = this.knee / PRECISION_STEPS;
    for (let r = 1; r < JBW_SCENT_RADIUS; r++) {
      if ((kernel[this.steady + jbwScentIndex(r, 0)] ?? 0) < floor) {
        reach = r;
        break;
      }
    }
    this.reach = reach;
    this.texture = makeDataTexture(this.data, FIELD_SIZE, FIELD_SIZE, 'orchard-scent');
    this.texture.wrapS = ClampToEdgeWrapping;
    this.texture.wrapT = ClampToEdgeWrapping;
    this.texture.anisotropy = Math.max(1, anisotropy);
  }

  get origin(): readonly [number, number] {
    return [this.cornerX, this.cornerY];
  }

  get radius(): number {
    return this.reach;
  }

  scar(x: number, y: number, step: number): void {
    const key = scarKey(x, y);
    if (this.scars.has(key)) return;
    this.scars.set(key, step);
    this.dirty = true;
  }

  depleted(x: number, y: number): boolean {
    return this.scars.has(scarKey(x, y));
  }

  resetScars(): void {
    this.scars.clear();
    this.dirty = true;
  }

  update(agentX: number, agentY: number, step: number): void {
    const drift = Math.max(Math.abs(agentX - this.splatX), Math.abs(agentY - this.splatY));
    if (this.dirty || !(drift <= FIELD_MARGIN - this.reach) || step - this.splatStep >= STALENESS_STEPS) {
      this.splat(agentX, agentY, step);
    }
    this.texture.needsUpdate = true;
  }

  private splat(agentX: number, agentY: number, step: number): void {
    this.accum.fill(0);
    this.cornerX = agentX - FIELD_SIZE / 2;
    this.cornerY = agentY - FIELD_SIZE / 2;
    this.splatX = agentX;
    this.splatY = agentY;
    this.splatStep = step;
    this.dirty = false;
    this.prune(step);

    const reach = this.reach;
    const firstX = Math.floor((this.cornerX - reach) / JBW_BLOCK);
    const lastX = Math.floor((this.cornerX + FIELD_SIZE + reach) / JBW_BLOCK);
    const firstY = Math.floor((this.cornerY - reach) / JBW_BLOCK);
    const lastY = Math.floor((this.cornerY + FIELD_SIZE + reach) / JBW_BLOCK);

    for (let by = firstY; by <= lastY; by++) {
      for (let bx = firstX; bx <= lastX; bx++) {
        const block = jbwBlockField(this.seed, bx, by);
        for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
          const at = jbwCandidate(block, bx, by, slot);
          const item = jbwItemAt(this.seed, at[0], at[1]);
          if (item === JBW_EMPTY) continue;
          const scent = JBW_SCENT[item];
          if (!scent || (scent[0] === 0 && scent[1] === 0 && scent[2] === 0)) continue;
          this.deposit(at[0], at[1], scent, step);
        }
      }
    }
    this.encode();
  }

  private prune(step: number): void {
    for (const [key, removed] of this.scars) {
      if (step - removed >= JBW_DESTRUCTION_LIFETIME) this.scars.delete(key);
    }
  }

  private deposit(
    wx: number,
    wy: number,
    scent: readonly [number, number, number],
    step: number,
  ): void {
    const removed = this.scars.get(scarKey(wx, wy));
    let decayed = -1;
    if (removed !== undefined) {
      const age = step - removed;
      if (age >= JBW_DESTRUCTION_LIFETIME) return;
      decayed = (age < 0 ? 0 : age) * JBW_SCENT_CELLS;
    }
    const reach = this.reach;
    const tx = wx - this.cornerX;
    const ty = wy - this.cornerY;
    const x0 = Math.max(0, tx - reach);
    const x1 = Math.min(FIELD_SIZE - 1, tx + reach);
    const y0 = Math.max(0, ty - reach);
    const y1 = Math.min(FIELD_SIZE - 1, ty + reach);
    for (let y = y0; y <= y1; y++) {
      const dy = y - ty;
      const row = y * FIELD_SIZE;
      for (let x = x0; x <= x1; x++) {
        const cell = jbwScentIndex(x - tx, dy);
        let value = this.kernel[this.steady + cell] ?? 0;
        if (decayed >= 0) value -= this.kernel[decayed + cell] ?? 0;
        if (value <= 0) continue;
        const at = (row + x) * 3;
        this.accum[at] = (this.accum[at] ?? 0) + scent[0] * value;
        this.accum[at + 1] = (this.accum[at + 1] ?? 0) + scent[1] * value;
        this.accum[at + 2] = (this.accum[at + 2] ?? 0) + scent[2] * value;
      }
    }
  }

  private encode(): void {
    const scale = 1 / this.knee;
    const norm = 255 / Math.log1p(SCENT_PEAK * this.knee * scale);
    for (let i = 0, n = FIELD_SIZE * FIELD_SIZE; i < n; i++) {
      const at = i * 3;
      const to = i * 4;
      this.data[to] = Math.min(255, Math.log1p((this.accum[at] ?? 0) * scale) * norm) | 0;
      this.data[to + 1] = Math.min(255, Math.log1p((this.accum[at + 1] ?? 0) * scale) * norm) | 0;
      this.data[to + 2] = Math.min(255, Math.log1p((this.accum[at + 2] ?? 0) * scale) * norm) | 0;
    }
  }



  dispose(): void {
    this.texture.dispose();
    this.scars.clear();
  }
}
