import {
  jbwBlockField,
  jbwCandidate,
  jbwField,
  jbwItemAt,
  jbwScentIndex,
  jbwTypeOf,
  jbwWallLine,
  JBW_BLOCK,
  JBW_BLOCK_ITEMS,
  JBW_DESTRUCTION_LIFETIME,
  JBW_EMPTY,
  JBW_SCENT,
  JBW_SCENT_CELLS,
  JBW_SCENT_RADIUS,
  JBW_WALL,
} from '@sim/envs/jellybeanworld';

export const FIELD_TEXELS = 1024;
export const FIELD_COVER = FIELD_TEXELS / 2;
export const FIELD_EDGE = 0.94;
export const TILE_CELLS = 64;
export const FIELD_TILES = FIELD_TEXELS / TILE_CELLS;
export const TILE_BYTES = TILE_CELLS * TILE_CELLS * 4;

export const PRECISION_STEPS = 512;
export const SCENT_PEAK = Math.max(...JBW_SCENT.map((s) => Math.max(s[0], s[1], s[2])));

const CELL_KEY_BIAS = 1 << 22;
const CELL_KEY_SPAN = 1 << 23;

export function cellKey(x: number, y: number): number {
  return (x + CELL_KEY_BIAS) * CELL_KEY_SPAN + (y + CELL_KEY_BIAS);
}

export function buildScentKernel(): Float32Array {
  const radius = JBW_SCENT_RADIUS;
  const cells = JBW_SCENT_CELLS;
  const lifetime = JBW_DESTRUCTION_LIFETIME;
  const decay = 0.4;
  const diffusion = 0.14;
  const response = new Float32Array(cells * lifetime);
  response[0] = 1;
  for (let t = 1; t < lifetime; t++) {
    const prev = (t - 1) * cells;
    const curr = t * cells;
    for (let i = 0; i < cells; i++) response[curr + i] = decay * (response[prev + i] ?? 0);
    response[curr] = (response[curr] ?? 0) + 1;
    response[curr + cells - 1] =
      (response[curr + cells - 1] ?? 0) + 2 * diffusion * (response[prev + cells - 2] ?? 0);
    for (let y = 0; y < radius - 1; y++) {
      const at = curr + cells - radius + y;
      response[at] =
        (response[at] ?? 0) +
        diffusion *
          ((response[prev + jbwScentIndex(radius - 2, y)] ?? 0) +
            (response[prev + jbwScentIndex(radius - 1, y + 1)] ?? 0) +
            (response[prev + jbwScentIndex(radius - 1, y - 1)] ?? 0));
    }
    for (let x = 0; x < radius - 1; x++) {
      for (let y = 0; y <= x; y++) {
        const at = curr + jbwScentIndex(x, y);
        response[at] =
          (response[at] ?? 0) +
          diffusion *
            ((response[prev + jbwScentIndex(x + 1, y)] ?? 0) +
              (response[prev + jbwScentIndex(x - 1, y)] ?? 0) +
              (response[prev + jbwScentIndex(x, y + 1)] ?? 0) +
              (response[prev + jbwScentIndex(x, y - 1)] ?? 0));
      }
    }
  }
  return response;
}

export interface KernelBundle {
  response: Float32Array;
  steady: number;
  knee: number;
  reach: number;
  settleAge: number;
}

export function bundleKernel(response: Float32Array): KernelBundle {
  const steady = (JBW_DESTRUCTION_LIFETIME - 1) * JBW_SCENT_CELLS;
  const knee = Math.max(1e-6, response[steady] ?? 1);
  const floor = knee / PRECISION_STEPS;
  let reach = JBW_SCENT_RADIUS;
  for (let r = 1; r < JBW_SCENT_RADIUS; r++) {
    if ((response[steady + jbwScentIndex(r, 0)] ?? 0) < floor) {
      reach = r;
      break;
    }
  }
  let settle = JBW_DESTRUCTION_LIFETIME;
  for (let t = 0; t < JBW_DESTRUCTION_LIFETIME; t++) {
    let residual = 0;
    const at = t * JBW_SCENT_CELLS;
    for (let i = 0; i < JBW_SCENT_CELLS; i++) {
      const gap = (response[steady + i] ?? 0) - (response[at + i] ?? 0);
      if (gap > residual) residual = gap;
    }
    if (residual < floor) {
      settle = t;
      break;
    }
  }
  return { response, steady, knee, reach, settleAge: settle };
}

export interface FieldLog {
  count: number;
  steps: Int32Array;
  xs: Int32Array;
  ys: Int32Array;
}

export const EMPTY_FIELD_LOG: FieldLog = {
  count: 0,
  steps: new Int32Array(0),
  xs: new Int32Array(0),
  ys: new Int32Array(0),
};

export interface TilePalette {
  scentA: readonly [number, number, number];
  scentB: readonly [number, number, number];
  scentC: readonly [number, number, number];
  wall: readonly [number, number, number];
}

function firstSlotCollides(block: readonly number[], bx: number, by: number, slot: number): boolean {
  const at = jbwCandidate(block, bx, by, slot);
  for (let prior = 0; prior < slot; prior++) {
    const p = jbwCandidate(block, bx, by, prior);
    if (p[0] === at[0] && p[1] === at[1]) return true;
  }
  return false;
}

export function renderTile(
  seed: number,
  tileX: number,
  tileY: number,
  playhead: number,
  collected: ReadonlyMap<number, number>,
  kernel: KernelBundle,
  palette: TilePalette,
  out: Uint8Array,
): Uint8Array {
  const wx0 = tileX * TILE_CELLS;
  const wy0 = tileY * TILE_CELLS;
  const accum = new Float32Array(TILE_CELLS * TILE_CELLS * 3);
  const marks = new Float32Array(TILE_CELLS * TILE_CELLS * 4);
  const reach = kernel.reach;
  const firstBX = Math.floor((wx0 - reach) / JBW_BLOCK);
  const lastBX = Math.floor((wx0 + TILE_CELLS - 1 + reach) / JBW_BLOCK);
  const firstBY = Math.floor((wy0 - reach) / JBW_BLOCK);
  const lastBY = Math.floor((wy0 + TILE_CELLS - 1 + reach) / JBW_BLOCK);

  for (let by = firstBY; by <= lastBY; by++) {
    for (let bx = firstBX; bx <= lastBX; bx++) {
      const block = jbwBlockField(seed, bx, by);
      const line = jbwWallLine(seed, bx, by);
      let ecology: readonly number[] | null = null;
      for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
        if (firstSlotCollides(block, bx, by, slot)) continue;
        const at = jbwCandidate(block, bx, by, slot);
        if (line && (line[0] ? at[1] === line[1] : at[0] === line[1])) continue;
        ecology ??= jbwField(seed, Math.floor(bx / 4), Math.floor(by / 4), 64);
        const item = jbwTypeOf(block, ecology, slot);
        const scent = JBW_SCENT[item];
        if (!scent || (scent[0] === 0 && scent[1] === 0 && scent[2] === 0)) continue;
        const removed = collected.get(cellKey(at[0], at[1]));
        const age = removed !== undefined && removed <= playhead ? playhead - removed : -1;
        if (age >= JBW_DESTRUCTION_LIFETIME) continue;
        depositFine(accum, at[0] - wx0, at[1] - wy0, scent, age, kernel);
      }
      if (line) {
        for (let i = 0; i < JBW_BLOCK; i++) {
          const wx = line[0] ? bx * JBW_BLOCK + i : line[1];
          const wy = line[0] ? line[1] : by * JBW_BLOCK + i;
          const tx = wx - wx0;
          const ty = wy - wy0;
          if (tx < 0 || tx >= TILE_CELLS || ty < 0 || ty >= TILE_CELLS) continue;
          const cell = (ty * TILE_CELLS + tx) * 4;
          marks[cell] = palette.wall[0];
          marks[cell + 1] = palette.wall[1];
          marks[cell + 2] = palette.wall[2];
          marks[cell + 3] = 1;
        }
      }
    }
  }

  const norm = 1 / Math.log1p(SCENT_PEAK);
  const invKnee = 1 / kernel.knee;
  for (let i = 0, n = TILE_CELLS * TILE_CELLS; i < n; i++) {
    const a3 = i * 3;
    const a4 = i * 4;
    const l0 = Math.min(1, Math.log1p((accum[a3] ?? 0) * invKnee) * norm);
    const l1 = Math.min(1, Math.log1p((accum[a3 + 1] ?? 0) * invKnee) * norm);
    const l2 = Math.min(1, Math.log1p((accum[a3 + 2] ?? 0) * invKnee) * norm);
    const sum = l0 + l1 + l2;
    let r = 0;
    let g = 0;
    let b = 0;
    let mass = 0;
    if (sum > 0) {
      const inv = 1 / sum;
      r = (palette.scentA[0] * l0 + palette.scentB[0] * l1 + palette.scentC[0] * l2) * inv;
      g = (palette.scentA[1] * l0 + palette.scentB[1] * l1 + palette.scentC[1] * l2) * inv;
      b = (palette.scentA[2] * l0 + palette.scentB[2] * l1 + palette.scentC[2] * l2) * inv;
      mass = 1 - (1 - l0) * (1 - l1) * (1 - l2);
    }
    const mark = marks[a4 + 3] ?? 0;
    if (mark > 0) {
      const keep = 1 - mark;
      r = r * keep + (marks[a4] ?? 0) * mark;
      g = g * keep + (marks[a4 + 1] ?? 0) * mark;
      b = b * keep + (marks[a4 + 2] ?? 0) * mark;
      mass = Math.max(mass, mark);
    }
    out[a4] = Math.min(255, Math.max(0, r * 255)) | 0;
    out[a4 + 1] = Math.min(255, Math.max(0, g * 255)) | 0;
    out[a4 + 2] = Math.min(255, Math.max(0, b * 255)) | 0;
    out[a4 + 3] = Math.min(255, Math.max(0, mass * 255)) | 0;
  }
  return out;
}

function depositFine(
  accum: Float32Array,
  tx: number,
  ty: number,
  scent: readonly [number, number, number],
  age: number,
  kernel: KernelBundle,
): void {
  const reach = kernel.reach;
  const decayed = age >= 0 ? age * JBW_SCENT_CELLS : -1;
  const x0 = Math.max(0, tx - reach);
  const x1 = Math.min(TILE_CELLS - 1, tx + reach);
  const y0 = Math.max(0, ty - reach);
  const y1 = Math.min(TILE_CELLS - 1, ty + reach);
  for (let y = y0; y <= y1; y++) {
    const dy = y - ty;
    const row = y * TILE_CELLS;
    for (let x = x0; x <= x1; x++) {
      const idx = jbwScentIndex(x - tx, dy);
      let value = kernel.response[kernel.steady + idx] ?? 0;
      if (decayed >= 0) value -= kernel.response[decayed + idx] ?? 0;
      if (value <= 0) continue;
      const at = (row + x) * 3;
      accum[at] = (accum[at] ?? 0) + scent[0] * value;
      accum[at + 1] = (accum[at + 1] ?? 0) + scent[1] * value;
      accum[at + 2] = (accum[at + 2] ?? 0) + scent[2] * value;
    }
  }
}

export function scentAtCell(
  seed: number,
  collected: ReadonlyMap<number, number>,
  playhead: number,
  x: number,
  y: number,
  kernel: KernelBundle,
): [number, number, number] {
  const reach = kernel.reach;
  const out: [number, number, number] = [0, 0, 0];
  const firstBX = Math.floor((x - reach) / JBW_BLOCK);
  const lastBX = Math.floor((x + reach) / JBW_BLOCK);
  const firstBY = Math.floor((y - reach) / JBW_BLOCK);
  const lastBY = Math.floor((y + reach) / JBW_BLOCK);
  for (let by = firstBY; by <= lastBY; by++) {
    for (let bx = firstBX; bx <= lastBX; bx++) {
      const block = jbwBlockField(seed, bx, by);
      for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
        if (firstSlotCollides(block, bx, by, slot)) continue;
        const at = jbwCandidate(block, bx, by, slot);
        const dx = at[0] - x;
        const dy = at[1] - y;
        if (Math.abs(dx) > reach || Math.abs(dy) > reach) continue;
        const item = jbwItemAt(seed, at[0], at[1]);
        if (item === JBW_EMPTY || item === JBW_WALL) continue;
        const scent = JBW_SCENT[item];
        if (!scent) continue;
        const removed = collected.get(cellKey(at[0], at[1]));
        const age = removed !== undefined && removed <= playhead ? playhead - removed : -1;
        if (age >= JBW_DESTRUCTION_LIFETIME) continue;
        const idx = jbwScentIndex(dx, dy);
        let value = kernel.response[kernel.steady + idx] ?? 0;
        if (age >= 0) value -= kernel.response[age * JBW_SCENT_CELLS + idx] ?? 0;
        if (value <= 0) continue;
        out[0] += scent[0] * value;
        out[1] += scent[1] * value;
        out[2] += scent[2] * value;
      }
    }
  }
  return out;
}
