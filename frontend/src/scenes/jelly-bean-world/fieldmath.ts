import { toHalfFloat } from 'three/src/extras/DataUtils.js';
import {
  jbwBlockField,
  jbwCandidate,
  jbwField,
  jbwScentIndex,
  jbwTypeOf,
  jbwWallLine,
  JBW_BLOCK,
  JBW_BLOCK_ITEMS,
  JBW_DESTRUCTION_LIFETIME,
  JBW_SCENT,
  JBW_SCENT_CELLS,
  JBW_SCENT_RADIUS,
  JBW_SCENT_DECAY,
  JBW_SCENT_DIFFUSION,
  JBW_WALL,
} from '@sim/envs/jellybeanworld';

export const FIELD_TEXELS = 1024;
export const TILE_CELLS = 64;
export const FIELD_TILES = FIELD_TEXELS / TILE_CELLS;
export const TILE_VALUES = TILE_CELLS * TILE_CELLS * 4;


const CELL_KEY_BIAS = 1 << 22;
const CELL_KEY_SPAN = 1 << 23;

export function cellKey(x: number, y: number): number {
  return (x + CELL_KEY_BIAS) * CELL_KEY_SPAN + (y + CELL_KEY_BIAS);
}

export function buildScentKernel(): Float32Array {
  const radius = JBW_SCENT_RADIUS;
  const cells = JBW_SCENT_CELLS;
  const lifetime = JBW_DESTRUCTION_LIFETIME;
  const decay = JBW_SCENT_DECAY;
  const diffusion = JBW_SCENT_DIFFUSION;
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
  const floor = knee / 65536;
  const reach = JBW_SCENT_RADIUS - 1;
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

export function visitBlock(seed: number, bx: number, by: number, emit: (x: number, y: number, item: number) => void, walls = true): void {
  const block = jbwBlockField(seed, bx, by), line = jbwWallLine(seed, bx, by);
  if (walls && line) for (let i = 0; i < JBW_BLOCK; i++) emit(line[0] ? bx * JBW_BLOCK + i : line[1], line[0] ? line[1] : by * JBW_BLOCK + i, JBW_WALL);
  const ecology = jbwField(seed, Math.floor(bx / 4), Math.floor(by / 4), 64);
  for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
    const [x, y] = jbwCandidate(block, bx, by, slot);
    if (line && (line[0] ? y === line[1] : x === line[1])) continue;
    let duplicate = false;
    for (let i = 0; i < slot; i++) {
      const prior = jbwCandidate(block, bx, by, i);
      if (prior[0] === x && prior[1] === y) duplicate = true;
    }
    if (!duplicate) emit(x, y, jbwTypeOf(block, ecology, slot));
  }
}

function sources(seed: number, x0: number, y0: number, x1: number, y1: number, playhead: number, collected: ReadonlyMap<number, number>, kernel: KernelBundle, emit: (x: number, y: number, scent: readonly number[], offset: number) => void): void {
  const reach = kernel.reach;
  for (let by = Math.floor((y0 - reach) / JBW_BLOCK); by <= Math.floor((y1 + reach) / JBW_BLOCK); by++) {
    for (let bx = Math.floor((x0 - reach) / JBW_BLOCK); bx <= Math.floor((x1 + reach) / JBW_BLOCK); bx++) visitBlock(seed, bx, by, (x, y, item) => {
      if (x < x0 - reach || x > x1 + reach || y < y0 - reach || y > y1 + reach) return;
      const removed = collected.get(cellKey(x, y));
      const age = removed !== undefined && removed <= playhead ? playhead - removed : -1;
      if (age < JBW_DESTRUCTION_LIFETIME) emit(x, y, JBW_SCENT[item]!, age < 0 ? -1 : age * JBW_SCENT_CELLS);
    }, false);
  }
}

function concentration(kernel: KernelBundle, dx: number, dy: number, offset: number): number {
  const index = jbwScentIndex(dx, dy);
  return kernel.response[kernel.steady + index]! - (offset < 0 ? 0 : kernel.response[offset + index]!);
}

export function renderTile(seed: number, tileX: number, tileY: number, playhead: number, collected: ReadonlyMap<number, number>, kernel: KernelBundle): Uint16Array<ArrayBuffer> {
  const x0 = tileX * TILE_CELLS, y0 = tileY * TILE_CELLS, reach = kernel.reach;
  const accum = new Float32Array(TILE_CELLS * TILE_CELLS * 3), out = new Uint16Array(TILE_VALUES);
  sources(seed, x0, y0, x0 + TILE_CELLS - 1, y0 + TILE_CELLS - 1, playhead, collected, kernel, (x, y, scent, offset) => {
    for (let ty = Math.max(y0, y - reach); ty <= Math.min(y0 + TILE_CELLS - 1, y + reach); ty++) {
      for (let tx = Math.max(x0, x - reach); tx <= Math.min(x0 + TILE_CELLS - 1, x + reach); tx++) {
        const value = concentration(kernel, tx - x, ty - y, offset), at = ((ty - y0) * TILE_CELLS + tx - x0) * 3;
        for (let c = 0; c < 3; c++) accum[at + c] = accum[at + c]! + scent[c]! * value;
      }
    }
  });
  for (let i = 0; i < TILE_CELLS * TILE_CELLS; i++) {
    for (let c = 0; c < 3; c++) out[i * 4 + c] = toHalfFloat(accum[i * 3 + c]!);
    out[i * 4 + 3] = toHalfFloat(1);
  }
  return out;
}

export function scentAtCell(seed: number, collected: ReadonlyMap<number, number>, playhead: number, x: number, y: number, kernel: KernelBundle): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  sources(seed, x, y, x, y, playhead, collected, kernel, (sx, sy, scent, offset) => {
    const value = concentration(kernel, sx - x, sy - y, offset);
    for (let c = 0; c < 3; c++) out[c] = out[c]! + scent[c]! * value;
  });
  return out;
}
