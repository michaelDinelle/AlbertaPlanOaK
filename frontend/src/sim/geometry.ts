import type { EnvEntry } from './catalog';

export const GEOMETRY_WORDS = 19;
export const GEOMETRY_BYTES = GEOMETRY_WORDS * 4;

export interface GridCell {
  x: number;
  y: number;
}

export interface LaneGeometry {
  w: number;
  h: number;
  walls: Uint8Array;
  actionMask: Uint8Array;
}

export class GeometryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GeometryError';
  }
}

function unpack(dv: DataView, wordOffset: number, words: number, cells: number): Uint8Array {
  const out = new Uint8Array(cells);
  for (let i = 0; i < cells; i++) {
    const word = dv.getUint32(wordOffset + (i >> 5) * 4, true);
    out[i] = (word >>> (i & 31)) & 1;
  }
  void words;
  return out;
}

export function readLaneGeometry(dv: DataView, byteOffset: number, entry: EnvEntry): LaneGeometry {
  const w = dv.getUint32(byteOffset, true);
  const h = dv.getUint32(byteOffset + 4, true);
  const maskWords = dv.getUint32(byteOffset + 8, true);
  if (w !== entry.gridW || h !== entry.gridH) {
    throw new GeometryError(
      `${entry.envId} lane geometry is ${w}x${h} but the catalog declares ${entry.gridW}x${entry.gridH}`,
    );
  }
  if (maskWords * 32 < w * h) {
    throw new GeometryError(
      `${entry.envId} lane geometry carries ${maskWords} mask words for ${w * h} cells`,
    );
  }
  const cells = w * h;
  const wallsAt = byteOffset + 12;
  const maskAt = wallsAt + maskWords * 4;
  return {
    w,
    h,
    walls: unpack(dv, wallsAt, maskWords, cells),
    actionMask: unpack(dv, maskAt, maskWords, cells),
  };
}

export function isWall(geometry: LaneGeometry, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= geometry.w || y >= geometry.h) return true;
  return geometry.walls[y * geometry.w + x] === 1;
}
