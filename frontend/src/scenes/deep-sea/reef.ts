
import { CELL, cellToWorld, type GridSize, type Vec3Tuple } from '@kit/grid';
import { TOKENS, type Palette } from '@kit/tokens';
import type { FrameState } from '@kit/types';
import * as THREE from 'three';
import {
  DEEPSEA_N,
  deepSeaDecoder,
  deepSeaNextCol,
  type DeepSeaStatic,
} from '@sim/envs/deepsea';
import type { EnvGeometry, GridGeometry } from '@sim/types';


export {
  TERRACE_DESCENT,
  rowDrop,
  terraceDepth as reefDepthY,
  terraceDepthAt as reefDepthYAt,
  cellToTerraceWorld as cellToReefWorld,
  cellToTerraceVec as cellToReefVec,
  terraceShear as reefShear,
  terraceShelves,
} from '@kit/terrace';

export const SHELF_THICKNESS = 1.35;

export const TREASURE_ROW = DEEPSEA_N - 1;
export const TREASURE_COL = DEEPSEA_N - 1;

export function resolveGrid(geometry: EnvGeometry, seed = 0): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === DEEPSEA_N && g.h === DEEPSEA_N) return g;
  const built = deepSeaDecoder.geometry(deepSeaDecoder.initStatic(seed), seed).grid;
  if (built) return built;
  return {
    w: DEEPSEA_N,
    h: DEEPSEA_N,
    walls: new Uint8Array(DEEPSEA_N * DEEPSEA_N),
    goal: { x: TREASURE_COL, y: TREASURE_ROW },
    actionMask: new Uint8Array(DEEPSEA_N * DEEPSEA_N),
  };
}

export function reefReachable(grid: GridGeometry): Uint8Array {
  const w = grid.w;
  const h = grid.h;
  const mask =
    grid.actionMask && grid.actionMask.length >= w * h ? grid.actionMask : new Uint8Array(w * h);
  const st: DeepSeaStatic = { seed: 0, maskLo: 0, maskHi: 0, mask };
  const reach = new Uint8Array(w * h);
  reach[0] = 1;
  for (let r = 0; r + 1 < h; r++) {
    for (let c = 0; c < w; c++) {
      if ((reach[r * w + c] ?? 0) === 0) continue;
      for (let action = 0; action < 2; action++) {
        reach[(r + 1) * w + deepSeaNextCol(st, r, c, action)] = 1;
      }
    }
  }
  return reach;
}

export function isReef(reach: Uint8Array, grid: GridSize, c: number, r: number): boolean {
  if (c < 0 || r < 0 || c >= grid.w || r >= grid.h) return false;
  return (reach[r * grid.w + c] ?? 0) !== 0;
}

export function isReefLip(reach: Uint8Array, grid: GridSize, c: number, r: number): boolean {
  return isReef(reach, grid, c, r) && !isReef(reach, grid, c + 1, r);
}

export function currentDir(grid: GridGeometry, c: number, r: number): number {
  const mask = grid.actionMask;
  if (!mask) return 1;
  return (mask[r * grid.w + c] ?? 0) !== 0 ? 1 : -1;
}

export function rowAtZ(z: number, grid: GridSize): number {
  return Math.round(z / CELL + (grid.h - 1) / 2);
}

export function colAtX(x: number, grid: GridSize): number {
  return Math.round(x / CELL + (grid.w - 1) / 2);
}


export interface DiveCell {
  r: number;
  c: number;
  lastA: number;
}

interface DeepSeaLike {
  r?: unknown;
  c?: unknown;
  lastA?: unknown;
}

export function readDive(state: FrameState, grid: GridSize, out: DiveCell): DiveCell {
  const d = state.derived as DeepSeaLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const r = has && typeof d.r === 'number' ? d.r : (state.pose[0] ?? 0);
  const c = has && typeof d.c === 'number' ? d.c : (state.pose[1] ?? 0);
  const rawA = has && typeof d.lastA === 'number' ? d.lastA : Math.trunc(state.ctrl[0] ?? 0);
  out.r = clampInt(Number.isFinite(r) ? Math.round(r) : 0, grid.h - 1);
  out.c = clampInt(Number.isFinite(c) ? Math.round(c) : 0, grid.w - 1);
  out.lastA = (Number.isFinite(rawA) ? Math.trunc(rawA) : 0) & 1;
  return out;
}

function clampInt(v: number, hi: number): number {
  return v < 0 ? 0 : v > hi ? hi : v;
}

export function isDiveStep(prevC: number, prevR: number, next: DiveCell): boolean {
  return next.r === prevR + 1 && Math.abs(next.c - prevC) <= 1;
}


export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ReefSkin {
  shelfShallow: THREE.Color;
  shelfDeep: THREE.Color;
  shelfLip: THREE.Color;
  shelfTreasure: THREE.Color;
  kelp: THREE.Color;
  coral: THREE.Color;
  anemone: THREE.Color;
  shell: THREE.Color;
  shellRib: THREE.Color;
  mantle: THREE.Color;
  pearl: THREE.Color;
  silt: THREE.Color;
  caustic: THREE.Color;
  godRay: THREE.Color;
  bubble: THREE.Color;
}

export function reefSkin(palette: Palette, tokens: typeof TOKENS): ReefSkin {
  const accent = tokens.accents['deep-sea'] ?? tokens.uiCyan;
  const green = tokens.accents['memory-chain'] ?? palette.groundB;
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const dawnRose = tokens.accents['acrobot'] ?? blossom;
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const abyss = tokens.palettes.abyss;

  const shelfShallow = blendTokens(blendTokens(palette.groundB, accent, 0.78), palette.sun, 0.22);
  const shelfDeep = shade(blendTokens(palette.groundA, abyss.groundA, 0.72), 0.18);
  return {
    shelfShallow,
    shelfDeep,
    shelfLip: blendTokens(shelfShallow, palette.sun, 0.4),
    shelfTreasure: blendTokens(shelfDeep, gold, 0.26),
    kelp: shade(blendTokens(palette.groundB, green, 0.46), 0.1),
    coral: shade(blendTokens(blossom, palette.groundB, 0.4), 0.24),
    anemone: shade(blendTokens(dawnRose, palette.sun, 0.22), 0.18),
    shell: shade(blendTokens(palette.hemiSky, palette.sun, 0.2), 0.56),
    shellRib: shade(blendTokens(palette.sun, gold, 0.7), 0.44),
    mantle: shade(blendTokens(accent, green, 0.3), 0.3),
    pearl: blendTokens(palette.sun, tokens.celebrationGold, 0.24),
    silt: blendTokens(palette.hemiSky, palette.sun, 0.45),
    caustic: blendTokens(palette.sun, accent, 0.3),
    godRay: blendTokens(palette.sun, palette.skyTop, 0.3),
    bubble: blendTokens(accent, palette.sun, 0.35),
  };
}

export function shelfTint(skin: ReefSkin, grid: GridSize, c: number, r: number, out: THREE.Color): THREE.Color {
  const depth = grid.h > 1 ? clamp01(r / (grid.h - 1)) : 0;
  out.copy(skin.shelfShallow).lerp(skin.shelfDeep, depth);
  if (r === TREASURE_ROW && c === TREASURE_COL) return out.lerp(skin.shelfTreasure, 0.75);
  return out;
}


export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

export function mergeColoredParts(parts: readonly ColoredPart[], name: string): THREE.BufferGeometry {
  let total = 0;
  for (const p of parts) {
    total += p.geometry.index ? p.geometry.index.count : p.geometry.getAttribute('position').count;
  }
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const c = new THREE.Color();
  let o = 0;
  for (const part of parts) {
    const geo = part.geometry;
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.index;
    const n = index ? index.count : pos.count;
    c.set(part.color);
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      positions[o * 3] = pos.getX(vi);
      positions[o * 3 + 1] = pos.getY(vi);
      positions[o * 3 + 2] = pos.getZ(vi);
      if (nor) {
        normals[o * 3] = nor.getX(vi);
        normals[o * 3 + 1] = nor.getY(vi);
        normals[o * 3 + 2] = nor.getZ(vi);
      }
      colors[o * 3] = c.r;
      colors[o * 3 + 1] = c.g;
      colors[o * 3 + 2] = c.b;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeBoundingSphere();
  g.name = name;
  for (const p of parts) p.geometry.dispose();
  return g;
}

export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}
