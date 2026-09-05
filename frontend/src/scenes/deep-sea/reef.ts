import { CELL, type GridSize } from '@kit/grid';
import { blendTokens, clamp01, shade } from '@kit/mix';
import { TOKENS, type Palette } from '@kit/tokens';
import type { FrameState } from '@kit/types';
import * as THREE from 'three';
import { DEEPSEA_ACTION, DEEPSEA_COL, DEEPSEA_N, DEEPSEA_ROW, deepSeaNextCol, type DeepSeaStatic } from '@sim/envs/deepsea';
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

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== DEEPSEA_N || grid.h !== DEEPSEA_N) throw new Error('Deep Sea grid is missing or invalid');
  return grid;
}

export function reefReachable(grid: GridGeometry): Uint8Array {
  const w = grid.w;
  const h = grid.h;
  const mask = grid.actionMask;
  if (!mask || mask.length !== w * h) throw new Error('Deep Sea action mask is missing or invalid');
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

export function readDive(state: FrameState, grid: GridSize, out: DiveCell): DiveCell {
  out.r = clampInt(Math.round(state.pose[DEEPSEA_ROW] ?? 0), grid.h - 1);
  out.c = clampInt(Math.round(state.pose[DEEPSEA_COL] ?? 0), grid.w - 1);
  out.lastA = Math.trunc(state.pose[DEEPSEA_ACTION] ?? 0) & 1;
  return out;
}

function clampInt(v: number, hi: number): number {
  return v < 0 ? 0 : v > hi ? hi : v;
}

export function isDiveStep(prevC: number, prevR: number, next: DiveCell): boolean {
  return next.r === prevR + 1 && Math.abs(next.c - prevC) <= 1;
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
