// Sunlit Descent — the reef's coordinate system, its reachable set and its palette.
//
// THE ONE IDEA OF THIS SCENE: DeepSea's state is `(r, c)` where **r is DEPTH**, and the
// art says so with the y axis. Every step of a dive increments r, so the reef is a
// STAIRCASE: row r's shelf top sits at exactly `reefDepthY(r) = -ROW_DROP · r`, and the
// diver rides down it. `pose = [r, c, 7, 7]` — r FIRST — so `cellToWorld(c, r)` is the
// only correct call. (The retired 2-D viewer transposed this and drew the dive sideways;
// every conversion in this scene goes through `cellToReefWorld` so it cannot recur.)
//
// THE REACHABLE TRIANGLE IS NOT TYPED IN. `reefReachable()` runs the DECODER's own
// transition (`deepSeaNextCol`, plus `r += 1` every step) forward from the env's
// deterministic init, so the reef is exactly the set of cells a dive can occupy. It comes
// out as `c ≤ r` for every mask — which is a RESULT here, not an assumption — and the
// upper-right of the board stays open blue water.
//
// THE DEPTH MAP IS AFFINE, WHICH IS WHY THE OVERLAY STILL LINES UP. `reefDepthY(row)` is
// linear in the row index, and the row index is linear in world z, so "sink row r by
// ROW_DROP·r" is a shear of the flat grid plane. `reefShear()` hands that exact shear to
// the kit `GroundTruthOverlay`'s group: the overlay is still drawn purely from
// `EnvGeometry` + `FrameState` in grid space, and the scene's own grid→world map carries
// it down the staircase. Every cell-centred marker (agent cross, goal diamond, mask
// arrows) lands on its terrace to the last bit; the lattice becomes the staircase's
// wireframe.

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

// ---------------------------------------------------------------------------
// the depth axis
// ---------------------------------------------------------------------------

/**
 * World-y drop per row — the riser of one terrace.
 *
 * THE GEOMETRY THAT DECIDES THIS NUMBER, AND THE CAMERA WITH IT. Looking DOWN a receding
 * staircase from above, the recession lifts each next step by `spacing·sin(elev)` on
 * screen while the drop lowers it by `drop·cos(elev)` — and the condition for a shelf top
 * to be visible over the shelf in front of it is exactly `tan(elev) > drop/spacing`, i.e.
 * exactly the condition for those two terms to CANCEL. A receding dive therefore renders
 * as a flat slab or as a wall of risers, and nothing in between; the first three passes
 * of this scene rediscovered that the hard way.
 *
 * Turn the frame around and the two terms ADD: with the deep rows NEAREST, each step is
 * both lower and closer, so the staircase marches down the screen and every top stays in
 * view. That is the composed frame (see index.ts), and it is what lets the drop be a real
 * 0.5 per row — a 3.5-unit descent across an 8-unit board — instead of a hint.
 */
export const ROW_DROP = 0.5;

/**
 * Shelf slab thickness. It must EXCEED `ROW_DROP` or daylight shows between the terraces —
 * but the first pass at 0.58 (just enough to close the gaps) left the reef reading as
 * eight slabs FLOATING in open water. A shelf hanging 1.35 below its own top buries every
 * row behind it in rock, so the reef reads as one mass cut into steps and the open-water
 * side gets a real cliff face.
 */
export const SHELF_THICKNESS = 1.35;

/** The shelf top of a dive row. Row 0 is the surface; y decreases as r increases. */
export function reefDepthY(row: number): number {
  return -ROW_DROP * row;
}

/**
 * The same map read off a world z — what a wisp gliding BETWEEN two rows needs. At a cell
 * centre `z / CELL + (h − 1) / 2` is exactly the integer row, so this is bit-identical to
 * `reefDepthY(row)` on every landed frame (the alignment probes assert 0 error, not 1e-3).
 */
export function reefDepthYAt(z: number, grid: GridSize): number {
  return reefDepthY(z / CELL + (grid.h - 1) / 2);
}

/** Cell (column c, row r) → world [x, reefDepthY(r), z]. THE conversion for this scene. */
export function cellToReefWorld(c: number, r: number, grid: GridSize, out?: Vec3Tuple): Vec3Tuple {
  const t = cellToWorld(c, r, grid, out);
  t[1] = reefDepthY(r);
  return t;
}

/** `cellToReefWorld` straight into a Vector3 (allocation-free with a preallocated out). */
export function cellToReefVec(c: number, r: number, grid: GridSize, out: THREE.Vector3): THREE.Vector3 {
  const w = cellToReefWorld(c, r, grid, SCRATCH);
  return out.set(w[0], w[1], w[2]);
}

const SCRATCH: Vec3Tuple = [0, 0, 0];

/**
 * The scene's grid→world map as a matrix: `(x, y, z) → (x, y + reefDepthYAt(z), z)`.
 * A pure shear plus a lift — the exact transform that carries a FLAT grid-space drawing
 * (the ground-truth overlay) down the staircase without touching x or z.
 */
export function reefShear(grid: GridSize, out?: THREE.Matrix4): THREE.Matrix4 {
  const m = out ?? new THREE.Matrix4();
  const k = ROW_DROP / CELL;
  const lift = -ROW_DROP * ((grid.h - 1) / 2);
  return m.set(
    1, 0, 0, 0,
    0, 1, -k, lift,
    0, 0, 1, 0,
    0, 0, 0, 1,
  );
}

// ---------------------------------------------------------------------------
// the env's own truth
// ---------------------------------------------------------------------------

/** The treasure row: one step past the deepest occupied row, where the clam sits. */
export const TREASURE_ROW = DEEPSEA_N - 1;
/** The treasure column. */
export const TREASURE_COL = DEEPSEA_N - 1;

/**
 * The grid the scene draws. The host always hands us the decoder's geometry; the fallback
 * is the DECODER'S OWN `geometry()`, never a scene re-derivation, so a placeholder
 * geometry degrades to a correct 8×8 board with a real per-seed action mask.
 */
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

/**
 * The cells a dive can ever stand on, computed by running the DECODER's own transition
 * (`deepSeaNextCol` for both actions, `r += 1` every step) forward from the env's
 * deterministic init at (0, 0). No triangle is typed in anywhere: this is why the reef is
 * exactly the reachable set and the rest of the board is open water.
 *
 * Row `h − 1` is included: the diver never OCCUPIES it (leaving row `h − 2` wraps in the
 * same step), but it is the column the treasure is claimed in and the shelf the clam
 * stands on, so the reef is drawn one row past the dive.
 */
export function reefReachable(grid: GridGeometry): Uint8Array {
  const w = grid.w;
  const h = grid.h;
  const mask =
    grid.actionMask && grid.actionMask.length >= w * h ? grid.actionMask : new Uint8Array(w * h);
  // deepSeaNextCol reads only `mask`; the word pair is the decoder's own provenance record
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

/** The sunlit lip of a shelf: a reef cell whose next column is open water. */
export function isReefLip(reach: Uint8Array, grid: GridSize, c: number, r: number): boolean {
  return isReef(reach, grid, c, r) && !isReef(reach, grid, c + 1, r);
}

/** Which way action 1 carries the diver at a cell: +1 = right (+x), −1 = left. */
export function currentDir(grid: GridGeometry, c: number, r: number): number {
  const mask = grid.actionMask;
  if (!mask) return 1;
  return (mask[r * grid.w + c] ?? 0) !== 0 ? 1 : -1;
}

/** World (x, z) → the reef row/column under it (rounded, unclamped). */
export function rowAtZ(z: number, grid: GridSize): number {
  return Math.round(z / CELL + (grid.h - 1) / 2);
}

export function colAtX(x: number, grid: GridSize): number {
  return Math.round(x / CELL + (grid.w - 1) / 2);
}

// ---------------------------------------------------------------------------
// the frame's truth
// ---------------------------------------------------------------------------

/** The decoded dive state a frame carries. `lastA` is the ACTION, not the direction. */
export interface DiveCell {
  r: number;
  c: number;
  /** `ctrl[0] & 1` — action 1 is the one that costs `-0.01/N` (the right-move cost) */
  lastA: number;
}

interface DeepSeaLike {
  r?: unknown;
  c?: unknown;
  lastA?: unknown;
}

/**
 * The dive cell for a frame. sim-core's decoded `DeepSeaState` is the source; the raw
 * pose (`[r, c, 7, 7, …]` — ROW FIRST) is the fallback so the scene still stands up under
 * the harness's fixture loop, which carries no `derived`.
 */
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

/**
 * Is `next` one legal step of the dive after `prev`? The wrap back to the surface is NOT
 * (r resets to 0), and neither is a scrub across the record — both are hard cuts, never a
 * glide across six terraces.
 */
export function isDiveStep(prevC: number, prevR: number, next: DiveCell): boolean {
  return next.r === prevR + 1 && Math.abs(next.c - prevC) <= 1;
}

// ---------------------------------------------------------------------------
// color (STYLE.md: every color is a token or a blend of tokens)
// ---------------------------------------------------------------------------

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

/** Take a token blend down into shade. `night.skyTop` is STYLE.md's legal near-black. */
export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ReefSkin {
  /** shelf tint at row 0 — the sunlit shallows */
  shelfShallow: THREE.Color;
  /** shelf tint at the deepest row */
  shelfDeep: THREE.Color;
  /** the sunlit edge of each shelf (the frontier column of the reachable set) */
  shelfLip: THREE.Color;
  /** the treasure shelf under the clam */
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

/**
 * The reef palette. Everything is the `underwater` palette crossed with the shared accent
 * table — reef teal (`deep-sea`) for the living rock, firefly green (`memory-chain`) for
 * the kelp, blossom pink (`chain`) and dawn blossom (`acrobot`) for the corals and
 * anemones, temple gold (`multi-key-corridor`) on the clam's ribs. Sixteen scenes, one
 * accent table: the gallery still reads as one preserve, and this one reads as a reef.
 */
export function reefSkin(palette: Palette, tokens: typeof TOKENS): ReefSkin {
  const accent = tokens.accents['deep-sea'] ?? tokens.uiCyan;
  const green = tokens.accents['memory-chain'] ?? palette.groundB;
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const dawnRose = tokens.accents['acrobot'] ?? blossom;
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const abyss = tokens.palettes.abyss;

  // The reef has to WIN against the water it sits in. The underwater palette's fog and
  // horizon are both mid-blue, so a shelf shaded the way a daylight ground would be
  // vanishes into the column — the first pass was ~10% apart in value from the background
  // and the whole triangle stopped reading. These push hard toward the reef-teal accent
  // and keep their saturation; the DEPTH is carried by the ramp between them, not by
  // draining both ends into grey.
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

/**
 * The shelf tint of a row: the descent is legible from the rock alone.
 *
 * The ramp is LINEAR in the row index, and that is the whole point. A smoothstep here —
 * the obvious first choice — flattens both ends of the curve, so rows 0–2 came out one
 * colour and rows 5–7 another and the terraces in between were the only ones that read.
 * Linear gives every one of the seven risers the SAME slice of the range, which is what
 * turns eight steps of tint into eight visible steps of rock.
 */
export function shelfTint(skin: ReefSkin, grid: GridSize, c: number, r: number, out: THREE.Color): THREE.Color {
  const depth = grid.h > 1 ? clamp01(r / (grid.h - 1)) : 0;
  out.copy(skin.shelfShallow).lerp(skin.shelfDeep, depth);
  if (r === TREASURE_ROW && c === TREASURE_COL) return out.lerp(skin.shelfTreasure, 0.75);
  return out;
}

// ---------------------------------------------------------------------------
// geometry plumbing
// ---------------------------------------------------------------------------

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources — the four-rooms prop pattern. A hand-built clam wants its source
 * normals plus a per-part shell palette baked in, so one mesh + one material draws it.
 */
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

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has real amplitude when the window
 * closes, so every flourish would end on a one-frame pop (canon lesson 6).
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}
