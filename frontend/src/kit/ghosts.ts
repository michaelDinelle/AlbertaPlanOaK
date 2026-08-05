// kit/ghosts — THE POPULATION LAYER. One hero seed is an anecdote; the fleet is the
// result (AGENTS.md: "The population is the deliverable — never a single seed"). This
// module renders EVERY seed of one env at once, around the one artisan scene that is
// already on screen, as instanced ghosts: the whole distribution in one frame, with the
// hero still readable inside it.
//
// THREE PROMISES, in the order they matter:
//
//  1. ACCURACY. A ghost is placed from `EnvGeometry` + its pose and nothing else — the
//     same ground truth the hero uses (`geometry.cheetah.fk`, `cellToWorld`,
//     `geometry.classic.heightAt`/`anchors`, the `@sim/envs` kinematics helpers). Cheetah
//     ghosts run at their TRUE rootx on the hero's own track (FK x -> world x, FK z ->
//     world y, z = 0), so the spread along the track IS the population's spread.
//     The ONE presentation-only liberty is documented at `ghostCellOffset` below.
//
//  2. SCALE. Every ghost body part is one instance of one `InstancedMesh` bank, so the
//     draw count is a function of the CATEGORY, never of the ghost count: <= 8 draws for
//     cheetah (7 bones + head), <= 3 for grid, <= 3 for classic — identical at 8 ghosts
//     and at 8192. Quality tiers scale segment counts (instance detail), never banks.
//     TRIANGLES are the other half of that sentence and they are NOT flat: instancing
//     trades draw calls for geometry, so the load is exactly `maxGhosts ×
//     GHOST_TRIANGLES_PER_GHOST` (published below, reported live by `triangles()`, split
//     out by the host as `ghostTriangles`). Size a population against that number — a
//     1024-seed hero cheetah wall is 1.22M triangles, several times a scene's own budget.
//
//  3. ZERO PER-FRAME ALLOCATION. Every scratch (`Matrix4`, `Vector3`, `Quaternion`,
//     `Color`, the FK out-param, the pose store) is preallocated in the constructor;
//     `update()`, `setGhostFrame()`, `setRate()`, `setEmphasis()` and `glint()` allocate
//     nothing. Instance buffers are created once and only ever written in place.
//
// LEGIBILITY follows canon lesson 22b (see `kit/wisp`): the emitter banks are ADDITIVE
// and therefore can only ADD light to whatever is behind them, so on a bright floor they
// wash out and past the bloom threshold they smear (lesson 19). Grid ghosts therefore
// carry the wisp's own treatment at ghost scale — a dark under-silhouette shell and a
// soft ground-contact shade, both the accent blended most of the way to
// `palettes.night.skyTop` (lesson 9's legal near-black), so in a dark palette they are
// already the floor's value and vanish. Background-agnostic by construction.
//
// BRIGHTNESS is a budget, not a knob. `setRate` drives a quantile ramp
// (`night.skyTop` -> accent -> `celebrationGold` at the very top) and `setEmphasis`
// scales it: 0 = faint (a background swarm), 1 = readable, 2 = the leader, the only
// level allowed to rim past the bloom threshold. The 300 ms `glint` is the one sanctioned
// excursion above it — the celebration grammar's own beat (canon lesson 6: every envelope
// multiplies exp-decay by (1 - t/T), so nothing ends on a pop).
//
// The glint puffs a ghost's THICKNESS and brightness, never its LENGTH or its position:
// a ghost never leaves its true pose, not even mid-celebration.
//
// OWNERSHIP: a GhostLayer is HOST machinery. Its `group` is named `oakGhostLayer` and the
// host attaches it to the scene AFTER `init()` (see `theater/sceneHost`). Scenes must
// never read, move, traverse into or dispose it — see the note in `kit/types.ts`.

import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import type { EnvGeometry } from '@sim/types';
import { CHEETAH_BONES, FK_FLOATS, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import { acrobotPoints } from '@sim/envs/acrobot';
import { cartPoleTip } from '@sim/envs/cartpole';
import { pendulumTip } from '@sim/envs/pendulum';
import { CELL } from './grid';
import { emissiveMaterial, type EmissiveMaterial } from './materials';
import { radialSprite } from './textures';
import type { DesignTokens } from './tokens';
import type { QualityTier } from './types';

// ---------------------------------------------------------------------------
// contract
// ---------------------------------------------------------------------------

/** The three body plans. Matches `EnvCategory` in shared/envs.ts. */
export type GhostCategory = 'cheetah' | 'grid' | 'classic';

/** 0 = faint background swarm, 1 = readable, 2 = the leader (may rim past bloom). */
export type GhostEmphasis = 0 | 1 | 2;

/**
 * Triangles ONE ghost costs, by body plan and quality tier — the budget the draw count
 * cannot express, published so a caller sizes `maxGhosts` against a number instead of
 * discovering it on a GPU.
 *
 * `maxGhosts × this` is the layer's whole load (see `GhostLayer.triangles()`, which is
 * what the host reports as `HostFrameInfo.ghostTriangles`). Worked out for the hero
 * cheetah plan, the most expensive one: 64 ghosts = 76k triangles, 1024 = 1.22M, 8192 =
 * 9.7M. A scene manifest budget is typically ~220k, so a 1024-seed cheetah wall spends
 * several scene budgets on the population alone — at the `thumb` tier (360/ghost) it is
 * 369k, which is why the tier ladder scales instance DETAIL rather than the bank count.
 *
 * Entries are per-category upper bounds: classic plans differ by env (a mountain-car
 * ghost is one marker, an acrobot ghost adds two links), and the table quotes the widest.
 * `test/kit-ghosts.test.ts` asserts every built layer against it, so it cannot drift.
 */
export const GHOST_TRIANGLES_PER_GHOST: Readonly<
  Record<GhostCategory, Readonly<Record<QualityTier, number>>>
> = {
  cheetah: { hero: 1188, medium: 668, thumb: 360 },
  grid: { hero: 262, medium: 102, thumb: 102 },
  classic: { hero: 204, medium: 104, thumb: 104 },
};

export interface GhostLayerOptions {
  category: GhostCategory;
  envId: EnvId;
  /** ground truth from sim-core — ghosts NEVER re-derive geometry */
  geometry: EnvGeometry;
  /** instance capacity; the draw count does not depend on it */
  maxGhosts: number;
  tokens: DesignTokens;
  /** env accent, normally `tokens.accents[slug]` */
  accent: THREE.ColorRepresentation;
  tier: QualityTier;
  /** group name override — the host relies on the default */
  name?: string;
}

/** One instanced bank: one draw call, `maxGhosts` instances. */
export interface GhostBank {
  /** stable, e.g. `bone3`, `head`, `core`, `shell`, `shade`, `marker`, `link0` */
  readonly name: string;
  readonly mesh: THREE.InstancedMesh;
  /**
   * `emitter` banks are additive and carry the rate ramp in `instanceColor`;
   * `shade` banks are the dark local background and carry emphasis in SCALE.
   */
  readonly role: 'emitter' | 'shade';
}

// ---------------------------------------------------------------------------
// constants (all derived, none tunable per scene)
// ---------------------------------------------------------------------------

/** The reserved group name. Scenes must not touch a group with this name. */
export const GHOST_GROUP_NAME = 'oakGhostLayer';

/** Glint span, seconds (STYLE.md celebration grammar: the 300 ms beat). */
export const GHOST_GLINT_SECONDS = 0.3;
/** Exponential time constant inside the glint envelope. */
const GLINT_TAU = 0.12;
/** Peak extra brightness and extra radial thickness at the top of a glint. */
const GLINT_BRIGHT = 1.1;
const GLINT_SCALE = 0.45;

/** Emitter brightness per emphasis level. Index with `GhostEmphasis`. */
export const GHOST_EMPHASIS_GAIN: readonly [number, number, number] = [0.18, 0.45, 0.92];
/** Dark-bank size per emphasis level: a leader gets a full pocket, a faint ghost a hint. */
const GHOST_SHADE_SCALE: readonly [number, number, number] = [0.55, 0.8, 1];

/**
 * The value an emitter must stay under to avoid feeding bloom, matching the scenes'
 * `postConfig.bloomThreshold` (~0.85). Emphasis 2 and the glint are the sanctioned
 * excursions; 0 and 1 are checked against this in the tests.
 */
export const GHOST_BLOOM_THRESHOLD = 0.85;

/**
 * Presentation-only fan-out for CO-LOCATED grid ghosts, in cells. Two seeds standing on
 * the same cell would otherwise be one ghost, so each index takes a deterministic point
 * of a golden-angle (Vogel) spiral around its true cell centre. The radius is capped at
 * 0.22 cells — well inside the +-0.5 cell boundary — so a ghost NEVER renders in a cell
 * it is not in. This offset is the only place in this file where a ghost is not exactly
 * at its ground-truth position, and it is bounded by construction.
 */
export const GHOST_CELL_OFFSET_MAX = 0.22;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** Rate ramp: dark -> accent up to the knee, accent -> celebration gold above it. */
const RAMP_KNEE = 0.82;
/** Value floor of the ramp — what makes it monotone in luminance for ANY accent. */
const RAMP_VALUE_MIN = 0.35;

/** Grid ghost size relative to the hero wisp (radius, height, contact shade). */
const GRID_GHOST_SCALE = 0.6;
/** Wisp core radius the grid ghost is scaled from. */
const WISP_CORE_RADIUS = 0.16;
/** Under-silhouette shell radius, in core radii (kit/wisp SILHOUETTE_SIZE). */
const SHELL_SIZE = 1.4;
/** Contact-shade diameter, in core radii (kit/wisp SHADE_SIZE). */
const SHADE_SIZE = 5.6;
/** The contact shade lifts off the floor plane so it never z-fights it. */
const SHADE_LIFT = 0.014;
/** How far the accent is carried toward `night.skyTop` for the dark banks. */
const SHADE_BLEND = 0.93;
/** How far the ember accent is carried toward night for a numerically diverged seed. */
const DEAD_BLEND = 0.62;

/** Bank material opacities. Emitters are additive so their weight lives in the colour. */
const EMITTER_OPACITY = 1;
const SHELL_OPACITY = 0.55;
const SHADE_OPACITY = 0.4;

/** Classic-control marker/link sizes, in world units (classic worlds are ~2-5 across). */
const CLASSIC_MARKER_RADIUS = 0.07;
const CLASSIC_LINK_THICKNESS = 0.03;

const AXIS_Z = /* @__PURE__ */ new THREE.Vector3(0, 0, 1);

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return Number.isFinite(x) ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
}

/** Triangles in ONE copy of a bank's geometry — indexed or not, the same three's renderer counts. */
function bankTriangles(geometry: THREE.BufferGeometry): number {
  const index = geometry.getIndex();
  const count = index !== null ? index.count : (geometry.getAttribute('position')?.count ?? 0);
  return Math.floor(count / 3);
}

/** Instance detail multiplier: thumb halves it, medium sits between. */
function tierDetail(tier: QualityTier): number {
  return tier === 'thumb' ? 0.5 : tier === 'medium' ? 0.75 : 1;
}

/** Icosahedron subdivision at a tier — floor, so thumb never rounds back up to hero. */
function icoDetail(base: number, detail: number): number {
  return Math.max(0, Math.floor(base * detail));
}

/** Radial-sprite resolution at a tier (hero 64, medium 48, thumb 32). */
function spriteSize(detail: number): number {
  return detail <= 0.5 ? 32 : detail < 1 ? 48 : 64;
}

/**
 * The presentation-only polar fan-out for grid ghost `i` of `n`, in CELLS.
 * A Vogel spiral: unique per index, uniformly dense, bounded by GHOST_CELL_OFFSET_MAX.
 * Writes [dx, dz] into `out`.
 */
export function ghostCellOffset(i: number, n: number, out: Float32Array | number[]): void {
  const count = Math.max(1, n);
  const angle = i * GOLDEN_ANGLE;
  const radius = GHOST_CELL_OFFSET_MAX * Math.sqrt(Math.min(1, (i + 0.5) / count));
  out[0] = Math.cos(angle) * radius;
  out[1] = Math.sin(angle) * radius;
}

/** The glint envelope: exp decay x (1 - t/T), zero outside [0, T] (canon lesson 6). */
export function ghostGlintEnvelope(age: number): number {
  if (!(age >= 0) || age >= GHOST_GLINT_SECONDS) return 0;
  return Math.exp(-age / GLINT_TAU) * (1 - age / GHOST_GLINT_SECONDS);
}

type ClassicKind = 'mountain-car' | 'pendulum' | 'acrobot' | 'cart-pole';

const CLASSIC_KINDS: Partial<Record<EnvId, ClassicKind>> = {
  'MountainCar-cont-v0': 'mountain-car',
  'Pendulum-cont-v0': 'pendulum',
  'Acrobot-cont-v0': 'acrobot',
  'CartPole-cont-v0': 'cart-pole',
};

/** Pose slots each body plan actually reads — the ones that must be finite. */
function poseNeedOf(category: GhostCategory, kind: ClassicKind): number {
  if (category === 'cheetah') return 9;
  if (category === 'grid') return 2;
  return kind === 'acrobot' ? 2 : kind === 'cart-pole' ? 3 : 1;
}

const POSE_STRIDE = 9;

// ---------------------------------------------------------------------------
// the layer
// ---------------------------------------------------------------------------

/**
 * Every seed of one env, drawn around the hero scene.
 *
 * ```ts
 * const ghosts = makeGhostLayer({
 *   category: 'cheetah', envId, geometry, maxGhosts: seeds.length,
 *   tokens: TOKENS, accent: TOKENS.accents['half-cheetah'], tier,
 * });
 * host.scene.add(ghosts.group);            // the HOST does this, never a scene
 * ghosts.setGhostFrame(i, pose, alive);    // per seed, per frame
 * ghosts.setRate(i, rate01);
 * ghosts.setEmphasis(leader, 2);
 * ghosts.update(dtReal, hostClockSeconds); // host clock only
 * ```
 */
export class GhostLayer {
  /** add this ONE object to the scene; it stays at the origin */
  readonly group = new THREE.Group();
  readonly category: GhostCategory;
  readonly envId: EnvId;
  readonly tier: QualityTier;
  /** instance capacity — the draw count is independent of it */
  readonly maxGhosts: number;
  readonly banks: readonly GhostBank[];

  private readonly geometry: EnvGeometry;
  private readonly classicKind: ClassicKind;
  private readonly poseNeed: number;

  // ---- per-ghost state (all preallocated; nothing here ever grows) ----
  /** last FINITE pose per ghost — what a diverged seed freezes at */
  private readonly poses: Float32Array;
  private readonly posed: Uint8Array;
  private readonly aliveFlags: Uint8Array;
  private readonly rates: Float32Array;
  private readonly emphasis: Uint8Array;
  /** host-clock second each glint started; -Infinity = never */
  private readonly glintAt: Float64Array;
  /** compact list of ghosts with a live glint (avoids an O(n) sweep per frame) */
  private readonly glintList: Int32Array;
  private readonly glinting: Uint8Array;
  private glintCount = 0;

  // ---- density normalization (the co-located-blowout guard) ----
  // Additive emitter banks SUM: 63 seeds parked on one cell (every run's first frames)
  // otherwise render as a white-out. Each emphasis-0 emitter is scaled by
  // 1/sqrt(#ghosts sharing its quantized anchor), so a lone ghost is untouched, a
  // co-located pack sums to ~sqrt(k) of one ghost (bright, readable, never clipped),
  // and an emphasized (hover/leader) ghost keeps full gain so interaction still pops.
  private readonly densityGain: Float32Array;
  private readonly dTableKey: Float64Array;
  private readonly dTableCount: Int32Array;
  private readonly dTableGen: Int32Array;
  private readonly dTableMask: number;
  private dGen = 0;
  private densityDirty = false;

  // ---- scratch ----
  private readonly poseScratch = new Float32Array(POSE_STRIDE);
  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly pointOut = new Float32Array(4);
  private readonly offsetOut = new Float32Array(2);
  private readonly mat = new THREE.Matrix4();
  private readonly pos = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scl = new THREE.Vector3();
  private readonly color = new THREE.Color();

  // ---- palette ----
  private readonly rampLow = new THREE.Color();
  private readonly rampMid = new THREE.Color();
  private readonly rampHigh = new THREE.Color();
  private readonly shadeColor = new THREE.Color();
  private readonly deadColor = new THREE.Color();
  private readonly deadShadeColor = new THREE.Color();
  /** the emitter materials' own colour — white, so `instanceColor` carries everything */
  private readonly white = new THREE.Color();

  // ---- resources ----
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly materials: EmissiveMaterial[] = [];
  private readonly sprites: THREE.Texture[] = [];
  private readonly bankList: GhostBank[] = [];

  // ---- derived sizes ----
  private readonly boneRadius: number;
  private readonly headRadius: number;
  private readonly gridRadiusScale: number;
  private readonly gridHeight: number;
  private readonly gridShadeSize: number;

  private time = 0;
  private disposed = false;

  constructor(opts: GhostLayerOptions) {
    this.category = opts.category;
    this.envId = opts.envId;
    this.tier = opts.tier;
    this.geometry = opts.geometry;
    this.classicKind = CLASSIC_KINDS[opts.envId] ?? 'mountain-car';
    this.poseNeed = poseNeedOf(opts.category, this.classicKind);
    const n = Math.max(0, Math.floor(opts.maxGhosts));
    this.maxGhosts = n;
    const capacity = Math.max(1, n);

    this.group.name = opts.name ?? GHOST_GROUP_NAME;

    this.poses = new Float32Array(capacity * POSE_STRIDE);
    this.posed = new Uint8Array(capacity);
    this.aliveFlags = new Uint8Array(capacity).fill(1);
    this.rates = new Float32Array(capacity);
    this.emphasis = new Uint8Array(capacity);
    this.glintAt = new Float64Array(capacity).fill(-Infinity);
    this.glintList = new Int32Array(capacity);
    this.glinting = new Uint8Array(capacity);

    this.densityGain = new Float32Array(capacity).fill(1);
    let tableSize = 1;
    while (tableSize < capacity * 2) tableSize <<= 1;
    this.dTableKey = new Float64Array(tableSize);
    this.dTableCount = new Int32Array(tableSize);
    this.dTableGen = new Int32Array(tableSize);
    this.dTableMask = tableSize - 1;

    const tokens = opts.tokens;
    const accent = new THREE.Color(opts.accent);
    const night = new THREE.Color(tokens.palettes.night.skyTop);
    this.rampLow.copy(night);
    this.rampMid.copy(accent);
    this.rampHigh.set(tokens.celebrationGold);
    this.shadeColor.copy(accent).lerp(night, SHADE_BLEND);
    this.white.set(tokens.palettes.noon.sun);
    // "dark red" as a LEGAL token blend (STYLE.md: scene colours are blends of tokens):
    // the alpenglow ember accent carried most of the way to night.
    this.deadColor
      .set(tokens.accents['mountain-car'] ?? tokens.celebrationGold)
      .lerp(night, DEAD_BLEND);
    this.deadShadeColor.copy(this.deadColor).lerp(night, SHADE_BLEND);

    const cheetah = this.geometry.cheetah ?? cheetahGeometry;
    this.boneRadius = cheetah.radius;
    this.headRadius = cheetah.headRadius;
    this.gridRadiusScale = WISP_CORE_RADIUS * GRID_GHOST_SCALE;
    this.gridHeight = tokens.wispHeight * GRID_GHOST_SCALE;
    this.gridShadeSize = this.gridRadiusScale * SHADE_SIZE;

    const detail = tierDetail(opts.tier);
    if (opts.category === 'cheetah') this.buildCheetahBanks(capacity, detail);
    else if (opts.category === 'grid') this.buildGridBanks(capacity, detail);
    else this.buildClassicBanks(capacity, detail);

    this.banks = this.bankList;

    // Nothing is posed yet, so every instance starts collapsed (a zero-scale matrix
    // submits degenerate triangles) with its resting colour already written — three's
    // instance buffers are zero-filled, which is not a valid transform.
    for (let i = 0; i < capacity; i++) {
      this.hideGhost(i);
      this.applyTint(i);
    }
    for (const bank of this.bankList) {
      bank.mesh.count = n;
      bank.mesh.instanceMatrix.needsUpdate = true;
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
  }

  // -------------------------------------------------------------------------
  // public API
  // -------------------------------------------------------------------------

  /**
   * Place ghost `i` at `pose`. `alive === false` is a numerically diverged seed: it
   * FREEZES at its last finite pose and turns dark red — the death certificate, drawn.
   * A non-finite pose freezes the same way even when the caller still believes the seed
   * is alive, so a NaN never teleports a ghost to the origin.
   */
  setGhostFrame(i: number, pose: Float32Array, alive: boolean): void {
    if (!this.valid(i)) return;
    const wasAlive = this.aliveFlags[i] === 1;
    this.aliveFlags[i] = alive ? 1 : 0;
    if (alive && this.finiteEnough(pose)) {
      const o = i * POSE_STRIDE;
      for (let k = 0; k < POSE_STRIDE; k++) this.poses[o + k] = pose[k] ?? 0;
      this.posed[i] = 1;
      this.applyGhost(i);
    } else if (this.posed[i] === 1) {
      // frozen: the transform is already the last finite pose, but a glint or an
      // emphasis change may have left a stale scale on it
      this.applyGhost(i);
    } else {
      this.hideGhost(i);
    }
    if (wasAlive !== alive) this.applyTint(i);
    this.densityDirty = true;
  }

  /**
   * Where seed `i` sits in the population, in [0, 1] — pass its average-reward-rate
   * QUANTILE, not the raw rate. The ramp is a quantile ramp on purpose: reward rates have
   * heavy tails (that is why the headline statistic is an IQM), so a linear map of the
   * raw rate would paint the whole fleet one colour and one rocket seed gold.
   */
  setRate(i: number, rate01: number): void {
    if (!this.valid(i)) return;
    const next = clamp01(rate01);
    if (this.rates[i] === next) return;
    this.rates[i] = next;
    this.applyTint(i);
  }

  /** 0 = faint, 1 = readable, 2 = leader. Drives brightness and dark-bank size. */
  setEmphasis(i: number, level: GhostEmphasis): void {
    if (!this.valid(i)) return;
    const next = level === 2 ? 2 : level === 1 ? 1 : 0;
    if (this.emphasis[i] === next) return;
    this.emphasis[i] = next;
    this.applyGhost(i);
    this.applyTint(i);
  }

  /**
   * Fire ghost `i`'s 300 ms celebration beat. Rate-limited per ghost: a glint already in
   * flight is not restarted (a seed collecting five rewards in five frames pulses once,
   * it does not strobe). Dead ghosts never glint.
   */
  glint(i: number): void {
    if (!this.valid(i)) return;
    if (this.aliveFlags[i] !== 1 || this.posed[i] !== 1) return;
    const age = this.time - (this.glintAt[i] ?? -Infinity);
    if (age < GHOST_GLINT_SECONDS) return;
    this.glintAt[i] = this.time;
    if (this.glinting[i] !== 1) {
      this.glinting[i] = 1;
      this.glintList[this.glintCount++] = i;
    }
    this.applyGhost(i);
    this.applyTint(i);
  }

  /**
   * `dt` is wall seconds (unused by the envelopes, which are pure functions of the host
   * clock — canon lesson 7: pause must freeze a celebration mid-air); `time` is the host
   * playback clock in seconds. Allocates nothing.
   */
  update(dt: number, time: number): void {
    void dt;
    this.time = Number.isFinite(time) ? time : this.time;
    if (this.densityDirty) {
      this.densityDirty = false;
      this.recomputeDensity();
    }
    if (this.glintCount === 0) return;
    let write = 0;
    for (let k = 0; k < this.glintCount; k++) {
      const i = this.glintList[k] ?? 0;
      const age = this.time - (this.glintAt[i] ?? -Infinity);
      this.applyGhost(i);
      this.applyTint(i);
      if (age >= 0 && age < GHOST_GLINT_SECONDS) {
        this.glintList[write++] = i;
      } else {
        // one last apply at zero envelope (above) retires it cleanly
        this.glinting[i] = 0;
      }
    }
    this.glintCount = write;
  }

  /** Draw calls this layer submits — a function of the category, not of the ghost count. */
  draws(): number {
    if (!this.group.visible) return 0;
    let n = 0;
    for (const bank of this.bankList) if (bank.mesh.visible && bank.mesh.count > 0) n++;
    return n;
  }

  /**
   * Triangles this layer submits at its CURRENT instance count.
   *
   * The draw count is flat in the ghost count; the triangle count is not, and instancing
   * is exactly the trade that makes it so — one draw, `count` copies of the geometry. So
   * this is the number that decides whether a population is affordable, and the one a
   * caller must size `maxGhosts` against (`GhostLayer.trianglesPerGhost`, or the table in
   * `GHOST_TRIANGLES_PER_GHOST`). It is reported separately from the scene's own load
   * (`HostFrameInfo.ghostTriangles`) because the ghost group renders INSIDE the scene: a
   * manifest budget is a budget for the artisan scene, and a population drawn around it
   * is not overspending — it is a second, declared cost.
   */
  triangles(): number {
    if (!this.group.visible) return 0;
    let n = 0;
    for (const bank of this.bankList) {
      if (!bank.mesh.visible || bank.mesh.count <= 0) continue;
      n += bankTriangles(bank.mesh.geometry) * bank.mesh.count;
    }
    return n;
  }

  /** Triangles ONE ghost costs at this tier — `maxGhosts × this` is the layer's load. */
  trianglesPerGhost(): number {
    let n = 0;
    for (const bank of this.bankList) n += bankTriangles(bank.mesh.geometry);
    return n;
  }

  /** Full GPU release; `makeGhostLayer` -> `dispose()` must round-trip clean. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    for (const m of this.materials) m.dispose();
    this.materials.length = 0;
    for (const t of this.sprites) t.dispose();
    this.sprites.length = 0;
    for (const bank of this.bankList) {
      bank.mesh.count = 0;
      bank.mesh.dispose();
    }
    this.bankList.length = 0;
    this.group.removeFromParent();
    this.group.clear();
  }

  // -------------------------------------------------------------------------
  // colour
  // -------------------------------------------------------------------------

  /**
   * The quantile ramp, written into `out`: `night.skyTop` -> accent up to the knee, then
   * accent -> `celebrationGold` in the very top band, all times a value ramp.
   *
   * The value ramp is what makes the result MONOTONE in luminance for every one of the 16
   * accents: luminance is a linear functional of the (linear-space) channels, so both
   * legs are affine in `t`, and the rising value factor dominates the small dip some
   * bright accents take on the way to gold.
   */
  rampColor(t: number, out: THREE.Color): THREE.Color {
    const x = clamp01(t);
    if (x <= RAMP_KNEE) out.copy(this.rampLow).lerp(this.rampMid, RAMP_KNEE > 0 ? x / RAMP_KNEE : 1);
    else out.copy(this.rampMid).lerp(this.rampHigh, (x - RAMP_KNEE) / (1 - RAMP_KNEE));
    return out.multiplyScalar(RAMP_VALUE_MIN + (1 - RAMP_VALUE_MIN) * x);
  }

  /** The colour ghost `i` currently writes into `instanceColor` on its emitter banks. */
  emitterColor(i: number, out: THREE.Color): THREE.Color {
    const emph = (this.emphasis[i] ?? 0) as 0 | 1 | 2;
    const gain = GHOST_EMPHASIS_GAIN[emph] ?? GHOST_EMPHASIS_GAIN[0];
    if (this.aliveFlags[i] !== 1) return out.copy(this.deadColor).multiplyScalar(gain);
    this.rampColor(this.rates[i] ?? 0, out);
    // Density normalization applies to the faint crowd only: an emphasized ghost
    // (hover / leader) must pop out of the very pack that is being dimmed.
    const density = emph === 0 ? (this.densityGain[i] ?? 1) : 1;
    return out.multiplyScalar(gain * density * (1 + GLINT_BRIGHT * this.glintOf(i)));
  }

  private glintOf(i: number): number {
    if (this.glinting[i] !== 1) return 0;
    return ghostGlintEnvelope(this.time - (this.glintAt[i] ?? -Infinity));
  }

  /**
   * Quantized anchor key for density bucketing. The quantum is the scale at which
   * additive overlap becomes a legibility problem, per category: ~a body length for the
   * cheetah track, the exact cell for grids, a state-space sliver for classic dials.
   */
  private densityKey(i: number): number {
    const o = i * POSE_STRIDE;
    const x = this.poses[o] ?? 0;
    if (this.category === 'grid') {
      return Math.round(x) * 4096 + Math.round(this.poses[o + 1] ?? 0);
    }
    if (this.category === 'cheetah') return Math.round(x / 1.5);
    return Math.round(x / 0.15);
  }

  /** One pass over the open-addressed count table, then gains. Allocates nothing. */
  private recomputeDensity(): void {
    const gen = ++this.dGen;
    const mask = this.dTableMask;
    for (let i = 0; i < this.maxGhosts; i++) {
      if (this.posed[i] !== 1 || this.aliveFlags[i] !== 1) continue;
      const key = this.densityKey(i);
      let slot = (((key * 2654435761) | 0) >>> 0) & mask;
      for (;;) {
        if (this.dTableGen[slot] !== gen) {
          this.dTableGen[slot] = gen;
          this.dTableKey[slot] = key;
          this.dTableCount[slot] = 1;
          break;
        }
        if (this.dTableKey[slot] === key) {
          this.dTableCount[slot] = (this.dTableCount[slot] ?? 0) + 1;
          break;
        }
        slot = (slot + 1) & mask;
      }
    }
    for (let i = 0; i < this.maxGhosts; i++) {
      let gain = 1;
      if (this.posed[i] === 1 && this.aliveFlags[i] === 1) {
        const key = this.densityKey(i);
        let slot = (((key * 2654435761) | 0) >>> 0) & mask;
        for (;;) {
          if (this.dTableGen[slot] !== gen) break;
          if (this.dTableKey[slot] === key) {
            gain = 1 / Math.sqrt(this.dTableCount[slot] ?? 1);
            break;
          }
          slot = (slot + 1) & mask;
        }
      }
      if (Math.abs(gain - (this.densityGain[i] ?? 1)) > 1e-3) {
        this.densityGain[i] = gain;
        this.applyTint(i);
      }
    }
  }

  private applyTint(i: number): void {
    const emitter = this.emitterColor(i, this.color);
    for (const bank of this.bankList) {
      if (bank.role !== 'emitter') continue;
      bank.mesh.setColorAt(i, emitter);
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
    const shade = this.aliveFlags[i] === 1 ? this.shadeColor : this.deadShadeColor;
    for (const bank of this.bankList) {
      if (bank.role !== 'shade') continue;
      bank.mesh.setColorAt(i, shade);
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
  }

  // -------------------------------------------------------------------------
  // transforms
  // -------------------------------------------------------------------------

  private valid(i: number): boolean {
    return !this.disposed && Number.isInteger(i) && i >= 0 && i < this.maxGhosts;
  }

  private finiteEnough(pose: ArrayLike<number>): boolean {
    for (let k = 0; k < this.poseNeed; k++) {
      if (!Number.isFinite(pose[k] ?? Number.NaN)) return false;
    }
    return true;
  }

  private applyGhost(i: number): void {
    if (this.posed[i] !== 1) {
      this.hideGhost(i);
      return;
    }
    const o = i * POSE_STRIDE;
    for (let k = 0; k < POSE_STRIDE; k++) this.poseScratch[k] = this.poses[o + k] ?? 0;
    const puff = 1 + GLINT_SCALE * this.glintOf(i);
    if (this.category === 'cheetah') this.applyCheetah(i, puff);
    else if (this.category === 'grid') this.applyGrid(i, puff);
    else this.applyClassic(i, puff);
    for (const bank of this.bankList) bank.mesh.instanceMatrix.needsUpdate = true;
  }

  private applyCheetah(i: number, puff: number): void {
    const fk = (this.geometry.cheetah ?? cheetahGeometry).fk(this.poseScratch, this.fkOut);
    const r = this.boneRadius * puff;
    for (let b = 0; b < CHEETAH_BONES.length; b++) {
      const bone = CHEETAH_BONES[b];
      const mesh = this.bankList[b]?.mesh;
      if (!bone || !mesh) continue;
      const a0 = bone[0] * 2;
      const a1 = bone[1] * 2;
      this.setSegment(mesh, i, fk[a0] ?? 0, fk[a0 + 1] ?? 0, fk[a1] ?? 0, fk[a1 + 1] ?? 0, r);
    }
    const head = this.bankList[CHEETAH_BONES.length]?.mesh;
    if (head) {
      const h = FK_POINT_INDEX.head * 2;
      this.setPoint(head, i, fk[h] ?? 0, fk[h + 1] ?? 0, 0, this.headRadius * puff);
    }
  }

  private applyGrid(i: number, puff: number): void {
    const grid = this.geometry.grid;
    const w = grid?.w ?? 1;
    const h = grid?.h ?? 1;
    ghostCellOffset(i, this.maxGhosts, this.offsetOut);
    const wx = ((this.poseScratch[0] ?? 0) - (w - 1) / 2) * CELL + (this.offsetOut[0] ?? 0) * CELL;
    const wz = ((this.poseScratch[1] ?? 0) - (h - 1) / 2) * CELL + (this.offsetOut[1] ?? 0) * CELL;
    const emph = (this.emphasis[i] ?? 0) as 0 | 1 | 2;
    const dark = GHOST_SHADE_SCALE[emph] ?? GHOST_SHADE_SCALE[0];

    const core = this.bankList[0]?.mesh;
    if (core) this.setOrb(core, i, wx, this.gridHeight, wz, this.gridRadiusScale * puff);
    const shell = this.bankList[1]?.mesh;
    if (shell) {
      // emphasis widens the dark RING; it never shrinks the shell inside the orb it is
      // supposed to be the background of, at any emphasis level
      const ring = 1 + (SHELL_SIZE - 1) * dark;
      this.setOrb(shell, i, wx, this.gridHeight, wz, this.gridRadiusScale * ring * puff);
    }
    const shade = this.bankList[2]?.mesh;
    if (shade) {
      this.pos.set(wx, SHADE_LIFT, wz);
      this.quat.identity();
      const s = this.gridShadeSize * puff * dark;
      this.scl.set(s, 1, s);
      this.mat.compose(this.pos, this.quat, this.scl);
      shade.setMatrixAt(i, this.mat);
    }
  }

  private applyClassic(i: number, puff: number): void {
    const marker = this.bankList[0]?.mesh;
    const link0 = this.bankList[1]?.mesh;
    const link1 = this.bankList[2]?.mesh;
    const r = CLASSIC_MARKER_RADIUS * puff;
    const t = CLASSIC_LINK_THICKNESS * puff;
    const anchors = this.geometry.classic?.anchors;
    const pivot = anchors?.['pivot'];
    const px = pivot?.[0] ?? 0;
    const py = pivot?.[1] ?? 0;
    const p = this.pointOut;

    switch (this.classicKind) {
      case 'mountain-car': {
        const x = this.poseScratch[0] ?? 0;
        const heightAt = this.geometry.classic?.heightAt;
        const y = heightAt ? heightAt(x) : 0;
        if (marker) this.setPoint(marker, i, x, y, 0, r);
        return;
      }
      case 'pendulum': {
        pendulumTip(this.poseScratch[0] ?? 0, p);
        const bx = p[0] ?? 0;
        const by = p[1] ?? 0;
        if (link0) this.setSegment(link0, i, px, py, bx, by, t);
        if (marker) this.setPoint(marker, i, bx, by, 0, r);
        return;
      }
      case 'cart-pole': {
        const x = this.poseScratch[0] ?? 0;
        cartPoleTip(x, this.poseScratch[2] ?? 0, p);
        const tx = p[0] ?? 0;
        const ty = p[1] ?? 0;
        if (link0) this.setSegment(link0, i, x, py, tx, ty, t);
        if (marker) this.setPoint(marker, i, x, py, 0, r);
        return;
      }
      case 'acrobot': {
        acrobotPoints(this.poseScratch[0] ?? 0, this.poseScratch[1] ?? 0, p);
        const ex = p[0] ?? 0;
        const ey = p[1] ?? 0;
        const tx = p[2] ?? 0;
        const ty = p[3] ?? 0;
        if (link0) this.setSegment(link0, i, px, py, ex, ey, t);
        if (link1) this.setSegment(link1, i, ex, ey, tx, ty, t);
        if (marker) this.setPoint(marker, i, tx, ty, 0, r);
        return;
      }
    }
  }

  /**
   * A capsule/box spanning (ax, ay) -> (bx, by) in the planar x-y world, at z = 0.
   * The unit geometry runs along local +y, so the rotation is the one that maps +y onto
   * the segment and the y SCALE is the segment's true length: the glint's `thickness`
   * touches x/z only, so both endpoints stay exactly on ground truth.
   */
  private setSegment(
    mesh: THREE.InstancedMesh,
    i: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    thickness: number,
  ): void {
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.sqrt(dx * dx + dy * dy);
    this.pos.set((ax + bx) * 0.5, (ay + by) * 0.5, 0);
    this.quat.setFromAxisAngle(AXIS_Z, Math.atan2(-dx, dy));
    this.scl.set(thickness, len > 1e-6 ? len : 1e-6, thickness);
    this.mat.compose(this.pos, this.quat, this.scl);
    mesh.setMatrixAt(i, this.mat);
  }

  /** A unit-radius sphere placed at a planar (x, y) point, z = depth. */
  private setPoint(
    mesh: THREE.InstancedMesh,
    i: number,
    x: number,
    y: number,
    z: number,
    radius: number,
  ): void {
    this.pos.set(x, y, z);
    this.quat.identity();
    this.scl.setScalar(radius);
    this.mat.compose(this.pos, this.quat, this.scl);
    mesh.setMatrixAt(i, this.mat);
  }

  /** A unit-radius sphere placed at a full 3-D world point (grid orbs). */
  private setOrb(
    mesh: THREE.InstancedMesh,
    i: number,
    x: number,
    y: number,
    z: number,
    radius: number,
  ): void {
    this.setPoint(mesh, i, x, y, z, radius);
  }

  /** Collapse every bank's instance `i` — a zero-scale matrix draws nothing. */
  private hideGhost(i: number): void {
    this.pos.set(0, 0, 0);
    this.quat.identity();
    this.scl.set(0, 0, 0);
    this.mat.compose(this.pos, this.quat, this.scl);
    for (const bank of this.bankList) {
      bank.mesh.setMatrixAt(i, this.mat);
      bank.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  // -------------------------------------------------------------------------
  // banks
  // -------------------------------------------------------------------------

  private addBank(
    name: string,
    role: 'emitter' | 'shade',
    geometry: THREE.BufferGeometry,
    material: EmissiveMaterial,
    capacity: number,
    renderOrder: number,
  ): GhostBank {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.name = `${GHOST_GROUP_NAME}:${name}`;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // create the colour buffer up front so `setColorAt` never allocates on the hot path
    const colors = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3).fill(1), 3);
    colors.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = colors;
    // instance bounds are not tracked, and a treadmill scene moves the world under the
    // ghosts — culling on the base geometry's bounds would pop them out at random
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = renderOrder;
    this.geometries.push(geometry);
    this.materials.push(material);
    this.group.add(mesh);
    const bank: GhostBank = { name, mesh, role };
    this.bankList.push(bank);
    return bank;
  }

  /** Additive, unlit, white — `instanceColor` carries the whole ramp (canon lesson 13). */
  private emitterMaterial(name: string, map?: THREE.Texture): EmissiveMaterial {
    return emissiveMaterial(this.white, 1, {
      transparent: true,
      opacity: EMITTER_OPACITY,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      map: map ?? null,
      name: `${GHOST_GROUP_NAME}:${name}`,
    });
  }

  /** Normal-blended dark backing — the emitter's local background, not its neighbour. */
  private shadeMaterial(name: string, opacity: number, map?: THREE.Texture): EmissiveMaterial {
    return emissiveMaterial(this.white, 1, {
      transparent: true,
      opacity,
      depthWrite: false,
      map: map ?? null,
      name: `${GHOST_GROUP_NAME}:${name}`,
    });
  }

  private buildCheetahBanks(capacity: number, detail: number): void {
    const capSeg = Math.max(2, Math.round(4 * detail));
    const radialSeg = Math.max(4, Math.round(8 * detail));
    for (let b = 0; b < CHEETAH_BONES.length; b++) {
      // unit radius, unit shaft: the instance matrix carries the real radius and length
      const geo = new THREE.CapsuleGeometry(1, 1, capSeg, radialSeg);
      geo.name = `${GHOST_GROUP_NAME}:bone${b}`;
      this.addBank(`bone${b}`, 'emitter', geo, this.emitterMaterial(`bone${b}`), capacity, -2);
    }
    const head = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    head.name = `${GHOST_GROUP_NAME}:head`;
    this.addBank('head', 'emitter', head, this.emitterMaterial('head'), capacity, -2);
  }

  private buildGridBanks(capacity: number, detail: number): void {
    const core = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    core.name = `${GHOST_GROUP_NAME}:core`;
    this.addBank('core', 'emitter', core, this.emitterMaterial('core'), capacity, -2);

    // the dark side of the glow, at ghost scale (canon lesson 22b)
    const shell = new THREE.IcosahedronGeometry(1, icoDetail(1, detail));
    shell.name = `${GHOST_GROUP_NAME}:shell`;
    this.addBank('shell', 'shade', shell, this.shadeMaterial('shell', SHELL_OPACITY), capacity, -3);

    const sprite = radialSprite(spriteSize(detail), 1);
    this.sprites.push(sprite);
    const plane = new THREE.PlaneGeometry(1, 1);
    plane.rotateX(-Math.PI / 2);
    plane.name = `${GHOST_GROUP_NAME}:shade`;
    this.addBank(
      'shade',
      'shade',
      plane,
      this.shadeMaterial('shade', SHADE_OPACITY, sprite),
      capacity,
      -3,
    );
  }

  private buildClassicBanks(capacity: number, detail: number): void {
    const marker = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    marker.name = `${GHOST_GROUP_NAME}:marker`;
    this.addBank('marker', 'emitter', marker, this.emitterMaterial('marker'), capacity, -2);

    const links = this.classicKind === 'acrobot' ? 2 : this.classicKind === 'mountain-car' ? 0 : 1;
    for (let k = 0; k < links; k++) {
      const geo = new THREE.BoxGeometry(1, 1, 1);
      geo.name = `${GHOST_GROUP_NAME}:link${k}`;
      this.addBank(`link${k}`, 'emitter', geo, this.emitterMaterial(`link${k}`), capacity, -2);
    }
  }
}

/** Build a GhostLayer. See the class doc for the usage shape. */
export function makeGhostLayer(opts: GhostLayerOptions): GhostLayer {
  return new GhostLayer(opts);
}
