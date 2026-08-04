// kit/wisp — THE shared avatar. Every one of the 10 grid environments is the same
// forest spirit in a different place; that is what makes the gallery read as one
// preserve (STYLE.md: "One sun logic, one avatar, one celebration grammar").
//
// Anatomy: a soft orb core (wispMaterial glow), a petal/leaf skirt that counter-rotates
// under it, and a trailing ribbon that remembers where the wisp has been. Height is
// tokens.wispHeight (0.35) — the orb center sits exactly there above the floor plane.
//
// Accuracy: the wisp's world position is `HopMotion.position`, which is a straight-line
// arc between two `cellToWorld` cells and nothing else. Delete the skirt and the ribbon
// and the remaining orb is still exactly on the true cell.
//
// Rig (STYLE.md "skins are children of exact bones"):
//   group → avatar (= HopMotion.position) → body (yaw, y = height EXACTLY) →
//     ├── core      the probe object. NOTHING decorative ever moves it.
//     ├── silhouette the dark side of the glow — see LEGIBILITY below
//     └── decor     the idle bob lives here — halo, skirt, carry anchor.
//   group → shade   the ground-contact pool (floor plane, never the hop arc)
// The bob used to ride on `body`, which meant the probe object oscillated ±0.018 —
// 18× the 1e-3 assertAligned tolerance, so no grid scene could ever have passed.
//
// LEGIBILITY (canon lesson 22b, measured 2026-08-03). The core, the halo and the ribbon
// are all ADDITIVE: they can only ever add light to whatever is behind them. Across the
// 11 grid scenes the floors run from 27.9 to 189.8 sRGB, and on the bright end the
// creature's peak luminance and its background land within 1.25× of each other — the
// wisp dissolves into the floor of four-rooms and chain. Luminance cannot buy that back
// (more gain just crosses the bloom threshold and smears, canon lesson 19). VALUE can:
// a soft dark under-silhouette behind the orb and a soft contact shade on the floor give
// the wisp its own local background, so the contrast is read against the rim — dark in
// every palette — instead of against the floor. Both are the wisp's OWN accent blended
// most of the way to `palettes.night.skyTop` (canon lesson 9's legal near-black), so on a
// dark floor they are already the floor's value and vanish. Background-agnostic by
// construction: no palette branch, no per-scene tuning, one treatment for all 11 scenes.

import * as THREE from 'three';
import type { GridCell } from '@sim/types';
import { HopMotion, cellToWorldVec, type GridSize, type HopMotionOptions } from './grid';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity, wispMaterial } from './materials';
import type { EmissiveMaterial, OakMaterial, WispMaterial } from './materials';
import { radialSprite } from './textures';
import { TOKENS } from './tokens';
import type { QualityTier } from './types';

/** A facing direction: yaw radians, or a grid-plane delta (dx = east, dy = south). */
export type FaceDir = number | { dx: number; dy: number };

/** Yaw (radians about +y) that points along a grid delta. dy is +z (south). */
export function yawFromGridDelta(dx: number, dy: number): number {
  if (dx === 0 && dy === 0) return 0;
  return Math.atan2(dx, dy);
}

export interface WispOptions {
  /** accent color, normally tokens.accents[slug] */
  color: THREE.ColorRepresentation;
  tier?: QualityTier;
  /** orb center height (default tokens.wispHeight) */
  height?: number;
  /** orb radius (default 0.16) */
  radius?: number;
  /** petal/leaf skirt (default true above thumb) */
  skirt?: boolean;
  /** trailing ribbon (default true above thumb) */
  ribbon?: boolean;
  /** ribbon sample count at hero (default 24) */
  ribbonLength?: number;
  /** dark under-silhouette behind the orb — the bright-floor legibility organ (default true) */
  silhouette?: boolean;
  /** soft ground-contact shade on the floor plane under the avatar (default true) */
  contactShade?: boolean;
  hop?: HopMotionOptions;
  name?: string;
}

const RIBBON_MIN = 6;

/** How far the accent is carried toward `night.skyTop` for the dark side of the glow. */
const SHADE_BLEND = 0.93;
/**
 * Under-silhouette RADIUS, in core radii — a shell just outside the orb and well inside
 * the halo (2.4 r). What buys contrast here is not luminance under the core (measured on
 * four-rooms: the additive core plus ACES compression eat a 0.5-alpha darkening for
 * −0.055 L) but the orb's own HUE: over a bright floor the core's additive green landed
 * on 174-sRGB gold and washed to white, and over the shell it stays green.
 */
const SILHOUETTE_SIZE = 1.4;
/**
 * Contact-shade diameter, in core radii — the organ that actually moves the number. It
 * lies on the FLOOR, outside the core/halo/bloom footprint, so its darkening survives
 * tone mapping and gives the creature a pocket to read against.
 */
const SHADE_SIZE = 5.6;
/** The shade lifts off the floor plane by this much so it never z-fights the floor. */
const SHADE_LIFT = 0.014;
/** Peak darkness of each, before `brightness`. */
const SILHOUETTE_OPACITY = 0.8;
const SHADE_OPACITY = 0.5;
/** Falloff exponent of the contact-shade sprite: 1 = a linear cone, soft to the edge. */
const SHADE_FALLOFF = 1;

/**
 * The avatar.
 *
 * ```ts
 * const wisp = new Wisp({ color: tokens.accents['four-rooms'], tier });
 * scene.add(wisp.group);
 * wisp.setCell(x, y, grid);            // init / teleport
 * wisp.hopTo({ x, y });                // on a state change
 * wisp.update(frame.dtReal, ctx.time());
 * ```
 */
export class Wisp {
  /** add this ONE object to the scene; it stays at the origin */
  readonly group = new THREE.Group();
  /** the moving avatar — its position IS `HopMotion.position` (world space) */
  readonly avatar = new THREE.Group();
  /** the exact state-bearing object — probes() should register this */
  readonly core: THREE.Mesh;
  readonly motion: HopMotion;

  private readonly body = new THREE.Group();
  /** everything that breathes, bobs or spins — never the core */
  private readonly decor = new THREE.Group();
  private readonly skirt: THREE.Mesh | null;
  private readonly carryAnchor = new THREE.Group();
  private readonly ribbon: THREE.Line | null;
  private readonly ribbonPositions: Float32Array | null;
  private readonly ribbonAttr: THREE.BufferAttribute | null;
  private readonly ribbonCount: number;
  private ribbonFilled = false;

  private readonly coreMaterial: WispMaterial;
  private readonly skirtMaterial: OakMaterial | null;
  private readonly ribbonMaterial: THREE.LineBasicMaterial | null;
  private readonly haloMaterial: EmissiveMaterial;
  private readonly halo: THREE.Mesh;
  private readonly geometries: THREE.BufferGeometry[] = [];

  // the dark side of the glow (see LEGIBILITY at the top of the file)
  private readonly silhouette: THREE.Mesh | null;
  private readonly silhouetteMaterial: EmissiveMaterial | null;
  private readonly shade: THREE.Mesh | null;
  private readonly shadeMaterial: EmissiveMaterial | null;
  private readonly shadeSprite: THREE.Texture | null;
  private readonly shadeSize: number;

  private readonly height: number;
  private time = 0;
  private targetYaw = 0;
  private yaw = 0;
  private brightness = 1;
  private targetBrightness = 1;
  private flareAmount = 0;
  private carried: THREE.Object3D | null = null;
  private readonly baseIntensity: number;
  /** hop-arc falloff of the contact shade, 1 when landed (see `syncShade`) */
  private shadeLift = 1;

  constructor(opts: WispOptions, grid: GridSize = { w: 1, h: 1 }) {
    const tier = opts.tier;
    const hero = tier !== 'thumb';
    const detail = tier === 'thumb' ? 0.35 : tier === 'medium' ? 0.65 : 1;
    this.height = opts.height ?? TOKENS.wispHeight;
    const radius = opts.radius ?? 0.16;
    this.group.name = opts.name ?? 'oakWisp';
    this.motion = new HopMotion(grid, opts.hop);

    // ---- orb core ----
    const coreGeo = new THREE.IcosahedronGeometry(radius, tier === 'thumb' ? 1 : 2);
    const haloGeo = new THREE.IcosahedronGeometry(radius * 2.4, tier === 'thumb' ? 0 : 1);
    this.geometries.push(coreGeo, haloGeo);

    this.coreMaterial = wispMaterial(opts.color, { intensity: 1.9, rimPower: 2.1, tier, name: 'oakWispCore' });
    this.baseIntensity = 0.7;
    this.haloMaterial = emissiveMaterial(opts.color, this.baseIntensity, {
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'oakWispHalo',
    });

    this.decor.name = 'oakWispDecor';
    this.core = new THREE.Mesh(coreGeo, this.coreMaterial);
    this.core.name = 'oakWispCore';
    this.halo = new THREE.Mesh(haloGeo, this.haloMaterial);
    this.halo.name = 'oakWispHalo';
    // the core hangs off the exact bone; every decoration hangs off `decor`
    this.body.add(this.core, this.decor);
    this.decor.add(this.halo);

    // ---- the dark side of the glow ----
    const wantsSilhouette = opts.silhouette ?? true;
    const wantsShade = opts.contactShade ?? true;
    this.shadeSprite = wantsShade ? radialSprite(tier === 'thumb' ? 32 : 64, SHADE_FALLOFF) : null;
    const shadeColor = new THREE.Color(opts.color).lerp(new THREE.Color(TOKENS.palettes.night.skyTop), SHADE_BLEND);
    this.shadeSize = radius * SHADE_SIZE;

    if (wantsSilhouette) {
      // A SHELL, not a billboard. The silhouette has to read the same from every rig in
      // the gallery and `Wisp.update()` is never handed a camera, so it must be
      // view-independent — and a sphere is. (The obvious camera-free alternative, a
      // sizeAttenuated Points sprite, was built and measured first: three scales point
      // size by `height/2`, NOT by the focal length, so at fov 40 it renders 2.75× under
      // its nominal world size and the rim came out a 3 px fringe. Geometry, not points.)
      // It reuses the CORE geometry at a scale — no second geometry, and the ring width
      // stays constant because it breathes with the orb.
      this.silhouetteMaterial = emissiveMaterial(shadeColor, 1, {
        transparent: true,
        opacity: SILHOUETTE_OPACITY,
        depthWrite: false,
        name: 'oakWispSilhouette',
      });
      this.silhouette = new THREE.Mesh(coreGeo, this.silhouetteMaterial);
      this.silhouette.name = 'oakWispSilhouette';
      this.silhouette.scale.setScalar(SILHOUETTE_SIZE);
      // BEFORE the additive core and halo — it is their background, not their neighbour
      this.silhouette.renderOrder = -1;
      this.body.add(this.silhouette);
    } else {
      this.silhouette = null;
      this.silhouetteMaterial = null;
    }

    if (wantsShade && this.shadeSprite) {
      const geo = new THREE.PlaneGeometry(1, 1);
      geo.rotateX(-Math.PI / 2);
      geo.name = 'oakWispShade';
      this.geometries.push(geo);
      this.shadeMaterial = emissiveMaterial(shadeColor, 1, {
        transparent: true,
        opacity: SHADE_OPACITY,
        depthWrite: false,
        map: this.shadeSprite,
        name: 'oakWispShade',
      });
      this.shade = new THREE.Mesh(geo, this.shadeMaterial);
      this.shade.name = 'oakWispShade';
      this.shade.renderOrder = -1;
      // the shade belongs to the FLOOR, not to the hop arc: it lives in group space and
      // is placed from the avatar's x/z every frame (`syncShade`)
      this.shade.position.y = SHADE_LIFT;
      this.shade.scale.setScalar(this.shadeSize);
      this.group.add(this.shade);
    } else {
      this.shade = null;
      this.shadeMaterial = null;
    }

    // ---- petal / leaf skirt ----
    const wantsSkirt = (opts.skirt ?? true) && hero;
    if (wantsSkirt) {
      const petals = Math.max(3, Math.round(6 * detail));
      const skirtGeo = buildSkirt(petals, radius);
      this.geometries.push(skirtGeo);
      this.skirtMaterial = oakMaterial({
        color: opts.color,
        roughness: 0.7,
        flatShading: true,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthWrite: false,
        vertexColors: true,
        toonSteps: 2,
        rimColor: opts.color,
        rimStrength: 0.5,
        tier,
        name: 'oakWispSkirt',
      });
      this.skirt = new THREE.Mesh(skirtGeo, this.skirtMaterial);
      this.skirt.name = 'oakWispSkirt';
      this.skirt.position.y = -radius * 0.7;
      this.decor.add(this.skirt);
    } else {
      this.skirt = null;
      this.skirtMaterial = null;
    }

    // ---- carry anchor (a token orbits just in front of the orb) ----
    this.carryAnchor.name = 'oakWispCarry';
    this.carryAnchor.position.set(0, radius * 0.2, radius * 1.5);
    this.decor.add(this.carryAnchor);

    this.body.position.y = this.height;
    this.avatar.name = 'oakWispAvatar';
    this.avatar.add(this.body);
    this.group.add(this.avatar);

    // ---- trailing ribbon ----
    const wantsRibbon = (opts.ribbon ?? true) && hero;
    if (wantsRibbon) {
      this.ribbonCount = Math.max(RIBBON_MIN, Math.round((opts.ribbonLength ?? 24) * detail));
      this.ribbonPositions = new Float32Array(this.ribbonCount * 3);
      const geo = new THREE.BufferGeometry();
      this.ribbonAttr = new THREE.BufferAttribute(this.ribbonPositions, 3);
      this.ribbonAttr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('position', this.ribbonAttr);
      geo.setDrawRange(0, this.ribbonCount);
      this.geometries.push(geo);
      this.ribbonMaterial = new THREE.LineBasicMaterial({
        color: new THREE.Color(opts.color),
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        fog: false,
      });
      this.ribbonMaterial.name = 'oakWispRibbon';
      this.ribbon = new THREE.Line(geo, this.ribbonMaterial);
      this.ribbon.name = 'oakWispRibbon';
      this.ribbon.frustumCulled = false;
      // the trail lives in SCENE space (a sibling of the avatar), so it stays behind
      this.group.add(this.ribbon);
    } else {
      this.ribbon = null;
      this.ribbonPositions = null;
      this.ribbonAttr = null;
      this.ribbonMaterial = null;
      this.ribbonCount = 0;
    }

    this.avatar.position.copy(this.motion.position);
    this.resetRibbon();
    this.syncShade();
  }

  /** world position of the avatar (the orb sits `height` above it) */
  get position(): THREE.Vector3 {
    return this.avatar.position;
  }

  /** the destination cell (what sim-core says) */
  get cellX(): number {
    return this.motion.cellX;
  }

  get cellY(): number {
    return this.motion.cellY;
  }

  /** the object a scene should hand to `AlignmentProbe` (world pos = orb center) */
  get probeObject(): THREE.Object3D {
    return this.core;
  }

  /** Bind (or rebind) the grid and TELEPORT to a cell — no easing, scrub-safe. */
  setCell(x: number, y: number, grid: GridSize): void {
    this.motion.setGrid(grid);
    this.motion.snap(x, y);
    this.avatar.position.copy(this.motion.position);
    this.resetRibbon();
    this.syncShade();
  }

  /** Start a straight hop to a cell (STYLE.md: straight hop only, never path-planned). */
  hopTo(cell: GridCell | { x: number; y: number }, now?: number): void {
    this.motion.hopTo(cell.x, cell.y, now ?? this.time);
    this.avatar.position.copy(this.motion.position);
    this.syncShade();
  }

  /** Face a direction: yaw radians, or a grid delta { dx, dy }. */
  face(dir: FaceDir): void {
    this.targetYaw = typeof dir === 'number' ? dir : yawFromGridDelta(dir.dx, dir.dy);
  }

  /**
   * Carry a token (a `KeyToken.group`, a pearl, an ember) or `null` to let go.
   * Returns whatever was being carried, so the scene can put it back where it belongs.
   */
  carry(token: THREE.Object3D | null): THREE.Object3D | null {
    const previous = this.carried;
    if (previous) {
      this.carryAnchor.remove(previous);
      previous.position.set(0, 0, 0);
    }
    this.carried = token;
    if (token) {
      token.position.set(0, 0, 0);
      this.carryAnchor.add(token);
    }
    return previous;
  }

  get carrying(): THREE.Object3D | null {
    return this.carried;
  }

  /** Fade toward a resting glow (POMaze fog, MemoryChain darkness). */
  dim(amount = 0.45): void {
    this.targetBrightness = Math.max(0, amount);
  }

  /** Back to full glow. */
  undim(): void {
    this.targetBrightness = 1;
  }

  /** Celebration beat: a bright flash that decays over ~0.5 s. */
  flare(strength = 1): void {
    this.flareAmount = Math.max(this.flareAmount, strength);
  }

  /**
   * Hard snap for `scrubbed` / `teleport`: land the hop, kill the ribbon history, drop
   * every eased value to its target. No easing may cross a scrub (STYLE.md).
   */
  snap(): void {
    this.motion.snap();
    this.avatar.position.copy(this.motion.position);
    this.yaw = this.targetYaw;
    this.body.rotation.y = this.yaw;
    this.brightness = this.targetBrightness;
    this.flareAmount = 0;
    this.resetRibbon();
    this.syncShade();
    this.applyBrightness();
  }

  /** `dt` is wall seconds (ambient easing); `time` is the host playback clock. */
  update(dt: number, time: number): void {
    this.time = time;
    this.motion.update(time);
    this.avatar.position.copy(this.motion.position);

    // facing: shortest-arc damped turn
    const k = 1 - Math.exp(-12 * Math.max(0, dt));
    let d = this.targetYaw - this.yaw;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    this.yaw += d * k;
    this.body.rotation.y = this.yaw;

    // idle bob rides on the DECOR node, never on the bone: `body` stays at exactly
    // `height` so core.worldPos is the cell truth to the last bit. The orb keeps its
    // own life through a breath in SCALE — scale never moves an object's origin, so
    // the probe reads the same number whether the wisp is inhaling or not.
    this.decor.position.y = Math.sin(time * 1.9) * 0.018;
    const breath = 1 + Math.sin(time * 1.9) * 0.03;
    this.core.scale.setScalar(breath);
    // the shell breathes WITH the orb, so the dark ring keeps its width
    this.silhouette?.scale.setScalar(SILHOUETTE_SIZE * breath);
    if (this.skirt) {
      this.skirt.rotation.y = -time * 0.9;
      this.skirt.scale.setScalar(1 + Math.sin(time * 2.4) * 0.05);
    }
    this.carryAnchor.rotation.y = time * 1.2;

    if (this.flareAmount > 0) this.flareAmount = Math.max(0, this.flareAmount - dt / 0.5);
    this.brightness += (this.targetBrightness - this.brightness) * (1 - Math.exp(-6 * Math.max(0, dt)));
    this.applyBrightness();
    this.coreMaterial.oak.setTime(time);
    this.syncShade();
    this.pushRibbon();
  }

  dispose(): void {
    this.carry(null);
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.coreMaterial.dispose();
    this.haloMaterial.dispose();
    this.skirtMaterial?.dispose();
    this.ribbonMaterial?.dispose();
    this.silhouetteMaterial?.dispose();
    this.shadeMaterial?.dispose();
    this.shadeSprite?.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }

  // ---- internals ----

  private applyBrightness(): void {
    const gain = this.brightness * (1 + 1.8 * this.flareAmount);
    this.coreMaterial.oak.setRim(1.9 * gain);
    this.coreMaterial.pulse.value = this.flareAmount;
    setEmissiveIntensity(this.haloMaterial, this.baseIntensity * gain);
    if (this.ribbonMaterial) this.ribbonMaterial.opacity = 0.4 * this.brightness;
    if (this.skirtMaterial) this.skirtMaterial.opacity = 0.85 * this.brightness;
    // the dark side dims with the creature (POMaze fog, MemoryChain darkness) but is
    // deliberately deaf to `flare`: a celebration brightens a wisp, it does not deepen
    // the hole under it
    if (this.silhouetteMaterial) this.silhouetteMaterial.opacity = SILHOUETTE_OPACITY * this.brightness;
    if (this.shadeMaterial) this.shadeMaterial.opacity = SHADE_OPACITY * this.brightness * this.shadeLift;
  }

  /**
   * The contact shade is a pure function of the avatar's position: it lies on the FLOOR
   * plane (group space, y = SHADE_LIFT) under the wisp's x/z, and shrinks and fades with
   * the height of the hop arc — exactly what a contact shadow does. No history, so it is
   * scrub-safe by construction (canon lesson 2).
   */
  private syncShade(): void {
    const shade = this.shade;
    if (!shade) return;
    const lift = Math.max(0, this.avatar.position.y);
    const k = 1 / (1 + 2.6 * lift);
    shade.position.x = this.avatar.position.x;
    shade.position.z = this.avatar.position.z;
    shade.scale.setScalar(this.shadeSize * k);
    this.shadeLift = k;
    if (this.shadeMaterial) this.shadeMaterial.opacity = SHADE_OPACITY * this.brightness * k;
  }

  private resetRibbon(): void {
    const p = this.ribbonPositions;
    if (!p || !this.ribbonAttr) return;
    const x = this.avatar.position.x;
    const y = this.avatar.position.y + this.height;
    const z = this.avatar.position.z;
    for (let i = 0; i < this.ribbonCount; i++) {
      p[i * 3] = x;
      p[i * 3 + 1] = y;
      p[i * 3 + 2] = z;
    }
    this.ribbonAttr.needsUpdate = true;
    this.ribbonFilled = true;
  }

  private pushRibbon(): void {
    const p = this.ribbonPositions;
    if (!p || !this.ribbonAttr) return;
    if (!this.ribbonFilled) this.resetRibbon();
    // shift back one sample (fixed-size in-place move, no allocation)
    p.copyWithin(3, 0, (this.ribbonCount - 1) * 3);
    p[0] = this.avatar.position.x;
    p[1] = this.avatar.position.y + this.height;
    p[2] = this.avatar.position.z;
    this.ribbonAttr.needsUpdate = true;
  }
}

/** A ring of tapered petals, cupped downward. Built once per wisp. */
function buildSkirt(petals: number, radius: number): THREE.BufferGeometry {
  const positions = new Float32Array(petals * 3 * 3);
  const colors = new Float32Array(petals * 3 * 3);
  const inner = radius * 0.55;
  const outer = radius * 1.9;
  for (let i = 0; i < petals; i++) {
    const a0 = (i / petals) * Math.PI * 2;
    const a1 = ((i + 0.62) / petals) * Math.PI * 2;
    const mid = (a0 + a1) * 0.5;
    const o = i * 9;
    positions[o] = Math.cos(a0) * inner;
    positions[o + 1] = 0;
    positions[o + 2] = Math.sin(a0) * inner;
    positions[o + 3] = Math.cos(a1) * inner;
    positions[o + 4] = 0;
    positions[o + 5] = Math.sin(a1) * inner;
    positions[o + 6] = Math.cos(mid) * outer;
    positions[o + 7] = -radius * 0.5;
    positions[o + 8] = Math.sin(mid) * outer;
    for (let v = 0; v < 3; v++) {
      const k = v === 2 ? 0.35 : 1;
      colors[o + v * 3] = k;
      colors[o + v * 3 + 1] = k;
      colors[o + v * 3 + 2] = k;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.name = 'oakWispSkirt';
  return g;
}

/** Place any object at a cell (KeyToken drop points, beacons, camera targets). */
export function placeAtCell(object: THREE.Object3D, x: number, y: number, grid: GridSize, height = 0): void {
  cellToWorldVec(x, y, grid, object.position);
  object.position.y = height;
}
