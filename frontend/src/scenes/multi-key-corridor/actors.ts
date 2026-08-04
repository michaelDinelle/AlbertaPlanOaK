// Temple of Three Gates — the state-bearing actors (MultiKeyCorridor-cont-v0).
//
// Four kinds of thing live here, and all four obey the same accuracy law: the object a
// probe registers is a bare `root` whose world position is EXACTLY the cell sim-core
// says it is (y = 0, no hover, no bob, no growth). Every skin — every petal of mist,
// every rising stone, every scale-in — hangs off a CHILD. Delete the decoration and
// what remains is a correctly-placed empty node.
//
//   1. THE THREE GATES. One interface (`GateHandle`), three completely different
//      mechanisms, because the three gates are the three chapters of the env:
//        gate 0 · BRAMBLE — thorned canes laced across the doorway; opening unwinds
//                 them back onto the piers.
//        gate 1 · RIVER — a channel of water crossing the passage; opening surfaces
//                 three stepping stones out of it.
//        gate 2 · MIST — two curtain halves; opening parts them and lets them thin.
//      A gate KEEPS NO CLOCK AND NO HISTORY. `setProgress` and `setFlare` are pure
//      setters; the scene computes both from the frame's own `doorOpen[j]` bit and the
//      PLAYBACK-clock age of the door event (index.ts `gatePhase`). A paused frame
//      therefore freezes the mechanism exactly as it freezes the wisp, playback speed
//      scales it like every other beat in the temple, and a scrub — whose clock is
//      settled — lands on the state the frame describes with no easing at all.
//
//   2. THE KEY ALTARS. A pedestal at `state.keys[j]` — which the decoder REPLAYS from
//      the ctrl stream, since key cells are not in the record — carrying the gem while
//      `!hasKey[j]`. The pedestal stays at the cell when the key is taken (the socket it
//      came from is honest), and hard-snaps to the newly drawn cell on a respawn.
//
//   3. THE CARRY RACK. Three motes orbiting the wisp's carry anchor, one per key held.
//      This is decoration only: it is the ALTAR that is probed.
//
//   4. THE SHRINE. The relic on the live goal cell, growing in after every respawn.

import { GoalBeacon, KeyToken } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { fbmField, makeDataTexture } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import { makeWater, type WaterHandle } from '@kit/water';
import * as THREE from 'three';
import {
  PORTAL_LINTEL_Y,
  blendTokens,
  clamp01,
  mergeColoredParts,
  type ColoredPart,
  type TempleSkin,
} from './world';

/**
 * Seconds for a gate to swing fully open — measured on the PLAYBACK clock, so it is
 * five 8 Hz control steps of the stream, at whatever speed the stream is being played.
 */
export const GATE_SECONDS = 0.5;

/** How long the "this gate just gave way" beat burns, also in playback seconds. */
export const GATE_FLARE_SECONDS = 0.6;

// ---------------------------------------------------------------------------
// the shared gate contract
// ---------------------------------------------------------------------------

export interface GateHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(door cell), y = 0, always */
  root: THREE.Object3D;
  /** 0 = shut, 1 = fully open — what the mechanism is currently posed at */
  readonly progress: number;
  /**
   * Pose the mechanism. A PURE SETTER holding no clock and no history: the scene hands
   * it `gatePhase(doorOpen[j], playbackTime − doorAt[j])`, so the same frame at the same
   * playback time always produces the same gate, and a settled clock lands on the frame's
   * bit exactly (0 or 1) — which is what makes a scrub, a teleport and a relock hard cuts.
   */
  setProgress(progress01: number): void;
  /** 0..1 flourish weight for the beat when this gate is the one that gave way */
  setFlare(amount01: number): void;
  /** ctx.time() — the idle life (bramble tremble, water shimmer, mist breath) */
  setTime(t: number): void;
  dispose(): void;
}

export interface GateOptions {
  tier: QualityTier;
  palette: Palette;
  skin: TempleSkin;
  /** the passage runs along ±x (read from the walls, never assumed) */
  facesX: boolean;
  /** the shared temple stone material (vertex-colored) */
  stone: OakMaterial;
  name?: string;
}

/** eased 0..1 with a smooth start and end — a gate never snaps to a stop */
function ease(t: number): number {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
}

// ---------------------------------------------------------------------------
// gate 0 — the bramble
// ---------------------------------------------------------------------------

/**
 * Thorned canes laced across the doorway. Closed, eight canes span the passage at eight
 * heights; opening rotates each one up onto its pier, shortens it and lets it thin
 * away. They are eight INSTANCES of one thorned-cane geometry, re-posed from a single
 * scalar, so the whole mechanism is one draw call at every tier.
 */
export function buildBrambleGate(opts: GateOptions): GateHandle {
  const { tier, palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'brambleGate';
  const root = new THREE.Object3D();
  root.name = 'brambleGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'brambleGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  const canes = tier === 'thumb' ? 4 : tier === 'medium' ? 6 : 8;
  const geometry = makeCaneGeometry(tier, skin.bramble, skin.brambleLeaf);
  const material: OakMaterial = oakMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.88,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    // the canes tremble where they are laced; the phase comes from the instance origin
    wind: { amplitude: 0.012, frequency: 1.35, height: 1.1 },
    tier,
    name: 'brambleCane',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, canes);
  mesh.name = 'brambleCanes';
  mesh.castShadow = tier !== 'thumb';
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  pivot.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'ZYX');
  const scale = new THREE.Vector3(1, 1, 1);
  const flareColor = (skin.keys[0] ?? new THREE.Color(0xffffff)).clone();
  let progress = 0;
  let flareAmount = 0;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < canes; i++) {
      const t = canes > 1 ? i / (canes - 1) : 0.5;
      // closed: laced across the doorway at rising heights, alternating lean
      const lean = (i % 2 === 0 ? 1 : -1) * (0.12 + 0.1 * t);
      const y = 0.1 + t * (PORTAL_LINTEL_Y - 0.2);
      const side = i % 2 === 0 ? -1 : 1;
      // open: swung up onto its own pier, shortened, and turned edge-on
      position.set(0, y + e * 0.06 * side, side * e * 0.36);
      euler.set(lean * (1 - e) + e * side * 1.1, e * side * 0.5, 0);
      quaternion.setFromEuler(euler);
      const span = 1 - 0.86 * e;
      scale.set(1 - 0.3 * e, 1 - 0.3 * e, span);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  apply();

  const applyFlare = (): void => {
    // the canes glow faintly as they part — the only light a bramble ever gives, and it
    // is written EVERY time from the amount, so amount 0 puts the emissive back to black
    material.emissive.copy(flareColor).multiplyScalar(flareAmount * 0.6);
  };
  applyFlare();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      applyFlare();
    },
    setTime(t) {
      oakUniforms(material)?.setTime(t);
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * One thorned cane: a tapered bar along ±z with thorns, and — the part that makes the
 * gate READ — a run of leaves lying nearly flat.
 *
 * A cane is a 0.03-radius rod, and this scene's camera is a 3/4 aerial: seen from above,
 * eight bare rods across a doorway project about as much area as a pencil sketch, and
 * the bramble gate was invisible in the composed frame. Leaves are horizontal surface,
 * which is the only kind an overhead camera actually collects.
 */
function makeCaneGeometry(
  tier: QualityTier,
  dark: THREE.ColorRepresentation,
  light: THREE.ColorRepresentation,
): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const bar = new THREE.CylinderGeometry(0.03, 0.036, 0.94, tier === 'thumb' ? 4 : 6, 1);
  bar.rotateX(Math.PI / 2);
  parts.push({ geometry: bar, color: dark });

  const thorns = tier === 'thumb' ? 2 : 4;
  for (let i = 0; i < thorns; i++) {
    const t = (i + 0.5) / thorns;
    const thorn = new THREE.ConeGeometry(0.022, 0.075, 4, 1);
    thorn.rotateZ(i % 2 === 0 ? Math.PI / 2.4 : -Math.PI / 2.4);
    thorn.rotateY(i * 1.9);
    thorn.translate(0, 0, (t - 0.5) * 0.86);
    parts.push({ geometry: thorn, color: dark });
  }

  const leaves = tier === 'thumb' ? 3 : 6;
  for (let i = 0; i < leaves; i++) {
    const t = (i + 0.5) / leaves;
    const side = i % 2 === 0 ? 1 : -1;
    const leaf = new THREE.CircleGeometry(0.085, 4);
    leaf.scale(1, 0.52, 1);
    leaf.rotateX(-Math.PI / 2 + side * 0.35); // nearly flat: area for the aerial camera
    leaf.rotateY(side * 0.7 + i * 0.5);
    leaf.translate(side * 0.075, 0.02, (t - 0.5) * 0.82);
    parts.push({ geometry: leaf, color: i % 3 === 0 ? dark : light });
  }
  return mergeColoredParts(parts, 'brambleCane');
}

// ---------------------------------------------------------------------------
// gate 1 — the river
// ---------------------------------------------------------------------------

/**
 * A channel of water crossing the passage. The basin runs UNDER the flanking wall
 * cells, so the water reads as a river the temple was built over rather than as a
 * puddle in a doorway. Opening surfaces three stepping stones out of it; closing sinks
 * them again. The water is the kit's — a scene spends nothing from its one-bespoke-
 * shader budget by using it.
 */
export function buildRiverGate(opts: GateOptions): GateHandle {
  const { tier, palette, skin } = opts;
  const underwater = TOKENS.palettes.underwater;
  const group = new THREE.Group();
  group.name = opts.name ?? 'riverGate';
  const root = new THREE.Object3D();
  root.name = 'riverGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'riverGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  // The ford's caustic web has exactly two states and both are pure: the idle whisper it
  // was BUILT with, and that plus the flare the scene is currently asking for. There is
  // no third, remembered value — the old `0.5 + 1.4·flare` write left 0.5 latched on the
  // uniform forever, and no scrub could clear it.
  const CAUSTIC_IDLE = tier === 'thumb' ? 0 : 0.26;
  const CAUSTIC_FLARE = tier === 'thumb' ? 0 : 1.4;

  // ---- the basin: a stone lip around a sunken channel ----
  const basinGeometry = makeBasinGeometry(skin);
  const basinMesh = new THREE.Mesh(basinGeometry, opts.stone);
  basinMesh.name = 'riverBasin';
  basinMesh.castShadow = false;
  basinMesh.receiveShadow = tier !== 'thumb';
  pivot.add(basinMesh);

  // ---- the water ----
  // The colours are a cross-palette blend into `underwater` — STYLE.md's legal
  // off-palette move — because morning's own blues are pale enough to read as grey
  // stone next to a temple pier, and a river that reads as stone is not a river.
  //
  // `depthFalloff` is the trap here: the kit's depth gradient runs on VIEW depth, not
  // water depth, so at this scene's 17-unit standoff a 3.5-unit falloff painted the
  // whole basin its deepest colour. 45 keeps the ford in its shallows where a ford
  // belongs.
  const water: WaterHandle = makeWater({
    // NOTE: no `shallow`/`deep` here on purpose — see `setColors` below. The kit's
    // constructor lerps whatever it is given 35% toward `palette.groundB` and 55%
    // toward `palette.skyBottom`, which is right for open water under the scene's own
    // sky and wrong for a ford that has to stay blue in an olive-and-stone temple.
    // one cell wide, and long enough in z to slip under both flanking wall blocks, so
    // the ford reads as a channel the temple was built over rather than a puddle
    size: [0.99, 2.1],
    palette,
    tier,
    // a 2-unit strip: the kit's default 96 segments would be 18k triangles of pure waste
    segments: 26,
    sky: palette.skyHorizon,
    amplitude: 0.011,
    waveScale: 2.1,
    speed: 0.9,
    fresnel: 0,
    caustics: CAUSTIC_IDLE,
    depthFalloff: 45,
    opacity: 0.72,
    level: 0.035,
    name: 'riverWater',
  });
  // The ford is a HALF-CELL of blue in an olive-and-stone frame, and every additive
  // term the kit stacks on water — caustics, fresnel, a 1.6 specular glint tuned for
  // open water seen from a shore — desaturates it toward white before ACES has even
  // had its turn. A pure-red diagnostic rendered SALMON here. So: the glint is off, the
  // caustics are a whisper, and the colour is `underwater` at full strength. It is the
  // one saturated hue in the temple and it has to survive.
  //
  // `setColors` is the kit's VERBATIM override — the one place a scene can state a water
  // colour and get exactly that colour, with no palette lerp on top — so the ford is the
  // cross-palette blend it is supposed to be and not the morning sky twice over. Passing
  // `shallow`/`deep` to the constructor instead lands on #d2e5d5 / #e7ece2: a pale mint
  // wedge indistinguishable from the mist gate two chambers along.
  water.setColors(
    // shallow: underwater's own cerulean carried a third of the way to its reef teal —
    // the colour a hand's depth of clean water over a dark bed actually is
    blendTokens(underwater.skyTop, underwater.groundB, 0.32),
    // deep: straight underwater trench blue, for the troughs and the far lip
    underwater.skyBottom,
  );
  water.uniforms.uSpecular.value = 0;
  pivot.add(water.mesh);

  // ---- the three stepping stones ----
  const stoneCount = 3;
  const stoneGeometry = new THREE.CylinderGeometry(0.19, 0.22, 0.14, tier === 'thumb' ? 5 : 7, 1);
  const stoneMaterial: OakMaterial = oakMaterial({
    // wet stone, not temple stone: a sunk stone in `stoneLight` is three pale discs
    // glowing through a 60%-opaque surface, and they read as the water instead of the
    // water reading as water
    color: blendTokens(skin.stoneLight, skin.riverBed, 0.4),
    roughness: 0.95,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.28,
    tier,
    name: 'riverSteppingStone',
  });
  const stones = new THREE.InstancedMesh(stoneGeometry, stoneMaterial, stoneCount);
  stones.name = 'riverSteppingStones';
  stones.castShadow = tier !== 'thumb';
  stones.receiveShadow = tier !== 'thumb';
  stones.frustumCulled = false;
  pivot.add(stones);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3(1, 1, 1);
  let progress = 0;
  let flareAmount = 0;
  const SUNK_Y = -0.03;
  const SURFACE_Y = 0.1;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < stoneCount; i++) {
      const t = (i - (stoneCount - 1) / 2) / stoneCount;
      // stones step ACROSS the passage (along x); a small z stagger makes it a ford
      // rather than a bridge
      const stagger = (i % 2 === 0 ? 1 : -1) * 0.11;
      // they surface in sequence, near side first
      const local = clamp01((e - i * 0.12) / 0.76);
      position.set(t * 0.66, SUNK_Y + (SURFACE_Y - SUNK_Y) * ease(local), stagger);
      euler.set(0, i * 0.7, 0);
      quaternion.setFromEuler(euler);
      const s = 0.72 + 0.28 * ease(local);
      scale.set(s, 1, s);
      matrix.compose(position, quaternion, scale);
      stones.setMatrixAt(i, matrix);
    }
    stones.instanceMatrix.needsUpdate = true;
  };
  apply();

  const applyFlare = (): void => {
    // the ford flashes as the stones break the surface, and settles back onto the exact
    // value it was constructed with — the idle web, never a remembered peak
    water.uniforms.uCausticStrength.value = CAUSTIC_IDLE + CAUSTIC_FLARE * flareAmount;
  };
  applyFlare();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      applyFlare();
    },
    setTime(t) {
      water.setTime(t);
      oakUniforms(stoneMaterial)?.setTime(t);
    },
    dispose() {
      stones.dispose();
      stoneGeometry.dispose();
      stoneMaterial.dispose();
      water.dispose();
      basinGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * The river's bed: one dark slab just under the surface.
 *
 * THE FORD IS BRIM-FULL, NOT SUNKEN, and that is an accuracy decision as much as an art
 * one. Every gate cell has a wall cell immediately south of it, and from this scene's
 * aerial anything set into the floor of a gate is behind that wall: the first pass put
 * the surface 60 mm down in a channel with stone kerbs and it was, from the composed
 * frame, invisible — a magenta-water diagnostic proved the mesh was there and simply
 * could not be seen. At floor level the water is exactly as visible as the flagstone it
 * replaces, which is the most a one-cell river can ask for.
 */
function makeBasinGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  // No kerbs. The adjacent flagstones ARE the banks, and a pair of stone lips inside a
  // one-cell channel spends a third of the only water this scene has on masonry.
  // Shallow, not a well: the bed has to read THROUGH the water, or the ford is a black
  // hole in the floor rather than a river.
  // A DARK bed, seen THROUGH a 60%-opaque surface, is what makes the ford read: the
  // temple floor is pale stone, so the one thing that must not happen is a pale patch of
  // water on it. Dark blue under a bright caustic web reads as water at forty pixels.
  const bed = new THREE.BoxGeometry(0.99, 0.06, 2.1);
  bed.translate(0, 0.005, 0);
  parts.push({ geometry: bed, color: skin.riverBed });
  return mergeColoredParts(parts, 'riverBasin');
}

// ---------------------------------------------------------------------------
// gate 2 — the mist
// ---------------------------------------------------------------------------

/**
 * Two curtain halves of standing mist over a pool of it. Opening parts the halves along
 * the doorway, thins them, and lets the pool shrink back to the piers.
 *
 * Each half is one or two soft quads whose alpha is BAKED (fbm × an edge feather)
 * rather than scrolled, so a quad can be moved without dragging its own silhouette
 * around with it — the feather travels with the mesh and the edges never show.
 *
 * THE POOL IS NOT DECORATION. This scene is a 3/4 aerial, and a vertical curtain
 * presents almost no area to a camera looking down at it: the first pass had a
 * beautiful mist gate that was, from the composed frame, an empty doorway. The
 * ground-hugging plane is the half of the mechanism the camera can actually see, and it
 * carries the same openness.
 */
export function buildMistGate(opts: GateOptions): GateHandle {
  const { tier, palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'mistGate';
  const root = new THREE.Object3D();
  root.name = 'mistGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'mistGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  const layers = tier === 'thumb' ? 1 : 2;
  const halves = 2;
  const geometry = new THREE.PlaneGeometry(1.02, PORTAL_LINTEL_Y - 0.06);
  geometry.name = 'mistCurtain';
  geometry.rotateY(Math.PI / 2); // the curtain spans the doorway (±z), facing ±x

  // near-white (noon.sun is the sanctioned white) with a morning-blue cast
  const color = blendTokens(TOKENS.palettes.noon.sun, palette.hemiSky, 0.45);
  const textures: THREE.DataTexture[] = [];
  const materials: EmissiveMaterial[] = [];
  const meshes: THREE.Mesh[] = [];
  for (let l = 0; l < layers; l++) {
    const texture = makeMistTexture(0x51ed270b + l * 977);
    textures.push(texture);
    const material = emissiveMaterial(color, 0.95, {
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      side: THREE.DoubleSide,
      name: `mistCurtainLayer${l}`,
    });
    material.alphaMap = texture;
    materials.push(material);
    for (let h = 0; h < halves; h++) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `mistCurtain${l}_${h}`;
      mesh.renderOrder = 2;
      mesh.frustumCulled = false;
      pivot.add(mesh);
      meshes.push(mesh);
    }
  }

  // the pool: the horizontal half of the mechanism, and the half this camera sees
  const poolGeometry = new THREE.PlaneGeometry(0.96, 0.86);
  poolGeometry.name = 'mistPool';
  poolGeometry.rotateX(-Math.PI / 2);
  const poolMaterial = emissiveMaterial(color, 0.9, {
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
    side: THREE.DoubleSide,
    name: 'mistPool',
  });
  poolMaterial.alphaMap = textures[0] ?? null;
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'mistPool';
  pool.renderOrder = 2;
  pool.frustumCulled = false;
  pool.position.y = 0.06;
  pivot.add(pool);

  let progress = 0;
  let flareAmount = 0;
  let breath = 0;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < meshes.length; i++) {
      const mesh = meshes[i] as THREE.Mesh;
      const layer = Math.floor(i / halves);
      const side = i % halves === 0 ? -1 : 1;
      const drift = Math.sin(breath * (0.7 + 0.3 * layer) + layer * 2.1 + side) * 0.05;
      // parting: each half slides to its pier and shrinks to nothing across the gap
      mesh.position.set(layer === 0 ? 0.03 : -0.03, (PORTAL_LINTEL_Y - 0.06) * 0.5 + drift * 0.6, side * (0.26 + 0.3 * e) + drift);
      mesh.scale.set(1, 1 + 0.04 * Math.sin(breath * 0.8 + layer), Math.max(0.001, 0.55 - 0.5 * e));
    }
    for (let l = 0; l < materials.length; l++) {
      const material = materials[l] as EmissiveMaterial;
      const fade = (1 - e) * (0.5 + 0.14 * Math.sin(breath * 0.9 + l * 1.7));
      material.opacity = Math.max(0, fade);
      setEmissiveIntensity(material, 0.95 * (1 + 1.6 * flareAmount));
    }
    // the pool draws back to the piers and fades as the gate gives way
    pool.scale.set(1, 1, Math.max(0.001, 1 - 0.86 * e));
    pool.position.y = 0.06 + 0.012 * Math.sin(breath * 0.7);
    poolMaterial.opacity = Math.max(0, (1 - e) * (0.42 + 0.1 * Math.sin(breath * 0.6)));
    setEmissiveIntensity(poolMaterial, 0.9 * (1 + 1.6 * flareAmount));
  };
  apply();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      apply();
    },
    setTime(t) {
      breath = t;
      apply();
    },
    dispose() {
      geometry.dispose();
      poolGeometry.dispose();
      poolMaterial.dispose();
      for (const m of materials) m.dispose();
      for (const t of textures) t.dispose();
      materials.length = 0;
      textures.length = 0;
      meshes.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * A mist alpha map: fbm × a radial edge feather, so the quad's own rectangle can never
 * be seen. Built from the kit's fbm field — no image, no network, 128² of RGBA.
 */
function makeMistTexture(seed: number): THREE.DataTexture {
  const size = 128;
  const field = fbmField(seed, { size, frequency: 3, octaves: 4, gain: 0.55 });
  const rgba = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const u = (x / (size - 1)) * 2 - 1;
      const v = (y / (size - 1)) * 2 - 1;
      // feather hardest at the top and the sides; the mist pools at the floor
      const edge = Math.max(0, 1 - u * u * u * u) * Math.max(0, 1 - Math.pow(Math.max(0, v), 3));
      const n = Math.max(0, Math.min(1, (field.data[i] ?? 0) * 1.75 - 0.42));
      const a = Math.round(255 * Math.max(0, Math.min(1, n * edge)));
      rgba[i * 4] = 255;
      rgba[i * 4 + 1] = 255;
      rgba[i * 4 + 2] = 255;
      rgba[i * 4 + 3] = a;
    }
  }
  const texture = makeDataTexture(rgba, size, size, 'mistCurtainAlpha');
  return texture;
}

// ---------------------------------------------------------------------------
// the thumb doorway bank — three gates, one draw
// ---------------------------------------------------------------------------

export interface ThumbGateBankOptions {
  tier: QualityTier;
  skin: TempleSkin;
  /** true when the passage runs along ±x, per gate, read from the walls */
  facesX: readonly boolean[];
  count: number;
  name?: string;
}

export interface ThumbGateBank {
  /** add this ONE group to the scene, alongside the per-gate groups */
  group: THREE.Group;
  /** one handle per gate, in door order — the same contract the mechanisms honour */
  gates: readonly GateHandle[];
  dispose(): void;
}

/**
 * Height of the thumb doorway plug. Deliberately LOW: the mist gate learned this lesson
 * first (a tall curtain presents almost no area to a camera looking down at it), and the
 * thumb rig is the most aerial camera in the gallery. The plug fills the doorway CELL,
 * so what the wall view sees is a coloured plate in the gap that is simply gone once the
 * gate opens.
 */
const THUMB_LEAF_H = PORTAL_LINTEL_Y * 0.3;

/**
 * The three gates AT THUMB, as one InstancedMesh.
 *
 * The bramble, the river and the mist are three mechanisms because they are the three
 * chapters of the env — at hero and medium they stay exactly that. At thumb they cost
 * SEVEN of the forty draw calls STYLE.md allows the whole wall-view frame, to render
 * three ~10 px thresholds: the bramble's canes, the river's basin + water + stones, the
 * mist's curtain + pool. So thumb collapses all three into one instanced doorway — one
 * leaf per gate, tinted to its own chapter (bramble green / river blue / mist white) so
 * the three chapters still read, retracting into the lintel as the gate opens and
 * ZERO-SCALED once it is fully open (canon lesson 14: a cell that wants something else
 * collapses its instance; the array stays untouched).
 *
 * Every handle is still a real `GateHandle`: its `root` is a bare node on the door cell
 * (the probes and `assertAligned` never notice the difference), and `setProgress` stays
 * a pure setter with no clock. `setFlare` is the one thing that does not survive — the
 * thumb tier runs with no post chain, so a bloom pulse has nothing to feed.
 */
export function buildThumbGateBank(opts: ThumbGateBankOptions): ThumbGateBank {
  const group = new THREE.Group();
  group.name = opts.name ?? 'thumbGateBank';
  const count = Math.max(1, opts.count);

  // Spans ±z with its face along ±x, exactly like the mist gate's curtain, so the shared
  // `facesX ? 0 : PI/2` pivot rotation puts it ACROSS the passage and not along it.
  const geometry = new THREE.BoxGeometry(0.84, THUMB_LEAF_H, 1.0);
  geometry.translate(0, THUMB_LEAF_H / 2, 0); // pivot at the floor: it retracts upward
  geometry.name = 'thumbGateLeaf';
  const material = oakMaterial({
    color: 0xffffff, // the per-gate hue rides instanceColor (grid canon, lesson 13)
    roughness: 0.85,
    flatShading: true,
    toonSteps: 2,
    tier: opts.tier,
    name: 'thumbGateLeaf',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = 'thumbGateLeaves';
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  group.add(mesh);

  // chapter tints: the woody bramble, the river's own bed-read water, the mist's white
  const tints: readonly THREE.Color[] = [
    gateChapterTint(opts.skin, 0),
    gateChapterTint(opts.skin, 1),
    gateChapterTint(opts.skin, 2),
  ];

  const roots: THREE.Object3D[] = [];
  const progresses = new Float32Array(count);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();

  const apply = (j: number): void => {
    const root = roots[j];
    if (!root) return;
    // eased, and identically 0 at progress 1 — a fully open gate submits no area at all
    const shut = 1 - ease(progresses[j] ?? 0);
    position.copy(root.position);
    euler.set(0, opts.facesX[j] === true ? 0 : Math.PI / 2, 0);
    quaternion.setFromEuler(euler);
    scale.set(1, shut, 1);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(j, matrix);
    mesh.instanceMatrix.needsUpdate = true;
  };

  const gates: GateHandle[] = [];
  for (let j = 0; j < count; j++) {
    const gateGroup = new THREE.Group();
    gateGroup.name = `gate${j}`;
    const root = new THREE.Object3D();
    root.name = `thumbGateRoot${j}`;
    gateGroup.add(root);
    roots.push(root);
    const tint = tints[j] ?? tints[0] ?? new THREE.Color(0xffffff);
    mesh.setColorAt(j, tint);
    gates.push({
      group: gateGroup,
      root,
      get progress(): number {
        return progresses[j] ?? 0;
      },
      setProgress(progress01: number) {
        progresses[j] = clamp01(progress01);
        apply(j);
      },
      setFlare() {
        /* no post chain at thumb — nothing for a bloom pulse to feed */
      },
      setTime() {
        /* the idle life (tremble, shimmer, breath) is a hero/medium luxury */
      },
      dispose() {
        gateGroup.removeFromParent();
        gateGroup.clear();
      },
    });
    apply(j);
  }
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  return {
    group,
    gates,
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * The chapter tint of gate `j` — the one colour each full mechanism is built around, so
 * the collapsed thumb doorway still says bramble / river / mist at a glance.
 */
export function gateChapterTint(s: TempleSkin, j: number): THREE.Color {
  if (j === 0) return blendTokens(s.brambleLeaf, s.bramble, 0.35);
  if (j === 1) return blendTokens(s.riverBed, TOKENS.palettes.morning.hemiSky, 0.55);
  return blendTokens(TOKENS.palettes.noon.sun, TOKENS.palettes.morning.hemiSky, 0.45);
}

// ---------------------------------------------------------------------------
// the key altars
// ---------------------------------------------------------------------------

export interface KeyAltarOptions {
  tier: QualityTier;
  palette: Palette;
  skin: TempleSkin;
  /** which key this altar holds: 0 seed, 1 pearl, 2 ember */
  index: number;
  stone: OakMaterial;
  name?: string;
}

export interface KeyAltarHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(state.keys[j]), y = 0, always */
  root: THREE.Object3D;
  /** the gem's height — the scene lifts a point light to it on a pickup */
  readonly gemHeight: number;
  /** the key is on the wisp, not on this altar */
  setCarried(carried: boolean): void;
  /** 0..1 scale-in after a respawn (a new key does not slide in from the old cell) */
  setGrow(grow01: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

/**
 * A pedestal on the key's exact cell with the gem hovering over it. The pedestal does
 * NOT move when the key is taken — the empty socket is the honest picture of a key that
 * has been picked up — and hard-snaps to the new cell when the env draws one.
 */
export function buildKeyAltar(opts: KeyAltarOptions): KeyAltarHandle {
  const { tier, palette, skin, index } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? `keyAltar${index}`;
  const root = new THREE.Object3D();
  root.name = `keyAltarRoot${index}`;
  group.add(root);

  // growth + bob live on the decoration node, never on the root
  const decor = new THREE.Group();
  decor.name = `keyAltarDecor${index}`;
  root.add(decor);

  const color = skin.keys[index] ?? new THREE.Color(0xffffff);
  const pedestalGeometry = makePedestalGeometry(skin);
  const pedestal = new THREE.Mesh(pedestalGeometry, opts.stone);
  pedestal.name = `keyPedestal${index}`;
  pedestal.castShadow = tier !== 'thumb';
  pedestal.receiveShadow = tier !== 'thumb';
  decor.add(pedestal);

  // the kit token, parented at the local origin so it inherits the exact cell
  const token = new KeyToken({
    color,
    tier,
    radius: 0.1,
    hover: 0.42,
    intensity: index === 1 ? 1.5 : 1.9, // the pearl is white: it needs less gain to read
    name: `keyToken${index}`,
  });
  if (tier === 'thumb') {
    // the thumb tier has 40 draw calls for the whole temple and the gem is 8 px across:
    // its additive halo buys nothing there and costs one call per key
    const halo = token.group.getObjectByName('oakKeyTokenHalo');
    if (halo) halo.visible = false;
  }
  decor.add(token.group);

  let carried = false;
  const gemHeight = 0.42;

  return {
    group,
    root,
    gemHeight,
    setCarried(v) {
      carried = v;
      token.setVisible(!v);
    },
    setGrow(grow01) {
      decor.scale.setScalar(clamp01(grow01));
    },
    setTime(t) {
      // a slow sway on the socket's flowers; the pedestal itself is stone and still
      decor.rotation.y = Math.sin(t * 0.4 + index) * 0.05;
    },
    update(dt, time) {
      if (!carried) token.update(dt, time);
    },
    dispose() {
      token.dispose();
      pedestalGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A lotus socket: a stepped stone base with four petal lobes cupping the gem. */
function makePedestalGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const base = new THREE.CylinderGeometry(0.18, 0.23, 0.09, 8, 1);
  base.translate(0, 0.045, 0);
  parts.push({ geometry: base, color: skin.stoneDark });
  const drum = new THREE.CylinderGeometry(0.1, 0.14, 0.13, 8, 1);
  drum.translate(0, 0.155, 0);
  parts.push({ geometry: drum, color: skin.stoneLight });
  for (let i = 0; i < 4; i++) {
    const petal = new THREE.ConeGeometry(0.07, 0.14, 4, 1);
    petal.rotateX(0.85);
    petal.rotateY((i / 4) * Math.PI * 2);
    petal.translate(Math.cos((i / 4) * Math.PI * 2) * 0.1, 0.25, Math.sin((i / 4) * Math.PI * 2) * 0.1);
    parts.push({ geometry: petal, color: skin.stoneLight });
  }
  return mergeColoredParts(parts, 'keyPedestal');
}

// ---------------------------------------------------------------------------
// the carry rack — three motes orbiting the wisp
// ---------------------------------------------------------------------------

export interface CarryRackHandle {
  /** hand this to `Wisp.carry()` — it rides the avatar's carry anchor */
  group: THREE.Group;
  setCarried(index: number, carried: boolean): void;
  setTime(t: number): void;
  dispose(): void;
}

/**
 * The keys the wisp is holding, one mote per key, orbiting on the kit's carry anchor
 * (which spins on `ctx.time()`, so a paused frame holds them mid-orbit). Pure
 * decoration: what a test checks is the ALTAR, and what the frame states is
 * `hasKey[j]`.
 *
 * THREE MOTES, ONE DRAW. They are instances of one octahedron with per-instance colour
 * (three defines `USE_INSTANCING_COLOR` the moment an InstancedMesh has an
 * `instanceColor`, with no `vertexColors` needed — the same trick `buildCellField` uses
 * for per-cell hues), and "not carried" is a zero-scale instance. Three separate meshes
 * cost three draws in every frame of every tier, and the thumb tier has forty to spend
 * on the whole temple.
 */
export function buildCarryRack(skin: TempleSkin, tier: QualityTier): CarryRackHandle {
  const group = new THREE.Group();
  group.name = 'keyCarryRack';
  const geometry = new THREE.OctahedronGeometry(0.062, tier === 'thumb' ? 0 : 1);
  // white base: the HDR gain per key is baked into its instance colour, so the pearl
  // (which is already near-white) does not out-bloom the seed and the ember
  const material = emissiveMaterial(0xffffff, 1, { name: 'carriedKeys' });
  const mesh = new THREE.InstancedMesh(geometry, material, 3);
  mesh.name = 'carriedKeys';
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(9), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < 3; i++) {
      c.copy(skin.keys[i] ?? new THREE.Color(0xffffff)).multiplyScalar(i === 1 ? 1.5 : 2.0);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  group.add(mesh);

  const carried = [false, false, false];
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3(1, 1, 1);

  const apply = (t: number): void => {
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      const bob = Math.sin(t * 2.2 + i * 2.1) * 0.03;
      position.set(Math.cos(a) * 0.2, 0.02 * (i - 1) + bob, Math.sin(a) * 0.2);
      euler.set(t * 0.9 + i, t * 1.3, 0);
      quaternion.setFromEuler(euler);
      scale.setScalar(carried[i] ? 1 : 0);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  apply(0);

  let time = 0;
  return {
    group,
    setCarried(index, on) {
      if (index < 0 || index > 2) return;
      if (carried[index] === on) return;
      carried[index] = on;
      apply(time);
    },
    setTime(t) {
      time = t;
      apply(t);
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// the shrine — the relic on the live goal cell
// ---------------------------------------------------------------------------

export interface ShrineOptions {
  tier: QualityTier;
  palette: Palette;
  skin: TempleSkin;
  stone: OakMaterial;
  name?: string;
}

export interface ShrineHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(gx, gy), y = 0, always */
  root: THREE.Object3D;
  /** local height of the relic — the scene lifts its point light to it */
  readonly crownHeight: number;
  /** 0..1 grow-in after a respawn; 0..1 "the wisp is close" opening */
  setState(near01: number, grow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

/**
 * A stepped plinth with the relic turning over it. `root` never moves off the goal
 * cell; growth is a scale on `decor`, so a relic that is 3% grown is still exactly
 * where sim-core says the goal is.
 */
export function buildShrine(opts: ShrineOptions): ShrineHandle {
  const { tier, palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sanctumShrine';
  const root = new THREE.Object3D();
  root.name = 'shrineRoot';
  group.add(root);
  const decor = new THREE.Group();
  decor.name = 'shrineDecor';
  root.add(decor);

  const plinthGeometry = makeShrineGeometry(skin, tier);
  const plinth = new THREE.Mesh(plinthGeometry, opts.stone);
  plinth.name = 'shrinePlinth';
  plinth.castShadow = tier !== 'thumb';
  plinth.receiveShadow = tier !== 'thumb';
  decor.add(plinth);

  // the kit beacon, demoted to a decoration child at the local origin — never setCell'd
  const beacon = new GoalBeacon({
    color: skin.relic,
    tier,
    height: 1.0,
    radius: 0.11,
    intensity: 0.5,
    name: 'shrineGlow',
  });
  decor.add(beacon.group);

  const CROWN = 0.86;
  const relicGeometry = new THREE.OctahedronGeometry(0.125, tier === 'thumb' ? 0 : 1);
  // The relic is the one thing in this temple that crosses the bloom threshold, and it
  // is deliberately modest: stacked on the beacon column and its ground disc, a fat
  // emitter turns the whole sanctum into one white ball three cells across.
  const RELIC_HDR = 1.3;
  const relicMaterial: EmissiveMaterial = emissiveMaterial(skin.relic, RELIC_HDR, { name: 'shrineRelic' });
  const relic = new THREE.Mesh(relicGeometry, relicMaterial);
  relic.name = 'shrineRelic';
  relic.position.y = CROWN;
  decor.add(relic);

  // a ring of four gold blades around the relic — the silhouette that says "shrine"
  const bladeGeometry = new THREE.BoxGeometry(0.03, 0.2, 0.06);
  const bladeMaterial: OakMaterial = oakMaterial({
    color: blendTokens(skin.relic, palette.groundB, 0.25),
    roughness: 0.6,
    metalness: 0.15,
    flatShading: true,
    rimColor: skin.relic,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.34,
    tier,
    name: 'shrineBlades',
  });
  const blades = new THREE.InstancedMesh(bladeGeometry, bladeMaterial, 4);
  blades.name = 'shrineBlades';
  blades.castShadow = false;
  blades.frustumCulled = false;
  {
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2;
      position.set(Math.cos(a) * 0.17, CROWN, Math.sin(a) * 0.17);
      euler.set(0, -a, 0.5);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      blades.setMatrixAt(i, matrix);
    }
    blades.instanceMatrix.needsUpdate = true;
  }
  decor.add(blades);

  let near = 0;

  return {
    group,
    root,
    crownHeight: CROWN,
    setState(near01, grow01) {
      near = clamp01(near01);
      const grow = clamp01(grow01);
      decor.scale.setScalar(grow);
      relic.scale.setScalar(0.82 + 0.3 * near);
      setEmissiveIntensity(relicMaterial, RELIC_HDR * (0.6 + 0.7 * near) * (0.3 + 0.7 * grow));
    },
    flare(strength = 1) {
      beacon.flare(strength);
    },
    setTime(t) {
      relic.rotation.y = t * 0.8;
      relic.position.y = CROWN + Math.sin(t * 1.5) * 0.03;
      blades.rotation.y = -t * 0.35;
      decor.rotation.y = Math.sin(t * 0.3) * 0.04 * (0.3 + 0.7 * near);
      oakUniforms(bladeMaterial)?.setTime(t);
    },
    update(dt, time) {
      beacon.update(dt, time);
    },
    dispose() {
      beacon.dispose();
      blades.dispose();
      bladeGeometry.dispose();
      bladeMaterial.dispose();
      relicGeometry.dispose();
      relicMaterial.dispose();
      plinthGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A stepped octagonal plinth with a lotus lip — where the relic stands. */
function makeShrineGeometry(skin: TempleSkin, tier: QualityTier): THREE.BufferGeometry {
  const radial = tier === 'thumb' ? 6 : 8;
  const parts: ColoredPart[] = [];
  const step0 = new THREE.CylinderGeometry(0.33, 0.4, 0.11, radial, 1);
  step0.translate(0, 0.055, 0);
  parts.push({ geometry: step0, color: skin.stoneDark });
  const step1 = new THREE.CylinderGeometry(0.25, 0.31, 0.12, radial, 1);
  step1.translate(0, 0.17, 0);
  parts.push({ geometry: step1, color: skin.stoneLight });
  const drum = new THREE.CylinderGeometry(0.125, 0.18, 0.32, radial, 1);
  drum.translate(0, 0.39, 0);
  parts.push({ geometry: drum, color: skin.stoneLight });
  const lip = new THREE.CylinderGeometry(0.22, 0.14, 0.08, radial, 1);
  lip.translate(0, 0.59, 0);
  parts.push({ geometry: lip, color: skin.stoneDark });
  for (let i = 0; i < radial; i++) {
    const a = (i / radial) * Math.PI * 2;
    const petal = new THREE.ConeGeometry(0.06, 0.16, 4, 1);
    petal.rotateX(0.95);
    petal.rotateY(a);
    petal.translate(Math.cos(a) * 0.2, 0.64, Math.sin(a) * 0.2);
    parts.push({ geometry: petal, color: skin.stoneLight });
  }
  return mergeColoredParts(parts, 'shrinePlinth');
}

