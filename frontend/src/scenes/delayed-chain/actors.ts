// The Long Night Road — the lantern road, the end waymarker and the start milestone.
//
// ACCURACY. Every lantern stands on the x of its own cell: instance i is placed at
// `cellToWorld(i, 0)` with only a z offset onto the far verge, and the paying cell's
// lantern is a bespoke prop whose ROOT is exactly `cellToWorld(goal, 0)` with y = 0 —
// the alignment probe. Delete every stone, every cage and every glow and what remains is
// thirty-two correctly-placed empty nodes. (The instanced posts are asserted cell by cell
// in the test by reading their instance matrices, which is the only honest way to check
// a prop that has no Object3D of its own.)
//
// ONE DRAW PER LAYER. Thirty-one posts, thirty-one lamps, thirty-one halos and
// thirty-one ground pools are four InstancedMeshes. Per-lantern brightness — the whole
// point of this scene — rides on `instanceColor`, which three multiplies into the
// fragment colour for a MeshBasicMaterial (four-rooms lesson 13, applied to emitters
// instead of floors). Igniting the road costs 31 float writes, not 31 materials.

import { GoalBeacon } from '@kit/grid';
import { cellToWorld, type GridSize } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { hash1f } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import {
  LAMP_REACH,
  LANTERN_Z,
  blendTokens,
  clamp01,
  mergeColoredParts,
  propMaterial,
  setPropTime,
  shade,
  type ColoredPart,
  type MoorSkin,
} from './world';

/** post height; the lamp hangs from an arm just under the top */
export const POST_HEIGHT = 1.78;
/** world y of the flame inside the cage — the burst and the point lights ride this */
export const LAMP_Y = 1.5;
/** HDR gain of a fully-lit lamp. Night is dark: this is the one thing that blooms. */
export const LAMP_HDR = 2.6;
/**
 * The warm ball around a lit flame. SMALL — the kit's cheapest icosphere (detail 0) has a
 * HEXAGONAL silhouette, and the first pass hung a 0.26-radius one on every lamp: from the
 * road camera the near lanterns turned into paper hexagons the size of the cage they were
 * meant to be glowing inside. At 0.12 with a smooth shell the halo is a highlight and the
 * host's bloom pass does the spreading, which is what bloom is for.
 */
const HALO_HDR = 0.75;
/**
 * The light a lit lamp throws on the flagstones. Deliberately WEAK: eight lit lanterns
 * overlap along the road during an ignition wave, and additive discs at the first pass's
 * 0.62 stacked into a continuous white carpet — the road stopped reading as stone. At
 * 0.34 each lamp keeps its own pool of warm light and the stone underneath still has a
 * colour.
 */
const POOL_HDR = 0.46;

export interface LanternRoadOptions {
  tier: QualityTier;
  palette: Palette;
  skin: MoorSkin;
  grid: GridSize;
  /** the paying cell — its lantern is the bespoke one (the probe) */
  goalCell: number;
  /** lane seed: the only randomness (post lean and weathering) */
  seed: number;
  /** the scene accent (aurora teal) — the goal beacon is a thread of the sky */
  accent: THREE.ColorRepresentation;
  name?: string;
}

export interface LanternRoadHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(goalCell, 0), y = 0, forever */
  endRoot: THREE.Object3D;
  /** world position of the end lamp's flame (the celebration's origin) */
  readonly endLamp: THREE.Vector3;
  /** lanterns on the road (= grid.w) */
  readonly count: number;
  /** world x of a lantern (fractional cells allowed — the wave front rides this) */
  lanternX(cell: number): number;
  /** 0 = cold iron, 1 = fully alight. Call `commit()` once after a sweep. */
  setGlow(cell: number, v01: number): void;
  commit(): void;
  /** celebration beat on the paying cell's own beacon */
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

/**
 * One lantern, local origin at the foot of its post: a tapered iron shaft, a collar, an
 * arm reaching out over the road and a four-post cage under a conical cap. Silhouette
 * first — at night this is a black shape against an aurora, and the shape is all there is.
 */
export function lanternGeometry(skin: MoorSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(4, Math.round(6 * detail));
  const parts: ColoredPart[] = [];

  const foot = new THREE.CylinderGeometry(0.075, 0.115, 0.11, radial);
  foot.translate(0, 0.055, 0);
  parts.push({ geometry: foot, color: skin.stoneDark });

  const shaft = new THREE.CylinderGeometry(0.03, 0.048, POST_HEIGHT, radial);
  shaft.translate(0, POST_HEIGHT * 0.5, 0);
  parts.push({ geometry: shaft, color: skin.iron });

  const collar = new THREE.CylinderGeometry(0.052, 0.052, 0.055, radial);
  collar.translate(0, 1.16, 0);
  parts.push({ geometry: collar, color: skin.ironLight });

  const arm = new THREE.BoxGeometry(0.038, 0.038, LAMP_REACH + 0.08);
  arm.translate(0, POST_HEIGHT - 0.08, (LAMP_REACH + 0.08) * 0.5 - 0.04);
  parts.push({ geometry: arm, color: skin.iron });

  const hook = new THREE.BoxGeometry(0.028, 0.13, 0.028);
  hook.translate(0, POST_HEIGHT - 0.15, LAMP_REACH);
  parts.push({ geometry: hook, color: skin.iron });

  const cap = new THREE.ConeGeometry(0.125, 0.1, radial);
  cap.translate(0, LAMP_Y + 0.15, LAMP_REACH);
  parts.push({ geometry: cap, color: skin.ironLight });

  const tray = new THREE.CylinderGeometry(0.092, 0.074, 0.035, radial);
  tray.translate(0, LAMP_Y - 0.11, LAMP_REACH);
  parts.push({ geometry: tray, color: skin.ironLight });

  // four uprights, so the flame is SEEN between them rather than sealed in a can
  const bars = tier === 'thumb' ? 2 : 4;
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2 + Math.PI / 4;
    const bar = new THREE.BoxGeometry(0.017, 0.23, 0.017);
    bar.translate(Math.cos(a) * 0.075, LAMP_Y + 0.015, LAMP_REACH + Math.sin(a) * 0.075);
    parts.push({ geometry: bar, color: skin.iron });
  }

  return mergeColoredParts(parts, 'lanternPost');
}

/**
 * THE END OF THE ROAD. A heaped moorland cairn under a leaning standing stone, planted
 * ACROSS THE ROAD'S OWN AXIS half a cell past the paying cell — not off on the verge,
 * where the first pass put it and where it read as two pale panels lost behind the lantern
 * line. The road physically stops at a stone, so from any cell of the walk the answer to
 * "how much further" is a silhouette standing in the aurora with the path running into it.
 * The agent never occupies cell 31 (the env wraps out of it in the step that pays), so
 * nothing is being hidden or blocked: the terminus is exactly as far as the road goes.
 *
 * IT IS A PILE, NOT A TOWER. The second pass stacked concentric cylinders and rimmed them
 * in the accent, and the result read as a teal-painted easel standing on the moor — the
 * silhouette of a cairn is IRREGULAR, so the stones here are faceted icosahedra, each
 * squashed and spun on its own hash, heaped into a rough cone. Moonlit granite carries it;
 * the rim only edges it.
 */
function cairnGeometry(skin: MoorSkin, tier: QualityTier, seed: number): THREE.BufferGeometry {
  // detail 0 everywhere: 20 flat faces per boulder is the scene's low-poly language, and a
  // detail-1 rock under flat shading rounds off into a pebble
  const detail = 0;
  const parts: ColoredPart[] = [];
  const base = skin.stone;
  const dark = shade(skin.stoneDark, 0.1);

  // a heaped cone of boulders: radius shrinks and the ring spins as the pile rises
  const rings: readonly { y: number; r: number; n: number; s: number }[] = [
    { y: 0.16, r: 0.38, n: 5, s: 0.23 },
    { y: 0.42, r: 0.26, n: 4, s: 0.2 },
    { y: 0.64, r: 0.14, n: 3, s: 0.17 },
    { y: 0.8, r: 0, n: 1, s: 0.15 },
  ];
  let k = 0;
  for (const ring of rings) {
    for (let i = 0; i < ring.n; i++) {
      const h1 = hash1f(k, seed ^ 0x51a3d7);
      const h2 = hash1f(k, seed ^ 0x2ce19b);
      const a = (i / ring.n) * Math.PI * 2 + ring.y * 3.1 + h1 * 0.5;
      const rock = new THREE.IcosahedronGeometry(ring.s * (0.82 + 0.36 * h2), detail);
      rock.scale(1, 0.72 + 0.22 * h1, 0.88);
      rock.rotateY(h1 * 3.14);
      rock.rotateZ(h2 * 0.8 - 0.4);
      rock.translate(Math.cos(a) * ring.r, ring.y + 0.05 * h2, Math.sin(a) * ring.r * 0.8);
      parts.push({ geometry: rock, color: k % 3 === 1 ? dark : base });
      k++;
    }
  }

  // The standing stone: a leaning, tapered slab about twice the cairn's height — the shape
  // you see first from anywhere on the road. Tapered, because a true rectangle at this size
  // reads as a headstone rather than as a stone somebody stood on end.
  const marker = new THREE.CylinderGeometry(0.16, 0.25, 1.16, 5);
  marker.scale(1, 1, 0.62);
  marker.rotateZ(0.15);
  marker.rotateY(-0.34);
  marker.translate(-0.42, 0.57, 0.1);
  parts.push({ geometry: marker, color: base });
  const wedge = new THREE.IcosahedronGeometry(0.19, detail);
  wedge.scale(1, 0.6, 0.9);
  wedge.translate(-0.3, 0.09, -0.24);
  parts.push({ geometry: wedge, color: dark });
  return mergeColoredParts(parts, 'endCairn');
}

/**
 * The milestone where the wisp always starts again: a leaning waymarker slab with a moss
 * cap, bedded in two half-sunk kerb stones. It is a SLAB, not a stub — the first shape
 * here was a squat cylinder with a domed top that read, from the road camera, as a bin
 * standing next to the agent. A tilted stone tablet says "mile zero" in silhouette alone,
 * which is the only language a night scene has.
 */
function milestoneGeometry(skin: MoorSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(5, Math.round(7 * detail));
  const parts: ColoredPart[] = [];

  // Kept SMALL and shaded: the wisp respawns on this cell every single lap and stands a
  // metre from the stone with a point light on it, so a pale full-size slab here renders
  // as a blank white billboard beside the agent. Half a metre of dark granite reads as a
  // waymarker and lets the wisp stay the brightest thing on the road.
  // TAPERED, so its silhouette is a stone and not a chimney: the road camera stands beside
  // this thing on every lap, and a straight-sided box in the near foreground read as a
  // brick pillar cropping the avatar.
  const slab = new THREE.CylinderGeometry(0.1, 0.15, 0.52, 4);
  slab.rotateY(Math.PI * 0.25 + 0.16);
  slab.rotateZ(-0.12);
  slab.scale(1, 1, 0.72);
  slab.translate(0, 0.25, 0);
  parts.push({ geometry: slab, color: shade(skin.stone, 0.42) });

  // the weathered top: a thin wedge of moss, tilted with the slab
  const cap = new THREE.CylinderGeometry(0.085, 0.105, 0.05, 4);
  cap.rotateY(Math.PI * 0.25 + 0.16);
  cap.rotateZ(-0.12);
  cap.scale(1, 1, 0.72);
  cap.translate(0.033, 0.51, 0);
  parts.push({ geometry: cap, color: shade(skin.moss, 0.3) });

  const kerbA = new THREE.CylinderGeometry(0.1, 0.13, 0.08, radial);
  kerbA.translate(0.17, 0.03, 0.12);
  parts.push({ geometry: kerbA, color: shade(skin.stoneDark, 0.25) });
  const kerbB = new THREE.CylinderGeometry(0.08, 0.1, 0.06, radial);
  kerbB.translate(-0.14, 0.025, -0.15);
  parts.push({ geometry: kerbB, color: shade(skin.stoneDark, 0.25) });

  const bed = new THREE.CylinderGeometry(0.22, 0.26, 0.05, radial);
  bed.translate(0, 0.02, 0);
  parts.push({ geometry: bed, color: shade(skin.stoneDark, 0.25) });
  return mergeColoredParts(parts, 'startMilestone');
}

export function buildLanternRoad(opts: LanternRoadOptions): LanternRoadHandle {
  const { tier, palette, skin, grid, goalCell, seed, accent: aurora } = opts;
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = opts.name ?? 'lanternRoad';
  const count = grid.w;
  const instanceCount = Math.max(1, count - 1);

  // ---- geometry, shared by the instanced road and the bespoke end lantern ----
  const postGeometry = lanternGeometry(skin, tier);
  const glassGeometry = new THREE.IcosahedronGeometry(0.058, tier === 'thumb' ? 0 : 1);
  const haloGeometry = new THREE.IcosahedronGeometry(0.12, tier === 'thumb' ? 1 : 2);
  const poolGeometry = new THREE.CircleGeometry(0.86, Math.max(10, Math.round(24 * detail)));
  poolGeometry.rotateX(-Math.PI / 2);
  // THE POOL IS A LIGHT, NOT A DISC. A flat additive circle paints a hard-edged ellipse of
  // paint on the flagstones — during an ignition wave the road became a row of overlapping
  // white plates. The kit's radial sprite gives the disc a real r-falloff in its alpha, so
  // each lamp lays down a soft pool that fades into the stone it is lighting.
  const poolTexture = radialSprite(64, 1.7);

  const postMaterial: OakMaterial = propMaterial(palette, tier, 'lanternIron');
  const glassMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, LAMP_HDR, { name: 'lanternFlame' });
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, HALO_HDR, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternHalo',
  });
  // the pool keeps the flame's own warmth — a lamp lights stone gold, not blue-white
  const poolColor = blendTokens(skin.flame, palette.accentLight, 0.1);
  const poolMaterial: EmissiveMaterial = emissiveMaterial(poolColor, POOL_HDR, {
    map: poolTexture,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternPool',
  });

  // ---- the instanced road (every cell except the paying one) ----
  const posts = new THREE.InstancedMesh(postGeometry, postMaterial, instanceCount);
  posts.name = 'lanternPosts';
  posts.castShadow = false; // a 0.06-unit shaft is sub-texel: it would shadow as dashes
  posts.receiveShadow = shadows;
  posts.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3).fill(1), 3);

  const glass = new THREE.InstancedMesh(glassGeometry, glassMaterial, instanceCount);
  glass.name = 'lanternFlames';
  glass.castShadow = false;
  glass.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const halo = new THREE.InstancedMesh(haloGeometry, haloMaterial, instanceCount);
  halo.name = 'lanternHalos';
  halo.castShadow = false;
  halo.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const pool = new THREE.InstancedMesh(poolGeometry, poolMaterial, instanceCount);
  pool.name = 'lanternPools';
  pool.castShadow = false;
  pool.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const instanceOf = new Int32Array(count).fill(-1);
  {
    const matrix = new THREE.Matrix4();
    const lamp = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const weather = new THREE.Color();
    const world: [number, number, number] = [0, 0, 0];
    let n = 0;
    for (let cell = 0; cell < count; cell++) {
      if (cell === goalCell) continue;
      cellToWorld(cell, 0, grid, world);
      const lean = hash1f(cell, seed ^ 0x2f5a17) - 0.5;
      const wear = hash1f(cell, seed ^ 0x7c1d93);
      position.set(world[0], 0, LANTERN_Z);
      euler.set(0, lean * 0.22, lean * 0.05, 'YXZ');
      quaternion.setFromEuler(euler);
      scale.setScalar(0.94 + 0.12 * wear);
      matrix.compose(position, quaternion, scale);
      posts.setMatrixAt(n, matrix);
      // the lamp layers share the post's transform, so a leaning post's flame,
      // halo and light pool lean with it
      lamp.copy(matrix);
      glass.setMatrixAt(n, lamp);
      halo.setMatrixAt(n, lamp);
      pool.setMatrixAt(n, lamp);
      weather.setRGB(0.9 + 0.2 * wear, 0.9 + 0.2 * wear, 0.92 + 0.18 * wear);
      posts.setColorAt(n, weather);
      instanceOf[cell] = n;
      n++;
    }
    posts.count = n;
    glass.count = n;
    halo.count = n;
    pool.count = n;
  }
  // the lamp layers sit at the cage, not at the post's foot
  glassGeometry.translate(0, LAMP_Y, LAMP_REACH);
  haloGeometry.translate(0, LAMP_Y, LAMP_REACH);
  poolGeometry.translate(0, 0.014, LAMP_REACH);

  for (const mesh of [posts, glass, halo, pool]) {
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    mesh.frustumCulled = false;
    group.add(mesh);
  }

  // ---- the paying cell: a bespoke lantern on an exact root ----
  const endRoot = new THREE.Object3D();
  endRoot.name = 'endLanternRoot';
  {
    const world: [number, number, number] = [0, 0, 0];
    cellToWorld(goalCell, 0, grid, world);
    endRoot.position.set(world[0], 0, world[2]);
  }
  group.add(endRoot);

  const endPost = new THREE.Mesh(postGeometry, postMaterial);
  endPost.name = 'endLanternPost';
  endPost.position.set(0, 0, LANTERN_Z);
  endPost.castShadow = false;
  endPost.receiveShadow = shadows;
  endRoot.add(endPost);

  const endGlassMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, LAMP_HDR, { name: 'endLanternFlame' });
  const endGlass = new THREE.Mesh(glassGeometry, endGlassMaterial);
  endGlass.name = 'endLanternFlame';
  endGlass.position.copy(endPost.position);
  endRoot.add(endGlass);

  const endHaloMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, HALO_HDR, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'endLanternHalo',
  });
  const endHalo = new THREE.Mesh(haloGeometry, endHaloMaterial);
  endHalo.name = 'endLanternHalo';
  endHalo.position.copy(endPost.position);
  endRoot.add(endHalo);

  const endPoolMaterial: EmissiveMaterial = emissiveMaterial(poolColor, POOL_HDR, {
    map: poolTexture,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'endLanternPool',
  });
  const endPool = new THREE.Mesh(poolGeometry, endPoolMaterial);
  endPool.name = 'endLanternPool';
  endPool.position.copy(endPost.position);
  endRoot.add(endPool);

  const cairnGeo = cairnGeometry(skin, tier, seed);
  // The terminus gets its OWN material: the same stone as everything else, but rimmed in
  // the aurora's teal at more than double the scene's rim strength. It is the one prop the
  // walk is aimed at, it stands under the only sky-glow on the moor, and a cool token rim
  // is the honest way to say "the light up there is touching this" — a lit-from-nowhere
  // albedo bump would not be.
  const cairnMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.9,
    // moonlight FIRST, aurora only at the edge: a stone rimmed hard in the accent stops
    // being granite and starts being neon (measured — the teal-easel pass)
    rimColor: blendTokens(palette.hemiSky, aurora, 0.45),
    rimStrength: 0.38,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.2,
    tier,
    name: 'endCairnStone',
  });
  const cairn = new THREE.Mesh(cairnGeo, cairnMaterial);
  cairn.name = 'endCairn';
  // half a cell PAST the paying cell, standing in the road's own axis: the terminus
  cairn.position.set(0.74, 0, -0.14);
  cairn.rotation.y = 0.42;
  cairn.castShadow = shadows;
  cairn.receiveShadow = shadows;
  endRoot.add(cairn);

  // The env's own goal marker, kept DELIBERATELY faint: a thread of aurora-teal at the
  // road's end so the walk has an address, never bright enough to out-shine the wisp.
  // Measured down twice from the kit defaults (1.6 tall, intensity 2.2): the beacon is an
  // ADDITIVE column, so at 1.15 × 0.42 it saturated to a white fluorescent tube standing
  // beside the cairn and stole the eye from the agent. 0.82 × 0.17, tinted to the aurora's
  // own teal, reads as a thread of the sky reaching down to the paying cell.
  const beacon = new GoalBeacon({
    color: blendTokens(palette.accentLight, aurora, 0.55),
    tier,
    // A GLOW-MARK, not a column. Third measured pass: even at 0.82 the additive core stood
    // in front of the cairn like a fluorescent tube and made the terminus read as clutter.
    // The cairn is the vertical; the paying cell only needs to be MARKED, and a 0.2 nub
    // with its ground disc does that without competing with anything.
    height: 0.2,
    radius: 0.085,
    intensity: 0.3,
    name: 'endBeacon',
  });
  endRoot.add(beacon.group);

  // ---- brightness plumbing ----
  const glassColors = glass.instanceColor as THREE.InstancedBufferAttribute;
  const haloColors = halo.instanceColor as THREE.InstancedBufferAttribute;
  const poolColors = pool.instanceColor as THREE.InstancedBufferAttribute;
  const endLamp = new THREE.Vector3(endRoot.position.x, LAMP_Y, LANTERN_Z + LAMP_REACH);
  const worldScratch: [number, number, number] = [0, 0, 0];

  return {
    group,
    endRoot,
    endLamp,
    count,

    lanternX(cell) {
      return cellToWorld(cell, 0, grid, worldScratch)[0];
    },

    setGlow(cell, v01) {
      const v = clamp01(v01);
      const id = cell >= 0 && cell < count ? (instanceOf[cell] ?? -1) : -1;
      if (id >= 0) {
        const g = glassColors.array as Float32Array;
        const h = haloColors.array as Float32Array;
        const p = poolColors.array as Float32Array;
        const o = id * 3;
        g[o] = v;
        g[o + 1] = v;
        g[o + 2] = v;
        const hv = v * 0.62;
        h[o] = hv;
        h[o + 1] = hv;
        h[o + 2] = hv;
        const pv = v * 0.8;
        p[o] = pv;
        p[o + 1] = pv;
        p[o + 2] = pv;
        return;
      }
      if (cell === goalCell) {
        setEmissiveIntensity(endGlassMaterial, LAMP_HDR * v);
        setEmissiveIntensity(endHaloMaterial, HALO_HDR * v * 0.62);
        setEmissiveIntensity(endPoolMaterial, POOL_HDR * v * 0.8);
      }
    },

    commit() {
      glassColors.needsUpdate = true;
      haloColors.needsUpdate = true;
      poolColors.needsUpdate = true;
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      setPropTime(postMaterial, t);
      setPropTime(cairnMaterial, t);
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      posts.dispose();
      glass.dispose();
      halo.dispose();
      pool.dispose();
      postGeometry.dispose();
      glassGeometry.dispose();
      haloGeometry.dispose();
      poolGeometry.dispose();
      poolTexture.dispose();
      cairnGeo.dispose();
      postMaterial.dispose();
      glassMaterial.dispose();
      haloMaterial.dispose();
      poolMaterial.dispose();
      endGlassMaterial.dispose();
      endHaloMaterial.dispose();
      endPoolMaterial.dispose();
      cairnMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// the wisp's own lantern-glow
// ---------------------------------------------------------------------------

export interface WispGlowHandle {
  /** a scene-space billboard: the scene drives its position from the wisp, nothing else */
  mesh: THREE.Mesh;
  /** 0 = out, 1 = the standing glow; the celebration pushes it past 1 */
  setGain(g: number): void;
  dispose(): void;
}

/**
 * THE ONE THING THAT MAKES A NIGHT SCENE'S AVATAR READ. The kit's wisp halo is a
 * detail-1 icosphere, and this scene's camera stands 8.4 units away — close enough that
 * the halo's SILHOUETTE IS A VISIBLE DECAGON, a flat green plate pasted over the moor.
 * On the two exemplars the same avatar is 17–20 units off and the facets never surface;
 * here they are the first thing the eye finds.
 *
 * So the scene HIDES the kit halo mesh (`oakWispHalo`, a named child of the kit rig — not
 * `wisp.dim()`, which dims the core, the skirt and the ribbon along with it) and hangs
 * this in its place: a camera-facing quad carrying the kit's own procedural radial sprite,
 * additively blended, with a true r² falloff and no silhouette at all. The orb core — a
 * detail-2 icosphere on the kit's wisp material — keeps the kit's full 1.9 rim intensity,
 * so the agent is still the brightest thing on the road; it is just a lantern now instead
 * of a sticker.
 */
export function buildWispGlow(opts: {
  color: THREE.ColorRepresentation;
  tier: QualityTier;
  /** world radius of the glow disc */
  radius?: number;
  intensity?: number;
  name?: string;
}): WispGlowHandle {
  const radius = opts.radius ?? 0.95;
  const intensity = opts.intensity ?? 0.5;
  const texture = radialSprite(opts.tier === 'thumb' ? 32 : 64, 2.6);
  const geometry = new THREE.PlaneGeometry(radius * 2, radius * 2);
  const material = emissiveMaterial(opts.color, intensity, {
    map: texture,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'wispGlowMaterial',
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'wispGlow';
  mesh.frustumCulled = false;
  // behind the lantern flames in the transparent queue, so a lit lamp still burns through
  mesh.renderOrder = -1;

  return {
    mesh,
    setGain(g) {
      setEmissiveIntensity(material, intensity * (g > 0 ? g : 0));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      mesh.removeFromParent();
    },
  };
}

// ---------------------------------------------------------------------------
// the start milestone
// ---------------------------------------------------------------------------

export interface MilestoneOptions {
  tier: QualityTier;
  palette: Palette;
  skin: MoorSkin;
  grid: GridSize;
  /** the cell the stream always restarts on */
  cell: number;
  name?: string;
}

export interface MilestoneHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(cell, 0), y = 0 */
  root: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

/** The stone at cell 0 — the only landmark of the start, and the wrap's landing mark. */
export function buildMilestone(opts: MilestoneOptions): MilestoneHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'startMilestone';
  const root = new THREE.Object3D();
  root.name = 'milestoneRoot';
  {
    const world: [number, number, number] = [0, 0, 0];
    cellToWorld(opts.cell, 0, opts.grid, world);
    root.position.set(world[0], 0, world[2]);
  }
  group.add(root);

  const geometry = milestoneGeometry(opts.skin, opts.tier);
  const material = propMaterial(opts.palette, opts.tier, 'milestoneStone');
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'milestone';
  // Planted on the verge OPPOSITE the lantern line, clear of the path the wisp walks —
  // and pushed a metre out, because the camera stands on this side and the agent respawns
  // here: any closer and the stone crosses in front of the avatar on every lap.
  mesh.position.set(-0.34, 0, 1.32);
  mesh.rotation.y = 0.62;
  mesh.castShadow = opts.tier !== 'thumb';
  mesh.receiveShadow = opts.tier !== 'thumb';
  root.add(mesh);

  return {
    group,
    root,
    setTime(t) {
      setPropTime(material, t);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
