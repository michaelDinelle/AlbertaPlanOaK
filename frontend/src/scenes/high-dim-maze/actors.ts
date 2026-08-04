// The Slot Canyons — the actors (HighDimMaze-cont-v0).
//
// Four things move in this canyon, and each of them is a pure function of the frame:
//
//  * THE OASIS (the goal). `root` is the probe object: its world position is EXACTLY
//    `cellToWorld(gx, gy)` with y = 0 and nothing else. The basin, the water, the reeds,
//    the shaft of light and the kit `GoalBeacon` glow all hang off `alcove`, a child that
//    carries the grow-in scale. Delete every decoration and a correctly-placed empty node
//    remains.
//  * THE SCARF. HighDimMaze is the one grid env with a FACING (pose[4] = lastDir), so the
//    wisp is the only one in the preserve that is not radially symmetric. The scarf is
//    the reason a still frame tells you which way the agent is pointed. It hangs off
//    `wisp.avatar` — the exact cell node — and carries its own damped yaw, so the probe
//    that node feeds stays bit-exact (THE WISP PROBE RULE).
//  * THE GUST. `dash` (action 5) is a two-cell sprint inside one 8 Hz step; the hop
//    itself is a straight line along the corridor, and this adds the speed lines that
//    make it read as speed plus the ground wake that says which cells were crossed.
//  * THE SETTLE. `stay` (action 4) is the wisp holding station: the scarf goes slack, a
//    faint ring of disturbed sand hangs under it, and dust sifts down.
//
// ZERO ALLOCATION: every per-frame path here writes into preallocated Matrix4 / Vector3 /
// Color scratch. Every clock is `ctx.time()`.

import { GoalBeacon } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import { makeWater, type WaterHandle } from '@kit/water';
import * as THREE from 'three';
import { clamp01, mergeColoredParts, type CanyonSkin, type ColoredPart } from './world';

// ---------------------------------------------------------------------------
// the oasis — the goal
// ---------------------------------------------------------------------------

/** basin outer radius; a shade under half a cell, so the alcove owns its cell and no more */
const BASIN_R = 0.44;
const BASIN_RIM_Y = 0.13;
const POOL_R = 0.315;
const POOL_Y = 0.052;
/** the shaft climbs to the canyon rim — it is how the eye finds the goal from across the maze */
const SHAFT_TOP = 1.62;
/**
 * The shaft's HDR gain. It is the one thing in this canyon that crosses the 0.85 bloom
 * threshold, and it is deliberately modest: at 2.4 the shaft, the beacon column and a
 * sunlit terracotta floor stacked up and bloomed into a white pillar three cells wide.
 * The goal must read as WATER IN SHADE with light falling on it, not as a lens flare.
 */
const SHAFT_HDR = 0.62;
/**
 * The most the grow-in scale may reach. It is a guard on garbage input, NOT the shape of
 * the beat: `growth()` peaks at 1.06 and settles to 1, and clamping at 1 (the first pass)
 * is what killed the bounce.
 */
const GROW_CEILING = 1.2;

export interface OasisOptions {
  tier: QualityTier;
  palette: Palette;
  skin: CanyonSkin;
  name?: string;
}

export interface OasisHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(gx, gy), y = 0, always */
  root: THREE.Object3D;
  /** local height of the shaft's core — the scene lifts its point light to it */
  readonly crownHeight: number;
  /** the whole visual state: grow-in from the last respawn, and how near the wisp is */
  setState(grow01: number, near01: number): void;
  flare(strength?: number): void;
  /** ctx.time() — water, shaft breath and reed sway */
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildOasis(opts: OasisOptions): OasisHandle {
  const { tier, palette, skin } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'oasisAlcove';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'oasisRoot';
  group.add(root);

  // ---- everything else hangs off the growth node ----
  const alcove = new THREE.Object3D();
  alcove.name = 'oasisGrow';
  root.add(alcove);

  // ---- the kit goal beacon, demoted to a decoration child at the local origin ----
  // Never `setCell`ed: parented at (0,0,0) under `root`, it inherits the exact goal cell.
  const beacon = new GoalBeacon({
    color: skin.shaft,
    tier,
    height: 0.46,
    radius: 0.08,
    intensity: 0.26,
    name: 'oasisGlow',
  });
  alcove.add(beacon.group);

  // ---- the basin: a broken ring of wet rock around a sunken pool ----
  const basinParts: ColoredPart[] = [];
  {
    const stones = tier === 'thumb' ? 6 : 11;
    for (let i = 0; i < stones; i++) {
      const a = (i / stones) * Math.PI * 2;
      const wobble = 0.86 + 0.3 * ((i * 5) % 7) / 7;
      const stone = new THREE.BoxGeometry(0.2 * wobble, BASIN_RIM_Y * wobble, 0.13);
      stone.rotateY(-a);
      stone.translate(Math.cos(a) * BASIN_R * 0.92, BASIN_RIM_Y * 0.5 * wobble, Math.sin(a) * BASIN_R * 0.92);
      basinParts.push({
        geometry: stone,
        color: i % 3 === 0 ? (skin.strata[1] ?? skin.stone) : (skin.strata[2] ?? skin.stone),
      });
    }
    // a dark, damp floor under the water so the pool never shows the sand through it
    const bed = new THREE.CylinderGeometry(POOL_R * 1.08, POOL_R * 0.8, 0.09, Math.max(6, Math.round(16 * detail)));
    bed.translate(0, 0.02, 0);
    basinParts.push({ geometry: bed, color: skin.waterDeep });
  }
  const basinGeometry = mergeColoredParts(basinParts, 'oasisBasin');
  const basinMaterial: OakMaterial = oakMaterial({
    // the same white base the canyon's rock uses, so the alcove is graded like its cliff
    color: palette.sun,
    vertexColors: true,
    flatShading: true,
    roughness: 0.7,
    rimColor: skin.rim,
    rimStrength: 0.45,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.28,
    tier,
    name: 'oasisBasinMaterial',
  });
  const basin = new THREE.Mesh(basinGeometry, basinMaterial);
  basin.name = 'oasisBasin';
  basin.castShadow = tier !== 'thumb';
  basin.receiveShadow = tier !== 'thumb';
  alcove.add(basin);

  // ---- the water ----
  const water: WaterHandle = makeWater({
    size: POOL_R * 2,
    palette,
    tier,
    segments: 18,
    shallow: skin.water,
    deep: skin.waterDeep,
    sky: palette.skyHorizon,
    caustic: palette.sun,
    amplitude: 0.012,
    waveScale: 9,
    speed: 0.7,
    fresnel: 0.6,
    caustics: 0.5,
    depthFalloff: 1.2,
    opacity: 0.94,
    level: POOL_Y,
    name: 'oasisWater',
  });
  water.mesh.name = 'oasisWater';
  alcove.add(water.mesh);

  // ---- reeds: the only green in the canyon, and that is why the eye finds the goal ----
  const reedGeometry = makeReedGeometry(tier);
  const reedMaterial: OakMaterial = oakMaterial({
    color: skin.reed,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.78,
    rimColor: skin.rim,
    rimStrength: 0.4,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.3,
    wind: { amplitude: 0.05, frequency: 1.7, height: 0.34 },
    tier,
    name: 'oasisReedMaterial',
  });
  const reedCount = tier === 'thumb' ? 5 : 10;
  const reeds = new THREE.InstancedMesh(reedGeometry, reedMaterial, reedCount);
  reeds.name = 'oasisReeds';
  reeds.castShadow = false;
  reeds.frustumCulled = false;
  reeds.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(reedCount * 3), 3);
  {
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const s = new THREE.Vector3();
    const c = new THREE.Color();
    for (let i = 0; i < reedCount; i++) {
      // a fixed arc on the shaded (north) lip: deterministic, so a scrub rebuilds it
      const a = -2.7 + (i / Math.max(1, reedCount - 1)) * 3.6 + ((i * 3) % 5) * 0.06;
      const r = BASIN_R * (0.82 + 0.12 * (((i * 7) % 4) / 3));
      p.set(Math.cos(a) * r, 0.02, Math.sin(a) * r);
      e.set(0.1 * Math.sin(i * 2.1), a + 1.5, 0.14 * Math.cos(i * 1.3));
      q.setFromEuler(e);
      const k = 0.95 + 0.6 * (((i * 5) % 6) / 5);
      s.set(k, k * (0.85 + 0.35 * (((i * 11) % 7) / 6)), k);
      m.compose(p, q, s);
      reeds.setMatrixAt(i, m);
      const tone = 0.82 + 0.3 * (((i * 13) % 5) / 4);
      c.setRGB(tone, tone, tone * 0.94);
      reeds.setColorAt(i, c);
    }
    reeds.instanceMatrix.needsUpdate = true;
    if (reeds.instanceColor) reeds.instanceColor.needsUpdate = true;
  }
  alcove.add(reeds);

  // ---- the shaft: afternoon light finding its way down into the slot ----
  const shaftGeometry = new THREE.CylinderGeometry(
    POOL_R * 2.5,
    POOL_R * 0.6,
    SHAFT_TOP,
    Math.max(6, Math.round(18 * detail)),
    1,
    true,
  );
  shaftGeometry.translate(0, SHAFT_TOP * 0.5 + POOL_Y, 0);
  const shaftMaterial: EmissiveMaterial = emissiveMaterial(skin.shaft, SHAFT_HDR, {
    transparent: true,
    opacity: 0.16,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'oasisShaft',
  });
  const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
  shaft.name = 'oasisShaft';
  shaft.renderOrder = 2;
  alcove.add(shaft);

  const crownHeight = 0.5;

  return {
    group,
    root,
    crownHeight,

    setState(grow01, near01) {
      // THE GROW-IN IS ALLOWED TO OVERSHOOT. A `clamp01` here flattened every value above
      // 1, which silently deleted the bounce the caller's `growth()` computes — the alcove
      // rose to exactly 1 and stopped, and the documented "wells up with a little bounce"
      // beat rendered as nothing at all. The ceiling is a guard against garbage, not a
      // shape: it sits above the envelope's own peak (1.06) and nowhere near it.
      const grow = grow01 > 0 ? (grow01 < GROW_CEILING ? grow01 : GROW_CEILING) : 0;
      const near = clamp01(near01);
      // growth is a SCALE on the decoration node — the root never moves, so an oasis that
      // is 3% grown is still exactly on its cell
      alcove.scale.setScalar(grow);
      // the LIGHT does not bounce with it: the shaft is a level, not a spring, so its terms
      // read the settled 0..1 growth and the overshoot stays purely geometric
      const lit = clamp01(grow);
      // the shaft strengthens as the wisp closes in: nothing the agent does not already
      // know (its own cell and the goal's), and it makes the last two cells feel inevitable
      setEmissiveIntensity(shaftMaterial, SHAFT_HDR * (0.55 + 0.7 * near) * (0.2 + 0.8 * lit));
      shaftMaterial.opacity = (0.115 + 0.09 * near) * lit;
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      water.setTime(t);
      oakUniforms(reedMaterial)?.setTime(t);
      oakUniforms(basinMaterial)?.setTime(t);
      // the shaft breathes with the dust drifting through it
      const breath = 1 + 0.07 * Math.sin(t * 0.9) + 0.03 * Math.sin(t * 2.3);
      shaft.scale.set(breath, 1, breath);
      shaft.rotation.y = t * 0.11;
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      water.dispose();
      reeds.dispose();
      reedGeometry.dispose();
      reedMaterial.dispose();
      basinGeometry.dispose();
      basinMaterial.dispose();
      shaftGeometry.dispose();
      shaftMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** One reed: three tapering blades from a common root, graded dark at the base. */
function makeReedGeometry(tier: QualityTier): THREE.BufferGeometry {
  const blades = tier === 'thumb' ? 2 : 3;
  const positions = new Float32Array(blades * 2 * 3 * 3);
  const colors = new Float32Array(blades * 2 * 3 * 3);
  let o = 0;
  for (let b = 0; b < blades; b++) {
    const a = (b / blades) * Math.PI * 1.1 - 0.4;
    const h = 0.3 + 0.14 * ((b * 5) % 3) / 2;
    const wide = 0.028;
    const lean = 0.1 + 0.07 * ((b * 7) % 4) / 3;
    const dx = Math.cos(a) * wide;
    const dz = Math.sin(a) * wide;
    const tipX = Math.cos(a + 0.5) * lean;
    const tipZ = Math.sin(a + 0.5) * lean;
    const midX = tipX * 0.45;
    const midZ = tipZ * 0.45;
    const pts: readonly (readonly [number, number, number, number])[] = [
      [-dx, 0, -dz, 0.45],
      [dx, 0, dz, 0.45],
      [midX + dx * 0.5, h * 0.55, midZ + dz * 0.5, 0.85],
      [-dx, 0, -dz, 0.45],
      [midX + dx * 0.5, h * 0.55, midZ + dz * 0.5, 0.85],
      [tipX, h, tipZ, 1.2],
    ];
    for (const [x, y, z, k] of pts) {
      positions[o * 3] = x;
      positions[o * 3 + 1] = y;
      positions[o * 3 + 2] = z;
      colors[o * 3] = k;
      colors[o * 3 + 1] = k;
      colors[o * 3 + 2] = k;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'oasisReed';
  return g;
}

// ---------------------------------------------------------------------------
// the scarf — pose[4] made visible
// ---------------------------------------------------------------------------

export interface ScarfOptions {
  tier: QualityTier;
  color: THREE.ColorRepresentation;
  /** the wisp's orb height — the scarf ties on just under it */
  height: number;
  name?: string;
}

export interface ScarfHandle {
  /** parent this ONE node to `wisp.avatar` (the exact cell node) */
  group: THREE.Group;
  segments: number;
  /**
   * `yaw` is the facing this frame, `slack` in [0,1] is how still the wisp is (1 = a
   * settled `stay`, 0 = a full dash), `speed` in [0,1] is how hard the air is moving.
   */
  setState(yaw: number, slack: number, speed: number, time: number): void;
  dispose(): void;
}

/**
 * A windswept scarf: a chain of tapering panels streaming BEHIND the facing direction.
 * One InstancedMesh, one draw; the chain is walked on the CPU into preallocated scratch,
 * which is five `compose()` calls a frame and buys a silhouette that reads at a glance.
 */
export function buildScarf(opts: ScarfOptions): ScarfHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'wispScarf';
  const segments = opts.tier === 'thumb' ? 0 : opts.tier === 'medium' ? 5 : 7;
  group.position.y = opts.height;

  if (segments === 0) {
    return {
      group,
      segments,
      setState() {
        /* thumb wears no scarf */
      },
      dispose() {
        group.removeFromParent();
        group.clear();
      },
    };
  }

  const SEG_LEN = 0.2;
  const geometry = makeScarfPanelGeometry(SEG_LEN);
  const material: OakMaterial = oakMaterial({
    color: opts.color,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    roughness: 0.6,
    rimColor: opts.color,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.42,
    tier: opts.tier,
    name: 'wispScarfMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, segments);
  mesh.name = 'wispScarfPanels';
  mesh.castShadow = false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(segments * 3), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < segments; i++) {
      const k = 1 - (i / segments) * 0.45;
      c.setRGB(k, k, k);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  group.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    segments,
    setState(yaw, slack, speed, time) {
      group.rotation.y = yaw;
      // the chain: each panel hangs a little further back, lifts with the airflow and
      // wags with a phase that runs down the length
      let x = 0;
      let y = 0;
      let z = 0;
      let heading = 0;
      const lift = 0.42 * (1 - slack) + 0.1;
      const wag = 0.45 + 0.6 * speed;
      for (let i = 0; i < segments; i++) {
        const t = (i + 1) / segments;
        // the wave runs DOWN the length with a long phase lag; a short lag and a big
        // per-segment turn made a lightning bolt, not a ribbon
        const phase = time * (1.9 + 1.6 * speed) - i * 0.52;
        const swing = Math.sin(phase) * (0.1 + 0.26 * t) * wag * (0.35 + 0.65 * (1 - slack));
        const droop = -(0.12 + 0.42 * t) * slack;
        heading += swing * 0.3;
        const pitch = droop + lift * (0.5 - 0.5 * t) + 0.07 * Math.cos(phase * 0.8);
        // step BACKWARD along the current heading (local -z is behind the facing)
        const dx = Math.sin(heading) * SEG_LEN;
        const dz = -Math.cos(heading) * SEG_LEN;
        const dy = Math.sin(pitch) * SEG_LEN * 0.5;
        position.set(x + dx * 0.5, y + dy * 0.5, z + dz * 0.5);
        euler.set(pitch, heading, swing * 0.35);
        quaternion.setFromEuler(euler);
        const taper = 1 - t * 0.62;
        scale.set(taper, 1, 1);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
        x += dx;
        y += dy;
        z += dz;
      }
      mesh.instanceMatrix.needsUpdate = true;
      oakUniforms(material)?.setTime(time);
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

/** One scarf panel: a flat trapezoid lying along local −z, graded toward its trailing edge. */
function makeScarfPanelGeometry(length: number): THREE.BufferGeometry {
  const w0 = 0.105;
  const w1 = 0.078;
  const verts: readonly (readonly [number, number, number, number])[] = [
    [-w0, 0, length * 0.5, 1],
    [w0, 0, length * 0.5, 1],
    [w1, 0, -length * 0.5, 0.72],
    [-w0, 0, length * 0.5, 1],
    [w1, 0, -length * 0.5, 0.72],
    [-w1, 0, -length * 0.5, 0.72],
  ];
  const positions = new Float32Array(verts.length * 3);
  const colors = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i];
    if (!v) continue;
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
    colors[i * 3] = v[3];
    colors[i * 3 + 1] = v[3];
    colors[i * 3 + 2] = v[3];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'wispScarfPanel';
  return g;
}

// ---------------------------------------------------------------------------
// the gust — dash (action 5) and stay (action 4)
// ---------------------------------------------------------------------------

export interface GustOptions {
  tier: QualityTier;
  skin: CanyonSkin;
  color: THREE.ColorRepresentation;
  name?: string;
}

export interface GustHandle {
  /** add this ONE group to the scene (world space) */
  group: THREE.Group;
  /**
   * The speed lines that trail a dashing wisp. `from`/`to` are the world XZ of the two
   * cells; `progress` is the hop's own 0→1; `amount` in [0,1] fades the whole thing.
   */
  setLines(
    fromX: number,
    fromZ: number,
    toX: number,
    toZ: number,
    y: number,
    progress: number,
    amount: number,
  ): void;
  /** the sand wake lying on the traversed cells; `amount` is its fade envelope */
  setWake(fromX: number, fromZ: number, toX: number, toZ: number, amount: number): void;
  /** the disturbed-sand ring under a settled wisp */
  setSettle(x: number, z: number, amount: number, time: number): void;
  /** kill everything with no easing (scrub / teleport) */
  clear(): void;
  dispose(): void;
}

/**
 * The dash and stay trappings, pooled into three preallocated meshes. Every one of them
 * is invisible at amount 0 and never allocates.
 */
export function buildGust(opts: GustOptions): GustHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'wispGust';
  const lines = opts.tier === 'thumb' ? 0 : opts.tier === 'medium' ? 4 : 6;

  // ---- speed lines: thin additive streaks strung along the dash segment ----
  const lineGeometry = new THREE.PlaneGeometry(1, 0.022);
  lineGeometry.rotateX(-Math.PI / 2);
  const lineMaterial: EmissiveMaterial = emissiveMaterial(opts.color, 1.1, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    name: 'wispSpeedLines',
  });
  const lineMesh = new THREE.InstancedMesh(lineGeometry, lineMaterial, Math.max(1, lines));
  lineMesh.name = 'wispSpeedLines';
  lineMesh.frustumCulled = false;
  lineMesh.visible = false;
  lineMesh.count = lines;
  lineMesh.renderOrder = 3;
  group.add(lineMesh);

  // ---- the sand wake: a stretched smear on the floor along the cells crossed ----
  const wakeGeometry = new THREE.PlaneGeometry(1, 1, 1, 1);
  wakeGeometry.rotateX(-Math.PI / 2);
  const wakeMaterial: OakMaterial = oakMaterial({
    color: opts.skin.sandSun,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    roughness: 1,
    toonSteps: 2,
    toonSoftness: 0.5,
    toonFloor: 0.6,
    tier: opts.tier,
    name: 'wispSandWake',
  });
  const wake = new THREE.Mesh(wakeGeometry, wakeMaterial);
  wake.name = 'wispSandWake';
  wake.visible = false;
  wake.renderOrder = 1;
  group.add(wake);

  // ---- the settle ring: disturbed sand under a wisp holding station ----
  const ringGeometry = new THREE.RingGeometry(0.2, 0.36, opts.tier === 'thumb' ? 10 : 26);
  ringGeometry.rotateX(-Math.PI / 2);
  const ringMaterial: EmissiveMaterial = emissiveMaterial(opts.skin.dust, 0.55, {
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    name: 'wispSettleRing',
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.name = 'wispSettleRing';
  ring.visible = false;
  ring.renderOrder = 1;
  group.add(ring);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    setLines(fromX, fromZ, toX, toZ, y, progress, amount) {
      if (lines === 0 || amount <= 0.001) {
        lineMesh.visible = false;
        return;
      }
      lineMesh.visible = true;
      const dx = toX - fromX;
      const dz = toZ - fromZ;
      const span = Math.hypot(dx, dz);
      const yaw = Math.atan2(dx, dz);
      // the head of the streak sits where the wisp is; the tail lags behind it
      const headX = fromX + dx * progress;
      const headZ = fromZ + dz * progress;
      for (let i = 0; i < lines; i++) {
        const t = (i + 0.5) / lines;
        const lag = 0.12 + 0.42 * t;
        const cx = headX - dx * lag * 0.5;
        const cz = headZ - dz * lag * 0.5;
        const off = ((i % 2 === 0 ? 1 : -1) * (0.05 + 0.09 * t));
        position.set(cx + Math.cos(yaw) * off, y + (i % 3) * 0.045 - 0.03, cz - Math.sin(yaw) * off);
        euler.set(0, yaw, 0);
        quaternion.setFromEuler(euler);
        scale.set(Math.max(0.12, span * (0.3 + 0.4 * t)), 1, 1 - t * 0.4);
        matrix.compose(position, quaternion, scale);
        lineMesh.setMatrixAt(i, matrix);
      }
      lineMesh.instanceMatrix.needsUpdate = true;
      setEmissiveIntensity(lineMaterial, 1.1 * amount);
      lineMaterial.opacity = 0.55 * amount;
    },
    setWake(fromX, fromZ, toX, toZ, amount) {
      if (amount <= 0.001) {
        wake.visible = false;
        return;
      }
      wake.visible = true;
      const dx = toX - fromX;
      const dz = toZ - fromZ;
      const span = Math.max(0.35, Math.hypot(dx, dz));
      wake.position.set((fromX + toX) * 0.5, 0.014, (fromZ + toZ) * 0.5);
      wake.rotation.y = Math.atan2(dx, dz);
      wake.scale.set(0.44 + 0.16 * amount, 1, span + 0.5);
      wakeMaterial.opacity = 0.4 * amount;
    },
    setSettle(x, z, amount, time) {
      if (amount <= 0.001) {
        ring.visible = false;
        return;
      }
      ring.visible = true;
      const breath = 1 + 0.09 * Math.sin(time * 1.6);
      ring.position.set(x, 0.016, z);
      ring.scale.set(breath, 1, breath);
      ring.rotation.y = time * 0.25;
      setEmissiveIntensity(ringMaterial, 0.55 * amount);
      ringMaterial.opacity = 0.42 * amount;
    },
    clear() {
      lineMesh.visible = false;
      wake.visible = false;
      ring.visible = false;
      wakeMaterial.opacity = 0;
      ringMaterial.opacity = 0;
    },
    dispose() {
      lineMesh.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      wakeGeometry.dispose();
      wakeMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
