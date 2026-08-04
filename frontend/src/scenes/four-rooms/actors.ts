// Four Seasons Courtyard — the lantern-flower (the goal).
//
// ACCURACY: `root` is the probe object. Its world position is EXACTLY
// `cellToWorld(gx, gy)` with y = 0 and nothing else — no hover, no bob, no lean. The
// stem, the petals, the pistil flame and the kit `GoalBeacon`'s glow all hang off `bud`
// and `glow`, children that carry the growth scale and the idle sway. Delete every
// decoration and what remains is a correctly-placed empty node.
//
// THE PETALS ARE ONE DRAW. A flower opens by rotating each petal outward, which normally
// costs one node per petal; here they are six INSTANCES of a single three-triangle petal,
// re-posed every frame from `(openness)` into a preallocated Matrix4. Six matrices per
// frame is free, and the bud stays a single draw call at every tier.
//
// IDEMPOTENCE: openness and growth are pure functions of `(state, ctx.time(), the last
// goal event)`. Nothing here integrates, accumulates or remembers a position, so calling
// `setState` twice with the same arguments leaves the same world — and a scrub that snaps
// growth to 1 lands on exactly the frame a long-settled lantern would show.

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
import * as THREE from 'three';
import { blendTokens, clamp01, type ColoredPart } from './world';

/**
 * Stem height. The bud has to be READ, not just seen: from the scene's 17.8-unit
 * standoff a 0.3-unit flower is twelve pixels, and twelve pixels of petal under a
 * blooming emissive is a white dot. At 0.44 the open lantern spans most of its cell —
 * the goal is legible as a flower from the composed frame, which is the whole job of a
 * goal marker.
 */
const BUD_Y = 0.44;
const PETAL_SCALE = 1.55;
const PETALS = 6;
/** petal pitch from vertical: a tight bud at 0, a wide-open lantern at 1 */
const PITCH_CLOSED = 0.1;
const PITCH_OPEN = 1.16;
/**
 * The flame's HDR gain. It is the ONE thing in this courtyard that crosses the 0.85
 * bloom threshold, and it is deliberately modest: at 2.9 (the first pass) the additive
 * beacon column, the ground disc and the flame stacked on top of a sunlit winter floor
 * and bloomed into a featureless white ball three cells across. The goal must read as a
 * FLOWER holding a light, not as a light where a flower used to be.
 */
const PISTIL_HDR = 0.98;

export interface LanternOptions {
  tier: QualityTier;
  palette: Palette;
  /** petal color (the spring blossom of the courtyard) */
  blossom: THREE.ColorRepresentation;
  /** the flame inside the bud — the one HDR emitter of the goal */
  flame: THREE.ColorRepresentation;
  /** stem / leaf green */
  stem: THREE.ColorRepresentation;
  name?: string;
}

export interface LanternHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(gx, gy), y = 0, always */
  root: THREE.Object3D;
  /** local height of the flame — the scene lifts its point light to it */
  readonly crownHeight: number;
  /** the whole visual state, from the frame and the last goal event and nothing else */
  setState(open01: number, grow01: number): void;
  /** a kindling pop when a new lantern grows in */
  flare(strength?: number): void;
  /** ctx.time() — the idle sway and the material clocks */
  setTime(t: number): void;
  /** beacon breathing + flare decay (dt is wall seconds, time the playback clock) */
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildLantern(opts: LanternOptions): LanternHandle {
  const { tier, palette, blossom, flame, stem } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'lanternFlower';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'lanternRoot';
  group.add(root);

  // ---- the kit goal beacon, demoted to a decoration child at the local origin ----
  // It is never `setCell`ed: parented at (0,0,0) under `root`, it inherits the exact goal
  // cell, and scaling it is then pure decoration.
  const beacon = new GoalBeacon({
    color: flame,
    tier,
    height: 0.5,
    radius: 0.085,
    intensity: 0.3,
    name: 'lanternGlow',
  });
  root.add(beacon.group);

  // ---- the bud: stem, petals, flame ----
  const bud = new THREE.Object3D();
  bud.name = 'lanternBud';
  root.add(bud);

  const stemParts: ColoredPart[] = [];
  {
    const shaft = new THREE.CylinderGeometry(0.019, 0.038, BUD_Y, Math.max(4, Math.round(6 * detail)));
    shaft.translate(0, BUD_Y * 0.5, 0);
    stemParts.push({ geometry: shaft, color: stem });
    // two leaves, so the silhouette says "plant" even before the petals open
    for (const side of [-1, 1]) {
      const leaf = new THREE.CircleGeometry(0.11, Math.max(4, Math.round(6 * detail)));
      leaf.scale(1, 0.5, 1);
      leaf.rotateX(-Math.PI / 2.3);
      leaf.rotateY(side > 0 ? 0.6 : Math.PI - 0.6);
      leaf.translate(side * 0.08, BUD_Y * 0.32, side * 0.04);
      stemParts.push({ geometry: leaf, color: blendTokens(stem, palette.groundB, 0.3) });
    }
  }
  const stemGeometry = mergeStem(stemParts);
  const stemMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.82,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.28,
    tier,
    name: 'lanternStemMaterial',
  });
  const stemMesh = new THREE.Mesh(stemGeometry, stemMaterial);
  stemMesh.name = 'lanternStem';
  stemMesh.castShadow = tier !== 'thumb';
  bud.add(stemMesh);

  const petalGeometry = makePetalGeometry();
  const petalMaterial: OakMaterial = oakMaterial({
    color: blossom,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: flame,
    rimStrength: 0.42,
    toonSteps: 2,
    toonSoftness: 0.38,
    toonFloor: 0.35,
    tier,
    name: 'lanternPetalMaterial',
  });
  const petals = new THREE.InstancedMesh(petalGeometry, petalMaterial, PETALS);
  petals.name = 'lanternPetals';
  petals.castShadow = false;
  petals.frustumCulled = false;
  petals.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(PETALS * 3), 3);
  {
    // a fixed per-petal tint + size, so a scrubbed replay rebuilds the same flower
    const c = new THREE.Color();
    for (let i = 0; i < PETALS; i++) {
      const k = 0.86 + 0.22 * (((i * 7) % 5) / 4);
      c.setRGB(k, k, k);
      petals.setColorAt(i, c);
    }
    if (petals.instanceColor) petals.instanceColor.needsUpdate = true;
  }
  bud.add(petals);

  // small and only just over the bloom threshold. A flame reads as a POINT of light: a
  // fat emitter at 2.9 HDR bloomed into a 60 px star that swallowed the whole flower,
  // and the goal stopped being a lantern-flower and became a lens flare.
  const pistilGeometry = new THREE.IcosahedronGeometry(0.055, tier === 'thumb' ? 0 : 1);
  const pistilMaterial: EmissiveMaterial = emissiveMaterial(flame, PISTIL_HDR, { name: 'lanternFlame' });
  const pistil = new THREE.Mesh(pistilGeometry, pistilMaterial);
  pistil.name = 'lanternFlame';
  pistil.position.set(0, BUD_Y + 0.065, 0);
  bud.add(pistil);

  // ---- preallocated scratch: setState runs every frame and may not allocate ----
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3(0, BUD_Y, 0);
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);
  let openness = 0;

  const crownHeight = BUD_Y + 0.12;

  return {
    group,
    root,
    crownHeight,

    setState(open01, grow01) {
      const open = clamp01(open01);
      const grow = clamp01(grow01);
      openness = open;

      // growth is a SCALE on the decoration node — the root never moves, so a lantern
      // that is 3% grown is still exactly on its cell
      bud.scale.setScalar(grow);
      beacon.group.scale.setScalar(grow);

      const pitch = PITCH_CLOSED + (PITCH_OPEN - PITCH_CLOSED) * open;
      for (let i = 0; i < PETALS; i++) {
        const yaw = (i / PETALS) * Math.PI * 2;
        // a tighter, taller petal when closed; it broadens as the lantern opens
        const s = PETAL_SCALE * (0.94 + 0.16 * open + 0.06 * (((i * 3) % 4) / 3));
        euler.set(pitch, yaw, 0);
        quaternion.setFromEuler(euler);
        scale.set(s, PETAL_SCALE * (1.06 - 0.14 * open), s);
        matrix.compose(position, quaternion, scale);
        petals.setMatrixAt(i, matrix);
      }
      petals.instanceMatrix.needsUpdate = true;

      // the flame swells as the lantern opens; it is the scene's brightest emitter and
      // the only thing here that crosses the bloom threshold
      pistil.scale.setScalar(0.78 + 0.34 * open);
      setEmissiveIntensity(pistilMaterial, PISTIL_HDR * (0.55 + 0.75 * open) * (0.25 + 0.75 * grow));
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      // a slow lean in the noon air — pure ctx.time(), so a paused frame holds it
      bud.rotation.z = Math.sin(t * 0.85) * 0.055 * (0.4 + 0.6 * openness);
      bud.rotation.x = Math.cos(t * 0.63) * 0.042 * (0.4 + 0.6 * openness);
      oakUniforms(petalMaterial)?.setTime(t);
      oakUniforms(stemMaterial)?.setTime(t);
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      petals.dispose();
      petalGeometry.dispose();
      petalMaterial.dispose();
      stemGeometry.dispose();
      stemMaterial.dispose();
      pistilGeometry.dispose();
      pistilMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * One petal: base at the origin, growing up +y and curling forward +z. Three triangles,
 * because a petal is read entirely from the taper between its shoulder and its tip. The
 * vertex grade (dark at the base, hot at the tip) is what makes six flat instances look
 * like a lantern lit from inside.
 */
function makePetalGeometry(): THREE.BufferGeometry {
  const b0: readonly [number, number, number] = [-0.042, 0, 0];
  const b1: readonly [number, number, number] = [0.042, 0, 0];
  const m0: readonly [number, number, number] = [-0.074, 0.105, 0.03];
  const m1: readonly [number, number, number] = [0.074, 0.105, 0.03];
  const tip: readonly [number, number, number] = [0, 0.238, 0.078];
  const tris: readonly (readonly (readonly [number, number, number])[])[] = [
    [b0, b1, m1],
    [b0, m1, m0],
    [m0, m1, tip],
  ];
  const grade: readonly number[] = [0.5, 0.5, 0.86, 0.5, 0.86, 0.86, 0.86, 0.86, 1.12];
  const positions = new Float32Array(9 * 3);
  const colors = new Float32Array(9 * 3);
  let o = 0;
  for (const tri of tris) {
    for (const v of tri) {
      positions[o * 3] = v[0];
      positions[o * 3 + 1] = v[1];
      positions[o * 3 + 2] = v[2];
      const k = grade[o] ?? 1;
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
  g.name = 'lanternPetal';
  return g;
}

/** Merge the stem parts with a flat per-part vertex color (the world.ts pattern). */
function mergeStem(parts: readonly ColoredPart[]): THREE.BufferGeometry {
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
  g.name = 'lanternStem';
  for (const p of parts) p.geometry.dispose();
  return g;
}
