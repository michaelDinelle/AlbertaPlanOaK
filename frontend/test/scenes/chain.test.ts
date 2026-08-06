/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld, isWall } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import { CHAIN_END, CHAIN_L, chainDecoder } from '@sim/envs/chain';
import type { EnvGeometry } from '@sim/types';

import createScene, {
  LANTERN_DRY,
  LANTERN_Z,
  flashEnvelope,
  groundedLanternOffset,
  readPos,
  stoneGlow,
  treeBloom,
} from '../../src/scenes/chain/index';
import { fixtures } from '../../src/scenes/chain/fixtures';
import { chainDerived, chainPose, manifest } from '../../src/scenes/chain/manifest';
import { blossomTreeCrownHeight, blossomTreeLimbs, FERRY_SECONDS } from '../../src/scenes/chain/actors';
import {
  BANK_RISE,
  CAP_HEIGHT,
  CAP_R_FOOT,
  CAP_R_TOP,
  CAP_R_WAIST,
  CAP_SCALE_MAX,
  CAP_SCALE_MIN,
  CAP_WAIST_FRAC,
  CREEK_HALF,
  PLINTH_DROP,
  PLINTH_HALF,
  WATER_LEVEL,
  buildWorld,
  capApothem,
  capSides,
  creekSkin,
  makeCreekHeight,
  resolveGrid,
  stoneCapGeometry,
  stoneX,
} from '../../src/scenes/chain/world';
import { LANTERN_BASE_RADIUS } from '../../src/scenes/chain/actors';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 64;

function geometryOf(): EnvGeometry {
  return chainDecoder.geometry(chainDecoder.initStatic(SEED), SEED);
}

async function live(tier: QualityTier = 'hero'): Promise<{ module: SceneModule; ctx: StubSceneContext }> {
  const module = createScene();
  const ctx = makeSceneContext({ geometry: geometryOf(), seed: SEED, tier });
  await module.init(ctx);
  return { module, ctx };
}

function fixtureNamed(name: string): SceneFixture {
  const f = fixtures.find((x) => x.name === name);
  if (!f) throw new Error(`fixture "${name}" missing`);
  return f;
}

function posFixture(name: string, pos: number, step = 0, action = 1): SceneFixture {
  return { name, pose: chainPose(pos), ctrl: [action, 0, 0, 0, 0, 0], derived: chainDerived(pos), step };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function capColor(module: SceneModule, index: number): THREE.Color {
  const mesh = module.scene.getObjectByName('chainStoneCaps');
  if (!(mesh instanceof THREE.InstancedMesh)) throw new Error('stone caps missing');
  const c = new THREE.Color();
  mesh.getColorAt(index, c);
  return c;
}

function heartScale(module: SceneModule): number {
  return module.scene.getObjectByName('blossomTreeHeart')?.scale.x ?? -1;
}

export interface NdcBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function ndcBounds(module: SceneModule, camera: THREE.Camera, names: readonly string[]): NdcBounds {
  const out: NdcBounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  const v = new THREE.Vector3();
  const m4 = new THREE.Matrix4();
  const scale = new THREE.Vector3();
  const take = (x: number, y: number, z: number, m: THREE.Matrix4): void => {
    v.set(x, y, z).applyMatrix4(m).project(camera);
    out.minX = Math.min(out.minX, v.x);
    out.maxX = Math.max(out.maxX, v.x);
    out.minY = Math.min(out.minY, v.y);
    out.maxY = Math.max(out.maxY, v.y);
  };
  for (const name of names) {
    const object = module.scene.getObjectByName(name);
    if (!object) throw new Error(`mesh "${name}" missing`);
    object.updateWorldMatrix(true, false);
    const mesh = object as THREE.Mesh;
    mesh.geometry.computeBoundingBox();
    mesh.geometry.computeBoundingSphere();
    const box = mesh.geometry.boundingBox;
    const radius = mesh.geometry.boundingSphere?.radius ?? 0;
    if (!box) continue;
    if (object instanceof THREE.InstancedMesh) {
      for (let i = 0; i < object.count; i++) {
        object.getMatrixAt(i, m4);
        scale.setFromMatrixScale(m4);
        m4.premultiply(object.matrixWorld);
        const r = radius * Math.max(scale.x, scale.y, scale.z);
        for (const dx of [-r, r]) for (const dy of [-r, r]) for (const dz of [-r, r]) take(dx, dy, dz, m4);
      }
    } else {
      for (const x of [box.min.x, box.max.x])
        for (const y of [box.min.y, box.max.y])
          for (const z of [box.min.z, box.max.z]) take(x, y, z, object.matrixWorld);
    }
  }
  return out;
}

function instancedNamed(module: SceneModule, name: string): THREE.InstancedMesh {
  const o = module.scene.getObjectByName(name);
  if (!(o instanceof THREE.InstancedMesh)) throw new Error(`instanced mesh "${name}" missing`);
  return o;
}

function worldY(object: THREE.Object3D): number {
  let y = 0;
  let node: THREE.Object3D | null = object;
  while (node) {
    y += node.position.y;
    node = node.parent;
  }
  return y;
}

function plinthClearance(module: SceneModule, tier: QualityTier): number {
  const caps = instancedNamed(module, 'chainStoneCaps');
  const plinths = instancedNamed(module, 'oakCellFloorMesh');
  plinths.geometry.computeBoundingBox();
  const box = plinths.geometry.boundingBox;
  if (!box) throw new Error('plinth has no bounding box');
  const half = box.max.x;
  const plinthTop = worldY(plinths) + box.max.y;
  const sides = capSides(tier);

  const m4 = new THREE.Matrix4();
  const capPos = new THREE.Vector3();
  const capQuat = new THREE.Quaternion();
  const capScale = new THREE.Vector3();
  const slabPos = new THREE.Vector3();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');

  let clearance = Infinity;
  for (let i = 0; i < caps.count; i++) {
    caps.getMatrixAt(i, m4);
    m4.decompose(capPos, capQuat, capScale);
    euler.setFromQuaternion(capQuat, 'YXZ');
    const yaw = euler.y;
    plinths.getMatrixAt(i, m4);
    slabPos.setFromMatrixPosition(m4);

    const capFoot = capPos.y - CAP_HEIGHT * capScale.y;
    for (let s = 0; s <= 16; s++) {
      const y = plinthTop + ((capFoot - plinthTop) * s) / 16;
      const localY = (y - capPos.y) / capScale.y;
      const waist = -CAP_HEIGHT * CAP_WAIST_FRAC;
      const r =
        localY >= 0
          ? CAP_R_TOP
          : localY >= waist
            ? CAP_R_TOP + (CAP_R_WAIST - CAP_R_TOP) * (localY / waist)
            : CAP_R_WAIST + (CAP_R_FOOT - CAP_R_WAIST) * ((localY - waist) / (-CAP_HEIGHT - waist));
      const apothem = r * capScale.x * Math.cos(Math.PI / sides);
      for (const sx of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const dx = slabPos.x + sx * half - capPos.x;
          const dz = slabPos.z + sz * half - capPos.z;
          const lx = dx * Math.cos(yaw) - dz * Math.sin(yaw);
          const lz = dx * Math.sin(yaw) + dz * Math.cos(yaw);
          let outside = -Infinity;
          for (let k = 0; k < sides; k++) {
            const a = ((k + 0.5) / sides) * Math.PI * 2;
            const d = lx * Math.cos(a) + lz * Math.sin(a) - apothem;
            if (d > outside) outside = d;
          }
          if (-outside < clearance) clearance = -outside;
        }
      }
    }
  }
  return clearance;
}


describe('chain / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('Chain-cont-v0');
    expect(manifest.slug).toBe('chain');
    expect(manifest.accent).toBe('chain');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a pose the record can actually contain', () => {
    const p = manifest.thumbFrame.pose;
    expect(p[0]).toBeGreaterThanOrEqual(0);
    expect(p[0]).toBeLessThan(CHAIN_END);
    expect(p[1]).toBe(CHAIN_END);
    expect(manifest.thumbFrame.derived).toEqual(chainDerived(p[0] ?? 0));
  });

  test('ships at least three canonical poses, none of them the un-observable stone 15', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const pos = f.pose[0] ?? -1;
      expect(pos).toBeGreaterThanOrEqual(0);
      expect(pos).toBeLessThan(CHAIN_END);
      expect(f.pose[1]).toBe(CHAIN_END);
      expect(f.derived).toEqual(chainDerived(pos));
    }
  });

  test('the fixtures cover the start, mid-crossing, the last observed stone and the wrap', () => {
    expect(fixtureNamed('first-stone').pose[0]).toBe(0);
    expect(fixtureNamed('last-stone').pose[0]).toBe(CHAIN_END - 1);
    expect(fixtureNamed('wrapped-to-start').pose[0]).toBe(0);
    const mid = fixtureNamed('midstream').pose[0] ?? 0;
    expect(mid).toBeGreaterThan(2);
    expect(mid).toBeLessThan(CHAIN_END - 2);
  });
});

describe('chain / the corridor is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own 16x1 corridor", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(CHAIN_L);
    expect(grid.h).toBe(1);
    expect(grid.goal).toEqual({ x: CHAIN_END, y: 0 });
    for (let x = 0; x < grid.w; x++) expect(isWall(grid, x, 0)).toBe(false);
  });

  test('the ford is exactly sixteen stones, one world unit apart, centred on the origin', () => {
    const grid = resolveGrid(geometryOf());
    expect(grid.w).toBe(16);
    for (let i = 0; i < grid.w; i++) {
      expect(stoneX(i, grid)).toBeCloseTo((i - 7.5) * CELL, 10);
    }
    expect(stoneX(0, grid)).toBeCloseTo(-stoneX(CHAIN_END, grid), 10);
  });

  test('readPos prefers the decoder state and falls back to the raw pose', () => {
    const grid = resolveGrid(geometryOf());
    const withDerived = { pose: new Float32Array([9, 15, 0, 0, 0, 0, 0, 0, 0]), ctrl: new Float32Array(6), done: false, step: 0, simTime: 0, derived: chainDerived(4) };
    expect(readPos(withDerived, grid)).toBe(4);
    const poseOnly = { ...withDerived, derived: null };
    expect(readPos(poseOnly, grid)).toBe(9);
    const wild = { ...withDerived, derived: chainDerived(999) };
    expect(readPos(wild, grid)).toBe(CHAIN_L - 1);
    const negative = { ...withDerived, derived: chainDerived(-4) };
    expect(readPos(negative, grid)).toBe(0);
  });
});

describe('chain / the creek and the stones', () => {
  test('a stone cap presents its crown at EXACTLY y = 0 (the wisp stands on it)', () => {
    const g = stoneCapGeometry(7, 0.5);
    const pos = g.getAttribute('position');
    let top = -Infinity;
    let bottom = Infinity;
    let maxRadius = 0;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      if (y > top) top = y;
      if (y < bottom) bottom = y;
      maxRadius = Math.max(maxRadius, Math.hypot(pos.getX(i), pos.getZ(i)));
    }
    expect(top).toBe(0);
    expect(bottom).toBeCloseTo(-0.5, 10);
    expect(maxRadius).toBeLessThan(CELL * 0.5);
    g.dispose();
  });

  test('the plinth is CONTAINED by its cap — the derivation', () => {
    for (const [tier, sides] of [
      ['hero', 7],
      ['medium', 6],
      ['thumb', 5],
    ] as const) {
      expect(`${tier}:${capSides(tier)}`).toBe(`${tier}:${sides}`);
      expect(PLINTH_HALF * Math.SQRT2).toBeLessThan(capApothem(sides, CAP_SCALE_MIN));
    }
    expect(capApothem(5, CAP_SCALE_MIN)).toBeLessThan(CAP_R_TOP);
    expect(PLINTH_HALF * 2).toBeGreaterThan(0.3);
  });

  test('...and the measured ford agrees, at every tier, on every stone, at every depth', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      const caps = instancedNamed(module, 'chainStoneCaps');
      const m4 = new THREE.Matrix4();
      const scale = new THREE.Vector3();
      for (let i = 0; i < caps.count; i++) {
        caps.getMatrixAt(i, m4);
        scale.setFromMatrixScale(m4);
        expect(scale.x).toBeGreaterThanOrEqual(CAP_SCALE_MIN - 1e-9);
        expect(scale.x).toBeLessThanOrEqual(CAP_SCALE_MAX + 1e-9);
        expect(scale.z).toBeCloseTo(scale.x, 12);
      }
      const clearance = plinthClearance(module, tier);
      expect(`${tier} clearance=${clearance > 0.02}`).toBe(`${tier} clearance=true`);
      expect(clearance).toBeGreaterThan(0.02);
      module.dispose();
    }
  });

  test('the plinth hides under the crown, not above it', () => {
    expect(PLINTH_DROP).toBeGreaterThan(0);
    expect(PLINTH_DROP).toBeLessThan(Math.abs(WATER_LEVEL));
  });

  test('the creek bed is a ford: submerged across the crossing, dry beyond both banks', () => {
    const h = makeCreekHeight(SEED);
    const grid = resolveGrid(geometryOf());
    for (let i = 0; i < grid.w; i++) {
      expect(h(stoneX(i, grid), 0)).toBeLessThan(WATER_LEVEL - 0.2);
    }
    for (const side of [-1, 1]) {
      const x = side * (CREEK_HALF + BANK_RISE + 1.5);
      for (let j = -4; j <= 4; j++) expect(h(x, j * 3)).toBeGreaterThan(WATER_LEVEL + 0.3);
    }
  });

  test('the skin is all token blends, and no two stone tones collide', () => {
    const skin = creekSkin(TOKENS.palettes.afternoon, TOKENS);
    const hexes = new Set([skin.stoneDry.getHex(), skin.stoneWet.getHex(), skin.stoneDeep.getHex(), skin.stoneGlow.getHex()]);
    expect(hexes.size).toBe(4);
    for (const c of [skin.stoneDry, skin.stoneWet, skin.bed, skin.bank, skin.bankHigh]) {
      expect(Math.max(c.r, c.g, c.b)).toBeLessThan(0.78);
    }
  });

  test('the blossom-tree skeleton puts every twig above the stone it grows from', () => {
    const limbs = blossomTreeLimbs();
    expect(limbs.length).toBeGreaterThan(8);
    const twigs = limbs.filter((l) => l.twig);
    expect(twigs.length).toBe(12);
    for (const limb of limbs) {
      expect(limb.a[1]).toBeGreaterThanOrEqual(0);
      expect(limb.b[1]).toBeGreaterThan(0);
    }
    expect(blossomTreeCrownHeight()).toBeGreaterThan(2.2);
  });
});

describe('chain / the bank is ground, not the y = 0 plane', () => {
  function terrainFor(seed: number): { sample: (x: number, z: number) => number; world: ReturnType<typeof buildWorld> } {
    const geometry = chainDecoder.geometry(chainDecoder.initStatic(seed), seed);
    const scene = new THREE.Scene();
    const world = buildWorld(makeSceneContext({ geometry, seed, tier: 'hero' }), scene, resolveGrid(geometry));
    return { sample: (x, z) => world.terrain.sampleHeight(x, z), world };
  }

  test('the ford lantern is planted on the bank, dry, on every lane', async () => {
    for (const seed of [64, 65, 66, 70]) {
      const { sample, world } = terrainFor(seed);
      const geometry = chainDecoder.geometry(chainDecoder.initStatic(seed), seed);
      const grid = resolveGrid(geometry);
      const module = createScene();
      await module.init(makeSceneContext({ geometry, seed, tier: 'hero' }));
      module.update(toFrameUpdate(fixtureNamed('midstream')));

      const bodyNode = module.scene.getObjectByName('fordLanternBody');
      if (!bodyNode) throw new Error('lantern art missing');
      bodyNode.updateWorldMatrix(true, false);
      const art = new THREE.Vector3().setFromMatrixPosition(bodyNode.matrixWorld);

      expect(`seed${seed} dry`).toBe(`seed${seed} ${art.y >= WATER_LEVEL + LANTERN_DRY - 1e-9 ? 'dry' : `wet(${art.y.toFixed(3)})`}`);
      let lowest = Infinity;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        const g = sample(art.x + Math.cos(a) * LANTERN_BASE_RADIUS, art.z + Math.sin(a) * LANTERN_BASE_RADIUS);
        expect(g).toBeGreaterThanOrEqual(art.y - 1e-6);
        if (g < lowest) lowest = g;
      }
      expect(lowest - art.y).toBeLessThan(1e-6);
      expect(art.z).toBeCloseTo(LANTERN_Z, 10);
      expect(art.x).toBeLessThan(stoneX(0, grid));
      const root = worldOf(module, 'ford-lantern');
      const truth: [number, number, number] = [0, 0, 0];
      cellToWorld(-1, 0, grid, truth);
      expect(root.x).toBe(truth[0]);
      expect(root.y).toBe(0);
      expect(root.z).toBe(truth[2]);

      module.dispose();
      world.dispose();
    }
  });

  test('the lantern walks out only as far as the bank makes it, and never off the scan', () => {
    const flat = groundedLanternOffset(() => 0.4, -8.5);
    expect(flat[0]).toBeCloseTo(-0.45, 10);
    expect(flat[1]).toBeCloseTo(0.4, 10);
    const shore = (x: number): number => (x < -10 ? 0.5 : -0.8);
    const walked = groundedLanternOffset((x) => shore(x), -8.5);
    expect(walked[0]).toBeLessThan(-1.0);
    expect(walked[1]).toBeCloseTo(0.5, 10);
    const drowned = groundedLanternOffset((x) => -0.9 - Math.abs(x + 9) * 0.01, -8.5);
    expect(Number.isFinite(drowned[0])).toBe(true);
    expect(Number.isFinite(drowned[1])).toBe(true);
    expect(drowned[2]).toBeCloseTo(LANTERN_Z, 10);
  });

  test('every scatter is rooted on the ground — the shallow reeds included', () => {
    const { sample, world } = terrainFor(SEED);
    const beds: string[] = [];
    for (const name of ['shallowReeds', 'creekReedsNear', 'creekReedsFar', 'bankMeadow']) {
      const group = world.group.getObjectByName(name);
      if (!group) throw new Error(`scatter "${name}" missing`);
      const m4 = new THREE.Matrix4();
      const pos = new THREE.Vector3();
      const scale = new THREE.Vector3();
      let checked = 0;
      let emergent = 0;
      let worst = 0;
      group.traverse((o) => {
        if (!(o instanceof THREE.InstancedMesh)) return;
        o.geometry.computeBoundingBox();
        const top = o.geometry.boundingBox?.max.y ?? 0;
        for (let i = 0; i < o.count; i++) {
          o.getMatrixAt(i, m4);
          pos.setFromMatrixPosition(m4);
          scale.setFromMatrixScale(m4);
          worst = Math.max(worst, Math.abs(pos.y - sample(pos.x, pos.z)));
          if (pos.y + top * scale.y > WATER_LEVEL) emergent++;
          checked++;
        }
      });
      expect(checked).toBeGreaterThan(0);
      beds.push(`${name}: rooted=${worst < 1e-6}`);
      if (name === 'shallowReeds') {
        expect(emergent / checked).toBeGreaterThan(0.4);
      }
    }
    expect(beds).toEqual([
      'shallowReeds: rooted=true',
      'creekReedsNear: rooted=true',
      'creekReedsFar: rooted=true',
      'bankMeadow: rooted=true',
    ]);
    world.dispose();
  });
});

describe('chain / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('blossom-tree') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('ford-lantern') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      for (const name of ['wisp', 'wisp-anchor', 'blossom-tree', 'ford-lantern']) {
        expect(report.worst.get(name) ?? 1).toBeLessThan(1e-9);
      }
      module.dispose();
    }
  });

  test('the wisp lands on EVERY stone of the ford, including the one it is never observed on', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < grid.w; i++) {
      module.update(toFrameUpdate(posFixture(`stone${i}`, i, i)));
      const w = worldOf(module, 'wisp-anchor');
      cellToWorld(i, 0, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      expect(worldOf(module, 'wisp').y).toBeCloseTo(TOKENS.wispHeight, 10);
    }
    module.dispose();
  });

  test('the tree owns the paying stone and the lantern the cell upstream of the first', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    module.update(toFrameUpdate(fixtureNamed('midstream')));

    const tree = worldOf(module, 'blossom-tree');
    cellToWorld(CHAIN_END, 0, grid, truth);
    expect(tree.x).toBeCloseTo(truth[0], 10);
    expect(tree.y).toBeCloseTo(0, 10);
    expect(tree.z).toBeCloseTo(truth[2], 10);

    const lantern = worldOf(module, 'ford-lantern');
    cellToWorld(-1, 0, grid, truth);
    expect(lantern.x).toBeCloseTo(truth[0], 10);
    expect(lantern.y).toBeCloseTo(0, 10);
    expect(lantern.x).toBeCloseTo(stoneX(0, grid) - CELL, 10);
    module.dispose();
  });
});

describe('chain / the hop is bounded, and the wrap is a cut', () => {
  test('a hop stays between the two stones and lands within one control step', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(posFixture('a', 6, 0)));
    cellToWorld(6, 0, grid, from);
    cellToWorld(7, 0, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(posFixture('b', 7, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(posFixture('b', 7, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp-anchor');
      expect(w.x).toBeGreaterThanOrEqual(from[0] - 1e-6);
      expect(w.x).toBeLessThanOrEqual(to[0] + 1e-6);
      expect(w.z).toBeCloseTo(0, 6);
      expect(w.y).toBeGreaterThanOrEqual(-1e-6);
    }
    const landed = worldOf(module, 'wisp-anchor');
    expect(landed.x).toBeCloseTo(to[0], 10);
    expect(landed.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('the wrap is a HARD CUT: on the goal frame the wisp is already on stone 0', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];

    ctx.setTime(20);
    module.update(toFrameUpdate(posFixture('walking', CHAIN_END - 1, 45), { scrubbed: false }));
    ctx.setTime(20.125);
    module.update(
      toFrameUpdate(posFixture('paid', 0, 46), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(0, 0, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);

    ctx.setTime(20.19);
    module.update(toFrameUpdate(posFixture('paid', 0, 46), { scrubbed: false }));
    expect(worldOf(module, 'wisp-anchor').x).toBeCloseTo(truth[0], 10);
    module.dispose();
  });

  test('a scrub snaps the wisp clean across the ford with no glide', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(posFixture('a', 1, 0), { scrubbed: false }));
    module.update(toFrameUpdate(posFixture('far', 13, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(13, 0, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });
});

describe('chain / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module, ctx } = await live();
    module.update(toFrameUpdate(fixtureNamed('first-stone')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 80; i++) {
      ctx.setTime(i * 0.125);
      module.update(
        toFrameUpdate(posFixture('walk', i % CHAIN_END, i), {
          scrubbed: false,
          events: i % 15 === 14 ? [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }] : [],
        }),
      );
    }
    let after = 0;
    module.scene.traverse(() => after++);
    expect(after).toBe(count);
    module.dispose();
  });

  test('celebrations fire ONLY from frame.events', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    ctx.setTime(1);
    module.update(toFrameUpdate(posFixture('almost', CHAIN_END - 1, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(1.125);
    module.update(toFrameUpdate(posFixture('home', 0, 101), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);

    ctx.setTime(1.25);
    module.update(
      toFrameUpdate(posFixture('paid', 0, 102), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('the bloom happens at STONE 15, not where the wrapped agent is standing', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(3);
    module.update(toFrameUpdate(posFixture('walking', CHAIN_END - 1, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(posFixture('paid', 0, 201), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );

    const flash = module.scene.getObjectByName('oakPoint2');
    expect(flash).toBeDefined();
    if (!flash) return;
    expect(flash.position.x).toBeGreaterThan(stoneX(CHAIN_END, grid) - 0.01);
    expect(flash.position.z).toBeCloseTo(0, 10);
    expect(flash.position.y).toBeGreaterThan(1);
    expect((flash as THREE.PointLight).intensity).toBeGreaterThan(0);
    expect(flash.position.x - stoneX(0, grid)).toBeGreaterThan(CHAIN_END - 1);

    expect(worldOf(module, 'wisp-anchor').x).toBeCloseTo(stoneX(0, grid), 10);
    expect(heartScale(module)).toBeCloseTo(0.86, 6);
    module.dispose();
  });

  test('a scrub clears the bloom, the ferry, the ripples and every warm stone', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');

    const rest: number[] = [];
    for (let i = 0; i < CHAIN_L; i++) rest.push(capColor(module, i).getHex());

    ctx.setTime(1);
    module.update(toFrameUpdate(posFixture('step', 5, 300), { scrubbed: false }));
    expect(capColor(module, 5).getHex()).not.toBe(rest[5]);

    module.update(
      toFrameUpdate(posFixture('paid', 0, 301), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    expect(heartScale(module)).toBeGreaterThan(0.8);

    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('midstream'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(heartScale(module)).toBeCloseTo(0.42, 6);
    for (let i = 0; i < CHAIN_L; i++) {
      expect(`stone${i}=${capColor(module, i).getHex()}`).toBe(`stone${i}=${rest[i]}`);
    }
    module.dispose();
  });

  test('the warm stone cools on its own within a couple of seconds', async () => {
    const { module, ctx } = await live();
    const rest = capColor(module, 9).getHex();
    ctx.setTime(50);
    module.update(toFrameUpdate(posFixture('step', 9, 400), { scrubbed: false }));
    expect(capColor(module, 9).getHex()).not.toBe(rest);
    ctx.setTime(53.5);
    module.update(toFrameUpdate(posFixture('step', 9, 400), { scrubbed: false }));
    expect(capColor(module, 9).getHex()).toBe(rest);
    module.dispose();
  });

  test('the envelopes are pure, bounded, and settle to nothing', () => {
    for (const fn of [treeBloom, stoneGlow, flashEnvelope]) {
      expect(fn(-Infinity)).toBe(0);
      expect(fn(Number.NaN)).toBe(0);
      expect(fn(0)).toBeGreaterThan(0.9);
      expect(fn(1e6)).toBe(0);
      for (let i = 0; i <= 60; i++) {
        const v = fn((i / 60) * 4);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
    expect(treeBloom(2.3)).toBeGreaterThan(0.3);
    expect(treeBloom(12)).toBeLessThan(0.01);
    expect(FERRY_SECONDS).toBeLessThan(2.3);
  });
});

describe('chain / lifecycle', () => {
  test('init -> dispose round-trips clean three times', async () => {
    const report = await assertInitDisposeRoundTrip(createScene, {
      geometry: geometryOf(),
      seed: SEED,
      tier: 'hero',
      fixtures,
      cycles: 3,
    });
    expect(report.census.length).toBe(3);
    expect((report.census[0]?.geometries.size ?? 0) > 0).toBe(true);
  });

  test('every tier round-trips clean too', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      await assertInitDisposeRoundTrip(createScene, {
        geometry: geometryOf(),
        seed: SEED,
        tier,
        fixtures,
        cycles: 2,
      });
    }
  });

  test('setDebug toggles the ground-truth overlay onto the scene camera', async () => {
    const { module } = await live();
    module.setDebug(true);
    module.update(toFrameUpdate(fixtureNamed('last-stone')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'creekGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'creekGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the composed frame holds the whole crown, not just its centre point', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(fixtureNamed('midstream')));
    module.update(
      toFrameUpdate(posFixture('paid', 0, 1), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    module.resize({ width: 1600, height: 900, dpr: 1 });
    const cam = module.camera;
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();

    const v = new THREE.Vector3();
    for (let i = 0; i < grid.w; i++) {
      v.set(stoneX(i, grid), 0, 0).project(cam);
      expect(Math.abs(v.x)).toBeLessThan(0.85);
      expect(Math.abs(v.y)).toBeLessThan(0.85);
    }

    const crown = ndcBounds(module, cam, ['blossomTreeBark', 'blossomTreePetals']);
    expect(`crown ndcX max ${crown.maxX < 0.94}`).toBe('crown ndcX max true');
    expect(crown.maxX).toBeLessThan(0.94);
    expect(crown.minX).toBeGreaterThan(-0.94);
    expect(crown.maxY).toBeLessThan(0.94);
    expect(crown.minY).toBeGreaterThan(-0.94);
    expect(crown.maxX).toBeGreaterThan(0.6);

    const lantern = ndcBounds(module, cam, ['fordLanternMesh']);
    expect(Math.max(Math.abs(lantern.minX), Math.abs(lantern.maxX))).toBeLessThan(0.96);
    expect(Math.max(Math.abs(lantern.minY), Math.abs(lantern.maxY))).toBeLessThan(0.96);

    expect(cam.position.y).toBeGreaterThan(4);
    expect(cam.position.y).toBeLessThan(14);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.275);
    expect(dir.y).toBeGreaterThan(-0.616);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    v.set(stoneX(CHAIN_END, grid), blossomTreeCrownHeight(), 0).project(cam);
    expect(Math.abs(v.x)).toBeLessThan(0.94);
    module.dispose();
  });

  test('the thumb frame holds the whole ford AT A SQUARE CROP (the wall-view contract)', async () => {
    const { module } = await live('thumb');
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(posFixture('thumb', manifest.thumbFrame.pose[0] ?? 12, 42)));

    for (const [w, h, limit] of [
      [900, 900, 0.94],
      [1600, 900, 0.94],
    ] as const) {
      module.resize({ width: w, height: h, dpr: 1 });
      const cam = module.camera;
      cam.updateMatrixWorld(true);
      cam.updateProjectionMatrix();
      const v = new THREE.Vector3();
      for (let i = 0; i < grid.w; i++) {
        v.set(stoneX(i, grid), 0, 0).project(cam);
        expect(`${w}x${h} stone${i} inside=${Math.abs(v.x) < limit && Math.abs(v.y) < limit}`).toBe(
          `${w}x${h} stone${i} inside=true`,
        );
      }
      const ford = ndcBounds(module, cam, ['chainStoneCaps']);
      expect(`${w}x${h} caps inside=${Math.max(Math.abs(ford.minX), Math.abs(ford.maxX)) < limit}`).toBe(
        `${w}x${h} caps inside=true`,
      );
      const crown = ndcBounds(module, cam, ['blossomTreeBark', 'blossomTreePetals']);
      expect(Math.max(Math.abs(crown.minX), Math.abs(crown.maxX))).toBeLessThan(limit);
      expect(Math.max(Math.abs(crown.minY), Math.abs(crown.maxY))).toBeLessThan(limit);
    }
    module.dispose();
  });
});

describe('chain / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'chain (thumb, overlay on)');
    module.dispose();
  });
});
