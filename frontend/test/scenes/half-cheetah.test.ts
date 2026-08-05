/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { mulberry32, rngAt } from '@kit/rng';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import { halfCheetahDecoder, CHEETAH_DT, CHEETAH_TELEPORT_DX } from '@sim/envs/halfcheetah';
import {
  CHEETAH_BONES,
  CHEETAH_HEAD_RADIUS,
  CHEETAH_RADIUS,
  FK_POINT_INDEX,
  ROOT_Z0,
  fk,
} from '@sim/fk/cheetah';
import type { EnvGeometry } from '@sim/types';

import createScene, { CAM_OFFSET, TRACK_Y, envelope } from '../../src/scenes/half-cheetah/index';
import { REFERENCE_SPEED } from '../../src/scenes/half-cheetah/actors';
import { fixtures } from '../../src/scenes/half-cheetah/fixtures';
import { manifest } from '../../src/scenes/half-cheetah/manifest';
import {
  GROUND_PERIOD,
  LANE_HALF_Z,
  TILE,
  bankKey,
  savannaHeight,
} from '../../src/scenes/half-cheetah/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameState,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 0;

function geometryOf(): EnvGeometry {
  return halfCheetahDecoder.geometry(halfCheetahDecoder.initStatic(SEED), SEED);
}

async function live(
  tier: QualityTier = 'hero',
  seed: number = SEED,
  geometry: EnvGeometry = geometryOf(),
): Promise<{ module: SceneModule; ctx: StubSceneContext }> {
  const module = createScene();
  const ctx = makeSceneContext({ geometry, seed, tier });
  await module.init(ctx);
  return { module, ctx };
}

function lowestSurfaceY(module: SceneModule, match: (name: string) => boolean): number {
  const v = new THREE.Vector3();
  let lowest = Infinity;
  module.scene.updateMatrixWorld(true);
  module.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || !match(mesh.name)) return;
    const pos = mesh.geometry.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      v.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
      if (v.y < lowest) lowest = v.y;
    }
  });
  return lowest;
}

const isTail = (name: string): boolean => name.startsWith('cheetahTailMesh');
const isBone = (name: string): boolean => name.startsWith('cheetah_');

function atSpeed(fixture: SceneFixture, velX: number): SceneFixture {
  const derived = (fixture.derived as Record<string, unknown> | undefined) ?? {};
  return { ...fixture, derived: { ...derived, velX } };
}

function bankMatrices(module: SceneModule, name: string): number[] {
  const mesh = module.scene.getObjectByName(name) as THREE.InstancedMesh | undefined;
  if (!mesh?.isInstancedMesh) throw new Error(`bank "${name}" missing`);
  return [...mesh.instanceMatrix.array];
}

function fixtureNamed(name: string): SceneFixture {
  const f = fixtures.find((x) => x.name === name);
  if (!f) throw new Error(`fixture "${name}" missing`);
  return f;
}

function poseOf(fixture: SceneFixture): Float32Array {
  const p = new Float32Array(9);
  for (let i = 0; i < 9; i++) p[i] = fixture.pose[i] ?? 0;
  return p;
}

function worldOf(object: THREE.Object3D): THREE.Vector3 {
  object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(object.matrixWorld);
}

function worldSnapshot(module: SceneModule): number[] {
  const out: number[] = [];
  module.scene.updateMatrixWorld(true);
  module.scene.traverse((o) => {
    out.push(o.visible ? 1 : 0);
    for (const v of o.matrixWorld.elements) out.push(v);
    const inst = o as THREE.InstancedMesh;
    if (inst.isInstancedMesh && inst.instanceMatrix) {
      const a = inst.instanceMatrix.array;
      for (let i = 0; i < a.length; i += 7) out.push(a[i] ?? 0);
    }
  });
  return out;
}


describe('half-cheetah / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('HalfCheetah-v4');
    expect(manifest.slug).toBe('half-cheetah');
    expect(manifest.accent).toBe('half-cheetah');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a REAL recorded pose (the hero stride), not a posed one', () => {
    const stride = fixtureNamed('stride-plant');
    expect([...manifest.thumbFrame.pose]).toEqual([...stride.pose]);
    const truth = fk(poseOf(stride));
    expect(truth[FK_POINT_INDEX.toe_b * 2 + 1] ?? 1).toBeLessThan(CHEETAH_RADIUS);
    expect(truth[FK_POINT_INDEX.toe_f * 2 + 1] ?? 0).toBeGreaterThan(0.15);
  });

  test('ships at least three canonical poses, all physically plausible', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const out = fk(poseOf(f));
      const rootY = out[1] ?? 0;
      expect(rootY).toBeGreaterThan(0.15);
      expect(rootY).toBeLessThan(1.2);
      expect(rootY).toBeCloseTo(ROOT_Z0 + (f.pose[1] ?? 0), 6);
      expect(out[FK_POINT_INDEX.toe_b * 2 + 1] ?? 0).toBeGreaterThan(-CHEETAH_RADIUS);
      expect(out[FK_POINT_INDEX.toe_f * 2 + 1] ?? 0).toBeGreaterThan(-CHEETAH_RADIUS);
    }
  });

  test('the fixtures cover both ends of the stream: a drop, a stride, and 431 m out', () => {
    const airborne = fixtures.find((f) => (fk(poseOf(f))[FK_POINT_INDEX.toe_b * 2 + 1] ?? 0) > 0.15);
    expect(airborne).toBeDefined();
    const far = fixtures.find((f) => Math.abs(f.pose[0] ?? 0) > 400);
    expect(far).toBeDefined();
    const fast = fixtures.find((f) => Math.abs((f.derived as { velX: number }).velX) > 0.9);
    expect(fast).toBeDefined();
  });
});

describe('half-cheetah / the savanna floor is the contact plane', () => {
  test('savannaHeight is IDENTICALLY 0 across the whole running lane, at any distance', () => {
    for (const x of [-40, -1.3, 0, 0.7, 12.5, 189.54, 431.1, 1871.3]) {
      for (const z of [0, 0.2, -0.4, LANE_HALF_Z, -LANE_HALF_Z]) {
        expect(savannaHeight(x, z)).toBe(0);
      }
    }
  });

  test('every fixture toe stands on ground the scene actually renders as y = 0', () => {
    for (const f of fixtures) {
      const out = fk(poseOf(f));
      for (const name of ['toe_b', 'toe_f'] as const) {
        const tx = out[FK_POINT_INDEX[name] * 2] ?? 0;
        expect(savannaHeight(tx, 0)).toBe(0);
      }
    }
  });

  test('the plain really does roll once you leave the lane (it is not a flat extrusion)', () => {
    let deviation = 0;
    for (let i = 0; i <= 60; i++) {
      deviation = Math.max(deviation, Math.abs(savannaHeight(i * 0.9, -26)));
    }
    expect(deviation).toBeGreaterThan(0.2);
  });

  test('the height field is periodic in x, which is what makes the treadmill seamless', () => {
    for (const z of [-3, -12, -30, 5, 18]) {
      for (let i = 0; i < 40; i++) {
        const x = i * 1.7;
        expect(savannaHeight(x + GROUND_PERIOD, z)).toBeCloseTo(savannaHeight(x, z), 9);
        expect(savannaHeight(x - 3 * GROUND_PERIOD, z)).toBeCloseTo(savannaHeight(x, z), 9);
      }
    }
  });
});

describe('half-cheetah / alignment', () => {
  test('every probe lands on ground truth EXACTLY at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    for (const name of ['torso', 'toe-back', 'toe-front', 'head']) {
      expect(report.worst.get(name) ?? 1).toBe(0);
    }
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      for (const name of ['torso', 'toe-back', 'toe-front', 'head']) {
        expect(report.worst.get(name) ?? 1).toBe(0);
      }
      module.dispose();
    }
  });

  test('the torso probe is exactly (rootx, ROOT_Z0 + rootz) across a swept stride', async () => {
    const { module } = await live();
    for (let i = 0; i <= 40; i++) {
      const rootX = -3 + i * 9.7;
      const rootZ = -0.16 + (i % 7) * 0.03;
      const pose = [rootX, rootZ, 0.08, 0.2, -0.1, 0.05, -0.2, 0.15, -0.05];
      module.update(toFrameUpdate({ name: `sweep${i}`, pose, step: i }));
      const probe = module.probes()[0];
      expect(probe).toBeDefined();
      if (!probe) continue;
      const world = worldOf(probe.object);
      expect(world.x).toBe(Math.fround(rootX));
      expect(world.y).toBeCloseTo(ROOT_Z0 + Math.fround(rootZ), 6);
      expect(world.z).toBe(0);
    }
    module.dispose();
  });
});

describe('half-cheetah / the coat is a rigid skin on the exact skeleton', () => {
  test('every bone mesh spans exactly the two FK joints it claims', async () => {
    const { module } = await live();
    const local = new THREE.Vector3();
    const world = new THREE.Vector3();
    for (const f of fixtures) {
      module.update(toFrameUpdate(f));
      const truth = fk(poseOf(f));
      module.scene.updateMatrixWorld(true);
      for (let b = 0; b < CHEETAH_BONES.length; b++) {
        const bone = CHEETAH_BONES[b];
        if (!bone) continue;
        const mesh = module.scene.getObjectByName(boneName(b));
        expect(mesh).toBeDefined();
        if (!mesh) continue;
        const ax = truth[bone[0] * 2] ?? 0;
        const ay = truth[bone[0] * 2 + 1] ?? 0;
        const bx = truth[bone[1] * 2] ?? 0;
        const by = truth[bone[1] * 2 + 1] ?? 0;
        const half = Math.hypot(bx - ax, by - ay) * 0.5;
        for (const [sign, tx, ty] of [
          [-1, ax, ay],
          [1, bx, by],
        ] as const) {
          local.set(0, sign * half, 0);
          world.copy(local).applyMatrix4(mesh.matrixWorld);
          expect(world.x).toBeCloseTo(tx, 4);
          expect(world.y).toBeCloseTo(ty, 4);
          expect(world.z).toBeCloseTo(0, 9);
        }
      }
    }
    module.dispose();
  });

  test('the head node yaws by the torso pitch — read off the bones, not the pose', async () => {
    const { module } = await live();
    const node = module.scene.getObjectByName('cheetahHeadNode');
    expect(node).toBeDefined();
    if (!node) return;
    for (const f of fixtures) {
      module.update(toFrameUpdate(f));
      const truth = fk(poseOf(f));
      const dx = (truth[FK_POINT_INDEX.ft * 2] ?? 0) - (truth[FK_POINT_INDEX.bt * 2] ?? 0);
      const dy = (truth[FK_POINT_INDEX.ft * 2 + 1] ?? 0) - (truth[FK_POINT_INDEX.bt * 2 + 1] ?? 0);
      expect(node.rotation.z).toBe(Math.atan2(dy, dx));
      expect(node.rotation.z).toBeCloseTo(-(f.pose[2] ?? 0), 4);
      expect(node.position.lengthSq()).toBe(0);
    }
    module.dispose();
  });

  test('the foot contact visuals are a pure function of toe height x speed', async () => {
    const { module } = await live();
    const scuffs = ['cheetahScuff0', 'cheetahScuff1'];

    module.update(toFrameUpdate(fixtureNamed('reset-drop')));
    for (const name of scuffs) {
      expect(module.scene.getObjectByName(name)?.visible).toBe(false);
    }

    module.update(toFrameUpdate(fixtureNamed('stride-extend')));
    expect(module.scene.getObjectByName('cheetahScuff1')?.visible).toBe(false);

    module.update(toFrameUpdate(fixtureNamed('flail-sprawl')));
    for (const name of scuffs) {
      expect(module.scene.getObjectByName(name)?.visible).toBe(true);
    }

    const truth = fk(poseOf(fixtureNamed('flail-sprawl')));
    const back = module.scene.getObjectByName('cheetahScuff0');
    const front = module.scene.getObjectByName('cheetahScuff1');
    expect(back?.position.x).toBeCloseTo(truth[FK_POINT_INDEX.toe_b * 2] ?? 0, 6);
    expect(front?.position.x).toBeCloseTo(truth[FK_POINT_INDEX.toe_f * 2] ?? 0, 6);
    module.dispose();
  });
});

describe('half-cheetah / the tail never breaks the contact plane', () => {
  const SPEEDS = [0, 0.3, 1.1];

  test('no tail surface is ever below y = 0, at any fixture and any speed', async () => {
    const { module } = await live();
    for (const f of fixtures) {
      for (const velX of SPEEDS) {
        module.update(toFrameUpdate(atSpeed(f, velX)));
        const tail = lowestSurfaceY(module, isTail);
        expect(`${f.name}@${velX}: ${tail >= 0}`).toBe(`${f.name}@${velX}: true`);
        expect(tail).toBeGreaterThanOrEqual(CHEETAH_RADIUS);
      }
    }
    module.dispose();
  });

  test('the tail is never the LOWEST part of the animal', async () => {
    const { module } = await live();
    for (const f of fixtures) {
      for (const velX of SPEEDS) {
        module.update(toFrameUpdate(atSpeed(f, velX)));
        const tail = lowestSurfaceY(module, isTail);
        const coat = lowestSurfaceY(module, isBone);
        expect(`${f.name}@${velX}: ${tail > coat}`).toBe(`${f.name}@${velX}: true`);
      }
    }
    module.dispose();
  });

  test('the droop still reads: hanging at rest, streamed out at speed', async () => {
    const { module } = await live();
    const stride = fixtureNamed('stride-plant');
    module.update(toFrameUpdate(atSpeed(stride, 0)));
    const rest = lowestSurfaceY(module, isTail);
    module.update(toFrameUpdate(atSpeed(stride, 1.1)));
    const run = lowestSurfaceY(module, isTail);
    expect(run).toBeGreaterThan(rest + 0.15);
    const hip = fk(poseOf(stride))[FK_POINT_INDEX.bt * 2 + 1] ?? 0;
    expect(rest).toBeLessThan(hip - 0.15);
    module.dispose();
  });

  test('every physically reachable pose keeps the tail above the plane', async () => {
    const { module } = await live();
    const rng = mulberry32(0x7a11);
    const pose = new Float32Array(9);
    let worst = Infinity;
    let checked = 0;
    for (let n = 0; n < 240; n++) {
      pose[0] = (rng() - 0.5) * 40;
      pose[1] = -0.35 + rng() * 0.85;
      pose[2] = (rng() - 0.5) * 3.2;
      for (let i = 3; i < 9; i++) pose[i] = (rng() - 0.5) * 2.4;
      const truth = fk(pose);
      if ((truth[FK_POINT_INDEX.bt * 2 + 1] ?? 0) < 0.062) continue;
      for (const velX of SPEEDS) {
        module.update(toFrameUpdate({ name: `sweep${n}`, pose: [...pose], step: n, derived: { velX } }));
        worst = Math.min(worst, lowestSurfaceY(module, isTail));
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(300);
    expect(`worst tail y over ${checked} poses: ${worst >= 0}`).toBe(`worst tail y over ${checked} poses: true`);
    module.dispose();
  });

  test('the solved tail is still a pure function of state (scrub-safe, no history)', async () => {
    const { module } = await live();
    const stride = fixtureNamed('stride-plant');
    const slow = fixtureNamed('nose-dip');
    module.update(toFrameUpdate(atSpeed(stride, 0.2), { scrubbed: false }));
    const a = worldSnapshot(module);
    module.update(toFrameUpdate(atSpeed(slow, 1.1), { scrubbed: false }));
    module.update(toFrameUpdate(atSpeed(fixtureNamed('reset-drop'), 0), { scrubbed: true }));
    module.update(toFrameUpdate(atSpeed(stride, 0.2), { scrubbed: false }));
    expect(worldSnapshot(module)).toEqual(a);
    module.dispose();
  });
});

describe('half-cheetah / the treadmill is a pure function of rootx', () => {
  test('scrubbing a kilometre away and back rebuilds the identical world', async () => {
    const { module } = await live();
    const at = (x: number): SceneFixture => ({
      name: `at${x}`,
      pose: [x, -0.12, 0.02, 0.1, 0.05, -0.05, -0.1, 0.08, -0.02],
      step: 0,
    });
    module.update(toFrameUpdate(at(0)));
    const before = worldSnapshot(module);
    module.update(toFrameUpdate(at(1000)));
    const away = worldSnapshot(module);
    module.update(toFrameUpdate(at(0)));
    const after = worldSnapshot(module);
    expect(after).toEqual(before);
    expect(away).not.toEqual(before);
    module.dispose();
  });

  test('the ground mesh only ever sits on a period multiple (that is why the snap hides)', async () => {
    const { module } = await live();
    const ground = module.scene.getObjectByName('savannaGround');
    expect(ground).toBeDefined();
    if (!ground) return;
    for (const x of [0, 5.5, 47.9, 120, 189.54, 431.1, -60]) {
      module.update(toFrameUpdate({ name: `g${x}`, pose: [x, -0.1, 0, 0, 0, 0, 0, 0, 0], step: 0 }));
      const offset = ground.position.x / GROUND_PERIOD;
      expect(Math.abs(offset - Math.round(offset))).toBeLessThan(1e-9);
      expect(Math.abs(ground.position.x - x)).toBeLessThanOrEqual(GROUND_PERIOD / 2 + 1e-6);
    }
    module.dispose();
  });

  test('the cairn ruler really is every CAIRN_SPACING metres', async () => {
    const { module } = await live();
    module.update(toFrameUpdate({ name: 'mid', pose: [96.4, -0.1, 0, 0, 0, 0, 0, 0, 0], step: 0 }));
    const cairns = module.scene.getObjectByName('savannaCairns') as THREE.InstancedMesh | undefined;
    expect(cairns).toBeDefined();
    if (!cairns) return;
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const xs: number[] = [];
    for (let i = 0; i < cairns.count; i++) {
      cairns.getMatrixAt(i, m);
      p.setFromMatrixPosition(m);
      xs.push(p.x);
    }
    xs.sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i++) {
      expect((xs[i] ?? 0) - (xs[i - 1] ?? 0)).toBeCloseTo(10, 6);
    }
    expect(xs[0] ?? 0).toBeLessThan(96.4);
    expect(xs[xs.length - 1] ?? 0).toBeGreaterThan(96.4);
    module.dispose();
  });

  test('every bank is keyed to the LANE SEED — two lanes are two plains', async () => {
    const BANKS = ['savannaGrass', 'savannaAcacias', 'savannaRocks', 'savannaCairns'];
    const at = (x: number): SceneFixture => ({
      name: `seedcheck${x}`,
      pose: [x, -0.12, 0.02, 0.1, 0.05, -0.05, -0.1, 0.08, -0.02],
      step: 0,
    });
    const a = await live('hero', 0);
    const b = await live('hero', 987654);
    const again = await live('hero', 0);
    for (const m of [a, b, again]) m.module.update(toFrameUpdate(at(240)));
    for (const bank of BANKS) {
      expect(`${bank} differs by seed: ${bankMatrices(b.module, bank).join() !== bankMatrices(a.module, bank).join()}`).toBe(
        `${bank} differs by seed: true`,
      );
      expect(bankMatrices(again.module, bank)).toEqual(bankMatrices(a.module, bank));
    }
    for (const m of [a, b, again]) m.module.dispose();
  });

  test('each bank draws its OWN stream (a name length is not an identity)', () => {
    expect('savannaGrass'.length).toBe('savannaRocks'.length);
    expect(bankKey('savannaGrass')).not.toBe(bankKey('savannaRocks'));
    expect(bankKey('savannaGrass')).not.toBe(bankKey('savannaAcacias'));
    expect(bankKey('savannaRocks')).not.toBe(bankKey('savannaAcacias'));
    for (const tile of [0, 1, 317, -12]) {
      const grass = rngAt(tile, bankKey('savannaGrass'), SEED);
      const rocks = rngAt(tile, bankKey('savannaRocks'), SEED);
      const trees = rngAt(tile, bankKey('savannaAcacias'), SEED);
      const draw = (r: () => number): string => `${r()},${r()},${r()}`;
      const g = draw(grass);
      expect(draw(rocks)).not.toBe(g);
      expect(draw(trees)).not.toBe(g);
    }
    const once = rngAt(317, bankKey('savannaGrass'), SEED);
    const twice = rngAt(317, bankKey('savannaGrass'), SEED);
    expect(twice()).toBe(once());
  });

  test('no scattered prop ever stands inside the runner’s silhouette', async () => {
    const { module } = await live();
    const rocks = module.scene.getObjectByName('savannaRocks') as THREE.InstancedMesh | undefined;
    expect(rocks).toBeDefined();
    if (!rocks) return;
    rocks.geometry.computeBoundingBox();
    const propTop = rocks.geometry.boundingBox?.max.y ?? 0;
    expect(propTop).toBeGreaterThan(0.1);
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    let seen = 0;
    for (const x of [0, 61.5, 240, 431.1, 1200]) {
      module.update(toFrameUpdate({ name: `props${x}`, pose: [x, -0.1, 0, 0, 0, 0, 0, 0, 0], step: 0 }));
      for (let i = 0; i < rocks.count; i++) {
        rocks.getMatrixAt(i, m);
        m.decompose(p, q, s);
        if (s.lengthSq() === 0) continue;
        seen++;
        const top = propTop * s.y;
        if (Math.abs(p.z) < 2.5) expect(`near prop top ${top < 0.45}`).toBe('near prop top true');
        if (p.z > 0) expect(`camera-side prop top ${top < 0.3}`).toBe('camera-side prop top true');
      }
    }
    expect(seen).toBeGreaterThan(50);
    module.dispose();
  });

  test('TILE is a real tile: the scatter window moves with the runner', async () => {
    const { module } = await live();
    const grass = () => module.scene.getObjectByName('savannaGrass') as THREE.InstancedMesh | undefined;
    const centroid = (): number => {
      const mesh = grass();
      if (!mesh) return Number.NaN;
      const m = new THREE.Matrix4();
      const p = new THREE.Vector3();
      let sum = 0;
      let n = 0;
      for (let i = 0; i < mesh.count; i++) {
        mesh.getMatrixAt(i, m);
        p.setFromMatrixPosition(m);
        if (p.lengthSq() === 0) continue;
        sum += p.x;
        n++;
      }
      return n > 0 ? sum / n : Number.NaN;
    };
    module.update(toFrameUpdate({ name: 'a', pose: [0, -0.1, 0, 0, 0, 0, 0, 0, 0], step: 0 }));
    const a = centroid();
    module.update(toFrameUpdate({ name: 'b', pose: [200 * TILE, -0.1, 0, 0, 0, 0, 0, 0, 0], step: 0 }));
    const b = centroid();
    expect(Math.abs(b - a - 200 * TILE)).toBeLessThan(0.5);
    module.dispose();
  });
});

describe('half-cheetah / the sim owns the animal’s dimensions', () => {
  function scaledGeometry(radius: number, headRadius: number): EnvGeometry {
    return { kind: 'cheetah', cheetah: { fk, radius, headRadius } };
  }

  function halfExtent(module: SceneModule, name: string, axis: 'x' | 'y' | 'z'): number {
    const mesh = module.scene.getObjectByName(name) as THREE.Mesh | undefined;
    if (!mesh) throw new Error(`mesh "${name}" missing`);
    const pos = mesh.geometry.getAttribute('position');
    let m = 0;
    for (let i = 0; i < pos.count; i++) {
      const v = axis === 'x' ? pos.getX(i) : axis === 'y' ? pos.getY(i) : pos.getZ(i);
      m = Math.max(m, Math.abs(v));
    }
    return m;
  }

  test('the coat CONTAINS the capsule the physics simulates, at whatever radius it has', async () => {
    const base = await live();
    expect(halfExtent(base.module, 'cheetah_footBack', 'x')).toBe(Math.fround(CHEETAH_RADIUS));
    base.module.dispose();

    const fat = await live('hero', SEED, scaledGeometry(CHEETAH_RADIUS * 2, CHEETAH_HEAD_RADIUS));
    expect(halfExtent(fat.module, 'cheetah_footBack', 'x')).toBe(Math.fround(CHEETAH_RADIUS * 2));
    fat.module.dispose();

    const tailTip = await live();
    expect(halfExtent(tailTip.module, 'cheetahTailMesh3', 'x')).toBeLessThan(CHEETAH_RADIUS);
    tailTip.module.dispose();
  });

  test('the skull IS the sim’s head sphere (the overlay draws that same circle)', async () => {
    const base = await live();
    expect(halfExtent(base.module, 'cheetahSkull', 'x')).toBeCloseTo(CHEETAH_HEAD_RADIUS * 1.08, 6);
    base.module.dispose();

    const big = await live('hero', SEED, scaledGeometry(CHEETAH_RADIUS, CHEETAH_HEAD_RADIUS * 2));
    expect(halfExtent(big.module, 'cheetahSkull', 'x')).toBeCloseTo(CHEETAH_HEAD_RADIUS * 2 * 1.08, 6);
    const baseFace = await live();
    const smallFace = halfExtent(baseFace.module, 'cheetahFace', 'x');
    expect(halfExtent(big.module, 'cheetahFace', 'x')).toBeCloseTo(smallFace * 2, 6);
    baseFace.module.dispose();
    big.module.dispose();
  });

  test('a changed model does not move a single probe (the FK still owns the joints)', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, {
      geometry: scaledGeometry(CHEETAH_RADIUS * 2, CHEETAH_HEAD_RADIUS * 2),
      seed: SEED,
      tier: 'hero',
    });
    for (const name of ['torso', 'toe-back', 'toe-front', 'head']) {
      expect(report.worst.get(name) ?? 1).toBe(0);
    }
    module.dispose();
  });
});

describe('half-cheetah / update contract', () => {
  test('update is idempotent on the same state (probes)', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update is idempotent on the WHOLE world matrix, treadmill included', async () => {
    const { module, ctx } = await live();
    for (const f of fixtures) {
      ctx.setTime(3.25);
      module.update(toFrameUpdate(f, { scrubbed: false, events: [] }));
      const a = worldSnapshot(module);
      module.update(toFrameUpdate(f, { scrubbed: false, events: [] }));
      const b = worldSnapshot(module);
      expect(b).toEqual(a);
    }
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('stride-extend')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 50; i++) {
      module.update(toFrameUpdate(fixtures[i % fixtures.length] as SceneFixture, { scrubbed: false }));
    }
    let after = 0;
    module.scene.traverse(() => after++);
    expect(after).toBe(count);
    module.dispose();
  });

  test('celebrations NEVER fire: HalfCheetah has no +1 to celebrate', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    ctx.setTime(1);
    for (const f of fixtures) module.update(toFrameUpdate(f, { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(fixtureNamed('stride-extend'), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(3);
    module.update(
      toFrameUpdate(fixtureNamed('reset-drop'), { scrubbed: false, events: [{ type: 'teleport' }] }),
    );
    expect(pool.stats.bursts).toBe(0);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(fixtureNamed('reset-drop'), { scrubbed: false, events: [{ type: 'teleport' }] }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('stride-extend'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });

  test('teleport is a hard snap: pose AND camera cut in the same frame', async () => {
    const { module, ctx } = await live();
    const torso = (): THREE.Vector3 => {
      const p = module.probes()[0];
      if (!p) throw new Error('no torso probe');
      return worldOf(p.object);
    };

    ctx.setTime(1);
    const stride = fixtureNamed('stride-extend');
    module.update(toFrameUpdate(stride, { scrubbed: false }));
    expect(torso().x).toBe(Math.fround(stride.pose[0] ?? 0));

    ctx.setTime(1.05);
    const respawn = fixtureNamed('reset-drop');
    expect(Math.abs((stride.pose[0] ?? 0) - (respawn.pose[0] ?? 0))).toBeGreaterThan(CHEETAH_TELEPORT_DX);
    module.update(
      toFrameUpdate(respawn, {
        scrubbed: false,
        prev: toFrameState(stride),
        events: [{ type: 'teleport' }],
      }),
    );
    expect(torso().x).toBe(Math.fround(respawn.pose[0] ?? 0));
    expect(module.camera.position.x).toBeCloseTo((respawn.pose[0] ?? 0) + (CAM_OFFSET[0] ?? 0), 5);
    expect(module.camera.position.y).toBeCloseTo(TRACK_Y + (CAM_OFFSET[1] ?? 0), 5);
    module.dispose();
  });

  test('the superluminal guard rejects a host-interpolated mid-teleport pose', async () => {
    const { module, ctx } = await live();
    const torsoX = (): number => {
      const p = module.probes()[0];
      if (!p) throw new Error('no torso probe');
      return worldOf(p.object).x;
    };
    ctx.setTime(1);
    const stride = fixtureNamed('stride-extend');
    module.update(toFrameUpdate(stride, { scrubbed: false }));
    const at = Math.fround(stride.pose[0] ?? 0);
    expect(torsoX()).toBe(at);

    const slide: SceneFixture = {
      name: 'mid-slide',
      pose: [95, -0.1, 0, 0.1, 0.05, 0, -0.05, 0.1, 0],
      step: stride.step ?? 0,
    };
    module.update(toFrameUpdate(slide, { scrubbed: false }));
    expect(torsoX()).toBe(at);

    const next: SceneFixture = {
      name: 'next-step',
      pose: [(stride.pose[0] ?? 0) + 0.048, -0.1, 0, 0.1, 0.05, 0, -0.05, 0.1, 0],
      step: (stride.step ?? 0) + 1,
    };
    module.update(toFrameUpdate(next, { scrubbed: false }));
    expect(torsoX()).toBe(Math.fround(next.pose[0] ?? 0));
    module.dispose();
  });

  test('the flash envelope never ends on a pop', () => {
    expect(envelope(-1, 0.34, 2.6)).toBe(0);
    expect(envelope(0, 0.34, 2.6)).toBeCloseTo(1, 6);
    expect(envelope(0.34, 0.34, 2.6)).toBe(0);
    expect(envelope(0.339, 0.34, 2.6)).toBeLessThan(0.002);
  });
});

describe('half-cheetah / lifecycle', () => {
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
    expect((report.census[0]?.textures.size ?? 0) > 0).toBe(true);
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
    module.update(toFrameUpdate(fixtureNamed('stride-extend')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'savannaGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'savannaGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });
});

const BONE_MESH_NAMES: readonly string[] = [
  'cheetah_torso',
  'cheetah_thighBack',
  'cheetah_shinBack',
  'cheetah_footBack',
  'cheetah_thighFront',
  'cheetah_shinFront',
  'cheetah_footFront',
];

function boneName(index: number): string {
  return BONE_MESH_NAMES[index] ?? `cheetah_${index}`;
}

describe('half-cheetah / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'half-cheetah (thumb, overlay on)');
    module.dispose();
  });
});
