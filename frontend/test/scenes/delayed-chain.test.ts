/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { cellToWorld } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L, delayedChainDecoder } from '@sim/envs/delayedchain';
import type { EnvGeometry } from '@sim/types';

import createScene, {
  LANTERN_SPAN,
  PULSE_STRENGTH,
  WAVE_SPAN,
  WAVE_TRAVEL,
  envelope,
  flicker,
  lanternGlow,
} from '../../src/scenes/delayed-chain/index';
import { fixtures } from '../../src/scenes/delayed-chain/fixtures';
import { manifest } from '../../src/scenes/delayed-chain/manifest';
import { LAMP_Y, POST_HEIGHT } from '../../src/scenes/delayed-chain/actors';
import {
  AURORA_FIRST_CELL,
  LANTERN_Z,
  ROAD_HALF,
  ROAD_HALF_Z,
  goalCellOf,
  makeMoorHeight,
  moonPools,
  moonlightAt,
  moorSkin,
  resolveGrid,
} from '../../src/scenes/delayed-chain/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 120;

function geometryOf(): EnvGeometry {
  return delayedChainDecoder.geometry(delayedChainDecoder.initStatic(SEED), SEED);
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

function poseFixture(name: string, pos: number, lastA: number, step = 0): SceneFixture {
  return {
    name,
    pose: [pos, DELAYEDCHAIN_END, lastA, 0, 0, 0, 0, 0, 0],
    ctrl: [lastA, 0, 0, 0, 0, 0],
    derived: { pos, lastA, len: DELAYEDCHAIN_L },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function meshNamed(module: SceneModule, name: string): THREE.Object3D {
  const o = module.scene.getObjectByName(name);
  if (!o) throw new Error(`object "${name}" missing from the scene graph`);
  return o;
}

function flameColors(module: SceneModule): Float32Array {
  const mesh = meshNamed(module, 'lanternFlames') as THREE.InstancedMesh;
  const attr = mesh.instanceColor;
  if (!attr) throw new Error('lanternFlames carries no instanceColor');
  return attr.array as Float32Array;
}

function endFlameGain(module: SceneModule): number {
  const mesh = meshNamed(module, 'endLanternFlame') as THREE.Mesh;
  const material = mesh.material as THREE.MeshBasicMaterial;
  return Math.max(material.color.r, material.color.g, material.color.b);
}


describe('delayed-chain / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('DelayedChain-cont-v0');
    expect(manifest.slug).toBe('delayed-chain');
    expect(manifest.accent).toBe('delayed-chain');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
    expect(TOKENS.accents[manifest.accent]).toBeDefined();
  });

  test('the thumbFrame is a legal DelayedChain state', () => {
    const p = manifest.thumbFrame.pose;
    expect(p[0]).toBeGreaterThanOrEqual(0);
    expect(p[0]).toBeLessThan(DELAYEDCHAIN_END);
    expect(p[1]).toBe(DELAYEDCHAIN_END);
    expect([0, 1]).toContain(p[2] ?? -1);
  });

  test('ships at least three canonical poses, all legal corridor states', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const pos = f.pose[0] ?? -1;
      expect(pos).toBeGreaterThanOrEqual(0);
      expect(pos).toBeLessThan(DELAYEDCHAIN_L);
      expect(f.pose[1]).toBe(DELAYEDCHAIN_END);
      expect([0, 1]).toContain(f.pose[2] ?? -1);
    }
  });

  test('the fixtures cover the last observed cell AND the wrap frame', () => {
    expect(fixtures.some((f) => f.pose[0] === DELAYEDCHAIN_END - 1)).toBe(true);
    const wrap = fixtureNamed('the-wrap');
    expect(wrap.pose[0]).toBe(0);
    expect(wrap.pose[2]).toBe(1);
  });
});

describe('delayed-chain / the corridor is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own 32×1 road", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(DELAYEDCHAIN_L);
    expect(grid.h).toBe(1);
    expect(goalCellOf(grid)).toBe(DELAYEDCHAIN_END);
    let walls = 0;
    for (let i = 0; i < grid.w * grid.h; i++) walls += grid.walls[i] ?? 0;
    expect(walls).toBe(0);
  });

  test('the road really is 32 cells of one world unit, centred on the origin', () => {
    const grid = resolveGrid(geometryOf());
    expect(cellToWorld(0, 0, grid)[0]).toBeCloseTo(-ROAD_HALF + 0.5, 12);
    expect(cellToWorld(DELAYEDCHAIN_END, 0, grid)[0]).toBeCloseTo(ROAD_HALF - 0.5, 12);
    expect(cellToWorld(5, 0, grid)[2]).toBe(0);
  });

  test('the moor is identically flat under the whole path, and rolls off it', () => {
    const h = makeMoorHeight(SEED);
    for (let i = 0; i <= 64; i++) {
      const x = -ROAD_HALF - 2 + ((2 * ROAD_HALF + 4) * i) / 64;
      for (const z of [-ROAD_HALF_Z, -0.2, 0, 0.2, ROAD_HALF_Z]) {
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const z of [-14, -9, 9, 14]) wander = Math.max(wander, Math.abs(h(3, z)));
    expect(wander).toBeGreaterThan(0.05);
  });

  test('the moonlight pools are deterministic, bounded and sparse', () => {
    const a = moonPools(SEED);
    const b = moonPools(SEED);
    const c = moonPools(SEED + 1);
    expect(a.map((p) => p.cell)).toEqual(b.map((p) => p.cell));
    expect(a.map((p) => p.cell)).not.toEqual(c.map((p) => p.cell));
    let lit = 0;
    for (let cell = 0; cell < DELAYEDCHAIN_L; cell++) {
      const m = moonlightAt(cell, a);
      expect(m).toBeGreaterThanOrEqual(0);
      expect(m).toBeLessThanOrEqual(1);
      if (m > 0.5) lit++;
    }
    expect(lit).toBeLessThan(DELAYEDCHAIN_L / 2);
  });

  test('every colour of the skin is a distinct blend, and the flame is the warm one', () => {
    const skin = moorSkin(TOKENS.palettes.night, TOKENS);
    expect(skin.flagstoneMoonlit.getHex()).not.toBe(skin.flagstone.getHex());
    expect(skin.flame.r).toBeGreaterThan(skin.flame.b);
    expect(skin.moor.b).toBeGreaterThan(skin.moor.r);
    expect(skin.iron.r + skin.iron.g + skin.iron.b).toBeLessThan(
      skin.flagstone.r + skin.flagstone.g + skin.flagstone.b,
    );
  });
});

describe('delayed-chain / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('end-lantern') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('start-milestone') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('end-lantern') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY one of the 32 cells, exactly', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let pos = 0; pos < DELAYEDCHAIN_L; pos++) {
      module.update(toFrameUpdate(poseFixture(`cell${pos}`, pos, 1, pos)));
      const w = worldOf(module, 'wisp');
      cellToWorld(pos, 0, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 12);
      expect(w.y).toBeCloseTo(0, 12);
      expect(w.z).toBeCloseTo(0, 12);
    }
    module.dispose();
  });

  test('all 32 lantern posts stand on their own cell centre, on the far verge', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const goal = goalCellOf(grid);
    const posts = meshNamed(module, 'lanternPosts') as THREE.InstancedMesh;
    expect(posts.count).toBe(DELAYEDCHAIN_L - 1);

    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const seen = new Set<number>();
    const truth: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < posts.count; i++) {
      posts.getMatrixAt(i, matrix);
      matrix.decompose(position, quaternion, scale);
      let cell = -1;
      for (let c = 0; c < DELAYEDCHAIN_L; c++) {
        if (Math.abs(cellToWorld(c, 0, grid, truth)[0] - position.x) < 1e-9) {
          cell = c;
          break;
        }
      }
      expect(cell).toBeGreaterThanOrEqual(0);
      expect(seen.has(cell)).toBe(false);
      seen.add(cell);
      expect(position.y).toBe(0);
      expect(position.z).toBeCloseTo(LANTERN_Z, 6);
      expect(position.z).toBeLessThan(-ROAD_HALF_Z);
    }
    const end = worldOf(module, 'end-lantern');
    cellToWorld(goal, 0, grid, truth);
    expect(end.x).toBeCloseTo(truth[0], 12);
    seen.add(goal);
    expect(seen.size).toBe(DELAYEDCHAIN_L);
    module.dispose();
  });

  test('the aurora never reaches back over the walk', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const curtain = meshNamed(module, 'auroraCurtain') as THREE.Mesh;
    curtain.updateWorldMatrix(true, false);
    curtain.geometry.computeBoundingBox();
    const box = (curtain.geometry.boundingBox as THREE.Box3).clone().applyMatrix4(curtain.matrixWorld);
    const firstAllowed = cellToWorld(AURORA_FIRST_CELL, 0, grid)[0];
    expect(box.min.x).toBeGreaterThanOrEqual(firstAllowed);
    expect(box.min.x).toBeGreaterThan(cellToWorld(goalCellOf(grid), 0, grid)[0]);
    expect(box.max.y).toBeGreaterThan(4);
    module.dispose();
  });

  test('the aurora is a CANOPY over the terminus, not a weather front across the moor', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const curtain = meshNamed(module, 'auroraCurtain') as THREE.Mesh;
    curtain.updateWorldMatrix(true, false);
    curtain.geometry.computeBoundingBox();
    const box = (curtain.geometry.boundingBox as THREE.Box3).clone().applyMatrix4(curtain.matrixWorld);

    expect(box.max.z - box.min.z).toBeLessThanOrEqual(DELAYEDCHAIN_L);

    ctx.setTime(12);
    module.update(toFrameUpdate(poseFixture('mid', 16, 1, 250)));
    const cam = module.camera;
    cam.updateMatrixWorld(true);
    const truth = cellToWorld(goalCellOf(grid), 0, grid);
    const p = new THREE.Vector3();
    const goalNdc = p.set(truth[0], LAMP_Y, truth[2]).project(cam).x;
    for (const z of [box.min.z, 0, box.max.z]) {
      p.set(box.min.x, 4, z).project(cam);
      expect(p.x).toBeGreaterThan(goalNdc - 0.6);
    }
    p.set(box.min.x, 4, 0).project(cam);
    expect(Math.abs(p.x - goalNdc)).toBeLessThan(0.35);
    const goalY = new THREE.Vector3(truth[0], LAMP_Y, truth[2]).project(cam).y;
    p.set(box.min.x, 4, 0).project(cam);
    expect(p.y).toBeGreaterThan(goalY);
    module.dispose();
  });
});

describe('delayed-chain / the walk is a hop, the wrap is a cut', () => {
  test('a hop stays inside the segment between the two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 9, 1, 0)));
    cellToWorld(9, 0, grid, from);
    cellToWorld(10, 0, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 10, 1, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 10, 1, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp');
      expect(w.x).toBeGreaterThanOrEqual(from[0] - 1e-6);
      expect(w.x).toBeLessThanOrEqual(to[0] + 1e-6);
      expect(w.z).toBeCloseTo(0, 6);
      expect(w.y).toBeGreaterThanOrEqual(-1e-6);
    }
    const landed = worldOf(module, 'wisp');
    expect(landed.x).toBeCloseTo(to[0], 12);
    expect(landed.y).toBeCloseTo(0, 12);
    module.dispose();
  });

  test('the wrap teleport is a hard cut: no glide across the moor', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(20);
    module.update(toFrameUpdate(poseFixture('end', 30, 1, 475), { scrubbed: false }));
    ctx.setTime(20.125);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 476), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp');
    expect(w.x).toBeCloseTo(cellToWorld(0, 0, grid)[0], 12);
    expect(w.y).toBeCloseTo(0, 12);
    module.dispose();
  });

  test('a scrub snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('a', 2, 1, 0), { scrubbed: false }));
    module.update(toFrameUpdate(poseFixture('far', 28, 1, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp');
    expect(w.x).toBeCloseTo(cellToWorld(28, 0, grid)[0], 12);
    expect(w.z).toBeCloseTo(0, 12);
    expect(w.y).toBeCloseTo(0, 12);
    module.dispose();
  });

  test('the wisp leans into the way it stepped, and only that far', async () => {
    const { module, ctx } = await live();
    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('east', 5, 1, 10)));
    const east = (meshNamed(module, 'oakWispAvatar') as THREE.Object3D).rotation.z;
    module.update(toFrameUpdate(poseFixture('west', 5, 0, 11)));
    const west = (meshNamed(module, 'oakWispAvatar') as THREE.Object3D).rotation.z;
    expect(east).toBeLessThan(0);
    expect(west).toBeGreaterThan(0);
    expect(Math.abs(east)).toBeLessThan(0.35);
    expect(Math.abs(east)).toBeCloseTo(Math.abs(west), 12);
    module.dispose();
  });
});

describe('delayed-chain / the avatar is the brightest thing on the road', () => {
  test('only the kit HALO MESH is switched off — nothing about the wisp is dimmed', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('mid-moor')));

    const core = meshNamed(module, 'oakWispCore') as THREE.Mesh;
    const coreMaterial = core.material as THREE.ShaderMaterial;
    expect((coreMaterial.uniforms.uIntensity as { value: number }).value).toBeCloseTo(1.9, 10);

    const ribbon = meshNamed(module, 'oakWispRibbon') as THREE.Line;
    expect((ribbon.material as THREE.LineBasicMaterial).opacity).toBeCloseTo(0.4, 10);
    const skirt = meshNamed(module, 'oakWispSkirt') as THREE.Mesh;
    expect((skirt.material as THREE.Material).opacity).toBeCloseTo(0.85, 10);

    expect(meshNamed(module, 'oakWispHalo').visible).toBe(false);
    expect(meshNamed(module, 'wispGlow').visible).toBe(true);
    module.dispose();
  });

  test('the celebration flare still brightens the core, and settles back to kit brightness', async () => {
    const { module, ctx } = await live();
    ctx.setTime(40);
    module.update(toFrameUpdate(poseFixture('run', 30, 1, 900), { scrubbed: false }));
    const core = meshNamed(module, 'oakWispCore') as THREE.Mesh;
    const rim = (): number => ((core.material as THREE.ShaderMaterial).uniforms.uIntensity as { value: number }).value;
    const idle = rim();
    ctx.setTime(40.125);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 901), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(rim()).toBeGreaterThan(idle);
    for (let k = 0; k < 60; k++) {
      ctx.setTime(40.125 + (k + 1) / 60);
      module.update(toFrameUpdate(poseFixture('after', 1, 1, 902 + k), { scrubbed: false }));
    }
    expect(rim()).toBeCloseTo(idle, 6);
    module.dispose();
  });
});

describe('delayed-chain / the ignition wave', () => {
  test('lanternGlow is dark before the goal, and everywhere outside its window', () => {
    for (const i of [0, 7, 31]) {
      expect(lanternGlow(-Infinity, i, DELAYEDCHAIN_L)).toBe(0);
      expect(lanternGlow(Number.NaN, i, DELAYEDCHAIN_L)).toBe(0);
      expect(lanternGlow(-0.01, i, DELAYEDCHAIN_L)).toBe(0);
      expect(lanternGlow(WAVE_SPAN + 0.001, i, DELAYEDCHAIN_L)).toBe(0);
      expect(lanternGlow(999, i, DELAYEDCHAIN_L)).toBe(0);
    }
  });

  test('the wave starts at the END and arrives at the START — never the other way', () => {
    const n = DELAYEDCHAIN_L;
    const early = 0.2;
    expect(lanternGlow(early, n - 1, n)).toBeGreaterThan(0.5);
    expect(lanternGlow(early, 0, n)).toBe(0);
    expect(lanternGlow(early, 16, n)).toBe(0);
    const mid = WAVE_TRAVEL * 0.5 + 0.2;
    expect(lanternGlow(mid, Math.round(n / 2), n)).toBeGreaterThan(0);
    expect(lanternGlow(mid, 0, n)).toBe(0);
    const late = WAVE_TRAVEL + 0.2;
    expect(lanternGlow(late, 0, n)).toBeGreaterThan(0.5);
    let previous = -1;
    for (let i = n - 1; i >= 0; i--) {
      const ignites = ((n - 1 - i) / (n - 1)) * WAVE_TRAVEL;
      expect(ignites).toBeGreaterThan(previous);
      previous = ignites;
      expect(lanternGlow(ignites - 1e-6, i, n)).toBe(0);
      expect(lanternGlow(ignites + 0.2, i, n)).toBeGreaterThan(0);
    }
  });

  test('every lantern kindles, holds and gutters out to exactly zero', () => {
    const n = DELAYEDCHAIN_L;
    for (const i of [0, 5, 18, 31]) {
      const start = ((n - 1 - i) / (n - 1)) * WAVE_TRAVEL;
      let peak = 0;
      for (let k = 0; k <= 200; k++) {
        const age = start + (LANTERN_SPAN * k) / 200;
        const g = lanternGlow(age, i, n);
        expect(g).toBeGreaterThanOrEqual(0);
        expect(g).toBeLessThanOrEqual(1);
        peak = Math.max(peak, g);
      }
      expect(peak).toBeCloseTo(1, 6);
      expect(lanternGlow(start + LANTERN_SPAN, i, n)).toBeCloseTo(0, 6);
    }
  });

  test('flicker and envelope are pure and bounded', () => {
    expect(flicker(3.25, 7)).toBe(flicker(3.25, 7));
    for (let k = 0; k < 40; k++) {
      const f = flicker(k * 0.37, k);
      expect(f).toBeGreaterThan(0.7);
      expect(f).toBeLessThanOrEqual(1.0001);
    }
    expect(envelope(-Infinity, 1, 2)).toBe(0);
    expect(envelope(Number.NaN, 1, 2)).toBe(0);
    expect(envelope(0, 1, 2)).toBe(1);
    expect(envelope(1.0001, 1, 2)).toBe(0);
  });

  test('the road really ignites end-first, and a scrub extinguishes it instantly', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const goal = goalCellOf(grid);

    ctx.setTime(30);
    module.update(toFrameUpdate(poseFixture('walking', 30, 1, 475), { scrubbed: false }));
    let colors = flameColors(module);
    for (let i = 0; i < colors.length; i++) expect(colors[i]).toBe(0);
    expect(endFlameGain(module)).toBe(0);

    ctx.setTime(30.125);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 476), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    ctx.setTime(30.125 + 0.25);
    module.update(toFrameUpdate(poseFixture('after', 0, 1, 477), { scrubbed: false }));
    expect(endFlameGain(module)).toBeGreaterThan(0);
    colors = flameColors(module);
    expect(colors[0]).toBe(0);

    ctx.setTime(30.125 + WAVE_TRAVEL + 0.25);
    module.update(toFrameUpdate(poseFixture('after', 1, 1, 478), { scrubbed: false }));
    colors = flameColors(module);
    expect(colors[0] ?? 0).toBeGreaterThan(0.5);

    ctx.setTime(9);
    module.update(toFrameUpdate(poseFixture('scrubbed', 12, 0, 90), { scrubbed: true }));
    colors = flameColors(module);
    for (let i = 0; i < colors.length; i++) expect(colors[i]).toBe(0);
    expect(endFlameGain(module)).toBe(0);
    expect(goal).toBe(DELAYEDCHAIN_END);
    module.dispose();
  });

  test('the wave is over for good after its own span', async () => {
    const { module, ctx } = await live();
    ctx.setTime(50);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 476), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    ctx.setTime(50 + WAVE_SPAN + 0.5);
    module.update(toFrameUpdate(poseFixture('later', 3, 1, 500), { scrubbed: false }));
    const colors = flameColors(module);
    for (let i = 0; i < colors.length; i++) expect(colors[i]).toBe(0);
    expect(endFlameGain(module)).toBe(0);
    module.dispose();
  });
});

describe('delayed-chain / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('road-start')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 60; i++) {
      module.update(toFrameUpdate(fixtures[i % fixtures.length] as SceneFixture, { scrubbed: false }));
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
    module.update(toFrameUpdate(poseFixture('doorstep', 30, 1, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 101), {
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

  test('the pay-off blooms at the GOAL cell, not where the wrapped agent is', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const goalX = cellToWorld(goalCellOf(grid), 0, grid)[0];

    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('run', 30, 1, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 201), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );

    const ripple = meshNamed(module, 'oakCelebrationRipple0');
    expect(ripple.visible).toBe(true);
    expect(ripple.position.x).toBeCloseTo(goalX, 10);
    expect(ripple.position.z).toBeCloseTo(0, 10);
    expect(Math.abs(ripple.position.x - worldOf(module, 'wisp').x)).toBeGreaterThan(DELAYEDCHAIN_L / 2);

    const light = module.scene.getObjectByName('oakPoint3') as THREE.PointLight | undefined;
    expect(light).toBeDefined();
    if (!light) return;
    expect(light.position.x).toBeCloseTo(goalX, 10);
    expect(light.position.y).toBeCloseTo(LAMP_Y, 10);
    expect(light.intensity).toBeGreaterThan(0);
    module.dispose();
  });

  test('the pay-off pulse is the DECLARED weight, not the pool default', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;
    expect(PULSE_STRENGTH).toBeLessThan(1);
    ctx.setTime(7);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 800), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.pulse.bloom).toBeCloseTo(PULSE_STRENGTH, 12);
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 300), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('mid-moor'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });
});

describe('delayed-chain / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('mid-moor')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'roadGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'roadGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera is a road picture: shallow, low, and looking down the moor', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('mid-moor')));
    const cam = module.camera;
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(cam.position.y).toBeGreaterThan(1.4);
    expect(cam.position.y).toBeLessThan(3.6);
    expect(dir.y).toBeLessThan(-0.02);
    expect(dir.y).toBeGreaterThan(-0.3);
    expect(dir.x).toBeGreaterThan(0.6);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the camera walks with the wisp and hard-cuts on the wrap', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(5);
    module.update(toFrameUpdate(poseFixture('end', 30, 1, 475)));
    const atEnd = module.camera.position.x;
    expect(atEnd).toBeGreaterThan(cellToWorld(20, 0, grid)[0]);

    ctx.setTime(5.125);
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 476), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const atStart = module.camera.position.x;
    expect(atStart).toBeLessThan(cellToWorld(4, 0, grid)[0]);
    expect(atEnd - atStart).toBeGreaterThan(DELAYEDCHAIN_L / 2);
    module.dispose();
  });

  test('medium keeps the two lights that matter: the agent and the far end', async () => {
    const { module } = await live('medium');
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(fixtureNamed('mid-moor')));

    const fill = module.scene.getObjectByName('oakPoint1') as THREE.PointLight | undefined;
    expect(fill).toBeDefined();
    if (!fill) return;
    expect(fill.intensity).toBeGreaterThan(0);
    expect(fill.position.x).toBeGreaterThan(cellToWorld(DELAYEDCHAIN_END - 4, 0, grid)[0]);
    expect(fill.position.y).toBeGreaterThan(2);

    expect(module.scene.getObjectByName('oakPoint2')).toBeUndefined();
    expect(module.scene.getObjectByName('oakPoint3')).toBeUndefined();
    module.update(
      toFrameUpdate(poseFixture('wrap', 0, 1, 476), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(Number.isFinite(worldOf(module, 'wisp').x)).toBe(true);
    module.dispose();
  });

  test('the thumb frame holds the WHOLE road AND the aurora — at EVERY wall aspect', async () => {
    const { module } = await live('thumb');
    module.update(toFrameUpdate(poseFixture('mid', 16, 1, 0)));
    const cam = module.camera;

    const grid = resolveGrid(geometryOf());
    const curtain = meshNamed(module, 'auroraCurtain') as THREE.Mesh;
    curtain.updateWorldMatrix(true, false);
    curtain.geometry.computeBoundingBox();
    const box = (curtain.geometry.boundingBox as THREE.Box3).clone().applyMatrix4(curtain.matrixWorld);
    const p = new THREE.Vector3();

    for (const [width, height] of [
      [1600, 900],
      [800, 600],
      [600, 600],
    ] as const) {
      module.resize({ width, height, dpr: 1 });
      cam.updateMatrixWorld(true);
      for (const cell of [0, DELAYEDCHAIN_END]) {
        const truth = cellToWorld(cell, 0, grid);
        for (const y of [truth[1], POST_HEIGHT]) {
          p.set(truth[0], y, truth[2]).project(cam);
          expect(Math.abs(p.x)).toBeLessThanOrEqual(0.9);
          expect(Math.abs(p.y)).toBeLessThanOrEqual(0.9);
          expect(p.z).toBeLessThan(1);
        }
      }
      p.set(box.min.x, 6, 0).project(cam);
      expect(Math.abs(p.x)).toBeLessThanOrEqual(1);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(1);
    }

    expect(cam.position.length()).toBeGreaterThan(25);
    module.dispose();
  });

  test('the thumb tier is a fixed frame (no pointer listeners, no follow)', async () => {
    const { module, ctx } = await live('thumb');
    module.update(toFrameUpdate(poseFixture('a', 2, 1, 0)));
    const a = module.camera.position.clone();
    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('b', 29, 1, 400), { scrubbed: false }));
    const b = module.camera.position.clone();
    expect(a.distanceTo(b)).toBeCloseTo(0, 10);
    module.dispose();
  });

  test("the start pose really is the env's deterministic init", () => {
    const start = fixtureNamed('road-start');
    const init = delayedChainDecoder.initFrame(SEED, delayedChainDecoder.initStatic(SEED));
    expect(start.pose[0]).toBe(init.pose[0]);
    expect(start.pose[1]).toBe(init.pose[1]);
    expect(start.pose[2]).toBe(init.pose[2]);
  });
});

describe('delayed-chain / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'delayed-chain (thumb, overlay on)');
    module.dispose();
  });
});
