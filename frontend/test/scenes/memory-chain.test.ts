/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld } from '@kit/grid';
import { oakUniforms, tierEffectScale, type EmissiveMaterial } from '@kit/materials';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  MEMORYCHAIN_END,
  MEMORYCHAIN_L,
  memoryChainDecoder,
  type MemoryChainState,
} from '@sim/envs/memorychain';
import type { EnvGeometry } from '@sim/types';

import createScene, {
  cueVisibleAt,
  envelope,
  floodProgress,
  floodRunning,
  posOf,
} from '../../src/scenes/memory-chain/index';
import {
  MOUTH_RIM,
  mouthCarriers,
  mouthStoneTint,
  mouthTierGain,
  type MouthCarriers,
} from '../../src/scenes/memory-chain/actors';
import { fixtures } from '../../src/scenes/memory-chain/fixtures';
import { manifest, memoryChainDerived, memoryChainPose } from '../../src/scenes/memory-chain/manifest';
import {
  CAVERN_X0,
  CAVERN_X1,
  CHOICE_CELL,
  CUE_CELL,
  caveSkin,
  cavernSurface,
  cavernWidthAt,
  readChoice,
  readCue,
  readPos,
  resolveGrid,
} from '../../src/scenes/memory-chain/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  measureBudget,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 88;

function geometryOf(): EnvGeometry {
  return memoryChainDecoder.geometry(memoryChainDecoder.initStatic(SEED), SEED);
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

function poseFixture(name: string, pos: number, cue: number, action = 0, step = 0): SceneFixture {
  return {
    name,
    pose: memoryChainPose(pos, cue),
    ctrl: [action, 0, 0, 0, 0, 0],
    derived: memoryChainDerived(pos, cue, action) satisfies MemoryChainState,
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function meshNamed(module: SceneModule, name: string): THREE.Mesh {
  const o = module.scene.getObjectByName(name);
  if (!o) throw new Error(`object "${name}" missing`);
  return o as THREE.Mesh;
}

function emissiveOf(module: SceneModule, name: string): EmissiveMaterial {
  return meshNamed(module, name).material as EmissiveMaterial;
}

function drawnIntensity(module: SceneModule, name: string): number {
  const mesh = meshNamed(module, name);
  if (!mesh.visible) return 0;
  const c = (mesh.material as EmissiveMaterial).color;
  return Math.max(c.r, c.g, c.b);
}


const CUE_HEXES: readonly number[] = [TOKENS.cueCyan, TOKENS.cueMagenta];

function hueOf(c: THREE.Color): [number, number, number] {
  const m = Math.max(c.r, c.g, c.b);
  if (m <= 1e-5) return [0, 0, 0];
  return [c.r / m, c.g / m, c.b / m];
}

export function isCueColored(c: THREE.Color): boolean {
  if (Math.max(c.r, c.g, c.b) <= 1e-5) return false;
  const h = hueOf(c);
  const probe = new THREE.Color();
  for (const hex of CUE_HEXES) {
    const t = hueOf(probe.set(hex));
    const d = Math.hypot(h[0] - t[0], h[1] - t[1], h[2] - t[2]);
    if (d < 0.22) return true;
  }
  return false;
}

interface ShaderLike {
  uniforms?: Record<string, { value?: unknown } | undefined>;
}

function cueLeaks(module: SceneModule): string[] {
  const out: string[] = [];
  module.scene.traverse((o) => {
    if (!o.visible) return;
    for (let p = o.parent; p; p = p.parent) if (!p.visible) return;

    const light = o as THREE.PointLight;
    if (light.isPointLight === true) {
      if (light.intensity > 1e-6 && isCueColored(light.color)) out.push(o.name || 'point-light');
      return;
    }

    const withMaterial = o as THREE.Mesh & { material?: THREE.Material };
    const material = withMaterial.material;
    if (!material) return;

    const basic = material as THREE.MeshBasicMaterial;
    if (basic.isMeshBasicMaterial === true) {
      if (isCueColored(basic.color)) out.push(o.name || 'emissive-mesh');
      return;
    }

    const std = material as THREE.MeshStandardMaterial;
    if (std.isMeshStandardMaterial === true) {
      const handle = oakUniforms(std);
      const rim = handle?.uniforms.uRimStrength.value ?? 0;
      const rimLit = rim > 1e-6 && isCueColored(handle?.uniforms.uRimColor.value ?? new THREE.Color(0, 0, 0));
      const emissiveLit = std.emissiveIntensity > 1e-6 && isCueColored(std.emissive);
      if (rimLit || emissiveLit || isCueColored(std.color)) out.push(o.name || 'lit-mesh');
      return;
    }

    const shader = material as unknown as ShaderLike;
    const uniforms = shader.uniforms;
    if (!uniforms) return;
    const color = uniforms.uColor?.value;
    if (!(color instanceof THREE.Color)) return;
    const density = uniforms.uDensity?.value;
    if (typeof density === 'number' && density <= 1e-6) return;
    const intensity = uniforms.uIntensity?.value;
    if (typeof intensity === 'number' && intensity <= 1e-6) return;
    if (isCueColored(color)) out.push(o.name || 'particle-field');
  });
  return out;
}

const MOUTH_GLOWS = ['exitMouthCyanThroat', 'exitMouthMagentaThroat'];
const MOUTH_OBJECTS = [...MOUTH_GLOWS, 'exitMouthCyanFrame', 'exitMouthMagentaFrame'];

function expectOnlyMouthsPaintCue(module: SceneModule, label: string): void {
  const leaks = cueLeaks(module).sort();
  expect(`${label} strangers:${leaks.filter((n) => !MOUTH_OBJECTS.includes(n)).join(',')}`).toBe(
    `${label} strangers:`,
  );
  for (const glow of MOUTH_GLOWS) {
    expect(`${label} ${glow} drawn:${leaks.includes(glow)}`).toBe(`${label} ${glow} drawn:true`);
  }
}

function hueDistance(c: THREE.Color, hex: number): number {
  const a = hueOf(c);
  const b = hueOf(new THREE.Color(hex));
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function carriersOf(module: SceneModule, name: string): MouthCarriers {
  return mouthCarriers(module.scene, name);
}



describe('memory-chain / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('MemoryChain-cont-v0');
    expect(manifest.slug).toBe('memory-chain');
    expect(manifest.accent).toBe('memory-chain');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal passage state, and it is a LIT one', () => {
    const p = manifest.thumbFrame.pose;
    expect(p[0]).toBeGreaterThanOrEqual(0);
    expect(p[0]).toBeLessThanOrEqual(MEMORYCHAIN_END);
    expect(p[1]).toBe(MEMORYCHAIN_END);
    expect(p[2] === 0 || p[2] === 1).toBe(true);
    expect(cueVisibleAt(p[0] ?? -1)).toBe(true);
  });

  test('ships at least three canonical poses, all inside the corridor', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const pos = f.pose[0] ?? -1;
      const cue = f.pose[2] ?? -1;
      expect(pos).toBeGreaterThanOrEqual(0);
      expect(pos).toBeLessThanOrEqual(MEMORYCHAIN_END);
      expect(cue === 0 || cue === 1).toBe(true);
    }
  });

  test('the fixtures cover both cues at cell 0, a hidden-cue walk, and the choice cell', () => {
    const lit = fixtures.filter((f) => (f.pose[0] ?? -1) === CUE_CELL);
    expect(new Set(lit.map((f) => f.pose[2])).size).toBe(2);
    expect(fixtures.some((f) => (f.pose[0] ?? 0) > 0 && (f.pose[0] ?? 0) < MEMORYCHAIN_END)).toBe(true);
    expect(fixtures.some((f) => (f.pose[0] ?? -1) === CHOICE_CELL)).toBe(true);
  });

  test('BOTH endings ship as canonical poses, and they are distinguishable', () => {
    const win = fixtureNamed('correct-choice-respawn');
    const loss = fixtureNamed('wrong-choice-respawn');
    for (const f of [win, loss]) {
      expect(f.pose[0]).toBe(CUE_CELL);
      expect(Math.trunc(f.ctrl?.[0] ?? -1) & 1).toBe(1);
    }
    expect(win.pose[2]).not.toBe(loss.pose[2]);
  });

  test('every fixture step is a REAL row of the recording it claims', () => {
    const url = new URL(
      '../../../.oak-benchmarks/frontend-demo-16env-20260803/MemoryChain-cont-v0/seed88/actions.bin',
      import.meta.url,
    );
    const path = decodeURIComponent(url.pathname);
    if (!existsSync(path)) {
      console.log(`\n  memory-chain: recording absent, fixture provenance not re-checked\n    ${path}\n`);
      return;
    }
    const buf = readFileSync(path);
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    const rows = Math.floor(f32.length / 16);
    const statics = memoryChainDecoder.initStatic(SEED);
    const init = memoryChainDecoder.initFrame(SEED, statics);
    for (const fixture of fixtures) {
      const step = fixture.step ?? 0;
      const frame =
        step < 0
          ?
            init
          : memoryChainDecoder.decodeFrame(step, f32.slice(step * 16, step * 16 + 16), statics, init);
      expect(step).toBeLessThan(rows);
      const read = { pose: frame.pose, ctrl: frame.ctrl, derived: frame.state };
      expect(`${fixture.name}@${step} pos:${readPos(read)}`).toBe(
        `${fixture.name}@${step} pos:${fixture.pose[0]}`,
      );
      expect(`${fixture.name}@${step} cue:${readCue(read)}`).toBe(
        `${fixture.name}@${step} cue:${fixture.pose[2]}`,
      );
      expect(`${fixture.name}@${step} action:${readChoice(read)}`).toBe(
        `${fixture.name}@${step} action:${Math.trunc(fixture.ctrl?.[0] ?? 0) & 1}`,
      );
      expect(`${fixture.name} derived`).toBe(`${fixture.name} derived`);
      expect(fixture.derived).toEqual(frame.state);
    }
  });
});

describe('memory-chain / the passage is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own 32 × 1 corridor", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(MEMORYCHAIN_L);
    expect(grid.h).toBe(1);
    let walls = 0;
    for (let i = 0; i < grid.walls.length; i++) walls += grid.walls[i] ?? 0;
    expect(walls).toBe(0);
    expect(grid.goal?.x).toBe(MEMORYCHAIN_END);
  });

  test('the cue cell and the choice cell are the env constants, not scene guesses', () => {
    expect(CUE_CELL).toBe(0);
    expect(CHOICE_CELL).toBe(MEMORYCHAIN_END);
    expect(CHOICE_CELL).toBe(MEMORYCHAIN_L - 1);
  });

  test('the cavern is a closed arch that clears the walkway and reaches both banks', () => {
    const surf = { z: 0, y: 0, nz: 0, ny: 0 };
    for (let i = 0; i <= 40; i++) {
      const x = CAVERN_X0 + ((CAVERN_X1 - CAVERN_X0) * i) / 40;
      const w = cavernWidthAt(x, SEED);
      expect(w).toBeGreaterThan(4.6);
      cavernSurface(x, 0, SEED, surf);
      expect(surf.z).toBeCloseTo(-w, 10);
      expect(surf.y).toBeCloseTo(0, 10);
      expect(surf.nz).toBeCloseTo(1, 6);
      cavernSurface(x, 1, SEED, surf);
      expect(surf.z).toBeCloseTo(w, 10);
      cavernSurface(x, 0.5, SEED, surf);
      expect(surf.z).toBeCloseTo(0, 6);
      expect(surf.y).toBeGreaterThan(6);
      expect(surf.ny).toBeCloseTo(-1, 6);
    }
  });

  test('the cave palette carries NO cue colour: the cave itself is colourless', () => {
    const skin = caveSkin(TOKENS.palettes.cave, TOKENS);
    for (const [name, color] of Object.entries(skin)) {
      if (!(color instanceof THREE.Color)) continue;
      expect(`${name}:${isCueColored(color)}`).toBe(`${name}:false`);
    }
  });

  test('the state readers are the env packing, and they clamp instead of inventing', () => {
    const f = poseFixture('probe', 7, 1, 1);
    const state = {
      pose: Float32Array.from(f.pose),
      ctrl: Float32Array.from(f.ctrl ?? []),
      derived: f.derived,
    };
    expect(readPos(state)).toBe(7);
    expect(readCue(state)).toBe(1);
    expect(readChoice(state)).toBe(1);
    const broken = { pose: Float32Array.from([Number.NaN, 31, 9]), ctrl: new Float32Array(6), derived: null };
    expect(readPos(broken)).toBe(0);
    expect(readCue(broken)).toBe(0);
    const past = { pose: Float32Array.from([99, 31, 1]), ctrl: new Float32Array(6), derived: null };
    expect(readPos(past)).toBe(MEMORYCHAIN_END);
  });
});

describe('memory-chain / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('cue-crystal') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('exit-cyan') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('exit-magenta') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('cue-crystal') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('exit-magenta') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY one of the 32 stones', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let pos = 0; pos < MEMORYCHAIN_L; pos++) {
      module.update(toFrameUpdate(poseFixture(`cell${pos}`, pos, pos & 1, 0, pos)));
      const w = worldOf(module, 'wisp-anchor');
      cellToWorld(pos, 0, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(0, 10);
      expect(worldOf(module, 'wisp').y).toBeCloseTo(TOKENS.wispHeight, 10);
    }
    module.dispose();
  });

  test('the crystal owns cell 0 and BOTH mouths own cell 31, for every frame', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const cue: [number, number, number] = [0, 0, 0];
    const choice: [number, number, number] = [0, 0, 0];
    cellToWorld(CUE_CELL, 0, grid, cue);
    cellToWorld(CHOICE_CELL, 0, grid, choice);
    for (const f of fixtures) {
      module.update(toFrameUpdate(f));
      const c = worldOf(module, 'cue-crystal');
      expect(c.x).toBeCloseTo(cue[0], 10);
      expect(c.y).toBeCloseTo(0, 10);
      expect(c.z).toBeCloseTo(0, 10);
      for (const name of ['exit-cyan', 'exit-magenta']) {
        const m = worldOf(module, name);
        expect(m.x).toBeCloseTo(choice[0], 10);
        expect(m.y).toBeCloseTo(0, 10);
        expect(m.z).toBeCloseTo(0, 10);
      }
    }
    expect(choice[0] - cue[0]).toBeCloseTo(MEMORYCHAIN_END * CELL, 10);
    module.dispose();
  });
});

describe('memory-chain / THE HONESTY RULE', () => {
  test('the crystal is lit ONLY on cell 0 — for both cues, at every cell, at every tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module, ctx } = await live(tier);
      ctx.setTime(5);
      for (const cue of [0, 1] as const) {
        for (let pos = 0; pos < MEMORYCHAIN_L; pos++) {
          module.update(toFrameUpdate(poseFixture(`c${cue}p${pos}`, pos, cue)));
          const glow = drawnIntensity(module, 'cueCrystalGlow');
          const halo = drawnIntensity(module, 'cueCrystalHalo');
          const light = module.scene.getObjectByName('oakPoint1') as THREE.PointLight | undefined;
          if (pos === 0) {
            expect(`${tier} p${pos} glow>0:${glow > 0}`).toBe(`${tier} p${pos} glow>0:true`);
            expect(halo).toBeGreaterThan(0);
            if (light) expect(light.intensity).toBeGreaterThan(0);
          } else {
            expect(`${tier} p${pos} glow:${glow}`).toBe(`${tier} p${pos} glow:0`);
            expect(`${tier} p${pos} halo:${halo}`).toBe(`${tier} p${pos} halo:0`);
            if (light) expect(`${tier} p${pos} light:${light.intensity}`).toBe(`${tier} p${pos} light:0`);
          }
        }
      }
      module.dispose();
    }
  });

  test('while pos > 0 the ONLY cue-coloured things in the frame are the two mouths, at every tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module, ctx } = await live(tier);
      ctx.setTime(9);
      for (const cue of [0, 1] as const) {
        for (let pos = 1; pos < MEMORYCHAIN_L; pos++) {
          module.update(toFrameUpdate(poseFixture(`leak${cue}-${pos}`, pos, cue)));
          expectOnlyMouthsPaintCue(module, `${tier} cue${cue} pos${pos}`);
        }
      }
      module.dispose();
    }
  });

  test('the two mouths are lit EXACTLY equally at rest, whatever the cue is, at every tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module, ctx } = await live(tier);
      ctx.setTime(13);
      for (const cue of [0, 1] as const) {
        for (const pos of [0, 1, 15, MEMORYCHAIN_END]) {
          module.update(toFrameUpdate(poseFixture(`sym${cue}-${pos}`, pos, cue, cue)));
          const c = carriersOf(module, 'exitMouthCyan');
          const m = carriersOf(module, 'exitMouthMagenta');
          const key = `${tier} cue${cue} pos${pos}`;
          expect(`${key} rim:${c.rim === m.rim}`).toBe(`${key} rim:true`);
          expect(`${key} glow:${c.glow === m.glow}`).toBe(`${key} glow:true`);
          expect(`${key} stoneEmissive:${c.stoneEmissive === m.stoneEmissive}`).toBe(
            `${key} stoneEmissive:true`,
          );
          expect(`${key} total:${c.total === m.total}`).toBe(`${key} total:true`);
          expect(c.glow).toBeGreaterThan(0);
        }
      }
      module.dispose();
    }
  });

  test('on cell 0 the crystal carries the cue token exactly, and nothing else does', async () => {
    const { module, ctx } = await live();
    ctx.setTime(2);
    for (const [cue, hex] of [
      [0, TOKENS.cueCyan],
      [1, TOKENS.cueMagenta],
    ] as const) {
      module.update(toFrameUpdate(poseFixture(`lit${cue}`, 0, cue)));
      expect(emissiveOf(module, 'cueCrystalGlow').baseColor.getHex()).toBe(hex);
      expect(emissiveOf(module, 'cueCrystalHalo').baseColor.getHex()).toBe(hex);
      const light = module.scene.getObjectByName('oakPoint1') as THREE.PointLight;
      expect(light.color.getHex()).toBe(hex);
      const allowed = [...MOUTH_OBJECTS, 'cueCrystalGlow', 'cueCrystalHalo', 'oakPoint1'];
      const leaks = cueLeaks(module).sort();
      expect(leaks.filter((n) => !allowed.includes(n))).toEqual([]);
      for (const name of ['cueCrystalGlow', 'cueCrystalHalo', 'oakPoint1']) {
        expect(`cue${cue} ${name}:${leaks.includes(name)}`).toBe(`cue${cue} ${name}:true`);
      }
    }
    module.dispose();
  });

  test('EVERY tier paints both mouths, in their own colours — the thumb blocker, locked', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      module.update(toFrameUpdate(poseFixture('mid', 17, 1, 1)));
      const pairs = [
        ['exitMouthCyan', TOKENS.cueCyan, TOKENS.cueMagenta],
        ['exitMouthMagenta', TOKENS.cueMagenta, TOKENS.cueCyan],
      ] as const;
      for (const [name, own, other] of pairs) {
        const c = carriersOf(module, name);
        expect(`${tier} ${name} total>0:${c.total > 0}`).toBe(`${tier} ${name} total>0:true`);
        expect(`${tier} ${name} glow>0:${c.glow > 0}`).toBe(`${tier} ${name} glow>0:true`);
        expect(`${tier} ${name} glow is own token:${hueDistance(c.glowColor, own) < 1e-6}`).toBe(
          `${tier} ${name} glow is own token:true`,
        );
        const rimAlive = c.rim > 1e-6;
        const stoneAlive = mouthStoneTint(tier) > 0;
        expect(`${tier} ${name} rim-or-stone:${rimAlive || stoneAlive}`).toBe(
          `${tier} ${name} rim-or-stone:true`,
        );
        if (rimAlive) {
          expect(`${tier} ${name} rim own:${hueDistance(c.rimColor, own) < 1e-6}`).toBe(
            `${tier} ${name} rim own:true`,
          );
        }
        if (stoneAlive) {
          expect(
            `${tier} ${name} stone own:${hueDistance(c.stone, own) < hueDistance(c.stone, other)}`,
          ).toBe(`${tier} ${name} stone own:true`);
          expect(
            `${tier} ${name} emissive own:${
              hueDistance(c.stoneEmissiveColor, own) < hueDistance(c.stoneEmissiveColor, other)
            }`,
          ).toBe(`${tier} ${name} emissive own:true`);
        }
      }
      expect(mouthTierGain(tier) >= 1).toBe(true);
      module.dispose();
    }
    expect(mouthTierGain('thumb')).toBeGreaterThan(mouthTierGain('hero'));
    expect(mouthStoneTint('thumb')).toBeGreaterThan(mouthStoneTint('hero'));
  });

  test('the hero arch stone is untouched neutral rock — the tint is a THUMB compensation', async () => {
    for (const tier of ['hero', 'medium'] as const) {
      const { module } = await live(tier);
      module.update(toFrameUpdate(poseFixture('mid', 9, 0, 1)));
      for (const name of ['exitMouthCyan', 'exitMouthMagenta']) {
        const c = carriersOf(module, name);
        expect(`${tier} ${name} albedo white:${c.stone.getHex()}`).toBe(`${tier} ${name} albedo white:16777215`);
        expect(c.rim).toBeCloseTo(MOUTH_RIM * tierEffectScale(tier).rim, 10);
      }
      module.dispose();
    }
  });

  test('the quartz body never changes: a dark crystal looks the same either way', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(poseFixture('cyanWalk', 9, 0)));
    const a = (meshNamed(module, 'cueCrystalQuartz').material as THREE.MeshStandardMaterial).color.getHex();
    module.update(toFrameUpdate(poseFixture('magentaWalk', 9, 1)));
    const b = (meshNamed(module, 'cueCrystalQuartz').material as THREE.MeshStandardMaterial).color.getHex();
    expect(b).toBe(a);
    expect(isCueColored(new THREE.Color(a))).toBe(false);
    module.dispose();
  });

  test('the fireflies are the accent and are never tinted by the cue', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(poseFixture('walk', 20, 1)));
    for (const name of ['passageFireflies', 'vaultDust', 'choiceFlood']) {
      const points = module.scene.getObjectByName(name) as THREE.Points;
      const uniforms = (points.material as THREE.ShaderMaterial).uniforms;
      const color = uniforms.uColor?.value as THREE.Color;
      expect(`${name}:${isCueColored(color)}`).toBe(`${name}:false`);
    }
    module.dispose();
  });
});

describe('memory-chain / the hop is bounded, and the respawn is a hard cut', () => {
  test('a one-stone step stays between the two stones and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];
    cellToWorld(5, 0, grid, from);
    cellToWorld(6, 0, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 5, 0, 1, 0)));
    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 6, 0, 1, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 6, 0, 1, 1), { scrubbed: false }));
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

  test('a respawn TELEPORTS: 31 stones in one frame with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(0, 0, grid, truth);

    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('end', MEMORYCHAIN_END, 1, 1), { scrubbed: false }));
    ctx.setTime(4.125);
    module.update(
      toFrameUpdate(poseFixture('respawn', 0, 0, 1, 1), {
        scrubbed: false,
        events: [{ type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('the magnitude guard snaps the same jump even with the events dropped', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(0, 0, grid, truth);
    ctx.setTime(7);
    module.update(toFrameUpdate(poseFixture('end', MEMORYCHAIN_END, 0, 1), { scrubbed: false }));
    ctx.setTime(7.125);
    module.update(toFrameUpdate(poseFixture('respawn', 0, 1, 1), { scrubbed: false }));
    expect(worldOf(module, 'wisp-anchor').x).toBeCloseTo(truth[0], 10);
    module.dispose();
  });

  test('a scrub snaps the wisp anywhere in the passage', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('a', 2, 0), { scrubbed: false }));
    module.update(toFrameUpdate(poseFixture('far', 27, 1, 1, 9000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(27, 0, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });
});

describe('memory-chain / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('cue-cyan-lit')));
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
    module.update(toFrameUpdate(poseFixture('atChoice', MEMORYCHAIN_END, 1, 1, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('scored', 0, 0, 1, 101), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('the +1 blooms where the reward was EARNED (cell 31), not where the agent respawned', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const choice: [number, number, number] = [0, 0, 0];
    cellToWorld(CHOICE_CELL, 0, grid, choice);

    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('run', MEMORYCHAIN_END, 1, 1, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 0, 0, 1, 201), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
      }),
    );

    const ripple = module.scene.getObjectByName('oakCelebrationRipple0');
    expect(ripple).toBeDefined();
    expect(ripple?.position.x ?? 0).toBeCloseTo(choice[0], 6);
    expect(worldOf(module, 'wisp-anchor').x).toBeCloseTo(-choice[0], 6);
    const flood = module.scene.getObjectByName('choiceFlood') as THREE.Points;
    expect(flood.position.x).toBeCloseTo(choice[0], 6);
    module.dispose();
  });

  test('the winning colour is the ACTION the agent took, and it floods the cave', async () => {
    for (const [action, hex] of [
      [0, TOKENS.cueCyan],
      [1, TOKENS.cueMagenta],
    ] as const) {
      const { module, ctx } = await live();
      const grid = resolveGrid(geometryOf());
      const choice: [number, number, number] = [0, 0, 0];
      const cueCell: [number, number, number] = [0, 0, 0];
      cellToWorld(CHOICE_CELL, 0, grid, choice);
      cellToWorld(CUE_CELL, 0, grid, cueCell);

      ctx.setTime(20);
      module.update(
        toFrameUpdate(poseFixture('scored', 0, 1, action, 300), {
          scrubbed: false,
          events: [{ type: 'goal' }, { type: 'cue', value: 1 }, { type: 'teleport' }, { type: 'done' }],
        }),
      );
      const flood = module.scene.getObjectByName('choiceFlood') as THREE.Points;
      const uniforms = (flood.material as THREE.ShaderMaterial).uniforms;
      expect((uniforms.uColor?.value as THREE.Color).getHex()).toBe(hex);
      expect(uniforms.uDensity?.value).toBe(1);
      const cyan = drawnIntensity(module, 'exitMouthCyanThroat');
      const magenta = drawnIntensity(module, 'exitMouthMagentaThroat');
      if (action === 0) expect(cyan).toBeGreaterThan(magenta);
      else expect(magenta).toBeGreaterThan(cyan);

      ctx.setTime(20 + 0.95);
      module.update(toFrameUpdate(poseFixture('after', 0, 1, action, 301), { scrubbed: false }));
      const mid = flood.position.x;
      expect(mid).toBeLessThan(choice[0]);
      expect(mid).toBeGreaterThan(cueCell[0]);
      ctx.setTime(20 + 2.4);
      module.update(toFrameUpdate(poseFixture('done', 0, 1, action, 302), { scrubbed: false }));
      expect(uniforms.uDensity?.value).toBe(0);
      expect(flood.position.x).toBeCloseTo(choice[0], 6);
      module.dispose();
    }
  });

  test('a win and a loss in ONE event buffer each resolve on their own evidence', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(50);
    module.update(
      toFrameUpdate(poseFixture('burst', 0, 1, 1, 600), {
        scrubbed: false,
        events: [
          { type: 'goal' },
          { type: 'cue', value: 0 },
          { type: 'teleport' },
          { type: 'done' },
          { type: 'cue', value: 1 },
          { type: 'teleport' },
          { type: 'done' },
        ],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    expect(pool.pulse.sigh).toBeGreaterThan(0);
    module.dispose();
  });

  test('a lone goal in a coalesced buffer never sighs', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(60);
    module.update(
      toFrameUpdate(poseFixture('twoWins', 0, 0, 1, 601), {
        scrubbed: false,
        events: [
          { type: 'goal' },
          { type: 'cue', value: 0 },
          { type: 'teleport' },
          { type: 'done' },
          { type: 'goal' },
          { type: 'cue', value: 1 },
          { type: 'teleport' },
          { type: 'done' },
        ],
      }),
    );
    expect(pool.stats.bursts).toBe(2);
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('a wrong choice sighs and douses BOTH mouths together', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(30);
    const restCyan = drawnIntensity(module, 'exitMouthCyanThroat');
    module.update(
      toFrameUpdate(poseFixture('lost', 0, 0, 1, 400), {
        scrubbed: false,
        events: [{ type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.sigh).toBeGreaterThan(0);
    expect(pool.stats.bursts).toBe(0);
    const cyan = drawnIntensity(module, 'exitMouthCyanThroat');
    const magenta = drawnIntensity(module, 'exitMouthMagentaThroat');
    expect(cyan).toBeLessThan(restCyan);
    expect(`equal:${cyan === magenta}`).toBe('equal:true');
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 0, 0, 1, 500), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('carrying-the-cue'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    const flood = module.scene.getObjectByName('choiceFlood') as THREE.Points;
    expect((flood.material as THREE.ShaderMaterial).uniforms.uDensity?.value).toBe(0);
    expect(drawnIntensity(module, 'cueCrystalGlow')).toBe(0);
    module.dispose();
  });

  test('the envelopes are pure, bounded and settle on 0', () => {
    expect(envelope(-Infinity, 1, 2)).toBe(0);
    expect(envelope(Number.NaN, 1, 2)).toBe(0);
    expect(envelope(0, 1, 2)).toBe(1);
    expect(envelope(99, 1, 2)).toBe(0);
    for (let i = 0; i <= 50; i++) {
      const e = envelope((i / 50) * 1.35, 1.35, 2.3);
      expect(e).toBeGreaterThanOrEqual(0);
      expect(e).toBeLessThanOrEqual(1);
    }
    expect(floodProgress(-Infinity)).toBe(0);
    expect(floodProgress(99)).toBe(0);
    expect(floodProgress(0)).toBe(0);
    expect(floodProgress(0.95)).toBeGreaterThan(0.4);
    expect(floodProgress(0.95)).toBeLessThan(0.6);
    expect(floodRunning(0)).toBe(true);
    expect(floodRunning(-Infinity)).toBe(false);
    expect(floodRunning(Number.NaN)).toBe(false);
    expect(floodRunning(1.89)).toBe(true);
    expect(floodRunning(1.91)).toBe(false);
  });

  test('posOf is the only position the scene ever draws from', () => {
    const f = poseFixture('x', 19, 1, 1);
    const state = {
      pose: Float32Array.from(f.pose),
      ctrl: Float32Array.from(f.ctrl ?? []),
      done: false,
      step: 0,
      simTime: 0,
      derived: f.derived,
    };
    expect(posOf(state)).toBe(19);
  });
});

describe('memory-chain / lifecycle', () => {
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

  test('every tier renders inside its declared budget', async () => {
    const rows: string[] = [];
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      module.update(toFrameUpdate(fixtureNamed('cue-magenta-lit')));
      const idle = measureBudget(module);
      module.update(
        toFrameUpdate(poseFixture('scored', 0, 0, 1, 900), {
          scrubbed: false,
          events: [{ type: 'goal' }, { type: 'cue', value: 0 }, { type: 'teleport' }, { type: 'done' }],
        }),
      );
      const busy = measureBudget(module);
      module.setDebug(true);
      module.update(toFrameUpdate(fixtureNamed('cue-magenta-lit')));
      const debug = measureBudget(module);
      rows.push(
        `${tier.padEnd(6)} idle ${idle.draws}/${(idle.triangles / 1000).toFixed(1)}k · ` +
          `celebrating ${busy.draws}/${(busy.triangles / 1000).toFixed(1)}k · ` +
          `overlay ${debug.draws}/${(debug.triangles / 1000).toFixed(1)}k`,
      );
      expect(busy.draws).toBeLessThanOrEqual(manifest.budget.drawCalls);
      expect(busy.triangles).toBeLessThanOrEqual(manifest.budget.triangles);
      if (tier === 'thumb') assertThumbBudget(module, 'memory-chain (overlay on)');
      module.dispose();
    }
    console.log(`\n  memory-chain budget (scene-graph upper bound)\n    ${rows.join('\n    ')}\n`);
  });

  test('setDebug toggles the ground-truth overlay onto the scene camera', async () => {
    const { module } = await live();
    module.setDebug(true);
    module.update(toFrameUpdate(fixtureNamed('carrying-the-cue')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'caveGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'caveGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera looks DOWN the passage and holds both of its ends', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(fixtureNamed('cue-cyan-lit')));
    const cam = module.camera;
    const ends: [number, number, number] = [0, 0, 0];
    cellToWorld(CUE_CELL, 0, grid, ends);
    expect(cam.position.x).toBeLessThan(ends[0]);
    expect(cam.position.y).toBeGreaterThan(3);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.x).toBeGreaterThan(0.5);
    expect(dir.y).toBeLessThan(0);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);

    cam.aspect = 16 / 9;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    const p = new THREE.Vector3();
    for (const cell of [CUE_CELL, CHOICE_CELL]) {
      cellToWorld(cell, 0, grid, ends);
      p.set(ends[0], 0.5, 0).project(cam);
      expect(`cell${cell} x:${Math.abs(p.x) < 0.95}`).toBe(`cell${cell} x:true`);
      expect(`cell${cell} y:${Math.abs(p.y) < 0.95}`).toBe(`cell${cell} y:true`);
      expect(p.z).toBeLessThan(1);
    }
    module.dispose();
  });

  test('the thumb tier still holds the whole passage', async () => {
    const { module } = await live('thumb');
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(fixtureNamed('cue-magenta-lit')));
    const cam = module.camera;
    cam.aspect = 1;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    const ends: [number, number, number] = [0, 0, 0];
    const p = new THREE.Vector3();
    for (const cell of [CUE_CELL, CHOICE_CELL]) {
      cellToWorld(cell, 0, grid, ends);
      p.set(ends[0], 0.5, 0).project(cam);
      expect(`thumb cell${cell}:${Math.abs(p.x) < 1 && Math.abs(p.y) < 1}`).toBe(`thumb cell${cell}:true`);
    }
    module.dispose();
  });
});
