/// <reference types="bun" />
// Headless tests for kit/ghosts — THE POPULATION LAYER.
//
// There is no WebGL context under `bun test`, so nothing here renders. What IS tested is
// everything the layer can get wrong without a GPU, and that turns out to be every claim
// the module makes:
//
//   * SCALE     — the draw count is a function of the CATEGORY, never of the ghost count:
//                 identical at 64 and at 512 instances, and inside the per-category cap.
//                 The TRIANGLE count is the opposite — strictly linear in the ghost count
//                 — so it is measured against the layer's own published per-ghost table
//                 (`GHOST_TRIANGLES_PER_GHOST`), which is what a caller sizes `maxGhosts`
//                 against. A flat draw count is not a free population.
//   * ACCURACY  — cheetah ghost bones decompose back to `fk()` truth (position, heading
//                 and length) for 3 poses x 8 ghosts, each ghost at its own TRUE rootx;
//                 grid ghosts land inside their own cell for 64 co-located seeds; classic
//                 markers sit exactly on the `@sim/envs` kinematics helpers.
//   * COLOUR    — the rate ramp is monotone in luminance for all 16 accents, emphasis 0
//                 and 1 stay under the bloom threshold, and a diverged seed freezes.
//   * LIFECYCLE — build -> dispose round-trips clean three times, and `update()` allocates
//                 no THREE objects and never reallocates an instance buffer.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import {
  GHOST_BLOOM_THRESHOLD,
  GHOST_CELL_OFFSET_MAX,
  GHOST_EMPHASIS_GAIN,
  GHOST_GLINT_SECONDS,
  GHOST_GROUP_NAME,
  GHOST_TRIANGLES_PER_GHOST,
  GhostLayer,
  ghostCellOffset,
  ghostGlintEnvelope,
  makeGhostLayer,
  type GhostCategory,
  type GhostLayerOptions,
} from '@kit/ghosts';
import { CELL, cellToWorld, worldToCell } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import type { EnvGeometry } from '@sim/types';
import { CHEETAH_BONES, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import { acrobotPoints } from '@sim/envs/acrobot';
import { cartPoleTip } from '@sim/envs/cartpole';
import { mountainCarHeight } from '@sim/envs/mountaincar';
import { pendulumTip } from '@sim/envs/pendulum';

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const CHEETAH_GEOMETRY: EnvGeometry = { kind: 'cheetah', cheetah: cheetahGeometry };

function gridGeometry(w = 11, h = 11): EnvGeometry {
  return { kind: 'grid', grid: { w, h, walls: new Uint8Array(w * h) } };
}

/** The classic geometries, taken from the decoders so the anchors are the real ones. */
const CLASSIC_GEOMETRY: EnvGeometry = {
  kind: 'classic',
  classic: {
    heightAt: mountainCarHeight,
    anchors: { pivot: [0, 0, 0] },
  },
};

function makeLayer(over: Partial<GhostLayerOptions> = {}): GhostLayer {
  const category: GhostCategory = over.category ?? 'cheetah';
  const geometry =
    over.geometry ??
    (category === 'cheetah' ? CHEETAH_GEOMETRY : category === 'grid' ? gridGeometry() : CLASSIC_GEOMETRY);
  return makeGhostLayer({
    category,
    envId: over.envId ?? (category === 'cheetah' ? 'HalfCheetah-v4' : category === 'grid' ? 'FourRooms-cont-v0' : 'MountainCar-cont-v0'),
    geometry,
    maxGhosts: over.maxGhosts ?? 8,
    tokens: TOKENS,
    accent: over.accent ?? TOKENS.accents['half-cheetah'] ?? 0xffb347,
    tier: over.tier ?? 'hero',
    ...(over.name === undefined ? {} : { name: over.name }),
  });
}

function pose9(values: readonly number[]): Float32Array {
  const p = new Float32Array(9);
  for (let i = 0; i < 9; i++) p[i] = values[i] ?? 0;
  return p;
}

const CHEETAH_POSES: readonly Float32Array[] = [
  pose9([0, 0, 0, 0, 0, 0, 0, 0, 0]),
  pose9([3.5, 0.12, -0.21, 0.44, -0.31, 0.16, -0.52, 0.27, 0.05]),
  pose9([-17.25, -0.06, 0.63, -0.9, 0.72, -0.4, 0.85, -0.66, 0.31]),
];

/** Decompose instance `i` of `bank` into position / quaternion / scale. */
function readInstance(
  layer: GhostLayer,
  bankName: string,
  i: number,
): { pos: THREE.Vector3; quat: THREE.Quaternion; scale: THREE.Vector3 } {
  const bank = layer.banks.find((b) => b.name === bankName);
  if (!bank) throw new Error(`no bank named ${bankName} (have ${layer.banks.map((b) => b.name).join(', ')})`);
  const m = new THREE.Matrix4();
  bank.mesh.getMatrixAt(i, m);
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  m.decompose(pos, quat, scale);
  return { pos, quat, scale };
}

/**
 * The RAW instance matrix. `Matrix4.decompose` reports a degenerate (zero-scale)
 * transform as scale (1,1,1) — it bails on `det === 0` — so "this ghost is hidden" has to
 * be read off the elements, not off the decomposition.
 */
function readMatrix(layer: GhostLayer, bankName: string, i: number): THREE.Matrix4 {
  const bank = layer.banks.find((b) => b.name === bankName);
  if (!bank) throw new Error(`no bank named ${bankName}`);
  const m = new THREE.Matrix4();
  bank.mesh.getMatrixAt(i, m);
  return m;
}

/** Largest absolute entry of the 3x3 linear block — 0 means "draws nothing". */
function linearMagnitude(m: THREE.Matrix4): number {
  const e = m.elements;
  let max = 0;
  for (const k of [0, 1, 2, 4, 5, 6, 8, 9, 10]) max = Math.max(max, Math.abs(e[k] ?? 0));
  return max;
}

function readColor(layer: GhostLayer, bankName: string, i: number): THREE.Color {
  const bank = layer.banks.find((b) => b.name === bankName);
  if (!bank) throw new Error(`no bank named ${bankName}`);
  const c = new THREE.Color();
  bank.mesh.getColorAt(i, c);
  return c;
}

/** Rec.709 luminance of whatever space three keeps the channels in. */
function luminance(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

// ---------------------------------------------------------------------------
// scale: the draw budget
// ---------------------------------------------------------------------------

describe('kit/ghosts — draw budget', () => {
  const CAPS: Record<GhostCategory, number> = { cheetah: 8, grid: 3, classic: 3 };

  test('the draw count is a function of the category, not of the ghost count', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      const small = makeLayer({ category, maxGhosts: 64 });
      const large = makeLayer({ category, maxGhosts: 512 });
      expect(small.draws()).toBe(large.draws());
      expect(large.draws()).toBeLessThanOrEqual(CAPS[category]);
      expect(large.draws()).toBeGreaterThan(0);
      // every bank really does carry every ghost — the cap is instancing, not truncation
      for (const bank of large.banks) expect(bank.mesh.count).toBe(512);
      for (const bank of small.banks) expect(bank.mesh.count).toBe(64);
      small.dispose();
      large.dispose();
    }
  });

  test('the cheetah plan is 7 bones + 1 head, all instanced', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 512 });
    expect(layer.banks.length).toBe(CHEETAH_BONES.length + 1);
    expect(layer.banks.map((b) => b.name)).toEqual([
      'bone0', 'bone1', 'bone2', 'bone3', 'bone4', 'bone5', 'bone6', 'head',
    ]);
    for (const bank of layer.banks) {
      expect((bank.mesh as { isInstancedMesh?: boolean }).isInstancedMesh).toBe(true);
      expect(bank.mesh.frustumCulled).toBe(false);
      expect(bank.mesh.castShadow).toBe(false);
      expect((bank.mesh.material as THREE.Material).depthWrite).toBe(false);
      expect(bank.mesh.renderOrder).toBeLessThan(0); // behind the hero
    }
    expect(layer.draws()).toBe(8);
    layer.dispose();
  });

  test('the grid plan is the wisp treatment at ghost scale: core + shell + contact shade', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 256 });
    expect(layer.banks.map((b) => b.name)).toEqual(['core', 'shell', 'shade']);
    expect(layer.banks.map((b) => b.role)).toEqual(['emitter', 'shade', 'shade']);
    expect(layer.draws()).toBe(3);
    layer.dispose();
  });

  test('classic plans stay within 3 banks and carry the links each env actually has', () => {
    const plans: Record<string, string[]> = {
      'MountainCar-cont-v0': ['marker'],
      'Pendulum-cont-v0': ['marker', 'link0'],
      'CartPole-cont-v0': ['marker', 'link0'],
      'Acrobot-cont-v0': ['marker', 'link0', 'link1'],
    };
    for (const [envId, names] of Object.entries(plans)) {
      const layer = makeLayer({
        category: 'classic',
        envId: envId as GhostLayerOptions['envId'],
        maxGhosts: 512,
      });
      expect(layer.banks.map((b) => b.name)).toEqual(names);
      expect(layer.draws()).toBe(names.length);
      expect(layer.draws()).toBeLessThanOrEqual(3);
      layer.dispose();
    }
  });

  test('a hidden group draws nothing and maxGhosts 0 is legal', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 0 });
    expect(layer.draws()).toBe(0);
    layer.dispose();

    const live = makeLayer({ category: 'grid', maxGhosts: 4 });
    expect(live.draws()).toBe(3);
    live.group.visible = false;
    expect(live.draws()).toBe(0);
    live.dispose();
  });

  test('tier scaling halves instance DETAIL, never the draw count', () => {
    const counts = new Map<QualityTier, { draws: number; verts: number }>();
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const layer = makeLayer({ category: 'cheetah', maxGhosts: 64, tier });
      let verts = 0;
      for (const bank of layer.banks) verts += bank.mesh.geometry.getAttribute('position')?.count ?? 0;
      counts.set(tier, { draws: layer.draws(), verts });
      layer.dispose();
    }
    const hero = counts.get('hero');
    const medium = counts.get('medium');
    const thumb = counts.get('thumb');
    expect(hero?.draws).toBe(8);
    expect(medium?.draws).toBe(8);
    expect(thumb?.draws).toBe(8);
    expect((thumb?.verts ?? 0)).toBeLessThan((hero?.verts ?? 0) / 2);
    expect((medium?.verts ?? 0)).toBeLessThan(hero?.verts ?? 0);
    expect((medium?.verts ?? 0)).toBeGreaterThan(thumb?.verts ?? 0);
  });

  test('the group carries the reserved name the host and STYLE.md agree on', () => {
    const layer = makeLayer();
    expect(layer.group.name).toBe(GHOST_GROUP_NAME);
    expect(GHOST_GROUP_NAME).toBe('oakGhostLayer');
    layer.dispose();
  });
});

// ---------------------------------------------------------------------------
// scale: the TRIANGLE budget — the half the draw count cannot express
// ---------------------------------------------------------------------------

describe('kit/ghosts — triangle budget', () => {
  /** three's own rule: indexed geometry counts index/3, otherwise position/3 */
  function trianglesOf(layer: GhostLayer): number {
    let n = 0;
    for (const bank of layer.banks) {
      const index = bank.mesh.geometry.getIndex();
      const verts = index !== null ? index.count : (bank.mesh.geometry.getAttribute('position')?.count ?? 0);
      n += Math.floor(verts / 3) * bank.mesh.count;
    }
    return n;
  }

  test('triangles are LINEAR in the ghost count while draws stay flat', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      const counts = [64, 128, 1024];
      const layers = counts.map((maxGhosts) => makeLayer({ category, maxGhosts }));
      const [a, b, c] = layers;
      if (!a || !b || !c) throw new Error('layer build failed');
      // the claim the commit made and the one it did not
      expect(a.draws()).toBe(c.draws());
      expect(c.triangles()).toBe(a.triangles() * 16);
      expect(b.triangles()).toBe(a.triangles() * 2);
      // and each agrees with an independent count off the raw buffers
      for (const layer of layers) expect(layer.triangles()).toBe(trianglesOf(layer));
      for (const layer of layers) layer.dispose();
    }
  });

  test('trianglesPerGhost is the published table, and maxGhosts × it is the load', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      for (const tier of ['hero', 'medium', 'thumb'] as const) {
        const layer = makeLayer({ category, maxGhosts: 1024, tier });
        const per = layer.trianglesPerGhost();
        const quoted = GHOST_TRIANGLES_PER_GHOST[category][tier];
        // the table is a per-category upper bound (classic plans differ by env)
        expect(`${category}/${tier}: ${per} <= ${quoted}`).toBe(`${category}/${tier}: ${per} <= ${quoted}`);
        expect(per).toBeLessThanOrEqual(quoted);
        expect(layer.triangles()).toBe(per * 1024);
        layer.dispose();
      }
    }
    // the two plans that do not vary by env are quoted EXACTLY, so the table is not
    // silently generous: this is the number a caller sizes a population against.
    for (const [category, tier, want] of [
      ['cheetah', 'hero', 1188],
      ['cheetah', 'medium', 668],
      ['cheetah', 'thumb', 360],
      ['grid', 'hero', 262],
      ['grid', 'medium', 102],
      ['grid', 'thumb', 102],
    ] as const) {
      const layer = makeLayer({ category, maxGhosts: 8, tier });
      expect(`${category}/${tier} = ${layer.trianglesPerGhost()}`).toBe(`${category}/${tier} = ${want}`);
      expect(GHOST_TRIANGLES_PER_GHOST[category][tier]).toBe(want);
      layer.dispose();
    }
  });

  test('the widest classic plan (acrobot: marker + 2 links) is the one the table quotes', () => {
    const plans: [string, number][] = [
      ['MountainCar-cont-v0', 180],
      ['CartPole-cont-v0', 192],
      ['Pendulum-cont-v0', 192],
      ['Acrobot-cont-v0', 204],
    ];
    for (const [envId, want] of plans) {
      const layer = makeLayer({
        category: 'classic',
        envId: envId as GhostLayerOptions['envId'],
        maxGhosts: 64,
        tier: 'hero',
      });
      expect(`${envId}: ${layer.trianglesPerGhost()}`).toBe(`${envId}: ${want}`);
      expect(layer.trianglesPerGhost()).toBeLessThanOrEqual(GHOST_TRIANGLES_PER_GHOST.classic.hero);
      layer.dispose();
    }
    expect(GHOST_TRIANGLES_PER_GHOST.classic.hero).toBe(204);
  });

  test('a tier drop cuts triangles, never banks — the ladder is instance detail', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      const hero = makeLayer({ category, maxGhosts: 256, tier: 'hero' });
      const thumb = makeLayer({ category, maxGhosts: 256, tier: 'thumb' });
      expect(thumb.draws()).toBe(hero.draws());
      expect(thumb.triangles()).toBeLessThan(hero.triangles());
      hero.dispose();
      thumb.dispose();
    }
  });

  test('a hidden group, an empty layer and a disposed layer all cost zero triangles', () => {
    const empty = makeLayer({ category: 'grid', maxGhosts: 0 });
    expect(empty.triangles()).toBe(0);
    expect(empty.trianglesPerGhost()).toBeGreaterThan(0); // the plan exists; nobody is in it
    empty.dispose();

    const live = makeLayer({ category: 'cheetah', maxGhosts: 32 });
    expect(live.triangles()).toBe(32 * live.trianglesPerGhost());
    live.group.visible = false;
    expect(live.triangles()).toBe(0);
    expect(live.draws()).toBe(0);
    live.group.visible = true;
    live.dispose();
    expect(live.triangles()).toBe(0);
    expect(live.draws()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// accuracy: cheetah
// ---------------------------------------------------------------------------

describe('kit/ghosts — cheetah accuracy', () => {
  test('every bone decomposes back to fk() truth for 3 poses x 8 ghosts', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 8 });
    const radius = cheetahGeometry.radius;
    let worst = 0;

    for (const base of CHEETAH_POSES) {
      // each ghost races at its OWN rootx — the population spreads along the real track
      const poses: Float32Array[] = [];
      for (let g = 0; g < 8; g++) {
        const p = pose9(Array.from(base));
        p[0] = (base[0] ?? 0) + g * 1.7;
        poses.push(p);
        layer.setGhostFrame(g, p, true);
      }

      for (let g = 0; g < 8; g++) {
        const truth = cheetahGeometry.fk(poses[g] as Float32Array);
        for (let b = 0; b < CHEETAH_BONES.length; b++) {
          const bone = CHEETAH_BONES[b] as readonly [number, number];
          const ax = truth[bone[0] * 2] ?? 0;
          const ay = truth[bone[0] * 2 + 1] ?? 0;
          const bx = truth[bone[1] * 2] ?? 0;
          const by = truth[bone[1] * 2 + 1] ?? 0;
          const { pos, quat, scale } = readInstance(layer, `bone${b}`, g);

          // FK x -> world x, FK z -> world y, z = 0 (the hero's own planar mapping)
          const posErr = pos.distanceTo(
            new THREE.Vector3((ax + bx) / 2, (ay + by) / 2, 0),
          );
          const len = Math.hypot(bx - ax, by - ay);
          const lenErr = Math.abs(scale.y - len);
          // the unit geometry runs along +y; the rotation must map it onto the bone
          const axis = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
          const dirErr = axis.distanceTo(new THREE.Vector3((bx - ax) / len, (by - ay) / len, 0));

          worst = Math.max(worst, posErr, lenErr, dirErr);
          expect(posErr).toBeLessThan(1e-3);
          expect(lenErr).toBeLessThan(1e-3);
          expect(dirErr).toBeLessThan(1e-3);
          // the radial scale is the SIM's own capsule radius (instanceMatrix is f32)
          expect(scale.x).toBeCloseTo(radius, 6);
          expect(scale.z).toBeCloseTo(radius, 6);
        }

        const head = readInstance(layer, 'head', g);
        const hx = truth[FK_POINT_INDEX.head * 2] ?? 0;
        const hy = truth[FK_POINT_INDEX.head * 2 + 1] ?? 0;
        const headErr = head.pos.distanceTo(new THREE.Vector3(hx, hy, 0));
        worst = Math.max(worst, headErr);
        expect(headErr).toBeLessThan(1e-3);
        expect(head.scale.x).toBeCloseTo(cheetahGeometry.headRadius, 6);
      }
    }
    expect(worst).toBeLessThan(1e-3);
    layer.dispose();
  });

  test('ghosts sit at TRUE rootx — a shared origin would collapse them', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 4 });
    const xs: number[] = [];
    for (let g = 0; g < 4; g++) {
      const p = pose9([g * 12.5, 0.05, 0, 0, 0, 0, 0, 0, 0]);
      layer.setGhostFrame(g, p, true);
      xs.push(readInstance(layer, 'head', g).pos.x);
    }
    for (let g = 1; g < 4; g++) {
      expect((xs[g] ?? 0) - (xs[g - 1] ?? 0)).toBeCloseTo(12.5, 5);
    }
    layer.dispose();
  });

  test('a glint puffs thickness and brightness but never moves a bone endpoint', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 2 });
    const p = CHEETAH_POSES[1] as Float32Array;
    layer.setGhostFrame(0, p, true);
    layer.setRate(0, 0.6);
    layer.update(1 / 60, 10);
    const before = readInstance(layer, 'bone0', 0);
    const colorBefore = readColor(layer, 'bone0', 0);

    layer.glint(0);
    layer.update(1 / 60, 10.01);
    const during = readInstance(layer, 'bone0', 0);
    const colorDuring = readColor(layer, 'bone0', 0);

    expect(during.pos.distanceTo(before.pos)).toBeLessThan(1e-9);
    expect(Math.abs(during.scale.y - before.scale.y)).toBeLessThan(1e-9); // length exact
    expect(during.scale.x).toBeGreaterThan(before.scale.x); // thickness puffs
    expect(luminance(colorDuring)).toBeGreaterThan(luminance(colorBefore));

    // ...and it retires cleanly at the end of its 300 ms span (canon lesson 6)
    layer.update(1 / 60, 10 + GHOST_GLINT_SECONDS + 0.01);
    const after = readInstance(layer, 'bone0', 0);
    expect(after.scale.x).toBeCloseTo(before.scale.x, 9);
    expect(luminance(readColor(layer, 'bone0', 0))).toBeCloseTo(luminance(colorBefore), 9);
    layer.dispose();
  });
});

// ---------------------------------------------------------------------------
// accuracy: grid
// ---------------------------------------------------------------------------

describe('kit/ghosts — grid accuracy', () => {
  test('64 co-located ghosts all render INSIDE their true cell', () => {
    const grid = { w: 11, h: 11 };
    const layer = makeLayer({ category: 'grid', maxGhosts: 64, geometry: gridGeometry(11, 11) });
    const cell = { x: 3, y: 7 };
    const p = pose9([cell.x, cell.y, 9, 9]);
    for (let g = 0; g < 64; g++) layer.setGhostFrame(g, p, true);

    const [tx, , tz] = cellToWorld(cell.x, cell.y, grid);
    let maxOffset = 0;
    const seen = new Set<string>();
    for (let g = 0; g < 64; g++) {
      const { pos } = readInstance(layer, 'core', g);
      const d = Math.hypot(pos.x - tx, pos.z - tz);
      maxOffset = Math.max(maxOffset, d);
      // the strongest statement: it decodes back to the cell the agent is actually in
      expect(worldToCell(pos.x, pos.z, grid)).toEqual(cell);
      seen.add(`${pos.x.toFixed(6)},${pos.z.toFixed(6)}`);
    }
    // bounded by construction, and every seed is separable
    expect(maxOffset).toBeLessThanOrEqual(GHOST_CELL_OFFSET_MAX * CELL + 1e-6);
    expect(maxOffset).toBeLessThan(0.5 * CELL);
    expect(seen.size).toBe(64);
    layer.dispose();
  });

  test('the fan-out is a deterministic function of the index alone', () => {
    const a = new Float32Array(2);
    const b = new Float32Array(2);
    for (const i of [0, 1, 7, 63, 511]) {
      ghostCellOffset(i, 512, a);
      ghostCellOffset(i, 512, b);
      expect([a[0], a[1]]).toEqual([b[0], b[1]]);
      expect(Math.hypot(a[0] ?? 0, a[1] ?? 0)).toBeLessThanOrEqual(GHOST_CELL_OFFSET_MAX + 1e-9);
    }
  });

  test('the core rides at ghost height and the contact shade lies on the floor', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 4, geometry: gridGeometry(9, 9) });
    layer.setGhostFrame(0, pose9([4, 4]), true);
    const core = readInstance(layer, 'core', 0);
    const shell = readInstance(layer, 'shell', 0);
    const shade = readInstance(layer, 'shade', 0);
    expect(core.pos.y).toBeGreaterThan(0);
    expect(core.pos.y).toBeLessThan(TOKENS.wispHeight);
    expect(shell.pos.y).toBeCloseTo(core.pos.y, 6);
    expect(shell.scale.x).toBeGreaterThan(core.scale.x); // a shell, not a billboard
    expect(shade.pos.y).toBeGreaterThan(0);
    expect(shade.pos.y).toBeLessThan(0.05); // just off the floor plane, never z-fighting
    expect(shade.pos.x).toBeCloseTo(core.pos.x, 6);
    expect(shade.pos.z).toBeCloseTo(core.pos.z, 6);
    layer.dispose();
  });

  test('the shell stays OUTSIDE the orb it backs, at every emphasis level', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 2, geometry: gridGeometry(9, 9) });
    layer.setGhostFrame(0, pose9([4, 4]), true);
    for (const level of [0, 1, 2] as const) {
      layer.setEmphasis(0, level);
      const core = readInstance(layer, 'core', 0);
      const shell = readInstance(layer, 'shell', 0);
      expect(shell.scale.x).toBeGreaterThan(core.scale.x);
    }
    layer.dispose();
  });

  test('grid ghosts read the grid they are given, not a hard-coded one', () => {
    const wide = makeLayer({ category: 'grid', maxGhosts: 1, geometry: gridGeometry(32, 1) });
    wide.setGhostFrame(0, pose9([31, 0]), true);
    const { pos } = readInstance(wide, 'core', 0);
    const [tx, , tz] = cellToWorld(31, 0, { w: 32, h: 1 });
    expect(Math.hypot(pos.x - tx, pos.z - tz)).toBeLessThanOrEqual(GHOST_CELL_OFFSET_MAX * CELL + 1e-6);
    wide.dispose();
  });
});

// ---------------------------------------------------------------------------
// accuracy: classic
// ---------------------------------------------------------------------------

describe('kit/ghosts — classic accuracy', () => {
  test('MountainCar ghosts sit on the terrain curve at their own position', () => {
    const layer = makeLayer({ category: 'classic', envId: 'MountainCar-cont-v0', maxGhosts: 8 });
    for (let g = 0; g < 8; g++) {
      const x = -1.2 + (g / 7) * 1.8;
      layer.setGhostFrame(g, pose9([x, 0.01, 0.5]), true);
      const { pos } = readInstance(layer, 'marker', g);
      expect(pos.x).toBeCloseTo(x, 6);
      expect(pos.y).toBeCloseTo(mountainCarHeight(x), 6);
      expect(pos.z).toBeCloseTo(0, 9);
    }
    layer.dispose();
  });

  test('Pendulum ghosts hang their bob exactly on pendulumTip, link spanning the pivot', () => {
    const layer = makeLayer({ category: 'classic', envId: 'Pendulum-cont-v0', maxGhosts: 6 });
    for (let g = 0; g < 6; g++) {
      const th = -Math.PI + (g / 5) * 2 * Math.PI;
      layer.setGhostFrame(g, pose9([th, 1.2, 0.4]), true);
      const tip = pendulumTip(th);
      const bob = readInstance(layer, 'marker', g);
      expect(bob.pos.x).toBeCloseTo(tip[0] ?? 0, 6);
      expect(bob.pos.y).toBeCloseTo(tip[1] ?? 0, 6);
      const link = readInstance(layer, 'link0', g);
      expect(link.pos.x).toBeCloseTo((tip[0] ?? 0) / 2, 6);
      expect(link.pos.y).toBeCloseTo((tip[1] ?? 0) / 2, 6);
      expect(link.scale.y).toBeCloseTo(Math.hypot(tip[0] ?? 0, tip[1] ?? 0), 6);
    }
    layer.dispose();
  });

  test('Acrobot ghosts place both links on acrobotPoints', () => {
    const layer = makeLayer({ category: 'classic', envId: 'Acrobot-cont-v0', maxGhosts: 5 });
    for (let g = 0; g < 5; g++) {
      const th1 = -1.1 + g * 0.5;
      const th2 = 0.7 - g * 0.4;
      layer.setGhostFrame(g, pose9([th1, th2, 0.2, -0.3]), true);
      const pts = acrobotPoints(th1, th2);
      const ex = pts[0] ?? 0;
      const ey = pts[1] ?? 0;
      const tx = pts[2] ?? 0;
      const ty = pts[3] ?? 0;

      const link0 = readInstance(layer, 'link0', g);
      expect(link0.pos.x).toBeCloseTo(ex / 2, 6);
      expect(link0.pos.y).toBeCloseTo(ey / 2, 6);
      expect(link0.scale.y).toBeCloseTo(Math.hypot(ex, ey), 6);

      const link1 = readInstance(layer, 'link1', g);
      expect(link1.pos.x).toBeCloseTo((ex + tx) / 2, 6);
      expect(link1.pos.y).toBeCloseTo((ey + ty) / 2, 6);
      expect(link1.scale.y).toBeCloseTo(Math.hypot(tx - ex, ty - ey), 6);

      const bob = readInstance(layer, 'marker', g);
      expect(bob.pos.x).toBeCloseTo(tx, 6);
      expect(bob.pos.y).toBeCloseTo(ty, 6);
    }
    layer.dispose();
  });

  test('CartPole ghosts put the cart on the track and the pole on cartPoleTip', () => {
    const layer = makeLayer({ category: 'classic', envId: 'CartPole-cont-v0', maxGhosts: 5 });
    for (let g = 0; g < 5; g++) {
      const x = -2.4 + g * 1.2;
      const th = -0.2 + g * 0.1;
      layer.setGhostFrame(g, pose9([x, 0.1, th, -0.05]), true);
      const tip = cartPoleTip(x, th);
      const cart = readInstance(layer, 'marker', g);
      expect(cart.pos.x).toBeCloseTo(x, 6);
      expect(cart.pos.y).toBeCloseTo(0, 6);
      const pole = readInstance(layer, 'link0', g);
      expect(pole.pos.x).toBeCloseTo((x + (tip[0] ?? 0)) / 2, 6);
      expect(pole.pos.y).toBeCloseTo((tip[1] ?? 0) / 2, 6);
      expect(pole.scale.y).toBeCloseTo(Math.hypot((tip[0] ?? 0) - x, tip[1] ?? 0), 6);
    }
    layer.dispose();
  });
});

// ---------------------------------------------------------------------------
// colour
// ---------------------------------------------------------------------------

describe('kit/ghosts — the rate ramp', () => {
  test('is monotone in luminance for every one of the 16 accents', () => {
    const c = new THREE.Color();
    for (const [slug, hex] of Object.entries(TOKENS.accents)) {
      const layer = makeLayer({ category: 'grid', maxGhosts: 1, accent: hex });
      let previous = -Infinity;
      for (let k = 0; k <= 512; k++) {
        const lum = luminance(layer.rampColor(k / 512, c));
        if (!(lum >= previous - 1e-9)) {
          throw new Error(`${slug}: ramp dipped at t=${k / 512} (${lum} < ${previous})`);
        }
        previous = lum;
      }
      layer.dispose();
    }
  });

  test('runs night -> accent -> celebration gold, gold only in the very top band', () => {
    const accent = TOKENS.accents['four-rooms'] ?? 0x8fd47a;
    const layer = makeLayer({ category: 'grid', maxGhosts: 1, accent });
    const c = new THREE.Color();
    const low = layer.rampColor(0, c).clone();
    const mid = layer.rampColor(0.82, c).clone();
    const top = layer.rampColor(1, c).clone();
    const night = new THREE.Color(TOKENS.palettes.night.skyTop);
    const gold = new THREE.Color(TOKENS.celebrationGold);

    // the bottom of the ramp is night, dimmed by the value floor
    expect(low.getHex()).not.toBe(gold.getHex());
    expect(luminance(low)).toBeLessThan(luminance(night) + 1e-6);
    // the knee is the env's own accent at full value
    expect(mid.r / Math.max(mid.g, 1e-9)).toBeCloseTo(
      new THREE.Color(accent).r / Math.max(new THREE.Color(accent).g, 1e-9),
      3,
    );
    // and only the very top is gold
    expect(top.getHexString()).toBe(gold.getHexString());
    layer.dispose();
  });

  test('setRate drives instanceColor monotonically on every emitter bank', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 2 });
    layer.setGhostFrame(0, CHEETAH_POSES[1] as Float32Array, true);
    layer.setEmphasis(0, 1);
    let previous = -Infinity;
    for (let k = 0; k <= 64; k++) {
      layer.setRate(0, k / 64);
      const head = luminance(readColor(layer, 'head', 0));
      const bone = luminance(readColor(layer, 'bone3', 0));
      expect(head).toBeCloseTo(bone, 12); // one ghost, one colour, every bank
      expect(head).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = head;
    }
    layer.dispose();
  });

  test('emphasis 0 and 1 stay under the bloom threshold; 2 is the only one allowed to rim', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 1, accent: TOKENS.accents['memory-chain'] ?? 0x9ff0a8 });
    const c = new THREE.Color();
    const peak = (level: 0 | 1 | 2): number => {
      layer.setEmphasis(0, level);
      let max = 0;
      for (let k = 0; k <= 128; k++) {
        layer.setRate(0, k / 128);
        layer.emitterColor(0, c);
        max = Math.max(max, c.r, c.g, c.b);
      }
      return max;
    };
    expect(peak(0)).toBeLessThan(GHOST_BLOOM_THRESHOLD);
    expect(peak(1)).toBeLessThan(GHOST_BLOOM_THRESHOLD);
    expect(peak(2)).toBeGreaterThan(peak(1));
    expect(peak(2)).toBeCloseTo(GHOST_EMPHASIS_GAIN[2], 2);
    layer.dispose();
  });

  test('emphasis brightens the emitter and grows the dark pocket under a grid ghost', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 2 });
    layer.setGhostFrame(0, pose9([5, 5]), true);
    layer.setRate(0, 0.5);
    const read = (level: 0 | 1 | 2): { lum: number; shade: number } => {
      layer.setEmphasis(0, level);
      return {
        lum: luminance(readColor(layer, 'core', 0)),
        shade: readInstance(layer, 'shade', 0).scale.x,
      };
    };
    const faint = read(0);
    const readable = read(1);
    const leader = read(2);
    expect(readable.lum).toBeGreaterThan(faint.lum);
    expect(leader.lum).toBeGreaterThan(readable.lum);
    expect(readable.shade).toBeGreaterThan(faint.shade);
    expect(leader.shade).toBeGreaterThan(readable.shade);
    layer.dispose();
  });
});

// ---------------------------------------------------------------------------
// divergence + glint semantics
// ---------------------------------------------------------------------------

describe('kit/ghosts — divergence and the glint', () => {
  test('a numerically diverged seed freezes at its last finite pose and turns dark red', () => {
    const layer = makeLayer({ category: 'cheetah', maxGhosts: 2 });
    const good = CHEETAH_POSES[1] as Float32Array;
    layer.setGhostFrame(0, good, true);
    layer.setRate(0, 0.9);
    const alive = readInstance(layer, 'head', 0);
    const aliveColor = readColor(layer, 'head', 0);

    const dead = pose9([999, Number.NaN, Number.NaN, 0, 0, 0, 0, 0, 0]);
    layer.setGhostFrame(0, dead, false);
    const frozen = readInstance(layer, 'head', 0);
    const deadColor = readColor(layer, 'head', 0);

    expect(frozen.pos.distanceTo(alive.pos)).toBeLessThan(1e-9); // frozen, not teleported
    expect(deadColor.getHexString()).not.toBe(aliveColor.getHexString());
    expect(deadColor.r).toBeGreaterThan(deadColor.g);
    expect(deadColor.r).toBeGreaterThan(deadColor.b);
    // and it is dark: a death certificate, not a celebration
    expect(luminance(deadColor)).toBeLessThan(luminance(aliveColor));
    layer.dispose();
  });

  test('a non-finite pose freezes even when the caller still believes the seed is alive', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 2 });
    layer.setGhostFrame(0, pose9([4, 6]), true);
    const before = readInstance(layer, 'core', 0);
    layer.setGhostFrame(0, pose9([Number.NaN, Number.POSITIVE_INFINITY]), true);
    const after = readInstance(layer, 'core', 0);
    expect(after.pos.distanceTo(before.pos)).toBeLessThan(1e-9);
    layer.dispose();
  });

  test('an unposed ghost is collapsed to a degenerate transform, so it draws nothing', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 4 });
    layer.setGhostFrame(0, pose9([1, 1]), true);
    for (const bank of layer.banks) {
      expect(linearMagnitude(readMatrix(layer, bank.name, 3))).toBe(0);
      expect(linearMagnitude(readMatrix(layer, bank.name, 0))).toBeGreaterThan(0);
    }
    layer.dispose();
  });

  test('the glint envelope decays to exactly zero at the end of its span', () => {
    expect(ghostGlintEnvelope(0)).toBeCloseTo(1, 12);
    expect(ghostGlintEnvelope(GHOST_GLINT_SECONDS)).toBe(0);
    expect(ghostGlintEnvelope(GHOST_GLINT_SECONDS + 1)).toBe(0);
    expect(ghostGlintEnvelope(-1)).toBe(0);
    let previous = 1;
    for (let k = 1; k <= 60; k++) {
      const v = ghostGlintEnvelope((k / 60) * GHOST_GLINT_SECONDS);
      expect(v).toBeLessThanOrEqual(previous);
      previous = v;
    }
  });

  test('glints are rate-limited per ghost and dead seeds never glint', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 4 });
    layer.setGhostFrame(0, pose9([2, 2]), true);
    layer.setGhostFrame(1, pose9([3, 3]), true);
    layer.setRate(0, 0.4);
    layer.setRate(1, 0.4);
    layer.update(1 / 60, 5);

    const rest = luminance(readColor(layer, 'core', 0));
    layer.glint(0);
    layer.update(1 / 60, 5.02);
    const first = luminance(readColor(layer, 'core', 0));
    expect(first).toBeGreaterThan(rest);

    // a second glint mid-flight does NOT restart the envelope (no strobing)
    layer.glint(0);
    layer.update(1 / 60, 5.04);
    const second = luminance(readColor(layer, 'core', 0));
    expect(second).toBeLessThan(first);

    // ...but once the span is over, the ghost can glint again
    layer.update(1 / 60, 5 + GHOST_GLINT_SECONDS + 0.01);
    layer.glint(0);
    layer.update(1 / 60, 5 + GHOST_GLINT_SECONDS + 0.02);
    expect(luminance(readColor(layer, 'core', 0))).toBeGreaterThan(rest);

    // a numerically diverged seed is inert
    layer.setGhostFrame(1, pose9([3, 3]), false);
    const deadRest = luminance(readColor(layer, 'core', 1));
    layer.glint(1);
    layer.update(1 / 60, 6);
    expect(luminance(readColor(layer, 'core', 1))).toBeCloseTo(deadRest, 12);
    layer.dispose();
  });

  test('a backwards clock (a scrub) retires a glint instead of stranding it', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 2 });
    layer.setGhostFrame(0, pose9([1, 1]), true);
    layer.setRate(0, 0.5);
    layer.update(1 / 60, 20);
    const rest = luminance(readColor(layer, 'core', 0));
    layer.glint(0);
    layer.update(1 / 60, 20.01);
    expect(luminance(readColor(layer, 'core', 0))).toBeGreaterThan(rest);
    layer.update(1 / 60, 3); // scrubbed backwards
    expect(luminance(readColor(layer, 'core', 0))).toBeCloseTo(rest, 9);
    layer.dispose();
  });

  test('out-of-range indices are ignored, not thrown', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 2 });
    expect(() => {
      layer.setGhostFrame(-1, pose9([0, 0]), true);
      layer.setGhostFrame(2, pose9([0, 0]), true);
      layer.setGhostFrame(1.5, pose9([0, 0]), true);
      layer.setRate(9, 1);
      layer.setEmphasis(9, 2);
      layer.glint(9);
      layer.update(1 / 60, 1);
    }).not.toThrow();
    layer.dispose();
  });
});

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

interface Resources {
  geometries: Set<THREE.BufferGeometry>;
  materials: Set<THREE.Material>;
  textures: Set<THREE.Texture>;
}

function collectResources(root: THREE.Object3D): Resources {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) geometries.add(mesh.geometry);
    const material = mesh.material;
    if (!material) return;
    for (const m of Array.isArray(material) ? material : [material]) {
      materials.add(m);
      const map = (m as THREE.MeshBasicMaterial).map;
      if (map) textures.add(map);
    }
  });
  return { geometries, materials, textures };
}

/**
 * three stamps a monotonically increasing id on every Object3D, BufferGeometry and
 * Material it constructs, so one probe of each before and after a workload MEASURES how
 * many were built in between. (`Material.id` is real at runtime but missing from
 * @types/three, hence the cast.)
 */
function probeIds(): { object: number; geometry: number; material: number } {
  return {
    object: new THREE.Object3D().id,
    geometry: new THREE.BufferGeometry().id,
    material: (new THREE.MeshBasicMaterial() as unknown as { id: number }).id,
  };
}

describe('kit/ghosts — lifecycle', () => {
  test('build -> dispose round-trips clean three times, releasing every resource', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      for (let round = 0; round < 3; round++) {
        const layer = makeLayer({ category, maxGhosts: 32 });
        for (let g = 0; g < 32; g++) layer.setGhostFrame(g, pose9([g * 0.1, 0.2, 0.1]), true);
        layer.update(1 / 60, round);

        const resources = collectResources(layer.group);
        expect(resources.geometries.size).toBeGreaterThan(0);
        expect(resources.materials.size).toBeGreaterThan(0);

        const disposed = new Map<object, number>();
        const watch = (r: { dispose(): void }): void => {
          const original = r.dispose.bind(r);
          disposed.set(r, 0);
          r.dispose = (): void => {
            disposed.set(r, (disposed.get(r) ?? 0) + 1);
            original();
          };
        };
        for (const g of resources.geometries) watch(g);
        for (const m of resources.materials) watch(m);
        for (const t of resources.textures) watch(t);

        layer.dispose();
        for (const [resource, count] of disposed) {
          if (count < 1) throw new Error(`${category}: a ${resource.constructor.name} was never disposed`);
        }
        expect(layer.group.children.length).toBe(0);
        expect(layer.draws()).toBe(0);
        // dispose is idempotent — a double teardown must not double-free
        expect(() => layer.dispose()).not.toThrow();
      }
    }
  });

  test('a disposed layer refuses further writes instead of throwing', () => {
    const layer = makeLayer({ category: 'grid', maxGhosts: 4 });
    layer.dispose();
    expect(() => {
      layer.setGhostFrame(0, pose9([1, 1]), true);
      layer.setRate(0, 1);
      layer.setEmphasis(0, 2);
      layer.glint(0);
      layer.update(1 / 60, 1);
    }).not.toThrow();
  });

  test('update() allocates no THREE objects and never reallocates an instance buffer', () => {
    for (const category of ['cheetah', 'grid', 'classic'] as const) {
      const layer = makeLayer({ category, maxGhosts: 128 });
      const poses: Float32Array[] = [];
      for (let g = 0; g < 128; g++) {
        poses.push(pose9([g * 0.03 - 1, 0.2 + g * 0.001, 0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.1]));
        layer.setGhostFrame(g, poses[g] as Float32Array, true);
        layer.setRate(g, g / 128);
      }
      layer.update(1 / 60, 0);

      const buffers = new Map<string, { matrix: ArrayLike<number>; color: ArrayLike<number> | null }>();
      for (const bank of layer.banks) {
        buffers.set(bank.name, {
          matrix: bank.mesh.instanceMatrix.array,
          color: bank.mesh.instanceColor?.array ?? null,
        });
      }

      // A stray `new THREE.Vector3` would not show up in this probe, but a stray mesh,
      // geometry or material — the allocations that actually cost — would.
      const before = probeIds();
      for (let frame = 0; frame < 240; frame++) {
        const t = frame / 60;
        for (let g = 0; g < 128; g++) {
          layer.setGhostFrame(g, poses[g] as Float32Array, g !== 3);
          layer.setRate(g, ((g + frame) % 128) / 128);
          layer.setEmphasis(g, ((g + frame) % 3) as 0 | 1 | 2);
        }
        layer.glint(frame % 128);
        layer.update(1 / 60, t);
      }
      const after = probeIds();
      expect(after.object - before.object).toBe(1);
      expect(after.geometry - before.geometry).toBe(1);
      expect(after.material - before.material).toBe(1);

      for (const bank of layer.banks) {
        const seen = buffers.get(bank.name);
        if (!seen) throw new Error(`bank ${bank.name} vanished`);
        expect(bank.mesh.instanceMatrix.array === seen.matrix).toBe(true);
        expect((bank.mesh.instanceColor?.array ?? null) === seen.color).toBe(true);
      }
      layer.dispose();
    }
  });

  test('makeGhostLayer and the class constructor agree', () => {
    const a = makeGhostLayer({
      category: 'grid',
      envId: 'Chain-cont-v0',
      geometry: gridGeometry(16, 1),
      maxGhosts: 4,
      tokens: TOKENS,
      accent: TOKENS.accents['chain'] ?? 0xf7a8b8,
      tier: 'medium',
    });
    const b = new GhostLayer({
      category: 'grid',
      envId: 'Chain-cont-v0',
      geometry: gridGeometry(16, 1),
      maxGhosts: 4,
      tokens: TOKENS,
      accent: TOKENS.accents['chain'] ?? 0xf7a8b8,
      tier: 'medium',
    });
    expect(a.draws()).toBe(b.draws());
    expect(a.banks.map((x) => x.name)).toEqual(b.banks.map((x) => x.name));
    a.dispose();
    b.dispose();
  });
});

// ---------------------------------------------------------------------------
// density normalization (the co-located-blowout guard)
// ---------------------------------------------------------------------------

describe('GhostLayer density normalization', () => {
  const grid = (): EnvGeometry => ({
    kind: 'grid',
    grid: { w: 11, h: 11, walls: new Uint8Array(121) },
  });

  const lum = (c: THREE.Color): number => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

  const mk = (n: number) =>
    makeGhostLayer({
      category: 'grid',
      envId: 'FourRooms-cont-v0',
      geometry: grid(),
      maxGhosts: n,
      tokens: TOKENS,
      accent: TOKENS.accents['four-rooms'] ?? 0x8fd47a,
      tier: 'medium',
    });

  test('a co-located pack dims each faint emitter by 1/sqrt(k); a lone ghost is untouched', () => {
    const lone = mk(1);
    lone.setGhostFrame(0, new Float32Array([5, 5, 0, 0, 0, 0, 0, 0, 0]), true);
    lone.setRate(0, 0.5);
    lone.update(0.016, 0);
    const loneLum = lum(lone.emitterColor(0, new THREE.Color()));
    lone.dispose();

    const pack = mk(64);
    const pose = new Float32Array([5, 5, 0, 0, 0, 0, 0, 0, 0]);
    for (let i = 0; i < 64; i++) {
      pack.setGhostFrame(i, pose, true);
      pack.setRate(i, 0.5);
    }
    pack.update(0.016, 0);
    const packLum = lum(pack.emitterColor(7, new THREE.Color()));
    expect(packLum).toBeGreaterThan(0);
    expect(packLum).toBeCloseTo(loneLum / 8, 3); // 1/sqrt(64)

    // total summed brightness stays bounded: 64 * (1/8) = 8x one ghost, not 64x
    expect(64 * packLum).toBeLessThan(9 * loneLum);
    pack.dispose();
  });

  test('spread ghosts keep full gain; emphasis exempts a ghost from density dimming', () => {
    const layer = mk(64);
    const pose = new Float32Array(9);
    for (let i = 0; i < 64; i++) {
      pose[0] = i % 8;
      pose[1] = Math.floor(i / 8);
      layer.setGhostFrame(i, pose, true);
      layer.setRate(i, 0.5);
    }
    layer.update(0.016, 0);
    const spreadLum = lum(layer.emitterColor(3, new THREE.Color()));

    // now co-locate everyone; the emphasized one keeps full (higher) gain
    pose[0] = 2; pose[1] = 2;
    for (let i = 0; i < 64; i++) layer.setGhostFrame(i, pose, true);
    layer.setEmphasis(5, 1);
    layer.update(0.016, 0.1);
    const dimmed = lum(layer.emitterColor(3, new THREE.Color()));
    const emphasized = lum(layer.emitterColor(5, new THREE.Color()));
    expect(dimmed).toBeLessThan(spreadLum * 0.2);
    expect(emphasized).toBeGreaterThan(spreadLum); // emphasis-1 gain > emphasis-0, undimmed
    layer.dispose();
  });
});
