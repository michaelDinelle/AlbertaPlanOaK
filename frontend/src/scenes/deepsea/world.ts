import * as THREE from 'three/webgpu';
import { abs, clamp, cos, exp, float, floor, fract, fwidth, length, materialEmissive, max, mix, positionWorld, pow, sin, smoothstep, uniform, vec3 } from 'three/tsl';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { parameter } from '../../data/envs';
import { makeLandscape } from '@kit/landscape';
import type { LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { SkyHandle } from '@kit/sky';
import type { SceneContext } from '@kit/types';
import { flowerGeometry, rockGeometry } from '@kit/vegetation';
import type { WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';
import type { SurfaceSample, ViewCoverage } from '@kit/view';
import { isReef, reefDepthY, reefSkin, shelfTint, terraceShelves, SHELF_THICKNESS, TERRACE_DESCENT, type ReefSkin } from './reef';

export function causticLighting(material: THREE.MeshStandardNodeMaterial, strength = 0.075): { setTime(t: number): void } {
  const clock = uniform(0);
  const p = positionWorld.xz.mul(5.8);
  const causticA = sin(p.x.add(sin(p.y.mul(1.2).add(clock.mul(0.28)))));
  const causticB = sin(p.y.sub(sin(p.x.mul(0.8).sub(clock.mul(0.22)))));
  const caustic = mix(float(0.045), pow(max(0, abs(causticA.add(causticB)).mul(2.4).oneMinus()), 5), smoothstep(0.3, 1.2, length(fwidth(p))).oneMinus());
  material.emissiveNode = materialEmissive.add(vec3(0.42, 0.91, 1).mul(caustic).mul(strength).mul(exp(abs(positionWorld.y).mul(-0.08)).mul(0.3).add(0.7)));
  return { setTime(t) { clock.value = t; } };
}

export function shelfLandscape(grid: GridGeometry, depth: number): { height: (p: THREE.Node<'vec2'>) => THREE.Node<'float'>; detail: (s: SurfaceSample) => THREE.Node<'vec3'>; heightAt(x: number, z: number): number } {
  const span = (grid.h - 1) * CELL;
  const half = grid.w * CELL / 2;
  const drop = TERRACE_DESCENT / (grid.h - 1) * 0.5 + 0.08;
  const smooth = (a: number, b: number, v: number): number => {
    const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  return {
    height(p) {
      const row = clamp(p.y.add(span / 2).div(CELL), 0, grid.h - 1);
      const terrace = float(-TERRACE_DESCENT / (grid.h - 1)).mul(floor(row).add(smoothstep(0.25, 0.75, fract(row)))).sub(drop);
      const along = p.x.add(p.y);
      const coast = smoothstep(CELL * 0.7, CELL * 3.7, p.x.sub(p.y).sub(2.2).sub(sin(along.mul(0.25)).mul(1.2)).sub(sin(along.mul(0.57)).mul(0.7))).oneMinus();
      const shelf = terrace.add(smoothstep(half, half + 12, max(abs(p.x), abs(p.y))).mul(sin(p.x.mul(0.09))).mul(sin(p.y.mul(0.07))).mul(1.2));
      const bed = float(-depth).add(sin(p.x.mul(0.035)).mul(cos(p.y.mul(0.03))).mul(1.5));
      return mix(mix(bed, shelf, coast), shelf.add(1).add(sin(p.y.mul(0.06)).mul(4)), smoothstep(half + 5, half + 28, p.x));
    },
    detail: (s) => mix(s.a, s.b, clamp(s.p.z.add(span / 2).div(span), 0, 1)).mul(sin(s.p.y.mul(3.1).add(sin(s.p.x.mul(0.3).add(s.p.z.mul(0.2))))).mul(0.06).add(0.94)),
    heightAt(x, z) {
      const row = Math.min(grid.h - 1, Math.max(0, (z + span / 2) / CELL));
      const outer = smooth(half, half + 12, Math.max(Math.abs(x), Math.abs(z)));
      const shelf = -TERRACE_DESCENT / (grid.h - 1) * (Math.floor(row) + smooth(0.25, 0.75, row % 1)) - drop + outer * Math.sin(x * 0.09) * Math.sin(z * 0.07) * 1.2;
      const bed = -depth + Math.sin(x * 0.035) * Math.cos(z * 0.03) * 1.5;
      const coast = 1 - smooth(CELL * 0.7, CELL * 3.7, x - z - 2.2 - Math.sin((x + z) * 0.25) * 1.2 - Math.sin((x + z) * 0.57) * 0.7);
      const bank = smooth(half + 5, half + 28, x);
      return (bed + (shelf - bed) * coast) * (1 - bank) + (shelf + 1 + Math.sin(z * 0.06) * 4) * bank;
    },
  };
}

export function coralGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < 7; i++) {
    const angle = -1.25 + i * 0.42;
    const direction = new THREE.Vector3(Math.sin(angle) * 0.65, 0.7 + Math.cos(angle) * 0.2, Math.sin(i * 2.4) * 0.15).normalize();
    const stem = new THREE.CylinderGeometry(0.012, 0.037, 0.6, 5);
    stem.translate(0, 0.3, 0);
    stem.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, direction));
    parts.push({ geometry: stem, color: 0xffffff });
    for (const side of [-1, 1]) {
      const branch = new THREE.CylinderGeometry(0.006, 0.016, 0.23, 4);
      branch.rotateZ(side * 0.55);
      branch.translate(direction.x * 0.35 + side * 0.05, direction.y * 0.35 + 0.1, direction.z * 0.35);
      parts.push({ geometry: branch, color: 0xffffff });
    }
  }
  return mergeColoredParts(parts, 'staghornCoral');
}

export function makeKelpGeometry(): THREE.BufferGeometry {
  const detail = 1;
  const blades = Math.max(3, Math.round(5 * detail));
  const segments = Math.max(2, Math.round(4 * detail));
  const quads = blades * segments;
  const positions = new Float32Array(quads * 6 * 3);
  const colors = new Float32Array(quads * 6 * 3);
  let o = 0;

  const push = (x: number, y: number, z: number, k: number): void => {
    positions[o * 3] = x;
    positions[o * 3 + 1] = y;
    positions[o * 3 + 2] = z;
    colors[o * 3] = k;
    colors[o * 3 + 1] = k;
    colors[o * 3 + 2] = k;
    o++;
  };

  for (let b = 0; b < blades; b++) {
    const yaw = (b / blades) * Math.PI * 2 + ((b * 7) % 5) * 0.19;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const height = .74 * (0.72 + 0.28 * (((b * 3) % 4) / 3));
    const offset = 0.06 + 0.05 * (((b * 5) % 3) / 2);
    for (let s = 0; s < segments; s++) {
      const t0 = s / segments;
      const t1 = (s + 1) / segments;
      const y0 = t0 * height;
      const y1 = t1 * height;
      const w0 = 0.055 * (1 - 0.78 * t0);
      const w1 = 0.055 * (1 - 0.78 * t1);
      const s0 = Math.sin(t0 * 2.4) * 0.055;
      const s1 = Math.sin(t1 * 2.4) * 0.055;
      const k0 = 0.42 + 0.7 * t0;
      const k1 = 0.42 + 0.7 * t1;
      const ax = cos * (offset + s0);
      const az = sin * (offset + s0);
      const bx = cos * (offset + s1);
      const bz = sin * (offset + s1);
      const nx = -sin;
      const nz = cos;
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(ax + nx * w0, y0, az + nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(bx - nx * w1, y1, bz - nz * w1, k1);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'reefKelpClump';
  return g;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  skin: ReefSkin;
  silt: AmbientParticlesHandle;
  rush: AmbientParticlesHandle;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, reach: Uint8Array): WorldHandle {
  const palette = ctx.tokens.palettes.underwater;
  const skin = reefSkin(palette, ctx.tokens);
  const slip = parameter(ctx.environment, 'slip'), half = grid.w * CELL * 0.5;
  const land = makeLandscape(scene, 'sunlitDescent', palette, 'underwater', {
    sky: { topSpread: 0.8, bottomSpread: 0.7 },
    light: { sunDir: [-0.22, 0.95, 0.22], pointCount: 3, pointDistance: 9 },
    fogDensity: 0.012,
    weather: { preset: 'bubbles', count: 220 + Math.round(slip * 4000), seed: ctx.seed, color: skin.bubble, size: 1.1, intensity: 1.3, rise: 0.5 + slip * 6, swirl: 0.2 + slip * 6, bounds: { y: -1.5, halfX: half + 4, halfY: 3, halfZ: half + 4 } },
  });
  const { world: w, sky, rig, fog } = land;
  sky.setColors(0x91ece1, 0x217fa2, 0x05294c);
  sky.add((dir, time) => vec3(0.66, 1, 0.87).mul(pow(max(dir.y, 0), 8)).mul(sin(dir.x.mul(30).add(dir.z.mul(27)).add(time.mul(0.17))).mul(0.16).add(1.25)));
  fog.color.set(0x247b96);
  rig.sun.color.set(0xd6f4ff); rig.sun.intensity = 1.9;
  rig.hemi.color.set(0x7fd0ea); rig.hemi.groundColor.set(0x0e3a5c); rig.hemi.intensity *= 1.4;
  rig.setShadowFocus({ x: 0, y: -1.75, z: 0 }, 12);
  const tint = new THREE.Color();
  const field = w.own(buildCellField(grid, {
    floorColor: 0xffffff,
    wallColor: 0xffffff,
    floorFilter: (x, y) => isReef(reach, grid, x, y),
    floorTint: (x, y) => shelfTint(skin, grid, x, y, tint).clone(),
    floorInset: 0.006,
    floorThickness: SHELF_THICKNESS,
    jitter: 0.018,
    rimColor: 0xacefe0,
    rimStrength: 0.24,
    seed: ctx.seed,
    name: 'reefShelves',
  }));
  terraceShelves(field, grid);
  w.time(field);
  w.time(causticLighting(field.floorMaterial));
  w.add(field.group);
  const shelf = shelfLandscape(grid, 6.5);
  const heightAt = shelf.heightAt;
  const floor = land.surface({ follow: false,
    name: 'reefEscarpment', colorA: skin.shelfShallow, colorB: skin.shelfDeep, segments: 112,
    height: shelf.height,
    detail: shelf.detail,
  });
  w.time(causticLighting(floor.material, 0.018));
  const outcropGeometry = rockGeometry(2.7, 1);
  const rockMaterial = oakMaterial({ color: 0xffffff, flatShading: true, roughness: 0.9, rimColor: 0xacefe0, rimStrength: 0.24 });
  w.time(causticLighting(rockMaterial, 0.028));
  land.scatter({
    name: 'reefOutcrops', seed: ctx.seed, tileSize: 18, perTile: 2,
    geometry: outcropGeometry, material: rockMaterial, colorA: 0x325e62, colorB: 0x63938d,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 2; i++) {
        const x = (tx + rng()) * 18;
        const z = (tz + rng()) * 18;
        if (Math.abs(x) < half + 7 && Math.abs(z) < half + 7) continue;
        write(x, heightAt(x, z) - 0.15, z, rng() * 6.28, 0.8 + rng() * 1.6, 0.8 + rng() * 0.6, rng());
      }
    },
  });
  [
    { geometry: makeKelpGeometry(), color: skin.kelp, count: 72, scale: 0.85, name: 'reefKelp' },
    { geometry: flowerGeometry(5), color: skin.anemone, count: 48, scale: 1, name: 'reefAnemones' },
  ].forEach((spec, index) => {
    const geometry = spec.geometry;
    const material = oakMaterial({ color: spec.color, vertexColors: true, flatShading: true, roughness: 0.8, side: THREE.DoubleSide, rimColor: 0xacefe0, rimStrength: 0.22, wind: { amplitude: 0.025, frequency: 0.65, height: 0.74 } });
    land.scatter({
      name: spec.name, seed: ctx.seed ^ (0x7ee1 + index), tileSize: 12, perTile: spec.count,
      geometry, material, colorA: 0x9ac2b0, colorB: 0xd5dfbd,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < spec.count; i++) {
          const x = (tx + rng()) * 12;
          const z = (tz + rng()) * 12;
          const column = Math.round(x / CELL + (grid.w - 1) / 2);
          const row = Math.round(z / CELL + (grid.h - 1) / 2);
          const onReef = isReef(reach, grid, column, row);
          const centerX = (column - (grid.w - 1) / 2) * CELL;
          const centerZ = (row - (grid.h - 1) / 2) * CELL;
          if (onReef && Math.hypot(x - centerX, z - centerZ) < (column === grid.w - 1 && row === grid.h - 1 ? 0.8 : 0.24)) continue;
          write(x, onReef ? reefDepthY(row, grid) : heightAt(x, z), z, rng() * 6.28, spec.scale * (0.8 + rng() * 0.6), 0.8 + rng() * 0.4, rng());
        }
      },
    });
  });
  const silt = w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 64, bounds: { halfX: 0.35, halfY: 0.2, halfZ: 0.35 },
    seed: ctx.seed, color: skin.silt, size: 0.9, intensity: 1.1, rise: -0.07, name: 'reefSilt',
  })));
  const rush = w.time(w.own(makeAmbientParticles({
    preset: 'bubbles', count: 100, bounds: { halfX: 0.26, halfY: 3.6, halfZ: 0.26 },
    seed: ctx.seed ^ 0x2c58, color: skin.bubble, size: 1.5, intensity: 2, rise: 2.4, name: 'resurfaceBubbles',
  })));
  silt.setDensity(0);
  rush.setDensity(0);
  w.add(silt.points);
  w.add(rush.points);
  return w.seal({
    sky, rig, field, skin, silt, rush,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) { land.setView(coverage, camera); },
  });
}

export function treasureCell(grid: GridGeometry): { c: number; r: number } {
  return { c: grid.goal?.x ?? grid.w - 1, r: grid.goal?.y ?? grid.h - 1 };
}
