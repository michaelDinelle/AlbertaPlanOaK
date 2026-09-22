import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { makeMeadow, type MeadowOptions } from '@kit/meadow';
import type { AmbientParticlesOptions } from '@kit/particles';
import { propMaterials } from '@kit/props';
import type { PaletteSkyOptions } from '@kit/sky';
import type { DesignTokens, PaletteName } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { ViewCoverage } from '@kit/view';
import type { WorldBuilder } from '@kit/world';

export interface CourtOptions {
  name: string;
  palette: PaletteName;
  sky?: PaletteSkyOptions;
  weather?: Omit<AmbientParticlesOptions, 'name'>;
  fogExtent?: number;
  sun?: number;
  hemi?: number;
  points?: number;
  pointDistance?: number;
  focus: { x: number; y: number; z: number; radius: number };
  meadow: Omit<MeadowOptions, 'name'>;
}

export function memoryCourt(ctx: SceneContext, scene: THREE.Scene, opts: CourtOptions) {
  const palette = ctx.tokens.palettes[opts.palette], accent = ctx.tokens.accents[ctx.environment.family] ?? ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, opts.name, palette, opts.palette, {
    sky: { sunSize: 0.05, sunHdr: 7, sunGlow: 0.42, sunGlowPower: 20, topSpread: 0.62, bottomSpread: 0.3, ...opts.sky },
    light: { pointCount: opts.points ?? 2, pointDistance: opts.pointDistance ?? 4 },
    fogExtent: opts.fogExtent ?? 140,
    weather: opts.weather,
  });
  const { world, rig } = land;
  rig.sun.intensity = opts.sun ?? 1.9;
  rig.hemi.intensity = opts.hemi ?? 0.6;
  rig.setShadowFocus(opts.focus, opts.focus.radius);
  const meadow = makeMeadow(world, ctx, { name: opts.name, ...opts.meadow });
  const materials = propMaterials(world);
  return {
    land, world, rig, palette, accent, meadow, materials, heightAt: meadow.heightAt,
    setView(coverage: ViewCoverage, camera: THREE.Camera) { land.setView(coverage, camera, meadow); },
  };
}

export type MemoryCourt = ReturnType<typeof memoryCourt>;

export function glyphSet(world: WorldBuilder, tokens: DesignTokens) {
  const shapes = [new THREE.BoxGeometry(0.44, 0.44, 0.44), new THREE.ConeGeometry(0.3, 0.56, 3), new THREE.SphereGeometry(0.27, 18, 12), new THREE.OctahedronGeometry(0.33)].map(shape => world.own(shape));
  const tints = [tokens.cueCyan, tokens.cueMagenta, tokens.celebrationGold, 0x9fea9a];
  return { shapes, tints };
}

export function glyphStones(world: WorldBuilder, glyphs: ReturnType<typeof glyphSet>, count: number, at: (index: number, mesh: THREE.Mesh) => void) {
  return Array.from({ length: count }, (_, index) => {
    const tint = glyphs.tints[index]!;
    const material = world.own(new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.5, roughness: 0.35 }));
    const mesh = world.add(new THREE.Mesh(glyphs.shapes[index]!, material));
    mesh.castShadow = true;
    at(index, mesh);
    return { mesh, material, tint };
  });
}

const scale = new THREE.Vector3();

export function instanced(world: WorldBuilder, geometry: THREE.BufferGeometry, material: THREE.Material, count: number, name: string, shadow = true) {
  const mesh = world.add(world.own(new THREE.InstancedMesh(world.own(geometry), material, count)));
  mesh.name = name;
  mesh.castShadow = mesh.receiveShadow = shadow;
  const matrix = new THREE.Matrix4(), color = new THREE.Color();
  return {
    mesh,
    place(i: number, x: number, y: number, z: number, sx = 1, sy = sx, sz = sx, yaw = 0) {
      matrix.makeRotationY(yaw).scale(scale.set(sx, sy, sz)).setPosition(x, y, z);
      mesh.setMatrixAt(i, matrix);
    },
    paint(i: number, tint: THREE.ColorRepresentation, intensity = 1) { mesh.setColorAt(i, color.set(tint).multiplyScalar(intensity)); },
    commit(bounds = true) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      if (bounds) mesh.computeBoundingSphere();
    },
  };
}

export type Instanced = ReturnType<typeof instanced>;

export function lampRow(world: WorldBuilder, count: number, radius: number, name: string) {
  return instanced(world, new THREE.SphereGeometry(radius, 12, 8), world.own(new THREE.MeshBasicMaterial({ color: 0xffffff })), count, name, false);
}

export function stoneRoad(world: WorldBuilder, material: THREE.Material, count: number, spacing: number, zStart: number, name: string) {
  const size = Math.min(0.86, spacing * 0.9);
  const road = instanced(world, new THREE.BoxGeometry(1, 0.08, 1), material, count, name);
  for (let i = 0; i < count; i++) { road.place(i, 0, 0.03, zStart - i * spacing, size, 1, size * 0.92); road.paint(i, 0xffffff); }
  road.commit();
  return road;
}

export const outcomeGlow = (age: number, rate = 1.1): number => age >= 0 ? Math.exp(-age * rate) : 0;
