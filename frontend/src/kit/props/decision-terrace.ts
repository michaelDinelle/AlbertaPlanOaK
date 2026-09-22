import * as THREE from 'three';
import { makeHeightField } from '../heightfield';
import { makeLandscape, makeSceneTerrain } from '../landscape';
import { modelParts } from '../robot';
import type { PaletteName } from '../tokens';
import type { SceneContext } from '../types';
import { propMaterials } from './materials';

export interface TerraceOptions {
  mood: PaletteName;
  halfX: number;
  halfZ: number;
  ground: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation];
  amplitude?: number;
  paving?: boolean;
  landscape?: Parameters<typeof makeLandscape>[4];
}

export function makeTerrace(ctx: SceneContext, scene: THREE.Scene, opts: TerraceOptions) {
  const family = ctx.environment.family, palette = ctx.tokens.palettes[opts.mood], accent = ctx.tokens.accents[family] ?? ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, `${family}Terrace`, palette, opts.mood, { fogExtent: 90, ...opts.landscape });
  const { world, rig } = land;
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, Math.max(opts.halfX, opts.halfZ) * 2);
  const amplitude = opts.amplitude ?? 1.4;
  const field = makeHeightField({ base: -0.05, court: { halfX: opts.halfX + 0.5, halfZ: opts.halfZ + 0.5, rise: 0.25 }, octaves: [{ wavelength: 27, amplitude, seed: ctx.seed }, { wavelength: 9, amplitude: amplitude * 0.25, seed: ctx.seed ^ 0x7 }] });
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: `${family}TerraceGround`, seed: ctx.seed, colorA: opts.ground[0], colorB: opts.ground[1], segments: 112, radius: 420, height: field.glsl, detail: 'color *= 0.94 + 0.06 * landNoise(p.xz * 1.3);' }, field.heightAt, { roughness: 0.95, rimColor: palette.skyHorizon, rimStrength: 0.14, toonSteps: 4 })));
  world.add(terrain.mesh);
  const m = propMaterials(world), parts = modelParts(world);
  if (opts.paving ?? true) {
    const slab = new THREE.Group(), pitch = 1.1, nx = Math.ceil(opts.halfX * 2 / pitch), nz = Math.ceil(opts.halfZ * 2 / pitch);
    world.add(slab);
    parts.box(slab, m.shale, 0, -0.09, 0, opts.halfX * 2 + 0.3, 0.1, opts.halfZ * 2 + 0.3, 0.02);
    for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
      const wear = 0.9 + ((i * 5 + j * 3) % 4) * 0.02;
      parts.box(slab, (i * 3 + j * 7) % 5 === 3 ? m.shale : m.sandstone, (i + 0.5) * pitch - opts.halfX, -0.03, (j + 0.5) * pitch - opts.halfZ, pitch * 0.92 * wear, 0.06, pitch * 0.92 * wear, 0.015);
    }
    for (const side of [-1, 1]) {
      parts.box(slab, m.brass, side * (opts.halfX + 0.1), 0.0, 0, 0.06, 0.03, opts.halfZ * 2 + 0.3, 0.005);
      parts.box(slab, m.brass, 0, 0.0, side * (opts.halfZ + 0.1), opts.halfX * 2 + 0.3, 0.03, 0.06, 0.005);
    }
    parts.build();
  }
  return { land, world, rig, palette, accent, m, parts, heightAt: field.heightAt };
}

export type Terrace = ReturnType<typeof makeTerrace>;

export const yawToward = (fromX: number, fromZ: number, toX: number, toZ: number): number => Math.atan2(toX - fromX, toZ - fromZ);

export function flameMaterial(color: THREE.ColorRepresentation, intensity: number): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
}
