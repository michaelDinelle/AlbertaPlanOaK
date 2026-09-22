import * as THREE from 'three';
import { makeHeightField } from '../heightfield';
import { aspenTree, makeSceneTerrain, type Landscape } from '../landscape';
import type { Palette } from '../tokens';
import type { SceneContext } from '../types';

export function groves(land: Landscape, ctx: SceneContext, foliage: string[], halfX: number, halfZ: number, heightAt: (x: number, z: number) => number, perTile = 5) {
  const tree = aspenTree(land.world, foliage);
  return land.scatter({
    name: `${ctx.environment.family}Groves`, geometry: tree.geometry, material: tree.material, depth: tree.depth, tileSize: 10, perTile, seed: ctx.seed, colorA: 0xffffff, colorB: 0xd6d6d6, castShadow: true,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < perTile; i++) {
        const x = (tx + rng()) * 10, z = (tz + rng()) * 10;
        if (Math.abs(x) < halfX && Math.abs(z) < halfZ) continue;
        const grove = 0.5 + 0.5 * Math.sin(x * 0.31 + Math.cos(z * 0.27) * 1.7);
        if (rng() > 0.25 + grove * 0.75) continue;
        write(x, heightAt(x, z) - 0.08, z, rng() * Math.PI * 2, 0.75 + rng() * 0.6, 0.9 + rng() * 0.4, rng());
      }
    },
  });
}

export interface CourtOptions {
  name: string;
  seed: number;
  halfX: number;
  halfZ: number;
  colorA: THREE.ColorRepresentation;
  colorB: THREE.ColorRepresentation;
  palette: Palette;
  rise?: number;
  amplitude?: number;
  snow?: number;
}

export function court(land: Landscape, options: CourtOptions) {
  const amplitude = options.amplitude ?? 1, seed = options.seed >>> 0;
  const field = makeHeightField({ base: -0.05, court: { halfX: options.halfX, halfZ: options.halfZ, rise: options.rise ?? 0.25 }, octaves: [{ wavelength: 27, amplitude: 1.4 * amplitude, seed }, { wavelength: 9, amplitude: 0.35 * amplitude, seed: seed ^ 0x7 }] });
  const snow = options.snow ?? 0;
  const detail = snow > 0
    ? `float drift = smoothstep(-0.35, 0.45, p.y + 0.25 + (landNoise(p.xz * 0.7) - 0.5) * 0.5); color = mix(color, vec3(0.90, 0.93, 0.99), drift * ${snow.toFixed(3)}); color *= 0.96 + 0.04 * landNoise(p.xz * 1.9);`
    : 'color *= 0.94 + 0.06 * landNoise(p.xz * 1.3);';
  const terrain = land.world.own(land.world.time(makeSceneTerrain(
    { follow: false, name: `${options.name}Ground`, seed, colorA: options.colorA, colorB: options.colorB, segments: 112, radius: 420, height: field.glsl, detail },
    field.heightAt,
    { roughness: 0.95, rimColor: options.palette.skyHorizon, rimStrength: 0.14, toonSteps: 4 },
  )));
  land.world.add(terrain.mesh);
  return { terrain, heightAt: field.heightAt };
}
