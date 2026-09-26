import * as THREE from 'three/webgpu';
import { alpineWorld, type AlpineWorld } from '@kit/alpine';
import { terrainNoise as noise } from '@kit/mountains';
import { makeAmbientParticles } from '@kit/particles';
import type { SceneContext } from '@kit/types';

export const WATER_LEVEL = -1.85;
const smooth = THREE.MathUtils.smoothstep;

function riverChannel(z: number) {
  const center = Math.sin(z * 0.016) * 9 + Math.sin(z * 0.047) * 2;
  const width = 5.5 + smooth(Math.abs(z), 25, 190) * 6 + noise(z * 0.03, 5) * smooth(Math.abs(z), 0, 80) * 2;
  return { center, width };
}

export function riverHeight(x: number, z: number): number {
  const { center, width } = riverChannel(z);
  const shore = Math.abs(x - center) - width;
  const bank = smooth(shore, -1, 3.5);
  const hills = smooth(shore, 6, 45) * (5 + noise(x * 0.024 + 7, z * 0.018) ** 2 * 19);
  const foothills = smooth(Math.abs(z), 160, 850) * smooth(shore, 12, 85) * (15 + 54 * noise(x * 0.006, z * 0.008) ** 2);
  return WATER_LEVEL - 1.4 + bank * 3.5 + hills + foothills + bank * (noise(x * 0.22, z * 0.22) - 0.5) * 0.16;
}

export type WorldHandle = AlpineWorld;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  return alpineWorld(ctx, scene, {
    name: 'albertaRiver', palette: ctx.tokens.palettes.dusk, mood: 'dusk',
    sky: { sunDir: [-0.45, 0.24, 0.86], sunSize: 0.022, sunHdr: 1.7, sunGlow: 0.18, topSpread: 0.3, stars: { intensity: 1.6, density: 160, threshold: 0.976, size: 0.07, color: 0xffecd7 } },
    skyColors: [0x303a76, 0xffb780, 0xc49190], fog: 0xd5a89c, fogDensity: 0.00024,
    sun: { color: 0xffc991, intensity: 2.4 }, hemi: { sky: 0xb8c9eb, ground: 0x6d6958, intensity: 1.8 },
    shadow: { focus: [0, 0.3, 0], radius: 13 },
    heightAt: riverHeight, waterLevel: WATER_LEVEL,
    mountains: { rotation: -0.12, position: [-180, -20, 0], name: 'bowValleyRockies' },
    water: { size: [320, 4096], segments: 256, sky: 0xd5a89c, amplitude: 0.055, waveScale: 1.65, speed: 1.2, depthFalloff: 1.6, caustics: 0.045, colors: [0x65bfaa, 0x147c79], specular: 0.22 },
    fluid: { bounds: [-42, -96, 84, 192], flow: [0, 1.1] },
    seedStream: [9, 31],
    trees(rng, place) {
      let count = 0;
      for (let i = 0; i < 1500 && count < 560; i++) {
        const near = i % 3 !== 0, z = near ? rng() * 290 - 100 : -rng() * 850 - 160;
        const { center, width } = riverChannel(z);
        const x = near ? center + (i % 2 ? -1 : 1) * (width + 7 + rng() ** 1.5 * 34) : (rng() - 0.5) * 750;
        const y = riverHeight(x, z);
        if (y < 0.24 || (Math.abs(x) < 24 && Math.abs(z) < 15) || noise(x * 0.03, z * 0.03) < 0.28) continue;
        const scale = 0.72 + rng() * 1.15;
        place(x, y - 0.16, z, [scale * 1.06, scale, scale * 1.06], rng() * Math.PI * 2);
        count++;
      }
    },
    stones(rng, place) {
      let count = 0;
      for (let i = 0; i < 1500 && count < 200; i++) {
        const x = (rng() - 0.5) * 135, z = (rng() - 0.5) * 170, y = riverHeight(x, z);
        if (y < WATER_LEVEL - 0.65 || y > 4 || (Math.abs(x) < 13 && Math.abs(z) < 1.7)) continue;
        const size = 0.3 + rng() ** 2 * 2.5, radius = size * 0.46;
        const bed = Math.min(y, riverHeight(x - radius, z), riverHeight(x + radius, z), riverHeight(x, z - radius), riverHeight(x, z + radius));
        place(x, bed - 0.1, z, [size * 1.15, size, size * 1.15], rng() * Math.PI * 2);
        count++;
      }
    },
    grass(rng, place) {
      for (let i = 0; i < 6500; i++) {
        const x = (rng() - 0.5) * 150, z = (rng() - 0.5) * 180, y = riverHeight(x, z);
        if (y < WATER_LEVEL + 0.7 || y > 7 || (Math.abs(x) < 18 && Math.abs(z) < 1.8)) continue;
        const scale = 0.65 + rng() * 0.9;
        place(x, y, z, [scale, scale, scale], rng() * Math.PI * 2);
      }
    },
    grassColor: 0x8ca36b,
    extras(world) {
      const fireflies = world.time(world.own(makeAmbientParticles({ preset: 'fireflies', count: 72, seed: ctx.seed, color: 0xffe6a0, size: 1.3, intensity: 1.8, bounds: { y: 1.1, halfX: 13, halfY: 0.6, halfZ: 10 }, name: 'riverbankFireflies' })));
      const positions = fireflies.points.geometry.getAttribute('aPosition');
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i), { center, width } = riverChannel(positions.getZ(i));
        positions.setX(i, center + Math.sign(x) * (width + 2 + Math.abs(x) * 0.25));
      }
      world.add(fireflies.points);
    },
  });
}
