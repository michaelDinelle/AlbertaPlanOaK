import * as THREE from 'three';
import { alpineWorld, type AlpineWorld } from '@kit/alpine';
import { makeHeightField } from '@kit/heightfield';
import { blendTokens } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes } from '@kit/particles';
import { lamp, propMaterials } from '@kit/props';
import type { SceneContext } from '@kit/types';
import { AB_L1, AB_L2 } from '@sim/envs/acrobot';
import type { EnvGeometry } from '@sim/types';
import { parameter } from '../../data/envs';

export const WATER_Y = -(AB_L1 + AB_L2) - 0.42;
export const PIER_Z = -0.7;
export const PIER_X = 2.6;
export const DECK_Y = 0.6;
const smooth = THREE.MathUtils.smoothstep;

const octave = (wavelength: number, seed: number) => {
  const field = makeHeightField({ octaves: [{ wavelength, amplitude: 0.5, seed }] });
  return (x: number, z: number): number => field.heightAt(x, z) + 0.5;
};
const shoreNoise = octave(18, 0x51), hillNoise = octave(125, 0x52), farNoise = octave(140, 0x53), rippleNoise = octave(3.3, 0x54), groveNoise = octave(40, 0x55);

export function pivotAnchor(geometry: EnvGeometry): readonly [number, number, number] {
  const anchor = geometry.classic?.anchors?.pivot;
  if (!anchor) throw new Error('Acrobot pivot anchor required');
  return anchor;
}

export function lakeHeight(x: number, z: number): number {
  const radius = Math.hypot((x + 8 + Math.sin(z * 0.018) * 7) / 92, (z + 105) / 129);
  const shore = (radius - 1) * 92 + (shoreNoise(x, z) - 0.5) * 6;
  const bank = Math.min(7, Math.max(-14, shore * 0.16));
  const hills = smooth(shore, 6, 120) * (3 + 18 * hillNoise(x + 8, z) ** 2);
  const headland = Math.exp(-Math.pow((z - 8) / 65, 2)) * (7.2 * Math.exp(-Math.pow((x + 11) / 6, 2)) + 6.2 * Math.exp(-Math.pow((x - 17) / 9, 2)));
  const foothills = smooth(-z, 220, 650) * (12 + 34 * farNoise(x + 30, z) ** 2);
  return WATER_Y + bank + hills + headland + foothills + smooth(shore, -3, 4) * (rippleNoise(x, z) - 0.5) * 0.32;
}

export interface WorldHandle extends AlpineWorld {
  lamps: ReturnType<typeof lamp>[];
  rimLight: THREE.PointLight;
  accentLight: THREE.PointLight;
  positionOnly: boolean;
  slip: number;
}

export async function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  const accent = ctx.tokens.accents['acrobot'] ?? ctx.tokens.celebrationGold, palette = ctx.tokens.palettes.night;
  const positionOnly = parameter(ctx.environment, 'positionOnly') !== 0, slip = Math.fround(parameter(ctx.environment, 'slip'));
  const lamps: ReturnType<typeof lamp>[] = [];
  const rimLight = new THREE.PointLight(0xffc98a, 2.4, 9, 2), accentLight = new THREE.PointLight(accent, 1.8, 7, 2);
  const world = await alpineWorld(ctx, scene, {
    name: 'alpineLake', palette, mood: 'night',
    sky: { sunDir: [-0.6, 0.48, 0.64], sunSize: 0.016, sunHdr: 2, sunGlow: 0.045, topSpread: 0.25, stars: { intensity: 2.8, density: 180, threshold: 0.974, size: 0.075, color: 0xdbeaff } },
    skyColors: [0x101c48, 0x718bae, 0x455977], fog: 0x718bae, fogDensity: 0.00018,
    sun: { color: 0xc3dcff, intensity: 1.7 }, hemi: { sky: 0xa8c9e9, ground: 0x485b61, intensity: 1.8 },
    shadow: { focus: [0, -1, 0], radius: 7 }, aurora: { color: 0x71ebd2, intensity: 0.8, gain: 1.15 },
    heightAt: lakeHeight, waterLevel: WATER_Y,
    mountains: { region: 'moraine-lake', rotation: Math.PI - 0.35, position: [490, 10, -2150], name: 'canadianRockies' },
    water: { size: 1024, segments: 256, sky: 0x718bae, amplitude: 0.045, waveScale: 1.3, speed: 0.85, depthFalloff: 3.5, caustics: 0.025, colors: [0x479baf, 0x15466f], specular: 0.25 },
    fluid: { bounds: [-128, -256, 256, 288] },
    seedStream: [4, 17],
    trees(rng, place) {
      let count = 0;
      for (let i = 0; i < 1700 && count < 800; i++) {
        const angle = rng() * Math.PI * 2, radius = 1.02 + rng() ** 1.7 * 0.9, near = i < 160;
        const x = near ? (i % 2 ? -12 : 19) + (rng() - 0.5) * 12 : i > 1000 ? (rng() - 0.5) * 900 : Math.cos(angle) * 100 * radius - 8;
        const z = near ? -25 + rng() * 52 : i > 1000 ? -280 - rng() * 450 : Math.sin(angle) * 137 * radius - 105;
        const height = lakeHeight(x, z);
        if (height < WATER_Y + 0.85 || height > 85 || Math.hypot(x, z) < 9 || (!near && groveNoise(x, z) < 0.25)) continue;
        const yaw = rng() * Math.PI * 2, lean = (rng() - 0.5) * 0.04, scale = (near ? 0.8 : 1.2) + rng() * 1.2;
        place(x, height - 0.2, z, [scale * 1.15, scale, scale * 1.15], yaw, [0, lean]);
        count++;
      }
    },
    stones(rng, place) {
      let count = 0;
      for (let i = 0; i < 1500 && count < 260; i++) {
        const angle = rng() * Math.PI * 2, radius = rng() ** 1.9 * 180;
        const x = Math.cos(angle) * radius, z = Math.sin(angle) * radius, height = lakeHeight(x, z);
        if (height < WATER_Y - 2.8 || height > WATER_Y + 5) continue;
        const scale = 0.25 + rng() ** 2 * 4, tiltX = (rng() - 0.5) * 0.3, yaw = rng() * Math.PI * 2, tiltZ = (rng() - 0.5) * 0.3, squash = 0.7 + rng() * 0.5;
        place(x, height - 0.09 * scale, z, [scale, scale * squash, scale], yaw, [tiltX, tiltZ]);
        count++;
      }
    },
    grass(rng, place) {
      for (let i = 0; i < 5200; i++) {
        const x = (rng() - 0.5) * 150, z = (rng() - 0.5) * 150, height = lakeHeight(x, z);
        if (height < WATER_Y + 0.3 || height > WATER_Y + 6) continue;
        const yaw = rng() * Math.PI * 2, scale = 0.7 + rng() * 1.1;
        place(x, height, z, [scale, scale, scale], yaw);
      }
    },
    grassColor: 0x87a174,
    extras(w) {
      const materials = propMaterials(w);
      for (const side of [-1, 1]) {
        const l = lamp(w, blendTokens(accent, 0xffc98a, 0.35), 0.95, materials);
        l.group.position.set(side * (PIER_X + 0.28), DECK_Y, PIER_Z - 0.05);
        l.group.scale.setScalar(0.55);
        w.add(l.group);
        l.lit(positionOnly ? 0 : 0.7);
        lamps.push(l);
      }
      rimLight.position.set(1.8, 1.6, -3.2);
      accentLight.position.set(-1.4, WATER_Y + 0.5, 1.6);
      w.add(rimLight); w.add(accentLight);
      w.own({ dispose() { rimLight.dispose(); accentLight.dispose(); } });
      const mist = makeMistPlanes({ size: [26, 16], color: blendTokens(0x718bae, 0xa8c9e9, 0.45), seed: ctx.seed, layers: 3, baseY: WATER_Y + 0.05, spread: 0.4, opacity: 0.055, speed: 0.007, scale: 2.4, nearFade: 3, name: 'lakeMist' });
      mist.group.position.set(0, 0, -3);
      w.add(mist.group); w.time(w.own(mist));
      const snow = makeAmbientParticles({ preset: 'snow', count: 520, bounds: { x: 0, y: 1.2, z: -1.5, halfX: 8, halfY: 3.2, halfZ: 6 }, seed: ctx.seed ^ 0x51ab, color: 0xeef4ff, size: 0.25, intensity: 1.1, opacity: 0.7, rise: -0.32, swirl: 0.12, name: 'lakeSnow' });
      w.add(snow.points); w.size(w.time(w.own(snow)));
    },
  });
  return { ...world, lamps, rimLight, accentLight, positionOnly, slip };
}
