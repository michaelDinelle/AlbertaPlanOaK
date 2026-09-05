import * as THREE from 'three';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { makeWater } from '@kit/water';
import { makeWorld, type WorldCore } from '@kit/world';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L } from '@sim/envs/delayedchain';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import { extendWater } from '../half-cheetah/world';
import { grassClumpGeometry } from '@kit/vegetation';
import { buildAurora, type AuroraHandle } from '@kit/aurora';

export const LANTERN_Z = -0.62;
export const LAMP_REACH = 0.3;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== DELAYEDCHAIN_L || grid.h !== 1) throw new Error('Delayed Chain grid is missing or invalid');
  return grid;
}

export function goalCellOf(grid: GridGeometry): number {
  return grid.goal?.x ?? grid.w - 1;
}

export interface MoorSkin {
  stone: THREE.Color;
  stoneDark: THREE.Color;
  moss: THREE.Color;
  iron: THREE.Color;
  ironLight: THREE.Color;
  flame: THREE.Color;
}

export function moorSkin(palette: Palette, tokens: typeof TOKENS): MoorSkin {
  const moon = palette.sun;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundB;

  return {
    stone: blendTokens(palette.groundB, moon, 0.26),
    stoneDark: shade(blendTokens(palette.groundB, palette.skyHorizon, 0.4), 0.24),
    moss: shade(blendTokens(palette.groundA, moss, 0.34), 0.3),
    iron: shade(blendTokens(palette.skyTop, palette.groundB, 0.42), 0.28),
    ironLight: blendTokens(shade(blendTokens(palette.skyTop, palette.groundB, 0.42), 0.28), palette.hemiSky, 0.4),
    flame: blendTokens(tokens.celebrationGold, moon, 0.16),
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  skin: MoorSkin;
  aurora: AuroraHandle;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const w = makeWorld('moonlitMoor');
  const palette = ctx.tokens.palettes.night;
  const skin = moorSkin(palette, ctx.tokens);
  const sky = w.time(w.own(skyFromPalette(palette, 'night', {
    sunDir: [-0.52, 0.62, 0.59], sunSize: 0.022, sunHdr: 3.2, topSpread: 0.7,
    stars: { intensity: 2.5, density: 190, threshold: 0.976, size: 0.05 },
  })));
  sky.setColors(0x071426, 0x314f60, 0x162938);
  w.add(sky.mesh);
  const aurora = w.time(w.own(buildAurora(sky, 0x74d9bc)));
  const rig = w.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 4, pointDistance: 9 }));
  rig.hemi.intensity = 1.35;
  rig.sun.intensity = 1.6;
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, grid.w * CELL * 0.6);
  w.add(rig.group);
  const fog = new THREE.FogExp2(0x304950, 0.005);
  scene.fog = fog;
  const half = grid.w * CELL * 0.5;
  const heightAt = (x: number, z: number): number => {
    const route = 1 - smoothstep01((Math.abs(x) - half) / 4);
    const outside = 1 - route * (1 - smoothstep01((Math.abs(z) - 1) / 5));
    const mire = (z - 10 - Math.sin(x * 0.01) * 5) / 4.2;
    return -0.05 + outside * (1.4 + Math.sin(x * 0.023) * 0.45 + Math.sin(z * 0.029) * 0.5 + Math.sin(x * 0.017 + z * 0.024) * 0.35 - 2.1 * Math.exp(-mire * mire));
  };
  const groundHeight = `float route = 1.-smoothstep(${half.toFixed(3)},${(half + 4).toFixed(3)},abs(p.x)); float outside = 1.-route*(1.-smoothstep(1.,6.,abs(p.y))); float mire = (p.y-10.-sin(p.x*.01)*5.)/4.2; return -.05 + outside*(1.4 + sin(p.x*.023)*.45 + sin(p.y*.029)*.5 + sin(p.x*.017+p.y*.024)*.35 - 2.1*exp(-mire*mire));`;
  const floor = w.time(w.own(makeContinuousSurface({
    name: 'rollingHeatherMoor', colorA: 0x607d78, colorB: 0x3e575f, segments: 112,
    height: groundHeight,
    detail: 'float heather = smoothstep(.42,.64,landNoise(p.xz*.065)); color = mix(color,vec3(.28,.2,.32),heather*.32);',
  })));
  w.add(floor.mesh);
  const water = w.time(w.own(makeWater({
    size: 2, segments: 2, palette, level: -0.16,
    shallow: 0x234050, deep: 0x102638, sky: 0x598079, caustic: 0x65ad9c,
    amplitude: 0.01, waveScale: 0.5, speed: 0.16, fresnel: 0.35, caustics: 0.03,
    depthFalloff: 0.7, opacity: 0.94, sunDir: rig.sunDirection, name: 'moorPools',
  })));
  water.setColors(0x234050, 0x102638);
  w.add(water.mesh);
  water.material.fragmentShader = `float moorBed(vec2 p) {${groundHeight}}\n${water.material.fragmentShader}`;
  const field = w.own(buildCellField(grid, {
    floorColor: 0x90a7a1, wallColor: 0x90a7a1, floorInset: 0.065, floorThickness: 0.15, jitter: 0.045,
    floorTint: x => new THREE.Color().setScalar(0.76 + 0.18 * Math.sin(x * 0.62)),
    rimColor: 0x8dbeac, rimStrength: 0.22, seed: ctx.seed, name: 'moonlitFlagstoneRoad',
  }));
  w.time(field);
  w.add(field.group);
  const vegetationMaterials: OakMaterial[] = [];
  const heather = new THREE.IcosahedronGeometry(0.55, 1);
  heather.scale(1, 0.25, 1);
  heather.translate(0, 0.14, 0);
  const groundCover = [
    { geometry: grassClumpGeometry(9, 0.25, 0.07), count: 12, scale: 0.9, a: 0x879786, b: 0x63817a, name: 'moorGrass' },
    { geometry: heather, count: 24, scale: 1.8, a: 0x897b99, b: 0x646e84, name: 'moorHeather' },
  ].map((spec, index) => {
    const geometry = w.own(spec.geometry);
    const material = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, side: THREE.DoubleSide, roughness: 0.9, wind: { amplitude: 0.045, frequency: 1.15, height: 0.45 }, rimColor: 0x8aafad, rimStrength: 0.12 }));
    vegetationMaterials.push(material);
    const bank = w.own(makeSceneryBank({
      name: spec.name, seed: ctx.seed + index * 717, tileSize: 6, perTile: spec.count,
      geometry, material, colorA: spec.a, colorB: spec.b,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < spec.count; i++) {
          const x = (tx + rng()) * 6;
          const z = (tz + rng()) * 6;
          const y = heightAt(x, z);
          const patch = .5 + .5 * Math.sin(x * 0.19 + Math.sin(z * 0.14) * 2.1);
          if ((Math.abs(x) < half + 1 && Math.abs(z) < (index === 1 ? 2 : 1.05)) || y < -0.1 || (index === 1 && patch < 0.52)) continue;
          write(x, y, z, rng() * 6.28, spec.scale * (0.65 + rng() * 0.7), 0.8 + rng() * 0.55, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const motes = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'fireflies', count: 100, bounds: { y: 0.75, halfX: grid.w * CELL * 0.6, halfY: 0.7, halfZ: 4 },
    seed: ctx.seed ^ 0x727e, color: 0xc6d6b2, size: 1.5, intensity: 1.4, name: 'moorFireflies',
  }))));
  w.add(motes.points);
  return w.seal({
    sky, rig, field, skin, aurora,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      floor.setView(coverage);
      extendWater(water.mesh, coverage, 2, 2, '-0.16 - moorBed(vWorld.xz)');
      for (const bank of groundCover) bank.setView(coverage);
      fog.density = 2.8 / coverage.far;
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
    },
  }, { time(t) { for (const material of vegetationMaterials) oakUniforms(material)?.setTime(t); } });
}

export function propMaterial(palette: Palette, name: string): OakMaterial {
  return oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.9, rimColor: palette.hemiSky, rimStrength: 0.3, name });
}

export function setPropTime(material: THREE.Material | null, t: number): void {
  oakUniforms(material)?.setTime(t);
}
