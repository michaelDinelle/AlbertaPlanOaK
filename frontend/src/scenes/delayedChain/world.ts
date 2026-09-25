import * as THREE from 'three/webgpu';
import { abs, exp, float, mix, sin, smoothstep, vec3 } from 'three/tsl';
import { buildAurora, type AuroraHandle } from '@kit/aurora';
import { CELL, buildCellField } from '@kit/grid';
import { makeLandscape } from '@kit/landscape';
import type { LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { grassClumpGeometry } from '@kit/vegetation';
import type { ViewCoverage } from '@kit/view';
import { extendWater, makeWater } from '@kit/water';
import type { WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';

type N<T extends string> = THREE.Node<T>;

export const LANTERN_Z = -0.62;
export const LAMP_REACH = 0.3;
const POOL_LEVEL = -0.16;

export interface MoorSkin {
  stone: THREE.Color;
  stoneDark: THREE.Color;
  moss: THREE.Color;
  iron: THREE.Color;
  ironLight: THREE.Color;
  flame: THREE.Color;
}

function moorSkin(palette: Palette): MoorSkin {
  const iron = shade(blendTokens(palette.skyTop, palette.groundB, 0.42), 0.28);
  return {
    stone: blendTokens(palette.groundB, palette.sun, 0.26),
    stoneDark: shade(blendTokens(palette.groundB, palette.skyHorizon, 0.4), 0.24),
    moss: shade(blendTokens(palette.groundA, TOKENS.accents.ProceduralMaze!, 0.34), 0.3),
    iron,
    ironLight: blendTokens(iron, palette.hemiSky, 0.4),
    flame: blendTokens(TOKENS.celebrationGold, palette.sun, 0.16),
  };
}

export interface WorldHandle extends WorldCore {
  rig: LightRig;
  skin: MoorSkin;
  aurora: AuroraHandle;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette = ctx.tokens.palettes.night;
  const land = makeLandscape(scene, 'moonlitMoor', palette, 'night', {
    sky: { sunDir: [-0.52, 0.62, 0.59], sunSize: 0.022, sunHdr: 3.2, topSpread: 0.7, stars: { intensity: 2.5, density: 190, threshold: 0.976, size: 0.05 } },
    light: { pointCount: 4, pointDistance: 9 },
  });
  const { world: w, sky, rig, fog } = land;
  sky.setColors(0x071426, 0x314f60, 0x162938);
  const aurora = w.time(w.own(buildAurora(sky, 0x74d9bc)));
  rig.hemi.intensity = 1.35;
  rig.sun.intensity = 1.6;
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, grid.w * CELL * 0.6);
  fog.color.set(0x304950);

  const half = grid.w * CELL * 0.5;
  const heightAt = (x: number, z: number): number => {
    const route = 1 - smoothstep01((Math.abs(x) - half) / 4);
    const outside = 1 - route * (1 - smoothstep01((Math.abs(z) - 1) / 5));
    const mire = (z - 10 - Math.sin(x * 0.01) * 5) / 4.2;
    return -0.05 + outside * (1.4 + Math.sin(x * 0.023) * 0.45 + Math.sin(z * 0.029) * 0.5 + Math.sin(x * 0.017 + z * 0.024) * 0.35 - 2.1 * Math.exp(-mire * mire));
  };
  const moorHeight = (p: N<'vec2'>): N<'float'> => {
    const route = smoothstep(half, half + 4, abs(p.x)).oneMinus();
    const outside = route.mul(smoothstep(1, 6, abs(p.y)).oneMinus()).oneMinus();
    const mire = p.y.sub(10).sub(sin(p.x.mul(0.01)).mul(5)).div(4.2);
    const rolls = sin(p.x.mul(0.023)).mul(0.45).add(sin(p.y.mul(0.029)).mul(0.5)).add(sin(p.x.mul(0.017).add(p.y.mul(0.024))).mul(0.35));
    return outside.mul(rolls.add(1.4).sub(exp(mire.mul(mire).negate()).mul(2.1))).sub(0.05);
  };
  const poolDepth = (p: N<'vec2'>): N<'float'> => float(POOL_LEVEL).sub(moorHeight(p));

  land.surface({
    name: 'rollingHeatherMoor', colorA: 0x607d78, colorB: 0x3e575f, segments: 112, height: moorHeight,
    detail: (s) => mix(s.color, vec3(0.28, 0.2, 0.32), smoothstep(0.42, 0.64, s.noise(s.p.xz.mul(0.065))).mul(0.32)),
  });
  const pools = w.time(w.own(makeWater({
    size: 2, segments: 2, palette, level: POOL_LEVEL,
    shallow: 0x234050, deep: 0x102638, sky: 0x598079, caustic: 0x65ad9c,
    amplitude: 0.01, waveScale: 0.5, speed: 0.16, fresnel: 0.35, caustics: 0.03,
    depthFalloff: 0.7, opacity: 0.94, sunDir: rig.sunDirection, name: 'moorPools',
  })));
  pools.setColors(0x234050, 0x102638);
  w.add(pools.mesh);

  const field = w.time(w.own(buildCellField(grid, {
    floorColor: 0x90a7a1, wallColor: 0x90a7a1, floorInset: 0.065, floorThickness: 0.15, jitter: 0.045,
    floorTint: (x) => new THREE.Color().setScalar(0.76 + 0.18 * Math.sin(x * 0.62)),
    rimColor: 0x8dbeac, rimStrength: 0.22, seed: ctx.seed, name: 'moonlitFlagstoneRoad',
  })));
  w.add(field.group);

  const heather = new THREE.IcosahedronGeometry(0.55, 1).scale(1, 0.25, 1).translate(0, 0.14, 0);
  [
    { geometry: grassClumpGeometry(9, 0.25, 0.07), count: 12, scale: 0.9, a: 0x879786, b: 0x63817a, clearX: 1, clearZ: 1.05, patchy: false, name: 'moorGrass' },
    { geometry: heather, count: 24, scale: 1.8, a: 0x897b99, b: 0x646e84, clearX: 2.4, clearZ: 2, patchy: true, name: 'moorHeather' },
  ].forEach((spec, index) => land.scatter({
    name: spec.name, seed: ctx.seed + index * 717, tileSize: 6, perTile: spec.count,
    geometry: spec.geometry, colorA: spec.a, colorB: spec.b,
    material: oakMaterial({ color: 0xffffff, vertexColors: true, side: THREE.DoubleSide, roughness: 0.9, wind: { amplitude: 0.045, frequency: 1.15, height: 0.45 }, rimColor: 0x8aafad, rimStrength: 0.12 }),
    fill(tx, tz, rng, write) {
      for (let i = 0; i < spec.count; i++) {
        const x = (tx + rng()) * 6, z = (tz + rng()) * 6, y = heightAt(x, z);
        if ((Math.abs(x) < half + spec.clearX && Math.abs(z) < spec.clearZ) || y < -0.1 || (spec.patchy && Math.sin(x * 0.19 + Math.sin(z * 0.14) * 2.1) < 0.04)) continue;
        write(x, y, z, rng() * 6.28, spec.scale * (0.65 + rng() * 0.7), 0.8 + rng() * 0.55, rng());
      }
    },
  }));

  const motes = w.time(w.own(makeAmbientParticles({
    preset: 'fireflies', count: 100, bounds: { y: 0.75, halfX: grid.w * CELL * 0.6, halfY: 0.7, halfZ: 4 },
    seed: ctx.seed ^ 0x727e, color: 0xc6d6b2, size: 1.5, intensity: 1.4, name: 'moorFireflies',
  })));
  w.add(motes.points);

  return w.seal<WorldHandle>({
    rig,
    skin: moorSkin(palette),
    aurora,
    setView(coverage: ViewCoverage, camera: THREE.Camera) {
      land.setView(coverage, camera);
      extendWater(pools, coverage, 2, 2, poolDepth);
    },
  });
}
