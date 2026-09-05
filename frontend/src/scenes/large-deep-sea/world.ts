import * as THREE from 'three';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { terraceShelves } from '@kit/terrace';
import { TOKENS } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { makeWorld, type WorldCore } from '@kit/world';
import { LARGE_DEEPSEA_N } from '@sim/envs/largedeepsea';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import { causticLighting, coralGeometry, shelfLandscape } from '../deep-sea/world';

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  if (!geometry.grid || geometry.grid.w !== LARGE_DEEPSEA_N || geometry.grid.h !== LARGE_DEEPSEA_N) throw new Error('Large Deep Sea grid is missing or invalid');
  return geometry.grid;
}

export function resolveMask(grid: GridGeometry): Uint8Array {
  const mask = grid.actionMask;
  if (!mask || mask.length !== grid.w * grid.h) throw new Error('Large Deep Sea action mask is missing or invalid');
  return mask;
}

export interface AbyssSkin {
  timber: THREE.Color;
  timberDark: THREE.Color;
  heart: THREE.Color;
  snow: THREE.Color;
  silt: THREE.Color;
}

export function abyssSkin(tokens: typeof TOKENS): AbyssSkin {
  const abyss = tokens.palettes.abyss;
  const water = tokens.palettes.underwater;
  const accent = tokens.accents['large-deep-sea'] ?? tokens.uiCyan;
  const rockLip = blendTokens(abyss.groundB, water.groundA, 0.45);
  const snow = blendTokens(water.accentLight, abyss.hemiSky, 0.4);

  return {
    timber: (blendTokens(tokens.palettes.dusk.groundA, abyss.groundB, 0.45)).multiplyScalar(0.85),
    timberDark: (blendTokens(abyss.groundA, tokens.palettes.dusk.groundA, 0.3)).multiplyScalar(0.45),
    heart: blendTokens(tokens.celebrationGold, accent, 0.3),
    snow,
    silt: blendTokens(rockLip, snow, 0.34),
  };
}

function jellyGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const bell = new THREE.SphereGeometry(0.45, 10, 5, 0, Math.PI * 2, 0, Math.PI * 0.6);
  bell.scale(1, 0.7, 1);
  parts.push({ geometry: bell, color: 0xffffff });
  for (let i = 0; i < 6; i++) {
    const a = i * Math.PI / 3;
    const points = [0, 1, 2, 3].map(j => new THREE.Vector3(Math.cos(a + j * 0.22) * (0.28 - j * 0.045), -j * 0.35, Math.sin(a + j * 0.22) * 0.25));
    const trail = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 5, 0.018, 3, false);
    parts.push({ geometry: trail, color: 0xab77c9 });
  }
  return mergeColoredParts(parts, 'jellyBellAndTendrils');
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  skin: AbyssSkin;
  rush: AmbientParticlesHandle;
  silt: AmbientParticlesHandle;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, _mask: Uint8Array): WorldHandle {
  const w = makeWorld('theAbyss');
  const palette = ctx.tokens.palettes.abyss;
  const skin = abyssSkin(ctx.tokens);
  const sky = w.time(w.own(skyFromPalette(palette, 'abyss', { topSpread: 0.85, bottomSpread: 0.65 })));
  sky.setColors(0x378caa, 0x0c304e, 0x040e29);
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: [-0.3, 0.95, -0.16], pointCount: 2, pointDistance: 12 }));
  rig.hemi.color.set(0x7ab8cd);
  rig.hemi.groundColor.set(0x325d73);
  rig.hemi.intensity = 1.7;
  rig.sun.color.set(0xa4cee0);
  rig.sun.intensity = 1.65;
  rig.setShadowFocus({ x: 0, y: -1.5, z: 0 }, 16);
  w.add(rig.group);
  scene.fog = new THREE.FogExp2(0x123b56, 0.013);
  const field = w.own(buildCellField(grid, {
    floorColor: 0xffffff, wallColor: 0xffffff,
    floorFilter: (x, y) => x <= y,
    floorTint: (_x, y) => new THREE.Color(0x83c0c6).lerp(new THREE.Color(0x39627c), y / (grid.h - 1)),
    floorInset: 0.006, floorThickness: 2.8, jitter: 0.018,
    rimColor: 0x83c4c8, rimStrength: 0.2, seed: ctx.seed, name: 'descendingTrenchShelves',
  }));
  terraceShelves(field, grid);
  w.time(field);
  w.time(causticLighting(field.floorMaterial, 0.025));
  w.add(field.group);
  const half = LARGE_DEEPSEA_N * CELL / 2;
  const shelf = shelfLandscape(grid, 11);
  const heightAt = shelf.heightAt;
  const floor = w.time(w.own(makeContinuousSurface({
    name: 'abyssalEscarpment', colorA: 0x83c0c6, colorB: 0x39627c, segments: 112,
    height: shelf.height,
    detail: shelf.detail,
  })));
  w.add(floor.mesh);
  const cliffGeo = w.own(new THREE.IcosahedronGeometry(1, 1));
  cliffGeo.translate(0, 0.8, 0);
  const cliffMat = w.own(oakMaterial({ color: 0xffffff, flatShading: true, roughness: 0.95, rimColor: 0x5a97ad, rimStrength: 0.22 }));
  const cliffs = w.own(makeSceneryBank({
    name: 'trenchButtresses', seed: ctx.seed, tileSize: 28, perTile: 2,
    geometry: cliffGeo, material: cliffMat, colorA: 0x17324b, colorB: 0x426978,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 2; i++) {
        const x = (tx + rng()) * 28;
        const z = (tz + rng()) * 28;
        if (Math.abs(x) < half + 24) continue;
        write(x, heightAt(x, z) - 3, z, rng() * 6.28, 3 + rng() * 4, 1 + rng() * 0.8, rng());
      }
    },
  }));
  w.add(cliffs.mesh);
  const coralGeo = w.own(coralGeometry());
  const coralMat = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.7, emissive: 0x264660, emissiveIntensity: 0.22, wind: { amplitude: 0.05, frequency: 0.45 } }));
  const coral = w.own(makeSceneryBank({
    name: 'abyssSeaFans', seed: ctx.seed ^ 0x2e1a, tileSize: 18, perTile: 6,
    geometry: coralGeo, material: coralMat, colorA: 0x657ca2, colorB: 0x956791,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 6; i++) {
        const x = (tx + rng()) * 18;
        const z = (tz + rng()) * 18;
        if (Math.abs(x) < half + 1 && Math.abs(z) < half + 1 && x <= z + 0.6) continue;
        write(x, heightAt(x, z), z, rng() * 6.28, 1.2 + rng() * 2, 1 + rng(), rng());
      }
    },
  }));
  w.add(coral.mesh);
  const jellyGeo = w.own(jellyGeometry());
  const jellyMat = w.own(oakMaterial({ vertexColors: true, color: 0xade6f2, emissive: 0x6eb2d5, emissiveIntensity: 0.8, roughness: 0.25, side: THREE.DoubleSide }));
  const clock = { value: 0 };
  const compile = jellyMat.onBeforeCompile;
  const cache = jellyMat.customProgramCacheKey();
  jellyMat.onBeforeCompile = (shader, renderer) => {
    compile.call(jellyMat, shader, renderer);
    shader.uniforms.uMarineTime = clock;
    shader.vertexShader = `uniform float uMarineTime;\n${shader.vertexShader}`.replace('#include <begin_vertex>', `#include <begin_vertex>
float pulse = sin(uMarineTime*1.3 + instanceMatrix[3].x*.7 + instanceMatrix[3].z*.3);
transformed.xz *= 1. + .07*pulse;
transformed.y += .18 * pulse + .05 * sin(position.y*4. + uMarineTime);`);
  };
  jellyMat.customProgramCacheKey = () => `${cache}:marinePulse`;
  const jellies = w.own(makeSceneryBank({
    name: 'driftingJellyfish', seed: ctx.seed ^ 0x15ee, tileSize: 24, perTile: 1,
    geometry: jellyGeo, material: jellyMat, colorA: 0xb1e8ef, colorB: 0x959edb,
    fill(tx, tz, rng, write) {
      const x = (tx + rng()) * 24;
      const z = (tz + rng()) * 24;
      write(x, Math.max(heightAt(x, z) + 4, -1) + rng() * 7, z, rng() * 6.28, 0.7 + rng() * 1.2, 1, rng());
    },
  }));
  w.add(jellies.mesh);
  const snow = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'snow', count: 260, bounds: { y: 1, halfX: 45, halfY: 12, halfZ: 45 },
    seed: ctx.seed, color: skin.snow, size: 0.85, opacity: 0.45, rise: -0.1, name: 'marineSnow',
  }))));
  const silt = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 60, bounds: { halfX: 0.44, halfY: 0.15, halfZ: 0.44 },
    seed: ctx.seed ^ 0x3ee, color: skin.silt, size: 1.1, intensity: 1.3, rise: -0.09, name: 'diveSilt',
  }))));
  const rush = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'bubbles', count: 96, bounds: { halfX: 0.32, halfY: 1.5, halfZ: 0.32 },
    seed: ctx.seed ^ 0x7a91, color: skin.snow, size: 1.7, intensity: 2.4, rise: 1.7, name: 'resurfaceRush',
  }))));
  for (const particles of [snow, silt, rush]) w.add(particles.points);
  silt.setDensity(0);
  rush.setDensity(0);
  return w.seal({
    sky, rig, field, skin, rush, silt,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      floor.setView(coverage);
      cliffs.setView(coverage);
      coral.setView(coverage);
      jellies.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
      snow.points.position.set(coverage.x, 0, coverage.z);
    },
  }, { time(t) { clock.value = t; oakUniforms(coralMat)?.setTime(t); } });
}
