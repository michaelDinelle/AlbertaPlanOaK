import * as THREE from 'three/webgpu';
import { positionGeometry, sin, uniform, vec3, vec4 } from 'three/tsl';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { makeLandscape } from '@kit/landscape';
import type { LightRig } from '@kit/lighting';
import { oakMaterial, setDeform } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { SkyHandle } from '@kit/sky';
import { terraceShelves } from '@kit/terrace';
import type { TOKENS } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { ViewCoverage } from '@kit/view';
import type { WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';
import { isReef } from '../reef';
import { causticLighting, coralGeometry, shelfLandscape } from '../world';

export interface AbyssSkin {
  timber: THREE.Color;
  timberDark: THREE.Color;
  heart: THREE.Color;
  snow: THREE.Color;
  silt: THREE.Color;
}

export function abyssSkin(tokens: typeof TOKENS, accent: number): AbyssSkin {
  const abyss = tokens.palettes.abyss;
  const water = tokens.palettes.underwater;
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

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, reach: Uint8Array, accent: number): WorldHandle {
  const palette = ctx.tokens.palettes.abyss;
  const skin = abyssSkin(ctx.tokens, accent);
  const land = makeLandscape(scene, 'theAbyss', palette, 'abyss', {
    sky: { topSpread: 0.85, bottomSpread: 0.65 },
    light: { sunDir: [-0.3, 0.95, -0.16], pointCount: 2, pointDistance: 12 },
    fogDensity: 0.013,
  });
  const { world: w, sky, rig, fog } = land;
  sky.setColors(0x378caa, 0x0c304e, 0x040e29);
  rig.hemi.color.set(0x7ab8cd);
  rig.hemi.groundColor.set(0x325d73);
  rig.hemi.intensity = 1.7;
  rig.sun.color.set(0xa4cee0);
  rig.sun.intensity = 1.65;
  rig.setShadowFocus({ x: 0, y: -1.5, z: 0 }, 16);
  fog.color.set(0x123b56);
  const shallow = new THREE.Color(0x83c0c6), deep = new THREE.Color(0x39627c);
  const field = w.own(buildCellField(grid, {
    floorColor: 0xffffff, wallColor: 0xffffff,
    floorFilter: (x, y) => isReef(reach, grid, x, y),
    floorTint: (_x, y) => shallow.clone().lerp(deep, y / (grid.h - 1)),
    floorInset: 0.006, floorThickness: 2.8, jitter: 0.018,
    rimColor: 0x83c4c8, rimStrength: 0.2, seed: ctx.seed, name: 'descendingTrenchShelves',
  }));
  terraceShelves(field, grid);
  w.time(field);
  w.time(causticLighting(field.floorMaterial, 0.025));
  w.add(field.group);
  const half = grid.w * CELL / 2;
  const shelf = shelfLandscape(grid, 11);
  const heightAt = shelf.heightAt;
  land.surface({
    name: 'abyssalEscarpment', colorA: shallow, colorB: deep, segments: 112,
    height: shelf.height,
    detail: shelf.detail,
  });
  const cliffGeo = new THREE.IcosahedronGeometry(1, 1);
  cliffGeo.translate(0, 0.8, 0);
  land.scatter({
    name: 'trenchButtresses', seed: ctx.seed, tileSize: 28, perTile: 2,
    geometry: cliffGeo, material: oakMaterial({ color: 0xffffff, flatShading: true, roughness: 0.95, rimColor: 0x5a97ad, rimStrength: 0.22 }), colorA: 0x17324b, colorB: 0x426978,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 2; i++) {
        const x = (tx + rng()) * 28;
        const z = (tz + rng()) * 28;
        if (Math.abs(x) < half + 24) continue;
        write(x, heightAt(x, z) - 3, z, rng() * 6.28, 3 + rng() * 4, 1 + rng() * 0.8, rng());
      }
    },
  });
  land.scatter({
    name: 'abyssSeaFans', seed: ctx.seed ^ 0x2e1a, tileSize: 18, perTile: 6,
    geometry: coralGeometry(), material: oakMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.7, emissive: 0x264660, emissiveIntensity: 0.22, wind: { amplitude: 0.05, frequency: 0.45 } }), colorA: 0x657ca2, colorB: 0x956791,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 6; i++) {
        const x = (tx + rng()) * 18;
        const z = (tz + rng()) * 18;
        if (Math.abs(x) < half + 1 && Math.abs(z) < half + 1 && x <= z + 0.6) continue;
        write(x, heightAt(x, z), z, rng() * 6.28, 1.2 + rng() * 2, 1 + rng(), rng());
      }
    },
  });
  const clock = uniform(0);
  const jellyMat = oakMaterial({ vertexColors: true, color: 0xade6f2, emissive: 0x6eb2d5, emissiveIntensity: 0.8, roughness: 0.25, side: THREE.DoubleSide });
  setDeform(jellyMat, (matrix) => {
    const origin = matrix ? matrix.mul(vec4(0, 0, 0, 1)).xyz : vec3(0);
    const pulse = sin(clock.mul(1.3).add(origin.x.mul(0.7)).add(origin.z.mul(0.3)));
    const swell = pulse.mul(0.07).add(1);
    return { position: vec3(positionGeometry.x.mul(swell), positionGeometry.y.add(pulse.mul(0.18)).add(sin(positionGeometry.y.mul(4).add(clock)).mul(0.05)), positionGeometry.z.mul(swell)) };
  });
  land.scatter({
    name: 'driftingJellyfish', seed: ctx.seed ^ 0x15ee, tileSize: 24, perTile: 1,
    geometry: jellyGeometry(), material: jellyMat, colorA: 0xb1e8ef, colorB: 0x959edb,
    fill(tx, tz, rng, write) {
      const x = (tx + rng()) * 24;
      const z = (tz + rng()) * 24;
      write(x, Math.max(heightAt(x, z) + 4, -1) + rng() * 7, z, rng() * 6.28, 0.7 + rng() * 1.2, 1, rng());
    },
  });
  const snow = w.time(w.own(makeAmbientParticles({
    preset: 'snow', count: 260, bounds: { y: 1, halfX: 45, halfY: 12, halfZ: 45 },
    seed: ctx.seed, color: skin.snow, size: 0.85, opacity: 0.45, rise: -0.1, name: 'marineSnow',
  })));
  const silt = w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 60, bounds: { halfX: 0.44, halfY: 0.15, halfZ: 0.44 },
    seed: ctx.seed ^ 0x3ee, color: skin.silt, size: 1.1, intensity: 1.3, rise: -0.09, name: 'diveSilt',
  })));
  const rush = w.time(w.own(makeAmbientParticles({
    preset: 'bubbles', count: 96, bounds: { halfX: 0.32, halfY: 1.5, halfZ: 0.32 },
    seed: ctx.seed ^ 0x7a91, color: skin.snow, size: 1.7, intensity: 2.4, rise: 1.7, name: 'resurfaceRush',
  })));
  for (const particles of [snow, silt, rush]) w.add(particles.points);
  silt.setDensity(0);
  rush.setDensity(0);
  const drift = { setView(coverage: ViewCoverage) { snow.points.position.set(coverage.x, 0, coverage.z); } };
  return w.seal({
    sky, rig, field, skin, rush, silt,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) { land.setView(coverage, camera, drift); },
  }, { time(t) { clock.value = t; } });
}
