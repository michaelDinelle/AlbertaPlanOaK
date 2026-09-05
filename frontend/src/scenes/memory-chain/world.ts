import * as THREE from 'three';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { makeAmbientParticles } from '@kit/particles';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import type { SceneContext } from '@kit/types';
import { makeWater } from '@kit/water';
import { makeWorld, type WorldCore } from '@kit/world';
import { MEMORYCHAIN_CUE, MEMORYCHAIN_END, MEMORYCHAIN_POS } from '@sim/envs/memorychain';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';
import { extendWater } from '../half-cheetah/world';

export const CUE_CELL = 0;
export const CHOICE_CELL = MEMORYCHAIN_END;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== MEMORYCHAIN_END + 1 || grid.h !== 1) throw new Error('Memory Chain grid is missing or invalid');
  return grid;
}

export interface CaveSkin {
  stone: THREE.Color;
  rim: THREE.Color;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  skin: CaveSkin;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const w = makeWorld('limestoneCavernLake');
  const palette = ctx.tokens.palettes.cave;
  const skin = { stone: new THREE.Color(0xa8b7ae), rim: new THREE.Color(0xadcfd0) };
  const sky = w.time(w.own(skyFromPalette(palette, 'cave', { topSpread: 0.8, bottomSpread: 0.5 })));
  sky.setColors(0x192e3b, 0x2e5b64, 0x112b3a);
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: [-0.2, 0.85, 0.42], pointCount: 3, pointDistance: 12 }));
  rig.hemi.color.set(0xa6c7c8);
  rig.hemi.intensity = 1.6;
  rig.hemi.groundColor.set(0x466c73);
  rig.sun.color.set(0xd5e3ca);
  rig.sun.intensity = 1.45;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, grid.w * CELL * 0.6);
  w.add(rig.group);
  scene.fog = new THREE.FogExp2(0x274853, 0.0085);
  const water = w.time(w.own(makeWater({
    size: 2, segments: 2, palette, level: -0.22,
    shallow: 0x386576, deep: 0x102a43, sky: 0x6997a7, caustic: 0x41606a,
    amplitude: 0.012, waveScale: 0.6, speed: 0.2, fresnel: 0.26, caustics: 0.025,
    depthFalloff: 0.85, opacity: 0.96, sunDir: rig.sunDirection, name: 'cavernLake',
  })));
  water.setColors(0x386576, 0x102a43);
  w.add(water.mesh);
  const half = (grid.w - 1) * CELL / 2;
  const groundHeight = `float width = 8.2 + sin(p.x*.14)*1.1 + sin(p.x*.34)*.6;
float lake = length(vec2(p.x/${(half + 0.6).toFixed(4)},(p.y+sin(p.x*.1)*.8)/width));
float passage = abs(p.y-sin(p.x*.12)*.45)/1.8;
float bed = -1.1 + smoothstep(.7,.96,min(lake,passage))*1.15 + smoothstep(1.05,2.4,lake)*smoothstep(1.,1.7,passage)*(2.2+sin(p.x*.11)*sin(p.y*.07));
float wall = smoothstep(${(half + 1).toFixed(4)},${(half + 6).toFixed(4)},abs(p.x))*(1.-smoothstep(8.,12.,abs(p.y)))*smoothstep(1.2,2.8,abs(p.y))*6.;
float road = 1.-smoothstep(${(half + 0.6).toFixed(4)},${(half + 1.5).toFixed(4)},abs(p.x));
return mix(bed+wall,clamp(bed+wall,-.4,-.04),road*(1.-smoothstep(.4,.85,abs(p.y))));`;
  const floor = w.time(w.own(makeContinuousSurface({
    name: 'flowstoneLakeMargins', colorA: skin.stone, colorB: 0x66828a, segments: 160,
    height: groundHeight,
  })));
  w.add(floor.mesh);
  water.material.fragmentShader = `float lakeBed(vec2 p) {${groundHeight}}\n${water.material.fragmentShader}`;
  const field = w.own(buildCellField(grid, {
    floorColor: skin.stone, wallColor: skin.stone, floorInset: 0.025, floorThickness: 0.42, jitter: 0.015,
    rimColor: 0xa7ddd5, rimStrength: 0.25, seed: ctx.seed, name: 'luminousCauseway',
  }));
  field.floorMaterial.emissive.set(0x5a958e);
  field.floorMaterial.emissiveIntensity = 0.18;
  w.add(field.group);
  w.time(field);
  const vaultGeometry = w.own(new THREE.BufferGeometry());
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const stone = new THREE.Color();
  const low = new THREE.Color(0x97b4ad);
  const high = new THREE.Color(0x647e86);
  for (let ring = 0; ring <= 24; ring++) {
    const x = ring - 12;
    for (let arc = 0; arc <= 32; arc++) {
      const angle = arc / 32 * Math.PI;
      const rise = Math.sin(angle);
      const phase = x * Math.PI / 12;
      const fold = Math.cos(phase * 2 + Math.sin(angle * 3) * 0.25) * 0.65 + Math.cos(phase * 5 + angle * 0.4) * 0.25;
      const width = 12 + fold * (0.5 + Math.pow(rise, 0.65) * 1.7);
      const y = -0.7 + rise * (10.6 + Math.cos(phase) * 0.7 + fold);
      positions.push(x, y, -Math.cos(angle) * width);
      stone.copy(low).lerp(high, rise * 0.75);
      colors.push(stone.r, stone.g, stone.b);
      if (ring < 24 && arc < 32) {
        const a = ring * 33 + arc;
        const b = a + 33;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
  }
  vaultGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  vaultGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  vaultGeometry.setIndex(indices);
  vaultGeometry.computeVertexNormals();
  const vaultMaterial = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.97, rimColor: skin.rim, rimStrength: 0.18, emissive: 0x283f48, emissiveIntensity: 0.04, toonSteps: 3, toonSoftness: 0.32, toonFloor: 0.36, side: THREE.FrontSide }));
  const vault = w.own(makeSceneryBank({
    name: 'continuousLimestoneVault', seed: ctx.seed, tileSize: 24, perTile: 1,
    geometry: vaultGeometry, material: vaultMaterial, colorA: 0xffffff, colorB: 0xffffff,
    fill(tx, tz, _rng, write) {
      if (tz === 0) write((tx + 0.5) * 24, 0, 0, 0, 1, 1, 0);
    },
  }));
  w.add(vault.mesh);
  const motes = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'fireflies', count: 220, bounds: { y: 2, halfX: 28, halfY: 3, halfZ: 8 },
    seed: ctx.seed, color: 0xaedbba, size: 1.7, intensity: 1.8, opacity: 0.75, rise: 0.02, name: 'caveMotes',
  }))));
  w.add(motes.points);
  return w.seal({
    sky, rig, field, skin,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      extendWater(water.mesh, coverage, 2, 2, '-0.22 - lakeBed(vWorld.xz)');
      floor.setView(coverage);
      vault.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
    },
  });
}

export function readPos(state: { pose: Float32Array }): number {
  return Math.max(0, Math.min(MEMORYCHAIN_END, Math.round(state.pose[MEMORYCHAIN_POS] ?? 0)));
}

export function readCue(state: { pose: Float32Array }): 0 | 1 {
  return Math.round(state.pose[MEMORYCHAIN_CUE] ?? 0) === 1 ? 1 : 0;
}
