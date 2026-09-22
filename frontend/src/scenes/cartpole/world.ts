import * as THREE from 'three';
import { makeHeightField } from '@kit/heightfield';
import { extendVegetation, makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { blendTokens, TOKEN_WHITE } from '@kit/mix';
import { makeMistPlanes } from '@kit/particles';
import { hoodooGeometry, lamp, propMaterials } from '@kit/props';
import { iceMaterial } from '@kit/props/ice';
import { modelParts } from '@kit/robot';
import { streamFor } from '@kit/rng';
import type { SceneContext } from '@kit/types';
import { excludeAny, excludeBand, scatterGrass, scatterRocks, scatterTrees } from '@kit/vegetation';
import type { WorldCore } from '@kit/world';
import { CP_TRACK_HALF } from '@sim/envs/cartpole';
import { parameter } from '../../data/envs';

export const RAIL_Y = 1.22;
export const COURT_HALF_X = CP_TRACK_HALF + 1.6;
export const COURT_HALF_Z = 2.2;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.noon, accent = ctx.tokens.accents['cartpole'] ?? ctx.tokens.celebrationGold;
  const slip = parameter(ctx.environment, 'slip') > 0;
  const land = makeLandscape(scene, 'prairieRail', palette, 'noon', {
    sky: { sunDir: [0.42, 0.78, -0.46], sunSize: 0.035, sunHdr: 7, sunGlow: 0.3, sunGlowPower: 26, topSpread: 0.55, bottomSpread: 0.3, radius: 320 },
    light: { pointCount: 2, pointDistance: 3.2, sunDistanceScale: 3 },
    fogExtent: 140,
  });
  const { world, sky, rig } = land;
  sky.setColors(0x5aa6e6, 0xcfe3ec, 0xe8e2c4);
  rig.sun.color.set(0xfff1d2); rig.sun.intensity = 2.9;
  rig.hemi.color.set(0xbfe0f5); rig.hemi.groundColor.set(0x8a8a5a); rig.hemi.intensity = 0.9;
  rig.setShadowFocus({ x: 0, y: 0.6, z: 0 }, 4.2);
  const field = makeHeightField({ base: 0, court: { halfX: COURT_HALF_X, halfZ: COURT_HALF_Z, rise: 0.3 }, octaves: [{ wavelength: 46, amplitude: 1.5, seed: ctx.seed }, { wavelength: 13, amplitude: 0.32, seed: ctx.seed ^ 0x3c }] });
  const prairieA = blendTokens(palette.groundA, 0xb8a860, 0.35), prairieB = blendTokens(palette.groundB, 0xe0cf84, 0.3);
  const terrain = world.own(world.time(makeSceneTerrain({
    follow: false, name: 'prairieGround', seed: ctx.seed, colorA: prairieA, colorB: prairieB, segments: 128, radius: 420, height: field.glsl,
    detail: 'float sway = sin(p.x * 0.9 + sin(p.z * 0.7) * 2.0) * sin(p.z * 1.3); color *= 0.93 + 0.07 * sway + 0.05 * landNoise(p.xz * 1.9);',
  }, field.heightAt, { roughness: 0.95, rimColor: palette.skyHorizon, rimStrength: 0.12, toonSteps: 4, toonSoftness: 0.35 })));
  world.add(terrain.mesh);
  const heightAt = field.heightAt;
  const materials = propMaterials(world), parts = modelParts(world), { rod } = parts;
  const timber = world.own(oakMaterial({ color: 0x8a6a45, roughness: 0.85, rimColor: palette.sun, rimStrength: 0.3, toonSteps: 3, name: 'railTimber' }));
  const rail = new THREE.Group();
  rail.name = 'woodenRail';
  const railLength = CP_TRACK_HALF * 2 + 0.6;
  parts.box(rail, timber, 0, RAIL_Y - 0.11, 0, railLength, 0.09, 0.34, 0.015);
  for (let i = 0; i <= 8; i++) {
    const x = -railLength / 2 + i * railLength / 8;
    parts.box(rail, timber, x, RAIL_Y - 0.19, 0, 0.11, 0.16, 0.52, 0.012);
    if (i % 2 === 1) for (const side of [-1, 1]) {
      rod(rail, timber, new THREE.Vector3(x + side * 0.05, RAIL_Y - 0.24, side * 0.2), new THREE.Vector3(x + side * 0.14, heightAt(x + side * 0.14, side * 0.42) - 0.05, side * 0.42), 0.035);
      rod(rail, timber, new THREE.Vector3(x + side * 0.05, RAIL_Y - 0.24, side * 0.2), new THREE.Vector3(x - side * 0.14, heightAt(x - side * 0.14, side * 0.42) - 0.05, side * 0.42), 0.035);
    }
  }
  for (const x of [-railLength / 2 + railLength / 8, railLength / 2 - railLength / 8]) parts.box(rail, timber, x, RAIL_Y - 0.62, 0, 0.09, 0.05, 0.86, 0.01);
  for (const side of [-1, 1]) {
    const x = side * (railLength / 2 + 0.24);
    parts.box(rail, materials.sandstone, x, RAIL_Y / 2 - 0.06, 0, 0.5, RAIL_Y - 0.12, 0.66, 0.04);
    parts.box(rail, materials.brass, x, RAIL_Y + 0.02, 0, 0.56, 0.05, 0.72, 0.01);
    parts.box(rail, materials.sandstone, x, RAIL_Y + 0.3, 0, 0.3, 0.5, 0.3, 0.03);
    parts.box(rail, materials.sandstone, x, 0.1, 0, 0.7, 0.2, 0.86, 0.04);
  }
  parts.build();
  const railTop = new THREE.Mesh(world.own(new THREE.BoxGeometry(railLength, 0.024, 0.09)), slip ? iceMaterial(world, 0xe4f6ff, 'railIce') : materials.brass);
  railTop.position.set(0, RAIL_Y - 0.052, 0);
  railTop.receiveShadow = true;
  rail.add(railTop);
  world.add(rail);
  const hoodoo = world.own(hoodooGeometry());
  const hoodooMaterial = world.own(oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.96, rimColor: palette.sun, rimStrength: 0.22, toonSteps: 3, toonSoftness: 0.3, toonFloor: 0.2, name: 'hoodooStone' }));
  const hoodoos = world.own(new THREE.InstancedMesh(hoodoo, hoodooMaterial, 5));
  hoodoos.castShadow = hoodoos.receiveShadow = true;
  {
    const rng = streamFor(ctx.seed, 'hoodoos'), dummy = new THREE.Object3D();
    const spots: readonly (readonly [number, number, number, number])[] = [[-9, -14, 1.6, 0.3], [2, -17, 2.1, -0.4], [11, -13, 1.5, 1.2], [-15, -6, 1.3, 2.4], [16, -3, 1.4, -1.1]];
    spots.forEach(([x, z, s, yaw], i) => {
      dummy.position.set(x, heightAt(x, z) - 0.2, z);
      dummy.rotation.set(0, yaw + (rng() - 0.5) * 0.4, 0);
      dummy.scale.set(s, s * (0.9 + rng() * 0.3), s);
      dummy.updateMatrix();
      hoodoos.setMatrixAt(i, dummy.matrix);
    });
    hoodoos.computeBoundingSphere();
  }
  world.add(hoodoos);
  const lamps = [-1, 1].map(side => {
    const l = lamp(world, accent, 1.25, materials);
    l.group.position.set(side * (railLength / 2 + 0.24), RAIL_Y + 0.02, 0);
    l.group.scale.setScalar(0.7);
    world.add(l.group);
    l.lit(0.35);
    return l;
  });
  const grassColor = blendTokens(0xb8b86a, palette.groundB, 0.4);
  const railBand = excludeBand(0, 0.55, 0.2);
  const grass = scatterGrass({
    count: 2600, area: { x: 0, z: 0, halfX: COURT_HALF_X + 3, halfZ: COURT_HALF_Z + 3 }, seed: ctx.seed ^ 0x7f19,
    exclude: railBand, heightAt, color: grassColor, scale: [0.16, 0.34], stretch: [0.9, 1.7], hueJitter: 0.24,
    wind: 0.06, windFrequency: 1.2, rimColor: palette.sun, rimStrength: 0.35, name: 'prairieGrass',
  });
  world.add(grass.group); world.time(world.own(grass));
  const far = world.own(extendVegetation(grass, {
    name: 'prairieFar', seed: ctx.seed ^ 0x7f19, tileSize: 18, perTile: 520, scale: [0.18, 0.4], stretch: [0.9, 1.6], heightAt,
    exclude: (x, z) => Math.abs(x) < COURT_HALF_X + 3 && Math.abs(z) < COURT_HALF_Z + 3 || Math.sin(x * 0.13 + z * 0.19) + Math.cos(z * 0.08 - x * 0.05) < -0.55,
  }));
  for (const mesh of far.meshes) world.add(mesh);
  const rocks = scatterRocks({
    count: 70, area: { x: 0, z: -4, halfX: 12, halfZ: 7 }, seed: ctx.seed ^ 0x2b17, exclude: excludeAny(railBand, (x, z) => Math.abs(x) < COURT_HALF_X && Math.abs(z) < COURT_HALF_Z),
    heightAt, color: blendTokens(0xc9b08a, palette.hemiGround, 0.3), scale: [0.08, 0.3], stretch: [0.5, 0.9], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.25, castShadow: true, name: 'prairieStones',
  });
  world.add(rocks.group); world.time(world.own(rocks));
  const trees = scatterTrees({
    archetypes: ['broadleaf'], count: 40, area: { x: 0, z: -9, halfX: 16, halfZ: 6 }, seed: ctx.seed ^ 0x51ab,
    exclude: (x, z) => Math.abs(x) < 6 && z > -7, heightAt, maxSlope: 0.6, color: blendTokens(0x7d9d4a, palette.groundB, 0.25), scale: [0.5, 0.95], stretch: [0.9, 1.3], hueJitter: 0.2, wind: 0.025, rimColor: palette.sun, rimStrength: 0.25, castShadow: true, name: 'prairieAspens',
  });
  world.add(trees.group); world.time(world.own(trees));
  const haze = makeMistPlanes({ size: [22, 10], color: blendTokens(palette.fog, TOKEN_WHITE, 0.35), seed: ctx.seed, layers: 3, baseY: 0.05, spread: 0.5, opacity: 0.05, speed: 0.01, scale: 2, nearFade: 3, name: 'prairieHaze' });
  haze.group.position.set(0, 0, -6);
  world.add(haze.group); world.time(world.own(haze));
  rig.points.forEach((point, i) => { point.color.set(accent); point.intensity = 0; point.position.set(i === 0 ? -1 : 1, RAIL_Y + 0.7, 0.3); });
  const extra = { land, rig, heightAt, lamps, materials, accent, slip, palette, setView(coverage: Parameters<typeof land.setView>[0], camera: THREE.Camera) { land.setView(coverage, camera, terrain, far); } };
  return world.seal<WorldCore & typeof extra>(extra);
}

export type PrairieWorld = ReturnType<typeof buildWorld>;
