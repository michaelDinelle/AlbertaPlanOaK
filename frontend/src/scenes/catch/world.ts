import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { emissiveMaterial, oakMaterial } from '@kit/materials';
import { makeMeadow } from '@kit/meadow';
import { blendTokens, TOKEN_WHITE } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { lamp, propMaterials, vineStrand } from '@kit/props';
import { engravedPanel, modelParts } from '@kit/robot';
import { streamFor } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { parameter } from '../../data/envs';

export const PADDLE_Y = 0.32;
export const COURT_DEPTH = 2.6;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.dusk, accent = ctx.tokens.accents['catch'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const width = parameter(ctx.environment, 'width'), height = parameter(ctx.environment, 'height'), noise = parameter(ctx.environment, 'noise'), scale = parameter(ctx.environment, 'scale');
  const cell = 0.5 * (scale > 1 ? 1.8 : 1);
  const courtW = width * cell, top = PADDLE_Y + (height - 1) * cell, postH = top + 0.9;
  const colX = (col: number): number => (col - (width - 1) / 2) * cell;
  const rowY = (row: number): number => PADDLE_Y + (height - 1 - row) * cell;
  const land = makeLandscape(scene, 'pergolaCourt', palette, 'dusk', {
    sky: { sunDir: [0.62, 0.2, -0.76], sunSize: 0.05, sunHdr: 7, sunGlow: 0.42, sunGlowPower: 20, topSpread: 0.62, bottomSpread: 0.3, radius: 320 },
    light: { pointCount: 1, pointDistance: 3.5, sunDistanceScale: 3 },
    fogExtent: 90,
    weather: { preset: 'pollen', count: 420, seed: ctx.seed, color: 0xfff1c8, size: 1.3, intensity: 0.9, bounds: { y: postH / 2, halfX: courtW / 2 + 1.5, halfY: postH / 2, halfZ: 2 } },
  });
  const { world, rig } = land;
  rig.sun.color.set(0xffb96a); rig.sun.intensity = 2.1;
  rig.hemi.color.set(0x8f9ad0); rig.hemi.groundColor.set(0x5a4634); rig.hemi.intensity = 0.7;
  rig.setShadowFocus({ x: 0, y: postH / 2, z: 0 }, Math.max(4, courtW * 0.7, postH * 0.65));
  const meadow = makeMeadow(world, ctx, { name: 'pergolaMeadow', halfX: courtW / 2 + 2.6, halfZ: 3.4, ground: blendTokens(palette.groundA, 0x6e6a44, 0.4), grass: blendTokens(palette.groundB, 0x8a9450, 0.45), trees: [0xc98a4a, 0xa8b060, 0xd9a35a, 0x7f9a5a], amplitude: 1.3 });
  const materials = propMaterials(world), parts = modelParts(world);
  const court = new THREE.Group();
  court.name = 'pergolaCourt';
  parts.box(court, materials.sandstone, 0, -0.16, 0, courtW + 1.8, 0.32, COURT_DEPTH + 0.8, 0.04);
  for (let i = 0; i <= width; i++) parts.box(court, materials.shale, colX(i - 0.5), 0.005, 0, 0.03, 0.012, COURT_DEPTH + 0.5, 0.004);
  parts.box(court, materials.brass, 0, PADDLE_Y - 0.14, 0, courtW + 0.3, 0.05, 0.14, 0.01);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    parts.box(court, materials.sandstone, sx * (courtW / 2 + 0.55), postH / 2, sz * (COURT_DEPTH / 2 + 0.05), 0.36, postH, 0.36, 0.04);
    parts.box(court, materials.brass, sx * (courtW / 2 + 0.55), postH + 0.04, sz * (COURT_DEPTH / 2 + 0.05), 0.46, 0.06, 0.46, 0.01);
  }
  for (const sz of [-1, 1]) parts.box(court, materials.brass, 0, postH + 0.14, sz * (COURT_DEPTH / 2 + 0.05), courtW + 1.5, 0.12, 0.2, 0.02);
  const slats = Math.max(3, Math.round((courtW + 1.4) / 0.7));
  for (let i = 0; i < slats; i++) parts.box(court, materials.shale, -(courtW + 1.4) / 2 + (i + 0.5) * (courtW + 1.4) / slats, postH + 0.3, 0, 0.14, 0.14, COURT_DEPTH + 1, 0.02);
  parts.build();
  {
    const rng = streamFor(ctx.seed, 'pergolaVines');
    const strands = Math.max(3, Math.round(courtW / 0.9));
    for (let s = 0; s < strands; s++) {
      const x = -courtW / 2 + (s + 0.5) * courtW / strands + (rng() - 0.5) * 0.3, z = (s % 2 ? 1 : -1) * (COURT_DEPTH / 2 + 0.02), drop = 0.7 + rng() * 0.9;
      const points: [number, number, number][] = [];
      for (let k = 0; k < 5; k++) points.push([x + Math.sin(s * 2.1 + k * 1.3) * 0.12, postH + 0.24 - drop * k / 4, z + Math.sin(k * 1.7 + s) * 0.05 * (s % 2 ? 1 : -1)]);
      const strand = new THREE.Mesh(world.own(vineStrand(points, 0.016, 8)), materials.vine);
      strand.castShadow = true;
      court.add(strand);
    }
  }
  world.add(court);
  const lamps = [-1, 1].map(side => {
    const l = lamp(world, accent, 1.5, materials);
    l.group.position.set(side * (courtW / 2 + 1.05), 0, COURT_DEPTH / 2 + 0.6);
    l.group.scale.setScalar(0.85);
    world.add(l.group);
    l.lit(0.5);
    return l;
  });
  const paddle = new THREE.Group();
  paddle.name = 'brassPaddle';
  const paddleGlass = materials.glass(accent, 0.8);
  parts.box(paddle, materials.brass, 0, PADDLE_Y, 0, cell * 0.92, 0.09, 0.34, 0.02);
  parts.box(paddle, paddleGlass, 0, PADDLE_Y + 0.06, 0, cell * 0.7, 0.03, 0.12, 0.008);
  parts.add(paddle, engravedPanel(0.16 * cell / 0.5, 0.07, 0.02, 0.01, 0.07).translate(0, PADDLE_Y, 0.18), world.own(oakMaterial({ color: 0xe8d8b8, vertexColors: true, roughness: 0.5, metalness: 0.2, rimColor: palette.sun, rimStrength: 0.5, toonSteps: 3, name: 'paddleArmor' })));
  parts.build();
  world.add(paddle);
  const seedRadius = 0.13 * cell / 0.5;
  const seedMaterial = world.own(emissiveMaterial(blendTokens(gold, TOKEN_WHITE, 0.2), 2.8, { name: 'fallingSeed' }));
  const seed = new THREE.Mesh(world.own(new THREE.IcosahedronGeometry(seedRadius, 1).scale(1, 1.35, 1)), seedMaterial);
  seed.name = 'fallingSeed';
  seed.castShadow = true;
  const sprite = world.own(radialSprite(48, 2.4));
  const haloMaterial = world.own(emissiveMaterial(blendTokens(gold, accent, 0.4), 1.2, { map: sprite, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, name: 'seedHalo' }));
  const halo = new THREE.Mesh(world.own(new THREE.PlaneGeometry(1, 1)), haloMaterial);
  halo.scale.setScalar(seedRadius * 7);
  seed.add(halo);
  world.add(seed);
  const glitter = makeAmbientParticles({ preset: 'fireflies', count: noise > 0 ? 380 : 0, bounds: { y: top / 2 + 0.2, halfX: courtW / 2 + 0.6, halfY: top / 2 + 0.3, halfZ: 1.4 }, seed: ctx.seed ^ 0x51ab, color: blendTokens(gold, TOKEN_WHITE, 0.5), size: 0.24, intensity: 2.2, opacity: 0.8, swirl: 0.35, name: 'rewardShimmer' });
  world.add(glitter.points); world.size(world.time(world.own(glitter)));
  const seedLight = rig.points[0];
  if (seedLight) { seedLight.color.set(gold); seedLight.intensity = 1.6; }
  const extra = {
    accent, gold, palette, width, height, noise, scale, cell, courtW, top, colX, rowY, paddle, paddleGlass, seed, seedMaterial, halo, haloMaterial, glitter, lamps, seedLight,
    setView(coverage: Parameters<typeof land.setView>[0], camera: THREE.Camera) { land.setView(coverage, camera, meadow); halo.quaternion.copy(camera.quaternion); },
  };
  return world.seal<WorldCore & typeof extra>(extra);
}

export type PergolaWorld = ReturnType<typeof buildWorld>;
