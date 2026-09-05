import * as THREE from 'three';
import { CELL, buildCellField, cellToWorldVec, isWall, excludeCellCenters, type CellFieldHandle } from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { blendTokens, clamp01, shade, smoothstep01 } from '@kit/mix';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { makeAmbientParticles } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { coniferTreeGeometry, scatterGrass, scatterRocks } from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import { POMAZE_H, POMAZE_W } from '@sim/envs/pomaze';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';

export const BUBBLE_OUTER = CELL * 2;
export const REVEAL_RADIUS = 3.2;
const SUN_DIR: readonly [number, number, number] = [0.62, 0.55, -0.556];

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== POMAZE_W || grid.h !== POMAZE_H || grid.walls.length !== grid.w * grid.h) {
    throw new Error('PO Maze requires complete recorded grid geometry');
  }
  return grid;
}

export interface ForestSkin {
  litterA: THREE.Color;
  litterB: THREE.Color;
  spruce: THREE.Color;
  spruceFar: THREE.Color;
  fern: THREE.Color;
  stone: THREE.Color;
  stoneDark: THREE.Color;
  haze: THREE.Color;
  motes: THREE.Color;
  bloom: THREE.Color;
  moth: THREE.Color;
}

export function forestSkin(palette: Palette, tokens: typeof TOKENS): ForestSkin {
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const gold = tokens.celebrationGold;
  const haze = shade(blendTokens(palette.fog, palette.skyTop, 0.62), 0.62);
  return {
    litterA: shade(blendTokens(palette.groundA, haze, 0.4), 0.53),
    litterB: shade(blendTokens(palette.groundB, haze, 0.35), 0.53),
    spruce: shade(blendTokens(palette.groundA, silver, 0.08), 0.68),
    spruceFar: shade(blendTokens(palette.groundA, haze, 0.5), 0.5),
    fern: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.54),
    stone: shade(blendTokens(silver, palette.hemiGround, 0.4), 0.44),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.2), 0.58),
    haze,
    motes: blendTokens(palette.sun, silver, 0.3),
    bloom: blendTokens(silver, palette.skyTop, 0.55),
    moth: blendTokens(gold, silver, 0.34),
  };
}

interface ForestReveal {
  setBubble(x: number, z: number, revealX: number, revealZ: number, amount: number): void;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  spruce: ForestReveal;
  skin: ForestSkin;
  setFogLift(amount: number): void;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette = ctx.tokens.palettes.dawn;
  const skin = forestSkin(palette, ctx.tokens);
  const w = makeWorld('fogboundForest');
  const sky = w.time(w.own(skyFromPalette(palette, 'dawn', {
    sunDir: SUN_DIR,
    sunHdr: 3.4,
    sunGlow: 0.5,
    sunGlowPower: 9,
  })));
  sky.setColors(blendTokens(palette.skyTop, skin.haze, 0.2), blendTokens(palette.skyHorizon, skin.haze, 0.35), skin.haze);
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 4, pointDistance: 5.5, sunDistanceScale: 2.6 }));
  w.add(rig.group);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, 5);
  const fog = applyFog(scene, palette, POMAZE_W * 2.6);
  fog.color.copy(skin.haze);
  let density = 3.8 / 256;
  const halfX = grid.w * CELL / 2;
  const halfZ = grid.h * CELL / 2;
  const heightAt = (x: number, z: number): number => -0.04 + smoothstep01(Math.max(Math.abs(x) - halfX, Math.abs(z) - halfZ) / 6)
    * (Math.sin(x * 0.07) * Math.cos(z * 0.085) * 0.6 + Math.sin(x * 0.19 + z * 0.13) * 0.15);
  const surface = w.time(w.own(makeContinuousSurface({
    name: 'forestLitter', seed: ctx.seed,
    colorA: skin.litterA, colorB: skin.litterB,
    height: `return -0.04 + smoothstep(0.0, 6.0, max(abs(p.x) - ${halfX.toFixed(3)}, abs(p.y) - ${halfZ.toFixed(3)}))
      * (sin(p.x * 0.07) * cos(p.y * 0.085) * 0.6 + sin(p.x * 0.19 + p.y * 0.13) * 0.15);`,
    detail: 'color *= 0.97 + 0.06 * landNoise(p.xz * 5.0);',
  })));
  w.add(surface.mesh);
  const field = w.time(w.own(buildCellField(grid, {
    floorColor: palette.sun,
    wallColor: palette.sun,
    floorUnderWalls: true,
    floorTint: (x, y) => blendTokens(skin.litterA, skin.litterB, hash2f(x, y, ctx.seed)),
    wallHeight: 0,
    floorInset: 0.035,
    floorThickness: 0.12,
    jitter: 0.08,
    seed: ctx.seed,
    name: 'forestPaths',
  })));
  w.add(field.group);
  for (const mesh of [...field.wallMeshes, ...field.floorMeshes]) mesh.visible = false;
  const agent = { value: new THREE.Vector2() };
  const reveal = { value: new THREE.Vector3() };
  const time = { value: 0 };
  const material = w.own(oakMaterial({
    color: skin.spruce, vertexColors: true, roughness: 0.95, flatShading: true,
    rimColor: skin.haze, rimStrength: 0.5, toonSteps: 2, toonSoftness: 0.34, toonFloor: 0.22,
    wind: { amplitude: 0.024, frequency: 0.75, height: 1.8 }, name: 'forestSpruce',
  }));
  const revealedMaterial = w.own(material.clone());
  const spruceCompile = material.onBeforeCompile;
  revealedMaterial.customProgramCacheKey = () => `${material.customProgramCacheKey()}|forest-reveal`;
  revealedMaterial.onBeforeCompile = (shader, renderer) => {
    spruceCompile.call(material, shader, renderer);
    shader.uniforms.uForestAgent = agent;
    shader.uniforms.uForestReveal = reveal;
    shader.uniforms.uForestTime = time;
    shader.vertexShader = `
      uniform vec2 uForestAgent;
      uniform vec3 uForestReveal;
      uniform float uForestTime;
    ` + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vec3 worldCenter = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
      vec2 cellCenter = floor(worldCenter.xz + 0.5);
      vec2 agentDistance = abs(cellCenter - uForestAgent);
      float nearAgent = 1.0 - smoothstep(1.0, ${BUBBLE_OUTER.toFixed(2)}, max(agentDistance.x, agentDistance.y));
      float nearBloom = uForestReveal.z * (1.0 - smoothstep(1.3, 3.2, length(worldCenter.xz - uForestReveal.xy)));
      float visibility = max(nearAgent, nearBloom);
      transformed *= vec3(visibility * (0.55 + 0.45 * visibility), visibility, visibility * (0.55 + 0.45 * visibility));
      transformed.x += sin(uForestTime * 0.8 + worldCenter.z * 0.7) * transformed.y * 0.009;
    `);
  };
  const shadowMaterial = w.own(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking }));
  shadowMaterial.onBeforeCompile = revealedMaterial.onBeforeCompile;
  shadowMaterial.customProgramCacheKey = () => 'forest-reveal-shadow';
  const geometry = w.own(coniferTreeGeometry(2));
  const bark = new THREE.Color(0x4e3e2d);
  bark.setRGB(bark.r / skin.spruce.r, bark.g / skin.spruce.g, bark.b / skin.spruce.b);
  const treeColors = geometry.getAttribute('color');
  for (let i = 0; i < 11 * 12; i++) treeColors.setXYZ(i, bark.r, bark.g, bark.b);
  const walls: [number, number][] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) if (isWall(grid, x, y)) walls.push([x, y]);
  }
  const trees = w.own(new THREE.InstancedMesh(geometry, revealedMaterial, walls.length * 3));
  trees.name = 'revealedSpruce';
  trees.frustumCulled = false;
  trees.castShadow = true;
  trees.customDepthMaterial = shadowMaterial;
  const transform = new THREE.Object3D();
  const tint = new THREE.Color();
  walls.forEach(([x, y], cell) => {
    for (let i = 0; i < 3; i++) {
      cellToWorldVec(x, y, grid, transform.position);
      const h1 = hash2f(x * 7 + i, y * 13 + i * 3, ctx.seed ^ 0x2f6b);
      const h2 = hash2f(x * 3 + i * 5, y * 11 + i, ctx.seed ^ 0x91c3);
      const h3 = hash2f(x + i * 17, y + i * 29, ctx.seed ^ 0x5ab1);
      const angle = (i + 0.5) / 3 * Math.PI * 2 + h1 * 1.1;
      const radius = 0.27 + h2 * 0.08;
      const scale = 0.34 + h3 * 0.18 + (i === 0 ? 0.06 : 0);
      transform.position.x += Math.cos(angle) * radius;
      transform.position.z += Math.sin(angle) * radius;
      transform.scale.set(scale, scale * (1.45 + h1), scale);
      transform.rotation.set((h3 - 0.5) * 0.14, h2 * Math.PI * 2, (h3 - 0.5) * 0.1);
      transform.updateMatrix();
      trees.setMatrixAt(cell * 3 + i, transform.matrix);
      trees.setColorAt(cell * 3 + i, tint.setScalar(0.88 + h3 * 0.24));
    }
  });
  trees.instanceMatrix.needsUpdate = true;
  w.add(trees);
  const outer = w.own(makeSceneryBank({
    name: 'continuingSpruce', geometry, material, tileSize: 12, perTile: 100,
    seed: ctx.seed ^ 0x77c1, colorA: 0xe3e9dd, colorB: 0x9caba3,
    receiveShadow: true,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 100; i++) {
        const x = (tx + (i % 10 + rng()) / 10) * 12;
        const z = (tz + (Math.floor(i / 10) + rng()) / 10) * 12;
        if (Math.abs(x) < halfX + 0.35 && Math.abs(z) < halfZ + 0.35) continue;
        const scale = 0.42 + rng() * 0.26;
        write(x, heightAt(x, z), z, rng() * Math.PI * 2, scale, 1.5 + rng() * 0.8, rng());
      }
    },
  }));
  w.add(outer.mesh);
  const groveParts: ColoredPart[] = [];
  for (let i = 0; i < 12; i++) {
    const tree = coniferTreeGeometry(0);
    const colors = tree.getAttribute('color');
    for (let j = 0; j < 5 * 12; j++) colors.setXYZ(j, bark.r, bark.g, bark.b);
    const scale = 0.42 + hash2f(i, 10, ctx.seed) * 0.26;
    tree.scale(scale, scale * (1.5 + hash2f(i, 11, ctx.seed) * 0.8), scale);
    tree.translate(Math.sin(i * 13.3) * 2, 0, Math.cos(i * 8.7) * 2);
    groveParts.push({ geometry: tree, color: new THREE.Color().setScalar(0.8 + (i % 4) * 0.06) });
  }
  const groveColors = groveParts.flatMap(({ geometry: tree }) => Array.from(tree.getAttribute('color').array));
  const groveGeometry = w.own(mergeColoredParts(groveParts, 'spruceGrove'));
  groveGeometry.setAttribute('color', new THREE.Float32BufferAttribute(groveColors, 3));
  const farForest = w.own(makeSceneryBank({
    name: 'continuingSpruceGroves', geometry: groveGeometry, material,
    tileSize: 96, perTile: 48, seed: ctx.seed ^ 0x631, colorA: 0xe3e9dd, colorB: 0x9caba3,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 48; i++) {
        const x = (tx + rng()) * 96;
        const z = (tz + rng()) * 96;
        if (Math.abs(x) < halfX + 64 && Math.abs(z) < halfZ + 64) continue;
        write(x, heightAt(x, z), z, rng() * Math.PI * 2, 1, 1, rng());
      }
    },
  }));
  w.add(farForest.mesh);
  const ferns = w.time(w.own(scatterGrass({
    count: 5200,
    area: { halfX: halfX + 2.5, halfZ: halfZ + 2.5 },
    seed: ctx.seed ^ 0x1f4d,
    exclude: excludeCellCenters(grid, 0.35),
    color: skin.fern,
    scale: [0.11, 0.27],
    stretch: [0.7, 1.6],
    wind: 0.04, rimColor: skin.haze, rimStrength: 0.34, hueJitter: 0.24,
    name: 'forestFerns',
  })));
  w.add(ferns.group);
  const stones = w.time(w.own(scatterRocks({
    count: 240, area: { halfX: halfX + 6, halfZ: halfZ + 6 }, seed: ctx.seed ^ 0x6b23,
    exclude: excludeCellCenters(grid, 0.34), heightAt, color: skin.stone,
    scale: [0.12, 0.34], stretch: [0.45, 0.85], hueJitter: 0.22,
    rimColor: skin.haze, rimStrength: 0.3, castShadow: false, name: 'forestStones',
  })));
  w.add(stones.group);
  const spores = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'pollen', count: 340,
    bounds: { x: 0, y: 1.2, z: 0, halfX, halfY: 1.1, halfZ },
    seed: ctx.seed ^ 0x7d05,
    color: skin.motes,
    opacity: 0.36,
    size: 0.8,
    intensity: 1.6,
    rise: 0.04,
    swirl: 0.2,
    name: 'forestFireflies',
  }))));
  w.add(spores.points);
  return w.seal({
    sky,
    rig,
    field,
    skin,
    spruce: {
      setBubble(x, z, revealX, revealZ, amount) {
        agent.value.set(x, z);
        reveal.value.set(revealX, revealZ, amount);
      },
    },
    setFogLift(amount) { fog.density = density * (1 - clamp01(amount) * 0.35); },
    setView(coverage, camera) {
      density = 3.8 / coverage.far;
      surface.setView(coverage);
      outer.setView(coverage);
      farForest.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
      rig.setShadowFocus({ x: agent.value.x, y: 0.3, z: agent.value.y }, 5);
    },
  }, {
    time(t) { time.value = t; oakUniforms(material)?.setTime(t); },
  });
}
