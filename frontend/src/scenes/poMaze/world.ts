import * as THREE from 'three/webgpu';
import { abs, clamp, cos, floor, length, max, modelWorldMatrix, positionGeometry, sin, smoothstep, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { CELL, cellToWorldVec, excludeCellCenters, isWall } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { makeLandscape } from '@kit/landscape';
import { oakMaterial, oakUniforms, setDeform } from '@kit/materials';
import { blendTokens, clamp01, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import type { Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { coniferTreeGeometry, scatterGrass, scatterRocks } from '@kit/vegetation';
import { makeSceneryBank, type ViewCoverage } from '@kit/view';
import type { WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';

export const BUBBLE_OUTER = CELL * 2;
const BLOOM_INNER = 1.3;
const BLOOM_OUTER = 3.2;
const SUN_DIR = [0.62, 0.55, -0.556] as const;
const WIND = { amplitude: 0.024, frequency: 0.75, height: 1.8 };
const BARK = 0x4e3e2d;

export type ForestSkin = ReturnType<typeof forestSkin>;
export type WorldHandle = ReturnType<typeof buildWorld>;

export function forestSkin(palette: Palette, silver: number) {
  const haze = shade(blendTokens(palette.fog, palette.skyTop, 0.62), 0.62);
  return {
    litterA: shade(blendTokens(palette.groundA, haze, 0.4), 0.53),
    litterB: shade(blendTokens(palette.groundB, haze, 0.35), 0.53),
    spruce: shade(blendTokens(palette.groundA, silver, 0.08), 0.68),
    fern: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.54),
    stone: shade(blendTokens(silver, palette.hemiGround, 0.4), 0.44),
    haze,
    motes: blendTokens(palette.sun, silver, 0.3),
    bloom: blendTokens(silver, palette.skyTop, 0.55),
  };
}

function barkTrunk(geometry: THREE.BufferGeometry, vertices: number, bark: THREE.Color): THREE.BufferGeometry {
  const colors = geometry.getAttribute('color');
  for (let i = 0; i < vertices; i++) colors.setXYZ(i, bark.r, bark.g, bark.b);
  return geometry;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, silver: number) {
  const palette = ctx.tokens.palettes.dawn;
  const skin = forestSkin(palette, silver);
  const land = makeLandscape(scene, 'fogboundForest', palette, 'dawn', {
    sky: { sunDir: SUN_DIR, sunHdr: 3.4, sunGlow: 0.5, sunGlowPower: 9 },
    light: { pointCount: 4, pointDistance: 5.5, sunDistanceScale: 2.6 },
    fogExtent: grid.w * 2.6,
  });
  const { world: w, sky, rig, fog } = land;
  sky.setColors(blendTokens(palette.skyTop, skin.haze, 0.2), blendTokens(palette.skyHorizon, skin.haze, 0.35), skin.haze);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, 5);
  rig.sun.intensity = 1.5;
  rig.hemi.color.copy(blendTokens(skin.haze, silver, 0.45));
  rig.hemi.groundColor.copy(skin.litterB);
  rig.hemi.intensity = 2;
  fog.color.copy(blendTokens(skin.haze, silver, 0.35));

  const halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
  const heightAt = (x: number, z: number): number => -0.04 + smoothstep01(Math.max(Math.abs(x) - halfX, Math.abs(z) - halfZ) / 6)
    * (Math.sin(x * 0.07) * Math.cos(z * 0.085) * 0.6 + Math.sin(x * 0.19 + z * 0.13) * 0.15);
  land.surface({
    name: 'forestLitter', seed: ctx.seed, colorA: skin.litterA, colorB: skin.litterB,
    height: (p) => smoothstep(0, 6, max(abs(p.x).sub(halfX), abs(p.y).sub(halfZ)))
      .mul(sin(p.x.mul(0.07)).mul(cos(p.y.mul(0.085))).mul(0.6).add(sin(p.x.mul(0.19).add(p.y.mul(0.13))).mul(0.15))).sub(0.04),
    detail: (s) => s.color.mul(s.noise(s.p.xz.mul(5)).mul(0.06).add(0.97)),
  });

  const spruce = { color: skin.spruce, vertexColors: true, roughness: 0.95, flatShading: true, rimColor: skin.haze, rimStrength: 0.5, toonSteps: 2, toonSoftness: 0.34, toonFloor: 0.22 };
  const material = w.own(oakMaterial({ ...spruce, wind: WIND, name: 'forestSpruce' }));
  const revealed = w.own(oakMaterial({ ...spruce, name: 'revealedSpruce' }));
  const agent = uniform(new THREE.Vector2()), bloom = uniform(new THREE.Vector3()), clock = uniform(0);
  const origin = vec2((grid.w - 1) / 2, (grid.h - 1) / 2);
  setDeform(revealed, (matrix) => {
    const root = modelWorldMatrix.mul(matrix ? matrix.mul(vec4(0, 0, 0, 1)) : vec4(0, 0, 0, 1)).xyz;
    const cell = floor(root.xz.div(CELL).add(origin).add(0.5)).sub(origin).mul(CELL);
    const away = abs(cell.sub(agent));
    const nearAgent = smoothstep(CELL, BUBBLE_OUTER, max(away.x, away.y)).oneMinus();
    const nearBloom = bloom.z.mul(smoothstep(BLOOM_INNER, BLOOM_OUTER, length(root.xz.sub(bloom.xy))).oneMinus());
    const seen = max(nearAgent, nearBloom);
    const girth = seen.mul(seen.mul(0.45).add(0.55));
    const grown = positionGeometry.mul(vec3(girth, seen, girth));
    const sway = clamp(grown.y.div(WIND.height), 0, 1);
    const phase = clock.mul(WIND.frequency).add(root.x.mul(0.7)).add(root.z.mul(1.3));
    const gust = vec3(sin(phase), 0, cos(phase.mul(0.83)).mul(0.6)).mul(sway.mul(sway).mul(WIND.amplitude));
    const lean = vec3(sin(clock.mul(0.8).add(root.z.mul(0.7))).mul(grown.y).mul(0.009), 0, 0);
    return { position: grown.add(gust).add(lean) };
  });

  const bark = new THREE.Color(BARK);
  bark.setRGB(bark.r / skin.spruce.r, bark.g / skin.spruce.g, bark.b / skin.spruce.b);
  const geometry = w.own(barkTrunk(coniferTreeGeometry(2), 11 * 12, bark));
  const walls: [number, number][] = [];
  for (let y = 0; y < grid.h; y++) for (let x = 0; x < grid.w; x++) if (isWall(grid, x, y)) walls.push([x, y]);
  const trees = w.add(w.own(instancedMesh(geometry, revealed, walls.length * 3)));
  trees.name = 'revealedSpruce';
  trees.frustumCulled = false;
  trees.castShadow = true;
  const transform = new THREE.Object3D(), tint = new THREE.Color();
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

  const outer = w.own(makeSceneryBank({
    name: 'continuingSpruce', geometry, material, tileSize: 12, perTile: 100,
    seed: ctx.seed ^ 0x77c1, colorA: 0xe3e9dd, colorB: 0x9caba3, receiveShadow: true,
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
    const tree = barkTrunk(coniferTreeGeometry(0), 5 * 12, bark);
    const scale = 0.42 + hash2f(i, 10, ctx.seed) * 0.26;
    tree.scale(scale, scale * (1.5 + hash2f(i, 11, ctx.seed) * 0.8), scale);
    tree.translate(Math.sin(i * 13.3) * 2, 0, Math.cos(i * 8.7) * 2);
    groveParts.push({ geometry: tree, color: 0xffffff });
  }
  const groveColors = groveParts.flatMap(({ geometry: tree }) => Array.from(tree.getAttribute('color').array));
  const groveGeometry = w.own(mergeColoredParts(groveParts, 'spruceGrove'));
  groveGeometry.setAttribute('color', new THREE.Float32BufferAttribute(groveColors, 3));
  const groves = w.own(makeSceneryBank({
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
  w.add(groves.mesh);

  const ferns = w.time(w.own(scatterGrass({
    count: 5200, area: { halfX: halfX + 2.5, halfZ: halfZ + 2.5 }, seed: ctx.seed ^ 0x1f4d,
    exclude: excludeCellCenters(grid, 0.35), color: skin.fern, scale: [0.11, 0.27], stretch: [0.7, 1.6],
    wind: 0.04, rimColor: skin.haze, rimStrength: 0.34, hueJitter: 0.24, name: 'forestFerns',
  })));
  w.add(ferns.group);
  const stones = w.time(w.own(scatterRocks({
    count: 240, area: { halfX: halfX + 6, halfZ: halfZ + 6 }, seed: ctx.seed ^ 0x6b23,
    exclude: excludeCellCenters(grid, 0.34), heightAt, color: skin.stone,
    scale: [0.12, 0.34], stretch: [0.45, 0.85], hueJitter: 0.22,
    rimColor: skin.haze, rimStrength: 0.3, castShadow: false, name: 'forestStones',
  })));
  w.add(stones.group);
  const spores = w.time(w.own(makeAmbientParticles({
    preset: 'pollen', count: 340, bounds: { x: 0, y: 1.2, z: 0, halfX, halfY: 1.1, halfZ }, seed: ctx.seed ^ 0x7d05,
    color: skin.motes, opacity: 0.36, size: 0.8, intensity: 1.6, rise: 0.04, swirl: 0.2, name: 'forestFireflies',
  })));
  w.add(spores.points);

  let density = fog.density, lift = 0;
  const handle = {
    rig,
    skin,
    setBubble(x: number, z: number, bloomX: number, bloomZ: number, amount: number) {
      agent.value.set(x, z);
      bloom.value.set(bloomX, bloomZ, amount);
    },
    setFogLift(amount: number) {
      lift = clamp01(amount);
      fog.density = density * (1 - lift * 0.35);
    },
    setView(coverage: ViewCoverage, camera: THREE.Camera) {
      land.setView(coverage, camera, outer, groves);
      density = 6 / coverage.far;
      fog.density = density * (1 - lift * 0.35);
      rig.setShadowFocus({ x: agent.value.x, y: 0.3, z: agent.value.y }, 5);
    },
  };
  return w.seal<WorldCore & typeof handle>(handle, {
    time(t) {
      clock.value = t;
      oakUniforms(material)?.setTime(t);
    },
  });
}
