import * as THREE from 'three';
import { CSM } from 'three/addons/csm/CSM.js';
import { buildAurora } from '@kit/aurora';
import { valueNoise2 } from '@kit/textures';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { winterWeather } from './ambience';
import { FIELD_TEXELS, TILE_CELLS } from './fieldmath';
import { ItemField } from './items';
import { ScentField } from './scentfield';
import type { ViewCoverage } from '@kit/view';

export type ParklandWorld = ReturnType<typeof buildWorld>;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const land = makeLandscape(scene, 'winterCandyLand', ctx.tokens.palettes.night, 'night', {
    sky: { sunDir: [-0.42, 0.17, -0.89], moon: true, sunColor: 0xe2ecff, sunSize: 0.021, sunHdr: 1.4, sunGlow: 0.025, topSpread: 0.36, stars: { intensity: 3.5, density: 180, threshold: 0.97, size: 0.1, twinkle: 0.1, color: 0xdcecff } },
    light: { pointCount: 0, shadowMapSize: 2048 }, fogDensity: 0.00018,
  });
  const { world, sky, rig, fog } = land;
  sky.setColors(0x10142f, 0x80506f, 0x414a6c);
  world.time(world.own(buildAurora(sky, 0xf28ac9, 2.6))).setGain(2.1);
  rig.sun.color.set(0xc4d9ff); rig.sun.intensity = 1.8; rig.sun.castShadow = false;
  rig.hemi.color.set(0xc9c9ef); rig.hemi.groundColor.set(0x6b527c); rig.hemi.intensity = 1.3;
  fog.color.set(0x775970);
  const weather = winterWeather(world, ctx.seed), vision = { value: new Float32Array(9) }, agent = { value: new THREE.Vector2() };
  const heightAt = (x: number, z: number) => 0.45 + (valueNoise2(x / 44, z / 44, 8192, ctx.seed) - 0.5) * 3.6 + (valueNoise2(x / 140, z / 140, 8192, ctx.seed ^ 0x51ed) - 0.5) * 12 + valueNoise2((x + z * 0.4) / 9, z / 22, 8192, ctx.seed ^ 0xa817) ** 3 * 1.4;
  const scent = world.own(new ScentField(ctx.renderer, ctx.seed, ctx.environment)), debug = { value: 0 };
  const terrain = world.own(world.time(makeSceneTerrain({
    name: 'sculptedSnow', seed: ctx.seed, segments: 256, radius: 8192,
    colorA: 0xffffff, colorB: 0xeff3ff,
    height: `return 0.45 + (worldNoise(p / 44.0, 8192.0, ${ctx.seed >>> 0}u) - 0.5) * 3.6 + (worldNoise(p / 140.0, 8192.0, ${(ctx.seed ^ 0x51ed) >>> 0}u) - 0.5) * 12.0 + pow(worldNoise(vec2(p.x + p.y * 0.4, p.y) / vec2(9.0, 22.0), 8192.0, ${(ctx.seed ^ 0xa817) >>> 0}u), 3.0) * 1.4;`,
  }, heightAt, { roughness: 1, toonSteps: 0 })));
  const compile = terrain.material.onBeforeCompile;
  terrain.material.onBeforeCompile = (shader, renderer) => {
    compile(shader, renderer);
    Object.assign(shader.uniforms, { uScent: { value: scent.texture }, uScentOrigin: { value: scent.origin }, uDebug: debug, uVision: vision, uAgent: agent });
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
uniform sampler2D uScent;
uniform vec2 uScentOrigin;
uniform float uDebug, uVision[9];
uniform vec2 uAgent;

`)
      .replace('#include <color_fragment>', `#include <color_fragment>
float drift = landNoise(vLandPosition.xz * vec2(0.22, 0.08));
float crust = landNoise(vLandPosition.xz * 2.5);
vec3 snowColor = mix(vec3(0.72, 0.83, 1.0), vec3(1.0, 0.91, 0.97), drift);
diffuseColor.rgb *= snowColor * (0.94 + 0.06 * crust);
`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
vec2 scentUV = (vLandPosition.xz - uScentOrigin + 0.5) / ${FIELD_TEXELS.toFixed(1)};
vec4 field = texture2D(uScent, scentUV);
vec2 edge = smoothstep(vec2(0.0), vec2(0.025), scentUV) * smoothstep(vec2(0.0), vec2(0.025), 1.0 - scentUV);
float inside = edge.x * edge.y * field.a;
vec3 concentration = max(field.rgb, vec3(0.0));
float mass = dot(concentration, vec3(1.0));
vec3 scentColor = concentration / max(max(concentration.r, max(concentration.g, concentration.b)), 0.00001);
float strength = (1.0 - exp(-mass * 0.38)) * inside;
float level = log2(1.0 + mass * 4.0) * 1.6;
float width = max(fwidth(level), 0.015);
float contour = 1.0 - smoothstep(width * 0.45, width * 1.1, abs(fract(level + 0.5) - 0.5));
contour *= smoothstep(0.2, 0.7, level) * (1.0 - smoothstep(0.35, 1.0, width));
diffuseColor.rgb = mix(diffuseColor.rgb, scentColor * 0.45, strength * (0.035 + uDebug * 0.42));
totalEmissiveRadiance += scentColor * strength * (0.008 + contour * (0.055 + 0.36 * uDebug));
vec2 gridWidth = max(fwidth(vLandPosition.xz), vec2(0.00001));
vec2 grid = abs(fract(vLandPosition.xz) - 0.5) / gridWidth;
float line = (1.0 - smoothstep(0.25, 1.25, min(grid.x, grid.y))) * (1.0 - smoothstep(0.25, 0.9, max(gridWidth.x, gridWidth.y)));
vec2 survey = abs(fract((vLandPosition.xz + 0.5) / 16.0 - 0.5) - 0.5) * 16.0 / gridWidth;
float major = 1.0 - smoothstep(0.5, 1.5, min(survey.x, survey.y));
totalEmissiveRadiance += vec3(0.2, 0.59, 0.7) * max(line * 0.48, major * 0.72) * uDebug;
vec2 localVision = vLandPosition.xz - uAgent, cell = floor(localVision + 0.5);
if (all(lessThan(abs(cell), vec2(1.5))) && dot(cell, cell) > 0.0) {
  float signal = 1.0 - exp(-uVision[int((cell.y + 1.0) * 3.0 + cell.x + 1.0)]);
  float border = 0.5 - max(abs(localVision.x - cell.x), abs(localVision.y - cell.y));
  float edgeWidth = max(fwidth(border), 0.015);
  float outline = 1.0 - smoothstep(edgeWidth, edgeWidth * 2.2, border);
  vec3 visionColor = vec3(0.13, 0.66, 0.84);
  totalEmissiveRadiance += visionColor * (0.09 + uDebug * 0.85) * (outline + signal * 0.14);
  diffuseColor.rgb = mix(diffuseColor.rgb, visionColor * 0.5, (0.03 + uDebug * 0.16) * (0.3 + signal));
}
`);
  };
  world.add(terrain.mesh);
  const items = new ItemField(world, ctx.seed, heightAt);
  world.own({ dispose() {}, staticInstances: items.staticInstances });
  let shadows: CSM | undefined, projection = '';
  world.own({ dispose() { shadows?.remove(); for (const light of shadows?.lights ?? []) light.dispose(); shadows?.dispose(); } });
  const handle = {
    items, scent, heightAt, debug, vision, agent,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      land.setView(coverage, camera, terrain);
      weather.setView(camera.position.x, heightAt(camera.position.x, camera.position.z), camera.position.z);
      if (!shadows) {
        rig.sun.removeFromParent();
        shadows = new CSM({ camera, parent: world.group, cascades: 3, maxFar: 2304, shadowMapSize: 2048, lightDirection: rig.sunDirection.clone().negate(), lightIntensity: 1.8, lightMargin: 80, lightFar: 6000, shadowBias: -0.000015, mode: 'custom', customSplitsCallback: (_count, near, far, breaks) => { for (const depth of [Math.min(40, far * 0.16), Math.min(260, far * 0.52), far]) breaks.push(Math.max(near, depth) / far); } });
        shadows.fade = true;
        for (const light of shadows.lights) { light.color.copy(rig.sun.color); light.shadow.normalBias = 0.025; }
        const materials = new Set<THREE.Material>();
        scene.traverse(object => { if (object instanceof THREE.Mesh) for (const material of Array.isArray(object.material) ? object.material : [object.material]) if (material instanceof THREE.MeshStandardMaterial) materials.add(material.userData.sourceMaterial ?? material); });
        for (const material of items.materials) if (material instanceof THREE.MeshStandardMaterial) materials.add(material);
        for (const material of materials) {
          const before = material.onBeforeCompile, key = material.customProgramCacheKey(); shadows.setupMaterial(material); const cascade = material.onBeforeCompile;
          const cascadeUniforms: Record<string, THREE.IUniform> = {};
          material.customProgramCacheKey = () => `${key}|cascades`;
          material.onBeforeCompile = (shader, renderer) => {
            before.call(material, shader, renderer); cascade.call(material, shader, renderer);
            for (const name of ['CSM_cascades', 'cameraNear', 'shadowFar']) shader.uniforms[name] = cascadeUniforms[name] ??= shader.uniforms[name]!;

          };
        }
      }
      const next = `${camera.aspect}:${camera.fov}:${camera.near}:${camera.far}`;
      if (projection !== next) { shadows.updateFrustums(); projection = next; }
      shadows.update();
      const shadowDepth = Math.round((coverage.x * rig.sunDirection.x + coverage.z * rig.sunDirection.z) / TILE_CELLS) * TILE_CELLS;
      for (let i = 0; i < shadows.cascades; i++) {
        const light = shadows.lights[i]!, frustum = shadows.frustums[i]!, matrix = camera.matrixWorld.elements;
        const shift = shadowDepth + light.shadow.camera.far / 2 - light.position.dot(rig.sunDirection);
        light.position.addScaledVector(rig.sunDirection, shift); light.target.position.addScaledVector(rig.sunDirection, shift);
        let bottom = Infinity;
        for (const vertices of [frustum.vertices.near, frustum.vertices.far]) for (const vertex of vertices) bottom = Math.min(bottom, matrix[1]! * vertex.x + matrix[5]! * vertex.y + matrix[9]! * vertex.z + matrix[13]!);
        light.shadow.autoUpdate = bottom <= items.canopyHeight + (3.6 + 12) / 2;
        light.shadow.needsUpdate = !light.shadow.map;
        light.updateMatrixWorld(); light.target.updateMatrixWorld(); light.shadow.updateMatrices(light);
      }
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
