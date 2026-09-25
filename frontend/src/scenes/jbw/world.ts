import * as THREE from 'three/webgpu';
import { abs, all, clamp, dot, exp, float, floor, fract, fwidth, int, log2, max, min, mix, select, smoothstep, texture, uniform, uniformArray, vec2, vec3 } from 'three/tsl';
import { buildAurora } from '@kit/aurora';
import { makeHeightField } from '@kit/heightfield';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { winterWeather } from './ambience';
import { FIELD_TEXELS } from './fieldmath';
import { ItemField } from './items';
import { ScentField } from './scentfield';
import type { ViewCoverage } from '@kit/view';

export type ParklandWorld = ReturnType<typeof buildWorld>;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const land = makeLandscape(scene, 'winterCandyLand', ctx.tokens.palettes.night, 'night', {
    sky: { sunDir: [-0.42, 0.17, -0.89], moon: true, sunColor: 0xe2ecff, sunSize: 0.021, sunHdr: 1.4, sunGlow: 0.025, topSpread: 0.36, stars: { intensity: 3.5, density: 180, threshold: 0.97, size: 0.1, twinkle: 0.1, color: 0xdcecff } },
    light: { pointCount: 0 }, fogDensity: 0.00018,
  });
  const { world, sky, rig, fog } = land;
  sky.setColors(0x10142f, 0x80506f, 0x414a6c);
  world.time(world.own(buildAurora(sky, 0xf28ac9, 2.6))).setGain(2.1);
  rig.sun.color.set(0xc4d9ff); rig.sun.intensity = 1.8; rig.sun.shadow.normalBias = 0.03;
  rig.hemi.color.set(0xc9c9ef); rig.hemi.groundColor.set(0x6b527c); rig.hemi.intensity = 1.3;
  fog.color.set(0x775970);
  const weather = winterWeather(world, ctx.seed), visionNode = uniformArray(new Array<number>(9).fill(0), 'float' as const), vision = { value: visionNode.array as number[] }, agent = uniform(new THREE.Vector2());
  const field = makeHeightField({ base: 0.45, octaves: [{ wavelength: 140, amplitude: 6, seed: ctx.seed ^ 0x51ed }, { wavelength: 44, amplitude: 1.8, seed: ctx.seed }, { wavelength: 14, amplitude: 0.4, seed: ctx.seed ^ 0xa817 }] });
  const heightAt = field.heightAt;
  const scent = world.own(new ScentField(ctx.renderer, ctx.seed, ctx.environment)), debug = uniform(0), origin = uniform(scent.origin);
  let emissive: THREE.Node<'vec3'> = vec3(0);
  const terrain = world.own(world.time(makeSceneTerrain({
    name: 'sculptedSnow', seed: ctx.seed, segments: 256, radius: 8192,
    colorA: 0xffffff, colorB: 0xeff3ff, height: field.height,
    detail: (s) => {
      const p = s.p.xz;
      const snow = mix(vec3(0.72, 0.83, 1.0), vec3(1.0, 0.91, 0.97), s.noise(p.mul(vec2(0.22, 0.08))));
      const lit = snow.mul(s.noise(p.mul(2.5)).mul(0.06).add(0.94)).mul(s.color);
      const scentUV = p.sub(origin).add(0.5).div(FIELD_TEXELS);
      const sample = texture(scent.texture, scentUV);
      const edge = smoothstep(vec2(0), vec2(0.025), scentUV).mul(smoothstep(vec2(0), vec2(0.025), scentUV.oneMinus()));
      const concentration = max(sample.rgb, vec3(0));
      const mass = dot(concentration, vec3(1));
      const scentColor = concentration.div(max(max(concentration.r, max(concentration.g, concentration.b)), 0.00001));
      const strength = exp(mass.mul(-0.38)).oneMinus().mul(edge.x.mul(edge.y).mul(sample.a));
      const level = log2(mass.mul(4).add(1)).mul(1.6);
      const width = max(fwidth(level), 0.015);
      const contour = smoothstep(width.mul(0.45), width.mul(1.1), abs(fract(level.add(0.5)).sub(0.5))).oneMinus().mul(smoothstep(0.2, 0.7, level).mul(smoothstep(0.35, 1, width).oneMinus()));
      const gridWidth = max(fwidth(p), vec2(0.00001));
      const grid = abs(fract(p).sub(0.5)).div(gridWidth);
      const line = smoothstep(0.25, 1.25, min(grid.x, grid.y)).oneMinus().mul(smoothstep(0.25, 0.9, max(gridWidth.x, gridWidth.y)).oneMinus());
      const survey = abs(fract(p.add(0.5).div(16).sub(0.5)).sub(0.5)).mul(16).div(gridWidth);
      const major = smoothstep(0.5, 1.5, min(survey.x, survey.y)).oneMinus();
      const local = p.sub(agent), cell = floor(local.add(0.5));
      const within = select(all(abs(cell).lessThan(vec2(1.5))).and(dot(cell, cell).greaterThan(0)), float(1), float(0));
      const signal = exp(visionNode.element(int(clamp(cell.y.add(1).mul(3).add(cell.x).add(1), 0, 8))).negate()).oneMinus();
      const border = float(0.5).sub(max(abs(local.x.sub(cell.x)), abs(local.y.sub(cell.y))));
      const edgeWidth = max(fwidth(border), 0.015);
      const outline = smoothstep(edgeWidth, edgeWidth.mul(2.2), border).oneMinus();
      const visionColor = vec3(0.13, 0.66, 0.84);
      emissive = scentColor.mul(strength).mul(contour.mul(debug.mul(0.36).add(0.055)).add(0.008))
        .add(vec3(0.2, 0.59, 0.7).mul(max(line.mul(0.48), major.mul(0.72))).mul(debug))
        .add(visionColor.mul(debug.mul(0.85).add(0.09)).mul(outline.add(signal.mul(0.14))).mul(within));
      return mix(mix(lit, scentColor.mul(0.45), strength.mul(debug.mul(0.42).add(0.035))), visionColor.mul(0.5), debug.mul(0.16).add(0.03).mul(signal.add(0.3)).mul(within));
    },
  }, heightAt, { roughness: 1, toonSteps: 0 })));
  terrain.material.emissiveNode = emissive;
  world.add(terrain.mesh);
  const items = new ItemField(world, ctx.seed, heightAt);
  const handle = {
    items, scent, heightAt, debug, vision, agent, lights: rig,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      land.setView(coverage, camera, terrain);
      weather.setView(camera.position.x, heightAt(camera.position.x, camera.position.z), camera.position.z);
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
