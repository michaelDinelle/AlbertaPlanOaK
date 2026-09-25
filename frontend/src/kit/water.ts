import * as THREE from 'three/webgpu';
import { Fn, abs, attribute, cameraPosition, cameraProjectionMatrix, clamp, cos, dot, exp, float, fract, fwidth, length, max, min, mix, modelViewMatrix, modelWorldMatrix, normalize, positionGeometry, positionView, positionWorld, pow, reflector, sin, smoothstep, uniform, varying, vec2, vec3, vec4 } from 'three/tsl';
import type { Palette } from './tokens';
import type { ViewCoverage } from './view';

type N<T extends string> = THREE.Node<T>;

const REFLECTION_DISTORTION = 0.055;

export interface WaterOptions {
  size: number | readonly [number, number];
  palette: Palette;
  segments?: number;
  shallow?: THREE.ColorRepresentation;
  deep?: THREE.ColorRepresentation;
  sky?: THREE.ColorRepresentation;
  caustic?: THREE.ColorRepresentation;
  amplitude?: number;
  waveScale?: number;
  speed?: number;
  fresnel?: number;
  caustics?: number;
  depthFalloff?: number;
  opacity?: number;
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  level?: number;
  name?: string;
  bedHeight?: (x: number, z: number) => number;
}

export interface WaterFluid {
  sample(p: N<'vec2'>): N<'vec4'>;
  flow: N<'vec2'>;
}

export type WaterHandle = ReturnType<typeof makeWater>;

type Wave = readonly [number, number, number, number];

const WAVES: readonly Wave[] = [
  [1.0, 0.15, 0.9, 0.55],
  [-0.35, 1.0, 1.35, 0.42],
  [0.7, -0.7, 2.6, 0.95],
  [-0.9, -0.45, 4.1, 1.35],
];
const WAVE_AMPS: readonly number[] = [0.55, 0.3, 0.11, 0.06];

export function waterHeightAt(x: number, z: number, time: number, amplitude: number, waveScale: number, speed: number): number {
  let h = 0;
  for (let i = 0; i < WAVES.length; i++) {
    const w = WAVES[i] as Wave;
    const amp = WAVE_AMPS[i] as number;
    h += amp * Math.sin((x * w[0] + z * w[1]) * w[2] * waveScale + time * w[3] * speed);
  }
  return h * amplitude;
}

const heading = (w: Wave) => vec2(w[0], w[1]);

function waveHeight(p: N<'vec2'>, t: N<'float'>, scale: N<'float'>, speed: N<'float'>): N<'float'> {
  return WAVES.reduce<N<'float'>>((h, w, i) => h.add(sin(dot(p, heading(w)).mul(w[2]).mul(scale).add(t.mul(w[3]).mul(speed))).mul(WAVE_AMPS[i]!)), float(0));
}

function waveSlope(p: N<'vec2'>, t: N<'float'>, scale: N<'float'>, speed: N<'float'>): N<'vec2'> {
  return WAVES.reduce<N<'vec2'>>((g, w, i) => {
    const k = scale.mul(w[2]);
    return g.add(cos(dot(p, heading(w)).mul(k).add(t.mul(w[3]).mul(speed))).mul(WAVE_AMPS[i]!).mul(k).mul(heading(w)));
  }, vec2(0));
}

function caustic(p: N<'vec2'>, t: N<'float'>): N<'float'> {
  const a = p.mul(1.7).add(vec2(t.mul(0.09), t.mul(0.06))), b = p.mul(2.3).sub(vec2(t.mul(0.07), t.mul(0.11)));
  const ca = abs(fract(a.x.add(a.y)).sub(0.5)).add(abs(fract(a.x.sub(a.y)).sub(0.5)));
  const cb = abs(fract(b.x.add(b.y)).sub(0.5)).add(abs(fract(b.x.sub(b.y)).sub(0.5)));
  return pow(clamp(min(ca, cb).mul(2.4), 0, 1).oneMinus(), 3);
}

interface Surface {
  world: ((p: N<'vec2'>) => N<'float'>) | null;
  fluid: WaterFluid | null;
  shade(): void;
}

const surfaces = new WeakMap<THREE.Material, Surface>();

export function makeWater(opts: WaterOptions) {
  const sizeX = typeof opts.size === 'number' ? opts.size : opts.size[0];
  const sizeZ = typeof opts.size === 'number' ? opts.size : opts.size[1];
  const seg = Math.max(2, Math.round(opts.segments ?? 96));
  const p = opts.palette;

  const amplitude = opts.amplitude ?? 0.035;
  const waveScale = opts.waveScale ?? 1;
  const speed = opts.speed ?? 1;

  const sun = opts.sunDir instanceof THREE.Vector3 ? opts.sunDir.clone() : new THREE.Vector3(...(opts.sunDir ?? [0, 1, 0]));
  if (sun.lengthSq() < 1e-12) sun.set(0, 1, 0);
  sun.normalize();

  const uniforms = {
    uTime: uniform(0),
    uShallow: uniform(new THREE.Color(opts.shallow ?? p.skyHorizon).lerp(new THREE.Color(p.groundB), 0.35)),
    uDeep: uniform(new THREE.Color(opts.deep ?? p.fog).lerp(new THREE.Color(p.skyBottom), 0.55)),
    uSky: uniform(new THREE.Color(opts.sky ?? p.skyHorizon)),
    uCaustic: uniform(new THREE.Color(opts.caustic ?? p.sun)),
    uAmplitude: uniform(amplitude),
    uWaveScale: uniform(waveScale),
    uSpeed: uniform(speed),
    uFresnel: uniform(opts.fresnel ?? 0.55),
    uCausticStrength: uniform(opts.caustics ?? 0.35),
    uDepthFalloff: uniform(opts.depthFalloff ?? 9),
    uOpacity: uniform(opts.opacity ?? 0.86),
    uSunDir: uniform(sun),
    uSpecular: uniform(1.6),
  };

  const geometry = new THREE.PlaneGeometry(sizeX, sizeZ, seg, seg);
  geometry.name = 'oakWater';
  if (opts.bedHeight) {
    const positions = geometry.getAttribute('position'), depths = new Float32Array(positions.count);
    const stretch = (p: number, size: number) => Math.sign(p) * (4 * Math.abs(p * 2 / size) + (size / 2 - 4) * Math.abs(p * 2 / size) ** 3);
    for (let i = 0; i < positions.count; i++) {
      const x = stretch(positions.getX(i), sizeX), z = -stretch(positions.getY(i), sizeZ);
      positions.setXY(i, x, -z);
      depths[i] = (opts.level ?? 0) - opts.bedHeight(x, z);
    }
    geometry.setAttribute('waterDepth', new THREE.BufferAttribute(depths, 1));
    const indices: number[] = [], source = geometry.index!.array;
    for (let i = 0; i < source.length; i += 3) {
      const a = source[i]!, b = source[i + 1]!, c = source[i + 2]!;
      if (Math.max(depths[a]!, depths[b]!, depths[c]!) > -0.3) indices.push(a, b, c);
    }
    geometry.setIndex(indices);
  }
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.NodeMaterial();
  material.name = opts.name ?? 'oakWater';
  material.transparent = true;
  material.depthWrite = false;
  const mesh = new THREE.Mesh(geometry, material);
  const mirror = opts.bedHeight ? reflector({ resolutionScale: 0.5, samples: 4 }) : null;
  if (mirror) {
    mirror.target.rotation.x = -Math.PI / 2;
    mesh.add(mirror.target);
  }

  const u = uniforms;
  const surface: Surface = {
    world: null,
    fluid: null,
    shade() {
      const { world, fluid } = surface;
      const wp = modelWorldMatrix.mul(vec4(positionGeometry, 1)).xz;
      const height = varying(fluid ? fluid.sample(wp).r : waveHeight(wp, u.uTime, u.uWaveScale, u.uSpeed).mul(u.uAmplitude));
      const lean = waveSlope(wp, u.uTime, u.uWaveScale, u.uSpeed).mul(u.uAmplitude);
      const normal = varying(normalize(vec3(lean.x.negate(), 1, lean.y.negate())));
      const clip = cameraProjectionMatrix.mul(modelViewMatrix.mul(vec4(positionGeometry, 1)));
      const projected = varying(vec3(clip.w.sub(clip.x).mul(0.5), clip.y.add(clip.w).mul(0.5), clip.w));

      const at = positionWorld.xz, view = normalize(cameraPosition.sub(positionWorld)), footprint = length(fwidth(at));
      const depth = attribute<'float'>('waterDepth', 'float').add(height), flowing = fluid?.sample(at);
      const wave = WAVES.reduce<N<'vec3'>>((sum, w, i) => {
        const k = u.uWaveScale.mul(w[2]), phase = dot(at, heading(w)).mul(k).add(u.uTime.mul(w[3]).mul(u.uSpeed));
        const size = u.uAmplitude.mul(WAVE_AMPS[i]!).div(footprint.mul(footprint).mul(k).mul(k).add(1));
        return sum.add(vec3(cos(phase).mul(size).mul(k).mul(heading(w)), sin(phase).mul(size)));
      }, vec3(0));
      const slope = (flowing?.gb ?? waveSlope(at, u.uTime, u.uWaveScale, u.uSpeed).mul(u.uAmplitude))
        .add(waveSlope(fluid ? at.sub(fluid.flow.mul(u.uTime)) : at, u.uTime, u.uWaveScale.mul(5), u.uSpeed.mul(1.3)).mul(u.uAmplitude).mul(0.035).div(footprint.mul(8).add(1)));
      const n = normalize(world ? vec3(wave.x.negate(), 1, wave.y.negate()) : mirror ? vec3(slope.x.negate(), 1, slope.y.negate()) : normal);

      const byDepth = exp((world ? max(0, world(at)) : mirror ? depth : positionView.z.negate()).negate().div(max(u.uDepthFalloff, 0.001))).oneMinus();
      const byTrough = clamp(float(0.5).sub((world ? wave.z : height).mul(6)), 0, 1);
      const glint = caustic(at.mul(u.uWaveScale), u.uTime.mul(u.uSpeed));
      const water = mix(u.uShallow, u.uDeep, clamp(byDepth.mul(0.75).add(byTrough.mul(0.35)), 0, 1))
        .add(u.uCaustic.mul((world ? glint.mul(smoothstep(0.12, 0.7, footprint.mul(u.uWaveScale)).oneMinus()) : glint).mul(u.uCausticStrength).mul(byDepth.mul(0.7).oneMinus())));

      const f = mirror ? float(0.975).mul(pow(max(dot(n, view), 0).oneMinus(), 5)).add(0.025) : pow(clamp(dot(n, view).oneMinus(), 0, 1), 3);
      const coverage = f.add(byDepth.mul(f.oneMinus()));
      if (mirror) {
        const uv = clamp(projected.xy.div(projected.z).add(n.xz.mul(REFLECTION_DISTORTION).mul(smoothstep(0, 0.8, depth))), 0.002, 0.998);
        mirror.uvNode = vec2(uv.x, uv.y.oneMinus());
      }
      const lit = mirror ? mirror.rgb.mul(f).add(water.mul(byDepth).mul(f.oneMinus())).div(max(coverage, 0.001)) : mix(water, u.uSky, f.mul(u.uFresnel));
      const spec = pow(clamp(dot(n, normalize(normalize(u.uSunDir).add(view))), 0, 1), world ? 36 : 96).mul(u.uSpecular);
      const shine = lit.add(u.uSky.mul(world ? spec.mul(0.4).div(footprint.mul(footprint).mul(4).add(1)) : spec));
      const col = mirror && flowing ? shine.add(vec3(0.4, 0.48, 0.46).mul(flowing.a).mul(0.15)) : shine;
      const alpha = mirror ? coverage.mul(smoothstep(0, 0.035, depth)) : clamp(u.uOpacity.add(f.mul(0.35)), 0, 1);

      material.positionNode = world ? null : positionGeometry.add(vec3(0, height, 0));
      material.fragmentNode = Fn(() => {
        if (mirror) depth.lessThanEqual(0).discard();
        return vec4(col, alpha);
      })();
      material.needsUpdate = true;
    },
  };
  surface.shade();
  surfaces.set(material, surface);
  mesh.name = opts.name ?? 'oakWater';
  mesh.position.y = opts.level ?? 0;
  mesh.receiveShadow = false;
  mesh.castShadow = false;
  mesh.renderOrder = 1;

  return {
    mesh,
    material,
    uniforms,
    setTime(t: number) {
      uniforms.uTime.value = t;
    },
    setLevel(y: number) {
      mesh.position.y = y;
    },
    setColors(shallow: THREE.ColorRepresentation, deep: THREE.ColorRepresentation) {
      uniforms.uShallow.value.set(shallow);
      uniforms.uDeep.value.set(deep);
    },
    setSunDirection(x: number, y: number, z: number) {
      uniforms.uSunDir.value.set(x, y, z);
      if (uniforms.uSunDir.value.lengthSq() < 1e-12) uniforms.uSunDir.value.set(0, 1, 0);
      uniforms.uSunDir.value.normalize();
    },
    sampleHeight(x: number, z: number, time: number) {
      return mesh.position.y + waterHeightAt(x, z, time, amplitude, waveScale, speed);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      mirror?.dispose();
      mesh.removeFromParent();
    },
  };
}

export function flowWater(water: WaterHandle, fluid: WaterFluid): void {
  const surface = surfaces.get(water.material)!;
  surface.fluid = fluid;
  surface.shade();
}

export function extendWater(water: WaterHandle, coverage: ViewCoverage, width: number, depth: number, bedDepth: (p: N<'vec2'>) => N<'float'>): void {
  const surface = surfaces.get(water.material)!;
  if (!surface.world) {
    surface.world = bedDepth;
    surface.shade();
  }
  water.mesh.position.x = coverage.x;
  water.mesh.position.z = coverage.z;
  water.mesh.scale.set(coverage.radius * 2 / width, 1, coverage.radius * 2 / depth);
  water.mesh.frustumCulled = false;
}
