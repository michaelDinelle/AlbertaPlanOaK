import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, cos, float, length, max, mod, positionView, pow, screenDPR, sin, smoothstep, step, texture, uniform, uv, varying, vec2, vec3 } from 'three/tsl';
import { fbm } from './textures';
import { mulberry32 } from './rng';

export type ParticlePreset = 'fireflies' | 'pollen' | 'snow' | 'embers' | 'bubbles' | 'dust';

export interface ParticlePresetSpec {
  count: number;
  size: number;
  color: number;
  intensity: number;
  rise: number;
  swirl: number;
  swirlFreq: number;
  twinkle: number;
  softness: number;
  additive: boolean;
  opacity: number;
}

export const PARTICLE_PRESETS: Record<ParticlePreset, ParticlePresetSpec> = Object.freeze({
  fireflies: { count: 220, size: 9, color: 0x9ff0a8, intensity: 2.6, rise: 0.06, swirl: 0.45, swirlFreq: 0.55, twinkle: 0.85, softness: 2.4, additive: true, opacity: 1 },
  pollen: { count: 900, size: 4, color: 0xffe9b0, intensity: 1.5, rise: 0.09, swirl: 0.3, swirlFreq: 0.35, twinkle: 0.35, softness: 2.0, additive: true, opacity: 0.75 },
  snow: { count: 1400, size: 6, color: 0xf2f8ff, intensity: 1.1, rise: -0.55, swirl: 0.26, swirlFreq: 0.7, twinkle: 0.12, softness: 1.6, additive: false, opacity: 0.85 },
  embers: { count: 420, size: 6, color: 0xff9a4d, intensity: 3.0, rise: 0.65, swirl: 0.22, swirlFreq: 1.1, twinkle: 0.7, softness: 2.6, additive: true, opacity: 1 },
  bubbles: { count: 600, size: 7, color: 0xbfe8ff, intensity: 1.4, rise: 0.75, swirl: 0.16, swirlFreq: 0.9, twinkle: 0.25, softness: 1.4, additive: true, opacity: 0.7 },
  dust: { count: 1100, size: 3, color: 0xffe0b8, intensity: 1.2, rise: 0.02, swirl: 0.18, swirlFreq: 0.22, twinkle: 0.2, softness: 1.8, additive: true, opacity: 0.6 },
});

export interface ParticleBounds {
  x?: number;
  y?: number;
  z?: number;
  halfX: number;
  halfY: number;
  halfZ: number;
}

export interface AmbientParticlesOptions {
  preset: ParticlePreset;
  bounds: ParticleBounds;
  seed: number;
  count?: number;
  color?: THREE.ColorRepresentation;
  size?: number;
  intensity?: number;
  opacity?: number;
  rise?: number;
  swirl?: number;
  twinkle?: number;
  name?: string;
}

export interface AmbientParticlesHandle {
  points: THREE.Sprite;
  material: THREE.PointsNodeMaterial;
  count: number;
  setTime(t: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  setIntensity(i: number): void;
  setDensity(f01: number): void;
  dispose(): void;
}

const MAX_POINT_PIXELS = 511;

export const pointSize = (pixels: THREE.Node<'float'>): THREE.Node<'float'> => clamp(pixels.mul(screenDPR), 1, MAX_POINT_PIXELS).div(screenDPR);

export function makeAmbientParticles(opts: AmbientParticlesOptions): AmbientParticlesHandle {
  const spec = PARTICLE_PRESETS[opts.preset];
  const count = Math.max(0, Math.round(opts.count ?? spec.count));
  const b = opts.bounds;
  const cx = b.x ?? 0;
  const cy = b.y ?? 0;
  const cz = b.z ?? 0;

  const positions = new Float32Array(Math.max(1, count) * 3);
  const seeds = new Float32Array(Math.max(1, count));
  const phases = new Float32Array(Math.max(1, count));
  const sizes = new Float32Array(Math.max(1, count));

  const rng = mulberry32((opts.seed | 0) ^ 0x1f83d9ab);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (rng() * 2 - 1) * b.halfX;
    positions[i * 3 + 1] = (rng() * 2 - 1) * b.halfY;
    positions[i * 3 + 2] = (rng() * 2 - 1) * b.halfZ;
    seeds[i] = rng();
    phases[i] = rng();
    sizes[i] = 0.55 + rng() * 0.9;
  }

  const time = uniform(0), size = uniform(opts.size ?? spec.size), halfY = uniform(b.halfY), rise = uniform(opts.rise ?? spec.rise);
  const swirl = uniform(opts.swirl ?? spec.swirl), swirlFreq = uniform(spec.swirlFreq), twinkle = uniform(opts.twinkle ?? spec.twinkle), density = uniform(1);
  const tint = uniform(new THREE.Color(opts.color ?? spec.color)), intensity = uniform(opts.intensity ?? spec.intensity), softness = uniform(spec.softness), opacity = uniform(opts.opacity ?? spec.opacity);
  const p = attribute<'vec3'>('aPosition', 'vec3'), seed = attribute<'float'>('aSeed', 'float');
  const s = time.mul(swirlFreq).add(attribute<'float'>('aPhase', 'float').mul(6.2831853));
  const live = step(seed, density);
  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: spec.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    fog: false,
    sizeAttenuation: false,
    alphaTest: 0.004,
    positionNode: vec3(
      p.x.add(sin(s).mul(swirl).mul(seed.add(0.45))),
      mod(p.y.add(halfY).add(rise.mul(time)), max(halfY.mul(2), 0.0001)).sub(halfY),
      p.z.add(cos(s.mul(0.83).add(seed.mul(3.1))).mul(swirl).mul(seed.add(0.45))),
    ),
    sizeNode: pointSize(live.mul(size).mul(attribute<'float'>('aSize', 'float')).mul(float(60).div(max(positionView.z.negate(), 0.001)))),
    colorNode: tint.mul(intensity),
    opacityNode: pow(clamp(length(uv().sub(0.5)).mul(2).oneMinus(), 0, 1), softness)
      .mul(varying(twinkle.mul(0.5).mul(sin(time.mul(2.3).add(seed.mul(31))).add(1)).oneMinus().mul(live))).mul(opacity),
  });
  material.name = opts.name ?? `oakParticles:${opts.preset}`;

  const points = new THREE.Sprite(material as unknown as THREE.SpriteMaterial);
  const geometry = points.geometry.clone();
  geometry.setAttribute('aPosition', new THREE.InstancedBufferAttribute(positions, 3));
  geometry.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seeds, 1));
  geometry.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1));
  geometry.setAttribute('aSize', new THREE.InstancedBufferAttribute(sizes, 1));
  points.geometry = geometry;
  points.count = count;
  points.name = material.name;
  points.position.set(cx, cy, cz);
  points.frustumCulled = false;
  points.visible = count > 0;

  return {
    points,
    material,
    count,
    setTime(t) {
      time.value = t;
    },
    setColor(color) {
      tint.value.set(color);
    },
    setIntensity(i) {
      intensity.value = i;
    },
    setDensity(f01) {
      density.value = f01 < 0 ? 0 : f01 > 1 ? 1 : f01;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      points.removeFromParent();
    },
  };
}

export interface MistPlanesOptions {
  size: number | readonly [number, number];
  color: THREE.ColorRepresentation;
  seed: number;
  layers?: number;
  baseY?: number;
  spread?: number;
  opacity?: number;
  speed?: number;
  scale?: number;
  nearFade?: number;
  name?: string;
}

export interface MistPlanesHandle {
  group: THREE.Group;
  material: THREE.MeshBasicNodeMaterial;
  layers: number;
  setTime(t: number): void;
  setOpacity(o: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  dispose(): void;
}

export function makeMistPlanes(opts: MistPlanesOptions): MistPlanesHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'oakMist';
  const layers = Math.max(0, Math.round(opts.layers ?? 5));
  const sizeX = typeof opts.size === 'number' ? opts.size : opts.size[0];
  const sizeZ = typeof opts.size === 'number' ? opts.size : opts.size[1];
  const baseY = opts.baseY ?? 0.15;
  const spread = opts.spread ?? 1.4;

  const noise = fbm(opts.seed ^ 0x2c1b3a95, { size: 256, frequency: 4, octaves: 4, gain: 0.55 });
  const tint = uniform(new THREE.Color(opts.color)), time = uniform(0), opacity = uniform(opts.opacity ?? 0.28);
  const speed = uniform(opts.speed ?? 0.012), scale = uniform(opts.scale ?? 2.5), nearFade = uniform(opts.nearFade ?? 3);
  const st = uv().mul(scale), drift = time.mul(speed), e = abs(uv().sub(0.5)).mul(2);
  const a = texture(noise, st.add(vec2(drift, drift.mul(0.6)))).r;
  const b = texture(noise, st.mul(1.7).sub(vec2(drift.mul(0.8), 0))).r;
  const material = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
    alphaTest: 0.004,
    colorNode: tint,
    opacityNode: clamp(a.mul(0.65).add(b.mul(0.55)).sub(0.28), 0, 1)
      .mul(smoothstep(0.55, 1, e.x).oneMinus().mul(smoothstep(0.55, 1, e.y).oneMinus()))
      .mul(smoothstep(0, max(nearFade, 0.001), positionView.z.negate()))
      .mul(opacity),
  });
  material.name = 'oakMistPlane';

  const geometry = new THREE.PlaneGeometry(sizeX, sizeZ);
  geometry.rotateX(-Math.PI / 2);
  const rng = mulberry32((opts.seed | 0) ^ 0x6b43a9f1);

  for (let i = 0; i < layers; i++) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `oakMistPlane${i}`;
    const t = layers > 1 ? i / (layers - 1) : 0;
    mesh.position.y = baseY + t * spread;
    mesh.position.x = (rng() * 2 - 1) * sizeX * 0.04;
    mesh.position.z = (rng() * 2 - 1) * sizeZ * 0.04;
    mesh.rotation.y = rng() * Math.PI * 2;
    mesh.renderOrder = 1;
    mesh.frustumCulled = false;
    group.add(mesh);
  }
  group.visible = layers > 0;

  return {
    group,
    material,
    layers,
    setTime(t) {
      time.value = t;
    },
    setOpacity(o) {
      opacity.value = o;
    },
    setColor(color) {
      tint.value.set(color);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      noise.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
