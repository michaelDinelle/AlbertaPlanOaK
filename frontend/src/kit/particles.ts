import * as THREE from 'three';
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
  name?: string;
}

export interface AmbientParticlesHandle {
  points: THREE.Points;
  material: THREE.ShaderMaterial;
  count: number;
  setTime(t: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  setIntensity(i: number): void;
  setDensity(f01: number): void;
  setPixelRatio(ratio: number): void;
  dispose(): void;
}

const AMBIENT_VERT =  `
#include <common>
uniform float uTime;
uniform float uPixelRatio;
uniform float uSize;
uniform vec3 uHalf;
uniform float uRise;
uniform float uSwirl;
uniform float uSwirlFreq;
uniform float uTwinkle;
uniform float uDensity;
attribute float aSeed;
attribute float aPhase;
attribute float aSize;
varying float vAlpha;
void main() {
	vec3 p = position;

	float span = max( uHalf.y * 2.0, 0.0001 );
	float y = mod( ( p.y + uHalf.y ) + uRise * uTime, span ) - uHalf.y;

	float s = uTime * uSwirlFreq + aPhase * 6.2831853;
	p.x += sin( s ) * uSwirl * ( 0.45 + aSeed );
	p.z += cos( s * 0.83 + aSeed * 3.1 ) * uSwirl * ( 0.45 + aSeed );
	p.y = y;

	vec4 mv = modelViewMatrix * vec4( p, 1.0 );
	float twinkle = 1.0 - uTwinkle * 0.5 * ( 1.0 + sin( uTime * 2.3 + aSeed * 31.0 ) );
	float live = step( aSeed, uDensity );
	vAlpha = twinkle * live;
	gl_PointSize = live * uSize * aSize * uPixelRatio * ( 60.0 / max( -mv.z, 0.001 ) );
	gl_Position = projectionMatrix * mv;
}
`;

const AMBIENT_FRAG =  `
#include <common>
uniform vec3 uColor;
uniform float uIntensity;
uniform float uSoftness;
uniform float uOpacity;
varying float vAlpha;
void main() {
	vec2 d = gl_PointCoord - vec2( 0.5 );
	float r = length( d ) * 2.0;
	float a = pow( clamp( 1.0 - r, 0.0, 1.0 ), uSoftness ) * vAlpha * uOpacity;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( uColor * uIntensity, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

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

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geometry.setDrawRange(0, count);
  geometry.boundingSphere = new THREE.Sphere(
    new THREE.Vector3(0, 0, 0),
    Math.sqrt(b.halfX * b.halfX + b.halfY * b.halfY + b.halfZ * b.halfZ) + Math.max(1, spec.swirl * 2),
  );

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: 1 },
      uSize: { value: opts.size ?? spec.size },
      uHalf: { value: new THREE.Vector3(b.halfX, b.halfY, b.halfZ) },
      uRise: { value: opts.rise ?? spec.rise },
      uSwirl: { value: opts.swirl ?? spec.swirl },
      uSwirlFreq: { value: spec.swirlFreq },
      uTwinkle: { value: spec.twinkle },
      uDensity: { value: 1 },
      uColor: { value: new THREE.Color(opts.color ?? spec.color) },
      uIntensity: { value: opts.intensity ?? spec.intensity },
      uSoftness: { value: spec.softness },
      uOpacity: { value: opts.opacity ?? spec.opacity },
    },
    vertexShader: AMBIENT_VERT,
    fragmentShader: AMBIENT_FRAG,
    transparent: true,
    depthWrite: false,
    blending: spec.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    fog: false,
  });
  material.name = opts.name ?? `oakParticles:${opts.preset}`;

  const points = new THREE.Points(geometry, material);
  points.name = material.name;
  points.position.set(cx, cy, cz);
  points.frustumCulled = false;
  points.visible = count > 0;

  return {
    points,
    material,
    count,
    setTime(t) {
      (material.uniforms.uTime as { value: number }).value = t;
    },
    setColor(color) {
      (material.uniforms.uColor as { value: THREE.Color }).value.set(color);
    },
    setIntensity(i) {
      (material.uniforms.uIntensity as { value: number }).value = i;
    },
    setDensity(f01) {
      (material.uniforms.uDensity as { value: number }).value = f01 < 0 ? 0 : f01 > 1 ? 1 : f01;
    },
    setPixelRatio(ratio) {
      (material.uniforms.uPixelRatio as { value: number }).value = ratio > 0 ? ratio : 1;
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
  material: THREE.ShaderMaterial;
  layers: number;
  setTime(t: number): void;
  setOpacity(o: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  dispose(): void;
}

const MIST_VERT =  `
#include <common>
varying vec2 vMistUv;
varying float vDepth;
void main() {
	vMistUv = uv;
	vec4 mv = modelViewMatrix * vec4( position, 1.0 );
	vDepth = -mv.z;
	gl_Position = projectionMatrix * mv;
}
`;

const MIST_FRAG =  `
#include <common>
uniform sampler2D uNoise;
uniform vec3 uColor;
uniform float uTime;
uniform float uOpacity;
uniform float uSpeed;
uniform float uScale;
uniform float uNearFade;
varying vec2 vMistUv;
varying float vDepth;
void main() {
	vec2 uv = vMistUv * uScale;
	float a = texture2D( uNoise, uv + vec2( uTime * uSpeed, uTime * uSpeed * 0.6 ) ).r;
	float b = texture2D( uNoise, uv * 1.7 - vec2( uTime * uSpeed * 0.8, 0.0 ) ).r;
	float n = clamp( a * 0.65 + b * 0.55 - 0.28, 0.0, 1.0 );

	vec2 e = abs( vMistUv - 0.5 ) * 2.0;
	float edge = ( 1.0 - smoothstep( 0.55, 1.0, e.x ) ) * ( 1.0 - smoothstep( 0.55, 1.0, e.y ) );

	float near = smoothstep( 0.0, max( uNearFade, 0.001 ), vDepth );

	float alpha = n * edge * near * uOpacity;
	if ( alpha <= 0.004 ) discard;
	gl_FragColor = vec4( uColor, alpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export function makeMistPlanes(opts: MistPlanesOptions): MistPlanesHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'oakMist';
  const layers = Math.max(0, Math.round(opts.layers ?? 5));
  const sizeX = typeof opts.size === 'number' ? opts.size : opts.size[0];
  const sizeZ = typeof opts.size === 'number' ? opts.size : opts.size[1];
  const baseY = opts.baseY ?? 0.15;
  const spread = opts.spread ?? 1.4;

  const noise = fbm(opts.seed ^ 0x2c1b3a95, { size: 256, frequency: 4, octaves: 4, gain: 0.55 });
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: noise },
      uColor: { value: new THREE.Color(opts.color) },
      uTime: { value: 0 },
      uOpacity: { value: opts.opacity ?? 0.28 },
      uSpeed: { value: opts.speed ?? 0.012 },
      uScale: { value: opts.scale ?? 2.5 },
      uNearFade: { value: opts.nearFade ?? 3 },
    },
    vertexShader: MIST_VERT,
    fragmentShader: MIST_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: false,
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
      (material.uniforms.uTime as { value: number }).value = t;
    },
    setOpacity(o) {
      (material.uniforms.uOpacity as { value: number }).value = o;
    },
    setColor(color) {
      (material.uniforms.uColor as { value: THREE.Color }).value.set(color);
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
