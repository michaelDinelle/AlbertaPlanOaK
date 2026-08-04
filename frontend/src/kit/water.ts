// kit/water — ONE parameterized water material for every wet surface in the preserve:
// the Acrobot mirror pond, the CartPole lake, the Pendulum koi pond, the Chain creek,
// and the DeepSea / LargeDeepSea column surface.
//
// STYLE.md allows at most one bespoke ShaderMaterial PER SCENE; this is a kit material,
// so a scene spends nothing from that budget by using it. Everything is token-driven:
// the shallow/deep colors, the sky tint of the fresnel rim and the caustic color all
// come from a `Palette`, so the water in five different scenes reads as the same water
// under five different skies.
//
// One material does all of it:
//   * vertex: sum of four sines (two cross-swell, two ripple) with the analytic normal
//     derived from the same sum — the lighting can never disagree with the silhouette
//   * fragment: depth gradient (shallow -> deep by view-space depth AND by wave height),
//     a fresnel rim in the sky color, and a procedural caustic shimmer built from two
//     counter-scrolling triangle-wave lattices (no texture, no dependent reads)
//
// All animation comes from `uTime` = ctx.time().

import * as THREE from 'three';
import type { Palette } from './tokens';
import type { QualityTier } from './types';

export interface WaterUniforms {
  uTime: { value: number };
  uShallow: { value: THREE.Color };
  uDeep: { value: THREE.Color };
  uSky: { value: THREE.Color };
  uCaustic: { value: THREE.Color };
  uAmplitude: { value: number };
  uWaveScale: { value: number };
  uSpeed: { value: number };
  uFresnel: { value: number };
  uCausticStrength: { value: number };
  uDepthFalloff: { value: number };
  uOpacity: { value: number };
  uSunDir: { value: THREE.Vector3 };
  uSpecular: { value: number };
}

export interface WaterOptions {
  /** world extent: a number is square, or [x, z] */
  size: number | readonly [number, number];
  palette: Palette;
  tier?: QualityTier;
  /** mesh segments at hero tier (default 96); tier-scaled */
  segments?: number;
  /** override the palette-derived shallow color */
  shallow?: THREE.ColorRepresentation;
  /** override the palette-derived deep color */
  deep?: THREE.ColorRepresentation;
  /** sky tint for the fresnel rim (default palette.skyHorizon) */
  sky?: THREE.ColorRepresentation;
  /** caustic shimmer color (default palette.sun) */
  caustic?: THREE.ColorRepresentation;
  /** wave height in world units (default 0.035) */
  amplitude?: number;
  /** spatial frequency multiplier (default 1) */
  waveScale?: number;
  /** temporal speed multiplier (default 1) */
  speed?: number;
  /** 0..1 rim strength (default 0.55) */
  fresnel?: number;
  /** 0..1 caustic strength (default 0.35) */
  caustics?: number;
  /** how fast the color reaches `deep` with view depth, world units (default 9) */
  depthFalloff?: number;
  opacity?: number;
  /** unit vector toward the sun (sky.sunDirection) — drives the specular glint */
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  /** surface height (default 0) */
  level?: number;
  name?: string;
}

export interface WaterHandle {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  uniforms: WaterUniforms;
  /** ctx.time() — the ONLY clock */
  setTime(t: number): void;
  setLevel(y: number): void;
  setColors(shallow: THREE.ColorRepresentation, deep: THREE.ColorRepresentation): void;
  setSunDirection(x: number, y: number, z: number): void;
  /** the rendered surface height at a world point — keep floaters ON the water */
  sampleHeight(x: number, z: number, time: number): number;
  dispose(): void;
}

/** hero 1 / medium 0.55 / thumb 0.22 — vertex density only, never the look. */
export function waterSegmentScale(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.22 : tier === 'medium' ? 0.55 : 1;
}

// The four waves, shared by the GPU and the CPU sampler so a lantern floating on the
// pond sits exactly on the rendered surface (the kit/terrain sampleHeight contract).
const WAVES: readonly (readonly [number, number, number, number])[] = [
  // [dirX, dirZ, frequency, speed]
  [1.0, 0.15, 0.9, 0.55],
  [-0.35, 1.0, 1.35, 0.42],
  [0.7, -0.7, 2.6, 0.95],
  [-0.9, -0.45, 4.1, 1.35],
];
const WAVE_AMPS: readonly number[] = [0.55, 0.3, 0.11, 0.06];

/** CPU twin of the vertex displacement — same constants, same result. */
export function waterHeightAt(x: number, z: number, time: number, amplitude: number, waveScale: number, speed: number): number {
  let h = 0;
  for (let i = 0; i < WAVES.length; i++) {
    const w = WAVES[i] as readonly [number, number, number, number];
    const amp = WAVE_AMPS[i] as number;
    h += amp * Math.sin((x * w[0] + z * w[1]) * w[2] * waveScale + time * w[3] * speed);
  }
  return h * amplitude;
}

const WAVE_GLSL = /* glsl */ `
const int OAK_WAVES = 4;
const vec4 OAK_WAVE[ OAK_WAVES ] = vec4[ OAK_WAVES ](
	vec4( 1.0, 0.15, 0.9, 0.55 ),
	vec4( -0.35, 1.0, 1.35, 0.42 ),
	vec4( 0.7, -0.7, 2.6, 0.95 ),
	vec4( -0.9, -0.45, 4.1, 1.35 )
);
const float OAK_WAVE_AMP[ OAK_WAVES ] = float[ OAK_WAVES ]( 0.55, 0.3, 0.11, 0.06 );

float oakWaveHeight( vec2 p, float t, float scale, float speed ) {
	float h = 0.0;
	for ( int i = 0; i < OAK_WAVES; i ++ ) {
		vec4 w = OAK_WAVE[ i ];
		h += OAK_WAVE_AMP[ i ] * sin( dot( p, w.xy ) * w.z * scale + t * w.w * speed );
	}
	return h;
}

vec2 oakWaveSlope( vec2 p, float t, float scale, float speed ) {
	vec2 g = vec2( 0.0 );
	for ( int i = 0; i < OAK_WAVES; i ++ ) {
		vec4 w = OAK_WAVE[ i ];
		float k = w.z * scale;
		float c = cos( dot( p, w.xy ) * k + t * w.w * speed );
		g += OAK_WAVE_AMP[ i ] * c * k * w.xy;
	}
	return g;
}
`;

const WATER_VERT = /* glsl */ `
#include <common>
#include <fog_pars_vertex>
uniform float uTime;
uniform float uAmplitude;
uniform float uWaveScale;
uniform float uSpeed;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vHeight;
varying float vDepth;
${WAVE_GLSL}
void main() {
	vec3 p = position;
	vec4 world4 = modelMatrix * vec4( p, 1.0 );
	vec2 wp = world4.xz;
	float h = oakWaveHeight( wp, uTime, uWaveScale, uSpeed ) * uAmplitude;
	world4.y += h;
	vec2 slope = oakWaveSlope( wp, uTime, uWaveScale, uSpeed ) * uAmplitude;
	vNormalW = normalize( vec3( -slope.x, 1.0, -slope.y ) );
	vWorld = world4.xyz;
	vHeight = h;
	vec4 mv = viewMatrix * world4;
	vDepth = -mv.z;
	gl_Position = projectionMatrix * mv;
	#ifdef USE_FOG
		vFogDepth = -mv.z;
	#endif
}
`;

const WATER_FRAG = /* glsl */ `
#include <common>
#include <fog_pars_fragment>
uniform vec3 uShallow;
uniform vec3 uDeep;
uniform vec3 uSky;
uniform vec3 uCaustic;
uniform float uTime;
uniform float uFresnel;
uniform float uCausticStrength;
uniform float uDepthFalloff;
uniform float uOpacity;
uniform float uWaveScale;
uniform float uSpeed;
uniform vec3 uSunDir;
uniform float uSpecular;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vHeight;
varying float vDepth;

// two counter-scrolling triangle lattices -> the classic pool-caustic web, no texture
float oakCaustic( vec2 p, float t ) {
	vec2 a = p * 1.7 + vec2( t * 0.09, t * 0.06 );
	vec2 b = p * 2.3 - vec2( t * 0.07, t * 0.11 );
	float ca = abs( fract( a.x + a.y ) - 0.5 ) + abs( fract( a.x - a.y ) - 0.5 );
	float cb = abs( fract( b.x + b.y ) - 0.5 ) + abs( fract( b.x - b.y ) - 0.5 );
	float c = 1.0 - clamp( min( ca, cb ) * 2.4, 0.0, 1.0 );
	return pow( c, 3.0 );
}

void main() {
	vec3 n = normalize( vNormalW );
	vec3 viewDir = normalize( cameraPosition - vWorld );

	// depth gradient: distance from the eye AND the wave trough both push toward uDeep
	float byDepth = 1.0 - exp( -vDepth / max( uDepthFalloff, 0.001 ) );
	float byTrough = clamp( 0.5 - vHeight * 6.0, 0.0, 1.0 );
	vec3 col = mix( uShallow, uDeep, clamp( byDepth * 0.75 + byTrough * 0.35, 0.0, 1.0 ) );

	// caustic shimmer, strongest in the shallows
	float caustic = oakCaustic( vWorld.xz * uWaveScale, uTime * uSpeed );
	col += uCaustic * ( caustic * uCausticStrength * ( 1.0 - byDepth * 0.7 ) );

	// sky-tinted fresnel rim
	float f = pow( clamp( 1.0 - dot( n, viewDir ), 0.0, 1.0 ), 3.0 );
	col = mix( col, uSky, f * uFresnel );

	// a single specular glint along the sun direction (HDR: this is what blooms)
	vec3 halfV = normalize( normalize( uSunDir ) + viewDir );
	float spec = pow( clamp( dot( n, halfV ), 0.0, 1.0 ), 96.0 );
	col += uSky * ( spec * uSpecular );

	float alpha = clamp( uOpacity + f * 0.35, 0.0, 1.0 );
	gl_FragColor = vec4( col, alpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}
`;

/**
 * Build a water surface.
 *
 * ```ts
 * const pond = makeWater({ size: 14, palette: tokens.palettes.dawn, tier, sunDir: sky.sunDirection });
 * scene.add(pond.mesh);
 * // update(): pond.setTime(ctx.time());
 * lantern.position.y = pond.sampleHeight(lx, lz, ctx.time());   // floats exactly
 * ```
 */
export function makeWater(opts: WaterOptions): WaterHandle {
  const sizeX = typeof opts.size === 'number' ? opts.size : opts.size[0];
  const sizeZ = typeof opts.size === 'number' ? opts.size : opts.size[1];
  const base = Math.max(2, Math.round(opts.segments ?? 96));
  const seg = Math.max(2, Math.round(base * waterSegmentScale(opts.tier)));
  const p = opts.palette;

  const amplitude = opts.amplitude ?? 0.035;
  const waveScale = opts.waveScale ?? 1;
  const speed = opts.speed ?? 1;

  const sun = new THREE.Vector3(0, 1, 0);
  const given = opts.sunDir;
  if (given) {
    const v = given as THREE.Vector3;
    if (typeof v.x === 'number') sun.set(v.x, v.y, v.z);
    else {
      const a = given as readonly [number, number, number];
      sun.set(a[0], a[1], a[2]);
    }
    if (sun.lengthSq() < 1e-12) sun.set(0, 1, 0);
    sun.normalize();
  }

  const uniforms: WaterUniforms = {
    uTime: { value: 0 },
    uShallow: { value: new THREE.Color(opts.shallow ?? p.skyHorizon).lerp(new THREE.Color(p.groundB), 0.35) },
    uDeep: { value: new THREE.Color(opts.deep ?? p.fog).lerp(new THREE.Color(p.skyBottom), 0.55) },
    uSky: { value: new THREE.Color(opts.sky ?? p.skyHorizon) },
    uCaustic: { value: new THREE.Color(opts.caustic ?? p.sun) },
    uAmplitude: { value: amplitude },
    uWaveScale: { value: waveScale },
    uSpeed: { value: speed },
    uFresnel: { value: opts.fresnel ?? 0.55 },
    uCausticStrength: { value: (opts.caustics ?? 0.35) * (opts.tier === 'thumb' ? 0 : 1) },
    uDepthFalloff: { value: opts.depthFalloff ?? 9 },
    uOpacity: { value: opts.opacity ?? 0.86 },
    uSunDir: { value: sun },
    uSpecular: { value: opts.tier === 'thumb' ? 0 : 1.6 },
  };

  const material = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      uniforms as unknown as Record<string, THREE.IUniform>,
    ]),
    vertexShader: WATER_VERT,
    fragmentShader: WATER_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    fog: true,
  });
  material.name = opts.name ?? 'oakWater';
  // UniformsUtils.merge CLONES every uniform, so re-point the handle at the live objects
  const live = material.uniforms as unknown as WaterUniforms;
  live.uTime.value = 0;
  live.uShallow.value.copy(uniforms.uShallow.value);
  live.uDeep.value.copy(uniforms.uDeep.value);
  live.uSky.value.copy(uniforms.uSky.value);
  live.uCaustic.value.copy(uniforms.uCaustic.value);
  live.uSunDir.value.copy(sun);

  const geometry = new THREE.PlaneGeometry(sizeX, sizeZ, seg, seg);
  geometry.rotateX(-Math.PI / 2);
  geometry.name = 'oakWater';

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'oakWater';
  mesh.position.y = opts.level ?? 0;
  mesh.receiveShadow = false;
  mesh.castShadow = false;
  mesh.renderOrder = 1;

  return {
    mesh,
    material,
    uniforms: live,
    setTime(t) {
      live.uTime.value = t;
    },
    setLevel(y) {
      mesh.position.y = y;
    },
    setColors(shallow, deep) {
      live.uShallow.value.set(shallow);
      live.uDeep.value.set(deep);
    },
    setSunDirection(x, y, z) {
      live.uSunDir.value.set(x, y, z);
      if (live.uSunDir.value.lengthSq() < 1e-12) live.uSunDir.value.set(0, 1, 0);
      live.uSunDir.value.normalize();
    },
    sampleHeight(x, z, time) {
      return mesh.position.y + waterHeightAt(x, z, time, amplitude, waveScale, speed);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}
