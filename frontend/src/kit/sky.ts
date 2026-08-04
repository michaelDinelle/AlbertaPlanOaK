// kit/sky — the one sky. STYLE.md: "makeGradientSky (or its underwater/cave
// parameterization) — exactly one sky", "A horizon or depth gradient is always visible",
// "HDR emissives (>1.0) are the only things that bloom" (the sun disc is one of them).
//
// An inward icosphere carrying a single ShaderMaterial: a three-stop vertical gradient,
// an optional HDR sun disc + glow, an optional star field, and a screen-space dither
// that kills banding across the big gradients. Disable the sun and it becomes the
// underwater / cave depth column (vertical gradient, no disc) with no second code path.
//
// The dome follows the camera in onBeforeRender (allocation-free), so scenes can move
// the camera anywhere without leaving the sky.

import * as THREE from 'three';
import type { Palette } from './tokens';
import type { QualityTier } from './types';

export interface SkyStarOptions {
  /** lattice cells per unit direction — higher = more, smaller stars */
  density?: number;
  /** star radius in cell units */
  size?: number;
  /** cell occupancy: higher = fewer stars (0..1) */
  threshold?: number;
  /** twinkle depth in [0,1] */
  twinkle?: number;
  color?: THREE.ColorRepresentation;
  /** HDR gain — >1 lets the brightest stars bloom */
  intensity?: number;
}

export interface GradientSkyOptions {
  top: THREE.ColorRepresentation;
  horizon: THREE.ColorRepresentation;
  bottom: THREE.ColorRepresentation;
  /** how far up the horizon color reaches (0..1 of the hemisphere) */
  topSpread?: number;
  /** how far down the horizon color reaches */
  bottomSpread?: number;

  /** omit (or pass null) for the underwater/cave depth column */
  sunDir?: THREE.Vector3 | readonly [number, number, number] | null;
  sunColor?: THREE.ColorRepresentation;
  /** angular radius of the disc, radians (~0.02 = a real sun, 0.05 = stylized) */
  sunSize?: number;
  /** HDR multiplier on the disc — this is what blooms */
  sunHdr?: number;
  /** broad scattering halo around the sun */
  sunGlow?: number;
  sunGlowPower?: number;

  stars?: boolean | SkyStarOptions;
  /** dome radius in world units (must sit inside the camera far plane is NOT required — depth test is off) */
  radius?: number;
  /** gradient dither amplitude (0 disables) */
  dither?: number;
  tier?: QualityTier;
  name?: string;
}

export interface SkyUniforms {
  uTop: { value: THREE.Color };
  uHorizon: { value: THREE.Color };
  uBottom: { value: THREE.Color };
  uTopSpread: { value: number };
  uBottomSpread: { value: number };
  uSunDir: { value: THREE.Vector3 };
  uSunColor: { value: THREE.Color };
  uSunSize: { value: number };
  uSunHdr: { value: number };
  uSunGlow: { value: number };
  uSunGlowPower: { value: number };
  uStarColor: { value: THREE.Color };
  uStarDensity: { value: number };
  uStarSize: { value: number };
  uStarThreshold: { value: number };
  uStarTwinkle: { value: number };
  uStarIntensity: { value: number };
  uDither: { value: number };
  uTime: { value: number };
}

export interface SkyHandle {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  uniforms: SkyUniforms;
  /** the sun direction the light rig should match (unit vector, points TOWARD the sun) */
  sunDirection: THREE.Vector3;
  setSunDirection(x: number, y: number, z: number): void;
  setColors(top: THREE.ColorRepresentation, horizon: THREE.ColorRepresentation, bottom: THREE.ColorRepresentation): void;
  setTime(t: number): void;
  dispose(): void;
}

const VERT = /* glsl */ `
varying vec3 vOakDir;
void main() {
	vOakDir = normalize( position );
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const FRAG_HEAD = /* glsl */ `
#include <common>
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uBottom;
uniform float uTopSpread;
uniform float uBottomSpread;
uniform float uDither;
uniform float uTime;
varying vec3 vOakDir;

float oakHash13( vec3 p ) {
	p = fract( p * 0.1031 );
	p += dot( p, p.zyx + 31.32 );
	return fract( ( p.x + p.y ) * p.z );
}
`;

const FRAG_SUN_DECL = /* glsl */ `
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uSunSize;
uniform float uSunHdr;
uniform float uSunGlow;
uniform float uSunGlowPower;
`;

const FRAG_STAR_DECL = /* glsl */ `
uniform vec3 uStarColor;
uniform float uStarDensity;
uniform float uStarSize;
uniform float uStarThreshold;
uniform float uStarTwinkle;
uniform float uStarIntensity;
`;

const FRAG_MAIN_OPEN = /* glsl */ `
void main() {
	vec3 dir = normalize( vOakDir );
	float h = clamp( dir.y, -1.0, 1.0 );
	vec3 col = uHorizon;
	// NOTE: every smoothstep here keeps edge0 < edge1 — reversed edges are undefined in GLSL.
	col = mix( col, uTop, smoothstep( 0.0, max( uTopSpread, 0.001 ), h ) );
	col = mix( col, uBottom, smoothstep( 0.0, max( uBottomSpread, 0.001 ), -h ) );
`;

const FRAG_SUN_APPLY = /* glsl */ `
	{
		vec3 sd = normalize( uSunDir );
		float cosA = dot( dir, sd );
		float sz = max( uSunSize, 0.002 );
		float disc = smoothstep( cos( sz * 1.4 ), cos( sz * 0.85 ), cosA );
		float glow = pow( max( cosA, 0.0 ), max( uSunGlowPower, 1.0 ) );
		col += uSunColor * ( disc * uSunHdr + glow * uSunGlow );
	}
`;

const FRAG_STAR_APPLY = /* glsl */ `
	{
		float above = smoothstep( -0.08, 0.12, dir.y );
		vec3 g = dir * uStarDensity;
		vec3 c = floor( g );
		float hcell = oakHash13( c );
		vec3 pt = vec3( oakHash13( c + 1.7 ), oakHash13( c + 3.3 ), oakHash13( c + 5.9 ) );
		float d = length( ( g - c ) - pt );
		float star = ( 1.0 - smoothstep( 0.0, max( uStarSize, 0.001 ), d ) ) * step( uStarThreshold, hcell );
		float tw = 1.0 - uStarTwinkle * 0.5 * ( 1.0 + sin( uTime * 2.7 + hcell * 63.0 ) );
		col += uStarColor * ( star * uStarIntensity * tw * above );
	}
`;

const FRAG_MAIN_CLOSE = /* glsl */ `
	col += ( oakHash13( vec3( gl_FragCoord.xy, 1.0 ) ) - 0.5 ) * uDither;
	gl_FragColor = vec4( col, 1.0 );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

function detailForTier(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 1 : 2;
}

function toVec3(v: THREE.Vector3 | readonly [number, number, number] | null | undefined, out: THREE.Vector3): THREE.Vector3 {
  if (!v) return out.set(0, 1, 0);
  const asVec = v as THREE.Vector3;
  if (typeof asVec.x === 'number') return out.set(asVec.x, asVec.y, asVec.z);
  const a = v as readonly [number, number, number];
  return out.set(a[0], a[1], a[2]);
}

/**
 * The one sky. Add `handle.mesh` to the scene; it renders first, never writes depth,
 * and follows the camera automatically.
 *
 * Underwater / cave: pass `sunDir: null` — the disc and its glow are compiled out and
 * what remains is a pure vertical depth column (top = surface light, bottom = the dark).
 */
export function makeGradientSky(opts: GradientSkyOptions): SkyHandle {
  const hasSun = opts.sunDir !== null && opts.sunDir !== undefined;
  const stars = opts.stars;
  const starOpts: SkyStarOptions | null = stars === true ? {} : stars ? stars : null;
  const hasStars = starOpts !== null && opts.tier !== 'thumb';

  const sunDirection = new THREE.Vector3();
  toVec3(opts.sunDir ?? null, sunDirection);
  if (sunDirection.lengthSq() < 1e-12) sunDirection.set(0, 1, 0);
  sunDirection.normalize();

  const uniforms: SkyUniforms = {
    uTop: { value: new THREE.Color(opts.top) },
    uHorizon: { value: new THREE.Color(opts.horizon) },
    uBottom: { value: new THREE.Color(opts.bottom) },
    uTopSpread: { value: opts.topSpread ?? 0.55 },
    uBottomSpread: { value: opts.bottomSpread ?? 0.35 },
    uSunDir: { value: sunDirection },
    uSunColor: { value: new THREE.Color(opts.sunColor ?? 0xffffff) },
    uSunSize: { value: opts.sunSize ?? 0.045 },
    uSunHdr: { value: opts.sunHdr ?? 6 },
    uSunGlow: { value: opts.sunGlow ?? 0.35 },
    uSunGlowPower: { value: opts.sunGlowPower ?? 48 },
    uStarColor: { value: new THREE.Color(starOpts?.color ?? 0xffffff) },
    uStarDensity: { value: starOpts?.density ?? 160 },
    uStarSize: { value: starOpts?.size ?? 0.06 },
    uStarThreshold: { value: starOpts?.threshold ?? 0.965 },
    uStarTwinkle: { value: starOpts?.twinkle ?? 0.35 },
    uStarIntensity: { value: starOpts?.intensity ?? 2.2 },
    uDither: { value: opts.dither ?? 0.006 },
    uTime: { value: 0 },
  };

  const fragment =
    FRAG_HEAD +
    (hasSun ? FRAG_SUN_DECL : '') +
    (hasStars ? FRAG_STAR_DECL : '') +
    FRAG_MAIN_OPEN +
    (hasSun ? FRAG_SUN_APPLY : '') +
    (hasStars ? FRAG_STAR_APPLY : '') +
    FRAG_MAIN_CLOSE;

  const material = new THREE.ShaderMaterial({
    uniforms: uniforms as unknown as Record<string, THREE.IUniform>,
    vertexShader: VERT,
    fragmentShader: fragment,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    fog: false,
  });
  material.name = opts.name ?? 'oakSky';

  const geometry = new THREE.IcosahedronGeometry(1, detailForTier(opts.tier));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'oakSky';
  mesh.scale.setScalar(opts.radius ?? 400);
  mesh.renderOrder = -1000;
  mesh.frustumCulled = false;
  mesh.matrixAutoUpdate = true;
  mesh.onBeforeRender = (_renderer, _scene, camera) => {
    mesh.position.copy(camera.position);
    mesh.updateMatrixWorld();
  };

  return {
    mesh,
    material,
    uniforms,
    sunDirection,
    setSunDirection(x: number, y: number, z: number) {
      sunDirection.set(x, y, z);
      if (sunDirection.lengthSq() < 1e-12) sunDirection.set(0, 1, 0);
      sunDirection.normalize();
    },
    setColors(top, horizon, bottom) {
      uniforms.uTop.value.set(top);
      uniforms.uHorizon.value.set(horizon);
      uniforms.uBottom.value.set(bottom);
    },
    setTime(t: number) {
      uniforms.uTime.value = t;
    },
    dispose() {
      mesh.onBeforeRender = () => {};
      geometry.dispose();
      material.dispose();
    },
  };
}

/**
 * Default sun elevation for a palette: dim palettes sit low (dusk/night), bright ones
 * high (noon). Derived from the palette itself so scenes never hand-tune a number.
 * Returns a unit vector pointing TOWARD the sun.
 */
export function paletteSunDirection(palette: Palette, out?: THREE.Vector3): THREE.Vector3 {
  const v = out ?? new THREE.Vector3();
  const k = Math.max(0, Math.min(1, palette.sunIntensity / 3));
  const elevation = 0.22 + 0.68 * k;
  const azimuth = Math.PI * 0.32;
  const horiz = Math.sqrt(Math.max(0, 1 - elevation * elevation));
  return v.set(Math.cos(azimuth) * horiz, elevation, Math.sin(azimuth) * horiz).normalize();
}

/** true for the palettes that are a water/rock column rather than a sky. */
export function isDepthColumnPalette(name: string): boolean {
  return name === 'underwater' || name === 'abyss' || name === 'cave';
}

export interface PaletteSkyOptions extends Partial<Omit<GradientSkyOptions, 'top' | 'horizon' | 'bottom'>> {
  /** force the depth-column parameterization (no sun disc) */
  depthColumn?: boolean;
}

/**
 * Token-driven sky: everything comes from the palette, so all 16 scenes read as one
 * day. `depthColumn` (or an underwater/abyss/cave palette) drops the sun disc.
 */
export function skyFromPalette(
  palette: Palette,
  paletteName: string,
  tier: QualityTier,
  overrides: PaletteSkyOptions = {},
): SkyHandle {
  const depth = overrides.depthColumn ?? isDepthColumnPalette(paletteName);
  const night = paletteName === 'night';
  const sunDir = depth ? null : (overrides.sunDir ?? paletteSunDirection(palette));
  const stars = overrides.stars ?? (night ? { intensity: 2.4 } : paletteName === 'abyss' ? false : false);
  const { depthColumn: _ignored, ...rest } = overrides;
  return makeGradientSky({
    top: palette.skyTop,
    horizon: palette.skyHorizon,
    bottom: palette.skyBottom,
    sunColor: palette.sun,
    sunHdr: night ? 2.5 : 6,
    tier,
    ...rest,
    sunDir,
    stars,
  });
}
