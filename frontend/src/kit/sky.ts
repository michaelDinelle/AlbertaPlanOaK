import * as THREE from 'three';
import type { Palette } from './tokens';

export interface SkyStarOptions {
  density?: number;
  size?: number;
  threshold?: number;
  twinkle?: number;
  color?: THREE.ColorRepresentation;
  intensity?: number;
}

export interface GradientSkyOptions {
  top: THREE.ColorRepresentation;
  horizon: THREE.ColorRepresentation;
  bottom: THREE.ColorRepresentation;
  topSpread?: number;
  bottomSpread?: number;

  sunDir?: THREE.Vector3 | readonly [number, number, number] | null;
  sunColor?: THREE.ColorRepresentation;
  sunSize?: number;
  sunHdr?: number;
  sunGlow?: number;
  sunGlowPower?: number;

  stars?: boolean | SkyStarOptions;
  radius?: number;
  dither?: number;
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
  sunDirection: THREE.Vector3;
  setSunDirection(x: number, y: number, z: number): void;
  setColors(top: THREE.ColorRepresentation, horizon: THREE.ColorRepresentation, bottom: THREE.ColorRepresentation): void;
  setTime(t: number): void;
  dispose(): void;
}

const VERT =  `
varying vec3 vOakDir;
void main() {
	vOakDir = normalize( position );
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const FRAG_HEAD =  `
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

const FRAG_SUN_DECL =  `
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uSunSize;
uniform float uSunHdr;
uniform float uSunGlow;
uniform float uSunGlowPower;
`;

const FRAG_STAR_DECL =  `
uniform vec3 uStarColor;
uniform float uStarDensity;
uniform float uStarSize;
uniform float uStarThreshold;
uniform float uStarTwinkle;
uniform float uStarIntensity;
`;

const FRAG_MAIN_OPEN =  `
void main() {
	vec3 dir = normalize( vOakDir );
	float h = clamp( dir.y, -1.0, 1.0 );
	vec3 col = uHorizon;
	col = mix( col, uTop, smoothstep( 0.0, max( uTopSpread, 0.001 ), h ) );
	col = mix( col, uBottom, smoothstep( 0.0, max( uBottomSpread, 0.001 ), -h ) );
`;

const FRAG_SUN_APPLY =  `
	{
		vec3 sd = normalize( uSunDir );
		float cosA = dot( dir, sd );
		float sz = max( uSunSize, 0.002 );
		float disc = smoothstep( cos( sz * 1.4 ), cos( sz * 0.85 ), cosA );
		float glow = pow( max( cosA, 0.0 ), max( uSunGlowPower, 1.0 ) );
		col += uSunColor * ( disc * uSunHdr + glow * uSunGlow );
	}
`;

const FRAG_STAR_APPLY =  `
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

const FRAG_MAIN_CLOSE =  `
	col += ( oakHash13( vec3( gl_FragCoord.xy, 1.0 ) ) - 0.5 ) * uDither;
	gl_FragColor = vec4( col, 1.0 );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

function toVec3(v: THREE.Vector3 | readonly [number, number, number] | null | undefined, out: THREE.Vector3): THREE.Vector3 {
  if (!v) return out.set(0, 1, 0);
  const asVec = v as THREE.Vector3;
  if (typeof asVec.x === 'number') return out.set(asVec.x, asVec.y, asVec.z);
  const a = v as readonly [number, number, number];
  return out.set(a[0], a[1], a[2]);
}

export function makeGradientSky(opts: GradientSkyOptions): SkyHandle {
  const hasSun = opts.sunDir !== null && opts.sunDir !== undefined;
  const stars = opts.stars;
  const starOpts: SkyStarOptions | null = stars === true ? {} : stars ? stars : null;
  const hasStars = starOpts !== null;

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

  const geometry = new THREE.IcosahedronGeometry(1, 2);
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

export function paletteSunDirection(palette: Palette, out?: THREE.Vector3): THREE.Vector3 {
  const v = out ?? new THREE.Vector3();
  const k = Math.max(0, Math.min(1, palette.sunIntensity / 3));
  const elevation = 0.22 + 0.68 * k;
  const azimuth = Math.PI * 0.32;
  const horiz = Math.sqrt(Math.max(0, 1 - elevation * elevation));
  return v.set(Math.cos(azimuth) * horiz, elevation, Math.sin(azimuth) * horiz).normalize();
}

export type PaletteSkyOptions = Partial<Omit<GradientSkyOptions, 'top' | 'horizon' | 'bottom'>>;

export function skyFromPalette(
  palette: Palette,
  paletteName: string,
  overrides: PaletteSkyOptions = {},
): SkyHandle {
  const depth = paletteName === 'underwater' || paletteName === 'abyss' || paletteName === 'cave';
  const night = paletteName === 'night';
  const sunDir = depth ? null : (overrides.sunDir ?? paletteSunDirection(palette));
  const stars = overrides.stars ?? (night ? { intensity: 2.4 } : false);
  return makeGradientSky({
    top: palette.skyTop,
    horizon: palette.skyHorizon,
    bottom: palette.skyBottom,
    sunColor: palette.sun,
    sunHdr: night ? 2.5 : 6,
    ...overrides,
    sunDir,
    stars,
  });
}
