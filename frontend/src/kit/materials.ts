import * as THREE from 'three';

export interface NumberUniform {
  value: number;
}
export interface ColorUniform {
  value: THREE.Color;
}

export interface OakUniforms {
  uSpotColor: ColorUniform;
  uSpotStrength: NumberUniform;
  uSpotScale: NumberUniform;
  uSpotRadius: NumberUniform;
  uSpotSeed: NumberUniform;
  uTime: NumberUniform;
  uWindAmp: NumberUniform;
  uWindFreq: NumberUniform;
  uWindHeight: NumberUniform;
  uRimColor: ColorUniform;
  uRimStrength: NumberUniform;
  uRimPower: NumberUniform;
  uToonSteps: NumberUniform;
  uToonSoftness: NumberUniform;
  uToonFloor: NumberUniform;
}

export interface OakHandle {
  readonly uniforms: OakUniforms;
  readonly flags: { toon: boolean; rim: boolean; wind: boolean };
  setTime(t: number): void;
  setWind(amplitude: number, frequency?: number): void;
  setRim(strength: number, color?: THREE.ColorRepresentation): void;
  setToon(steps: number, softness?: number): void;
}

export interface OakMaterial extends THREE.MeshStandardMaterial {
  readonly oak: OakHandle;
}

interface HasOak {
  oak?: unknown;
  userData?: { oak?: unknown };
}

function isOakHandle(v: unknown): v is OakHandle {
  return typeof v === 'object' && v !== null && 'uniforms' in v && 'setTime' in v;
}

export function oakUniforms(material: THREE.Material | null | undefined): OakHandle | null {
  if (!material) return null;
  const m = material as unknown as HasOak;
  if (isOakHandle(m.oak)) return m.oak;
  const ud = m.userData?.oak;
  return isOakHandle(ud) ? ud : null;
}

export function setMaterialsTime(materials: readonly (THREE.Material | null | undefined)[], t: number): void {
  for (let i = 0; i < materials.length; i++) {
    const h = oakUniforms(materials[i]);
    if (h) h.uniforms.uTime.value = t;
  }
}

export interface OakWindOptions {
  amplitude: number;
  frequency?: number;
  height?: number;
}

export interface OakSpotOptions {
  color: THREE.ColorRepresentation;
  strength: number;
  scale: number;
  radius: number;
  seed: number;
}

export interface OakMaterialOptions {
  spots?: OakSpotOptions;
  color?: THREE.ColorRepresentation;
  roughness?: number;
  metalness?: number;
  flatShading?: boolean;
  vertexColors?: boolean;
  transparent?: boolean;
  opacity?: number;
  side?: THREE.Side;
  depthWrite?: boolean;
  emissive?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
  map?: THREE.Texture | null;
  fog?: boolean;
  name?: string;

  toonSteps?: number;
  toonSoftness?: number;
  toonFloor?: number;

  rimColor?: THREE.ColorRepresentation;
  rimStrength?: number;
  rimPower?: number;

  wind?: OakWindOptions | number;

}

const DEFAULT_TOON_STEPS = 3;
const DEFAULT_TOON_SOFTNESS = 0.25;
const DEFAULT_RIM_POWER = 3;
const DEFAULT_WIND_FREQ = 1.1;
const DEFAULT_WIND_HEIGHT = 1;

const TOON_FN =  `
uniform float uToonSteps;
uniform float uToonSoftness;
uniform float uToonFloor;
float oakToonRamp( const in float x ) {
	float steps = max( uToonSteps, 1.0 );
	float f = clamp( x, 0.0, 1.0 ) * steps;
	float i = floor( f );
	float frac = f - i;
	float soft = clamp( uToonSoftness, 0.001, 1.0 ) * 0.5;
	float w = smoothstep( 0.5 - soft, 0.5 + soft, frac );
	float band = clamp( ( i + w ) / steps, 0.0, 1.0 );
	return mix( clamp( uToonFloor, 0.0, 1.0 ), 1.0, band );
}
`;

const RIM_DECL =  `
uniform vec3 uRimColor;
uniform float uRimStrength;
uniform float uRimPower;
`;

const RIM_APPLY =  `
{
	vec3 oakViewDir = normalize( vViewPosition );
	float oakFresnel = pow( clamp( 1.0 - dot( normal, oakViewDir ), 0.0, 1.0 ), uRimPower );
	outgoingLight += uRimColor * ( oakFresnel * uRimStrength );
}
`;

export const WIND_DECL = `
uniform float uTime;
uniform float uWindAmp;
uniform float uWindFreq;
uniform float uWindHeight;
vec3 oakWind(vec3 p, vec3 origin) {
  float sway = clamp(p.y / max(uWindHeight, 0.0001), 0.0, 1.0); sway *= sway;
  float phase = uTime * uWindFreq + origin.x * 0.7 + origin.z * 1.3;
  return p + vec3(sin(phase), 0.0, cos(phase * 0.83) * 0.6) * uWindAmp * sway;
}
`;

const WIND_APPLY = `
{
  vec3 oakOrigin = modelMatrix[3].xyz;
  #ifdef USE_INSTANCING
    oakOrigin = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
  #endif
  transformed = oakWind(transformed, oakOrigin);
}
`;

const SPOT_VERT_DECL =  `
varying vec3 vOakObj;
`;

const SPOT_VERT_APPLY =  `
vOakObj = position;
`;

const SPOT_DECL =  `
uniform vec3 uSpotColor;
uniform float uSpotStrength;
uniform float uSpotScale;
uniform float uSpotRadius;
uniform float uSpotSeed;
varying vec3 vOakObj;

vec3 oakSpotHash( vec3 c ) {
	vec3 p = vec3( dot( c, vec3( 127.1, 311.7, 74.7 ) ),
	               dot( c, vec3( 269.5, 183.3, 246.1 ) ),
	               dot( c, vec3( 113.5, 271.9, 124.6 ) ) );
	return fract( sin( p + uSpotSeed ) * 43758.5453 );
}

float oakSpots( vec3 p ) {
	vec3 cell = floor( p );
	vec3 f = fract( p );
	float best = 1e9;
	for ( int i = -1; i <= 1; i++ ) {
		for ( int j = -1; j <= 1; j++ ) {
			for ( int k = -1; k <= 1; k++ ) {
				vec3 o = vec3( float( i ), float( j ), float( k ) );
				vec3 site = o + oakSpotHash( cell + o ) - f;
				best = min( best, dot( site, site ) );
			}
		}
	}
	return 1.0 - smoothstep( 0.0, uSpotRadius * uSpotRadius, best );
}
`;

const SPOT_APPLY =  `
{
	float oakSpot = oakSpots( vOakObj * uSpotScale );
	diffuseColor.rgb = mix( diffuseColor.rgb, uSpotColor, oakSpot * uSpotStrength );
}
`;

const DIFFUSE_LINE = /reflectedLight\.directDiffuse\s*\+=\s*irradiance\s*\*\s*BRDF_Lambert\(\s*material\.(diffuseContribution|diffuseColor)\s*\);/;

let warnedToon = false;

export function patchToonFragment(fragment: string): string {
  const chunk = THREE.ShaderChunk.lights_physical_pars_fragment;
  if (typeof chunk !== 'string' || !DIFFUSE_LINE.test(chunk)) {
    if (!warnedToon) {
      warnedToon = true;
      console.warn('[kit/materials] toon ramp: lights_physical_pars_fragment shape changed; skipping ramp');
    }
    return fragment;
  }
  const patchedChunk = chunk.replace(
    DIFFUSE_LINE,
    'reflectedLight.directDiffuse += ( directLight.color * oakToonRamp( dotNL ) ) * BRDF_Lambert( material.$1 );',
  );
  return fragment.replace('#include <lights_physical_pars_fragment>', patchedChunk);
}

function normalizeWind(wind: OakWindOptions | number | undefined): OakWindOptions | null {
  if (wind === undefined) return null;
  if (typeof wind === 'number') return wind > 0 ? { amplitude: wind } : null;
  return wind.amplitude > 0 ? wind : null;
}

export function oakMaterial(opts: OakMaterialOptions = {}): OakMaterial {
  const wind = normalizeWind(opts.wind);
  const windAmp = wind ? wind.amplitude : 0;
  const rimStrength = opts.rimStrength ?? 0;

  const useToon = (opts.toonSteps ?? DEFAULT_TOON_STEPS) > 0;
  const useRim = rimStrength > 0;
  const useWind = wind !== null;
  const useSpots = opts.spots !== undefined;

  const uniforms: OakUniforms = {
    uTime: { value: 0 },
    uWindAmp: { value: windAmp },
    uWindFreq: { value: wind?.frequency ?? DEFAULT_WIND_FREQ },
    uWindHeight: { value: wind?.height ?? DEFAULT_WIND_HEIGHT },
    uRimColor: { value: new THREE.Color(opts.rimColor ?? 0xffffff) },
    uRimStrength: { value: rimStrength },
    uRimPower: { value: opts.rimPower ?? DEFAULT_RIM_POWER },
    uToonSteps: { value: opts.toonSteps ?? DEFAULT_TOON_STEPS },
    uToonSoftness: { value: opts.toonSoftness ?? DEFAULT_TOON_SOFTNESS },
    uToonFloor: { value: opts.toonFloor ?? 0 },
    uSpotColor: { value: new THREE.Color(opts.spots?.color ?? 0xffffff) },
    uSpotStrength: { value: opts.spots?.strength ?? 0 },
    uSpotScale: { value: opts.spots?.scale ?? 1 },
    uSpotRadius: { value: opts.spots?.radius ?? 0.5 },
    uSpotSeed: { value: opts.spots?.seed ?? 0 },
  };

  const handle: OakHandle = {
    uniforms,
    flags: { toon: useToon, rim: useRim, wind: useWind },
    setTime(t: number) {
      uniforms.uTime.value = t;
    },
    setWind(amplitude: number, frequency?: number) {
      uniforms.uWindAmp.value = amplitude;
      if (frequency !== undefined) uniforms.uWindFreq.value = frequency;
    },
    setRim(strength: number, color?: THREE.ColorRepresentation) {
      uniforms.uRimStrength.value = strength;
      if (color !== undefined) uniforms.uRimColor.value.set(color);
    },
    setToon(steps: number, softness?: number) {
      uniforms.uToonSteps.value = steps;
      if (softness !== undefined) uniforms.uToonSoftness.value = softness;
    },
  };

  const material = new THREE.MeshStandardMaterial({
    color: opts.color ?? 0xffffff,
    roughness: opts.roughness ?? 0.85,
    metalness: opts.metalness ?? 0.0,
    flatShading: opts.flatShading ?? false,
    vertexColors: opts.vertexColors ?? false,
    transparent: opts.transparent ?? false,
    opacity: opts.opacity ?? 1,
    side: opts.side ?? THREE.FrontSide,
    emissive: opts.emissive ?? 0x000000,
    emissiveIntensity: opts.emissiveIntensity ?? 1,
    map: opts.map ?? null,
    fog: opts.fog ?? true,
  }) as OakMaterial;

  if (opts.depthWrite !== undefined) material.depthWrite = opts.depthWrite;
  material.name = opts.name ?? 'oakMaterial';

  const cacheKey = `oak|${useToon ? 1 : 0}${useRim ? 1 : 0}${useWind ? 1 : 0}`;
  material.customProgramCacheKey = () => cacheKey;

  material.onBeforeCompile = (shader) => {
    if (useToon) {
      shader.uniforms.uToonSteps = uniforms.uToonSteps;
      shader.uniforms.uToonSoftness = uniforms.uToonSoftness;
      shader.uniforms.uToonFloor = uniforms.uToonFloor;
      shader.fragmentShader = patchToonFragment(shader.fragmentShader).replace(
        '#include <common>',
        `#include <common>\n${TOON_FN}`,
      );
    }
    if (useRim) {
      shader.uniforms.uRimColor = uniforms.uRimColor;
      shader.uniforms.uRimStrength = uniforms.uRimStrength;
      shader.uniforms.uRimPower = uniforms.uRimPower;
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${RIM_DECL}`)
        .replace('#include <opaque_fragment>', `${RIM_APPLY}\n#include <opaque_fragment>`);
    }
    if (useSpots) {
      shader.uniforms.uSpotColor = uniforms.uSpotColor;
      shader.uniforms.uSpotStrength = uniforms.uSpotStrength;
      shader.uniforms.uSpotScale = uniforms.uSpotScale;
      shader.uniforms.uSpotRadius = uniforms.uSpotRadius;
      shader.uniforms.uSpotSeed = uniforms.uSpotSeed;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${SPOT_VERT_DECL}`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>\n${SPOT_VERT_APPLY}`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${SPOT_DECL}`)
        .replace('#include <color_fragment>', `#include <color_fragment>\n${SPOT_APPLY}`);
    }
    if (useWind) {
      shader.uniforms.uTime = uniforms.uTime;
      shader.uniforms.uWindAmp = uniforms.uWindAmp;
      shader.uniforms.uWindFreq = uniforms.uWindFreq;
      shader.uniforms.uWindHeight = uniforms.uWindHeight;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${WIND_DECL}`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>\n${WIND_APPLY}`);
    }
  };

  Object.defineProperty(material, 'oak', { value: handle, enumerable: false, writable: false });
  material.userData.oak = handle;
  return material;
}

export function sculptureMaterial(opts: OakMaterialOptions): OakMaterial {
  return oakMaterial({ vertexColors: true, flatShading: true, toonSteps: 3, ...opts });
}

export interface EmissiveMaterialOptions {
  transparent?: boolean;
  opacity?: number;
  depthWrite?: boolean;
  side?: THREE.Side;
  blending?: THREE.Blending;
  name?: string;
  map?: THREE.Texture | null;
  fog?: boolean;
}

export interface EmissiveMaterial extends THREE.MeshBasicMaterial {
  readonly baseColor: THREE.Color;
}

export function emissiveMaterial(
  color: THREE.ColorRepresentation,
  intensity = 1.6,
  opts: EmissiveMaterialOptions = {},
): EmissiveMaterial {
  const base = new THREE.Color(color);
  const material = new THREE.MeshBasicMaterial({
    color: base.clone().multiplyScalar(Math.max(0, intensity)),
    transparent: opts.transparent ?? false,
    opacity: opts.opacity ?? 1,
    side: opts.side ?? THREE.FrontSide,
    fog: opts.fog ?? false,
    toneMapped: true,
    map: opts.map ?? null,
  }) as EmissiveMaterial;
  if (opts.depthWrite !== undefined) material.depthWrite = opts.depthWrite;
  if (opts.blending !== undefined) material.blending = opts.blending;
  material.name = opts.name ?? 'oakEmissive';
  Object.defineProperty(material, 'baseColor', { value: base, enumerable: false, writable: false });
  material.userData.oakEmissiveIntensity = intensity;
  return material;
}

export function setEmissiveIntensity(material: EmissiveMaterial, intensity: number): void {
  material.color.copy(material.baseColor).multiplyScalar(Math.max(0, intensity));
  material.userData.oakEmissiveIntensity = intensity;
}

export function setEmissiveColor(material: EmissiveMaterial, color: THREE.ColorRepresentation): void {
  material.baseColor.set(color);
  const i = typeof material.userData.oakEmissiveIntensity === 'number' ? material.userData.oakEmissiveIntensity : 1;
  material.color.copy(material.baseColor).multiplyScalar(Math.max(0, i));
}

export interface WispMaterialOptions {
  intensity?: number;
  rimPower?: number;
  name?: string;
}

export interface WispMaterial extends THREE.ShaderMaterial {
  readonly oak: OakHandle;
  readonly pulse: NumberUniform;
}

const WISP_VERT =  `
#include <common>
varying vec3 vOakNormalV;
varying vec3 vOakViewDir;
void main() {
	#include <beginnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <project_vertex>
	vOakNormalV = normalize( transformedNormal );
	vOakViewDir = - mvPosition.xyz;
}
`;

const WISP_FRAG =  `
#include <common>
uniform vec3 uColor;
uniform float uIntensity;
uniform float uRimPower;
uniform float uBreathe;
uniform float uTime;
uniform float uPulse;
varying vec3 vOakNormalV;
varying vec3 vOakViewDir;
void main() {
	float f = 1.0 - abs( dot( normalize( vOakNormalV ), normalize( vOakViewDir ) ) );
	float core = pow( 1.0 - f, 2.0 );
	float halo = pow( f, max( uRimPower, 0.01 ) );
	float breath = 1.0 + uBreathe * sin( uTime * 2.1 );
	float gain = uIntensity * breath * ( 1.0 + uPulse );
	vec3 col = uColor * ( core * 0.65 + halo ) * gain;
	float alpha = clamp( core * 0.7 + halo, 0.0, 1.0 );
	gl_FragColor = vec4( col, alpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export function wispMaterial(color: THREE.ColorRepresentation, opts: WispMaterialOptions = {}): WispMaterial {
  const uTime: NumberUniform = { value: 0 };
  const uPulse: NumberUniform = { value: 0 };
  const uniforms = {
    uColor: { value: new THREE.Color(color) },
    uIntensity: { value: opts.intensity ?? 1.8 },
    uRimPower: { value: opts.rimPower ?? 2.2 },
    uBreathe: { value: 0.12 },
    uTime,
    uPulse,
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: WISP_VERT,
    fragmentShader: WISP_FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  }) as WispMaterial;
  material.name = opts.name ?? 'oakWisp';

  const oakUniformSet: OakUniforms = {
    uTime,
    uWindAmp: { value: 0 },
    uWindFreq: { value: 0 },
    uWindHeight: { value: 1 },
    uRimColor: uniforms.uColor,
    uRimStrength: uniforms.uIntensity,
    uRimPower: uniforms.uRimPower,
    uToonSteps: { value: 0 },
    uToonSoftness: { value: 0 },
    uToonFloor: { value: 0 },
    uSpotColor: { value: new THREE.Color(0xffffff) },
    uSpotStrength: { value: 0 },
    uSpotScale: { value: 1 },
    uSpotRadius: { value: 0.5 },
    uSpotSeed: { value: 0 },
  };
  const handle: OakHandle = {
    uniforms: oakUniformSet,
    flags: { toon: false, rim: true, wind: false },
    setTime(t: number) {
      uTime.value = t;
    },
    setWind() {
    },
    setRim(strength: number, c?: THREE.ColorRepresentation) {
      uniforms.uIntensity.value = strength;
      if (c !== undefined) uniforms.uColor.value.set(c);
    },
    setToon() {
    },
  };
  Object.defineProperty(material, 'oak', { value: handle, enumerable: false, writable: false });
  Object.defineProperty(material, 'pulse', { value: uPulse, enumerable: false, writable: false });
  material.userData.oak = handle;
  return material;
}
