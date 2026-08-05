// kit/materials — the one stylization language for every lit surface.
//
// STYLE.md: "Stylized-PBR with soft toon ramping. kit/materials.oakMaterial() on every
// lit surface: soft 2–3-step NdotL ramp (softness ~0.25 — painterly, never hard cel),
// sky-tinted fresnel rim, optional vertex wind."
//
// Implementation notes that matter to scenes:
//  * `oakMaterial` is a real MeshStandardMaterial patched in `onBeforeCompile`, so it
//    keeps shadows, fog, instancing, vertex colors and env lighting for free. One
//    material instance can back any number of Mesh / InstancedMesh objects.
//  * `customProgramCacheKey` encodes the patch flags, so two differently-patched
//    materials never share a compiled program.
//  * Animation is driven through a per-material UNIFORM HANDLE, never by touching
//    shader internals: `oakUniforms(mat)?.setTime(ctx.time())` (or the convenience
//    `setMaterialTime`). The handle exists before the first compile and is reused by
//    every program compiled from that material.
//  * Vertex wind displaces the render pass only; the shadow depth material is
//    untouched, so wind-animated foliage casts its rest-pose shadow. That is a
//    deliberate cost/coherence trade for grass and canopies.

import * as THREE from 'three';
import type { QualityTier } from './types';

// ---------------------------------------------------------------------------
// uniform handle
// ---------------------------------------------------------------------------

export interface NumberUniform {
  value: number;
}
export interface ColorUniform {
  value: THREE.Color;
}

export interface OakUniforms {
  /** seconds — ALWAYS driven from ctx.time(), never a private clock */
  uTime: NumberUniform;
  uWindAmp: NumberUniform;
  uWindFreq: NumberUniform;
  /** world height at which sway reaches full amplitude */
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

/** A MeshStandardMaterial carrying the kit stylization + its uniform handle. */
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

/** The handle for any kit material, or null for a plain material. */
export function oakUniforms(material: THREE.Material | null | undefined): OakHandle | null {
  if (!material) return null;
  const m = material as unknown as HasOak;
  if (isOakHandle(m.oak)) return m.oak;
  const ud = m.userData?.oak;
  return isOakHandle(ud) ? ud : null;
}

/** Drive one material's clock. No-op for materials without a handle. */
export function setMaterialTime(material: THREE.Material | null | undefined, t: number): void {
  const h = oakUniforms(material);
  if (h) h.uniforms.uTime.value = t;
}

/** Drive a whole list (scenes keep one preallocated array of animated materials). */
export function setMaterialsTime(materials: readonly (THREE.Material | null | undefined)[], t: number): void {
  for (let i = 0; i < materials.length; i++) setMaterialTime(materials[i], t);
}

// ---------------------------------------------------------------------------
// oakMaterial
// ---------------------------------------------------------------------------

export interface OakWindOptions {
  /** world-unit sway at full height */
  amplitude: number;
  /** radians per second */
  frequency?: number;
  /** local height (object space) at which sway is full; below it, sway tapers to 0 */
  height?: number;
}

export interface OakMaterialOptions {
  color?: THREE.ColorRepresentation;
  roughness?: number;
  metalness?: number;
  flatShading?: boolean;
  vertexColors?: boolean;
  transparent?: boolean;
  opacity?: number;
  alphaTest?: number;
  side?: THREE.Side;
  depthWrite?: boolean;
  emissive?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
  map?: THREE.Texture | null;
  fog?: boolean;
  name?: string;

  /** 2–3 for the house look. 0 disables the ramp (physical diffuse). */
  toonSteps?: number;
  /** band blend width in [0,1]; ~0.25 is the house softness (never hard cel) */
  toonSoftness?: number;
  /** minimum diffuse response so shadowed faces stay painterly, not black */
  toonFloor?: number;

  /** sky-tinted fresnel rim — pass a palette sky color */
  rimColor?: THREE.ColorRepresentation;
  /** 0 disables the rim */
  rimStrength?: number;
  rimPower?: number;

  /** vertex wind; a bare number is the amplitude */
  wind?: OakWindOptions | number;

  /** scales rim/wind cost down for medium, off for thumb */
  tier?: QualityTier;
}

const DEFAULT_TOON_STEPS = 3;
const DEFAULT_TOON_SOFTNESS = 0.25;
const DEFAULT_RIM_POWER = 3;
const DEFAULT_WIND_FREQ = 1.1;
const DEFAULT_WIND_HEIGHT = 1;

/** rim/wind scale per tier — thumb drops both (fixed camera, tiny raster). */
export function tierEffectScale(tier: QualityTier | undefined): { rim: number; wind: number } {
  switch (tier) {
    case 'thumb':
      return { rim: 0, wind: 0 };
    case 'medium':
      return { rim: 0.75, wind: 0.7 };
    default:
      return { rim: 1, wind: 1 };
  }
}

const TOON_FN = /* glsl */ `
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

const RIM_DECL = /* glsl */ `
uniform vec3 uRimColor;
uniform float uRimStrength;
uniform float uRimPower;
`;

const RIM_APPLY = /* glsl */ `
{
	vec3 oakViewDir = normalize( vViewPosition );
	float oakFresnel = pow( clamp( 1.0 - dot( normal, oakViewDir ), 0.0, 1.0 ), uRimPower );
	outgoingLight += uRimColor * ( oakFresnel * uRimStrength );
}
`;

const WIND_DECL = /* glsl */ `
uniform float uTime;
uniform float uWindAmp;
uniform float uWindFreq;
uniform float uWindHeight;
`;

const WIND_APPLY = /* glsl */ `
{
	vec3 oakOrigin = modelMatrix[ 3 ].xyz;
	#ifdef USE_INSTANCING
		oakOrigin = ( modelMatrix * vec4( instanceMatrix[ 3 ].xyz, 1.0 ) ).xyz;
	#endif
	float oakSway = clamp( transformed.y / max( uWindHeight, 0.0001 ), 0.0, 1.0 );
	oakSway *= oakSway;
	float oakPhase = uTime * uWindFreq + oakOrigin.x * 0.7 + oakOrigin.z * 1.3;
	transformed.x += sin( oakPhase ) * uWindAmp * oakSway;
	transformed.z += cos( oakPhase * 0.83 ) * uWindAmp * 0.6 * oakSway;
}
`;

const DIFFUSE_LINE = /reflectedLight\.directDiffuse\s*\+=\s*irradiance\s*\*\s*BRDF_Lambert\(\s*material\.(diffuseContribution|diffuseColor)\s*\);/;

let warnedToon = false;

/**
 * Patch the physical lighting chunk so the DIFFUSE term reads a soft NdotL ramp while
 * specular stays physical. Returns the original fragment source unchanged (and warns
 * once) if three's chunk text ever moves — a stylization miss, never a crash.
 */
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

/**
 * The house material. Every lit surface in every scene is one of these.
 *
 * ```ts
 * const grass = oakMaterial({ color: p.groundB, rimColor: p.skyHorizon, rimStrength: 0.3,
 *                             wind: { amplitude: 0.06, frequency: 1.4, height: 0.6 }, tier });
 * // in update(): oakUniforms(grass)!.setTime(frame-independent ctx.time())
 * ```
 */
export function oakMaterial(opts: OakMaterialOptions = {}): OakMaterial {
  const scale = tierEffectScale(opts.tier);
  const wind = normalizeWind(opts.wind);
  const windAmp = wind ? wind.amplitude * scale.wind : 0;
  const rimStrength = (opts.rimStrength ?? 0) * scale.rim;

  const useToon = (opts.toonSteps ?? DEFAULT_TOON_STEPS) > 0;
  const useRim = rimStrength > 0;
  const useWind = windAmp > 0;

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

  if (opts.alphaTest !== undefined) material.alphaTest = opts.alphaTest;
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

// ---------------------------------------------------------------------------
// emissive (the only bloom feeders)
// ---------------------------------------------------------------------------

export interface EmissiveMaterialOptions {
  transparent?: boolean;
  opacity?: number;
  depthWrite?: boolean;
  side?: THREE.Side;
  blending?: THREE.Blending;
  name?: string;
  map?: THREE.Texture | null;
  /** false = the material ignores scene fog (through-the-fog markers) */
  fog?: boolean;
}

export interface EmissiveMaterial extends THREE.MeshBasicMaterial {
  /** unscaled token color (intensity is applied on top of this) */
  readonly baseColor: THREE.Color;
}

/**
 * An unlit HDR emitter. `intensity > 1` pushes the color past the bloom threshold —
 * per STYLE.md these are the ONLY things that bloom. Renders unfogged and unlit.
 */
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

/** Re-scale an emissive material's HDR intensity (lantern brightness, cue reveal). */
export function setEmissiveIntensity(material: EmissiveMaterial, intensity: number): void {
  material.color.copy(material.baseColor).multiplyScalar(Math.max(0, intensity));
  material.userData.oakEmissiveIntensity = intensity;
}

/** Tint an emissive material without losing its intensity (cue cyan ↔ magenta). */
export function setEmissiveColor(material: EmissiveMaterial, color: THREE.ColorRepresentation): void {
  material.baseColor.set(color);
  const i = typeof material.userData.oakEmissiveIntensity === 'number' ? material.userData.oakEmissiveIntensity : 1;
  material.color.copy(material.baseColor).multiplyScalar(Math.max(0, i));
}

// ---------------------------------------------------------------------------
// wisp — the shared avatar glow
// ---------------------------------------------------------------------------

export interface WispMaterialOptions {
  /** HDR multiplier; >1 blooms */
  intensity?: number;
  /** halo tightness */
  rimPower?: number;
  /** idle breathing depth in [0,1] */
  breathe?: number;
  tier?: QualityTier;
  name?: string;
}

export interface WispMaterial extends THREE.ShaderMaterial {
  readonly oak: OakHandle;
  /** 0 = idle, >0 = celebration flare (host/scene drives this from events) */
  readonly pulse: NumberUniform;
}

const WISP_VERT = /* glsl */ `
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

const WISP_FRAG = /* glsl */ `
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

/**
 * The Wisp avatar's body material: an additive, depth-write-free glow with a soft halo
 * and an idle breath. Shareable across every grid scene; drive with
 * `oakUniforms(mat)!.setTime(ctx.time())` and `mat.pulse.value` on celebration.
 */
export function wispMaterial(color: THREE.ColorRepresentation, opts: WispMaterialOptions = {}): WispMaterial {
  const scale = tierEffectScale(opts.tier);
  const uTime: NumberUniform = { value: 0 };
  const uPulse: NumberUniform = { value: 0 };
  const uniforms = {
    uColor: { value: new THREE.Color(color) },
    uIntensity: { value: opts.intensity ?? 1.8 },
    uRimPower: { value: opts.rimPower ?? 2.2 },
    uBreathe: { value: (opts.breathe ?? 0.12) * (scale.rim > 0 ? 1 : 0) },
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
  };
  const handle: OakHandle = {
    uniforms: oakUniformSet,
    flags: { toon: false, rim: true, wind: false },
    setTime(t: number) {
      uTime.value = t;
    },
    setWind() {
      /* wisps do not sway */
    },
    setRim(strength: number, c?: THREE.ColorRepresentation) {
      uniforms.uIntensity.value = strength;
      if (c !== undefined) uniforms.uColor.value.set(c);
    },
    setToon() {
      /* no ramp on an unlit glow */
    },
  };
  Object.defineProperty(material, 'oak', { value: handle, enumerable: false, writable: false });
  Object.defineProperty(material, 'pulse', { value: uPulse, enumerable: false, writable: false });
  material.userData.oak = handle;
  return material;
}
