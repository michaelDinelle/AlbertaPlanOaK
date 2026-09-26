import * as THREE from 'three/webgpu';
import { Fn, abs, clamp, color, cos, diffuseColor, dot, float, floor, instanceIndex, metalness, mix, modelWorldMatrix, normalLocal, normalView, positionGeometry, positionViewDirection, pow, screenCoordinate, screenSize, sin, smoothstep, storage, transformNormal, uniform, vec2, vec3, vec4 } from 'three/tsl';

type N<T extends string> = THREE.Node<T>;

export const glFragCoord = vec2(screenCoordinate.x, screenSize.y.sub(screenCoordinate.y));

export interface NumberUniform {
  value: number;
}

export interface Deformed {
  position: N<'vec3'>;
  normal?: N<'vec3'>;
  tint?: N<'vec3'>;
}

export type Deform = (matrix: N<'mat4'> | null) => Deformed;

const deforms = new WeakMap<THREE.Material, Deform>();
const handles = new WeakMap<THREE.Material, OakHandle>();

export function deformOf(material: THREE.Material): Deform | undefined {
  return deforms.get(material);
}

export function setDeform(material: THREE.NodeMaterial, deform: Deform): void {
  deforms.set(material, deform);
  material.positionNode = Fn((builder) => {
    const object = builder.object as THREE.InstancedMesh;
    if (!object.isInstancedMesh) {
      const { position, normal } = deform(null);
      if (normal) normalLocal.assign(normal);
      return position;
    }
    const matrices = object.instanceMatrix;
    if (!(matrices instanceof THREE.StorageInstancedBufferAttribute)) throw new Error(`${object.name || 'An instanced mesh'} must be built with instancedMesh() to carry a deformed material`);
    const matrix = storage(matrices, 'mat4', matrices.count).element(instanceIndex);
    const { position, normal } = deform(matrix);
    if (normal) normalLocal.assign(transformNormal(normal, matrix));
    return matrix.mul(vec4(position, 1)).xyz;
  })();
}

export interface OakHandle {
  readonly flags: { wind: boolean };
  readonly time: NumberUniform;
  setTime(t: number): void;
  setWind(amplitude: number, frequency?: number): void;
}

export interface OakMaterial extends THREE.MeshStandardNodeMaterial {
  readonly oak: OakHandle;
}

export function oakUniforms(material: THREE.Material | null | undefined): OakHandle | null {
  return (material && handles.get(material)) ?? null;
}

export function setMaterialsTime(materials: readonly (THREE.Material | null | undefined)[], t: number): void {
  for (const material of materials) oakUniforms(material)?.setTime(t);
}

export interface OakWindOptions {
  amplitude: number;
  frequency?: number;
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

type Ramp = (x: N<'float'>) => N<'float'>;

class ToonLighting extends THREE.PhysicalLightingModel {
  constructor(private readonly ramp: Ramp) {
    super();
  }

  direct(input: THREE.LightingModelDirectInput, builder: THREE.NodeBuilder): void {
    const lightDirection = input.lightDirection as N<'vec3'>, lightColor = input.lightColor as N<'vec3'>, diffuse = input.reflectedLight.directDiffuse as N<'vec3'>;
    super.direct({ ...input, reflectedLight: { ...input.reflectedLight, directDiffuse: vec3(0).toVar() } }, builder);
    const dotNL = normalView.dot(lightDirection).clamp();
    diffuse.addAssign(lightColor.mul(this.ramp(dotNL)).mul(diffuseColor.rgb.mul(metalness.oneMinus()).mul(1 / Math.PI)));
  }
}

class OakNodeMaterial extends THREE.MeshStandardNodeMaterial {
  ramp: Ramp | null = null;
  rim: N<'vec3'> | null = null;

  setupLightingModel(): THREE.PhysicalLightingModel {
    return this.ramp ? new ToonLighting(this.ramp) : new THREE.PhysicalLightingModel();
  }

  setupOutput(builder: THREE.NodeBuilder, outputNode: N<'vec4'>): N<'vec4'> {
    return super.setupOutput(builder, this.rim ? vec4(outputNode.rgb.add(this.rim), outputNode.a) : outputNode) as N<'vec4'>;
  }

  copy(source: OakNodeMaterial): this {
    super.copy(source);
    this.ramp = source.ramp;
    this.rim = source.rim;
    return this;
  }
}

function toonRamp(steps: number, softness: number, floorLevel: number): Ramp {
  const count = Math.max(steps, 1), soft = Math.min(Math.max(softness, 0.001), 1) * 0.5, base = Math.min(Math.max(floorLevel, 0), 1);
  return (x) => {
    const f = clamp(x, 0, 1).mul(count), i = floor(f);
    const w = smoothstep(0.5 - soft, 0.5 + soft, f.sub(i));
    return mix(float(base), float(1), clamp(i.add(w).div(count), 0, 1));
  };
}

function normalizeWind(wind: OakWindOptions | number | undefined): OakWindOptions | null {
  if (wind === undefined) return null;
  if (typeof wind === 'number') return wind > 0 ? { amplitude: wind } : null;
  return wind.amplitude > 0 ? wind : null;
}

export function oakMaterial(opts: OakMaterialOptions = {}): OakMaterial {
  const wind = normalizeWind(opts.wind);
  const time = uniform(0), amplitude = uniform(wind?.amplitude ?? 0), frequency = uniform(wind?.frequency ?? DEFAULT_WIND_FREQ);
  const material = new OakNodeMaterial({
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
  }) as OakNodeMaterial & OakMaterial;
  if (opts.depthWrite !== undefined) material.depthWrite = opts.depthWrite;
  material.name = opts.name ?? 'oakMaterial';

  const steps = opts.toonSteps ?? DEFAULT_TOON_STEPS;
  if (steps > 0) material.ramp = toonRamp(steps, opts.toonSoftness ?? DEFAULT_TOON_SOFTNESS, opts.toonFloor ?? 0);
  const rimStrength = opts.rimStrength ?? 0;
  if (rimStrength > 0) {
    const fresnel = pow(clamp(dot(normalView, positionViewDirection).oneMinus(), 0, 1), opts.rimPower ?? DEFAULT_RIM_POWER);
    material.rim = color(new THREE.Color(opts.rimColor ?? 0xffffff)).mul(fresnel.mul(rimStrength));
  }
  if (wind) {
    const height = Math.max(wind.height ?? DEFAULT_WIND_HEIGHT, 0.0001);
    setDeform(material, (matrix) => {
      const origin = modelWorldMatrix.mul(matrix ? matrix.mul(vec4(0, 0, 0, 1)) : vec4(0, 0, 0, 1)).xyz;
      const sway = clamp(positionGeometry.y.div(height), 0, 1);
      const phase = time.mul(frequency).add(origin.x.mul(0.7)).add(origin.z.mul(1.3));
      return { position: positionGeometry.add(vec3(sin(phase), 0, cos(phase.mul(0.83)).mul(0.6)).mul(amplitude).mul(sway.mul(sway))) };
    });
  }

  const handle: OakHandle = {
    flags: { wind: wind !== null },
    time,
    setTime(t) { time.value = t; },
    setWind(a, f) {
      amplitude.value = a;
      if (f !== undefined) frequency.value = f;
    },
  };
  Object.defineProperty(material, 'oak', { value: handle, enumerable: false, writable: false });
  handles.set(material, handle);
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

export interface WispMaterial extends THREE.MeshBasicNodeMaterial {
  readonly time: NumberUniform;
  readonly intensity: NumberUniform;
  readonly pulse: NumberUniform;
}

export function wispMaterial(tint: THREE.ColorRepresentation, opts: WispMaterialOptions = {}): WispMaterial {
  const time = uniform(0), pulse = uniform(0), intensity = uniform(opts.intensity ?? 1.8);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }) as THREE.MeshBasicNodeMaterial & WispMaterial;
  const f = abs(dot(normalView, positionViewDirection)).oneMinus();
  const core = pow(f.oneMinus(), 2), halo = pow(f, Math.max(opts.rimPower ?? 2.2, 0.01));
  const gain = intensity.mul(sin(time.mul(2.1)).mul(0.12).add(1)).mul(pulse.add(1));
  material.colorNode = color(new THREE.Color(tint)).mul(core.mul(0.65).add(halo)).mul(gain);
  material.opacityNode = clamp(core.mul(0.7).add(halo), 0, 1);
  material.name = opts.name ?? 'oakWisp';
  Object.defineProperty(material, 'time', { value: time, enumerable: false, writable: false });
  Object.defineProperty(material, 'intensity', { value: intensity, enumerable: false, writable: false });
  Object.defineProperty(material, 'pulse', { value: pulse, enumerable: false, writable: false });
  return material;
}
