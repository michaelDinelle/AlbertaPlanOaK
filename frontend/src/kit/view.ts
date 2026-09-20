import * as THREE from 'three';
import { oakUniforms } from './materials';
import { DEFAULT_ORBIT_LIMITS, type OrbitRig, type OrbitRigOptions, type OrbitState } from './rigs';
import { makeTileBank, type TileBankOptions } from './tiles';
import type { FrameUpdate, SceneContext, SceneSize } from './types';

export type ViewAnchor = readonly [number, number, number];

export interface SceneBounds {
  min: ViewAnchor;
  max: ViewAnchor;
}

export interface ViewCoverage {
  x: number;
  z: number;
  radius: number;
  nearRadius: number;
  far: number;
  frustum: THREE.Frustum;
}

export interface SceneView {
  options: OrbitRigOptions;
  update(rig: OrbitRig, dtReal: number, anchor?: ViewAnchor): ViewCoverage;
  resize(size: SceneSize, rig: OrbitRig): void;
}

export function makeSceneView(
  ctx: SceneContext,
  bounds: SceneBounds,
  anchor: ViewAnchor,
  composition: OrbitState,
  minFar = 256,
): SceneView {
  const radius = Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ) / 2;
  const tanV = Math.tan(ctx.tokens.fovHero * Math.PI / 360);
  const coverage: ViewCoverage = { x: anchor[0], z: anchor[2], radius: 256, nearRadius: composition.distance, far: 256, frustum: new THREE.Frustum() };
  const projection = new THREE.Matrix4();
  let attachedRig: OrbitRig | null = null;
  let aspectScale = 1;
  const options: OrbitRigOptions = {
    azimuth: composition.azimuth,
    polar: composition.polar,
    target: anchor,
    distance: composition.distance,
    limits: { maxDistance: Math.max(composition.distance, Math.min(DEFAULT_ORBIT_LIMITS.maxDistance, composition.distance * 4)), maxPolar: Math.max(DEFAULT_ORBIT_LIMITS.maxPolar, composition.polar) },
  };
  return {
    options,
    update(rig, dtReal, next = anchor) {
      if (attachedRig !== rig) {
        const pan = rig.pan;
        rig.pan = (dx, dy) => pan(dx * aspectScale, dy * aspectScale);
        attachedRig = rig;
      }
      aspectScale = Math.max(1, (16 / 9) / rig.camera.aspect);
      rig.setTarget(next[0], next[1], next[2]);
      rig.update(dtReal);
      rig.camera.position.sub(rig.target).multiplyScalar(aspectScale).add(rig.target);
      const distance = rig.state.distance * aspectScale;
      const tanD = Math.hypot(tanV, tanV * rig.camera.aspect);
      const position = rig.camera.position;
      const level = (next[1] - position.y) / (rig.target.y - position.y);
      const reach = Math.max(0, level);
      coverage.x = position.x + (rig.target.x - position.x) * reach;
      coverage.z = position.z + (rig.target.z - position.z) * reach;
      coverage.nearRadius = distance * Math.max(1, reach) * (1 + tanD);
      coverage.far = Math.max(minFar, Math.max(radius, coverage.nearRadius) * 6);
      coverage.radius = coverage.far * Math.hypot(1, tanD) + distance;
      if (rig.camera.far !== coverage.far) {
        rig.camera.far = coverage.far;
        rig.camera.updateProjectionMatrix();
      }
      rig.camera.updateMatrixWorld();
      projection.multiplyMatrices(rig.camera.projectionMatrix, rig.camera.matrixWorldInverse);
      coverage.frustum.setFromProjectionMatrix(projection);
      return coverage;
    },
    resize(size, rig) {
      const aspect = size.width / Math.max(1, size.height);
      rig.resize(aspect);
    },
  };
}

const TRACK_SECONDS = 0.5;

export function trackPose(frame: FrameUpdate, slot: number): number {
  const half = Math.ceil(TRACK_SECONDS * frame.controlHz);
  const centre = Math.floor(frame.position);
  let lo = centre - half, hi = centre + half + 1;
  for (let i = centre + 1; i <= hi; i++) if (frame.frameAt(i).reset) { hi = i - 1; break; }
  for (let i = centre; i > lo; i--) if (frame.frameAt(i).reset) { lo = i; break; }
  let sum = 0, weight = 0;
  for (let i = centre - half; i <= centre + half + 1; i++) {
    const w = 1 + Math.cos(Math.PI * Math.min(1, Math.abs(i - frame.position) / (half + 1)));
    sum += w * (frame.frameAt(Math.min(hi, Math.max(lo, i))).pose[slot] ?? 0);
    weight += w;
  }
  return sum / weight;
}

export interface ContinuousSurfaceOptions {
  name: string;
  seed?: number;
  colorA: THREE.ColorRepresentation;
  colorB: THREE.ColorRepresentation;
  height?: string;
  detail?: string;
  roughness?: number;
  metalness?: number;
  emissive?: THREE.ColorRepresentation;
  segments?: number;
  radius?: number;
}

export interface ContinuousSurface {
  mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>;
  material: THREE.MeshStandardMaterial;
  setView(coverage: ViewCoverage): void;
  setTime(time: number): void;
  dispose(): void;
}

const SURFACE_NOISE = `
float landHash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7)) + uLandSeed) * 43758.5453123);
}
float landNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(landHash(i), landHash(i + vec2(1.0, 0.0)), f.x),
    mix(landHash(i + vec2(0.0, 1.0)), landHash(i + vec2(1.0)), f.x), f.y);
}
`;

export function makeContinuousSurface(opts: ContinuousSurfaceOptions): ContinuousSurface {
  const segments = opts.segments ?? 96;
  const geometry = new THREE.PlaneGeometry(2, 2, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const uniforms = {
    uLandCenter: { value: new THREE.Vector2() },
    uLandRadius: { value: opts.radius ?? DEFAULT_ORBIT_LIMITS.maxDistance * 16 },
    uLandSeed: { value: (opts.seed ?? 0) % 65536 },
    uLandTime: { value: 0 },
    uLandA: { value: new THREE.Color(opts.colorA) },
    uLandB: { value: new THREE.Color(opts.colorB) },
  };
  const height = opts.height ?? 'return 0.0;';
  const declarations = `
uniform vec2 uLandCenter;
uniform float uLandRadius;
uniform float uLandSeed;
uniform float uLandTime;
varying vec3 vLandPosition;
${SURFACE_NOISE}
float landHeight(vec2 p) {
  ${height}
}
float landOffset(float p) {
  return sign(p) * (exp(abs(p) * log(1.0 + uLandRadius)) - 1.0);
}
`;
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: opts.roughness ?? 0.94,
    metalness: opts.metalness ?? 0,
    emissive: opts.emissive ?? 0,
    side: THREE.DoubleSide,
  });
  material.name = `${opts.name}Material`;
  material.customProgramCacheKey = () => `${opts.name}|${height}|${opts.detail ?? ''}`;
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${declarations}`)
      .replace('#include <beginnormal_vertex>', `
vec2 landP = uLandCenter + vec2(landOffset(position.x), landOffset(position.z));
vec2 sampleStep = vec2(0.15, 0.0);
vec3 objectNormal = normalize(vec3(
  landHeight(landP - sampleStep.xy) - landHeight(landP + sampleStep.xy), 0.3,
  landHeight(landP - sampleStep.yx) - landHeight(landP + sampleStep.yx)));
`)
      .replace('#include <begin_vertex>', `
vec3 transformed = vec3(landP.x, landHeight(landP), landP.y);
vLandPosition = transformed;
`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
uniform float uLandSeed;
uniform float uLandTime;
uniform vec3 uLandA;
uniform vec3 uLandB;
varying vec3 vLandPosition;
${SURFACE_NOISE}
`)
      .replace('#include <color_fragment>', `#include <color_fragment>
vec3 p = vLandPosition;
float t = uLandTime;
float landTone = 0.55 * landNoise(p.xz * 0.027) + 0.3 * landNoise(p.xz * 0.113) + 0.15 * landNoise(p.xz * 0.47);
vec3 color = mix(uLandA, uLandB, landTone);
${opts.detail ?? ''}
diffuseColor.rgb *= color;
`);
  };
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name;
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  return {
    mesh,
    material,
    setView(coverage) {
      const spacing = Math.expm1(2 * Math.log1p(uniforms.uLandRadius.value) / segments);
      uniforms.uLandCenter.value.set(
        Math.round(coverage.x / spacing) * spacing,
        Math.round(coverage.z / spacing) * spacing,
      );
    },
    setTime(time) {
      uniforms.uLandTime.value = time;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

export type SceneryBankOptions = TileBankOptions & {
  depth?: THREE.Material;
  tileSize: number;
};

export function makeSceneryBank(opts: SceneryBankOptions): {
  mesh: THREE.Group;
  setView(coverage: ViewCoverage): void;
  dispose(): void;
  staticInstances: import('./instances').SpatialInstances[];
} {
  const oak = oakUniforms(opts.material);
  const bank = makeTileBank(opts);
  return {
    mesh: bank.mesh,
    staticInstances: opts.castShadow && !oak?.flags.wind && !(opts.material instanceof THREE.ShaderMaterial) ? [bank.instances] : [],
    setView(coverage) {
      bank.setOrigin(Math.floor(coverage.x / opts.tileSize), Math.floor(coverage.z / opts.tileSize), Math.ceil(coverage.nearRadius / opts.tileSize) + 8);
    },
    dispose() { bank.dispose(); },
  };
}
