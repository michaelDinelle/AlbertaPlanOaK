import * as THREE from 'three/webgpu';
import { abs, dot, exp, float, floor, fract, log, max, mix, normalize, positionGeometry, sign, sin, uniform, varying, vec2, vec3 } from 'three/tsl';
import { oakMaterial, setDeform, type OakMaterial, type OakMaterialOptions } from './materials';
import { DEFAULT_ORBIT_LIMITS, type OrbitRig, type OrbitRigOptions, type OrbitState } from './rigs';
import { makeTileBank, type TileBankOptions } from './tiles';
import type { FrameUpdate, SceneContext, SceneSize } from './types';

type N<T extends string> = THREE.Node<T>;

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
    sum += w * (frame.poseAt(Math.min(hi, Math.max(lo, i)))[slot] ?? 0);
    weight += w;
  }
  return sum / weight;
}

export interface SurfaceSample {
  p: N<'vec3'>;
  t: N<'float'>;
  a: N<'vec3'>;
  b: N<'vec3'>;
  color: N<'vec3'>;
  noise(q: N<'vec2'>): N<'float'>;
}

export interface ContinuousSurfaceOptions {
  name: string;
  seed?: number;
  colorA: THREE.ColorRepresentation;
  colorB: THREE.ColorRepresentation;
  height?: (p: N<'vec2'>, spacing: N<'float'>) => N<'float'>;
  detail?: (s: SurfaceSample) => N<'vec3'>;
  roughness?: number;
  metalness?: number;
  emissive?: THREE.ColorRepresentation;
  segments?: number;
  radius?: number;
  follow?: boolean;
  anchor?: readonly [number, number];
  finish?: Omit<OakMaterialOptions, 'wind'>;
}

export interface ContinuousSurface {
  mesh: THREE.Mesh<THREE.PlaneGeometry, OakMaterial>;
  material: OakMaterial;
  setView(coverage: ViewCoverage): void;
  setTime(time: number): void;
  dispose(): void;
}

export function makeContinuousSurface(opts: ContinuousSurfaceOptions): ContinuousSurface {
  const segments = opts.segments ?? 96;
  const geometry = new THREE.PlaneGeometry(2, 2, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const follow = opts.follow ?? true, anchor = opts.anchor ?? [0, 0];
  const center = uniform(new THREE.Vector2(anchor[0], anchor[1])), radius = uniform(opts.radius ?? DEFAULT_ORBIT_LIMITS.maxDistance * 16);
  const seed = uniform((opts.seed ?? 0) % 65536), time = uniform(0), height = opts.height ?? (() => float(0));
  const scale = log(radius.add(1)), offset = (x: N<'float'>) => sign(x).mul(exp(abs(x).mul(scale)).sub(1));
  const land = center.add(vec2(offset(positionGeometry.x), offset(positionGeometry.z))), reach = land.sub(center);
  const spacing = scale.mul(max(abs(reach.x), abs(reach.y)).add(1)).mul(2).div(uniform(segments));
  const step = max(0.15, spacing.mul(0.5)), dx = vec2(step, 0), dz = vec2(0, step);
  const position = vec3(land.x, height(land, spacing), land.y);
  const material = oakMaterial({ ...(opts.finish ?? { roughness: opts.roughness ?? 0.94, toonSteps: 0 }), color: 0xffffff, metalness: opts.metalness ?? 0, emissive: opts.emissive ?? 0, side: THREE.DoubleSide, name: `${opts.name}Material` });
  setDeform(material, () => ({ position, normal: normalize(vec3(height(land.sub(dx), spacing).sub(height(land.add(dx), spacing)), step.mul(2), height(land.sub(dz), spacing).sub(height(land.add(dz), spacing)))) }));
  const hash = (q: N<'vec2'>) => fract(sin(dot(q, vec2(127.1, 311.7)).add(seed)).mul(43758.5453123));
  const noise = (q: N<'vec2'>) => {
    const i = floor(q), s = fract(q), f = s.mul(s).mul(float(3).sub(s.mul(2)));
    return mix(mix(hash(i), hash(i.add(vec2(1, 0))), f.x), mix(hash(i.add(vec2(0, 1))), hash(i.add(1)), f.x), f.y);
  };
  const rgb = (value: THREE.ColorRepresentation) => vec3(new THREE.Vector3().setFromColor(new THREE.Color(value)));
  const p = varying(position), a = rgb(opts.colorA), b = rgb(opts.colorB);
  const tone = mix(a, b, noise(p.xz.mul(0.027)).mul(0.55).add(noise(p.xz.mul(0.113)).mul(0.3)).add(noise(p.xz.mul(0.47)).mul(0.15)));
  material.colorNode = opts.detail?.({ p, t: time, a, b, color: tone, noise }) ?? tone;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name;
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  return {
    mesh,
    material,
    setView(coverage) {
      if (!follow) return;
      const spacing = Math.expm1(2 * Math.log1p(radius.value) / segments);
      center.value.set(
        Math.round(coverage.x / spacing) * spacing,
        Math.round(coverage.z / spacing) * spacing,
      );
    },
    setTime(t) {
      time.value = t;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

export type SceneryBankOptions = TileBankOptions & {
  tileSize: number;
};

export function makeSceneryBank(opts: SceneryBankOptions): {
  mesh: THREE.Group;
  setView(coverage: ViewCoverage): void;
  dispose(): void;
} {
  const bank = makeTileBank(opts);
  return {
    mesh: bank.mesh,
    setView(coverage) {
      bank.setOrigin(Math.floor(coverage.x / opts.tileSize), Math.floor(coverage.z / opts.tileSize), Math.ceil(coverage.nearRadius / opts.tileSize) + 8);
    },
    dispose() { bank.dispose(); },
  };
}
