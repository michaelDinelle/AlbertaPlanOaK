import * as THREE from 'three';
import { oakUniforms } from '@kit/materials';
import { DEFAULT_ORBIT_LIMITS, type OrbitRig, type OrbitRigOptions, type OrbitState } from '@kit/rigs';
import { makeTileBank, type TileBankOptions } from '@kit/tiles';
import type { SceneContext, SceneSize } from '@kit/types';

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
  maxDistance = DEFAULT_ORBIT_LIMITS.maxDistance,
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
    follow: 0,
    limits: { maxDistance: Math.max(maxDistance, composition.distance * 4) },
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
      coverage.nearRadius = Math.max(radius, distance * Math.max(1, reach) * (1 + tanD));
      coverage.far = Math.max(256, coverage.nearRadius * 6);
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
    uLandRadius: { value: DEFAULT_ORBIT_LIMITS.maxDistance * 16 },
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

export type SceneryBankOptions = Omit<TileBankOptions, 'tilesX' | 'tilesZ' | 'backX' | 'backZ'> & {
  tileSize: number;
};

export function makeSceneryBank(opts: SceneryBankOptions): {
  mesh: THREE.InstancedMesh;
  setView(coverage: ViewCoverage): void;
  dispose(): void;
} {
  const center = { value: new THREE.Vector2() };
  const material = opts.material.clone();
  const oak = oakUniforms(opts.material);
  if (oak) {
    Object.defineProperty(material, 'oak', { value: oak });
    material.userData.oak = oak;
  }
  const compile = opts.material.onBeforeCompile;
  material.customProgramCacheKey = () => `${opts.material.customProgramCacheKey()}|scenery-edge|${opts.tileSize}`;
  material.onBeforeCompile = (shader, renderer) => {
    compile.call(opts.material, shader, renderer);
    shader.uniforms.uSceneryCenter = center;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSceneryPosition;')
      .replace('#include <project_vertex>', `
vec4 sceneryPosition = vec4(transformed, 1.0);
#ifdef USE_INSTANCING
sceneryPosition = instanceMatrix * sceneryPosition;
#endif
vSceneryPosition = (modelMatrix * sceneryPosition).xyz;
#include <project_vertex>
`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform vec2 uSceneryCenter;\nvarying vec3 vSceneryPosition;')
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
float sceneryFade = 1.0 - smoothstep(${(opts.tileSize * 6).toFixed(3)}, ${(opts.tileSize * 7.2).toFixed(3)}, length(vSceneryPosition.xz - uSceneryCenter));
if (sceneryFade < fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453)) discard;
`);
  };
  const bank = makeTileBank({
    ...opts,
    material,
    tilesX: 17,
    tilesZ: 17,
    backX: 8,
    backZ: 8,
  });
  const source = bank.mesh;
  const capacity = source.instanceMatrix.count;
  const mesh = new THREE.InstancedMesh(opts.geometry, material, capacity);
  mesh.name = opts.name;
  mesh.castShadow = source.castShadow;
  mesh.receiveShadow = source.receiveShadow;
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage);
  opts.geometry.computeBoundingSphere();
  const bounds = opts.geometry.boundingSphere!;
  const spheres = new Float32Array(capacity * 4);
  const indices = new Uint32Array(capacity);
  const planes = new Float64Array(24).fill(Number.NaN);
  let sourceVersion = -1;
  let active = 0;
  let centerX = Number.NaN;
  let centerZ = Number.NaN;
  return {
    mesh,
    setView(coverage) {
      center.value.set(coverage.x, coverage.z);
      bank.setOrigin(Math.floor(coverage.x / opts.tileSize), Math.floor(coverage.z / opts.tileSize));
      const matrices = source.instanceMatrix.array;
      const colors = source.instanceColor!.array;
      let dirty = sourceVersion !== source.instanceMatrix.version;
      if (dirty) {
        sourceVersion = source.instanceMatrix.version;
        active = 0;
        for (let i = 0; i < capacity; i++) {
          const m = i * 16;
          const sx = Math.hypot(matrices[m]!, matrices[m + 1]!, matrices[m + 2]!);
          const sy = Math.hypot(matrices[m + 4]!, matrices[m + 5]!, matrices[m + 6]!);
          const sz = Math.hypot(matrices[m + 8]!, matrices[m + 9]!, matrices[m + 10]!);
          const scale = Math.max(sx, sy, sz);
          if (scale === 0) continue;
          const c = bounds.center;
          const s = active * 4;
          indices[active++] = i;
          spheres[s] = matrices[m]! * c.x + matrices[m + 4]! * c.y + matrices[m + 8]! * c.z + matrices[m + 12]!;
          spheres[s + 1] = matrices[m + 1]! * c.x + matrices[m + 5]! * c.y + matrices[m + 9]! * c.z + matrices[m + 13]!;
          spheres[s + 2] = matrices[m + 2]! * c.x + matrices[m + 6]! * c.y + matrices[m + 10]! * c.z + matrices[m + 14]!;
          spheres[s + 3] = bounds.radius * scale * 1.2 + 1;
        }
      }
      for (let i = 0; i < 6; i++) {
        const plane = coverage.frustum.planes[i]!;
        const p = i * 4;
        if (planes[p] !== plane.normal.x || planes[p + 1] !== plane.normal.y || planes[p + 2] !== plane.normal.z || planes[p + 3] !== plane.constant) dirty = true;
        planes[p] = plane.normal.x;
        planes[p + 1] = plane.normal.y;
        planes[p + 2] = plane.normal.z;
        planes[p + 3] = plane.constant;
      }
      if (centerX !== coverage.x || centerZ !== coverage.z) dirty = true;
      if (!dirty) return;
      centerX = coverage.x;
      centerZ = coverage.z;
      const output = mesh.instanceMatrix.array;
      const tints = mesh.instanceColor!.array;
      let visible = 0;
      for (let i = 0; i < active; i++) {
        const s = i * 4;
        const x = spheres[s]!;
        const y = spheres[s + 1]!;
        const z = spheres[s + 2]!;
        const r = spheres[s + 3]!;
        const distance = Math.hypot(x - coverage.x, z - coverage.z);
        if (distance - r > opts.tileSize * 7.2) continue;
        let outside = false;
        if (!mesh.castShadow || distance > coverage.nearRadius * 2 + r) {
          for (let p = 0; p < 24; p += 4) {
            if (planes[p]! * x + planes[p + 1]! * y + planes[p + 2]! * z + planes[p + 3]! < -r) {
              outside = true;
              break;
            }
          }
        }
        if (outside) continue;
        const index = indices[i]!;
        for (let k = 0; k < 16; k++) output[visible * 16 + k] = matrices[index * 16 + k]!;
        for (let k = 0; k < 3; k++) tints[visible * 3 + k] = colors[index * 3 + k]!;
        visible++;
      }
      mesh.count = visible;
      mesh.instanceMatrix.clearUpdateRanges();
      mesh.instanceMatrix.addUpdateRange(0, visible * 16);
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor!.clearUpdateRanges();
      mesh.instanceColor!.addUpdateRange(0, visible * 3);
      mesh.instanceColor!.needsUpdate = true;
    },
    dispose() {
      bank.dispose();
      mesh.dispose();
      mesh.removeFromParent();
      material.dispose();
    },
  };
}
