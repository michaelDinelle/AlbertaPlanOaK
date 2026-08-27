import * as THREE from 'three';
import { paletteSunDirection } from './sky';
import type { Palette } from './tokens';

export { paletteSunDirection };

export const MAX_POINT_LIGHTS = 4;
export const SHADOW_MAP_SIZE = 2048;
const SHADOW_BIAS = -0.0006;
const SHADOW_BIAS_RANGE = 104;

interface ShadowExtent {
  halfX: number;
  halfY: number;
  reach: number;
  depth: number;
}

export interface LightRigOptions {
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  pointCount?: number;
  pointDistance?: number;
  pointDecay?: number;
  sunDistanceScale?: number;
  shadowMapSize?: number | readonly [number, number];
  name?: string;
}

export interface LightRig {
  group: THREE.Group;
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  points: THREE.PointLight[];
  sunDirection: THREE.Vector3;
  setShadowFocus(center: { x: number; y: number; z: number }, radius: number): void;
  setShadowExtent(
    center: { x: number; y: number; z: number },
    halfX: number,
    halfY: number,
    reach: number,
    depth: number,
  ): void;
  setSunDirection(x: number, y: number, z: number): void;
  dispose(): void;
}

export function makeLightRig(palette: Palette, opts: LightRigOptions = {}): LightRig {
  const group = new THREE.Group();
  group.name = opts.name ?? 'oakLightRig';

  const sunDirection = new THREE.Vector3();
  const given = opts.sunDir;
  if (given) {
    const asVec = given as THREE.Vector3;
    if (typeof asVec.x === 'number') sunDirection.set(asVec.x, asVec.y, asVec.z);
    else {
      const a = given as readonly [number, number, number];
      sunDirection.set(a[0], a[1], a[2]);
    }
    if (sunDirection.lengthSq() < 1e-12) paletteSunDirection(palette, sunDirection);
    else sunDirection.normalize();
  } else {
    paletteSunDirection(palette, sunDirection);
  }

  const sun = new THREE.DirectionalLight(palette.sun, palette.sunIntensity);
  sun.name = 'oakSun';
  sun.castShadow = true;
  const mapSize = opts.shadowMapSize ?? SHADOW_MAP_SIZE;
  if (typeof mapSize === 'number') sun.shadow.mapSize.set(mapSize, mapSize);
  else sun.shadow.mapSize.set(mapSize[0], mapSize[1]);
  sun.shadow.bias = SHADOW_BIAS;
  sun.shadow.normalBias = 0.02;
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 100;
  group.add(sun);
  group.add(sun.target);

  const hemi = new THREE.HemisphereLight(palette.hemiSky, palette.hemiGround, palette.hemiIntensity);
  hemi.name = 'oakHemi';
  group.add(hemi);

  const count = Math.max(0, Math.min(MAX_POINT_LIGHTS, Math.floor(opts.pointCount ?? MAX_POINT_LIGHTS)));
  const points: THREE.PointLight[] = [];
  for (let i = 0; i < count; i++) {
    const p = new THREE.PointLight(palette.accentLight, 0, opts.pointDistance ?? 12, opts.pointDecay ?? 2);
    p.name = `oakPoint${i}`;
    p.castShadow = false;
    points.push(p);
    group.add(p);
  }

  const sunDistanceScale = opts.sunDistanceScale ?? 2.5;
  let extent: ShadowExtent | null = null;
  const focus = new THREE.Vector3();

  const rig: LightRig = {
    group,
    sun,
    hemi,
    points,
    sunDirection,
    setShadowExtent(center, halfX, halfY, reach, depth) {
      extent = { halfX, halfY, reach, depth };
      focus.set(center.x, center.y, center.z);
      sun.target.position.copy(focus);
      sun.target.updateMatrixWorld();
      sun.position.copy(focus).addScaledVector(sunDirection, reach);
      const cam = sun.shadow.camera;
      cam.left = -halfX;
      cam.right = halfX;
      cam.top = halfY;
      cam.bottom = -halfY;
      cam.near = Math.max(0.05, reach - depth);
      cam.far = reach + depth;
      cam.updateProjectionMatrix();
      sun.shadow.bias = SHADOW_BIAS * (SHADOW_BIAS_RANGE / Math.max(1, cam.far - cam.near));
    },
    setShadowFocus(center, radius) {
      extent = null;
      sun.shadow.bias = SHADOW_BIAS;
      const r = Math.max(0.5, radius);
      focus.set(center.x, center.y, center.z);
      sun.target.position.copy(focus);
      sun.target.updateMatrixWorld();
      const dist = r * sunDistanceScale;
      sun.position.copy(focus).addScaledVector(sunDirection, dist);
      const cam = sun.shadow.camera;
      cam.left = -r;
      cam.right = r;
      cam.top = r;
      cam.bottom = -r;
      cam.near = Math.max(0.05, dist - r * 2);
      cam.far = dist + r * 2;
      cam.updateProjectionMatrix();
    },
    setSunDirection(x, y, z) {
      sunDirection.set(x, y, z);
      if (sunDirection.lengthSq() < 1e-12) sunDirection.set(0, 1, 0);
      sunDirection.normalize();
      if (extent) {
        rig.setShadowExtent(
          sun.target.position,
          extent.halfX,
          extent.halfY,
          extent.reach,
          extent.depth,
        );
        return;
      }
      rig.setShadowFocus(sun.target.position, sun.shadow.camera.right);
    },
    dispose() {
      for (const p of points) {
        p.dispose();
        group.remove(p);
      }
      points.length = 0;
      sun.shadow.dispose();
      sun.dispose();
      hemi.dispose();
      group.remove(sun);
      group.remove(sun.target);
      group.remove(hemi);
      group.removeFromParent();
    },
  };

  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 10);
  return rig;
}

export function makeFog(palette: Palette, extent: number): THREE.FogExp2 {
  const reach = Math.max(0.001, palette.fogDensityHint * Math.max(0.001, extent));
  const density = 1.7308 / reach;
  return new THREE.FogExp2(palette.fog, density);
}

export function applyFog(scene: THREE.Scene, palette: Palette, extent: number): THREE.FogExp2 {
  const fog = makeFog(palette, extent);
  scene.fog = fog;
  return fog;
}
