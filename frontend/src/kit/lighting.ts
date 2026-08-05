
import * as THREE from 'three';
import { paletteSunDirection } from './sky';
import type { Palette } from './tokens';
import type { QualityTier } from './types';

export { paletteSunDirection };

export function maxPointLights(tier: QualityTier): number {
  return tier === 'hero' ? 4 : tier === 'medium' ? 2 : 0;
}

export function shadowMapSize(tier: QualityTier): number {
  return tier === 'hero' ? 2048 : tier === 'medium' ? 1024 : 0;
}

export interface LightRigOptions {
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  pointCount?: number;
  pointDistance?: number;
  pointDecay?: number;
  shadowRadius?: number;
  shadowCenter?: readonly [number, number, number];
  sunDistanceScale?: number;
  name?: string;
}

export interface LightRig {
  group: THREE.Group;
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  points: THREE.PointLight[];
  sunDirection: THREE.Vector3;
  setShadowFocus(center: { x: number; y: number; z: number }, radius: number): void;
  setSunDirection(x: number, y: number, z: number): void;
  dispose(): void;
}

export function makeLightRig(palette: Palette, tier: QualityTier, opts: LightRigOptions = {}): LightRig {
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
  const mapSize = shadowMapSize(tier);
  sun.castShadow = mapSize > 0;
  if (sun.castShadow) {
    sun.shadow.mapSize.set(mapSize, mapSize);
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.02;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 100;
  }
  group.add(sun);
  group.add(sun.target);

  const hemi = new THREE.HemisphereLight(palette.hemiSky, palette.hemiGround, palette.hemiIntensity);
  hemi.name = 'oakHemi';
  group.add(hemi);

  const budget = maxPointLights(tier);
  const count = Math.max(0, Math.min(budget, Math.floor(opts.pointCount ?? budget)));
  const points: THREE.PointLight[] = [];
  for (let i = 0; i < count; i++) {
    const p = new THREE.PointLight(palette.accentLight, 0, opts.pointDistance ?? 12, opts.pointDecay ?? 2);
    p.name = `oakPoint${i}`;
    p.castShadow = false;
    points.push(p);
    group.add(p);
  }

  const sunDistanceScale = opts.sunDistanceScale ?? 2.5;
  const focus = new THREE.Vector3();

  const rig: LightRig = {
    group,
    sun,
    hemi,
    points,
    sunDirection,
    setShadowFocus(center, radius) {
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

  const c = opts.shadowCenter;
  rig.setShadowFocus(c ? { x: c[0], y: c[1], z: c[2] } : { x: 0, y: 0, z: 0 }, opts.shadowRadius ?? 10);
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

export function configureRendererShadows(renderer: THREE.WebGLRenderer, tier: QualityTier): void {
  const on = shadowMapSize(tier) > 0;
  renderer.shadowMap.enabled = on;
  renderer.shadowMap.type = THREE.PCFShadowMap;
}
