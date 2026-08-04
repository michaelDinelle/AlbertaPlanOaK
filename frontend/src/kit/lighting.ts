// kit/lighting — the light budget, enforced by construction.
//
// STYLE.md: "makeLightRig presets — exactly one shadow-casting DirectionalLight, one
// HemisphereLight, ≤4 PointLights (hero). No other lights." and "Fog always present".
//
// The rig is created ONCE per scene and never adds or removes lights afterwards: point
// lights are created at intensity 0 and stay `visible` so the light counts (and hence
// the compiled programs) never change mid-playback. Scenes claim a point light, set its
// color/position/intensity, and drive it.

import * as THREE from 'three';
import { paletteSunDirection } from './sky';
import type { Palette } from './tokens';
import type { QualityTier } from './types';

// One implementation of the palette→sun-elevation rule, shared with the sky so the key
// light and the sun disc can never disagree.
export { paletteSunDirection };

/** hero 4 / medium 2 / thumb 0 — the hard budget from STYLE.md. */
export function maxPointLights(tier: QualityTier): number {
  return tier === 'hero' ? 4 : tier === 'medium' ? 2 : 0;
}

/** hero 2048 / medium 1024 / thumb 0 (no shadow). */
export function shadowMapSize(tier: QualityTier): number {
  return tier === 'hero' ? 2048 : tier === 'medium' ? 1024 : 0;
}

export interface LightRigOptions {
  /** unit vector pointing TOWARD the sun; defaults to the palette-derived elevation */
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  /** point lights to allocate (clamped to the tier max) */
  pointCount?: number;
  /** point light falloff distance (0 = infinite) */
  pointDistance?: number;
  pointDecay?: number;
  /** initial shadow focus */
  shadowRadius?: number;
  shadowCenter?: readonly [number, number, number];
  /** sun distance from the focus center, as a multiple of the focus radius */
  sunDistanceScale?: number;
  name?: string;
}

export interface LightRig {
  /** add this ONE object to the scene */
  group: THREE.Group;
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  /** length === min(opts.pointCount ?? tier max, tier max) */
  points: THREE.PointLight[];
  /** unit vector pointing toward the sun (shared with the sky when scenes pass it in) */
  sunDirection: THREE.Vector3;
  /** re-aim the key light and re-fit the shadow frustum to the region that matters */
  setShadowFocus(center: { x: number; y: number; z: number }, radius: number): void;
  setSunDirection(x: number, y: number, z: number): void;
  dispose(): void;
}

/**
 * The whole light budget for one scene.
 *
 * ```ts
 * const rig = makeLightRig(tokens.palettes.dusk, ctx.tier, { sunDir: sky.sunDirection });
 * scene.add(rig.group);
 * rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 8);   // the playfield
 * ```
 */
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

// ---------------------------------------------------------------------------
// fog — always present (STYLE.md), always token-derived
// ---------------------------------------------------------------------------

/**
 * FogExp2 whose density is derived from `palette.fogDensityHint` ("fraction of scene
 * extent where fog reaches full density"): at `hint * extent` world units, ~95% of the
 * surface color is gone. No hand-tuned magic number in any scene.
 */
export function makeFog(palette: Palette, extent: number): THREE.FogExp2 {
  const reach = Math.max(0.001, palette.fogDensityHint * Math.max(0.001, extent));
  const density = 1.7308 / reach; // sqrt(ln(20)) — transmittance 0.05 at `reach`
  return new THREE.FogExp2(palette.fog, density);
}

/** Set the scene's fog from tokens. Background stays null — the sky dome owns it. */
export function applyFog(scene: THREE.Scene, palette: Palette, extent: number): THREE.FogExp2 {
  const fog = makeFog(palette, extent);
  scene.fog = fog;
  return fog;
}

/**
 * HOST-ONLY helper. The shadow map type is a renderer setting and scenes must never
 * touch the renderer (STYLE.md), so the host calls this once per tier switch. Kept here
 * so the shadow contract (PCF, on for hero/medium, off for thumb) lives in one file.
 * three 0.185 deprecated PCFSoftShadowMap and silently substitutes PCFShadowMap — we set
 * PCFShadowMap directly to keep the identical result without the console warning.
 */
export function configureRendererShadows(renderer: THREE.WebGLRenderer, tier: QualityTier): void {
  const on = shadowMapSize(tier) > 0;
  renderer.shadowMap.enabled = on;
  renderer.shadowMap.type = THREE.PCFShadowMap;
}
