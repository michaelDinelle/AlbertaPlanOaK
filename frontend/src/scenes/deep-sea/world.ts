import { terraceShelves } from '@kit/terrace';

import {
  buildCellField,
  cellToWorld,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { makeFog, makeLightRig, type LightRig } from '@kit/lighting';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle, type MistPlanesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeDataTexture } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { excludeAny, scatterFlowers, scatterRocks, type ExcludeFn } from '@kit/vegetation';
import * as THREE from 'three';
import type { GridGeometry } from '@sim/types';
import {
  rowDrop,
  blendTokens,
  SHELF_THICKNESS,
  TREASURE_COL,
  TREASURE_ROW,
  cellToReefWorld,
  colAtX,
  currentDir,
  isReef,
  isReefLip,
  reefDepthY,
  reefSkin,
  rowAtZ,
  shelfTint,
  type ReefSkin,
} from './reef';


export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.83, 0.32];
export const SHADOW_RADIUS = 7.6;
export const FOG_EXTENT = 96;
export const CAMERA_AZIMUTH = 0.78;

export const SURFACE_Y = 7.2;
export const SHAFT_LENGTH = 9.5;
export const SHAFT_RADIUS_MIN = 2.0;
export const SHAFT_RADIUS_MAX = 8.0;


export function shelfHeightAt(_x: number, z: number, grid: GridSize): number {
  return reefDepthY(rowAtZ(z, grid), grid);
}

export function excludeOpenWater(grid: GridSize, reach: Uint8Array): ExcludeFn {
  return (x, z) => !isReef(reach, grid, colAtX(x, grid), rowAtZ(z, grid));
}

export function excludeCellCenters(grid: GridSize, radius: number): ExcludeFn {
  const r2 = radius * radius;
  const ox = (grid.w - 1) / 2;
  const oz = (grid.h - 1) / 2;
  return (x, z) => {
    const cx = Math.round(x + ox);
    const cz = Math.round(z + oz);
    const dx = x - (cx - ox);
    const dz = z - (cz - oz);
    return dx * dx + dz * dz < r2;
  };
}




export const KELP_LEAN = 0.42;
export const KELP_HEIGHT = 0.74;

export function makeKelpGeometry(): THREE.BufferGeometry {
  const detail = 1;
  const blades = Math.max(3, Math.round(5 * detail));
  const segments = Math.max(2, Math.round(4 * detail));
  const quads = blades * segments;
  const positions = new Float32Array(quads * 6 * 3);
  const colors = new Float32Array(quads * 6 * 3);
  let o = 0;

  const push = (x: number, y: number, z: number, k: number): void => {
    positions[o * 3] = x;
    positions[o * 3 + 1] = y;
    positions[o * 3 + 2] = z;
    colors[o * 3] = k;
    colors[o * 3 + 1] = k;
    colors[o * 3 + 2] = k;
    o++;
  };

  for (let b = 0; b < blades; b++) {
    const yaw = (b / blades) * Math.PI * 2 + ((b * 7) % 5) * 0.19;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const height = KELP_HEIGHT * (0.72 + 0.28 * (((b * 3) % 4) / 3));
    const offset = 0.06 + 0.05 * (((b * 5) % 3) / 2);
    for (let s = 0; s < segments; s++) {
      const t0 = s / segments;
      const t1 = (s + 1) / segments;
      const y0 = t0 * height;
      const y1 = t1 * height;
      const w0 = 0.055 * (1 - 0.78 * t0);
      const w1 = 0.055 * (1 - 0.78 * t1);
      const s0 = Math.sin(t0 * 2.4) * 0.055;
      const s1 = Math.sin(t1 * 2.4) * 0.055;
      const k0 = 0.42 + 0.7 * t0;
      const k1 = 0.42 + 0.7 * t1;
      const ax = cos * (offset + s0);
      const az = sin * (offset + s0);
      const bx = cos * (offset + s1);
      const bz = sin * (offset + s1);
      const nx = -sin;
      const nz = cos;
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(ax + nx * w0, y0, az + nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(bx - nx * w1, y1, bz - nz * w1, k1);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'reefKelpClump';
  return g;
}

export interface KelpHandle {
  mesh: THREE.InstancedMesh;
  material: OakMaterial;
  count: number;
  setTime(t: number): void;
  dispose(): void;
}

export function buildKelp(
  grid: GridGeometry,
  reach: Uint8Array,
  skin: ReefSkin,
  palette: Palette,
  seed: number,
): KelpHandle {
  const geometry = makeKelpGeometry();
  const material = oakMaterial({
    color: skin.kelp,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.88,
    rimColor: palette.hemiSky,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.3,
    wind: { amplitude: 0.05, frequency: 0.62, height: KELP_HEIGHT },
    name: 'reefKelp',
  });

  let capacity = 0;
  for (let r = 0; r < grid.h; r++) {
    for (let c = 0; c < grid.w; c++) if (isReef(reach, grid, c, r)) capacity++;
  }

  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, capacity));
  mesh.name = 'reefKelp';
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  mesh.count = 0;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, capacity) * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'ZYX');
  const scale = new THREE.Vector3();
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];
  let n = 0;

  for (let r = 0; r < grid.h; r++) {
    for (let c = 0; c < grid.w; c++) {
      if (!isReef(reach, grid, c, r)) continue;
      cellToReefWorld(c, r, grid, world);
      const jitter = hash2f(c, r, seed ^ 0x4b3f19);
      const jitter2 = hash2f(c, r, seed ^ 0x1d77c3);
      const dir = currentDir(grid, c, r);
      position.set(world[0] + (jitter - 0.5) * 0.16, world[1], world[2] + (jitter2 - 0.5) * 0.16);
      euler.set(0, jitter * Math.PI * 2, -dir * KELP_LEAN * (0.78 + 0.34 * jitter2));
      quaternion.setFromEuler(euler);
      const s = 0.78 + 0.42 * jitter2;
      scale.set(s, s * (0.86 + 0.3 * jitter), s);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(n, matrix);
      const depth = grid.h > 1 ? r / (grid.h - 1) : 0;
      const k = (1.1 - 0.42 * depth) * (0.88 + 0.24 * jitter);
      color.setRGB(k, k * (1 + 0.05 * (jitter2 - 0.5)), k * (1 - 0.05 * (jitter2 - 0.5)));
      mesh.setColorAt(n, color);
      n++;
    }
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  return {
    mesh,
    material,
    count: n,
    setTime(t) {
      oakUniforms(material)?.setTime(t);
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}


const CAUSTIC_VERT =  `
#include <common>
varying vec3 vReefWorld;
varying vec2 vReefUv;
void main() {
	vReefUv = uv;
	vec4 world4 = modelMatrix * vec4( position, 1.0 );
	vReefWorld = world4.xyz;
	gl_Position = projectionMatrix * viewMatrix * world4;
}
`;

const CAUSTIC_FRAG =  `
#include <common>
uniform float uTime;
uniform vec3 uColor;
uniform float uStrength;
uniform float uScale;
uniform float uSpeed;
uniform float uTop;
uniform float uRange;
varying vec3 vReefWorld;
varying vec2 vReefUv;

float reefCaustic( vec2 p, float t ) {
	vec2 a = p * 1.7 + vec2( t * 0.09, t * 0.06 );
	vec2 b = p * 2.3 - vec2( t * 0.07, t * 0.11 );
	float ca = abs( fract( a.x + a.y ) - 0.5 ) + abs( fract( a.x - a.y ) - 0.5 );
	float cb = abs( fract( b.x + b.y ) - 0.5 ) + abs( fract( b.x - b.y ) - 0.5 );
	float c = 1.0 - clamp( min( ca, cb ) * 2.4, 0.0, 1.0 );
	return pow( c, 3.0 );
}

void main() {
	float web = reefCaustic( vReefWorld.xz * uScale, uTime * uSpeed );
	// The surface light thins with depth, and it has to thin FAST. This decal is additive,
	// so a lazy falloff re-lights the deep shelves in surface-teal and erases the whole
	// depth ramp the rock is tinted with — the reef went flat for three passes on exactly
	// this. Cubed, the bottom terrace keeps ~4% of the web and the top keeps all of it.
	float depth = clamp( ( vReefWorld.y - ( uTop - uRange ) ) / max( uRange, 0.001 ), 0.0, 1.0 );
	// feather each shelf quad so its silhouette never shows as a bright rectangle
	vec2 e = abs( vReefUv - 0.5 ) * 2.0;
	float edge = ( 1.0 - smoothstep( 0.62, 1.0, e.x ) ) * ( 1.0 - smoothstep( 0.62, 1.0, e.y ) );
	float a = web * uStrength * ( 0.04 + 0.96 * depth * depth * depth ) * edge;
	if ( a <= 0.003 ) discard;
	gl_FragColor = vec4( uColor * a, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export interface CausticHandle {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  setTime(t: number): void;
  dispose(): void;
}

export function buildCaustics(
  grid: GridSize,
  reach: Uint8Array,
  skin: ReefSkin,
): CausticHandle {
  const rows: { x0: number; x1: number; z: number; y: number }[] = [];
  const world: Vec3Tuple = [0, 0, 0];
  for (let r = 0; r < grid.h; r++) {
    let lo = -1;
    let hi = -1;
    for (let c = 0; c < grid.w; c++) {
      if (!isReef(reach, grid, c, r)) continue;
      if (lo < 0) lo = c;
      hi = c;
    }
    if (lo < 0) continue;
    cellToWorld(lo, r, grid, world);
    const x0 = world[0] - 0.5;
    const z = world[2];
    cellToWorld(hi, r, grid, world);
    rows.push({ x0, x1: world[0] + 0.5, z, y: reefDepthY(r, grid) + 0.016 });
  }

  const positions = new Float32Array(rows.length * 6 * 3);
  const uvs = new Float32Array(rows.length * 6 * 2);
  let o = 0;
  for (const row of rows) {
    const z0 = row.z - 0.5;
    const z1 = row.z + 0.5;
    const quad: readonly (readonly [number, number, number, number])[] = [
      [row.x0, z0, 0, 0],
      [row.x1, z0, 1, 0],
      [row.x1, z1, 1, 1],
      [row.x0, z0, 0, 0],
      [row.x1, z1, 1, 1],
      [row.x0, z1, 0, 1],
    ];
    for (const v of quad) {
      positions[o * 3] = v[0];
      positions[o * 3 + 1] = row.y;
      positions[o * 3 + 2] = v[1];
      uvs[o * 2] = v[2];
      uvs[o * 2 + 1] = v[3];
      o++;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.computeBoundingSphere();
  geometry.name = 'reefCaustics';

  const top = reefDepthY(0, grid);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(skin.caustic) },
      uStrength: { value: 1.55 },
      uScale: { value: 0.62 },
      uSpeed: { value: 1.0 },
      uTop: { value: top },
      uRange: { value: rowDrop(grid) * Math.max(1, grid.h - 1) + 0.001 },
    },
    vertexShader: CAUSTIC_VERT,
    fragmentShader: CAUSTIC_FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    fog: false,
  });
  material.name = 'reefCaustics';

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'reefCaustics';
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;

  return {
    mesh,
    material,
    setTime(t) {
      (material.uniforms.uTime as { value: number }).value = t;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}


export function makeShaftSprite(size = 64): THREE.DataTexture {
  const s = Math.max(8, size);
  const data = new Uint8Array(s * s * 4);
  for (let y = 0; y < s; y++) {
    const v = (y + 0.5) / s;
    const along = Math.pow(v, 1.5);
    for (let x = 0; x < s; x++) {
      const u = (x + 0.5) / s;
      const across = Math.pow(Math.cos((u - 0.5) * Math.PI), 2.2);
      const a = Math.max(0, Math.min(1, along * across));
      const o = (y * s + x) * 4;
      data[o] = 255;
      data[o + 1] = 255;
      data[o + 2] = 255;
      data[o + 3] = (a * 255) | 0;
    }
  }
  const tex = makeDataTexture(data, s, s, 'reefShaft');
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  return tex;
}

export interface GodRayHandle {
  group: THREE.Group;
  material: EmissiveMaterial;
  setTime(t: number): void;
  dispose(): void;
}

export function buildGodRays(
  sunDir: THREE.Vector3,
  skin: ReefSkin,
  seed: number,
): GodRayHandle {
  const group = new THREE.Group();
  group.name = 'reefGodRays';
  const shafts = 9;
  const planes = 3;
  const positions = new Float32Array(shafts * planes * 6 * 3);
  const uvs = new Float32Array(shafts * planes * 6 * 2);
  let o = 0;

  for (let i = 0; i < shafts; i++) {
    const h1 = hash2f(i, 11, seed ^ 0x2f6b1d);
    const h2 = hash2f(i, 29, seed ^ 0x7c1a55);
    const h3 = hash2f(i, 47, seed ^ 0x39d0e7);
    const radius = SHAFT_RADIUS_MIN + (SHAFT_RADIUS_MAX - SHAFT_RADIUS_MIN) * h1;
    const angle = h2 * Math.PI * 2;
    const cx = Math.cos(angle) * radius;
    const cz = Math.sin(angle) * radius;
    const width = 0.34 + 0.62 * h3;
    const length = SHAFT_LENGTH * (0.6 + 0.55 * h1);
    for (let p = 0; p < planes; p++) {
      const a = (p / planes) * Math.PI + h3 * 0.7;
      const dx = Math.cos(a) * width * 0.5;
      const dz = Math.sin(a) * width * 0.5;
      const quad: readonly (readonly [number, number, number, number, number])[] = [
        [cx - dx, 0, cz - dz, 0, 1],
        [cx + dx, 0, cz + dz, 1, 1],
        [cx + dx, -length, cz + dz, 1, 0],
        [cx - dx, 0, cz - dz, 0, 1],
        [cx + dx, -length, cz + dz, 1, 0],
        [cx - dx, -length, cz - dz, 0, 0],
      ];
      for (const v of quad) {
        positions[o * 3] = v[0];
        positions[o * 3 + 1] = v[1];
        positions[o * 3 + 2] = v[2];
        uvs[o * 2] = v[3];
        uvs[o * 2 + 1] = v[4];
        o++;
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.computeBoundingSphere();
  geometry.name = 'reefGodRays';

  const sprite = makeShaftSprite(64);
  const material = emissiveMaterial(skin.godRay, 0.95, {
    map: sprite,
    transparent: true,
    opacity: 0.11,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    name: 'reefGodRays',
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'reefGodRayShafts';
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), sunDir.clone().normalize());
  mesh.position.y = SURFACE_Y;
  group.add(mesh);

  return {
    group,
    material,
    setTime(t) {
      group.rotation.y = Math.sin(t * 0.055) * 0.09;
      material.opacity = 0.095 + 0.028 * Math.sin(t * 0.31);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      sprite.dispose();
      mesh.removeFromParent();
      group.removeFromParent();
      group.clear();
    },
  };
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  kelp: KelpHandle;
  skin: ReefSkin;
  silt: AmbientParticlesHandle;
  rush: AmbientParticlesHandle;
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

export function buildWorld(
  ctx: SceneContext,
  scene: THREE.Scene,
  grid: GridGeometry,
  reach: Uint8Array,
): WorldHandle {
  const ctxTokens = ctx.tokens;
  const palette: Palette = ctxTokens.palettes.underwater;
  const skin = reefSkin(palette, ctx.tokens);
  const group = new THREE.Group();
  group.name = 'sunlitDescent';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const pixelated: AmbientParticlesHandle[] = [];

  const sky = skyFromPalette(palette, 'underwater', {
    topSpread: 0.34,
    bottomSpread: 0.88,
    dither: 0.012,
    radius: 300,
  });
  sky.setColors(
    blendTokens(palette.skyTop, palette.sun, 0.3),
    blendTokens(palette.skyHorizon, palette.skyBottom, 0.62),
    blendTokens(palette.skyBottom, ctxTokens.palettes.abyss.skyBottom, 0.62),
  );
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: SUN_DIR,
    pointCount: 3,
    pointDistance: 9,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: reefDepthY((grid.h - 1) / 2, grid), z: 0 }, SHADOW_RADIUS);

  const fog = makeFog(palette, FOG_EXTENT);
  fog.color.set(blendTokens(palette.fog, palette.skyBottom, 0.62));
  scene.fog = fog;

  const tint = new THREE.Color();
  const white = ctxTokens.palettes.noon.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorFilter: (x, y) => isReef(reach, grid, x, y),
      floorTint: (x, y) => {
        shelfTint(skin, grid, x, y, tint);
        return isReefLip(reach, grid, x, y) ? tint.clone().lerp(skin.shelfLip, 0.45) : tint.clone();
      },
      floorInset: 0.022,
      floorThickness: SHELF_THICKNESS,
      jitter: 0.05,
      rimColor: palette.hemiSky,
      rimStrength: 0.55,
      seed: ctx.seed,
      name: 'reefShelves',
    },
  );
  terraceShelves(field, grid);
  const floorMesh = field.floorMeshes[0];
  if (floorMesh) floorMesh.castShadow = true;
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const kelp = buildKelp(grid, reach, skin, palette, ctx.seed);
  group.add(kelp.mesh);
  owned.push(kelp);
  timed.push(kelp);

  const offReef = excludeAny(excludeOpenWater(grid, reach), excludeCellCenters(grid, 0.34));
  const reefArea = { x: 0, z: 0, halfX: (grid.w * 0.5) - 0.1, halfZ: (grid.h * 0.5) - 0.1 };
  const coral = scatterRocks({
    count: 220,
    area: reefArea,
    seed: ctx.seed ^ 0x51c0,
    exclude: offReef,
    heightAt: (x, z) => shelfHeightAt(x, z, grid),
    color: skin.coral,
    scale: [0.08, 0.2],
    stretch: [0.5, 1.5],
    hueJitter: 0.3,
    rimColor: palette.hemiSky,
    rimStrength: 0.36,
    castShadow: false,
    name: 'reefCoral',
  });
  group.add(coral.group);
  owned.push(coral);
  timed.push(coral);

  const anemones = scatterFlowers({
    count: 120,
    area: reefArea,
    seed: ctx.seed ^ 0x3a91,
    exclude: offReef,
    heightAt: (x, z) => shelfHeightAt(x, z, grid),
    color: skin.anemone,
    scale: [0.2, 0.42],
    stretch: [0.7, 1.2],
    hueJitter: 0.26,
    wind: 0.035,
    windFrequency: 0.7,
    rimColor: palette.hemiSky,
    rimStrength: 0.44,
    castShadow: false,
    name: 'reefAnemones',
  });
  group.add(anemones.group);
  owned.push(anemones);
  timed.push(anemones);

  const caustics = buildCaustics(grid, reach, skin);
  group.add(caustics.mesh);
  owned.push(caustics);
  timed.push(caustics);

  let godRays: GodRayHandle | null = null;
  {
    godRays = buildGodRays(rig.sunDirection, skin, ctx.seed);
    group.add(godRays.group);
    owned.push(godRays);
    timed.push(godRays);
  }

  const haze: MistPlanesHandle = makeMistPlanes({
    size: [34, 34],
    color: skin.bubble,
    seed: ctx.seed ^ 0x1b7d,
    layers: 3,
    baseY: -4.6,
    spread: 9.4,
    opacity: 0.03,
    speed: 0.009,
    scale: 2.2,
    nearFade: 5,
    name: 'reefHaze',
  });
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const bubbles = makeAmbientParticles({
    preset: 'bubbles',
    count: 150,
    bounds: { x: 0, y: -0.4, z: 0, halfX: 4.6, halfY: 4.4, halfZ: 4.6 },
    seed: ctx.seed ^ 0x6ec2,
    color: skin.bubble,
    size: 1.0,
    intensity: 1.25,
    opacity: 0.3,
    rise: 0.5,
    swirl: 0.14,
    name: 'reefBubbles',
  });
  group.add(bubbles.points);
  owned.push(bubbles);
  timed.push(bubbles);
  pixelated.push(bubbles);

  const silt = makeAmbientParticles({
    preset: 'dust',
    count: 110,
    bounds: { x: 0, y: 0, z: 0, halfX: 0.34, halfY: 0.2, halfZ: 0.34 },
    seed: ctx.seed ^ 0x77a3,
    color: skin.silt,
    size: 0.85,
    intensity: 1.1,
    opacity: 0.42,
    rise: -0.07,
    swirl: 0.06,
    name: 'reefSilt',
  });
  silt.setDensity(0);
  group.add(silt.points);
  owned.push(silt);
  timed.push(silt);
  pixelated.push(silt);

  const rush = makeAmbientParticles({
    preset: 'bubbles',
    count: 220,
    bounds: { x: 0, y: 0, z: 0, halfX: 0.26, halfY: 3.6, halfZ: 0.26 },
    seed: ctx.seed ^ 0x2c58,
    color: skin.bubble,
    size: 1.5,
    intensity: 2.0,
    opacity: 0.72,
    rise: 2.4,
    swirl: 0.1,
    name: 'reefResurfaceRush',
  });
  rush.setDensity(0);
  group.add(rush.points);
  owned.push(rush);
  timed.push(rush);
  pixelated.push(rush);

  return {
    group,
    sky,
    rig,
    field,
    kelp,
    skin,
    silt,
    rush,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < pixelated.length; i++) pixelated[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      pixelated.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

export function treasureCell(grid: GridGeometry): { c: number; r: number } {
  const goal = grid.goal;
  if (goal) return { c: goal.x, r: goal.y };
  return { c: TREASURE_COL, r: TREASURE_ROW };
}
