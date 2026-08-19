import { TERRACE_DESCENT, cellToTerraceWorld, terraceShelves } from '@kit/terrace';

import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { blendTokens, clamp01, smoothstep01, TOKEN_WHITE } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  LARGE_DEEPSEA_N,
  largeDeepSeaMask,
  largeDeepSeaMaskWords,
} from '@sim/envs/largedeepsea';
import type { EnvGeometry, GridGeometry } from '@sim/types';

export const ABYSS_N = LARGE_DEEPSEA_N;
export const TRENCH_HALF = (ABYSS_N * CELL) / 2;

export const SHELF_THICKNESS = 3.2;
export const BED_NORTH = -0.9;
export const BED_SOUTH = -2.1;
export const TERRAIN_SIZE = 88;
export const TERRAIN_SEGMENTS = 160;
export const WALL_RUN = 15;
export const WALL_HEIGHT = 3.4;
export const ROCK_GAIN = 2.2;

export const FOG_EXTENT = 68;

export const SUN_DIR: readonly [number, number, number] = [-0.3, 0.93, -0.22];
export const SHADOW_RADIUS = 13.5;

export const CAMERA_AZIMUTH = -2.52;

export function hue(c: THREE.Color): THREE.Color {
  const m = Math.max(c.r, c.g, c.b, 1e-6);
  return c.multiplyScalar(1 / m);
}

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === ABYSS_N && g.h === ABYSS_N) return g;
  return {
    w: ABYSS_N,
    h: ABYSS_N,
    walls: new Uint8Array(ABYSS_N * ABYSS_N),
    goal: { x: ABYSS_N - 1, y: ABYSS_N - 1 },
  };
}

export function resolveMask(grid: GridGeometry, seed: number): Uint8Array {
  const m = grid.actionMask;
  if (m && m.length >= grid.w * grid.h) return m;
  return largeDeepSeaMask(largeDeepSeaMaskWords(seed));
}

export function isShelfCell(x: number, y: number): boolean {
  return x <= y;
}

export interface AbyssSkin {
  shelfLit: THREE.Color;
  shelfMid: THREE.Color;
  shelfDeep: THREE.Color;
  bed: THREE.Color;
  bedDeep: THREE.Color;
  rock: THREE.Color;
  rockLip: THREE.Color;
  coral: THREE.Color;
  coralTip: THREE.Color;
  timber: THREE.Color;
  timberDark: THREE.Color;
  anglerBody: THREE.Color;
  heart: THREE.Color;
  jellyBell: THREE.Color;
  jellyTrail: THREE.Color;
  snow: THREE.Color;
  silt: THREE.Color;
}

export function abyssSkin(tokens: typeof TOKENS): AbyssSkin {
  const abyss = tokens.palettes.abyss;
  const water = tokens.palettes.underwater;
  const accent = tokens.accents['large-deep-sea'] ?? tokens.uiCyan;
  const reef = tokens.accents['deep-sea'] ?? water.groundB;

  const rockLip = hue(blendTokens(abyss.groundB, water.groundA, 0.45));
  const snow = blendTokens(water.accentLight, abyss.hemiSky, 0.4);

  return {
    shelfLit: hue(blendTokens(water.groundB, reef, 0.35)),
    shelfMid: hue(blendTokens(water.groundA, abyss.groundB, 0.4)),
    shelfDeep: hue(blendTokens(abyss.skyTop, tokens.cueMagenta, 0.06)),
    bed: hue(blendTokens(abyss.groundB, water.groundA, 0.32)),
    bedDeep: hue(blendTokens(abyss.groundA, abyss.skyBottom, 0.4)),
    rock: hue(blendTokens(abyss.groundB, abyss.skyTop, 0.3)),
    rockLip,
    coral: hue(blendTokens(abyss.groundB, water.groundA, 0.4)),
    coralTip: blendTokens(accent, water.groundB, 0.42),
    timber: hue(blendTokens(tokens.palettes.dusk.groundA, abyss.groundB, 0.45)).multiplyScalar(0.85),
    timberDark: hue(blendTokens(abyss.groundA, tokens.palettes.dusk.groundA, 0.3)).multiplyScalar(0.45),
    anglerBody: hue(blendTokens(abyss.groundA, abyss.skyBottom, 0.45)).multiplyScalar(0.34),
    heart: blendTokens(tokens.celebrationGold, accent, 0.3),
    jellyBell: blendTokens(accent, water.skyTop, 0.28),
    jellyTrail: blendTokens(tokens.cueMagenta, accent, 0.55),
    snow,
    silt: blendTokens(rockLip, snow, 0.34),
  };
}

export function shelfTint(row: number, n: number, skin: AbyssSkin, out?: THREE.Color): THREE.Color {
  const c = out ?? new THREE.Color();
  const t = n > 1 ? clamp01(row / (n - 1)) : 0;
  const handover = 0.45;
  if (t <= handover) {
    c.copy(skin.shelfLit).lerp(skin.shelfMid, smoothstep01(t / handover));
  } else {
    const k = (t - handover) / (1 - handover);
    c.copy(skin.shelfMid).lerp(skin.shelfDeep, k * k * (1.35 - 0.35 * k));
  }
  return attenuate(c, t);
}

export const EXTINCTION: readonly [number, number, number] = [3.3, 2.9, 2.4];
export const SURFACE_GAIN = 11;

export function attenuate(c: THREE.Color, t: number): THREE.Color {
  const d = clamp01(t);
  c.r *= SURFACE_GAIN * Math.exp(-EXTINCTION[0] * d);
  c.g *= SURFACE_GAIN * Math.exp(-EXTINCTION[1] * d);
  c.b *= SURFACE_GAIN * Math.exp(-EXTINCTION[2] * d);
  return c;
}

export function makeTrenchHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const outside = Math.max(Math.abs(x), Math.abs(z)) - TRENCH_HALF;
    const wall = smoothstep01(outside / WALL_RUN);
    const bedT = clamp01((z + TRENCH_HALF) / (2 * TRENCH_HALF));
    const bed = BED_NORTH + (BED_SOUTH - BED_NORTH) * bedT - TERRACE_DESCENT * bedT;
    const n1 = valueNoise2(x * 0.062 + 3.1, z * 0.062 + 8.7, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.173 + 1.7, z * 0.173 + 4.3, 64, seed ^ 0x51ed270b) - 0.5;
    const relief = n1 * 2.1 + n2 * 0.75;
    return bed + relief * (0.18 + 6.2 * wall) + wall * WALL_HEIGHT;
  };
}

export function makeTrenchColor(skin: AbyssSkin): (h: number, slope: number, x: number, z: number) => THREE.Color {
  const scratch = new THREE.Color();
  const climb = new THREE.Color();
  return (h, _slope, _x, z) => {
    const depthT = clamp01((z + TRENCH_HALF) / (2 * TRENCH_HALF));
    const wallT = clamp01((h - (BED_SOUTH - TERRACE_DESCENT)) / (WALL_HEIGHT * 0.7));
    scratch.copy(skin.bed).lerp(skin.bedDeep, depthT * depthT * (1.4 - 0.4 * depthT));
    attenuate(scratch, depthT).multiplyScalar(0.62);
    if (wallT <= 0) return scratch;
    climb
      .copy(skin.rock)
      .lerp(skin.rockLip, clamp01((wallT - 0.5) * 2))
      .multiplyScalar(ROCK_GAIN * (0.42 + 0.58 * (1 - depthT)));
    return scratch.lerp(climb, smoothstep01(wallT));
  };
}

const JELLY_VERT =  `
#include <common>
uniform float uTime;
uniform float uDrift;
uniform float uPulse;
attribute vec2 aJelly;   // x = phase 0..1, y = stroke speed
attribute float aGrade;  // 1 at the crown, 0 at the ribbon tips
varying float vGrade;
varying vec3 vNormalV;
varying vec3 vViewDir;
void main() {
	float phase = aJelly.x * 6.2831853;
	float speed = aJelly.y;
	float stroke = sin( uTime * speed + phase );
	float lag = sin( uTime * speed - 1.05 + phase );

	vec3 p = position;
	float ribbon = 1.0 - aGrade;
	p.y *= 1.0 + uPulse * stroke * aGrade;
	p.xz *= 1.0 - uPulse * 0.55 * stroke * aGrade;
	p.x += ribbon * ribbon * lag * 0.11;
	p.z += ribbon * ribbon * cos( uTime * speed * 0.73 + phase ) * 0.085;

	vec4 local = instanceMatrix * vec4( p, 1.0 );
	vec3 origin = instanceMatrix[ 3 ].xyz;
	local.x += sin( uTime * 0.19 + origin.z * 0.55 + phase ) * uDrift;
	local.y += sin( uTime * 0.27 + phase ) * uDrift * 0.55;
	local.z += cos( uTime * 0.23 + origin.x * 0.47 + phase ) * uDrift;

	vec4 mv = modelViewMatrix * local;
	vNormalV = normalize( normalMatrix * ( mat3( instanceMatrix ) * normal ) );
	vViewDir = - mv.xyz;
	vGrade = aGrade;
	gl_Position = projectionMatrix * mv;
}
`;

const JELLY_FRAG =  `
#include <common>
uniform vec3 uBell;
uniform vec3 uTrail;
uniform float uIntensity;
varying float vGrade;
varying vec3 vNormalV;
varying vec3 vViewDir;
void main() {
	float f = 1.0 - abs( dot( normalize( vNormalV ), normalize( vViewDir ) ) );
	float rim = pow( f, 1.9 );
	vec3 col = mix( uTrail, uBell, vGrade ) * ( 0.28 + 1.45 * rim ) * uIntensity;
	float a = clamp( ( 0.16 + 0.6 * rim ) * ( 0.22 + 0.78 * vGrade ), 0.0, 1.0 );
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( col, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export interface JellySwarmHandle {
  mesh: THREE.InstancedMesh;
  material: THREE.ShaderMaterial;
  count: number;
  setTime(t: number): void;
  dispose(): void;
}

export function makeJellyGeometry(detail: number): THREE.BufferGeometry {
  const segments = Math.max(6, Math.round(10 * detail));
  const rings = Math.max(2, Math.round(3 * detail));
  const ribbons = Math.max(3, Math.round(5 * detail));
  const RAD = 0.13;
  const BELL = 0.17;
  const TAIL = 0.34;

  const positions: number[] = [];
  const grades: number[] = [];

  const bellPoint = (ring: number, seg: number): readonly [number, number, number] => {
    const t = ring / rings;
    const theta = t * Math.PI * 0.54;
    const a = (seg / segments) * Math.PI * 2;
    const scallop = 1 + 0.08 * Math.sin(a * segments * 0.5) * t;
    const r = Math.sin(theta) * RAD * scallop;
    const y = Math.cos(theta) * BELL;
    return [Math.cos(a) * r, y, Math.sin(a) * r];
  };

  const push = (p: readonly [number, number, number], g: number): void => {
    positions.push(p[0], p[1], p[2]);
    grades.push(g);
  };

  for (let ring = 0; ring < rings; ring++) {
    const g0 = 1 - 0.28 * (ring / rings);
    const g1 = 1 - 0.28 * ((ring + 1) / rings);
    for (let seg = 0; seg < segments; seg++) {
      const a = bellPoint(ring, seg);
      const b = bellPoint(ring, seg + 1);
      const c = bellPoint(ring + 1, seg + 1);
      const d = bellPoint(ring + 1, seg);
      push(a, g0);
      push(c, g1);
      push(b, g0);
      push(a, g0);
      push(d, g1);
      push(c, g1);
    }
  }

  const rimY = Math.cos(Math.PI * 0.54) * BELL;
  const rimR = Math.sin(Math.PI * 0.54) * RAD;
  for (let i = 0; i < ribbons; i++) {
    const a = (i / ribbons) * Math.PI * 2 + 0.31;
    const cx = Math.cos(a);
    const cz = Math.sin(a);
    const nx = -cz;
    const nz = cx;
    const w = 0.05;
    const len = TAIL * (0.72 + 0.28 * (((i * 5) % 3) / 2));
    const x0 = cx * rimR * 0.86;
    const z0 = cz * rimR * 0.86;
    const x1 = cx * rimR * 0.34;
    const z1 = cz * rimR * 0.34;
    const top: readonly [number, number, number] = [x0 - nx * w, rimY, z0 - nz * w];
    const top2: readonly [number, number, number] = [x0 + nx * w, rimY, z0 + nz * w];
    const mid: readonly [number, number, number] = [x1 - nx * w * 0.5, rimY - len * 0.55, z1 - nz * w * 0.5];
    const mid2: readonly [number, number, number] = [x1 + nx * w * 0.5, rimY - len * 0.55, z1 + nz * w * 0.5];
    const tip: readonly [number, number, number] = [x1 * 0.4, rimY - len, z1 * 0.4];
    push(top, 0.55);
    push(mid, 0.26);
    push(top2, 0.55);
    push(top2, 0.55);
    push(mid, 0.26);
    push(mid2, 0.26);
    push(mid, 0.26);
    push(tip, 0);
    push(mid2, 0.26);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  g.setAttribute('aGrade', new THREE.BufferAttribute(new Float32Array(grades), 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'abyssJelly';
  return g;
}

export interface JellySwarmOptions {
  grid: GridSize;
  count: number;
  seed: number;
  bell: THREE.ColorRepresentation;
  trail: THREE.ColorRepresentation;
  intensity: number;
  name?: string;
}

export function buildJellySwarm(opts: JellySwarmOptions): JellySwarmHandle {
  const detail = 1;
  const count = Math.max(1, opts.count);
  const geometry = makeJellyGeometry(detail);

  const jelly = new Float32Array(count * 2);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uDrift: { value: 0.42 },
      uPulse: { value: 0.2 },
      uBell: { value: new THREE.Color(opts.bell) },
      uTrail: { value: new THREE.Color(opts.trail) },
      uIntensity: { value: opts.intensity },
    },
    vertexShader: JELLY_VERT,
    fragmentShader: JELLY_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  material.name = opts.name ?? 'abyssJellySwarm';

  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = opts.name ?? 'abyssJellySwarm';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;

  let placed = 0;
  let attempt = 0;
  while (placed < count && attempt < count * 40) {
    const h = hash2f(attempt, 17, opts.seed ^ 0x4a11ef);
    const h2 = hash2f(attempt, 41, opts.seed ^ 0x1d05b3);
    const h3 = hash2f(attempt, 73, opts.seed ^ 0x77c0de);
    const h4 = hash2f(attempt, 109, opts.seed ^ 0x2be115);
    attempt++;
    const cx = h * opts.grid.w;
    const cy = h2 * opts.grid.h;
    const world: Vec3Tuple = [0, 0, 0];
    cellToWorld(cx - 0.5, cy - 0.5, opts.grid, world);
    const depth = clamp01(cy / opts.grid.h);
    const open = isShelfCell(Math.floor(cx), Math.floor(cy)) ? 0 : 1;
    const y = 0.55 - open * 1.1 + h3 * (0.9 + 1.5 * depth);
    const s = 0.58 + h4 * 0.62;
    position.set(world[0], y, world[2]);
    euler.set((h3 - 0.5) * 0.5, h * Math.PI * 2, (h4 - 0.5) * 0.5);
    quaternion.setFromEuler(euler);
    scale.set(s, s * (0.85 + h3 * 0.4), s);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(placed, matrix);
    jelly[placed * 2] = h4;
    jelly[placed * 2 + 1] = 0.55 + h3 * 0.75;
    placed++;
  }
  mesh.count = placed;
  mesh.instanceMatrix.needsUpdate = true;
  geometry.setAttribute('aJelly', new THREE.InstancedBufferAttribute(jelly, 2));

  return {
    mesh,
    material,
    count: placed,
    setTime(t) {
      (material.uniforms.uTime as { value: number }).value = t;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

export const CORAL_LEAN = 0.32;
export const CORAL_HEIGHT = 0.52;

const LEAN_AXIS = new THREE.Vector3(0, 0, 1);

export interface CoralFieldHandle {
  mesh: THREE.InstancedMesh;
  setTime(t: number): void;
  dispose(): void;
}

function coralWhip(
  height: number,
  base: number,
  tip: number,
  x: number,
  z: number,
  tilt: number,
  yaw: number,
): THREE.BufferGeometry {
  const whip = new THREE.CylinderGeometry(tip, base, height, 5, 1, true);
  whip.translate(0, height / 2, 0);
  whip.rotateZ(tilt);
  whip.rotateY(yaw);
  whip.translate(x, 0.02, z);
  return whip;
}

export function makeCoralGeometry(): THREE.BufferGeometry {
  const holdfast = new THREE.CylinderGeometry(0.045, 0.075, 0.05, 6, 1);
  holdfast.translate(0, 0.025, 0);
  const parts: THREE.BufferGeometry[] = [
    coralWhip(CORAL_HEIGHT, 0.022, 0.005, 0, 0, 0, 0),
    coralWhip(0.33, 0.016, 0.004, 0.045, 0.02, 0.42, 0.7),
    coralWhip(0.26, 0.014, 0.004, -0.04, -0.03, -0.5, -0.4),
    holdfast,
  ];

  const geometry = mergeColoredParts(
    parts.map((part) => ({ geometry: part, color: TOKEN_WHITE })),
    'abyssCoral',
  );
  const position = geometry.getAttribute('position');
  const color = geometry.getAttribute('color');
  for (let i = 0; i < position.count; i++) {
    const k = 0.45 + 1.25 * clamp01(position.getY(i) / CORAL_HEIGHT);
    color.setXYZ(i, k, k, k);
  }
  return geometry;
}

export interface CoralFieldOptions {
  grid: GridSize;
  mask: Uint8Array;
  skin: AbyssSkin;
  seed: number;
  name?: string;
}

export function buildCoralField(opts: CoralFieldOptions): CoralFieldHandle {
  const { grid, mask, skin } = opts;
  const geometry = makeCoralGeometry();

  const material = oakMaterial({
    color: skin.coral,
    vertexColors: true,
    flatShading: true,
    roughness: 0.85,
    side: THREE.DoubleSide,
    rimColor: skin.coralTip,
    rimStrength: 0.65,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    wind: { amplitude: 0.028, frequency: 0.62, height: 0.5 },
    name: 'abyssCoralMaterial',
  });

  let capacity = 0;
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) if (isShelfCell(x, y)) capacity++;
  }

  const mesh = new THREE.InstancedMesh(geometry, material, capacity);
  mesh.name = opts.name ?? 'abyssCoral';
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const lean = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const shelf: Vec3Tuple = [0, 0, 0];
  const tint = new THREE.Color();
  const biolume = new THREE.Color();

  let n = 0;
  for (let y = 0; y < grid.h; y++) {
    const rowColor = shelfTint(y, grid.h, skin).multiplyScalar(0.8);
    const rowGlow = 0.06 + 0.2 * clamp01(y / (grid.h - 1));
    for (let x = 0; x < grid.w; x++) {
      if (!isShelfCell(x, y)) continue;
      const h = hash2f(x, y, opts.seed ^ 0x0c02a1);
      const h2 = hash2f(x, y, opts.seed ^ 0x5eaf00);
      cellToTerraceWorld(x, y, grid, shelf);
      position.set(
        shelf[0] + (0.24 + 0.12 * h) * (h2 > 0.5 ? 1 : -1),
        shelf[1],
        shelf[2] + (0.22 + 0.13 * h2) * (h > 0.5 ? 1 : -1),
      );
      const current = (mask[y * grid.w + x] ?? 0) !== 0 ? 1 : -1;
      lean.setFromAxisAngle(LEAN_AXIS, -current * CORAL_LEAN * (0.8 + 0.4 * h));
      const s = 0.7 + 0.55 * h2;
      scale.set(s, s * (0.85 + 0.4 * h), s);
      mesh.setMatrixAt(n, matrix.compose(position, lean, scale));
      biolume.copy(skin.coralTip).multiplyScalar(rowGlow);
      mesh.setColorAt(n, tint.copy(rowColor).add(biolume));
      n++;
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  return {
    mesh,
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

export interface AnglerSite {
  x: number;
  y: number;
  hover: number;
  scale: number;
  phase: number;
}

export function anglerSites(grid: GridSize, seed: number): AnglerSite[] {
  const a = hash2f(3, 11, seed ^ 0x104e);
  const b = hash2f(7, 23, seed ^ 0x2a71);
  const c = hash2f(13, 5, seed ^ 0x39b2);
  const d = hash2f(17, 29, seed ^ 0x4c08);
  const lastRow = Math.max(1, grid.h - 2);
  const rowA = Math.min(lastRow, 6 + Math.floor(a * 4));
  const rowB = Math.min(lastRow, 11 + Math.floor(c * 4));
  const colA = Math.max(2, Math.min(rowA - 1, rowA - 1 - Math.floor(b * 3)));
  const colB = Math.max(4, Math.min(rowB - 1, rowB - 2 - Math.floor(d * 5)));
  return [
    { x: colA, y: rowA, hover: 1.4, scale: 0.95, phase: 0.6 },
    { x: colB, y: rowB, hover: 1.75, scale: 1.25, phase: 3.1 },
  ];
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  coral: CoralFieldHandle;
  jellies: JellySwarmHandle;
  rush: AmbientParticlesHandle;
  silt: AmbientParticlesHandle;
  skin: AbyssSkin;
}

export function buildWorld(
  ctx: SceneContext,
  scene: THREE.Scene,
  grid: GridGeometry,
  mask: Uint8Array,
): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.abyss;
  const water: Palette = ctx.tokens.palettes.underwater;
  const skin = abyssSkin(ctx.tokens);
  const w = makeWorld('theAbyss');

  const sky = skyFromPalette(palette, 'abyss', {
    topSpread: 0.42,
    bottomSpread: 0.62,
    radius: 320,
    dither: 0.01,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: SUN_DIR,
    pointCount: 4,
    pointDistance: 9,
    sunDistanceScale: 2.4,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: -0.3, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    seed: ctx.seed,
    heightFn: makeTrenchHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.07,
    slopeColor: skin.rock,
    slopeStrength: 0.65,
    slopeSharpness: 1.8,
    colorRamp: makeTrenchColor(skin),
    materialOptions: {
      rimColor: water.skyHorizon,
      rimStrength: 0.22,
      toonSteps: 3,
      toonSoftness: 0.32,
      toonFloor: 0.3,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'trenchBed',
  });
  w.add(terrain.mesh);
  w.own(terrain);

  const field = buildCellField(
    grid,
    {
      floorColor: TOKEN_WHITE,
      wallColor: TOKEN_WHITE,
      floorFilter: isShelfCell,
      floorTint: (_x, y) => shelfTint(y, grid.h, skin),
      floorInset: 0.035,
      floorThickness: SHELF_THICKNESS,
      jitter: 0.08,
      rimColor: water.skyTop,
      rimStrength: 0.18,
      seed: ctx.seed,
      name: 'abyssShelves',
    },
  );
  terraceShelves(field, grid);
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const coral = buildCoralField({ grid, mask, skin, seed: ctx.seed });
  w.add(coral.mesh);
  w.time(w.own(coral));

  const jellyCount = 46;
  const jellies = buildJellySwarm({
    grid,
    count: jellyCount,
    seed: ctx.seed,
    bell: skin.jellyBell,
    trail: skin.jellyTrail,
    intensity: 1.35,
    name: 'abyssJellies',
  });
  w.add(jellies.mesh);
  w.time(w.own(jellies));

  const snow = makeAmbientParticles({
    preset: 'snow',
    count: 330,
    bounds: { x: 0, y: 2.1, z: 0, halfX: TRENCH_HALF, halfY: 3.0, halfZ: TRENCH_HALF },
    seed: ctx.seed ^ 0x5f0a,
    color: skin.snow,
    size: 0.8,
    intensity: 0.85,
    opacity: 0.32,
    rise: -0.11,
    swirl: 0.12,
    name: 'marineSnow',
  });
  w.add(snow.points);
  w.size(w.time(w.own(snow)));

  const bubbles = makeAmbientParticles({
    preset: 'bubbles',
    count: 120,
    bounds: { x: -TRENCH_HALF * 0.45, y: 1.9, z: -TRENCH_HALF * 0.55, halfX: 3.4, halfY: 2.2, halfZ: 3.0 },
    seed: ctx.seed ^ 0x2b17,
    color: water.accentLight,
    size: 1.0,
    intensity: 1.35,
    opacity: 0.4,
    rise: 0.5,
    swirl: 0.1,
    name: 'shelfBubbles',
  });
  w.add(bubbles.points);
  w.size(w.time(w.own(bubbles)));

  const silt = makeAmbientParticles({
    preset: 'dust',
    count: 120,
    bounds: { x: 0, y: 0, z: 0, halfX: 0.44, halfY: 0.13, halfZ: 0.44 },
    seed: ctx.seed ^ 0x1d5e,
    color: skin.silt,
    size: 1.1,
    intensity: 1.3,
    opacity: 0.42,
    rise: -0.09,
    swirl: 0.07,
    name: 'diveSilt',
  });
  silt.setDensity(0);
  w.add(silt.points);
  w.size(w.time(w.own(silt)));

  const rush = makeAmbientParticles({
    preset: 'bubbles',
    count: 210,
    bounds: { x: 0, y: 0, z: 0, halfX: 0.32, halfY: 1.5, halfZ: 0.32 },
    seed: ctx.seed ^ 0x7a91,
    color: water.accentLight,
    size: 1.7,
    intensity: 2.4,
    opacity: 0.8,
    rise: 1.7,
    swirl: 0.07,
    name: 'resurfaceRush',
  });
  rush.setDensity(0);
  w.add(rush.points);
  w.size(w.time(w.own(rush)));

  const haze = makeMistPlanes({
    size: [TERRAIN_SIZE * 0.26, TERRAIN_SIZE * 0.26],
    color: blendTokens(water.skyHorizon, palette.skyTop, 0.45),
    seed: ctx.seed ^ 0x3c40,
    layers: 4,
    baseY: 2.4,
    spread: 4.4,
    opacity: 0.07,
    speed: 0.006,
    scale: 1.8,
    nearFade: 6,
    name: 'abyssHaze',
  });
  haze.group.position.set(-TRENCH_HALF * 0.55, 0, -TRENCH_HALF * 0.55);
  w.add(haze.group);
  w.time(w.own(haze));

  return w.seal({
    sky,
    rig,
    terrain,
    field,
    coral,
    jellies,
    rush,
    silt,
    skin,
  });
}
