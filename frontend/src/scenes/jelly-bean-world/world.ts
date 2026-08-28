import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { emissiveMaterial, oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01, luminance, shade } from '@kit/mix';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { makeTileBank } from '@kit/tiles';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { flowerGeometry, grassClumpGeometry } from '@kit/vegetation';
import { radialSprite } from '@kit/textures';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  buildBeacon,
  HALO_GLOW,
  HALO_WHITEN,
  DECAL_CAPACITY,
  DRAW_RADIUS,
  ITEM_CAPACITY,
  ITEM_GLOW,
  ITEM_KINDS,
  ITEM_MOTION,
  orchardItemGeometry,
  type BeaconHandle,
} from './actors';
import {
  JBW_BANANA,
  JBW_COLOR,
  JBW_JELLY,
  JBW_ONION,
  JBW_TREE,
  JBW_TRUFFLE,
  JBW_WALL,
  jbwCollectible,
} from '@sim/envs/jellybeanworld';
import { FIELD_SIZE, type ScentField } from './scent';

export const GROUND_PERIOD = 256;
export const GROUND_EXTENT = 1900;
export const GROUND_SEGMENTS = 560;
export const FOG_EXTENT = 1980;
export const RELIEF = 6.2;
export const SCATTER_TILE = 24;
export const SCATTER_SPAN = 33;
export const SCATTER_BACK = 16;
export const SHADOW_REACH = 300;
export const SHADOW_SPAN = 108;
const SLAB_LOW = -10;
const SLAB_HIGH = 15;
const SHADOW_MAP_X = 4096;
const SHADOW_MAP_Y = 1365;
const MAP_MARK = 0.42;
const ZERO_MOTION = { hover: 0, bob: 0, spin: 0 };
const DECAL_DIM = 0.22;
const BEACON_CAPACITY = 512;
const SHELL_SCALE = 1.06;

export const CAM_DISTANCE = 78;
export const CAM_MAX_DISTANCE = 460;
export const MAP_START = 190;
export const MAP_FULL = 420;
export const CAM_NEAR = 0.6;
export const CAM_FAR = 3600;
export const CAM_AZIMUTH = 0.68;
export const SUN_AZIMUTH = CAM_AZIMUTH + Math.PI;
export const CAM_POLAR = 1.2;
export const CAM_FOLLOW = 2.6;

const FRONT_WIDTH = 44;

export interface OrchardSkin {
  grassLow: THREE.Color;
  grassHigh: THREE.Color;
  soil: THREE.Color;
  banana: THREE.Color;
  onion: THREE.Color;
  jelly: THREE.Color;
  truffle: THREE.Color;
  stone: THREE.Color;
  moss: THREE.Color;
  bark: THREE.Color;
  canopyLow: THREE.Color;
  canopyHigh: THREE.Color;
  papery: THREE.Color;
  bloom: THREE.Color;
  scentA: THREE.Color;
  scentB: THREE.Color;
  scentC: THREE.Color;
  shaft: THREE.Color;
  ripe: THREE.Color;
  spoiled: THREE.Color;
  front: THREE.Color;
  mapGround: THREE.Color;
  legendary: THREE.Color;
}

function itemGlow(item: number, skin: OrchardSkin, target: number): number {
  if (!jbwCollectible(item)) return 0;
  const luma = luminance(itemHue(item, skin));
  if (!(luma > 0)) throw new Error(`orchard hue ${item} has no luminance`);
  return target / luma;
}

function itemHue(item: number, skin: OrchardSkin): THREE.Color {
  if (item === JBW_BANANA) return skin.banana;
  if (item === JBW_ONION) return skin.onion;
  if (item === JBW_JELLY) return skin.jelly;
  if (item === JBW_TRUFFLE) return skin.truffle;
  if (item === JBW_TREE) return skin.canopyLow;
  return skin.stone;
}

function paperHue(item: number, palette: Palette, k: number): THREE.Color {
  const c = JBW_COLOR[item];
  if (!c) throw new Error(`no paper colour for item ${item}`);
  const base = new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
  return blendTokens(base, palette.sun, k);
}

export function orchardSkin(palette: Palette, accent: THREE.ColorRepresentation): OrchardSkin {
  const accents = TOKENS.accents;
  const leaf = accents['four-rooms'] ?? TOKENS.celebrationGold;
  const ember = accents['high-dim-maze'] ?? TOKENS.celebrationGold;
  return {
    grassLow: blendTokens(blendTokens(palette.groundA, leaf, 0.72), palette.hemiGround, 0.2),
    grassHigh: blendTokens(blendTokens(palette.groundB, leaf, 0.56), palette.sun, 0.14),
    soil: shade(blendTokens(blendTokens(palette.groundA, leaf, 0.55), ember, 0.16), 0.12),
    banana: paperHue(JBW_BANANA, palette, 0.14),
    onion: paperHue(JBW_ONION, palette, 0.14),
    jelly: paperHue(JBW_JELLY, palette, 0.12),
    truffle: paperHue(JBW_TRUFFLE, palette, 0.16),
    stone: paperHue(JBW_WALL, palette, 0.3),
    moss: blendTokens(leaf, palette.hemiGround, 0.55),
    bark: shade(blendTokens(ember, palette.hemiGround, 0.6), 0.3),
    canopyLow: paperHue(JBW_TREE, palette, 0.22),
    canopyHigh: paperHue(JBW_TREE, palette, 0.3),
    papery: blendTokens(palette.skyBottom, palette.sun, 0.35),
    bloom: blendTokens(palette.sun, TOKENS.celebrationGold, 0.4),
    scentA: paperHue(JBW_JELLY, palette, 0.34),
    scentB: paperHue(JBW_BANANA, palette, 0.3),
    scentC: paperHue(JBW_ONION, palette, 0.24),
    shaft: blendTokens(palette.sun, accent, 0.18),
    ripe: blendTokens(TOKEN_ONE, TOKENS.celebrationGold, 0.22),
    spoiled: blendTokens(TOKEN_ONE, palette.hemiSky, 0.62),
    front: blendTokens(palette.hemiSky, palette.skyTop, 0.4),
    mapGround: shade(blendTokens(palette.groundA, leaf, 0.5), 0.42),
    legendary: blendTokens(TOKENS.celebrationGold, ember, 0.42),
  };
}

const TOKEN_ONE = 0xffffff;

export interface ScentPaint {
  uScentMap: { value: THREE.Texture | null };
  uScentOrigin: { value: THREE.Vector2 };
  uScentSize: { value: number };
  uScentA: { value: THREE.Color };
  uScentB: { value: THREE.Color };
  uScentC: { value: THREE.Color };
  uScentGain: { value: number };
  uShaftColor: { value: THREE.Color };
  uShaftGain: { value: number };
  uScentGlow: { value: number };
  uFrontEdge: { value: number };
  uFrontWidth: { value: number };
  uFrontColor: { value: THREE.Color };
  uFrontMix: { value: number };
  uAgent: { value: THREE.Vector2 };
  uMapBlend: { value: number };
  uMapGround: { value: THREE.Color };
  uGridGain: { value: number };
  uGridColor: { value: THREE.Color };
}

const PAINT_DECL = `
uniform sampler2D uScentMap;
uniform vec2 uScentOrigin;
uniform float uScentSize;
uniform vec3 uScentA;
uniform vec3 uScentB;
uniform vec3 uScentC;
uniform float uScentGain;
uniform vec3 uShaftColor;
uniform float uShaftGain;
uniform float uScentGlow;
uniform float uFrontEdge;
uniform float uFrontWidth;
uniform vec3 uFrontColor;
uniform float uFrontMix;
uniform vec2 uAgent;
uniform float uMapBlend;
uniform vec3 uMapGround;
uniform float uGridGain;
uniform vec3 uGridColor;
varying vec3 vOrchard;

float orchardLattice( const in vec2 p ) {
	vec2 w = fwidth( p );
	vec2 edge = abs( fract( p ) - 0.5 ) / max( w, vec2( 1e-4 ) );
	vec2 line = vec2( 1.0 ) - smoothstep( vec2( 0.4 ), vec2( 1.4 ), edge );
	vec2 att = vec2( 1.0 ) / ( vec2( 1.0 ) + 4.0 * w );
	return max( line.x * att.x, line.y * att.y );
}

float orchardVisibility( const in vec2 p ) {
	vec2 d = abs( p - uAgent );
	vec2 w = fwidth( p );
	vec2 inside = vec2( 1.0 ) - smoothstep( vec2( 1.5 ) - w, vec2( 1.5 ) + w, d );
	return inside.x * inside.y;
}
`;

const PAINT_TINT = `
vec2 orchardUv = ( vOrchard.xz - uScentOrigin ) / uScentSize;
vec4 orchardScent = texture2D( uScentMap, orchardUv );
vec2 orchardEdge = abs( orchardUv - 0.5 ) * 2.0;
float orchardInField = ( 1.0 - smoothstep( 0.84, 1.0, orchardEdge.x ) ) * ( 1.0 - smoothstep( 0.84, 1.0, orchardEdge.y ) );
vec3 orchardLift = orchardScent.rgb * orchardInField;
float orchardSum = orchardLift.r + orchardLift.g + orchardLift.b;
vec3 orchardTint = ( uScentA * orchardLift.r + uScentB * orchardLift.g + uScentC * orchardLift.b ) / max( orchardSum, 0.001 );
float orchardMass = 1.0 - ( 1.0 - orchardLift.r ) * ( 1.0 - orchardLift.g ) * ( 1.0 - orchardLift.b );
diffuseColor.rgb = mix( diffuseColor.rgb, orchardTint, clamp( orchardMass * uScentGain, 0.0, 1.0 ) );
vec3 orchardFlat = mix( uMapGround, orchardTint, clamp( orchardMass * 1.35, 0.0, 1.0 ) );
float orchardFront = uFrontMix * ( 1.0 - smoothstep( 0.0, uFrontWidth, abs( vOrchard.x + vOrchard.z - uFrontEdge ) ) );
diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * uFrontColor, orchardFront );
`;

const PAINT_SHAFT = `
totalEmissiveRadiance += uShaftColor * orchardVisibility( vOrchard.xz ) * uShaftGain;
totalEmissiveRadiance += orchardTint * orchardMass * uScentGlow;
`;

export interface ItemMotion {
  hover: number;
  bob: number;
  spin: number;
}

const FLOAT_DECL = `
uniform float uItemZoom;
uniform float uItemTime;
uniform float uItemHover;
uniform float uItemBob;
uniform float uItemSpin;
`;

const FLOAT_APPLY = `
transformed *= uItemZoom;
{
	vec3 orchardOrigin = instanceMatrix[ 3 ].xyz;
	float orchardPhase = orchardOrigin.x * 0.7 + orchardOrigin.z * 1.3;
	float orchardAngle = uItemTime * uItemSpin + orchardPhase;
	float orchardCos = cos( orchardAngle );
	float orchardSin = sin( orchardAngle );
	transformed.xz = vec2(
		orchardCos * transformed.x - orchardSin * transformed.z,
		orchardSin * transformed.x + orchardCos * transformed.z
	);
	transformed.y += uItemZoom * ( uItemHover + sin( uItemTime * 2.1 + orchardPhase ) * uItemBob );
}
`;

export function animateItems(
  material: OakMaterial,
  zoom: { value: number },
  time: { value: number },
  motion: ItemMotion,
): void {
  const base = material.onBeforeCompile;
  const baseKey = material.customProgramCacheKey;
  material.customProgramCacheKey = () => `${baseKey ? baseKey.call(material) : ''}|float`;
  const hover = { value: motion.hover };
  const bob = { value: motion.bob };
  const spin = { value: motion.spin };
  material.onBeforeCompile = (shader, renderer) => {
    base.call(material, shader, renderer);
    shader.uniforms.uItemZoom = zoom;
    shader.uniforms.uItemTime = time;
    shader.uniforms.uItemHover = hover;
    shader.uniforms.uItemBob = bob;
    shader.uniforms.uItemSpin = spin;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${FLOAT_DECL}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${FLOAT_APPLY}`);
  };
}

export function paintScent(material: OakMaterial, paint: ScentPaint): void {
  const base = material.onBeforeCompile;
  const baseKey = material.customProgramCacheKey;
  material.customProgramCacheKey = () => `${baseKey ? baseKey.call(material) : ''}|orchard`;
  material.onBeforeCompile = (shader, renderer) => {
    base.call(material, shader, renderer);
    shader.uniforms.uScentMap = paint.uScentMap;
    shader.uniforms.uScentOrigin = paint.uScentOrigin;
    shader.uniforms.uScentSize = paint.uScentSize;
    shader.uniforms.uScentA = paint.uScentA;
    shader.uniforms.uScentB = paint.uScentB;
    shader.uniforms.uScentC = paint.uScentC;
    shader.uniforms.uScentGain = paint.uScentGain;
    shader.uniforms.uShaftColor = paint.uShaftColor;
    shader.uniforms.uShaftGain = paint.uShaftGain;
    shader.uniforms.uScentGlow = paint.uScentGlow;
    shader.uniforms.uFrontEdge = paint.uFrontEdge;
    shader.uniforms.uFrontWidth = paint.uFrontWidth;
    shader.uniforms.uFrontColor = paint.uFrontColor;
    shader.uniforms.uFrontMix = paint.uFrontMix;
    shader.uniforms.uAgent = paint.uAgent;
    shader.uniforms.uMapBlend = paint.uMapBlend;
    shader.uniforms.uMapGround = paint.uMapGround;
    shader.uniforms.uGridGain = paint.uGridGain;
    shader.uniforms.uGridColor = paint.uGridColor;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vOrchard;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvOrchard = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${PAINT_DECL}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${PAINT_TINT}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${PAINT_SHAFT}`)
      .replace(
        '#include <opaque_fragment>',
        'outgoingLight = mix( outgoingLight, orchardFlat, uMapBlend );\noutgoingLight += uGridColor * orchardLattice( vOrchard.xz ) * uGridGain;\n#include <opaque_fragment>',
      );
  };
}

export function orchardHeight(seed: number, x: number, z: number): number {
  const a = valueNoise2(x / 32, z / 32, GROUND_PERIOD / 32, seed);
  const b = valueNoise2(x / 16, z / 16, GROUND_PERIOD / 16, seed ^ 0x51ed);
  return (a - 0.5) * RELIEF * 2 + (b - 0.5) * RELIEF * 0.44;
}

export interface OrchardWorld extends WorldCore {
  rig: LightRig;
  sky: SkyHandle;
  terrain: TerrainHandle;
  paint: ScentPaint;
  items: Map<number, THREE.InstancedMesh>;
  decals: THREE.InstancedMesh;
  shells: Map<number, THREE.InstancedMesh>;
  cellColor: Float32Array;
  beacon: BeaconHandle;
  setOrigin(x: number, z: number): void;
  setHour(hour: number): void;
  setRipeness(jellyRipe: number): void;
  decalTint(item: number, out: THREE.Color): THREE.Color;
  setFront(edge: number, mix: number): void;
  setScentOrigin(x: number, y: number): void;
  setSight(x: number, y: number): void;
  setBreeze(amount: number): void;
  heightAt(x: number, z: number): number;
  setMap(blend: number, distance: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, field: ScentField): OrchardWorld {
  const w = makeWorld('windfallOrchard');
  const palette = TOKENS.palettes.dusk;
  const pale = TOKENS.palettes.afternoon;
  const accent = TOKENS.accents['jelly-bean-world'] ?? TOKENS.celebrationGold;
  const skin = orchardSkin(palette, accent);

  const sky = w.time(skyFromPalette(palette, 'dusk', { radius: 1800, sunSize: 0.042 }));
  w.add(sky.mesh);
  w.own(sky);

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 2,
    pointDistance: 14,
    sunDistanceScale: 3.2,
    shadowMapSize: [SHADOW_MAP_X, SHADOW_MAP_Y],
  });
  rig.sun.shadow.normalBias = 0.4;
  rig.sun.shadow.radius = 2.4;
  w.add(rig.group);
  w.own(rig);

  const fog = applyFog(scene, palette, FOG_EXTENT);

  const paint: ScentPaint = {
    uScentMap: { value: field.texture },
    uScentOrigin: { value: new THREE.Vector2(0, 0) },
    uScentSize: { value: FIELD_SIZE },
    uScentA: { value: skin.scentA.clone() },
    uScentB: { value: skin.scentB.clone() },
    uScentC: { value: skin.scentC.clone() },
    uScentGain: { value: 0.55 },
    uShaftColor: { value: skin.shaft.clone() },
    uShaftGain: { value: 0.2 },
    uScentGlow: { value: 0.07 },
    uFrontEdge: { value: -1e6 },
    uFrontWidth: { value: FRONT_WIDTH },
    uFrontColor: { value: skin.front.clone() },
    uFrontMix: { value: 0 },
    uAgent: { value: new THREE.Vector2(0, 0) },
    uMapBlend: { value: 0 },
    uMapGround: { value: skin.mapGround.clone() },
    uGridGain: { value: 0 },
    uGridColor: { value: new THREE.Color(TOKENS.uiCyan) },
  };

  const terrain = makeTerrain({
    size: [GROUND_EXTENT, GROUND_EXTENT],
    segments: [GROUND_SEGMENTS, GROUND_SEGMENTS],
    seed: ctx.seed,
    heightFn: (x, z) => orchardHeight(ctx.seed, x, z),
    colorRamp: [
      { t: 0, color: skin.soil },
      { t: 0.42, color: skin.grassLow },
      { t: 1, color: skin.grassHigh },
    ],
    colorNoise: 0.02,
    slopeColor: skin.soil,
    slopeStrength: 0.14,
    slopeSharpness: 2.1,
    receiveShadow: true,
    name: 'orchardGround',
    materialOptions: {
      flatShading: false,
      roughness: 0.94,
      rimColor: palette.skyHorizon,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.42,
      toonFloor: 0.24,
    },
  });
  w.add(terrain.mesh);
  w.own(terrain);
  paintScent(terrain.material as OakMaterial, paint);

  const itemZoom = { value: 1 };
  const itemTime = { value: 0 };
  let groundX = 0;
  let groundZ = 0;
  const surfaceAt = (x: number, z: number): number => terrain.sampleHeight(x - groundX, z - groundZ);
  let hourNow = 0;
  const items = new Map<number, THREE.InstancedMesh>();
  for (const kind of ITEM_KINDS) {
    const geometry = orchardItemGeometry(kind, skin);
    w.own({ dispose: () => geometry.dispose() });
    const material = oakMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.9,
      rimColor: palette.skyHorizon,
      rimStrength: 0.34,
      toonSteps: 3,
      toonSoftness: 0.4,
      toonFloor: 0.3,
      emissive: itemHue(kind, skin),
      emissiveIntensity: itemGlow(kind, skin, ITEM_GLOW),
      name: `orchardItem${kind}`,
    });
    w.own(material);
    animateItems(material, itemZoom, itemTime, ITEM_MOTION[kind] ?? ZERO_MOTION);
    const mesh = new THREE.InstancedMesh(geometry, material, ITEM_CAPACITY[kind] ?? 128);
    mesh.name = `orchardItems${kind}`;
    mesh.castShadow = (ITEM_MOTION[kind]?.hover ?? 0) === 0;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.count = 0;
    w.add(mesh);
    w.own({ dispose: () => mesh.dispose() });
    items.set(kind, mesh);
  }

  const decalGeometry = new THREE.PlaneGeometry(1, 1);
  decalGeometry.rotateX(-Math.PI / 2);
  w.own({ dispose: () => decalGeometry.dispose() });
  const decalMaterial = emissiveMaterial(TOKEN_ONE, 0.9, {
    map: w.own(radialSprite(64, 2.2)),
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'orchardDecal',
  });
  w.own(decalMaterial);
  const decals = new THREE.InstancedMesh(decalGeometry, decalMaterial, DECAL_CAPACITY);
  decals.name = 'orchardDecals';
  decals.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(DECAL_CAPACITY * 3), 3);
  decals.castShadow = false;
  decals.receiveShadow = false;
  decals.frustumCulled = false;
  decals.renderOrder = 2;
  decals.count = 0;
  w.add(decals);
  w.own({ dispose: () => decals.dispose() });

  const shells = new Map<number, THREE.InstancedMesh>();
  for (const kind of ITEM_KINDS) {
    const motion = ITEM_MOTION[kind] ?? ZERO_MOTION;
    if (motion.hover <= 0) continue;
    const source = items.get(kind);
    if (!source) continue;
    const shellGeometry = source.geometry.clone();
    shellGeometry.scale(SHELL_SCALE, SHELL_SCALE, SHELL_SCALE);
    w.own({ dispose: () => shellGeometry.dispose() });
    const tint = blendTokens(itemHue(kind, skin), TOKEN_ONE, HALO_WHITEN);
    const shellMaterial = emissiveMaterial(tint, HALO_GLOW / luminance(tint), {
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      name: `orchardShell${kind}`,
    });
    w.own(shellMaterial);
    animateItems(shellMaterial as unknown as OakMaterial, itemZoom, itemTime, motion);
    const shell = new THREE.InstancedMesh(shellGeometry, shellMaterial, ITEM_CAPACITY[kind] ?? 128);
    shell.name = `orchardShells${kind}`;
    shell.castShadow = false;
    shell.receiveShadow = false;
    shell.frustumCulled = false;
    shell.renderOrder = 4;
    shell.count = 0;
    w.add(shell);
    w.own({ dispose: () => shell.dispose() });
    shells.set(kind, shell);
  }

  const cellColor = new Float32Array((Math.max(...ITEM_KINDS) + 1) * 3);
  for (const kind of ITEM_KINDS) {
    const hue = itemHue(kind, skin);
    cellColor[kind * 3] = hue.r;
    cellColor[kind * 3 + 1] = hue.g;
    cellColor[kind * 3 + 2] = hue.b;
  }

  const beacon = w.time(buildBeacon(skin.legendary, BEACON_CAPACITY));
  w.add(beacon.mesh);
  w.add(beacon.pool);
  w.own(beacon);

  const grassGeometry = grassClumpGeometry(4, 0.4, 0.05);
  w.own({ dispose: () => grassGeometry.dispose() });
  const grassMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: palette.skyHorizon,
    rimStrength: 0.2,
    toonSteps: 3,
    toonSoftness: 0.5,
    toonFloor: 0.5,
    wind: { amplitude: 0.02, frequency: 1.05, height: 0.5 },
    name: 'orchardGrass',
  });
  w.own(grassMaterial);
  const grassBank = makeTileBank({
    geometry: grassGeometry,
    material: grassMaterial,
    tilesX: SCATTER_SPAN,
    tilesZ: SCATTER_SPAN,
    backX: SCATTER_BACK,
    backZ: SCATTER_BACK,
    perTile: 96,
    colorA: skin.grassLow,
    colorB: skin.grassHigh,
    seed: ctx.seed,
    name: 'orchardGrassBank',
    fill: (tx, tz, rng, write) => {
      const x0 = tx * SCATTER_TILE;
      const z0 = tz * SCATTER_TILE;
      for (let i = 0; i < 96; i++) {
        const x = x0 + rng() * SCATTER_TILE;
        const z = z0 + rng() * SCATTER_TILE;
        write(
          x,
          surfaceAt(x, z),
          z,
          rng() * Math.PI,
          0.52 + rng() * 0.66,
          0.72 + rng() * 0.7,
          rng(),
        );
      }
    },
  });
  w.add(grassBank.mesh);
  w.own(grassBank);

  const flowerGeo = flowerGeometry(5);
  w.own({ dispose: () => flowerGeo.dispose() });
  const flowerMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: palette.skyHorizon,
    rimStrength: 0.26,
    toonSteps: 3,
    toonSoftness: 0.5,
    toonFloor: 0.46,
    wind: { amplitude: 0.024, frequency: 1.3, height: 0.4 },
    name: 'orchardFlowers',
  });
  w.own(flowerMaterial);
  const flowerBank = makeTileBank({
    geometry: flowerGeo,
    material: flowerMaterial,
    tilesX: SCATTER_SPAN,
    tilesZ: SCATTER_SPAN,
    backX: SCATTER_BACK,
    backZ: SCATTER_BACK,
    perTile: 7,
    colorA: skin.bloom,
    colorB: skin.papery,
    seed: ctx.seed ^ 0x2c91,
    name: 'orchardFlowerBank',
    fill: (tx, tz, rng, write) => {
      const x0 = tx * SCATTER_TILE;
      const z0 = tz * SCATTER_TILE;
      for (let i = 0; i < 7; i++) {
        if (rng() > 0.62) continue;
        const x = x0 + rng() * SCATTER_TILE;
        const z = z0 + rng() * SCATTER_TILE;
        write(
          x,
          surfaceAt(x, z),
          z,
          rng() * Math.PI * 2,
          0.42 + rng() * 0.4,
          0.9 + rng() * 0.4,
          rng(),
        );
      }
    },
  });
  w.add(flowerBank.mesh);
  w.own(flowerBank);

  const pollen: AmbientParticlesHandle = w.time(
    makeAmbientParticles({
      preset: 'pollen',
      seed: ctx.seed ^ 0x77ab,
      bounds: { halfX: 120, halfY: 16, halfZ: 120, y: 15 },
      color: skin.bloom,
    }),
  );
  w.add(pollen.points);
  w.own(pollen);

  const mist: MistPlanesHandle = w.time(
    makeMistPlanes({
      size: [GROUND_EXTENT * 0.8, 74],
      seed: ctx.seed ^ 0x4d31,
      layers: 4,
      baseY: 3.4,
      spread: 640,
      color: blendTokens(palette.fog, palette.sun, 0.24),
      opacity: 0.03,
    }),
  );
  w.add(mist.group);
  w.own(mist);

  const sunDir = new THREE.Vector3();
  const baseAzimuth = SUN_AZIMUTH;
  const fogBase = fog.density;
  const grassWind = grassMaterial.oak;
  const flowerWind = flowerMaterial.oak;
  const lightProbe = new THREE.Object3D();
  const probePoint = new THREE.Vector3();
  const shadowCentre = new THREE.Vector3();

  const fitShadow = (cx: number, cz: number): void => {
    const elevation = Math.max(0.02, Math.min(1, rig.sunDirection.y));
    const horizon = Math.sqrt(Math.max(1e-6, 1 - elevation * elevation));
    const halfSlab = (SLAB_HIGH - SLAB_LOW) / 2;
    const halfX = SHADOW_SPAN;
    const halfY = SHADOW_SPAN * elevation + halfSlab * horizon;
    const depth = SHADOW_SPAN * horizon + halfSlab * elevation;
    shadowCentre.set(cx, (SLAB_LOW + SLAB_HIGH) / 2, cz);
    lightProbe.position.copy(shadowCentre).addScaledVector(rig.sunDirection, SHADOW_REACH);
    lightProbe.lookAt(shadowCentre);
    lightProbe.updateMatrixWorld();
    probePoint.copy(shadowCentre).applyMatrix4(lightProbe.matrixWorld.clone().invert());
    const qx = (2 * halfX) / SHADOW_MAP_X;
    const qy = (2 * halfY) / SHADOW_MAP_Y;
    probePoint.x = Math.round(probePoint.x / qx) * qx;
    probePoint.y = Math.round(probePoint.y / qy) * qy;
    probePoint.applyMatrix4(lightProbe.matrixWorld);
    rig.setShadowExtent(probePoint, halfX, halfY, SHADOW_REACH, depth + halfSlab);
  };

  const setHour = (hour: number): void => {
    const t = 0.7 + 0.3 * clamp01(hour);
    hourNow = t;
    sky.setColors(
      blendTokens(pale.skyTop, palette.skyTop, t),
      blendTokens(pale.skyHorizon, palette.skyHorizon, t),
      blendTokens(pale.skyBottom, palette.skyBottom, t),
    );
    rig.sun.color.copy(blendTokens(pale.sun, palette.sun, t));
    rig.sun.intensity = pale.sunIntensity + (palette.sunIntensity - pale.sunIntensity) * t;
    rig.hemi.color.copy(blendTokens(pale.hemiSky, palette.hemiSky, t));
    rig.hemi.groundColor.copy(blendTokens(pale.hemiGround, palette.hemiGround, t));
    rig.hemi.intensity = pale.hemiIntensity + (palette.hemiIntensity - pale.hemiIntensity) * t;
    fog.color.copy(blendTokens(pale.fog, palette.fog, t));
    fog.density = fogBase * (1 - 0.08 * t);
    const elevation = 0.36 - 0.1 * t;
    const radial = Math.sqrt(Math.max(1e-4, 1 - elevation * elevation));
    sunDir.set(Math.sin(baseAzimuth) * radial, elevation, Math.cos(baseAzimuth) * radial);
    sky.setSunDirection(sunDir.x, sunDir.y, sunDir.z);
    rig.setSunDirection(sunDir.x, sunDir.y, sunDir.z);
    paint.uScentGain.value = 0.42 + 0.28 * t;
    paint.uShaftGain.value = 0.12 + 0.2 * t;
    paint.uScentGlow.value = 0.04 + 0.09 * t;
    for (let i = 0; i < rig.points.length; i++) {
      const light = rig.points[i];
      if (light) light.intensity = 0.08 + 0.9 * t;
    }
    pollen.setDensity(0.24 + 0.76 * t);
  };

  const ripeColor = new THREE.Color();
  const ripeness = new Float32Array(cellColor.length / 3).fill(1);
  const setRipeness = (jellyRipe: number): void => {
    const k = clamp01(jellyRipe);
    ripeness[JBW_JELLY] = k;
    ripeness[JBW_ONION] = 1 - k;
  };

  setHour(0.4);
  setRipeness(1);

  return w.seal<OrchardWorld>(
    {
      rig,
      sky,
      terrain,
      paint,
      items,
      decals,
      shells,
      cellColor,
      beacon,
      setOrigin(x, z) {
        const sx = Math.round(x / GROUND_PERIOD) * GROUND_PERIOD;
        const sz = Math.round(z / GROUND_PERIOD) * GROUND_PERIOD;
        groundX = sx;
        groundZ = sz;
        terrain.mesh.position.set(sx, 0, sz);
        grassBank.setOrigin(Math.floor(x / SCATTER_TILE), Math.floor(z / SCATTER_TILE));
        flowerBank.setOrigin(Math.floor(x / SCATTER_TILE), Math.floor(z / SCATTER_TILE));
        pollen.points.position.set(x, 0, z);
        mist.group.position.set(x, 0, z);
        sky.mesh.position.set(x, 0, z);
        fitShadow(x, z);
      },
      setHour,
      setRipeness,
      decalTint(item, out) {
        const t = item * 3;
        const ripe = ripeness[item] ?? 1;
        ripeColor.copy(skin.spoiled).lerp(skin.ripe, ripe);
        return out
          .setRGB(cellColor[t] ?? 1, cellColor[t + 1] ?? 1, cellColor[t + 2] ?? 1)
          .multiply(ripeColor)
          .multiplyScalar(DECAL_DIM + (1 - DECAL_DIM) * ripe);
      },
      setFront(edge, mix) {
        paint.uFrontEdge.value = edge;
        paint.uFrontMix.value = clamp01(mix);
      },
      setScentOrigin(x, y) {
        paint.uScentOrigin.value.set(x, y);
      },
      setSight(x, y) {
        paint.uAgent.value.set(x, y);
      },
      setMap(blend, distance) {
        const k = clamp01(blend);
        paint.uMapBlend.value = k;
        fog.density = fogBase * (1 - 0.08 * hourNow) * (1 - k);
        itemZoom.value = 1 + (Math.max(distance, MAP_START) / MAP_START - 1) * k * MAP_MARK;
        paint.uScentGain.value = (0.42 + 0.28 * hourNow) * (1 - 0.72 * k);
        paint.uScentGlow.value = (0.04 + 0.09 * hourNow) * (1 - 0.85 * k);
      },
      heightAt: surfaceAt,
      setBreeze(amount) {
        grassWind.setWind(0.016 + 0.028 * clamp01(amount));
        flowerWind.setWind(0.02 + 0.03 * clamp01(amount));
      },
    },
    {
      time(t) {
        itemTime.value = t;
        const drift = hash2f(1, 2, ctx.seed) * 0.4;
        pollen.points.rotation.y = Math.sin(t * 0.02 + drift) * 0.05;
      },
    },
  );
}
