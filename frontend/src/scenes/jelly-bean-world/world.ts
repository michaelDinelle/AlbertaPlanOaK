import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01, shade } from '@kit/mix';
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
import { makeWorld, type WorldCore } from '@kit/world';
import { ITEM_CAPACITY, ITEM_KINDS, orchardItemGeometry } from './actors';
import {
  JBW_BANANA,
  JBW_COLOR,
  JBW_FIELD_OF_VIEW,
  JBW_JELLY,
  JBW_ONION,
  JBW_TREE,
  JBW_TRUFFLE,
  JBW_VISUAL_RANGE,
  JBW_WALL,
} from '@sim/envs/jellybeanworld';
import { FIELD_SIZE, type ScentField } from './scent';

export const DRAW_RADIUS = 520;
export const GROUND_PERIOD = 256;
export const GROUND_EXTENT = 1400;
export const GROUND_SEGMENTS = 512;
export const FOG_EXTENT = 1320;
export const RELIEF = 6.2;
export const SCATTER_TILE = 24;
export const SCATTER_SPAN = 33;
export const SCATTER_BACK = 16;
export const SHADOW_REACH = 300;
export const SHADOW_SPAN = 260;
const SLAB_LOW = -10;
const SLAB_HIGH = 15;
const SHADOW_MAP = 2048;
const MAP_MARK = 0.42;

export const CAM_DISTANCE = 78;
export const CAM_MAX_DISTANCE = 620;
export const MAP_START = 190;
export const MAP_FULL = 540;
export const CAM_NEAR = 0.6;
export const CAM_FAR = 3600;
export const CAM_AZIMUTH = 0.68;
export const SUN_AZIMUTH = CAM_AZIMUTH + Math.PI;
export const CAM_POLAR = 1.2;
export const CAM_FOLLOW = 2.6;

const FRONT_WIDTH = 44;
export const OCCLUDER_LIMIT = 64;

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
  uFacing: { value: number };
  uHalfFov: { value: number };
  uViewRange: { value: number };
  uOccluderCount: { value: number };
  uOccluders: { value: Float32Array };
  uMapBlend: { value: number };
  uMapGround: { value: THREE.Color };
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
uniform float uFacing;
uniform float uHalfFov;
uniform float uViewRange;
uniform int uOccluderCount;
uniform vec3 uOccluders[ 64 ];
uniform float uMapBlend;
uniform vec3 uMapGround;
varying vec3 vOrchard;

float orchardWrap( const in float a ) {
	return a - 6.2831853 * floor( ( a + 3.1415927 ) / 6.2831853 );
}

float orchardVisibility( const in vec2 p ) {
	vec2 d = p - uAgent;
	float dist = length( d );
	if ( dist >= uViewRange ) return 0.0;
	float beta = dist <= 0.5 ? 1.5707963 : asin( 0.5 / dist );
	float span = 2.0 * beta;
	float delta = orchardWrap( atan( d.y, d.x ) - uFacing );
	float seen = min( delta + beta, uHalfFov ) - max( delta - beta, -uHalfFov );
	if ( seen <= 0.0 ) return 0.0;
	float blocked = 0.0;
	for ( int i = 0; i < 64; i++ ) {
		if ( i >= uOccluderCount ) break;
		vec3 o = uOccluders[ i ];
		vec2 od = o.xy - uAgent;
		float odist = length( od );
		if ( odist >= dist ) continue;
		float ob = odist <= 0.5 ? 1.5707963 : asin( 0.5 / odist );
		float oa = orchardWrap( atan( od.y, od.x ) - uFacing );
		float overlap = min( delta + beta, oa + ob ) - max( delta - beta, oa - ob );
		if ( overlap <= 0.0 ) continue;
		blocked += o.z * ( overlap / span );
	}
	float edge = 1.0 - smoothstep( uViewRange * 0.62, uViewRange, dist );
	return ( seen / span ) * max( 0.0, 1.0 - blocked ) * edge;
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

export function scaleItems(material: OakMaterial, zoom: { value: number }): void {
  const base = material.onBeforeCompile;
  const baseKey = material.customProgramCacheKey;
  material.customProgramCacheKey = () => `${baseKey ? baseKey.call(material) : ''}|zoom`;
  material.onBeforeCompile = (shader, renderer) => {
    base.call(material, shader, renderer);
    shader.uniforms.uItemZoom = zoom;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uItemZoom;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\ntransformed *= uItemZoom;');
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
    shader.uniforms.uFacing = paint.uFacing;
    shader.uniforms.uHalfFov = paint.uHalfFov;
    shader.uniforms.uViewRange = paint.uViewRange;
    shader.uniforms.uOccluderCount = paint.uOccluderCount;
    shader.uniforms.uOccluders = paint.uOccluders;
    shader.uniforms.uMapBlend = paint.uMapBlend;
    shader.uniforms.uMapGround = paint.uMapGround;
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
        'outgoingLight = mix( outgoingLight, orchardFlat, uMapBlend );\n#include <opaque_fragment>',
      );
  };
}

export function orchardHeight(seed: number, x: number, z: number): number {
  const a = valueNoise2(x / 32, z / 32, GROUND_PERIOD / 32, seed);
  const b = valueNoise2(x / 8, z / 8, GROUND_PERIOD / 8, seed ^ 0x51ed);
  const c = valueNoise2(x / 3, z / 3, GROUND_PERIOD / 3, seed ^ 0x2b17);
  return (a - 0.5) * RELIEF * 2 + (b - 0.5) * RELIEF * 0.42 + (c - 0.5) * RELIEF * 0.06;
}

export interface OrchardWorld extends WorldCore {
  rig: LightRig;
  sky: SkyHandle;
  terrain: TerrainHandle;
  paint: ScentPaint;
  items: Map<number, THREE.InstancedMesh>;
  itemMaterials: Map<number, OakMaterial>;
  setOrigin(x: number, z: number): void;
  setHour(hour: number): void;
  setRipeness(jellyRipe: number): void;
  setFront(edge: number, mix: number): void;
  setScentOrigin(x: number, y: number): void;
  setSight(x: number, y: number, facing: number, count: number): void;
  setBreeze(amount: number): void;
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
  });
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
    uFacing: { value: 0 },
    uHalfFov: { value: JBW_FIELD_OF_VIEW / 2 },
    uViewRange: { value: JBW_VISUAL_RANGE },
    uOccluderCount: { value: 0 },
    uOccluders: { value: new Float32Array(OCCLUDER_LIMIT * 3) },
    uMapBlend: { value: 0 },
    uMapGround: { value: skin.mapGround.clone() },
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
  let hourNow = 0;
  const items = new Map<number, THREE.InstancedMesh>();
  const itemMaterials = new Map<number, OakMaterial>();
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
      name: `orchardItem${kind}`,
    });
    w.own(material);
    scaleItems(material, itemZoom);
    const mesh = new THREE.InstancedMesh(geometry, material, ITEM_CAPACITY[kind] ?? 128);
    mesh.name = `orchardItems${kind}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.count = 0;
    w.add(mesh);
    w.own({ dispose: () => mesh.dispose() });
    items.set(kind, mesh);
    itemMaterials.set(kind, material);
  }

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
          orchardHeight(ctx.seed, x, z),
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
          orchardHeight(ctx.seed, x, z),
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
  const lightProbe = new THREE.Object3D();
  const probePoint = new THREE.Vector3();
  const shadowCentre = new THREE.Vector3();

  const fitShadow = (cx: number, cz: number): void => {
    shadowCentre.set(cx, (SLAB_LOW + SLAB_HIGH) / 2, cz);
    lightProbe.position.copy(shadowCentre).addScaledVector(rig.sunDirection, SHADOW_REACH);
    lightProbe.lookAt(shadowCentre);
    lightProbe.updateMatrixWorld();
    lightProbe.matrixWorld.invert();
    let halfX = 1;
    let halfY = 1;
    let depth = 1;
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const px = cx + Math.cos(a) * SHADOW_SPAN;
      const pz = cz + Math.sin(a) * SHADOW_SPAN;
      for (let h = 0; h < 2; h++) {
        probePoint.set(px, h === 0 ? SLAB_LOW : SLAB_HIGH, pz);
        probePoint.applyMatrix4(lightProbe.matrixWorld);
        halfX = Math.max(halfX, Math.abs(probePoint.x));
        halfY = Math.max(halfY, Math.abs(probePoint.y));
        depth = Math.max(depth, Math.abs(probePoint.z + SHADOW_REACH));
      }
    }
    const qx = (2 * halfX) / SHADOW_MAP;
    const qy = (2 * halfY) / SHADOW_MAP;
    rig.setShadowExtent(
      {
        x: Math.round(shadowCentre.x / qx) * qx,
        y: shadowCentre.y,
        z: Math.round(shadowCentre.z / qy) * qy,
      },
      halfX,
      halfY,
      SHADOW_REACH,
      depth + 4,
    );
  };
  const baseAzimuth = SUN_AZIMUTH;
  const grassWind = grassMaterial.oak;
  const flowerWind = flowerMaterial.oak;
  const fogBase = fog.density;

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
  const setRipeness = (jellyRipe: number): void => {
    const k = clamp01(jellyRipe);
    const jellyMaterial = itemMaterials.get(3);
    const onionMaterial = itemMaterials.get(2);
    if (jellyMaterial) {
      ripeColor.copy(skin.spoiled).lerp(skin.ripe, k);
      jellyMaterial.color.copy(ripeColor);
      jellyMaterial.emissive.copy(skin.jelly).multiplyScalar(0.16 * k);
    }
    if (onionMaterial) {
      ripeColor.copy(skin.ripe).lerp(skin.spoiled, k);
      onionMaterial.color.copy(ripeColor);
      onionMaterial.emissive.copy(skin.onion).multiplyScalar(0.16 * (1 - k));
    }
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
      itemMaterials,
      setOrigin(x, z) {
        const sx = Math.round(x / GROUND_PERIOD) * GROUND_PERIOD;
        const sz = Math.round(z / GROUND_PERIOD) * GROUND_PERIOD;
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
      setFront(edge, mix) {
        paint.uFrontEdge.value = edge;
        paint.uFrontMix.value = clamp01(mix);
      },
      setScentOrigin(x, y) {
        paint.uScentOrigin.value.set(x, y);
      },
      setSight(x, y, facing, count) {
        paint.uAgent.value.set(x, y);
        paint.uFacing.value = facing;
        paint.uOccluderCount.value = count;
      },
      setMap(blend, distance) {
        const k = clamp01(blend);
        paint.uMapBlend.value = k;
        fog.density = fogBase * (1 - 0.08 * hourNow) * (1 - k);
        sky.mesh.visible = k < 0.995;
        itemZoom.value = 1 + (Math.max(distance, MAP_START) / MAP_START - 1) * k * MAP_MARK;
      },
      setBreeze(amount) {
        grassWind.setWind(0.016 + 0.028 * clamp01(amount));
        flowerWind.setWind(0.02 + 0.03 * clamp01(amount));
      },
    },
    {
      time(t) {
        const drift = hash2f(1, 2, ctx.seed) * 0.4;
        pollen.points.rotation.y = Math.sin(t * 0.02 + drift) * 0.05;
      },
    },
  );
}
