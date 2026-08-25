import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from './lighting';
import { blendTokens } from './mix';
import { skyFromPalette, type SkyHandle } from './sky';
import { makeTerrain, type TerrainHandle } from './terrain';
import { valueNoise2 } from './textures';
import { TOKENS, type Palette, type PaletteName } from './tokens';
import type { SceneContext } from './types';
import { makeWorld, type WorldBuilder, type WorldCore } from './world';

export const GROUNDS_PERIOD = 24;
const NOISE_CELLS = 8;

export interface GroundsOptions {
  paletteName: PaletteName;
  name: string;
  extent?: number;
  relief?: number;
  shadowRadius?: number;
  groundLift?: number;
}

export interface GroundsHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  setOrigin(x: number, z: number): void;
}

export interface GroundsParts {
  w: WorldBuilder;
  palette: Palette;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  shadowRadius: number;
  setOrigin(x: number, z: number): void;
}

export function groundsHeightFn(seed: number, relief: number): (x: number, z: number) => number {
  const scale = NOISE_CELLS / GROUNDS_PERIOD;
  return (x, z) =>
    relief * (valueNoise2(x * scale, z * scale, NOISE_CELLS, seed) - 0.5) +
    relief * 0.4 * (valueNoise2(x * scale * 3 + 11.3, z * scale * 3 + 4.7, NOISE_CELLS * 3, seed ^ 0x5bd1) - 0.5);
}

export function buildGrounds(ctx: SceneContext, scene: THREE.Scene, opts: GroundsOptions): GroundsParts {
  const palette = TOKENS.palettes[opts.paletteName];
  const w = makeWorld(opts.name);
  const extent = opts.extent ?? 64;
  const relief = opts.relief ?? 0.14;
  const shadowRadius = opts.shadowRadius ?? 5.2;
  const groundLift = opts.groundLift ?? 0;

  const sky = skyFromPalette(palette, opts.paletteName, { radius: 300 });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 2,
    pointDistance: 9,
    sunDistanceScale: 2.4,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, shadowRadius);

  applyFog(scene, palette, extent * 2.4);

  const rampColor = new THREE.Color();
  const groundLow = blendTokens(palette.groundA, palette.hemiGround, 0.2);
  const groundHigh = blendTokens(palette.groundB, palette.hemiSky, 0.12);
  const heightFn = groundsHeightFn(ctx.seed, relief);
  const terrain = makeTerrain({
    size: [extent, extent],
    segments: [110, 110],
    seed: ctx.seed,
    heightFn: (x, z) => groundLift + heightFn(x, z),
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: groundHigh,
    slopeStrength: 0.4,
    slopeSharpness: 2.2,
    colorRamp: (h) => {
      const k = relief > 0 ? Math.min(1, Math.max(0, (h - groundLift) / relief + 0.5)) : 0.5;
      rampColor.copy(groundLow).lerp(groundHigh, k);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.hemiSky,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.34,
      toonFloor: 0.06,
    },
    receiveShadow: true,
    castShadow: false,
    name: `${opts.name}Ground`,
  });
  w.add(terrain.mesh);
  w.own(terrain);

  const setOrigin = (x: number, z: number): void => {
    terrain.mesh.position.x = Math.round(x / GROUNDS_PERIOD) * GROUNDS_PERIOD;
    terrain.mesh.position.z = Math.round(z / GROUNDS_PERIOD) * GROUNDS_PERIOD;
    rig.setShadowFocus({ x, y: groundLift, z }, shadowRadius);
  };

  return { w, palette, sky, rig, terrain, shadowRadius, setOrigin };
}
