/// <reference types="bun" />
// Headless unit tests for the nature-kit CORE (kit/rng, textures, materials, sky,
// lighting, post, terrain). No WebGL context exists under `bun test`, so anything that
// needs a live renderer (EffectComposer construction, shader compilation) is exercised
// through its PURE parts instead:
//   * rng / textures / terrain — fully deterministic CPU math, tested directly
//   * materials — the onBeforeCompile patch is run against three's real ShaderLib source
//     and the resulting GLSL is asserted, which catches a chunk rename in three without
//     a GPU
//   * post — tier policy + shader definition
//   * lighting/sky — object graphs (light counts, budgets, uniform wiring)

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import {
  chance,
  gaussian,
  hash1i,
  hash2f,
  hash2i,
  hash3i,
  inDisk,
  jitter,
  mulberry32,
  onCircle,
  pick,
  pickWeighted,
  range,
  rangeInt,
  rngAt,
  shuffle,
  streamFor,
} from '@kit/rng';
import {
  MAX_TEXTURE_SIZE,
  canvasTexture,
  clampSize,
  fbm,
  fbmField,
  grainField,
  hasCanvas2D,
  radialSprite,
  rampData,
  rampTexture,
  sampleRamp,
  valueNoise2,
  voronoi,
  voronoiField,
} from '@kit/textures';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  patchToonFragment,
  setEmissiveColor,
  setEmissiveIntensity,
  setMaterialTime,
  setMaterialsTime,
  tierEffectScale,
  wispMaterial,
} from '@kit/materials';
import { isDepthColumnPalette, makeGradientSky, paletteSunDirection, skyFromPalette } from '@kit/sky';
import { applyFog, configureRendererShadows, makeFog, makeLightRig, maxPointLights, shadowMapSize } from '@kit/lighting';
import {
  OakGradeShader,
  bloomResolutionScale,
  buildComposer,
  gradeDefaults,
  msaaSamples,
  pulsedBloomStrength,
  usesPost,
} from '@kit/post';
import { makeTerrain, rectFlattenWeight, terrainSegmentScale } from '@kit/terrain';
import { TOKENS } from '@kit/tokens';
import type { QualityTier } from '@kit/types';

const TIERS: QualityTier[] = ['hero', 'medium', 'thumb'];

// ---------------------------------------------------------------------------
// rng
// ---------------------------------------------------------------------------

describe('kit/rng', () => {
  test('mulberry32 is deterministic and in range', () => {
    const a = mulberry32(12345);
    const b = mulberry32(12345);
    const seq: number[] = [];
    for (let i = 0; i < 512; i++) {
      const x = a();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      seq.push(x);
    }
    for (let i = 0; i < 512; i++) expect(b()).toBe(seq[i] as number);
  });

  test('different seeds give different streams', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    let same = 0;
    for (let i = 0; i < 64; i++) if (a() === b()) same++;
    expect(same).toBe(0);
  });

  test('mean of a long stream is ~0.5 (no gross bias)', () => {
    const r = mulberry32(7);
    let sum = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) sum += r();
    expect(Math.abs(sum / n - 0.5)).toBeLessThan(0.01);
  });

  test('range / rangeInt / jitter / chance stay inside their bounds', () => {
    const r = mulberry32(99);
    for (let i = 0; i < 500; i++) {
      const v = range(r, -3, 5);
      expect(v).toBeGreaterThanOrEqual(-3);
      expect(v).toBeLessThan(5);
      const n = rangeInt(r, 2, 6);
      expect(Number.isInteger(n)).toBe(true);
      expect(n).toBeGreaterThanOrEqual(2);
      expect(n).toBeLessThanOrEqual(5);
      const j = jitter(r, 0.25);
      expect(Math.abs(j)).toBeLessThanOrEqual(0.25);
      expect(typeof chance(r, 0.5)).toBe('boolean');
    }
    expect(rangeInt(mulberry32(1), 4, 4)).toBe(4);
  });

  test('pick / pickWeighted / shuffle are deterministic per seed', () => {
    const items = ['a', 'b', 'c', 'd'] as const;
    const picks = (seed: number) => {
      const r = mulberry32(seed);
      return [pick(r, items), pick(r, items), pick(r, items)].join('');
    };
    expect(picks(4)).toBe(picks(4));
    expect(() => pick(mulberry32(1), [])).toThrow();

    const r = mulberry32(3);
    const w = pickWeighted(r, ['x', 'y'], [0, 1]);
    expect(w).toBe('y');
    expect(pickWeighted(mulberry32(1), ['x', 'y'], [0, 0])).toBe('x');

    const s1 = shuffle(mulberry32(11), [1, 2, 3, 4, 5, 6]);
    const s2 = shuffle(mulberry32(11), [1, 2, 3, 4, 5, 6]);
    expect(s1).toEqual(s2);
    expect([...s1].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test('gaussian is finite and roughly standard', () => {
    const r = mulberry32(17);
    let sum = 0;
    let sum2 = 0;
    const n = 8000;
    for (let i = 0; i < n; i++) {
      const g = gaussian(r);
      expect(Number.isFinite(g)).toBe(true);
      sum += g;
      sum2 += g * g;
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.08);
    expect(Math.abs(sum2 / n - 1)).toBeLessThan(0.12);
  });

  test('onCircle/inDisk write in place and stay on/in the unit circle', () => {
    const r = mulberry32(5);
    const out: [number, number] = [0, 0];
    for (let i = 0; i < 200; i++) {
      onCircle(r, out);
      expect(Math.abs(Math.hypot(out[0], out[1]) - 1)).toBeLessThan(1e-12);
      inDisk(r, out);
      expect(Math.hypot(out[0], out[1])).toBeLessThanOrEqual(1);
    }
  });

  test('hash2i is a pure uint32 function of (x, y, seed)', () => {
    expect(hash2i(3, 4, 1)).toBe(hash2i(3, 4, 1));
    expect(hash2i(3, 4, 1)).not.toBe(hash2i(4, 3, 1));
    expect(hash2i(3, 4, 1)).not.toBe(hash2i(3, 4, 2));
    for (let x = -20; x < 20; x++) {
      for (let y = -20; y < 20; y++) {
        const h = hash2i(x, y, 7);
        expect(Number.isInteger(h)).toBe(true);
        expect(h).toBeGreaterThanOrEqual(0);
        expect(h).toBeLessThanOrEqual(0xffffffff);
        const f = hash2f(x, y, 7);
        expect(f).toBeGreaterThanOrEqual(0);
        expect(f).toBeLessThan(1);
      }
    }
    expect(hash1i(9, 2)).toBe(hash1i(9, 2));
    expect(hash3i(1, 2, 3, 4)).toBe(hash3i(1, 2, 3, 4));
    expect(hash3i(1, 2, 3, 4)).not.toBe(hash3i(1, 2, 4, 4));
  });

  test('hash2i spreads over a lattice (no obvious clustering)', () => {
    const buckets = new Array<number>(16).fill(0);
    for (let x = 0; x < 64; x++) {
      for (let y = 0; y < 64; y++) {
        const b = Math.min(15, Math.floor(hash2f(x, y, 42) * 16));
        buckets[b] = (buckets[b] as number) + 1;
      }
    }
    const expected = (64 * 64) / 16;
    for (const b of buckets) expect(Math.abs(b - expected)).toBeLessThan(expected * 0.35);
  });

  test('rngAt gives a tile the same stream regardless of visit order', () => {
    const forward: number[] = [];
    for (let i = 0; i < 8; i++) forward.push(rngAt(i, 3, 1234)());
    const backward: number[] = [];
    for (let i = 7; i >= 0; i--) backward.unshift(rngAt(i, 3, 1234)());
    expect(forward).toEqual(backward);
  });

  test('streamFor separates named sub-streams of one seed', () => {
    const grass = streamFor(77, 'grass');
    const rocks = streamFor(77, 'rocks');
    expect(grass()).not.toBe(rocks());
    expect(streamFor(77, 'grass')()).toBe(streamFor(77, 'grass')());
  });
});

// ---------------------------------------------------------------------------
// textures
// ---------------------------------------------------------------------------

describe('kit/textures', () => {
  test('clampSize enforces the ≤1024 power-of-two contract', () => {
    expect(clampSize(4096)).toBe(MAX_TEXTURE_SIZE);
    expect(clampSize(1024)).toBe(1024);
    expect(clampSize(300)).toBe(256);
    expect(clampSize(1)).toBe(4);
    expect(clampSize(Number.NaN)).toBe(4);
  });

  test('valueNoise2 tiles seamlessly on its period', () => {
    for (let i = 0; i < 20; i++) {
      const x = i * 0.37;
      const y = i * 0.11;
      expect(valueNoise2(x, y, 8, 3)).toBeCloseTo(valueNoise2(x + 8, y, 8, 3), 12);
      expect(valueNoise2(x, y, 8, 3)).toBeCloseTo(valueNoise2(x, y + 8, 8, 3), 12);
    }
  });

  test('fbmField is deterministic, in [0,1], and seed-sensitive', () => {
    const a = fbmField(5, { size: 32, frequency: 4, octaves: 3 });
    const b = fbmField(5, { size: 32, frequency: 4, octaves: 3 });
    const c = fbmField(6, { size: 32, frequency: 4, octaves: 3 });
    expect(a.size).toBe(32);
    expect(a.data.length).toBe(32 * 32);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
    let diff = 0;
    for (let i = 0; i < a.data.length; i++) {
      const v = a.data[i] as number;
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      if (v !== (c.data[i] as number)) diff++;
    }
    expect(diff).toBeGreaterThan(a.data.length * 0.9);
  });

  test('fbmField is not constant (real variation)', () => {
    const { data } = fbmField(1, { size: 64, frequency: 4, octaves: 4 });
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    expect(max - min).toBeGreaterThan(0.25);
  });

  test('ridged fbm stays in range', () => {
    const { data } = fbmField(2, { size: 32, ridged: true, octaves: 3 });
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  test('voronoiField returns co-registered f1/edge/cell channels', () => {
    const v = voronoiField(9, { size: 32, cells: 4 });
    expect(v.f1.length).toBe(32 * 32);
    expect(v.edge.length).toBe(32 * 32);
    expect(v.cell.length).toBe(32 * 32);
    let maxEdge = 0;
    for (let i = 0; i < v.f1.length; i++) {
      expect(v.f1[i] as number).toBeGreaterThanOrEqual(0);
      expect(v.f1[i] as number).toBeLessThanOrEqual(1);
      maxEdge = Math.max(maxEdge, v.edge[i] as number);
    }
    expect(maxEdge).toBeGreaterThan(0);
    const again = voronoiField(9, { size: 32, cells: 4 });
    expect(Array.from(v.f1)).toEqual(Array.from(again.f1));
  });

  test('rampData interpolates between stops and pins the ends', () => {
    const stops = [
      { t: 0, color: 0x000000 },
      { t: 1, color: 0xffffff },
    ];
    const data = rampData(stops, 64);
    expect(data.length).toBe(64 * 4);
    expect(data[0] as number).toBeLessThan(8);
    expect(data[63 * 4] as number).toBeGreaterThan(247);
    for (let i = 1; i < 64; i++) {
      expect(data[i * 4] as number).toBeGreaterThanOrEqual(data[(i - 1) * 4] as number);
    }
  });

  test('sampleRamp honours unsorted-safe lookup, clamping and alpha', () => {
    const out = new Uint8Array(4);
    const stops = [
      { t: 0.25, color: 0xff0000, alpha: 0 },
      { t: 0.75, color: 0x0000ff, alpha: 1 },
    ];
    sampleRamp(stops, 0, out);
    expect([out[0], out[1], out[2], out[3]]).toEqual([255, 0, 0, 0]);
    sampleRamp(stops, 1, out);
    expect([out[0], out[1], out[2], out[3]]).toEqual([0, 0, 255, 255]);
    sampleRamp(stops, 0.5, out);
    expect(out[0] as number).toBeCloseTo(128, -1);
    expect(out[2] as number).toBeCloseTo(128, -1);
    expect(out[3] as number).toBeCloseTo(128, -1);
    sampleRamp([], 0.5, out);
    expect(out[3]).toBe(255);
  });

  test('generators build real DataTextures headlessly', () => {
    const f = fbm(3, { size: 32 });
    expect(f.image.width).toBe(32);
    expect(f.image.height).toBe(32);
    expect(f.version).toBeGreaterThan(0); // needsUpdate is write-only; version proves the upload flag
    const v = voronoi(3, { size: 16, cells: 3 });
    expect(v.image.width).toBe(16);
    const r = rampTexture([{ t: 0, color: TOKENS.accents['mountain-car'] as number }], 32);
    expect(r.image.width).toBe(32);
    expect(r.image.height).toBe(1);
    expect(r.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(r.wrapS).toBe(THREE.ClampToEdgeWrapping);
    const s = radialSprite(32, 2);
    expect(s.image.width).toBe(32);
    for (const t of [f, v, r, s]) t.dispose();
  });

  test('grainField is centered and bounded', () => {
    const { data } = grainField(4, { size: 32, strength: 0.5 });
    let sum = 0;
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      sum += v;
    }
    expect(Math.abs(sum / data.length - 0.5)).toBeLessThan(0.05);
  });

  test('canvas generation is guarded headlessly', () => {
    expect(hasCanvas2D()).toBe(false);
    expect(canvasTexture(32, () => {})).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// materials
// ---------------------------------------------------------------------------

/** Run a material's onBeforeCompile against three's real physical shader source. */
function compilePatch(material: THREE.Material): { vertexShader: string; fragmentShader: string; uniforms: Record<string, { value: unknown }> } {
  const lib = THREE.ShaderLib.physical as { vertexShader: string; fragmentShader: string };
  const shader = {
    name: 'test',
    uniforms: {} as Record<string, { value: unknown }>,
    vertexShader: lib.vertexShader,
    fragmentShader: lib.fragmentShader,
    defines: {},
  };
  const fn = material.onBeforeCompile as unknown as (s: typeof shader) => void;
  fn.call(material, shader);
  return shader;
}

describe('kit/materials', () => {
  test('patchToonFragment replaces the physical diffuse term', () => {
    const src = THREE.ShaderLib.physical.fragmentShader;
    const out = patchToonFragment(src);
    expect(out).not.toBe(src);
    expect(out).toContain('oakToonRamp( dotNL )');
    expect(out).toContain('reflectedLight.directDiffuse +=');
    // specular must stay physical
    expect(out).toContain('reflectedLight.directSpecular += irradiance *');
    expect(out).not.toContain('#include <lights_physical_pars_fragment>');
  });

  test('oakMaterial defaults: toon ramp on, rim/wind off, uniform handle present', () => {
    const m = oakMaterial({ color: 0x88aa66 });
    expect(m).toBeInstanceOf(THREE.MeshStandardMaterial);
    const h = oakUniforms(m);
    expect(h).not.toBeNull();
    expect(h?.flags).toEqual({ toon: true, rim: false, wind: false });
    const shader = compilePatch(m);
    expect(shader.fragmentShader).toContain('float oakToonRamp');
    expect(shader.fragmentShader).toContain('uToonSoftness');
    expect(shader.vertexShader).not.toContain('uWindAmp');
    expect(shader.uniforms.uToonSteps?.value).toBe(3);
    expect(shader.uniforms.uToonSoftness?.value).toBeCloseTo(0.25, 10);
    m.dispose();
  });

  test('rim + wind patches inject only when enabled, and share uniform objects', () => {
    const m = oakMaterial({
      color: 0x557733,
      rimColor: 0x9fd4ff,
      rimStrength: 0.4,
      wind: { amplitude: 0.08, frequency: 1.5, height: 0.7 },
      tier: 'hero',
    });
    const h = oakUniforms(m);
    expect(h?.flags).toEqual({ toon: true, rim: true, wind: true });
    const shader = compilePatch(m);
    expect(shader.fragmentShader).toContain('uRimStrength');
    expect(shader.fragmentShader).toContain('oakFresnel');
    expect(shader.vertexShader).toContain('uWindAmp');
    expect(shader.vertexShader).toContain('#ifdef USE_INSTANCING');
    // the same uniform object the handle drives is the one the program binds
    expect(shader.uniforms.uTime).toBe(h?.uniforms.uTime as unknown as { value: unknown });
    h?.setTime(4.5);
    expect(shader.uniforms.uTime?.value).toBe(4.5);
    setMaterialTime(m, 9);
    expect(shader.uniforms.uTime?.value).toBe(9);
    setMaterialsTime([m, null, undefined], 11);
    expect(shader.uniforms.uTime?.value).toBe(11);
    h?.setWind(0.2, 2);
    expect(shader.uniforms.uWindAmp?.value).toBe(0.2);
    expect(shader.uniforms.uWindFreq?.value).toBe(2);
    h?.setRim(0.1, 0xff0000);
    expect(shader.uniforms.uRimStrength?.value).toBe(0.1);
    h?.setToon(2, 0.4);
    expect(shader.uniforms.uToonSteps?.value).toBe(2);
    m.dispose();
  });

  test('tier scaling: thumb drops rim and wind entirely', () => {
    expect(tierEffectScale('hero')).toEqual({ rim: 1, wind: 1 });
    expect(tierEffectScale('thumb')).toEqual({ rim: 0, wind: 0 });
    const thumb = oakMaterial({ rimColor: 0xffffff, rimStrength: 1, wind: 0.2, tier: 'thumb' });
    expect(oakUniforms(thumb)?.flags).toEqual({ toon: true, rim: false, wind: false });
    const shader = compilePatch(thumb);
    expect(shader.fragmentShader).not.toContain('oakFresnel');
    expect(shader.vertexShader).not.toContain('uWindAmp');
    const medium = oakMaterial({ rimColor: 0xffffff, rimStrength: 1, wind: 0.2, tier: 'medium' });
    expect(oakUniforms(medium)?.uniforms.uRimStrength.value).toBeCloseTo(0.75, 10);
    expect(oakUniforms(medium)?.uniforms.uWindAmp.value).toBeCloseTo(0.14, 10);
    thumb.dispose();
    medium.dispose();
  });

  test('customProgramCacheKey separates patch variants', () => {
    const plain = oakMaterial({});
    const rim = oakMaterial({ rimColor: 0xffffff, rimStrength: 0.5 });
    const wind = oakMaterial({ wind: 0.1 });
    const k = (m: THREE.Material) => m.customProgramCacheKey();
    expect(k(plain)).not.toBe(k(rim));
    expect(k(plain)).not.toBe(k(wind));
    expect(k(rim)).not.toBe(k(wind));
    expect(k(oakMaterial({ rimColor: 0x00ff00, rimStrength: 0.9 }))).toBe(k(rim));
    for (const m of [plain, rim, wind]) m.dispose();
  });

  test('one material instance is shareable across meshes (instancing-safe)', () => {
    const m = oakMaterial({ color: 0xffffff, wind: 0.05 });
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const a = new THREE.Mesh(geo, m);
    const b = new THREE.InstancedMesh(geo, m, 16);
    expect(a.material).toBe(m);
    expect(b.material).toBe(m);
    oakUniforms(m)?.setTime(2);
    expect(oakUniforms(b.material as THREE.Material)?.uniforms.uTime.value).toBe(2);
    b.dispose();
    geo.dispose();
    m.dispose();
  });

  test('emissiveMaterial is HDR (>1) and re-tintable without losing intensity', () => {
    const e = emissiveMaterial(0x808080, 2);
    expect(e.color.r).toBeGreaterThan(e.baseColor.r);
    expect(e.color.r / e.baseColor.r).toBeCloseTo(2, 6);
    expect(e.fog).toBe(false);
    setEmissiveIntensity(e, 4);
    expect(e.color.r / e.baseColor.r).toBeCloseTo(4, 6);
    setEmissiveColor(e, TOKENS.cueMagenta);
    expect(e.baseColor.getHex(THREE.SRGBColorSpace)).toBe(TOKENS.cueMagenta);
    expect(e.color.r / e.baseColor.r).toBeCloseTo(4, 6);
    e.dispose();
  });

  test('wispMaterial is additive, depth-safe, and time/pulse driven', () => {
    const w = wispMaterial(TOKENS.uiCyan, { intensity: 2 });
    expect(w.blending).toBe(THREE.AdditiveBlending);
    expect(w.depthWrite).toBe(false);
    expect(w.transparent).toBe(true);
    expect(w.fragmentShader).toContain('uPulse');
    oakUniforms(w)?.setTime(3);
    expect(w.uniforms.uTime?.value).toBe(3);
    w.pulse.value = 1;
    expect(w.uniforms.uPulse?.value).toBe(1);
    w.dispose();
  });

  test('oakUniforms returns null for a plain material', () => {
    const plain = new THREE.MeshBasicMaterial();
    expect(oakUniforms(plain)).toBeNull();
    expect(oakUniforms(null)).toBeNull();
    setMaterialTime(plain, 1); // must not throw
    plain.dispose();
  });
});

// ---------------------------------------------------------------------------
// sky
// ---------------------------------------------------------------------------

describe('kit/sky', () => {
  test('gradient sky is a BackSide, depth-free dome that follows the camera', () => {
    const sky = makeGradientSky({ top: 0x223366, horizon: 0xffaa66, bottom: 0x332211, sunDir: [0, 1, 0] });
    expect(sky.material.side).toBe(THREE.BackSide);
    expect(sky.material.depthWrite).toBe(false);
    expect(sky.material.depthTest).toBe(false);
    expect(sky.mesh.renderOrder).toBeLessThan(0);
    expect(sky.mesh.frustumCulled).toBe(false);

    const camera = new THREE.PerspectiveCamera();
    camera.position.set(3, 4, 5);
    const renderer = null as unknown as THREE.WebGLRenderer;
    const scene = new THREE.Scene();
    sky.mesh.onBeforeRender(renderer, scene, camera, sky.mesh.geometry, sky.material, {} as never);
    expect(sky.mesh.position.toArray()).toEqual([3, 4, 5]);
    sky.dispose();
  });

  test('sun disc is HDR and compiled out for a depth column', () => {
    const withSun = makeGradientSky({ top: 1, horizon: 2, bottom: 3, sunDir: [0, 1, 0], sunHdr: 6 });
    expect(withSun.material.fragmentShader).toContain('uSunHdr');
    expect(withSun.uniforms.uSunHdr.value).toBeGreaterThan(1);

    const column = makeGradientSky({ top: 1, horizon: 2, bottom: 3, sunDir: null });
    expect(column.material.fragmentShader).not.toContain('uSunHdr');
    expect(column.material.fragmentShader).toContain('uBottom');
    withSun.dispose();
    column.dispose();
  });

  test('stars compile in only when asked, and never at thumb tier', () => {
    const night = makeGradientSky({ top: 1, horizon: 2, bottom: 3, stars: true, tier: 'hero' });
    expect(night.material.fragmentShader).toContain('uStarDensity');
    const day = makeGradientSky({ top: 1, horizon: 2, bottom: 3 });
    expect(day.material.fragmentShader).not.toContain('uStarDensity');
    const thumb = makeGradientSky({ top: 1, horizon: 2, bottom: 3, stars: true, tier: 'thumb' });
    expect(thumb.material.fragmentShader).not.toContain('uStarDensity');
    for (const s of [night, day, thumb]) s.dispose();
  });

  test('setSunDirection normalizes and never yields a zero vector', () => {
    const sky = makeGradientSky({ top: 1, horizon: 2, bottom: 3, sunDir: [0, 1, 0] });
    sky.setSunDirection(0, 0, 0);
    expect(sky.sunDirection.length()).toBeCloseTo(1, 10);
    sky.setSunDirection(3, 0, 4);
    expect(sky.sunDirection.length()).toBeCloseTo(1, 10);
    expect(sky.sunDirection.x).toBeCloseTo(0.6, 10);
    expect(sky.uniforms.uSunDir.value).toBe(sky.sunDirection);
    sky.setTime(1.25);
    expect(sky.uniforms.uTime.value).toBe(1.25);
    sky.setColors(0x010203, 0x040506, 0x070809);
    expect(sky.uniforms.uTop.value.getHex(THREE.SRGBColorSpace)).toBe(0x010203);
    sky.dispose();
  });

  test('palette sun elevation tracks the time of day', () => {
    const noon = paletteSunDirection(TOKENS.palettes.noon);
    const dusk = paletteSunDirection(TOKENS.palettes.dusk);
    const night = paletteSunDirection(TOKENS.palettes.night);
    expect(noon.y).toBeGreaterThan(dusk.y);
    expect(dusk.y).toBeGreaterThan(night.y);
    expect(noon.length()).toBeCloseTo(1, 10);
  });

  test('skyFromPalette drops the sun for water/cave palettes', () => {
    expect(isDepthColumnPalette('underwater')).toBe(true);
    expect(isDepthColumnPalette('abyss')).toBe(true);
    expect(isDepthColumnPalette('cave')).toBe(true);
    expect(isDepthColumnPalette('dusk')).toBe(false);

    const dusk = skyFromPalette(TOKENS.palettes.dusk, 'dusk', 'hero');
    expect(dusk.material.fragmentShader).toContain('uSunHdr');
    expect(dusk.uniforms.uTop.value.getHex(THREE.SRGBColorSpace)).toBe(TOKENS.palettes.dusk.skyTop);

    const sea = skyFromPalette(TOKENS.palettes.underwater, 'underwater', 'hero');
    expect(sea.material.fragmentShader).not.toContain('uSunHdr');

    const night = skyFromPalette(TOKENS.palettes.night, 'night', 'hero');
    expect(night.material.fragmentShader).toContain('uStarDensity');
    for (const s of [dusk, sea, night]) s.dispose();
  });
});

// ---------------------------------------------------------------------------
// lighting
// ---------------------------------------------------------------------------

describe('kit/lighting', () => {
  test('tier budgets match STYLE.md exactly', () => {
    expect(maxPointLights('hero')).toBe(4);
    expect(maxPointLights('medium')).toBe(2);
    expect(maxPointLights('thumb')).toBe(0);
    expect(shadowMapSize('hero')).toBe(2048);
    expect(shadowMapSize('medium')).toBe(1024);
    expect(shadowMapSize('thumb')).toBe(0);
  });

  test('every tier builds exactly one directional + one hemisphere + ≤budget points', () => {
    for (const tier of TIERS) {
      const rig = makeLightRig(TOKENS.palettes.dusk, tier);
      const lights: THREE.Light[] = [];
      rig.group.traverse((o) => {
        const l = o as THREE.Light;
        if (l.isLight) lights.push(l);
      });
      const dir = lights.filter((l) => (l as THREE.DirectionalLight).isDirectionalLight);
      const hemi = lights.filter((l) => (l as THREE.HemisphereLight).isHemisphereLight);
      const points = lights.filter((l) => (l as THREE.PointLight).isPointLight);
      expect(dir.length).toBe(1);
      expect(hemi.length).toBe(1);
      expect(points.length).toBe(maxPointLights(tier));
      expect(rig.points.length).toBe(maxPointLights(tier));
      expect(lights.length).toBe(2 + maxPointLights(tier));

      const shadowCasters = lights.filter((l) => l.castShadow);
      expect(shadowCasters.length).toBe(tier === 'thumb' ? 0 : 1);
      if (tier !== 'thumb') {
        expect(rig.sun.shadow.mapSize.width).toBe(shadowMapSize(tier));
      }
      // point lights start silent but VISIBLE, so light counts never change mid-run
      for (const p of rig.points) {
        expect(p.intensity).toBe(0);
        expect(p.visible).toBe(true);
        expect(p.castShadow).toBe(false);
      }
      rig.dispose();
      expect(rig.points.length).toBe(0);
    }
  });

  test('pointCount is clamped to the tier budget', () => {
    const rig = makeLightRig(TOKENS.palettes.noon, 'medium', { pointCount: 9 });
    expect(rig.points.length).toBe(2);
    rig.dispose();
    const one = makeLightRig(TOKENS.palettes.noon, 'hero', { pointCount: 1 });
    expect(one.points.length).toBe(1);
    one.dispose();
  });

  test('rig colors and intensities come from the palette', () => {
    const p = TOKENS.palettes.night;
    const rig = makeLightRig(p, 'hero');
    expect(rig.sun.color.getHex(THREE.SRGBColorSpace)).toBe(p.sun);
    expect(rig.sun.intensity).toBe(p.sunIntensity);
    expect(rig.hemi.intensity).toBe(p.hemiIntensity);
    expect(rig.hemi.color.getHex(THREE.SRGBColorSpace)).toBe(p.hemiSky);
    expect(rig.hemi.groundColor.getHex(THREE.SRGBColorSpace)).toBe(p.hemiGround);
    expect((rig.points[0] as THREE.PointLight).color.getHex(THREE.SRGBColorSpace)).toBe(p.accentLight);
    rig.dispose();
  });

  test('setShadowFocus aims the key light and fits the frustum', () => {
    const rig = makeLightRig(TOKENS.palettes.morning, 'hero', { sunDir: [0, 1, 0] });
    rig.setShadowFocus({ x: 2, y: 0, z: -3 }, 6);
    expect(rig.sun.target.position.toArray()).toEqual([2, 0, -3]);
    expect(rig.sun.position.x).toBeCloseTo(2, 10);
    expect(rig.sun.position.z).toBeCloseTo(-3, 10);
    expect(rig.sun.position.y).toBeCloseTo(6 * 2.5, 10);
    const cam = rig.sun.shadow.camera;
    expect(cam.right).toBe(6);
    expect(cam.left).toBe(-6);
    expect(cam.top).toBe(6);
    expect(cam.bottom).toBe(-6);
    expect(cam.far).toBeGreaterThan(cam.near);
    expect(rig.sun.target.parent).toBe(rig.group);
    rig.dispose();
  });

  test('setSunDirection keeps the focus and re-normalizes', () => {
    const rig = makeLightRig(TOKENS.palettes.dawn, 'hero');
    rig.setShadowFocus({ x: 1, y: 0, z: 1 }, 4);
    rig.setSunDirection(0, 5, 0);
    expect(rig.sunDirection.length()).toBeCloseTo(1, 10);
    expect(rig.sun.target.position.toArray()).toEqual([1, 0, 1]);
    expect(rig.sun.position.y).toBeCloseTo(4 * 2.5, 10);
    rig.dispose();
  });

  test('fog density is derived from the palette hint', () => {
    const p = TOKENS.palettes.dusk;
    const fog = makeFog(p, 40);
    expect(fog).toBeInstanceOf(THREE.FogExp2);
    expect(fog.color.getHex(THREE.SRGBColorSpace)).toBe(p.fog);
    const reach = p.fogDensityHint * 40;
    // ~95% extinction at hint*extent
    expect(Math.exp(-Math.pow(fog.density * reach, 2))).toBeCloseTo(0.05, 3);
    // denser palettes (higher hint means fog closes in later) stay monotone in extent
    expect(makeFog(p, 80).density).toBeLessThan(fog.density);

    const scene = new THREE.Scene();
    applyFog(scene, p, 20);
    expect(scene.fog).toBeInstanceOf(THREE.FogExp2);
    expect(scene.background).toBeNull();
  });

  test('configureRendererShadows sets PCF per tier (host-only helper)', () => {
    // three 0.185 deprecated PCFSoftShadowMap (silently substitutes PCFShadowMap);
    // the kit sets PCFShadowMap directly to keep the identical result warning-free.
    const fake = { shadowMap: { enabled: false, type: THREE.BasicShadowMap } } as unknown as THREE.WebGLRenderer;
    configureRendererShadows(fake, 'hero');
    expect(fake.shadowMap.enabled).toBe(true);
    expect(fake.shadowMap.type).toBe(THREE.PCFShadowMap);
    configureRendererShadows(fake, 'thumb');
    expect(fake.shadowMap.enabled).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// post
// ---------------------------------------------------------------------------

describe('kit/post', () => {
  test('tier policy: half-res hero bloom, quarter medium, none thumb', () => {
    expect(bloomResolutionScale('hero')).toBe(0.5);
    expect(bloomResolutionScale('medium')).toBe(0.25);
    expect(bloomResolutionScale('thumb')).toBe(0);
    expect(msaaSamples('hero')).toBe(4);
    expect(msaaSamples('medium')).toBe(0);
    expect(msaaSamples('thumb')).toBe(0);
    expect(usesPost('hero')).toBe(true);
    expect(usesPost('thumb')).toBe(false);
  });

  test('bloom pulse is monotone and returns to base', () => {
    expect(pulsedBloomStrength(0.5, 0)).toBe(0.5);
    expect(pulsedBloomStrength(0.5, 1)).toBeGreaterThan(0.5);
    expect(pulsedBloomStrength(0.5, 0.5)).toBeGreaterThan(pulsedBloomStrength(0.5, 0.25));
    expect(pulsedBloomStrength(0.5, 2)).toBe(pulsedBloomStrength(0.5, 1));
    expect(pulsedBloomStrength(0.5, -1)).toBe(0.5);
  });

  test('grade defaults are token-driven and tier-scaled', () => {
    const cfg = { bloomStrength: 0.5, bloomThreshold: 0.85, fogTint: TOKENS.palettes.dusk.fog };
    const hero = gradeDefaults(cfg, 'hero');
    const medium = gradeDefaults(cfg, 'medium');
    expect(hero.tint).toBe(cfg.fogTint);
    expect(hero.vignette).toBeGreaterThan(medium.vignette);
    expect(hero.grain).toBeGreaterThan(medium.grain);
    expect(hero.gamma).toBeGreaterThan(0.5);
    expect(hero.gamma).toBeLessThanOrEqual(1);
  });

  test('grade shader carries lift/tint/vignette/ordered-dither and no external asset', () => {
    expect(OakGradeShader.fragmentShader).toContain('uVignette');
    expect(OakGradeShader.fragmentShader).toContain('oakBayer8');
    expect(OakGradeShader.fragmentShader).toContain('uTint');
    expect(OakGradeShader.fragmentShader).toContain('uLift');
    expect(OakGradeShader.uniforms.tDiffuse.value).toBeNull();
    expect(OakGradeShader.fragmentShader).not.toContain('http');
  });

  test('thumb tier builds no composer and renders straight through', () => {
    let rendered = 0;
    let touchedRenderer = 0;
    const fake = {
      toneMapping: THREE.NoToneMapping,
      toneMappingExposure: 0,
      render: () => {
        rendered++;
      },
      setPixelRatio: () => {
        touchedRenderer++;
      },
      setSize: () => {
        touchedRenderer++;
      },
    } as unknown as THREE.WebGLRenderer;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const handle = buildComposer(
      fake,
      scene,
      camera,
      { bloomStrength: 0.5, bloomThreshold: 0.85, fogTint: 0xffffff },
      'thumb',
      { width: 320, height: 200, dpr: 1 },
    );
    expect(handle.composer).toBeNull();
    expect(handle.bloom).toBeNull();
    expect(handle.gradeUniforms).toBeNull();
    // exposure/tone mapping are owned here, not by scenes
    expect(fake.toneMapping).toBe(THREE.ACESFilmicToneMapping);
    expect(fake.toneMappingExposure).toBe(TOKENS.exposure);
    handle.setBloomPulse(1); // no-op, must not throw
    handle.render(0.016);
    expect(rendered).toBe(1);
    // resize touches the post chain only — canvas/renderer sizing is host-owned
    handle.resize(640, 480, 2);
    expect(touchedRenderer).toBe(0);
    handle.dispose();
  });
});

// ---------------------------------------------------------------------------
// terrain
// ---------------------------------------------------------------------------

describe('kit/terrain', () => {
  const wave = (x: number, z: number) => Math.sin(3 * x) * 0.45 + 0.55 + Math.cos(z) * 0.1;

  test('segment counts scale with tier', () => {
    expect(terrainSegmentScale('hero')).toBe(1);
    expect(terrainSegmentScale('medium')).toBe(0.6);
    expect(terrainSegmentScale('thumb')).toBe(0.3);
    const hero = makeTerrain({ size: 8, segments: 40, tier: 'hero' });
    const medium = makeTerrain({ size: 8, segments: 40, tier: 'medium' });
    const thumb = makeTerrain({ size: 8, segments: 40, tier: 'thumb' });
    expect(hero.segmentsX).toBe(40);
    expect(medium.segmentsX).toBe(24);
    expect(thumb.segmentsX).toBe(12);
    expect(thumb.mesh.receiveShadow).toBe(false);
    expect(hero.mesh.receiveShadow).toBe(true);
    for (const t of [hero, medium, thumb]) t.dispose();
  });

  test('sampleHeight matches heightFn at every lattice vertex', () => {
    const t = makeTerrain({ size: [12, 6], segments: [48, 24], heightFn: wave, tier: 'hero' });
    const dx = 12 / t.segmentsX;
    const dz = 6 / t.segmentsZ;
    for (let j = 0; j <= t.segmentsZ; j++) {
      for (let i = 0; i <= t.segmentsX; i++) {
        const x = -6 + i * dx;
        const z = -3 + j * dz;
        expect(t.sampleHeight(x, z)).toBeCloseTo(wave(x, z), 6);
      }
    }
    t.dispose();
  });

  test('mesh vertices sit exactly on sampleHeight (the accuracy contract)', () => {
    const t = makeTerrain({ size: 10, segments: 32, heightFn: wave, tier: 'hero' });
    const pos = t.geometry.getAttribute('position');
    for (let i = 0; i < pos.count; i += 7) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      expect(y).toBeCloseTo(t.sampleHeight(x, z), 6);
    }
    t.dispose();
  });

  test('bilinear interpolation is exact for a linear height field', () => {
    const lin = (x: number, z: number) => 0.25 * x - 0.5 * z + 1;
    const t = makeTerrain({ size: 8, segments: 8, heightFn: lin });
    for (const [x, z] of [
      [0.13, -1.7],
      [-3.9, 2.02],
      [1.5, 1.5],
      [0, 0],
    ] as const) {
      expect(t.sampleHeight(x, z)).toBeCloseTo(lin(x, z), 9);
    }
    t.dispose();
  });

  test('sampleHeight clamps outside the terrain instead of extrapolating', () => {
    const t = makeTerrain({ size: 4, segments: 4, heightFn: wave });
    expect(t.sampleHeight(-999, 0)).toBeCloseTo(t.sampleHeight(-2, 0), 9);
    expect(t.sampleHeight(999, 0)).toBeCloseTo(t.sampleHeight(2, 0), 9);
    expect(t.sampleHeight(0, -999)).toBeCloseTo(t.sampleHeight(0, -2), 9);
    expect(Number.isFinite(t.sampleHeight(0, 999))).toBe(true);
    t.dispose();
  });

  test('flat terrain by default; normals point up', () => {
    const t = makeTerrain({ size: 6, segments: 6 });
    expect(t.minHeight).toBe(0);
    expect(t.maxHeight).toBe(0);
    expect(t.sampleHeight(1.3, -2.1)).toBe(0);
    const n = t.sampleNormal(1.3, -2.1, new THREE.Vector3());
    expect(n.y).toBeCloseTo(1, 9);
    t.dispose();
  });

  test('sampleNormal follows the slope of the height field', () => {
    const ramp = (x: number) => 0.5 * x;
    const t = makeTerrain({ size: 8, segments: 32, heightFn: ramp });
    const n = t.sampleNormal(0, 0, new THREE.Vector3());
    expect(n.length()).toBeCloseTo(1, 9);
    expect(n.x).toBeLessThan(0); // uphill towards +x tilts the normal towards -x
    expect(n.y).toBeGreaterThan(0);
    expect(n.z).toBeCloseTo(0, 9);
    t.dispose();
  });

  test('flattenMask carves an exact inset with a feathered edge', () => {
    const rect = { x: 0, z: 0, halfX: 2, halfZ: 2, feather: 1 };
    expect(rectFlattenWeight(rect, 0, 0)).toBe(1);
    expect(rectFlattenWeight(rect, 2, 0)).toBe(1);
    expect(rectFlattenWeight(rect, 3.5, 0)).toBe(0);
    const mid = rectFlattenWeight(rect, 2.5, 0);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);

    const t = makeTerrain({
      size: 12,
      segments: 48,
      heightFn: () => 2,
      flattenMask: rect,
      flattenHeight: 0,
    });
    expect(t.sampleHeight(0, 0)).toBeCloseTo(0, 9);
    expect(t.sampleHeight(1.9, 1.9)).toBeCloseTo(0, 9);
    expect(t.sampleHeight(5.5, 5.5)).toBeCloseTo(2, 9);
    t.dispose();
  });

  test('flattenMask accepts a function mask', () => {
    const t = makeTerrain({
      size: 8,
      segments: 16,
      heightFn: () => 1,
      flattenMask: (x) => (x < 0 ? 1 : 0),
      flattenHeight: -0.5,
    });
    expect(t.sampleHeight(-3, 0)).toBeCloseTo(-0.5, 9);
    expect(t.sampleHeight(3, 0)).toBeCloseTo(1, 9);
    t.dispose();
  });

  test('geometry is indexed, colored, and wound +y up', () => {
    const t = makeTerrain({
      size: 4,
      segments: 4,
      heightFn: wave,
      colorRamp: [
        { t: 0, color: TOKENS.palettes.dusk.groundA },
        { t: 1, color: TOKENS.palettes.dusk.groundB },
      ],
      slopeColor: 0x333333,
      slopeStrength: 0.5,
      colorNoise: 0.03,
      seed: 3,
    });
    const index = t.geometry.getIndex();
    expect(index).not.toBeNull();
    expect(index?.count).toBe(4 * 4 * 2 * 3);
    const color = t.geometry.getAttribute('color');
    expect(color.itemSize).toBe(3);
    expect(color.count).toBe(5 * 5);
    for (let i = 0; i < color.count; i++) {
      expect(color.getX(i)).toBeGreaterThanOrEqual(0);
      expect(color.getX(i)).toBeLessThanOrEqual(1.05);
    }
    // first triangle must face +y
    const pos = t.geometry.getAttribute('position');
    const a = new THREE.Vector3().fromBufferAttribute(pos, index?.getX(0) as number);
    const b = new THREE.Vector3().fromBufferAttribute(pos, index?.getX(1) as number);
    const c = new THREE.Vector3().fromBufferAttribute(pos, index?.getX(2) as number);
    const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a));
    expect(n.y).toBeGreaterThan(0);
    t.dispose();
  });

  test('terrain is deterministic per seed and disposes cleanly', () => {
    const build = () => makeTerrain({ size: 6, segments: 12, heightFn: wave, seed: 42, jitter: 0.05, colorNoise: 0.1 });
    const a = build();
    const b = build();
    const pa = a.geometry.getAttribute('position').array as Float32Array;
    const pb = b.geometry.getAttribute('position').array as Float32Array;
    expect(Array.from(pa)).toEqual(Array.from(pb));

    let disposed = 0;
    a.geometry.addEventListener('dispose', () => disposed++);
    a.material.addEventListener('dispose', () => disposed++);
    a.dispose();
    expect(disposed).toBe(2);
    b.dispose();
  });

  test('an externally supplied material is not disposed by the terrain', () => {
    const shared = oakMaterial({ color: 0xffffff });
    let disposed = 0;
    shared.addEventListener('dispose', () => disposed++);
    const t = makeTerrain({ size: 4, segments: 4, material: shared });
    expect(t.material).toBe(shared);
    t.dispose();
    expect(disposed).toBe(0);
    shared.dispose();
    expect(disposed).toBe(1);
  });

  test('jitter never moves the border vertices (tiles stay seamless)', () => {
    const t = makeTerrain({ size: 4, segments: 4, jitter: 0.2, seed: 9 });
    const pos = t.geometry.getAttribute('position');
    const n = t.segmentsX + 1;
    for (let j = 0; j <= t.segmentsZ; j++) {
      for (let i = 0; i <= t.segmentsX; i++) {
        if (i !== 0 && i !== t.segmentsX && j !== 0 && j !== t.segmentsZ) continue;
        const idx = j * n + i;
        expect(pos.getX(idx)).toBeCloseTo(-2 + i * 1, 9);
        expect(pos.getZ(idx)).toBeCloseTo(-2 + j * 1, 9);
      }
    }
    t.dispose();
  });
});
