/// <reference types="bun" />

import { expect } from 'bun:test';
import * as THREE from 'three';
import { TOKENS } from '@kit/tokens';
import type {
  AlignmentProbe,
  FrameState,
  FrameUpdate,
  QualityTier,
  SceneContext,
  SceneEvent,
  SceneFactory,
  SceneFixture,
  SceneModule,
  SceneSize,
} from '@kit/types';
import type { EnvGeometry } from '@sim/types';


export interface StubRendererCapabilities {
  isWebGL2: boolean;
  maxTextureSize: number;
  maxAttributes: number;
  maxSamples: number;
  precision: string;
  getMaxAnisotropy(): number;
  getMaxPrecision(): string;
}

export function stubRenderer(): THREE.WebGLRenderer {
  const capabilities: StubRendererCapabilities = {
    isWebGL2: true,
    maxTextureSize: 8192,
    maxAttributes: 16,
    maxSamples: 4,
    precision: 'highp',
    getMaxAnisotropy: () => 16,
    getMaxPrecision: () => 'highp',
  };
  return { capabilities, outputColorSpace: THREE.SRGBColorSpace } as unknown as THREE.WebGLRenderer;
}

export interface StubCanvas {
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
  listeners: Map<string, number>;
  addEventListener(type: string, listener: unknown, options?: unknown): void;
  removeEventListener(type: string, listener: unknown, options?: unknown): void;
  setPointerCapture(id: number): void;
  releasePointerCapture(id: number): void;
  getBoundingClientRect(): { left: number; top: number; width: number; height: number };
}

export function stubCanvas(width = 1280, height = 720): THREE.WebGLRenderer['domElement'] {
  const listeners = new Map<string, number>();
  const canvas: StubCanvas = {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    listeners,
    addEventListener(type) {
      listeners.set(type, (listeners.get(type) ?? 0) + 1);
    },
    removeEventListener(type) {
      listeners.set(type, Math.max(0, (listeners.get(type) ?? 0) - 1));
    },
    setPointerCapture() {
    },
    releasePointerCapture() {
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
  };
  return canvas as unknown as HTMLCanvasElement;
}

export interface SceneContextOptions {
  geometry: EnvGeometry;
  tier?: QualityTier;
  seed?: number;
  size?: Partial<SceneSize>;
  time?: number | (() => number);
  canvas?: THREE.WebGLRenderer['domElement'];
}

export interface StubSceneContext extends SceneContext {
  setTime(t: number): void;
}

export function makeSceneContext(opts: SceneContextOptions): StubSceneContext {
  let now = typeof opts.time === 'number' ? opts.time : 0;
  const timeFn = typeof opts.time === 'function' ? opts.time : (): number => now;
  const size: SceneSize = {
    width: opts.size?.width ?? 1280,
    height: opts.size?.height ?? 720,
    dpr: opts.size?.dpr ?? 1,
  };
  const ctx: StubSceneContext = {
    renderer: stubRenderer(),
    canvas: opts.canvas ?? stubCanvas(size.width, size.height),
    size,
    tier: opts.tier ?? 'hero',
    tokens: TOKENS,
    geometry: opts.geometry,
    seed: opts.seed ?? 0,
    time: timeFn,
    setTime(t: number) {
      now = t;
    },
  };
  return ctx;
}


export function toFrameState(fixture: SceneFixture): FrameState {
  const pose = new Float32Array(9);
  for (let i = 0; i < 9; i++) pose[i] = fixture.pose[i] ?? 0;
  const ctrl = new Float32Array(6);
  if (fixture.ctrl) for (let i = 0; i < 6; i++) ctrl[i] = fixture.ctrl[i] ?? 0;
  return {
    pose,
    ctrl,
    done: false,
    step: fixture.step ?? 0,
    simTime: (fixture.step ?? 0) / 30,
    derived: fixture.derived ?? null,
  };
}

export interface FrameUpdateOptions {
  prev?: FrameState | null;
  events?: readonly SceneEvent[];
  dtReal?: number;
  scrubbed?: boolean;
}

export function toFrameUpdate(fixture: SceneFixture, opts: FrameUpdateOptions = {}): FrameUpdate {
  return {
    state: toFrameState(fixture),
    prev: opts.prev ?? null,
    events: opts.events ?? [],
    dtReal: opts.dtReal ?? 1 / 60,
    scrubbed: opts.scrubbed ?? true,
  };
}


export interface AssertAlignedOptions extends SceneContextOptions {
  tolerance?: number;
  context?: StubSceneContext;
  alreadyInitialized?: boolean;
}

export interface AlignedReport {
  context: StubSceneContext;
  worst: Map<string, number>;
}

const DEFAULT_TOLERANCE = 1e-3;

export async function assertAligned(
  module: SceneModule,
  fixtures: readonly SceneFixture[],
  opts: AssertAlignedOptions,
): Promise<AlignedReport> {
  expect(fixtures.length).toBeGreaterThanOrEqual(3);
  const context = opts.context ?? makeSceneContext(opts);
  if (!opts.alreadyInitialized) await module.init(context);

  const tolerance = opts.tolerance ?? DEFAULT_TOLERANCE;
  const worst = new Map<string, number>();
  const world = new THREE.Vector3();

  for (const fixture of fixtures) {
    const frame = toFrameUpdate(fixture);
    module.update(frame);
    const probes = module.probes();
    expect(probes.length).toBeGreaterThanOrEqual(2);
    for (const probe of probes) {
      const truth = probe.expectWorldPos(frame.state);
      probe.object.updateWorldMatrix(true, false);
      world.setFromMatrixPosition(probe.object.matrixWorld);
      const dx = world.x - (truth[0] ?? 0);
      const dy = world.y - (truth[1] ?? 0);
      const dz = world.z - (truth[2] ?? 0);
      const err = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const key = `${probe.name}`;
      worst.set(key, Math.max(worst.get(key) ?? 0, err));
      const tol = probe.tolerance ?? tolerance;
      if (!(err <= tol)) {
        throw new Error(
          `probe "${probe.name}" misaligned on fixture "${fixture.name}": ` +
            `world (${world.x.toFixed(6)}, ${world.y.toFixed(6)}, ${world.z.toFixed(6)}) ` +
            `vs truth (${(truth[0] ?? 0).toFixed(6)}, ${(truth[1] ?? 0).toFixed(6)}, ${(truth[2] ?? 0).toFixed(6)}) ` +
            `= ${err.toExponential(3)} > ${tol}`,
        );
      }
    }
  }
  return { context, worst };
}

export function probeNames(module: SceneModule): string[] {
  return module.probes().map((p: AlignmentProbe) => p.name);
}


export interface ResourceCensus {
  geometries: Set<THREE.BufferGeometry>;
  materials: Set<THREE.Material>;
  textures: Set<THREE.Texture>;
  objects: number;
}

export function censusOf(root: THREE.Object3D): ResourceCensus {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  let objects = 0;
  root.traverse((o) => {
    objects++;
    const withGeo = o as { geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] };
    if (withGeo.geometry) geometries.add(withGeo.geometry);
    const mat = withGeo.material;
    if (Array.isArray(mat)) for (const m of mat) materials.add(m);
    else if (mat) materials.add(mat);
  });
  for (const m of materials) {
    const record = m as unknown as Record<string, unknown>;
    for (const key of Object.keys(record)) {
      const value = record[key];
      if (value instanceof THREE.Texture) textures.add(value);
    }
    const uniforms = (m as unknown as { uniforms?: Record<string, { value?: unknown }> }).uniforms;
    if (uniforms) {
      for (const key of Object.keys(uniforms)) {
        const value = uniforms[key]?.value;
        if (value instanceof THREE.Texture) textures.add(value);
      }
    }
  }
  return { geometries, materials, textures, objects };
}

export interface DisposalTracker {
  disposed: Set<unknown>;
  restore(): void;
}

export function trackDisposals(): DisposalTracker {
  const disposed = new Set<unknown>();
  const geoProto = THREE.BufferGeometry.prototype;
  const matProto = THREE.Material.prototype;
  const texProto = THREE.Texture.prototype;
  const originalGeo = geoProto.dispose;
  const originalMat = matProto.dispose;
  const originalTex = texProto.dispose;
  geoProto.dispose = function patchedGeo(this: THREE.BufferGeometry) {
    disposed.add(this);
    return originalGeo.call(this);
  };
  matProto.dispose = function patchedMat(this: THREE.Material) {
    disposed.add(this);
    return originalMat.call(this);
  };
  texProto.dispose = function patchedTex(this: THREE.Texture) {
    disposed.add(this);
    return originalTex.call(this);
  };
  return {
    disposed,
    restore() {
      geoProto.dispose = originalGeo;
      matProto.dispose = originalMat;
      texProto.dispose = originalTex;
    },
  };
}

export interface RoundTripOptions extends SceneContextOptions {
  cycles?: number;
  fixtures?: readonly SceneFixture[];
}

export interface RoundTripReport {
  census: ResourceCensus[];
}

export async function assertInitDisposeRoundTrip(
  factory: SceneFactory,
  opts: RoundTripOptions,
): Promise<RoundTripReport> {
  const cycles = opts.cycles ?? 3;
  const census: ResourceCensus[] = [];
  const tracker = trackDisposals();
  try {
    for (let c = 0; c < cycles; c++) {
      const module = factory();
      const ctx = makeSceneContext(opts);
      await module.init(ctx);
      if (opts.fixtures) for (const f of opts.fixtures) module.update(toFrameUpdate(f));

      const live = censusOf(module.scene);
      census.push(live);

      module.dispose();

      expect(module.scene.children.length).toBe(0);
      for (const g of live.geometries) {
        if (!tracker.disposed.has(g)) {
          throw new Error(`cycle ${c}: geometry "${g.name || '(unnamed)'}" was never disposed`);
        }
      }
      for (const m of live.materials) {
        if (!tracker.disposed.has(m)) {
          throw new Error(`cycle ${c}: material "${m.name || '(unnamed)'}" was never disposed`);
        }
      }
      for (const t of live.textures) {
        if (!tracker.disposed.has(t)) {
          throw new Error(`cycle ${c}: texture "${t.name || '(unnamed)'}" was never disposed`);
        }
      }
    }
  } finally {
    tracker.restore();
  }

  const first = census[0];
  if (first) {
    for (let c = 1; c < census.length; c++) {
      const cur = census[c];
      if (!cur) continue;
      expect(`cycle${c}:geometries=${cur.geometries.size}`).toBe(`cycle${c}:geometries=${first.geometries.size}`);
      expect(`cycle${c}:materials=${cur.materials.size}`).toBe(`cycle${c}:materials=${first.materials.size}`);
      expect(`cycle${c}:objects=${cur.objects}`).toBe(`cycle${c}:objects=${first.objects}`);
    }
  }
  return { census };
}


export interface DrawBudget {
  draws: number;
  triangles: number;
}

export function measureBudget(module: SceneModule): DrawBudget {
  let draws = 0;
  let triangles = 0;
  module.scene.traverse((o) => {
    if (!o.visible) return;
    for (let p = o.parent; p; p = p.parent) if (!p.visible) return;
    const drawable = o as THREE.Mesh & {
      isMesh?: boolean;
      isPoints?: boolean;
      isLine?: boolean;
      isInstancedMesh?: boolean;
      count?: number;
      geometry?: THREE.BufferGeometry;
    };
    if (!drawable.isMesh && !drawable.isPoints && !drawable.isLine) return;
    const geometry = drawable.geometry;
    if (!geometry) return;
    if (drawable.isInstancedMesh === true && (drawable.count ?? 0) === 0) return;
    draws++;
    if (drawable.isMesh !== true) return;
    const index = geometry.index;
    const position = geometry.getAttribute('position');
    const verts = index ? index.count : (position?.count ?? 0);
    const instances = drawable.isInstancedMesh === true ? (drawable.count ?? 1) : 1;
    triangles += Math.floor(verts / 3) * instances;
  });
  return { draws, triangles };
}

export const THUMB_MAX_DRAWS = 40;
export const THUMB_MAX_TRIANGLES = 50_000;

export function assertThumbBudget(module: SceneModule, label: string): DrawBudget {
  const budget = measureBudget(module);
  if (budget.draws > THUMB_MAX_DRAWS || budget.triangles > THUMB_MAX_TRIANGLES) {
    throw new Error(
      `${label}: thumb tier submits ${budget.draws} draws / ${budget.triangles} tris, ` +
        `over STYLE.md's ${THUMB_MAX_DRAWS} / ${THUMB_MAX_TRIANGLES}`,
    );
  }
  expect(budget.draws).toBeLessThanOrEqual(THUMB_MAX_DRAWS);
  expect(budget.triangles).toBeLessThanOrEqual(THUMB_MAX_TRIANGLES);
  return budget;
}


export function probeSnapshot(module: SceneModule): number[] {
  const out: number[] = [];
  for (const probe of module.probes()) {
    probe.object.updateWorldMatrix(true, false);
    for (const v of probe.object.matrixWorld.elements) out.push(v);
  }
  return out;
}

export function assertUpdateIdempotent(module: SceneModule, fixture: SceneFixture): void {
  module.update(toFrameUpdate(fixture, { scrubbed: false, events: [] }));
  const a = probeSnapshot(module);
  module.update(toFrameUpdate(fixture, { scrubbed: false, events: [] }));
  const b = probeSnapshot(module);
  expect(b).toEqual(a);
}
