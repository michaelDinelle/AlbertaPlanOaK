import * as THREE from 'three';
import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
  type CelebrationPoolOptions,
} from './celebrate';
import { GroundTruthOverlay, type GroundTruthOptions } from './debug';
import { makeOrbitRig, type OrbitRig, type OrbitRigOptions } from './rigs';
import type { FrameUpdate, SceneContext, SceneModule, SceneSize } from './types';
import type { WorldCore } from './world';
import { SCENES } from '../scenes/table';

export interface Stage<W extends WorldCore> {
  ctx: SceneContext;
  world: W;
  pool: CelebrationPool;
  rig: OrbitRig;
  time: number;
}

export abstract class StageScene<W extends WorldCore> implements SceneModule {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  protected ctx: SceneContext | null = null;
  protected world: W | null = null;
  protected rig: OrbitRig | null = null;
  protected pool: CelebrationPool | null = null;
  protected overlay: GroundTruthOverlay | null = null;
  protected debugOn = false;

  protected abstract readonly stageName: string;
  protected abstract buildWorld(ctx: SceneContext): W | Promise<W>;
  protected abstract buildActors(ctx: SceneContext, world: W): void;
  protected abstract rigOptions(ctx: SceneContext): OrbitRigOptions;
  protected abstract overlayOptions(ctx: SceneContext): GroundTruthOptions;
  protected abstract start(ctx: SceneContext, world: W): void;
  protected abstract tick(frame: FrameUpdate, stage: Stage<W>): void;
  protected abstract drop(): void;

  protected poolOptions(_ctx: SceneContext): CelebrationPoolOptions {
    return {};
  }

  protected accentOf(ctx: SceneContext): number {
    return ctx.tokens.accents[SCENES[ctx.environment.family]!.slug] ?? ctx.tokens.celebrationGold;
  }

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = `${this.stageName}Scene`;
    const world = await this.buildWorld(ctx);
    this.world = world;
    this.scene.add(world.group);
    this.buildActors(ctx, world);
    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      fov: ctx.tokens.fovHero,
      name: `${this.stageName}OrbitRig`,
      ...this.rigOptions(ctx),
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();
    const pool = new CelebrationPool({
      seed: ctx.seed,
      name: `${this.stageName}Celebrations`,
      burstSize: SCENES[ctx.environment.family]!.burst,
      ...this.poolOptions(ctx),
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;
    const overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      name: `${this.stageName}GroundTruth`,
      ...this.overlayOptions(ctx),
    });
    this.scene.add(overlay.group);
    overlay.visible = this.debugOn;
    this.overlay = overlay;
    world.setPixelRatio(ctx.size.dpr);
    this.start(ctx, world);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !pool || !rig) return;
    this.tick(frame, { ctx, world, pool, rig, time: ctx.time() });
    if (this.debugOn && this.overlay) this.overlay.setState(frame.state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  resetCamera(): void {
    this.rig?.reset();
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.drop();
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
  }
}
