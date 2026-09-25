import * as THREE from 'three/webgpu';
import { CelebrationPool, attachCelebrations, disposeCelebrations, type CelebrationPoolOptions } from './celebrate';
import { GroundTruthOverlay, type GroundTruthOptions } from './debug';
import { makeLight2D, type Light2D } from './light2d';
import { accentOf } from './stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneModule, ScenePost, SceneSize } from './types';

const MARGIN = 0.06;
const HEIGHT = 50;


export interface Stage2D {
  ctx: SceneContext;
  pool: CelebrationPool;
  light: Light2D;
  time: number;
}

export abstract class Stage2DScene implements SceneModule {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 2 * HEIGHT);
  legend: LegendSpec | null = null;
  protected ctx: SceneContext | null = null;
  protected pool: CelebrationPool | null = null;
  protected light: Light2D | null = null;
  protected overlay: GroundTruthOverlay | null = null;
  protected debugOn = false;
  private readonly area = new THREE.Box2();
  private size: SceneSize = { width: 16, height: 9, dpr: 1 };

  abstract readonly post: ScenePost;
  abstract readonly controlHz: number;
  protected abstract readonly stageName: string;
  protected abstract extent(ctx: SceneContext): THREE.Box2;
  protected abstract ambient(ctx: SceneContext): THREE.Color;
  protected abstract build(ctx: SceneContext, stage: Stage2D): void | Promise<void>;
  protected abstract tick(frame: FrameUpdate, stage: Stage2D): void;

  protected overlayOptions(_ctx: SceneContext): GroundTruthOptions {
    return {};
  }

  protected poolOptions(_ctx: SceneContext): CelebrationPoolOptions {
    return {};
  }

  protected accentOf(ctx: SceneContext): number {
    return accentOf(ctx);
  }

  protected drop(): void {}

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.size = ctx.size;
    this.scene.name = `${this.stageName}Scene`;
    this.area.copy(this.extent(ctx));
    const centre = this.area.getCenter(new THREE.Vector2());
    this.camera.position.set(centre.x, HEIGHT, centre.y);
    this.camera.up.set(0, 0, -1);
    this.camera.lookAt(centre.x, 0, centre.y);
    this.fit();
    const pool = new CelebrationPool({ seed: ctx.seed, name: `${this.stageName}Celebrations`, ...this.poolOptions(ctx) });
    attachCelebrations(this.scene, pool);
    this.pool = pool;
    const overlay = new GroundTruthOverlay(ctx.geometry, { camera: this.camera, name: `${this.stageName}GroundTruth`, ...this.overlayOptions(ctx) });
    this.scene.add(overlay.group);
    overlay.visible = this.debugOn;
    this.overlay = overlay;
    const light = makeLight2D(ctx.renderer, this.scene, this.camera, this.ambient(ctx));
    light.resize(ctx.size);
    light.setEnabled(!this.debugOn);
    this.light = light;
    await this.build(ctx, { ctx, pool, light, time: ctx.time() });
    light.render();
  }

  update(frame: FrameUpdate): void {
    const { ctx, pool, light } = this;
    if (!ctx || !pool || !light) return;
    this.tick(frame, { ctx, pool, light, time: ctx.time() });
    if (this.debugOn && this.overlay) this.overlay.update(frame);
    light.render();
  }

  resize(size: SceneSize): void {
    const sized = size.width !== this.size.width || size.height !== this.size.height || size.dpr !== this.size.dpr;
    this.size = size;
    this.fit();
    if (sized) this.light?.resize(size);
  }

  resetCamera(): void {
    this.fit();
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
    this.light?.setEnabled(!on);
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.drop();
    this.light?.dispose();
    this.light = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.ctx = null;
  }

  private fit(): void {
    const aspect = this.size.width / this.size.height;
    const width = this.area.max.x - this.area.min.x, depth = this.area.max.y - this.area.min.y;
    const half = Math.max(depth, width / aspect) * (1 + MARGIN) / 2;
    this.camera.left = -half * aspect;
    this.camera.right = half * aspect;
    this.camera.top = half;
    this.camera.bottom = -half;
    this.camera.updateProjectionMatrix();
  }
}
