import type { FrameUpdate, SceneContext, SceneFactory, SceneModule, SceneSize } from '@kit/types';
import { parameter } from '../../data/envs';
import { IvyGateScene } from './garden/index';
import { TempleOfThreeGatesScene } from './temple/index';

const LOOKS: Record<number, () => SceneModule> = {
  1: () => new IvyGateScene(),
  3: () => new TempleOfThreeGatesScene(),
};

class KeyCorridorScene implements SceneModule {
  private look: SceneModule | null = null;

  private get chosen(): SceneModule {
    if (!this.look) throw new Error('Key Corridor has not chosen its look');
    return this.look;
  }

  get scene() { return this.chosen.scene; }
  get camera() { return this.chosen.camera; }
  get post() { return this.chosen.post; }
  get controlHz() { return this.chosen.controlHz; }
  get legend() { return this.look?.legend ?? null; }

  async init(ctx: SceneContext): Promise<void> {
    const doors = parameter(ctx.environment, 'doors'), make = LOOKS[doors];
    if (!make) throw new Error(`${ctx.environment.envId} has no look for ${doors} doors`);
    this.look = make();
    await this.look.init(ctx);
  }

  update(frame: FrameUpdate): void { this.chosen.update(frame); }
  resize(size: SceneSize): void { this.chosen.resize(size); }
  resetCamera(): void { this.chosen.resetCamera(); }
  setDebug(on: boolean): void { this.chosen.setDebug(on); }

  dispose(): void {
    this.look?.dispose();
    this.look = null;
  }
}

export default ((() => new KeyCorridorScene()) satisfies SceneFactory);
