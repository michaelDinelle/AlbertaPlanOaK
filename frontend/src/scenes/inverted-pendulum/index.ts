import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { IP_CART_X, IP_POLE_LEN, IP_POLE_TH } from '@sim/envs/invertedpendulum';
import { buildCart, type CartHandle } from './actors';
import { buildWorld, RAIL_Y, type WorldHandle } from './world';

const CAM_BASE: readonly [number, number, number] = [0.5, 1.05, 4.1];
const CAM_LOOK: readonly [number, number, number] = [0, RAIL_Y + 0.3, 0];
const FLASH_SECONDS = 0.34;

class RailYardScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'railYard';

  private cart: CartHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const cart = buildCart({
      palette: ctx.tokens.palettes.noon,
      accent: this.accentOf(ctx),
    });
    this.cart = cart;
    this.scene.add(cart.group);
  }

  protected rigOptions(): OrbitRigOptions {
    const home = orbitFromPose(CAM_BASE, CAM_LOOK);
    return {
      target: CAM_LOOK,
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      near: 0.05,
      far: 420,
      rotateSpeed: 0.006,
      zoomSpeed: 0.0016,
      damping: 4,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        const x = state.pose[IP_CART_X] ?? 0;
        const th = state.pose[IP_POLE_TH] ?? 0;
        this.overlayPos[0] = x + IP_POLE_LEN * Math.sin(th);
        this.overlayPos[1] = RAIL_Y + 0.1 + IP_POLE_LEN * Math.cos(th);
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.cart?.setState(0, 0);
    world.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const cart = this.cart;
    if (!cart) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    cart.setState(state.pose[IP_CART_X] ?? 0, state.pose[IP_POLE_TH] ?? 0);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    cart.setGlow(flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    cart.setTime(time);
    rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.cart?.dispose();
    this.cart = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new RailYardScene()) satisfies SceneFactory);
