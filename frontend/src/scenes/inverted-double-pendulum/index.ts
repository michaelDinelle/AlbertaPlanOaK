import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { IDP_CART_X, IDP_HINGE1, IDP_HINGE2, IDP_LINK_LEN } from '@sim/envs/inverteddoublependulum';
import { buildDoubleCart, type DoubleCartHandle } from './actors';
import { buildWorld, RAIL_Y, type WorldHandle } from './world';

const CAM_BASE: readonly [number, number, number] = [0.6, 1.4, 4.9];
const CAM_LOOK: readonly [number, number, number] = [0, RAIL_Y + 0.55, 0];
const FLASH_SECONDS = 0.34;

class DoubleRailYardScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'doubleRailYard';

  private cart: DoubleCartHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const cart = buildDoubleCart({
      palette: ctx.tokens.palettes.afternoon,
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
        const x = state.pose[IDP_CART_X] ?? 0;
        const h1 = state.pose[IDP_HINGE1] ?? 0;
        const h2 = h1 + (state.pose[IDP_HINGE2] ?? 0);
        this.overlayPos[0] = x + IDP_LINK_LEN * Math.sin(h1) + IDP_LINK_LEN * Math.sin(h2);
        this.overlayPos[1] = RAIL_Y + 0.1 + IDP_LINK_LEN * Math.cos(h1) + IDP_LINK_LEN * Math.cos(h2);
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.cart?.setState(0, 0, 0);
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

    cart.setState(
      state.pose[IDP_CART_X] ?? 0,
      state.pose[IDP_HINGE1] ?? 0,
      state.pose[IDP_HINGE2] ?? 0,
    );

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

export default ((() => new DoubleRailYardScene()) satisfies SceneFactory);
