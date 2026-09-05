import * as THREE from 'three';
import { PULSE_SECONDS } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import type { GridGeometry } from '@sim/types';
import { CHAIN_L, CHAIN_POS } from '@sim/envs/chain';
import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { buildCrossing, buildRover, chainX, deckHeight } from './actors';
import { buildWorld, type WorldHandle } from './world';

class RiverCrossingScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'riverCrossing';
  private view: SceneView | null = null;
  private rover: ReturnType<typeof buildRover> | null = null;
  private crossing: ReturnType<typeof buildCrossing> | null = null;
  private grid!: GridGeometry;
  private position = -1;
  private origin = 0;
  private movedAt = 0;
  private goalAt = -Infinity;
  private readonly goal = new THREE.Vector3();
  private readonly cell = { x: 0, y: 0 };

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    const grid = ctx.geometry.grid;
    if (!grid || grid.w !== CHAIN_L || grid.h !== 1 || !grid.goal) throw new Error('Chain grid is missing or invalid');
    this.grid = grid;
    const end = chainX(grid.goal.x, grid);
    this.goal.set(end, deckHeight(end) + 0.8, 0);
    this.view = makeSceneView(ctx, { min: [-11.8, -2, -2], max: [11.8, 3.2, 2] }, [0, 0.35, 0], { distance: 21, azimuth: 0.22, polar: 1.41 });
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(): void {
    this.rover = buildRover();
    this.crossing = buildCrossing(this.grid);
    this.scene.add(this.rover.group, this.crossing.group);
  }

  protected rigOptions(): OrbitRigOptions {
    return { ...this.view!.options, far: 5000, limits: { ...this.view!.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: state => { this.cell.x = Math.round(state.pose[CHAIN_POS]!); return this.cell; }, goalCell: () => this.grid.goal! };
  }

  protected start(): void {
    this.rover!.setPosition(chainX(0, this.grid), true);
    this.crossing!.setState(0, 0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const rover = this.rover!, position = Math.round(frame.state.pose[CHAIN_POS]!);
    let achievedAt: number | null = null;
    const teleport = frame.events.some(event => event.type === 'teleport');
    pool.update(frame.dtReal, time);
    if (frame.scrubbed || teleport || this.position < 0) {
      rover.setPosition(chainX(position, this.grid), true);
      this.position = position;
      this.origin = rover.position.x;
      this.movedAt = time;
      if (frame.scrubbed) {
        pool.clear(); this.goalAt = -Infinity;
        const log = frame.eventLog;
        let low = 0, high = log.count;
        while (low < high) {
          const mid = (low + high) >>> 1;
          if (log.steps[mid]! <= frame.state.step) low = mid + 1;
          else high = mid;
        }
        for (let i = low - 1; i >= 0; i--) if (log.types[log.kinds[i]!] === 'goal') {
          achievedAt = (log.steps[i]! + 1) * frame.state.simTime / frame.state.step;
          break;
        }
      }
    } else if (position !== this.position) {
      this.origin = rover.position.x;
      this.position = position;
      this.movedAt = time;
    }
    const blend = THREE.MathUtils.smoothstep(time - this.movedAt, 0, 0.11);
    rover.setPosition(THREE.MathUtils.lerp(this.origin, chainX(position, this.grid), blend), frame.scrubbed || teleport);
    for (const event of frame.events) if (event.type === 'goal') achievedAt = time;
    if (achievedAt !== null) {
      this.goalAt = achievedAt;
      pool.burst(this.goal, 0xb8ecd8, 0.4, achievedAt);
      pool.pulse.fire(Math.max(0, 0.16 - (time - achievedAt) / PULSE_SECONDS));
      pool.update(0, time);
    }
    this.crossing!.setState(position, Math.exp(-(time - this.goalAt) / 1.4));
    world.setTime(time);
    world.setView(this.view!.update(rig, frame.dtReal), this.camera);
  }

  protected drop(): void {
    this.rover?.dispose(); this.rover = null;
    this.crossing?.dispose(); this.crossing = null;
  }
}

export default ((() => new RiverCrossingScene()) satisfies SceneFactory);
