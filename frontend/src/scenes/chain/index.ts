import { PULSE_SECONDS } from '@kit/celebrate';
import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { blendTokens, shade } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { CHAIN_END, CHAIN_POS } from '@sim/envs/chain';
import {
  LANTERN_BASE_HEIGHT,
  LANTERN_BASE_RADIUS,
  buildBlossomTree,
  buildFordLantern,
  buildPetalFerry,
  type BlossomTreeHandle,
  type FordLanternHandle,
  type PetalFerryHandle,
} from './actors';
import {
  WATER_LEVEL,
  buildWorld,
  creekSkin,
  resolveGrid,
  stoneX,
  type WorldHandle,
} from './world';


const BURST_SCALE = 0.68;
const RIPPLE_SCALE = 0.6;
const PULSE_STRENGTH = 0.32;
const FLASH_SECONDS = 1.25;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.2;

const BLOOM_TAU = 2.35;
const GLOW_SPAN = 2.2;
const GLOW_TAU = 0.8;

export const LANTERN_Z = -2.35;
export const LANTERN_DX_NEAR = -0.45;
export const LANTERN_DX_FAR = -3.1;
export const LANTERN_DRY = 0.1;
export const LANTERN_LEVEL = LANTERN_BASE_HEIGHT * 0.6;
const LANTERN_WALK_COST = 0.1;

export function groundedLanternOffset(
  sampleHeight: (x: number, z: number) => number,
  rootX: number,
): [number, number, number] {
  let bestDx = LANTERN_DX_NEAR;
  let bestLow = -Infinity;
  let bestScore = Infinity;
  let bestDry = false;
  for (let dx = LANTERN_DX_NEAR; dx >= LANTERN_DX_FAR - 1e-6; dx -= 0.05) {
    const x = rootX + dx;
    let low = sampleHeight(x, LANTERN_Z);
    let high = low;
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2;
      const h = sampleHeight(
        x + Math.cos(a) * LANTERN_BASE_RADIUS,
        LANTERN_Z + Math.sin(a) * LANTERN_BASE_RADIUS,
      );
      if (h < low) low = h;
      if (h > high) high = h;
    }
    const dry = low >= WATER_LEVEL + LANTERN_DRY;
    if (dry && high - low <= LANTERN_LEVEL) return [dx, low, LANTERN_Z];
    const score = high - low + LANTERN_WALK_COST * -dx;
    if ((dry && !bestDry) || (dry === bestDry && score < bestScore)) {
      bestDry = dry;
      bestScore = score;
      bestLow = low;
      bestDx = dx;
    }
  }
  return [bestDx, bestLow, LANTERN_Z];
}

export function readPos(state: FrameState, grid: GridSize): number {
  const p = Math.round(state.pose[CHAIN_POS] ?? 0);
  return p < 0 ? 0 : p > grid.w - 1 ? grid.w - 1 : p;
}

class SixteenStonesScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'sixteenStones';
  private view: SceneView | null = null;

  private wisp: Wisp | null = null;
  private tree: BlossomTreeHandle | null = null;
  private lantern: FordLanternHandle | null = null;
  private ferry: PetalFerryHandle | null = null;
  private grid!: GridGeometry;

  private goalAt = -Infinity;
  private visitedAt = new Float32Array(0);
  private lastPos = -1;

  private readonly burstPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: CHAIN_END, y: 0 };
  private blossom = 0xf7a8b8;
  private gold = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.view = makeSceneView(ctx, { min: [stoneX(0, grid) - 3, -1, -3], max: [stoneX(grid.w - 1, grid) + 3, 4, 3] }, [1.6, 0.45, 0], { distance: 17.2, azimuth: 0.35, polar: 1.17 });
    this.visitedAt = new Float32Array(grid.w).fill(-Infinity);
    this.lastPos = -1;
    this.goalCell.x = grid.goal?.x ?? grid.w - 1;
    this.goalCell.y = grid.goal?.y ?? 0;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.afternoon;
    const skin = creekSkin(palette, ctx.tokens);
    const accent = this.accentOf(ctx);
    this.blossom = skin.blossom.getHex();
    this.gold = ctx.tokens.celebrationGold;
    const ember = blendTokens(ctx.tokens.celebrationGold, accent, 0.4).getHex();

    const treeLight = world.rig.points[0];
    if (treeLight) treeLight.color.set(ember);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const flashLight = world.rig.points[2];
    if (flashLight) flashLight.color.set(this.blossom);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'creekWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const tree = buildBlossomTree({
      palette,
      blossom: shade(blendTokens(this.blossom, palette.groundB, 0.14), 0.24).getHex(),
      bark: skin.bark.getHex(),
      moss: skin.moss.getHex(),
      ember,
      name: 'blossomTree',
    });
    this.tree = tree;
    cellToWorldVec(this.goalCell.x, this.goalCell.y, grid, tree.root.position);
    this.scene.add(tree.group);

    const lantern = buildFordLantern({
      palette,
      stone: skin.stoneDry.getHex(),
      moss: skin.moss.getHex(),
      offset: groundedLanternOffset((x, z) => world.terrain.sampleHeight(x, z), stoneX(-1, grid)),
      name: 'fordLantern',
    });
    this.lantern = lantern;
    cellToWorldVec(-1, 0, grid, lantern.root.position);
    this.scene.add(lantern.group);

    const ferry = buildPetalFerry({
      palette,
      blossom: shade(blendTokens(this.blossom, palette.sun, 0.12), 0.14).getHex(),
      from: [stoneX(this.goalCell.x, grid) + 0.24, tree.crownHeight * 0.72, 0],
      to: [stoneX(0, grid), WATER_LEVEL + 0.14, 0],
      name: 'petalFerry',
    });
    this.ferry = ferry;
    this.scene.add(ferry.mesh);
  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Scene view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.grid;
    const overlayCell: GridCell = { x: 0, y: 0 };
    const goalCell = this.goalCell;
    return {
      agentCell: (state) => {
        overlayCell.x = readPos(state, grid);
        overlayCell.y = 0;
        return overlayCell;
      },
      goalCell: () => goalCell,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    if (this.view && this.rig) world.setView(this.view.update(this.rig, 0), this.camera);
    world.caps.clearGlow();
    world.petalMotes.setDensity(0);
    this.wisp?.setCell(0, 0, this.grid);
    this.tree?.setState(0, 0);
    this.tree?.setTime(0);
    this.lantern?.setTime(0);
    this.ferry?.clear();
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const tree = this.tree;
    const lantern = this.lantern;
    const ferry = this.ferry;
    if (!wisp || !tree || !lantern || !ferry) return;

    const state = frame.state;
    const grid = this.grid;

    pool.update(frame.dtReal, time);

    const pos = readPos(state, grid);

    if (frame.scrubbed) {
      pool.clear();
      ferry.clear();
      this.goalAt = -Infinity;
      this.visitedAt.fill(-Infinity);
      world.caps.clearGlow();
      wisp.setCell(pos, 0, grid);
      wisp.snap();
      this.lastPos = pos;
      const log = frame.eventLog;
      let low = 0;
      let high = log.count;
      while (low < high) {
        const mid = (low + high) >>> 1;
        if ((log.steps[mid] ?? Infinity) <= state.step) low = mid + 1;
        else high = mid;
      }
      const seconds = state.step > 0 ? state.simTime / state.step : 0;
      for (let i = low - 1; i >= 0; i--) {
        const at = ((log.steps[i] ?? -1) + 1) * seconds;
        if (log.types[log.kinds[i] ?? 0] === 'goal') {
          this.goalAt = at;
          cellToWorldVec(this.goalCell.x, this.goalCell.y, grid, this.groundPos);
          this.burstPos.copy(this.groundPos);
          this.burstPos.x += 0.24;
          this.burstPos.y = tree.crownHeight * 0.78;
          this.groundPos.y = WATER_LEVEL;
          pool.burst(this.burstPos, this.blossom, BURST_SCALE, at);
          pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
          ferry.fire(at);
          pool.pulse.reset();
          pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - at) / PULSE_SECONDS));
          pool.update(0, time);
          break;
        }
      }
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        cellToWorldVec(this.goalCell.x, this.goalCell.y, grid, this.groundPos);
        this.burstPos.copy(this.groundPos);
        this.burstPos.x += 0.24;
        this.burstPos.y = tree.crownHeight * 0.78;
        this.groundPos.y = WATER_LEVEL;
        pool.burst(this.burstPos, this.blossom, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(0.72);
        this.goalAt = time;
        ferry.fire(time);
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(pos, 0, grid);
      wisp.snap();
    } else if (pos !== wisp.cellX) {
      wisp.hopTo(this.cellFor(pos), time);
    }
    if (pos !== this.lastPos) {
      if (pos >= 0 && pos < this.visitedAt.length) this.visitedAt[pos] = time;
      this.lastPos = pos;
    }
    wisp.update(frame.dtReal, time);

    for (let i = 0; i < this.visitedAt.length; i++) {
      world.caps.setGlow(i, stoneGlow(time - (this.visitedAt[i] ?? -Infinity)));
    }

    const bloom = treeBloom(time - this.goalAt);
    tree.setState(bloom, time);
    tree.setTime(time);
    lantern.setTime(time);
    ferry.setTime(time);
    world.petalMotes.setDensity(bloom * 0.85);

    const flash = flashEnvelope(time - this.goalAt);
    const treeLight = world.rig.points[0];
    if (treeLight) {
      treeLight.position.set(tree.root.position.x + 0.2, tree.crownHeight * 0.62, 0);
      treeLight.intensity = 0.08 + 0.62 * bloom;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.7 * flash;
    }
    const flashLight = world.rig.points[2];
    if (flashLight) {
      flashLight.position.copy(this.burstPos);
      flashLight.intensity = 1.5 * flash;
    }

    world.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.view && this.rig && this.world) {
      this.view.resize(size, this.rig);
      this.world.setView(this.view.update(this.rig, 0), this.camera);
    }
  }

  protected drop(): void {
    this.ferry?.dispose();
    this.ferry = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.tree?.dispose();
    this.tree = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.goalAt = -Infinity;
    this.visitedAt = new Float32Array(0);
    this.lastPos = -1;
  }

  private cellFor(pos: number): GridCell {
    this.agentCell.x = pos;
    this.agentCell.y = 0;
    return this.agentCell;
  }
}

export function treeBloom(age: number): number {
  if (!(age >= 0)) return 0;
  const v = Math.exp(-age / BLOOM_TAU);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function stoneGlow(age: number): number {
  if (!(age >= 0) || age > GLOW_SPAN) return 0;
  const v = Math.exp(-age / GLOW_TAU) * (1 - age / GLOW_SPAN);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function flashEnvelope(age: number): number {
  if (!(age >= 0) || age > FLASH_SECONDS) return 0;
  const v = Math.exp(-2.6 * age) * (1 - age / FLASH_SECONDS);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export default ((() => new SixteenStonesScene()) satisfies SceneFactory);
