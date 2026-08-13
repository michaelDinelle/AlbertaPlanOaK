
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { CHAIN_END } from '@sim/envs/chain';
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
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  WATER_LEVEL,
  blendTokens,
  buildWorld,
  creekSkin,
  resolveGrid,
  shade,
  stoneX,
  type WorldHandle,
} from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 1.17;
const CAM_DISTANCE = 17.2;
const CAM_TARGET: readonly [number, number, number] = [1.6, 0.45, 0];
const CAM_YAW_RANGE = 0.5;
const CAM_DISTANCE_THUMB = 26.5;
const CAM_TARGET_THUMB: readonly [number, number, number] = [2.1, 0.45, 0];
const CAM_AZIMUTH_THUMB = 0.32;
const CAM_POLAR_THUMB = 1.12;

const BURST_SCALE = 0.68;
const RIPPLE_SCALE = 0.6;
const BURST_PARTICLES = 120;
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

interface ChainLike {
  pos?: unknown;
  len?: unknown;
}

export function readPos(state: FrameState, grid: GridSize): number {
  const d = state.derived as ChainLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const raw = has && typeof d.pos === 'number' ? d.pos : (state.pose[0] ?? 0);
  const p = Number.isFinite(raw) ? Math.round(raw) : 0;
  return p < 0 ? 0 : p > grid.w - 1 ? grid.w - 1 : p;
}

class SixteenStonesScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private tree: BlossomTreeHandle | null = null;
  private lantern: FordLanternHandle | null = null;
  private ferry: PetalFerryHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

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

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'sixteenStonesScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.visitedAt = new Float32Array(grid.w).fill(-Infinity);
    this.lastPos = -1;
    this.goalCell.x = grid.goal?.x ?? grid.w - 1;
    this.goalCell.y = grid.goal?.y ?? 0;

    const palette = ctx.tokens.palettes.afternoon;
    const skin = creekSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.blossom = skin.blossom.getHex();
    this.gold = ctx.tokens.celebrationGold;
    const ember = blendTokens(ctx.tokens.celebrationGold, accent, 0.4).getHex();

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    const treeLight = world.rig.points[0];
    if (treeLight) treeLight.color.set(ember);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const flashLight = world.rig.points[2];
    if (flashLight) flashLight.color.set(this.blossom);

    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'creekWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const tree = buildBlossomTree({
      tier: ctx.tier,
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
      tier: ctx.tier,
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
      tier: ctx.tier,
      palette,
      blossom: shade(blendTokens(this.blossom, palette.sun, 0.12), 0.14).getHex(),
      from: [stoneX(this.goalCell.x, grid) + 0.24, tree.crownHeight * 0.72, 0],
      to: [stoneX(0, grid), WATER_LEVEL + 0.14, 0],
      name: 'petalFerry',
    });
    this.ferry = ferry;
    this.scene.add(ferry.mesh);

    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: thumb
        ? [CAM_TARGET_THUMB[0], CAM_TARGET_THUMB[1], CAM_TARGET_THUMB[2]]
        : [CAM_TARGET[0], CAM_TARGET[1], CAM_TARGET[2]],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'creekOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'creekCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayCell: GridCell = { x: 0, y: 0 };
    const goalCell = this.goalCell;
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        overlayCell.x = readPos(state, grid);
        overlayCell.y = 0;
        return overlayCell;
      },
      goalCell: () => goalCell,
      name: 'creekGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    world.caps.clearGlow();
    world.petalMotes.setDensity(0);
    wisp.setCell(0, 0, grid);
    tree.setState(0, 0);
    tree.setTime(0);
    lantern.setTime(0);
    ferry.clear();
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const tree = this.tree;
    const lantern = this.lantern;
    const ferry = this.ferry;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !tree || !lantern || !ferry || !pool || !rig) return;

    const time = ctx.time();
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
      rig.snap();
      this.lastPos = pos;
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
      rig.snap();
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

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.ferry?.dispose();
    this.ferry = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.tree?.dispose();
    this.tree = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
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

const createScene: SceneFactory = () => new SixteenStonesScene();
export default createScene;
