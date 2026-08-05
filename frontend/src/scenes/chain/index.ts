// Sixteen Stones — Chain-cont-v0. The palate cleanser of the OaK Preserve.
//
// THE ONE IDEA: the whole environment is ONE NUMBER — `pos`, an integer in 0..15 — so the
// whole scene is one line: sixteen stepping stones across an afternoon creek. There is no
// map to read, no key to carry, no failure to survive. The wisp hops on; the bare tree on
// the far stone flowers; the petals come back down the water. Everything else is air, reeds
// and light.
//
// WHAT THE ENV ACTUALLY DOES, AND WHAT THE ART OWES IT
//  * Reaching stone 15 pays the +1, sets `done` and WRAPS to stone 0 in the same step. The
//    record therefore never contains pos 15, and every `done` frame is simultaneously the
//    reward AND a teleport back to the start. sim-core reflects that exactly: the decoder
//    emits `goal`, `teleport` and `done` together, and the pose on that frame is already 0.
//  * So the celebration fires at STONE 15 — where the reward was earned, under the tree —
//    while the wisp is HARD-SNAPPED to stone 0 on the same frame. That is the one place
//    this scene's celebration position deliberately differs from four-rooms' "bloom where
//    the agent is": here the agent is no longer where it scored, and pretending otherwise
//    would bloom the wrong end of the creek. It is test-locked.
//  * `done` here is a REWARD, not a failure. This scene never sighs.
//  * The petal ferry is DECORATION and is built to be obviously so: it launches from the
//    tree AFTER the wisp has already been snapped home, and it trails behind. Easing the
//    avatar across sixteen stones it was never on would be a lie about a teleport, and
//    STYLE.md forbids it.
//
// THE FLOURISH CLOCKS. Three, all measured on `ctx.time()`, all cleared by a scrub:
// `goalAt` (the tree's bloom and the petal ferry), and `visitedAt[i]` (the faint warmth a
// stone keeps for a couple of seconds after being stepped on — the brief's one sanctioned
// piece of history, and the reason `clearGlow()` exists). Everything else in the frame is a
// pure function of the frame's own `pos`.

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, type GridSize, type Vec3Tuple } from '@kit/grid';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  AlignmentProbe,
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

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A low 3/4 from downstream, ~29° above the water. The angle is the whole
// composition: at 49° (the first pass) the creek became a plan-view sheet with no horizon in
// it at all, and a ford seen from above is a dotted line. At 29° the stones keep their
// silhouettes, the water keeps its surface, and the bend downstream closes the top of the
// frame with a far shore.
// The POLAR is the sky. Looking down at 29° the frame's top edge still points 9° below the
// horizontal, and the wooded spit downstream (crest ~6° below horizontal) filled the frame
// edge to edge — no sky anywhere. At polar 1.17 the pitch is 23°, the top edge points 3°
// below horizontal, and the ridge crest falls under it with a band of afternoon sky above.
// The azimuth matters twice over. At 0.55 the camera stood at x = 7.2 — level with the tree
// on stone 15, which loomed into the right third of the frame at four times its true size
// and lost half its crown off the edge. Swinging downstream to 0.35 puts the camera at
// x = 6.6 and the tree back in perspective with the ford it belongs to.
//
// THE SUBJECT IS NOT THE ROW OF STONES, IT IS THE ROW PLUS ITS TWO BOOKENDS (review fix).
// The first pass sized the frame from the stone CENTRES — 16 × cos(0.35) = 15.0 units — and
// then tested the composition by projecting one point, the crown's centre. Both were wrong
// in the same direction: the tree is 2 units of bough WIDER than the stone it stands on and
// the lantern is a unit off the other end, so the real subject spans ≈ 20 units, and at
// d = 14.6 the crown ran off the right edge at ndcX 1.13 (3 of 72 blossom clusters outside
// the frame, the far bough tip at 1.03 — visible in the review captures).
// The distance does the work and the target barely moves, because pushing the target toward
// the tree instead drags the near bank into the right third and the crown ends up drawn
// against reeds rather than water. Measured at the real 16:9 aspect over the whole crown
// BOUNDING BOX: d = 17.2, target x = 1.6 lands stones −0.65…+0.55, crown +0.37…+0.86 (whole,
// against open water), lantern −0.79…−0.67. Everything the scene owns is inside |ndcX| < 0.87.
// The framing test now projects the crown box and the lantern box, not points.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 1.17;
const CAM_DISTANCE = 17.2;
const CAM_TARGET: readonly [number, number, number] = [1.6, 0.45, 0];
const CAM_YAW_RANGE = 0.5;
/**
 * Thumb pulls back and stops moving, and its contract is the hard one: a SQUARE crop (the
 * wall view) must still hold all sixteen stones. A 1:1 crop of a 40° vertical FOV is 1.78×
 * narrower than 16:9, so the shipped 19.5 — composed and checked at 16:9 only — put stone 0
 * at ndcX −1.045, off the frame entirely, with the tree sliced at the other end.
 * The wall view is also the one frame that has to be composed for the SQUARE and not for the
 * strip, so it gets its own target. And the contract is the STONES, not their centres: at
 * d = 25 the centres fitted while stone 0's cap sat on the edge at −0.965. At d = 26.5 with
 * target x = 2.1 the square holds the whole cap row at −0.93…+0.63 and the whole crown at
 * +0.36…+0.86, and the 16:9 crop of the same rig is a wider, calmer version of it. The
 * lantern is a hero bookend and is allowed off the wall-view edge — pulling back far enough
 * to hold that too (d ≈ 32) shrinks the ford to a thread.
 */
const CAM_DISTANCE_THUMB = 26.5;
const CAM_TARGET_THUMB: readonly [number, number, number] = [2.1, 0.45, 0];
const CAM_AZIMUTH_THUMB = 0.32;
const CAM_POLAR_THUMB = 1.12;

// Celebration weights. This ford pays out every ~2.3 s of playback (the recorded lanes score
// ~5400 times in 100k steps), so the grammar has to survive being fired constantly: all
// three beats, none of them loud. At full kit strength the creek strobes.
const BURST_SCALE = 0.68;
const RIPPLE_SCALE = 0.6;
const BURST_PARTICLES = 120;
const PULSE_STRENGTH = 0.32;
/** the celebration light and burst decay over this many playback seconds */
const FLASH_SECONDS = 1.25;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
/** stones over water want a real hop, a little higher than a courtyard's */
const HOP_HEIGHT = 0.2;

/**
 * The tree's memory of the last +1. exp decay with a 2.35 s constant: at the recorded pace
 * (a goal every ~2.3 s) the tree lives around 0.4–1.0 and pulses with every crossing, and a
 * lane that stops scoring goes visibly bare inside ten seconds. It is a readout of the
 * agent, and it is derived from nothing but the event stream.
 */
const BLOOM_TAU = 2.35;
/** a stepped-on stone keeps its warmth this long */
const GLOW_SPAN = 2.2;
const GLOW_TAU = 0.8;

// ---------------------------------------------------------------------------
// the lantern's footing — derived from the terrain the scene itself built
// ---------------------------------------------------------------------------

/** the art's fixed z: off the crossing line, upstream, where it cannot read as a stone */
export const LANTERN_Z = -2.35;
/** the art walks outward from the crossing between these two x offsets, looking for bank */
export const LANTERN_DX_NEAR = -0.45;
export const LANTERN_DX_FAR = -3.1;
/** how far the base plate must clear the water before the bank counts as bank */
export const LANTERN_DRY = 0.1;
/**
 * A stone base BEDS into a bank; it does not bury itself in one. The plate rests on the
 * lowest ground it covers, so the ground under it may not vary by more than the plate's own
 * height or the whole base disappears into the slope — which is exactly what happens on the
 * creek's ramp (a full unit of rise across one footprint).
 */
export const LANTERN_LEVEL = LANTERN_BASE_HEIGHT * 0.6;
/** walking further from the ford costs this much "levelness" per world unit (the tiebreak) */
const LANTERN_WALK_COST = 0.1;

/**
 * Where the lantern's ART stands, given the scene's own height field. The probe ROOT is and
 * stays `cellToWorld(-1, 0)` at y = 0 — this offset moves only the child that carries the art.
 *
 * The bank edge MEANDERs ±0.55 with the lane seed, so a typed offset cannot be on it: at the
 * shipped (-0.45, 0, -2.35) the lantern stood 0.55 above the bed and 0.17 above the WATER at
 * seed 64, and over open water in 11 of 14 sampled lanes. So the art walks outward, looking
 * for the first footing that is both DRY and LEVEL: dry because a bank lantern is not a
 * stepping stone, level because the base plate sits on the LOWEST ground it covers (nothing
 * may float) and a 1-unit-per-footprint ramp would swallow it. If the lane's bank never
 * offers a level shelf, the least-sloped dry footing wins, with a small penalty for walking
 * away from the ford — the lantern is a bookend, and a bookend at the edge of the world is
 * not one.
 */
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
    // the first DRY AND LEVEL footing wins outright: it is the closest one to the ford
    if (dry && high - low <= LANTERN_LEVEL) return [dx, low, LANTERN_Z];
    // otherwise keep the best fallback: dry beats wet, then flattest, then nearest
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

/**
 * The agent's stone for a frame. sim-core's decoded state is the source; the raw pose
 * (`[pos, 15, …]`) is the fallback so the scene still stands up under the harness's fixture
 * loop, which carries no `derived`.
 */
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

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private goalAt = -Infinity;
  private visitedAt = new Float32Array(0);
  private lastPos = -1;

  // ---- preallocated scratch: update() may not allocate ----
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

    // This scene claims all three of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const treeLight = world.rig.points[0];
    if (treeLight) treeLight.color.set(ember);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const flashLight = world.rig.points[2];
    if (flashLight) flashLight.color.set(this.blossom);

    // ---- the shared avatar ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a shade over the 0.16 default: from a 13.6-unit standoff, over water that carries
        // its own specular sparkle, the default orb loses the eye
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'creekWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // ---- the blossom tree, on the paying stone the agent is never observed at ----
    const tree = buildBlossomTree({
      tier: ctx.tier,
      palette,
      // a DEEP rose, not the pale accent: under a 2.7 key with a rim light of its own, the
      // raw token pink renders as white paper
      blossom: shade(blendTokens(this.blossom, palette.groundB, 0.14), 0.24).getHex(),
      bark: skin.bark.getHex(),
      moss: skin.moss.getHex(),
      ember,
      name: 'blossomTree',
    });
    this.tree = tree;
    cellToWorldVec(this.goalCell.x, this.goalCell.y, grid, tree.root.position);
    this.scene.add(tree.group);

    // ---- the ford lantern: the composition's other bookend, one cell upstream ----
    // its ART is planted on the bank the world just built; its ROOT is the derived cell
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

    // ---- the petal ferry: tree crown -> the start stone ----
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

    // ---- camera ----
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

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'creekCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    const overlayCell: GridCell = { x: 0, y: 0 };
    const goalCell = this.goalCell;
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        overlayCell.x = readPos(state, grid);
        overlayCell.y = 0;
        return overlayCell;
      },
      // the paying cell never appears in the stream, but it is still where the +1 lives
      goalCell: () => goalCell,
      name: 'creekGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
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

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth: one integer
    const pos = readPos(state, grid);

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity
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

    // 4. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // The +1 was earned at the PAYING STONE, under the tree — and the pose on this very
        // frame has already wrapped to stone 0. Read the celebration off the goal cell, not
        // off the agent: the agent is standing at the other end of the creek by now.
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
      // `done` in this env IS the reward (reaching the end wraps the stream). Never sigh.
    }

    // 5. the avatar. A teleport is a HARD CUT — the wrap is not a sixteen-stone glide.
    if (teleported) {
      wisp.setCell(pos, 0, grid);
      wisp.snap();
      rig.snap();
    } else if (pos !== wisp.cellX) {
      wisp.hopTo(this.cellFor(pos), time);
    }
    if (pos !== this.lastPos) {
      // the stone the wisp has just arrived on takes its warmth
      if (pos >= 0 && pos < this.visitedAt.length) this.visitedAt[pos] = time;
      this.lastPos = pos;
    }
    wisp.update(frame.dtReal, time);

    // 6. the stones' fading warmth — a decay clock, cleared by every scrub
    for (let i = 0; i < this.visitedAt.length; i++) {
      world.caps.setGlow(i, stoneGlow(time - (this.visitedAt[i] ?? -Infinity)));
    }

    // 7. the tree: bloom is the last +1 and nothing else
    const bloom = treeBloom(time - this.goalAt);
    tree.setState(bloom, time);
    tree.setTime(time);
    lantern.setTime(time);
    ferry.setTime(time);
    world.petalMotes.setDensity(bloom * 0.85);

    // 8. the three point lights this scene claims from the rig
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

    // 9. one clock for the whole creek
    world.setTime(time);

    // 10. camera easing runs in WALL time so a paused frame is still explorable
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

  probes(): AlignmentProbe[] {
    const grid = this.grid;
    const wisp = this.wisp;
    const tree = this.tree;
    const lantern = this.lantern;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;

    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const treeOut: Vec3Tuple = [0, 0, 0];
    const lanternOut: Vec3Tuple = [0, 0, 0];
    const goal = this.goalCell;

    // THE PROBE RULE FOR EVERY GRID SCENE: `assertUpdateIdempotent` compares probe objects'
    // whole WORLD MATRIX, not just their translation, so a probe may carry no eased value
    // of any kind. This creek's wisp is a radially symmetric spirit and never calls
    // `face()`, so BOTH of its anchors stay exact and probeable; every bob, breath and
    // ribbon lives on `decor`. The tree's lean lives on `crown`, the lantern's placement
    // offset on `body` — the two roots carry position only, forever.
    return [
      {
        // the orb centre: exactly `wispHeight` over the true stone
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          cellToWorld(readPos(state, grid), 0, grid, wispOut);
          wispOut[1] = height;
          return wispOut;
        },
      },
      {
        // and the avatar node under it: the stone's top face, y = 0, position only
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => cellToWorld(readPos(state, grid), 0, grid, anchorOut),
      },
      {
        // the blossom tree stands on the PAYING stone — the cell the record never shows
        name: 'blossom-tree',
        object: tree?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(goal.x, goal.y, grid, treeOut),
      },
      {
        // the ford lantern is planted one cell UPSTREAM of the first stone, derived from
        // the same conversion as everything else
        name: 'ford-lantern',
        object: lantern?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(-1, 0, grid, lanternOut),
      },
    ];
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

  /** allocation-free cell handle for `hopTo` */
  private cellFor(pos: number): GridCell {
    this.agentCell.x = pos;
    this.agentCell.y = 0;
    return this.agentCell;
  }
}

// ---------------------------------------------------------------------------
// envelopes — every one of them measured on the playback clock
// ---------------------------------------------------------------------------

/**
 * How far into flower the tree stands. A never-fired clock (`-Infinity`) is 0, which is what
 * makes a scrub land on a genuinely bare tree instead of replaying a bloom it never saw.
 * Pure exponential: no window, because a lane that is scoring keeps the tree in flower and a
 * lane that has stopped lets it fall bare, and both of those are true things about the run.
 */
export function treeBloom(age: number): number {
  if (!(age >= 0)) return 0;
  const v = Math.exp(-age / BLOOM_TAU);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * The warmth a stone keeps after the wisp steps off it. The linear tail is not cosmetic: a
 * bare exp() still has real amplitude when the window closes, so every stone in the ford
 * would end its glow on a one-frame pop.
 */
export function stoneGlow(age: number): number {
  if (!(age >= 0) || age > GLOW_SPAN) return 0;
  const v = Math.exp(-age / GLOW_TAU) * (1 - age / GLOW_SPAN);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** The celebration flash: the two point lights that answer a +1, and nothing else. */
export function flashEnvelope(age: number): number {
  if (!(age >= 0) || age > FLASH_SECONDS) return 0;
  const v = Math.exp(-2.6 * age) * (1 - age / FLASH_SECONDS);
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

const createScene: SceneFactory = () => new SixteenStonesScene();
export default createScene;
