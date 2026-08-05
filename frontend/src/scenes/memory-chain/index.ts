// The Firefly Cave — MemoryChain-cont-v0.
//
// THE ONE IDEA: THE CUE IS THE SCENE, AND THE CUE IS ONLY WHERE THE AGENT CAN SEE IT.
// bsuite's Memory Chain shows the agent a single bit at position 0 and then hides it for
// thirty-one steps, at the end of which the agent's action IS its answer. So this cave is
// built as one long honest sentence: a crystal that burns cyan or magenta while — and only
// while — the wisp stands on cell 0; thirty-one stepping stones of darkness; and two dim
// mouths at the far end, one of each colour, lit exactly equally, forever, because they
// are the CHOICE and not the answer.
//
// THE HONESTY RULE, stated as code. `state.cue` is read every frame (it is right there in
// the pose), but the ONLY consumers are:
//   * `crystal.setGlow(pos === 0 ? … : 0)` — a hard gate, not a fade. Step off cell 0 and
//     the emitters are hidden, their intensity is zero and the crystal's point light is
//     zero in the same frame. Nothing eases, nothing lingers, nothing is tinted: not the
//     fireflies, not the wisp, not the water, not the stone, not the mouths.
//   * the celebration, whose colour comes from `readChoice(state)` — the action the agent
//     actually took. On a `goal` frame that bit IS the cue that was in force (the decoder
//     fires `goal` exactly when `action & 1 === prev.cue`), so the reveal is the reward
//     revealing itself, which is the one moment colour is allowed to flood the cave.
// `cueLeaks()` in the test file walks the live scene graph at EVERY tier and fails on any
// cue-coloured channel that reaches the raster while `pos > 0` and does not belong to a
// mouth: emissive mesh colours, point lights, particle uniforms, and — the family the first
// instrument was blind to — a lit surface's fresnel rim, albedo and self-emission.
//
// WHERE THE CELEBRATION HAPPENS. FourRooms' canon lesson 18 says the `goal` bloom belongs
// where the AGENT is, because there the goal respawns under the agent's feet. Here the
// agent TELEPORTS: by the time the +1 is reported it is already back on cell 0. The
// principle behind the lesson — bloom where the reward was earned — puts the burst and the
// ripple on cell 31, and the swarm then sweeps the whole passage back to the wisp. Both
// ends of the sentence light up, in order, in the winning colour.

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
import {
  buildCueCrystal,
  buildExitMouth,
  type CueCrystalHandle,
  type ExitMouthHandle,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  CHOICE_CELL,
  CUE_CELL,
  buildWorld,
  clamp01,
  readChoice,
  readCue,
  readPos,
  resolveGrid,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. The passage is 32 units long and 1 wide — the hardest shape in the whole
// preserve to compose. The camera stands on the near bank, behind and above the crystal
// end, and looks DOWN the passage: the cue is a big near object, the choice is a pair of
// pinpricks at the vanishing point, and the thirty-one stones between them are the
// distance the agent has to carry a single bit.
// The numbers are not taste. From (−23, 8.5, 14) looking at (−4, 1, 0), cell 0 lands at
// NDC (−0.65, −0.64), cell 16 at (0.15, 0.01) and cell 31 at (0.45, 0.26) on a 16:9 40°
// frame: the passage sweeps corner to centre with a 2.5:1 depth ratio, which is as flat as
// a 32-unit corridor can be made without pulling so far back that the crystal stops
// reading. Pulled in to 18 units the far third fell off the top of the frame.
const CAM_TARGET: readonly [number, number, number] = [-4, 1.0, 0];
const CAM_DISTANCE = 24.8;
const CAM_POLAR = 1.263;
const CAM_YAW_RANGE = 0.55;
/** thumb pulls back and stops moving: a square-ish crop must still hold all 32 stones */
const CAM_TARGET_THUMB: readonly [number, number, number] = [-2.6, 1.3, 0];
const CAM_DISTANCE_THUMB = 29.5;
const CAM_AZIMUTH_THUMB = -1.02;
const CAM_POLAR_THUMB = 1.21;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

// Celebration weights. The cave is dark and the celebration is the only colour it ever
// gets, so this one runs closer to full strength than the noon courtyard's — but the burst
// is 40 units from the camera, so its job is to mark the spot while the swarm does the
// travelling.
const BURST_SCALE = 0.72;
const RIPPLE_SCALE = 0.62;
const BURST_PARTICLES = 130;
// screen-level bloom pulse: goals land every 32 steps, so at boosted playback
// this fires constantly — it must read as a chime, not a strobe
const PULSE_STRENGTH = 0.3;
const SIGH_STRENGTH = 0.85;

/** the burst blooms at the mouths' throat height, not on the wet stone */
const BLOOM_Y = 1.25;

/** how long the winning mouth flares */
const FLARE_SECONDS = 1.35;
/** how long both mouths stay doused after a wrong choice */
const DOUSE_SECONDS = 1.15;
/** the crystal's re-kindle when the cue is redrawn under the wisp */
const KINDLE_SECONDS = 0.8;
/** the swarm's run from the choice cell back to the crystal */
const FLOOD_SECONDS = 1.9;

/** resting glow of each mouth — equal, always */
const MOUTH_REST = 1;
const MOUTH_FLARE = 3.4;

class FireflyCaveScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private crystal: CueCrystalHandle | null = null;
  private mouthCyan: ExitMouthHandle | null = null;
  private mouthMagenta: ExitMouthHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private kindleAt = -Infinity;
  private flareAt = -Infinity;
  private douseAt = -Infinity;
  private floodAt = -Infinity;
  /** which mouth won the last resolved choice (0 = cyan, 1 = magenta) */
  private winner: 0 | 1 = 0;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly cell: GridCell = { x: 0, y: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: CHOICE_CELL, y: 0 };
  private readonly choicePos = new THREE.Vector3();
  private readonly bloomPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly cueColors: [THREE.Color, THREE.Color] = [
    new THREE.Color(TOKENS.cueCyan),
    new THREE.Color(TOKENS.cueMagenta),
  ];
  private cueX = 0;
  private choiceX = 0;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'fireflyCaveScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.cueColors[0].set(ctx.tokens.cueCyan);
    this.cueColors[1].set(ctx.tokens.cueMagenta);

    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    const anchor: Vec3Tuple = [0, 0, 0];
    this.cueX = cellToWorld(CUE_CELL, 0, grid, anchor)[0];
    this.choiceX = cellToWorld(CHOICE_CELL, 0, grid, anchor)[0];
    this.choicePos.set(this.choiceX, 0, 0);

    // ---- the shared avatar: the one warm-green thing in a cold cave ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a little over the 0.16 default: at a 25-unit standoff through cave fog the
        // default orb is a dozen pixels, and the wisp is the only object whose exact
        // position the viewer is supposed to be able to read
        radius: 0.21,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'caveWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // ---- the cue crystal on cell 0 ----
    const crystal = buildCueCrystal({
      tier: ctx.tier,
      skin: world.skin,
      cueColors: [ctx.tokens.cueCyan, ctx.tokens.cueMagenta],
      name: 'cueCrystal',
    });
    this.crystal = crystal;
    this.scene.add(crystal.group);
    cellToWorldVec(CUE_CELL, 0, grid, crystal.root.position);

    // ---- the two mouths on cell 31 ----
    // Cyan is the cue the decoder calls 0 ("left") and magenta the one it calls 1
    // ("right"); travelling +x with +y up, left is −z. So the mouths stand where their
    // own names say they do.
    const mouthCyan = buildExitMouth({
      tier: ctx.tier,
      skin: world.skin,
      color: ctx.tokens.cueCyan,
      side: -1,
      name: 'exitMouthCyan',
    });
    const mouthMagenta = buildExitMouth({
      tier: ctx.tier,
      skin: world.skin,
      color: ctx.tokens.cueMagenta,
      side: 1,
      name: 'exitMouthMagenta',
    });
    this.mouthCyan = mouthCyan;
    this.mouthMagenta = mouthMagenta;
    this.scene.add(mouthCyan.group, mouthMagenta.group);
    cellToWorldVec(CHOICE_CELL, 0, grid, mouthCyan.root.position);
    cellToWorldVec(CHOICE_CELL, 0, grid, mouthMagenta.root.position);

    // ---- the four point lights this scene claims from the rig ----
    // Their positions and intensities are state-driven; only the crystal's colour ever
    // changes, and it is dead (intensity 0) whenever cell 0 is not the agent's own cell.
    const wispLight = world.rig.points[0];
    if (wispLight) {
      wispLight.color.set(accent);
      wispLight.intensity = 0;
    }
    const crystalLight = world.rig.points[1];
    if (crystalLight) {
      crystalLight.color.copy(this.cueColors[0]);
      crystalLight.intensity = 0;
      crystalLight.position.set(this.cueX, crystal.coreHeight, crystal.offsetZ);
    }
    const choiceLight = world.rig.points[2];
    if (choiceLight) {
      // the neutral blend of BOTH mouths: a light that favoured either one would be a leak
      choiceLight.color.copy(this.cueColors[0]).lerp(this.cueColors[1], 0.5);
      choiceLight.intensity = 0;
      choiceLight.position.set(this.choiceX + 0.4, mouthCyan.glowHeight, 0);
    }
    const floodLight = world.rig.points[3];
    if (floodLight) {
      floodLight.color.copy(this.cueColors[0]);
      floodLight.intensity = 0;
      floodLight.position.set(this.choiceX, 1.2, 0);
    }

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: thumb ? CAM_TARGET_THUMB : CAM_TARGET,
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAMERA_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0014,
      name: 'caveOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      // the sigh curtain falls over the choice end, which is where the choice was lost
      sighRadius: 3.4,
      sighHeight: 4,
      name: 'caveCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    const overlayCell = this.overlayCell;
    const goalCell = this.goalCell;
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        overlayCell.x = readPos(state);
        overlayCell.y = 0;
        return overlayCell;
      },
      goalCell: () => goalCell,
      name: 'caveGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(CUE_CELL, 0, grid);
    crystal.setCue(0);
    crystal.setGlow(1);
    crystal.setTime(0);
    mouthCyan.setGlow(MOUTH_REST);
    mouthMagenta.setGlow(MOUTH_REST);
    mouthCyan.setTime(0);
    mouthMagenta.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const crystal = this.crystal;
    const cyan = this.mouthCyan;
    const magenta = this.mouthMagenta;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !crystal || !cyan || !magenta || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth: two integers, and that is the whole env
    const pos = readPos(state);
    const cue = readCue(state);
    this.cell.x = pos;
    this.cell.y = 0;

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.kindleAt = -Infinity;
      this.flareAt = -Infinity;
      this.douseAt = -Infinity;
      this.floodAt = -Infinity;
      wisp.setCell(pos, 0, grid);
      wisp.snap();
      rig.snap();
    }

    // 4. events — celebrations are NEVER inferred from the state stream.
    //    The decoder emits them in the order [goal?, cue, teleport, done], so a single
    //    forward pass with a `scored` flag distinguishes "done because it was right" from
    //    "done because it was wrong" without looking at the state at all.
    //    `scored` is PER CHOICE, not per buffer. `kit/harness` drains up to
    //    MAX_CATCHUP_FRAMES frames into one `frame.events` (4× speed, a stalled frame, a
    //    backgrounded tab), so the array can be [goal,cue,teleport,done, cue,teleport,done]
    //    — a win and a loss arriving together. A buffer-wide flag swallowed the second,
    //    unrewarded `done` and the loss rendered as a win with no sigh at all; clearing the
    //    flag on each `done` makes every choice in the burst resolve on its own evidence.
    let teleported = false;
    let scored = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        scored = true;
        // the action the agent took IS the cue that was in force — see readChoice()
        this.winner = readChoice(state);
        const color = this.cueColors[this.winner];
        this.flareAt = time;
        this.floodAt = time;
        world.flood.setColor(color);
        this.bloomPos.set(this.choiceX, BLOOM_Y, 0);
        pool.burst(this.bloomPos, color, BURST_SCALE, time);
        pool.ripple(this.choicePos, color, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(0.6);
        const floodLight = world.rig.points[3];
        if (floodLight) floodLight.color.copy(color);
      } else if (ev.type === 'cue') {
        // the respawn redraws the cue; the crystal's COLOUR is a pure read of the state,
        // so all this event does is strike the match
        this.kindleAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      } else if (ev.type === 'done') {
        if (!scored) {
          // a wrong choice: both mouths go out together and the cave sighs
          this.douseAt = time;
          this.bloomPos.set(this.choiceX, BLOOM_Y, 0);
          pool.sigh(SIGH_STRENGTH, this.bloomPos, time);
        }
        // this choice is resolved; whatever follows in the same buffer is a NEW one
        scored = false;
      }
    }

    // 5. the avatar: one straight hop per cell change, never a path plan. The respawn is a
    //    31-cell jump, so `teleport` is a hard cut — and the magnitude guard catches the
    //    same jump even if the event were ever lost, because a wisp gliding the length of
    //    the passage would be a lie about where the agent is.
    if (teleported || Math.abs(pos - wisp.cellX) > 1) {
      wisp.setCell(pos, 0, grid);
      wisp.snap();
    } else if (pos !== wisp.cellX) {
      wisp.hopTo(this.cell, time);
    }
    wisp.update(frame.dtReal, time);

    // 6. THE CUE. Colour from the state, existence from the position. Nothing else.
    const kindle = envelope(time - this.kindleAt, KINDLE_SECONDS, 3.4);
    const atCue = pos === CUE_CELL;
    crystal.setCue(cue);
    // the memoranda beat: while the agent READS the bit the crystal is the
    // brightest thing in the cave, with a slow breath that says "this, remember
    // this" — the honesty gate (zero off cell 0) is unchanged
    crystal.setGlow(atCue ? 1.3 + 0.45 * kindle + 0.18 * Math.sin(time * 2.1) : 0);
    crystal.setTime(time);

    // 7. the two mouths: equal at rest, doused together on a loss, and only ever
    //    asymmetric during the flare that resolves a win
    const flare = envelope(time - this.flareAt, FLARE_SECONDS, 2.3);
    const douse = envelope(time - this.douseAt, DOUSE_SECONDS, 1.6);
    // at the choice cell the QUESTION is being asked: both mouths breathe up
    // together — equal by construction, so no cue information leaks
    const asking = pos === CHOICE_CELL ? 0.8 : 0;
    const rest = MOUTH_REST * (1 + asking) * (1 - 0.94 * douse);
    cyan.setGlow(rest + (this.winner === 0 ? MOUTH_FLARE * flare : 0));
    magenta.setGlow(rest + (this.winner === 1 ? MOUTH_FLARE * flare : 0));
    cyan.setTime(time);
    magenta.setTime(time);

    // 8. the swarm: the one moment colour floods the cave, running the whole passage from
    //    the mouth that won back to the crystal the bit came from.
    //    `running` and `progress` are deliberately two functions: the swarm is at
    //    progress 0 on the frame it is FIRED, and "progress > 0" would have parked it for
    //    exactly that frame — the one frame it is meant to erupt.
    const floodAge = time - this.floodAt;
    const flooding = floodRunning(floodAge);
    const flood = floodProgress(floodAge);
    if (flooding) {
      const eased = 1 - (1 - flood) * (1 - flood);
      world.flood.points.position.x = this.choiceX + (this.cueX - this.choiceX) * eased;
      world.flood.setDensity(1);
      world.flood.setIntensity(2.4 * (1 - flood * flood));
    } else {
      world.flood.points.position.x = this.choiceX;
      world.flood.setDensity(0);
    }

    // 9. the four point lights
    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += ctx.tokens.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 2.4 + 2.6 * flare;
    }
    const crystalLight = world.rig.points[1];
    if (crystalLight) {
      crystalLight.color.copy(this.cueColors[cue]);
      // THE honesty line: zero the instant the agent is not standing on cell 0
      crystalLight.intensity = atCue ? 4.2 + 2.2 * kindle : 0;
    }
    const choiceLight = world.rig.points[2];
    if (choiceLight) {
      choiceLight.intensity = 2.4 * (1 - 0.94 * douse) + 3.5 * flare;
    }
    const floodLight = world.rig.points[3];
    if (floodLight) {
      floodLight.position.x = world.flood.points.position.x;
      floodLight.intensity = flooding ? 3.4 * (1 - flood * flood) : 0;
    }

    // 10. one clock for the whole cave
    world.setTime(time);

    // 11. camera easing runs in WALL time so a paused frame is still explorable
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
    const grid: GridSize = this.grid;
    const wisp = this.wisp;
    const crystal = this.crystal;
    const cyan = this.mouthCyan;
    const magenta = this.mouthMagenta;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;

    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const crystalOut: Vec3Tuple = [0, 0, 0];
    const cyanOut: Vec3Tuple = [0, 0, 0];
    const magentaOut: Vec3Tuple = [0, 0, 0];

    // This wisp never calls `face()` (a corridor spirit has a ribbon to show which way it
    // is going and nothing to turn toward), so canon lesson 16 does not bite and BOTH
    // anchors stay bit-exact: `core` is the orb centre at exactly `wispHeight`, `avatar`
    // is the cell itself at y = 0.
    return [
      {
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          cellToWorld(readPos(state), 0, grid, wispOut);
          wispOut[1] = height;
          return wispOut;
        },
      },
      {
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => cellToWorld(readPos(state), 0, grid, anchorOut),
      },
      {
        // the cue crystal owns the env's own start cell
        name: 'cue-crystal',
        object: crystal?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(CUE_CELL, 0, grid, crystalOut),
      },
      {
        // both mouths stand on the env's own choice cell; the splay is decoration
        name: 'exit-cyan',
        object: cyan?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(CHOICE_CELL, 0, grid, cyanOut),
      },
      {
        name: 'exit-magenta',
        object: magenta?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(CHOICE_CELL, 0, grid, magentaOut),
      },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.mouthCyan?.dispose();
    this.mouthCyan = null;
    this.mouthMagenta?.dispose();
    this.mouthMagenta = null;
    this.crystal?.dispose();
    this.crystal = null;
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
    this.kindleAt = -Infinity;
    this.flareAt = -Infinity;
    this.douseAt = -Infinity;
    this.floodAt = -Infinity;
  }
}

// ---------------------------------------------------------------------------
// envelopes — every one of them measured on the playback clock
// ---------------------------------------------------------------------------

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has real amplitude when the window
 * closes, so every flourish would end on a one-frame pop (canon lesson 6).
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

/**
 * Whether the swarm is in the air. A never-fired clock (`-Infinity`), a NaN and any age
 * past the window are all false, which is what makes a scrub land on a cave with no swarm
 * in it instead of replaying a flood it never saw.
 */
export function floodRunning(age: number): boolean {
  return age >= 0 && age < FLOOD_SECONDS;
}

/** How far along that run it is, 0 → 1. Meaningless unless `floodRunning(age)`. */
export function floodProgress(age: number): number {
  if (!(age >= 0) || age >= FLOOD_SECONDS) return 0;
  return clamp01(age / FLOOD_SECONDS);
}

/** True when the cue crystal is allowed to carry a colour: the agent is looking at it. */
export function cueVisibleAt(pos: number): boolean {
  return pos === CUE_CELL;
}

/** The cue a frame carries — exported so the test can assert the honesty rule directly. */
export function cueOf(state: FrameState): 0 | 1 {
  return readCue(state);
}

/** The passage cell a frame carries. */
export function posOf(state: FrameState): number {
  return readPos(state);
}

const createScene: SceneFactory = () => new FireflyCaveScene();
export default createScene;
