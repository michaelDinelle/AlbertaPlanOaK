import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { slipDraw } from '@kit/props/ice';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { CLIFF_X, CLIFF_Y, cliffDecoder } from '@sim/envs/cliff';
import type { EnvFrame, GridCell } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { coastMaps, coastMaterial, FACE, layStones, LOOKS, stoneMaterial, WALL, wallSegments, type CoastMaps, type CoastUniforms, type Look } from './coast';
import { BEAM, beamMaterial, FIGURE, figureMaterial, flat, LAMP, LANTERN, lighthouse, POST, POST_LAMP, postMaterial, PRINT, printMaterial, skidMaterial, SPLASH, splashMaterial, type Level, type Lighthouse } from './figures';

const ARROWS = ['↑ north', '↓ south', '← west', 'east →'] as const;
const DX = [0, 0, -1, 1] as const;
const DY = [1, -1, 0, 0] as const;
const ACCENT = TOKENS.accents.CliffWalk!;

const HOP = 0.1;
const STEP_OFF = 0.09;
const DROP = 0.3;
const SINK = 0.14;
const FALL = STEP_OFF + DROP + SINK;
const ARRIVE = 0.5;
const GHOSTS = 10;
const TRAIL = 48;
const SKIDS = 12;
const SKID_LIFE = 2.4;
const SPLASH_LIFE = 0.55;
const FLARE_LIFE = 1.4;
const GLOW_LIFE = 0.35;
const WEAR_TAU = 300;
const WEAR_SPAN = 1200;
const WEAR_SCALE = 10;
const ROUTE = 240;
const SIZE = 1.2;

const ease = (t: number) => t * t * (3 - 2 * t);
const worn = (visits: number) => 1 - Math.exp(-visits / WEAR_SCALE);
const headingOf = (action: number) => Math.atan2(-DX[action]!, DY[action]!);
const turn = (from: number, to: number, k: number) => from + (Math.atan2(Math.sin(to - from), Math.cos(to - from))) * k;
const signed = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace('-', '−');

interface Ghost { born: number; fall: boolean; heading: number; spin: number; splashed: boolean; from: THREE.Vector3; edge: THREE.Vector3; land: THREE.Vector3 }
interface Skid { born: number; at: THREE.Vector3; heading: number; span: number }
interface Move { intent: number; exec: number; px: number; py: number; tx: number; ty: number }
interface Batch { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute }

class CliffWalkScene extends Stage2DScene {
  post: ScenePost = { bloom: 0.45, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'cliffWalk';

  private grid: GridSize = { w: 12, h: 4 };
  private look: Look = LOOKS.dusk;
  private slip = 0;
  private hazards: GridCell[] = [];
  private hazardSet = new Set<number>();
  private goal: GridCell = { x: 0, y: 0 };
  private start: GridCell | null = null;
  private maps: CoastMaps | null = null;
  private coast: CoastUniforms | null = null;
  private tower: Lighthouse | null = null;
  private beam: THREE.Mesh | null = null;
  private beamStrength: Level | null = null;
  private lamp = uniform(1);
  private lantern: THREE.Mesh | null = null;
  private signpost: THREE.Mesh | null = null;
  private signLamp: THREE.Mesh | null = null;
  private signGlow: Level | null = null;
  private walker: Batch | null = null;
  private shades: Batch | null = null;
  private prints: Batch | null = null;
  private slides: Batch | null = null;
  private splashes: Batch | null = null;
  private textures: THREE.Texture[] = [];

  private readonly ghosts: Ghost[] = Array.from({ length: GHOSTS }, () => ({ born: -Infinity, fall: true, heading: 0, spin: 1, splashed: true, from: new THREE.Vector3(), edge: new THREE.Vector3(), land: new THREE.Vector3() }));
  private readonly skids: Skid[] = Array.from({ length: SKIDS }, () => ({ born: -Infinity, at: new THREE.Vector3(), heading: 0, span: 0 }));
  private ghostCursor = 0;
  private skidCursor = 0;

  private readonly walk = { from: new THREE.Vector3(), to: new THREE.Vector3(), push: new THREE.Vector3(), t: 1, kind: 'rest' as 'rest' | 'glide' | 'bump' | 'skid', turnFrom: 0, heading: 0 };
  private readonly shown = new THREE.Vector3();
  private angle = 0;
  private step = -1;
  private simTime = 0;
  private clock = 0;
  private flared = -Infinity;
  private respawned = -Infinity;
  private wear = new Float32Array(0);
  private wearAt = -Infinity;
  private wearStale = true;
  private wearDirty = false;

  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { x: 0, y: 0, intent: -1, exec: -1, reward: 0, step: 0, harm: undefined as number | undefined, goal: undefined as number | undefined, looked: -Infinity };

  override legend: LegendSpec = {
    title: 'CLIFF WALK',
    rows: [
      { key: 'at', label: 'walker', tint: ACCENT, value: () => `column ${this.ledger.x + 1} · row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.intent < 0 ? '—' : this.ledger.exec === this.ledger.intent ? ARROWS[this.ledger.intent]! : `${ARROWS[this.ledger.intent]} · slid ${ARROWS[this.ledger.exec]}` },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => signed(this.ledger.reward) },
      { key: 'fall', label: 'last fall', tint: TOKENS.failureBlue, value: () => this.since('harm') },
      { key: 'goal', label: 'last arrival', tint: 0xffe2a8, value: () => this.since('goal') },
    ],
    note: () => {
      const d = this.cliffDistance(this.ledger.x, this.ledger.y);
      return `${d} step${d === 1 ? '' : 's'} from the cliff`;
    },
    bar: () => this.ledger.x / Math.max(1, this.grid.w - 1),
  };

  private since(type: 'harm' | 'goal'): string {
    const found = this.ledger[type];
    return found === undefined ? 'none yet' : found === this.ledger.step ? 'now' : `${formatCount(this.ledger.step - found)} steps ago`;
  }

  private cliffDistance(x: number, y: number): number {
    let best = Infinity;
    for (const cell of this.hazards) best = Math.min(best, Math.abs(cell.x - x) + Math.abs(cell.y - y));
    return best;
  }

  private row(y: number): number {
    return this.grid.h - 1 - y;
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, this.row(y), this.grid, out);
  }

  private inside(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < this.grid.w && y < this.grid.h;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.grid = { w: constant(env, 'CLIFF_WIDTH'), h: constant(env, 'CLIFF_HEIGHT') };
    this.slip = Math.fround(parameter(env, 'slip'));
    this.look = this.slip > 0 ? LOOKS.frost : LOOKS.dusk;
    this.post = this.slip > 0 ? { bloom: 0.5, fog: { from: ['dawn', 'hemiSky'] } } : { bloom: 0.45, fog: { from: ['dusk', 'fog'] } };
    const halfW = this.grid.w / 2, halfH = this.grid.h / 2;
    return new THREE.Box2(new THREE.Vector2(-halfW - 0.35, -halfH - 0.8), new THREE.Vector2(halfW + 0.35, halfH + 2.3));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(this.look.ambient).multiplyScalar(this.look.ambientLevel);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 }, goal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[CLIFF_X]!;
        agent.y = this.row(state.pose[CLIFF_Y]!);
        return agent;
      },
      goalCell: () => {
        goal.x = this.goal.x;
        goal.y = this.row(this.goal.y);
        return goal;
      },
      debug: (frame, kit) => {
        const marks = kit.marks('hazards', this.hazards.length);
        for (const cell of this.hazards) {
          this.centre(cell.x, cell.y, this.at);
          marks.set(this.at.x, 0.05, this.at.z, 0.13, 0xff6b5a);
        }
        if (this.start) {
          this.centre(this.start.x, this.start.y, this.at);
          kit.marks('start', 1).set(this.at.x, 0.05, this.at.z, 0.13, 0x86f0b4);
        }
        const worn = kit.marks('wear', this.grid.w * this.grid.h);
        for (let y = 0; y < this.grid.h; y++) for (let x = 0; x < this.grid.w; x++) {
          const level = this.wornLevel(x, y);
          if (level < 0.02) continue;
          this.centre(x, y, this.at);
          worn.set(this.at.x + 0.3, 0.05, this.at.z + 0.3, 0.02 + 0.1 * level, 0xc6a07a);
        }
        const route = kit.curve('route', ROUTE + 1, TOKENS.uiCyan), step = frame.state.step;
        const cells: EnvFrame[] = [];
        for (let s = step; s >= Math.max(0, step - ROUTE); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          cells.push(f);
          if (f.done) break;
        }
        for (let i = cells.length - 1; i >= 0; i--) {
          const f = cells[i]!;
          this.centre(f.words[0]!, f.words[1]!, this.at);
          route.point(this.at.x, 0.06, this.at.z);
        }
        const intent = kit.curve('intent', 5, TOKENS.uiCyan), exec = kit.curve('executed', 5, 0xffa05a);
        const now = frame.frameAt(step), before = frame.frameAt(step - 1);
        if (step < 1 || now.step !== step || before.step !== step - 1) return;
        const chosen = Math.trunc(now.actions[0]!) & 3, draw = slipDraw(this.ctx!.seed, step, this.slip), taken = draw ? draw[1]! & 3 : chosen;
        this.centre(before.words[0]!, before.words[1]!, this.other);
        this.arrow(intent, this.other, chosen, 0.62);
        if (taken !== chosen) this.arrow(exec, this.other, taken, 0.5);
      },
    };
  }

  private arrow(curve: { point(x: number, y: number, z: number): void }, from: THREE.Vector3, action: number, length: number): void {
    const dx = DX[action]!, dz = -DY[action]!, tip = { x: from.x + dx * length, z: from.z + dz * length };
    curve.point(from.x, 0.07, from.z);
    curve.point(tip.x, 0.07, tip.z);
    curve.point(tip.x - dx * 0.14 - dz * 0.1, 0.07, tip.z - dz * 0.14 + dx * 0.1);
    curve.point(tip.x, 0.07, tip.z);
    curve.point(tip.x - dx * 0.14 + dz * 0.1, 0.07, tip.z - dz * 0.14 - dx * 0.1);
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const env = ctx.environment, { w, h } = this.grid, look = this.look, lit = light.lit;
    if (!env.goal || !env.hazards) throw new Error(`${env.envId} lacks its goal or hazards`);
    this.goal = { x: env.goal[0], y: env.goal[1] };
    this.hazards = env.hazards.map(([x, y]) => ({ x, y }));
    this.hazardSet = new Set(this.hazards.map((cell) => cell.y * w + cell.x));
    this.wear = new Float32Array(w * h);

    const maps = coastMaps(this.grid, new Set(this.hazards.map((cell) => this.row(cell.y) * w + cell.x)), ctx.seed);
    this.maps = maps;
    this.textures.push(maps.mask, maps.wear, maps.mottle, maps.grain);
    const coast = coastMaterial(this.grid, maps, look, lit);
    this.coast = coast.uniforms;
    const ground = new THREE.Mesh(flat(80, 80), coast.material);
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground);

    const segments = wallSegments(this.grid, maps.sea), stones = layStones(segments, ctx.seed);
    const wall = instancedMesh(flat(0.1, 0.2), stoneMaterial(look, maps.grain, lit), stones.matrices.length);
    wall.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(stones.matrices.length * 3), 3);
    stones.matrices.forEach((matrix, i) => {
      wall.setMatrixAt(i, matrix);
      wall.setColorAt(i, new THREE.Color().setScalar(stones.tones[i]!));
    });
    wall.renderOrder = 4;
    wall.frustumCulled = false;
    this.scene.add(wall);
    for (const run of segments) {
      const span = run.from.distanceTo(run.to), shade = light.occluder(flat(span, WALL * 0.7));
      shade.position.set((run.from.x + run.to.x) / 2, 0.01, (run.from.y + run.to.y) / 2);
      shade.rotation.y = Math.atan2(-(run.to.y - run.from.y), run.to.x - run.from.x);
      shade.renderOrder = 1;
      this.scene.add(shade);
    }

    const tower = lighthouse(this.accentOf(ctx), look, lit, this.ambient().multiplyScalar(1.15));
    this.centre(this.goal.x, this.goal.y, tower.mesh.position);
    tower.mesh.renderOrder = 5;
    this.tower = tower;
    const lamp = light.emitter(new THREE.CircleGeometry(0.08, 32).rotateX(-Math.PI / 2), rgb(0xffe2b0).mul(this.lamp));
    lamp.position.set(tower.mesh.position.x + LAMP.x, 0.02, tower.mesh.position.z - LAMP.y);
    lamp.renderOrder = 2;
    const haze = beamMaterial();
    this.beamStrength = haze.strength;
    const beam = new THREE.Mesh(flat(BEAM, 2.4), haze.material);
    beam.position.copy(lamp.position).setY(0);
    beam.renderOrder = 9;
    beam.frustumCulled = false;
    this.beam = beam;
    this.scene.add(tower.mesh, lamp, beam);

    const sign = postMaterial(lit);
    this.signGlow = sign.glow;
    this.signpost = new THREE.Mesh(flat(POST, POST), sign.material);
    this.signpost.renderOrder = 5;
    this.signpost.visible = false;
    this.signLamp = light.emitter(new THREE.CircleGeometry(0.045, 24).rotateX(-Math.PI / 2), rgb(0xffc47a).mul(9).mul(sign.glow));
    this.signLamp.renderOrder = 2;
    this.signLamp.visible = false;
    this.scene.add(this.signpost, this.signLamp);

    const make = (geometry: THREE.BufferGeometry, name: string, material: THREE.Material, count: number, order: number): Batch => {
      const state = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, state);
      const mesh = instancedMesh(geometry, material, count);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return { mesh, state };
    };
    this.prints = make(flat(PRINT, PRINT), 'aPrint', printMaterial(look, lit), TRAIL, 1);
    this.slides = make(flat(1, 0.2), 'aSkid', skidMaterial(lit), SKIDS, 2);
    this.splashes = make(flat(SPLASH, SPLASH), 'aSplash', splashMaterial(look), GHOSTS, 3);

    const figure = figureMaterial(this.accentOf(ctx), look, lit);
    this.shades = make(flat(FIGURE, FIGURE), 'aFigure', figure, GHOSTS, 6);
    this.walker = make(flat(FIGURE, FIGURE), 'aFigure', figure, 1, 7);
    this.lantern = light.emitter(new THREE.CircleGeometry(0.055, 24).rotateX(-Math.PI / 2), rgb(0xffb35c).mul(11));
    this.lantern.renderOrder = 3;
    this.scene.add(this.lantern);
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = Math.max(time - this.simTime, frame.dtReal, 0);
    this.simTime = time;
    this.clock += frame.dtReal;
    pool.update(frame.dtReal, time);
    const step = frame.state.step;
    if (frame.scrubbed || step < this.step || this.start === null) this.settle(frame, step);
    else for (let s = this.step + 1; s <= step; s++) this.advance(frame, s, pool, time);
    if (step !== this.step) this.footprints(frame, step);
    this.step = step;
    this.ledger.step = step;
    if (this.clock - this.ledger.looked > 1) {
      this.ledger.harm = lastEvent(frame.eventLog, 'harm', step);
      this.ledger.goal = lastEvent(frame.eventLog, 'goal', step);
      this.ledger.looked = this.clock;
    }
    if (this.wearStale && this.clock - this.wearAt > 0.4) this.rebuildWear(frame, step);
    if (this.wearDirty) this.paintWear();

    this.animateWalker(dt);
    this.animateGhosts(pool, time);
    this.animateSkids(time);
    const flare = Math.max(0, 1 - (time - this.flared) / FLARE_LIFE), glow = Math.max(0, 1 - (time - this.respawned) / GLOW_LIFE);
    const heading = this.clock * 0.55;
    if (this.tower) {
      this.tower.flare.value = flare;
      this.tower.heading.value = heading;
    }
    if (this.beam) this.beam.rotation.y = heading;
    if (this.beamStrength) this.beamStrength.value = (this.look.frost ? 0.16 : 0.12) * (1 + 2 * flare);
    this.lamp.value = 7 * (1 + 2.5 * flare);
    if (this.signGlow) this.signGlow.value = 1 + 0.9 * glow + 0.08 * Math.sin(this.clock * 9.1) * Math.sin(this.clock * 3.7);
    if (this.coast) this.coast.time.value = this.clock;
  }

  private settle(frame: FrameUpdate, step: number): void {
    if (this.start === null) {
      const first = frame.frameAt(0);
      this.start = { x: first.words[0]!, y: first.words[1]! };
      this.coast?.start.value.set(this.start.x, this.row(this.start.y));
      const post = this.signpost, lampMesh = this.signLamp;
      if (post && lampMesh) {
        this.centre(this.start.x, this.start.y, post.position).add(this.other.set(-0.24, 0, -0.2));
        lampMesh.position.set(post.position.x + POST_LAMP.x, 0.02, post.position.z - POST_LAMP.y);
        post.visible = lampMesh.visible = true;
      }
    }
    const now = frame.frameAt(step);
    this.centre(now.words[0]!, now.words[1]!, this.shown);
    this.walk.from.copy(this.shown);
    this.walk.to.copy(this.shown);
    this.walk.t = 1;
    this.walk.kind = 'rest';
    const action = step > 0 ? Math.trunc(now.actions[0]!) & 3 : 3;
    this.angle = this.walk.heading = this.walk.turnFrom = headingOf(action);
    for (const ghost of this.ghosts) ghost.born = -Infinity;
    for (const skid of this.skids) skid.born = -Infinity;
    this.flared = this.respawned = -Infinity;
    for (let s = Math.max(1, step - Math.ceil(SKID_LIFE * frame.controlHz)); s <= step; s++) {
      const later = frame.frameAt(s), earlier = frame.frameAt(s - 1);
      if (later.step === s && earlier.step === s - 1) this.effects(later, earlier, s, this.move(later, earlier, s), frame.controlHz, null, 0);
    }
    this.wearStale = true;
    this.wearAt = -Infinity;
    this.ledger.looked = -Infinity;
    this.record(step, now);
    this.step = -1;
  }

  private move(now: EnvFrame, before: EnvFrame, s: number): Move {
    const intent = Math.trunc(now.actions[0]!) & 3, draw = slipDraw(this.ctx!.seed, s, this.slip), exec = draw ? draw[1]! & 3 : intent;
    const px = before.words[0]!, py = before.words[1]!;
    return { intent, exec, px, py, tx: px + DX[exec]!, ty: py + DY[exec]! };
  }

  private record(step: number, now: EnvFrame): void {
    const ledger = this.ledger;
    ledger.x = now.words[0]!;
    ledger.y = now.words[1]!;
    ledger.reward = now.reward;
    if (step < 1) {
      ledger.intent = ledger.exec = -1;
      return;
    }
    ledger.intent = Math.trunc(now.actions[0]!) & 3;
    const draw = slipDraw(this.ctx!.seed, step, this.slip);
    ledger.exec = draw ? draw[1]! & 3 : ledger.intent;
  }

  private advance(frame: FrameUpdate, s: number, pool: CelebrationPool, time: number): void {
    const now = frame.frameAt(s), before = frame.frameAt(s - 1);
    if (now.step !== s || before.step !== s - 1) {
      this.settle(frame, now.step);
      return;
    }
    const move = this.move(now, before, s), { intent, exec, px, py, tx, ty } = move;
    this.record(s, now);
    this.visit(now.words[0]!, now.words[1]!);
    this.effects(now, before, s, move, frame.controlHz, pool, time);
    const walk = this.walk;
    walk.turnFrom = this.angle;
    walk.heading = headingOf(intent);
    if (now.done) {
      this.centre(now.words[0]!, now.words[1]!, this.shown);
      walk.from.copy(this.shown);
      walk.to.copy(this.shown);
      walk.t = 1;
      walk.kind = 'rest';
      return;
    }
    const cx = now.words[0]!, cy = now.words[1]!, moved = cx !== px || cy !== py;
    if (moved ? cx !== tx || cy !== ty : this.inside(tx, ty)) throw new Error(`CliffWalk step ${s} went from (${px}, ${py}) to (${cx}, ${cy}) though ${ARROWS[exec]} leads to (${tx}, ${ty})`);
    walk.from.copy(this.shown);
    this.centre(cx, cy, walk.to);
    walk.push.set(DX[exec]!, 0, -DY[exec]!);
    walk.kind = !moved ? 'bump' : exec !== intent ? 'skid' : 'glide';
    walk.t = 0;
  }

  private effects(now: EnvFrame, before: EnvFrame, s: number, move: Move, hz: number, pool: CelebrationPool | null, time: number): void {
    const { intent, exec, px, py, tx, ty } = move, born = s / hz;
    for (const event of cliffDecoder.deriveEvents(before, now)) {
      if (event.type === 'harm' || event.type === 'goal') this.ledger[event.type] = s;
      if (event.type === 'harm') {
        if (!this.inside(tx, ty) || !this.hazardSet.has(ty * this.grid.w + tx)) throw new Error(`CliffWalk step ${s} fell from (${px}, ${py}) moving ${ARROWS[exec]} onto a cell that is not cliff`);
        this.depart(move, true, born, pool === null);
      }
      if (event.type === 'goal') {
        if (tx !== this.goal.x || ty !== this.goal.y) throw new Error(`CliffWalk step ${s} reached the goal from (${px}, ${py}) moving ${ARROWS[exec]}`);
        this.depart(move, false, born, true);
        this.flared = born;
        if (pool) {
          this.centre(tx, ty, this.at);
          pool.burst(this.at.setY(38), TOKENS.celebrationGold, 0.9, time);
          pool.ripple(this.at.setY(0), TOKENS.celebrationGold, 0.32, time);
        }
      }
      if (event.type === 'teleport') this.respawned = born;
    }
    if (exec === intent) return;
    const cx = now.words[0]!, cy = now.words[1]!, moved = !now.done && (cx !== px || cy !== py);
    this.slide(px, py, moved ? cx : tx, moved ? cy : ty, now.done ? 0.5 : moved ? 1 : 0.34, born);
  }

  private depart({ intent, exec, px, py, tx, ty }: Move, fall: boolean, born: number, quiet: boolean): void {
    const ghost = this.ghosts[this.ghostCursor]!;
    this.ghostCursor = (this.ghostCursor + 1) % GHOSTS;
    this.centre(px, py, ghost.from);
    this.centre(tx, ty, this.other);
    ghost.edge.copy(ghost.from).lerp(this.other, fall ? 0.66 : 1);
    ghost.land.copy(ghost.edge).add(this.at.set(DX[exec]! * 0.12, 0, FACE + 0.22 - DY[exec]! * 0.12));
    ghost.fall = fall;
    ghost.born = born;
    ghost.heading = headingOf(intent);
    ghost.spin = exec === 2 ? 1 : -1;
    ghost.splashed = quiet;
  }

  private slide(px: number, py: number, tx: number, ty: number, reach: number, born: number): void {
    const skid = this.skids[this.skidCursor]!;
    this.skidCursor = (this.skidCursor + 1) % SKIDS;
    this.centre(px, py, this.at);
    this.centre(tx, ty, this.other);
    this.other.sub(this.at).multiplyScalar(reach);
    skid.span = Math.max(0.2, this.other.length());
    skid.heading = Math.atan2(-this.other.z, this.other.x);
    skid.at.copy(this.at).addScaledVector(this.other, 0.5);
    skid.born = born;
  }

  private animateWalker(dt: number): void {
    const walker = this.walker, walk = this.walk;
    if (!walker) return;
    walk.t = Math.min(1, walk.t + dt / HOP);
    const k = walk.kind === 'skid' ? walk.t : ease(walk.t);
    if (walk.kind === 'bump') this.shown.copy(walk.to).addScaledVector(walk.push, 0.17 * Math.sin(Math.PI * walk.t));
    else this.shown.lerpVectors(walk.from, walk.to, k);
    this.angle = turn(walk.turnFrom, walk.heading, ease(Math.min(1, walk.t * 1.6)));
    const wobble = walk.kind === 'skid' ? 0.35 * Math.sin(Math.PI * walk.t) : 0;
    const bob = walk.kind === 'glide' ? 1 + 0.06 * Math.sin(Math.PI * walk.t) : 1;
    this.quaternion.setFromAxisAngle(this.axis, this.angle + wobble);
    walker.mesh.setMatrixAt(0, this.matrix.compose(this.shown, this.quaternion, this.scale.setScalar(bob * SIZE)));
    walker.state.setXYZW(0, 1, 0, 1, 0);
    walker.mesh.instanceMatrix.needsUpdate = true;
    walker.state.needsUpdate = true;
    if (this.lantern) this.lantern.position.copy(this.other.set(LANTERN.x, 0, -LANTERN.y).applyQuaternion(this.quaternion).multiplyScalar(bob * SIZE).add(this.shown)).setY(0.02);
  }

  private animateGhosts(pool: CelebrationPool, time: number): void {
    const shades = this.shades, splashes = this.splashes;
    if (!shades || !splashes) return;
    const impact = STEP_OFF + DROP;
    this.ghosts.forEach((ghost, i) => {
      const t = time - ghost.born, splash = (t - impact) / SPLASH_LIFE;
      const shown = ghost.fall && splash >= 0 && splash < 1;
      splashes.mesh.setMatrixAt(i, shown ? this.matrix.makeTranslation(ghost.land.x, 0, ghost.land.z) : this.matrix.makeScale(0, 0, 0));
      splashes.state.setXYZW(i, Math.max(0, splash), shown ? 1 : 0, 0, 0);
      if (ghost.fall && !ghost.splashed && splash >= 0) {
        ghost.splashed = true;
        pool.ripple(this.other.copy(ghost.land), this.look.foam, 0.13, time);
      }
      if (!(t >= 0 && t < (ghost.fall ? FALL : ARRIVE))) {
        shades.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
        shades.state.setXYZW(i, 0, 0, 0, 0);
        return;
      }
      let size = 1, sink = 0, alpha = 0.85, spin = 0, glow = 0;
      if (ghost.fall) {
        const walk = Math.min(1, t / STEP_OFF), drop = Math.min(1, Math.max(0, (t - STEP_OFF) / DROP)), under = Math.max(0, (t - impact) / SINK);
        this.at.lerpVectors(ghost.from, ghost.edge, 1 - (1 - walk) ** 2).lerp(ghost.land, drop * drop);
        size = 1 - 0.5 * drop;
        sink = drop;
        alpha = 0.85 * (1 - under);
        spin = ghost.spin * 2.2 * drop * drop;
      } else {
        const walk = Math.min(1, t / 0.16), fade = Math.max(0, (t - 0.16) / (ARRIVE - 0.16));
        this.at.lerpVectors(ghost.from, ghost.edge, ease(walk));
        size = 1 - 0.45 * fade;
        alpha = 0.85 * (1 - fade * fade);
        glow = Math.min(1, fade * 1.6);
      }
      this.quaternion.setFromAxisAngle(this.axis, ghost.heading + spin);
      shades.mesh.setMatrixAt(i, this.matrix.compose(this.at, this.quaternion, this.scale.setScalar(size * SIZE)));
      shades.state.setXYZW(i, alpha, sink, ghost.fall ? 0.4 : 1, glow);
    });
    for (const batch of [shades, splashes]) {
      batch.mesh.instanceMatrix.needsUpdate = true;
      batch.state.needsUpdate = true;
    }
  }

  private animateSkids(time: number): void {
    const slides = this.slides;
    if (!slides) return;
    this.skids.forEach((skid, i) => {
      const t = time - skid.born, life = 1 - t / SKID_LIFE;
      if (!(t >= 0 && life > 0)) {
        slides.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
        slides.state.setXYZW(i, 0, 1, 0, 0);
        return;
      }
      this.quaternion.setFromAxisAngle(this.axis, skid.heading);
      slides.mesh.setMatrixAt(i, this.matrix.compose(skid.at, this.quaternion, this.scale.set(skid.span, 1, 1)));
      slides.state.setXYZW(i, 0.85 * Math.min(1, life * 1.6), skid.span, 0, 0);
    });
    slides.mesh.instanceMatrix.needsUpdate = true;
    slides.state.needsUpdate = true;
  }

  private footprints(frame: FrameUpdate, step: number): void {
    const prints = this.prints;
    if (!prints) return;
    for (let i = 0; i < TRAIL; i++) {
      const s = step - i, now = s >= 1 ? frame.frameAt(s) : null, before = s >= 1 ? frame.frameAt(s - 1) : null;
      const x = now?.words[0], y = now?.words[1], bx = before?.words[0], by = before?.words[1];
      if (!now || !before || now.step !== s || before.step !== s - 1 || now.done || (x === bx && y === by)) {
        prints.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
        prints.state.setXYZW(i, 0, 0, 0, 0);
        continue;
      }
      this.centre(x!, y!, this.at);
      this.quaternion.setFromAxisAngle(this.axis, Math.atan2(-(x! - bx!), y! - by!));
      prints.mesh.setMatrixAt(i, this.matrix.compose(this.at, this.quaternion, this.scale.setScalar(1)));
      prints.state.setXYZW(i, 0.6 * (1 - i / TRAIL) ** 1.2, s % 2 ? 1 : -1, 0, 0);
    }
    prints.mesh.instanceMatrix.needsUpdate = true;
    prints.state.needsUpdate = true;
  }

  private visit(x: number, y: number): void {
    const decay = Math.exp(-1 / WEAR_TAU);
    for (let i = 0; i < this.wear.length; i++) this.wear[i]! *= decay;
    this.wear[this.row(y) * this.grid.w + x]! += 1;
    this.wearDirty = true;
  }

  private rebuildWear(frame: FrameUpdate, step: number): void {
    this.wear.fill(0);
    this.wearAt = this.clock;
    this.wearStale = false;
    for (let i = 0; i < WEAR_SPAN && step - i >= 0; i++) {
      const f = frame.frameAt(step - i);
      if (f.step !== step - i) {
        this.wearStale = true;
        continue;
      }
      this.wear[this.row(f.words[1]!) * this.grid.w + f.words[0]!]! += Math.exp(-i / WEAR_TAU);
    }
    this.paintWear();
  }

  private wornLevel(x: number, y: number): number {
    return worn(this.wear[this.row(y) * this.grid.w + x] ?? 0);
  }

  private paintWear(): void {
    const texture = this.maps?.wear;
    if (!texture) return;
    const data = texture.image.data as Uint8Array;
    for (let i = 0; i < this.wear.length; i++) data[i * 4] = Math.round(255 * worn(this.wear[i]!));
    texture.needsUpdate = true;
    this.wearDirty = false;
  }
}

export default (() => new CliffWalkScene()) satisfies SceneFactory;
