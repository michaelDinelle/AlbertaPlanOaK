import * as THREE from 'three/webgpu';
import { vec3 } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { LIGHT_LAYER, type Light2D } from '@kit/light2d';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import {
  LOST, POM_CELL, RIGHT, RIGID, TIED, UP, WON, WOOD,
  actOf, aliveOf, ammoOf, blastOf, bombsOf, cellOf, collapsedRings, fieldOf, flameOf, gameOf, hiddenOf, itemOf, kickOf,
  partnerOf, pomShape, resultOf, ringOf, seatOf, sightingsOf, terrainOf, tickOf, STEP_X, STEP_Y, type PomBomb, type PomShape,
} from '@sim/envs/pommerman';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import {
  BADGE_H, BADGE_W, BOMB_QUAD, BOMBER_QUAD, FX, FX_QUAD, HUE, ITEM_HUES, RIM, Batch, badgeShade, bomberShade, bombShade, flat, fogShade, fxShade, makeBoard,
  type Board, type Fog,
} from './look';

const TICK_HZ = 4;
const HISTORY = 6;
const WARN_TICKS = 30;
const FACING_SCAN = 16;
const LIFE = { [FX.SPLINTER]: 0.7, [FX.PICKUP]: 0.8, [FX.PUFF]: 1.0, [FX.DUST]: 0.45, [FX.SLAM]: 0.9 } as Record<number, number>;
const GHOST_LIFE = 1.1;
const POP = 0.16;
const SPAWN = 0.55;
const SIZE = 1.14;
const GOLD = TOKENS.celebrationGold;
const FFA_RIVALS = [0x5aa8ff, 0xc58cff, 0x62dca0] as const;
const TEAM_RIVALS = [0x5aa8ff, 0xa58bff] as const;
const PARTNER = 0xf2c94c;
const HEADING = [Math.PI, 0, Math.PI, Math.PI / 2, -Math.PI / 2, Math.PI] as const;
const ACTIONS = ['· stay', '↑ up', '↓ down', '← left', '→ right', '● bomb'] as const;
const RESULTS = ['—', 'won', 'lost', 'tied'] as const;

const signed = (value: number) => formatSigned(value).replace('-', '−');
const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;

type Kind = 'splinter' | 'pickup' | 'burn' | 'death' | 'place' | 'slam' | 'reveal';
interface Happening { kind: Kind; cell: number; arg: number }
interface Tick {
  step: number;
  game: number;
  tick: number;
  start: number;
  words: Uint32Array;
  view: Uint32Array;
  bombs: PomBomb[];
  facing: number[];
  happenings: Happening[];
}

class PommermanScene extends Stage2DScene {
  post: ScenePost = { bloom: 0.36, fog: { from: ['dusk', 'fog'] } };
  controlHz = TICK_HZ;
  protected readonly stageName = 'pommerman';
  override legend: LegendSpec | null = null;

  private shape!: PomShape;
  private grid: GridSize = { w: 11, h: 11 };
  private accent = 0xff9448;
  private board: Board | null = null;
  private bombs: Batch | null = null;
  private bombers: Batch | null = null;
  private fx: Batch | null = null;
  private badges: Batch | null = null;
  private fog: Fog | null = null;
  private fogMesh: THREE.Mesh | null = null;
  private lamps: THREE.InstancedMesh | null = null;
  private lampUsed = 0;
  private readonly owned: { dispose(): void }[] = [];

  private readonly ticks = new Map<number, Tick>();
  private painted = -1;
  private lastTick = -1;
  private celebrated = -1;
  private log: FrameUpdate['eventLog'] | null = null;

  private readonly at = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly color = new THREE.Color();
  private readonly matrix = new THREE.Matrix4();
  private readonly ledger = { step: 0, x: 0, y: 0, act: 0, ammo: 0, blast: 0, kick: false, rivals: 0, rivalSeats: 0, partner: false, reward: 0, result: 0, game: 0, tick: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.shape = pomShape(env);
    this.grid = { w: this.shape.side, h: this.shape.side };
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor();
    const x = this.shape.side / 2 + RIM + 0.4 + BADGE_W + 0.2, z = this.shape.side / 2 + RIM + 0.2;
    return new THREE.Box2(new THREE.Vector2(-x, -z), new THREE.Vector2(x, z));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xc8d0ea).multiplyScalar(0.8);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighSize: 34, sighRadius: 1.3, sighHeight: 0.9 };
  }

  private legendFor(): LegendSpec {
    const l = this.ledger, shape = this.shape;
    return {
      title: 'POMMERMAN',
      rows: [
        { key: 'you', label: 'agent', tint: this.accent, value: () => `column ${l.x + 1} · row ${l.y + 1}` },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTIONS[l.act] ?? '—' },
        { key: 'arsenal', label: 'arsenal', tint: HUE.flameMid, value: () => `${l.ammo} bomb${l.ammo === 1 ? '' : 's'} · blast ${l.blast}${l.kick ? ' · kick' : ''}` },
        { key: 'field', label: shape.team ? 'teams' : shape.seats === 2 ? 'rival' : 'rivals', tint: FFA_RIVALS[0], value: () => this.field() },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(l.reward) },
        { key: 'last', label: 'last game', tint: GOLD, value: () => RESULTS[l.result] ?? '—' },
      ],
      note: () => this.record(),
      bar: () => l.tick / shape.tickLimit,
    };
  }

  private field(): string {
    const l = this.ledger;
    if (this.shape.team) return `partner ${l.partner ? 'standing' : 'down'} · ${l.rivals} of ${l.rivalSeats} rivals`;
    if (this.shape.seats === 2) return l.rivals ? 'standing' : 'down';
    return `${l.rivals} of ${l.rivalSeats} standing`;
  }

  private record(): string {
    const l = this.ledger, log = this.log, shape = this.shape;
    let tally = '';
    if (log) {
      const won = countEvents(log, 'goal', -1, l.step), lost = countEvents(log, 'harm', -1, l.step), games = countEvents(log, 'done', -1, l.step);
      tally = ` · ${formatCount(won)} W · ${formatCount(lost)} L · ${formatCount(games - won - lost)} T`;
    }
    const rings = collapsedRings(shape, l.tick), due = shape.collapseStart + rings * shape.collapseEvery;
    const closing = shape.collapseRings && rings < shape.collapseRings ? ` · walls close in ${formatCount(Math.max(0, due - l.tick))}` : shape.collapseRings ? ' · arena closed' : '';
    return `game ${formatCount(l.game + 1)} · tick ${formatCount(l.tick)} / ${formatCount(shape.tickLimit)}${closing}${tally}`;
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        const cell = state.pose[POM_CELL]!;
        agent.x = cell % this.shape.side;
        agent.y = Math.floor(cell / this.shape.side);
        return agent;
      },
      debug: (frame, kit) => this.groundTruth(frame, kit),
    };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    const shape = this.shape, side = shape.side, lit = light.lit;
    this.scene.background = new THREE.Color(HUE.void);
    pool.group.traverse((object) => { object.renderOrder = 20; });

    const board = makeBoard(side, shape.flameLife, lit);
    this.board = board;
    this.owned.push(board);
    const ground = new THREE.Mesh(flat(side + 60, side + 40), board.ground);
    ground.renderOrder = 0;
    ground.frustumCulled = false;
    this.scene.add(ground);
    this.owned.push(ground.geometry);

    const occluder = light.occluder(flat(side + 2 * RIM, side + 2 * RIM));
    const shade = occluder.material as THREE.MeshBasicNodeMaterial;
    shade.opacityNode = board.walls();
    shade.alphaTest = 0.5;
    this.lightProxy(occluder);
    const fire = board.fire(), flames = light.emitter(flat(side, side), fire.color);
    const glow = flames.material as THREE.MeshBasicNodeMaterial;
    glow.opacityNode = fire.mask;
    glow.alphaTest = 0.5;
    this.lightProxy(flames);
    this.lamps = this.lampBank(light, 4 + shape.bombs + shape.cells);

    const time = board.time;
    this.bombs = this.batch(new Batch(shape.bombs, BOMB_QUAD, BOMB_QUAD, ['aBomb', 'aTint'], bombShade(lit, time), 4));
    this.fx = this.batch(new Batch(96, FX_QUAD, FX_QUAD, ['aFx', 'aTint'], fxShade(), 5));
    this.bombers = this.batch(new Batch(shape.seats, BOMBER_QUAD, BOMBER_QUAD, ['aBody', 'aTint', 'aMark'], bomberShade(lit, time, this.accent), 6));
    if (shape.fog) {
      const fog = fogShade(this.accent, time);
      this.fog = fog;
      const mesh = new THREE.Mesh(flat(side + 2 * RIM, side + 2 * RIM), fog.material);
      mesh.renderOrder = 8;
      mesh.frustumCulled = false;
      this.fogMesh = mesh;
      this.scene.add(mesh);
      this.owned.push(mesh.geometry, fog.material);
    }
    this.badges = this.batch(new Batch(shape.seats, BADGE_W, BADGE_H, ['aSeat', 'aTint', 'aMeta'], badgeShade(lit, time, this.accent, Math.max(shape.maxAmmo, shape.maxBlast)), 9));
  }

  private lightProxy(mesh: THREE.Mesh): void {
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.owned.push(mesh.geometry, mesh.material as THREE.Material);
  }

  private lampBank(light: Light2D, count: number): THREE.InstancedMesh {
    const template = light.emitter(new THREE.CircleGeometry(1, 20).rotateX(-Math.PI / 2), vec3(1));
    const mesh = instancedMesh(template.geometry, template.material, count);
    mesh.layers.set(LIGHT_LAYER);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    mesh.count = 0;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.owned.push(template.geometry, template.material as THREE.Material, mesh);
    return mesh;
  }

  private batch(batch: Batch): Batch {
    this.scene.add(batch.mesh);
    this.owned.push(batch);
    return batch;
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    if (this.fogMesh) this.fogMesh.visible = !on;
  }

  protected override drop(): void {
    for (const item of this.owned) item.dispose();
    this.owned.length = 0;
    this.ticks.clear();
  }

  private centre(cell: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(cell % this.shape.side, Math.floor(cell / this.shape.side), this.grid, out);
  }

  private frame(frame: FrameUpdate, step: number): EnvFrame | null {
    if (step < 0) return null;
    const found = frame.frameAt(step);
    return found.step === step ? found : null;
  }

  private hue(words: ArrayLike<number>, seat: number): number {
    const me = seatOf(words), shape = this.shape;
    if (seat === me) return this.accent;
    if (shape.team) return seat === partnerOf(shape, me) ? PARTNER : TEAM_RIVALS[seat === ((me + 1) & 3) ? 0 : 1]!;
    let rank = 0;
    for (let s = 0; s < seat; s++) if (s !== me) rank++;
    return FFA_RIVALS[rank]!;
  }

  private tickAt(frame: FrameUpdate, step: number, fallback: EnvFrame): Tick {
    const cached = this.ticks.get(step);
    if (cached) return cached;
    const shape = this.shape, source = this.frame(frame, step) ?? fallback;
    const words = Uint32Array.from(source.words), view = Uint32Array.from(source.view), game = gameOf(words), tick = tickOf(words);
    const before = tick > 0 ? this.frame(frame, step - 1) : null;
    const prev = before && gameOf(before.words) === game && tickOf(before.words) === tick - 1 ? before : null;
    const bombs = bombsOf(view, shape);
    const record: Tick = { step, game, tick, start: step - tick, words, view, bombs, facing: this.facing(frame, step, words), happenings: prev ? this.happenings(prev, words, view, bombs, tick) : [] };
    if (prev || tick === 0) {
      this.ticks.set(step, record);
      if (this.ticks.size > 96) for (const key of [...this.ticks.keys()].sort((a, b) => Math.abs(b - step) - Math.abs(a - step)).slice(0, 32)) this.ticks.delete(key);
    }
    return record;
  }

  private facing(frame: FrameUpdate, step: number, words: Uint32Array): number[] {
    const shape = this.shape, out = new Array<number>(shape.seats).fill(Math.PI), found = new Array<boolean>(shape.seats).fill(false), game = gameOf(words);
    let source: ArrayLike<number> | null = words;
    for (let j = 0; j < FACING_SCAN && source; j++) {
      for (let s = 0; s < shape.seats; s++) {
        const act = actOf(source, shape, s);
        if (!found[s] && act >= UP && act <= RIGHT) { out[s] = HEADING[act]!; found[s] = true; }
      }
      if (found.every(Boolean) || tickOf(source) === 0) break;
      const f = this.frame(frame, step - j - 1);
      source = f && gameOf(f.words) === game ? f.words : null;
    }
    return out;
  }

  private happenings(prev: EnvFrame, words: Uint32Array, view: Uint32Array, bombs: PomBomb[], tick: number): Happening[] {
    const shape = this.shape, out: Happening[] = [], side = shape.side;
    const rings = collapsedRings(shape, tick), slammed = rings > collapsedRings(shape, tick - 1) ? rings - 1 : -1;
    for (let c = 0; c < shape.cells; c++) {
      const a = fieldOf(prev.view, c), b = fieldOf(view, c);
      if (slammed >= 0 && ringOf(shape, c % side, Math.floor(c / side)) === slammed) {
        if (terrainOf(b) !== RIGID) throw new Error(`Pommerman tick ${tick} collapses ring ${slammed} but leaves cell ${c} open`);
        out.push({ kind: 'slam', cell: c, arg: 0 });
        continue;
      }
      if (terrainOf(a) === WOOD && terrainOf(b) !== WOOD) out.push({ kind: 'splinter', cell: c, arg: 0 });
      const shown = hiddenOf(a) ? 0 : itemOf(a), now = hiddenOf(b) ? 0 : itemOf(b);
      if (shown && !now) {
        let taker = -1;
        for (let s = 0; s < shape.seats; s++) if (aliveOf(prev.words, s) && aliveOf(words, s) && cellOf(words, s) === c) taker = s;
        out.push({ kind: taker >= 0 ? 'pickup' : 'burn', cell: c, arg: shown });
      }
      if (hiddenOf(a) && itemOf(a) && now) out.push({ kind: 'reveal', cell: c, arg: now });
    }
    for (let s = 0; s < shape.seats; s++) if (aliveOf(prev.words, s) && !aliveOf(words, s)) out.push({ kind: 'death', cell: cellOf(words, s), arg: s });
    for (const bomb of bombs) if (bomb.life === shape.bombLife) out.push({ kind: 'place', cell: bomb.cell, arg: bomb.owner });
    return out;
  }

  private history(frame: FrameUpdate, now: Tick, fallback: EnvFrame): Tick[] {
    const out = [now];
    for (let j = 1; j < HISTORY; j++) {
      const step = now.step - j;
      if (step < now.start) break;
      const f = this.frame(frame, step);
      if (!f || gameOf(f.words) !== now.game) break;
      out.push(this.tickAt(frame, step, fallback));
    }
    return out;
  }

  private paint(ticks: Tick[]): void {
    const board = this.board!, shape = this.shape, side = shape.side, now = ticks[0]!;
    const cells = board.cells.image.data as Float32Array, marks = board.marks.image.data as Float32Array;
    const rings = collapsedRings(shape, now.tick), due = shape.collapseStart + rings * shape.collapseEvery;
    const warn = shape.collapseRings && rings < shape.collapseRings && due >= now.tick ? clamp01(1 - (due - now.tick) / WARN_TICKS) : 0;
    for (let c = 0; c < shape.cells; c++) {
      const field = fieldOf(now.view, c), terrain = terrainOf(field), ring = ringOf(shape, c % side, Math.floor(c / side));
      cells[4 * c] = terrain === RIGID && ring < rings ? 3 : terrain;
      cells[4 * c + 1] = flameOf(field);
      cells[4 * c + 2] = hiddenOf(field) ? 0 : itemOf(field);
      cells[4 * c + 3] = 0;
      marks[4 * c] = marks[4 * c + 1] = marks[4 * c + 2] = -1;
      marks[4 * c + 3] = ring === rings ? warn : 0;
    }
    ticks.forEach((record, j) => {
      if (j > 3) return;
      for (const happening of record.happenings) {
        const slot = happening.kind === 'splinter' ? 0 : happening.kind === 'slam' ? 1 : happening.kind === 'reveal' ? 2 : -1;
        const at = 4 * happening.cell + slot;
        if (slot >= 0 && marks[at]! < 0) marks[at] = j;
      }
    });
    board.cells.needsUpdate = true;
    board.marks.needsUpdate = true;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.log = frame.eventLog;
    const board = this.board!, step = frame.state.step, now = this.frame(frame, step);
    board.time.value = time;
    if (!now) return;
    const current = this.tickAt(frame, step, now), ticks = this.history(frame, current, now);
    const peek = this.frame(frame, step + 1), next = peek && !peek.done && gameOf(peek.words) === current.game ? peek : null;
    const u = clamp01(frame.position - step);
    board.u.value = u;
    board.start.value = (frame.position - current.start) / this.controlHz;
    if (current.step !== this.painted) {
      this.paint(ticks);
      this.painted = current.step;
    }
    if (!frame.scrubbed) this.celebrate(frame, current, pool, time);
    this.lastTick = current.step;
    this.lampUsed = 0;
    this.fx!.begin();
    this.drawBombs(frame, current, next, u);
    this.drawBombers(frame, current, next, ticks, u);
    this.drawFx(frame, current, ticks);
    this.drawBadges(current);
    this.drawFog(current, next, u);
    this.fx!.commit();
    this.commitLamps();
    this.account(now, current);
  }

  private celebrate(frame: FrameUpdate, current: Tick, pool: CelebrationPool, time: number): void {
    if (current.step - this.lastTick === 1) {
      for (const happening of current.happenings) {
        if (happening.kind !== 'death') continue;
        this.centre(happening.cell, this.at);
        pool.ripple(this.at, this.hue(current.words, happening.arg), 0.32, time);
      }
    }
    if (!frame.events.some((event) => event.type === 'done')) return;
    const ended = lastEvent(frame.eventLog, 'done', frame.state.step);
    if (ended === undefined || ended === this.celebrated) return;
    this.celebrated = ended;
    const end = this.frame(frame, ended), last = this.frame(frame, ended - 1);
    if (!end) return;
    const result = resultOf(end.words);
    if (last) this.centre(cellOf(last.words, seatOf(last.words)), this.at);
    else this.at.set(0, 0, 0);
    if (result === WON) {
      pool.burst(this.to.copy(this.at).setY(30), GOLD, 1.3, time);
      pool.ripple(this.at, GOLD, 0.5, time);
    } else if (result === LOST) {
      pool.sigh(0.8, this.at, time);
      pool.ripple(this.at, TOKENS.failureBlue, 0.42, time);
    } else if (result === TIED) pool.ripple(this.at.set(0, 0, 0), 0xdcf4ff, 0.8, time);
  }

  private lamp(x: number, z: number, radius: number, color: number, gain: number): void {
    const lamps = this.lamps!;
    if (this.lampUsed >= lamps.instanceMatrix.count) return;
    const i = this.lampUsed++;
    lamps.setMatrixAt(i, this.matrix.makeScale(radius, 1, radius).setPosition(x, 0, z));
    lamps.setColorAt(i, this.color.set(color).multiplyScalar(gain));
  }

  private commitLamps(): void {
    const lamps = this.lamps!;
    lamps.count = this.lampUsed;
    lamps.instanceMatrix.needsUpdate = true;
    if (lamps.instanceColor) lamps.instanceColor.needsUpdate = true;
  }

  private drawBombs(frame: FrameUpdate, current: Tick, next: EnvFrame | null, u: number): void {
    const bombs = this.bombs!, shape = this.shape, hz = this.controlHz;
    const moved = new Map<number, number>();
    if (next) for (const bomb of bombsOf(next.view, shape)) {
      if (!bomb.dir) continue;
      const x = bomb.cell % shape.side - STEP_X[bomb.dir]!, y = Math.floor(bomb.cell / shape.side) - STEP_Y[bomb.dir]!;
      moved.set(y * shape.side + x, bomb.cell);
    }
    bombs.begin();
    const age = (frame.position - current.step) / hz;
    for (const bomb of current.bombs) {
      this.centre(bomb.cell, this.at);
      const target = moved.get(bomb.cell);
      if (target !== undefined) this.at.lerp(this.centre(target, this.to), u);
      const i = bombs.put(this.at.x, this.at.z);
      const spawn = bomb.life === shape.bombLife ? clamp01(age / POP) : 1;
      bombs.set('aBomb', i, Math.max(0, bomb.life - u) / shape.bombLife, spawn, 0, 0);
      if (target !== undefined) {
        const dx = target % shape.side - bomb.cell % shape.side, dy = Math.floor(target / shape.side) - Math.floor(bomb.cell / shape.side);
        this.putFx(this.at.x, this.at.z, FX.STREAK, 0, 0, Math.atan2(-dy, dx), 0xdfe6f0);
      }
      this.color.set(this.hue(current.words, bomb.owner));
      bombs.set('aTint', i, this.color.r, this.color.g, this.color.b, 0);
      const fuse = Math.max(0, bomb.life - u) / shape.bombLife, flicker = 0.75 + 0.25 * Math.sin(frame.position * 9.1 + bomb.cell);
      this.lamp(this.at.x + 0.24 + 0.05 * fuse, this.at.z - 0.24 - 0.2 * fuse, 0.07, HUE.spark, 2.4 * flicker * spawn);
      if (fuse < 0.4) this.lamp(this.at.x, this.at.z, 0.2, HUE.danger, 1.6 * (1 - fuse / 0.4));
    }
    bombs.commit();
  }

  private drawBombers(frame: FrameUpdate, current: Tick, next: EnvFrame | null, ticks: Tick[], u: number): void {
    const bombers = this.bombers!, shape = this.shape, hz = this.controlHz, me = seatOf(current.words);
    const spawn = (frame.position - current.start) / hz;
    bombers.begin();
    const order = [...Array(shape.seats).keys()].filter((s) => s !== me).concat(me);
    for (const seat of order) {
      const hue = this.hue(current.words, seat);
      this.color.set(hue);
      if (aliveOf(current.words, seat)) {
        const from = cellOf(current.words, seat), to = next ? cellOf(next.words, seat) : from;
        this.centre(from, this.at);
        let heading = current.facing[seat]!, stride = 0;
        if (to !== from) {
          this.centre(to, this.to);
          const dx = this.to.x - this.at.x, dz = this.to.z - this.at.z;
          heading = turn(heading, Math.atan2(-dx, -dz), smoothstep01(Math.min(1, u / 0.3)));
          stride = Math.sin(u * Math.PI * 2);
          this.at.lerp(this.to, u);
        } else if (next) {
          const act = actOf(next.words, shape, seat);
          if (act >= UP && act <= RIGHT) heading = turn(heading, HEADING[act]!, smoothstep01(Math.min(1, u / 0.3)));
        }
        const flash = this.flash(frame, ticks, seat);
        const i = bombers.put(this.at.x, this.at.z, SIZE);
        bombers.set('aBody', i, heading, stride, 1, 0);
        bombers.set('aTint', i, this.color.r, this.color.g, this.color.b, seat === me ? 1 : 0);
        bombers.set('aMark', i, spawn < SPAWN ? spawn / SPAWN : 0, 0, flash, 0);
        this.lamp(this.at.x, this.at.z, seat === me ? 0.2 : 0.15, hue, seat === me ? 4.2 : 2.6);
        continue;
      }
      for (const record of ticks) {
        const death = record.happenings.find((h) => h.kind === 'death' && h.arg === seat);
        if (!death) continue;
        const age = (frame.position - record.step) / hz;
        if (age >= GHOST_LIFE) break;
        this.centre(death.cell, this.at);
        const t = age / GHOST_LIFE, i = bombers.put(this.at.x, this.at.z, SIZE);
        bombers.set('aBody', i, record.facing[seat]!, 0, 1 - t * t, 1);
        bombers.set('aTint', i, this.color.r, this.color.g, this.color.b, 0);
        bombers.set('aMark', i, 0, smoothstep01(t), 0, 0);
        break;
      }
    }
    bombers.commit();
  }

  private flash(frame: FrameUpdate, ticks: Tick[], seat: number): number {
    for (const record of ticks.slice(0, 2)) {
      const age = (frame.position - record.step) / this.controlHz;
      for (const h of record.happenings) if (h.kind === 'pickup' && cellOf(record.words, seat) === h.cell) return Math.max(0, 1 - age / 0.3);
    }
    return 0;
  }

  private putFx(x: number, z: number, kind: number, t: number, seed: number, arg: number, hue: number): void {
    const fx = this.fx!;
    if (fx.used >= fx.capacity) return;
    const i = fx.put(x, z);
    fx.set('aFx', i, kind, t, seed, arg);
    this.color.set(hue);
    fx.set('aTint', i, this.color.r, this.color.g, this.color.b, 0);
  }

  private drawFx(frame: FrameUpdate, current: Tick, ticks: Tick[]): void {
    const shape = this.shape, hz = this.controlHz;
    const put = (cell: number, kind: number, t: number, seed: number, arg: number, hue: number) => {
      this.centre(cell, this.at);
      this.putFx(this.at.x, this.at.z, kind, t, seed, arg, hue);
    };
    for (const record of ticks) {
      const age = (frame.position - record.step) / hz;
      for (const h of record.happenings) {
        const kind = h.kind === 'splinter' ? FX.SPLINTER : h.kind === 'pickup' ? FX.PICKUP : h.kind === 'burn' || h.kind === 'death' ? FX.PUFF : h.kind === 'place' ? FX.DUST : h.kind === 'slam' ? FX.SLAM : -1;
        if (kind < 0) continue;
        const t = age / LIFE[kind]!;
        if (t < 0 || t >= 1) continue;
        const hue = kind === FX.SPLINTER ? HUE.plank : kind === FX.PICKUP ? ITEM_HUES[h.arg]! : kind === FX.PUFF ? 0x5b5856 : kind === FX.SLAM ? HUE.seam : 0x9aa0a8;
        put(h.cell, kind, t, (record.step * 7 + h.cell) % 97, h.arg, hue);
      }
    }
    for (let c = 0; c < shape.cells; c++) {
      const field = fieldOf(current.view, c), item = hiddenOf(field) ? 0 : itemOf(field);
      if (!item || terrainOf(field)) continue;
      this.centre(c, this.at);
      this.lamp(this.at.x, this.at.z, 0.13, ITEM_HUES[item]!, 1.3);
    }
  }

  private drawBadges(current: Tick): void {
    const badges = this.badges!, shape = this.shape, me = seatOf(current.words), half = shape.seats / 2;
    badges.begin();
    const x = shape.side / 2 + RIM + 0.4 + BADGE_W / 2, pitch = BADGE_H + 0.35;
    for (let seat = 0; seat < shape.seats; seat++) {
      const column = seat < half ? -1 : 1, row = seat % half, z = (row - (half - 1) / 2) * pitch;
      const i = badges.put(column * x, z);
      const alive = aliveOf(current.words, seat) ? 1 : 0;
      badges.set('aSeat', i, ammoOf(current.words, shape, seat), blastOf(current.words, shape, seat), kickOf(current.words, shape, seat) ? 1 : 0, alive);
      this.color.set(this.hue(current.words, seat));
      badges.set('aTint', i, this.color.r, this.color.g, this.color.b, seat === me ? 1 : 0);
      badges.set('aMeta', i, -column, seat === me ? 2 : seat === partnerOf(shape, me) ? 1 : 0, 0, 0);
    }
    badges.commit();
  }

  private drawFog(current: Tick, next: EnvFrame | null, u: number): void {
    const fog = this.fog, shape = this.shape;
    if (!fog) return;
    const me = seatOf(current.words);
    const from = cellOf(current.words, me), to = next ? cellOf(next.words, me) : from, side = shape.side, low = -(side - 1) / 2;
    const x = from % side + (to % side - from % side) * u, y = Math.floor(from / side) + (Math.floor(to / side) - Math.floor(from / side)) * u;
    fog.rect.value.set(low + Math.max(0, x - shape.fog) - 0.5, low + Math.max(0, y - shape.fog) - 0.5, low + Math.min(side - 1, x + shape.fog) + 0.5, low + Math.min(side - 1, y + shape.fog) + 0.5);
    fog.strength.value = 0.62;
  }

  private account(now: EnvFrame, current: Tick): void {
    const l = this.ledger, shape = this.shape, words = current.words, me = seatOf(words), cell = cellOf(words, me);
    l.step = now.step;
    l.x = cell % shape.side;
    l.y = Math.floor(cell / shape.side);
    l.act = actOf(words, shape, me);
    l.ammo = ammoOf(words, shape, me);
    l.blast = blastOf(words, shape, me);
    l.kick = kickOf(words, shape, me);
    l.reward = now.reward;
    l.result = resultOf(now.words);
    l.game = gameOf(words);
    l.tick = current.tick;
    let rivals = 0, seats = 0;
    for (let s = 0; s < shape.seats; s++) {
      if (s === me || s === partnerOf(shape, me)) continue;
      seats++;
      if (aliveOf(words, s)) rivals++;
    }
    l.rivals = rivals;
    l.rivalSeats = seats;
    l.partner = shape.team && aliveOf(words, partnerOf(shape, me));
    for (const seen of sightingsOf(now.observations, shape, me)) {
      if ((seen.code > 0) !== (seen.seat === partnerOf(shape, me))) throw new Error(`Pommerman step ${now.step} observes seat ${seen.seat} as ${seen.code > 0 ? 'a partner' : 'a rival'} against the team the scene draws`);
      const at = cellOf(words, seen.seat);
      if (at % shape.side - cell % shape.side !== seen.dx || Math.floor(at / shape.side) - Math.floor(cell / shape.side) !== seen.dy) throw new Error(`Pommerman step ${now.step} observes seat ${seen.seat} at (${seen.dx}, ${seen.dy}) from the agent, where the board puts it elsewhere`);
    }
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit): void {
    const shape = this.shape, side = shape.side, s = frame.state.step, f = this.frame(frame, s);
    if (!f) return;
    const words = f.words, view = f.view, rings = collapsedRings(shape, tickOf(words));
    const terrain = kit.marks('terrain', shape.cells), items = kit.marks('items', shape.cells), flames = kit.marks('flames', shape.cells);
    for (let c = 0; c < shape.cells; c++) {
      const field = fieldOf(view, c), kind = terrainOf(field), item = itemOf(field), flame = flameOf(field);
      this.centre(c, this.at);
      if (kind) terrain.set(this.at.x - 0.3, 0.05, this.at.z - 0.3, 0.09, kind === WOOD ? 0xc08a4a : ringOf(shape, c % side, Math.floor(c / side)) < rings ? 0xd0503a : 0x8fa6c4);
      if (item) items.set(this.at.x + 0.3, 0.05, this.at.z - 0.3, hiddenOf(field) ? 0.06 : 0.12, ITEM_HUES[item]!);
      if (flame) flames.set(this.at.x, 0.05, this.at.z + 0.3, 0.04 + 0.04 * flame, HUE.flameMid);
    }
    const bombs = bombsOf(view, shape), fuses = kit.marks('bombs', shape.bombs), paths = kit.curve('kicked bombs', 3 * shape.bombs, 0xdfe6f0);
    for (const bomb of bombs) {
      this.centre(bomb.cell, this.at);
      fuses.set(this.at.x, 0.06, this.at.z, 0.05 + 0.015 * bomb.life, this.hue(words, bomb.owner));
      if (!bomb.dir) continue;
      paths.point(this.at.x, 0.07, this.at.z);
      paths.point(this.at.x + 0.5 * STEP_X[bomb.dir]!, 0.07, this.at.z + 0.5 * STEP_Y[bomb.dir]!);
      paths.point(this.at.x, 0.07, this.at.z);
    }
    const bodies = kit.marks('bombers', shape.seats);
    for (let seat = 0; seat < shape.seats; seat++) {
      this.centre(cellOf(words, seat), this.at);
      bodies.set(this.at.x, 0.08, this.at.z, aliveOf(words, seat) ? 0.18 : 0.08, aliveOf(words, seat) ? this.hue(words, seat) : 0x6a7080);
    }
    const me = seatOf(words), obs = f.observations, home = cellOf(words, me);
    this.centre(home, this.at);
    const sight = kit.curve('sight', 5, TOKENS.uiCyan), reach = shape.sight + 0.5;
    for (const [dx, dz] of [[-reach, -reach], [reach, -reach], [reach, reach], [-reach, reach], [-reach, -reach]] as const) sight.point(this.at.x + dx, 0.07, this.at.z + dz);
    const sightings = sightingsOf(obs, shape, me), reported = kit.marks('reported bombers', shape.seats - 1);
    for (const seen of sightings) reported.set(this.at.x + seen.dx, 0.09, this.at.z + seen.dy, seen.code > 0 ? 0.12 : 0.07, this.hue(words, seen.seat));
    const window = kit.curve('view window', 5, this.accent);
    if (shape.fog) {
      const x = home % side, y = Math.floor(home / side);
      const x0 = Math.max(0, x - shape.fog) - x - 0.5, x1 = Math.min(side - 1, x + shape.fog) - x + 0.5, z0 = Math.max(0, y - shape.fog) - y - 0.5, z1 = Math.min(side - 1, y + shape.fog) - y + 0.5;
      for (const [dx, dz] of [[x0, z0], [x1, z0], [x1, z1], [x0, z1], [x0, z0]] as const) window.point(this.at.x + dx, 0.07, this.at.z + dz);
    }
  }
}

export default (() => new PommermanScene()) satisfies SceneFactory;
