import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import type { Light2D } from '@kit/light2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  ANC_X, ANC_Y, ARCHETYPES, EV_ACQUIRED, EV_ALLY, EV_DEATH, EV_KILL, EV_LASTHIT, EV_LEVEL, EV_RESPAWN, EV_TOWER, FLAG_ATTACK, FLAG_DEAD,
  ancientsDecoder, ancientsLayout, campAggro, campHp, campX, campY, creepArc, creepCount, heroArch, heroFlags, heroGold, heroHp, heroLevel, heroMaxHp, heroTarget, heroX, heroY,
  laneHeading, leagueMember, memoryAge, memoryX, memoryY, towerHp, towerTarget, type AncientsLayout,
} from '@sim/envs/ancients';
import type { EnvFrame, SimEvent } from '@sim/types';
import { constantRecords, parameter } from '../../data/envs';
import { formatCount } from '../../format';
import { Board, buildFountains, buildGround, type Point } from './field';
import {
  ANCIENT_RADIUS, BADGE_DEPTH, BADGE_WIDTH, CREEP_RADIUS, CREEP_SPAN, CRIMSON, DIGIT_MASKS, HERO_RADIUS, HERO_SPAN, JADE, PLATE_DEPTH, PLATE_WIDTH, STRUCTURE_SPAN, SpriteBatch, TAUNT, TOWER_RADIUS,
  badgeShade, beamShade, creepShade, effectShade, heroShade, plateShade, structureShade,
} from './sprites';
import { FX_BOLT, FX_BURST, FX_FLASH, FX_MARK, FX_WAVE, Tracker, type FrameSource } from './track';

const TICKS_PER_SECOND = 10;
const MAX_SCAN = 600;
const REPLAY_SECONDS = 1.6;
const GLIDE = 16;
const GOLD = 0xffd166;
const TRUTH = 0x5ec1ff;
const TEAMS = [JADE, CRIMSON] as const;
const LIGHT_TEAMS = [0x52f0b0, 0xff5a44] as const;
const CAMP_OFFSETS = [[0, -0.36], [-0.34, 0.22], [0.34, 0.22]] as const;
const BADGE_SCALE = 1.3;
const SPIRIT_RING = 1.45;
const SPIRIT_SCALE = 0.72;
const STEER = 2.4;
const DEFEAT = 0x7a95c4;
const TEAM_COLORS = TEAMS.map((hex) => new THREE.Color(hex));
const GOLD_COLOR = new THREE.Color(GOLD);
const glow = (hex: number) => { const c = new THREE.Color(hex); return new THREE.Vector3(c.r, c.g, c.b); };
const TEAM_GLOW = LIGHT_TEAMS.map(glow);

function steering(act: Float32Array): [number, number] {
  let vx = Math.max(-1, Math.min(1, act[0]!)), vy = Math.max(-1, Math.min(1, act[1]!));
  const n = Math.hypot(vx, vy);
  if (n > 1) { vx /= n; vy /= n; }
  return [vx, vy];
}

class AncientsScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  controlHz = TICKS_PER_SECOND;
  protected readonly stageName = 'ancients';
  override legend: LegendSpec | null = null;

  private board!: Board;
  private layout!: AncientsLayout;
  private tracker!: Tracker;
  private ranged: boolean[] = [];
  private substeps = 1;
  private tickCap = 1;
  private batches: SpriteBatch[] = [];
  private heroes!: SpriteBatch;
  private badges!: SpriteBatch;
  private plates!: SpriteBatch;
  private creeps!: SpriteBatch;
  private structures!: SpriteBatch;
  private beams!: SpriteBatch;
  private effects!: SpriteBatch;
  private readonly clock = uniform(0);
  private last = -1;
  private lastTime = 0;
  private shown: Point[] = [];
  private readonly creepShown = new Map<number, Point & { seen: number }>();
  private generation = 0;
  private heroLights: { mesh: THREE.Mesh; tint: THREE.Vector3 }[] = [];
  private structureLights: { glow: THREE.Mesh; tint: THREE.Vector3; block: THREE.Mesh | null; s: number }[] = [];
  private flashLights: { mesh: THREE.Mesh; tint: THREE.Vector3 }[] = [];
  private campLights: { mesh: THREE.Mesh; tint: THREE.Vector3 }[] = [];
  private readonly scratch: Point = { x: 0, z: 0 };
  private readonly target: Point = { x: 0, z: 0 };
  private readonly celebration = new THREE.Vector3();
  private readonly ledger = { arch: 0, level: 1, hp: 0, maxHp: 0, dead: false, gold: 0, ours: 0, theirs: 0, wins: 0, losses: 0, draws: 0, game: 0, tick: 0, member: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.layout = ancientsLayout(env);
    this.board = new Board(env, this.layout);
    this.tracker = new Tracker(env, this.board, this.layout);
    this.substeps = parameter(env, 'substeps');
    this.tickCap = parameter(env, 'tickCap');
    this.controlHz = TICKS_PER_SECOND * this.substeps;
    this.ranged = constantRecords(env, 'ANC_ARCHETYPES').map((record) => record.ranged !== 0);
    this.shown = Array.from({ length: 2 * this.layout.heroes }, () => ({ x: 0, z: 0 }));
    return this.board.extent(2.2);
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xbac8ec).multiplyScalar(0.78);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighSize: 36, sighRadius: 1.4, sighHeight: 0.4 };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), board = this.board, layout = this.layout;
    this.scene.background = new THREE.Color(0x040507);
    pool.group.renderOrder = 20;
    this.scene.add(buildGround(board, light));
    const fountains = buildFountains(board, this.clock, light);
    this.batches.push(fountains);
    this.structures = new SpriteBatch('ancientsStructures', 2 * layout.own, STRUCTURE_SPAN, STRUCTURE_SPAN, ['aTower', 'aFx'], structureShade(this.clock, light.lit), 3);
    this.creeps = new SpriteBatch('ancientsCreeps', 2 * layout.laneSlots * 16 + 3 * layout.camps, 2 * CREEP_SPAN * CREEP_RADIUS, 2 * CREEP_SPAN * CREEP_RADIUS, ['aCreep'], creepShade, 4);
    this.beams = new SpriteBatch('ancientsBeams', 96, 1, 1, ['aBeam', 'aSpan'], beamShade(this.clock), 5, true);
    this.heroes = new SpriteBatch('ancientsHeroes', 2 * layout.heroes, 2 * HERO_SPAN * HERO_RADIUS, 2 * HERO_SPAN * HERO_RADIUS, ['aHero', 'aState', 'aKeys'], heroShade(this.clock, accent), 6);
    this.plates = new SpriteBatch('ancientsPlates', 2 * layout.own, PLATE_WIDTH, PLATE_DEPTH, ['aPlate'], plateShade, 7);
    this.badges = new SpriteBatch('ancientsBadges', 2 * layout.heroes, BADGE_WIDTH, BADGE_DEPTH, ['aBar', 'aBadge'], badgeShade(accent), 8);
    this.effects = new SpriteBatch('ancientsEffects', 256, 1, 1, ['aFx', 'aLife'], effectShade, 9, true);
    this.batches.push(this.structures, this.creeps, this.beams, this.heroes, this.plates, this.badges, this.effects);
    for (const batch of this.batches) this.scene.add(batch.mesh);
    this.buildLights(light, accent);
    this.buildLegend(accent);
  }

  private buildLights(light: Light2D, accent: number): void {
    const board = this.board, layout = this.layout;
    const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);
    const lamp = (radius: number) => {
      const tint = new THREE.Vector3();
      const mesh = light.emitter(disc(radius), uniform(tint));
      this.scene.add(mesh);
      return { mesh, tint };
    };
    for (let s = 0; s < 2 * layout.own; s++) {
      if (!board.exists(s)) continue;
      const ancient = s % layout.own === layout.own - 1, x = board.x(board.structureX[s]!), z = board.z(board.structureY[s]!);
      const crystal = lamp(ancient ? 0.42 : 0.2);
      crystal.mesh.position.set(x, 0, z);
      const block = ancient ? null : light.occluder(new THREE.PlaneGeometry(0.72, 0.72).rotateX(-Math.PI / 2));
      if (block) {
        block.position.set(x, 0, z);
        this.scene.add(block);
      }
      this.structureLights.push({ glow: crystal.mesh, tint: crystal.tint, block, s });
    }
    for (let side = 0; side < 2; side++) {
      const pool = lamp(0.34);
      pool.mesh.position.set(board.x(board.fountainX[side]!), 0, board.z(board.fountainY[side]!));
      pool.tint.copy(TEAM_GLOW[side]!).multiplyScalar(7);
    }
    for (let slot = 0; slot < 2 * layout.heroes; slot++) {
      const hero = lamp(slot === 0 ? 0.2 : 0.15);
      hero.tint.copy(slot === 0 ? glow(accent).multiplyScalar(12) : TEAM_GLOW[slot < layout.heroes ? 0 : 1]!.clone().multiplyScalar(4.5));
      this.heroLights.push(hero);
    }
    for (let i = 0; i < 6; i++) this.flashLights.push(lamp(0.3));
    if (layout.jungle) for (let i = 0; i < layout.camps; i++) {
      const camp = lamp(0.14);
      camp.tint.copy(glow(0xffa640).multiplyScalar(4));
      this.campLights.push(camp);
    }
  }

  private buildLegend(accent: number): void {
    const l = this.ledger;
    const rows: LegendSpec['rows'][number][] = [
      { key: 'hero', label: 'hero', tint: accent, value: () => `${ARCHETYPES[l.arch] ?? 'hero'} · lv ${l.level}` },
      { key: 'health', label: 'health', tint: 0x5fe38f, value: () => l.dead ? 'respawning' : `${formatCount(l.hp)} / ${formatCount(l.maxHp)}` },
      { key: 'gold', label: 'gold', tint: GOLD, value: () => formatCount(l.gold) },
      { key: 'towers', label: 'towers', tint: JADE, value: () => `${l.ours} vs ${l.theirs}` },
      { key: 'record', label: 'record', tint: 0xe6e2d6, value: () => `${l.wins} W · ${l.losses} L · ${l.draws} D` },
    ];
    if (parameter(this.ctx!.environment, 'league') !== 0) rows.push({ key: 'rival', label: 'rival', tint: CRIMSON, value: () => `league member ${l.member + 1}` });
    this.legend = {
      title: 'ANCIENTS',
      rows,
      note: () => `game ${formatCount(l.game + 1)} · tick ${formatCount(l.tick)} / ${formatCount(this.tickCap)}`,
      bar: () => l.tick / this.tickCap,
    };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.groundTruth(frame, kit) };
  }

  private source(frame: FrameUpdate): FrameSource {
    return (step) => {
      if (step < 0) return null;
      const found = frame.frameAt(step);
      return found.step === step ? found : null;
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const step = frame.state.step, source = this.source(frame), current = frame.frameAt(step);
    pool.update(frame.dtReal, time);
    this.clock.value = time;
    const gap = step - this.last;
    if (!(this.last >= 0 && gap >= 0 && gap <= MAX_SCAN && this.run(this.last, step, time, source, pool))) {
      pool.clear();
      let start = step;
      while (start > 0 && step - start < REPLAY_SECONDS * this.controlHz && source(start - 1)) if (source(start--)?.done) break;
      this.tracker.resync(source(start) ?? current, time - (step - start) / this.controlHz, source);
      if (!this.run(start, step, time, source, pool)) this.tracker.resync(current, time, source);
    }
    this.last = step;
    this.tracker.prune(time);
    const dt = Math.max(time - this.lastTime, frame.dtReal, 0);
    this.lastTime = time;
    this.draw(frame, current, time, dt, source);
  }

  private run(from: number, to: number, time: number, source: FrameSource, pool: CelebrationPool): boolean {
    for (let s = from + 1; s <= to; s++) {
      const prev = source(s - 1), next = source(s);
      if (!prev || !next) return false;
      const at = time - (to - s) / this.controlHz;
      if (next.done) this.tracker.resync(next, at, source);
      else this.tracker.advance(prev, next, at, TEAMS, source);
      for (const event of ancientsDecoder.deriveEvents(prev, next)) this.celebrate(event, next, pool, at);
    }
    return true;
  }

  private celebrate(event: SimEvent, frame: EnvFrame, pool: CelebrationPool, at: number): void {
    const board = this.board, tracker = this.tracker, spot = this.celebration;
    const place = (x: number, z: number) => spot.set(x, 0.05, z);
    const structure = (s: number) => place(board.x(board.structureX[s]!), board.z(board.structureY[s]!));
    const cheer = (color: number, size: number) => {
      tracker.emit(FX_WAVE, spot.x, spot.z, color, size * 1.6, at, 1.1);
      tracker.emit(FX_BURST, spot.x, spot.z, color, size, at, 0.9);
    };
    if (event.type === 'goal') {
      structure(board.ancient(1));
      pool.burst(spot, GOLD, 2.6, at);
      pool.ripple(spot, GOLD, 1.5, at);
      pool.pulse.fire(1);
      tracker.fall(board.ancient(1), GOLD, at);
    } else if (event.type === 'harm') {
      structure(board.ancient(0));
      pool.sigh(1, spot, at);
      pool.ripple(spot, DEFEAT, 1.2, at);
      tracker.fall(board.ancient(0), DEFEAT, at);
    } else if (event.type === 'done' && frame.reward === 0) {
      place(0, 0);
      pool.ripple(spot, 0xe6e2d6, 1.6, at);
    } else if (event.type === 'cue') {
      const fact = event.value === EV_KILL ? tracker.lastKill : event.value === EV_DEATH ? tracker.lastAgentDeath : event.value === EV_ALLY ? tracker.lastAllyDeath : event.value === EV_LASTHIT ? tracker.lastHit : null;
      if (fact) place(fact.x, fact.z);
      else place(board.x(heroX(frame.view, 0)), board.z(heroY(frame.view, 0)));
      if (event.value === EV_KILL) {
        pool.burst(spot, GOLD, 1.3, at);
        pool.ripple(spot, GOLD, 0.7, at);
        cheer(GOLD, 3.2);
      } else if (event.value === EV_DEATH) {
        pool.sigh(0.8, spot, at);
        cheer(DEFEAT, 3);
      } else if (event.value === EV_ALLY) {
        pool.sigh(0.4, spot, at);
        cheer(DEFEAT, 1.8);
      } else if (event.value === EV_LASTHIT) pool.burst(spot, GOLD, 0.45, at);
      else if (event.value === EV_LEVEL) pool.ripple(spot, GOLD, 0.55, at);
      else if (event.value === EV_ACQUIRED) {
        pool.ripple(spot, TAUNT, 0.45, at);
        cheer(TAUNT, 1.6);
      } else if (event.value === EV_RESPAWN) {
        place(board.x(board.fountainX[0]!), board.z(board.fountainY[0]!));
        pool.ripple(spot, JADE, 0.7, at);
      } else if (event.value === EV_TOWER && tracker.lastTowerFall) {
        const s = tracker.lastTowerFall.s;
        structure(s);
        if (s >= this.layout.own) {
          pool.burst(spot, GOLD, 1.4, at);
          pool.ripple(spot, GOLD, 0.9, at);
          cheer(GOLD, 3.4);
        } else {
          pool.sigh(0.6, spot, at);
          cheer(DEFEAT, 2.6);
        }
      }
    }
  }

  private glide(point: Point, x: number, z: number, snap: boolean, dt: number): void {
    if (snap) {
      point.x = x;
      point.z = z;
      return;
    }
    const k = 1 - Math.exp(-GLIDE * dt);
    point.x += (x - point.x) * k;
    point.z += (z - point.z) * k;
  }

  private draw(frame: FrameUpdate, current: EnvFrame, time: number, dt: number, source: FrameSource): void {
    const view = current.view, layout = this.layout, board = this.board, tracker = this.tracker, target = this.target, scratch = this.scratch;
    const nowTick = tracker.tickTime(current) + (frame.position - Math.floor(frame.position)) / this.substeps;
    const heroes = this.heroes, badges = this.badges, creeps = this.creeps, structures = this.structures, plates = this.plates, beams = this.beams, effects = this.effects;
    for (const batch of [heroes, badges, creeps, structures, plates, beams, effects]) batch.begin();

    const dead = [0, 0], fallen = [0, 0];
    for (let slot = 0; slot < 2 * layout.heroes; slot++) if (heroFlags(view, slot) & FLAG_DEAD) fallen[slot < layout.heroes ? 0 : 1]!++;
    for (let order = 1; order <= 2 * layout.heroes; order++) {
      const slot = order % (2 * layout.heroes);
      const side = slot < layout.heroes ? 0 : 1, flags = heroFlags(view, slot), gone = (flags & FLAG_DEAD) !== 0, point = this.shown[slot]!, light = this.heroLights[slot]!;
      if (gone) {
        const ring = dead[side]!;
        dead[side] = ring + 1;
        const angle = (side === 0 ? (-3 * Math.PI) / 4 : Math.PI / 4) + (ring - (fallen[side]! - 1) / 2) * 0.7;
        target.x = board.x(board.fountainX[side]!) + Math.cos(angle) * SPIRIT_RING;
        target.z = board.z(board.fountainY[side]!) - Math.sin(angle) * SPIRIT_RING;
      } else board.at(heroX(view, slot), heroY(view, slot), target);
      if (slot === 0 && !gone) {
        const px = board.x(frame.state.pose[ANC_X]!), pz = board.z(frame.state.pose[ANC_Y]!);
        if (Math.hypot(px - target.x, pz - target.z) < 1) { target.x = px; target.z = pz; }
      }
      this.glide(point, target.x, target.z, tracker.snapped[slot]! || slot === 0, dt);
      tracker.snapped[slot] = false;
      const veiled = layout.fog && side === 1 && !gone && memoryAge(view, layout, slot - layout.heroes) > 0;
      const flare = Math.max(0, 1 - (time - tracker.heroFlash[slot]!) / 0.18) * 0.8;
      const index = heroes.add(point.x, point.z, 0, gone ? SPIRIT_SCALE : 1);
      heroes.set('aHero', index, side, heroArch(view, slot), slot === 0 ? 1 : 0, gone ? 0.42 : veiled ? 0.5 : 1);
      heroes.set('aState', index, gone ? 0 : flags, veiled ? 1 : 0, gone ? 1 : 0, flare);
      if (slot === 0 && !gone) this.keys(frame, current, index);
      else heroes.set('aKeys', index, -1);
      light.mesh.visible = !gone && !veiled;
      light.mesh.position.set(point.x, 0, point.z);
      if (!gone) {
        const level = heroLevel(view, slot), mana = slot === 0 ? current.observations[1]! : -1;
        const badge = badges.add(point.x + 0.16, point.z - HERO_RADIUS - 0.5, 0, BADGE_SCALE);
        badges.set('aBar', badge, heroHp(view, slot), heroMaxHp(view, slot), mana, side);
        badges.set('aBadge', badge, level >= 10 ? DIGIT_MASKS[Math.floor(level / 10)]! : 0, DIGIT_MASKS[level % 10]!, slot === 0 ? 1 : 0, veiled ? 0.55 : 1);
      }
      if (!gone && flags & FLAG_ATTACK && tracker.locate(view, heroTarget(view, slot), scratch)) {
        if (!Number.isFinite(tracker.attackLand[slot]!)) tracker.landing(slot, current.step, source);
        const from = tracker.attackFrom[slot]!, land = tracker.attackLand[slot]!;
        const progress = Number.isFinite(from) && Number.isFinite(land) && land > from ? Math.min(1, Math.max(0, (nowTick - from) / (land - from))) : 0;
        const color = slot === 0 ? GOLD_COLOR : TEAM_COLORS[side]!;
        this.beam(point.x, point.z, scratch.x, scratch.z, color, 1, 0.3, 0.45);
        if (this.ranged[heroArch(view, slot)]) {
          const fx = effects.add(point.x + (scratch.x - point.x) * progress, point.z + (scratch.z - point.z) * progress, 0, 0.9);
          effects.set('aFx', fx, color.r * 2, color.g * 2, color.b * 2, FX_BOLT);
          effects.set('aLife', fx, 0, 1);
        }
      }
    }

    const live = ++this.generation;
    for (let fifo = 0; fifo < tracker.ids.length; fifo++) {
      if (!tracker.live(fifo)) continue;
      const side = fifo < layout.laneSlots ? 0 : 1, count = creepCount(view, layout, fifo);
      for (let k = 0; k < count; k++) {
        const arc = creepArc(view, layout, fifo, k);
        if (arc < 0) continue;
        const id = tracker.ids[fifo]![k]!;
        tracker.creepPoint(view, fifo, k, target);
        let shown = this.creepShown.get(id);
        if (!shown) {
          shown = { x: target.x, z: target.z, seen: live };
          this.creepShown.set(id, shown);
        }
        this.glide(shown, target.x, target.z, false, dt);
        shown.seen = live;
        const born = tracker.born.get(id) ?? -Infinity, hit = tracker.flash.get(id) ?? -Infinity;
        const index = creeps.add(shown.x, shown.z, -laneHeading(board.shape, fifo, arc));
        creeps.set('aCreep', index, side, Math.max(0, 1 - (time - hit) / 0.14), Math.min(1, Math.max(0, (time - born) / 0.3)), 0);
      }
    }
    for (const [id, shown] of this.creepShown) if (shown.seen !== live) this.creepShown.delete(id);

    if (layout.jungle) for (let i = 0; i < layout.camps; i++) {
      const x = board.x(campX(view, layout, i)), z = board.z(campY(view, layout, i)), angry = campAggro(view, layout, i) !== 0 ? 1 : 0;
      let alive = 0;
      for (let k = 0; k < 3; k++) {
        if (campHp(view, layout, i, k) === 0) continue;
        alive++;
        const index = creeps.add(x + CAMP_OFFSETS[k]![0], z + CAMP_OFFSETS[k]![1], 0, 1.25);
        creeps.set('aCreep', index, angry, Math.max(0, 1 - (time - tracker.campFlash[3 * i + k]!) / 0.14), 1, 1);
      }
      const lamp = this.campLights[i]!;
      lamp.mesh.visible = alive > 0;
      lamp.mesh.position.set(x, 0, z);
    }

    let ours = 0, theirs = 0;
    for (const lamp of this.structureLights) {
      const s = lamp.s, side = s < layout.own ? 0 : 1, ancient = s % layout.own === layout.own - 1, hp = towerHp(view, layout, s), radius = ancient ? ANCIENT_RADIUS : TOWER_RADIUS;
      const x = board.x(board.structureX[s]!), z = board.z(board.structureY[s]!), standing = hp > 0;
      if (standing && !ancient) side === 0 ? ours++ : theirs++;
      const index = structures.add(x, z, 0, radius);
      structures.set('aTower', index, side, ancient ? 1 : 0, hp / board.structureHp[s % layout.own]!, standing ? 1 : 0);
      structures.set('aFx', index, Math.max(0, 1 - (time - tracker.structureFlash[s]!) / 0.2));
      lamp.glow.visible = standing;
      lamp.tint.copy(TEAM_GLOW[side]!).multiplyScalar((ancient ? 10 : 7) * (0.55 + 0.45 * hp / board.structureHp[s % layout.own]!));
      if (lamp.block) lamp.block.visible = standing;
      if (!standing) continue;
      const plate = plates.add(x, z - radius - 0.36);
      plates.set('aPlate', plate, hp / board.structureHp[s % layout.own]!, side, ancient ? 0.85 : 0.52, 1);
      const code = towerTarget(view, layout, s);
      if (code && tracker.locate(view, code, scratch)) this.beam(x, z, scratch.x, scratch.z, TEAM_COLORS[side]!, 1, 0.36, 0.75);
    }

    if (layout.fog) for (let i = 0; i < layout.heroes; i++) {
      const age = memoryAge(view, layout, i);
      if (age === 0 || heroFlags(view, layout.heroes + i) & FLAG_DEAD) continue;
      const fx = effects.add(board.x(memoryX(view, layout, i)), board.z(memoryY(view, layout, i)), 0, 1.9);
      effects.set('aFx', fx, 1.2, 0.35, 0.3, FX_MARK);
      effects.set('aLife', fx, 0, Math.max(0.3, 1 - age / 900));
    }

    if (!(heroFlags(view, 0) & FLAG_DEAD)) {
      const [vx, vy] = steering(current.actions), agent = this.shown[0]!;
      if (Math.hypot(vx, vy) * STEER > 0.15) this.beam(agent.x, agent.z, agent.x + vx * STEER, agent.z + vy * STEER, GOLD_COLOR, 2, 0.5, 0.85);
    }

    for (const streak of tracker.streaks) {
      const t = (time - streak.start) / streak.life;
      this.beam(streak.x0, streak.z0, streak.x1, streak.z1, streak.color, streak.style, streak.width, (1 - t) * (1 - t) * 1.2);
    }

    let flashes = 0;
    for (let i = tracker.effects.length - 1; i >= 0; i--) {
      const e = tracker.effects[i]!, t = (time - e.start) / e.life;
      if (t < 0 || t >= 1) continue;
      const ease = 1 - (1 - t) * (1 - t), fx = effects.add(e.x0 + (e.x1 - e.x0) * ease, e.z0 + (e.z1 - e.z0) * ease, 0, e.size);
      effects.set('aFx', fx, e.color.r * 1.6, e.color.g * 1.6, e.color.b * 1.6, e.kind);
      effects.set('aLife', fx, t, 1);
      if (e.kind === FX_FLASH && e.size >= 2 && flashes < this.flashLights.length) {
        const lamp = this.flashLights[flashes++]!;
        lamp.mesh.visible = true;
        lamp.mesh.position.set(e.x0, 0, e.z0);
        lamp.tint.set(e.color.r, e.color.g, e.color.b).multiplyScalar(6 * (1 - t) * (1 - t));
      }
    }
    for (let i = flashes; i < this.flashLights.length; i++) this.flashLights[i]!.mesh.visible = false;

    for (const batch of [heroes, badges, creeps, structures, plates, beams, effects]) batch.commit();

    const l = this.ledger;
    l.arch = heroArch(view, 0);
    l.level = heroLevel(view, 0);
    l.hp = heroHp(view, 0);
    l.maxHp = heroMaxHp(view, 0);
    l.dead = (heroFlags(view, 0) & FLAG_DEAD) !== 0;
    l.gold = heroGold(view, 0);
    l.ours = ours;
    l.theirs = theirs;
    l.game = current.words[2]!;
    l.wins = current.words[3]!;
    l.losses = current.words[4]!;
    l.draws = current.words[5]!;
    l.tick = current.words[0]!;
    l.member = leagueMember(view);
  }

  private keys(frame: FrameUpdate, current: EnvFrame, index: number): void {
    const pressed = [0, 0, 0];
    let count = 0;
    for (let step = current.step; step > current.step - this.substeps && step >= 0; step--) {
      const act = frame.frameAt(step);
      if (act.step !== step || (step < current.step && act.done)) break;
      pressed[0]! += act.actions[2]! > 0 ? 1 : 0;
      pressed[1]! += act.actions[4]! > 0 ? 1 : 0;
      pressed[2]! += act.actions[5]! > 0 ? 1 : 0;
      count++;
    }
    const aim = current.actions[3]!, kind = aim < -1 / 3 ? 0 : aim > 1 / 3 ? 2 : 1;
    this.heroes.set('aKeys', index, pressed[0]! / Math.max(count, 1), kind, pressed[1]! / Math.max(count, 1), pressed[2]! / Math.max(count, 1));
  }

  private beam(x0: number, z0: number, x1: number, z1: number, color: THREE.Color, style: number, width: number, alpha: number): void {
    const dx = x1 - x0, dz = z1 - z0, length = Math.hypot(dx, dz);
    if (length < 0.05 || alpha <= 0) return;
    const index = this.beams.add((x0 + x1) / 2, (z0 + z1) / 2, Math.atan2(-dz, dx), length, width);
    this.beams.set('aBeam', index, color.r * 1.6, color.g * 1.6, color.b * 1.6, style);
    this.beams.set('aSpan', index, length, width, (x0 * 7.1 + z0 * 3.7) % 6.28, Math.min(1, alpha));
  }

  private groundTruth(frame: FrameUpdate, kit: DebugKit): void {
    const current = frame.frameAt(frame.state.step), view = current.view, layout = this.layout, board = this.board, tracker = this.tracker, at = this.scratch, target = this.target;
    const marks = kit.marks('ancientsUnits', 2 * layout.heroes + 2 * layout.laneSlots * 16 + 2 * layout.own + layout.camps + layout.heroes);
    for (let l = 0; l < layout.lanes; l++) {
      const curve = kit.curve(`ancientsLane${l}`, 3, TRUTH);
      for (const point of board.lanePath(l)) curve.point(point.x, 0.02, point.z);
    }
    for (let slot = 0; slot < 2 * layout.heroes; slot++) {
      const flags = heroFlags(view, slot), aim = kit.curve(`ancientsAim${slot}`, 2, slot === 0 ? GOLD : 0xffffff);
      board.at(heroX(view, slot), heroY(view, slot), at);
      marks.set(at.x, 0.05, at.z, flags & FLAG_DEAD ? 0.14 : 0.3, slot === 0 ? GOLD : slot < layout.heroes ? 0xb8ffe4 : 0xffc2bb);
      if (!(flags & FLAG_DEAD) && tracker.locate(view, heroTarget(view, slot), target)) {
        aim.point(at.x, 0.05, at.z);
        aim.point(target.x, 0.05, target.z);
      }
    }
    for (let fifo = 0; fifo < tracker.ids.length; fifo++) {
      if (!tracker.live(fifo)) continue;
      for (let k = 0; k < creepCount(view, layout, fifo); k++) {
        if (creepArc(view, layout, fifo, k) < 0) continue;
        tracker.creepPoint(view, fifo, k, at);
        marks.set(at.x, 0.05, at.z, 0.12, TRUTH);
      }
    }
    for (let s = 0; s < 2 * layout.own; s++) {
      const aim = kit.curve(`ancientsTower${s}`, 2, TRUTH);
      if (!board.exists(s) || towerHp(view, layout, s) === 0) continue;
      board.at(board.structureX[s]!, board.structureY[s]!, at);
      marks.set(at.x, 0.05, at.z, 0.36, TRUTH);
      if (tracker.locate(view, towerTarget(view, layout, s), target)) {
        aim.point(at.x, 0.05, at.z);
        aim.point(target.x, 0.05, target.z);
      }
    }
    if (layout.jungle) for (let i = 0; i < layout.camps; i++) {
      if (![0, 1, 2].some((k) => campHp(view, layout, i, k) > 0)) continue;
      board.at(campX(view, layout, i), campY(view, layout, i), at);
      marks.set(at.x, 0.05, at.z, 0.24, 0xffe0a0);
    }
    if (layout.fog) for (let i = 0; i < layout.heroes; i++) {
      if (memoryAge(view, layout, i) === 0) continue;
      board.at(memoryX(view, layout, i), memoryY(view, layout, i), at);
      marks.set(at.x, 0.05, at.z, 0.2, 0xff7a70);
    }
    const move = kit.curve('ancientsMove', 2, GOLD), [vx, vy] = steering(current.actions);
    board.at(heroX(view, 0), heroY(view, 0), at);
    if (!(heroFlags(view, 0) & FLAG_DEAD)) {
      move.point(at.x, 0.05, at.z);
      move.point(at.x + vx * STEER, 0.05, at.z + vy * STEER);
    }
  }

  protected override drop(): void {
    for (const batch of this.batches) batch.dispose();
    this.batches = [];
  }
}

export default (() => new AncientsScene()) satisfies SceneFactory;
