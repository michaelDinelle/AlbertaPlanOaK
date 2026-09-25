import * as THREE from 'three/webgpu';
import { constantList, constantRecords, parameter, type EnvDescriptor } from '../../data/envs';
import {
  FLAG_ATTACK, FLAG_DEAD, FLAG_SLOW, FLAG_STUN, FLAG_TAUNT, FLAG_WARD, KIND_CAMP, KIND_CREEP, KIND_HERO, KIND_TOWER,
  campHp, campX, campY, codeIndex, codeKind, creepArc, creepCount, creepFifoOf, creepHp, creepSlotOf, heroArch, heroFlags, heroGold, heroHp, heroLevel,
  heroMana, heroSlotOf, heroTarget, heroX, heroY, lanePoint, towerHp, towerTarget, type AncientsLayout,
} from '@sim/envs/ancients';
import type { EnvFrame } from '@sim/types';
import type { Board, Point } from './field';

export const FX_FLASH = 0;
export const FX_WAVE = 1;
export const FX_BOLT = 2;
export const FX_COIN = 3;
export const FX_BURST = 4;
export const FX_MARK = 5;

export const STREAK_TOWER = 0;
export const STREAK_BOLT = 3;
export const STREAK_TETHER = 4;

export interface Effect { kind: number; x0: number; z0: number; x1: number; z1: number; color: THREE.Color; size: number; start: number; life: number }
export interface Streak { style: number; x0: number; z0: number; x1: number; z1: number; color: THREE.Color; width: number; start: number; life: number }
export interface Fact { x: number; z: number; step: number }
export type FrameSource = (step: number) => EnvFrame | null;

export const SPELL = [0xff9a3c, 0xb98cff, 0xffe066, 0xd8cfbf, 0x7dffb0, 0xc07cff] as const;
const TOWER_JOBS = new Set([22, 23]);
const JOB_SLOTS = 20;
const FIFO = 16;

export class Tracker {
  readonly effects: Effect[] = [];
  readonly streaks: Streak[] = [];
  readonly ids: Int32Array[];
  readonly born = new Map<number, number>();
  readonly flash = new Map<number, number>();
  readonly heroFlash: number[];
  readonly structureFlash: number[];
  readonly campFlash: number[];
  readonly attackFrom: number[];
  readonly attackLand: number[];
  readonly snapped: boolean[];
  lastKill: Fact | null = null;
  lastAllyDeath: Fact | null = null;
  lastAgentDeath: Fact | null = null;
  lastTowerFall: { s: number; step: number } | null = null;
  lastHit: Fact | null = null;
  private nextId = 1;
  private readonly before: Int32Array[];
  private readonly hit = new Set<number>();
  private readonly dead = new Set<number>();
  private readonly jobs: number[];
  private readonly costs: number[][];
  private readonly substeps: number;
  private readonly instance: number;
  private readonly heroes: number;
  private readonly a: Point = { x: 0, z: 0 };
  private readonly b: Point = { x: 0, z: 0 };
  private readonly lane = { x: 0, y: 0 };

  constructor(env: EnvDescriptor, private readonly board: Board, private readonly layout: AncientsLayout) {
    this.jobs = constantList(env, 'ANC_JOBS');
    this.costs = constantRecords(env, 'ANC_ARCHETYPES').map((record) => record.cost as number[]);
    this.substeps = parameter(env, 'substeps');
    this.instance = env.instance;
    this.heroes = 2 * layout.heroes;
    this.ids = Array.from({ length: 2 * layout.laneSlots }, () => new Int32Array(FIFO));
    this.before = Array.from({ length: 2 * layout.laneSlots }, () => new Int32Array(FIFO));
    this.heroFlash = new Array(this.heroes).fill(-Infinity);
    this.structureFlash = new Array(2 * layout.own).fill(-Infinity);
    this.campFlash = new Array(3 * layout.camps).fill(-Infinity);
    this.attackFrom = new Array(this.heroes).fill(NaN);
    this.attackLand = new Array(this.heroes).fill(NaN);
    this.snapped = new Array(this.heroes).fill(true);
  }

  tickTime(frame: EnvFrame): number {
    return frame.words[0]! + frame.words[1]! / this.substeps;
  }

  live(fifo: number): boolean {
    return fifo % this.layout.laneSlots < this.layout.lanes;
  }

  creepPoint(view: Uint32Array, fifo: number, k: number, out: Point): Point {
    lanePoint(this.board.shape, fifo, creepArc(view, this.layout, fifo, k), this.lane);
    return this.board.at(this.lane.x, this.lane.y, out);
  }

  heroPoint(view: Uint32Array, slot: number, out: Point): Point {
    return this.board.at(heroX(view, slot), heroY(view, slot), out);
  }

  locate(view: Uint32Array, code: number, out: Point): boolean {
    const kind = codeKind(code), index = codeIndex(code), layout = this.layout;
    if (kind === KIND_HERO) {
      const slot = heroSlotOf(layout, index);
      if (slot >= this.heroes || heroFlags(view, slot) & FLAG_DEAD) return false;
      this.heroPoint(view, slot, out);
      return true;
    }
    if (kind === KIND_CREEP) {
      const fifo = creepFifoOf(code), k = creepSlotOf(code);
      if (fifo >= this.ids.length || !this.live(fifo) || k >= creepCount(view, layout, fifo)) return false;
      this.creepPoint(view, fifo, k, out);
      return true;
    }
    if (kind === KIND_TOWER) {
      if (index >= 2 * layout.own || towerHp(view, layout, index) === 0) return false;
      this.board.at(this.board.structureX[index]!, this.board.structureY[index]!, out);
      return true;
    }
    if (kind === KIND_CAMP && layout.jungle) {
      const camp = index & 3;
      this.board.at(campX(view, layout, camp), campY(view, layout, camp), out);
      return true;
    }
    return false;
  }

  resync(frame: EnvFrame, time: number, source: FrameSource): void {
    this.born.clear();
    for (let fifo = 0; fifo < this.ids.length; fifo++) {
      if (!this.live(fifo)) continue;
      const count = creepCount(frame.view, this.layout, fifo);
      for (let k = 0; k < count; k++) {
        const id = this.nextId++;
        this.ids[fifo]![k] = id;
        this.born.set(id, creepArc(frame.view, this.layout, fifo, k) >= 0 ? time - 10 : Infinity);
      }
    }
    this.effects.length = 0;
    this.streaks.length = 0;
    this.flash.clear();
    this.heroFlash.fill(-Infinity);
    this.structureFlash.fill(-Infinity);
    this.campFlash.fill(-Infinity);
    this.snapped.fill(true);
    for (let slot = 0; slot < this.heroes; slot++) {
      this.attackFrom[slot] = NaN;
      this.attackLand[slot] = NaN;
      if (!(heroFlags(frame.view, slot) & FLAG_ATTACK)) continue;
      this.origin(slot, frame.step, source);
      this.landing(slot, frame.step, source);
    }
  }

  private origin(slot: number, from: number, source: FrameSource): void {
    for (let step = from; step > Math.max(0, from - 16 * this.substeps); step--) {
      const here = source(step), earlier = source(step - 1);
      if (!here) return;
      if (here.done || !earlier || !(heroFlags(earlier.view, slot) & FLAG_ATTACK)) {
        this.attackFrom[slot] = this.tickTime(here);
        return;
      }
    }
  }

  landing(slot: number, from: number, source: FrameSource): void {
    for (let step = from + 1; step <= from + 16 * this.substeps; step++) {
      const frame = source(step);
      if (!frame || frame.done) return;
      if (!(heroFlags(frame.view, slot) & FLAG_ATTACK)) {
        this.attackLand[slot] = this.tickTime(frame);
        return;
      }
    }
  }

  private effect(kind: number, x0: number, z0: number, x1: number, z1: number, color: THREE.ColorRepresentation, size: number, start: number, life: number): void {
    if (this.effects.length >= 160) this.effects.shift();
    this.effects.push({ kind, x0, z0, x1, z1, color: new THREE.Color(color), size, start, life });
  }

  private streak(style: number, x0: number, z0: number, x1: number, z1: number, color: THREE.ColorRepresentation, width: number, start: number, life: number): void {
    if (this.streaks.length >= 48) this.streaks.shift();
    this.streaks.push({ style, x0, z0, x1, z1, color: new THREE.Color(color), width, start, life });
  }

  prune(time: number): void {
    const keep = <T extends { start: number; life: number }>(list: T[]) => {
      let n = 0;
      for (const item of list) if (time - item.start < item.life) list[n++] = item;
      list.length = n;
    };
    keep(this.effects);
    keep(this.streaks);
  }

  advance(prev: EnvFrame, next: EnvFrame, time: number, teams: readonly [number, number], source: FrameSource): void {
    const view = next.view, before = prev.view, layout = this.layout, a = this.a, b = this.b;
    const job = this.jobs[this.instance * JOB_SLOTS + prev.words[1]!] ?? 0;
    this.hit.clear();
    this.dead.clear();

    for (let fifo = 0; fifo < this.ids.length; fifo++) {
      if (!this.live(fifo)) continue;
      const ids = this.ids[fifo]!, old = this.before[fifo]!, na = creepCount(before, layout, fifo), nb = creepCount(view, layout, fifo);
      old.set(ids);
      let j = 0;
      for (let k = 0; k < nb; k++) {
        const arc = creepArc(view, layout, fifo, k), hp = creepHp(view, layout, fifo, k);
        while (j < na && (creepArc(before, layout, fifo, j) > arc || creepHp(before, layout, fifo, j) < hp)) this.creepDies(before, fifo, j++, time);
        if (j < na) {
          ids[k] = old[j]!;
          if (hp < creepHp(before, layout, fifo, j)) {
            this.flash.set(ids[k]!, time);
            this.hit.add(ids[k]!);
          }
          j++;
        } else {
          ids[k] = this.nextId++;
          this.born.set(ids[k]!, Infinity);
        }
        if (arc >= 0 && this.born.get(ids[k]!) === Infinity) this.born.set(ids[k]!, time);
      }
      while (j < na) this.creepDies(before, fifo, j++, time);
    }

    for (let slot = 0; slot < this.heroes; slot++) {
      const side = slot < layout.heroes ? 0 : 1, was = heroFlags(before, slot), now = heroFlags(view, slot), color = teams[side];
      this.heroPoint(before, slot, a);
      this.heroPoint(view, slot, b);
      if (!(was & FLAG_DEAD) && now & FLAG_DEAD) {
        this.effect(FX_BURST, a.x, a.z, a.x, a.z, color, 2.6, time, 0.8);
        this.effect(FX_FLASH, a.x, a.z, a.x, a.z, color, 2.2, time, 0.5);
        const fact = { x: a.x, z: a.z, step: next.step };
        if (slot === 0) this.lastAgentDeath = fact;
        else if (side === 0) this.lastAllyDeath = fact;
        else this.lastKill = fact;
        this.snapped[slot] = true;
        this.attackFrom[slot] = NaN;
        continue;
      }
      if (was & FLAG_DEAD && !(now & FLAG_DEAD)) {
        this.effect(FX_WAVE, b.x, b.z, b.x, b.z, color, 3.2, time, 0.9);
        this.snapped[slot] = true;
      }
      if (now & FLAG_DEAD) continue;
      if (Math.hypot(b.x - a.x, b.z - a.z) > 1.2) {
        this.streak(STREAK_BOLT, a.x, a.z, b.x, b.z, SPELL[heroArch(view, slot)]!, 0.5, time, 0.45);
        this.effect(FX_FLASH, a.x, a.z, a.x, a.z, SPELL[heroArch(view, slot)]!, 1.6, time, 0.4);
        this.snapped[slot] = true;
      }
      if (heroLevel(view, slot) > heroLevel(before, slot)) this.effect(FX_WAVE, b.x, b.z, b.x, b.z, 0xffd166, 2.4, time, 0.8);
      const gain = heroGold(view, slot) - heroGold(before, slot);
      if (gain > 1) this.effect(FX_COIN, b.x, b.z - 0.4, b.x, b.z - 1.5, 0xffcf4a, Math.min(0.95, 0.5 + gain / 400), time, 0.9);
      if (heroHp(view, slot) < heroHp(before, slot)) this.heroFlash[slot] = time;
      if (!(was & FLAG_ATTACK) && now & FLAG_ATTACK) {
        this.attackFrom[slot] = this.tickTime(next);
        this.attackLand[slot] = NaN;
        this.landing(slot, next.step, source);
      }
      if (was & FLAG_ATTACK && !(now & FLAG_ATTACK)) {
        const code = heroTarget(before, slot);
        if (this.struck(before, view, code, true) && this.locate(before, code, a)) {
          this.effect(FX_FLASH, a.x, a.z, a.x, a.z, side === 0 ? 0xfff1c9 : 0xffd0c8, 1, time, 0.3);
          if (slot === 0) this.lastHit = { x: a.x, z: a.z, step: next.step };
        }
      }
      const cost = this.costs[heroArch(view, slot)]!, spent = heroMana(before, slot) - heroMana(view, slot);
      if (spent >= Math.min(cost[0]!, cost[1]!) - 1) this.cast(before, view, slot, side, time);
    }

    for (let s = 0; s < 2 * layout.own; s++) {
      const was = towerHp(before, layout, s), now = towerHp(view, layout, s), color = teams[s < layout.own ? 0 : 1];
      if (now < was) this.structureFlash[s] = time;
      this.board.at(this.board.structureX[s]!, this.board.structureY[s]!, a);
      if (was > 0 && now === 0) {
        this.effect(FX_BURST, a.x, a.z, a.x, a.z, color, 4.2, time, 1.1);
        this.effect(FX_WAVE, a.x, a.z, a.x, a.z, 0xd9c7a0, 5.5, time, 1.2);
        this.effect(FX_FLASH, a.x, a.z, a.x, a.z, color, 4, time, 0.8);
        this.lastTowerFall = { s, step: next.step };
      }
      if (!TOWER_JOBS.has(job) || was === 0) continue;
      const after = towerTarget(view, layout, s), code = after || towerTarget(before, layout, s);
      if (!code || !this.struck(before, view, code, after === 0) || !this.locate(after ? view : before, code, b)) continue;
      this.streak(STREAK_TOWER, a.x, a.z, b.x, b.z, color, 0.36, time, 0.3);
      this.effect(FX_FLASH, b.x, b.z, b.x, b.z, color, 1.3, time, 0.3);
    }

    if (layout.jungle) for (let i = 0; i < layout.camps; i++) for (let k = 0; k < 3; k++) {
      const was = campHp(before, layout, i, k), now = campHp(view, layout, i, k);
      if (now < was) this.campFlash[3 * i + k] = time;
      if (was > 0 && now === 0) {
        this.board.at(campX(before, layout, i), campY(before, layout, i), a);
        this.effect(FX_BURST, a.x, a.z, a.x, a.z, 0xe0a646, 1.4, time, 0.6);
      }
    }
  }

  fall(s: number, color: number, time: number): void {
    this.board.at(this.board.structureX[s]!, this.board.structureY[s]!, this.a);
    const { x, z } = this.a;
    this.effect(FX_BURST, x, z, x, z, color, 7, time, 1.6);
    this.effect(FX_WAVE, x, z, x, z, color, 9, time, 1.8);
    this.effect(FX_FLASH, x, z, x, z, color, 6, time, 1.2);
  }

  emit(kind: number, x: number, z: number, color: number, size: number, time: number, life: number): void {
    this.effect(kind, x, z, x, z, color, size, time, life);
  }

  private struck(before: Uint32Array, view: Uint32Array, code: number, earlier: boolean): boolean {
    const kind = codeKind(code), index = codeIndex(code), layout = this.layout;
    if (kind === KIND_HERO) {
      const slot = heroSlotOf(layout, index);
      return slot < this.heroes && heroHp(view, slot) < heroHp(before, slot);
    }
    if (kind === KIND_CREEP) {
      const fifo = creepFifoOf(code), k = creepSlotOf(code);
      if (fifo >= this.ids.length || !this.live(fifo)) return false;
      const id = (earlier ? this.before : this.ids)[fifo]![k]!;
      return this.hit.has(id) || this.dead.has(id);
    }
    if (kind === KIND_TOWER) return index < 2 * layout.own && towerHp(view, layout, index) < towerHp(before, layout, index);
    if (kind === KIND_CAMP && layout.jungle) {
      const camp = index & 3;
      return [0, 1, 2].some((k) => campHp(view, layout, camp, k) < campHp(before, layout, camp, k));
    }
    return false;
  }

  private creepDies(view: Uint32Array, fifo: number, k: number, time: number): void {
    const id = this.before[fifo]![k]!;
    this.born.delete(id);
    this.flash.delete(id);
    this.dead.add(id);
    if (creepArc(view, this.layout, fifo, k) < 0) return;
    this.creepPoint(view, fifo, k, this.a);
    this.effect(FX_BURST, this.a.x, this.a.z, this.a.x, this.a.z, fifo < this.layout.laneSlots ? 0x9ff5cf : 0xffb3a8, 1.1, time, 0.45);
  }

  private cast(before: Uint32Array, view: Uint32Array, slot: number, side: number, time: number): void {
    const layout = this.layout, color = SPELL[heroArch(view, slot)]!, a = this.a, b = this.b;
    this.heroPoint(view, slot, a);
    this.effect(FX_WAVE, a.x, a.z, a.x, a.z, color, 3, time, 0.6);
    this.effect(FX_FLASH, a.x, a.z, a.x, a.z, color, 2.4, time, 0.5);
    for (let other = 0; other < this.heroes; other++) {
      if (other === slot || heroFlags(view, other) & FLAG_DEAD) continue;
      const ally = (other < layout.heroes ? 0 : 1) === side, gained = heroFlags(view, other) & ~heroFlags(before, other);
      const touched = ally ? heroHp(view, other) > heroHp(before, other) + 1 || (gained & FLAG_WARD) !== 0 : heroHp(view, other) < heroHp(before, other) || (gained & (FLAG_STUN | FLAG_SLOW | FLAG_TAUNT)) !== 0;
      if (!touched) continue;
      this.heroPoint(view, other, b);
      this.streak(ally ? STREAK_TETHER : STREAK_BOLT, a.x, a.z, b.x, b.z, color, ally ? 0.42 : 0.55, time, 0.5);
      this.effect(FX_FLASH, b.x, b.z, b.x, b.z, color, 1.8, time, 0.45);
    }
    for (let fifo = 0; fifo < this.ids.length; fifo++) {
      if (!this.live(fifo) || (fifo < layout.laneSlots ? 0 : 1) === side) continue;
      const count = creepCount(view, layout, fifo);
      for (let k = 0; k < count; k++) if (this.hit.has(this.ids[fifo]![k]!)) {
        this.creepPoint(view, fifo, k, b);
        this.effect(FX_FLASH, b.x, b.z, b.x, b.z, color, 1.2, time, 0.4);
      }
    }
  }
}
