import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { FULL, INTERACT, ITEM, OC_X, OC_Y, STAY, STEP_X, STEP_Y, TILE, actOf, aheadX, aheadY, cellOf, chanceOf, chefOf, clockOf, counterOf, facingOf, gameOf, heldOf, kitchenOf, onionsIn, potOf, stillOf, tileAt, xOf, yOf, type Kitchen } from '@sim/envs/overcooked';
import type { EnvFrame, GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { ACCENT, BROTH, CHEF, COMMIS, FLAME, GOLD, LAMP, LAMP_AT, ONION, PIECE, PLATE, chefMaterial, crateMaterial, flat, groundMaterial, itemMaterial, passMaterial, potMaterial, rackMaterial, type Level } from './art';

type Kind = 'give' | 'take' | 'fill';

interface Sheet { mesh: THREE.InstancedMesh; attributes: THREE.InstancedBufferAttribute[]; used: number }
interface Transfer { chef: number; kind: Kind; item: number; x: number; y: number; code: number }
interface Pass { x: number; y: number; angle: number; lamp: Level; served: number }
interface Pot { x: number; y: number; flame: Level; plopped: number; readied: number }

const ANGLE = [0, Math.PI, Math.PI / 2, -Math.PI / 2] as const;
const ACTIONS = ['↑ up', '↓ down', '← left', '→ right', 'stay', 'interact'] as const;
const HELD = ['empty-handed', 'onion', 'dish', 'soup'] as const;
const TINT = [0x8a8a8a, ONION, PLATE, BROTH] as const;
const LAND = 0.55;
const HAND = 0.35;
const SIZE = 1.35;
const FLARE_LIFE = 1.3;
const BELL_LIFE = 0.9;
const PLOP_LIFE = 0.5;
const PUFF_LIFE = 2.4;

const turn = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const signed = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace('-', '−');
const fade = (age: number, life: number) => (age >= 0 && age < life ? 1 - age / life : 0);
const steamy = (item: number) => (item === ITEM.SOUP ? 1 : 0);

class OvercookedScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'overcookedAI';

  private kitchen!: Kitchen;
  private grid: GridSize = { w: 5, h: 4 };
  private readonly time = uniform(0);
  private pots: Sheet | null = null;
  private passes: Sheet | null = null;
  private placed: Sheet | null = null;
  private chefs: Sheet | null = null;
  private carried: Sheet | null = null;
  private readonly stoves: Pot[] = [];
  private readonly windows: Pass[] = [];
  private textures: THREE.Texture[] = [];

  private step = -1;
  private fired = -1;
  private clock = 0;
  private cache: { from: number; to: number; transfers: Transfer[] } = { from: -1, to: -1, transfers: [] };
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly shown = new THREE.Vector3();
  private readonly hands = new THREE.Vector3();
  private readonly back = new THREE.Vector3();
  private readonly ledger = { step: -1, held: [0, 0], act: [-1, -1], blocked: [false, false], chance: 0, still: 0, pots: [0, 0], served: 0, reward: 0, clock: 0, game: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const kitchen = kitchenOf(ctx.environment);
    this.kitchen = kitchen;
    this.grid = { w: kitchen.w, h: kitchen.h };
    const halfW = kitchen.w / 2 + 0.3, halfH = kitchen.h / 2 + 0.3;
    return new THREE.Box2(new THREE.Vector2(-halfW, -halfH), new THREE.Vector2(halfW, halfH + 0.1 * kitchen.h));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff5e8).multiplyScalar(0.78);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      wallColor: 0x3b5b86,
      agentCell: (state) => {
        agent.x = state.pose[OC_X]!;
        agent.y = state.pose[OC_Y]!;
        return agent;
      },
      debug: (frame, kit) => this.debugDraw(frame, kit),
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const kitchen = this.kitchen, { w, tiles } = kitchen, lit = light.lit;
    const ground = groundMaterial(kitchen, lit);
    this.textures.push(...ground.textures);
    const floor = new THREE.Mesh(flat(80, 80), ground.material);
    floor.renderOrder = 0;
    floor.frustumCulled = false;
    this.scene.add(floor);

    const cells = (code: number) => [...tiles.keys()].filter((cell) => tiles[cell] === code).map((cell) => ({ x: cell % w, y: Math.floor(cell / w) }));
    const fixture = (material: THREE.Material, code: number) => {
      const spots = cells(code);
      const mesh = instancedMesh(flat(1, 1), material, spots.length);
      spots.forEach(({ x, y }, i) => mesh.setMatrixAt(i, this.matrix.makeTranslation(this.centre(x, y, this.at))));
      mesh.instanceMatrix.needsUpdate = true;
      mesh.renderOrder = 2;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
    };
    fixture(crateMaterial(lit), TILE.ONIONS);
    fixture(rackMaterial(lit), TILE.DISHES);

    for (const cell of kitchen.pots) {
      const x = cell % w, y = Math.floor(cell / w), level = uniform(0);
      const mesh = light.emitter(new THREE.RingGeometry(0.3, 0.42, 48).rotateX(-Math.PI / 2), rgb(FLAME).mul(level));
      const cover = light.occluder(new THREE.CircleGeometry(0.3, 40).rotateX(-Math.PI / 2));
      this.centre(x, y, mesh.position).setY(0.02);
      cover.position.copy(mesh.position);
      this.scene.add(mesh, cover);
      this.stoves.push({ x, y, flame: level, plopped: -Infinity, readied: -Infinity });
    }
    this.pots = this.sheet(flat(1, 1), potMaterial(lit, this.time), kitchen.pots.length, ['aPot', 'aPotFx'], 2);

    for (const { x, y } of cells(TILE.WINDOW)) {
      const toward = [0, 1, 2, 3].find((d) => tileAt(kitchen, x + STEP_X[d]!, y + STEP_Y[d]!) === TILE.FLOOR);
      if (toward === undefined) throw new Error(`${ctx.envId} serves through a window at (${x}, ${y}) that no floor reaches`);
      const angle = Math.atan2(STEP_X[toward]!, STEP_Y[toward]!), level = uniform(0);
      const mesh = light.emitter(flat(0.6, 0.05), rgb(LAMP).mul(level));
      this.centre(x, y, mesh.position).add(this.other.set(-Math.sin(angle), 0, -Math.cos(angle)).multiplyScalar(LAMP_AT - 0.052)).setY(0.02);
      mesh.rotation.y = angle;
      this.scene.add(mesh);
      this.windows.push({ x, y, angle, lamp: level, served: -Infinity });
    }
    this.passes = this.sheet(flat(1, 1), passMaterial(lit, this.time), this.windows.length, ['aPass'], 2);

    const item = itemMaterial(lit, this.time);
    this.placed = this.sheet(flat(PIECE, PIECE), item, kitchen.counters.length, ['aItem'], 4);
    this.chefs = this.sheet(flat(CHEF, CHEF), chefMaterial(lit, this.time), 4, ['aChef', 'aPose'], 5);
    this.carried = this.sheet(flat(PIECE, PIECE), item, 6, ['aItem'], 6);
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 3 : 9; });

    const ledger = this.ledger, pots = kitchen.pots.length;
    this.legend = {
      title: 'OVERCOOKED-AI',
      rows: [
        { key: 'you', label: 'you', tint: ACCENT, value: () => this.chefText(0) },
        { key: 'commis', label: 'commis', tint: COMMIS, value: () => this.chefText(1) },
        ...Array.from({ length: pots }, (_, p) => ({ key: `pot${p}`, label: pots > 1 ? `pot ${p + 1}` : 'pot', tint: BROTH, value: () => this.potText(ledger.pots[p]!) })),
        { key: 'served', label: 'served', tint: GOLD, value: () => `${formatCount(ledger.served)} soup${ledger.served === 1 ? '' : 's'} this game` },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(ledger.reward) },
      ],
      note: () => `game ${formatCount(ledger.game + 1)} · tick ${formatCount(ledger.clock)}`,
      bar: () => ledger.clock / kitchen.horizon,
    } satisfies LegendSpec;
  }

  private chefText(i: number): string {
    const { held, act, blocked, chance, still } = this.ledger, move = act[i]! < 0 ? '—' : `${ACTIONS[act[i]!]}${blocked[i] ? ' · blocked' : ''}`;
    const mood = i === 0 ? '' : chance ? ' · random step' : still > 0 ? ` · still ${still} of ${this.kitchen.patience}` : '';
    return `${HELD[held[i]!]} · ${move}${mood}`;
  }

  private potText(value: number): string {
    const { cookTime } = this.kitchen;
    if (value === FULL + cookTime) return 'soup ready';
    if (value > FULL) return `cooking ${value - FULL} of ${cookTime}`;
    return value === 0 ? 'empty' : `${value} of ${FULL} onions`;
  }

  private sheet(geometry: THREE.BufferGeometry, material: THREE.Material, count: number, names: readonly string[], order: number): Sheet {
    const attributes = names.map((name) => {
      const attribute = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attribute);
      return attribute;
    });
    const mesh = instancedMesh(geometry, material, Math.max(1, count));
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    return { mesh, attributes, used: 0 };
  }

  private put(sheet: Sheet, position: THREE.Vector3, angle: number, size: number, ...values: readonly (readonly number[])[]): void {
    const i = sheet.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    sheet.mesh.setMatrixAt(i, this.matrix.compose(position, this.quaternion, this.scale.setScalar(size)));
    values.forEach((row, k) => sheet.attributes[k]!.setXYZW(i, row[0]!, row[1]!, row[2]!, row[3]!));
  }

  private commit(sheet: Sheet): void {
    sheet.mesh.count = sheet.used;
    sheet.mesh.instanceMatrix.needsUpdate = true;
    for (const attribute of sheet.attributes) attribute.needsUpdate = true;
    sheet.used = 0;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private centre(x: number, y: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(x, y, this.grid, out);
  }

  private transfersOf(A: EnvFrame, B: EnvFrame): Transfer[] {
    if (this.cache.from === A.step && this.cache.to === B.step) return this.cache.transfers;
    const out: Transfer[] = [], boundary = B.done !== 0, kitchen = this.kitchen;
    for (let i = 0; i < 2; i++) {
      const a = chefOf(A.words, i), act = actOf(B.words, i), before = heldOf(a);
      const x = aheadX(a), y = aheadY(a), code = tileAt(kitchen, x, y);
      if (boundary) {
        if (act === INTERACT && before === ITEM.SOUP && code === TILE.WINDOW) out.push({ chef: i, kind: 'give', item: before, x, y, code });
        continue;
      }
      const b = chefOf(B.words, i), after = heldOf(b);
      if (act !== INTERACT || after === before) {
        if (after !== before) throw new Error(`OvercookedAI step ${B.step} changes chef ${i}'s hands from ${HELD[before]} to ${HELD[after]} without an interaction`);
        continue;
      }
      const kind: Kind = before !== ITEM.EMPTY && after === ITEM.EMPTY ? 'give' : before === ITEM.EMPTY ? 'take' : 'fill';
      const fits = kind === 'give' ? code === TILE.COUNTER || (code === TILE.POT && before === ITEM.ONION) || (code === TILE.WINDOW && before === ITEM.SOUP)
        : kind === 'take' ? code === TILE.COUNTER || (code === TILE.ONIONS && after === ITEM.ONION) || (code === TILE.DISHES && after === ITEM.DISH)
          : code === TILE.POT && before === ITEM.DISH && after === ITEM.SOUP;
      if (!fits || cellOf(a) !== cellOf(b) || facingOf(a) !== facingOf(b)) throw new Error(`OvercookedAI step ${B.step} turns chef ${i}'s ${HELD[before]} into ${HELD[after]} facing terrain ${code}`);
      out.push({ chef: i, kind, item: kind === 'take' ? after : before, x, y, code });
    }
    const served = out.filter((transfer) => transfer.kind === 'give' && transfer.code === TILE.WINDOW).length;
    if (Math.abs(B.reward - served * kitchen.soupReward) > 1e-6) throw new Error(`OvercookedAI step ${B.step} pays ${B.reward} for ${served} soups served at ${kitchen.soupReward} each`);
    this.cache = { from: A.step, to: B.step, transfers: out };
    return out;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    this.clock += frame.dtReal;
    this.time.value = this.clock;
    pool.update(frame.dtReal, time);
    const at = frame.state.step, A = frame.frameAt(at);
    if (A.step !== at) return;
    const after = frame.frameAt(at + 1), B = after.step === at + 1 ? after : A;
    const t = B === A ? 0 : clamp01(frame.position - at);
    if (frame.scrubbed || at < this.step || this.step < 0) {
      pool.clear();
      for (const stove of this.stoves) stove.plopped = stove.readied = -Infinity;
      for (const pass of this.windows) pass.served = -Infinity;
      this.fired = B !== A && t >= LAND ? B.step : at;
    } else {
      for (let s = this.fired + 1; s <= at; s++) this.effects(frame.frameAt(s - 1), frame.frameAt(s), s, pool, time);
      if (B !== A && t >= LAND && this.fired < B.step) this.effects(A, B, B.step, pool, time);
      this.fired = Math.max(this.fired, B !== A && t >= LAND ? B.step : at);
    }
    this.step = at;
    this.compose(A, B, t, time);
    this.record(frame, A);
  }

  private effects(A: EnvFrame, B: EnvFrame, s: number, pool: CelebrationPool, time: number): void {
    if (A.step !== s - 1 || B.step !== s) return;
    for (const transfer of this.transfersOf(A, B)) {
      if (transfer.kind !== 'give') continue;
      if (transfer.code === TILE.WINDOW) {
        const pass = this.windows.find((w) => w.x === transfer.x && w.y === transfer.y)!;
        pass.served = time;
        this.centre(pass.x, pass.y, this.at).add(this.back.set(-Math.sin(pass.angle), 0, -Math.cos(pass.angle)).multiplyScalar(0.45));
        pool.burst(this.at.setY(38), GOLD, 0.5, time);
      }
      if (transfer.code === TILE.POT) this.stoves.find((stove) => stove.x === transfer.x && stove.y === transfer.y)!.plopped = time;
    }
    if (B.done) return;
    const ready = FULL + this.kitchen.cookTime;
    this.stoves.forEach((stove, p) => {
      if (potOf(B.words, p) === ready && potOf(A.words, p) !== ready) stove.readied = time;
    });
  }

  private compose(A: EnvFrame, B: EnvFrame, t: number, time: number): void {
    const { pots, passes, placed, chefs, carried } = this;
    if (!pots || !passes || !placed || !chefs || !carried) return;
    const next = B !== A, boundary = next && B.done !== 0, kitchen = this.kitchen, transfers = next ? this.transfersOf(A, B) : [];
    const leave = boundary ? 1 - smoothstep01((t - 0.3) / 0.3) : 1, arrive = boundary ? smoothstep01((t - 0.5) / 0.35) : 0;

    kitchen.counters.forEach((cell, slot) => {
      const ia = counterOf(A.words, slot), ib = next ? counterOf(B.words, slot) : ia, x = cell % kitchen.w, y = Math.floor(cell / kitchen.w);
      const shown = boundary || ia === ib ? ia : t >= LAND ? ib : ITEM.EMPTY;
      if (shown === ITEM.EMPTY) return;
      const settle = !boundary && ia !== ib ? 1 + 0.14 * Math.sin(Math.PI * clamp01((t - LAND) / 0.3)) : 1;
      this.put(placed, this.centre(x, y, this.at), 0, settle, [shown, leave, steamy(shown), 0]);
    });
    this.commit(placed);

    this.drawPots(A, B, t, next, boundary, time);
    this.windows.forEach((pass) => {
      const flare = fade(time - pass.served, FLARE_LIFE) ** 2, bell = 1 - fade(time - pass.served, BELL_LIFE);
      this.put(passes, this.centre(pass.x, pass.y, this.at), pass.angle, 1, [flare, bell < 1 ? bell : 0, 0, 0]);
      pass.lamp.value = 1.4 + 2.2 * flare;
    });
    this.commit(passes);

    for (const i of [1, 0]) {
      const a = chefOf(A.words, i), act = next ? actOf(B.words, i) : -1, transfer = transfers.find((item) => item.chef === i);
      const pos = this.centre(xOf(a), yOf(a), this.shown);
      let angle: number = ANGLE[facingOf(a)]!, walk = 0, reach = act === INTERACT ? Math.sin(Math.PI * clamp01(t / 0.8)) : 0;
      let carry = heldOf(a) !== ITEM.EMPTY ? 1 : 0;
      if (next && !boundary) {
        const b = chefOf(B.words, i);
        if (cellOf(a) !== cellOf(b)) {
          pos.lerp(this.centre(xOf(b), yOf(b), this.other), smoothstep01(t));
          walk = t;
        } else if (act < STAY) pos.add(this.other.set(STEP_X[act]!, 0, STEP_Y[act]!).multiplyScalar(0.09 * Math.sin(Math.PI * clamp01(t / 0.6))));
        angle = turn(angle, ANGLE[facingOf(b)]!, smoothstep01(t / 0.35));
        if (transfer?.kind === 'give') carry = 1 - smoothstep01((t - 0.25) / (LAND - 0.25));
        if (transfer?.kind === 'take') carry = smoothstep01((t - 0.15) / (LAND - 0.15));
      } else if (boundary && act >= 0 && act < STAY) {
        pos.add(this.other.set(STEP_X[act]!, 0, STEP_Y[act]!).multiplyScalar(0.3 * smoothstep01(t / 0.5)));
        angle = turn(angle, ANGLE[act]!, smoothstep01(t / 0.35));
        walk = t;
      }
      const fumble = i === 1 && next ? THREE.MathUtils.lerp(chanceOf(A.words), chanceOf(B.words), smoothstep01(t / 0.3)) : i === 1 ? chanceOf(A.words) : 0;
      this.put(chefs, pos, angle, SIZE * (1 + 0.035 * Math.sin(Math.PI * walk)), [i, leave, walk, carry], [angle, reach, fumble, 1]);
      this.handsOf(pos, angle);
      if (boundary) {
        if (heldOf(a) !== ITEM.EMPTY && !transfer) this.put(carried, this.hands, 0, 1, [heldOf(a), leave, steamy(heldOf(a)), 0]);
        if (transfer) this.flight(transfer, t, transfer.code === TILE.WINDOW ? 1 : leave);
      } else if (transfer) this.flight(transfer, t, 1);
      else if (heldOf(a) !== ITEM.EMPTY) this.put(carried, this.hands, 0, 1, [heldOf(a), 1, steamy(heldOf(a)), 0]);
    }
    if (boundary) for (const i of [1, 0]) {
      const b = chefOf(B.words, i);
      this.put(chefs, this.centre(xOf(b), yOf(b), this.shown), ANGLE[facingOf(b)]!, SIZE * (1.15 - 0.15 * arrive), [i, arrive, 0, 0], [ANGLE[facingOf(b)]!, 0, 0, 1]);
    }
    this.commit(chefs);
    this.commit(carried);
  }

  private flight(transfer: Transfer, t: number, alpha: number): void {
    const carried = this.carried!, tile = this.centre(transfer.x, transfer.y, this.other), u = smoothstep01(t / LAND), arc = Math.sin(Math.PI * u);
    if (transfer.kind === 'fill') {
      this.at.copy(this.hands).lerp(tile, 0.42 * arc);
      this.put(carried, this.at, 0, 1 + 0.08 * arc, [u < 0.5 ? ITEM.DISH : ITEM.SOUP, alpha, u < 0.5 ? 0 : 1, arc]);
      return;
    }
    if (transfer.kind === 'take') {
      const lift = transfer.code === TILE.COUNTER ? 1 : 0.65 + 0.35 * u;
      this.at.copy(tile).lerp(this.hands, u);
      this.put(carried, this.at, 0, lift * (1 + 0.12 * arc), [transfer.item, alpha, steamy(transfer.item), arc]);
      return;
    }
    if (t < LAND) {
      const sink = transfer.code === TILE.POT ? 1 - 0.45 * u * u : 1;
      this.at.copy(this.hands).lerp(tile, u);
      this.put(carried, this.at, 0, sink * (1 + 0.12 * arc), [transfer.item, alpha, steamy(transfer.item), arc]);
      return;
    }
    if (transfer.code !== TILE.WINDOW) return;
    const pass = this.windows.find((w) => w.x === transfer.x && w.y === transfer.y)!, s = clamp01((t - LAND) / (1 - LAND));
    this.at.copy(tile).add(this.back.set(-Math.sin(pass.angle), 0, -Math.cos(pass.angle)).multiplyScalar(0.22 * smoothstep01(s)));
    this.put(carried, this.at, 0, 1 - 0.25 * s, [ITEM.SOUP, alpha * (1 - smoothstep01((s - 0.35) / 0.65)), 1, 0]);
  }

  private handsOf(pos: THREE.Vector3, angle: number): THREE.Vector3 {
    return this.hands.set(pos.x - Math.sin(angle) * HAND, 0, pos.z - Math.cos(angle) * HAND);
  }

  private drawPots(A: EnvFrame, B: EnvFrame, t: number, next: boolean, boundary: boolean, time: number): void {
    const pots = this.pots!, { cookTime } = this.kitchen, ready = FULL + cookTime;
    const cooked = (v: number) => (v > FULL ? (v - FULL) / cookTime : 0);
    this.stoves.forEach((stove, p) => {
      const va = potOf(A.words, p), vb = next ? potOf(B.words, p) : va;
      let shown = va, cook = cooked(va), fill = 1, grow = onionsIn(va);
      if (boundary) {
        shown = t < 0.5 ? va : vb;
        cook = cooked(shown);
        grow = onionsIn(shown);
        fill = t < 0.5 ? 1 - smoothstep01((t - 0.3) / 0.2) : 1;
      } else if (next) {
        const ticking = va > FULL && vb === va + 1;
        shown = t < LAND ? va : vb;
        cook = ticking ? THREE.MathUtils.lerp(cooked(va), cooked(vb), t) : t < LAND ? cooked(va) : va <= FULL && vb > FULL ? cooked(vb) * clamp01((t - LAND) / (1 - LAND)) : cooked(vb);
        grow = t < LAND ? onionsIn(va) : onionsIn(vb) > onionsIn(va) ? onionsIn(va) + smoothstep01((t - LAND) / 0.25) : onionsIn(vb);
      }
      const done = shown === ready ? 1 : 0, cooking = shown > FULL && !done ? 1 : 0;
      const plop = time - stove.plopped, puff = fade(time - stove.readied, PUFF_LIFE);
      this.put(pots, this.centre(stove.x, stove.y, this.at), 0, 1, [grow, cook, done, (cooking + 0.3 * done) * fill], [plop >= 0 && plop < PLOP_LIFE ? Math.max(1e-3, plop / PLOP_LIFE) : 0, puff, fade(time - stove.readied, 0.8), fill]);
      const flicker = 0.85 + 0.15 * Math.sin(this.clock * 13.7 + p * 2.1) * Math.sin(this.clock * 5.3 + p);
      stove.flame.value = (3.2 * cooking + 1.1 * done) * fill * flicker;
    });
    this.commit(pots);
  }

  private record(frame: FrameUpdate, A: EnvFrame): void {
    const ledger = this.ledger, agent = chefOf(A.words, 0), partner = chefOf(A.words, 1), opened = A.step > 0;
    ledger.held[0] = heldOf(agent);
    ledger.held[1] = heldOf(partner);
    ledger.act[0] = opened ? actOf(A.words, 0) : -1;
    ledger.act[1] = opened ? actOf(A.words, 1) : -1;
    ledger.chance = opened ? chanceOf(A.words) : 0;
    ledger.still = stillOf(partner);
    ledger.pots[0] = potOf(A.words, 0);
    ledger.pots[1] = potOf(A.words, 1);
    ledger.reward = A.reward;
    ledger.clock = clockOf(A.words);
    ledger.game = gameOf(A.words);
    if (ledger.step === A.step) return;
    ledger.step = A.step;
    const before = frame.frameAt(A.step - 1), known = opened && A.done === 0 && before.step === A.step - 1;
    for (let i = 0; i < 2; i++) ledger.blocked[i] = known && ledger.act[i]! < STAY && cellOf(chefOf(before.words, i)) === cellOf(chefOf(A.words, i));
    let served = 0;
    for (let s = A.step, n = ledger.clock; n > 0; s--, n--) {
      const f = s === A.step ? A : frame.frameAt(s);
      if (f.step !== s) break;
      served += Math.round(f.reward / this.kitchen.soupReward);
    }
    ledger.served = served;
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const A = frame.frameAt(frame.state.step);
    if (A.step !== frame.state.step) return;
    const kitchen = this.kitchen, at = this.at, ready = FULL + kitchen.cookTime;
    const mate = kit.marks('commis', 1), goods = kit.marks('counters', Math.max(1, kitchen.counters.length)), held = kit.marks('held', 2);
    const pips = kit.marks('onions', FULL * kitchen.pots.length), patience = kit.marks('patience', kitchen.patience);
    const partner = chefOf(A.words, 1);
    this.centre(xOf(partner), yOf(partner), at);
    mate.set(at.x, 0.06, at.z, 0.12, COMMIS);
    for (let i = 0; i < 2; i++) {
      const chef = chefOf(A.words, i), color = i === 0 ? ACCENT : COMMIS, curve = kit.curve(`facing${i}`, 5, color);
      this.centre(xOf(chef), yOf(chef), at);
      const dx = STEP_X[facingOf(chef)]!, dz = STEP_Y[facingOf(chef)]!, tipX = at.x + 0.8 * dx, tipZ = at.z + 0.8 * dz;
      curve.point(at.x, 0.07, at.z);
      curve.point(tipX, 0.07, tipZ);
      curve.point(tipX - 0.14 * dx - 0.1 * dz, 0.07, tipZ - 0.14 * dz + 0.1 * dx);
      curve.point(tipX, 0.07, tipZ);
      curve.point(tipX - 0.14 * dx + 0.1 * dz, 0.07, tipZ - 0.14 * dz - 0.1 * dx);
      if (heldOf(chef) !== ITEM.EMPTY) held.set(at.x + 0.3, 0.06, at.z - 0.3, 0.08, TINT[heldOf(chef)]!);
    }
    for (let n = 0; n < stillOf(partner); n++) {
      this.centre(xOf(partner), yOf(partner), at);
      patience.set(at.x - 0.3 + 0.12 * n, 0.06, at.z + 0.34, 0.04, COMMIS);
    }
    kitchen.counters.forEach((cell, slot) => {
      const item = counterOf(A.words, slot);
      if (item === ITEM.EMPTY) return;
      this.centre(cell % kitchen.w, Math.floor(cell / kitchen.w), at);
      goods.set(at.x, 0.06, at.z, 0.1, TINT[item]!);
    });
    this.stoves.forEach((stove, p) => {
      const value = potOf(A.words, p), arc = kit.curve(`cook${p}`, 41, value === ready ? GOLD : FLAME);
      this.centre(stove.x, stove.y, at);
      for (let n = 0; n < onionsIn(value); n++) pips.set(at.x - 0.12 + 0.12 * n, 0.06, at.z, 0.045, ONION);
      const turns = value > FULL ? (value - FULL) / kitchen.cookTime : 0;
      for (let n = 0; n <= 40 && turns > 0; n++) {
        const a = turns * 2 * Math.PI * n / 40;
        arc.point(at.x + 0.53 * Math.sin(a), 0.07, at.z - 0.53 * Math.cos(a));
      }
    });
  }
}

export default (() => new OvercookedScene()) satisfies SceneFactory;
