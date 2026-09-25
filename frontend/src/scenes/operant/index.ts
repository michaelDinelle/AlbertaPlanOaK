import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { coverage, rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';
import {
  BOX, BUZZ, CHUTE, CUP, HOUSE, INSIDE, LAMP, LEVER_DROP, LEVER_SPRITE, MAG_LAMP, PELLET, PIVOT, POCKETS, RIG, SPEAKER, TROUGH,
  backdropMaterial, buzzMaterial, flat, hazeMaterial, leverMaterial, pelletMaterial, place, rigMaterial, type RigGlow,
} from './art';
import { COST, FOOD, History, LIGHT, PRESS, type Totals } from './history';
import { RAT, RatRig, ratMaterial, ratShape } from './rat';
import { CONTACT, Chart, EVENT_H, EVENT_Y, PAPER, PEN_X, PILOT, REC, SWEEP, WINDOW, contactOf, levelY, paperX, recorderMaterial } from './recorder';

const ACCENT = TOKENS.accents.Operant!;
const GOLD = TOKENS.celebrationGold;
const BLUE = TOKENS.failureBlue;
const STATS = 2048;
const LOOK = 64;
const EAT = 2.4;
const FALL = 0.5;
const PELLETS = 4;
const REST = [-0.95, -1.08, -0.83] as const;
const PATH = [CHUTE[0], CHUTE[1], CHUTE[2], new THREE.Vector2(REST[0], TROUGH + PELLET * 0.8)] as const;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const has = (bits: number, bit: number) => bits >= 0 && (bits & bit) !== 0;
const signed = (value: number) => (Number.isFinite(value) ? (value > 0 ? `+${Number(value.toPrecision(3))}` : value < 0 ? `−${Number((-value).toPrecision(3))}` : '0') : '—');

interface Cup { pellets: number; bite: number; dip: number; landed: number }

class OperantScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 3;
  protected readonly stageName = 'operant';

  private cueRate = 1;
  private readonly history = new History();
  private readonly chart = new Chart();
  private readonly rig = new RatRig();
  private readonly glow: RigGlow = { lamp: uniform(0), house: uniform(1), magazine: uniform(0), buzz: uniform(0), wheel: uniform(0) };
  private readonly lampPower = uniform(0);
  private readonly magPower = uniform(0);
  private readonly buzzAge = uniform(0);
  private readonly haze = uniform(0);
  private lever: THREE.Object3D[] = [];
  private pellets: { mesh: THREE.InstancedMesh; state: THREE.InstancedBufferAttribute } | null = null;
  private readonly totals: Totals = { presses: 0, pellets: 0 };
  private readonly cup: Cup = { pellets: 0, bite: 0, dip: 0, landed: -Infinity };
  private readonly shown = { rear: 0, lever: 0, dip: 0, look: 0 };
  private readonly matrix = new THREE.Matrix4();
  private readonly point = new THREE.Vector2();
  private readonly world = new THREE.Vector3();
  private clock = 0;
  private pace = 0;
  private lastPosition = 0;
  private statsAt = -Infinity;
  private readonly ledger = { step: 0, light: false, press: false, reward: NaN, hit: NaN, dark: NaN, share: 0, pellets: 0, span: 0 };

  override legend: LegendSpec = {
    title: 'OPERANT',
    rows: [
      { key: 'light', label: 'light', tint: ACCENT, value: () => (this.ledger.light ? 'on' : 'off') },
      { key: 'lever', label: 'lever', tint: 0xd9dee3, value: () => (this.ledger.step < 1 ? '—' : this.ledger.press ? 'pressed' : 'still') },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => (this.ledger.step < 1 ? '—' : signed(this.ledger.reward)) },
      { key: 'answered', label: 'lights answered', tint: ACCENT, value: () => formatPercent(this.ledger.hit) },
      { key: 'dark', label: 'dark presses', tint: BLUE, value: () => formatPercent(this.ledger.dark, 2) },
      { key: 'pellets', label: 'pellets', tint: 0xd9a766, value: () => formatCount(this.ledger.pellets) },
    ],
    note: () => `light 1 step in ${formatCount(1 / this.cueRate)} · rates over ${formatCount(this.ledger.span)} steps`,
    bar: () => this.ledger.share,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.cueRate = parameter(ctx.environment, 'cueRate');
    return new THREE.Box2(new THREE.Vector2(-8.1, -3.8), new THREE.Vector2(8.2, 4.6));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x98a4bc).multiplyScalar(0.42);
  }

  protected override poolOptions() {
    return { burstSize: 26, sighRadius: 1.2, sighHeight: 1 };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    for (const child of pool.group.children) child.renderOrder = 11;
    this.scene.background = new THREE.Color(0x07090b);
    const lit = light.lit, accent = this.accentOf(ctx);
    const add = (mesh: THREE.Mesh, x: number, y: number, order: number) => {
      place(mesh, x, y);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return mesh;
    };
    add(new THREE.Mesh(flat(40, 26), backdropMaterial(lit)), 0, -0.5, 0);
    add(new THREE.Mesh(flat(RIG.x1 - RIG.x0, RIG.y1 - RIG.y0), rigMaterial(lit, accent, this.glow)), (RIG.x0 + RIG.x1) / 2, (RIG.y0 + RIG.y1) / 2, 1);
    add(new THREE.Mesh(flat(REC.x1 - REC.x0 + 0.1, REC.y1 - REC.y0 + 0.1), recorderMaterial(lit, this.chart)), (REC.x0 + REC.x1) / 2, (REC.y0 + REC.y1) / 2, 1);

    const leverMesh = add(new THREE.Mesh(flat(LEVER_SPRITE.width, LEVER_SPRITE.height).translate(-LEVER_SPRITE.pivot, 0, 0), leverMaterial(lit)), PIVOT.x, PIVOT.y, 2);
    const leverShade = add(light.occluder(flat(0.95, 0.1).translate(-0.425, 0, 0)), PIVOT.x, PIVOT.y, 1);
    this.lever = [leverMesh, leverShade];

    const pelletGeometry = flat(PELLET * 2.6, PELLET * 2.6);
    const state = new THREE.InstancedBufferAttribute(new Float32Array(PELLETS * 4), 4).setUsage(THREE.DynamicDrawUsage);
    pelletGeometry.setAttribute('aPellet', state);
    const pellets = instancedMesh(pelletGeometry, pelletMaterial(lit), PELLETS);
    pellets.renderOrder = 2;
    pellets.frustumCulled = false;
    this.scene.add(pellets);
    this.pellets = { mesh: pellets, state };

    const ratCentre = { x: (RAT.x0 + RAT.x1) / 2, y: (RAT.y0 + RAT.y1) / 2 }, ratSize = { w: RAT.x1 - RAT.x0, h: RAT.y1 - RAT.y0 };
    add(new THREE.Mesh(flat(ratSize.w, ratSize.h), ratMaterial(ratShape(this.rig, lit))), ratCentre.x, ratCentre.y, 3);
    const ratShade = light.occluder(flat(ratSize.w, ratSize.h));
    const shadeMaterial = ratShade.material as THREE.MeshBasicNodeMaterial;
    shadeMaterial.opacityNode = coverage(ratShape(this.rig, lit).silhouette);
    shadeMaterial.alphaTest = 0.5;
    add(ratShade, ratCentre.x, ratCentre.y, 1);

    const block = (x0: number, y0: number, x1: number, y1: number) => add(light.occluder(flat(x1 - x0, y1 - y0)), (x0 + x1) / 2, (y0 + y1) / 2, 1);
    block(BOX.x0, INSIDE.y0, INSIDE.x0, BOX.y1);
    block(BOX.x0, INSIDE.y1, BOX.x1, BOX.y1);
    block(INSIDE.x1, INSIDE.y0, BOX.x1, BOX.y1);
    block(BOX.x0, BOX.y0, BOX.x1, INSIDE.y0);

    const circle = (r: number) => new THREE.CircleGeometry(r, 24).rotateX(-Math.PI / 2);
    add(light.emitter(circle(0.17), rgb(accent).mul(this.lampPower)), LAMP.x - 0.05, LAMP.y, 2);
    add(light.emitter(circle(0.2), rgb(0xffe6c4).mul(this.glow.house.mul(15))), HOUSE.x, HOUSE.y - 0.06, 2);
    add(light.emitter(flat(INSIDE.x1 - INSIDE.x0 - 1.1, 0.05), rgb(0xc8d6ff).mul(this.glow.house.mul(6.5))), (INSIDE.x0 + INSIDE.x1) / 2 - 0.15, INSIDE.y1 - 0.04, 2);
    add(light.emitter(circle(0.06), rgb(0xffd890).mul(this.magPower)), CUP.x0 + 0.08, MAG_LAMP.y - 0.05, 2);
    add(light.emitter(circle(0.05), rgb(0xff8a3a).mul(this.chart.pilot.mul(1.5))), PILOT.x, PILOT.y, 2);

    add(new THREE.Mesh(flat(3.2, 3.2), hazeMaterial(accent, this.haze, 2.2)), LAMP.x - 0.2, LAMP.y, 5);
    add(new THREE.Mesh(flat(BUZZ, BUZZ), buzzMaterial(this.glow.buzz, this.buzzAge)), SPEAKER.x, SPEAKER.y, 5);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const tip = new THREE.Vector2();
    return {
      debug: (frame, kit) => {
        const history = this.history, i = frame.state.step, position = frame.position;
        history.bind(frame);
        const now = history.at(i), next = history.at(i + 1);
        kit.marks('light', 1).set(LAMP.x - 0.42, 0.4, -LAMP.y, 0.14, has(now, LIGHT) ? ACCENT : 0x4a5058);
        this.rig.leverPoint(1.0, 0, tip);
        kit.marks('press', 1).set(tip.x, 0.4, -(tip.y - 0.3), 0.12, has(next, PRESS) ? (has(next, FOOD) ? GOLD : BLUE) : 0x4a5058);
        kit.marks('pen', 1).set(PEN_X, 0.4, -levelY(this.chart.pen.value), 0.06, TOKENS.uiCyan);
        const cup = kit.marks('cup', 3);
        for (let j = 0; j < Math.min(3, this.cup.pellets); j++) cup.set(REST[j]!, 0.4, -(TROUGH + 0.3), 0.06, GOLD);
        const base = i - WINDOW - 2;
        let count = history.totals(frame.eventLog, base - 1, this.totals).presses;
        const trace = kit.curve('trace', 3 * (WINDOW + 4) + 2, TOKENS.uiCyan);
        const food = kit.marks('food', WINDOW + 4), cost = kit.marks('cost', WINDOW + 4), lights = kit.marks('lights', WINDOW + 4);
        let started = false;
        for (let k = Math.max(0, base); k <= i + 1; k++) {
          const bits = history.at(k);
          if (bits < 0) break;
          const x = paperX(contactOf(k), position);
          if (has(bits, LIGHT) && k <= i) {
            const lx = paperX(k + 0.5, position);
            if (lx >= PAPER.x0 && lx <= PEN_X) lights.set(lx, 0.4, -(EVENT_Y + EVENT_H), 0.03, ACCENT);
          }
          if (!has(bits, PRESS) || contactOf(k) > position) continue;
          const before = count % SWEEP;
          count++;
          if (x < PAPER.x0) continue;
          if (!started) trace.point(PAPER.x0, 0.4, -levelY(before));
          started = true;
          trace.point(x, 0.4, -levelY(before));
          trace.point(x, 0.4, -levelY(before + 1));
          if (before + 1 === SWEEP) trace.point(x, 0.4, -levelY(0));
          (has(bits, FOOD) ? food : cost).set(x, 0.4, -levelY(before + 1), 0.035, has(bits, FOOD) ? GOLD : BLUE);
        }
        if (!started) trace.point(PAPER.x0, 0.4, -levelY(count % SWEEP));
        trace.point(PEN_X, 0.4, -levelY(count % SWEEP));
      },
    };
  }

  protected override drop(): void {
    this.chart.texture.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private simulateCup(position: number, i: number): Cup {
    const history = this.history, cup = this.cup;
    let pellets = 0, bite = 0, since = NaN, stopped = -Infinity, landed = -Infinity;
    for (let m = i - LOOK; m <= i; m++) {
      const end = m === i ? position : m + 1;
      const rearing = history.has(m + 1, PRESS);
      if (rearing) {
        if (!Number.isNaN(since)) stopped = m;
        since = NaN;
      } else {
        let cursor = history.has(m, PRESS) ? m + CONTACT : m;
        while (pellets > 0 && cursor < end) {
          if (Number.isNaN(since)) since = cursor;
          const use = Math.min(end - cursor, EAT - bite);
          bite += use;
          cursor += use;
          if (bite >= EAT - 1e-9) {
            pellets--;
            bite = 0;
            if (pellets === 0) {
              stopped = cursor;
              since = NaN;
            }
          }
        }
      }
      if (m < i && history.has(m + 1, FOOD)) {
        pellets++;
        landed = m + 1;
      }
    }
    cup.pellets = pellets;
    cup.bite = bite;
    cup.landed = landed;
    cup.dip = Number.isNaN(since) ? 1 - smooth(0, 0.3, position - stopped) : smooth(0, 0.3, position - since);
    return cup;
  }

  private along(s: number, out: THREE.Vector2): THREE.Vector2 {
    let total = 0;
    for (let k = 1; k < PATH.length; k++) total += PATH[k]!.distanceTo(PATH[k - 1]!);
    let left = s * total;
    for (let k = 1; k < PATH.length; k++) {
      const span = PATH[k]!.distanceTo(PATH[k - 1]!);
      if (left <= span || k === PATH.length - 1) return out.copy(PATH[k - 1]!).lerp(PATH[k]!, Math.min(1, left / span));
      left -= span;
    }
    return out.copy(PATH[PATH.length - 1]!);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal, history = this.history;
    this.clock += dt;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * 5;
    history.bind(frame);
    const i = frame.state.step, position = Math.max(i, frame.position), t = Math.min(0.999999, position - i);
    const now = history.at(i), next = history.at(i + 1), prev = history.at(i - 1);
    const pressNext = has(next, PRESS), pressNow = has(now, PRESS), lightNow = has(now, LIGHT), lightPrev = has(prev, LIGHT), lightNext = has(next, LIGHT);

    const rear = pressNext ? (pressNow ? 1 : smooth(0.04, 0.34, t)) : pressNow ? 1 - smooth(0, 0.4, t) : 0;
    const lever = pressNext ? smooth(0.3, CONTACT, t) * (1 - smooth(0.55, 0.78, t)) : 0;
    const look = lightNow ? (lightPrev ? 1 : smooth(0, 0.22, t)) : lightPrev ? 1 - smooth(0, 0.4, t) : 0;
    const lamp = lightNow ? (lightNext ? 1 : 1 - smooth(0.92, 1, t)) : lightNext ? smooth(0.94, 1, t) : 0;
    const cup = this.simulateCup(position, i);

    const blend = frame.scrubbed ? 1 : 1 - Math.exp(-dt / (fast ? 0.12 : 0.028));
    const shown = this.shown;
    shown.rear += (rear - shown.rear) * blend;
    shown.lever += (lever - shown.lever) * blend;
    shown.dip += (cup.dip - shown.dip) * blend;
    shown.look += (look - shown.look) * blend;
    const c = this.clock;
    this.rig.pose({
      rear: shown.rear,
      lever: shown.lever,
      dip: shown.dip,
      look: shown.look,
      chew: Math.sin(c * 15),
      breath: 1 + 0.012 * Math.sin(c * 2 * Math.PI * 1.3),
      sniff: 0.035 * Math.sin(c * 9.1) * Math.max(0, Math.sin(c * 0.7)) * (1 - shown.rear) * (1 - shown.dip),
      sway: Math.sin(c * 0.9),
      glint: lamp,
    });
    for (const part of this.lever) part.rotation.y = shown.lever * LEVER_DROP;

    this.glow.lamp.value = lamp;
    this.lampPower.value = 34 * lamp;
    this.haze.value = 0.16 * lamp;
    const since = position - cup.landed, mag = since >= 0 && since < 8 ? Math.exp(-since / 1.6) : 0;
    this.glow.magazine.value = mag;
    this.magPower.value = 30 * mag;
    let buzz = 0, buzzAge = 0;
    for (const k of [i, i + 1]) {
      const age = position - contactOf(k);
      if (has(history.at(k), COST) && age >= 0 && age < 2.5) {
        const level = Math.exp(-age / 0.5);
        if (level > buzz) {
          buzz = level;
          buzzAge = age;
        }
      }
    }
    this.glow.buzz.value = buzz;
    this.buzzAge.value = buzzAge;
    const earned = history.totals(frame.eventLog, i, this.totals).pellets;
    const dispense = has(next, FOOD) ? smooth(0.4, 0.58, t) : 0;
    this.glow.wheel.value = -((earned + dispense) * 2 * Math.PI) / POCKETS;

    this.placePellets(has(next, FOOD) ? t : -1, cup, mag);

    if (i !== this.chart.filledAt || !this.chart.complete || frame.scrubbed) this.chart.fill(history, frame.eventLog, i);
    this.chart.aim(history, frame.eventLog, position);

    if (frame.scrubbed) pool.clear();
    else if (!fast) for (const event of frame.events) {
      if (event.type === 'goal') {
        pool.burst(this.world.set(CUP.x0 - 0.12, 0.6, -(CUP.y0 + CUP.y1) / 2), GOLD, 0.7, time);
        pool.ripple(this.world.set(CUP.x0 - 0.05, 0, -(CUP.y0 + CUP.y1) / 2), GOLD, 0.2, time);
      } else if (event.type === 'harm') pool.ripple(this.world.set(SPEAKER.x - 0.1, 0, -SPEAKER.y), BLUE, 0.2, time);
    }

    const ledger = this.ledger;
    ledger.step = i;
    ledger.light = lightNow;
    ledger.press = pressNow;
    ledger.reward = history.reward(i);
    ledger.pellets = earned;
    if (frame.scrubbed || this.clock - this.statsAt > 0.3) this.stats(i);
  }

  private placePellets(fall: number, cup: Cup, mag: number): void {
    const batch = this.pellets;
    if (!batch) return;
    const { mesh, state } = batch;
    const hide = (slot: number) => {
      mesh.setMatrixAt(slot, this.matrix.makeScale(0, 0, 0));
      state.setXYZW(slot, 0, 0, 0, 0);
    };
    if (fall >= FALL) {
      const s = (fall - FALL) / (1 - FALL);
      this.along(s * s, this.point);
      mesh.setMatrixAt(0, this.matrix.makeTranslation(this.point.x, 0, -this.point.y));
      state.setXYZW(0, 1, 0.8, 0, 0);
    } else hide(0);
    for (let j = 0; j < 3; j++) {
      if (j >= cup.pellets) {
        hide(j + 1);
        continue;
      }
      const size = j === 0 ? 1 - 0.75 * (cup.bite / EAT) : 1;
      this.matrix.makeScale(size, 1, size).setPosition(REST[j]!, 0, -(TROUGH + PELLET * 0.8 * size));
      mesh.setMatrixAt(j + 1, this.matrix);
      state.setXYZW(j + 1, 1, mag, 0, 0);
    }
    mesh.instanceMatrix.needsUpdate = true;
    state.needsUpdate = true;
  }

  private stats(i: number): void {
    const history = this.history, ledger = this.ledger;
    let lights = 0, hits = 0, dark = 0, darkPresses = 0, sum = 0, pay = NaN, span = 0;
    for (let k = Math.max(1, i - STATS + 1); k <= i; k++) {
      const bits = history.at(k), before = history.at(k - 1);
      if (bits < 0 || before < 0) continue;
      span++;
      sum += history.reward(k);
      if (has(bits, FOOD)) {
        hits++;
        pay = history.reward(k);
      }
      if (has(before, LIGHT)) lights++;
      else {
        dark++;
        if (has(bits, COST)) darkPresses++;
      }
    }
    ledger.span = span;
    ledger.hit = lights > 0 ? hits / lights : NaN;
    ledger.dark = dark > 0 ? darkPresses / dark : NaN;
    ledger.share = lights > 0 && Number.isFinite(pay) ? Math.min(1, Math.max(0, sum / (lights * pay))) : 0;
    this.statsAt = this.clock;
  }
}

export default (() => new OperantScene()) satisfies SceneFactory;
