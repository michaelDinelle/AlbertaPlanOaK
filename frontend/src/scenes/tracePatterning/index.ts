import * as THREE from 'three/webgpu';
import { Fn, abs, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, smoothstep, sqrt, step, texture, uniform, vec2, vec3 } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { CONDITIONING_COUNTDOWN, CONDITIONING_CUES, CONDITIONING_STIMULUS } from '@sim/envs/conditioning';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Level = THREE.UniformNode<'float', number>;

const WINDOW = 40;
const RING = 1024;
const STATS = 1000;
const X0 = -7.6;
const NOW = 1.6;
const COL = (NOW - X0) / WINDOW;
const HX = NOW + 0.44;
const LANES = [2.6, 1.6, 0.6, -0.9, -2.2] as const;
const CELL_H = 0.3;
const LAMP_R = 0.27;
const EX = 5.3;
const EY = 0.45;
const EW = 2.0;
const EH = 1.15;
const IRIS = new THREE.Vector2(-0.3, -0.06);
const IR = 0.66;
const NT = new THREE.Vector2(7.0, -2.05);
const NB = new THREE.Vector2(7.75, -3.3);
const TG = new THREE.Vector2(EX + 0.75, EY - 0.3);
const SWING = 0.45;

const VOID = 0x0a0e13;
const TAPE = 0x111920;
const GRID = 0x1b2530;
const HEAD = 0x1d2631;
const RIM = 0x3a4757;
const SOCKET = 0x151a22;
const SCLERA = 0xe6e0d3;
const LID = 0x76606f;
const LASH = 0x110c13;
const PUPIL = 0x05070a;
const NOZZLE = 0x5f6a78;
const PUFF = 0xe2edff;
const CORAL = 0xff6a57;
const CUE_C = 0xc9e052;
const CUES = [TOKENS.cueCyan, TOKENS.cueMagenta, CUE_C] as const;

class TracePatterningScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'tracePatterning';
  override legend: LegendSpec | null = null;

  private interval = 1;
  private readonly texels = new Float32Array(RING * 4);
  private readonly tags = new Float64Array(RING).fill(NaN);
  private readonly history = new THREE.DataTexture(this.texels, RING, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly head = uniform(0);
  private readonly alpha = uniform(0);
  private readonly lamps: Level[] = LANES.map(() => uniform(0));
  private readonly lid = uniform(0);
  private readonly puff = uniform(-1);
  private readonly flush = uniform(0);
  private readonly hit = uniform(0);
  private readonly point = new THREE.Vector3();
  private accent = 0xffffff;
  private readonly ledger = { step: 0, cues: 0, countdown: 0, blink: 0, reward: 0, puffs: 0, caught: 0, stray: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.interval = parameter(ctx.environment, 'interval');
    return new THREE.Box2(new THREE.Vector2(-8, -4), new THREE.Vector2(8, 4));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa8b4c8).multiplyScalar(0.6);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor();
    this.scene.background = new THREE.Color(VOID);
    pool.group.renderOrder = 10;
    this.history.magFilter = this.history.minFilter = THREE.NearestFilter;
    this.history.needsUpdate = true;
    const board = new THREE.Mesh(new THREE.PlaneGeometry(26, 16).rotateX(-Math.PI / 2), this.material());
    board.frustumCulled = false;
    this.scene.add(board);
  }

  protected override drop(): void {
    this.history.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private material(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const accent = rgb(this.accent), gold = rgb(TOKENS.celebrationGold), coral = rgb(CORAL), start = this.interval + 1;
    material.colorNode = Fn(() => {
      const p = vec2(positionWorld.x, positionWorld.z.negate());
      const px = max(fwidth(p.x), 1e-5);
      const cov = (d: N<'float'>) => clamp(float(0.5).sub(d.div(px)), 0, 1);
      const c = rgb(VOID).toVar();

      const tapeBox = sdBox(p.sub(vec2((X0 + NOW) / 2 - 0.3, 0.2)), vec2((NOW - X0) / 2 + 0.5, 3.3), 0.18);
      const tapeFade = smoothstep(X0 - 0.6, X0 + 1.6, p.x);
      c.assign(mix(c, rgb(TAPE), cov(tapeBox).mul(tapeFade)));

      const u = float(NOW).sub(p.x).div(COL);
      const t = this.alpha.sub(u);
      const rel = floor(t).add(1);
      const f = fract(t);
      const slot = mod(this.head.add(rel).add(RING), RING);
      const data = texture(this.history, vec2(slot.add(0.5).div(RING), 0.5));
      const cues = data.x, stim = data.y, blink = data.z, cd = data.w.sub(1);
      const inside = step(p.x, NOW).mul(tapeFade).mul(step(0.5, data.w)).mul(step(abs(p.y.sub(0.2)), 3.1));
      const cx = f.sub(0.5).mul(COL);

      const sep = min(f, float(1).sub(f)).mul(COL);
      c.assign(mix(c, rgb(GRID), cov(sep.sub(px.mul(0.5))).mul(step(p.x, NOW)).mul(tapeFade).mul(step(abs(p.y.sub(0.2)), 3.1)).mul(0.8)));
      const divide = abs(p.y.sub((LANES[2] + LANES[3]) / 2)).sub(0.012);
      c.assign(mix(c, rgb(RIM), cov(divide).mul(tapeFade).mul(step(p.x, NOW)).mul(0.7)));

      const trace = step(0.5, cd);
      const opening = step(abs(cd.sub(start)), 0.5);
      const traceLine = cov(abs(p.y.sub(LANES[3])).sub(0.03)).mul(trace).mul(mix(float(1), step(0, cx), opening));
      c.assign(mix(c, accent.mul(0.62), traceLine.mul(inside)));
      const bracket = cov(abs(cx).sub(0.026)).mul(step(p.y, LANES[0])).mul(step(LANES[3], p.y)).mul(opening);
      c.assign(mix(c, accent.mul(0.72), bracket.mul(inside)));

      const bit = (k: number) => mod(floor(cues.div(2 ** k)), 2);
      LANES.forEach((y, j) => {
        const dy = p.y.sub(y);
        const box = sdBox(vec2(cx, dy), vec2(COL * 0.36, CELL_H), 0.05);
        const on = j < 3 ? bit(j) : j === 3 ? stim : blink;
        const tint = j < 3 ? rgb(CUES[j]!) : j === 3 ? rgb(PUFF) : mix(coral, gold, stim);
        const shade = tint.mul(smoothstep(-CELL_H, CELL_H, dy).mul(0.18).add(0.64));
        c.assign(mix(c, tint.mul(0.26), cov(abs(dy).sub(0.016)).mul(step(p.x, NOW)).mul(tapeFade).mul(0.75)));
        c.assign(mix(c, shade, cov(box).mul(on).mul(inside)));
        if (j === 4) c.assign(mix(c, coral.mul(0.8), cov(abs(box).sub(0.028)).mul(stim).mul(float(1).sub(blink)).mul(inside)));
      });

      const headBox = sdBox(p.sub(vec2(HX, 0.2)), vec2(0.46, 3.25), 0.16);
      c.assign(mix(c, rgb(HEAD), cov(headBox)));
      c.assign(mix(c, rgb(RIM), cov(abs(headBox).sub(0.022))));
      LANES.forEach((y, j) => {
        const tint = j < 3 ? rgb(CUES[j]!) : j === 3 ? rgb(PUFF) : accent;
        const level = this.lamps[j]!;
        const r = length(p.sub(vec2(HX, y)));
        const d = r.sub(LAMP_R);
        c.assign(c.add(tint.mul(exp(max(d, 0).mul(-7)).mul(level).mul(0.3)).mul(step(0, d))));
        const core = mix(tint.mul(0.13), tint.mul(smoothstep(LAMP_R, 0, r).mul(0.6).add(0.9)), level);
        c.assign(mix(c, core, cov(d)));
        c.assign(mix(c, rgb(RIM).mul(1.3), cov(abs(d).sub(0.018))));
        const g = p.sub(vec2(HX, y));
        const seg = (ax: number, ay: number, bx: number, by: number) => sdSegment(g, vec2(ax, ay), vec2(bx, by), 0);
        const ring = (cx: number, cy: number, radius: number) => abs(length(g.sub(vec2(cx, cy))).sub(radius));
        const glyph = [
          () => min(min(seg(-0.1, -0.13, 0, 0.13), seg(0, 0.13, 0.1, -0.13)), seg(-0.055, -0.035, 0.055, -0.035)),
          () => min(min(seg(-0.075, -0.13, -0.075, 0.13), min(seg(-0.075, 0.13, -0.015, 0.13), seg(-0.075, 0, -0.005, 0))),
            min(seg(-0.075, -0.13, -0.005, -0.13), min(max(ring(-0.015, 0.065, 0.065), g.x.negate().sub(0.015)), max(ring(-0.005, -0.065, 0.065), g.x.negate().sub(0.005))))),
          () => max(ring(0.01, 0, 0.12), max(float(0.04).sub(g.x), abs(g.y).sub(0.07)).negate()),
          () => min(min(length(g.sub(vec2(-0.1, 0))), length(g)), length(g.sub(vec2(0.1, 0)))).sub(0.018),
          () => min(max(ring(0, 0.07, 0.13), g.y.sub(0.07)), min(seg(0, -0.06, 0, -0.12), min(seg(-0.075, -0.035, -0.11, -0.085), seg(0.075, -0.035, 0.11, -0.085)))),
        ][j]!();
        c.assign(mix(c, mix(tint.mul(0.75), rgb(HEAD).mul(0.6), level), cov(glyph.sub(0.022))));
      });

      const q = p.sub(vec2(EX, EY));
      const socket = length(q.div(vec2(EW + 0.75, EH + 0.95))).sub(1).mul(EH + 0.9);
      c.assign(mix(c, rgb(SOCKET), cov(socket)));
      const xn = q.x.div(EW);
      const top = float(1).sub(xn.mul(xn)).mul(EH);
      const slope = q.x.mul(-2 * EH / (EW * EW));
      const almond = abs(q.y).sub(top).div(sqrt(slope.mul(slope).add(1)));
      c.assign(c.add(gold.mul(exp(abs(almond).mul(-4.5)).mul(this.hit).mul(0.9)).mul(step(0, almond))));
      const ball = cov(almond);
      const depth = smoothstep(0, EH * 0.9, almond.negate());
      const sclera = mix(rgb(SCLERA).mul(depth.mul(0.3).add(0.62)), coral.mul(0.8), this.flush.mul(0.55));
      const e = q.sub(vec2(IRIS.x, IRIS.y));
      const r = length(e);
      const iris = mix(accent.mul(0.3), accent.mul(0.9), smoothstep(0.2, IR, r));
      const eye = sclera.toVar();
      eye.assign(mix(eye, iris, cov(r.sub(IR))));
      eye.assign(mix(eye, rgb(LASH), cov(abs(r.sub(IR)).sub(0.03))));
      eye.assign(mix(eye, rgb(PUPIL), cov(r.sub(float(0.28).add(this.flush.mul(-0.06))))));
      eye.assign(mix(eye, rgb(0xffffff).mul(0.92), cov(length(e.sub(vec2(-0.22, 0.24))).sub(0.1))));
      c.assign(mix(c, eye, ball));
      const drop = float(1).sub(this.lid.mul(2));
      const edge = top.mul(drop);
      const lidDist = q.y.sub(edge).div(sqrt(slope.mul(drop).mul(slope.mul(drop)).add(1)));
      const lidTone = rgb(LID).mul(smoothstep(0, 0.9, lidDist).mul(0.3).add(0.6));
      c.assign(mix(c, lidTone, cov(lidDist.negate()).mul(ball)));
      c.assign(mix(c, rgb(LASH), cov(abs(lidDist).sub(0.05)).mul(step(abs(xn), 1))));
      c.assign(mix(c, rgb(LASH), cov(abs(almond).sub(0.04))));
      const crease = abs(q.y.sub(top.mul(0.92).add(0.34)).div(sqrt(slope.mul(slope).add(1)))).sub(0.028);
      c.assign(mix(c, rgb(LID).mul(0.5), cov(crease).mul(smoothstep(0.95, 0.35, abs(xn)))));

      const hose = sdSegment(p, vec2(NB.x, NB.y), vec2(8.6, -4.6), 0.1);
      c.assign(mix(c, rgb(0x2b333d), cov(hose)));
      const nozzle = sdSegment(p, vec2(NB.x, NB.y), vec2(NT.x, NT.y), 0.2);
      c.assign(mix(c, rgb(NOZZLE), cov(nozzle)));
      c.assign(mix(c, rgb(RIM).mul(1.6), cov(abs(nozzle).sub(0.02))));
      c.assign(mix(c, rgb(0x1a2028), cov(length(p.sub(vec2(NT.x, NT.y))).sub(0.1))));

      const travel = clamp(this.puff, 0, 1).mul(step(0, this.puff)).mul(step(this.puff, 1));
      [0, 0.16, 0.32].forEach((lag, n) => {
        const k = clamp(this.puff.sub(lag), 0, 1);
        const at = mix(vec2(NT.x, NT.y), vec2(TG.x, TG.y), k);
        const blob = length(p.sub(at)).sub(k.mul(0.2).add(0.14 - n * 0.03));
        c.assign(mix(c, rgb(PUFF).mul(0.8), cov(blob).mul(travel).mul(step(lag, this.puff)).mul(0.85 - n * 0.2)));
      });
      const s = clamp(this.puff.sub(1), 0, 1), splash = step(1, this.puff);
      const ring = abs(length(p.sub(vec2(TG.x, TG.y))).sub(s.mul(0.75).add(0.25))).sub(float(1).sub(s).mul(0.07).add(0.01));
      c.assign(mix(c, rgb(PUFF).mul(0.8), cov(ring).mul(splash).mul(float(1).sub(s))));
      return c;
    })();
    return material;
  }

  private at(frame: FrameUpdate, s: number): EnvFrame | null {
    if (s < 0) return null;
    const found = frame.frameAt(s);
    return found.step === s ? found : null;
  }

  private slot(s: number): number {
    return ((s % RING) + RING) % RING;
  }

  private sync(frame: FrameUpdate, i: number): void {
    let dirty = false;
    for (let s = i - RING + 2; s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] === s) continue;
      const f = this.at(frame, s);
      if (!f) {
        if (this.texels[o + 3] !== 0) dirty = true;
        this.texels.fill(0, o, o + 4);
        this.tags[slot] = NaN;
        continue;
      }
      this.texels[o] = f.words[CONDITIONING_CUES]!;
      this.texels[o + 1] = f.words[CONDITIONING_STIMULUS]! ? 1 : 0;
      this.texels[o + 2] = Math.trunc(f.actions[0]!) & 1;
      this.texels[o + 3] = 1 + f.words[CONDITIONING_COUNTDOWN]!;
      this.tags[slot] = s;
      dirty = true;
    }
    if (dirty) this.history.needsUpdate = true;
  }

  private bits(f: EnvFrame): number[] {
    const cues = f.words[CONDITIONING_CUES]!;
    return [cues & 1, (cues >> 1) & 1, (cues >> 2) & 1, f.words[CONDITIONING_STIMULUS] ? 1 : 0, Math.trunc(f.actions[0]!) & 1];
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const i = frame.state.step, now = frame.frameAt(i);
    this.sync(frame, i);
    const next = this.at(frame, i + 1), alpha = next ? clamp01(frame.position - i) : 0;
    this.head.value = this.slot(i);
    this.alpha.value = alpha;
    const k = next ? smoothstep01((alpha - SWING) / (1 - SWING)) : 0;
    const a = this.bits(now), b = next ? this.bits(next) : a;
    this.lamps.forEach((lamp, j) => { lamp.value = a[j]! + (b[j]! - a[j]!) * k; });
    this.lid.value = a[4]! + (b[4]! - a[4]!) * k;
    this.puff.value = b[3] && next ? alpha : a[3] ? 1 + alpha : -1;
    this.flush.value = a[3] && !a[4] ? 1 - 0.7 * alpha : 0;
    this.hit.value = a[3] && a[4] ? 1 - 0.8 * alpha : 0;
    this.celebrate(frame, now, pool, time);
    this.tally(i, now);
  }

  private celebrate(frame: FrameUpdate, now: EnvFrame, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    for (const event of frame.events) {
      if (event.type === 'goal') {
        pool.burst(this.point.set(EX, 40, -EY), TOKENS.celebrationGold, 1.2, time);
        pool.ripple(this.point.set(EX, 0, -EY), TOKENS.celebrationGold, 0.55, time);
      } else if (event.type === 'cue' && now.words[CONDITIONING_COUNTDOWN] === this.interval + 1) {
        pool.ripple(this.point.set(HX, 0, -(LANES[0] + LANES[1]) / 2), this.accent, 0.22, time);
      }
    }
  }

  private tally(i: number, now: EnvFrame): void {
    const l = this.ledger;
    l.step = i;
    l.cues = now.words[CONDITIONING_CUES]!;
    l.countdown = now.words[CONDITIONING_COUNTDOWN]!;
    l.blink = Math.trunc(now.actions[0]!) & 1;
    l.reward = now.reward;
    let puffs = 0, caught = 0, stray = 0;
    for (let s = Math.max(0, i - STATS + 1); s <= i; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] !== s) continue;
      const stim = this.texels[o + 1]!, blink = this.texels[o + 2]!;
      puffs += stim;
      caught += stim * blink;
      stray += (1 - stim) * blink;
    }
    Object.assign(l, { puffs, caught, stray });
  }

  private legendFor(): LegendSpec {
    const l = this.ledger, names = ['A', 'B', 'C'];
    return {
      title: 'TRACE PATTERNING',
      rows: [
        { key: 'cues', label: 'cues now', tint: TOKENS.cueCyan, value: () => names.filter((_, k) => (l.cues >> k) & 1).join(' + ') || 'none' },
        { key: 'trace', label: 'trace', tint: this.accent, value: () => (l.countdown > 0 ? `puff in ${formatCount(l.countdown)} step${l.countdown === 1 ? '' : 's'}` : 'idle') },
        { key: 'blink', label: 'eye', tint: this.accent, value: () => (l.blink ? 'blinking' : 'open') },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
        { key: 'caught', label: `puffs blinked, last ${formatCount(STATS)}`, tint: TOKENS.celebrationGold, value: () => (l.puffs > 0 ? `${l.caught} of ${l.puffs}` : '—') },
        { key: 'stray', label: `false blinks, last ${formatCount(STATS)}`, tint: CORAL, value: () => formatCount(l.stray) },
      ],
      note: () => `only A + B without C starts the trace · puff ${formatCount(this.interval + 1)} steps later`,
      bar: () => (l.puffs > 0 ? l.caught / l.puffs : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const i = frame.state.step, alpha = this.alpha.value, top = this.interval + 1;
    const x = (s: number) => NOW - (alpha - (s - i) + 0.5) * COL;
    const profile = kit.curve('countdown', 2 * (WINDOW + 2), TOKENS.uiCyan);
    const due = kit.marks('due', WINDOW + 2);
    for (let s = i - WINDOW; s <= i + 1; s++) {
      const slot = this.slot(s), o = slot * 4;
      if (this.tags[slot] !== s) continue;
      const cx = x(s);
      if (cx - COL / 2 < X0 || cx + COL / 2 > NOW) continue;
      const h = LANES[3] - 0.4 + (0.8 * (this.texels[o + 3]! - 1)) / top;
      profile.point(cx - COL / 2, 0.06, -h);
      profile.point(cx + COL / 2, 0.06, -h);
      const stim = this.texels[o + 1]!, blink = this.texels[o + 2]!;
      if (stim || blink) due.set(cx, 0.06, -LANES[4], stim ? 0.16 : 0.08, stim ? (blink ? 0xffffff : CORAL) : CORAL);
    }
    kit.marks('lid', 1).set(EX, 0.06, -EY, 0.1 + 0.2 * this.lid.value, this.accent);
    kit.marks('target', 1).set(TG.x, 0.06, -TG.y, this.puff.value >= 0 ? 0.2 : 0.06, 0xffffff);
  }
}

export default (() => new TracePatterningScene()) satisfies SceneFactory;
