import * as THREE from 'three/webgpu';
import { abs, clamp, dot, float, floor, fract, fwidth, length, max, min, mix, positionWorld, round, smoothstep, step, texture, uniform, vec2 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { fractal, rgb, sdBox } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { walkBet, walkDest, walkPosition } from '@sim/envs/walkbet';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';

type N = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

const TRAIL = 30;
const DY = 0.33;
const TOP = -0.78;
const EVIDENCE_Y = 0.8;
const WINDOW = 2048;
const BET_SECONDS = 0.06;

const BOARD = 0x11171b;
const PAD = 0x2a343b;
const PAD_RIM = 0x3d4a52;
const PAPER = 0xc2b8a0;
const RULE = 0xaa9f86;
const INK = 0x5d5a52;
const LEFT = 0x3b93c4;
const RIGHT = 0xe0923e;
const LEFT_INK = 0x2a6f99;
const RIGHT_INK = 0xb5651f;
const CORAL = 0xd9534c;
const WALKER_RIM = 0x0d1114;

class WalkBetScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.35, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'walkBet';
  override legend: LegendSpec | null = null;

  private cells = 20;
  private drift = 0;
  private period = 1;
  private readonly rows = new Float32Array(TRAIL * 4);
  private readonly trail = new THREE.DataTexture(this.rows, TRAIL, 1, THREE.RGBAFormat, THREE.FloatType);
  private proof = new Float32Array(4);
  private evidence = new THREE.DataTexture(this.proof, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly u = {
    walker: uniform(0), live: uniform(0), alpha: uniform(0), bet: uniform(0), show: uniform(1), miss: uniform(0),
    flashL: uniform(0), flashR: uniform(0), edge: uniform(0), edgeOn: uniform(0),
  };
  private betLevel = 0;
  private scannedAt = -1;
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, position: 0, bet: -1, reward: 0, seen: 0, right: 0, moves: 0, ups: 0, lo: 0, hi: 20, known: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.cells = constant(ctx.environment, 'WALKBET_CELLS');
    this.period = constant(ctx.environment, 'WALKBET_PERIOD');
    this.drift = parameter(ctx.environment, 'drift');
    this.proof = new Float32Array((this.cells + 1) * 4);
    this.evidence = new THREE.DataTexture(this.proof, this.cells + 1, 1, THREE.RGBAFormat, THREE.FloatType);
    const half = this.cells / 2 + 1.2, bottom = TOP - TRAIL * DY - 0.55;
    return new THREE.Box2(new THREE.Vector2(-half, -1.9), new THREE.Vector2(half, -bottom));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    const accent = this.accentOf(ctx);
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(BOARD);
    pool.group.renderOrder = 10;
    for (const tex of [this.trail, this.evidence]) {
      tex.magFilter = tex.minFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
    }
    const board = new THREE.Mesh(new THREE.PlaneGeometry(120, 120).rotateX(-Math.PI / 2), this.material(accent));
    board.frustumCulled = false;
    board.position.set(0, 0, (TOP - TRAIL * DY) / -2);
    this.scene.add(board);
  }

  private material(accent: number): THREE.MeshBasicNodeMaterial {
    const u = this.u, cells = this.cells, h = cells / 2;
    const q = vec2(positionWorld.x, positionWorld.z.negate());
    const px = max(fwidth(q.x), 1e-5);
    const cover = (d: N): N => clamp(float(0.5).sub(d.div(px)), 0, 1);
    const seg = (p: V2, a: V2, b: V2, r: number): N => {
      const pa = p.sub(a), ba = b.sub(a);
      const t = clamp(dot(pa, ba).div(max(dot(ba, ba), 1e-6)), 0, 1);
      return length(pa.sub(ba.mul(t))).sub(r);
    };

    let color = rgb(BOARD).mul(fractal(q.mul(0.7), 3).mul(0.06).add(0.97));

    const tapeTop = TOP + 0.16, tapeBottom = TOP - TRAIL * DY - 0.12;
    const tapeD = sdBox(q.sub(vec2(0, (tapeTop + tapeBottom) / 2)), vec2(h + 0.55, (tapeTop - tapeBottom) / 2), 0.08);
    const age = float(TOP).sub(q.y).div(DY);
    const fade = smoothstep(TRAIL * 0.45, TRAIL, age);
    let paper = rgb(PAPER).mul(fractal(q.mul(5), 2).mul(0.05).add(0.97));
    const column = abs(fract(q.x.add(0.5)).sub(0.5)).sub(0.008);
    paper = mix(paper, rgb(RULE), cover(column).mul(0.55));
    paper = mix(paper, rgb(RULE).mul(0.85), cover(abs(q.x).sub(0.018)).mul(0.6));

    const k = clamp(round(age.sub(u.alpha)), 0, TRAIL - 1);
    const at = texture(this.trail, vec2(k.add(0.5).div(TRAIL), 0.5));
    const flags = at.w, valid = step(0.5, flags.mod(2)), hit = step(0.5, floor(flags.div(2)).mod(2)), done = step(3.5, flags);
    const yk = float(TOP).sub(k.add(u.alpha).mul(DY));
    const dash = step(0.5, fract(q.x.mul(2.5)));
    paper = mix(paper, rgb(INK), cover(abs(q.y.sub(yk)).sub(0.014)).mul(done).mul(valid).mul(dash).mul(0.55));

    const kk = clamp(floor(age.sub(u.alpha)), 0, TRAIL - 1);
    const bt = texture(this.trail, vec2(kk.add(0.5).div(TRAIL), 0.5));
    const ykk = float(TOP).sub(kk.add(u.alpha).mul(DY));
    const pathD = seg(q, vec2(bt.y.sub(h), ykk), vec2(bt.x.sub(h), ykk.sub(DY)), 0.028);
    const pathOn = step(0.5, bt.w.mod(2)).mul(step(u.alpha, age));
    const liveD = seg(q, vec2(u.live, float(TOP).sub(u.alpha.mul(DY))), vec2(u.walker, TOP), 0.028);
    const lineD = min(pathD.add(pathOn.oneMinus().mul(10)), liveD);
    paper = mix(paper, rgb(INK), cover(lineD).mul(0.8));

    const dotD = length(q.sub(vec2(at.y.sub(h), yk)));
    const ink = mix(rgb(LEFT_INK), rgb(RIGHT_INK), at.z);
    paper = mix(paper, rgb(CORAL), cover(abs(dotD.sub(0.14)).sub(0.028)).mul(valid).mul(hit.oneMinus()));
    paper = mix(paper, ink, cover(dotD.sub(0.09)).mul(valid));
    paper = mix(paper, rgb(PAPER), fade.mul(0.8));

    const tapeCover = cover(tapeD).mul(step(tapeBottom, q.y));
    color = mix(color, rgb(0x070a0c), cover(tapeD.sub(0.1)).mul(0.5));
    color = mix(color, paper, tapeCover);

    const c = clamp(round(q.x.add(h)), 0, cells);
    const lx = q.x.add(h).sub(c);
    const cup = step(c, 0.5).add(step(cells - 0.5, c));
    const padD = sdBox(vec2(lx, q.y), vec2(0.42, 0.36), 0.1);
    const home = cover(abs(length(vec2(lx, q.y)).sub(0.2)).sub(0.02)).mul(step(abs(c.sub(h)), 0.25));
    const groove = abs(q.y).sub(0.07).max(abs(q.x).sub(h));
    color = mix(color, rgb(0x080b0d), cover(groove));
    const tick = sdBox(vec2(lx, q.y.add(0.5)), vec2(0.025, 0.06)).add(step(0.5, abs(c.mod(5).sub(0))).mul(10));
    color = mix(color, rgb(PAD_RIM), cover(tick).mul(cup.oneMinus()));
    let pad = mix(rgb(PAD_RIM), rgb(PAD), cover(padD.add(0.05)));
    pad = mix(pad, rgb(PAD_RIM).mul(1.4), home);
    color = mix(color, pad, cover(padD).mul(cup.oneMinus()));

    const ev = texture(this.evidence, vec2(c.add(0.5).div(cells + 1), 0.5));
    const evD = sdBox(vec2(lx, q.y.sub(EVIDENCE_Y)), vec2(0.4, 0.085), 0.06);
    const evTint = mix(rgb(LEFT), rgb(RIGHT), ev.x);
    const evColor = mix(rgb(PAD), evTint.mul(0.85), ev.y.mul(ev.z.mul(0.7).add(0.3)));
    color = mix(color, evColor, cover(evD).mul(cup.oneMinus()));

    const rightCup = step(cells - 0.5, c);
    const cupTint = mix(rgb(LEFT), rgb(RIGHT), rightCup);
    const betOn = mix(u.bet.mul(-0.5).add(0.5), u.bet.mul(0.5).add(0.5), rightCup);
    const flash = mix(u.flashL, u.flashR, rightCup);
    const cupD = sdBox(vec2(lx, q.y.sub(0.2)), vec2(0.44, 0.62), 0.3);
    const wellD = sdBox(vec2(lx, q.y.sub(0.2)), vec2(0.28, 0.46), 0.2);
    const level = betOn.mul(0.45).add(0.3).add(flash.mul(0.5));
    let cupColor = cupTint.mul(level);
    cupColor = mix(cupColor, rgb(0x080b0d).add(cupTint.mul(flash.mul(0.9).add(betOn.mul(0.12)))), cover(wellD));
    color = mix(color, cupColor, cover(cupD).mul(cup));

    const edgeD = abs(q.x.sub(u.edge)).sub(0.03).max(abs(q.y.sub(0.25)).sub(0.85));
    color = mix(color, rgb(TOKENS.celebrationGold).mul(0.8), cover(edgeD).mul(u.edgeOn));
    const edgeTape = abs(q.x.sub(u.edge)).sub(0.02).max(step(0.5, fract(q.y.mul(3))).sub(0.5));
    color = mix(color, rgb(TOKENS.celebrationGold).mul(0.55), cover(edgeTape).mul(u.edgeOn).mul(tapeCover).mul(0.7));

    const w = vec2(u.walker, 0);
    const stalk = seg(q, vec2(u.walker, -0.3), vec2(u.walker, TOP + 0.02), 0.018);
    const nib = abs(length(q.sub(vec2(u.walker, TOP))).sub(0.06)).sub(0.022);
    color = mix(color, rgb(0x3a3f44), cover(min(stalk, nib)).mul(u.show));
    const wd = length(q.sub(w));
    const d = u.bet;
    const eye = length(q.sub(w.add(vec2(d.mul(0.11), 0)))).sub(0.075);
    const chev = min(seg(q, w.add(vec2(d.mul(0.46), 0.17)), w.add(vec2(d.mul(0.6), 0)), 0.036), seg(q, w.add(vec2(d.mul(0.6), 0)), w.add(vec2(d.mul(0.46), -0.17)), 0.036));
    const betTint = mix(rgb(LEFT), rgb(RIGHT), d.mul(0.5).add(0.5));
    let body = mix(rgb(WALKER_RIM), rgb(accent).mul(0.9), cover(wd.sub(0.25)));
    body = mix(body, rgb(WALKER_RIM), cover(eye));
    color = mix(color, body, cover(wd.sub(0.3)).mul(u.show));
    color = mix(color, betTint, cover(chev).mul(u.show));
    color = mix(color, rgb(CORAL), cover(abs(wd.sub(0.42)).sub(0.03)).mul(u.miss).mul(u.show));

    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = color;
    return material;
  }

  protected override drop(): void {
    this.trail.dispose();
    this.evidence.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private at(frame: FrameUpdate, s: number): EnvFrame | null {
    if (s < 0) return null;
    const found = frame.frameAt(s);
    return found.step === s ? found : null;
  }

  private scan(frame: FrameUpdate, i: number): void {
    if (this.scannedAt === i) return;
    this.scannedAt = i;
    const cells = this.cells, l = this.ledger;
    this.rows.fill(0);
    this.proof.fill(0);
    let seen = 0, right = 0, ups = 0;
    let after = this.at(frame, i);
    for (let s = i; s >= Math.max(1, i - WINDOW + 1) && after; s--) {
      const before = this.at(frame, s - 1);
      if (!before) break;
      const from = walkPosition(before), dest = walkDest(before, after, cells), bet = walkBet(after), ok = after.reward > 0 ? 1 : 0;
      seen++;
      right += ok;
      ups += dest > from ? 1 : 0;
      const o = from * 4;
      if (from <= cells && this.proof[o + 1] === 0) {
        this.proof[o] = ok ? bet : 1 - bet;
        this.proof[o + 1] = 1;
        this.proof[o + 2] = 1 - (i - s) / WINDOW;
      }
      const k = i - s;
      if (k < TRAIL) this.rows.set([from, dest, bet, 1 + 2 * ok + (after.done ? 4 : 0)], k * 4);
      after = before;
    }
    let lo = 0, hi = cells, known = 0;
    for (let c = 1; c < cells; c++) {
      if (this.proof[c * 4 + 1] === 0) continue;
      known++;
      if (this.proof[c * 4] === 1) hi = Math.min(hi, c);
      else lo = Math.max(lo, c);
    }
    Object.assign(l, { seen, right, moves: seen, ups, lo, hi, known });
    this.u.edgeOn.value = known > 0 && lo + 1 === hi ? 1 : 0;
    this.u.edge.value = hi - cells / 2 - 0.5;
    this.trail.needsUpdate = true;
    this.evidence.needsUpdate = true;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const i = frame.state.step, now = frame.frameAt(i), next = this.at(frame, i + 1), prev = this.at(frame, i - 1);
    const alpha = next ? clamp01(frame.position - i) : 0, cells = this.cells, h = cells / 2;
    this.scan(frame, i);
    const pos = walkPosition(now), dest = next ? walkDest(now, next, cells) : pos;
    const u = this.u;
    u.alpha.value = alpha;
    u.live.value = pos - h;
    u.walker.value = pos + (dest - pos) * smoothstep01(alpha) - h;
    u.show.value = next && next.done ? 1 - smoothstep01((alpha - 0.55) / 0.45) : 1;
    const bet = next ? walkBet(next) : -1, target = bet < 0 ? this.betLevel : bet * 2 - 1;
    this.betLevel = frame.scrubbed ? target : target + (this.betLevel - target) * Math.exp(-frame.dtReal / BET_SECONDS);
    u.bet.value = this.betLevel;
    const settle = 1 - smoothstep01(alpha);
    const fresh = i > 0 && !now.reset && prev;
    u.miss.value = fresh && now.reward <= 0 ? settle : 0;
    const end = fresh && now.done ? walkDest(prev, now, cells) : -1;
    u.flashL.value = end === 0 ? settle : 0;
    u.flashR.value = end === cells ? settle : 0;
    Object.assign(this.ledger, { step: i, position: pos, bet, reward: now.reward });
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    if (end < 0) return;
    for (const event of frame.events) {
      if (event.type !== 'goal') continue;
      pool.burst(this.point.set(end - h, 40, -0.2), TOKENS.celebrationGold, 1.2, time);
      pool.ripple(this.point.set(end - h, 0, -0.2), TOKENS.celebrationGold, 0.5, time);
    }
  }

  private threshold(): string {
    const l = this.ledger;
    if (l.known === 0) return '—';
    if (l.lo >= l.hi) return 'shifting';
    if (l.lo + 1 === l.hi) return l.hi >= this.cells ? 'left everywhere' : `right from cell ${l.hi}`;
    return `between cells ${l.lo + 1} and ${l.hi}`;
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger;
    const rows: LegendSpec['rows'][number][] = [
      { key: 'position', label: 'walker', tint: accent, value: () => `cell ${l.position} of ${this.cells}` },
      { key: 'bet', label: 'bet', tint: RIGHT, value: () => (l.bet < 0 ? '—' : l.bet ? 'right end' : 'left end') },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
      { key: 'right', label: `right bets, last ${formatCount(l.seen)}`, tint: accent, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      { key: 'edge', label: 'bets pay right', tint: TOKENS.celebrationGold, value: () => this.threshold() },
      { key: 'ups', label: `right steps, last ${formatCount(l.moves)}`, tint: LEFT, value: () => formatPercent(l.moves > 0 ? l.ups / l.moves : NaN) },
    ];
    return {
      title: 'WALK BET',
      rows,
      note: () => (this.drift ? `odds drift over ${formatCount(this.period)} steps · ends at 0 and ${this.cells}` : `fair coin · ends at 0 and ${this.cells}`),
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const h = this.cells / 2, alpha = this.u.alpha.value;
    kit.marks('walker', 1).set(this.u.walker.value, 0.06, 0, 0.3, 0xffffff);
    const bet = this.ledger.bet;
    if (bet >= 0) kit.marks('bet', 1).set(bet ? h : -h, 0.06, -0.2, 0.5, bet ? RIGHT : LEFT);
    const verdicts = kit.marks('verdicts', this.cells + 1);
    for (let c = 1; c < this.cells; c++) {
      const o = c * 4;
      if (this.proof[o + 1] === 1) verdicts.set(c - h, 0.06, -EVIDENCE_Y, 0.14, this.proof[o] ? RIGHT : LEFT);
    }
    if (this.u.edgeOn.value > 0) {
      const edge = kit.curve('edge', 2, TOKENS.celebrationGold);
      edge.point(this.u.edge.value, 0.06, 0.6);
      edge.point(this.u.edge.value, 0.06, -1.1);
    }
    const path = kit.marks('path', TRAIL);
    for (let k = 0; k < TRAIL; k++) {
      const o = k * 4;
      if (this.rows[o + 3]! < 1) continue;
      const hit = Math.floor(this.rows[o + 3]! / 2) % 2 === 1;
      path.set(this.rows[o + 1]! - h, 0.06, -(TOP - (k + alpha) * DY), 0.12, hit ? TOKENS.uiCyan : CORAL);
    }
    void frame;
  }
}

export default (() => new WalkBetScene()) satisfies SceneFactory;
