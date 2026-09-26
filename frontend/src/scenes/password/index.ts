import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, exp, float, floor, fwidth, int, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, step, uniform, uniformArray, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { PASSWORD_PHASE, passwordKey } from '@sim/envs/password';
import type { EnvFrame, EventStream } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Span = readonly [number, number];
type Moment = { col: number; row: number; kind: number };

const S = 1;
const R = 0.72;
const KS = 1;
const TICK = 0.18;
const LEDGER = 24;
const MARGIN = 0.75;
const PAD = 0.35;
const FAST = 5;
const TAU = Math.PI * 2;
const GOLD = TOKENS.celebrationGold;
const MISS = TOKENS.failureBlue;
const PLATE = 0x0c0f14;
const BODY = 0x262b33;
const RIM = 0x46505d;
const WELL = 0x07080b;
const DOOR = 0x161a21;
const RIVET = 0.55;
const STEEL = 0x9aa4b2;
const DIGITS = [0xff8a5c, 0xf2c94c, 0x5fd39a, 0xb58cff, 0xff6fae, 0x6fd6ff];
const WRONG = 0;
const RIGHT = 1;
const OPEN = 2;

const PRESS_OUT: Span = [0, 0.2];
const PRESS_IN: Span = [0.1, 0.3];
const PICK_OUT: Span = [0.05, 0.28];
const PICK_IN: Span = [0.22, 0.4];
const JUDGE: Span = [0.42, 0.55];
const SEAT: Span = [0.45, 0.62];
const DRAIN: Span = [0.5, 0.75];
const FLUSH: Span = [0.8, 1];
const SLIDE: Span = [0.55, 0.9];
const RISE: Span = [0.58, 0.78];
const FALL: Span = [0.35, 0.9];

const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const ease = (t: number, s: Span) => smoothstep01(span(t, s));
const slots = (count: number, fill: number) => {
  const values = new Array<number>(count).fill(fill);
  return { node: uniformArray(values, 'float' as const), values };
};

const pixel = fwidth(positionWorld.x);
const aa = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel)), 0, 1);
const halo = (d: N<'float'>, r: number): N<'float'> => exp(max(d, 0).div(r).negate());
const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

function pip(q: N<'vec2'>, n: N<'float'>, radius: number, size: number): N<'float'> {
  const off = float(1).sub(mod(n, 2)).mul(Math.PI).div(n);
  const j = floor(atan(q.x, q.y).sub(off).div(TAU).mul(n).add(0.5));
  const th = j.mul(TAU).div(n).add(off);
  const rr = min(n.sub(1), 1).mul(radius);
  return length(q.sub(vec2(sin(th), cos(th)).mul(rr))).sub(size);
}

class PasswordScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 3;
  protected readonly stageName = 'password';
  override legend: LegendSpec | null = null;

  private length = 1;
  private base = 1;
  private accent = 0;
  private bw = 1;
  private bh = 1;
  private ty = 1;
  private ky = 1;
  private ly = 1;
  private door: [number, number, number] = [1, 1, -1];
  private notch = slots(1, 0);
  private latch = slots(1, 0);
  private keys = slots(1, 0);
  private ledger = slots(LEDGER, -1);
  private readonly probe = uniform(0);
  private readonly pick = { col: uniform(-9), row: uniform(-9), alpha: uniform(0), kind: uniform(0), judged: uniform(0) };
  private readonly open = uniform(0);
  private board: THREE.Mesh | null = null;
  private pace = 0;
  private lastPosition = 0;
  private drawn = -1;
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, opened: 0, lost: 0 };
  private readonly row = { phase: 0, key: -1, kind: -1, reward: 0, code: '' };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const L = parameter(ctx.environment, 'length'), B = constant(ctx.environment, 'PASSWORD_BASE');
    if (!Number.isInteger(L) || L < 1 || !Number.isInteger(B) || B < 2 || B > DIGITS.length) throw new Error(`${ctx.environment.envId} has length ${L} and base ${B}`);
    this.length = L;
    this.base = B;
    this.bw = (L * S) / 2 + MARGIN;
    this.bh = (B * R) / 2 + PAD;
    this.ty = this.bh + 0.32;
    this.ky = -this.bh - 0.75;
    this.ly = this.ky - 0.72;
    const hx = Math.max(this.bw + 1.05, (B * KS) / 2 + 0.3, (LEDGER * TICK) / 2 + 0.3) + 0.45;
    this.door = [hx - 0.1, this.ty + 0.6, this.ly - 0.66];
    return new THREE.Box2(new THREE.Vector2(-hx, -(this.ty + 0.7)), new THREE.Vector2(hx, -(this.ly - 0.76)));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8a93b0);
  }

  private colX(c: number): number {
    return (c + 0.5) * S - (this.length * S) / 2;
  }

  private rowY(r: number): number {
    return (this.base * R) / 2 - (r + 0.5) * R;
  }

  private keyOf(frame: EnvFrame): number {
    return passwordKey(frame) % this.base;
  }

  private moment(before: EnvFrame | null, after: EnvFrame): Moment | null {
    const phase = after.words[PASSWORD_PHASE]!;
    const kind = after.reward > 0 ? OPEN : phase > 0 ? RIGHT : WRONG;
    const col = before ? before.words[PASSWORD_PHASE]! : kind === OPEN ? this.length - 1 : kind === RIGHT ? phase - 1 : -1;
    return col < 0 ? null : { col, row: this.keyOf(after), kind };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const L = this.length, i = frame.state.step, head = frame.frameAt(i);
        const code = kit.marks('code', L);
        for (let c = 0; c < L; c++) {
          const d = head.view[c]! % this.base;
          code.set(this.colX(c), 0.1, -this.rowY(d), 0.14, DIGITS[d]!);
        }
        kit.marks('phase', 1).set(this.colX(head.words[PASSWORD_PHASE]!), 0.1, -this.ty, 0.18, this.accent);
        if (i < 1) return;
        const back = frame.frameAt(i - 1), was = this.moment(back.step === i - 1 ? back : null, head);
        if (was) kit.marks('pressed', 1).set(this.colX(was.col), 0.1, -this.rowY(was.row), 0.24, was.kind === WRONG ? MISS : GOLD);
      },
    };
  }

  private digit(index: N<'float'>): N<'vec3'> {
    let c: N<'vec3'> = rgb(DIGITS[this.base - 1]!);
    for (let k = this.base - 2; k >= 0; k--) c = select(index.lessThan(k + 0.5), rgb(DIGITS[k]!), c);
    return c;
  }

  private shader(): N<'vec3'> {
    const L = this.length, B = this.base, bw = this.bw, bh = this.bh, accent = rgb(this.accent), gold = rgb(GOLD), miss = rgb(MISS);
    const p = vec2(positionWorld.x, positionWorld.z.negate());
    const grainy = noise(p.mul(3.1)).mul(0.07).add(noise(vec2(p.x.mul(0.7), p.y.mul(26))).mul(0.05)).add(0.94);
    let c: N<'vec3'> = rgb(PLATE).mul(grainy).mul(smoothstep(2, 9, length(p.sub(vec2(0, -0.6)))).mul(0.5).oneMinus());

    const [dhx, dtop, dbot] = this.door, dq = p.sub(vec2(0, (dtop + dbot) / 2)), dh = vec2(dhx, (dtop - dbot) / 2);
    const doorD = sdBox(dq, dh, 0.35);
    c = mix(c, rgb(DOOR).mul(grainy).mul(smoothstep((dbot - dtop) / 2, (dtop - dbot) / 2, dq.y).mul(0.16).add(0.9)), aa(doorD));
    c = mix(c, rgb(RIM).mul(0.75), aa(abs(doorD.add(0.04)).sub(0.02)));
    c = c.mul(smoothstep(-0.02, 0.05, abs(doorD.add(0.1)).sub(0.012)).mul(0.35).add(0.65));
    const inner = vec2(dhx - 0.3, (dtop - dbot) / 2 - 0.3);
    const horiz = abs(dq.y).sub(inner.y).greaterThan(abs(dq.x).sub(inner.x));
    const along = select(horiz, dq.x, dq.y), reach = select(horiz, inner.x, inner.y);
    const m = along.sub(clamp(floor(along.div(RIVET).add(0.5)), reach.div(RIVET).negate().floor().add(1), reach.div(RIVET).floor()).mul(RIVET));
    const rq = vec2(m, abs(sdBox(dq, inner, 0)));
    const rivetD = length(rq).sub(0.045);
    c = c.mul(smoothstep(-0.02, 0.06, length(rq.add(vec2(-0.02, 0.025))).sub(0.05)).mul(0.5).add(0.5));
    c = mix(c, mix(rgb(STEEL).mul(0.28), rgb(STEEL).mul(0.55), smoothstep(-0.05, 0.05, rq.x.negate().add(rq.y))), aa(rivetD));

    const bodyD = sdBox(p, vec2(bw, bh), 0.24);
    const shadow = smoothstep(-0.15, 0.55, sdBox(p.sub(vec2(0.12, -0.16)), vec2(bw, bh), 0.24)).oneMinus();
    c = c.mul(shadow.mul(0.65).oneMinus());

    const keeperQ = p.sub(vec2(bw + 0.72, 0));
    const keeperD = sdBox(keeperQ, vec2(0.3, 0.34), 0.09);
    c = mix(c, rgb(BODY).mul(0.8), aa(keeperD));
    c = mix(c, rgb(WELL), aa(sdBox(keeperQ, vec2(0.2, 0.24), 0.05)));
    c = mix(c, rgb(RIM), aa(abs(keeperD).sub(0.018)));

    const xs = bw - 0.3, xe = this.open.mul(-0.9).add(bw + 0.85);
    const boltD = sdBox(p.sub(vec2(xe.add(xs).mul(0.5), 0)), vec2(xe.sub(xs).mul(0.5), 0.17), 0.06);
    const boltShade = mix(rgb(STEEL).mul(0.45), rgb(STEEL).mul(0.85), smoothstep(-0.17, 0.12, p.y));
    c = mix(c, boltShade, aa(boltD));
    c = mix(c, rgb(STEEL).mul(0.3), aa(abs(boltD).sub(0.012)));

    const bodyFill = rgb(BODY).mul(grainy).mul(smoothstep(-bh, bh, p.y).mul(0.18).add(0.9));
    c = mix(c, bodyFill, aa(bodyD));
    c = mix(c, rgb(RIM), aa(abs(bodyD.add(0.07)).sub(0.03)));
    c = c.add(gold.mul(this.open.mul(0.55)).mul(halo(abs(bodyD.add(0.07)).sub(0.03), 0.05)));

    const ci = clamp(floor(p.x.add((L * S) / 2).div(S)), 0, L - 1);
    const ri = clamp(floor(float((B * R) / 2).sub(p.y).div(R)), 0, B - 1);
    const cx = ci.add(0.5).mul(S).sub((L * S) / 2), cy = float((B * R) / 2).sub(ri.add(0.5).mul(R));
    const q = p.sub(vec2(cx, cy));
    const rc = this.digit(ri);
    const lt = this.latch.node.element(int(ci));
    const isNotch = step(abs(this.notch.node.element(int(ci)).sub(ri)), 0.5);

    const strip = sdBox(p.sub(vec2(cx, 0)), vec2(0.46, bh - 0.14), 0.12);
    c = c.add(accent.mul(lt.mul(0.07)).mul(aa(strip)));
    const cellD = sdBox(q, vec2(0.36, 0.26), 0.07);
    c = mix(c, rgb(BODY).mul(0.62), aa(cellD));
    c = mix(c, rgb(RIM).mul(0.7), aa(abs(cellD).sub(0.01)).mul(smoothstep(-0.2, 0.2, q.y)));
    const wayD = sdBox(q, vec2(0.27, 0.085), 0.085);
    c = mix(c, rgb(WELL), aa(wayD).mul(isNotch));
    c = mix(c, rc.mul(0.5), aa(abs(wayD).sub(0.016)).mul(isNotch));
    c = c.add(rc.mul(lt.mul(0.3).mul(isNotch)).mul(halo(wayD, 0.05)));
    c = c.add(gold.mul(this.open.mul(0.3).mul(isNotch)).mul(halo(wayD, 0.05)));
    c = mix(c, mix(rc, vec3(1), 0.15).mul(0.95), aa(sdBox(q, vec2(0.22, 0.045), 0.045)).mul(lt).mul(isNotch));

    const lx = -(L * S) / 2 - 0.38;
    c = mix(c, rc.mul(0.8), aa(pip(p.sub(vec2(lx, cy)), ri.add(1), 0.1, 0.042)));

    const pk = this.pick, here = step(abs(ci.sub(pk.col)), 0.5).mul(step(abs(ri.sub(pk.row)), 0.5));
    const seated = pk.judged.mul(step(0.5, pk.kind));
    const pinD = sdCircle(q, 0.12);
    c = c.add(rc.mul(0.35).mul(halo(pinD, 0.05)).mul(pk.alpha).mul(here).mul(seated.oneMinus()));
    c = mix(c, mix(rc, vec3(1), 0.3).mul(0.9), aa(pinD).mul(pk.alpha).mul(here).mul(seated.oneMinus()));
    const verdictTint = select(pk.kind.greaterThan(1.5), gold, rc);
    c = mix(c, verdictTint.mul(0.95), aa(abs(cellD).sub(0.022)).mul(here).mul(seated).mul(pk.alpha));
    const crossD = min(sdSegment(q, vec2(-0.13, -0.13), vec2(0.13, 0.13), 0.028), sdSegment(q, vec2(-0.13, 0.13), vec2(0.13, -0.13), 0.028));
    const wrong = here.mul(pk.judged).mul(pk.alpha).mul(step(pk.kind, 0.5));
    c = c.add(miss.mul(0.22).mul(aa(cellD)).mul(wrong));
    c = mix(c, miss.mul(0.95), aa(crossD).mul(wrong));
    c = mix(c, miss.mul(0.8), aa(abs(cellD).sub(0.02)).mul(wrong));

    const pxw = this.probe.add(0.5).mul(S).sub((L * S) / 2);
    const bracket = abs(sdBox(p.sub(vec2(pxw, 0)), vec2(0.47, bh - 0.11), 0.14)).sub(0.02);
    c = mix(c, accent.mul(0.8), aa(bracket).mul(0.85));
    const cq = p.sub(vec2(pxw, this.ty));
    c = mix(c, accent.mul(0.85), aa(min(sdBox(cq, vec2(0.28, 0.1), 0.08), sdSegment(cq, vec2(0, -0.02), vec2(0, -0.24), 0.045))));
    c = mix(c, rgb(WELL), aa(sdCircle(cq, 0.04)));

    const kk = clamp(floor(p.x.div(KS).add(B / 2)), 0, B - 1);
    const kq = p.sub(vec2(kk.add(0.5).sub(B / 2).mul(KS), this.ky));
    const pr = this.keys.node.element(int(kk)), kc = this.digit(kk);
    let keyShadow: N<'float'> = float(9), keyGlow: N<'vec3'> = vec3(0);
    for (let k = 0; k < B; k++) {
      const at = p.sub(vec2((k + 0.5 - B / 2) * KS, this.ky)), down = this.keys.node.element(int(k));
      keyShadow = min(keyShadow, sdBox(at.sub(vec2(0.05, -0.08)), vec2(0.38, 0.34), 0.1));
      keyGlow = keyGlow.add(rgb(DIGITS[k]!).mul(down.mul(0.45)).mul(halo(sdBox(at.add(vec2(0, down.mul(0.035))), vec2(0.38, 0.34), 0.1), 0.12)));
    }
    c = c.mul(smoothstep(-0.1, 0.25, keyShadow).mul(0.6).add(0.4)).add(keyGlow);
    const capQ = kq.add(vec2(0, pr.mul(0.035)));
    const capD = sdBox(capQ, vec2(0.38, 0.34), 0.1);
    const face = mix(rgb(BODY).mul(1.25), kc.mul(0.42), 0.3);
    c = mix(c, mix(face, kc.mul(0.72), pr).mul(smoothstep(-0.34, 0.34, capQ.y).mul(0.2).add(0.88)), aa(capD));
    c = mix(c, rgb(RIM).mul(1.2), aa(abs(capD).sub(0.012)).mul(pr.oneMinus()));
    c = mix(c, mix(kc.mul(0.9), vec3(0.08), pr.mul(0.8)), aa(pip(capQ, kk.add(1), 0.13, 0.05)));

    const x0 = -(LEDGER * TICK) / 2;
    const li = clamp(floor(p.x.sub(x0).div(TICK)), 0, LEDGER - 1);
    const lq = p.sub(vec2(li.add(0.5).mul(TICK).add(x0), this.ly));
    const v = this.ledger.node.element(int(li)), has = step(0, v), kind = floor(v.div(B)), tc = this.digit(mod(max(v, 0), B));
    const tickC = select(kind.lessThan(0.5), mix(tc.mul(0.3), miss.mul(0.45), 0.35), tc.mul(0.9));
    c = mix(c, tickC, aa(sdBox(lq, vec2(0.05, 0.11), 0.04)).mul(has));
    c = mix(c, gold.mul(0.9), aa(sdCircle(lq.sub(vec2(0, 0.22)), 0.045)).mul(has).mul(step(1.5, kind)));
    c = mix(c, rgb(RIM), aa(sdSegment(p, vec2(x0, this.ly - 0.2), vec2(-x0, this.ly - 0.2), 0.008)));
    return c;
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    const L = this.length, B = this.base;
    this.accent = this.accentOf(ctx);
    this.notch = slots(L, 0);
    this.latch = slots(L, 0);
    this.keys = slots(B, 0);
    this.ledger = slots(LEDGER, -1);
    this.scene.background = new THREE.Color(PLATE);
    const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.shader();
    const mesh = new THREE.Mesh(flat(80, 60), material);
    mesh.position.set(0, 0, (this.ty + this.ly) / -2);
    mesh.renderOrder = 0;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.board = mesh;
    pool.group.traverse((object) => { object.renderOrder = 10; });

    const row = this.row, tally = this.tally;
    this.legend = {
      title: 'PASSWORD',
      rows: [
        { key: 'entered', label: 'entered', tint: this.accent, value: () => `${row.phase} of ${L}` },
        { key: 'pressed', label: 'pressed', tint: STEEL, value: () => (row.key < 0 ? '—' : `${row.key} · ${row.kind === OPEN ? 'opened' : row.kind === RIGHT ? 'right' : 'wrong'}`) },
        { key: 'code', label: 'hidden code', tint: DIGITS[2]!, value: () => row.code },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(row.reward) },
      ],
      note: () => `${formatCount(tally.opened)} opened · ${formatCount(tally.lost)} reset`,
      bar: () => clamp01((tally.opened * L) / Math.max(1, tally.through)),
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    if (!this.board) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const L = this.length, i = frame.state.step, head = frame.frameAt(i), ahead = frame.frameAt(i + 1);
    const next = !fast && ahead.step === i + 1 ? ahead : null, t = next ? clamp01(frame.position - i) : 0;
    const back = i >= 1 ? frame.frameAt(i - 1) : null, prev = back && back.step === i - 1 ? back : null;
    const phase = head.words[PASSWORD_PHASE]!;
    for (let c = 0; c < L; c++) this.notch.values[c] = head.view[c]! % this.base;

    const was = i >= 1 ? this.moment(prev, head) : null, will = next ? this.moment(head, next) : null;
    const latch = this.latch.values;
    for (let c = 0; c < L; c++) latch[c] = c < phase ? 1 : 0;
    if (will) {
      if (will.kind !== WRONG) latch[will.col] = Math.max(latch[will.col]!, ease(t, SEAT));
      const drain = will.kind === WRONG ? ease(t, DRAIN) : will.kind === OPEN ? ease(t, FLUSH) : 0;
      for (let c = 0; c < L; c++) latch[c] = latch[c]! * (1 - drain);
    }

    let open = head.reward > 0 ? (next ? 1 - ease(t, FALL) : 1) : 0;
    if (will && will.kind === OPEN) open = Math.max(open, ease(t, RISE));
    this.open.value = open;
    this.probe.value = phase + (next ? (next.words[PASSWORD_PHASE]! - phase) * ease(t, SLIDE) : 0);

    const pk = this.pick, shown = will && (t >= PICK_IN[0] || !was) ? will : was;
    if (shown) {
      pk.col.value = shown.col;
      pk.row.value = shown.row;
      pk.kind.value = shown.kind;
      pk.alpha.value = shown === will ? ease(t, PICK_IN) : next ? 1 - ease(t, PICK_OUT) : 1;
      pk.judged.value = shown === will ? ease(t, JUDGE) : 1;
    } else pk.alpha.value = 0;

    const keys = this.keys.values;
    keys.fill(0);
    const now = i >= 1 ? this.keyOf(head) : -1, then = next ? this.keyOf(next) : -1;
    if (now >= 0) keys[now] = next ? 1 - ease(t, PRESS_OUT) : 1;
    if (then >= 0) keys[then] = Math.max(keys[then]!, ease(t, PRESS_IN));

    if (this.drawn !== i) {
      this.drawn = i;
      const values = this.ledger.values;
      for (let n = 0; n < LEDGER; n++) {
        const j = i - (LEDGER - 1) + n, f = j >= 1 ? frame.frameAt(j) : null;
        values[n] = f && f.step === j ? this.keyOf(f) + this.base * (f.reward > 0 ? OPEN : f.words[PASSWORD_PHASE]! > 0 ? RIGHT : WRONG) : -1;
      }
    }

    if (!frame.scrubbed && !fast) {
      for (const item of frame.events) {
        if (item.type === 'goal') {
          pool.ripple({ x: 0, y: 0, z: 0 }, GOLD, 1.4, time);
          pool.burst({ x: this.bw - 0.2, y: 40, z: 0 }, GOLD, 0.8, time);
          pool.pulse.fire(0.4);
        } else if (item.type === 'harm' && was) pool.ripple({ x: this.colX(was.col), y: 0, z: -this.rowY(was.row) }, MISS, 0.5, time);
      }
    }

    this.score(frame.eventLog, i);
    Object.assign(this.row, {
      phase, key: was ? was.row : now, kind: was ? was.kind : -1, reward: head.reward,
      code: Array.from(this.notch.values.slice(0, L)).join(' '),
    });
  }

  private score(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, opened: 0, lost: 0 });
    if (at === tally.through) return;
    tally.opened += countEvents(log, 'goal', tally.through, at);
    tally.lost += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }

  protected override drop(): void {
    if (this.board) {
      this.board.geometry.dispose();
      (this.board.material as THREE.Material).dispose();
    }
    this.board = null;
  }
}

export default (() => new PasswordScene()) satisfies SceneFactory;
