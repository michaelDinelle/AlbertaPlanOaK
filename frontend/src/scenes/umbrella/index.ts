import * as THREE from 'three/webgpu';
import { Fn, abs, atan, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, sin, smoothstep, texture, uniform, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { UMBRELLA_HAVE, UMBRELLA_NEED, UMBRELLA_PHASE } from '@sim/envs/umbrella';
import type { EnvFrame, EventStream } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Span = readonly [number, number];

const TAU = Math.PI * 2;
const FAST = 5;
const G = -1.85;
const XA = -4.9;
const XB = 5.2;
const XD = 6.4;
const PIP_Y = -2.85;
const TEX = 64;
const WALK: Span = [0.12, 0.88];
const GRAB: Span = [0, 0.3];
const CLOSE: Span = [0.45, 0.85];
const OPEN: Span = [0.55, 0.95];
const REVEAL: Span = [0.35, 0.9];
const COIN_IN: Span = [0.7, 1];
const COIN_OUT: Span = [0.6, 1];
const BLINK: Span = [0, 0.15];
const GOLD = 0xf2c14e;
const MISS = 0x4f86d9;
const RAIN = 0x6f93c4;
const SUN = 0xffc54a;
const WINDOW = 0xffd27a;
const WALKER = 0x2f7f8f;
const CANOPY = 0xd94a7a;
const K = 1.35;

const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const ease = (t: number, s: Span) => smoothstep01(span(t, s));
const hash = (v: N<'vec2'>) => fract(sin(v.dot(vec2(127.1, 311.7))).mul(43758.5453));

class UmbrellaScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.35, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'umbrella';

  private length = 4;
  private distractors = 0;
  private pitch = 1;
  private accent = 0;
  private data: THREE.DataTexture | null = null;
  private readonly u = {
    wx: uniform(0), bob: uniform(0), swing: uniform(0), held: uniform(0), shutter: uniform(0), forecast: uniform(0),
    rain: uniform(0), sun: uniform(0), verdict: uniform(0), tint: uniform(new THREE.Vector3()), tile: uniform(-1), time: uniform(0),
  };
  private pace = 0;
  private lastPosition = 0;
  private readonly glow = new THREE.Color();
  private readonly point = new THREE.Vector3();
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, right: 0, wrong: 0 };
  private readonly row = { phase: 0, forecast: -1, have: 0, deciding: false, reward: 0, outcome: '—', lit: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.length = parameter(ctx.environment, 'length');
    this.distractors = parameter(ctx.environment, 'distractors');
    if (!Number.isInteger(this.length) || this.length < 2 || this.length > TEX) throw new Error(`${ctx.environment.envId} has length ${this.length}`);
    if (!Number.isInteger(this.distractors) || this.distractors < 0 || this.distractors > TEX || 4 + this.distractors !== ctx.environment.obs) throw new Error(`${ctx.environment.envId} has ${this.distractors} distractors for ${ctx.environment.obs} observations`);
    this.pitch = (XB - XA) / this.length;
    return new THREE.Box2(new THREE.Vector2(-8, -4.3), new THREE.Vector2(8, 3.8));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  private centre(p: number): number {
    return XA + (p + 0.5) * this.pitch;
  }

  private walkerX(s: number): number {
    const L = this.length;
    return s <= L - 1 ? this.centre(s) : this.centre(L - 1) + (XD - this.centre(L - 1)) * (s - L + 1);
  }

  private settled(frame: FrameUpdate, head: EnvFrame): { need: number; have: number; reward: number } | null {
    const at = head.step - head.words[UMBRELLA_PHASE]!;
    if (at < 1) return null;
    const S = frame.frameAt(at), B = frame.frameAt(at - 1);
    if (S.step !== at || !S.done || B.step !== at - 1) return null;
    return { need: B.words[UMBRELLA_NEED]!, have: B.words[UMBRELLA_HAVE]!, reward: S.reward };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, head = frame.frameAt(i), phase = head.words[UMBRELLA_PHASE]!;
        const x = this.centre(phase);
        kit.marks('agent', 1).set(x, 0.1, -(G - 0.25), 0.16, this.accent);
        kit.marks('need', 1).set(XD, 0.1, -2.3, 0.35, head.words[UMBRELLA_NEED] ? RAIN : SUN);
        kit.marks('have', 1).set(x, 0.1, -(G + 3.0), head.words[UMBRELLA_HAVE] ? 0.18 : 0.0001, this.accent);
        const coins = kit.marks('coins', this.length);
        for (let p = 1; p <= phase; p++) {
          const f = frame.frameAt(i - phase + p);
          if (f.step !== i - phase + p) continue;
          coins.set(this.centre(p), 0.1, -PIP_Y, Math.min(0.12, this.pitch * 0.3), f.reward > 0 ? GOLD : MISS);
        }
        const walked = kit.curve('walked', 2, this.accent);
        walked.point(this.centre(0), 0.1, -(G - 0.25));
        walked.point(x, 0.1, -(G - 0.25));
      },
    };
  }

  protected build(ctx: SceneContext, _stage: Stage2D): void {
    const L = this.length, D = this.distractors, pitch = this.pitch, U = this.u;
    this.accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(0x10141c);
    const data = new THREE.DataTexture(new Uint8Array(TEX * 2 * 4), TEX, 2, THREE.RGBAFormat, THREE.UnsignedByteType);
    data.minFilter = THREE.NearestFilter;
    data.magFilter = THREE.NearestFilter;
    data.needsUpdate = true;
    this.data = data;
    const sample = (index: N<'float'>, row: number) => texture(data, vec2(index.add(0.5).div(TEX), (row + 0.5) / 2));
    const accent = rgb(this.accent), canopyTint = rgb(CANOPY);

    const shade = Fn(() => {
      const p = vec2(positionWorld.x, positionWorld.z.negate()).toVar();
      const px = max(fwidth(p.x), 1e-5);
      const fill = (d: N<'float'>) => clamp(float(0.5).sub(d.div(px)), 0, 1);
      const col = vec3(0).toVar();
      const put = (c: N<'vec3'>, a: N<'float'>) => col.assign(mix(col, c, a));

      const sy = clamp(p.y.sub(G).div(4.5 - G), 0, 1);
      const base = mix(rgb(0x9aa5b4), rgb(0x55647d), sy), wet = mix(rgb(0x55606e), rgb(0x2b3342), sy), fair = mix(rgb(0xf4d9a6), rgb(0x4f93d6), sy);
      col.assign(mix(mix(base, wet, U.rain), fair, U.sun));

      const sd = p.sub(vec2(3.4, 3.1)), sr = length(sd);
      col.addAssign(rgb(SUN).mul(exp(sr.mul(-1.1)).mul(0.35).mul(U.sun)));
      const ray = fract(atan(sd.y, sd.x).div(TAU).mul(12).add(U.time.mul(0.03))).sub(0.5).abs();
      put(rgb(SUN), fill(max(ray.sub(0.13).mul(sr.mul(TAU / 12)), max(float(0.78).sub(sr), sr.sub(1.2)))).mul(U.sun));
      put(rgb(0xffe08a), fill(sr.sub(0.55)).mul(U.sun));

      const k = floor(p.x.add(8).div(1.3));
      const top = hash(vec2(k, 3.1)).mul(1.9).add(G + 1.2);
      const bx = abs(fract(p.x.add(8).div(1.3)).sub(0.5)).mul(1.3).sub(0.56);
      put(mix(rgb(0x7a8698), rgb(0x3c4655), U.rain.mul(0.6)), fill(max(bx, p.y.sub(top))));

      if (D > 0) {
        put(rgb(0x3f4859), fill(sdBox(p.sub(vec2(0, 0.35)), vec2(1.7, 2.2), 0.04)));
        const cx = clamp(floor(p.x.add(1.6).div(0.8)), 0, 3), ry = clamp(floor(float(2.4).sub(p.y).div(0.8)), 0, 2);
        const index = ry.mul(4).add(cx);
        const lit = sample(index, 1).r.mul(index.lessThan(D).select(1, 0));
        const wc = vec2(cx.add(0.5).mul(0.8).sub(1.6), float(2.4).sub(ry.add(0.5).mul(0.8)));
        put(mix(rgb(0x252b36), rgb(WINDOW), lit), fill(sdBox(p.sub(wc), vec2(0.24, 0.26), 0.03)));
      }

      const walk = mix(rgb(0xb8b2a7), rgb(0x7d8590), U.rain);
      put(walk, fill(p.y.sub(G)));
      const inStreet = fill(max(float(XA).sub(p.x), p.x.sub(XB)));
      const idx = clamp(floor(p.x.sub(XA).div(pitch)), 0, L - 1);
      const tcx = idx.add(0.5).mul(pitch).add(XA);
      const slab = sample(idx, 0);
      const band = max(p.y.sub(G), float(G - 0.5).sub(p.y));
      put(mix(walk, rgb(0xe9e2d2), 0.55), fill(max(band, abs(p.x.sub(tcx)).sub(pitch / 2 - 0.012))).mul(slab.b).mul(inStreet));
      put(accent, fill(max(band, abs(p.x.sub(tcx)).sub(pitch / 2 - 0.012))).mul(idx.equal(U.tile).select(0.45, 0)).mul(inStreet));
      const seam = abs(fract(p.x.sub(XA).div(pitch).add(0.5)).sub(0.5)).mul(pitch).sub(0.012);
      put(rgb(0x6e6a64), fill(max(seam, band)).mul(inStreet));
      put(rgb(0x8e8a83), fill(max(p.y.sub(G - 0.5), float(G - 0.62).sub(p.y))));
      put(mix(rgb(0x3a3d44), rgb(0x272c35), U.rain), fill(p.y.sub(G - 0.62)));
      put(rgb(0xd8c99a), fill(max(abs(p.y.add(3.3)).sub(0.035), abs(fract(p.x.div(0.9)).sub(0.5)).mul(0.9).sub(0.22))).mul(0.55));

      const rp = Math.min(0.2, pitch * 0.36);
      const coin = p.sub(vec2(tcx, PIP_Y)), cr = length(coin);
      put(rgb(GOLD), fill(cr.sub(rp)).mul(slab.r).mul(inStreet));
      put(rgb(0xb07d1c), fill(abs(cr.sub(rp * 0.62)).sub(rp * 0.1)).mul(slab.r).mul(inStreet));
      put(rgb(MISS), fill(cr.sub(rp)).mul(slab.g).mul(inStreet));
      put(rgb(0xd7e6ff), fill(sdBox(coin, vec2(rp * 0.5, rp * 0.13))).mul(slab.g).mul(inStreet));

      put(rgb(0xc9b79c), fill(sdBox(p.sub(vec2(-6.7, (G + 0.95) / 2)), vec2(1.2, (0.95 - G) / 2))));
      const roofY = p.y.sub(0.95);
      put(rgb(0x7a4b3a), fill(max(max(roofY.negate(), roofY.sub(1.15)), abs(p.x.add(6.7)).sub(float(1.4).sub(roofY.mul(1.15))).mul(0.66))));
      put(rgb(0x5b3a2e), fill(sdBox(p.sub(vec2(-5.95, G + 0.6)), vec2(0.3, 0.6), 0.05)));
      put(rgb(0xd9b25f), fill(sdCircle(p.sub(vec2(-5.75, G + 0.55)), 0.04)));

      const q = p.sub(vec2(-6.8, 0.2)), qr = length(q);
      const inside = fill(qr.sub(0.52));
      put(rgb(0x4a3528), fill(qr.sub(0.62)));
      put(mix(rgb(0x9fd0f5), rgb(0x8290a4), U.forecast), inside);
      const open = inside.mul(U.shutter.lessThan(0.999).select(1, 0));
      const icon = open.mul(float(1).sub(U.forecast));
      const sq = q.sub(vec2(0, 0.02)), sqr = length(sq);
      put(rgb(SUN), fill(sqr.sub(0.19)).mul(icon));
      const sray = fract(atan(sq.y, sq.x).div(TAU).mul(8)).sub(0.5).abs();
      put(rgb(SUN), fill(max(sray.sub(0.14).mul(sqr.mul(TAU / 8)), max(float(0.26).sub(sqr), sqr.sub(0.38)))).mul(icon));
      const rain = open.mul(U.forecast);
      const cloud = min(min(sdCircle(q.sub(vec2(-0.15, 0.08)), 0.16), sdCircle(q.sub(vec2(0.07, 0.15)), 0.2)), min(sdCircle(q.sub(vec2(0.22, 0.05)), 0.13), sdBox(q.sub(vec2(0.03, -0.01)), vec2(0.3, 0.07), 0.07)));
      put(rgb(0xeef2f6), fill(cloud).mul(rain));
      const drops = min(min(sdSegment(q, vec2(-0.16, -0.16), vec2(-0.2, -0.3), 0.028), sdSegment(q, vec2(0.04, -0.16), vec2(0, -0.3), 0.028)), sdSegment(q, vec2(0.24, -0.16), vec2(0.2, -0.3), 0.028));
      put(rgb(0x2f6fd0), fill(drops).mul(rain));
      const edge = float(0.52 - 1.04).mul(U.shutter).add(0.52);
      const cover = fill(edge.sub(q.y)).mul(inside);
      put(rgb(0x8b6b4f), cover);
      put(rgb(0x5e4633), fill(abs(fract(q.y.mul(8)).sub(0.5)).div(8).sub(0.012)).mul(cover));
      put(rgb(0x4a3528), fill(abs(q.y.sub(edge)).sub(0.025)).mul(inside).mul(U.shutter.greaterThan(0.01).select(1, 0)));

      const stand = float(1).sub(U.held);
      put(rgb(0x3d4852), fill(sdBox(p.sub(vec2(-5.28, G + 0.18)), vec2(0.11, 0.18), 0.02)));
      put(rgb(0x2a2d33), fill(sdSegment(p, vec2(-5.28, G + 0.12), vec2(-5.19, G + 0.98), 0.022)).mul(stand));
      put(canopyTint, fill(sdSegment(p, vec2(-5.27, G + 0.4), vec2(-5.21, G + 0.86), 0.07)).mul(stand));
      put(rgb(0x2a2d33), fill(abs(sdCircle(p.sub(vec2(-5.13, G + 1.0)), 0.06)).sub(0.018)).mul(fill(float(G + 1.0).sub(p.y))).mul(stand));

      put(rgb(0x93a89c), fill(sdBox(p.sub(vec2(6.7, (G + 1.3) / 2)), vec2(1.2, (1.3 - G) / 2))));
      put(rgb(0x6f8378), fill(sdBox(p.sub(vec2(6.7, 1.4)), vec2(1.28, 0.1))));
      const halo = length(p.sub(vec2(XD, G + 0.7)));
      col.assign(mix(col, U.tint, smoothstep(1.5, 0.2, halo).mul(U.verdict).mul(0.55)));
      put(rgb(0x2f3b3a), fill(sdBox(p.sub(vec2(XD, G + 0.62)), vec2(0.34, 0.62), 0.04)));
      put(mix(rgb(0x51615f), U.tint, U.verdict), fill(sdBox(p.sub(vec2(XD, G + 0.86)), vec2(0.22, 0.28), 0.03)));
      put(rgb(0xe3d3a1), fill(sdBox(p.sub(vec2(7.4, -0.55)), vec2(0.32, 0.45), 0.03)));
      const ax = p.x.sub(5.5);
      const scallop = length(vec2(fract(ax.div(0.3)).sub(0.5).mul(0.3), p.y.sub(0.38))).sub(0.15);
      const awning = max(min(sdBox(p.sub(vec2(6.7, 0.62)), vec2(1.3, 0.24)), max(scallop, p.y.sub(0.38))), abs(p.x.sub(6.7)).sub(1.3));
      put(rgb(0xf1e6d0), fill(awning));
      put(rgb(0x3f8f6b), fill(awning).mul(fill(abs(fract(ax.div(0.6)).sub(0.5)).mul(0.6).sub(0.15))));

      const wx = U.wx, lift = U.bob.add(G);
      const hip = vec2(wx, lift.add(0.45 * K));
      const legs = min(sdSegment(p, hip, vec2(wx.add(U.swing.mul(0.13 * K)), G + 0.05), 0.05 * K), sdSegment(p, hip, vec2(wx.sub(U.swing.mul(0.13 * K)), G + 0.05), 0.05 * K));
      put(rgb(0x2c3140), fill(legs));
      put(rgb(WALKER), fill(sdSegment(p, vec2(wx, lift.add(0.55 * K)), vec2(wx, lift.add(0.8 * K)), 0.15 * K)));
      put(rgb(0xe6bf98), fill(sdCircle(p.sub(vec2(wx, lift.add(1.08 * K))), 0.13 * K)));
      put(rgb(0x3b2a22), fill(max(sdCircle(p.sub(vec2(wx, lift.add(1.1 * K))), 0.135 * K), lift.add(1.12 * K).sub(p.y))));
      const hand = vec2(wx.add(mix(0.1 * K, 0.22 * K, U.held)), lift.add(mix(0.45 * K, 0.72 * K, U.held)));
      put(rgb(0x276b78), fill(sdSegment(p, vec2(wx.add(0.06 * K), lift.add(0.78 * K)), hand, 0.045 * K)));
      const cx = wx.add(0.22 * K), cy = lift.add(0.72 * K).add(U.held.mul(0.75 * K)), rc = U.held.mul(0.55 * K).add(0.001);
      const held = U.held.greaterThan(0.01).select(1, 0);
      put(rgb(0x2a2d33), fill(sdSegment(p, hand, vec2(cx, cy.add(rc.mul(0.15))), 0.024 * K)).mul(held));
      const cq = p.sub(vec2(cx, cy)), cl = length(cq);
      const canopy = max(cl.sub(rc), float(0).sub(cq.y));
      put(canopyTint, fill(canopy).mul(held));
      const rib = abs(fract(atan(cq.y, cq.x).div(Math.PI / 5).add(0.5)).sub(0.5)).mul(Math.PI / 5).mul(cl).sub(0.014);
      put(canopyTint.mul(0.55), fill(max(rib, canopy)).mul(held));
      put(rgb(0x2a2d33), fill(sdSegment(p, vec2(cx, cy.add(rc)), vec2(cx, cy.add(rc).add(0.1)), 0.022)).mul(held));

      const shelter = max(fill(abs(p.x.sub(cx)).sub(rc)).mul(fill(p.y.sub(cy))).mul(held), fill(abs(p.x.sub(6.7)).sub(1.3)).mul(fill(p.y.sub(0.3))));
      const rq = vec2(p.x.mul(7).add(p.y.mul(0.9)), p.y.mul(0.9).add(U.time.mul(10)));
      const cell = floor(rq), f = fract(rq), h = hash(cell);
      const streak = fill(abs(f.x.sub(h.mul(0.6).add(0.2))).div(7).sub(0.004)).mul(smoothstep(0, 0.2, f.y)).mul(smoothstep(0.75, 0.35, f.y));
      const on = hash(cell.add(7.3)).greaterThan(0.5).select(1, 0);
      put(rgb(0xc9dbef), streak.mul(on).mul(0.45).mul(U.rain).mul(float(1).sub(shelter)));
      return col;
    });

    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = shade();
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(16.6, 9.6).rotateX(-Math.PI / 2), material);
    this.scene.add(plane);

    const row = this.row, tally = this.tally;
    this.legend = {
      title: `UMBRELLA · ${L}${D ? ' · NOISE' : ''}`,
      rows: [
        { key: 'forecast', label: 'forecast', tint: RAIN, value: () => (row.forecast < 0 ? 'hidden' : row.forecast ? 'rain' : 'sun') },
        { key: 'umbrella', label: 'umbrella', tint: CANOPY, value: () => (row.deciding ? 'at the door' : row.have ? 'carried' : 'left home') },
        { key: 'block', label: 'block', tint: 0xe9e2d2, value: () => `${row.phase + 1} of ${L}` },
        { key: 'coin', label: 'reward', tint: GOLD, value: () => formatSigned(row.reward) },
        { key: 'last', label: 'last walk', tint: MISS, value: () => row.outcome },
        ...(D ? [{ key: 'noise', label: 'noise', tint: WINDOW, value: () => `${row.lit} of ${D} lit` }] : []),
      ],
      note: () => `${formatCount(tally.right)} right · ${formatCount(tally.wrong)} wrong`,
      bar: () => tally.right / Math.max(1, tally.right + tally.wrong),
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const data = this.data;
    if (!data) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const L = this.length, D = this.distractors, U = this.u, i = frame.state.step, head = frame.frameAt(i), ahead = frame.frameAt(i + 1);
    const next = !fast && ahead.step === i + 1 ? ahead : null, t = next ? clamp01(frame.position - i) : 0;
    const phase = head.words[UMBRELLA_PHASE]!, need = head.words[UMBRELLA_NEED]!, have = head.words[UMBRELLA_HAVE]!;
    const settling = next !== null && next.done !== 0;
    const stepping = next !== null && (settling || next.words[UMBRELLA_PHASE] === phase + 1);
    const w = stepping ? span(t, WALK) : 0, s = phase + (stepping ? ease(t, WALK) : 0);

    U.time.value = time;
    U.wx.value = this.walkerX(s);
    U.swing.value = Math.sin(TAU * w);
    U.bob.value = 0.035 * Math.abs(Math.sin(TAU * w));
    U.tile.value = s <= L - 0.5 ? Math.round(s) : -1;
    U.held.value = phase === 0 && next && !settling ? next.words[UMBRELLA_HAVE]! * ease(t, GRAB) : have;
    U.shutter.value = settling ? 1 - ease(t, OPEN) : phase > 0 ? 1 : next ? ease(t, CLOSE) : 0;
    U.forecast.value = settling ? next.observations[0]! : head.observations[0]!;

    let reveal = 0, shown = 0, verdict = 0;
    const last = this.settled(frame, head);
    if (settling) {
      reveal = ease(t, REVEAL);
      shown = need;
      verdict = next.reward;
    } else if (last && phase <= 1) {
      reveal = 1 - smoothstep01(clamp01((phase + t - 0.3) / 1.2));
      shown = last.need;
      verdict = last.reward;
    }
    U.rain.value = reveal * shown;
    U.sun.value = reveal * (1 - shown);
    U.verdict.value = reveal;
    this.glow.set(verdict > 0 ? GOLD : MISS);
    U.tint.value.set(this.glow.r, this.glow.g, this.glow.b);

    const bytes = data.image.data as Uint8Array;
    bytes.fill(0);
    const fade = settling ? 1 - ease(t, COIN_OUT) : 1;
    for (let p = 0; p < L; p++) if (p <= s) bytes[4 * p + 2] = Math.round(255 * fade);
    for (let p = 1; p <= phase; p++) {
      const f = frame.frameAt(i - phase + p);
      if (f.step !== i - phase + p) continue;
      bytes[4 * p + (f.reward > 0 ? 0 : 1)] = Math.round(255 * fade);
    }
    if (next && !settling && next.words[UMBRELLA_PHASE] === phase + 1) bytes[4 * (phase + 1) + (next.reward > 0 ? 0 : 1)] = Math.round(255 * ease(t, COIN_IN));
    let lit = 0;
    for (let d = 0; d < D; d++) {
      const a = head.observations[4 + d]!, b = next ? next.observations[4 + d]! : a;
      bytes[4 * (TEX + d)] = Math.round(255 * (a + (b - a) * ease(t, BLINK)));
      if (a > 0.5) lit++;
    }
    data.needsUpdate = true;

    if (!frame.scrubbed && !fast) {
      for (const item of frame.events) {
        if (item.type !== 'goal' && item.type !== 'harm') continue;
        this.point.set(XD, 0, -(G + 0.2));
        pool.ripple(this.point, item.type === 'goal' ? GOLD : MISS, item.type === 'goal' ? 1.2 : 0.7, time);
        if (item.type === 'goal') pool.pulse.fire(0.25);
      }
    }

    this.score(frame.eventLog, i);
    Object.assign(this.row, {
      phase, have, deciding: phase === 0, reward: head.reward, lit,
      forecast: phase === 0 ? head.observations[0]! : -1,
      outcome: last ? `${last.need ? 'rain' : 'sun'} · ${last.have ? 'umbrella' : 'none'} · ${last.reward > 0 ? 'right' : 'wrong'}` : '—',
    });
  }

  private score(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, right: 0, wrong: 0 });
    if (at === tally.through) return;
    tally.right += countEvents(log, 'goal', tally.through, at);
    tally.wrong += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.data?.dispose();
    this.data = null;
  }
}

export default (() => new UmbrellaScene()) satisfies SceneFactory;
