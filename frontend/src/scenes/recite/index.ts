import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { reciteCall, recitePhase, reciteRecalling, reciteSymbol } from '@sim/envs/recite';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';
import { GAP, GOLD, HEARD, IVORY, MISS, NAMES, PITCH, RECORD_Z, ROWS, SAID, SYM, boardMaterial, columnX, type Board } from './art';

type Span = readonly [number, number];

const APPEAR: Span = [0.28, 0.66];
const LISTEN_MOVE: Span = [0.1, 0.5];
const STAMP: Span = [0.05, 0.34];
const VERDICT: Span = [0.3, 0.5];
const RECITE_MOVE: Span = [0.5, 0.86];
const CLEAR: Span = [0.52, 0.72];
const SWAP = 0.72;
const ENTER: Span = [0.55, 0.95];
const WINDOW = 1000;
const LAMP = 1.5;
const NONE = -1;

const span = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const ease = (t: number, s: Span) => smoothstep01(span(t, s));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const named = (s: number) => (s === NONE ? '—' : NAMES[s]!);

class ReciteScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.28, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'recite';
  override legend: LegendSpec | null = null;

  private length = 4;
  private board: Board | null = null;
  private lamp: THREE.Mesh | null = null;
  private readonly lampTint = uniform(new THREE.Color(IVORY));
  private readonly lampColor = new THREE.Color();
  private readonly other = new THREE.Color();
  private readonly point = new THREE.Vector3();
  private base: EnvFrame | null = null;
  private readonly scores = new Int16Array(ROWS + 1);
  private readonly rated = { step: -1, count: -1 };
  private readonly ledger = { phase: '', shown: NONE, due: NONE, call: NONE, truth: NONE, reward: 0, last: NONE, seen: 0, right: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.length = parameter(ctx.environment, 'length');
    const w = this.length * PITCH / 2 + 0.9;
    return new THREE.Box2(new THREE.Vector2(-w, HEARD - 1.2), new THREE.Vector2(w, RECORD_Z + (ROWS - 1) * GAP + 0.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb4c4c8).multiplyScalar(0.62);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 24, sighSize: 12, sighRadius: 0.4, sighHeight: 0.8 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    this.legend = this.legendFor();
    pool.group.renderOrder = 10;
    const board = boardMaterial(this.length, light.lit);
    this.board = board;
    const box = this.extent(_ctx), centre = box.getCenter(new THREE.Vector2());
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(160, 160).rotateX(-Math.PI / 2), board.material);
    plane.position.set(centre.x, 0, centre.y);
    plane.frustumCulled = false;
    this.scene.add(plane);
    const lamp = light.emitter(new THREE.CircleGeometry(0.95, 40).rotateX(-Math.PI / 2), this.lampTint.mul(LAMP));
    lamp.frustumCulled = false;
    this.scene.add(lamp);
    this.lamp = lamp;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.board?.data.dispose();
  }

  private at(frame: FrameUpdate, s: number): EnvFrame | null {
    if (s < 0) return null;
    const found = frame.frameAt(s);
    return found.step === s ? found : null;
  }

  private check(frame: FrameUpdate, f: EnvFrame): void {
    const L = this.length, p = recitePhase(f), o = f.observations;
    const shown = p < L ? reciteSymbol(f, p) : NONE;
    for (let i = 0; i < 4; i++) if (o[i] !== (i === shown ? 1 : 0)) throw new Error(`Recite step ${f.step} shows the wrong symbol for phase ${p}`);
    const prev = f.step >= 1 ? this.at(frame, f.step - 1) : null;
    if (!prev) return;
    const q = recitePhase(prev);
    if ((q + 1) % (2 * L) !== p) throw new Error(`Recite step ${f.step} jumped from phase ${q} to ${p}`);
    const paid = q >= L && reciteCall(f) === reciteSymbol(prev, q - L) ? 1 : 0;
    if (f.reward !== paid) throw new Error(`Recite step ${f.step} paid ${f.reward} for calling ${named(reciteCall(f))} at phase ${q}`);
  }

  private spot(f: EnvFrame, out: THREE.Vector2): THREE.Vector2 {
    const L = this.length, p = recitePhase(f);
    return p < L ? out.set(columnX(L, p), HEARD) : out.set(columnX(L, p - L), SAID);
  }

  private lampOf(f: EnvFrame, out: THREE.Color): THREE.Color {
    const L = this.length, p = recitePhase(f);
    return p < L ? out.set(SYM[reciteSymbol(f, p)]!) : out.set(IVORY).multiplyScalar(0.5);
  }

  private put(row: number, j: number, x: number, y: number, z: number, w: number): void {
    const v = this.board!.values, i = (row * this.length + j) * 4;
    v[i] = x; v[i + 1] = y; v[i + 2] = z; v[i + 3] = w;
  }

  private fill(frame: FrameUpdate, base: EnvFrame, act: EnvFrame | null, f: number): void {
    const L = this.length, p = recitePhase(base), start = base.step - p;
    for (let j = 0; j < L; j++) {
      const heard = p >= L || j <= p ? 1 : 0;
      this.put(0, j, reciteSymbol(base, j), heard, p < L && j === p ? 1 : 0, 0);
      if (p < L || j >= p - L) continue;
      const c = this.at(frame, start + L + j + 1);
      if (!c) continue;
      const hit = c.reward > 0;
      this.put(0, j, reciteSymbol(base, j), 1, 0, 1);
      this.put(1, j, reciteCall(c), 1, hit ? 1 : -1, 0);
    }
    if (!act) return;
    if (p + 1 < L) {
      const a = ease(f, APPEAR);
      this.put(0, p + 1, reciteSymbol(base, p + 1), a, a, 0);
      this.put(0, p, reciteSymbol(base, p), 1, 1 - a, 0);
    } else if (p + 1 === L) {
      this.put(0, p, reciteSymbol(base, p), 1, 1 - ease(f, APPEAR), 0);
    } else {
      const j = p - L, v = ease(f, VERDICT), hit = act.reward > 0;
      this.put(0, j, reciteSymbol(base, j), 1, 0, v);
      this.put(1, j, reciteCall(act), ease(f, STAMP), (hit ? 1 : -1) * v, 0);
    }
  }

  private record(frame: FrameUpdate, end: number): void {
    const L = this.length;
    this.scores.fill(NONE);
    for (let r = 0; r <= ROWS; r++) {
      const e = end - r * 2 * L;
      if (e - 2 * L < 0) break;
      let hits = 0, whole = true;
      for (let j = 0; j < L; j++) {
        const c = this.at(frame, e - L + j + 1);
        if (!c || recitePhase(c) !== (L + j + 1) % (2 * L)) { whole = false; break; }
        const hit = c.reward > 0 ? 1 : 0;
        hits += hit;
        this.put(2 + r, j, 1, hit, 0, 0);
      }
      if (!whole) { for (let j = 0; j < L; j++) this.put(2 + r, j, 0, 0, 0, 0); break; }
      this.scores[r] = hits;
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const board = this.board, lamp = this.lamp, k = frame.state.step, now = this.at(frame, k);
    if (!board || !lamp || !now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0, L = this.length, p = recitePhase(now);
    this.check(frame, now);
    if (next) this.check(frame, next);
    this.celebrate(frame, pool, time, now);
    board.values.fill(0);
    const wiping = next !== null && recitePhase(next) === 0;
    if (wiping && f >= SWAP) {
      this.fill(frame, next, null, 0);
      board.fade.value = span(f, [SWAP, 1]);
      this.base = next;
    } else {
      this.fill(frame, now, next, f);
      board.fade.value = wiping ? 1 - ease(f, CLEAR) : 1;
      this.base = now;
    }
    board.enter.value = wiping ? ease(f, ENTER) : 1;
    this.record(frame, wiping ? k + 1 : k - p);
    board.commit();

    const from = this.spot(now, new THREE.Vector2());
    const m = next ? ease(f, p + 1 < L ? LISTEN_MOVE : RECITE_MOVE) : 0;
    if (next) from.lerp(this.spot(next, new THREE.Vector2()), m);
    board.cursor.value.copy(from);
    lamp.position.set(from.x, 0.02, from.y);
    this.lampOf(now, this.lampColor);
    if (next) this.lampColor.lerp(this.lampOf(next, this.other), m);
    this.lampTint.value.copy(this.lampColor);

    const prev = this.at(frame, k - 1), judged = prev !== null && reciteRecalling(prev);
    Object.assign(this.ledger, {
      phase: p < L ? `listening ${p + 1} of ${L}` : `reciting ${p - L + 1} of ${L}`,
      shown: p < L ? reciteSymbol(now, p) : NONE, due: p >= L ? reciteSymbol(now, p - L) : NONE,
      call: judged ? reciteCall(now) : NONE, truth: judged ? reciteSymbol(prev, recitePhase(prev) - L) : NONE, reward: now.reward,
      last: this.scores[wiping && f < SWAP ? 1 : 0]!,
    });
    this.rate(frame, k);
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number, now: EnvFrame): void {
    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    const L = this.length, j = (recitePhase(now) + 2 * L - 1) % (2 * L) - L;
    if (j < 0) return;
    const x = columnX(L, j);
    for (const e of frame.events) {
      if (e.type === 'goal') {
        pool.burst(this.point.set(x, 40, SAID), GOLD, 0.35, time);
        pool.ripple(this.point.set(x, 0.03, SAID), GOLD, 0.45, time);
      } else if (e.type === 'harm') pool.ripple(this.point.set(x, 0.03, SAID), MISS, 0.3, time);
      else if (e.type === 'done' && countEvents(frame.eventLog, 'goal', now.step - 2 * L, now.step) >= L) {
        pool.burst(this.point.set(0, 38, (HEARD + SAID) / 2), GOLD, 1.1, time);
        pool.ripple(this.point.set(0, 0.03, (HEARD + SAID) / 2), GOLD, 1.6, time);
      }
    }
  }

  private rate(frame: FrameUpdate, k: number): void {
    const log = frame.eventLog;
    if (k === this.rated.step && log.count === this.rated.count) return;
    const right = countEvents(log, 'goal', k - WINDOW, k), wrong = countEvents(log, 'harm', k - WINDOW, k);
    this.ledger.right = right;
    this.ledger.seen = right + wrong;
    this.rated.step = k;
    this.rated.count = log.count;
  }

  private legendFor(): LegendSpec {
    const l = this.ledger, L = this.length;
    return {
      title: 'RECITE',
      rows: [
        { key: 'phase', label: 'phase', tint: IVORY, value: () => l.phase },
        { key: 'shown', label: 'showing', tint: TOKENS.uiCyan,value: () => named(l.shown) },
        { key: 'due', label: 'due now', tint: TOKENS.cueMagenta, value: () => named(l.due) },
        { key: 'call', label: 'last call', tint: IVORY, value: () => (l.call === NONE ? '—' : `${named(l.call)} for ${named(l.truth)} · ${l.reward > 0 ? '+1' : '0'}`) },
        { key: 'last', label: 'last recital', tint: GOLD, value: () => (l.last === NONE ? '—' : `${l.last} of ${L}`) },
        { key: 'right', label: `right, last ${formatCount(WINDOW)}`, tint: GOLD, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      ],
      note: () => `hear ${formatCount(L)} symbols, then recite them in order · chance ${formatPercent(0.25, 0)}`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const base = this.base, board = this.board;
    if (!base || !board) return;
    const L = this.length, p = recitePhase(base), lift = 0.08, v = board.values;
    const heard = kit.marks('sequence', L), calls = kit.marks('calls', L);
    for (let j = 0; j < L; j++) {
      const x = columnX(L, j);
      heard.set(x + 0.4, lift, HEARD - 0.4, 0.09, SYM[reciteSymbol(base, j)]!);
      const i = (L + j) * 4;
      if (v[i + 1]! > 0.01) calls.set(x + 0.4, lift, SAID - 0.4, 0.09, SYM[v[i]!]!);
    }
    const target = p < L ? p : p - L, z = p < L ? HEARD : SAID, x = columnX(L, target), h = 0.62;
    const box = kit.curve('due', 5, p < L ? TOKENS.uiCyan : TOKENS.cueMagenta);
    for (const [dx, dz] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) box.point(x + dx * h, lift, z + dz * h);
    const link = kit.curve('link', 2, TOKENS.cueMagenta);
    link.point(x, lift, HEARD);
    link.point(x, lift, p < L ? HEARD : SAID);
  }
}

export default (() => new ReciteScene()) satisfies SceneFactory;
