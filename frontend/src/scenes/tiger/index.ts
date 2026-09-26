import * as THREE from 'three/webgpu';
import { mix, step, uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { tigerAction, tigerDoor, tigerHint, tigerListened, tigerMauled, tigerOpened, tigerTally } from '@sim/envs/tiger';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';
import { CHALK, DOOR, GAUGE, GOLD, GROWL, TORCH, disc, door, doorX, gauge, hall, worldZ } from './art';

type Span = readonly [number, number];
type At = (s: number) => EnvFrame | null;
interface Episode { start: number; n: number; tally: number }

const FAST = 5;
const GLIDE = 16;
const LOOKBACK = 4096;
const CLAW_LIFE = 1.4;
const CATCHUP = 64;

const ease = (t: number, [a, b]: Span) => smoothstep01(clamp01((t - a) / (b - a)));
const outcome = (f: EnvFrame) => (tigerMauled(f) ? 2 : 1);
const truth = (f: EnvFrame) => (tigerOpened(f) ? (tigerMauled(f) ? tigerAction(f) : 1 - tigerAction(f)) : tigerDoor(f));
const signed = (n: number) => (n > 0 ? `+${n}` : n < 0 ? `−${-n}` : '0');

class TigerScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'tiger';

  private ratio = 1;
  private accuracy = 0.5;
  private readonly time = uniform(0);
  private readonly flick = uniform(0);
  private readonly needle = uniform(Math.PI / 2);
  private readonly doors = [0, 1].map(() => ({ open: uniform(0), kind: uniform(0), growl: uniform(0), count: uniform(0), slash: uniform(0), time: this.time }));
  private readonly clawAt = [-Infinity, -Infinity];
  private readonly chambers: (THREE.Mesh | null)[] = [null, null];
  private pace = 0;
  private lastPosition = 0;
  private lastStep = -1;
  private cachedAt = -1;
  private head: Episode | null = null;
  private evidence: Episode | null = null;
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, listened: false, hint: 0, tally: 0, belief: 0.5, n: -1, last: '…' };

  override legend: LegendSpec | null = null;

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger;
    return {
      title: 'TIGER',
      rows: [
        { key: 'heard', label: 'growl heard', tint: GROWL, value: () => (l.step < 1 ? '—' : l.listened ? (l.hint ? 'behind right door' : 'behind left door') : 'none, a door opened') },
        { key: 'growls', label: 'growls left · right', tint: CHALK, value: () => (l.n < 0 ? '…' : `${formatCount((l.n - l.tally) / 2)} · ${formatCount((l.n + l.tally) / 2)}`) },
        { key: 'belief', label: 'tiger behind right', tint: accent, value: () => `${formatPercent(l.belief)} (tally ${signed(l.tally)})` },
        { key: 'last', label: 'last door opened', tint: GOLD, value: () => l.last },
      ],
      note: () => `each growl comes from the tiger's door ${formatPercent(this.accuracy, 0)} of the time`,
      bar: () => l.belief,
    };
  }

  private belief(tally: number): number {
    return 1 / (1 + Math.pow(this.ratio, tally));
  }

  private angle(tally: number): number {
    return Math.PI * (1 - this.belief(tally));
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.accuracy = parameter(ctx.environment, 'accuracy');
    this.ratio = (1 - this.accuracy) / this.accuracy;
    return new THREE.Box2(new THREE.Vector2(-6.9, -3.6), new THREE.Vector2(6.9, 3.6));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8a7a68);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, now = this.reader(frame)(i);
        if (!now) return;
        kit.marks('tiger', 1).set(doorX(truth(now)), 0.1, worldZ(DOOR.mid), 0.4, TOKENS.cueMagenta);
        if (tigerListened(now)) kit.marks('heard', 1).set(doorX(tigerHint(now)), 0.1, worldZ(DOOR.spring + 0.9), 0.16, TOKENS.uiCyan);
        const a = this.needle.value, r = GAUGE.r - 0.12;
        kit.marks('needle', 1).set(GAUGE.x + Math.cos(a) * r, 0.1, worldZ(GAUGE.y + Math.sin(a) * r), 0.1, accent);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit;
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(0x0a0705);
    this.scene.add(hall(lit, this.flick), door(-1, this.doors[0]!, lit), door(1, this.doors[1]!, lit), gauge(this.needle));
    const place = (mesh: THREE.Mesh, x: number, y: number) => {
      mesh.position.set(x, 0, worldZ(y));
      mesh.renderOrder = 1;
      this.scene.add(mesh);
      return mesh;
    };
    for (const side of [-1, 1]) place(light.emitter(disc(0.6), rgb(0xff9a3c).mul(this.flick.mul(0.4).add(2.4))), side * TORCH.x, TORCH.y + 0.3);
    this.doors.forEach((d, k) => (this.chambers[k] = place(light.emitter(disc(1.1), mix(rgb(0xffb347), rgb(0xff3a20), step(1.5, d.kind)).mul(d.open.mul(2.5))), doorX(k), -1.8)));
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private reader(frame: FrameUpdate): At {
    return (s) => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
  }

  private episode(s: number, at: At): Episode | null {
    const end = at(s);
    if (!end) return null;
    if (tigerOpened(end)) return { start: s, n: 0, tally: 0 };
    for (let g = s; g > s - LOOKBACK; g--) {
      if (g <= 0) return this.checked(s, 0, end);
      const f = at(g);
      if (!f) return null;
      if (tigerOpened(f)) return this.checked(s, g, end);
    }
    return null;
  }

  private checked(s: number, start: number, end: EnvFrame): Episode {
    const n = s - start, tally = tigerTally(end);
    if (Math.abs(tally) > n || (n - tally) % 2 !== 0) throw new Error(`Tiger step ${s} tallies ${tally} over ${n} listens`);
    return { start, n, tally };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal;
    pool.update(dt, time);
    this.time.value = time;
    this.flick.value = Math.sin(time * 7.3) * 0.6 + Math.sin(time * 12.1) * 0.4;
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, head = frame.frameAt(i), fast = this.pace > this.controlHz * FAST;
    const next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    if (this.cachedAt !== i) {
      this.cachedAt = i;
      this.head = this.episode(i, at);
      this.evidence = tigerOpened(head) ? this.episode(i - 1, at) : null;
    }
    const opened = tigerOpened(head), both = !!next && opened && tigerOpened(next);
    const open = [0, 0], kind = [0, 0], growl = [0, 0];
    if (opened) {
      const d = tigerAction(head);
      open[d] = next ? 1 - ease(t, both ? [0.02, 0.4] : [0.5, 0.92]) : 1;
      kind[d] = outcome(head);
    }
    if (next && tigerOpened(next)) {
      const d = tigerAction(next), o = ease(t, both ? [0.5, 0.9] : [0.12, 0.5]);
      if (o >= open[d]!) {
        open[d] = o;
        kind[d] = outcome(next);
      }
    }
    if (tigerListened(head)) growl[tigerHint(head)] = next ? 1 - ease(t, [0, 0.35]) : 1;
    if (next && tigerListened(next)) growl[tigerHint(next)] = Math.max(growl[tigerHint(next)]!, ease(t, [0.4, 0.75]));

    const shown = opened ? this.evidence : this.head;
    let tally = opened ? (this.evidence?.tally ?? 0) : tigerTally(head);
    let target = this.angle(tally);
    let episode = shown;
    if (next) {
      const k = opened ? ease(t, [0.5, 0.9]) : ease(t, [0.45, 0.8]);
      const after = tigerOpened(next) ? tally : tigerTally(next);
      target += (this.angle(after) - target) * k;
      if (t >= (opened ? 0.7 : 0.6) && !tigerOpened(next)) {
        tally = after;
        episode = this.head ? { start: this.head.start, n: this.head.n + 1, tally: after } : null;
      } else if (t >= 0.7 && opened) episode = { start: i, n: 0, tally: 0 };
      this.needle.value = target;
    } else {
      const k = frame.scrubbed ? 1 : 1 - Math.exp(-GLIDE * dt);
      this.needle.value += (target - this.needle.value) * k;
    }
    this.doors.forEach((d, k) => {
      d.open.value = open[k]!;
      if (this.chambers[k]) this.chambers[k].visible = open[k]! > 0.001;
      d.kind.value = kind[k]!;
      d.growl.value = growl[k]!;
      d.count.value = episode ? (episode.n + (k === 1 ? episode.tally : -episode.tally)) / 2 : -1;
    });

    this.celebrate(frame, i, at, pool, time);
    this.doors.forEach((d, k) => (d.slash.value = Math.max(0, 1 - (time - this.clawAt[k]!) / CLAW_LIFE)));
    const l = this.ledger;
    l.step = i;
    l.listened = tigerListened(head);
    l.hint = tigerHint(head);
    l.tally = tigerTally(head);
    l.belief = this.belief(l.tally);
    l.n = this.head ? this.head.n : -1;
    const last = this.head ? at(this.head.start) : null;
    l.last = !this.head ? '…' : this.head.start === 0 || !last ? 'none yet' : `${tigerAction(last) ? 'right' : 'left'}, ${tigerMauled(last) ? 'tiger −10' : 'treasure +1'}`;
  }

  private celebrate(frame: FrameUpdate, i: number, at: At, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed || i < this.lastStep) {
      pool.clear();
      this.clawAt.fill(-Infinity);
      this.lastStep = i;
      return;
    }
    for (let s = Math.max(this.lastStep + 1, i - CATCHUP); s <= i; s++) {
      const f = at(s);
      if (!f || !tigerOpened(f)) continue;
      const d = tigerAction(f);
      if (tigerMauled(f)) this.clawAt[d] = time;
      else pool.burst(this.point.set(doorX(d), 40, worldZ(-1.6)), TOKENS.celebrationGold, 0.8, time);
    }
    this.lastStep = i;
  }
}

export default (() => new TigerScene()) satisfies SceneFactory;
