import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { lastEvent } from '@sim/events';
import { hiddenModeStation, hiddenModeTuned } from '@sim/envs/hiddenmode';
import type { EnvFrame, EventStream } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';
import { BANDS, CABINET, CARRIER, DIAL, DIAL_Y, EYE, KNOB, MID, NEEDLE, PASS, PHOSPHOR, ROW, ROWS, SCREEN, STATION_Y, bandX, cabinet, desk, dial, disc, eye, flat, knob, screen } from './art';

type Span = readonly [number, number];
type At = (s: number) => EnvFrame | null;

const TUNE: Span = [0, 0.3];
const READ: Span = [0.22, 0.42];
const PRINT: Span = [0.3, 0.62];
const HOP: Span = [0.62, 0.95];
const FAST = 5;
const GLIDE = 16;
const LOOKBACK = 4096;
const FLARE = 0.9;
const NEEDLE_Y = (NEEDLE.y0 + NEEDLE.y1) / 2;
const MISS = 0xff6b5a;

const ease = (t: number, [a, b]: Span) => smoothstep01(clamp01((t - a) / (b - a)));

function eventsKnown(stream: EventStream, from: number, to: number): boolean {
  for (let b = Math.floor(from / stream.blockFrames); b <= Math.floor(to / stream.blockFrames); b++) if (!stream.blocks[b]) return false;
  return true;
}

class HiddenModeScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'hiddenMode';

  private switchRate = 0;
  private readonly rows = new Float32Array((ROWS + 1) * 4);
  private readonly history = new THREE.DataTexture(this.rows, ROWS + 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly scroll = uniform(0);
  private readonly head = uniform(0);
  private readonly needle = uniform(MID);
  private readonly station = uniform(BANDS[0]);
  private readonly signal = uniform(0);
  private readonly turn = uniform(0);
  private readonly flare = uniform(0);
  private hopAt = -Infinity;
  private stationLamp: THREE.Mesh | null = null;
  private needleLamp: THREE.Mesh | null = null;
  private filled = '';
  private complete = false;
  private pace = 0;
  private lastPosition = 0;
  private lastStep = -1;
  private settled = -1;
  private readonly point = new THREE.Vector3();
  private readonly ledger = { step: 0, station: 0, tuned: -1, reward: 0, share: NaN, since: null as number | null, known: false };

  override legend: LegendSpec | null = null;

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, ago = (n: number) => (n === 0 ? 'this step' : `${formatCount(n)} ${n === 1 ? 'step' : 'steps'} ago`);
    return {
      title: 'HIDDEN MODE',
      rows: [
        { key: 'mode', label: 'hidden mode', tint: CARRIER, value: () => `station on band ${l.station}` },
        { key: 'tuned', label: 'tuned', tint: accent, value: () => (l.step > 0 ? `band ${l.tuned}` : '—') },
        { key: 'reward', label: 'reward', tint: PHOSPHOR, value: () => (l.step > 0 ? formatSigned(l.reward) : '—') },
        { key: 'share', label: `on station, last ${ROWS}`, tint: accent, value: () => formatPercent(l.share) },
        { key: 'switch', label: 'last switch', tint: CARRIER, value: () => (!l.known ? '…' : l.since === null ? 'none yet' : ago(l.since)) },
      ],
      note: () => `switches 1 in ${formatCount(1 / this.switchRate)} steps`,
      bar: () => (Number.isFinite(l.share) ? l.share : 0),
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.switchRate = parameter(ctx.environment, 'switchRate');
    return new THREE.Box2(new THREE.Vector2(-CABINET.w - 0.45, -CABINET.h - 0.45), new THREE.Vector2(CABINET.w + 0.45, CABINET.h + 0.45));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x8e96b8).multiplyScalar(1.1);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, scroll = this.scroll.value, at = this.reader(frame);
        const carrier = kit.marks('carrier', ROWS + 1), passband = kit.marks('passband', 2 * (ROWS + 1));
        for (let m = 0; m <= ROWS; m++) {
          const s = i + 1 - m, cur = at(s), prev = at(s - 1), y = SCREEN.y1 - (m - 0.5 + scroll) * ROW;
          if (s < 1 || !cur || !prev || y > SCREEN.y1 || y < SCREEN.y0) continue;
          const x = bandX(hiddenModeTuned(cur)), tint = cur.reward > 0 ? TOKENS.uiCyan : MISS;
          carrier.set(bandX(hiddenModeStation(prev)), 0.1, -y, 0.03, TOKENS.cueMagenta);
          passband.set(x - PASS, 0.1, -y, 0.026, tint);
          passband.set(x + PASS, 0.1, -y, 0.026, tint);
        }
        const now = at(i);
        if (!now) return;
        kit.marks('station', 1).set(bandX(hiddenModeStation(now)), 0.1, -STATION_Y, 0.12, TOKENS.cueMagenta);
        if (i >= 1) kit.marks('tuned', 1).set(bandX(hiddenModeTuned(now)), 0.1, -NEEDLE_Y, 0.08, accent);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit;
    this.legend = this.legendFor(accent);
    this.scene.background = new THREE.Color(0x050403);
    pool.group.renderOrder = 10;
    this.history.magFilter = this.history.minFilter = THREE.NearestFilter;
    this.history.needsUpdate = true;
    this.scene.add(desk(lit), cabinet(lit), screen(accent, this.history, this.scroll, this.head), dial(accent, { needle: this.needle, station: this.station, lock: this.signal, flare: this.flare }), eye(this.signal, lit), knob(this.turn, lit));
    const place = (mesh: THREE.Mesh, x: number, y: number, order: number) => {
      mesh.position.set(x, 0, -y);
      mesh.renderOrder = order;
      this.scene.add(mesh);
      return mesh;
    };
    place(light.emitter(flat(DIAL.x1 - DIAL.x0, DIAL.y1 - DIAL.y0), rgb(0xffc27a).mul(1.6)), (DIAL.x0 + DIAL.x1) / 2, DIAL_Y, 1);
    place(light.emitter(flat(SCREEN.x1 - SCREEN.x0, SCREEN.y1 - SCREEN.y0), rgb(0x303a8c).mul(0.5)), (SCREEN.x0 + SCREEN.x1) / 2, (SCREEN.y0 + SCREEN.y1) / 2, 1);
    place(light.emitter(disc(EYE.r * 0.78), rgb(PHOSPHOR).mul(this.signal.mul(1.4).add(1.4))), EYE.x, EYE.y, 1);
    place(light.occluder(disc(KNOB.r)), KNOB.x, KNOB.y, 1);
    this.stationLamp = place(light.emitter(disc(0.3), rgb(0xff8a30).mul(this.flare.mul(8).add(4))), BANDS[0], STATION_Y, 2);
    this.needleLamp = place(light.emitter(disc(0.1), rgb(accent).mul(this.signal.mul(4).add(0.8))), MID, NEEDLE_Y, 3);
  }

  protected override drop(): void {
    this.history.dispose();
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

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, head = frame.frameAt(i), fast = this.pace > this.controlHz * FAST;
    const next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    this.fill(i, next, at);

    const station = hiddenModeStation(head), tuned = i >= 1 ? hiddenModeTuned(head) : -1;
    let needle = tuned < 0 ? MID : bandX(tuned), stationX = bandX(station), signal = i >= 1 && head.reward > 0 ? 1 : 0, scroll = 0;
    if (next) {
      const after = hiddenModeStation(next);
      needle += (bandX(hiddenModeTuned(next)) - needle) * ease(t, TUNE);
      signal += ((next.reward > 0 ? 1 : 0) - signal) * ease(t, READ);
      scroll = ease(t, PRINT);
      if (after !== station) stationX += (bandX(after) - stationX) * ease(t, HOP);
    } else {
      const k = frame.scrubbed ? 1 : 1 - Math.exp(-GLIDE * dt);
      needle = this.needle.value + (needle - this.needle.value) * k;
      stationX = this.station.value + (stationX - this.station.value) * k;
      signal = this.signal.value + (signal - this.signal.value) * k;
    }
    this.needle.value = needle;
    this.station.value = stationX;
    this.signal.value = signal;
    this.scroll.value = scroll;
    this.head.value = i;
    this.turn.value = KNOB.swing * ((2 * (needle - BANDS[0])) / (BANDS[1] - BANDS[0]) - 1);
    if (this.stationLamp) this.stationLamp.position.x = stationX;
    if (this.needleLamp) this.needleLamp.position.x = needle;

    this.celebrate(frame, i, at, pool, time);
    this.flare.value = Math.max(0, 1 - (time - this.hopAt) / FLARE);
    const l = this.ledger;
    l.step = i;
    l.station = station;
    l.tuned = tuned;
    l.reward = head.reward;
    if (this.settled !== i) {
      const log = frame.eventLog, found = lastEvent(log, 'cue', i);
      l.known = eventsKnown(log, found ?? 0, i);
      l.since = found === undefined ? null : i - found;
      if (l.known) this.settled = i;
    }
  }

  private fill(i: number, next: EnvFrame | null, at: At): void {
    const key = `${i}:${next ? 1 : 0}`;
    if (key === this.filled && this.complete) return;
    const rows = this.rows;
    let prev = at(i - ROWS), hits = 0, count = 0;
    for (let m = ROWS; m >= 0; m--) {
      const s = i + 1 - m, cur = m === 0 ? next : at(s), o = m * 4;
      if (s >= 1 && cur && prev) {
        const carrier = hiddenModeStation(prev), tuned = hiddenModeTuned(cur), heard = cur.reward > 0 ? 1 : 0;
        if ((heard === 1) !== (tuned === carrier)) throw new Error(`HiddenMode step ${s} paid ${cur.reward} tuned to band ${tuned} while the station was on band ${carrier}`);
        rows[o] = carrier;
        rows[o + 1] = tuned;
        rows[o + 2] = heard;
        rows[o + 3] = 1;
        if (m > 0) {
          hits += heard;
          count++;
        }
      } else rows.fill(0, o, o + 4);
      prev = cur;
    }
    this.history.needsUpdate = true;
    this.filled = key;
    this.complete = count === Math.min(ROWS, i);
    this.ledger.share = count > 0 ? hits / count : NaN;
  }

  private celebrate(frame: FrameUpdate, i: number, at: At, pool: CelebrationPool, time: number): void {
    if (frame.scrubbed || i < this.lastStep) {
      pool.clear();
      this.hopAt = -Infinity;
      this.lastStep = i;
      return;
    }
    if (frame.events.some((event) => event.type === 'cue')) this.hopAt = time;
    for (let s = this.lastStep + 1; s <= i; s++) {
      const before = at(s - 1);
      if (before && this.relocked(s, before, at)) pool.burst(this.point.set(bandX(hiddenModeStation(before)), 38, -(SCREEN.y1 - 0.25)), TOKENS.celebrationGold, 0.55, time);
    }
    this.lastStep = i;
  }

  private relocked(s: number, before: EnvFrame, at: At): boolean {
    const now = at(s);
    if (!now || now.reward <= 0) return false;
    let hit = before;
    for (let g = s - 1; g >= 1 && g > s - LOOKBACK; g--) {
      const prior = at(g - 1);
      if (!prior) return false;
      if (hit.reward > 0) return hiddenModeStation(prior) !== hiddenModeStation(before);
      hit = prior;
    }
    return false;
  }
}

export default (() => new HiddenModeScene()) satisfies SceneFactory;
