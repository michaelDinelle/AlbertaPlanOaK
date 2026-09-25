import * as THREE from 'three/webgpu';
import { abs, clamp, exp2, float, floor, fract, fwidth, max, min, mix, mod, positionWorld, step, texture, uniform, vec2, vec3 } from 'three/tsl';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { cellBit, cellFlip } from '@sim/envs/cellular';
import type { EnvFrame } from '@sim/types';
import { constant } from '../../data/envs';
import { formatCount, formatPercent } from '../../format';

type N<T extends string> = THREE.Node<T>;

const ROWS = 22;
const HOLD = 0.35;
const WINDOW = 1000;
const FLASH = 0.35;
const LANE = 1.8;
const INK = 0x0a0e14;
const PANEL = 0x111922;
const DEAD = 0x1c2631;
const VOID = 0x141b24;
const GOLD = TOKENS.celebrationGold;
const LEVER = TOKENS.cueMagenta;
const BLUE = TOKENS.failureBlue;

const cover = (d: N<'float'>, aa: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(aa)), 0, 1);

interface Board {
  cells: number;
  hist: THREE.DataTexture;
  slide: N<'float'>;
  rule: N<'float'>;
  fired: N<'float'>;
  flash: N<'float'>;
  alive: number;
}

function boardMaterial(b: Board): THREE.MeshBasicNodeMaterial {
  const n = b.cells, width = n + 1, height = ROWS + 1;
  const x = positionWorld.x, z = positionWorld.z, aa = max(fwidth(x), 1e-4);
  let color: N<'vec3'> = rgb(INK);

  const panel = sdBox(vec2(x.sub(n / 2), z.sub(ROWS / 2)), vec2(n / 2 + LANE + 0.75, ROWS / 2 + 0.45), 0.5);
  color = mix(color, rgb(PANEL), cover(panel, aa));

  const t = float(ROWS).sub(z), rowF = t.sub(b.slide).add(1), age = floor(rowF), fz = fract(rowF);
  const inRows = cover(abs(t.sub(ROWS / 2)).sub(ROWS / 2), aa);
  const v = age.add(0.5).div(height);
  const fade = mix(float(1), float(0.3), clamp(t.sub(3).div(ROWS - 3), 0, 1));

  const cx = floor(x), col = mod(cx.add(n), n);
  const cell = texture(b.hist, vec2(col.add(0.5).div(width), v));
  const inGrid = cover(abs(x.sub(n / 2)).sub(n / 2 + 1), aa).mul(inRows);
  const real = cover(abs(x.sub(n / 2)).sub(n / 2), aa);
  const local = vec2(fract(x).sub(0.5), fz.sub(0.5));

  const leverBand = cover(abs(x.sub(0.5)).sub(0.5), aa).mul(inRows).mul(cell.a);
  const goldBand = cover(abs(x.sub(1.5)).sub(0.5), aa).mul(inRows).mul(cell.a);
  color = mix(color, rgb(LEVER).mul(0.16).add(rgb(PANEL).mul(0.84)), leverBand);
  color = mix(color, rgb(GOLD).mul(0.18).add(rgb(PANEL).mul(0.82)), goldBand);

  const tile = cover(sdBox(local, vec2(0.39, 0.39), 0.1), aa);
  const lit = rgb(b.alive).mul(float(0.66).add(local.y.mul(-0.22)).add(0.12));
  const tone = mix(rgb(DEAD), lit, cell.r);
  const shown = mix(rgb(VOID), tone, cell.a).mul(fade).mul(mix(float(0.38), float(1), real));
  color = mix(color, shown, tile.mul(inGrid));

  const ring = cover(abs(sdCircle(local, 0.2)).sub(0.05), aa).mul(cell.g).mul(inGrid);
  color = mix(color, rgb(LEVER).mul(0.82).mul(mix(float(0.45), float(1), real)), ring);

  const hold = float(1).sub(b.slide);
  const box = cover(abs(sdBox(local, vec2(0.47, 0.47), 0.12)).sub(0.035), aa).mul(real).mul(inRows).mul(hold);
  const neigh = step(0.5, cell.b).mul(step(cell.b, 1.5)), outcome = step(1.5, cell.b);
  color = mix(color, vec3(0.8, 0.82, 0.86), box.mul(neigh));
  color = mix(color, rgb(GOLD).mul(0.8).mul(float(1).add(b.flash.mul(0.12))), box.mul(outcome));

  const laneL = vec2(x.add(LANE), fz.sub(0.5));
  const lever = texture(b.hist, vec2(float(0.5).div(width), v));
  const inLeft = cover(abs(x.add(LANE)).sub(0.5), aa).mul(inRows);
  const chevron = min(sdSegment(laneL, vec2(-0.12, -0.2), vec2(0.13, 0), 0.055), sdSegment(laneL, vec2(-0.12, 0.2), vec2(0.13, 0), 0.055));
  color = mix(color, rgb(LEVER).mul(0.8).mul(fade), cover(chevron, aa).mul(lever.g).mul(inLeft));
  color = mix(color, rgb(DEAD).mul(fade), cover(sdCircle(laneL, 0.05), aa).mul(float(1).sub(lever.g)).mul(lever.a).mul(inLeft));

  const laneR = vec2(x.sub(n + LANE), fz.sub(0.5));
  const paid = texture(b.hist, vec2(float(n + 0.5).div(width), v));
  const inRight = cover(abs(x.sub(n + LANE)).sub(0.5), aa).mul(inRows).mul(paid.a);
  const newest = step(age, 1.5).mul(step(0.5, age));
  const glow = float(0.78).add(b.flash.mul(newest).mul(0.14));
  color = mix(color, rgb(GOLD).mul(glow).mul(fade), cover(sdCircle(laneR, 0.27), aa).mul(paid.r).mul(inRight));
  color = mix(color, rgb(BLUE).mul(0.8).mul(fade), cover(abs(sdCircle(laneR, 0.2)).sub(0.045), aa).mul(float(1).sub(paid.r)).mul(inRight));

  const nowLine = cover(abs(z.sub(ROWS + 0.12)).sub(0.03), aa).mul(real);
  color = mix(color, rgb(b.alive).mul(0.5), nowLine);

  const span = n + 2 * LANE + 1, sw = span / 8, sx = x.add(LANE + 0.5), slot = clamp(floor(sx.div(sw)), 0, 7), lx = sx.sub(slot.add(0.5).mul(sw));
  const pattern = float(7).sub(slot), pitch = Math.min(sw / 3.6, 0.72), half = pitch * 0.42;
  const top = ROWS + 1.35, bottom = top + pitch * 1.25, mid = (top + bottom) / 2;
  const inStrip = cover(abs(sx.sub(span / 2)).sub(span / 2), aa);
  const k = clamp(floor(lx.div(pitch).add(1.5)), 0, 2), lxx = lx.sub(k.sub(1).mul(pitch));
  const topBit = mod(floor(pattern.div(exp2(float(2).sub(k)))), 2);
  const outBit = mod(floor(b.rule.div(exp2(pattern))), 2);
  const on = float(1).sub(step(0.5, abs(pattern.sub(b.fired))));
  const frame = cover(abs(sdBox(vec2(lx, z.sub(mid)), vec2(pitch * 1.5 + 0.14, pitch * 1.05 + 0.14), 0.16)).sub(0.035), aa).mul(inStrip);
  color = mix(color, mix(rgb(DEAD).mul(1.4), rgb(GOLD).mul(0.8), on), frame.mul(mix(float(0.6), float(1), on)));
  const topTile = cover(sdBox(vec2(lxx, z.sub(top)), vec2(half, half), half * 0.3), aa).mul(inStrip);
  const outTile = cover(sdBox(vec2(lx, z.sub(bottom)), vec2(half, half), half * 0.3), aa).mul(inStrip);
  const dim = mix(float(0.6), float(1), on);
  color = mix(color, mix(rgb(DEAD).mul(1.25), rgb(b.alive).mul(0.72), topBit).mul(dim), topTile);
  color = mix(color, mix(rgb(DEAD).mul(1.25), rgb(b.alive).mul(0.72), outBit).mul(dim), outTile);

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  return material;
}

class CellularScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.2, fog: { from: ['night', 'skyTop'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'cellular';
  override legend: LegendSpec | null = null;

  private cells = 16;
  private rule = 110;
  private data = new Float32Array(4);
  private hist = new THREE.DataTexture(this.data, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly slide = uniform(0);
  private readonly fired = uniform(-1);
  private readonly flash = uniform(0);
  private readonly ruleU = uniform(0);
  private readonly point = new THREE.Vector3();
  private filled = -1;
  private filledNext = -1;
  private shown = -1;
  private landed = -Infinity;
  private readonly ledger = { step: 0, flip: -1, scored: 0, reward: 0, alive: 0, seen: 0, right: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.cells = constant(ctx.environment, 'CELLULAR_CELLS');
    this.rule = constant(ctx.environment, 'CELLULAR_RULE');
    this.data = new Float32Array((this.cells + 1) * (ROWS + 1) * 4);
    this.hist = new THREE.DataTexture(this.data, this.cells + 1, ROWS + 1, THREE.RGBAFormat, THREE.FloatType);
    return new THREE.Box2(new THREE.Vector2(-LANE - 1.2, -0.9), new THREE.Vector2(this.cells + LANE + 1.2, ROWS + 3.6));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xffffff);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 18, sighSize: 12, sighRadius: 0.3, sighHeight: 0.8 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { pool }: Stage2D): void {
    this.legend = this.legendFor(this.accentOf(ctx));
    this.scene.background = new THREE.Color(INK);
    pool.group.renderOrder = 10;
    this.hist.magFilter = this.hist.minFilter = THREE.NearestFilter;
    this.hist.needsUpdate = true;
    this.ruleU.value = this.rule;
    const board: Board = { cells: this.cells, hist: this.hist, slide: this.slide, rule: this.ruleU, fired: this.fired, flash: this.flash, alive: this.accentOf(ctx) };
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(200, 200).rotateX(-Math.PI / 2), boardMaterial(board));
    mesh.position.set(this.cells / 2, 0, ROWS / 2);
    mesh.frustumCulled = false;
    mesh.renderOrder = 0;
    this.scene.add(mesh);
  }

  protected override drop(): void {
    this.hist.dispose();
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

  private pattern(prev: EnvFrame, f: EnvFrame): number {
    return ((cellBit(prev, 0) ^ cellFlip(f)) << 2) | (cellBit(prev, 1) << 1) | cellBit(prev, 2);
  }

  private check(prev: EnvFrame | null, f: EnvFrame): void {
    const scored = cellBit(f, 1);
    if ((f.reward !== 0 && f.reward !== 1) || f.reward !== scored || (prev && ((this.rule >>> this.pattern(prev, f)) & 1) !== scored)) {
      throw new Error(`Cellular step ${f.step} paid ${f.reward} while cell 1 is ${scored}${prev ? ` after neighbourhood ${this.pattern(prev, f).toString(2).padStart(3, '0')}` : ''}`);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), prev = this.at(frame, k - 1), f = next ? clamp01(frame.position - k) : 0;
    const slide = smoothstep01((f - HOLD) / (1 - HOLD));
    if (k !== this.shown) {
      this.landed = !frame.scrubbed && k === this.shown + 1 ? time : -Infinity;
      this.shown = k;
    }
    this.fill(frame, k, next);
    this.slide.value = slide;
    this.flash.value = Math.exp(-(time - this.landed) / FLASH);
    if (k > 0) this.check(prev, now);
    this.fired.value = k > 0 && prev ? this.pattern(prev, now) : -1;

    let alive = 0;
    for (let i = 0; i < this.cells; i++) alive += cellBit(now, i);
    const right = countEvents(frame.eventLog, 'goal', k - WINDOW, k), wrong = countEvents(frame.eventLog, 'harm', k - WINDOW, k);
    Object.assign(this.ledger, { step: k, flip: k > 0 ? cellFlip(now) : -1, scored: cellBit(now, 1), reward: now.reward, alive, seen: right + wrong, right });

    if (frame.scrubbed) {
      pool.clear();
      return;
    }
    for (const event of frame.events) {
      if (event.type === 'goal') pool.burst(this.point.set(this.cells + LANE, 40, ROWS - 0.5), GOLD, 0.22, time);
      else if (event.type === 'harm') pool.ripple(this.point.set(1.5, 0.05, ROWS - 0.5), BLUE, 0.7, time);
    }
  }

  private fill(frame: FrameUpdate, k: number, next: EnvFrame | null): void {
    const nextStep = next ? next.step : -1;
    if (this.filled === k && this.filledNext === nextStep) return;
    const data = this.data, n = this.cells, width = n + 1;
    data.fill(0);
    let later: EnvFrame | null = null;
    for (let a = 0; a <= ROWS; a++) {
      const s = k + 1 - a, f = a === 0 ? next : this.at(frame, s), row = a * width * 4;
      if (f) {
        for (let i = 0; i < n; i++) {
          const o = row + i * 4;
          data[o] = cellBit(f, i);
          data[o + 3] = 1;
        }
        if (later) data[row + 1] = cellFlip(later);
        if (s >= 1) {
          data[row + n * 4] = f.reward > 0 ? 1 : 0;
          data[row + n * 4 + 3] = 1;
        }
        if (a === 1 && s >= 1) data[row + 1 * 4 + 2] = 2;
        if (a === 2 && k >= 1) for (let i = 0; i < 3; i++) data[row + i * 4 + 2] = 1;
      }
      later = f;
    }
    this.hist.needsUpdate = true;
    this.filled = k;
    this.filledNext = nextStep;
  }

  private legendFor(accent: number): LegendSpec {
    const l = this.ledger, n = this.cells;
    return {
      title: 'CELLULAR',
      rows: [
        { key: 'flip', label: 'lever · cell 0', tint: LEVER, value: () => (l.flip < 0 ? '—' : l.flip ? 'flip' : 'keep') },
        { key: 'scored', label: 'scored · cell 1', tint: GOLD, value: () => (l.scored ? 'alive' : 'dead') },
        { key: 'alive', label: 'alive cells', tint: accent, value: () => `${l.alive} / ${n}` },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => (l.step < 1 ? '—' : l.reward > 0 ? '+1' : '0') },
        { key: 'right', label: `alive, last ${formatCount(WINDOW)}`, tint: GOLD, value: () => formatPercent(l.seen > 0 ? l.right / l.seen : NaN) },
      ],
      note: () => `rule ${this.rule} · ${formatCount(n)}-cell ring`,
      bar: () => (l.seen > 0 ? l.right / l.seen : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const n = this.cells, slide = this.slide.value;
    const marks = kit.marks('cells', n * (ROWS + 1)), lanes = kit.marks('reward', ROWS + 1), levers = kit.marks('lever', ROWS + 1);
    let later: EnvFrame | null = null;
    for (let a = 0; a <= ROWS; a++) {
      const s = k + 1 - a, f = a === 0 ? this.at(frame, k + 1) : this.at(frame, s), t = a - 0.5 + slide, z = ROWS - t;
      const shown = f && t >= 0 && t <= ROWS;
      if (shown) {
        for (let i = 0; i < n; i++) if (cellBit(f, i)) marks.set(i + 0.5, 0.08, z, 0.12, 0xffffff);
        if (s >= 1) lanes.set(n + LANE, 0.08, z, 0.14, f.reward > 0 ? GOLD : BLUE);
        if (later && cellFlip(later)) levers.set(-LANE, 0.08, z, 0.14, LEVER);
      }
      later = f;
    }
    if (k > 0) kit.marks('scored', 1).set(1.5, 0.08, ROWS - 0.5 - slide, cellBit(now, 1) ? 0.24 : 0.08, GOLD);
  }
}

export default (() => new CellularScene()) satisfies SceneFactory;
