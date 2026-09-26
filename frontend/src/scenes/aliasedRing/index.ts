import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, normalize, positionWorld, pow, sin, smoothstep, sqrt, step, texture, uniform, vec2, vec3 } from 'three/tsl';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { fractal, noise, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { aliasedCell, aliasedClaim, aliasedColour, aliasedTarget } from '@sim/envs/aliased';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount } from '../../format';

type N<T extends string> = THREE.Node<T>;

const TAU = Math.PI * 2;
const R = 10;
const HALF = 13.9;
const H = 16;
const HS = 0.92;
const WINDOW = 1000;
const QUICK = 14;
const PEARL = 0xcfc8b8;
const ONYX = 0x1a1c24;
const BRASS = 0xd2a656;
const GOLD = 0xc9a048;
const LIGHT = new THREE.Vector3(-0.45, -0.55, 0.7).normalize();
const HALFWAY = LIGHT.clone().add(new THREE.Vector3(0, 0, 1)).normalize();

class AliasedRingScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['cave', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'aliasedRing';
  override legend: LegendSpec | null = null;

  private cells = 1;
  private rb = 1;
  private hit = 1;
  private miss = -0.1;
  private cellData = new Float32Array(4);
  private cellTex = new THREE.DataTexture(this.cellData, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly histData = new Float32Array((H + 1) * 4);
  private readonly histTex = new THREE.DataTexture(this.histData, H + 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly slider = uniform(new THREE.Vector3(0, -R, 1));
  private readonly press = uniform(0);
  private readonly shift = uniform(0);
  private readonly flashCell = uniform(-1);
  private readonly flashAt = uniform(new THREE.Vector2(0, -R));
  private readonly goldAt = uniform(new THREE.Vector2(0, -R));
  private readonly flash = uniform(0);
  private readonly gold = uniform(0);
  private readonly dim = uniform(1);
  private filled = -1;
  private view = '';
  private wall = 0;
  private rate = 0;
  private lastPosition = NaN;
  private readonly point = new THREE.Vector3();
  private readonly ledger = { cell: 0, sees: 0, target: 0, act: -1, reward: 0, found: 0, claims: 0, ones: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.cells = parameter(ctx.environment, 'cells');
    this.hit = constant(ctx.environment, 'ALIASED_CLAIM_REWARD');
    this.miss = constant(ctx.environment, 'ALIASED_FALSE_CLAIM');
    this.rb = Math.min((0.36 * TAU * R) / this.cells, 1.45);
    this.cellData = new Float32Array(this.cells * 4);
    this.cellTex = new THREE.DataTexture(this.cellData, this.cells, 1, THREE.RGBAFormat, THREE.FloatType);
    return new THREE.Box2(new THREE.Vector2(-HALF, -HALF), new THREE.Vector2(HALF, HALF));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb8c4bc).multiplyScalar(0.5);
  }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    this.dim.value = on ? 0.35 : 1;
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 34, sighSize: 18, sighRadius: 0.3, sighHeight: 0.9 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(_ctx: SceneContext, { pool }: Stage2D): void {
    this.legend = this.legendFor();
    this.scene.background = new THREE.Color(0x07100e);
    pool.group.renderOrder = 10;
    for (const tex of [this.cellTex, this.histTex]) {
      tex.magFilter = tex.minFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
    }
    const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.shade();
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(400, 400).rotateX(-Math.PI / 2), material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
  }

  private shade(): N<'vec3'> {
    const p = vec2(positionWorld.x, positionWorld.z);
    const px = fwidth(positionWorld.x).max(1e-5);
    const fill = (d: N<'float'>): N<'float'> => clamp(d.div(px).negate().add(0.5), 0, 1);
    const soft = (d: N<'float'>, w: number): N<'float'> => float(1).sub(smoothstep(-w, w, d));
    const n = float(this.cells), rb = float(this.rb), radial = length(p);
    const L = vec3(LIGHT.x, LIGHT.y, LIGHT.z), Hv = vec3(HALFWAY.x, HALFWAY.y, HALFWAY.z);
    const sphere = (q: N<'vec2'>, r: N<'float'> | number, albedo: N<'vec3'>, sheen: number): N<'vec3'> => {
      const nq = q.div(r), nz = sqrt(max(float(1).sub(dot(nq, nq)), 0)), nrm = vec3(nq.x, nq.y, nz);
      const diff = clamp(dot(nrm, L), 0, 1), spec = pow(clamp(dot(nrm, Hv), 0, 1), 36);
      return albedo.mul(diff.mul(0.66).add(0.34)).add(vec3(0.6, 0.64, 0.72).mul(spec.mul(sheen)));
    };

    const weave = clamp(fractal(p.mul(0.55), 4).mul(0.7).add(noise(p.mul(9)).mul(0.3)), 0, 1);
    let col: N<'vec3'> = mix(rgb(0x122a24), rgb(0x1f4a3e), weave);
    col = col.mul(float(1).sub(smoothstep(8, 20, radial).mul(0.5)));
    col = mix(col, col.mul(0.6), soft(abs(radial.sub(R).sub(0.08)).sub(0.1), 0.08));
    col = mix(col, rgb(0x6f5a3c), fill(abs(radial.sub(R)).sub(0.05)));

    const a = fract(atan(p.x, p.y.negate()).div(TAU).add(1));
    const i = min(floor(a.mul(n)), n.sub(1)), ca = i.add(0.5).div(n).mul(TAU);
    const q = p.sub(vec2(sin(ca).mul(R), cos(ca).mul(-R))), dist = length(q);
    const cell = texture(this.cellTex, vec2(i.add(0.5).div(n), 0.5));
    const bit = cell.r, mark = cell.g;
    col = col.mul(float(1).sub(soft(length(q.sub(vec2(rb.mul(0.16), rb.mul(0.24)))).sub(rb), this.rb * 0.3).mul(0.6)));
    const here = float(1).sub(step(0.5, abs(i.sub(this.flashCell)))).mul(this.flash);
    const glow = mark.mul(this.gold);
    let bead: N<'vec3'> = mix(sphere(q, rb, rgb(ONYX), 0.55), sphere(q, rb, rgb(PEARL), 0.1), bit);
    bead = bead.add(mix(rgb(0x4a5670), rgb(0xb5a3d0), bit).mul(mix(float(0.5), float(0.16), bit).mul(float(1).sub(sqrt(max(float(1).sub(dot(q, q).div(rb.mul(rb))), 0))))));
    bead = mix(bead, sphere(q, rb, rgb(TOKENS.failureBlue).mul(0.8), 0.2), here.mul(0.7));
    bead = mix(bead, sphere(q, rb, rgb(GOLD), 0.2), glow.mul(0.65));
    const halo = (at: N<'vec2'>): N<'float'> => exp(max(length(p.sub(at)).sub(rb), 0).div(rb.mul(0.8)).negate());
    col = col.add(rgb(TOKENS.failureBlue).mul(halo(this.flashAt).mul(this.flash).mul(0.35))).add(rgb(GOLD).mul(halo(this.goldAt).mul(this.gold).mul(0.45)));
    col = mix(col, bead, fill(dist.sub(rb)));
    col = mix(col, rgb(GOLD), fill(abs(dist.sub(rb.mul(1.2))).sub(max(rb.mul(0.06), px))).mul(mark).mul(0.9));
    const inward = normalize(this.goldAt), tickR = R - this.rb * 1.35 - 0.12;
    col = mix(col, rgb(GOLD), fill(sdSegment(p, inward.mul(tickR), inward.mul(tickR - 0.3 - this.rb * 0.3), max(rb.mul(0.07), 0.05))).mul(0.9));

    const s = vec2(this.slider.x, this.slider.y), scale = this.slider.z, sq = p.sub(s);
    const sr = rb.mul(1.42).mul(scale).mul(float(1).sub(this.press.mul(0.14))), sw = rb.mul(0.17).mul(scale);
    const out = normalize(s), pinA = s.add(out.mul(sr.add(sw))), pinB = s.add(out.mul(sr.add(sw).add(scale.mul(0.3 + this.rb * 0.3))));
    const shape = (o: N<'vec2'>): N<'float'> => min(abs(length(sq.sub(o)).sub(sr)).sub(sw), sdSegment(p.sub(o), pinA, pinB, max(sw.mul(0.75), scale.mul(0.05))));
    const shown = smoothstep(0, 0.12, scale);
    col = col.mul(float(1).sub(soft(shape(vec2(rb.mul(0.14), rb.mul(0.2))), this.rb * 0.12).mul(0.55).mul(shown)));
    const face = dot(normalize(sq.add(vec2(1e-5, 0))), vec2(-0.6, -0.8)).mul(0.5).add(0.5);
    const brass = mix(rgb(0x6a4e20), rgb(BRASS), face).add(rgb(0x3a2c10).mul(this.press));
    col = mix(col, brass, fill(shape(vec2(0, 0))).mul(shown));

    const W = H * HS, hr = 0.29, plate = sdBox(p.sub(vec2(0, 0.24)), vec2(W / 2 + 0.35, 0.86), 0.45);
    col = mix(col, col.mul(0.55), fill(plate));
    col = mix(col, rgb(0x3d3322), fill(abs(plate).sub(0.02)));
    const ux = p.x.add(W / 2).div(HS).add(this.shift), j = floor(ux);
    const inside = step(0, j).mul(step(j, float(H))).mul(step(plate, 0));
    const h = texture(this.histTex, vec2(clamp(j, 0, H).add(0.5).div(H + 1), 0.5));
    const hq = vec2(p.x.sub(j.add(0.5).sub(this.shift).mul(HS).sub(W / 2)), p.y);
    const dot2 = mix(sphere(hq, hr, rgb(ONYX), 0.55), sphere(hq, hr, rgb(PEARL), 0.1), h.r);
    const live = inside.mul(h.b).mul(h.a);
    col = mix(col, dot2, fill(length(hq).sub(hr)).mul(live));
    const verdict = mix(rgb(GOLD), rgb(TOKENS.failureBlue), step(1.5, h.g));
    col = mix(col, verdict, fill(length(hq.sub(vec2(0, 0.6))).sub(0.1)).mul(live).mul(step(0.5, h.g)));
    const caret = vec2(p.x.sub((H - 0.5) * HS - W / 2), p.y.add(0.52));
    col = mix(col, rgb(BRASS), fill(sdSegment(caret, vec2(-0.16, 0), vec2(0.16, 0), 0.035)));
    return col.mul(this.dim);
  }

  protected override drop(): void {
    this.cellTex.dispose();
    this.histTex.dispose();
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

  private ring(cell: number, out: THREE.Vector3): THREE.Vector3 {
    const angle = (TAU * (cell + 0.5)) / this.cells;
    return out.set(R * Math.sin(angle), 0, -R * Math.cos(angle));
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.wall += frame.dtReal;
    if (frame.dtReal > 0 && Number.isFinite(this.lastPosition)) {
      const speed = Math.abs(frame.position - this.lastPosition) / frame.dtReal;
      this.rate += (speed - this.rate) * Math.min(1, frame.dtReal * 4);
    }
    this.lastPosition = frame.position;
    const k = frame.state.step, now = this.at(frame, k);
    if (!now) return;
    const next = this.at(frame, k + 1), f = next ? clamp01(frame.position - k) : 0, quick = this.rate > QUICK;
    if (k !== this.filled) this.fill(frame, k, now);
    this.strip(frame, k, next, f);
    this.pose(now, next, quick ? 0 : f);
    if (frame.scrubbed) pool.clear();
    else if (!quick) this.celebrate(frame, now, pool, time);
  }

  private fill(frame: FrameUpdate, k: number, now: EnvFrame): void {
    const cells = this.cells, data = this.cellData, target = aliasedTarget(now), key = now.view.join(',') + ':' + target;
    let ones = 0;
    for (let c = 0; c < cells; c++) {
      const bit = aliasedColour(now, c);
      ones += bit;
      data[c * 4] = bit;
      data[c * 4 + 1] = c === target ? 1 : 0;
    }
    if (key !== this.view) {
      this.view = key;
      this.cellTex.needsUpdate = true;
    }
    const cell = aliasedCell(now), from = Math.max(0, k - WINDOW), found = countEvents(frame.eventLog, 'goal', from, k);
    Object.assign(this.ledger, {
      cell, sees: now.observations[0]!, target, act: k > 0 ? aliasedClaim(now) : -1, reward: now.reward,
      found, claims: found + countEvents(frame.eventLog, 'harm', from, k), ones,
    });
    this.filled = k;
  }

  private strip(frame: FrameUpdate, k: number, next: EnvFrame | null, f: number): void {
    const data = this.histData;
    for (let j = 0; j <= H; j++) {
      const s = k - (H - 1) + j, e = j === H ? next : this.at(frame, s), o = j * 4;
      data[o] = e ? e.observations[0]! : 0;
      data[o + 1] = e && s > 0 ? (e.reward > 0 ? 1 : e.reward < 0 ? 2 : 0) : 0;
      data[o + 2] = e ? 1 : 0;
      data[o + 3] = j === 0 ? 1 - f : j === H ? f : 1;
    }
    this.shift.value = f;
    this.histTex.needsUpdate = true;
  }

  private pose(now: EnvFrame, next: EnvFrame | null, f: number): void {
    const cell = aliasedCell(now);
    let pos = cell, scale = 1, press = 0;
    if (next && f > 0) {
      if (aliasedClaim(next) === 0) pos = cell + smoothstep01(f);
      else if (next.reward > 0) {
        press = Math.sin(Math.PI * clamp01(f / 0.35));
        if (f < 0.62) scale = 1 - smoothstep01((f - 0.3) / 0.32);
        else {
          pos = aliasedCell(next);
          scale = smoothstep01((f - 0.62) / 0.38);
        }
      } else press = Math.sin(Math.PI * clamp01(f / 0.6));
    }
    this.ring(pos, this.point);
    this.slider.value.set(this.point.x, this.point.z, scale);
    this.press.value = press;
    this.ring(cell, this.point);
    this.flashAt.value.set(this.point.x, this.point.z);
    this.ring(aliasedTarget(now), this.point);
    this.goldAt.value.set(this.point.x, this.point.z);
    this.flashCell.value = now.reward < 0 ? cell : -1;
    this.flash.value = now.reward < 0 ? 1 - 0.8 * smoothstep01(f) : 0;
    this.gold.value = now.reward > 0 ? 1 - 0.7 * smoothstep01(f) : 0;
  }

  private celebrate(frame: FrameUpdate, now: EnvFrame, pool: Stage2D['pool'], time: number): void {
    for (const event of frame.events) {
      if (event.type === 'goal') {
        this.ring(aliasedTarget(now), this.point);
        pool.burst(this.point.setY(40), TOKENS.celebrationGold, 0.9, time);
        pool.ripple(this.point.setY(0.2), GOLD, 1.2 * this.rb, time);
      } else if (event.type === 'harm') {
        this.ring(aliasedCell(now), this.point);
        pool.ripple(this.point.setY(0.2), TOKENS.failureBlue, 0.8 * this.rb, time);
      }
    }
  }

  private legendFor(): LegendSpec {
    const l = this.ledger, colour = (bit: number) => (bit ? 'pearl · 1' : 'onyx · 0');
    return {
      title: 'ALIASED RING',
      rows: [
        { key: 'bead', label: 'agent on bead', tint: BRASS, value: () => `${l.cell + 1} of ${this.cells}` },
        { key: 'sees', label: 'sees', tint: PEARL, value: () => colour(l.sees) },
        { key: 'target', label: 'hidden mark', tint: GOLD, value: () => `bead ${l.target + 1}` },
        { key: 'act', label: 'last act', tint: BRASS, value: () => (l.act < 0 ? '—' : l.act ? 'claim' : 'walk') },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => (l.act < 0 ? '—' : l.reward > 0 ? `+${this.hit}` : l.reward < 0 ? `${this.miss.toFixed(1)}` : '0') },
        { key: 'found', label: `claims right, last ${formatCount(WINDOW)}`, tint: TOKENS.celebrationGold, value: () => (l.claims ? `${formatCount(l.found)} of ${formatCount(l.claims)}` : '—') },
      ],
      note: () => `${l.ones} pearl · ${this.cells - l.ones} onyx · the agent sees only the bead it is on`,
      bar: () => (l.claims ? l.found / l.claims : 0),
    };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const now = this.at(frame, frame.state.step);
    if (!now) return;
    const cells = this.cells, at = this.point, rb = this.rb;
    const cord = kit.curve('cord', 257, 0x8fa6c4);
    for (let c = 0; c <= 256; c++) cord.point(R * Math.sin((TAU * c) / 256), 0.05, -R * Math.cos((TAU * c) / 256));
    const colours = kit.marks('colours', cells);
    for (let c = 0; c < cells; c++) {
      this.ring(c, at);
      colours.set(at.x, 0.06, at.z, rb * 0.55, aliasedColour(now, c) ? 0xffffff : 0x4a5060);
    }
    this.ring(aliasedTarget(now), at);
    kit.marks('target', 1).set(at.x, 0.07, at.z, rb * 1.1, TOKENS.celebrationGold);
    this.ring(aliasedCell(now), at);
    kit.marks('agent', 1).set(at.x, 0.08, at.z, rb * 0.8, TOKENS.uiCyan);
  }
}

export default (() => new AliasedRingScene()) satisfies SceneFactory;
