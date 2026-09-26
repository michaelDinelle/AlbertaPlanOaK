import * as THREE from 'three/webgpu';
import { abs, dot, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, mod, select, smoothstep, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { G2048_CELLS, G2048_WIN, g2048Highest, g2048Tile } from '@sim/envs/g2048';
import type { EnvFrame, GridCell } from '@sim/types';
import { constantList } from '../../data/envs';
import { formatCount } from '../../format';

type N<T extends string> = THREE.Node<T>;

const MOVE_SECONDS = 0.11;
const LEAVE = 0.4;
const ARRIVE = 0.3;
const STAGGER = 0.25;
const POP = 0.45;
const NUDGE = 0.075;
const GHOST_SECONDS = 0.3;
const EDGE_SECONDS = 0.24;
const FLARE_SECONDS = 0.45;
const WAVE_SECONDS = 0.7;
const WALK = 2048;
const QUAD = 1.34;
const HALF = 0.435;
const ROUND = 0.085;
const TRAY = 2.2;
const RIM = 2.1;
const TABLE = 60;
const GLOW_FROM = 7;
const LABEL_PX = 256;
const FONT = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const RAMP = [0xf4ebe0, 0xf0dcc0, 0xf0a266, 0xee8753, 0xe96c4a, 0xd94d35, 0xf2cd66, 0xf0c252, 0xeeb641, 0xeca932, 0xe99b22] as const;
const OBSIDIAN = 0x2b2723;
const GRAPHITE = 0x6d6158;
const IVORY = 0xfffaf2;
const BRONZE = 0x5b3a12;
const GILT = 0xf6d27a;
const PRESSED = 0xffc978;
const BLOCKED = 0x7aa7ff;
const ARROWS: Record<string, string> = { '0,-1': '↑ up', '0,1': '↓ down', '-1,0': '← left', '1,0': 'right →' };

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const easeBack = (t: number) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2;

interface Look { body: THREE.Color; ink: THREE.Color; shine: THREE.Color; glow: number; emit: number }

function looks(highest: number): Look[] {
  const out: Look[] = [];
  for (let e = 1; e <= highest; e++) {
    const body = new THREE.Color(RAMP[e - 1] ?? OBSIDIAN), ink = new THREE.Color(e <= 2 ? GRAPHITE : e < GLOW_FROM ? IVORY : e <= RAMP.length ? BRONZE : GILT);
    const glow = e < GLOW_FROM ? 0 : 0.06 + 0.05 * (e - GLOW_FROM);
    out[e] = { body, ink, shine: e > RAMP.length ? ink : body, glow, emit: e < GLOW_FROM ? 0 : 0.55 + 0.15 * (e - GLOW_FROM) };
  }
  return out;
}

function labelAtlas(highest: number): { texture: THREE.DataTexture; columns: number; rows: number } {
  const columns = Math.ceil(Math.sqrt(highest)), rows = Math.ceil(highest / columns);
  const canvas = document.createElement('canvas');
  canvas.width = columns * LABEL_PX;
  canvas.height = rows * LABEL_PX;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('2048 labels need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  for (let e = 1; e <= highest; e++) {
    const text = String(2 ** e);
    context.font = `700 100px ${FONT}`;
    const probe = context.measureText(text);
    const size = 100 * Math.min(0.36 * LABEL_PX / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent), 0.7 * LABEL_PX / probe.width);
    context.font = `700 ${size}px ${FONT}`;
    const fit = context.measureText(text), i = e - 1;
    context.fillText(text, (i % columns + 0.5) * LABEL_PX, (Math.floor(i / columns) + 0.5) * LABEL_PX + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2);
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const texture = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return { texture, columns, rows };
}

const sameBoard = (a: EnvFrame, b: EnvFrame) => a.words.every((word, i) => word === b.words[i]);

class G2048Scene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'g2048';

  private grid: GridSize = { w: 4, h: 4 };
  private highest = G2048_CELLS + 1;
  private cellAt: GridCell[] = [];
  private push: GridCell[] = [];
  private rank: number[][] = [];
  private look: Look[] = [];
  private tiles: THREE.InstancedMesh | null = null;
  private lookAttr: THREE.InstancedBufferAttribute | null = null;
  private bodyAttr: THREE.InstancedBufferAttribute | null = null;
  private inkAttr: THREE.InstancedBufferAttribute | null = null;
  private lamps: { mesh: THREE.Mesh; tint: THREE.Color }[] = [];
  private edges: { warm: { value: number }; cool: { value: number }; glow: THREE.Mesh }[] = [];
  private readonly wave = { at: -Infinity, reach: 0, gain: 0, centre: uniform(new THREE.Vector2()), radius: uniform(0), power: uniform(0) };
  private textures: THREE.Texture[] = [];

  private readonly from = new Uint8Array(G2048_CELLS);
  private readonly to = new Uint8Array(G2048_CELLS);
  private readonly ghost = new Uint8Array(G2048_CELLS);
  private readonly flare = new Float32Array(G2048_CELLS);
  private ghostAt = -Infinity;
  private kind: 'rest' | 'move' | 'blocked' | 'reset' = 'rest';
  private action = 0;
  private progress = 1;
  private since = Infinity;
  private step = -1;
  private readonly game = { from: 0, to: -1, sum: 0, found: false };

  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private readonly ledger = { move: '—', reward: 0, best: 0, filled: 0 };

  override legend: LegendSpec = {
    title: '2048',
    rows: [
      { key: 'move', label: 'move', tint: 0xdcf4ff, value: () => this.ledger.move },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => this.ledger.reward > 0 ? `+${formatCount(this.ledger.reward)}` : formatCount(this.ledger.reward) },
      { key: 'score', label: 'score', tint: 0xedc46a, value: () => this.game.found ? formatCount(this.game.sum) : '—' },
      { key: 'best', label: 'best tile', tint: 0xf2d27e, value: () => this.ledger.best ? formatCount(2 ** this.ledger.best) : '—' },
    ],
    note: () => `${this.ledger.filled} of ${G2048_CELLS} cells`,
    bar: () => this.ledger.filled / G2048_CELLS,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const orient = constantList(ctx.environment, 'G2048_ORIENT'), cells = orient.length / 4, side = Math.sqrt(cells);
    if (cells !== G2048_CELLS || side * side !== cells) throw new Error(`${ctx.environment.envId} orients ${cells} cells`);
    this.grid = { w: side, h: side };
    this.cellAt = [];
    for (let j = 0; j < cells; j++) this.cellAt[orient[2 * cells + j]!] = { x: j % side, y: Math.floor(j / side) };
    for (let a = 0; a < 4; a++) {
      const lead = this.cellAt[orient[a * cells]!]!, tail = this.cellAt[orient[a * cells + side - 1]!]!;
      this.push[a] = { x: Math.sign(lead.x - tail.x), y: Math.sign(lead.y - tail.y) };
      this.rank[a] = [];
      for (let j = 0; j < cells; j++) this.rank[a]![orient[a * cells + j]!] = j % side;
    }
    return new THREE.Box2(new THREE.Vector2(-TRAY - 0.25, -TRAY - 0.25), new THREE.Vector2(TRAY + 0.25, TRAY + 0.75));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff1e0).multiplyScalar(0.8);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (frame, kit) => {
        const step = frame.state.step, now = frame.frameAt(step), pips = kit.marks('exponent', G2048_CELLS * this.highest);
        for (let c = 0; c < G2048_CELLS; c++) {
          const e = g2048Tile(now.words, c), cell = this.cellAt[c]!, centre = cellToWorldVec(cell.x, cell.y, this.grid, this.at);
          for (let k = 0; k < e; k++) pips.set(centre.x - 0.34 + (k % 6) * 0.1, 0.05, centre.z - 0.34 + Math.floor(k / 6) * 0.1, 0.035, 0x5ec1ff);
        }
        const moved = kit.curve('moved', 5, 0xffd166), blocked = kit.curve('blocked', 5, 0x7a95c4), before = step > 0 ? frame.frameAt(step - 1) : null;
        if (!before || before.step !== step - 1 || now.done) return;
        const dir = this.push[Math.trunc(now.actions[0]!) & 3]!, arrow = sameBoard(before, now) ? blocked : moved;
        const tip = { x: dir.x * 1.2, z: dir.y * 1.2 }, side = { x: -dir.y * 0.3, z: dir.x * 0.3 }, back = { x: dir.x * 0.9, z: dir.y * 0.9 };
        arrow.point(-tip.x, 0.1, -tip.z);
        arrow.point(tip.x, 0.1, tip.z);
        arrow.point(back.x + side.x, 0.1, back.z + side.z);
        arrow.point(tip.x, 0.1, tip.z);
        arrow.point(back.x - side.x, 0.1, back.z - side.z);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    this.look = looks(this.highest);
    this.scene.background = new THREE.Color(0x0c0907);

    const grain = fbm(ctx.seed, { size: 256, frequency: 4, octaves: 5 });
    this.textures.push(grain);
    const table = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const tp = uv().sub(0.5).mul(TABLE);
    const streak = texture(grain, vec2(tp.x.mul(0.02), tp.y.mul(0.5))).r;
    const fibre = texture(grain, vec2(tp.x.mul(0.08), tp.y.mul(2.4))).r;
    const rings = smoothstep(0.55, 1, fract(streak.mul(9)).sub(0.5).abs().mul(2).oneMinus());
    const wood = mix(rgb(0x1d130d), rgb(0x2f2016), smoothstep(0.3, 0.7, streak)).mul(fibre.mul(0.22).add(0.86)).mul(rings.mul(-0.1).add(1));
    table.colorNode = light.lit(wood.mul(smoothstep(2.6, 9, length(tp)).oneMinus().mul(0.7).add(0.3)));
    const tableMesh = new THREE.Mesh(flat(TABLE, TABLE), table);
    tableMesh.renderOrder = 0;

    const clay = fbm(ctx.seed ^ 0x2048, { size: 128, frequency: 8, octaves: 4 });
    this.textures.push(clay);
    const span = 2 * TRAY + 0.8, tray = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = uv().sub(0.5).mul(span), rim = sdBox(p, vec2(TRAY), 0.26), cover = coverage(rim);
    const drop = smoothstep(-0.08, 0.32, sdBox(p.sub(vec2(0.06, -0.12)), vec2(TRAY), 0.26)).oneMinus().mul(0.6);
    const q = p.sub(floor(p)).sub(0.5), inner = max(abs(p.x), abs(p.y)).lessThan(this.grid.w / 2);
    const well = select(inner, coverage(sdBox(q, vec2(HALF), ROUND)), float(0));
    const speck = texture(clay, p.mul(0.35)).r.mul(0.12).add(0.94);
    const bevel = smoothstep(-0.12, 0, rim).mul(p.y.div(TRAY).mul(0.14).sub(0.03)).add(1);
    const recess = smoothstep(HALF - 0.16, HALF, q.y).mul(-0.45).add(smoothstep(HALF - 0.1, HALF, q.y.negate()).mul(0.16)).add(1);
    const trayColor = mix(rgb(0x9a8b7a).mul(bevel), rgb(0x76685a).mul(recess), well).mul(speck);
    const trayAlpha = cover.add(cover.oneMinus().mul(drop));
    const crest = length(p.sub(this.wave.centre)).sub(this.wave.radius).div(0.14);
    const surge = rgb(0xffc766).mul(exp(crest.mul(crest).negate()).mul(this.wave.power));
    tray.colorNode = light.lit(trayColor).add(surge).mul(cover).div(max(trayAlpha, 1e-4));
    tray.opacityNode = trayAlpha;
    const trayMesh = new THREE.Mesh(flat(span, span), tray);
    trayMesh.renderOrder = 1;
    this.scene.add(tableMesh, trayMesh);

    for (let a = 0; a < 4; a++) {
      const dir = this.push[a]!, along = dir.x !== 0 ? 'z' : 'x', warm = uniform(0), cool = uniform(0);
      const tint = rgb(PRESSED).mul(warm).add(rgb(BLOCKED).mul(cool.mul(0.8)));
      const bar = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });
      const length = this.grid.w - 0.3, bp = uv().sub(0.5).mul(vec2(along === 'x' ? length + 0.2 : 0.3, along === 'x' ? 0.3 : length + 0.2));
      const half = along === 'x' ? vec2(length / 2, 0) : vec2(0, length / 2);
      bar.colorNode = tint.mul(1.25);
      bar.opacityNode = min(coverage(sdSegment(bp, half.negate(), half, 0.022)).add(exp(sdSegment(bp, half.negate(), half, 0).mul(-28)).mul(0.5)), 1);
      const strip = new THREE.Mesh(flat(along === 'x' ? length + 0.2 : 0.3, along === 'x' ? 0.3 : length + 0.2), bar);
      strip.position.set(dir.x * RIM, 0, dir.y * RIM);
      strip.renderOrder = 2;
      const glow = light.emitter(flat(along === 'x' ? length : 0.05, along === 'x' ? 0.05 : length), tint.mul(0.6));
      glow.position.copy(strip.position);
      glow.visible = false;
      this.edges[a] = { warm, cool, glow };
      this.scene.add(strip, glow);
    }

    const atlas = labelAtlas(this.highest);
    this.textures.push(atlas.texture);
    const count = 2 * G2048_CELLS;
    this.lookAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.bodyAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.inkAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const look = instancedDynamicBufferAttribute(this.lookAttr, 'vec4') as N<'vec4'>;
    const body = instancedDynamicBufferAttribute(this.bodyAttr, 'vec3') as N<'vec3'>;
    const ink = instancedDynamicBufferAttribute(this.inkAttr, 'vec3') as N<'vec3'>;
    const tile = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(QUAD), d = sdBox(t, vec2(HALF), ROUND), face = coverage(d);
    const shadow = smoothstep(-0.03, 0.08, sdBox(t.sub(vec2(0.02, -0.035)), vec2(HALF), ROUND)).oneMinus().mul(0.38);
    const glyph = vec2(mod(look.x, atlas.columns), floor(look.x.div(atlas.columns))).add(vec2(t.x.add(0.5), t.y.negate().add(0.5))).div(vec2(atlas.columns, atlas.rows));
    const letter = texture(atlas.texture, glyph).r.mul(face);
    const luma = vec3(0.2126, 0.7152, 0.0722), raised = dot(ink, luma).greaterThan(dot(body, luma));
    const emboss = select(raised, texture(atlas.texture, glyph.sub(vec2(0.018 / atlas.columns, 0.026 / atlas.rows))).r.mul(face).mul(0.3), float(0));
    const sheen = smoothstep(0, 0.75, length(t.sub(vec2(-0.22, 0.24)))).oneMinus().mul(0.07);
    const edge = smoothstep(-0.07, 0, d);
    const glaze = body.mul(edge.mul(t.y.div(HALF).mul(0.13).sub(0.05)).add(1).add(sheen));
    const surface = mix(glaze.mul(emboss.oneMinus()), ink, letter);
    const glowing = look.z.greaterThan(0);
    const shaded = select(glowing, surface.mul(look.z.add(0.95)), light.lit(surface));
    const faded = mix(shaded, vec3(dot(shaded, luma).mul(0.45)), look.w);
    const halo = select(glowing, exp(max(d, 0).mul(-18)).mul(look.z.mul(1.2).add(0.18)), float(0)).mul(look.w.oneMinus());
    const shade = select(glowing, float(0), shadow);
    const alpha = face.add(face.oneMinus().mul(min(shade.add(halo), 1)));
    const shine = select(dot(body, luma).lessThan(0.05), ink, body);
    tile.colorNode = faded.mul(face).add(shine.mul(1.6).mul(halo).mul(face.oneMinus())).div(max(alpha, 1e-4));
    tile.opacityNode = alpha.mul(look.y);
    const tiles = instancedMesh(flat(QUAD, QUAD), tile, count);
    tiles.renderOrder = 3;
    tiles.frustumCulled = false;
    this.tiles = tiles;
    this.scene.add(tiles);

    for (let c = 0; c < G2048_CELLS; c++) {
      const tint = new THREE.Color();
      const mesh = light.emitter(flat(0.82, 0.82), uniform(tint) as unknown as N<'vec3'>);
      mesh.visible = false;
      this.lamps.push({ mesh, tint });
      this.scene.add(mesh);
    }
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.tiles?.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.since += frame.dtReal;
    const step = frame.state.step;
    if (step !== this.step || frame.scrubbed) this.advance(frame, step, pool, time);
    this.tally(frame, step);
    this.progress = Math.min(1, this.progress + frame.dtReal / MOVE_SECONDS);
    const fade = Math.exp(-frame.dtReal / EDGE_SECONDS), ebb = Math.exp(-frame.dtReal / FLARE_SECONDS);
    for (let c = 0; c < G2048_CELLS; c++) this.flare[c]! *= ebb;
    for (const edge of this.edges) {
      edge.warm.value *= fade;
      edge.cool.value *= fade;
      edge.glow.visible = edge.warm.value + edge.cool.value > 0.02;
    }
    const wave = this.wave, age = (time - wave.at) / WAVE_SECONDS;
    wave.radius.value = 0.3 + wave.reach * (1 - (1 - clamp01(age)) ** 3);
    wave.power.value = age >= 0 && age < 1 ? wave.gain * (1 - age) ** 2 : 0;
    this.draw(time);
  }

  private advance(frame: FrameUpdate, step: number, pool: Stage2D['pool'], time: number): void {
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null, last = this.step;
    const known = before !== null && before.step === step - 1;
    const forward = !frame.scrubbed && known && step > last;
    const animate = forward && step === last + 1 && this.since >= MOVE_SECONDS;
    const reset = now.done !== 0, moved = known && !reset && !sameBoard(before, now);
    this.since = 0;
    this.step = step;
    if (frame.scrubbed) {
      this.ghostAt = this.wave.at = -Infinity;
      this.flare.fill(0);
    }
    this.action = Math.trunc(now.actions[0]!) & 3;
    this.kind = !animate ? 'rest' : reset ? 'reset' : moved ? 'move' : 'blocked';
    this.progress = animate ? 0 : 1;
    for (let c = 0; c < G2048_CELLS; c++) {
      const e = g2048Tile(now.words, c);
      if (e > this.highest) throw new Error(`A 2048 tile of 2^${e} cannot exist on ${G2048_CELLS} cells`);
      this.from[c] = animate && !reset ? g2048Tile(before!.words, c) : animate ? 0 : e;
      this.to[c] = e;
    }
    const best = g2048Highest(now.words);
    this.ledger.best = best;
    this.ledger.filled = this.to.reduce((n, e) => n + (e ? 1 : 0), 0);
    this.ledger.reward = frame.state.reward;
    this.ledger.move = known ? `${ARROWS[`${this.push[this.action]!.x},${this.push[this.action]!.y}`]}${moved || reset ? '' : ' · blocked'}` : '—';
    if (!forward) return;
    const edge = this.edges[this.action]!;
    if (moved || reset) edge.warm.value = 0.55 + 0.45 * Math.min(1, Math.log2(1 + now.reward) / G2048_WIN);
    else edge.cool.value = 1;
    const shown = frame.frameAt(last), ended = frame.events.some((event) => event.type === 'done');
    if (!ended && shown.step === last && best > g2048Highest(shown.words) && best >= GLOW_FROM) {
      this.surge(now, best, 1.6 + 0.25 * (best - GLOW_FROM), 1, time);
      pool.pulse.fire(0.15 + 0.03 * (best - GLOW_FROM));
    }
    for (const event of frame.events) {
      if (event.type === 'goal') {
        this.surge(now, best, 4.5, 2, time);
        pool.pulse.fire(0.6);
      }
      if (event.type === 'done') {
        for (let s = step; s > step - 16 && s > 0; s--) {
          const end = frame.frameAt(s), last = frame.frameAt(s - 1);
          if (end.step !== s || last.step !== s - 1 || !end.done) continue;
          for (let c = 0; c < G2048_CELLS; c++) this.ghost[c] = g2048Tile(last.words, c);
          this.ghostAt = time;
          break;
        }
      }
    }
  }

  private surge(frame: EnvFrame, e: number, reach: number, gain: number, time: number): void {
    const c = this.cellOf(frame, e), cell = this.cellAt[c]!, spot = cellToWorldVec(cell.x, cell.y, this.grid, this.at);
    this.flare[c] = gain;
    Object.assign(this.wave, { at: time, reach, gain });
    this.wave.centre.value.set(spot.x, -spot.z);
  }

  private cellOf(frame: EnvFrame, e: number): number {
    for (let c = 0; c < G2048_CELLS; c++) if (g2048Tile(frame.words, c) === e) return c;
    throw new Error(`No 2^${e} tile on the board`);
  }

  private tally(frame: FrameUpdate, step: number): void {
    const game = this.game;
    if (frame.scrubbed || step < game.to) Object.assign(game, { from: step, to: step, sum: 0, found: false });
    for (let s = game.to + 1; s <= step; s++) {
      const f = frame.frameAt(s);
      if (f.step !== s) { Object.assign(game, { from: step, to: step, sum: 0, found: false }); break; }
      if (f.done) Object.assign(game, { from: s, to: s, sum: 0, found: true });
      else { game.sum += f.reward; game.to = s; }
    }
    for (let budget = WALK; !game.found && budget > 0; budget--) {
      const f = frame.frameAt(game.from);
      if (f.step !== game.from) break;
      if (f.done || game.from === 0) game.found = true;
      else { game.sum += f.reward; game.from--; }
    }
  }

  private draw(time: number): void {
    const tiles = this.tiles, lookAttr = this.lookAttr, bodyAttr = this.bodyAttr, inkAttr = this.inkAttr;
    if (!tiles || !lookAttr || !bodyAttr || !inkAttr) return;
    const u = this.progress, dir = this.push[this.action]!, span = this.grid.w - 1;
    const bump = this.kind === 'move' ? Math.sin(Math.PI * u) : this.kind === 'blocked' ? Math.sin(Math.PI * Math.min(1, u * 1.7)) * 0.8 : 0;
    const ghostAge = (time - this.ghostAt) / GHOST_SECONDS;
    for (let c = 0; c < G2048_CELLS; c++) {
      const cell = this.cellAt[c]!;
      cellToWorldVec(cell.x, cell.y, this.grid, this.at);
      const g = this.ghost[c]!;
      if (ghostAge >= 0 && ghostAge < 1 && g) this.place(c, g, 0.95 - 0.4 * ghostAge, 0.7 * (1 - ghostAge) ** 1.5, 1);
      else this.place(c, 0, 0, 0, 0);

      const before = this.from[c]!, after = this.to[c]!;
      let e = after, size = 1, alpha = 1;
      if (before !== after) {
        const leave = smoothstep01(u / LEAVE), grow = clamp01((u - ARRIVE - (this.kind === 'move' ? STAGGER * this.rank[this.action]![c]! / span : 0)) / POP);
        if (grow <= 0) {
          e = before;
          size = 1 - 0.4 * leave;
          alpha = 1 - leave;
        } else size = before ? 0.6 + 0.4 * easeBack(grow) : easeBack(grow);
      }
      this.at.x += dir.x * NUDGE * bump;
      this.at.z += dir.y * NUDGE * bump;
      const flare = this.flare[c]!;
      this.place(G2048_CELLS + c, e, size, alpha, 0, flare);
      const lamp = this.lamps[c]!, glow = e ? this.look[e]!.emit + 0.6 * flare : 0;
      lamp.mesh.visible = glow > 0 && size > 0.05;
      if (lamp.mesh.visible) {
        lamp.mesh.position.copy(this.at);
        lamp.mesh.scale.setScalar(size);
        lamp.tint.copy(this.look[e]!.shine).multiplyScalar(glow * alpha);
      }
    }
    tiles.instanceMatrix.needsUpdate = true;
    lookAttr.needsUpdate = true;
    bodyAttr.needsUpdate = true;
    inkAttr.needsUpdate = true;
  }

  private place(i: number, e: number, size: number, alpha: number, ghost: number, flare = 0): void {
    const tiles = this.tiles!, look = e ? this.look[e]! : null;
    this.matrix.compose(this.at, this.quaternion, this.scale.setScalar(look && size > 0 ? size : 0));
    tiles.setMatrixAt(i, this.matrix);
    this.lookAttr!.setXYZW(i, e - 1, alpha, look?.glow ? look.glow + 0.15 * flare : 0, ghost);
    if (look) {
      this.bodyAttr!.setXYZ(i, look.body.r, look.body.g, look.body.b);
      this.inkAttr!.setXYZ(i, look.ink.r, look.ink.g, look.ink.b);
    }
  }
}

export default (() => new G2048Scene()) satisfies SceneFactory;
