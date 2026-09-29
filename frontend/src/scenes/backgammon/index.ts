import * as THREE from 'three/webgpu';
import { abs, atan, clamp, dot, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, mod, positionWorld, select, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01 } from '@kit/mix';
import { hash2f } from '@kit/rng';
import { coverage, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  BG_BAR, BG_DICE, BG_GAME_SLOT, BG_HIGH_SLOT, BG_LOW_SLOT, BG_MEMBER_SLOT, BG_MEN, BG_NEXT, BG_OFF, BG_PATH, BG_POINTS, BG_PROPOSAL, BG_PROPOSAL_SLOT, BG_PROPOSALS,
  BG_PROPOSALS_SLOT, BG_REPLY, BG_REPLY_DICE, BG_REPLY_HIGH_SLOT, BG_REPLY_LOW_SLOT, BG_STEPS, BG_TURN_SLOT, bgCounts, bgFrom, bgHops, bgMoveSlot, bgSlot, bgTo,
} from '@sim/envs/backgammon';
import { countEvents, lastEvent } from '@sim/events';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';

type N<T extends string> = THREE.Node<T>;

const PITCH = 1;
const LENGTH = 5;
const FULL = 5;
const BAR = 1.1;
const BAND = 1.5;
const RAIL = 0.5;
const LIP = 0.66;
const TRAY = 1.2;
const SPLIT = 0.16;
const HX = 6 * PITCH + BAR / 2;
const HZ = LENGTH + BAND / 2;
const TRAY_X = HX + RAIL + TRAY / 2;
const CASE_LEFT = -HX - RAIL;
const CASE_RIGHT = HX + 2 * RAIL + TRAY;
const CASE_HZ = HZ + LIP;
const TIP = 0.47;
const SLOPE = 1 / Math.hypot(1, TIP / LENGTH);
const RADIUS = 0.455;
const SLAB = 0.29;
const SPRITE = 1.5;
const GLYPH = 0.5;
const DIE = 0.7;
const DIE_SPRITE = 1.1;
const DICE_X = BAR / 2 + 3;
const COLS = 8;
const CELL_PX = 128;
const SIDE = BG_OFF + 1;
const THROWS = 2 * BG_STEPS;

const GLIDE = 11;
const FADE = 10;
const ROLL_SECONDS = 0.5;
const FINALE_SECONDS = 1.2;
const FINALE_STEPS = 15;
const MOVED_SECONDS = 0.9;
const FRESH_SECONDS = 1.4;

const TABLE_DARK = 0x0a0706;
const TABLE_LIGHT = 0x1d130d;
const WALNUT_DARK = 0x21130a;
const WALNUT_LIGHT = 0x4d2e19;
const FELT = 0x173b30;
const BONE = 0xd8c49b;
const CLARET = 0x7c2a24;
const BRASS = 0xc9a05a;
const AGENT_RIM = 0xf6cfa6;
const AGENT_INK = 0x2a150a;
const EBONY = 0x1e1e24;
const EBONY_RIM = 0x9aa0ad;
const IVORY = 0xf2eadb;
const GOLD = TOKENS.celebrationGold;
const OPPONENTS = ['random', 'pubeval', 'league'] as const;
const RESULTS = ['single', 'gammon', 'backgammon'] as const;

const SERIF = 'Palatino, "Palatino Linotype", "Book Antiqua", "Times New Roman", serif';
const SANS = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const columnX = (point: number): number => {
  const q = point <= 12 ? point : 25 - point;
  return q <= 6 ? HX - (q - 0.5) * PITCH : -BAR / 2 - (q - 6.5) * PITCH;
};

const spacing = (n: number): number => (n > FULL ? (LENGTH - PITCH) / (n - 1) : PITCH);

const rollText = (high: number, low: number): string => (high === low ? `double ${high}s` : `${high}·${low}`);

function slot(side: number, at: number, k: number, n: number, out: THREE.Vector3): THREE.Vector3 {
  if (at === BG_OFF) return out.set(TRAY_X, 0, side === 0 ? HZ - 0.22 - k * SLAB : -HZ + 0.22 + k * SLAB);
  const along = PITCH / 2 + k * spacing(n);
  if (at === BG_BAR) return out.set(0, 0, side === 0 ? -(BAND / 2 + along) : BAND / 2 + along);
  const inward = at > 12 ? 1 : -1;
  return out.set(columnX(at), 0, inward * (along - HZ));
}

interface Atlas { texture: THREE.DataTexture; rows: number }

function glyphAtlas(): Atlas {
  const labels: [string, string][] = [];
  for (let p = 1; p <= BG_POINTS; p++) labels.push([String(p), `700 100px ${SERIF}`]);
  for (let n = 1; n <= BG_MEN; n++) labels.push([String(n), `800 100px ${SANS}`]);
  const rows = Math.ceil(labels.length / COLS), canvas = document.createElement('canvas');
  canvas.width = COLS * CELL_PX;
  canvas.height = rows * CELL_PX;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Backgammon numerals need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  labels.forEach(([text, font], i) => {
    context.font = font;
    const probe = context.measureText(text);
    const size = 100 * Math.min(0.6 * CELL_PX / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent), 0.84 * CELL_PX / probe.width);
    context.font = font.replace('100px', `${size}px`);
    const fit = context.measureText(text);
    context.fillText(text, (i % COLS + 0.5) * CELL_PX, (Math.floor(i / COLS) + 0.5) * CELL_PX + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2);
  });
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const texture = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return { texture, rows };
}

const glyphAt = (atlas: Atlas, index: N<'float'>, local: N<'vec2'>): N<'float'> => {
  const inside = step(0, local.x).mul(step(local.x, 1)).mul(step(0, local.y)).mul(step(local.y, 1));
  const cell = vec2(mod(index, COLS), floor(index.div(COLS)));
  return texture(atlas.texture, cell.add(clamp(local, 0.004, 0.996)).div(vec2(COLS, atlas.rows))).r.mul(inside);
};

const composite = (color: N<'vec3'>, face: N<'float'>, under: N<'float'>, glow: N<'vec3'> = vec3(0)) => {
  const alpha = face.add(face.oneMinus().mul(under));
  return { color: color.mul(face).add(glow.mul(face.oneMinus())).div(max(alpha, 1e-4)), alpha };
};

interface Man { side: number; at: number; k: number; x: number; z: number; tx: number; tz: number; morph: number; lift: number; moved: number }
interface Ghost { x: number; z: number; alpha: number; aim: number }
interface Arrow { from: THREE.UniformNode<'vec2', THREE.Vector2>; to: THREE.UniformNode<'vec2', THREE.Vector2>; strength: THREE.UniformNode<'float', number>; aim: number }
interface Die { value: number; owner: number; x: number; angle: number }

class BackgammonScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dusk', 'fog'], blend: [[['night', 'fog'], 0.6]] } };
  readonly controlHz = 12;
  protected readonly stageName = 'backgammon';

  private accent = 0xc98a5e;
  private rival = 'opponent';
  private league = false;
  private readonly disposables: { dispose(): void }[] = [];

  private readonly men: Man[] = [];
  private readonly shown = new Int8Array(2 * SIDE);
  private pending: Int8Array | null = null;
  private finaleUntil = -Infinity;
  private finaleStep = -1;
  private position = -1;
  private pace = 1;
  private settled = false;
  private step = -1;
  private roll = '';
  private rollAt = -Infinity;
  private replyAt = -Infinity;
  private closing = false;

  private checkers: THREE.InstancedMesh | null = null;
  private lookAttr: THREE.InstancedBufferAttribute | null = null;
  private fxAttr: THREE.InstancedBufferAttribute | null = null;
  private diceMesh: THREE.InstancedMesh | null = null;
  private diceAttr: THREE.InstancedBufferAttribute | null = null;
  private diceLamp: { mesh: THREE.Mesh; tint: THREE.Color } | null = null;

  private readonly pointState = new THREE.DataTexture(new Float32Array(4 * (BG_BAR + 1)), BG_BAR + 1, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly pointTarget = new Float32Array(3 * (BG_BAR + 1));
  private readonly clock = uniform(0);
  private readonly arrows: Arrow[] = Array.from({ length: 2 * BG_STEPS }, () => ({ from: uniform(new THREE.Vector2()), to: uniform(new THREE.Vector2()), strength: uniform(0), aim: 0 }));

  private readonly ghosts: Ghost[] = Array.from({ length: BG_STEPS }, () => ({ x: 0, z: 0, alpha: 0, aim: 0 }));
  private readonly dice: Die[] = [];
  private diceFade = 1;
  private readonly order: Man[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly ledger = { proposal: '—', dice: '—', action: '—', reply: '—', pips: '—', last: '—', note: '', off: 0 };

  private legendFor(accent: number): LegendSpec {
    return {
    title: 'BACKGAMMON',
    rows: [
      { key: 'proposal', label: 'proposal', tint: accent, value: () => this.ledger.proposal },
      { key: 'dice', label: 'dice', tint: IVORY, value: () => this.ledger.dice },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
      { key: 'reply', label: 'reply', tint: EBONY_RIM, value: () => this.ledger.reply },
      { key: 'pips', label: 'pips', tint: BRASS, value: () => this.ledger.pips },
      { key: 'last', label: 'last game', tint: GOLD, value: () => this.ledger.last },
    ],
    note: () => this.ledger.note,
    bar: () => this.ledger.off / BG_MEN,
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor(this.accent);
    const opponent = parameter(ctx.environment, 'opponent');
    const name = OPPONENTS[opponent];
    if (name === undefined) throw new Error(`${ctx.environment.envId} names opponent ${opponent}, which backgammon does not seat`);
    this.rival = name;
    this.league = name === 'league';
    return new THREE.Box2(new THREE.Vector2(CASE_LEFT - 3, -CASE_HZ - 0.9), new THREE.Vector2(CASE_RIGHT + 0.4, CASE_HZ + 1.55));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff0dc).multiplyScalar(0.74);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighRadius: 4.5, sighHeight: 1.4 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const truth = new THREE.Vector3(), counts = new Int8Array(2 * BG_OFF);
    return {
      debug: (frame, kit) => {
        const words = frame.frameAt(frame.state.step).words, dice = words[BG_DICE]!, reply = words[BG_REPLY_DICE]!;
        const men = kit.marks('men', 2 * BG_MEN), pips = kit.marks('dice', 2 * 4 * 6);
        bgCounts(words, counts);
        for (let side = 0; side < 2; side++) {
          const tint = side ? 0xffd166 : TOKENS.uiCyan;
          for (let at = 1; at <= BG_OFF; at++) {
            const n = counts[bgSlot(side, at)]!;
            for (let k = 0; k < n; k++) {
              if (at === BG_OFF) slot(side, at, k, n, truth);
              else if (at === BG_BAR) truth.set(0, 0, (side ? 1 : -1) * (BAND / 2 + 0.25 + 0.3 * k));
              else truth.set(columnX(at), 0, (at > 12 ? 1 : -1) * (0.25 + 0.3 * k - HZ));
              men.set(truth.x + (side ? 0.18 : -0.18), 0.2, truth.z, 0.13, tint);
            }
          }
        }
        for (let side = 0; side < 2; side++) {
          const path = side ? words[BG_REPLY]! : words[BG_PATH]!, roll = side ? reply : dice;
          for (let i = 0; i < BG_STEPS; i++) {
            const arrow = kit.curve(`${side ? 'reply' : 'proposal'}${i}`, 2, side ? 0xffffff : TOKENS.uiCyan);
            if (i >= bgHops(path)) continue;
            slot(side, bgFrom(path, i, side), 0, 1, truth);
            arrow.point(truth.x, 0.2, truth.z);
            slot(side, bgTo(path, roll, i, side), 0, 1, truth);
            arrow.point(truth.x, 0.2, truth.z);
          }
          if (!roll) continue;
          const high = roll & 7, low = (roll >>> 3) & 7, count = high === low ? 4 : 2;
          for (let i = 0; i < count; i++) {
            const value = i === 0 || high === low ? high : low, cx = (side ? -DICE_X : DICE_X) + (i - (count - 1) / 2) * 1.02;
            for (let v = 0; v < value; v++) pips.set(cx - 0.25 + (v % 3) * 0.25, 0.2, (v < 3 ? -0.14 : 0.14), 0.07, side ? 0xffd166 : TOKENS.uiCyan);
          }
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    this.scene.background = new THREE.Color(TABLE_DARK);
    this.pointState.magFilter = this.pointState.minFilter = THREE.NearestFilter;
    this.disposables.push(this.pointState);
    const grain = fbm(ctx.seed ^ 0xbac6, { size: 256, frequency: 4, octaves: 5 }), atlas = glyphAtlas();
    this.disposables.push(grain, atlas.texture);
    const sample = (q: N<'vec2'>) => texture(grain, q).r;

    const table = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const tp = positionWorld.xz;
    const streak = sample(vec2(tp.x.mul(0.018), tp.y.mul(0.32))), fibre = sample(vec2(tp.x.mul(0.09), tp.y.mul(1.9)));
    table.colorNode = light.lit(mix(rgb(TABLE_DARK), rgb(TABLE_LIGHT), smoothstep(0.3, 0.75, streak)).mul(fibre.mul(0.3).add(0.8)));
    const tableMesh = new THREE.Mesh(flat(120, 120), table);
    tableMesh.renderOrder = 0;

    const boardMesh = new THREE.Mesh(flat(CASE_RIGHT - CASE_LEFT + 3, 2 * CASE_HZ + 3), this.boardMaterial(light, sample, atlas));
    boardMesh.position.x = (CASE_LEFT + CASE_RIGHT) / 2;
    boardMesh.renderOrder = 1;

    const arrow = new THREE.Mesh(flat(CASE_RIGHT - CASE_LEFT, 2 * CASE_HZ), this.arrowMaterial());
    arrow.position.x = (CASE_LEFT + CASE_RIGHT) / 2;
    arrow.renderOrder = 5;
    this.scene.add(tableMesh, boardMesh, arrow);
    this.disposables.push(tableMesh.geometry, table, boardMesh.geometry, boardMesh.material as THREE.Material, arrow.geometry, arrow.material as THREE.Material);

    const count = 2 * BG_MEN + this.ghosts.length;
    this.lookAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.fxAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const checkers = instancedMesh(flat(SPRITE, SPRITE), this.checkerMaterial(light, atlas), count);
    checkers.renderOrder = 3;
    checkers.frustumCulled = false;
    this.checkers = checkers;

    this.diceAttr = new THREE.InstancedBufferAttribute(new Float32Array(THROWS * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const dice = instancedMesh(flat(DIE_SPRITE, DIE_SPRITE), this.diceMaterial(light), THROWS);
    dice.renderOrder = 4;
    dice.frustumCulled = false;
    this.diceMesh = dice;
    this.scene.add(checkers, dice);
    for (const mesh of [checkers, dice]) this.disposables.push(mesh, mesh.geometry, mesh.material as THREE.Material);

    const diceTint = new THREE.Color();
    const diceLamp = light.emitter(flat(2.3, 0.1), uniform(diceTint) as unknown as N<'vec3'>);
    diceLamp.position.set(DICE_X, 0, 0);
    this.diceLamp = { mesh: diceLamp, tint: diceTint };
    this.scene.add(diceLamp);
    this.disposables.push(diceLamp.geometry, diceLamp.material as THREE.Material);

    for (let side = 0; side < 2; side++) for (let i = 0; i < BG_MEN; i++) this.men.push({ side, at: 0, k: 0, x: 0, z: 0, tx: 0, tz: 0, morph: 0, lift: 0, moved: -Infinity });
  }

  private boardMaterial(light: Stage2D['light'], sample: (q: N<'vec2'>) => N<'float'>, atlas: Atlas): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = positionWorld.xz, x = p.x, z = p.y;
    const ax = abs(x).sub(BAR / 2), az = abs(z);
    const col = clamp(floor(ax), 0, 5), u = ax.sub(col).sub(0.5), v = float(HZ).sub(az);
    const top = z.lessThan(0), qb = select(x.greaterThan(0), float(6).sub(col), col.add(7)), point = select(top, float(25).sub(qb), qb);
    const info = texture(this.pointState, vec2(point.add(0.5).div(BG_BAR + 1), 0.5));

    const vertical = abs(x).lessThan(BAR / 2).or(abs(x).greaterThan(HX));
    const along = select(vertical, vec2(z, x), vec2(x, z));
    const streak = sample(along.mul(vec2(0.034, 0.62))), fibre = sample(along.mul(vec2(0.21, 3.6)));
    const walnut = mix(rgb(WALNUT_DARK), rgb(WALNUT_LIGHT), smoothstep(0.22, 0.8, streak)).mul(fibre.mul(0.3).add(0.82));

    const dField = min(sdBox(p.sub(vec2(-BAR / 2 - 3, 0)), vec2(3, HZ), 0.03), sdBox(p.sub(vec2(BAR / 2 + 3, 0)), vec2(3, HZ), 0.03));
    const trayHalf = vec2(TRAY / 2, (HZ - SPLIT) / 2);
    const dTray = min(sdBox(p.sub(vec2(TRAY_X, -(HZ + SPLIT) / 2)), trayHalf, 0.1), sdBox(p.sub(vec2(TRAY_X, (HZ + SPLIT) / 2)), trayHalf, 0.1));
    const dCase = sdBox(p.sub(vec2((CASE_LEFT + CASE_RIGHT) / 2, 0)), vec2((CASE_RIGHT - CASE_LEFT) / 2, CASE_HZ), 0.3);

    const nap = sample(p.mul(0.7)).mul(0.16).add(sample(p.mul(5.1)).mul(0.1)).add(0.8);
    const felt = rgb(FELT).mul(nap);
    const dTri = abs(u).sub(float(TIP).mul(v.div(LENGTH).oneMinus())).mul(SLOPE);
    const tri = coverage(dTri);
    const tone = select(mod(point, 2).greaterThan(0.5), rgb(BONE), rgb(CLARET));
    const lengthwise = v.div(LENGTH).mul(-0.16).add(1.06);
    const inlay = smoothstep(0.045, 0.02, abs(dTri.add(0.06))).mul(0.22);
    const pointColor = tone.mul(lengthwise).mul(nap.mul(0.35).add(0.7)).mul(inlay.oneMinus()).add(rgb(BRASS).mul(inlay));
    const touched = info.x.mul(step(0, v));
    const recess = smoothstep(-0.32, 0, dField).mul(0.4);
    const field = mix(felt, pointColor, tri).mul(touched.mul(tri).mul(0.26).add(1)).mul(recess.oneMinus());

    const fieldCover = coverage(dField), trayCover = coverage(dTray);
    const bevel = smoothstep(-0.16, 0, dCase).mul(z.div(CASE_HZ).mul(-0.5).add(0.15));
    const hinge = coverage(abs(x).sub(0.022)).mul(fieldCover.oneMinus()).mul(0.55);
    const casing = walnut.mul(bevel.add(1)).mul(hinge.oneMinus());
    const trayFloor = felt.mul(0.55).mul(smoothstep(-0.28, 0, dTray).mul(-0.45).add(1));
    const brassField = coverage(abs(dField.sub(0.1)).sub(0.022)), brassTray = coverage(abs(dTray.sub(0.09)).sub(0.02));
    let albedo: N<'vec3'> = mix(mix(casing, field, fieldCover), trayFloor, trayCover);
    albedo = mix(albedo, rgb(BRASS), max(brassField, brassTray).mul(0.85));

    const cz = select(top, float(-(HZ + LIP / 2)), float(HZ + LIP / 2));
    const local = vec2(select(x.greaterThan(0), u, u.negate()).div(GLYPH).add(0.5), z.sub(cz).div(GLYPH).add(0.5));
    const rail = step(0, ax).mul(step(ax, 6)).mul(step(HZ, az));
    const ink = glyphAt(atlas, point.sub(1), local).mul(rail);
    albedo = mix(albedo, rgb(0xe3c48a), ink.mul(0.85));

    const edge = exp(abs(dTri).mul(-38)).mul(touched).mul(tri.add(0.35)).mul(fieldCover).mul(0.55);
    const barInfo = texture(this.pointState, vec2((BG_BAR + 0.5) / (BG_BAR + 1), 0.5));
    const barRim = exp(abs(sdBox(p.sub(vec2(0, -(BAND + LENGTH) / 2)), vec2(BAR / 2 - 0.12, LENGTH / 2), 0.25)).mul(-26));
    const rivalRim = exp(abs(sdBox(p.sub(vec2(0, (BAND + LENGTH) / 2)), vec2(BAR / 2 - 0.12, LENGTH / 2), 0.25)).mul(-26));
    const landing = rgb(GOLD).mul(ink.mul(info.y).mul(2.2).add(barRim.mul(barInfo.y).mul(0.9)));
    const answered = rgb(IVORY).mul(ink.mul(info.z).mul(1.1).add(rivalRim.mul(barInfo.z).mul(0.6)));
    const glow = landing.add(answered).add(rgb(this.accent).mul(edge));
    const shadow = smoothstep(0.9, -0.2, sdBox(p.sub(vec2((CASE_LEFT + CASE_RIGHT) / 2 + 0.12, 0.22)), vec2((CASE_RIGHT - CASE_LEFT) / 2, CASE_HZ), 0.3)).mul(0.7);
    const result = composite(light.lit(albedo).add(glow), coverage(dCase), shadow);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private checkerMaterial(light: Stage2D['light'], atlas: Atlas): THREE.MeshBasicNodeMaterial {
    const look = instancedDynamicBufferAttribute(this.lookAttr!, 'vec4') as N<'vec4'>;
    const fx = instancedDynamicBufferAttribute(this.fxAttr!, 'vec4') as N<'vec4'>;
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(SPRITE), morph = look.z, lift = look.w;
    const shape = (q: N<'vec2'>) => mix(length(q).sub(RADIUS), sdBox(q, vec2(RADIUS, SLAB * 0.42), 0.06), morph);
    const d = shape(t), face = coverage(d);
    const drop = vec2(0.045, -0.075).mul(lift.mul(2.4).add(1));
    const shadow = smoothstep(lift.mul(0.24).add(0.14), float(-0.06), shape(t.sub(drop))).mul(0.58);
    const opponent = look.x.greaterThan(0.5);
    const body = select(opponent, rgb(EBONY), rgb(this.accent)), rim = select(opponent, rgb(EBONY_RIM), rgb(AGENT_RIM));
    const r = length(t).div(RADIUS), facing = dot(t.div(max(length(t), 1e-4)), vec2(-0.6, 0.8));
    const bevel = smoothstep(0.78, 0.9, r), groove = exp(abs(r.sub(0.64)).mul(-42));
    const disc = mix(body.mul(r.mul(-0.14).add(1.06)), mix(body.mul(0.72), rim, facing.mul(0.5).add(0.5)), bevel.mul(0.85)).mul(groove.mul(-0.45).add(1));
    const edgeLine = exp(abs(t.y.sub(SLAB * 0.42 - 0.035)).mul(-45));
    const slab = mix(body.mul(0.78), body.mul(1.08), smoothstep(-0.12, 0.1, t.y)).add(rim.mul(edgeLine).mul(0.75));
    const glint = exp(length(t.sub(vec2(-0.15, 0.17))).mul(-15)).mul(0.26).mul(morph.oneMinus());
    const label = glyphAt(atlas, look.y, vec2(t.x.div(0.58).add(0.5), t.y.negate().div(0.58).add(0.5))).mul(step(0, look.y)).mul(morph.oneMinus());
    const albedo = mix(mix(disc, slab, morph).add(vec3(glint)), select(opponent, rgb(IVORY), rgb(AGENT_INK)), label);
    const halo = exp(max(d, 0).mul(-12)).mul(fx.z);
    const solid = composite(light.lit(albedo).add(rgb(GOLD).mul(fx.z.mul(0.35))), face, max(shadow, halo), rgb(GOLD).mul(halo.mul(1.4)));
    const ghost = fx.x.greaterThan(0.5);
    const dash = step(0.45, fract(atan(t.y, t.x).mul(14 / (2 * Math.PI))));
    const ring = coverage(abs(d.add(0.04)).sub(0.03)).mul(dash).add(exp(abs(d.add(0.04)).mul(-20)).mul(0.25));
    material.colorNode = select(ghost, rgb(this.accent).mul(1.9), solid.color);
    material.opacityNode = select(ghost, min(ring, 1), solid.alpha).mul(fx.y);
    return material;
  }

  private diceMaterial(light: Stage2D['light']): THREE.MeshBasicNodeMaterial {
    const die = instancedDynamicBufferAttribute(this.diceAttr!, 'vec4') as N<'vec4'>;
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(DIE_SPRITE), half = vec2(DIE / 2);
    const d = sdBox(t, half, 0.12), face = coverage(d);
    const shadow = smoothstep(0.2, -0.05, sdBox(t.sub(vec2(0.05, -0.08)), half, 0.12)).mul(0.55);
    const value = die.x, c = DIE * 0.27;
    const pip = (px: number, py: number, on: N<'bool'>) => select(on, length(t.sub(vec2(px, py))).sub(DIE * 0.095), float(9));
    const odd = mod(value, 2).greaterThan(0.5), two = value.greaterThan(1.5), four = value.greaterThan(3.5), six = value.greaterThan(5.5);
    const dp = min(min(min(pip(0, 0, odd), pip(-c, c, two)), min(pip(c, -c, two), pip(c, c, four))), min(min(pip(-c, -c, four), pip(-c, 0, six)), pip(c, 0, six)));
    const pips = coverage(dp);
    const opponent = die.y.greaterThan(0.5);
    const body = select(opponent, rgb(EBONY), rgb(0xe2ad7e)), ink = select(opponent, rgb(IVORY), rgb(AGENT_INK));
    const facing = dot(t.div(max(length(t), 1e-4)), vec2(-0.6, 0.8));
    const bevel = smoothstep(-0.1, 0, d).mul(facing.mul(0.3)).add(1);
    const well = smoothstep(0.02, -0.02, dp.sub(0.012)).mul(smoothstep(-0.03, 0, dp)).mul(0.25);
    const albedo = mix(body.mul(bevel).mul(well.oneMinus()), ink, pips);
    const gray = vec3(dot(albedo, vec3(0.2126, 0.7152, 0.0722)));
    const used = die.z;
    const shade = mix(albedo, gray.mul(0.55).add(0.09), used.mul(0.85));
    const halo = exp(max(d, 0).mul(-9)).mul(die.w);
    const tint = select(opponent, rgb(IVORY), rgb(this.accent));
    const result = composite(light.lit(shade), face, max(shadow, halo), tint.mul(halo.mul(1.3)));
    material.colorNode = result.color;
    material.opacityNode = result.alpha.mul(used.mul(-0.3).add(1));
    return material;
  }

  private arrowMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = positionWorld.xz;
    let agent: N<'float'> = float(0), rival: N<'float'> = float(0);
    this.arrows.forEach(({ from, to, strength }, i) => {
      const span = to.sub(from), reach = max(length(span), 1e-3), dir = span.div(reach), side = vec2(dir.y.negate(), dir.x);
      const along = dot(p.sub(from), dir);
      const dash = step(0.4, fract(along.div(0.32).sub(this.clock.mul(1.4))));
      const tip = to.sub(dir.mul(RADIUS + 0.08));
      const shaft = coverage(sdSegment(p, from.add(dir.mul(0.28)), tip.sub(dir.mul(0.1)), 0.034)).mul(dash);
      const head = coverage(min(sdSegment(p, tip, tip.sub(dir.mul(0.36)).add(side.mul(0.21)), 0.042), sdSegment(p, tip, tip.sub(dir.mul(0.36)).sub(side.mul(0.21)), 0.042)));
      if (i < BG_STEPS) agent = max(agent, max(shaft, head).mul(strength));
      else rival = max(rival, max(max(shaft, head), coverage(abs(length(p.sub(from)).sub(0.24)).sub(0.032))).mul(strength));
    });
    material.colorNode = rgb(this.accent).mul(1.9).mul(agent).add(rgb(IVORY).mul(1.5).mul(rival)).div(max(agent.add(rival), 1e-4));
    material.opacityNode = max(agent, rival);
    return material;
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.clock.value = time;
    const step = frame.state.step, advanced = frame.position - this.position;
    this.pace = !frame.scrubbed && frame.dtReal > 0 && advanced > 0 ? Math.min(20, Math.max(1, advanced / (frame.dtReal * frame.controlHz))) : 1;
    this.position = frame.position;
    if (step !== this.step) this.advance(frame, step, pool, time);
    if (this.pending && (time >= this.finaleUntil || step - this.finaleStep > FINALE_STEPS)) {
      this.apply(this.pending, false, time, false);
      this.pending = null;
    }
    this.draw(frame.dtReal * this.pace, time);
  }

  private countsAt(frame: FrameUpdate, step: number, out: Int8Array): Int8Array {
    const pose = step === frame.state.step ? frame.state.pose : frame.poseAt(step);
    for (let side = 0; side < 2; side++) for (let at = 1; at <= BG_OFF; at++) out[side * SIDE + at] = pose[bgSlot(side, at)]!;
    return out;
  }

  private advance(frame: FrameUpdate, step: number, pool: Stage2D['pool'], time: number): void {
    const pose = frame.state.pose;
    const sequential = !frame.scrubbed && this.step >= 0 && step > this.step, ended = lastEvent(frame.eventLog, 'done', step);
    const renewed = !sequential || (ended !== undefined && ended > this.step) || this.ledger.note === '';
    const counts = this.countsAt(frame, step, new Int8Array(2 * SIDE));
    const game = pose[BG_GAME_SLOT]!, turn = pose[BG_TURN_SLOT]!, roll = `${game}:${turn}`, rolled = roll !== this.roll;
    if (sequential) {
      if (ended !== undefined && ended > this.step) this.finale(frame, ended, pool, time);
      else if (!this.pending && rolled) this.sparks(counts, pool, time);
      if (this.pending) this.pending = counts;
      else this.apply(counts, false, time);
    } else {
      this.pending = null;
      this.finaleUntil = -Infinity;
      this.settled = false;
      this.apply(counts, true, time);
    }
    if (rolled) {
      this.rollAt = sequential ? time : -Infinity;
      this.replyAt = sequential ? time : -Infinity;
    }
    this.roll = roll;

    this.dice.length = 0;
    this.cast(pose[BG_HIGH_SLOT]!, pose[BG_LOW_SLOT]!, 0, game, turn);
    this.cast(pose[BG_REPLY_HIGH_SLOT]!, pose[BG_REPLY_LOW_SLOT]!, 1, game, turn);
    this.closing = pose[BG_PROPOSAL_SLOT]! + 1 >= pose[BG_PROPOSALS_SLOT]!;

    this.pointTarget.fill(0);
    for (let side = 0; side < 2; side++) for (let i = 0; i < BG_STEPS; i++) {
      const from = pose[bgMoveSlot(side, i)]!, to = pose[bgMoveSlot(side, i) + 1]!, arrow = this.arrows[side * BG_STEPS + i]!;
      arrow.aim = from < 0 ? 0 : 1;
      if (!side) this.ghosts[i]!.aim = arrow.aim;
      if (from < 0) continue;
      let k = counts[side * SIDE + from]!;
      for (let j = 0; j < i; j++) k += pose[bgMoveSlot(side, j)] === from ? 1 : 0;
      slot(side, from, k, k + 1, this.scratch);
      arrow.from.value.set(this.scratch.x, this.scratch.z);
      if (!side) {
        this.ghosts[i]!.x = this.scratch.x;
        this.ghosts[i]!.z = this.scratch.z;
      }
      const n = counts[side * SIDE + to]!;
      slot(side, to, Math.max(0, n - 1), n, this.scratch);
      arrow.to.value.set(this.scratch.x, this.scratch.z);
      if (from < BG_BAR && !side) this.pointTarget[3 * from] = 1;
      if (to <= BG_POINTS) this.pointTarget[3 * to + side + 1] = 1;
      if (to <= BG_POINTS && !side) this.pointTarget[3 * to] = 1;
      if (from === BG_BAR) this.pointTarget[3 * BG_BAR + side + 1] = 1;
    }

    this.ledgerFor(frame, step, counts, ended, renewed);
    this.step = step;
  }

  private cast(high: number, low: number, owner: number, game: number, turn: number): void {
    if (!high) return;
    const count = high === low ? 4 : 2;
    for (let i = 0; i < count; i++) this.dice.push({
      value: i === 0 || high === low ? high : low,
      owner,
      x: (owner ? -DICE_X : DICE_X) + (i - (count - 1) / 2) * 1.02,
      angle: (hash2f(game * 131 + turn, i + BG_STEPS * owner, 0x5eed) - 0.5) * 0.7,
    });
  }

  private finale(frame: FrameUpdate, ended: number, pool: Stage2D['pool'], time: number): void {
    const before = frame.frameAt(ended - 1), end = frame.frameAt(ended);
    const won = end.reward > 0, points = Math.abs(end.reward), winner = won ? 0 : 1;
    const tray = slot(winner, BG_OFF, BG_MEN - 1, BG_MEN, new THREE.Vector3());
    if (won) {
      pool.burst(tray, GOLD, 0.55 + 0.2 * points, time);
      pool.ripple(tray, GOLD, 0.4 + 0.15 * points, time);
      pool.pulse.fire(0.5 + 0.15 * points);
    } else {
      pool.burst(tray, IVORY, 0.3 + 0.1 * points, time);
      pool.sigh(0.4 + 0.15 * points, new THREE.Vector3(0, 0, 0), time);
    }
    if (before.step !== ended - 1) return;
    const final = this.countsAt(frame, ended - 1, new Int8Array(2 * SIDE));
    let aboard = 0;
    for (let at = 1; at <= BG_BAR; at++) {
      aboard += final[winner * SIDE + at]!;
      final[winner * SIDE + at] = 0;
    }
    final[winner * SIDE + BG_OFF] = final[winner * SIDE + BG_OFF]! + aboard;
    this.apply(final, false, time);
    this.finaleUntil = time + FINALE_SECONDS;
    this.finaleStep = ended;
    this.pending = final;
  }

  private sparks(counts: Int8Array, pool: Stage2D['pool'], time: number): void {
    for (let side = 0; side < 2; side++) {
      const base = side * SIDE;
      if (counts[base + BG_BAR]! > this.shown[base + BG_BAR]!) for (let at = 1; at <= BG_POINTS; at++) {
        if (counts[base + at]! >= this.shown[base + at]!) continue;
        slot(side, at, 0, 1, this.scratch);
        pool.burst(this.scratch, side ? this.accent : TOKENS.failureBlue, 0.32, time);
        pool.ripple(this.scratch, side ? this.accent : TOKENS.failureBlue, 0.16, time);
      }
      if (counts[base + BG_OFF]! > this.shown[base + BG_OFF]!) {
        slot(side, BG_OFF, counts[base + BG_OFF]! - 1, counts[base + BG_OFF]!, this.scratch);
        pool.burst(this.scratch, side ? IVORY : this.accent, 0.14, time);
      }
    }
  }

  private apply(counts: Int8Array, snap: boolean, time: number, marked = true): void {
    for (let side = 0; side < 2; side++) {
      const byAt: Man[][] = Array.from({ length: SIDE }, () => []), free: Man[] = [];
      for (const man of this.men) if (man.side === side) byAt[man.at]!.push(man);
      for (let at = 0; at < SIDE; at++) {
        const list = byAt[at]!.sort((a, b) => a.k - b.k), n = at ? counts[side * SIDE + at]! : 0;
        while (list.length > n) free.push(list.pop()!);
      }
      for (let at = 1; at < SIDE; at++) {
        const list = byAt[at]!, n = counts[side * SIDE + at]!;
        while (list.length < n) {
          const man = free.shift();
          if (!man) throw new Error(`Backgammon side ${side} shows more than ${BG_MEN} checkers`);
          man.at = at;
          man.moved = snap || !marked ? -Infinity : time;
          list.push(man);
        }
        list.forEach((man, k) => {
          man.k = k;
          slot(side, at, k, n, this.scratch);
          man.tx = this.scratch.x;
          man.tz = this.scratch.z;
          if (!snap) return;
          man.x = man.tx;
          man.z = man.tz;
          man.morph = at === BG_OFF ? 1 : 0;
        });
      }
      if (free.length) throw new Error(`Backgammon side ${side} shows ${BG_MEN - free.length} of its ${BG_MEN} checkers`);
    }
    this.shown.set(counts);
  }

  private ledgerFor(frame: FrameUpdate, step: number, counts: Int8Array, ended: number | undefined, renewed: boolean): void {
    const pose = frame.state.pose, index = pose[BG_PROPOSAL_SLOT]!, total = pose[BG_PROPOSALS_SLOT]!, replyHigh = pose[BG_REPLY_HIGH_SLOT]!;
    const member = pose[BG_MEMBER_SLOT]!, rival = this.league ? `league ${member + 1}` : this.rival;
    this.ledger.proposal = `${formatCount(index + 1)} of ${formatCount(total)}${this.closing ? ' · next plays' : ''}`;
    this.ledger.dice = rollText(pose[BG_HIGH_SLOT]!, pose[BG_LOW_SLOT]!);
    this.ledger.reply = replyHigh ? `${rival} ${rollText(replyHigh, pose[BG_REPLY_LOW_SLOT]!)}${pose[bgMoveSlot(1, 0)]! < 0 ? ' · passed' : ''}` : '—';
    const before = step > 0 ? frame.frameAt(step - 1) : null, now = frame.frameAt(step);
    if (!before || before.step !== step - 1) this.ledger.action = 'waiting';
    else this.ledger.action = Math.trunc(now.actions[0]!) === BG_NEXT && before.words[BG_PROPOSAL]! + 1 < before.words[BG_PROPOSALS]! ? 'next' : 'play';
    let mine = 0, theirs = 0;
    for (let at = 1; at <= BG_POINTS; at++) {
      mine += at * counts[at]!;
      theirs += (BG_BAR - at) * counts[SIDE + at]!;
    }
    this.ledger.pips = `${mine + BG_BAR * counts[BG_BAR]!} vs ${theirs + BG_BAR * counts[SIDE + BG_BAR]!}`;
    this.ledger.off = counts[BG_OFF]!;
    if (ended === undefined) this.ledger.last = '—';
    else {
      const end = frame.frameAt(ended), won = lastEvent(frame.eventLog, 'goal', ended) === ended;
      const points = end.step === ended ? Math.abs(end.reward) : 0;
      this.ledger.last = points ? `${won ? '+' : '−'}${points} ${RESULTS[points - 1]}` : won ? 'won' : 'lost';
    }
    if (renewed) {
      const wins = countEvents(frame.eventLog, 'goal', -1, step), losses = countEvents(frame.eventLog, 'harm', -1, step);
      this.ledger.note = `game ${formatCount(pose[BG_GAME_SLOT]! + 1)} · won ${formatCount(wins)} · lost ${formatCount(losses)}`;
    }
  }

  private draw(dt: number, time: number): void {
    const checkers = this.checkers, look = this.lookAttr, fx = this.fxAttr;
    if (!checkers || !look || !fx) return;
    const snap = !this.settled, glide = 1 - Math.exp(-GLIDE * dt), fade = snap ? 1 : 1 - Math.exp(-FADE * dt);
    this.settled = true;
    for (const man of this.men) {
      man.x += (man.tx - man.x) * glide;
      man.z += (man.tz - man.z) * glide;
      man.morph += ((man.at === BG_OFF ? 1 : 0) - man.morph) * fade;
      man.lift = clamp01(Math.hypot(man.tx - man.x, man.tz - man.z) / 1.2);
    }
    this.order.length = 0;
    this.order.push(...this.men);
    this.order.sort((a, b) => (a.lift > 0.04 ? 100 : 0) + a.k - ((b.lift > 0.04 ? 100 : 0) + b.k));
    const shown = this.shown;
    let i = 0;
    for (const man of this.order) {
      const n = shown[man.side * SIDE + man.at]!, labelled = man.at <= BG_POINTS && n > FULL && man.k === n - 1;
      this.matrix.compose(this.at.set(man.x, 0, man.z), this.quaternion.identity(), this.scale.setScalar(1 + 0.14 * man.lift));
      checkers.setMatrixAt(i, this.matrix);
      look.setXYZW(i, man.side, labelled ? BG_POINTS + n - 1 : -1, man.morph, man.lift);
      const age = time - man.moved;
      fx.setXYZW(i, 0, 1, age >= 0 && age < MOVED_SECONDS ? 0.55 * (1 - age / MOVED_SECONDS) ** 2 : 0, 0);
      i++;
    }
    const finale = this.pending !== null;
    for (const ghost of this.ghosts) {
      ghost.alpha += ((finale ? 0 : ghost.aim) - ghost.alpha) * fade;
      this.matrix.compose(this.at.set(ghost.x, 0, ghost.z), this.quaternion.identity(), this.scale.setScalar(ghost.alpha > 0.01 ? 1 : 0));
      checkers.setMatrixAt(i, this.matrix);
      look.setXYZW(i, 0, -1, 0, 0);
      fx.setXYZW(i, 1, ghost.alpha * 0.8, 0, 0);
      i++;
    }
    checkers.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
    fx.needsUpdate = true;

    this.diceFade += ((finale ? 0 : 1) - this.diceFade) * fade;
    const dice = this.diceMesh!, diceAttr = this.diceAttr!, rolling = clamp01((time - this.rollAt) / ROLL_SECONDS), settle = (1 - rolling) ** 3;
    const pulse = this.closing ? 0.28 + 0.14 * Math.sin(time * 5.5) : 0.1;
    for (let d = 0; d < THROWS; d++) {
      const die = this.dice[d];
      if (!die || this.diceFade < 0.02) {
        this.matrix.makeScale(0, 0, 0);
        dice.setMatrixAt(d, this.matrix);
        continue;
      }
      this.quaternion.setFromAxisAngle(this.axis, die.angle + settle * (d % 2 ? -4 : 4));
      this.matrix.compose(this.at.set(die.x + settle * (die.owner ? 0.8 : -0.8), 0, settle * (d % 2 ? 0.3 : -0.3)), this.quaternion, this.scale.setScalar((1 + 0.3 * settle - 0.14 * die.owner) * this.diceFade));
      dice.setMatrixAt(d, this.matrix);
      diceAttr.setXYZW(d, die.value, die.owner, die.owner, die.owner ? 0 : pulse);
    }
    dice.instanceMatrix.needsUpdate = true;
    diceAttr.needsUpdate = true;
    this.diceLamp!.tint.set(this.accent).multiplyScalar((this.closing ? 1.2 + 0.8 * Math.sin(time * 5.5) : 1.6) * this.diceFade);

    const values = this.pointState.image.data as Float32Array;
    for (let p = 0; p <= BG_BAR; p++) for (let c = 0; c < 3; c++) values[4 * p + c]! += (this.pointTarget[3 * p + c]! - values[4 * p + c]!) * fade;
    this.pointState.needsUpdate = true;
    const fresh = clamp01((time - this.replyAt) / FRESH_SECONDS);
    this.arrows.forEach((arrow, a) => {
      const aim = arrow.aim * (a < BG_STEPS ? 0.9 : 0.42 + 0.58 * (1 - fresh)) * this.diceFade;
      arrow.strength.value += (aim - arrow.strength.value) * fade;
    });
  }
}

export default (() => new BackgammonScene()) satisfies SceneFactory;
