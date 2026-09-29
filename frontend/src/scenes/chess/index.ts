import * as THREE from 'three/webgpu';
import { abs, atan, clamp, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, mod, positionWorld, select, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorld, type GridSize, type Vec3Tuple } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { clamp01 } from '@kit/mix';
import { coverage, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  CHESS_AGENT_SLOT, CHESS_BLACK, CHESS_CASTLE, CHESS_DRAW, CHESS_GAME, CHESS_KING, CHESS_KNIGHT, CHESS_LAST, CHESS_LOSS, CHESS_NEXT, CHESS_PASSANT, CHESS_PLY, CHESS_PROPOSED_TO_SLOT, CHESS_SQUARES,
  CHESS_STATE, CHESS_WIN, chessAgent, chessCount, chessFlip, chessIndex, chessMoveFrom, chessMoveLands, chessMovePiece, chessMovePromotion, chessMoveSpecial, chessMoveTaken, chessMoveTo, chessPiece,
  chessPlayed, chessProposal,
} from '@sim/envs/chess';
import { countEvents, lastEvent } from '@sim/events';
import type { EnvFrame, GridCell } from '@sim/types';
import { configurationName, constantList, parameter } from '../../data/envs';
import { formatCount } from '../../format';
import { Caption, GLYPH_COLUMNS, glyphAtlas } from './labels';
import { PIECE_KINDS, PIECE_SPAN, PIECE_TEXELS, pieceAtlas } from './pieces';

type N<T extends string> = THREE.Node<T>;

const HALF = 4;
const FRAME = 0.62;
const EDGE = HALF + FRAME;
const PIECE = 1.14;
const QUAD = (PIECE * PIECE_SPAN) / 100;
const INSET = 0.5 / PIECE_TEXELS;
const OUTLINE = 0.021;
const LIFT = 0.12;
const CURSOR_QUAD = 1.4;
const LABEL = 0.42;
const PANEL_X = 7.25;
const PLATE_H = 0.92;
const PLATE_W = 3.26;
const GAP = 0.3;
const EYE_CELL = 0.38;
const EYE_HALF = 4 * EYE_CELL + 0.1;
const MIDDLE = 2 * EYE_HALF + GAP + PLATE_H;
const EYE_Z = EYE_HALF - MIDDLE / 2;
const OFFER_Z = MIDDLE / 2 - PLATE_H / 2;
const RIVAL_Z = -EDGE + PLATE_H / 2;
const AGENT_Z = EDGE - PLATE_H / 2;
const TEXT_LEFT = -PLATE_W / 2 + 0.95;
const TEXT_W = PLATE_W - 1.75;
const LAMP_Z = -(HALF + FRAME / 2);
const BANNER_W = 5.8;
const BANNER_ASPECT = 4;
const BANNER_STEPS = 14;
const CATCHUP = 16;
const GHOSTS = 16;
const CAPACITY = 2 * 16 + GHOSTS + 1;
const BADGES = 3;

const GLIDE = 16;
const CURSOR_GLIDE = 26;
const FADE = 10;
const FLASH = 3;
const GHOST_SECONDS = 0.45;

const TABLE = 0x0c110f;
const MAPLE = 0xead3a8;
const WALNUT = 0x9f7654;
const FRAME_DARK = 0x2a1a10;
const FRAME_LIGHT = 0x4b2f1b;
const BOXWOOD = 0xdcc497;
const IVORY_TOP = 0xfdf9f0;
const IVORY_BASE = 0xdcceb3;
const IVORY_INK = 0x2a1f17;
const EBONY_TOP = 0x463d35;
const EBONY_BASE = 0x15110e;
const EBONY_INK = 0x050404;
const EBONY_LINE = 0xe8dbc1;
const EBONY_RIM = 0x8c7d6c;
const LAST_TINT = 0xd6c35a;
const FROM_TINT = 0x5fae86;
const CHECK_RED = 0xff4b3a;
const LAMP = 0xffb865;
const GLASS = 0x120f0c;
const WELL = 0x6a5c4c;
const GOLD = TOKENS.celebrationGold;

const FILES = 'abcdefgh';
const GLYPHS = [...FILES, '1', '2', '3', '4', '5', '6', '7', '8'];
const LETTERS = ['', '', 'N', 'B', 'R', 'Q', 'K'] as const;
const OUTCOMES = ['', 'checkmate', 'insufficient material', 'stalemate', 'fifty-move rule', 'threefold repetition'] as const;
const OPPONENTS = ['tablebase', 'random mover', 'alpha-beta'] as const;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const squareRing = (outer: number, inner: number): THREE.BufferGeometry => {
  const corner = (r: number) => [new THREE.Vector2(-r, -r), new THREE.Vector2(r, -r), new THREE.Vector2(r, r), new THREE.Vector2(-r, r)];
  const shape = new THREE.Shape(corner(outer));
  shape.holes.push(new THREE.Path(corner(inner).reverse()));
  return new THREE.ShapeGeometry(shape).rotateX(-Math.PI / 2);
};
const squareName = (square: number): string => `${FILES[square & 7]}${(square >> 3) + 1}`;
const moveName = (move: number): string => {
  const from = chessMoveFrom(move), to = chessMoveTo(move), promotion = chessMovePromotion(move);
  if (chessMoveSpecial(move) === CHESS_CASTLE) return to > from ? 'O-O' : 'O-O-O';
  return `${LETTERS[chessMovePiece(move)]}${squareName(from)}${chessMoveTaken(move) ? '×' : '–'}${squareName(to)}${promotion ? `=${LETTERS[promotion]}` : ''}`;
};
const victimOf = (move: number): number => (!chessMoveTaken(move) ? -1 : chessMoveSpecial(move) === CHESS_PASSANT ? chessMoveTo(move) ^ 8 : chessMoveTo(move));
const GRID: GridSize = { w: 2 * HALF, h: 2 * HALF };
const spot: Vec3Tuple = [0, 0, 0];
const viewX = (view: number): number => cellToWorld(view & 7, 7 - (view >> 3), GRID, spot)[0];
const viewZ = (view: number): number => cellToWorld(view & 7, 7 - (view >> 3), GRID, spot)[2];
const sameBoard = (a: EnvFrame, b: EnvFrame): boolean => {
  for (let i = 0; i < 8; i++) if (a.words[i] !== b.words[i]) return false;
  return true;
};

const stack = (layers: readonly (readonly [N<'vec3'>, N<'float'>])[]) => {
  let color: N<'vec3'> = vec3(0), alpha: N<'float'> = float(0);
  for (const [tint, cover] of layers) {
    color = tint.mul(cover).add(color.mul(cover.oneMinus()));
    alpha = cover.add(alpha.mul(cover.oneMinus()));
  }
  return { color: color.div(max(alpha, 1e-4)), alpha };
};

interface Piece { code: number; square: number; x: number; z: number; tx: number; tz: number; alpha: number; hop: number }
interface Ghost { code: number; x: number; z: number; at: number }

class Glyphs {
  readonly mesh: THREE.InstancedMesh;
  readonly look: THREE.InstancedBufferAttribute;
  readonly fx: THREE.InstancedBufferAttribute;
  private used = 0;
  private readonly matrix = new THREE.Matrix4();

  constructor(capacity: number, make: (look: THREE.InstancedBufferAttribute, fx: THREE.InstancedBufferAttribute) => THREE.Material, order: number) {
    this.look = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.fx = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.mesh = instancedMesh(flat(QUAD, QUAD), make(this.look, this.fx), capacity);
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  put(x: number, z: number, size: number, code: number, alpha: number, lift = 0, halo = 0, shadow = 1): void {
    const i = this.used++;
    this.matrix.makeScale(size, 1, size).setPosition(x, 0, z);
    this.mesh.setMatrixAt(i, this.matrix);
    this.look.setXYZW(i, code & 7, code >> 3, alpha, lift);
    this.fx.setXYZW(i, halo, shadow, 0, size);
  }

  end(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.look.needsUpdate = true;
    this.fx.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.dispose();
  }
}

class ChessScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.38, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'chess';

  private accent = 0xe8dcc0;
  private rival = 'opponent';
  private values: number[] = [];
  private readonly disposables: { dispose(): void }[] = [];
  private banner: Caption | null = null;
  private agentPlate: Caption | null = null;
  private offerPlate: Caption | null = null;

  private pieces: Piece[] = [];
  private ghosts: Ghost[] = [];
  private board: Glyphs | null = null;
  private hud: Glyphs | null = null;
  private cursorMesh: THREE.Mesh | null = null;
  private bannerMesh: THREE.Mesh | null = null;
  private readonly lantern = { mesh: null as THREE.Mesh | null, tint: new THREE.Color() };
  private readonly candle = { mesh: null as THREE.Mesh | null, tint: new THREE.Color() };
  private readonly alarm = { mesh: null as THREE.Mesh | null, tint: new THREE.Color() };

  private readonly flipU = uniform(0);
  private readonly lastFrom = uniform(new THREE.Vector2(-9, -9));
  private readonly lastTo = uniform(new THREE.Vector2(-9, -9));
  private readonly lastAmount = uniform(0);
  private readonly proposedFrom = uniform(new THREE.Vector2(-9, -9));
  private readonly proposedTo = uniform(new THREE.Vector2(-9, -9));
  private readonly proposedAmount = uniform(0);
  private readonly checkAt = uniform(new THREE.Vector2());
  private readonly checkAmount = uniform(0);
  private readonly lampAmount = uniform(0);
  private readonly offerTurn = uniform(0);
  private readonly bannerAmount = uniform(0);
  private readonly bannerTint = uniform(new THREE.Color(GOLD));
  private readonly eyeFrom = uniform(new THREE.Vector2(-9, -9));
  private readonly eyeTo = uniform(new THREE.Vector2(-9, -9));

  private step = -1;
  private flip = 0;
  private agent = 0;
  private proposal = 0;
  private offered = '';
  private reveal = 0;
  private replied = '';
  private flash = 0;
  private check = -1;
  private from = 0;
  private to = 0;
  private bannerFrom = -Infinity;
  private readonly shownCursor = new THREE.Vector2(NaN, NaN);
  private readonly aimCursor = new THREE.Vector2();
  private tallied = '';
  private tally = { wins: 0, losses: 0 };
  private readonly scratch = new THREE.Vector3();
  private readonly order: Piece[] = [];
  private readonly ledger = { turn: '—', proposal: '—', action: '—', last: '—', material: '—', games: '—', result: '—', note: '', clock: 0 };

  private legendFor(accent: number): LegendSpec {
    return {
      title: 'CHESS',
      rows: [
        { key: 'turn', label: 'turn', tint: LAMP, value: () => this.ledger.turn },
        { key: 'proposal', label: 'proposal', tint: accent, value: () => this.ledger.proposal },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
        { key: 'last', label: 'last moves', tint: LAST_TINT, value: () => this.ledger.last },
        { key: 'material', label: 'material', tint: 0xc9b48a, value: () => this.ledger.material },
        { key: 'games', label: 'games', tint: GOLD, value: () => this.ledger.games },
        { key: 'result', label: 'last game', tint: GOLD, value: () => this.ledger.result },
      ],
      note: () => this.ledger.note,
      bar: () => this.ledger.clock,
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor(this.accent);
    const opponent = parameter(ctx.environment, 'opponent'), name = OPPONENTS[opponent];
    if (name === undefined) throw new Error(`${ctx.environment.envId} seats opponent ${opponent}, which chess does not know`);
    this.rival = name;
    this.values = constantList(ctx.environment, 'CHESS_VALUE');
    return new THREE.Box2(new THREE.Vector2(-EDGE - 0.3, -EDGE - 0.3), new THREE.Vector2(PANEL_X + PLATE_W / 2 + 0.55, EDGE + 0.3));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff0da).multiplyScalar(0.7);
  }

  protected override poolOptions() {
    return { burstSize: 60, sighRadius: 3.5, sighHeight: 1.2 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        const view = state.pose[CHESS_PROPOSED_TO_SLOT]! ^ (state.pose[CHESS_AGENT_SLOT]! ? 63 : 0);
        agent.x = view & 7;
        agent.y = 7 - (view >> 3);
        return agent;
      },
      debug: (frame, kit) => {
        const words = frame.frameAt(frame.state.step).words, flip = chessFlip(words), agent = chessAgent(words), marks = kit.marks('pieces', 2 * 16);
        for (let square = 0; square < CHESS_SQUARES; square++) {
          const code = chessPiece(words, square);
          if (!code) continue;
          const view = square ^ flip;
          marks.set(viewX(view), 0.3, viewZ(view), 0.06 + 0.02 * (code & 7), code >> 3 === agent ? TOKENS.uiCyan : GOLD);
        }
        const arrow = (name: string, tint: number, from: number, to: number) => {
          const curve = kit.curve(name, 2, tint);
          if (from < 0 || to < 0) return;
          curve.point(viewX(from ^ flip), 0.3, viewZ(from ^ flip));
          curve.point(viewX(to ^ flip), 0.3, viewZ(to ^ flip));
        };
        const proposal = chessProposal(words), played = chessPlayed(words), last = words[CHESS_LAST]!;
        arrow('proposal', this.accent, chessMoveFrom(proposal), chessMoveTo(proposal));
        arrow('agent', TOKENS.uiCyan, played ? chessMoveFrom(played) : -1, chessMoveTo(played));
        arrow('opponent', LAST_TINT, (last & 127) - 1, ((last >>> 8) & 127) - 1);
        const alarm = kit.marks('check', 1);
        if ((last >>> 16) & 1) {
          const king = this.kingOf(words, agent);
          if (king >= 0) alarm.set(viewX(king ^ flip), 0.5, viewZ(king ^ flip), 0.2, CHECK_RED);
        }
      },
    };
  }

  private kingOf(words: ArrayLike<number>, color: number): number {
    for (let square = 0; square < CHESS_SQUARES; square++) if (chessPiece(words, square) === (CHESS_KING | (color << 3))) return square;
    return -1;
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    this.scene.background = new THREE.Color(TABLE);
    const grain = fbm(ctx.seed ^ 0xc4e55, { size: 256, frequency: 4, octaves: 5 }), atlas = pieceAtlas(), glyphs = glyphAtlas(GLYPHS);
    const banner = new Caption(BANNER_ASPECT), rival = new Caption(TEXT_W / PLATE_H, 192), agentPlate = new Caption(TEXT_W / PLATE_H, 192), offerPlate = new Caption(TEXT_W / PLATE_H, 192);
    this.banner = banner;
    this.agentPlate = agentPlate;
    this.offerPlate = offerPlate;
    this.disposables.push(grain, atlas, glyphs, banner.texture, rival.texture, agentPlate.texture, offerPlate.texture);
    rival.write(this.plate(this.rival.toUpperCase(), this.rivalNote(ctx)));
    const sample = (q: N<'vec2'>) => texture(grain, q).r;

    const table = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    const tp = positionWorld.xz;
    const nap = sample(tp.mul(0.35)).mul(0.5).add(sample(tp.mul(2.7)).mul(0.3)).add(sample(tp.mul(11)).mul(0.2));
    table.colorNode = light.lit(rgb(TABLE).mul(nap.mul(0.5).add(0.78)));
    this.add(new THREE.Mesh(flat(80, 80), table), -1);

    this.add(new THREE.Mesh(flat(2 * EDGE + 2, 2 * EDGE + 2), this.boardMaterial(light, sample, glyphs)), 0);

    const pieces = new Glyphs(CAPACITY, (look, fx) => this.glyphMaterial(atlas, look, fx, light), 3);
    const hud = new Glyphs(2 * 16 + BADGES, (look, fx) => this.glyphMaterial(atlas, look, fx, null), 7);
    this.board = pieces;
    this.hud = hud;
    for (const layer of [pieces, hud]) {
      this.scene.add(layer.mesh);
      this.disposables.push(layer);
    }

    this.cursorMesh = this.add(new THREE.Mesh(flat(CURSOR_QUAD, CURSOR_QUAD), this.cursorMaterial()), 8);
    this.add(new THREE.Mesh(flat(8 * EYE_CELL + 0.5, 8 * EYE_CELL + 0.5), this.eyeMaterial()), 6).position.set(PANEL_X, 0, EYE_Z);

    const lamp = this.add(new THREE.Mesh(flat(1.2, 1.2), this.lampMaterial()), 2);
    lamp.position.set(0, 0, LAMP_Z);
    this.add(new THREE.Mesh(flat(PLATE_W + 0.5, PLATE_H + 0.5), this.plateMaterial(rival.texture, this.lampAmount, null, LAMP)), 6).position.set(PANEL_X, 0, RIVAL_Z);
    this.add(new THREE.Mesh(flat(PLATE_W + 0.5, PLATE_H + 0.5), this.plateMaterial(offerPlate.texture, float(1), this.offerTurn, this.accent)), 6).position.set(PANEL_X, 0, OFFER_Z);
    this.add(new THREE.Mesh(flat(PLATE_W + 0.5, PLATE_H + 0.5), this.plateMaterial(agentPlate.texture, float(0.6), null, this.accent)), 6).position.set(PANEL_X, 0, AGENT_Z);

    this.bannerMesh = this.add(new THREE.Mesh(flat(BANNER_W + 0.6, BANNER_W / BANNER_ASPECT + 0.6), this.bannerMaterial(banner.texture)), 9);
    this.bannerMesh.visible = false;

    this.lantern.mesh = this.emit(light, squareRing(0.53, 0.47), this.lantern.tint);
    this.candle.mesh = this.emit(light, new THREE.CircleGeometry(0.15, 24).rotateX(-Math.PI / 2), this.candle.tint);
    this.candle.mesh.position.set(0, 0, LAMP_Z);
    this.alarm.mesh = this.emit(light, new THREE.CircleGeometry(0.22, 24).rotateX(-Math.PI / 2), this.alarm.tint);
  }

  private plate(name: string, note: string) {
    return [
      { text: name, font: '600', size: 0.36, baseline: 0.5, align: 'left' as const, x: 0, spacing: 0.03 },
      { text: note, font: 'italic 500', size: 0.24, baseline: 0.84, align: 'left' as const, x: 0 },
    ];
  }

  private rivalNote(ctx: SceneContext): string {
    const env = ctx.environment;
    if (this.rival === 'alpha-beta') return `depth ${parameter(env, 'depth')}+${parameter(env, 'quiescence')} · ${formatCount(parameter(env, 'nodes'))} nodes`;
    if (this.rival === 'tablebase') return 'longest defence';
    return 'any legal move';
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
    return mesh;
  }

  private emit(light: Stage2D['light'], geometry: THREE.BufferGeometry, tint: THREE.Color): THREE.Mesh {
    const mesh = light.emitter(geometry, uniform(tint) as unknown as N<'vec3'>);
    this.scene.add(mesh);
    this.disposables.push(geometry, mesh.material as THREE.Material);
    return mesh;
  }

  private boardMaterial(light: Stage2D['light'], sample: (q: N<'vec2'>) => N<'float'>, glyphs: THREE.Texture): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const p = positionWorld.xz, q = p.add(HALF), cell = floor(q), local = fract(q).sub(0.5);
    const dBoard = sdBox(p, vec2(HALF), 0), dCase = sdBox(p, vec2(EDGE), 0.16), inside = coverage(dBoard);
    const rank = float(7).sub(cell.y), dark = mod(cell.x.add(rank), 2).lessThan(0.5);
    const tone = sample(cell.mul(0.173).add(0.37)), shift = sample(cell.mul(0.291).add(0.61));
    const streak = sample(vec2(q.x.mul(0.05).add(shift), q.y.mul(0.9).add(tone.mul(3)))), fibre = sample(vec2(q.x.mul(0.3).add(tone), q.y.mul(5.5)));
    const wood = smoothstep(0.25, 0.8, streak).mul(0.6).add(fibre.mul(0.4));
    const maple = rgb(MAPLE).mul(wood.mul(0.1).add(0.93)).mul(tone.mul(0.08).add(0.97));
    const walnut = rgb(WALNUT).mul(wood.mul(0.2).add(0.86)).mul(tone.mul(0.12).add(0.95));
    const seam = coverage(max(abs(local.x), abs(local.y)).sub(0.494));
    let squares: N<'vec3'> = select(dark, walnut, maple).mul(seam.mul(0.14).add(0.86));

    const same = (target: N<'vec2'>) => step(abs(cell.x.sub(target.x)), 0.5).mul(step(abs(cell.y.sub(target.y)), 0.5));
    const last = max(same(this.lastFrom).mul(0.75), same(this.lastTo)).mul(this.lastAmount);
    squares = mix(squares, rgb(LAST_TINT).mul(select(dark, float(0.82), float(1.02))), last.mul(0.46));
    squares = mix(squares, rgb(FROM_TINT).mul(select(dark, float(0.8), float(1.05))), same(this.proposedFrom).mul(this.proposedAmount).mul(0.5));
    squares = mix(squares, rgb(this.accent).mul(select(dark, float(0.78), float(1))), same(this.proposedTo).mul(this.proposedAmount).mul(0.32));

    const vertical = abs(p.x).greaterThan(abs(p.y));
    const along = select(vertical, vec2(p.y, p.x), vec2(p.x, p.y));
    const frameStreak = sample(along.mul(vec2(0.045, 0.8))), frameFibre = sample(along.mul(vec2(0.25, 4.2)));
    const frameWood = mix(rgb(FRAME_DARK), rgb(FRAME_LIGHT), smoothstep(0.25, 0.8, frameStreak)).mul(frameFibre.mul(0.28).add(0.84));
    const bevel = smoothstep(-0.12, 0, dCase).mul(p.y.add(p.x.mul(0.4)).div(EDGE).mul(-0.35).add(0.05));
    const inlay = coverage(abs(dBoard.sub(0.065)).sub(0.014));
    const lip = coverage(abs(dBoard.sub(0.004)).sub(0.006));
    let frame: N<'vec3'> = frameWood.mul(bevel.add(1));
    frame = mix(frame, rgb(BOXWOOD).mul(0.92), inlay.mul(0.85));
    frame = mix(frame, rgb(0x120a05), lip.mul(0.7));

    const glyphAt = (index: N<'float'>, centre: N<'vec2'>) => {
      const t = p.sub(centre).div(LABEL).add(0.5), rows = GLYPHS.length / GLYPH_COLUMNS;
      const box = step(0, t.x).mul(step(t.x, 1)).mul(step(0, t.y)).mul(step(t.y, 1));
      const at = vec2(mod(index, GLYPH_COLUMNS).add(clamp(t.x, 0.01, 0.99)).div(GLYPH_COLUMNS), float(1).sub(floor(index.div(GLYPH_COLUMNS)).add(clamp(t.y, 0.01, 0.99)).div(rows)));
      return texture(glyphs, at).r.mul(box);
    };
    const file = clamp(floor(p.x.add(HALF)), 0, 7), row = clamp(floor(p.y.add(HALF)), 0, 7), flipped = this.flipU.greaterThan(0.5);
    const letter = select(flipped, float(7).sub(file), file), digit = select(flipped, row, float(7).sub(row)).add(8);
    const inkFile = glyphAt(letter, vec2(file.sub(HALF - 0.5), HALF + FRAME / 2)).mul(step(abs(p.x), HALF));
    const inkRank = glyphAt(digit, vec2(-(HALF + FRAME / 2), row.sub(HALF - 0.5))).mul(step(abs(p.y), HALF));
    frame = mix(frame, rgb(BOXWOOD), max(inkFile, inkRank).mul(0.88));

    const lampWell = sdCircle(p.sub(vec2(0, LAMP_Z)), 0.24);
    frame = mix(frame, rgb(0x0d0805), coverage(lampWell).mul(0.8));

    const albedo = mix(frame, squares, inside);
    const checkGlow = exp(length(p.sub(this.checkAt)).mul(-3.4)).mul(this.checkAmount).mul(inside);
    const lit = light.lit(albedo).add(rgb(CHECK_RED).mul(checkGlow.mul(0.9)));
    const shadow = smoothstep(0.55, -0.15, sdBox(p.sub(vec2(0.1, 0.16)), vec2(EDGE), 0.2)).mul(0.75);
    const result = stack([[vec3(0), shadow], [lit, coverage(dCase)]]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private glyphMaterial(atlas: THREE.Texture, look: THREE.InstancedBufferAttribute, fx: THREE.InstancedBufferAttribute, light: Stage2D['light'] | null): THREE.MeshBasicNodeMaterial {
    const L = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, F = instancedDynamicBufferAttribute(fx, 'vec4') as N<'vec4'>;
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const kind = L.x, black = L.y.greaterThan(0.5), alpha = L.z, lift = L.w, halo = F.x, shadowing = F.y, size = F.w;
    const t = uv();
    const at = (o: N<'vec2'>) => texture(atlas, vec2(kind.sub(1).add(clamp(o.x, INSET, 1 - INSET)).div(PIECE_KINDS), clamp(o.y, INSET, 1 - INSET)));
    const here = at(t), unit = size.mul(QUAD);
    const d = here.r.mul(unit), dd = here.g.mul(unit);
    const face = coverage(d);
    const edge = coverage(abs(d.add(size.mul(OUTLINE / 2))).sub(size.mul(OUTLINE / 2)));
    const line = coverage(dd).mul(face);
    const eps = 1.5 / PIECE_TEXELS;
    const nx = at(t.add(vec2(eps, 0))).r.sub(at(t.sub(vec2(eps, 0))).r).div(2 * eps);
    const rim = smoothstep(-0.05, -0.008, here.r).mul(face);
    const g = smoothstep(0.14, 0.88, t.y);
    const body = select(black, mix(rgb(EBONY_BASE), rgb(EBONY_TOP), g), mix(rgb(IVORY_BASE), rgb(IVORY_TOP), g));
    const sheen = select(black, rgb(EBONY_RIM).mul(0.55), vec3(0.1));
    let albedo: N<'vec3'> = body.add(sheen.mul(rim.mul(clamp(nx.negate(), 0, 1)))).mul(rim.mul(clamp(nx, 0, 1)).mul(-0.16).add(1));
    albedo = mix(albedo, select(black, rgb(EBONY_LINE), rgb(IVORY_INK)), line.mul(select(black, float(0.72), float(0.92))));
    albedo = mix(albedo, select(black, rgb(EBONY_INK), rgb(IVORY_INK)), edge);
    const shade = light ? light.lit(albedo) : albedo;
    const offset = vec2(0.024, -0.036).mul(lift.mul(1.8).add(1));
    const cast = smoothstep(lift.mul(0.03).add(0.022), float(-0.008), at(t.sub(offset)).r).mul(0.46).mul(shadowing);
    const ring = coverage(abs(here.r.sub(0.016)).sub(0.007)).mul(halo), bloom = exp(max(here.r, 0).mul(-60)).mul(halo).mul(0.35);
    const result = stack([[vec3(0), cast], [rgb(this.accent), bloom], [shade, face], [rgb(this.accent).mul(1.3), ring.mul(face.oneMinus())]]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha.mul(alpha);
    return material;
  }

  private cursorMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(CURSOR_QUAD), a = abs(t), r = 0.5, arm = 0.2, thick = 0.024;
    const d = min(sdBox(a.sub(vec2(r - arm / 2, r)), vec2(arm / 2, thick), 0.01), sdBox(a.sub(vec2(r, r - arm / 2)), vec2(thick, arm / 2), 0.01));
    const stroke = coverage(d), rimDark = coverage(d.sub(0.02));
    const glow = exp(max(d, 0).mul(-34)).mul(0.3);
    const ring = coverage(abs(sdBox(t, vec2(0.37), 0.1)).sub(0.011));
    const tint = rgb(this.accent).mul(1.3);
    const result = stack([[vec3(0), rimDark.mul(0.5)], [tint, glow], [tint, stroke], [tint, ring.mul(0.85)]]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private eyeMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const size = 8 * EYE_CELL + 0.5, t = uv().sub(0.5).mul(size), d = sdBox(t, vec2(EYE_HALF), 0.12);
    const g = t.div(EYE_CELL).add(4), cell = floor(g), local = fract(g).sub(0.5);
    const inside = step(abs(t.x), 4 * EYE_CELL).mul(step(abs(t.y), 4 * EYE_CELL));
    const dark = mod(cell.x.add(cell.y), 2).lessThan(0.5);
    const same = (target: N<'vec2'>) => step(abs(cell.x.sub(target.x)), 0.5).mul(step(abs(cell.y.sub(target.y)), 0.5));
    const tile = coverage(sdBox(local, vec2(0.46), 0.08)).mul(inside);
    let body: N<'vec3'> = mix(rgb(GLASS), select(dark, rgb(0x4a3b2d), rgb(0x6f5d48)), tile.mul(0.9));
    body = mix(body, rgb(FROM_TINT), tile.mul(same(this.eyeFrom)).mul(0.55));
    body = mix(body, rgb(this.accent), tile.mul(same(this.eyeTo)).mul(0.45));
    const rim = coverage(abs(d).sub(0.012));
    const shadow = smoothstep(0.3, -0.1, sdBox(t.sub(vec2(0.06, -0.08)), vec2(EYE_HALF), 0.12)).mul(0.55);
    const result = stack([[vec3(0), shadow], [mix(body, rgb(this.accent).mul(0.6), rim), coverage(d).mul(0.95)]]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private lampMaterial(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(1.2), r = length(t);
    const core = coverage(sdCircle(t, 0.13));
    const flame = exp(r.mul(-9)).mul(this.lampAmount);
    const angle = fract(atan(t.x, t.y.negate()).div(2 * Math.PI));
    const ring = coverage(abs(r.sub(0.2)).sub(0.022));
    const ticks = step(0.5, fract(angle.mul(12)));
    const sweep = smoothstep(0.35, 1, angle).mul(step(0.02, this.lampAmount.sub(0.3)));
    const brass = rgb(0xc9a05a).mul(0.55);
    const color = mix(brass, rgb(LAMP).mul(this.lampAmount.mul(2.4).add(0.4)), core).add(rgb(LAMP).mul(flame.mul(1.4)));
    const alpha = max(max(core, flame.mul(0.9)), ring.mul(ticks.mul(0.4).add(sweep.mul(0.6)).add(0.18)));
    material.colorNode = mix(color, rgb(LAMP).mul(sweep.mul(1.6).add(0.5)), ring.mul(core.oneMinus()));
    material.opacityNode = clamp(alpha, 0, 1);
    return material;
  }

  private plateMaterial(text: THREE.Texture, lamp: N<'float'>, angle: N<'float'> | null, tint: number): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const t = uv().sub(0.5).mul(vec2(PLATE_W + 0.5, PLATE_H + 0.5)), d = sdBox(t, vec2(PLATE_W / 2, PLATE_H / 2), 0.16);
    const tx = t.x.sub(TEXT_LEFT).div(TEXT_W), ty = t.y.div(PLATE_H).add(0.5);
    const ink = texture(text, vec2(tx, ty)).r.mul(step(0, tx)).mul(step(tx, 1)).mul(step(0, ty)).mul(step(ty, 1));
    const spot = t.sub(vec2(PLATE_W / 2 - 0.46, 0)), r = length(spot);
    const core = coverage(sdCircle(spot, 0.12)), flame = exp(r.mul(-10)).mul(lamp);
    const ring = coverage(abs(r.sub(0.21)).sub(0.018));
    const turn = angle ? fract(atan(spot.x, spot.y).div(2 * Math.PI).sub(angle)) : float(0);
    const ticks = step(0.5, fract(turn.mul(12)));
    const sweep = angle ? smoothstep(0.4, 1, turn).mul(step(0.5, lamp)) : float(0);
    const badge = coverage(sdCircle(t.sub(vec2(-PLATE_W / 2 + 0.5, 0)), 0.36));
    const body = mix(mix(rgb(GLASS), rgb(0x241d16), coverage(d.add(0.05)).mul(0.6)), rgb(WELL), badge.mul(0.9));
    const rim = coverage(abs(d).sub(0.012));
    const glow = rgb(tint).mul(lamp.mul(2.2).add(0.35));
    const shadow = smoothstep(0.3, -0.1, sdBox(t.sub(vec2(0.06, -0.08)), vec2(PLATE_W / 2, PLATE_H / 2), 0.16)).mul(0.55);
    const result = stack([
      [vec3(0), shadow],
      [mix(body, rgb(tint).mul(lamp.mul(0.5).add(0.45)), rim), coverage(d).mul(0.95)],
      [rgb(0xeee2c8), ink.mul(0.92)],
      [rgb(tint).mul(0.8), ring.mul(ticks.mul(0.35).add(sweep.mul(0.65)).add(0.2))],
      [glow, max(core, flame.mul(0.8))],
    ]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha;
    return material;
  }

  private bannerMaterial(text: THREE.Texture): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    const height = BANNER_W / BANNER_ASPECT, t = uv().sub(0.5).mul(vec2(BANNER_W + 0.6, height + 0.6));
    const d = sdBox(t, vec2(BANNER_W / 2, height / 2), height / 2);
    const ink = texture(text, t.div(vec2(BANNER_W, height)).add(0.5)).r.mul(step(abs(t.x), BANNER_W / 2)).mul(step(abs(t.y), height / 2));
    const rim = coverage(abs(d).sub(0.016));
    const body = mix(mix(rgb(GLASS), this.bannerTint.mul(0.9), rim), this.bannerTint.mul(1.25), ink);
    const shadow = smoothstep(0.4, -0.1, d.sub(0.05)).mul(0.5);
    const result = stack([[vec3(0), shadow], [body, coverage(d).mul(mix(float(0.74), float(1), max(ink, rim)))]]);
    material.colorNode = result.color;
    material.opacityNode = result.alpha.mul(this.bannerAmount);
    return material;
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.board = this.hud = null;
    this.banner = this.agentPlate = this.offerPlate = null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    const step = frame.state.step;
    if (step !== this.step || frame.scrubbed) this.advance(frame, step, pool, time);
    this.present(frame, time);
  }

  private place(square: number, flip: number, out: { x: number; z: number }): void {
    const view = square ^ flip;
    out.x = viewX(view);
    out.z = viewZ(view);
  }

  private advance(frame: FrameUpdate, step: number, pool: Stage2D['pool'], time: number): void {
    let snapped = frame.scrubbed || this.step < 0 || step <= this.step || step - this.step > CATCHUP;
    if (!snapped) {
      for (let i = this.step + 1; i <= step; i++) {
        const before = frame.frameAt(i - 1), after = frame.frameAt(i);
        if (before.step !== i - 1 || after.step !== i) { snapped = true; break; }
        if (after.done) this.finale(before, after, pool, time);
        if (after.done || !sameBoard(before, after)) this.match(before, after, pool, time);
      }
    }
    const now = frame.frameAt(step);
    if (snapped) this.snap(now);
    this.step = step;
    this.read(frame, now, snapped);
  }

  private snap(now: EnvFrame): void {
    const flip = chessFlip(now.words);
    this.pieces = [];
    this.ghosts = [];
    for (let square = 0; square < CHESS_SQUARES; square++) {
      const code = chessPiece(now.words, square);
      if (!code) continue;
      const piece: Piece = { code, square, x: 0, z: 0, tx: 0, tz: 0, alpha: 1, hop: 0 };
      this.place(square, flip, piece);
      piece.tx = piece.x;
      piece.tz = piece.z;
      this.pieces.push(piece);
    }
    this.shownCursor.set(NaN, NaN);
  }

  private match(before: EnvFrame, after: EnvFrame, pool: Stage2D['pool'], time: number): void {
    const flip = chessFlip(after.words), reset = after.done !== 0;
    const held = new Map<number, Piece>();
    for (const piece of this.pieces) held.set(piece.square, piece);
    const vacated: number[] = [], arrived: number[] = [];
    for (let square = 0; square < CHESS_SQUARES; square++) {
      const was = chessPiece(before.words, square), is = chessPiece(after.words, square);
      if (was === is && !reset) continue;
      if (was && held.has(square)) vacated.push(square);
      if (is) arrived.push(square);
    }
    const target = { x: 0, z: 0 }, moved: Piece[] = [];
    const claim = (square: number, fits: (piece: Piece) => boolean): Piece | null => {
      this.place(square, flip, target);
      let best = -1, reach = Infinity;
      vacated.forEach((from, i) => {
        const piece = held.get(from)!;
        if (!fits(piece)) return;
        const distance = Math.hypot(piece.tx - target.x, piece.tz - target.z);
        if (distance < reach) { reach = distance; best = i; }
      });
      if (best < 0) return null;
      const piece = held.get(vacated[best]!)!;
      vacated.splice(best, 1);
      return piece;
    };
    const pending: number[] = [];
    for (const square of arrived) {
      const code = chessPiece(after.words, square), piece = claim(square, (p) => p.code === code);
      if (piece) { piece.square = square; moved.push(piece); } else pending.push(square);
    }
    const fresh: number[] = [];
    for (const square of pending) {
      const code = chessPiece(after.words, square), piece = claim(square, (p) => (p.code & CHESS_BLACK) === (code & CHESS_BLACK));
      if (piece) { piece.code = code; piece.square = square; moved.push(piece); } else fresh.push(square);
    }
    for (const square of vacated) {
      const piece = held.get(square)!;
      this.ghosts.push({ code: piece.code, x: piece.x, z: piece.z, at: time });
      this.pieces.splice(this.pieces.indexOf(piece), 1);
      if (!reset) {
        this.scratch.set(piece.x, 0.2, piece.z);
        pool.burst(this.scratch, piece.code & CHESS_BLACK ? 0x9a8a76 : 0xfff2d6, 0.45, time);
      }
    }
    if (this.ghosts.length > GHOSTS) this.ghosts.splice(0, this.ghosts.length - GHOSTS);
    for (const piece of moved) {
      this.place(piece.square, flip, target);
      piece.hop = (piece.code & 7) === CHESS_KNIGHT && !reset ? 1 : 0;
      piece.tx = target.x;
      piece.tz = target.z;
    }
    for (const square of fresh) {
      const piece: Piece = { code: chessPiece(after.words, square), square, x: 0, z: 0, tx: 0, tz: 0, alpha: 0, hop: 0 };
      this.place(square, flip, piece);
      piece.tx = piece.x;
      piece.tz = piece.z;
      this.pieces.push(piece);
    }
  }

  private finale(before: EnvFrame, after: EnvFrame, pool: Stage2D['pool'], time: number): void {
    const last = after.words[CHESS_LAST]!, result = (last >>> 24) & 3, agent = chessAgent(before.words), flip = chessFlip(before.words);
    const at = { x: 0, z: 0 };
    const king = (color: number): THREE.Vector3 => {
      const square = this.kingOf(before.words, color);
      if (square < 0) throw new Error(`Chess game ${before.words[CHESS_GAME]} ended without a ${color ? 'black' : 'white'} king`);
      this.place(square, flip, at);
      return this.scratch.set(at.x, 0.2, at.z);
    };
    if (result === CHESS_WIN) {
      const spot = king(agent ^ 1);
      pool.burst(spot, GOLD, 1, time);
      pool.ripple(spot, GOLD, 0.8, time);
      pool.pulse.fire(0.8);
    } else if (result === CHESS_LOSS) {
      const spot = king(agent);
      pool.burst(spot, TOKENS.failureBlue, 0.5, time);
      pool.sigh(0.8, spot, time);
    } else if (result === CHESS_DRAW) {
      pool.ripple(king(agent), this.accent, 0.55, time);
      pool.ripple(king(agent ^ 1), this.accent, 0.55, time);
    }
  }

  private read(frame: FrameUpdate, now: EnvFrame, snapped: boolean): void {
    const words = now.words, step = now.step, last = words[CHESS_LAST]!, game = words[CHESS_GAME]!, ply = words[CHESS_PLY]!;
    const proposal = chessProposal(words), index = chessIndex(words), count = chessCount(words);
    this.flip = chessFlip(words);
    this.agent = chessAgent(words);
    this.proposal = proposal;
    const offered = `${game}:${ply}:${index}`;
    if (offered !== this.offered) this.reveal = snapped ? 1 : 0;
    this.offered = offered;
    this.offerTurn.value = index / count;
    this.agentPlate?.write(this.plate('AGENT', `plays ${this.agent ? 'black' : 'white'} · to move`));
    this.offerPlate?.write(this.plate(`PROPOSAL ${index + 1} / ${count}`, `${moveName(proposal)} · ${index + 1 < count ? 'play or next' : 'next plays it'}`));
    this.check = (last >>> 16) & 1 ? this.kingOf(words, this.agent) : -1;

    const from = last & 127, to = (last >>> 8) & 127, replied = `${game}:${ply}`;
    if (from !== this.from || to !== this.to) this.lastAmount.value = snapped ? 1 : 0;
    if (from && to && replied !== this.replied && !snapped) this.flash = 1;
    this.replied = replied;
    this.from = from;
    this.to = to;

    const done = lastEvent(frame.eventLog, 'done', step);
    this.bannerFrom = done === undefined ? -Infinity : done;
    const outcome = (last >>> 20) & 15, result = (last >>> 24) & 3;
    if (done !== undefined && step - done < BANNER_STEPS && this.banner) {
      const ended = frame.frameAt(done - 1), color = ended.step === done - 1 ? chessAgent(ended.words) : this.agent;
      const white = result === CHESS_WIN ? color === 0 : color === 1;
      this.banner.write([
        { text: result === CHESS_DRAW ? '½–½' : white ? '1–0' : '0–1', font: '600', size: 0.42, baseline: 0.5 },
        { text: OUTCOMES[outcome] ?? '', font: 'italic 500', size: 0.2, baseline: 0.83 },
      ]);
      this.bannerTint.value.set(result === CHESS_WIN ? GOLD : result === CHESS_LOSS ? TOKENS.failureBlue : this.accent);
    }
    this.ledgerFor(frame, now, outcome, result);
  }

  private ledgerFor(frame: FrameUpdate, now: EnvFrame, outcome: number, result: number): void {
    const words = now.words, agent = this.agent, ply = words[CHESS_PLY]!, index = chessIndex(words), count = chessCount(words);
    this.ledger.turn = `agent (${agent ? 'black' : 'white'}) to move`;
    this.ledger.proposal = `${moveName(this.proposal)} · ${index + 1} of ${count}`;
    const before = now.step > 0 ? frame.frameAt(now.step - 1) : null, next = Math.trunc(now.actions[0]!) === CHESS_NEXT;
    if (!before || before.step !== now.step - 1) this.ledger.action = '—';
    else if (!next) this.ledger.action = 'play';
    else this.ledger.action = chessIndex(before.words) + 1 < chessCount(before.words) ? 'next' : 'next · clock plays';
    const played = chessPlayed(words), from = this.from, to = this.to;
    const reply = from && to ? `reply ${LETTERS[chessPiece(words, to - 1) & 7] ?? ''}${squareName(from - 1)}–${squareName(to - 1)}` : '';
    this.ledger.last = [played ? `you ${moveName(played)}` : '', reply].filter(Boolean).join(' · ') || '—';
    let balance = 0;
    for (let s = 0; s < CHESS_SQUARES; s++) {
      const c = chessPiece(words, s);
      if (c) balance += ((c >> 3) === agent ? 1 : -1) * (this.values[c & 7] ?? 0);
    }
    this.ledger.material = balance === 0 ? 'even' : `${balance > 0 ? '+' : '−'}${Math.abs(balance / (this.values[1] || 1))}`;
    this.ledger.result = result ? `${result === CHESS_WIN ? '+1 won' : result === CHESS_LOSS ? '−1 lost' : '½ drawn'} · ${OUTCOMES[outcome] ?? ''}` : '—';
    this.ledger.clock = (index + 1) / count;
    const games = words[CHESS_GAME]!, key = `${games}:${frame.eventLog.count}`;
    if (key !== this.tallied) {
      this.tally = { wins: countEvents(frame.eventLog, 'goal', -1, now.step), losses: countEvents(frame.eventLog, 'harm', -1, now.step) };
      this.tallied = key;
    }
    const { wins, losses } = this.tally;
    this.ledger.games = `won ${formatCount(wins)} · drawn ${formatCount(games - wins - losses)} · lost ${formatCount(losses)}`;
    this.ledger.note = `${configurationName(this.ctx!.environment).split('-').slice(1).join(' ')} · game ${formatCount(games + 1)} · move ${Math.floor(ply / 2) + 1} · fifty-move clock ${(words[CHESS_STATE]! >>> 16) & 255}`;
  }

  private present(frame: FrameUpdate, time: number): void {
    const dt = frame.dtReal, glide = 1 - Math.exp(-GLIDE * dt), fade = 1 - Math.exp(-FADE * dt), quick = 1 - Math.exp(-CURSOR_GLIDE * dt);
    const flip = this.flip, board = this.board;
    if (!board) return;
    this.flipU.value = flip ? 1 : 0;
    const proposal = this.proposal, from = chessMoveFrom(proposal), to = chessMoveTo(proposal), victim = victimOf(proposal);
    this.reveal += (1 - this.reveal) * quick;

    board.begin();
    this.order.length = 0;
    for (const piece of this.pieces) {
      piece.x += (piece.tx - piece.x) * glide;
      piece.z += (piece.tz - piece.z) * glide;
      piece.alpha += (1 - piece.alpha) * fade;
      const away = Math.hypot(piece.tx - piece.x, piece.tz - piece.z);
      if (away < 0.002) { piece.x = piece.tx; piece.z = piece.tz; piece.hop = 0; }
      this.order.push(piece);
    }
    this.order.sort((a, b) => a.z - b.z + (a.square === from ? 50 : 0) - (b.square === from ? 50 : 0) + (Math.hypot(a.tx - a.x, a.tz - a.z) > 0.05 ? 25 : 0) - (Math.hypot(b.tx - b.x, b.tz - b.z) > 0.05 ? 25 : 0));
    this.ghosts = this.ghosts.filter((ghost) => time - ghost.at < GHOST_SECONDS);
    for (const ghost of this.ghosts) {
      const age = clamp01((time - ghost.at) / GHOST_SECONDS);
      board.put(ghost.x, ghost.z - 0.1 * age, 1 + 0.25 * age, ghost.code, (1 - age) ** 2, age, 0, 0.4);
    }
    for (const piece of this.order) {
      const lifted = piece.square === from, away = Math.hypot(piece.tx - piece.x, piece.tz - piece.z);
      const hop = piece.hop * Math.min(1, away) * 0.9 + (lifted ? this.reveal : 0);
      const alpha = piece.square === victim ? piece.alpha * (1 - 0.6 * this.reveal) : piece.alpha;
      board.put(piece.x, piece.z - LIFT * hop, 1 + 0.08 * hop, piece.code, alpha, hop, lifted ? 0.55 * this.reveal : 0, 1);
    }
    board.put(viewX(to ^ flip), viewZ(to ^ flip), 1, chessMoveLands(proposal) | (this.agent << 3), 0.5 * this.reveal, 0, 0.8 * this.reveal, 0);
    board.end();

    const fromView = from ^ flip, toView = to ^ flip;
    this.proposedFrom.value.set(fromView & 7, 7 - (fromView >> 3));
    this.proposedTo.value.set(toView & 7, 7 - (toView >> 3));
    this.proposedAmount.value = this.reveal;
    this.eyeFrom.value.set(fromView & 7, fromView >> 3);
    this.eyeTo.value.set(toView & 7, toView >> 3);
    this.aimCursor.set(viewX(toView), viewZ(toView));
    if (Number.isNaN(this.shownCursor.x)) this.shownCursor.copy(this.aimCursor);
    this.shownCursor.lerp(this.aimCursor, quick);
    this.cursorMesh!.position.set(this.shownCursor.x, 0, this.shownCursor.y);

    const flicker = 0.9 + 0.06 * Math.sin(time * 9.1) + 0.04 * Math.sin(time * 23.7 + 1.3);
    this.lantern.mesh!.position.set(this.shownCursor.x, 0, this.shownCursor.y);
    this.lantern.tint.set(this.accent).multiplyScalar(0.34 * flicker);
    this.flash *= Math.exp(-FLASH * dt);
    this.lampAmount.value = this.flash;
    this.candle.tint.set(LAMP).multiplyScalar((0.3 + 2.2 * this.flash) * flicker);

    if (this.from && this.to) {
      const a = (this.from - 1) ^ flip, b = (this.to - 1) ^ flip;
      this.lastFrom.value.set(a & 7, 7 - (a >> 3));
      this.lastTo.value.set(b & 7, 7 - (b >> 3));
    }
    this.lastAmount.value += ((this.from && this.to ? 1 : 0) - this.lastAmount.value) * fade;

    const alarm = this.check >= 0;
    if (alarm) {
      const view = this.check ^ flip;
      this.checkAt.value.set(viewX(view), viewZ(view));
      this.alarm.mesh!.position.set(viewX(view), 0, viewZ(view));
    }
    this.checkAmount.value += ((alarm ? 0.75 + 0.25 * Math.sin(time * 7) : 0) - this.checkAmount.value) * quick;
    this.alarm.tint.set(CHECK_RED).multiplyScalar(4 * this.checkAmount.value);
    this.alarm.mesh!.visible = this.checkAmount.value > 0.01;

    this.presentEye(frame);
    this.presentBanner(frame);
  }

  private presentEye(frame: FrameUpdate): void {
    const hud = this.hud!, obs = frame.state.observations;
    hud.begin();
    for (let i = 0; i < CHESS_SQUARES; i++) {
      const value = obs[i]!, kind = Math.round(Math.abs(value) * 6);
      if (!kind) continue;
      const black = value > 0 ? this.agent : this.agent ^ 1;
      hud.put(PANEL_X + ((i & 7) - 3.5) * EYE_CELL, EYE_Z + (3.5 - (i >> 3)) * EYE_CELL, 0.9 * EYE_CELL, kind | (black << 3), 1, 0, 0, 0.6);
    }
    const badgeX = PANEL_X - PLATE_W / 2 + 0.5;
    hud.put(badgeX, RIVAL_Z + 0.02, 0.74, CHESS_KING | ((this.agent ^ 1) << 3), 1, 0, 0, 0.5);
    hud.put(badgeX, OFFER_Z + 0.02, 0.74, chessMoveLands(this.proposal) | (this.agent << 3), 1, 0, 0, 0.5);
    hud.put(badgeX, AGENT_Z + 0.02, 0.74, CHESS_KING | (this.agent << 3), 1, 0, 0, 0.5);
    hud.end();
  }

  private presentBanner(frame: FrameUpdate): void {
    const mesh = this.bannerMesh!, age = frame.position - this.bannerFrom;
    const amount = age >= 0 && age < BANNER_STEPS ? Math.min(1, age * 3, (BANNER_STEPS - age) / 4) : 0;
    this.bannerAmount.value = amount;
    mesh.visible = amount > 0.01;
  }
}

export default (() => new ChessScene()) satisfies SceneFactory;
