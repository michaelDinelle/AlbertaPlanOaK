import * as THREE from 'three/webgpu';
import { abs, clamp, dot, exp, float, floor, fract, int, length, max, min, mix, screenCoordinate, select, smoothstep, step, uniform, uniformArray, uv, vec2 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import type { Light2D } from '@kit/light2d';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { CATCH_BALL_X, CATCH_BALL_Y, CATCH_PADDLE } from '@sim/envs/catch';
import { SIM_EVENT_TYPES, type EventStream, type GridCell } from '@sim/types';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;

const ACTIONS = ['← left', 'stay', 'right →'] as const;
const BEZEL = 0.5;
const GUTTER = 0.9;
const BALL = 0.25;
const LENS = 0.3;
const ORB = 1.8;
const CRADLE_W = 1.9;
const CRADLE_H = 1.9;
const BALL_CORE = 0xe6fbff;
const BALL_GLOW = 0x6fd6ff;
const CAUGHT = 0xffe7a8;
const MISS = TOKENS.failureBlue;
const MISS_COLOR = new THREE.Color(MISS);
const TAIL = 2.4;
const TUBE = 0.62;
const TUBE_GAP = 0.35;
const GOAL = SIM_EVENT_TYPES.indexOf('goal');
const HARM = SIM_EVENT_TYPES.indexOf('harm');

const dither = fract(fract(dot(floor(screenCoordinate.xy), vec2(0.06711056, 0.00583715))).mul(52.9829189)).sub(0.5).mul(0.0016);

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const additive = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });

const arc = (p: N<'vec2'>, aperture: number, radius: number, thickness: number): N<'float'> => {
  const q = vec2(abs(p.x), p.y.negate()), s = Math.sin(aperture), c = Math.cos(aperture);
  return select(q.x.mul(c).greaterThan(q.y.mul(s)), length(q.sub(vec2(s * radius, c * radius))), abs(length(q).sub(radius))).sub(thickness);
};

const orb = (core: N<'vec3'>, glow: N<'vec3'>, alpha: N<'float'>) => {
  const material = additive();
  const p = uv().sub(0.5).mul(ORB), d = sdCircle(p, BALL);
  const shine = smoothstep(0, BALL * 1.2, length(p.sub(vec2(-0.08, 0.08)))).oneMinus();
  material.colorNode = core.mul(coverage(d).mul(shine.mul(1.2).add(1))).add(glow.mul(exp(max(d, 0).mul(-8)).mul(0.5))).mul(alpha);
  return material;
};

class CatchScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.6, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'catch';

  private grid: GridSize = { w: 5, h: 10 };
  private accent = 0xffd166;
  private ball: THREE.Mesh | null = null;
  private paddle: THREE.Mesh | null = null;
  private landed: THREE.Mesh | null = null;
  private ballLight: THREE.Mesh | null = null;
  private paddleLight: THREE.Mesh | null = null;
  private landedLight: THREE.Mesh | null = null;
  private readonly ballCol = uniform(0);
  private readonly ballRow = uniform(0);
  private readonly aligned = uniform(0);
  private readonly left = uniform(0);
  private readonly right = uniform(0);
  private readonly flash = uniform(0);
  private readonly radiance = uniform(3);
  private readonly landedAlpha = uniform(1);
  private readonly shatter = uniform(0);
  private readonly landedTint = uniform(new THREE.Color(CAUGHT));
  private readonly landedEmit = uniform(new THREE.Color());
  private readonly scratch = new THREE.Vector3();
  private readonly tally = { log: null as EventStream | null, count: -1, through: -1, caught: 0, missed: 0 };
  private outcomes = uniformArray([0], 'float' as const);
  private readonly drawn = { step: -1, count: -1 };
  private readonly ledger = { paddle: 0, col: 0, row: 0, action: '—', reward: 0 };

  override legend: LegendSpec = {
    title: 'CATCH',
    rows: [
      { key: 'paddle', label: 'paddle', tint: TOKENS.accents.Catch!, value: () => `column ${this.ledger.paddle + 1} of ${this.grid.w}` },
      { key: 'ball', label: 'ball', tint: BALL_GLOW, value: () => `column ${this.ledger.col + 1} · row ${this.ledger.row + 1} of ${this.grid.h}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => `${formatCount(this.tally.caught)} caught · ${formatCount(this.tally.missed)} missed`,
    bar: () => this.tally.caught / Math.max(1, this.tally.caught + this.tally.missed),
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const grid = ctx.geometry.grid;
    if (!grid) throw new Error(`${ctx.environment.envId} lacks its board`);
    this.grid = { w: grid.w, h: grid.h };
    const x = grid.w / 2 + BEZEL + TUBE_GAP + TUBE + 0.3, top = -grid.h / 2 - BEZEL - 0.3, bottom = grid.h / 2 + GUTTER + BEZEL + 0.3 + 0.08 * grid.h;
    return new THREE.Box2(new THREE.Vector2(-x, top), new THREE.Vector2(x, bottom));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x5d6c94).multiplyScalar(0.9);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const paddle: GridCell = { x: 0, y: 0 }, ball: GridCell = { x: 0, y: 0 }, a = new THREE.Vector3(), b = new THREE.Vector3();
    return {
      agentCell: (state) => {
        paddle.x = state.pose[CATCH_PADDLE]!;
        paddle.y = this.grid.h - 1;
        return paddle;
      },
      goalCell: (state) => {
        ball.x = state.pose[CATCH_BALL_X]!;
        ball.y = state.pose[CATCH_BALL_Y]!;
        return ball;
      },
      debug: (frame, kit) => {
        const state = frame.state, x = state.pose[CATCH_BALL_X]!, h = this.grid.h;
        const fall = kit.curve('fall', 2, BALL_GLOW), landing = kit.marks('landing', 1);
        cellToWorldVec(x, state.pose[CATCH_BALL_Y]!, this.grid, a);
        cellToWorldVec(x, h - 1, this.grid, b);
        fall.point(a.x, 0.05, a.z);
        fall.point(b.x, 0.05, b.z);
        const before = state.done && state.step > 0 ? frame.frameAt(state.step - 1) : null;
        if (!before || before.step !== state.step - 1) return;
        const col = before.words[0]!;
        cellToWorldVec(col, h - 1, this.grid, a);
        landing.set(a.x, 0.1, a.z, 0.16, col === state.pose[CATCH_PADDLE] ? TOKENS.celebrationGold : MISS);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    this.accent = this.accentOf(ctx);
    this.scene.background = new THREE.Color(0x040508);
    this.scene.add(this.board(light), this.tube(light));

    const gold = rgb(this.accent);
    const cradle = additive();
    const p = uv().sub(0.5).mul(vec2(CRADLE_W, CRADLE_H)), cup = arc(p.sub(vec2(0, 0.06)), 1.2, 0.36, 0.065);
    const chevron = (side: number) => min(sdSegment(p, vec2(side * 0.58, 0.13), vec2(side * 0.72, 0), 0.032), sdSegment(p, vec2(side * 0.72, 0), vec2(side * 0.58, -0.13), 0.032));
    const arrows = coverage(chevron(-1)).mul(this.left).add(coverage(chevron(1)).mul(this.right));
    const halo = coverage(abs(sdCircle(p, this.flash.oneMinus().mul(0.55).add(0.35))).sub(0.03)).mul(this.flash.mul(this.flash));
    cradle.colorNode = gold.mul(coverage(cup).mul(this.flash.mul(0.6).add(1.25)).add(exp(max(cup, 0).mul(-10)).mul(this.flash.mul(0.4).add(0.35))).add(arrows).add(halo.mul(1.2)));
    const paddle = new THREE.Mesh(flat(CRADLE_W, CRADLE_H), cradle);
    paddle.renderOrder = 2;
    this.paddle = paddle;

    const splash = orb(this.landedTint as unknown as N<'vec3'>, this.landedTint as unknown as N<'vec3'>, this.landedAlpha);
    const sp = uv().sub(0.5).mul(ORB);
    const ticks = Array.from({ length: 8 }, (_, k) => {
      const a = (k + 0.5) * Math.PI / 4, c = Math.cos(a), s = Math.sin(a);
      return sdSegment(sp, vec2(c * 0.36, s * 0.36), vec2(c * 0.5, s * 0.5), 0.028);
    }).reduce((d, e) => min(d, e));
    splash.colorNode = (splash.colorNode as N<'vec3'>).add(rgb(MISS).mul(coverage(ticks).mul(this.shatter).mul(1.4)));
    const landed = new THREE.Mesh(flat(ORB, ORB), splash);
    landed.renderOrder = 3;
    landed.visible = false;
    this.landed = landed;

    const ball = new THREE.Mesh(flat(ORB, ORB), orb(rgb(BALL_CORE), rgb(BALL_GLOW), float(1)));
    ball.renderOrder = 3;
    this.ball = ball;

    this.ballLight = light.emitter(new THREE.CircleGeometry(0.17, 24).rotateX(-Math.PI / 2), rgb(BALL_GLOW).mul(9));
    this.paddleLight = light.emitter(flat(0.62, 0.12), gold.mul(this.radiance));
    this.landedLight = light.emitter(new THREE.CircleGeometry(0.15, 24).rotateX(-Math.PI / 2), this.landedEmit as unknown as N<'vec3'>);
    this.scene.add(paddle, landed, ball, this.ballLight, this.paddleLight, this.landedLight);
    pool.group.traverse((object) => { object.renderOrder = object instanceof THREE.Mesh ? 1 : 5; });
  }

  private board(light: Light2D): THREE.Mesh {
    const { w, h } = this.grid, width = w + 2 * BEZEL, depth = h + GUTTER + 2 * BEZEL;
    const q = vec2(uv().x.mul(width).sub(BEZEL), uv().y.oneMinus().mul(depth).sub(BEZEL));
    const half = vec2(w / 2, (h + GUTTER) / 2), centred = q.sub(half);
    const window = sdBox(centred, half.add(0.1), 0.18);
    const cell = floor(q), local = fract(q).sub(0.5);
    const inside = step(0, q.x).mul(step(q.x, w)).mul(step(0, q.y)).mul(step(q.y, h));
    const socket = sdCircle(local, LENS), lens = coverage(socket).mul(inside);
    const drain = coverage(sdBox(q.sub(vec2(w / 2, h + GUTTER / 2)), vec2(w / 2 - 0.08, GUTTER / 2 - 0.14), 0.22));
    const rail = coverage(abs(q.y.sub(h - 0.07)).sub(0.014)).mul(step(0.08, q.x)).mul(step(q.x, w - 0.08));
    const bezel = mix(rgb(0x4a4f5c), rgb(0x23262e), smoothstep(-BEZEL, h + GUTTER + BEZEL, q.y));
    let albedo = mix(bezel, rgb(0x0e1117), coverage(window));
    albedo = mix(albedo, rgb(0x6b7385), coverage(abs(window).sub(0.02)).mul(0.8));
    albedo = mix(albedo, mix(rgb(0x5a6376), rgb(0x20242d), smoothstep(-LENS, LENS, local.y)), lens);
    albedo = mix(albedo, rgb(0x030405), drain);
    albedo = mix(albedo, rgb(0x9a8350), rail);

    const column = select(abs(cell.x.sub(this.ballCol)).lessThan(0.5), float(1), float(0)).mul(inside);
    const dx = abs(q.x.sub(this.ballCol.add(0.5))), dy = this.ballRow.add(0.5).sub(q.y), taper = max(dy.div(TAIL).oneMinus().mul(BALL * 0.8), 1e-3);
    const tail = smoothstep(taper.mul(0.15), taper, dx).oneMinus().mul(step(0, dy)).mul(step(dy, this.ballRow)).mul(exp(dy.mul(-1.3))).mul(inside).mul(0.55);
    const ring = column.mul(select(cell.y.equal(h - 1), float(1), float(0))).mul(coverage(abs(sdCircle(local, 0.43)).sub(0.022))).mul(this.ballRow.div(h - 1).mul(0.7).add(0.3));
    const glow = rgb(BALL_GLOW).mul(tail).add(mix(rgb(BALL_GLOW), rgb(this.accent), this.aligned).mul(ring.mul(1.3)));

    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = light.lit(albedo).add(glow).add(dither);
    material.opacityNode = coverage(sdBox(centred, half.add(BEZEL), 0.45));
    const mesh = new THREE.Mesh(flat(width, depth), material);
    mesh.position.z = GUTTER / 2;
    mesh.renderOrder = 0;
    return mesh;
  }

  private tube(light: Light2D): THREE.Mesh {
    const { w, h } = this.grid, length = h + 0.36;
    this.outcomes = uniformArray(new Array<number>(h).fill(0), 'float' as const);
    const p = uv().sub(0.5).mul(vec2(TUBE, length)), qy = uv().y.oneMinus().mul(length).sub(0.18);
    const slot = floor(qy), local = vec2(p.x, fract(qy).sub(0.5));
    const outcome = this.outcomes.element(int(clamp(slot.negate().add(h - 1), 0, h - 1)));
    const shell = sdBox(p, vec2(TUBE / 2, length / 2), TUBE / 2);
    const lamp = coverage(sdCircle(local, 0.19)).mul(step(0, qy)).mul(step(qy, h));
    const albedo = mix(mix(rgb(0x4a4f5c), rgb(0x0e1117), coverage(shell.add(0.05))), mix(rgb(0x5a6376), rgb(0x20242d), smoothstep(-0.19, 0.19, local.y.negate())), lamp);
    const lit = mix(rgb(MISS).mul(0.9), rgb(this.accent).mul(1.3), step(0, outcome)).mul(abs(outcome)).mul(lamp);
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = light.lit(albedo).add(lit).add(dither);
    material.opacityNode = coverage(shell);
    const mesh = new THREE.Mesh(flat(TUBE, length), material);
    mesh.position.set(w / 2 + BEZEL + TUBE_GAP + TUBE / 2, 0, 0);
    mesh.renderOrder = 0;
    return mesh;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const ball = this.ball, paddle = this.paddle, landed = this.landed, ballLight = this.ballLight, paddleLight = this.paddleLight, landedLight = this.landedLight;
    if (!ball || !paddle || !landed || !ballLight || !paddleLight || !landedLight) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    const { h } = this.grid, state = frame.state, at = state.step;
    const now = frame.frameAt(at), next = frame.frameAt(at + 1), live = next.step === at + 1;
    const t = live ? clamp01(frame.position - at) : 0;
    const col = now.words[0]!, row = now.words[1]!, fall = row + ((live ? (next.done ? h - 1 : next.words[1]!) : row) - row) * t;
    const from = now.words[2]!, to = live ? next.words[2]! : from, x = from + (to - from) * (1 - (1 - t) ** 3);
    const action = live ? (Math.trunc(next.actions[0]!) >>> 0) % 3 : -1, blocked = action !== 1 && to === from;

    cellToWorldVec(col, fall, this.grid, ball.position);
    ballLight.position.copy(ball.position);
    cellToWorldVec(x, h - 1, this.grid, paddle.position);
    paddleLight.position.set(paddle.position.x, 0, paddle.position.z + 0.3);
    this.ballCol.value = col;
    this.ballRow.value = fall;
    this.aligned.value = clamp01(1 - Math.abs(x - col));
    const press = (blocked ? 0.4 : 1) * (1 - 0.5 * t);
    this.left.value = action === 0 ? press : 0;
    this.right.value = action === 2 ? press : 0;

    const before = now.done && at > 0 ? frame.frameAt(at - 1) : null;
    const dropped = before && before.step === at - 1 ? before.words[0]! : -1;
    const caught = dropped >= 0 && dropped === from;
    this.flash.value = caught ? (1 - t) ** 2 : 0;
    this.radiance.value = 3 + 3 * this.flash.value;
    this.shatter.value = dropped >= 0 && !caught ? 1 - t : 0;
    landed.visible = dropped >= 0 && !(caught && t >= 0.6);
    landedLight.visible = dropped >= 0 && !caught;
    if (caught) {
      cellToWorldVec(x, h - 1, this.grid, landed.position);
      landed.scale.setScalar(0.85 * (1 - smoothstep01(t / 0.6)));
      this.landedTint.value.set(CAUGHT);
      this.landedAlpha.value = 1;
    } else if (dropped >= 0) {
      cellToWorldVec(dropped, h - 1 + (0.5 + GUTTER / 2) * t * t, this.grid, landed.position);
      landed.scale.setScalar(1 - 0.25 * t);
      this.landedTint.value.set(BALL_GLOW).lerp(MISS_COLOR, Math.min(1, 0.65 + t));
      this.landedAlpha.value = 0.8 * (1 - t * t);
      landedLight.position.copy(landed.position);
      this.landedEmit.value.set(MISS).multiplyScalar(4 * (1 - t));
    }

    for (const event of frame.events) {
      if (event.type === 'goal') {
        cellToWorldVec(from, h - 1, this.grid, this.scratch);
        pool.burst(this.scratch, this.accent, 0.5, time);
        pool.pulse.fire(0.3);
      }
      if (event.type === 'harm' && dropped >= 0) {
        cellToWorldVec(dropped, h - 0.5 + GUTTER / 2, this.grid, this.scratch);
        pool.burst(this.scratch, MISS, 0.35, time);
      }
    }

    this.count(frame.eventLog, at);
    this.history(frame.eventLog, at);
    this.ledger.paddle = from;
    this.ledger.col = col;
    this.ledger.row = row;
    this.ledger.action = action < 0 ? '—' : `${ACTIONS[action]}${blocked ? ' · at the wall' : ''}`;
    this.ledger.reward = state.reward;
  }

  private history(log: EventStream, at: number): void {
    if (this.drawn.step === at && this.drawn.count === log.count) return;
    this.drawn.step = at;
    this.drawn.count = log.count;
    const values = this.outcomes.array as number[];
    values.fill(0);
    let k = 0;
    for (let b = Math.floor(at / log.blockFrames); b >= 0 && k < values.length; b--) {
      const block = log.blocks[b];
      if (!block) break;
      for (let i = block.steps.length - 1; i >= 0 && k < values.length; i--) {
        if (block.steps[i]! > at) continue;
        if (block.kinds[i] === GOAL) values[k++] = 1;
        else if (block.kinds[i] === HARM) values[k++] = -1;
      }
    }
  }

  private count(log: EventStream, at: number): void {
    const tally = this.tally;
    if (tally.log !== log || tally.count !== log.count || at < tally.through) Object.assign(tally, { log, count: log.count, through: -1, caught: 0, missed: 0 });
    if (at === tally.through) return;
    tally.caught += countEvents(log, 'goal', tally.through, at);
    tally.missed += countEvents(log, 'harm', tally.through, at);
    tally.through = at;
  }
}

export default (() => new CatchScene()) satisfies SceneFactory;
