import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { DebugCurve, GroundTruthOptions } from '@kit/debug';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { SLIME_AGENT, SLIME_BALL, SLIME_FRAME, SLIME_GAME, SLIME_HOLD, SLIME_LIVES, SLIME_MEMBER, SLIME_MEMORY, SLIME_POINT, SLIME_RIVAL, SLIME_RIVAL_LIVES, SLIME_VX, SLIME_VY, SLIME_X, SLIME_Y } from '@sim/envs/slime';
import { SIM_EVENT_TYPES, type EnvFrame, type EventStream, type SimEvent } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { BOARD, CEILING, EYE, LAMPS, LENS, PALETTE, SEVEN, ball, chevrons, fence, flat, hall, halo, lay, pop, puff, ribbon, ring, scoreboard, shadow, slime, sprite, strip, type Court, type Ribbon } from './art';

const SLOTS = [SLIME_AGENT, SLIME_RIVAL] as const;
const FACING = [-1, 1] as const;
const MOVES = [0, -1, 1, 0, -1, 1] as const;
const ACTIONS = ['stand', '← run', '→ run', '↑ jump', '← run · jump', '→ run · jump'] as const;
const TRAIL = 14;
const SUB = 3;
const POPS = 6;
const PUFFS = 4;
const PUFF_LIFE = 0.5;
const POP_LIFE = 0.45;
const FLASH_LIFE = 0.9;
const RESULT_LIFE = 2.6;
const TOUCH = 2e-3;
const LAMP = 36;
const GLOW = 0.55;
const BALL_GLOW = 3.2;
const CHEVRON_DROP = 2.3;
const TALLY_SECONDS = 0.5;
const GOLD = PALETTE.gold;

const word = (frame: EnvFrame, at: number): number => wordFloat(frame.words[at]!);
const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);
const signed = (value: number): string => formatSigned(value).replace('-', '−');
const fade = (age: number): number => (age >= 0 && age < 1 ? (1 - age) ** 2 : 0);

function recent(stream: EventStream, kind: SimEvent['type'], step: number, span: number): number | null {
  const code = SIM_EVENT_TYPES.indexOf(kind);
  for (let b = Math.floor(step / stream.blockFrames); b >= 0 && b >= Math.floor((step - span) / stream.blockFrames); b--) {
    const block = stream.blocks[b];
    if (!block) continue;
    for (let i = block.steps.length - 1; i >= 0; i--) {
      const at = block.steps[i]!;
      if (at <= step && at > step - span && block.kinds[i] === code) return at;
    }
  }
  return null;
}

function clock(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

type Scalar = THREE.UniformNode<'float', number>;
type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
interface Pop { mesh: THREE.Mesh; age: Scalar; tint: Tint }
interface Puff { mesh: THREE.Mesh; age: Scalar }
interface Flash { light: THREE.Mesh; tint: Tint; strength: Scalar }

class SlimeVolleyScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  controlHz = 30;
  protected readonly stageName = 'slimeVolley';

  private court: Court = { half: 1, ground: 0, top: 1, stub: 0.5, radius: 1, ball: 0.5 };
  private reach = 1;
  private nudge = 0;
  private postReach = 1;
  private hold = 1;
  private frames = 1;
  private lives = 1;
  private inner = 0;
  private outer = 1;
  private serveY = 1;
  private league = false;
  private positions = false;
  private accent = 0xffffff;
  private readonly rig = new THREE.Group();
  private readonly bodies: THREE.Group[] = [];
  private readonly mirrors: THREE.Mesh[] = [];
  private readonly shades: THREE.Mesh[] = [];
  private readonly looks = [uniform(new THREE.Vector2()), uniform(new THREE.Vector2())] as const;
  private readonly flashes = [uniform(0), uniform(0)] as const;
  private readonly lids = [uniform(0), uniform(0)] as const;
  private readonly clock = uniform(0);
  private readonly spin = uniform(0);
  private readonly countdown = uniform(1);
  private readonly ballGlow = uniform(1);
  private readonly dir = uniform(0);
  private readonly jump = uniform(0);
  private readonly flow = uniform(0);
  private readonly lifeCounts = [uniform(0), uniform(0)] as const;
  private readonly lifePops = [uniform(0), uniform(0)] as const;
  private readonly digits = [uniform(0), uniform(0), uniform(0)] as const;
  private readonly member = uniform(0);
  private readonly result = uniform(new THREE.Vector3());
  private readonly resultGlow = uniform(0);
  private readonly pops: Pop[] = [];
  private readonly puffs: Puff[] = [];
  private readonly sides: Flash[] = [];
  private ballParts: THREE.Object3D[] = [];
  private ballMirror: THREE.Mesh | null = null;
  private ballShade: THREE.Mesh | null = null;
  private serve: THREE.Mesh | null = null;
  private input: THREE.Mesh | null = null;
  private trail: Ribbon | null = null;
  private readonly path = new Float32Array(TRAIL * SUB * 2);
  private readonly xs = new Float64Array(TRAIL + 2);
  private readonly ys = new Float64Array(TRAIL + 2);
  private readonly shown = { bx: 0, by: 0, x: [0, 0], y: [0, 0] };
  private tints: readonly THREE.Color[] = [];
  private readonly spot = new THREE.Vector3();
  private readonly tally = { at: -Infinity, won: 0, lost: 0 };
  private wall = 0;
  private readonly ledger = { agent: 0, rival: 0, member: 0, action: '—', ball: '—', reward: 0, game: 0, frame: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.controlHz = constant(env, 'SLIME_RATE');
    this.court = {
      half: constant(env, 'SLIME_HALF'),
      ground: constant(env, 'SLIME_GROUND'),
      top: constant(env, 'SLIME_TOP'),
      stub: constant(env, 'SLIME_STUB'),
      radius: constant(env, 'SLIME_RADIUS'),
      ball: constant(env, 'SLIME_BALL'),
    };
    this.reach = constant(env, 'SLIME_REACH');
    this.nudge = constant(env, 'SLIME_NUDGE');
    this.postReach = constant(env, 'SLIME_POST');
    this.hold = constant(env, 'SLIME_HOLD');
    this.frames = constant(env, 'SLIME_FRAMES');
    this.lives = constant(env, 'SLIME_LIVES');
    this.inner = constant(env, 'SLIME_INNER');
    this.outer = constant(env, 'SLIME_OUTER');
    this.serveY = this.court.ground + constant(env, 'SLIME_SERVE_HEIGHT');
    this.league = parameter(env, 'league') !== 0;
    this.positions = parameter(env, 'positions') !== 0;
    if (this.frames / this.controlHz >= 600) throw new Error(`${env.envId} lasts ${this.frames / this.controlHz} s, beyond the scoreboard's single minute digit`);
    const { half } = this.court;
    return new THREE.Box2(new THREE.Vector2(-half - 1.6, -24.6), new THREE.Vector2(half + 1.6, 3.4));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa6b4d6).multiplyScalar(0.78);
  }

  protected override poolOptions() {
    return { burstSize: 40 };
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, red = 0xff6b57, slate = 0x8fa6c4, mint = 0x86f0b4;
    return {
      debug: (frame, kit) => {
        const { half, ground, top, stub, radius: R, ball: r } = this.court, shown = this.shown, step = frame.state.step;
        const floor = kit.curve('court', 4, slate);
        for (const [x, y] of [[-half, CEILING], [-half, ground], [half, ground], [half, CEILING]] as const) floor.point(x, 0.05, -y);
        this.capsule(kit.curve('fence', 42, slate), ground, top, stub);
        for (const [name, from, to] of [['agentRange', this.inner, this.outer], ['rivalRange', -this.outer, -this.inner]] as const) {
          const range = kit.curve(name, 6, name === 'agentRange' ? this.accent : PALETTE.rival);
          for (const [x, y] of [[from, ground - 0.2], [from, ground - 0.45], [from, ground - 0.325], [to, ground - 0.325], [to, ground - 0.2], [to, ground - 0.45]] as const) range.point(x, 0.05, -y);
        }
        const marks = kit.marks('points', 8);
        marks.set(0, 0.05, -this.serveY, 0.12, mint);
        for (let i = 0; i < 2; i++) {
          const x = shown.x[i]!, y = shown.y[i]!, tint = i === 0 ? this.accent : PALETTE.rival;
          this.dome(kit.curve(i === 0 ? 'agentDome' : 'rivalDome', 36, tint), x, y, R);
          this.circle(kit.curve(i === 0 ? 'agentReach' : 'rivalReach', 49, slate), x, y, this.reach);
          marks.set(x, 0.05, -y, 0.1, tint);
        }
        this.circle(kit.curve('ball', 33, gold), shown.bx, shown.by, r);
        marks.set(shown.bx, 0.05, -shown.by, 0.08, gold);
        const now = frame.frameAt(step), next = frame.frameAt(step + 1);
        const velocity = kit.curve('velocity', 2, cyan), vx = word(now, SLIME_BALL + SLIME_VX), vy = word(now, SLIME_BALL + SLIME_VY);
        velocity.point(shown.bx, 0.05, -shown.by);
        velocity.point(shown.bx + vx * 0.2, 0.05, -(shown.by + vy * 0.2));
        const trace = kit.curve('trace', 91, cyan);
        for (let s = step; s >= Math.max(0, step - 90); s--) {
          const f = frame.frameAt(s);
          if (f.step !== s) break;
          trace.point(word(f, SLIME_BALL), 0.05, -word(f, SLIME_BALL + 1));
          if (f.words[SLIME_HOLD] !== 0) break;
        }
        const touched = kit.marks('touches', 4), mask = now.step === step && now.words[SLIME_HOLD] === 0 ? this.contact(now) : 0;
        for (let bit = 0; bit < 4; bit++) if (mask & (1 << bit)) {
          const [x, y] = this.touchPoint(now, bit);
          touched.set(x, 0.06, -y, 0.14, red);
        }
        const memory = kit.curve('memory', 12, PALETTE.rival), mx = shown.x[1]!, my = shown.y[1]! + R + 1.2;
        for (let k = 0; k < 4; k++) {
          const value = word(now, SLIME_MEMORY + k), x = mx + (k - 1.5) * 0.35;
          memory.point(x, 0.05, -my);
          memory.point(x, 0.05, -(my + value * 0.8));
          memory.point(x, 0.05, -my);
        }
        const act = next.step === step + 1 ? Math.trunc(next.actions[0]!) : -1, move = act >= 0 && act < 6 ? MOVES[act]! : 0, hop = act >= 3;
        const arrow = kit.curve('action', 10, cyan), ax = shown.x[0]!, ay = shown.y[0]! + R + 0.6;
        if (move !== 0) for (const [x, y] of [[ax, ay], [ax + move * 1.2, ay], [ax + move * 0.95, ay + 0.2], [ax + move * 1.2, ay], [ax + move * 0.95, ay - 0.2]] as const) arrow.point(x, 0.05, -y);
        if (hop) for (const [x, y] of [[ax, ay], [ax, ay + 1.1], [ax - 0.2, ay + 0.85], [ax, ay + 1.1], [ax + 0.2, ay + 0.85]] as const) arrow.point(x, 0.05, -y);
      },
    };
  }

  private capsule(curve: DebugCurve, ground: number, top: number, stub: number): void {
    curve.point(stub, 0.05, -ground);
    curve.point(stub, 0.05, -top);
    for (let i = 1; i < 40; i++) {
      const a = Math.PI * i / 40;
      curve.point(stub * Math.cos(a), 0.05, -(top + stub * Math.sin(a)));
    }
    curve.point(-stub, 0.05, -ground);
  }

  private circle(curve: DebugCurve, x: number, y: number, radius: number): void {
    const count = curve.capacity - 1;
    for (let i = 0; i <= count; i++) {
      const a = 2 * Math.PI * i / count;
      curve.point(x + radius * Math.cos(a), 0.05, -(y + radius * Math.sin(a)));
    }
  }

  private dome(curve: DebugCurve, x: number, y: number, radius: number): void {
    const count = curve.capacity - 2;
    for (let i = 0; i <= count; i++) {
      const a = Math.PI * i / count;
      curve.point(x + radius * Math.cos(a), 0.05, -(y + radius * Math.sin(a)));
    }
    curve.point(x + radius, 0.05, -y);
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const accent = this.accentOf(ctx), lit = light.lit, court = this.court, { half, ground, top, stub, radius: R, ball: r } = court;
    this.accent = accent;
    this.tints = [accent, PALETTE.rival, PALETTE.post, PALETTE.padEdge, GOLD, TOKENS.failureBlue, 0xdcf4ff].map((tint) => new THREE.Color(tint));
    this.scene.background = new THREE.Color(PALETTE.void);
    this.scene.add(this.rig);
    this.rig.add(sprite(-70, -20, 70, 45, 0, hall(lit, court, accent)));
    for (const x of [-LAMPS[1], -LAMPS[0], LAMPS[0], LAMPS[1]]) {
      const lamp = light.emitter(flat(LENS.width, 0.36), rgb(PALETTE.lens).mul(LAMP));
      lamp.position.set(x, 0, -(LENS.y + 0.11));
      this.rig.add(lamp);
    }

    for (const sign of [-1, 1]) {
      const tint = uniform(new THREE.Vector3()), strength = uniform(0);
      const glow = sprite(-half / 2, -1.6, half / 2, 1.6, 1, strip(half, tint, strength), true);
      glow.position.set(sign * half / 2, 0, -ground);
      const flare = light.emitter(flat(half - 1, 0.3), tint.mul(strength).mul(4));
      flare.position.set(sign * half / 2, 0, -(ground - 0.15));
      this.rig.add(glow, flare);
      this.sides.push({ light: flare, tint, strength });
    }

    const colors = [accent, PALETTE.rival] as const;
    for (const i of [0, 1] as const) {
      const u = { look: this.looks[i], flash: this.flashes[i], lid: this.lids[i], clock: this.clock };
      const shade = sprite(-1, -1, 1, 1, 1.5, shadow(uniform(0.72)));
      const mirror = sprite(-R - 0.2, -R - 0.2, R + 0.2, 0.2, 2, slime(court, colors[i], FACING[i], u, true));
      const body = new THREE.Group();
      body.add(
        sprite(-R - 0.2, -0.2, R + 0.2, R + 0.2, 6, slime(court, colors[i], FACING[i], u, false)),
        light.emitter(new THREE.CircleGeometry(R, 40, 0, Math.PI).rotateX(-Math.PI / 2), rgb(colors[i]).mul(this.flashes[i].mul(2.5).add(GLOW))),
      );
      this.shades.push(shade);
      this.mirrors.push(mirror);
      this.bodies.push(body);
      this.rig.add(shade, mirror, body);
    }

    this.rig.add(sprite(-stub - 0.4, ground - 0.4, stub + 0.4, top + stub + 0.2, 3, fence(lit, court, accent)));
    const pole = light.occluder(flat(2 * stub, top - ground));
    pole.position.set(0, 0, -(ground + top) / 2);
    const cap = light.occluder(new THREE.CircleGeometry(stub, 24).rotateX(-Math.PI / 2));
    cap.position.set(0, 0, -top);
    this.rig.add(pole, cap);

    const trail = ribbon(TRAIL * SUB, PALETTE.glow);
    trail.mesh.renderOrder = 4;
    this.trail = trail;
    this.rig.add(trail.mesh);

    this.ballShade = sprite(-1, -1, 1, 1, 1.5, shadow(uniform(0.6)));
    this.ballMirror = sprite(-r - 0.1, -r - 0.1, r + 0.1, r + 0.1, 2, ball(court, this.spin, true));
    this.ballParts = [
      sprite(-2.4, -2.4, 2.4, 2.4, 7, halo(PALETTE.glow, r, 2.4, this.ballGlow), true),
      sprite(-r - 0.1, -r - 0.1, r + 0.1, r + 0.1, 8, ball(court, this.spin, false)),
      light.emitter(new THREE.CircleGeometry(r * 0.95, 24).rotateX(-Math.PI / 2), rgb(PALETTE.glow).mul(this.ballGlow.mul(BALL_GLOW))),
    ];
    this.serve = sprite(-1.2, -1.2, 1.2, 1.2, 9, ring(this.countdown), true);
    this.rig.add(this.ballShade, this.ballMirror, ...this.ballParts, this.serve);

    for (let i = 0; i < POPS; i++) {
      const age = uniform(1), tint = uniform(new THREE.Vector3(1, 1, 1));
      const mesh = sprite(-1.4, -1.4, 1.4, 1.4, 9, pop(age, tint), true);
      mesh.visible = false;
      this.pops.push({ mesh, age, tint });
      this.rig.add(mesh);
    }

    for (let i = 0; i < PUFFS; i++) {
      const age = uniform(1), mesh = sprite(-2.6, -0.3, 2.6, 0.9, 2.2, puff(age, R));
      mesh.visible = false;
      this.puffs.push({ mesh, age });
      this.rig.add(mesh);
    }

    this.input = sprite(-0.8, -0.4, 0.8, 0.85, 2.5, chevrons(accent, this.dir, this.jump, this.flow), true);
    this.rig.add(this.input);

    const centre = (BOARD.bottom + BOARD.top) / 2;
    const board = sprite(-BOARD.half - 0.3, BOARD.bottom - centre - 0.3, BOARD.half + 0.3, CEILING - centre + 0.1, 10, scoreboard(lit, accent, {
      lives: this.lifeCounts, pops: this.lifePops, digits: this.digits, member: this.member, result: this.result, glow: this.resultGlow,
    }, this.league, centre, this.lives));
    board.position.set(0, 0, -centre);
    this.rig.add(board);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    this.legend = this.describe(accent);
  }

  private describe(accent: number): LegendSpec {
    const l = this.ledger, tally = this.tally;
    return {
      title: 'SLIME VOLLEY',
      rows: [
        { key: 'lives', label: 'lives', tint: accent, value: () => `agent ${l.agent} · rival ${l.rival}` },
        { key: 'rival', label: 'rival', tint: PALETTE.rival, value: () => (this.league ? `league member ${l.member + 1}` : 'evolved net') },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => l.action },
        { key: 'ball', label: 'ball', tint: PALETTE.ball, value: () => l.ball },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(l.reward) },
        { key: 'points', label: 'points', tint: GOLD, value: () => `${formatCount(tally.won)} won · ${formatCount(tally.lost)} lost` },
      ],
      note: () => `game ${formatCount(l.game + 1)} · ${clock(Math.ceil(Math.max(0, this.frames - l.frame) / this.controlHz))} left${this.positions ? ' · sees positions only' : ''}`,
      bar: () => l.frame / this.frames,
    };
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const trail = this.trail, serve = this.serve, input = this.input;
    if (!trail || !serve || !input) return;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    this.wall += frame.dtReal;
    const hz = frame.controlHz, at = frame.position, step = frame.state.step, pose = frame.state.pose;
    const now = frame.frameAt(step), next = frame.frameAt(step + 1), ahead = next.step === step + 1;
    const cut = ahead && (next.reset || next.words[SLIME_POINT] !== now.words[SLIME_POINT]);
    const shown = this.shown, { ground, radius: R, ball: r } = this.court;
    shown.bx = cut ? word(now, SLIME_BALL + SLIME_X) : pose[SLIME_BALL + SLIME_X]!;
    shown.by = cut ? word(now, SLIME_BALL + SLIME_Y) : pose[SLIME_BALL + SLIME_Y]!;
    this.clock.value = time;

    for (const i of [0, 1] as const) {
      const x = pose[SLOTS[i] + SLIME_X]!, y = pose[SLOTS[i] + SLIME_Y]!, lift = y - ground;
      shown.x[i] = x;
      shown.y[i] = y;
      this.bodies[i]!.position.set(x, 0, -y);
      this.mirrors[i]!.position.set(x, 0, -(2 * ground - y));
      const spread = 1 / (1 + 0.14 * lift), shade = this.shades[i]!;
      shade.position.set(x, 0, -(ground - 0.1));
      shade.scale.set(R * 1.1 * spread, 1, 0.26 * spread);
      const ex = x + FACING[i] * EYE.x * R, ey = y + EYE.y * R, dx = shown.bx - ex, dy = shown.by - ey, d = Math.hypot(dx, dy) || 1, k = (EYE.radius - EYE.pupil) * R - 0.02;
      this.looks[i].value.set(dx / d * k, dy / d * k);
      this.flashes[i].value = 0;
      this.lids[i].value = 0;
    }

    const bx = shown.bx, by = shown.by, height = by - r - ground;
    for (const part of this.ballParts) part.position.set(bx, 0, -by);
    this.ballMirror?.position.set(bx, 0, -(2 * ground - by));
    if (this.ballShade) {
      const spread = 1 / (1 + 0.12 * height);
      this.ballShade.position.set(bx, 0, -(ground - 0.1));
      this.ballShade.scale.set(0.8 * spread, 1, 0.2 * spread);
    }
    this.spin.value = bx / r;
    const hold = now.words[SLIME_HOLD]!;
    serve.visible = hold > 0;
    serve.position.set(bx, 0, -by);
    this.countdown.value = clamp((hold - (at - step)) / this.hold, 0, 1);
    this.ballGlow.value = 1;

    this.stroke(frame, step, at, ahead && !cut ? next : null);
    this.touches(frame, step, at, hz);
    this.dust(frame, step, at, hz);
    this.points(frame, step, at, hz);
    this.scoreboard(frame, now, step, at, hz);

    const act = ahead ? Math.trunc(next.actions[0]!) : Math.trunc(now.actions[0]!), known = act >= 0 && act < 6;
    this.dir.value = known ? MOVES[act]! : 0;
    this.jump.value = known && act >= 3 ? 1 : 0;
    this.flow.value = time * 2.2;
    input.position.set(shown.x[0]!, 0, -(ground - CHEVRON_DROP));

    for (const event of frame.events) this.celebrate(event, frame, step, pool, time);

    const l = this.ledger;
    l.agent = now.words[SLIME_LIVES]!;
    l.rival = now.words[SLIME_RIVAL_LIVES]!;
    l.member = now.words[SLIME_MEMBER]!;
    l.game = now.words[SLIME_GAME]!;
    l.frame = now.words[SLIME_FRAME]!;
    l.reward = frame.state.reward;
    l.action = known && step > 0 ? ACTIONS[act]! : '—';
    const vx = word(now, SLIME_BALL + SLIME_VX), vy = word(now, SLIME_BALL + SLIME_VY);
    l.ball = hold > 0 ? `serve in ${(hold / hz).toFixed(1)} s` : `${Math.hypot(vx, vy).toFixed(1)} u/s · ${word(now, SLIME_BALL) < 0 ? 'rival side' : 'agent side'}`;
    if (frame.scrubbed || this.wall - this.tally.at > TALLY_SECONDS) {
      this.tally.won = countEvents(frame.eventLog, 'goal', -1, step);
      this.tally.lost = countEvents(frame.eventLog, 'harm', -1, step);
      this.tally.at = this.wall;
    }
  }

  private stroke(frame: FrameUpdate, step: number, at: number, head: EnvFrame | null): void {
    const trail = this.trail!, { xs, ys, path } = this, base = step - TRAIL;
    let start = step;
    for (let k = step; k >= Math.max(0, base); k--) {
      const f = frame.frameAt(k);
      if (f.step !== k) break;
      xs[k - base] = word(f, SLIME_BALL + SLIME_X);
      ys[k - base] = word(f, SLIME_BALL + SLIME_Y);
      start = k;
      if (f.words[SLIME_HOLD] !== 0) break;
    }
    if (head) {
      xs[step + 1 - base] = word(head, SLIME_BALL + SLIME_X);
      ys[step + 1 - base] = word(head, SLIME_BALL + SLIME_Y);
    }
    let used = 0;
    for (let j = 0; j < trail.points; j++) {
      const tau = at - j / SUB;
      if (tau < start) break;
      const k = Math.min(step, Math.floor(tau)), u = k === step && !head ? 0 : tau - k, i = k - base;
      path[2 * j] = xs[i]! + (xs[i + 1]! - xs[i]!) * u;
      path[2 * j + 1] = ys[i]! + (ys[i + 1]! - ys[i]!) * u;
      used = j + 1;
    }
    if (used === 0) {
      path[0] = this.shown.bx;
      path[1] = this.shown.by;
    }
    lay(trail, path, Math.max(1, used), this.court.ball * 0.42, 0.02);
  }

  private contact(f: EnvFrame): number {
    const { half, ground, top, radius: R, ball: r } = this.court, bx = word(f, SLIME_BALL + SLIME_X), by = word(f, SLIME_BALL + SLIME_Y);
    let mask = 0;
    for (const i of [0, 1] as const) {
      const dx = bx - word(f, SLOTS[i] + SLIME_X), dy = by - word(f, SLOTS[i] + SLIME_Y);
      const gap = dy >= 0 ? Math.hypot(dx, dy) - (this.reach + this.nudge) : Math.hypot(dx - clamp(dx, -R, R), dy) - (r + this.nudge);
      if (Math.abs(gap) < TOUCH) mask |= 1 << i;
    }
    if (Math.abs(Math.hypot(bx, by - clamp(by, ground, top)) - (this.postReach + this.nudge)) < TOUCH) mask |= 4;
    if (Math.abs(Math.abs(bx) - (half - r - this.nudge)) < TOUCH) mask |= 8;
    return mask;
  }

  private touchPoint(f: EnvFrame, bit: number): readonly [number, number] {
    const { half, ground, top, stub, radius: R } = this.court, bx = word(f, SLIME_BALL + SLIME_X), by = word(f, SLIME_BALL + SLIME_Y);
    if (bit < 2) {
      const x = word(f, SLOTS[bit]! + SLIME_X), y = word(f, SLOTS[bit]! + SLIME_Y), dx = bx - x, dy = by - y;
      if (dy < 0) return [x + clamp(dx, -R, R), y];
      const d = Math.hypot(dx, dy) || 1;
      return [x + dx / d * R, y + dy / d * R];
    }
    if (bit === 2) {
      const cy = clamp(by, ground, top), d = Math.hypot(bx, by - cy) || 1;
      return [bx / d * stub, cy + (by - cy) / d * stub];
    }
    return [Math.sign(bx) * half, by];
  }

  private touches(frame: FrameUpdate, step: number, at: number, hz: number): void {
    const span = Math.ceil(POP_LIFE * hz);
    let used = 0;
    for (let k = 0; k <= span && used < POPS; k++) {
      const s = step - k;
      if (s < 1) break;
      const f = frame.frameAt(s);
      if (f.step !== s) break;
      if (f.reset) break;
      if (f.words[SLIME_HOLD] !== 0) continue;
      const mask = this.contact(f);
      if (!mask) continue;
      const age = (at - s) / hz / POP_LIFE;
      for (let bit = 0; bit < 4 && used < POPS; bit++) {
        if (!(mask & (1 << bit))) continue;
        const slot = this.pops[used++]!, [x, y] = this.touchPoint(f, bit), tint = this.tints[bit]!;
        slot.mesh.visible = true;
        slot.mesh.position.set(x, 0, -y);
        slot.mesh.scale.setScalar(bit < 2 ? 1 : 0.6);
        slot.age.value = age;
        slot.tint.value.set(tint.r, tint.g, tint.b);
        if (bit < 2) {
          const flash = this.flashes[bit as 0 | 1];
          flash.value = Math.max(flash.value, fade(age * 1.6));
          this.ballGlow.value = Math.max(this.ballGlow.value, 1 + 1.4 * fade(age * 2));
        }
      }
    }
    for (let i = used; i < POPS; i++) this.pops[i]!.mesh.visible = false;
  }

  private dust(frame: FrameUpdate, step: number, at: number, hz: number): void {
    const ground = this.court.ground, span = Math.ceil(PUFF_LIFE * hz);
    let used = 0, later = frame.frameAt(step);
    for (let k = 0; k <= span && used < PUFFS && later.step === step - k; k++) {
      const s = step - k, earlier = frame.frameAt(s - 1);
      if (s < 1 || later.reset || earlier.step !== s - 1) break;
      for (const i of [0, 1] as const) {
        const y = word(later, SLOTS[i] + SLIME_Y), before = word(earlier, SLOTS[i] + SLIME_Y);
        if (y !== ground || before === ground || used >= PUFFS) continue;
        const slot = this.puffs[used++]!;
        slot.mesh.visible = true;
        slot.mesh.position.set(word(later, SLOTS[i] + SLIME_X), 0, -ground);
        slot.age.value = (at - s) / hz / PUFF_LIFE;
      }
      later = earlier;
    }
    for (let i = used; i < PUFFS; i++) this.puffs[i]!.mesh.visible = false;
  }

  private points(frame: FrameUpdate, step: number, at: number, hz: number): void {
    const span = Math.ceil(FLASH_LIFE * hz), scored = recent(frame.eventLog, 'goal', step, span), conceded = recent(frame.eventLog, 'harm', step, span);
    for (const side of this.sides) {
      side.strength.value = 0;
      side.light.visible = false;
    }
    this.lifePops[0].value = this.lifePops[1].value = 0;
    const last = Math.max(scored ?? -1, conceded ?? -1);
    if (last < 0) return;
    const ours = last === scored, age = (at - last) / hz / FLASH_LIFE, glow = fade(age), tint = this.tints[ours ? 0 : 1]!;
    const side = this.sides[ours ? 0 : 1]!;
    side.strength.value = glow;
    side.light.visible = glow > 0.01;
    side.tint.value.set(tint.r, tint.g, tint.b);
    this.lifePops[ours ? 0 : 1].value = glow;
    const loser = ours ? 1 : 0, droop = age < 1 ? Math.min(1, age * 8) * (1 - Math.max(0, (age - 0.55) / 0.45)) : 0;
    this.lids[loser].value = 0.62 * droop;
    this.flashes[ours ? 0 : 1].value = Math.max(this.flashes[ours ? 0 : 1].value, 0.5 * glow);
  }

  private scoreboard(frame: FrameUpdate, now: EnvFrame, step: number, at: number, hz: number): void {
    this.lifeCounts[0].value = now.words[SLIME_RIVAL_LIVES]!;
    this.lifeCounts[1].value = now.words[SLIME_LIVES]!;
    const left = Math.ceil(Math.max(0, this.frames - now.words[SLIME_FRAME]!) / hz), seconds = left % 60;
    this.digits[0].value = SEVEN[Math.floor(left / 60)]!;
    this.digits[1].value = SEVEN[Math.floor(seconds / 10)]!;
    this.digits[2].value = SEVEN[seconds % 10]!;
    this.member.value = SEVEN[(now.words[SLIME_MEMBER]! + 1) % 10]!;
    this.resultGlow.value = 0;
    const ended = recent(frame.eventLog, 'done', step, Math.ceil(RESULT_LIFE * hz));
    if (ended === null) return;
    const outcome = this.outcome(frame, ended);
    if (outcome === null) return;
    const age = (at - ended) / hz / RESULT_LIFE, tint = this.tints[outcome > 0 ? 4 : outcome < 0 ? 1 : 6]!;
    this.result.value.set(tint.r, tint.g, tint.b);
    this.resultGlow.value = fade(age) * (0.75 + 0.25 * Math.cos(age * 40));
  }

  private outcome(frame: FrameUpdate, ended: number): number | null {
    const before = frame.frameAt(ended - 1), final = frame.frameAt(ended);
    if (before.step !== ended - 1 || final.step !== ended) return null;
    const mine = before.words[SLIME_LIVES]! - (final.reward < 0 ? 1 : 0), theirs = before.words[SLIME_RIVAL_LIVES]! - (final.reward > 0 ? 1 : 0);
    return Math.sign(mine - theirs);
  }

  private celebrate(event: SimEvent, frame: FrameUpdate, step: number, pool: Stage2D['pool'], time: number): void {
    const ground = this.court.ground;
    if (event.type === 'goal' || event.type === 'harm') {
      const ours = event.type === 'goal', at = recent(frame.eventLog, event.type, step, 16);
      if (at === null) return;
      const before = frame.frameAt(at - 1), r = this.court.ball;
      const x = before.step === at - 1 ? word(before, SLIME_BALL + SLIME_X) : ours ? -this.court.half / 2 : this.court.half / 2;
      const landed = ours ? Math.min(x, -r) : Math.max(x, r);
      pool.ripple(this.spot.set(landed, 0, -ground), ours ? GOLD : TOKENS.failureBlue, 0.55, time);
      pool.burst(this.spot.set(landed, 40, -(ground + 0.4)), ours ? GOLD : TOKENS.failureBlue, ours ? 2.2 : 1, time);
      return;
    }
    if (event.type !== 'done') return;
    const at = recent(frame.eventLog, 'done', step, 16), outcome = at === null ? null : this.outcome(frame, at);
    if (outcome === null || outcome === 0) return;
    const centre = (BOARD.bottom + BOARD.top) / 2;
    pool.burst(this.spot.set(0, 40, -centre), outcome > 0 ? GOLD : TOKENS.failureBlue, outcome > 0 ? 3.5 : 1.2, time);
    pool.ripple(this.spot.set(0, 0, -centre), outcome > 0 ? GOLD : TOKENS.failureBlue, 1.6, time);
  }

  protected override drop(): void {
    this.rig.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.rig.clear();
    this.bodies.length = this.mirrors.length = this.shades.length = this.pops.length = this.puffs.length = this.sides.length = 0;
    this.ballParts = [];
    this.ballMirror = this.ballShade = this.serve = this.input = null;
    this.trail = null;
  }
}

export default (() => new SlimeVolleyScene()) satisfies SceneFactory;
