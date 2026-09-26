import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  ELEVEN_BALL_X, ELEVEN_BALL_Y, ELEVEN_BALL_Z, ELEVEN_CORNER, ELEVEN_DIVING, ELEVEN_DOWN, ELEVEN_FREEKICK, ELEVEN_KICKOFF, ELEVEN_NORMAL, ELEVEN_OFFSIDE,
  ELEVEN_PLAYER, elevenCoarse, elevenDecoder, elevenState, elevenTeamSlots, readEleven, type ElevenState,
} from '@sim/envs/eleven';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BALL, BALL_SPRITE, ballMaterial, ballShadowMaterial, CURSOR, cursorMaterial, flat, KITS, LIFT, numerals, SHADOW_SPRITE, SPRITE, TOKEN, tokenMaterial, tokenRimMaterial, tokenShadowMaterial,
  TETHER, tetherMaterial, trailMaterial, type CursorLevels,
} from './figures';
import { aimOf, BOARD, HEAD, NET, pitchLevels, pitchMaterial, restartLevels, restartMaterial, RUN_X, RUN_Y, SPAN, towers, type Dims, type PitchLevels, type RestartLevels } from './pitch';

const MODES = ['open play', 'kickoff', 'goal kick', 'free kick', 'corner', 'throw-in', 'penalty'] as const;
const ARROWS = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'] as const;
const SPEED = 6;
const TOWER = 6;
const TRAIL = 40;
const LOOKBACK = 1200;
const SPRAY_MODES = new Set([ELEVEN_FREEKICK, ELEVEN_CORNER]);
const GOLD = TOKENS.celebrationGold;
const MISS = TOKENS.failureBlue;

interface Memo { bit: number; since: number; through: number; heading: number }
interface Onset { key: string; step: number; ready: number }

const clock = (seconds: number): string => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

class ElevenScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['night', 'fog'] } };
  controlHz = 60;
  protected readonly stageName = 'eleven';

  private dims: Dims = { length: 0, width: 0, post: 0, bar: 0, box: 0, boxWidth: 0, area: 0, areaWidth: 0, mark: 0, circle: 0 };
  private perSide = 1;
  private teamSlots = 1;
  private drill = false;
  private league = false;
  private edge = 1;
  private dt = 1;
  private accent = 0x7ddc8a;

  private pitch: PitchLevels | null = null;
  private restart: RestartLevels | null = null;
  private cursorLevels: CursorLevels | null = null;
  private tokens: THREE.InstancedMesh | null = null;
  private shadows: THREE.InstancedMesh | null = null;
  private rims: THREE.InstancedMesh | null = null;
  private look: THREE.InstancedBufferAttribute | null = null;
  private motion: THREE.InstancedBufferAttribute | null = null;
  private trail: THREE.InstancedMesh | null = null;
  private trailState: THREE.InstancedBufferAttribute | null = null;
  private cursor: THREE.Mesh | null = null;
  private ball: THREE.Mesh | null = null;
  private ballShadow: THREE.Mesh | null = null;
  private halo: THREE.Mesh | null = null;
  private flash: THREE.Mesh | null = null;
  private tether: THREE.Mesh | null = null;
  private readonly span = uniform(1);
  private readonly flow = uniform(0);
  private readonly spin = uniform(0);
  private readonly height = uniform(0);
  private readonly haloPower = uniform(0.5);
  private readonly flashPower = uniform(0);
  private readonly flashTint = uniform(new THREE.Color(GOLD));
  private readonly disposables: { dispose(): void }[] = [];

  private readonly cur = elevenState();
  private readonly nxt = elevenState();
  private readonly probe = elevenState();
  private readonly history = [elevenState(), elevenState()];
  private readonly hold = new Float32Array(elevenDecoder.slots);
  private readonly exact = new Float32Array(elevenDecoder.slots);
  private order: number[] = [];
  private memos: Memo[] = [];
  private onsetMemo: Onset | null = null;
  private lastStep = -1;
  private lastActive = -1;
  private lastBall = new THREE.Vector2();
  private turn = 0;
  private pingAt = -Infinity;
  private netAt = -Infinity;
  private cheerAt = -Infinity;
  private flashAt = -Infinity;
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly scale = new THREE.Vector3();
  private readonly at = new THREE.Vector3();
  private readonly drawn = new THREE.Vector3();
  private readonly points: THREE.Vector3[] = Array.from({ length: TRAIL + 1 }, () => new THREE.Vector3());
  private readonly ledger = { clock: 0, us: 0, them: 0, half: 0, mode: 0, team: 0, ready: 0, ball: '—', agent: '—', control: '—', reward: 0, game: 0, wins: 0, losses: 0, draws: 0, member: 0 };

  private legendFor(): LegendSpec {
    const ledger = this.ledger;
    return {
      title: 'ELEVEN',
      rows: [
        this.drill
          ? { key: 'clock', label: 'clock', tint: this.accent, value: () => `${clock(ledger.clock)} of ${clock(this.edge * this.dt)}` }
          : { key: 'score', label: 'score', tint: this.accent, value: () => `${ledger.us} – ${ledger.them} · ${ledger.half ? '2nd' : '1st'} half · ${clock(ledger.clock)}` },
        { key: 'play', label: 'play', tint: 0xf5f8ff, value: () => this.playText() },
        { key: 'ball', label: 'ball', tint: 0xf6f7f2, value: () => ledger.ball },
        { key: 'agent', label: 'agent', tint: this.accent, value: () => ledger.agent },
        { key: 'control', label: 'control', tint: 0xdcf4ff, value: () => ledger.control },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(ledger.reward) },
      ],
      note: () => this.recordText(),
      bar: () => clamp01(ledger.clock / (this.edge * this.dt)),
    };
  }

  private playText(): string {
    const { mode, team, ready } = this.ledger;
    if (mode === ELEVEN_NORMAL) return MODES[0];
    const who = team === 0 ? 'us' : 'them';
    return `${MODES[mode] ?? `restart ${mode}`} · ${who}${ready > 0 ? ` · ready in ${(ready * this.dt).toFixed(1)} s` : ''}`;
  }

  private recordText(): string {
    const { game, wins, losses, draws, member } = this.ledger;
    if (this.drill) return `attempt ${formatCount(game + 1)} · scored ${formatCount(wins)} · stopped ${formatCount(losses)}`;
    const record = `won ${formatCount(wins)} · drawn ${formatCount(draws)} · lost ${formatCount(losses)}`;
    return this.league ? `league member ${member + 1} · game ${formatCount(game + 1)} · ${record}` : `game ${formatCount(game + 1)} · ${record}`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, pitch = constantList(env, 'ELEVEN_PITCH');
    if (pitch.length !== 10) throw new Error(`${env.envId} measures its pitch with ${pitch.length} numbers, not 10`);
    const [length, width, post, bar, box, boxWidth, area, areaWidth, mark, circle] = pitch as [number, number, number, number, number, number, number, number, number, number];
    this.dims = { length, width, post, bar, box, boxWidth, area, areaWidth, mark, circle };
    this.perSide = parameter(env, 'perSide');
    this.drill = parameter(env, 'drill') !== 0;
    this.league = parameter(env, 'league') !== 0;
    this.teamSlots = elevenTeamSlots(env);
    this.edge = parameter(env, 'ticks') * parameter(env, 'slices');
    this.dt = constant(env, 'ELE_DT');
    this.controlHz = 1 / this.dt;
    this.accent = this.accentOf(ctx);
    this.memos = Array.from({ length: 2 * this.perSide }, () => ({ bit: 0, since: -1, through: -2, heading: 0 }));
    this.legend = this.legendFor();
    const x = length + RUN_X + BOARD + 1, y = width + RUN_Y + BOARD + 1.2;
    return new THREE.Box2(new THREE.Vector2(-x, -y), new THREE.Vector2(x, y));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xdde6ff).multiplyScalar(0.74);
  }

  protected override poolOptions() {
    return { burstSize: 16 };
  }

  private hasKeeper(team: number): boolean {
    return !(this.drill && team === 0);
  }

  private viewOf(slot: number): number {
    if (slot >= 2 * this.teamSlots) return -1;
    const team = slot >= this.teamSlots ? 1 : 0, index = slot - team * this.teamSlots;
    return index < this.perSide ? team * this.perSide + index : -1;
  }

  private numberOf(view: number): number {
    return (view < this.perSide ? view : view - this.perSide) + 1;
  }

  private sideOf(view: number): string {
    const team = view < this.perSide ? 0 : 1, index = view - team * this.perSide;
    return `${team === 0 ? 'our' : 'their'} ${index === 0 && this.hasKeeper(team) ? 'keeper' : `#${index + 1}`}`;
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const d = this.dims, L = d.length, W = d.width;
    this.scene.background = new THREE.Color(0x06080b);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const keep = <T extends { dispose(): void }>(item: T): T => { this.disposables.push(item); return item; };

    const pitch = pitchLevels();
    this.pitch = pitch;
    const ground = new THREE.Mesh(keep(flat(SPAN, SPAN)), keep(pitchMaterial(d, light.lit, pitch)));
    ground.renderOrder = 0;
    ground.frustumCulled = false;

    const restart = restartLevels();
    this.restart = restart;
    const overlayWidth = 2 * (L + RUN_X), overlayDepth = 2 * (W + RUN_Y);
    const marks = new THREE.Mesh(keep(flat(overlayWidth, overlayDepth)), keep(restartMaterial(d, restart, overlayWidth, overlayDepth)));
    marks.renderOrder = 1;
    marks.frustumCulled = false;
    this.scene.add(ground, marks);

    for (const tower of towers(d)) {
      const lamp = light.emitter(keep(flat(HEAD.x, HEAD.y)), rgb(0xfff1d6).mul(TOWER));
      keep(lamp.material as THREE.Material);
      lamp.position.set(tower.x, 0, -tower.y);
      lamp.rotation.y = aimOf(tower);
      this.scene.add(lamp);
    }
    const halo = light.emitter(keep(new THREE.CircleGeometry(0.8, 32).rotateX(-Math.PI / 2)), rgb(this.accent).mul(this.haloPower));
    keep(halo.material as THREE.Material);
    this.halo = halo;
    const flash = light.emitter(keep(flat(NET, 2 * d.post)), (this.flashTint as unknown as THREE.Node<'vec3'>).mul(this.flashPower));
    keep(flash.material as THREE.Material);
    flash.visible = false;
    this.flash = flash;
    this.scene.add(halo, flash);

    const count = 2 * this.perSide, atlas = numerals(this.teamSlots - 1);
    keep(atlas.texture);
    const body = keep(flat(SPRITE, SPRITE));
    this.look = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.motion = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
    body.setAttribute('aLook', this.look);
    body.setAttribute('aMotion', this.motion);
    const shadows = keep(instancedMesh(body, keep(tokenShadowMaterial(towers(d))), count));
    const tokens = keep(instancedMesh(body, keep(tokenMaterial(atlas)), count));
    const rims = keep(instancedMesh(body, keep(tokenRimMaterial()), count));
    shadows.renderOrder = 2;
    tokens.renderOrder = 4;
    rims.renderOrder = 4.5;
    this.shadows = shadows;
    this.tokens = tokens;
    this.rims = rims;

    const streak = keep(flat(1, 1));
    this.trailState = new THREE.InstancedBufferAttribute(new Float32Array(TRAIL * 4), 4).setUsage(THREE.DynamicDrawUsage);
    streak.setAttribute('aTrail', this.trailState);
    const trail = keep(instancedMesh(streak, keep(trailMaterial()), TRAIL));
    trail.renderOrder = 3;
    this.trail = trail;

    const ballShadow = new THREE.Mesh(keep(flat(SHADOW_SPRITE, SHADOW_SPRITE)), keep(ballShadowMaterial(this.height)));
    ballShadow.renderOrder = 3;
    this.ballShadow = ballShadow;
    const ball = new THREE.Mesh(keep(flat(BALL_SPRITE, BALL_SPRITE)), keep(ballMaterial(this.spin)));
    ball.renderOrder = 6;
    this.ball = ball;
    const tether = new THREE.Mesh(keep(flat(1, TETHER)), keep(tetherMaterial(this.accent, this.span, this.flow)));
    tether.renderOrder = 3;
    tether.visible = false;
    this.tether = tether;
    const cursor = cursorMaterial(this.accent);
    this.cursorLevels = cursor.levels;
    const ring = new THREE.Mesh(keep(flat(CURSOR, CURSOR)), keep(cursor.material));
    ring.renderOrder = 5;
    this.cursor = ring;
    for (const mesh of [shadows, tokens, rims, trail, ballShadow, ball, ring, tether]) {
      mesh.frustumCulled = false;
      this.scene.add(mesh);
    }
    this.order = Array.from({ length: count }, (_, k) => k);
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
  }

  private onset(a: ElevenState, b: ElevenState): boolean {
    return b.game !== a.game || b.half !== a.half || (b.mode !== ELEVEN_NORMAL && (b.mode !== a.mode || b.team !== a.team || b.taker !== a.taker));
  }

  private loose(state: ElevenState): boolean {
    return state.owner >= 2 * this.teamSlots;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const step = frame.state.step, now = frame.frameAt(step);
    if (frame.scrubbed || step < this.lastStep) {
      pool.clear();
      this.pingAt = this.netAt = this.cheerAt = this.flashAt = -Infinity;
      this.lastActive = -1;
    }
    pool.update(frame.dtReal, time);
    if (now.step !== step || !this.pitch) return;
    readEleven(now, this.cur);
    const after = frame.frameAt(step + 1), ahead = after.step === step + 1 && after.done === 0;
    if (ahead) readEleven(after, this.nxt);
    let pose = frame.state.pose;
    if (ahead && this.onset(this.cur, this.nxt)) {
      this.hold.set(frame.poseAt(step));
      pose = this.hold;
    }
    this.pitch.time.value = time;
    this.celebrate(frame, pool, time);
    this.placePlayers(frame, pose, step, time);
    this.placeBall(pose);
    this.placeTrail(frame, pose, step);
    this.placeTether(frame, pose, time);
    this.placeRestart(frame, step);
    this.animate(time);
    this.record(frame, step);
    this.lastStep = step;
  }

  private placePlayers(frame: FrameUpdate, pose: Float32Array, step: number, time: number): void {
    const tokens = this.tokens, shadows = this.shadows, rims = this.rims, look = this.look, motion = this.motion, cursor = this.cursor, levels = this.cursorLevels, halo = this.halo;
    if (!tokens || !shadows || !rims || !look || !motion || !cursor || !levels || !halo) return;
    const cur = this.cur, count = 2 * this.perSide, active = this.viewOf(cur.active);
    let slot = 0;
    for (let k = 0; k < count; k++) if (k !== active) this.order[slot++] = k;
    if (active >= 0) this.order[slot++] = active;
    for (let i = 0; i < count; i++) {
      const k = this.order[i]!, player = cur.players[k]!, team = k < this.perSide ? 0 : 1, index = k - team * this.perSide;
      const keeper = index === 0 && this.hasKeeper(team);
      const x = pose[ELEVEN_PLAYER + 2 * k]!, y = pose[ELEVEN_PLAYER + 2 * k + 1]!;
      this.matrix.makeTranslation(x, 0, -y);
      tokens.setMatrixAt(i, this.matrix);
      shadows.setMatrixAt(i, this.matrix);
      rims.setMatrixAt(i, this.matrix);
      const bit = player.flags & ELEVEN_DOWN ? ELEVEN_DOWN : player.flags & ELEVEN_DIVING ? ELEVEN_DIVING : 0;
      look.setXYZW(i, 2 * team + (keeper ? 1 : 0), index, bit === ELEVEN_DOWN ? 1 : bit === ELEVEN_DIVING ? 2 : 0, bit ? this.headingOf(frame, step, k, bit) : 0);
      motion.setXYZW(i, Math.atan2(player.vy, player.vx), Math.min(1, Math.hypot(player.vx, player.vy) / SPEED), player.flags & ELEVEN_OFFSIDE ? 1 : 0, 0);
    }
    for (const mesh of [tokens, shadows, rims]) mesh.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
    motion.needsUpdate = true;

    cursor.visible = halo.visible = active >= 0;
    if (active < 0) return;
    const x = pose[ELEVEN_PLAYER + 2 * active]!, y = pose[ELEVEN_PLAYER + 2 * active + 1]!;
    cursor.position.set(x, 0, -y);
    halo.position.set(x, 0.01, -y);
    const control = frame.frameAt(step + 1), live = control.step === step + 1, mx = live ? Math.max(-1, Math.min(1, control.actions[0]!)) : 0, my = live ? Math.max(-1, Math.min(1, control.actions[1]!)) : 0;
    levels.aimX.value = mx;
    levels.aimY.value = my;
    levels.stamina.value = cur.players[active]!.stamina;
    if (cur.active !== this.lastActive && this.lastActive >= 0 && !frame.scrubbed) this.pingAt = time;
    this.lastActive = cur.active;
    levels.ping.value = clamp01(1 - (time - this.pingAt) / 0.4);
    levels.glow.value = cur.owner === cur.active ? 1.25 : 1;
  }

  private headingOf(frame: FrameUpdate, step: number, k: number, bit: number): number {
    const memo = this.memos[k]!;
    if (memo.bit === bit && memo.since <= step && step <= memo.through + 1) {
      memo.through = Math.max(memo.through, step);
      return memo.heading;
    }
    let since = step;
    while (since > 0 && step - since < LOOKBACK) {
      const earlier = frame.frameAt(since - 1);
      if (earlier.step !== since - 1 || !((earlier.view[2 * k + 1]! >>> 24) & bit)) break;
      since--;
    }
    const first = frame.frameAt(since), motion = first.view[2 * k + 1]!, vx = elevenCoarse(motion & 255), vy = elevenCoarse((motion >>> 8) & 255);
    const current = this.cur.players[k]!;
    const heading = bit === ELEVEN_DIVING
      ? (Math.abs(current.vy) > 0.3 ? Math.sign(current.vy) : Math.sign(this.cur.by - current.y) || 1) * Math.PI / 2
      : Math.hypot(vx, vy) > 0.2 ? Math.atan2(vy, vx) : Math.atan2(current.vy, current.vx);
    Object.assign(memo, { bit, since, through: step, heading });
    return heading;
  }

  private observed(state: ElevenState, observations: ArrayLike<number>): number {
    const count = 2 * this.perSide, L = this.dims.length;
    const x = (observations[0]! + observations[10]!) * L, y = (observations[1]! + observations[11]!) * L;
    let best = -1, gap = Infinity;
    for (let k = this.perSide; k < count; k++) {
      const player = state.players[k]!, distance = Math.hypot(player.x - x, player.y - y);
      if (distance < gap) { gap = distance; best = k; }
    }
    return gap < 0.05 ? best : -1;
  }

  private placeTether(frame: FrameUpdate, pose: Float32Array, time: number): void {
    const tether = this.tether, active = this.viewOf(this.cur.active);
    if (!tether) return;
    const rival = active >= 0 ? this.observed(this.cur, frame.frameAt(frame.state.step).observations) : -1;
    tether.visible = rival >= 0;
    if (rival < 0) return;
    const ax = pose[ELEVEN_PLAYER + 2 * active]!, ay = pose[ELEVEN_PLAYER + 2 * active + 1]!, bx = pose[ELEVEN_PLAYER + 2 * rival]!, by = pose[ELEVEN_PLAYER + 2 * rival + 1]!;
    const dx = bx - ax, dy = by - ay, reach = Math.hypot(dx, dy), start = TOKEN + 0.7, end = reach - TOKEN - 0.2, span = end - start;
    tether.visible = span > 0.6;
    if (!tether.visible) return;
    const ux = dx / reach, uy = dy / reach, mid = (start + end) / 2;
    tether.position.set(ax + ux * mid, 0, -(ay + uy * mid));
    tether.rotation.y = Math.atan2(uy, ux);
    tether.scale.set(span, 1, 1);
    this.span.value = span;
    this.flow.value = time * 1.2;
  }

  private ballAt(pose: Float32Array, state: ElevenState, out: THREE.Vector3): THREE.Vector3 {
    const bx = pose[ELEVEN_BALL_X]!, by = pose[ELEVEN_BALL_Y]!, bz = pose[ELEVEN_BALL_Z]!;
    const owner = this.viewOf(state.owner);
    if (owner >= 0 && !state.held) {
      const px = pose[ELEVEN_PLAYER + 2 * owner]!, py = pose[ELEVEN_PLAYER + 2 * owner + 1]!, dx = bx - px, dy = by - py, reach = Math.hypot(dx, dy);
      if (reach > 0.1) return out.set(px + dx / reach * (TOKEN + 0.55 * BALL), 0, -(py + dy / reach * (TOKEN + 0.55 * BALL)));
    }
    return out.set(bx, 0, -(by + bz * LIFT));
  }

  private placeBall(pose: Float32Array): void {
    const ball = this.ball, shadow = this.ballShadow;
    if (!ball || !shadow) return;
    const bx = pose[ELEVEN_BALL_X]!, by = pose[ELEVEN_BALL_Y]!, bz = pose[ELEVEN_BALL_Z]!;
    this.ballAt(pose, this.cur, ball.position);
    ball.scale.setScalar(1 + 0.04 * bz);
    const owned = !this.loose(this.cur) && !this.cur.held;
    shadow.position.set(owned ? ball.position.x : bx, 0, owned ? ball.position.z : -by);
    this.height.value = bz;
    const moved = Math.hypot(ball.position.x - this.lastBall.x, ball.position.z - this.lastBall.y);
    this.turn = (this.turn + (moved < 3 ? moved / BALL : 0)) % (2 * Math.PI);
    this.spin.value = this.turn;
    this.lastBall.set(ball.position.x, ball.position.z);
  }

  private placeTrail(frame: FrameUpdate, pose: Float32Array, step: number): void {
    const trail = this.trail, state = this.trailState;
    if (!trail || !state) return;
    let used = 0, later = this.cur;
    this.ballAt(pose, this.cur, this.points[0]!);
    for (let i = 0; i < TRAIL; i++) {
      const s = step - i, f = frame.frameAt(s);
      if (f.step !== s) break;
      const earlier = i === 0 ? this.cur : readEleven(f, this.history[i % 2]!);
      if ((i > 0 && this.onset(earlier, later)) || !this.loose(earlier) || !earlier.live) break;
      const head = this.points[i]!, point = this.points[i + 1]!.set(earlier.bx, 0, -(earlier.by + earlier.bz * LIFT)), span = head.distanceTo(point);
      const fade = 0.75 * clamp01((Math.hypot(earlier.bvx, earlier.bvy, earlier.bvz) - 3) / 14);
      if (span > 1e-4) {
        this.at.addVectors(head, point).multiplyScalar(0.5);
        this.quaternion.setFromAxisAngle(this.axis, Math.atan2(point.z - head.z, head.x - point.x));
        trail.setMatrixAt(used, this.matrix.compose(this.at, this.quaternion, this.scale.set(span + 0.5, 1, 0.5)));
        state.setXYZW(used, span, 0.5, fade * (1 - (i + 1) / TRAIL), fade * (1 - i / TRAIL));
        used++;
      }
      if (f.done) break;
      later = earlier;
    }
    trail.count = used;
    trail.instanceMatrix.needsUpdate = true;
    state.needsUpdate = true;
  }

  private restartOnset(frame: FrameUpdate, step: number): Onset {
    const cur = this.cur, key = `${cur.game}:${cur.half}:${cur.mode}:${cur.team}:${cur.taker}`, memo = this.onsetMemo;
    if (memo && memo.key === key && memo.step <= step && step - memo.step < LOOKBACK) return memo;
    let since = step, ready = cur.ready;
    while (since > 0 && step - since < LOOKBACK) {
      const earlier = frame.frameAt(since - 1);
      if (earlier.step !== since - 1 || earlier.done) break;
      const state = readEleven(earlier, this.probe);
      if (this.onset(state, cur) || state.mode !== cur.mode) break;
      ready = state.ready;
      since--;
    }
    this.onsetMemo = { key, step: since, ready };
    return this.onsetMemo;
  }

  private placeRestart(frame: FrameUpdate, step: number): void {
    const levels = this.restart;
    if (!levels) return;
    const cur = this.cur;
    levels.mode.value = cur.mode;
    if (cur.mode === ELEVEN_NORMAL) {
      levels.age.value = 99;
      levels.ready.value = 0;
      return;
    }
    const onset = this.restartOnset(frame, step);
    levels.age.value = (step - onset.step + clamp01(frame.position - step)) * this.dt;
    levels.ready.value = onset.ready > 0 ? cur.ready / (onset.ready + 1) : 0;
    levels.side.value = cur.team === 0 ? 1 : -1;
    levels.spot.value.set(cur.sx, cur.sy);
    levels.tint.value.set(KITS[cur.team === 0 ? 0 : 2]);
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    const pitch = this.pitch, flash = this.flash;
    if (!pitch || !flash) return;
    const scoredNow = frame.events.some((event) => event.type === 'goal' || event.type === 'harm');
    for (const event of frame.events) {
      if (event.type === 'done' && !scoredNow) this.whistle(frame, pool, time);
      if (event.type !== 'goal' && event.type !== 'harm') continue;
      const ours = event.type === 'goal', side = ours ? 1 : -1, step = frame.state.step;
      let scored = step;
      while (scored > 0 && step - scored < 16 && frame.frameAt(scored).reward === 0) scored--;
      const before = frame.frameAt(scored - 1), crossing = before.step === scored - 1 ? readEleven(before, this.probe).by : 0;
      const y = Math.max(-this.dims.post, Math.min(this.dims.post, crossing)), x = side * (this.dims.length + NET / 2);
      const color = ours ? GOLD : MISS;
      pool.burst({ x, y: 1, z: -y }, color, ours ? 3 : 1.8, time);
      pool.ripple({ x: side * this.dims.length, y: 0, z: -y }, color, ours ? 2.2 : 1.5, time);
      pool.pulse.fire(ours ? 0.2 : 0.08);
      pitch.netSide.value = side;
      pitch.netY.value = y;
      pitch.cheerSide.value = ours ? -1 : 1;
      pitch.celebrate.value.set(ours ? GOLD : KITS[2]);
      this.flashTint.value.set(ours ? GOLD : MISS);
      flash.position.set(x, 0.01, 0);
      this.netAt = this.cheerAt = this.flashAt = time;
    }
  }

  private whistle(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    const step = frame.state.step;
    let ended = step;
    while (ended > 0 && step - ended < 16 && frame.frameAt(ended).done === 0) ended--;
    const before = frame.frameAt(ended - 1);
    if (before.step !== ended - 1) return;
    const last = readEleven(before, this.probe);
    pool.ripple({ x: last.bx, y: 0, z: -last.by }, this.drill ? MISS : 0xf5f8ff, 0.9, time);
  }

  private animate(time: number): void {
    const pitch = this.pitch, flash = this.flash;
    if (!pitch || !flash) return;
    const net = time - this.netAt, cheer = time - this.cheerAt, glow = time - this.flashAt;
    pitch.netAge.value = net >= 0 ? clamp01(net / 1.4) : 1;
    pitch.cheer.value = cheer >= 0 ? clamp01(1 - cheer / 4) ** 0.6 * clamp01(cheer / 0.15) : 0;
    const power = glow >= 0 ? Math.exp(-glow * 1.8) : 0;
    this.flashPower.value = power;
    flash.visible = power > 0.02;
  }

  private record(frame: FrameUpdate, step: number): void {
    const cur = this.cur, ledger = this.ledger;
    ledger.clock = cur.clock * this.dt;
    ledger.us = cur.us;
    ledger.them = cur.them;
    ledger.half = cur.half;
    ledger.mode = cur.mode;
    ledger.team = cur.team;
    ledger.ready = cur.ready;
    ledger.game = cur.game;
    ledger.wins = cur.wins;
    ledger.losses = cur.losses;
    ledger.draws = cur.draws;
    ledger.member = cur.member;
    ledger.reward = frame.state.reward;
    const owner = this.viewOf(cur.owner), active = this.viewOf(cur.active);
    if (owner >= 0) ledger.ball = `${this.sideOf(owner)} ${cur.held ? 'holds it' : cur.mode === ELEVEN_NORMAL ? 'on the ball' : 'to take'}`;
    else ledger.ball = `loose · ${Math.hypot(cur.bvx, cur.bvy, cur.bvz).toFixed(1)} m/s${cur.bz > 0.05 ? ` · ${cur.bz.toFixed(1)} m up` : ''}`;
    if (active >= 0) {
      const player = cur.players[active]!;
      ledger.agent = `#${this.numberOf(active)} · stamina ${Math.round(player.stamina * 100)}%${player.flags & ELEVEN_DOWN ? ' · down' : ''}`;
    } else ledger.agent = '—';
    const control = frame.frameAt(step + 1);
    if (control.step !== step + 1) {
      ledger.control = '—';
      return;
    }
    const act = control.actions, mx = Math.max(-1, Math.min(1, act[0]!)), my = Math.max(-1, Math.min(1, act[1]!)), push = Math.hypot(mx, my);
    const arrow = ARROWS[((Math.round(Math.atan2(my, mx) / (Math.PI / 4)) % 8) + 8) % 8]!;
    ledger.control = `${arrow} ${Math.min(1, push).toFixed(2)}${act[2]! > 0 ? ' · sprint' : ''}`;
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const step = frame.state.step, now = frame.frameAt(step);
    if (now.step !== step) return;
    const s = readEleven(now, this.probe), d = this.dims, count = 2 * this.perSide;
    const players = kit.marks('players', count), flagged = kit.marks('offside', count);
    for (let k = 0; k < count; k++) {
      const player = s.players[k]!;
      players.set(player.x, 0.2, -player.y, 0.32, k < this.perSide ? TOKENS.uiCyan : TOKENS.cueMagenta);
      if (player.flags & ELEVEN_OFFSIDE) flagged.set(player.x, 0.2, -player.y - 1.4, 0.3, GOLD);
    }
    const pitch = kit.curve('pitch', 5, 0x8fa6c4);
    for (const [x, y] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) pitch.point(x * d.length, 0.1, -y * d.width);
    for (const sign of [-1, 1]) {
      const mouth = kit.curve(sign < 0 ? 'our goal mouth' : 'their goal mouth', 2, GOLD);
      mouth.point(sign * d.length, 0.1, -d.post);
      mouth.point(sign * d.length, 0.1, d.post);
    }
    const active = this.viewOf(s.active), cross = kit.curve('active', 5, this.accent);
    if (active >= 0) {
      const { x, y } = s.players[active]!, r = TOKEN + 0.6;
      for (const [dx, dy] of [[-r, -r], [r, r], [0, 0], [-r, r], [r, -r]] as const) cross.point(x + dx, 0.1, -y - dy);
      const link = (name: string, view: number, color: number) => {
        const curve = kit.curve(name, 2, color);
        if (view < 0) return;
        curve.point(x, 0.1, -y);
        curve.point(s.players[view]!.x, 0.1, -s.players[view]!.y);
      };
      link('observed opponent', this.observed(s, now.observations), TOKENS.cueMagenta);
      link('scanned', s.scan < count ? s.scan : -1, TOKENS.cueCyan);
    }
    const presser = this.viewOf(s.near);
    if (presser >= 0) kit.marks('nearest to ball', 1).set(s.players[presser]!.x, 0.2, -s.players[presser]!.y + 1.4, 0.26, TOKENS.uiCyan);
    kit.marks('ball', 1).set(s.bx, 0.2, -s.by, 0.2, 0xffffff);
    this.exact.set(frame.poseAt(step));
    this.ballAt(this.exact, s, this.drawn);
    const lift = kit.curve('ball as drawn', 2, 0xffffff);
    lift.point(s.bx, 0.1, -s.by);
    lift.point(this.drawn.x, 0.1, this.drawn.z);
    const velocity = kit.curve('ball velocity', 2, GOLD);
    velocity.point(s.bx, 0.1, -s.by);
    velocity.point(s.bx + 0.5 * s.bvx, 0.1, -(s.by + 0.5 * s.bvy));
    if (s.mode !== ELEVEN_NORMAL) {
      kit.marks('restart spot', 1).set(s.sx, 0.2, -s.sy, 0.3, GOLD);
      const ring = kit.curve('clearance', 65, GOLD), centre = s.mode === ELEVEN_KICKOFF ? { x: 0, y: 0 } : { x: s.sx, y: s.sy };
      if (SPRAY_MODES.has(s.mode) || s.mode === ELEVEN_KICKOFF) for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * 2 * Math.PI;
        ring.point(centre.x + d.circle * Math.cos(a), 0.1, -(centre.y + d.circle * Math.sin(a)));
      }
    }
  }
}

export default (() => new ElevenScene()) satisfies SceneFactory;
