import * as THREE from 'three';
import { envelope } from '@kit/mix';
import { lamp } from '@kit/props';
import { makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { HASHED_ACTION, HASHED_STATE, hashedReward } from '@sim/envs/hashed';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const TRAIL = 8;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const bits = parameter(ctx.environment, 'bits'), local = parameter(ctx.environment, 'local') === 1, actions = constant(ctx.environment, 'HASHED_ACTIONS');
  const side = 2 ** Math.ceil(bits / 2), rows = 2 ** Math.floor(bits / 2), cell = bits > 8 ? 0.09 : 0.62, halfX = side * cell / 2, halfZ = rows * cell / 2;
  const t = makeTerrace(ctx, scene, { mood: 'dawn', halfX: halfX + 1, halfZ: halfZ + 1, ground: [0x9fb0c2, 0xc9d4de], amplitude: 1.6, paving: false, landscape: {
    sky: { sunHdr: 5, sunGlow: 0.25, sunGlowPower: 40 }, light: { pointCount: 0 }, fogDensity: 0.005,
    weather: { preset: 'snow', count: 900, seed: ctx.seed, size: 1.6, opacity: 0.7, bounds: { y: 2.5, halfX: halfX + 4, halfY: 2.5, halfZ: halfZ + 4 } },
  } });
  const { world, rig, m, accent } = t;
  rig.sun.intensity = 2.1; rig.hemi.intensity = 0.5;
  const uniforms = {
    uCell: { value: cell }, uSide: { value: side }, uRows: { value: rows },
    uTrail: { value: Array.from({ length: TRAIL }, () => new THREE.Vector4(0, 0, 0, 1e9)) },
    uGlow: { value: new THREE.Color(accent) },
  };
  const floor = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
  floor.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 vLab;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvLab = position.xz;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
uniform float uCell; uniform float uSide; uniform float uRows; uniform vec4 uTrail[${TRAIL}]; uniform vec3 uGlow;
varying vec2 vLab;
float labHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
vec2 labCell = floor(vLab / uCell + vec2(uSide, uRows) * 0.5);
vec2 labF = fract(vLab / uCell + vec2(uSide, uRows) * 0.5);
float mortar = smoothstep(0.0, 0.08, labF.x) * smoothstep(0.0, 0.08, labF.y) * smoothstep(1.0, 0.92, labF.x) * smoothstep(1.0, 0.92, labF.y);
float tone = 0.55 + 0.45 * labHash(labCell);
diffuseColor.rgb *= mix(vec3(0.42, 0.4, 0.44), mix(vec3(0.72, 0.68, 0.6), vec3(0.9, 0.86, 0.78), tone), mortar);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
for (int i = 0; i < ${TRAIL}; i++) {
  vec2 d = (vLab - uTrail[i].xy) / (uCell * 0.9 + 0.08);
  float lit = exp(-dot(d, d)) * exp(-uTrail[i].w * 1.6);
  totalEmissiveRadiance += uGlow * lit * (0.6 + 0.5 * uTrail[i].z);
}`);
  };
  floor.customProgramCacheKey = () => 'hashedLabyrinth';
  const labyrinth = new THREE.Mesh(world.own(new THREE.PlaneGeometry(side * cell, rows * cell).rotateX(-Math.PI / 2)), world.own(floor));
  labyrinth.receiveShadow = true;
  world.add(labyrinth);
  const lamps = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sz]) => { const l = lamp(world, accent, 1.5, m); l.group.position.set(sx! * (halfX + 0.6), 0, sz! * (halfZ + 0.6)); world.add(l.group); l.lit(0.6); return l; });
  const wisp = world.own(new Wisp({ color: accent, radius: 0.3, height: 0.7, hop: { duration: 0.14, height: 0.35 }, name: 'hashedWalker' }));
  world.add(wisp.group);
  const cellAt = (state: number, out: THREE.Vector3) => out.set(((state % side) + 0.5) * cell - halfX, 0, (Math.floor(state / side) + 0.5) * cell - halfZ);
  const extra = { land: t.land, lights: rig, uniforms, lamps, wisp, cellAt, bits, local, actions, side, rows, cell, accent, gold: ctx.tokens.celebrationGold };
  return world.seal<WorldCore & typeof extra>(extra);
}

type LabyrinthWorld = ReturnType<typeof buildWorld>;

class HashLabyrinthScene extends StageScene<LabyrinthWorld> {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'hashLabyrinth';
  private view!: SceneView;
  private lastStep = -1;
  private arrivedAt = -Infinity;
  private readonly trailSteps = new Int32Array(TRAIL).fill(-1);
  private readonly at = new THREE.Vector3();
  private readonly ledger = { state: 0, action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'LABYRINTH',
    rows: [
      { key: 'state', label: 'cell', tint: 0xc7b7ff, value: () => `${formatCount(this.ledger.state + 1)} of ${formatCount(2 ** this.world!.bits)}` },
      { key: 'action', label: 'door taken', tint: 0xdcf4ff, value: () => `${this.ledger.action + 1} of ${this.world!.actions}` },
      { key: 'moves', label: 'doors lead', tint: 0xc7b7ff, value: () => (this.world!.local ? 'to nearby cells' : 'anywhere') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): LabyrinthWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!, halfX = world.side * world.cell / 2 + 1, halfZ = world.rows * world.cell / 2 + 1;
    this.view = makeSceneView(ctx, { min: [-halfX, -0.5, -halfZ], max: [halfX, 3, halfZ] }, [0, 0.4, 0], { distance: Math.max(9, halfX * 2.1), azimuth: 0.35, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [this.world!.side * this.world!.cell, this.world!.rows * this.world!.cell], agentPos: () => this.world!.wisp.group.position }; }
  protected start(_ctx: SceneContext, world: LabyrinthWorld): void { world.wisp.snap(); }
  protected tick(frame: FrameUpdate, { ctx, world, pool, time }: Stage<LabyrinthWorld>): void {
    const step = frame.state.step, state = Math.round(frame.state.pose[HASHED_STATE]!), action = Math.round(frame.state.pose[HASHED_ACTION]!);
    if (state < 0 || state >= 2 ** world.bits) throw new Error('HashedMDP state is outside its recorded range');
    this.ledger.state = state; this.ledger.action = action; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step; this.arrivedAt = time;
      world.cellAt(state, world.wisp.group.position);
      if (frame.scrubbed || frame.prev === null) { world.wisp.snap(); this.trailSteps.fill(-1); }
      for (let i = 0; i < TRAIL; i++) {
        const at = step - i, trailState = at < 0 ? -1 : Math.round(frame.frameAt(at).words[0]!);
        this.trailSteps[i] = at;
        const slot = world.uniforms.uTrail.value[i]!;
        if (trailState < 0) { slot.set(0, 0, 0, 1e9); continue; }
        world.cellAt(trailState, this.at);
        slot.set(this.at.x, this.at.z, i === 0 ? 1 : 0, i / frame.controlHz);
      }
      if (!frame.scrubbed && frame.state.reward !== 0) {
        this.at.copy(world.wisp.group.position).setY(0.6);
        if (frame.state.reward > 0) pool.burst(this.at, world.gold, Math.min(1, frame.state.reward), time);
        else pool.sigh(Math.min(1, -frame.state.reward), this.at, time);
      }
    }
    const arrive = envelope(time - this.arrivedAt, 1, 2);
    world.uniforms.uTrail.value[0]!.z = 1 + arrive;
    world.wisp.update(frame.dtReal, time);
    if (hashedReward(ctx.seed, state, action) > 0) world.wisp.flare(arrive);
    world.lamps.forEach((l, i) => l.lit(0.5 + 0.2 * Math.sin(time * 1.2 + i)));
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(this.rig!, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.arrivedAt = -Infinity; }
}

export default ((() => new HashLabyrinthScene()) satisfies SceneFactory);
