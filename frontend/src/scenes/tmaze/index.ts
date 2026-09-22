import * as THREE from 'three';
import { CELL } from '@kit/grid';
import { emissiveMaterial, setEmissiveIntensity } from '@kit/materials';
import { gate, lamp, relief } from '@kit/props';
import { glyphSet, instanced, memoryCourt, outcomeGlow } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { excludeAny, excludeRect, scatterTrees } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { TMAZE_CUE, TMAZE_POS } from '@sim/envs/tmaze';
import { lastEvent } from '@sim/events';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const GLYPH = ['cyan cube', 'magenta pyramid', 'gold sphere', 'green octahedron'] as const;

const WALK_SECONDS = 0.9;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, length = parameter(env, 'length'), choices = 2 ** parameter(env, 'cueBits');
  if (length < 2 || choices !== env.actions || choices > 4) throw new Error('TMaze geometry does not match its recorded configuration');
  const grid = { w: 1, h: length }, half = (length - 1) * CELL / 2, span = 1.8 * (choices - 1) + 0.6;
  const court = memoryCourt(ctx, scene, {
    name: 'tMazeOrchard', palette: 'afternoon', fogExtent: 150 + half * 2, points: 0,
    weather: { preset: 'pollen', count: 420, seed: ctx.seed, color: 0xfff1c8, size: 1.1, intensity: 0.8, bounds: { y: 1.2, halfX: span + 4, halfY: 1.2, halfZ: half + 4 } },
    focus: { x: 0, y: 0.5, z: -half * 0.2, radius: half + 5 },
    meadow: { halfX: span + 3.5, halfZ: half + 4.5, ground: 0xa5ad7a, grass: 0x7f9660, trees: [0xf2b8cf, 0xaed17a, 0xd8e39a, 0x8fbf74], amplitude: 1.2 },
  });
  const { world, materials: m, palette } = court;
  const cells = instanced(world, new THREE.BoxGeometry(0.84, 0.08, 0.78), m.sandstone, length, 'tMazeCells');
  for (let i = 0; i < length; i++) { cells.place(i, 0, 0.04, half - i * CELL); cells.paint(i, 0xffffff); }
  cells.commit();
  const kerbs = new THREE.Group(), parts = modelParts(world);
  for (const side of [-1, 1]) parts.box(kerbs, m.shale, side * 0.64, 0.09, 0, 0.14, 0.18, length * CELL + 0.4, 0.02);
  parts.build();
  world.add(kerbs);
  const orchard = scatterTrees({
    archetypes: ['broadleaf'], count: Math.min(60, length * 2 + 8), area: { halfX: 3.6, halfZ: half + 1.5 }, seed: ctx.seed,
    exclude: excludeAny(excludeRect(0, 0, 1.6, half + 2, 0), excludeRect(0, -half - 2.4, span + 3, 3, 0)),
    heightAt: court.heightAt, color: 0xe6b3cc, hueJitter: 0.3, scale: [0.6, 0.95], stretch: [0.9, 1.25], wind: 0.02, windFrequency: 0.8,
    rimColor: palette.skyHorizon, rimStrength: 0.3, castShadow: true, name: 'tMazeBlossoms',
  });
  world.add(orchard.group);
  world.time(world.own(orchard));
  const glyphs = glyphSet(world, ctx.tokens);
  const cueLamp = lamp(world, ctx.tokens.cueCyan, 1.4, m);
  cueLamp.group.position.set(-1.4, 0, half + 0.1);
  world.add(cueLamp.group);
  const cueMaterial = world.own(new THREE.MeshStandardMaterial({ color: ctx.tokens.cueCyan, emissive: ctx.tokens.cueCyan, emissiveIntensity: 1.6, roughness: 0.3 }));
  const cueGlyph = world.add(new THREE.Mesh(glyphs.shapes[0]!, cueMaterial));
  cueGlyph.position.set(-1.4, 2.15, half + 0.1);
  cueGlyph.scale.setScalar(0.55);
  const ghostMaterial = world.own(new THREE.MeshStandardMaterial({ color: ctx.tokens.cueCyan, emissive: ctx.tokens.cueCyan, emissiveIntensity: 1.1, transparent: true, opacity: 0.5, depthWrite: false }));
  const ghost = new THREE.Mesh(glyphs.shapes[0]!, ghostMaterial);
  ghost.scale.setScalar(0.4);
  const wall = relief(world, span * 2 + 5, 1.9, 0.5, m);
  wall.position.set(0, 0, -half - 2.7);
  world.add(wall);
  const gates = glyphs.tints.slice(0, choices).map((tint, choice) => {
    const x = (choice / (choices - 1) * 2 - 1) * span, z = -half - 1.9;
    const door = gate(world, 1.1, 1.7, tint, m);
    door.group.position.set(x, 0, z);
    world.add(door.group);
    const material = world.own(new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.8, roughness: 0.35 }));
    const glyph = world.add(new THREE.Mesh(glyphs.shapes[choice]!, material));
    glyph.position.set(x, 2.5, z);
    glyph.scale.setScalar(0.7);
    return { door, glyph, material, tint, at: new THREE.Vector3(x, 0.8, z) };
  });
  const wisp = world.own(new Wisp({ color: 0xfff1cf, radius: 0.32, height: 0.75, skirt: false, ribbon: false, silhouette: false, hop: { duration: 0.11, height: 0.14 }, name: 'tMazeTraveler' }, grid));
  world.add(wisp.group);
  wisp.carry(ghost);
  const walkerMaterial = world.own(emissiveMaterial(0xfff1cf, 1.6, { transparent: true, opacity: 0.8, depthWrite: false }));
  const walker = world.add(new THREE.Mesh(world.own(new THREE.IcosahedronGeometry(0.22, 2)), walkerMaterial));
  walker.visible = false;
  const extra = { court, grid, half, span, cells, cueLamp, cueGlyph, cueMaterial, ghost, ghostMaterial, glyphs, gates, wisp, walker, walkerMaterial };
  return world.seal<WorldCore & typeof extra>(extra);
}

type MazeWorld = ReturnType<typeof buildWorld>;

class TMazeScene extends StageScene<MazeWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'tMaze';
  private view!: SceneView;
  private outcomeStep: number | undefined;
  private position = -1;
  private cue = -1;
  private height = 0;
  private readonly cell = { x: 0, y: 0 };
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly ledger = { pos: 0, cue: 0, reward: 0, choice: '\u2014' };
  override legend: LegendSpec = {
    title: 'T-MAZE',
    rows: [
      { key: 'at', label: 'cell', tint: 0x9ff0a8, value: () => `${this.ledger.pos + 1} of ${this.world!.grid.h}` },
      { key: 'cue', label: 'cue', tint: 0x5ee0ff, value: () => (this.ledger.pos === 0 ? `lit: ${GLYPH[this.ledger.cue]!}` : `remember ${GLYPH[this.ledger.cue]!}`) },
      { key: 'choice', label: 'last choice', tint: 0xffd166, value: () => this.ledger.choice },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): MazeWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-world.span - 3, -0.5, -world.half - 4], max: [world.span + 3, 3.2, world.half + 3] }, [0, 0.4, -world.half * 0.15], { distance: Math.max(12, world.span * 2.6), azimuth: 0.55, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions() { return { agentCell: () => this.cell, goalCell: () => ({ x: 0, y: 0 }) }; }
  protected start(ctx: SceneContext, world: MazeWorld): void { this.height = ctx.tokens.wispHeight; world.wisp.setCell(0, world.grid.h - 1, world.grid); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<MazeWorld>): void {
    const pos = Math.round(frame.state.pose[TMAZE_POS]!), cue = Math.round(frame.state.pose[TMAZE_CUE]!);
    if (pos < 0 || pos >= world.grid.h || cue < 0 || cue >= world.gates.length) throw new Error('TMaze frame is outside its recorded geometry');
    this.cell.y = world.grid.h - 1 - pos;
    if (frame.scrubbed || frame.state.reset || Math.abs(world.wisp.cellY - this.cell.y) > 1) { world.wisp.setCell(0, this.cell.y, world.grid); world.wisp.snap(); }
    else if (world.wisp.cellY !== this.cell.y) world.wisp.hopTo(this.cell, time);
    world.wisp.update(frame.dtReal, time);
    if (this.position !== pos) {
      this.position = pos;
      for (let i = 0; i < world.grid.h; i++) world.cells.paint(i, i === pos ? 0xfff0b8 : i < pos ? 0xbfb8a0 : 0xffffff);
      world.cells.commit(false);
    }
    if (this.cue !== cue) {
      this.cue = cue;
      const tint = world.glyphs.tints[cue]!, shape = world.glyphs.shapes[cue]!;
      world.cueGlyph.geometry = shape;
      world.ghost.geometry = shape;
      for (const material of [world.cueMaterial, world.ghostMaterial, world.cueLamp.glass]) { material.color.set(tint); material.emissive.set(tint); }
      world.cueLamp.light.color.set(tint);
    }
    const atStart = pos === 0;
    world.cueLamp.lit(atStart ? 1 + 0.15 * Math.sin(time * 6) : 0);
    world.cueGlyph.visible = atStart;
    world.cueGlyph.rotation.y = time * 0.8;
    world.ghost.visible = !atStart;
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined ? null : frame.frameAt(step);
    const choice = outcome === null ? -1 : Math.trunc(outcome.actions[0]!) & (world.gates.length - 1);
    const hit = outcome !== null && outcome.reward > 0;
    this.ledger.pos = pos; this.ledger.cue = cue; this.ledger.reward = frame.state.reward; this.ledger.choice = choice < 0 ? '\u2014' : `${GLYPH[choice]!} gate \u2192 ${hit ? 'right' : 'wrong'}`;
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age), opening = age >= 0 ? Math.min(1, age * 2.5) * Math.exp(-age * 0.5) : 0;
    world.gates.forEach((entry, i) => {
      const chosen = i === choice;
      entry.door.lit(chosen ? (hit ? glow * 1.4 : -0.27 * glow) : 0);
      entry.door.open(chosen ? opening : 0);
      entry.material.emissiveIntensity = 0.8 + (chosen ? (hit ? 2.2 : -0.6) * glow : 0);
      entry.glyph.rotation.y = time * 0.35;
    });
    if (choice >= 0 && age >= 0 && age < WALK_SECONDS) {
      const t = age / WALK_SECONDS, ease = t * t * (3 - 2 * t), target = world.gates[choice]!.at;
      this.from.set(0, this.height, -world.half);
      this.to.set(target.x, this.height, target.z - 1);
      world.walker.position.lerpVectors(this.from, this.to, ease);
      world.walker.visible = true;
      setEmissiveIntensity(world.walkerMaterial, 1.6 * (1 - t));
    } else world.walker.visible = false;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && choice >= 0) {
      const entry = world.gates[choice]!;
      if (hit) pool.burst(entry.at, entry.tint, 0.65, time); else pool.sigh(0.35, entry.at, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal, [0, 0.4, Math.max(-world.half - 1.5, Math.min(world.half, world.wisp.position.z)) - 1.5]), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.position = -1; this.cue = -1; this.outcomeStep = undefined; }
}

export default ((() => new TMazeScene()) satisfies SceneFactory);
