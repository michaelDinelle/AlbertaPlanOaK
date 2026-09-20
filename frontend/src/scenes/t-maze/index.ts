import * as THREE from 'three';
import { makeLandscape } from '@kit/garden';
import { CELL } from '@kit/grid';
import { makeLightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { makeSceneView, type SceneView } from '@kit/view';
import { skyFromPalette } from '@kit/sky';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { makeWorld, type WorldCore } from '@kit/world';
import { Wisp } from '@kit/wisp';
import { lastEvent } from '@sim/events';
import { TMAZE_CUE, TMAZE_POS } from '@sim/envs/tmaze';
import { parameter } from '../../data/envs';

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const length = parameter(ctx.environment, 'length'), choices = ctx.environment.actions;
  if (length < 2 || choices !== 2 ** parameter(ctx.environment, 'cueBits')) throw new Error('TMaze geometry does not match its recorded configuration');
  const grid = { w: 1, h: length, walls: new Uint8Array(length), goal: { x: 0, y: 0 } };
  ctx.geometry = { grid };
  const corridor = (length - 1) * CELL, half = corridor / 2, span = 2.6 * choices + 1;
  const world = makeWorld('tMazeGarden'), palette = ctx.tokens.palettes.afternoon;
  const sky = world.time(world.own(skyFromPalette(palette, 'afternoon')));
  sky.setColors(0x86b8ce, 0xf3dfba, 0xbac6a1);
  world.add(sky.mesh);
  const rig = world.own(makeLightRig(palette, { sunDir: [-0.45, 0.85, 0.3], pointCount: 0, shadowMapSize: 2048 }));
  rig.sun.color.set(0xffe0a4); rig.sun.intensity = 2.5;
  rig.hemi.intensity = 1.8;
  rig.setShadowFocus({ x: 0, y: 0, z: -half * 0.2 }, Math.max(span + 3, half + 3));
  world.add(rig.group);
  scene.fog = new THREE.FogExp2(0xd0d5ad, 0.0025);
  const landscape = makeLandscape(world, ctx, { name: 'tMazeOrchard', halfX: span + 2, halfZ: half + 4, ground: 0xa5ad7a, grass: 0x7f9660, trees: [0xc69951, 0xaec57a, 0xc0ad67, 0x7d9d66], amplitude: 1.2 });
  const paving = world.own(oakMaterial({ color: 0xd8c9a7, roughness: 0.92, rimColor: 0xffe7af, rimStrength: 0.18 }));
  const stone = world.own(oakMaterial({ color: 0xbeb794, roughness: 0.88 }));
  const cap = world.own(oakMaterial({ color: 0xede0c1, roughness: 0.82 }));
  const bronze = world.own(oakMaterial({ color: 0x94784c, roughness: 0.5, metalness: 0.45 }));
  const foliage = world.own(oakMaterial({ color: 0x557647, roughness: 1, flatShading: true }));
  const unit = world.own(new THREE.BoxGeometry(1, 1, 1));
  const box = (material: THREE.Material, x: number, y: number, z: number, width: number, height: number, depth: number) => {
    const mesh = world.add(new THREE.Mesh(unit, material));
    mesh.position.set(x, y, z); mesh.scale.set(width, height, depth);
    mesh.castShadow = mesh.receiveShadow = true;
    return mesh;
  };
  box(paving, 0, -0.19, 1.3, 2.6, 0.38, corridor);
  box(paving, 0, -0.19, -half, span * 2 + 2.6, 0.38, 2.6);
  box(bronze, 0, 0.009, 1.3, 0.045, 0.018, corridor);
  box(bronze, 0, 0.009, -half, span * 2, 0.018, 0.045);
  for (const side of [-1, 1]) {
    box(stone, side * 1.55, 0.4, 1.3, 0.45, 0.8, corridor);
    box(cap, side * 1.55, 0.83, 1.3, 0.58, 0.1, corridor);
    box(foliage, side * 2.25, 0.45, 1.3, 0.8, 0.9, corridor);
    box(stone, side * (span / 2 + 1.45), 0.4, -half + 1.55, span - 0.3, 0.8, 0.45);
    box(cap, side * (span / 2 + 1.45), 0.83, -half + 1.55, span - 0.3, 0.1, 0.58);
    box(stone, side * (span + 1.55), 0.4, -half, 0.45, 0.8, 3.5);
    box(cap, side * (span + 1.55), 0.83, -half, 0.58, 0.1, 3.5);
  }
  const markers = world.own(new THREE.InstancedMesh(world.own(new THREE.BoxGeometry(0.8, 0.022, 0.7)), world.own(cap.clone()), length));
  const matrix = new THREE.Matrix4(), tint = new THREE.Color();
  for (let i = 0; i < length; i++) { markers.setMatrixAt(i, matrix.makeTranslation(0, 0.02, half - i * CELL)); markers.setColorAt(i, tint.set(0xd2c6a7)); }
  markers.computeBoundingSphere(); world.add(markers);
  const colors = [ctx.tokens.cueCyan, ctx.tokens.cueMagenta, ctx.tokens.celebrationGold, 0x9fea9a].slice(0, choices);
  const glyphs = [new THREE.BoxGeometry(0.55, 0.55, 0.2), new THREE.ConeGeometry(0.4, 0.65, 3), new THREE.SphereGeometry(0.3, 20, 12), new THREE.OctahedronGeometry(0.4)];
  glyphs.forEach(geometry => world.own(geometry));
  const gates = colors.map((color, choice) => {
    const x = (choice / (choices - 1) * 2 - 1) * span, z = -half - 1.4;
    const light = world.own(new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.7, roughness: 0.35 }));
    box(paving, x, -0.16, z - 0.4, 2.2, 0.32, 1.8);
    for (const side of [-1, 1]) { box(stone, x + side * 0.75, 1.2, z, 0.32, 2.4, 0.5); box(cap, x + side * 0.75, 2.38, z, 0.5, 0.16, 0.64); }
    box(cap, x, 2.6, z, 2.05, 0.3, 0.66);
    box(bronze, x, 0.06, z, 1.2, 0.08, 1.3);
    box(light, x, 0.12, z, 1.05, 0.04, 0.9);
    const glyph = world.add(new THREE.Mesh(glyphs[choice], light)); glyph.position.set(x, 1.3, z);
    return { light, glyph, position: new THREE.Vector3(x, 0.6, z) };
  });
  box(stone, -2.9, 0.45, half, 0.8, 0.9, 0.8);
  box(cap, -2.9, 0.94, half, 1.0, 0.12, 1.0);
  const cues = colors.map((color, choice) => {
    const material = world.own(new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.4 }));
    const mesh = world.add(new THREE.Mesh(glyphs[choice], material)); mesh.position.set(-2.9, 1.45, half); mesh.visible = false;
    return mesh;
  });
  const wisp = world.own(new Wisp({ color: 0xffefc7, radius: 0.23, skirt: false, ribbon: false, silhouette: false, hop: { duration: 0.1, height: 0.13 }, name: 'tMazeTraveler' }, grid));
  world.add(wisp.group);
  const extra = { grid, half, span, gates, cues, colors, wisp, markers, landscape, sky };
  return world.seal<WorldCore & typeof extra>(extra);
}

type MazeWorld = ReturnType<typeof buildWorld>;

class TMazeScene extends StageScene<MazeWorld> {
  protected readonly stageName = 'tMaze';
  private view!: SceneView;
  private outcomeStep: number | undefined;
  private position = -1;
  private readonly color = new THREE.Color();
  private readonly cell = { x: 0, y: 0 };

  protected buildWorld(ctx: SceneContext): MazeWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-world.span - 3, -0.5, -world.half - 4], max: [world.span + 3, 3.2, world.half + 3] }, [world.span * 0.4, 0.4, -0.6], { distance: Math.max(19, world.grid.h * 1.65, world.span * 3.5), azimuth: 0.38, polar: 0.58 });
    return this.view.options;
  }
  protected overlayOptions() { return { agentCell: () => this.cell, goalCell: () => ({ x: 0, y: 0 }) }; }
  protected start(_ctx: SceneContext, world: MazeWorld): void { world.wisp.setCell(0, world.grid.h - 1, world.grid); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<MazeWorld>): void {
    const pos = Math.round(frame.state.pose[TMAZE_POS]!), cue = Math.round(frame.state.pose[TMAZE_CUE]!);
    if (pos < 0 || pos >= world.grid.h || cue < 0 || cue >= world.gates.length) throw new Error('TMaze frame is outside its recorded geometry');
    this.cell.y = world.grid.h - 1 - pos;
    if (frame.scrubbed || frame.state.reset || Math.abs(world.wisp.cellY - this.cell.y) > 1) { world.wisp.setCell(0, this.cell.y, world.grid); world.wisp.snap(); }
    else if (world.wisp.cellY !== this.cell.y) world.wisp.hopTo(this.cell, time);
    world.wisp.update(frame.dtReal, time);
    if (this.position !== pos) {
      this.position = pos;
      for (let i = 0; i < world.grid.h; i++) world.markers.setColorAt(i, this.color.set(i === pos ? 0xffe3a0 : i < pos ? 0xbfb99a : 0xd2c6a7));
      world.markers.instanceColor!.needsUpdate = true;
    }
    world.cues.forEach((mesh, i) => { mesh.visible = pos === 0 && i === cue; mesh.rotation.y = time * 0.6; });
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined ? null : frame.frameAt(step);
    const choice = outcome === null ? -1 : Math.trunc(outcome.actions[0]!);
    if (outcome && (choice < 0 || choice >= world.gates.length)) throw new Error('TMaze choice is outside its recorded action space');
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = age >= 0 ? Math.exp(-age * 1.4) : 0;
    world.gates.forEach((gate, i) => { gate.light.emissiveIntensity = i === choice ? 0.7 + glow * (outcome!.reward > 0 ? 2.5 : -0.6) : 0.7; gate.glyph.rotation.y = time * 0.35; });
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && outcome) {
      if (outcome.reward > 0) pool.burst(world.gates[choice]!.position, world.colors[choice]!, 0.65, time);
      else pool.sigh(0.35, world.gates[choice]!.position, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.landscape.setView(this.view.update(rig, frame.dtReal), this.camera, world.sky);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.position = -1; this.outcomeStep = undefined; }
}

export default ((() => new TMazeScene()) satisfies SceneFactory);
