import * as THREE from 'three';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorld, cellToWorldVec } from '@kit/grid';
import { envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import { philoxDraw } from '@sim/philox';
import type { EventStream, GridCell, GridGeometry } from '@sim/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { buildCueCrystal, buildExitMouth, buildTraveler, type CueCrystalHandle, type ExitMouthHandle } from './actors';
import { buildWorld, CUE_CELL, CHOICE_CELL, readCue, readPos, resolveGrid, type WorldHandle } from './world';

interface Outcome {
  step: number;
  choice: 0 | 1;
  success: boolean;
}

class RecordedOutcomes {
  private stream: EventStream | null = null;
  private consumed = 0;
  private seed = Number.NaN;
  private goalStep = -Infinity;
  private readonly outcomes: Outcome[] = [];

  at(stream: EventStream, step: number, seed: number): Outcome | undefined {
    if (stream !== this.stream || stream.count < this.consumed || seed !== this.seed) {
      this.stream = stream;
      this.seed = seed;
      this.consumed = 0;
      this.goalStep = -Infinity;
      this.outcomes.length = 0;
    }
    for (; this.consumed < stream.count; this.consumed++) {
      const type = stream.types[stream.kinds[this.consumed] ?? 0];
      const eventStep = stream.steps[this.consumed] ?? -1;
      if (type === 'goal') this.goalStep = eventStep;
      if (type !== 'done') continue;
      const generation = this.outcomes.length;
      const cue = generation === 0 ? 0 : philoxDraw(seed >>> 0, 3, generation - 1, 0)[0] & 1;
      const success = this.goalStep === eventStep;
      this.outcomes.push({ step: eventStep, choice: (success ? cue : 1 - cue) as 0 | 1, success });
    }
    let low = 0;
    let high = this.outcomes.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if ((this.outcomes[mid]?.step ?? Infinity) <= step) low = mid + 1;
      else high = mid;
    }
    return this.outcomes[low - 1];
  }
}

class LimestoneLakeScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'limestoneLake';
  private view: SceneView | null = null;
  private grid!: GridGeometry;
  private traveler: ReturnType<typeof buildTraveler> | null = null;
  private wisp: Wisp | null = null;
  private cue: CueCrystalHandle | null = null;
  private readonly choices: ExitMouthHandle[] = [];
  private readonly outcomes = new RecordedOutcomes();
  private shownStep = -Infinity;
  private outcomeAt = -Infinity;
  private selected: 0 | 1 = 0;
  private succeeded = false;
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: CHOICE_CELL, y: 0 };
  private readonly effectPosition = new THREE.Vector3();

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.grid = resolveGrid(ctx.geometry);
    const start = cellToWorld(0, 0, this.grid)[0];
    const end = cellToWorld(CHOICE_CELL, 0, this.grid)[0];
    this.view = makeSceneView(ctx, { min: [start - 1.5, -0.4, -4], max: [end + 2.6, 3.5, 4] }, [0, 1, 0], { distance: 33.5, azimuth: -0.45, polar: 1.36 });
    return buildWorld(ctx, this.scene, this.grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const colors = [ctx.tokens.cueCyan, ctx.tokens.cueMagenta] as const;
    const wisp = new Wisp({ color: 0xffe4b0, radius: 0.2, skirt: false, ribbon: false, silhouette: false, hop: { duration: 0.11, height: 0.1 } }, this.grid);
    wisp.core.visible = false;
    const halo = wisp.group.getObjectByName('oakWispHalo');
    if (halo) halo.visible = false;
    this.wisp = wisp;
    this.traveler = buildTraveler();
    wisp.avatar.add(this.traveler.group);
    this.scene.add(wisp.group);
    this.cue = buildCueCrystal({ skin: world.skin, cueColors: colors });
    cellToWorldVec(CUE_CELL, 0, this.grid, this.cue.root.position);
    this.cue.root.position.z = -1.6;
    this.scene.add(this.cue.group);
    for (const choice of [0, 1] as const) {
      const alcove = buildExitMouth({ skin: world.skin, color: colors[choice], choice });
      cellToWorldVec(CHOICE_CELL, 0, this.grid, alcove.root.position);
      alcove.root.position.x += 0.85;
      alcove.root.position.z = choice === 0 ? -2.6 : 2.6;
      this.choices.push(alcove);
      this.scene.add(alcove.group);
    }
    world.rig.points[0]?.color.set(0xffe4b0);
    world.rig.points[1]?.color.set(colors[0]);
    world.rig.points[2]?.color.set(colors[1]);
  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Cavern view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentCell: state => { this.agentCell.x = readPos(state); return this.agentCell; },
      goalCell: () => this.goalCell,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.wisp?.setCell(CUE_CELL, 0, this.grid);
    world.setTime(0);
    if (this.rig && this.view) world.setView(this.view.update(this.rig, 0), this.camera);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const cue = this.cue;
    if (!wisp || !cue || !this.view) return;
    pool.update(frame.dtReal, time);
    const pos = readPos(frame.state);
    const outcome = this.outcomes.at(frame.eventLog, frame.state.step, ctx.seed);
    if (frame.scrubbed) {
      pool.clear();
      this.outcomeAt = -Infinity;
      this.shownStep = outcome?.step ?? -Infinity;
      wisp.setCell(pos, 0, this.grid);
      wisp.snap();
    }
    if (outcome) {
      const newlyShown = outcome.step !== this.shownStep && !frame.scrubbed;
      this.shownStep = outcome.step;
      this.selected = outcome.choice;
      this.succeeded = outcome.success;
      this.outcomeAt = (outcome.step + 1) * frame.state.simTime / frame.state.step;
      const alcove = this.choices[outcome.choice];
      if (alcove && newlyShown) {
        this.effectPosition.copy(alcove.root.position);
        this.effectPosition.y = 1.4;
        if (outcome.success) pool.burst(this.effectPosition, outcome.choice === 0 ? ctx.tokens.cueCyan : ctx.tokens.cueMagenta, 0.6, time);
        else pool.sigh(0.55, this.effectPosition, time);
      }
    }
    if (Math.abs(pos - wisp.cellX) > 1 || frame.events.some(event => event.type === 'teleport')) {
      wisp.setCell(pos, 0, this.grid);
      wisp.snap();
    } else if (pos !== wisp.cellX) {
      this.agentCell.x = pos;
      wisp.hopTo(this.agentCell, time);
    }
    wisp.update(frame.dtReal, time);
    cue.setCue(readCue(frame.state));
    cue.setGlow(pos === CUE_CELL ? 1 : 0);
    cue.setTime(time);
    const outcomeAge = time - this.outcomeAt;
    const flare = outcomeAge >= 0 ? envelope(Math.max(0, outcomeAge - 0.6), 2.2, 1.4) : 0;
    for (let i = 0; i < this.choices.length; i++) {
      const gain = 1 + (i === this.selected ? flare * (this.succeeded ? 2.5 : 0.8) : 0);
      this.choices[i]?.setGlow(gain, this.succeeded);
      this.choices[i]?.setTime(time);
      const light = world.rig.points[i + 1];
      const alcove = this.choices[i];
      if (light && alcove) {
        light.position.copy(alcove.root.position);
        light.position.y = 1.5;
        light.intensity = 1.1 * (i === this.selected && !this.succeeded ? 1 - flare * 0.85 : gain);
      }
    }
    const light = world.rig.points[0];
    if (light) { light.position.copy(wisp.position); light.position.y += 0.6; light.intensity = 1.6; }
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.view && this.rig && this.world) {
      this.view.resize(size, this.rig);
      this.world.setView(this.view.update(this.rig, 0), this.camera);
    }
  }

  protected drop(): void {
    for (const choice of this.choices) choice.dispose();
    this.choices.length = 0;
    this.cue?.dispose();
    this.cue = null;
    this.traveler?.dispose();
    this.traveler = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.outcomeAt = -Infinity;
    this.shownStep = -Infinity;
  }
}

export default ((() => new LimestoneLakeScene()) satisfies SceneFactory);
