import * as THREE from 'three';
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { GoalBeacon, buildCellField, type CellFieldHandle } from '@kit/grid';
import { Wisp } from '@kit/wisp';
import type { GridCell } from '@sim/types';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { terraceShelves } from '@kit/terrace';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { planePatch } from '@kit/gen/patch';
import { TOKENS } from '@kit/tokens';
import type {
  FrameUpdate,
  SceneContext,
  SceneManifest,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { entryFor, type EnvCatalog } from '@sim/catalog';
import type { GridGeometry } from '@sim/types';
import { makeSemanticFrame, type SemanticFrame } from './frame';
import { detailFor, type Detail, type EnvSpec, type Stage } from './spec';

const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.5;
const PULSE_STRENGTH = 0.5;

function manifestFor(spec: EnvSpec): SceneManifest {
  return {
    envId: spec.envId as SceneManifest['envId'],
    slug: spec.envId,
    title: spec.title,
    logline: spec.logline,
    accent: spec.accent,
    tiers: ['hero', 'medium', 'thumb'],
    thumbFrame: { pose: new Array<number>(9).fill(0) },
    postConfig: {
      bloomStrength: spec.bloom,
      bloomThreshold: 0.85,
      fogTint: TOKENS.palettes[spec.palette].fog,
    },
  };
}

export function makeScene(spec: EnvSpec, catalog: EnvCatalog): SceneModule {
  const scene = new THREE.Scene();
  const manifest = manifestFor(spec);
  let camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  let ctx: SceneContext | null = null;
  let detail: Detail = detailFor('hero');
  let rig: OrbitRig | null = null;
  let lights: LightRig | null = null;
  let sky: SkyHandle | null = null;
  let cells: CellFieldHandle | null = null;
  let ground: THREE.Mesh | null = null;
  let groundMaterial: OakMaterial | null = null;
  let pool: CelebrationPool | null = null;
  let overlay: GroundTruthOverlay | null = null;
  let stage: Stage | null = null;
  let wisp: Wisp | null = null;
  let beacon: GoalBeacon | null = null;
  let semantic: { frame: SemanticFrame; bind(state: FrameUpdate['state']): void } | null = null;
  let debugOn = false;

  const group = new THREE.Group();
  group.name = `${spec.envId}:stage`;
  const burstAt = new THREE.Vector3();
  const agentCell: GridCell = { x: 0, y: 0 };
  const goalCell: GridCell = { x: 0, y: 0 };

  return {
    manifest,
    scene,
    get camera() {
      return camera;
    },

    async init(context: SceneContext): Promise<void> {
      ctx = context;
      scene.name = `${spec.envId}:scene`;
      detail = detailFor(context.tier);

      const palette = TOKENS.palettes[spec.palette];
      sky = skyFromPalette(palette, spec.palette, context.tier);
      scene.add(sky.mesh);
      applyFog(scene, palette, spec.fogExtent);

      lights = makeLightRig(palette, context.tier, {
        sunDir: sky.sunDirection,
        pointCount: 3,
      });
      scene.add(lights.group);

      const entry = entryFor(catalog, spec.envId);
      const grid = context.geometry.grid;
      const extent = spec.ground.extent ?? 40;
      if (spec.ground.kind !== 'none') {
        groundMaterial = oakMaterial({
          color: spec.ground.color ?? palette.groundA,
          roughness: 0.95,
          toonSteps: 3,
          tier: context.tier,
          name: `${spec.envId}:ground`,
        });
        const segments = Math.max(2, Math.round(48 * detail.segments));
        const plate = planePatch(
          extent,
          extent,
          segments,
          segments,
          spec.ground.height,
          `${spec.envId}:groundGeometry`,
        );
        ground = new THREE.Mesh(plate, groundMaterial);
        ground.position.y = -0.02;
        ground.receiveShadow = detail.shadows;
        scene.add(ground);
      }
      if (spec.ground.kind === 'cells' && grid) {
        cells = buildCellField(
          grid,
          {
            floorColor: 0xffffff,
            wallColor: 0xffffff,
            floorTint: (x, y) => spec.ground.floorTint?.(x, y) ?? palette.groundA,
            wallTint: (x, y) => spec.ground.wallTint?.(x, y) ?? palette.groundB,
            floorInset: 0.035,
            jitter: 0.06,
            seed: context.seed,
            name: `${spec.envId}:cells`,
          },
          context.tier,
        );
        if (spec.ground.terraced) terraceShelves(cells, grid);
        group.add(cells.group);
      }

      semantic = makeSemanticFrame(entry, {
        w: grid?.w ?? 0,
        h: grid?.h ?? 0,
        walls: grid?.walls ?? new Uint8Array(0),
        actionMask: (grid as GridGeometry | undefined)?.actionMask ?? new Uint8Array(0),
      });

      rig = makeOrbitRig({
        canvas: context.canvas,
        target: spec.camera.target,
        azimuth: spec.camera.azimuth,
        polar: spec.camera.polar,
        distance: spec.camera.distance,
        ...(spec.camera.follow !== undefined ? { follow: spec.camera.follow } : {}),
        fov: TOKENS.fovHero,
        near: 0.1,
        far: 900,
        aspect: context.size.height > 0 ? context.size.width / context.size.height : 16 / 9,
        name: `${spec.envId}:rig`,
      });
      camera = rig.camera;
      rig.snap();

      pool = new CelebrationPool({
        tier: context.tier,
        seed: context.seed,
        name: `${spec.envId}:celebrations`,
      });
      attachCelebrations(scene, pool);
      pool.setPixelRatio(context.size.dpr);

      if (entry.isGrid && grid) {
        wisp = new Wisp(
          {
            color: TOKENS.accents[spec.accent] ?? TOKENS.celebrationGold,
            tier: context.tier,
            radius: 0.21,
            hop: { duration: 0.11, height: 0.17 },
            name: `${spec.envId}:agent`,
          },
          grid,
        );
        scene.add(wisp.group);
        if (semantic.frame.has('gx')) {
          beacon = new GoalBeacon({
            tier: context.tier,
            color: TOKENS.celebrationGold,
            name: `${spec.envId}:goal`,
          });
          scene.add(beacon.group);
        }
      }

      scene.add(group);
      const built = spec.build({ ...context, detail, group, scene });
      stage = built ?? null;

      overlay = new GroundTruthOverlay(context.geometry, {
        camera,
        name: `${spec.envId}:groundTruth`,
      });
      scene.add(overlay.group);
      overlay.visible = debugOn;
    },

    update(update: FrameUpdate): void {
      if (!ctx || !rig || !pool || !semantic) return;
      const time = ctx.time();
      semantic.bind(update.state);
      const frame = semantic.frame;

      pool.update(update.dtReal, time);
      if (update.scrubbed) {
        pool.clear();
        rig.snap();
      }

      for (const ev of update.events) {
        if (ev.type === 'goal') {
          burstAt.copy(wisp !== null ? wisp.position : rig.target);
          burstAt.y += TOKENS.wispHeight;
          pool.burst(burstAt, TOKENS.celebrationGold, BURST_SCALE, time);
          pool.ripple(rig.target, TOKENS.celebrationGold, RIPPLE_SCALE, time);
          pool.pulse.fire(PULSE_STRENGTH);
        }
      }

      if (wisp !== null) {
        frame.cell('agent', agentCell);
        if (update.scrubbed) {
          wisp.setCell(agentCell.x, agentCell.y, frame.geometry);
          wisp.snap();
        } else if (agentCell.x !== wisp.cellX || agentCell.y !== wisp.cellY) {
          wisp.hopTo(agentCell, time);
        }
        wisp.update(update.dtReal, time);
        if (spec.camera.follow !== undefined) {
          rig.follow(wisp.position.x, spec.camera.target[1], wisp.position.z);
        }
      }
      if (beacon !== null) {
        frame.cell('goal', goalCell);
        beacon.setCell(goalCell.x, goalCell.y, frame.geometry);
        beacon.update(update.dtReal, time);
      }

      stage?.update?.(frame, update.dtReal, time);
      rig.update(update.dtReal);
      if (debugOn && overlay) overlay.setState(update.state);
    },

    resize(size: SceneSize): void {
      rig?.resize(size.height > 0 ? size.width / size.height : 16 / 9);
      pool?.setPixelRatio(size.dpr);
      stage?.resize?.(size.dpr);
    },

    setDebug(on: boolean): void {
      debugOn = on;
      if (overlay) overlay.visible = on;
    },

    dispose(): void {
      stage?.dispose?.();
      stage = null;
      wisp?.dispose();
      wisp = null;
      beacon?.dispose();
      beacon = null;
      overlay?.dispose();
      overlay = null;
      cells?.dispose();
      cells = null;
      ground?.geometry.dispose();
      ground = null;
      groundMaterial?.dispose();
      groundMaterial = null;
      lights?.dispose();
      lights = null;
      sky?.dispose();
      sky = null;
      rig?.dispose();
      rig = null;
      disposeCelebrations(scene);
      pool = null;
      semantic = null;
      group.clear();
      scene.clear();
      scene.fog = null;
      ctx = null;
    },
  };
}
