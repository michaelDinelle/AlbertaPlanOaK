import { parseCatalog, type EnvCatalog } from '@sim/catalog';
import type { EnvId } from '@shared/envs';
import { makeLaneStream } from '../theater/laneStream';
import { makeLaneSource } from '../theater/laneSource';
import { makeGhostLayer } from '@kit/ghosts';
import { TOKENS } from '@kit/tokens';
import { mountSceneHost, type SceneHostHandle } from '../theater/sceneHost';
import { ENVS } from '@shared/envs';
import type { ViewHandle } from '../router';
import { apiUrl } from '../api';

const GHOST_POSE = new Float32Array(9);

async function runDetail(runId: string): Promise<{ catalog: EnvCatalog; steps: number }> {
  const res = await fetch(apiUrl(`/runs/${encodeURIComponent(runId)}`));
  if (!res.ok) throw new Error(`run ${runId} is ${res.status}`);
  const detail = (await res.json()) as { meta: { envCatalog: unknown[]; steps: number } };
  return { catalog: parseCatalog(detail.meta.envCatalog), steps: detail.meta.steps };
}

function query(): URLSearchParams {
  const hash = typeof window === 'undefined' ? '' : window.location.hash;
  const at = hash.indexOf('?');
  return new URLSearchParams(at < 0 ? '' : hash.slice(at + 1));
}

export const mount = async (host: HTMLElement): Promise<ViewHandle> => {
  const q = query();
  const runId = q.get('run') ?? '';
  const envId = (q.get('env') ?? 'FourRooms-cont-v0') as EnvId;
  const focus = Number(q.get('lane') ?? '0') || 0;

  const shell = document.createElement('div');
  shell.className = 'view stage-root';
  shell.style.cssText = 'position:absolute;inset:0;';
  host.appendChild(shell);

  const { catalog, steps } = await runDetail(runId);
  const stream = makeLaneStream({ runId, envId, catalog });
  await stream.ensure(0);

  const geometryBytes = stream.geometryBytes(focus);
  if (geometryBytes === null) throw new Error(`lane ${focus} has no geometry`);

  const source = makeLaneSource({
    stream,
    lane: focus,
    seed: stream.seedAt(focus),
    totalSteps: steps,
    geometryBytes,
  });

  const info = ENVS[envId];
  const ghosts = makeGhostLayer({
    category: info.category,
    envId,
    entry: stream.entry,
    geometry: source.geometry,
    maxGhosts: Math.max(0, stream.laneCount - 1),
    tokens: TOKENS,
    accent: TOKENS.accents[info.slug] ?? TOKENS.celebrationGold,
    tier: 'hero',
  });

  const others: number[] = [];
  for (let lane = 0; lane < stream.laneCount; lane++) {
    if (lane !== focus) others.push(lane);
  }
  for (let i = 0; i < others.length; i++) ghosts.setRate(i, 0.55);

  let handle: SceneHostHandle | null = null;
  handle = await mountSceneHost(shell, {
    envId,
    catalog,
    source,
    tier: 'hero',
    speed: 16,
    ghosts: { layer: ghosts },
    onFrame(frame) {
      const step = Math.max(0, Math.floor(frame.position) - 1);
      void stream.ensure(step);
      for (let i = 0; i < others.length; i++) {
        const lane = others[i];
        if (lane === undefined) continue;
        if (!stream.pose(lane, step, GHOST_POSE)) continue;
        ghosts.setGhostFrame(i, GHOST_POSE, true);
      }
    },
  });
  handle.play();

  return {
    destroy() {
      handle?.destroy();
      handle = null;
      ghosts.dispose();
      stream.destroy();
      shell.remove();
    },
  };
};
