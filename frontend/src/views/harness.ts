
import type { ViewMount } from '../router';
import { mountHarness, type HarnessHandle, type HarnessOptions } from '@kit/harness';
import type { QualityTier } from '@kit/types';

const TIERS: readonly string[] = ['hero', 'medium', 'thumb'];

function optionsFromQuery(): HarnessOptions {
  const opts: HarnessOptions = {};
  if (typeof window === 'undefined') return opts;
  const hash = window.location.hash;
  const qIndex = hash.indexOf('?');
  if (qIndex < 0) return opts;
  const params = new URLSearchParams(hash.slice(qIndex + 1));
  const tier = params.get('tier');
  if (tier && TIERS.includes(tier)) opts.tier = tier as QualityTier;
  const run = params.get('run');
  if (run) opts.runId = run;
  const steps = Number(params.get('steps'));
  if (Number.isFinite(steps) && steps > 0) opts.maxSteps = Math.floor(steps);
  const at = Number(params.get('at'));
  if (Number.isFinite(at) && at >= 0) opts.at = at;
  if (params.get('pause') === '1') opts.paused = true;
  if (params.get('debug') === '1') opts.debug = true;
  return opts;
}

export const mount: ViewMount = async (el, params) => {
  const slug = params.slug ?? '';
  const shell = document.createElement('div');
  shell.style.position = 'relative';
  shell.style.width = '100%';
  shell.style.height = '100vh';
  el.appendChild(shell);

  let handle: HarnessHandle | null = null;
  try {
    handle = await mountHarness(shell, slug, optionsFromQuery());
  } catch (cause) {
    const err = document.createElement('div');
    err.className = 'oak-err';
    err.style.padding = '32px';
    err.style.color = '#ffbe5c';
    err.textContent = `harness failed for "${slug}": ${String(cause)}`;
    shell.appendChild(err);
  }

  return {
    destroy() {
      handle?.destroy();
      handle = null;
      shell.remove();
    },
  };
};
