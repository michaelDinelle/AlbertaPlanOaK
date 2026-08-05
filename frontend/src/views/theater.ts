// #/theater — the hero replay view the whole app funnels into.
//
// Two modes behind one route, chosen by the URL and nothing else:
//
//   #/theater                        the picker — run, then env, then seed
//   #/theater/:id/:envId/:seed       playback — one seed, full bleed
//
// The URL is the source of truth. Picking a seed navigates; the router remounts;
// `destroy()` disposes the scene, the composer, the renderer, the rAF loop and
// the in-flight fetch. There is no hidden view state to get out of sync.
//
// The moving parts live next door so they can be tested without a GPU:
// `theater/loader` (data, deep link, progress, ticks, damage), `theater/player`
// (transport rules, tier governor) and `theater/sceneHost` (the WebGL host).
// This file is chrome: DOM, layout, and wiring.

import type { ViewHandle, ViewMount } from '../router';
import type { RecordingRef, RunDetail, RunIndexEntry } from '@shared/types';
import { ENVS, type EnvId } from '@shared/envs';
import type { QualityTier } from '@kit/types';
import type { SimEvent } from '@sim/types';
import { badge, el, emptyState, errorState, eyebrow, spinner, viewShell } from '../components/table';
import { envName, formatCount, formatStep, loadRunDetail, runIndex } from '../state';
import { accentCss } from './gallery';
import {
  DEFAULT_THEATER_RUN,
  envChips,
  loadTheaterRecording,
  makeProgress,
  parseTheaterParams,
  parseTheaterQuery,
  pickerRuns,
  recordingRef,
  seedsFor,
  seekTargetForTick,
  theaterHref,
  TICK_KINDS,
  type EnvChip,
  type LoadProgress,
  type TheaterRecording,
  type TheaterTarget,
  type TickKind,
  type TimelineTick,
} from '../theater/loader';
import {
  SPEEDS,
  TICKER_LIMIT,
  clampPosition,
  cycleTier,
  formatClock,
  keyAction,
  nextSpeed,
  pushTicker,
  stepPosition,
  type TickerEntry,
} from '../theater/player';
import { mountSceneHost, type HostFrameInfo, type SceneHostHandle } from '../theater/sceneHost';

const TIER_OPTIONS: readonly QualityTier[] = ['hero', 'medium', 'thumb'];

/**
 * CSS custom properties cannot be assigned through `el`'s style object —
 * `Object.assign(node.style, {'--x': v})` writes a JS expando, not a CSS
 * variable. Only `setProperty` reaches the cascade.
 */
function tint<T extends HTMLElement>(node: T, color: string): T {
  node.style.setProperty('--pick-accent', color);
  return node;
}

// ---------------------------------------------------------------------------
// styles — scoped to the view, injected once (the app stylesheet is not ours)
// ---------------------------------------------------------------------------

const CSS = `
.theater-root { height: 100vh; overflow: hidden; }
/* .theater-root is a flex item ('.view' is flex:1), so a plain height is overridden by
   the flex sizing — max-height actually clamps the used size and keeps the transport
   bar on-screen (E2E finding: without this the controls render below the fold). */
.theater-root.is-playback { max-height: 100vh; }
.theater-root > .view-body { max-width: none; padding: 0; min-height: 0; display: flex; flex-direction: column; }
.theater-root.is-picker { height: auto; overflow: visible; }
.theater-root.is-picker > .view-body { max-width: var(--maxw); margin: 0 auto; padding: 26px var(--pad) 72px; display: block; }

/* ---------- picker ---------- */
.theater-lede { color: var(--text-dim); max-width: 68ch; margin: 6px 0 22px; }
.theater-columns { display: grid; grid-template-columns: minmax(210px, 290px) minmax(230px, 1fr) minmax(190px, 300px); gap: 22px; align-items: start; }
@media (max-width: 900px) { .theater-columns { grid-template-columns: 1fr; } }
.theater-col { min-width: 0; display: flex; flex-direction: column; gap: 10px; }
.theater-list { display: flex; flex-direction: column; gap: 4px; max-height: 62vh; overflow-y: auto; padding-right: 2px; }
.pick {
  display: flex; flex-direction: column; gap: 3px; width: 100%; text-align: left;
  padding: 8px 10px; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: var(--r); cursor: pointer; color: var(--text);
}
.pick:hover { border-color: var(--border-strong); background: var(--bg-hover); text-decoration: none; }
.pick.is-on { border-color: var(--accent); background: var(--bg-hover); }
.pick-name { font-family: var(--mono); font-size: 12px; letter-spacing: -0.01em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pick-meta { font-family: var(--mono); font-size: 10px; color: var(--text-faint); letter-spacing: 0.04em; }
.pick-row { display: flex; align-items: baseline; gap: 8px; }
.pick-row .pick-name { flex: 1; min-width: 0; }
.pick-row .badge { flex: none; }
.pick-count { font-family: var(--mono); font-size: 10px; color: var(--text-faint); }
.pick-env { border-left: 3px solid var(--pick-accent, var(--border-strong)); }
.pick-env.is-on { border-left-color: var(--pick-accent, var(--accent)); }
.pick-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--pick-accent, var(--text-faint)); flex: none; }
.theater-seeds { display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr)); gap: 5px; max-height: 62vh; overflow-y: auto; }
.seed-pick {
  display: flex; flex-direction: column; gap: 1px; padding: 7px 9px; border-radius: var(--r);
  background: var(--bg-raised); border: 1px solid var(--border); color: var(--text);
}
.seed-pick:hover { border-color: var(--pick-accent, var(--border-strong)); background: var(--bg-hover); text-decoration: none; }
.seed-pick-num { font-family: var(--mono); font-size: 11.5px; }
.seed-pick-steps { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.1em; color: var(--text-faint); }

/* ---------- playback ---------- */
.theater { display: flex; flex: 1; min-height: 0; min-width: 0; }
.theater-main { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.theater-stage { position: relative; flex: 1; min-height: 0; overflow: hidden; background: #05070c; }
.theater-canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }

/* The stage tag sits over live art whose brightness we do not control (a noon
   courtyard is nearly white). A scrim, not a heavier font, is what keeps it
   legible without competing with the scene. */
.stage-tag {
  position: absolute; top: 0; left: 0; right: 0; padding: 12px 16px 26px; pointer-events: none;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.06em; color: rgba(233, 240, 250, 0.72);
  text-shadow: 0 1px 4px rgba(0, 0, 0, 0.85);
  background: linear-gradient(180deg, rgba(5, 7, 12, 0.55) 0%, rgba(5, 7, 12, 0) 100%);
}
.stage-tag b { display: block; font-weight: 500; font-size: 13px; letter-spacing: 0.01em; color: #f2f7ff; }

.stage-cover {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px; padding: 32px;
  background: radial-gradient(circle at 50% 45%, #0d1420 0%, #05070c 72%);
  text-align: center;
}
.stage-cover[hidden] { display: none; }
.cover-title { font-family: var(--mono); font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-dim); }
.cover-label { font-family: var(--mono); font-size: 11.5px; color: var(--text-faint); font-variant-numeric: tabular-nums; }
.cover-bar { width: min(360px, 60%); height: 3px; background: var(--bg-hover); border-radius: 2px; overflow: hidden; }
.cover-fill { height: 100%; width: 0%; background: var(--accent); transition: width 120ms linear; }
.cover-err { max-width: 60ch; }

.theater-banner {
  display: flex; align-items: center; gap: 8px; padding: 7px 14px;
  background: color-mix(in srgb, var(--viz-crit) 14%, var(--bg-raised));
  border-top: 1px solid color-mix(in srgb, var(--viz-crit) 45%, transparent);
  font-size: 12px; color: var(--text);
}
.theater-banner[hidden] { display: none; }
.theater-banner b { font-weight: 600; }

.theater-transport {
  display: flex; align-items: center; gap: 10px; flex: none;
  padding: 9px 14px; border-top: 1px solid var(--border); background: var(--bg-raised);
}
.tbtn {
  font-family: var(--mono); font-size: 11px; padding: 4px 9px; min-width: 34px;
  background: var(--bg); border: 1px solid var(--border); border-radius: var(--r);
  color: var(--text-dim); cursor: pointer;
}
.tbtn:hover { color: var(--text); border-color: var(--border-strong); }
.tbtn.is-on { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 55%, transparent); }
.tbtn-play { min-width: 58px; color: var(--text); }
.tsel {
  font-family: var(--mono); font-size: 11px; padding: 4px 6px; cursor: pointer;
  background: var(--bg); border: 1px solid var(--border); border-radius: var(--r); color: var(--text-dim);
}
.tsel:hover { color: var(--text); border-color: var(--border-strong); }
.treadout {
  font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  font-variant-numeric: tabular-nums; white-space: nowrap; flex: none;
}
.tsep { width: 1px; align-self: stretch; background: var(--border); flex: none; }

/* scrub bar + event ticks */
.scrub { position: relative; flex: 1; min-width: 120px; height: 26px; }
.scrub-track { position: absolute; left: 0; right: 0; top: 50%; height: 3px; margin-top: -1.5px; background: var(--bg-hover); border-radius: 2px; }
.scrub-fill { position: absolute; left: 0; top: 50%; height: 3px; margin-top: -1.5px; width: 0%; background: var(--accent); border-radius: 2px; }
.scrub-input {
  position: absolute; inset: 0; width: 100%; height: 100%; margin: 0;
  opacity: 0; cursor: pointer; z-index: 1;
}
.scrub-input:focus-visible + .scrub-track { box-shadow: 0 0 0 2px var(--accent); }
.scrub-head { position: absolute; top: 50%; width: 2px; height: 13px; margin: -6.5px 0 0 -1px; background: var(--text); border-radius: 1px; pointer-events: none; z-index: 3; }
.scrub-ticks { position: absolute; inset: 0; pointer-events: none; }
.tick {
  position: absolute; top: 50%; width: 3px; height: 15px; margin: -7.5px 0 0 -1.5px;
  border: 0; padding: 0; border-radius: 1px; cursor: pointer; z-index: 2; opacity: 0.78;
  pointer-events: auto;
}
.tick:hover, .tick:focus-visible { opacity: 1; height: 21px; margin-top: -10.5px; }
.tick-goal { background: var(--gold); }
.tick-door { background: var(--viz-ramp-4); }
.tick-key { background: #ffc94d; }
.tick-cue { background: #5ee0ff; }
.tick-done { background: var(--viz-crit); }
.tick-other { background: var(--text-faint); }
.scrub-tip {
  position: absolute; bottom: 26px; transform: translateX(-50%); z-index: 6;
  padding: 5px 8px; background: var(--bg-sunken); border: 1px solid var(--border-strong);
  border-radius: var(--r); font-family: var(--mono); font-size: 10.5px; color: var(--text);
  white-space: nowrap; pointer-events: none;
}
.scrub-tip[hidden] { display: none; }

/* ---------- side strip ---------- */
.theater-strip {
  width: 294px; flex: none; min-height: 0; overflow-y: auto;
  border-left: 1px solid var(--border); background: var(--bg-raised);
  padding: 16px 16px 28px; display: flex; flex-direction: column; gap: 18px;
}
@media (max-width: 1180px) { .theater-strip { width: 244px; } }
@media (max-width: 940px) { .theater-strip { display: none; } }
.strip-block { display: flex; flex-direction: column; gap: 7px; }
.strip-title { font-size: 16px; letter-spacing: -0.01em; color: var(--text); }
.strip-logline { font-size: 12px; color: var(--text-dim); line-height: 1.55; }
.strip-kv { display: flex; align-items: baseline; gap: 8px; font-family: var(--mono); font-size: 11px; }
.strip-kv dt { color: var(--text-faint); letter-spacing: 0.08em; text-transform: uppercase; font-size: 9.5px; width: 62px; flex: none; }
.strip-kv dd { margin: 0; color: var(--text); font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
.strip-warn {
  padding: 8px 10px; border-radius: var(--r); font-size: 11.5px; line-height: 1.5;
  background: color-mix(in srgb, var(--viz-crit) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--viz-crit) 40%, transparent); color: var(--text);
}
.strip-warn[hidden] { display: none; }
.ticker { display: flex; flex-direction: column; gap: 2px; font-family: var(--mono); font-size: 10.5px; }
.ticker-row { display: flex; align-items: baseline; gap: 8px; color: var(--text-dim); }
.ticker-row:last-child { color: var(--text); }
.ticker-step { color: var(--text-faint); font-variant-numeric: tabular-nums; min-width: 62px; }
.ticker-dot { width: 6px; height: 6px; border-radius: 50%; flex: none; align-self: center; }
.ticker-empty { font-family: var(--mono); font-size: 10.5px; color: var(--text-faint); }
.strip-keys { display: flex; flex-wrap: wrap; gap: 4px 10px; font-family: var(--mono); font-size: 10px; color: var(--text-faint); }
.strip-keys kbd {
  font: inherit; padding: 1px 5px; border-radius: 2px; color: var(--text-dim);
  background: var(--bg); border: 1px solid var(--border);
}
.strip-accent { height: 3px; border-radius: 2px; background: var(--pick-accent, var(--accent)); }
`;

function ensureStyle(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById('oak-theater-style')) return;
  const style = document.createElement('style');
  style.id = 'oak-theater-style';
  style.textContent = CSS;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// route entry
// ---------------------------------------------------------------------------

export const mount: ViewMount = (host, params) => {
  ensureStyle();
  const target = parseTheaterParams(params);
  return target ? mountPlayback(host, target) : mountPicker(host);
};

// ---------------------------------------------------------------------------
// picker
// ---------------------------------------------------------------------------

function mountPicker(host: HTMLElement): ViewHandle {
  const { root, body } = viewShell('theater');
  root.classList.add('theater-root', 'is-picker');
  host.append(root);
  let destroyed = false;

  body.append(
    el('div', { class: 'view-head' }, [
      eyebrow('Theater'),
      el('h1', { class: 'view-title', text: 'Replay a seed' }),
      el('p', {
        class: 'theater-lede',
        text:
          'Every recorded seed is one continuous stream of real experience — one observation, one action, one reward per step. ' +
          'Pick a run, an environment, and a seed; the scene plays that stream exactly as the fleet lived it.',
      }),
    ]),
  );

  const status = el('div', { class: 'view-status' }, [spinner('Loading runs')]);
  body.append(status);

  const runList = el('div', { class: 'theater-list' });
  const envList = el('div', { class: 'theater-list' });
  const seedList = el('div', { class: 'theater-seeds' });
  const envCount = el('span', { class: 'eyebrow-count' });
  const seedCount = el('span', { class: 'eyebrow-count' });

  const columns = el('div', { class: 'theater-columns' }, [
    el('div', { class: 'theater-col' }, [eyebrow('Runs with recordings'), runList]),
    el('div', { class: 'theater-col' }, [eyebrow('Environments', envCount), envList]),
    el('div', { class: 'theater-col' }, [eyebrow('Seeds', seedCount), seedList]),
  ]);
  columns.hidden = true;
  body.append(columns);

  let runs: RunIndexEntry[] = [];
  let selectedRun: string | null = null;
  let selectedEnv: EnvId | null = null;
  let recordings: readonly RecordingRef[] = [];
  let chips: EnvChip[] = [];

  function paintRuns(): void {
    runList.replaceChildren(
      ...runs.map((entry) => {
        const mark = entry.id === DEFAULT_THEATER_RUN
          ? badge('demo', 'good', 'The 16-env demo run')
          : entry.role
            ? badge(entry.role, entry.role)
            : null;
        const button = el('button', { class: `pick${entry.id === selectedRun ? ' is-on' : ''}`, type: 'button' }, [
          el('div', { class: 'pick-row' }, [
            el('div', { class: 'pick-name', text: entry.id, title: entry.id }),
            mark,
          ]),
          el('div', {
            class: 'pick-meta',
            text: [
              entry.envIds.length === 1 ? envName(entry.envIds[0] ?? '') : `${entry.envIds.length} envs`,
              `B ${entry.batch}`,
              `${formatStep(entry.steps)} steps`,
            ].join('  ·  '),
          }),
        ]);
        button.addEventListener('click', () => selectRun(entry.id));
        return button;
      }),
    );
  }

  function paintEnvs(): void {
    envCount.textContent = chips.length > 0 ? `${chips.length}` : '';
    if (chips.length === 0) {
      envList.replaceChildren(el('div', { class: 'pick-meta', text: 'no recorded envs' }));
      return;
    }
    envList.replaceChildren(
      ...chips.map((chip) => {
        const button = tint(
          el('button', {
            class: `pick pick-env${chip.envId === selectedEnv ? ' is-on' : ''}`,
            type: 'button',
          }, [
            el('div', { class: 'pick-row' }, [
              el('span', { class: 'pick-dot' }),
              el('span', { class: 'pick-name', text: chip.name }),
              el('span', { class: 'pick-count', text: `${chip.count}` }),
            ]),
            el('div', { class: 'pick-meta', text: `${ENVS[chip.envId].category} · ${ENVS[chip.envId].controlHz} Hz` }),
          ]),
          accentCss(chip.slug),
        );
        button.addEventListener('click', () => selectEnv(chip.envId));
        return button;
      }),
    );
  }

  function paintSeeds(): void {
    if (!selectedRun || !selectedEnv) {
      seedCount.textContent = '';
      seedList.replaceChildren(el('div', { class: 'pick-meta', text: 'pick an environment' }));
      return;
    }
    const envId = selectedEnv;
    const runId = selectedRun;
    const seeds = seedsFor(recordings, envId);
    const accent = accentCss(ENVS[envId].slug);
    seedCount.textContent = `${seeds.length}`;
    seedList.replaceChildren(
      ...seeds.map((seed) => {
        const ref = recordingRef(recordings, envId, seed);
        return tint(
          el('a', {
            class: 'seed-pick',
            href: theaterHref({ runId, envId, seed }),
            title: `Play ${ENVS[envId].name} seed ${seed}`,
          }, [
            el('span', { class: 'seed-pick-num', text: `seed ${seed}` }),
            el('span', { class: 'seed-pick-steps', text: ref ? formatStep(ref.steps) : '—' }),
          ]),
          accent,
        );
      }),
    );
  }

  function selectEnv(envId: EnvId): void {
    selectedEnv = envId;
    paintEnvs();
    paintSeeds();
  }

  function selectRun(id: string): void {
    selectedRun = id;
    paintRuns();
    chips = [];
    recordings = [];
    envList.replaceChildren(spinner('Reading recordings'));
    seedList.replaceChildren();
    loadRunDetail(id).then(
      (detail: RunDetail) => {
        if (destroyed || selectedRun !== id) return;
        recordings = detail.recordings ?? [];
        chips = envChips(recordings);
        selectedEnv = chips[0]?.envId ?? null;
        paintEnvs();
        paintSeeds();
      },
      (error: unknown) => {
        if (destroyed || selectedRun !== id) return;
        envList.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
      },
    );
  }

  function start(force: boolean): void {
    runIndex.load(force).then(
      (entries) => {
        if (destroyed) return;
        status.remove();
        runs = pickerRuns(entries);
        if (runs.length === 0) {
          body.append(
            emptyState(
              'No run has recordings yet',
              'Re-run the fleet with OAK_RECORD_ACTIONS=1 (or OAK_RECORD_SEEDS=3,8) to write the actions.bin streams the Theater replays.',
            ),
          );
          return;
        }
        columns.hidden = false;
        paintRuns();
        selectRun(runs[0]?.id ?? DEFAULT_THEATER_RUN);
      },
      (error: unknown) => {
        if (destroyed) return;
        if (status.parentNode === null) body.append(status);
        status.replaceChildren(
          errorState(error instanceof Error ? error.message : String(error), () => {
            status.replaceChildren(spinner('Retrying'));
            start(true);
          }),
        );
      },
    );
  }
  start(false);

  return {
    destroy() {
      destroyed = true;
      root.remove();
    },
  };
}

// ---------------------------------------------------------------------------
// playback
// ---------------------------------------------------------------------------

function kv(label: string, value: string, title?: string): HTMLElement {
  return el('div', { class: 'strip-kv' }, [
    el('dt', { text: label }),
    el('dd', { text: value, ...(title ? { title } : {}) }),
  ]);
}

function mountPlayback(host: HTMLElement, target: TheaterTarget): ViewHandle {
  const query = parseTheaterQuery(typeof window === 'undefined' ? '' : window.location.hash);
  const info = ENVS[target.envId];
  const accent = accentCss(info.slug);

  const { root, body } = viewShell('theater');
  root.classList.add('theater-root', 'is-playback');
  host.append(root);

  const abort = new AbortController();
  let destroyed = false;
  let sceneHost: SceneHostHandle | null = null;

  // ---- stage ----
  const stage = el('div', { class: 'theater-stage' });
  const stageTag = el('div', { class: 'stage-tag' }, [
    el('b', { text: info.name }),
    el('span', { text: `${target.runId} · seed ${target.seed}` }),
  ]);
  const coverTitle = el('div', { class: 'cover-title', text: 'Loading seed' });
  const coverFill = el('div', { class: 'cover-fill' });
  const coverBar = el('div', { class: 'cover-bar' }, [coverFill]);
  const coverLabel = el('div', { class: 'cover-label', text: 'reading recording meta' });
  const cover = el('div', { class: 'stage-cover' }, [coverTitle, coverBar, coverLabel]);
  stage.append(stageTag, cover);

  // ---- transport ----
  const playButton = el('button', { class: 'tbtn tbtn-play', type: 'button', text: 'pause' });
  const backButton = el('button', { class: 'tbtn', type: 'button', text: '‹', title: 'Step back one frame (←)' });
  const fwdButton = el('button', { class: 'tbtn', type: 'button', text: '›', title: 'Step on one frame (→)' });
  const speedSelect = el('select', { class: 'tsel', title: 'Playback speed ([ and ])' });
  for (const speed of SPEEDS) speedSelect.append(el('option', { value: String(speed), text: `${speed}×` }));
  speedSelect.value = '1';

  const scrubInput = el('input', { class: 'scrub-input', type: 'range' });
  scrubInput.min = '0';
  scrubInput.max = '1';
  scrubInput.step = 'any';
  scrubInput.value = '0';
  scrubInput.setAttribute('aria-label', 'Scrub');
  const scrubFill = el('div', { class: 'scrub-fill' });
  const scrubHead = el('div', { class: 'scrub-head' });
  const scrubTip = el('div', { class: 'scrub-tip' });
  scrubTip.hidden = true;
  const scrubTicks = el('div', { class: 'scrub-ticks' });
  const scrub = el('div', { class: 'scrub' }, [
    scrubInput,
    el('div', { class: 'scrub-track' }),
    scrubFill,
    scrubTicks,
    scrubHead,
    scrubTip,
  ]);

  const readout = el('div', { class: 'treadout', text: 'step — · 0:00.0 · f0/0' });
  const tierSelect = el('select', { class: 'tsel', title: 'Quality tier (t)' });
  for (const tier of TIER_OPTIONS) tierSelect.append(el('option', { value: tier, text: tier }));
  tierSelect.value = query.tier ?? 'hero';
  const debugButton = el('button', { class: 'tbtn', type: 'button', text: 'g', title: 'Ground-truth overlay (g)' });

  const transport = el('div', { class: 'theater-transport' }, [
    playButton,
    backButton,
    fwdButton,
    speedSelect,
    scrub,
    readout,
    el('div', { class: 'tsep' }),
    tierSelect,
    debugButton,
  ]);

  const banner = el('div', { class: 'theater-banner' }, [
    el('b', { text: 'Recording damaged.' }),
    el('span', { class: 'banner-text' }),
  ]);
  banner.hidden = true;

  // ---- side strip ----
  const stripTitle = el('div', { class: 'strip-title', text: info.name });
  const stripLogline = el('div', { class: 'strip-logline', text: '' });
  const stripFacts = el('div', { class: 'strip-block' });
  const stripHealth = el('div', { class: 'strip-warn' });
  stripHealth.hidden = true;
  const stripStats = el('div', { class: 'strip-block' });
  const tickerList = el('div', { class: 'ticker' }, [el('div', { class: 'ticker-empty', text: 'no events yet' })]);

  const strip = el('aside', { class: 'theater-strip' }, [
    el('div', { class: 'strip-block' }, [
      tint(el('div', { class: 'strip-accent' }), accent),
      eyebrow('Now playing'),
      stripTitle,
      stripLogline,
    ]),
    stripHealth,
    el('div', { class: 'strip-block' }, [eyebrow('Seed'), stripFacts]),
    el('div', { class: 'strip-block' }, [eyebrow('Events'), tickerList]),
    el('div', { class: 'strip-block' }, [eyebrow('Frame'), stripStats]),
    el('div', { class: 'strip-block' }, [
      eyebrow('Keys'),
      el('div', { class: 'strip-keys' }, [
        el('span', {}, [el('kbd', { text: 'space' }), ' play']),
        el('span', {}, [el('kbd', { text: '← →' }), ' step']),
        el('span', {}, [el('kbd', { text: '[ ]' }), ' speed']),
        el('span', {}, [el('kbd', { text: 'g' }), ' debug']),
        el('span', {}, [el('kbd', { text: 't' }), ' tier']),
      ]),
    ]),
    el('div', { class: 'strip-block' }, [
      el('a', { class: 'pick-meta', href: '#/theater', text: '‹ all recordings' }),
      el('a', { class: 'pick-meta', href: `#/run/${encodeURIComponent(target.runId)}`, text: '‹ run dashboard' }),
    ]),
  ]);

  body.append(
    el('div', { class: 'theater' }, [
      el('div', { class: 'theater-main' }, [stage, banner, transport]),
      strip,
    ]),
  );

  // ---- state the loop touches ----
  let frameCount = 1;
  let ticker: TickerEntry[] = [];
  let scrubbing = false;
  let statsAccum = 0;
  let tickerDirty = false;

  function setProgress(progress: LoadProgress): void {
    coverFill.style.width = `${(progress.fraction * 100).toFixed(1)}%`;
    coverLabel.textContent = progress.label;
  }

  function fail(message: string): void {
    cover.hidden = false;
    coverTitle.textContent = 'Could not play this seed';
    coverBar.hidden = true;
    coverLabel.hidden = true;
    cover.append(
      el('div', { class: 'cover-err' }, [
        errorState(message, undefined, {
          title: `${info.name} · seed ${target.seed}`,
          actions: el('a', { class: 'btn btn-quiet', href: '#/theater', text: 'Back to recordings' }),
        }),
      ]),
    );
  }

  function paintTicker(): void {
    if (ticker.length === 0) {
      tickerList.replaceChildren(el('div', { class: 'ticker-empty', text: 'no events yet' }));
      return;
    }
    tickerList.replaceChildren(
      ...ticker.map((entry) =>
        el('div', { class: 'ticker-row' }, [
          el('span', { class: 'ticker-step', text: `step ${entry.step}` }),
          // landmark kinds carry the scrub bar's colors; continuity events
          // (teleport / resurface) stay neutral so they read as punctuation
          el('span', { class: `ticker-dot tick-${TICK_KINDS.includes(entry.type as TickKind) ? entry.type : 'other'}` }),
          el('span', { text: entry.label }),
        ]),
      ),
    );
  }

  function paintTicks(ticks: readonly TimelineTick[]): void {
    scrubTicks.replaceChildren(
      ...ticks.map((tick) => {
        const node = el('button', {
          class: `tick tick-${tick.kind}`,
          type: 'button',
          title: tick.label,
          ariaLabel: `Seek to ${tick.label}`,
          style: { left: `${(tick.position * 100).toFixed(3)}%` },
        });
        node.addEventListener('click', (event) => {
          event.preventDefault();
          if (!sceneHost) return;
          sceneHost.seek(seekTargetForTick(tick));
          sceneHost.play();
          syncPlay();
        });
        node.addEventListener('mouseenter', () => {
          scrubTip.textContent = tick.label;
          scrubTip.style.left = `${(tick.position * 100).toFixed(3)}%`;
          scrubTip.hidden = false;
        });
        node.addEventListener('mouseleave', () => {
          scrubTip.hidden = true;
        });
        return node;
      }),
    );
  }

  function syncPlay(): void {
    playButton.textContent = sceneHost?.clock.playing ? 'pause' : 'play';
  }

  function onFrame(frame: HostFrameInfo): void {
    const fraction = frameCount > 1 ? frame.position / (frameCount - 1) : 0;
    scrubFill.style.width = `${(fraction * 100).toFixed(3)}%`;
    scrubHead.style.left = `${(fraction * 100).toFixed(3)}%`;
    if (!scrubbing) scrubInput.value = String(frame.position);

    statsAccum += frame.frameMs;
    if (tickerDirty) {
      tickerDirty = false;
      paintTicker();
    }
    if (statsAccum < 120) return;
    statsAccum = 0;
    readout.textContent =
      `step ${frame.step} · ${formatClock(frame.simTime)} · f${frame.index}/${frameCount - 1}`;
    const budget = frame.budget;
    const over = frame.draws > budget.drawCalls || frame.triangles > budget.triangles;
    stripStats.replaceChildren(
      kv('fps', frame.fps.toFixed(0)),
      // 0 means the governor's window is not warm yet — say "not measured",
      // never print a 0.0 ms frame time nobody could have rendered
      kv(
        'p95',
        frame.p95 > 0 ? `${frame.p95.toFixed(1)} ms` : '—',
        frame.p95Threshold > 0
          ? `rolling p95 frame time; two sustained seconds above ${frame.p95Threshold.toFixed(0)} ms drops the tier`
          : 'rolling p95 frame time (window not warm yet)',
      ),
      kv('draws', `${frame.draws} / ${budget.drawCalls}${over ? ' ⚠' : ''}`),
      kv('tris', `${(frame.triangles / 1000).toFixed(1)}k / ${(budget.triangles / 1000).toFixed(0)}k`),
      kv('tier', frame.tier),
      kv('speed', `${frame.speed}×`),
    );
    if (tierSelect.value !== frame.tier) tierSelect.value = frame.tier;
    syncPlay();
  }

  function onEvents(events: readonly SimEvent[], steps: readonly number[]): void {
    ticker = pushTicker(ticker, events, steps, TICKER_LIMIT);
    tickerDirty = true;
  }

  // ---- transport wiring ----
  playButton.addEventListener('click', () => {
    sceneHost?.toggle();
    syncPlay();
  });
  backButton.addEventListener('click', () => stepFrames(-1));
  fwdButton.addEventListener('click', () => stepFrames(1));
  speedSelect.addEventListener('change', () => sceneHost?.setSpeed(Number(speedSelect.value) || 1));
  tierSelect.addEventListener('change', () => {
    void sceneHost?.setTier(tierSelect.value as QualityTier);
  });
  debugButton.addEventListener('click', () => setDebug(!(sceneHost?.debug ?? false)));

  scrubInput.addEventListener('pointerdown', () => {
    scrubbing = true;
  });
  const endScrub = (): void => {
    scrubbing = false;
  };
  scrubInput.addEventListener('pointerup', endScrub);
  scrubInput.addEventListener('pointercancel', endScrub);
  scrubInput.addEventListener('blur', endScrub);
  scrubInput.addEventListener('input', () => {
    sceneHost?.seek(clampPosition(Number(scrubInput.value), frameCount));
  });

  function stepFrames(frames: number): void {
    if (!sceneHost) return;
    sceneHost.pause();
    sceneHost.seek(stepPosition(sceneHost.clock.position(), frames, frameCount));
    syncPlay();
  }

  function setDebug(on: boolean): void {
    sceneHost?.setDebug(on);
    debugButton.classList.toggle('is-on', on);
  }

  const onKey = (event: KeyboardEvent): void => {
    const node = event.target;
    if (node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement) {
      return;
    }
    const action = keyAction(event.key, event.shiftKey);
    if (!action || !sceneHost) return;
    event.preventDefault();
    switch (action.kind) {
      case 'toggle':
        sceneHost.toggle();
        syncPlay();
        break;
      case 'step':
        stepFrames(action.frames);
        break;
      case 'speed': {
        const speed = nextSpeed(sceneHost.clock.speed, action.direction);
        sceneHost.setSpeed(speed);
        speedSelect.value = String(speed);
        break;
      }
      case 'debug':
        setDebug(!sceneHost.debug);
        break;
      case 'tier': {
        const next = cycleTier(sceneHost.tier);
        tierSelect.value = next;
        void sceneHost.setTier(next);
        break;
      }
      case 'seek':
        sceneHost.seek(action.position === 'start' ? 0 : frameCount - 1);
        break;
    }
  };
  window.addEventListener('keydown', onKey);

  // ---- load, then mount the scene ----
  void (async () => {
    let recording: TheaterRecording;
    try {
      let expectedSteps = 0;
      try {
        const detail = await loadRunDetail(target.runId);
        expectedSteps = recordingRef(detail.recordings ?? [], target.envId, target.seed)?.steps ?? 0;
      } catch {
        // The index is a convenience (it gives the progress bar its denominator
        // before any header arrives); the recording endpoints are the truth.
        expectedSteps = 0;
      }
      if (destroyed) return;
      recording = await loadTheaterRecording({
        ...target,
        expectedSteps,
        ...(query.steps !== undefined ? { maxSteps: query.steps } : {}),
        signal: abort.signal,
        onProgress: setProgress,
      });
    } catch (cause) {
      if (destroyed || abort.signal.aborted) return;
      fail(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    if (destroyed) return;

    frameCount = recording.source.frames.length;
    scrubInput.max = String(Math.max(1, frameCount - 1));
    paintTicks(recording.ticks);

    stripFacts.replaceChildren(
      kv('env', info.name),
      kv('run', target.runId, target.runId),
      kv('seed', String(recording.decoded.seed)),
      kv('steps', formatCount(recording.steps)),
      kv('rate', `${recording.source.controlHz} Hz`),
      kv('action', info.action.kind === 'box' ? `box ${info.action.n}` : `discrete ${info.action.n}`),
      // The LANDMARKS on the scrub bar (goal/done/key/door/cue) — deliberately
      // not "events": teleport and resurface fire in the ticker and are not marked.
      kv(
        'marks',
        formatCount(recording.ticks.reduce((sum, tick) => sum + tick.count, 0)),
        'goal / done / key / door / cue events marked on the scrub bar',
      ),
      kv('bytes', `${(recording.bytes / (1024 * 1024)).toFixed(1)} MB`),
    );

    if (recording.health.message) {
      banner.hidden = false;
      const text = banner.querySelector('.banner-text');
      if (text) text.textContent = recording.health.message;
      stripHealth.hidden = false;
      stripHealth.textContent = recording.health.message;
    }

    setProgress(makeProgress('scene'));
    try {
      sceneHost = await mountSceneHost(stage, {
        envId: target.envId,
        source: recording.source,
        tier: query.tier ?? 'hero',
        ...(query.debug !== undefined ? { debug: query.debug } : {}),
        ...(query.paused !== undefined ? { paused: query.paused } : {}),
        ...(query.at !== undefined ? { at: query.at } : {}),
        ...(query.speed !== undefined ? { speed: query.speed } : {}),
        onFrame,
        onEvents,
        onReady(manifest) {
          stripTitle.textContent = manifest.title;
          stripLogline.textContent = manifest.logline;
          const tagTitle = stageTag.querySelector('b');
          if (tagTitle) tagTitle.textContent = manifest.title;
        },
        onTier(tier, reason) {
          tierSelect.value = tier;
          // The governor giving up quality is a fact about this machine, not a
          // failure — say so where the frame numbers are, and nowhere louder.
          stripStats.title = reason === 'auto'
            ? `auto-dropped to ${tier}: p95 frame time stayed over 20 ms`
            : `${tier} (chosen)`;
        },
      });
    } catch (cause) {
      if (destroyed) return;
      fail(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    if (destroyed) {
      sceneHost.destroy();
      sceneHost = null;
      return;
    }
    // The canvas must sit under the overlays the stage already holds.
    stage.insertBefore(sceneHost.canvas, stageTag);
    if (query.speed !== undefined) speedSelect.value = String(query.speed);
    debugButton.classList.toggle('is-on', sceneHost.debug);
    cover.hidden = true;
    syncPlay();
  })();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      abort.abort();
      window.removeEventListener('keydown', onKey);
      sceneHost?.destroy();
      sceneHost = null;
      root.remove();
    },
  };
}
