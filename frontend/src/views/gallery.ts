// #/runs — the run gallery.
//
// It used to be the home page; `#/` is the Wall now, and this is the index you
// go to when you want the ~600 roots rather than one run playing. `GALLERY_PATH`
// is the single constant every "back to the list" link in the app reads, so the
// route can move again without leaving orphan links behind.
//
// The signature element is the env spectrum: a fixed-order strip of accent
// cells, one per env in the run, widths proportional to seed share. It uses the
// same per-env accents the 3D scenes use (kit/tokens.ts), so a run's identity in
// the gallery matches its identity in the theater. The strip is an accent, never
// the only encoding — the env names are always printed beside it.

import type { ViewMount } from '../router';
import type { RunDetail, RunIndexEntry } from '@shared/types';
import { TOKENS } from '@kit/tokens';
import { ENVS, isEnvId, type EnvId } from '@shared/envs';
import {
  EMPTY_FILTER,
  aggregateSeries,
  envIdsInIndex,
  envIdsOf,
  envName,
  envShares,
  envSlug,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatStep,
  loadRunDetail,
  matchesFilter,
  policiesIn,
  primaryResult,
  roles as rolesResource,
  runIndex,
  sortRuns,
  type EnvShare,
  type GalleryFilter,
} from '../state';
import {
  actionButton,
  badge,
  el,
  emptyState,
  errorState,
  eyebrow,
  spinner,
  viewShell,
} from '../components/table';
import { sparkline, type SparklineHandle } from '../components/sparkline';

/** Where the gallery lives. The Wall owns `#/`. */
export const GALLERY_PATH = '#/runs';

export function accentCss(slug: string): string {
  const value = TOKENS.accents[slug];
  if (value === undefined) return '#5c6a80';
  return `#${value.toString(16).padStart(6, '0')}`;
}

/**
 * The strip: one cell per env, ordered by the canonical env table.
 *
 * Two inputs, deliberately unequal in what they claim:
 *   * `EnvShare[]` — TRUE apportionment from `envShares(meta)` (i.e. from
 *     `meta.envSchedule`): cell widths are the real seed fractions and each cell
 *     names its seed count.
 *   * a run-index entry — IDENTITY ONLY: equal-width cells and no seed numbers,
 *     because the index carries `envIds` and `batch` but never the split between
 *     them. Inventing `batch / envIds.length` there printed confident fiction on
 *     every mixed run (the default schedule is 1024/128/64/64/64/64, not 6×235).
 *     Callers that later hydrate the detail swap the strip for the share form.
 */
export function envSpectrum(source: Pick<RunIndexEntry, 'envIds'> | readonly EnvShare[]): HTMLElement {
  let shares: readonly EnvShare[] | null = null;
  let ids: EnvId[];
  if ('envIds' in source) {
    ids = envIdsOf(source);
  } else {
    shares = source;
    ids = source.map((share) => share.envId);
  }
  const strip = el('div', {
    class: 'spectrum',
    ariaLabel: ids.length > 0 ? `Environments: ${ids.map((id) => envName(id)).join(', ')}` : 'No environments',
    role: 'img',
  });
  if (ids.length === 0) {
    strip.append(el('span', { class: 'spectrum-cell', style: { flexGrow: '1', background: '#232c3d' } }));
    return strip;
  }
  ids.forEach((envId, index) => {
    const info = ENVS[envId];
    const share = shares?.[index];
    strip.append(
      el('span', {
        class: 'spectrum-cell',
        title: share ? `${info.name} — ${formatCount(share.lanes)} seeds` : info.name,
        style: {
          flexGrow: String(share ? Math.max(0.05, share.fraction) : 1),
          background: accentCss(info.slug),
        },
      }),
    );
  });
  return strip;
}

function roleBadge(role: RunIndexEntry['role']): HTMLElement | null {
  if (!role) return null;
  const labels: Record<string, string> = {
    bestKernel: 'best kernel',
    candidate: 'candidate',
    parent: 'parent',
  };
  const titles: Record<string, string> = {
    bestKernel: 'The current promoted best kernel',
    candidate: 'The run under evaluation',
    parent: 'The best kernel this lineage came from',
  };
  return badge(labels[role] ?? role, role, titles[role] ?? role);
}

function metaLine(entry: RunIndexEntry): HTMLElement {
  const envCount = entry.envIds.length;
  const bits = [
    envCount === 1 ? envName(entry.envIds[0] ?? '') : `${envCount} envs`,
    `B ${entry.batch}`,
    `${formatStep(entry.steps)} steps`,
    formatAgo(entry.startedAt),
  ];
  if (entry.ablationLabel && entry.ablationLabel !== 'none') bits.push(`ablate ${entry.ablationLabel}`);
  return el('div', { class: 'card-meta', text: bits.join('  ·  ') });
}

/**
 * Which filters are actually narrowing the list, in plain words, each with the
 * edit that would undo it.
 *
 * The zero-match state has to answer "why is this empty?", and with six
 * independent controls (search, env, and four toggles) the honest answer is the
 * list of what is currently on — a user who set `recordings` twenty runs ago and
 * then typed a name sees an empty grid and blames the search box.
 */
export interface FilterFacet {
  key: keyof GalleryFilter;
  label: string;
}

export function activeFacets(filter: GalleryFilter): FilterFacet[] {
  const facets: FilterFacet[] = [];
  const search = filter.search.trim();
  if (search) facets.push({ key: 'search', label: `name “${search}”` });
  if (filter.envId) facets.push({ key: 'envId', label: envName(filter.envId) });
  if (filter.completeOnly) facets.push({ key: 'completeOnly', label: 'complete' });
  if (filter.recordingsOnly) facets.push({ key: 'recordingsOnly', label: 'recordings' });
  if (filter.inFlightOnly) facets.push({ key: 'inFlightOnly', label: 'running' });
  if (filter.rolesOnly) facets.push({ key: 'rolesOnly', label: 'roles' });
  return facets;
}

interface Card {
  root: HTMLElement;
  entry: RunIndexEntry;
  hydrate(): void;
  destroy(): void;
}

/**
 * There are ~600 roots on disk. Cards hydrate one detail fetch each, so a fast
 * scroll can queue hundreds — cap the concurrency so the index request and the
 * card the user is actually looking at are never stuck behind the backlog.
 */
const HYDRATION_SLOTS = 5;
let activeHydrations = 0;
const hydrationQueue: (() => void)[] = [];

function pump(): void {
  while (activeHydrations < HYDRATION_SLOTS && hydrationQueue.length > 0) {
    const job = hydrationQueue.shift();
    if (!job) return;
    activeHydrations++;
    job();
  }
}

function queueHydration(job: () => Promise<unknown>): void {
  hydrationQueue.push(() => {
    void job().finally(() => {
      activeHydrations--;
      pump();
    });
  });
  pump();
}

/** A repaint replaces every card, so anything still queued is for a dead one. */
function clearHydrationQueue(): void {
  hydrationQueue.length = 0;
}

/**
 * Where a run's "theater" affordance points.
 *
 * The run index knows THAT a root has recordings (`hasRecordings`) but not which
 * seed — env and seed only arrive with the detail fetch. So the link starts on
 * the run's own recordings tab, which always resolves and always lists every
 * replayable seed, and is upgraded to the seed-level theater route the moment
 * hydration names one. It is never left pointing at a route that does not exist.
 */
export function theaterHref(runId: string, recording?: { envId: string; seed: number }): string {
  const run = encodeURIComponent(runId);
  if (!recording) return `#/run/${run}?tab=recordings`;
  return `#/theater/${run}/${encodeURIComponent(recording.envId)}/${recording.seed}`;
}

function buildCard(entry: RunIndexEntry): Card {
  const iqmValue = el('span', { class: 'kv-value', text: '—' });
  const posValue = el('span', { class: 'kv-value', text: '—' });
  const sparkHolder = el('div', { class: 'card-spark' });
  let spark: SparklineHandle | null = null;
  let hydrated = false;

  const chips = el('div', { class: 'card-chips' });
  if (entry.inFlight) chips.append(badge('running', 'live', 'Heartbeats are still arriving'));
  if (!entry.complete && !entry.inFlight) {
    chips.append(badge('incomplete', 'warn', 'Stopped before its declared horizon — never gate on this'));
  }
  // `hasSummary` is pure file existence (the index only stats the file). A root whose
  // summary.json is PRESENT but unparseable would otherwise show no badge at all while
  // every figure reads "n/a" — the card asserting the summary is fine and that it has
  // no data. Hydration below knows the difference and adds the missing cue.
  if (!entry.hasSummary) chips.append(badge('no summary', 'quiet', 'fleet.py report has not been run on this root'));

  const envList = entry.envIds.map((id) => envName(id)).join(', ');

  // identity-only until hydration hands us meta.envSchedule (see envSpectrum)
  let spectrum = envSpectrum(entry);

  const card = el('a', { class: 'card', href: `#/run/${encodeURIComponent(entry.id)}` }, [
    spectrum,
    el('div', { class: 'card-body' }, [
      el('div', { class: 'card-head' }, [
        el('span', { class: 'card-name', text: entry.id, title: entry.id }),
        roleBadge(entry.role),
      ]),
      metaLine(entry),
      el('div', { class: 'card-figures' }, [
        el('div', { class: 'card-kv' }, [
          el('div', { class: 'kv' }, [el('span', { class: 'kv-label', text: 'IQM' }), iqmValue]),
          el('div', { class: 'kv' }, [el('span', { class: 'kv-label', text: 'positive' }), posValue]),
        ]),
        sparkHolder,
      ]),
      el('div', { class: 'card-envs', text: envList, title: envList }),
      chips,
    ]),
  ]);

  const root = el('div', { class: `card-shell${entry.hasRecordings ? ' has-theater' : ''}` }, [card]);
  const theater = entry.hasRecordings
    ? el('a', {
        class: 'card-theater',
        href: theaterHref(entry.id),
        title: 'Replay a recorded seed of this run in the theater',
      }, [el('span', { text: 'theater' }), el('span', { text: '↗' })])
    : null;
  if (theater) root.append(theater);

  return {
    root,
    entry,
    hydrate() {
      if (hydrated) return;
      hydrated = true;
      queueHydration(() => loadRunDetail(entry.id).then(
        (detail: RunDetail) => {
          // The detail carries the real per-env seed split; upgrade the strip.
          const shares = envShares(detail.meta);
          if (shares.length > 0) {
            const next = envSpectrum(shares);
            spectrum.replaceWith(next);
            spectrum = next;
          }

          // …and the first recorded seed, which upgrades the theater link from
          // "the list of replayable seeds" to "this seed, playing". The index and
          // the detail run the same servable scan, so `hasRecordings` with an
          // empty list should be impossible — if it ever happens, withdraw the
          // affordance rather than advertise a replay that is not there.
          const first = detail.recordings[0];
          if (theater && first) {
            theater.setAttribute('href', theaterHref(entry.id, first));
            theater.title = `Replay ${envName(first.envId)} seed ${first.seed}`
              + `${detail.recordings.length > 1 ? ` · ${detail.recordings.length} recorded seeds` : ''}`;
          } else if (theater) {
            theater.remove();
            root.classList.remove('has-theater');
          }

          // Present-but-rejected summary: say so on the card, next to the n/a it causes.
          if (detail.summaryError === 'unreadable' && entry.hasSummary) {
            chips.append(badge(
              'summary unreadable',
              'warn',
              'summary.json exists on this root but is not valid JSON, so it was rejected — '
                + 'the figures here are unavailable, not zero',
            ));
          }

          const result = primaryResult(detail.summary);
          const iqm = result?.rewardRateIQM;
          if (typeof iqm === 'number') {
            iqmValue.textContent = (iqm > 0 ? '+' : '') + formatRate(iqm);
            iqmValue.classList.add(iqm > 0 ? 'tone-pos' : iqm < 0 ? 'tone-neg' : 'tone-dim');
          } else {
            iqmValue.textContent = 'n/a';
            iqmValue.classList.add('tone-dim');
          }
          const positive = result?.positiveFraction;
          if (typeof positive === 'number') posValue.textContent = formatPercent(positive, 1);

          // The sparkline must trace the SAME policy the IQM/positive figures came
          // from. Unfiltered, a multi-policy run (oak + zero + random) interleaves
          // three populations into one fabricated line.
          const policies = policiesIn(detail.aggregate);
          const summaryPolicy = result?.policy;
          const policy = summaryPolicy && policies.includes(summaryPolicy) ? summaryPolicy : policies[0];
          const series = aggregateSeries(detail.aggregate, policy);
          if (series.steps.length >= 2) {
            spark = sparkline({
              values: series.p50,
              height: 34,
              ariaLabel: `Median reward rate${policies.length > 1 && policy ? ` for policy ${policy}` : ''}`
                + ` over ${formatStep(series.steps[series.steps.length - 1])} steps`,
            });
            sparkHolder.replaceChildren(spark.root);
          } else {
            sparkHolder.replaceChildren(el('span', { class: 'card-spark-empty', text: 'no heartbeats' }));
          }
        },
        () => {
          iqmValue.textContent = 'n/a';
          iqmValue.classList.add('tone-dim');
          sparkHolder.replaceChildren(el('span', { class: 'card-spark-empty', text: 'unavailable' }));
        },
      ));
    },
    destroy() {
      spark?.destroy();
      root.remove();
    },
  };
}

interface Toggle {
  root: HTMLElement;
  /** drive the control from outside (the zero-match state clears filters) */
  set(value: boolean, notify?: boolean): void;
}

function toggle(label: string, title: string, initial: boolean, onChange: (value: boolean) => void): Toggle {
  let value = initial;
  const button = el('button', { class: 'chip', text: label, title, ariaPressed: String(value) });
  const apply = (next: boolean, notify: boolean): void => {
    value = next;
    button.classList.toggle('is-on', value);
    button.setAttribute('aria-pressed', String(value));
    if (notify) onChange(value);
  };
  apply(initial, false);
  button.addEventListener('click', () => apply(!value, true));
  return { root: button, set: (next, notify = false) => apply(next, notify) };
}

export const mount: ViewMount = (host) => {
  const liveCount = el('span', { class: 'header-live' });
  const { root, body } = viewShell('gallery', liveCount);
  host.append(root);

  const filter: GalleryFilter = { ...EMPTY_FILTER };
  let grouped = false;
  let entries: RunIndexEntry[] = [];
  let cards: Card[] = [];
  let destroyed = false;

  const search = el('input', { class: 'input', type: 'search', placeholder: 'Filter runs…', ariaLabel: 'Filter runs by name' });
  const envSelect = el('select', { class: 'input input-select', ariaLabel: 'Filter by environment' });
  const toggles: Partial<Record<keyof GalleryFilter, Toggle>> = {
    completeOnly: toggle('complete', 'Only runs that reached their declared horizon', false, (v) => {
      filter.completeOnly = v;
      paint();
    }),
    recordingsOnly: toggle('recordings', 'Only runs with per-seed recordings for the theater', false, (v) => {
      filter.recordingsOnly = v;
      paint();
    }),
    inFlightOnly: toggle('running', 'Only runs still emitting heartbeats', false, (v) => {
      filter.inFlightOnly = v;
      paint();
    }),
    rolesOnly: toggle('roles', 'Only the best kernel, candidate and parent', false, (v) => {
      filter.rolesOnly = v;
      paint();
    }),
  };
  const groupToggle = toggle('group by env', 'Group runs into sections by environment', false, (v) => {
    grouped = v;
    paint();
  });
  const controls = el('div', { class: 'controls' }, [
    search,
    envSelect,
    toggles.completeOnly?.root,
    toggles.recordingsOnly?.root,
    toggles.inFlightOnly?.root,
    toggles.rolesOnly?.root,
    groupToggle.root,
  ]);

  /** Undo one facet, or every facet. Grouping is a layout choice, not a filter. */
  function clearFacet(key: keyof GalleryFilter | 'all'): void {
    if (key === 'all' || key === 'search') {
      filter.search = '';
      search.value = '';
    }
    if (key === 'all' || key === 'envId') {
      filter.envId = null;
      envSelect.value = '';
    }
    for (const name of ['completeOnly', 'recordingsOnly', 'inFlightOnly', 'rolesOnly'] as const) {
      if (key !== 'all' && key !== name) continue;
      filter[name] = false;
      toggles[name]?.set(false);
    }
    paint();
  }

  const count = el('div', { class: 'result-count' });
  const grid = el('div', { class: 'gallery' });
  const status = el('div', { class: 'view-status' }, [spinner('Reading .oak-benchmarks')]);

  body.append(
    el('div', { class: 'view-head' }, [
      el('h1', { class: 'view-title', text: 'Run gallery' }),
      el('p', { class: 'view-lede', text: 'Every fleet root on disk. Cards fill in their population numbers as they scroll into view.' }),
    ]),
    controls,
    count,
    status,
    grid,
  );

  search.addEventListener('input', () => {
    filter.search = search.value;
    paint();
  });
  envSelect.addEventListener('change', () => {
    filter.envId = envSelect.value || null;
    paint();
  });

  const observer = typeof IntersectionObserver === 'undefined'
    ? null
    : new IntersectionObserver(
        (records) => {
          for (const record of records) {
            if (!record.isIntersecting) continue;
            const card = cards.find((c) => c.root === record.target);
            card?.hydrate();
            observer?.unobserve(record.target);
          }
        },
        { rootMargin: '240px 0px' },
      );

  function paint(): void {
    if (destroyed) return;
    clearHydrationQueue();
    observer?.disconnect();
    for (const card of cards) card.destroy();
    cards = [];

    const visible = sortRuns(entries.filter((entry) => matchesFilter(entry, filter)));
    count.textContent = `${visible.length} of ${entries.length} runs`;

    if (visible.length === 0) {
      grid.replaceChildren();
      const facets = activeFacets(filter);
      if (entries.length === 0) {
        status.replaceChildren(
          emptyState(
            'No fleet roots on disk',
            'The server found nothing under .oak-benchmarks. Point it elsewhere with OAK_BENCH_DIR, '
              + 'or produce a run with tools/run_fleet.sh.',
          ),
        );
        return;
      }
      const readout = el('div', { class: 'filter-readout' }, [
        el('span', { class: 'control-label', text: 'active' }),
        ...facets.map((facet) => {
          const chip = el('button', { class: 'filter-chip', type: 'button', title: `Remove the ${facet.label} filter` }, [
            el('span', { text: facet.label }),
            el('span', { class: 'filter-chip-x', text: '×' }),
          ]);
          chip.addEventListener('click', () => clearFacet(facet.key));
          return chip;
        }),
      ]);
      status.replaceChildren(
        emptyState(
          'Nothing matches',
          `None of the ${formatCount(entries.length)} runs on disk pass ${facets.length === 1 ? 'this filter' : 'all of these filters'}. `
            + 'Drop one below, or clear them all.',
          [readout, actionButton('Clear all filters', () => clearFacet('all'), { quiet: true })],
        ),
      );
      return;
    }
    status.replaceChildren();

    const build = (entry: RunIndexEntry): HTMLElement => {
      const card = buildCard(entry);
      cards.push(card);
      if (observer) observer.observe(card.root);
      else card.hydrate();
      return card.root;
    };

    if (!grouped) {
      grid.className = 'gallery';
      grid.replaceChildren(...visible.map(build));
      return;
    }

    grid.className = 'gallery-groups';
    const groups = new Map<string, RunIndexEntry[]>();
    for (const entry of visible) {
      const key = entry.envIds.length === 1 ? entry.envIds[0] ?? 'unknown' : `Mixed · ${entry.envIds.length} envs`;
      const list = groups.get(key) ?? [];
      list.push(entry);
      groups.set(key, list);
    }
    const sections: HTMLElement[] = [];
    for (const [key, list] of [...groups.entries()].sort((a, b) => b[1].length - a[1].length)) {
      const title = isEnvId(key) ? ENVS[key].name : key;
      const section = el('section', { class: 'gallery-group' }, [
        eyebrow(title, el('span', { class: 'eyebrow-count', text: String(list.length) })),
        el('div', { class: 'gallery' }, list.map(build)),
      ]);
      if (isEnvId(key)) section.style.setProperty('--section-accent', accentCss(envSlug(key)));
      sections.push(section);
    }
    grid.replaceChildren(...sections);
  }

  function fillEnvOptions(): void {
    envSelect.replaceChildren(el('option', { value: '', text: 'All environments' }));
    for (const id of envIdsInIndex(entries)) {
      envSelect.append(el('option', { value: id, text: ENVS[id].name }));
    }
    envSelect.value = filter.envId ?? '';
  }
  // Populate before the first fetch resolves: an option-less <select> collapses to
  // a ~20px stub, which is what the whole control row looked like whenever the
  // index request failed.
  fillEnvOptions();

  /** One load path for the first attempt and every retry — same success, same failure. */
  function loadIndex(force: boolean): void {
    runIndex.load(force).then(
      (list) => {
        if (destroyed) return;
        entries = list;
        const running = list.filter((entry) => entry.inFlight).length;
        liveCount.replaceChildren(
          ...(running > 0
            ? [
                el('span', { class: 'live-pip' }),
                el('a', { class: 'header-live-link', href: '#/watch', text: `${running} running` }),
              ]
            : []),
        );
        fillEnvOptions();
        paint();
      },
      (error: unknown) => {
        if (destroyed) return;
        grid.replaceChildren();
        count.textContent = '';
        status.replaceChildren(
          errorState(
            error instanceof Error ? error.message : String(error),
            () => {
              status.replaceChildren(spinner('Retrying'));
              loadIndex(true);
            },
            {
              title: 'Could not reach the API',
              // The one failure a reader can act on without leaving the page:
              // Vite serves this bundle on :5173 whether or not the API on :5174
              // is alive, so a dead API looks exactly like an empty benchmarks dir.
              actions: [
                el('span', {
                  class: 'empty-detail',
                  text: 'The dev server proxies /api to :5174 — start it with `bun run dev`.',
                }),
              ],
            },
          ),
        );
      },
    );
  }

  loadIndex(false);

  // roles are a nice-to-have: the index already carries `role`
  rolesResource.load().catch(() => undefined);

  return {
    destroy() {
      destroyed = true;
      observer?.disconnect();
      for (const card of cards) card.destroy();
      cards = [];
      root.remove();
    },
  };
};
