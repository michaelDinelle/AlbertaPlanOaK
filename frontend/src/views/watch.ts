// #/watch and #/watch/:id — the live wall.
//
// A fleet heartbeat lands every 10k steps, so "live" here means a chart that
// grows a point every few seconds for half an hour. The view is built for that
// tempo: nothing animates, the numbers just become true.

import type { ViewMount } from '../router';
import type { AggregateRow, RunIndexEntry } from '@shared/types';
import { ApiError, openLive, type LiveConnection } from '../api';
import {
  absorbAggregateRow,
  aggregateSeries,
  envName,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  formatWall,
  loadRunDetail,
  policiesIn,
  runIndex,
  sortRuns,
} from '../state';
import {
  badge,
  breadcrumb,
  el,
  emptyState,
  errorState,
  eyebrow,
  linkButton,
  spinner,
  statTile,
  viewShell,
} from '../components/table';
import { VIZ, band, timeSeries, type ChartHandle } from '../components/chart';
import { envSpectrum } from './gallery';

function progressBar(fraction: number): HTMLElement {
  const fill = el('span', { class: 'progress-fill' });
  fill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return el('span', { class: 'progress' }, [fill]);
}

// ---------------- run list ----------------

function mountList(host: HTMLElement): { destroy(): void } {
  const { root, body } = viewShell('watch');
  host.append(root);
  let destroyed = false;

  const status = el('div', { class: 'view-status' }, [spinner('Looking for running fleets')]);
  /** labels the fallback list so six finished runs never read as six live ones */
  const listHead = el('div');
  listHead.hidden = true;
  const list = el('div', { class: 'watch-list' });

  body.append(
    el('div', { class: 'view-head' }, [
      el('h1', { class: 'view-title', text: 'Watch' }),
      el('p', { class: 'view-lede', text: 'Fleets still emitting heartbeats. Open one to follow its population live.' }),
    ]),
    status,
    listHead,
    list,
  );

  function row(entry: RunIndexEntry, live: boolean): HTMLElement {
    const fraction = entry.steps > 0 ? entry.maxStep / entry.steps : 0;
    return el('a', { class: 'watch-row', href: `#/watch/${encodeURIComponent(entry.id)}` }, [
      envSpectrum(entry),
      el('div', { class: 'watch-row-body' }, [
        el('div', { class: 'watch-row-head' }, [
          el('span', { class: 'card-name', text: entry.id }),
          live ? badge('running', 'live') : entry.complete ? badge('complete', 'good') : badge('stalled', 'warn'),
        ]),
        el('div', {
          class: 'card-meta',
          text: [
            entry.envIds.length === 1 ? envName(entry.envIds[0] ?? '') : `${entry.envIds.length} envs`,
            `B ${entry.batch}`,
            formatAgo(entry.startedAt),
          ].join('  ·  '),
        }),
        el('div', { class: 'watch-progress' }, [
          progressBar(fraction),
          el('span', { class: 'watch-progress-text', text: `${formatStep(entry.maxStep)} / ${formatStep(entry.steps)}` }),
        ]),
      ]),
    ]);
  }

  function paint(entries: readonly RunIndexEntry[]): void {
    const running = sortRuns(entries.filter((entry) => entry.inFlight));
    if (running.length > 0) {
      status.replaceChildren();
      listHead.hidden = false;
      listHead.replaceChildren(
        eyebrow('Live now', el('span', { class: 'eyebrow-count', text: String(running.length) })),
      );
      list.replaceChildren(...running.map((entry) => row(entry, true)));
      return;
    }
    if (entries.length === 0) {
      status.replaceChildren(
        emptyState(
          'No fleet roots on disk',
          'The server found nothing under .oak-benchmarks. Point it elsewhere with OAK_BENCH_DIR, '
            + 'or produce a run with tools/run_fleet.sh.',
        ),
      );
      listHead.hidden = true;
      list.replaceChildren();
      return;
    }
    // Not an error: no fleet is running most of the time. Say so, then offer the
    // next most useful thing rather than an empty page.
    status.replaceChildren(
      emptyState(
        'Nothing is running',
        'No fleet root has been written to in the last two minutes. The most recent runs are below — '
          + 'their live views replay what is already on disk.',
        [linkButton('Browse every run', '#/runs')],
      ),
    );
    listHead.hidden = false;
    listHead.replaceChildren(
      eyebrow('Most recent', el('span', { class: 'eyebrow-note', text: 'finished — replayed from disk' })),
    );
    list.replaceChildren(...sortRuns(entries).slice(0, 6).map((entry) => row(entry, false)));
  }

  function load(force: boolean): void {
    runIndex.load(force).then(
      (entries) => {
        if (!destroyed) paint(entries);
      },
      (error: unknown) => {
        if (destroyed) return;
        listHead.hidden = true;
        list.replaceChildren();
        status.replaceChildren(
          errorState(error instanceof Error ? error.message : String(error), () => {
            status.replaceChildren(spinner('Retrying'));
            load(true);
          }, {
            title: 'Could not reach the API',
            actions: [linkButton('Back to the gallery', '#/runs')],
          }),
        );
      },
    );
  }

  load(false);

  return {
    destroy() {
      destroyed = true;
      root.remove();
    },
  };
}

// ---------------- live view ----------------

function mountLive(host: HTMLElement, id: string): { destroy(): void } {
  const pip = el('span', { class: 'live-pip is-waiting' });
  const { root, body } = viewShell('watch', el('span', { class: 'header-live' }, [pip, el('span', { text: 'live' })]));
  host.append(root);

  let destroyed = false;
  let horizon = 0;
  let frame = 0;
  const charts: ChartHandle[] = [];
  /**
   * Keyed `policy:step`, NOT step. A run with oak + zero + random emits three
   * heartbeats per step; keyed by step alone they overwrite each other and the
   * "live population" chart becomes a splice of three different populations —
   * a series no run ever produced.
   */
  const rows = new Map<string, AggregateRow>();
  /** the policy on screen; `null` until the first row tells us what exists */
  let policy: string | null = null;

  const statusChips = el('div', { class: 'run-chips' });
  const policyControl = el('div', { class: 'controls controls-inline' });
  policyControl.hidden = true;
  let policySelect: HTMLSelectElement | null = null;
  const banner = el('div', { class: 'banner' });
  banner.hidden = true;
  const statRow = el('div', { class: 'stat-row' });
  const progressHost = el('div', { class: 'watch-progress watch-progress-lg' });
  const status = el('div', { class: 'view-status' }, [spinner('Connecting')]);

  const bandHost = el('div');
  const throughputHost = el('div');
  const livesHost = el('div');
  /**
   * Three eyebrows over three empty divs is not an empty state — until a
   * heartbeat lands there is nothing to title. The whole stack stays hidden and
   * appears in one move when the first row arrives.
   */
  const chartStack = el('div', { class: 'block-stack' }, [
    el('section', { class: 'block' }, [
      eyebrow('Average reward rate', el('span', { class: 'eyebrow-note', text: 'P10 · median · P90, all seeds' })),
      bandHost,
    ]),
    el('div', { class: 'block-grid' }, [
      el('section', { class: 'block' }, [eyebrow('Throughput'), throughputHost]),
      el('section', { class: 'block' }, [eyebrow('Non-diverged seeds'), livesHost]),
    ]),
  ]);
  chartStack.hidden = true;

  const headActions = el('div', { class: 'run-head-actions' }, [
    linkButton('Full dashboard', `#/run/${encodeURIComponent(id)}`, {
      title: 'Survival, peak & retention, per-env slices and the seed ledger',
    }),
  ]);

  body.append(
    el('div', { class: 'view-head' }, [
      el('div', { class: 'run-title-row' }, [
        breadcrumb([{ label: 'Watch', href: '#/watch' }], { trailing: true }),
        el('h1', { class: 'view-title', text: id }),
        headActions,
      ]),
      statusChips,
      policyControl,
      progressHost,
    ]),
    banner,
    status,
    statRow,
    chartStack,
  );

  let bandChart: ChartHandle | null = null;
  let spsChart: ChartHandle | null = null;
  let livesChart: ChartHandle | null = null;

  function repaint(): void {
    if (destroyed || rows.size === 0) return;
    syncPolicies();
    const shown = policy;
    const ordered = [...rows.values()].filter((row) => shown === null || row.policy === shown);
    if (ordered.length === 0) return;
    status.replaceChildren();
    status.hidden = true;
    chartStack.hidden = false;
    const series = aggregateSeries(ordered, shown ?? undefined);
    const last = ordered.reduce((a, b) => (b.step > a.step ? b : a));

    statRow.replaceChildren(
      statTile('median reward rate', formatSigned(last.deliveredRateP50, 4), {
        hint: `P10 ${formatSigned(last.deliveredRateP10, 3)} · P90 ${formatSigned(last.deliveredRateP90, 3)}`,
        tone: last.deliveredRateP50 > 0 ? 'pos' : last.deliveredRateP50 < 0 ? 'neg' : 'dim',
      }),
      statTile('% seeds > 0', formatPercent(last.positiveRateFraction, 1), {
        hint: `${formatCount(last.positiveRateLaneCount)} of ${formatCount(last.alive + last.dead)}`,
      }),
      statTile('non-diverged', formatCount(last.alive), {
        hint: `${formatCount(last.dead)} numerically diverged`,
        tone: last.dead > 0 ? 'neg' : 'neutral',
      }),
      statTile('best seed', formatRate(last.bestLaneDeliveredRate), { hint: `seed ${formatCount(last.bestLaneSeed)}` }),
      statTile('throughput', `${formatStep(last.aggStepsPerSecond)}/s`, { hint: formatWall(last.wallSeconds) }),
      statTile('step', formatStep(last.step), { hint: horizon > 0 ? `of ${formatStep(horizon)}` : 'horizon unknown' }),
    );

    if (horizon > 0) {
      progressHost.replaceChildren(
        progressBar(last.step / horizon),
        el('span', { class: 'watch-progress-text', text: `${formatPercent(last.step / horizon, 0)} of horizon` }),
      );
    }

    if (!bandChart) {
      bandChart = band({
        steps: series.steps, p10: series.p10, p50: series.p50, p90: series.p90,
        height: 260, zeroLine: true, yUnit: 'reward / step', xUnit: 'steps',
      });
      charts.push(bandChart);
      bandHost.replaceChildren(bandChart.root);
    } else {
      bandChart.setData([series.steps, series.p90, series.p10, series.p50] as unknown as Parameters<ChartHandle['setData']>[0]);
    }

    if (!spsChart) {
      spsChart = timeSeries({
        steps: series.steps,
        lines: [{ label: 'aggregate steps/s', values: series.sps }],
        height: 150,
        yFormat: (v) => formatStep(v),
        yUnit: 'steps / second',
      });
      charts.push(spsChart);
      throughputHost.replaceChildren(spsChart.root);
    } else {
      spsChart.setData([series.steps, series.sps] as unknown as Parameters<ChartHandle['setData']>[0]);
    }

    if (!livesChart) {
      livesChart = timeSeries({
        steps: series.steps,
        lines: [
          { label: 'non-diverged', values: series.alive, stroke: VIZ.series },
          { label: 'diverged', values: series.dead, stroke: VIZ.crit },
        ],
        height: 150,
        yFormat: (v) => formatCount(v),
        yUnit: 'seeds',
        legend: true,
      });
      charts.push(livesChart);
      livesHost.replaceChildren(livesChart.root);
    } else {
      livesChart.setData([series.steps, series.alive, series.dead] as unknown as Parameters<ChartHandle['setData']>[0]);
    }
  }

  function schedule(): void {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(repaint);
  }

  function absorb(row: AggregateRow): void {
    // keyed (policy, step) — see state.aggregateKey
    absorbAggregateRow(rows, row);
  }

  /** Multi-policy runs get a selector; single-policy runs never see one. */
  function syncPolicies(): void {
    const names = policiesIn([...rows.values()]);
    if (names.length === 0) return;
    if (policy === null || !names.includes(policy)) policy = names[0] ?? null;
    if (names.length < 2) {
      policyControl.hidden = true;
      return;
    }
    if (!policySelect) {
      policySelect = el('select', { class: 'input input-select', ariaLabel: 'Policy' });
      policySelect.addEventListener('change', () => {
        policy = policySelect?.value ?? policy;
        repaint();
      });
      policyControl.replaceChildren(el('span', { class: 'control-label', text: 'policy' }), policySelect);
    }
    const options = [...policySelect.options].map((option) => option.value);
    if (options.join(',') !== names.join(',')) {
      policySelect.replaceChildren(...names.map((name) => el('option', { value: name, text: name })));
    }
    policySelect.value = policy ?? names[0] ?? '';
    policyControl.hidden = false;
  }

  function finish(message: string, kind: 'good' | 'warn'): void {
    if (missing) return; // a run that does not exist has not "completed"
    banner.hidden = false;
    banner.className = `banner banner-${kind}`;
    banner.replaceChildren(
      el('strong', { text: message }),
      el('a', { class: 'link-quiet', href: `#/run/${encodeURIComponent(id)}`, text: 'Open the full dashboard' }),
    );
    pip.classList.remove('is-waiting');
    pip.classList.add('is-off');
  }

  /**
   * The live view has two independent sources — the detail fetch (identity,
   * horizon, backlog) and the SSE stream — and either can fail on its own. When
   * the detail 404s the run simply is not there, and saying "no live stream,
   * showing what is on disk" over a permanently blank page is a lie; that state
   * wins over the transport message.
   */
  let missing = false;

  function showBlankState(): void {
    if (rows.size > 0) return; // real data on screen beats any message about its absence
    status.hidden = false;
    if (missing) {
      pip.classList.remove('is-waiting');
      pip.classList.add('is-off');
      headActions.remove(); // the dashboard would 404 on the same id
      status.replaceChildren(
        errorState(
          'Nothing under .oak-benchmarks answers to this id, so there is no stream to follow and no history to replay.',
          undefined,
          {
            title: `No run named “${id}”`,
            actions: [linkButton('Back to Watch', '#/watch'), linkButton('Browse every run', '#/runs')],
          },
        ),
      );
      return;
    }
    status.replaceChildren(
      errorState('No live stream for this run — showing whatever is already on disk.', undefined, {
        title: 'Not streaming',
        actions: [linkButton('Full dashboard', `#/run/${encodeURIComponent(id)}`)],
      }),
    );
  }

  loadRunDetail(id).then(
    (detail) => {
      if (destroyed) return;
      horizon = detail.meta.steps;
      statusChips.replaceChildren(
        badge(`B ${detail.meta.batch}`, 'quiet'),
        badge(`${formatStep(detail.meta.steps)} steps`, 'quiet'),
        badge(`ablate ${detail.meta.ablationLabel ?? 'none'}`, 'quiet'),
        badge(formatAgo(detail.meta.startedAt ?? null), 'quiet'),
      );
      for (const row of detail.aggregate) absorb(row);
      schedule();
    },
    (error: unknown) => {
      if (destroyed) return;
      missing = error instanceof ApiError && error.status === 404;
      if (missing) showBlankState();
    },
  );

  const connection: LiveConnection = openLive(id, {
    open: () => pip.classList.remove('is-waiting'),
    hello: (payload) => {
      for (const row of payload.rows ?? []) absorb(row);
      schedule();
    },
    row: (row) => {
      absorb(row);
      schedule();
    },
    status: (state) => {
      if (state.maxStep > 0 && horizon === 0) horizon = state.maxStep;
      if (state.complete) finish('Run complete. ', 'good');
      else if (!state.inFlight) finish('Stream stopped before the horizon. ', 'warn');
    },
    done: () => finish('Stream closed. ', 'good'),
    error: () => {
      pip.classList.add('is-waiting');
      showBlankState();
    },
  });

  if (connection.closed) {
    status.replaceChildren(
      errorState('This browser has no EventSource support, so the population cannot be followed live.', undefined, {
        title: 'No live transport',
        actions: [linkButton('Full dashboard', `#/run/${encodeURIComponent(id)}`)],
      }),
    );
  }

  return {
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      connection.close();
      for (const chart of charts) chart.destroy();
      charts.length = 0;
      root.remove();
    },
  };
}

export const mount: ViewMount = (host, params) => {
  const id = params['id'];
  return id ? mountLive(host, id) : mountList(host);
};
