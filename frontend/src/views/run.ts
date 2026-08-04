// #/run/:id — the run dashboard.
//
// House rule this view enforces in its layout: metrics are NEVER pooled across
// environments. The overview shows the fleet-wide aggregate (which the fleet
// itself computes) and the per-env tab gives every env its own section on its
// own scale. The only cross-env chart is positivity, because "fraction of lanes
// above zero" is the one quantity that means the same thing in CartPole and in
// HalfCheetah.

import type { ViewMount } from '../router';
import type { FleetColumnar, RunDetail, SummaryPolicyResult } from '@shared/types';
import { ENVS, isEnvId } from '@shared/envs';
import { ApiError, api } from '../api';
import {
  aggregateSeries,
  capLanes,
  envName,
  envShares,
  envSliceResult,
  envSlug,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  formatWall,
  kmCurve,
  laneSeries,
  loadRunDetail,
  pickField,
  policiesIn,
  policyView,
  rankLanes,
  type LaneSeries,
  type PolicyView,
} from '../state';
import {
  badge,
  breadcrumb,
  dataTable,
  el,
  emptyState,
  errorState,
  eyebrow,
  linkButton,
  spinner,
  statTile,
  viewShell,
  type TableHandle,
} from '../components/table';
import { VIZ, band, spaghetti, steppedLine, timeSeries, type ChartHandle } from '../components/chart';
import { bars, parseQuartiles, rangeBars, type RangeRow } from '../components/bars';
import { accentCss, envSpectrum, theaterHref } from './gallery';

interface Disposable {
  destroy(): void;
}

const LANE_CAP = 140;

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function tone(value: number | null): 'pos' | 'neg' | 'dim' {
  if (value === null) return 'dim';
  return value > 0 ? 'pos' : value < 0 ? 'neg' : 'dim';
}

export const mount: ViewMount = (host, params) => {
  const id = params['id'] ?? '';
  const { root, body } = viewShell('gallery');
  host.append(root);

  const disposables: Disposable[] = [];
  let destroyed = false;

  function track<T extends Disposable>(item: T): T {
    disposables.push(item);
    return item;
  }

  // The failure path gets the same chrome as the success path: a run id in the URL
  // is a claim about something on disk, and "that root is not there" is an answer
  // worth presenting, not a stack trace in a box.
  const head = el('div', { class: 'view-head' }, [
    el('div', { class: 'run-title-row' }, [
      breadcrumb([{ label: 'Gallery', href: '#/' }], { trailing: true }),
      el('h1', { class: 'run-title', text: id || 'run' }),
    ]),
  ]);
  const status = el('div', { class: 'view-status' }, [spinner(`Loading ${id}`)]);
  body.append(status);

  function fail(error: unknown, retry: boolean): void {
    if (head.parentNode === null) body.prepend(head);
    const notFound = error instanceof ApiError && error.status === 404;
    if (notFound) {
      // A 404 is settled — there is nothing to retry, only somewhere else to go.
      status.replaceChildren(
        errorState(
          'Nothing under .oak-benchmarks answers to this id. It may have been renamed, moved or deleted; '
            + 'the gallery lists every root the server can see.',
          undefined,
          { title: `No run named “${id}”`, actions: [linkButton('Back to the gallery', '#/')] },
        ),
      );
      return;
    }
    const message = error instanceof Error ? error.message : String(error);
    status.replaceChildren(
      errorState(
        message,
        retry
          ? () => {
              status.replaceChildren(spinner('Retrying'));
              loadRunDetail(id, true).then(
                (detail) => {
                  head.remove();
                  status.remove();
                  render(detail);
                },
                (again: unknown) => fail(again, false),
              );
            }
          : undefined,
        {
          title: 'Could not load this run',
          actions: [linkButton('Back to the gallery', '#/')],
        },
      ),
    );
  }

  loadRunDetail(id).then(
    (detail) => {
      if (destroyed) return;
      status.remove();
      render(detail);
    },
    (error: unknown) => {
      if (!destroyed) fail(error, true);
    },
  );

  function render(detail: RunDetail): void {
    const meta = detail.meta;
    const policies = policiesIn(detail.aggregate);
    let policy = policies[0] ?? meta.policies[0] ?? 'oak';
    const shares = envShares(meta);
    const mixed = shares.length > 1;
    const horizon = meta.steps;

    /**
     * EVERY policy-dependent number resolves through this (see state.policyView).
     * A policy switch must RE-DERIVE the figures — redrawing the charts while the
     * hero IQM, positivity, survival and completeness still describe the previous
     * policy is the worst kind of wrong: confidently mislabelled.
     */
    const forPolicy = (name: string): PolicyView => policyView(detail, name);

    // ---------------- header ----------------
    // `slot` is display:contents — the badges it holds stay direct flex items of
    // the row, so a policy switch can swap them without a layout seam.
    const completeSlot = el('span', { class: 'slot' });
    const policySlot = el('span', { class: 'slot' });
    const bannerSlot = el('div', { class: 'slot' });

    // The one outbound link the dashboard owes the theater. It names the lane it
    // will open rather than the route, because "theater" alone hides the fact
    // that a 1408-lane run replays exactly the handful that were recorded.
    const firstRecording = detail.recordings[0];
    const theaterAction = firstRecording
      ? linkButton(
          `Theater · ${envName(firstRecording.envId)} seed ${firstRecording.seed}`,
          theaterHref(detail.id, firstRecording),
          {
            title: `${formatCount(detail.recordings.length)} recorded lane${detail.recordings.length === 1 ? '' : 's'}`
              + ' — the Recordings tab lists them all',
          },
        )
      : null;

    body.append(
      el('section', { class: 'run-head' }, [
        envSpectrum(shares),
        el('div', { class: 'run-head-body' }, [
          el('div', { class: 'run-title-row' }, [
            breadcrumb([{ label: 'Gallery', href: '#/' }], { trailing: true }),
            el('h1', { class: 'run-title', text: detail.id }),
            completeSlot,
            theaterAction ? el('div', { class: 'run-head-actions' }, [theaterAction]) : null,
          ]),
          el('div', {
            class: 'run-envs',
            text: shares.map((s) => `${envName(s.envId)} ×${Math.round(s.lanes)}`).join('   ·   '),
          }),
          el('div', { class: 'run-chips' }, [
            badge(`B ${meta.batch}`, 'quiet', 'lanes in the fleet'),
            badge(`${formatStep(horizon)} steps`, 'quiet', 'declared horizon'),
            badge(
              `ablate ${meta.ablationLabel ?? 'none'}`,
              meta.ablationLabel && meta.ablationLabel !== 'none' ? 'warn' : 'quiet',
              'OAK_ABLATE for this run',
            ),
            policySlot,
            badge(formatAgo(meta.startedAt ?? null), 'quiet', meta.startedAt ?? undefined),
          ]),
        ]),
      ]),
      bannerSlot,
    );

    function syncPolicyHeader(): void {
      const view = forPolicy(policy);
      completeSlot.replaceChildren(
        view.complete
          ? badge('complete', 'good')
          : badge('incomplete', 'warn', 'Did not reach its declared horizon'),
      );
      policySlot.replaceChildren(badge(policy, 'quiet', 'policy'));
      bannerSlot.replaceChildren(
        ...(view.complete
          ? []
          : [
              el('div', { class: 'banner banner-warn' }, [
                el('strong', { text: 'Incomplete run. ' }),
                el('span', {
                  text: `Reached ${formatStep(view.maxStep)} of ${formatStep(horizon)} declared steps. fleet.py gate refuses these — read it, do not promote on it.`,
                }),
              ]),
            ]),
      );
    }
    syncPolicyHeader();

    // ---------------- lanes state (shared with the per-env tab) ----------------
    let selectedEnv: string = shares[0]?.envId ?? meta.envId;
    let requestLaneEnv: ((envId: string) => void) | null = null;

    // ---------------- tab machinery ----------------
    /**
     * A panel owns everything it built. Charts are kept apart because they also
     * need refresh() on tab show; `handles` is every other disposable (bars,
     * rangeBars, tables, fetch-scoped state) — all of them hold ResizeObservers,
     * so leaving them in the VIEW-level list leaked one observer per policy
     * switch, still watching a detached node.
     */
    interface Panel {
      node: HTMLElement;
      built: boolean;
      charts: ChartHandle[];
      handles: Disposable[];
    }
    interface PanelScope {
      /** register a chart and return its root (refreshed on show, destroyed on rebuild) */
      chart(handle: ChartHandle): HTMLElement;
      /** register any other disposable this panel owns */
      hold<T extends Disposable>(item: T): T;
    }
    const tabs: { key: string; label: string; build(panel: HTMLElement, scope: PanelScope): void }[] = [];
    const panels = new Map<string, Panel>();
    const tabBar = el('div', { class: 'tabs', role: 'tablist' });
    const panelHost = el('div', { class: 'panels' });

    function addTab(key: string, label: string, build: (panel: HTMLElement, scope: PanelScope) => void): void {
      tabs.push({ key, label, build });
      panels.set(key, { node: el('div', { class: 'panel' }), built: false, charts: [], handles: [] });
    }

    function scopeFor(panel: Panel): PanelScope {
      return {
        chart(handle) {
          panel.charts.push(handle);
          return handle.root;
        },
        hold(item) {
          panel.handles.push(item);
          return item;
        },
      };
    }

    /** Tear every panel back down to unbuilt — the policy switch and view destroy. */
    function resetPanels(): void {
      requestLaneEnv = null;
      for (const panel of panels.values()) {
        for (const chart of panel.charts) chart.destroy();
        panel.charts.length = 0;
        for (const item of panel.handles) item.destroy();
        panel.handles.length = 0;
        panel.built = false;
        panel.node.replaceChildren();
      }
    }
    track({ destroy: resetPanels });

    function currentTab(): string {
      const query = window.location.hash.split('?')[1] ?? '';
      const requested = new URLSearchParams(query).get('tab');
      return requested && panels.has(requested) ? requested : tabs[0]?.key ?? 'overview';
    }

    function show(key: string): void {
      for (const [tabKey, panel] of panels) {
        const isActive = tabKey === key;
        panel.node.hidden = !isActive;
        if (isActive && !panel.built) {
          panel.built = true;
          tabs.find((tab) => tab.key === tabKey)?.build(panel.node, scopeFor(panel));
        }
        if (isActive) for (const chart of panel.charts) chart.refresh();
      }
      for (const button of tabBar.querySelectorAll('button')) {
        const isActive = button.dataset['tab'] === key;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-selected', String(isActive));
      }
      const base = `#/run/${encodeURIComponent(id)}`;
      if (window.location.hash.split('?')[0] === base) history.replaceState(null, '', `${base}?tab=${key}`);
    }

    // ---------------- overview ----------------
    addTab('overview', 'Overview', (panel, scope) => {
      const chart = (handle: ChartHandle): HTMLElement => scope.chart(handle);
      // derived HERE, not in render(): this closure re-runs on every policy switch
      const { result } = forPolicy(policy);
      const series = aggregateSeries(detail.aggregate, policy);
      const iqm = num(result?.iqmDeliveredAll);
      const deaths = detail.deathsCount;
      const divergence = num((result?.['divergence'] as { rate?: number } | undefined)?.rate);
      const lanes = num(result?.lanes) ?? meta.batch;

      panel.append(
        el('div', { class: 'hero' }, [
          el('div', { class: 'hero-figure' }, [
            el('div', { class: 'hero-label', text: 'IQM delivered rate · all lanes' }),
            el('div', { class: `hero-value tone-${tone(iqm)}`, text: iqm === null ? 'n/a' : formatSigned(iqm, 4) }),
            el('div', {
              class: 'hero-hint',
              text: iqm === null
                ? 'No summary.json on this root — run tools/fleet.py report.'
                : `reward per step · interquartile mean over ${formatCount(lanes)} lanes`,
            }),
          ]),
          el('div', { class: 'hero-stats' }, [
            statTile('positive lanes', formatPercent(num(result?.positiveFraction), 1), {
              hint: `${formatCount(num(result?.['positiveLanes']))} above zero`,
            }),
            statTile('sustaining', formatPercent(num(result?.sustainingFraction), 1), { hint: 'holding ≥ ½ of zenith' }),
            statTile('divergence', divergence === null ? '—' : formatPercent(divergence, 2), {
              hint: `${formatCount(deaths)} death ${deaths === 1 ? 'certificate' : 'certificates'}`,
              tone: deaths > 0 ? 'neg' : 'neutral',
            }),
            statTile('best lane', formatRate(num(result?.bestLaneRate)), {
              hint: `seed ${formatCount(num(result?.bestLaneSeed))}`,
            }),
            statTile('reward / GFLOP', formatRate(num(result?.rewardPerGFLOP)), { hint: 'the SOTA-target metric' }),
            statTile(
              'throughput',
              num(result?.aggStepsPerSecond) === null ? '—' : `${formatStep(num(result?.aggStepsPerSecond))}/s`,
              { hint: formatWall(series.wallSeconds.at(-1) ?? null) },
            ),
          ]),
        ]),
      );

      if (mixed) {
        panel.append(
          el('p', {
            class: 'note',
            text: `This fleet mixes ${shares.length} environments, so every rate below pools scales that are not comparable — a CartPole lane caps at 1.0/step and a maze lane at 0.002. Read them as the fleet's own aggregate; the Per env tab keeps each environment on its own scale.`,
          }),
        );
      }

      const bandBlock = el('section', { class: 'block' }, [
        eyebrow('Population progress', el('span', { class: 'eyebrow-note', text: 'delivered reward rate · P10 · median · P90' })),
      ]);
      if (series.steps.length >= 2) {
        bandBlock.append(
          chart(band({
            steps: series.steps,
            p10: series.p10,
            p50: series.p50,
            p90: series.p90,
            height: 240,
            zeroLine: true,
            yUnit: 'reward / step',
            xUnit: 'steps',
          })),
        );
      } else {
        bandBlock.append(emptyState('No aggregate heartbeats', 'aggregate.jsonl is empty or missing for this root.'));
      }
      panel.append(bandBlock);

      if (series.steps.length >= 2) {
        panel.append(
          el('div', { class: 'block-grid' }, [
            el('section', { class: 'block' }, [
              eyebrow('Lanes above zero'),
              chart(timeSeries({
                steps: series.steps,
                lines: [{ label: 'positive fraction', values: series.positivity }],
                height: 150,
                yFormat: (v) => formatPercent(v, 0),
                yRange: () => [0, 1],
                yUnit: 'fraction of lanes',
              })),
            ]),
            el('section', { class: 'block' }, [
              eyebrow('Alive vs dead lanes'),
              chart(timeSeries({
                steps: series.steps,
                lines: [
                  { label: 'alive', values: series.alive, stroke: VIZ.series },
                  { label: 'dead', values: series.dead, stroke: VIZ.crit },
                ],
                height: 150,
                yFormat: (v) => formatCount(v),
                yUnit: 'lanes',
                legend: true,
              })),
            ]),
          ]),
        );
      }

      const grid = el('div', { class: 'block-grid' });

      const tiers = result?.tierFractions as Record<string, number> | undefined;
      if (tiers) {
        const items = [
          { key: 'gt0', label: '> 0   walking' },
          { key: 'gt1', label: '> 1   milestone' },
          { key: 'gt5', label: '> 5   ~SAC' },
          { key: 'gt10', label: '> 10  SOTA' },
        ]
          .filter((entry) => typeof tiers[entry.key] === 'number')
          .map((entry) => ({ label: entry.label, value: tiers[entry.key] ?? 0 }));
        if (items.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Reward tiers', el('span', { class: 'eyebrow-note', text: 'fraction of lanes' })),
              scope.hold(bars({ items, domain: [0, 1], ramp: true, format: (v) => formatPercent(v, 1), labelWidth: 130 })).root,
            ]),
          );
        }
      }

      const distribution = result?.['rateDistribution'] as Record<string, number> | undefined;
      if (distribution) {
        const items = ['p10', 'p25', 'p50', 'p75', 'p90', 'p99']
          .filter((key) => typeof distribution[key] === 'number')
          .map((key) => ({
            label: key.toUpperCase(),
            value: distribution[key] ?? 0,
            display: formatSigned(distribution[key] ?? 0),
          }));
        if (items.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Rate distribution', el('span', { class: 'eyebrow-note', text: 'reward / step' })),
              scope.hold(bars({ items, labelWidth: 56, format: (v) => formatSigned(v) })).root,
            ]),
          );
        }
      }

      const plasticity = result?.['plasticity'] as Record<string, unknown> | undefined;
      if (plasticity) {
        const rows: RangeRow[] = [];
        const zenithStep = parseQuartiles(plasticity['zenithStepQ']);
        const zenithRate = parseQuartiles(plasticity['zenithRateQ']);
        const decay = parseQuartiles(plasticity['decayRatioQ']);
        if (zenithStep) {
          rows.push({
            label: 'zenith step', q1: zenithStep[0], q2: zenithStep[1], q3: zenithStep[2],
            domain: [0, Math.max(horizon, zenithStep[2])], format: formatStep, hint: 'when each lane peaked',
          });
        }
        if (zenithRate) {
          rows.push({
            label: 'zenith rate', q1: zenithRate[0], q2: zenithRate[1], q3: zenithRate[2],
            domain: [0, Math.max(zenithRate[2] * 1.15, 1e-9)], format: (v) => formatRate(v), hint: 'peak reward rate per lane',
          });
        }
        if (decay) {
          rows.push({
            label: 'decay ratio', q1: decay[0], q2: decay[1], q3: decay[2],
            domain: [0, 1], format: (v) => formatRate(v, 3), hint: 'final ÷ zenith — 1 means no post-zenith decay',
          });
        }
        if (rows.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Plasticity', el('span', { class: 'eyebrow-note', text: 'quartiles · each row on its own scale' })),
              scope.hold(rangeBars({ rows })).root,
            ]),
          );
        }
      }
      if (grid.childElementCount > 0) panel.append(grid);

      const survival = result?.survival;
      if (survival && Object.keys(survival).length > 0) {
        const curve = kmCurve(survival, horizon);
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Survival', el('span', { class: 'eyebrow-note', text: `Kaplan-Meier · RMST ${formatStep(num(result?.['rmst']))}` })),
            chart(steppedLine({
              steps: curve.steps,
              values: curve.fraction,
              height: 160,
              yFormat: (v) => formatPercent(v, 2),
              yRange: (min) => [Math.min(min, 0.98), 1],
              yUnit: 'lanes surviving',
            })),
          ]),
        );
      } else {
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Survival'),
            emptyState('No deaths', 'Every lane reached the horizon — there is no curve to draw.'),
          ]),
        );
      }
    });

    // ---------------- per env ----------------
    if (mixed) {
      addTab('envs', 'Per env', (panel, scope) => {
        // envSlices[env] is one result PER POLICY, same as summary.results
        const sliceFor = (envId: string): SummaryPolicyResult | null =>
          envSliceResult(detail.summary, envId, policy);
        panel.append(
          el('p', {
            class: 'note',
            text: 'Reward rates are not comparable between environments — CartPole tops out at 1.0/step and HalfCheetah does not. Only positivity is shown side by side; every other number stays inside its own section.',
          }),
        );

        const withData = shares.filter((share) => sliceFor(share.envId));
        if (withData.length === 0) {
          panel.append(emptyState('No per-env slices', 'summary.json has no envSlices block for this root.'));
          return;
        }

        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Positive lanes by environment', el('span', { class: 'eyebrow-note', text: 'the one comparable quantity' })),
            scope.hold(bars({
              items: withData.map((share) => {
                const slice = sliceFor(share.envId);
                return {
                  label: envName(share.envId),
                  value: num(slice?.positiveFraction) ?? 0,
                  display: formatPercent(num(slice?.positiveFraction), 0),
                  hint: `${formatCount(num(slice?.lanes))} lanes`,
                  onClick: () => document.getElementById(`env-${envSlug(share.envId)}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
                };
              }),
              domain: [0, 1],
              format: (v) => formatPercent(v, 0),
            })).root,
          ]),
        );

        for (const share of withData) {
          const slice = sliceFor(share.envId);
          const info = isEnvId(share.envId) ? ENVS[share.envId] : null;
          const inspect = el('button', { class: 'btn btn-quiet', text: 'Inspect lanes' });
          inspect.addEventListener('click', () => {
            selectedEnv = share.envId;
            show('lanes');
            requestLaneEnv?.(share.envId);
          });
          const section = el('section', { class: 'env-section' }, [
            el('div', { class: 'env-head' }, [
              el('h2', { class: 'env-name', text: envName(share.envId) }),
              el('span', { class: 'env-sub', text: info ? `${info.action.kind} ${info.action.n} · ${info.category}` : '' }),
              inspect,
            ]),
            el('div', { class: 'stat-row' }, [
              statTile('IQM', formatSigned(num(slice?.iqmDeliveredAll), 4), {
                tone: tone(num(slice?.iqmDeliveredAll)), hint: 'all lanes',
              }),
              statTile('recent IQM', formatSigned(num(slice?.['iqmRecent']), 4), { hint: 'last window' }),
              statTile('positive', formatPercent(num(slice?.positiveFraction), 0), {
                hint: `${formatCount(num(slice?.lanes))} lanes`,
              }),
              statTile('best lane', formatRate(num(slice?.bestLaneRate)), {
                hint: `seed ${formatCount(num(slice?.bestLaneSeed))}`,
              }),
              statTile('igniters', formatCount(num(slice?.['igniters'])), {
                hint: `${formatCount(num(slice?.['ignitersSustained']))} sustained`,
              }),
              statTile('reward / GFLOP', formatRate(num(slice?.['bestRewardPerGFLOP'])), { hint: 'best lane' }),
            ]),
          ]);
          section.id = `env-${envSlug(share.envId)}`;
          section.style.setProperty('--section-accent', accentCss(envSlug(share.envId)));
          panel.append(section);
        }
      });
    }

    // ---------------- lanes ----------------
    addTab('lanes', 'Lanes', (panel, scope) => {
      let columnar: FleetColumnar | null = null;
      let laneChart: ChartHandle | null = null;
      let laneTable: TableHandle<LaneSeries> | null = null;
      let shown: LaneSeries[] = [];
      let stale = false;
      scope.hold({
        destroy() {
          stale = true; // a policy switch rebuilds this panel; drop the in-flight fetch
          laneChart?.destroy();
          laneTable?.destroy();
        },
      });

      const controls = el('div', { class: 'controls controls-inline' });
      const laneStatus = el('div', { class: 'view-status' }, [spinner('Reading fleet.jsonl')]);
      const chartHost = el('section', { class: 'block' });
      const tableHost = el('section', { class: 'block' });
      let select: HTMLSelectElement | null = null;

      if (mixed) {
        select = el('select', { class: 'input input-select', ariaLabel: 'Environment' });
        for (const share of shares) select.append(el('option', { value: share.envId, text: envName(share.envId) }));
        select.value = selectedEnv;
        select.addEventListener('change', () => {
          selectedEnv = select?.value ?? selectedEnv;
          paintLanes();
        });
        controls.append(el('span', { class: 'control-label', text: 'environment' }), select);
        requestLaneEnv = (envId) => {
          if (!select) return;
          selectedEnv = envId;
          select.value = envId;
          paintLanes();
        };
      }

      panel.append(controls, laneStatus, chartHost, tableHost);

      function paintLanes(): void {
        if (!columnar) return;
        laneStatus.hidden = false;
        const field = pickField(columnar, ['health.rewardRate', 'rewardRate', 'rewardSum']);
        if (!field) {
          laneStatus.replaceChildren(emptyState('No lane series', 'The fleet endpoint returned no columnar fields.'));
          return;
        }
        const all = laneSeries(columnar, field, mixed ? selectedEnv : undefined);
        if (all.length === 0) {
          laneStatus.replaceChildren(emptyState('No lanes for this environment'));
          chartHost.replaceChildren();
          tableHost.replaceChildren();
          return;
        }
        laneStatus.replaceChildren();
        laneStatus.hidden = true;
        shown = capLanes(all, LANE_CAP);
        const capped = shown.length < all.length;
        const label = field.split('.').pop() ?? field;

        laneChart?.destroy();
        laneChart = spaghetti({
          steps: columnar.steps,
          lanes: shown.map((lane) => ({ key: String(lane.lane), label: `seed ${lane.seed}`, values: lane.values })),
          height: 280,
          zeroLine: true,
          yUnit: label,
          xUnit: 'steps',
          onLaneFocus: (key) => laneTable?.highlight(key),
        });
        chartHost.replaceChildren(
          eyebrow('Per-lane trajectories', el('span', {
            class: 'eyebrow-note',
            text: capped
              ? `${shown.length} of ${all.length} lanes · extremes and median always kept`
              : `${all.length} lanes`,
          })),
          laneChart.root,
        );
        laneChart.refresh();

        const recordedSeeds = new Set(
          detail.recordings.filter((r) => !mixed || r.envId === selectedEnv).map((r) => r.seed),
        );
        laneTable?.destroy();
        laneTable = dataTable<LaneSeries>({
          columns: [
            { key: 'seed', label: 'seed', value: (row) => row.seed, align: 'left', render: (row) => String(row.seed) },
            { key: 'lane', label: 'lane', value: (row) => row.lane },
            {
              key: 'rate',
              label,
              value: (row) => row.last,
              render: (row) => el('span', { class: `tone-${tone(row.last)}`, text: formatSigned(row.last, 4) }),
            },
            {
              key: 'replay',
              label: 'replay',
              value: (row) => (recordedSeeds.has(row.seed) ? 1 : 0),
              render: (row) => recordedSeeds.has(row.seed)
                ? el('a', {
                    class: 'link-quiet',
                    href: `#/theater/${encodeURIComponent(id)}/${encodeURIComponent(row.envId)}/${row.seed}`,
                    text: 'watch',
                  })
                : el('span', { class: 'tone-dim', text: '—' }),
            },
          ],
          rows: rankLanes(all),
          sortKey: 'rate',
          sortDirection: 'desc',
          rowKey: (row) => String(row.lane),
          onRowHover: (row) => {
            if (!row) {
              laneChart?.focusSeries(null);
              return;
            }
            const index = shown.findIndex((lane) => lane.lane === row.lane);
            laneChart?.focusSeries(index >= 0 ? index : null);
          },
          pageSize: 24,
          emptyLabel: 'No lanes',
        });
        tableHost.replaceChildren(
          eyebrow('Lane ledger', el('span', { class: 'eyebrow-note', text: 'sortable · hover to trace a lane above' })),
          laneTable.root,
        );
      }

      // The server filters fleet.jsonl by policy (defaulting to `oak`), so the
      // selected policy has to travel with the request or a zero/random view
      // silently reads oak's lanes.
      api.fleet(id, { fields: ['health.rewardRate'], policy }).then(
        (data) => {
          if (destroyed || stale) return;
          columnar = data;
          paintLanes();
        },
        (error: unknown) => {
          if (destroyed || stale) return;
          laneStatus.hidden = false;
          laneStatus.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
        },
      );
    });

    // ---------------- recordings ----------------
    addTab('recordings', detail.recordings.length > 0 ? `Recordings · ${detail.recordings.length}` : 'Recordings', (panel) => {
      if (detail.recordings.length === 0) {
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Recordings', el('span', { class: 'eyebrow-note', text: 'theater replay sources' })),
            emptyState(
              'No recordings on this root',
              'The theater replays per-lane actions.bin streams (record v3) and this run wrote none. '
                + 'Re-run the fleet with OAK_RECORD_ACTIONS=1 for every lane, or OAK_RECORD_SEEDS=3,8 for a couple.',
              [linkButton('Find runs that have them', '#/', { title: 'The gallery has a “recordings” filter' })],
            ),
          ]),
        );
        return;
      }
      panel.append(
        eyebrow(
          'Recordings',
          el('span', {
            class: 'eyebrow-note',
            text: `${formatCount(detail.recordings.length)} replayable lanes · pick one to open the theater`,
          }),
        ),
      );
      const byEnv = new Map<string, RunDetail['recordings']>();
      for (const recording of detail.recordings) {
        const list = byEnv.get(recording.envId) ?? [];
        list.push(recording);
        byEnv.set(recording.envId, list);
      }
      for (const [envId, list] of byEnv) {
        const seeds = el('div', { class: 'seed-grid' });
        for (const recording of [...list].sort((a, b) => a.seed - b.seed)) {
          seeds.append(
            el('a', {
              class: 'seed-chip',
              href: `#/theater/${encodeURIComponent(id)}/${encodeURIComponent(recording.envId)}/${recording.seed}`,
              title: `${formatCount(recording.steps)} recorded steps`,
            }, [
              el('span', { class: 'seed-num', text: `seed ${recording.seed}` }),
              el('span', { class: 'seed-steps', text: formatStep(recording.steps) }),
            ]),
          );
        }
        const section = el('section', { class: 'env-section' }, [
          el('div', { class: 'env-head' }, [
            el('h2', { class: 'env-name', text: envName(envId) }),
            el('span', { class: 'env-sub', text: `${list.length} lane${list.length === 1 ? '' : 's'}` }),
          ]),
          seeds,
        ]);
        section.style.setProperty('--section-accent', accentCss(envSlug(envId)));
        panel.append(section);
      }
    });

    // ---------------- assemble ----------------
    for (const tab of tabs) {
      const button = el('button', { class: 'tab', text: tab.label, role: 'tab', dataset: { tab: tab.key } });
      button.addEventListener('click', () => show(tab.key));
      tabBar.append(button);
      const panel = panels.get(tab.key);
      if (panel) panelHost.append(panel.node);
    }

    if (policies.length > 1) {
      const policySelect = el('select', { class: 'input input-select', ariaLabel: 'Policy' });
      for (const name of policies) policySelect.append(el('option', { value: name, text: name }));
      policySelect.value = policy;
      policySelect.addEventListener('change', () => {
        policy = policySelect.value;
        // numbers AND charts AND the header rebuild together — nothing survives
        // a policy switch except the run's own identity.
        resetPanels();
        syncPolicyHeader();
        show(currentTab());
      });
      tabBar.append(
        el('div', { class: 'tabs-trailing' }, [el('span', { class: 'control-label', text: 'policy' }), policySelect]),
      );
    }

    body.append(tabBar, panelHost);
    show(currentTab());
  }

  return {
    destroy() {
      destroyed = true;
      for (const item of disposables) item.destroy();
      disposables.length = 0;
      root.remove();
    },
  };
};
