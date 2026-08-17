import { useMemo } from 'react';
import uPlot from 'uplot';
import type { PlasticityTrack, RunDetail, RunSummary } from '@shared/types';
import { Card, CardHead, LegendChip } from '../../components/ui/card';
import { cssVar, monoFont, tooltipPlugin, withAlpha, zeroLinePlugin } from '../../charts/plugins';
import { timeSeriesOptions, type OptionsBuilder, type SeriesLine } from '../../charts/specs';
import { ENV_SYNC_KEY } from '../../charts/sync';
import { useUPlot } from '../../charts/useUPlot';
import { formatPercent, formatRate } from '../../lib/format';
import { envLanes } from '../../lib/run';
import { useSelection } from '../../state/selectionStore';
import { useTapeSnapshot, type DeriveSpec } from '../../tape/store';

const POLICY_TOKENS = [
  { key: 'oak', name: 'OaK', token: '--color-oak' },
  { key: 'ppo', name: 'PPO', token: '--color-ppo' },
  { key: 'random', name: 'random', token: '--color-random' },
] as const;

function ChartCell({
  title,
  titleHint,
  legend,
  note,
  chartRef,
}: {
  title: string;
  titleHint?: string;
  legend?: React.ReactNode;
  note: string | null;
  chartRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <Card>
      <CardHead title={<span title={titleHint}>{title}</span>}>{legend}</CardHead>
      <div className="relative h-40">
        <div ref={chartRef} className="absolute inset-0" />
        {note !== null && (
          <div className="absolute inset-0 flex items-center justify-center px-4">
            <span className="rounded-sm bg-raised/85 px-2 py-1 text-center font-mono text-2xs text-ink-faint">{note}</span>
          </div>
        )}
      </div>
    </Card>
  );
}

function zFormat(value: number): string {
  return value.toFixed(2);
}

function RphCell({ runs, envId }: { runs: (RunSummary | null)[]; envId: string | null }) {
  const keeper = runs.find((s) => s !== null) ?? null;
  const { lines, data } = useMemo(() => {
    if (keeper === null || envId === null) return { lines: [] as SeriesLine[], data: [[]] as unknown as uPlot.AlignedData };
    const x = keeper.curveSteps;
    const lines: SeriesLine[] = [];
    const columns: (number | null)[][] = [];
    runs.forEach((summary, i) => {
      const meta = POLICY_TOKENS[i];
      const curves = summary?.envs[envId]?.curves.rateZ;
      if (meta === undefined || curves === undefined) return;
      if (summary === null || summary.curveSteps.length !== x.length) return;
      const color = cssVar(meta.token);
      for (const [pct, track, strong] of [
        ['p5', curves.p5, false],
        ['p95', curves.p95, false],
        ['p50', curves.p50, true],
      ] as const) {
        if (track.length !== x.length) continue;
        lines.push({
          label: `${meta.name} ${pct}`,
          stroke: strong ? color : withAlpha(color, 0.38),
          width: strong ? 1.6 : 1,
        });
        columns.push(track);
      }
    });
    return { lines, data: [x, ...columns] as unknown as uPlot.AlignedData };
  }, [runs, keeper, envId]);

  const build = useMemo<OptionsBuilder>(
    () =>
      timeSeriesOptions({
        lines,
        plugins: [tooltipPlugin({ yFormat: zFormat, xUnit: 'steps' }), zeroLinePlugin()],
        yFormat: zFormat,
      }),
    [lines],
  );
  const chart = useUPlot({ build, data, deps: [envId, lines.map((l) => l.label).join(',')], syncKey: ENV_SYNC_KEY });
  return (
    <ChartCell
      title="Run percentiles — rate (z)"
      titleHint="Windowed final-rate z curves of each policy's actual 5th, 50th and 95th-percentile seeds"
      legend={
        <>
          {POLICY_TOKENS.map((p) => (
            <LegendChip key={p.key} swatch={`var(${p.token})`} label={p.name} />
          ))}
        </>
      }
      note={lines.length === 0 ? 'no summary on this run yet' : null}
      chartRef={chart.ref}
    />
  );
}

function RhoCell({ detail }: { detail: RunDetail | null }) {
  const sel = useSelection();
  const lanes = useMemo(
    () => (detail === null || sel.envId === null ? null : envLanes(detail.meta, sel.envId)),
    [detail, sel.envId],
  );
  const ratesSpec = useMemo<(DeriveSpec & { mode: 'rates' }) | null>(
    () => (lanes === null ? null : { mode: 'rates', lanes: lanes.lanes }),
    [lanes],
  );
  const rhoSpec = useMemo<(DeriveSpec & { mode: 'rho' }) | null>(
    () => (lanes === null ? null : { mode: 'rho', lanes: lanes.lanes }),
    [lanes],
  );
  const rates = useTapeSnapshot(sel.runId, sel.envId, 'rates', ratesSpec);
  const rho = useTapeSnapshot(sel.runId, sel.envId, 'rho', rhoSpec);

  const data = useMemo<uPlot.AlignedData>(() => {
    if (rates === null || rho === null) return [[], [], []] as unknown as uPlot.AlignedData;
    const n = Math.min(rho.x.length, rates.iqm.length);
    return [rho.x.slice(0, n), rates.iqm.slice(0, n), rho.iqm.slice(0, n)] as unknown as uPlot.AlignedData;
  }, [rates, rho]);

  const build = useMemo<OptionsBuilder>(
    () =>
      timeSeriesOptions({
        lines: [
          { label: 'measured IQM', stroke: cssVar('--color-oak'), width: 1.6 },
          { label: 'agent ρ̄ IQM', stroke: cssVar('--color-ramp-4'), width: 1.2, dash: [4, 3] },
        ],
        plugins: [tooltipPlugin({ xUnit: 'steps' }), zeroLinePlugin()],
      }),
    [],
  );
  const chart = useUPlot({ build, data, deps: [sel.runId, sel.envId], syncKey: ENV_SYNC_KEY });
  return (
    <ChartCell
      title="ρ̄ vs measured"
      titleHint="The agent's tracked average-reward estimate ρ̄ against the reward rate measured from the tape"
      legend={
        <>
          <LegendChip swatch="var(--color-oak)" label="measured" />
          <LegendChip swatch="var(--color-ramp-4)" label="ρ̄" dashed />
        </>
      }
      note={rates === null || rho === null ? 'reading the tape…' : null}
      chartRef={chart.ref}
    />
  );
}

function SurvivalCell({ detail }: { detail: RunDetail | null }) {
  const sel = useSelection();
  const lanes = useMemo(
    () => (detail === null || sel.envId === null ? null : envLanes(detail.meta, sel.envId)),
    [detail, sel.envId],
  );
  const spec = useMemo<(DeriveSpec & { mode: 'survival' }) | null>(
    () => (lanes === null ? null : { mode: 'survival', lanes: lanes.lanes }),
    [lanes],
  );
  const survival = useTapeSnapshot(sel.runId, sel.envId, 'survival', spec);

  const data = useMemo<uPlot.AlignedData>(() => {
    if (survival === null) return [[], [], []] as unknown as uPlot.AlignedData;
    return [survival.x, survival.surviving, survival.positive] as unknown as uPlot.AlignedData;
  }, [survival]);

  const build = useMemo<OptionsBuilder>(
    () =>
      timeSeriesOptions({
        lines: [
          { label: 'alive', stroke: cssVar('--color-good'), width: 1.6, stepped: true },
          { label: 'cum reward > 0', stroke: withAlpha(cssVar('--color-ramp-2'), 0.8), width: 1, stepped: true },
        ],
        plugins: [tooltipPlugin({ yFormat: (v) => formatPercent(v, 0), xUnit: 'steps' })],
        yFormat: (v) => formatPercent(v, 0),
        yRange: () => [0, 1.04],
      }),
    [],
  );
  const chart = useUPlot({ build, data, deps: [sel.runId, sel.envId], syncKey: ENV_SYNC_KEY });
  return (
    <ChartCell
      title="Survival"
      titleHint="Fraction of this environment's 64 lanes still alive, and the fraction whose cumulative reward is positive"
      legend={
        <>
          <LegendChip swatch="var(--color-good)" label="alive" />
          <LegendChip swatch="var(--color-ramp-2)" label="reward > 0" />
        </>
      }
      note={survival === null ? 'reading the tape…' : null}
      chartRef={chart.ref}
    />
  );
}

interface PlasticityLine {
  line: SeriesLine;
  values: (number | null)[];
}

function plasticityLine(
  tracks: Record<string, PlasticityTrack> | undefined,
  field: string,
  length: number,
  line: SeriesLine,
): PlasticityLine | null {
  const track = tracks?.[field];
  if (track === undefined || track.iqm.length !== length) return null;
  return { line, values: track.iqm };
}

function PlasticityCell({ keeper, ppo, envId }: { keeper: RunSummary | null; ppo: RunSummary | null; envId: string | null }) {
  const { lines, data, hasAny, hasTracks } = useMemo(() => {
    const x = (keeper?.plasticitySteps.length ?? 0) > 0 ? keeper?.plasticitySteps : ppo?.plasticitySteps;
    if (x === undefined || x.length === 0 || envId === null) {
      return { lines: [] as SeriesLine[], data: [[]] as unknown as uPlot.AlignedData, hasAny: false, hasTracks: false };
    }
    const oakColor = cssVar('--color-oak');
    const ppoColor = cssVar('--color-ppo');
    const candidates = [
      plasticityLine(keeper?.envs[envId]?.plasticity, 'dormantFraction', x.length, {
        label: 'OaK dormant', stroke: oakColor, width: 1.6,
      }),
      plasticityLine(ppo?.envs[envId]?.plasticity, 'dormantFraction', x.length, {
        label: 'PPO dormant', stroke: ppoColor, width: 1.6,
      }),
      plasticityLine(keeper?.envs[envId]?.plasticity, 'erank', x.length, {
        label: 'OaK erank', stroke: withAlpha(oakColor, 0.75), width: 1.1, dash: [4, 3], scale: 'e',
      }),
      plasticityLine(ppo?.envs[envId]?.plasticity, 'erank', x.length, {
        label: 'PPO erank', stroke: withAlpha(ppoColor, 0.75), width: 1.1, dash: [4, 3], scale: 'e',
      }),
    ].filter((c): c is PlasticityLine => c !== null);
    const hasAny = candidates.some((c) => c.values.some((v) => v !== null && Number.isFinite(v)));
    return {
      lines: candidates.map((c) => c.line),
      data: [x, ...candidates.map((c) => c.values)] as unknown as uPlot.AlignedData,
      hasAny,
      hasTracks: candidates.length > 0,
    };
  }, [keeper, ppo, envId]);

  const build = useMemo<OptionsBuilder>(
    () => (size) => {
      const options = timeSeriesOptions({
        lines,
        plugins: [tooltipPlugin({ yFormat: (v) => formatRate(v, 3), xUnit: 'steps' })],
        yFormat: (v) => formatPercent(v, 0),
        yRange: () => [0, 1.04],
      })(size);
      options.scales = {
        ...options.scales,
        e: { range: (_u, min, max) => uPlot.rangeNum(Math.min(0, min), Math.max(1, max), 0.08, true) },
      };
      options.axes = [
        ...(options.axes ?? []),
        {
          scale: 'e',
          side: 1,
          stroke: cssVar('--color-ink-faint'),
          font: monoFont(10),
          grid: { show: false },
          ticks: { show: false },
          size: 36,
          values: (_u, splits) => splits.map((v) => formatRate(v, 2)),
        },
      ];
      return options;
    },
    [lines],
  );
  const chart = useUPlot({ build, data, deps: [envId, lines.map((l) => l.label).join(',')], syncKey: ENV_SYNC_KEY });
  return (
    <ChartCell
      title="Plasticity — dormant · erank"
      titleHint="Dormant-feature fraction (left axis) and effective rank of the representation (right axis), IQM across lanes"
      legend={
        <>
          <LegendChip swatch="var(--color-oak)" label="OaK" />
          <LegendChip swatch="var(--color-ppo)" label="PPO" />
          <LegendChip swatch="var(--color-ink-dim)" label="erank" dashed dim />
        </>
      }
      note={hasAny ? null : hasTracks
        ? 'plasticity lands with the next instrumented run'
        : 'no plasticity block in this summary'}
      chartRef={chart.ref}
    />
  );
}

export function EnvDiagnostics({
  detail,
  ppo,
  random,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
}) {
  const sel = useSelection();
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );
  return (
    <>
      <RphCell runs={summaries} envId={sel.envId} />
      <RhoCell detail={detail} />
      <SurvivalCell detail={detail} />
      <PlasticityCell keeper={detail?.summary ?? null} ppo={ppo?.summary ?? null} envId={sel.envId} />
    </>
  );
}
