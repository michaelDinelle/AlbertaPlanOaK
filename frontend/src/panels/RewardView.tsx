import { useMemo, useRef } from 'react';
import { clsx } from 'clsx';
import { Area, Line } from 'recharts';
import type { Policy, RunDetail } from '../data/types';
import { formatStep } from '../lib/format';
import { envLanes } from '../lib/run';
import { heroSeed } from '../lib/selection';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { retentionWindow, type BinnedRates } from '../tape/derive';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import {
  AXIS_WIDTH,
  CHART_MARGIN,
  Chip,
  Label,
  BEST_TOKEN,
  POLICY_ORDER,
  POLICY_TOKENS,
  Playhead,
  TXT,
  TimeChart,
  bandProps,
  bandRows,
  lineProps,
  seriesRows,
} from './charts';
import { SeedList } from './SeedList';
import { useStepScrub } from './useStepScrub';

const SPARSE_SAMPLES = 4;

type SeriesKey = Policy | 'best';

function sparseOrigin(points: readonly { x: number }[], horizon: number): number {
  if (points.length === 0 || points.length >= SPARSE_SAMPLES) return 0;
  const first = points[0]?.x ?? 0;
  return first >= horizon ? 0 : first;
}

export function RewardView({
  runs,
  best,
  active,
}: {
  runs: Record<Policy, RunDetail | null>;
  best: RunDetail | null;
  active: RunDetail | null;
}) {
  const sel = useSelection();
  const envId = sel.envId;
  const lanes = useMemo(
    () => (active === null || envId === null ? null : envLanes(active.meta, envId)),
    [active, envId],
  );
  const horizon = active?.meta.steps ?? runs.oak?.meta.steps ?? 0;
  const spec = useMemo<(DeriveSpec & { mode: 'windowedRates' }) | null>(
    () => (lanes === null || active === null ? null : { mode: 'windowedRates', lanes: lanes.lanes, win: retentionWindow(active.meta.steps) }),
    [lanes, active],
  );
  const rates = useTapeSnapshot(active?.id ?? null, envId, 'windowedRates', spec);

  const compareSpec = (run: RunDetail | null): (DeriveSpec & { mode: 'windowedRates' }) | null =>
    run === null || envId === null ? null : { mode: 'windowedRates', lanes: envLanes(run.meta, envId).lanes, win: retentionWindow(run.meta.steps) };
  const oakCompare = useMemo(() => compareSpec(runs.oak), [runs.oak, envId]);
  const ppoCompare = useMemo(() => compareSpec(runs.ppo), [runs.ppo, envId]);
  const randomCompare = useMemo(() => compareSpec(runs.random), [runs.random, envId]);
  const bestCompare = useMemo(() => compareSpec(best), [best, envId]);
  const compared: Record<SeriesKey, BinnedRates | null> = {
    oak: useTapeSnapshot(oakCompare === null ? null : runs.oak?.id ?? null, envId, 'windowedRates', oakCompare),
    ppo: useTapeSnapshot(ppoCompare === null ? null : runs.ppo?.id ?? null, envId, 'windowedRates', ppoCompare),
    random: useTapeSnapshot(randomCompare === null ? null : runs.random?.id ?? null, envId, 'windowedRates', randomCompare),
    best: useTapeSnapshot(bestCompare === null ? null : best?.id ?? null, envId, 'windowedRates', bestCompare),
  };

  const hero = heroSeed(sel.selection);
  const heroIdx = hero === null || lanes === null ? -1 : lanes.seeds.indexOf(hero);
  const tone = POLICY_TOKENS[sel.policy];

  const avgPts = useMemo(() => (rates === null ? [] : seriesRows(rates.x, rates.iqm)), [rates]);
  const band = useMemo(() => (rates === null ? [] : bandRows(rates.x, rates.p10, rates.p90)), [rates]);
  const heroPts = useMemo(
    () => (rates === null || heroIdx < 0 ? [] : seriesRows(rates.x, rates.perLane[heroIdx] ?? [])),
    [rates, heroIdx],
  );
  const primaryLabel = STR.reward.average(STR.policy[sel.policy]);
  const summaryLines = useMemo(
    () => [
      ...POLICY_ORDER.filter((policy) => policy !== sel.policy && runs[policy] !== null).map((policy) => ({ key: policy as SeriesKey, label: STR.policy[policy], token: POLICY_TOKENS[policy] })),
      ...(best === null ? [] : [{ key: 'best' as SeriesKey, label: STR.best, token: BEST_TOKEN }]),
    ].map((item) => {
      const rates = compared[item.key];
      return { ...item, pts: rates === null ? null : seriesRows(rates.x, rates.iqm) };
    }),
    [runs, best, sel.policy, compared.oak, compared.ppo, compared.random, compared.best],
  );

  const xMax = Math.max(1, horizon);
  const xMin = sparseOrigin(avgPts, horizon);
  const chartRef = useRef<HTMLDivElement | null>(null);
  const scrub = useStepScrub<HTMLDivElement>(
    () => ({ min: xMin, max: xMax }),
    { left: CHART_MARGIN.left + AXIS_WIDTH, right: CHART_MARGIN.right },
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="px-2">
        <Label hint={STR.reward.titleHint(formatStep(retentionWindow(horizon)))}>{STR.reward.title}</Label>
      </div>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-2">
        <Chip swatch={tone} label={primaryLabel} />
        <Chip swatch="var(--color-seed)" label={STR.reward.hero} />
        <Chip swatch={tone} label={STR.reward.range} block />
        {summaryLines.map(({ key, label, token, pts }) => pts !== null && <Chip key={key} swatch={token} label={STR.reward.average(label)} />)}
      </div>
      <div
        ref={chartRef}
        className="relative cursor-ew-resize touch-none select-none rounded-md transition-colors hover:bg-white/5"
        {...scrub}
      >
        <TimeChart
          label={STR.reward.title}
          height={188}
          xMin={xMin}
          xMax={xMax}
          xLabel={STR.axis.steps}
          yLabel={STR.axis.rewardRate}
          overlay={<Playhead total={xMax} from={xMin} />}
        >
          <Area {...bandProps(tone)} data={band} />
          <Line {...lineProps(tone, 1.6)} name={primaryLabel} data={avgPts} />
          <Line {...lineProps('var(--color-seed)', 1.2)} name={STR.reward.hero} data={heroPts} />
          {summaryLines.map(
            ({ key, label, token, pts }) =>
              pts !== null && <Line key={key} {...lineProps(token, 1.2)} name={STR.reward.average(label)} data={pts} />,
          )}
        </TimeChart>
        {rates === null && (
          <div className={clsx('absolute inset-0 flex items-center justify-center', TXT.bodyDim)}>
            {active === null ? STR.reward.waiting : STR.reward.reading}
          </div>
        )}
      </div>
      <SeedList active={active} />
    </div>
  );
}
