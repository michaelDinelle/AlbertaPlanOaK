import { useMemo, useRef } from 'react';
import { clsx } from 'clsx';
import { Area, Line } from 'recharts';
import type { Policy, RunDetail } from '@shared/types';
import { formatStep } from '../lib/format';
import { envLanes } from '../lib/run';
import { heroSeed } from '../lib/selection';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { retentionWindow } from '../tape/derive';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { scrubToStep } from '../theater/controller';
import { navigate } from '../url/useUrlState';
import {
  AXIS_WIDTH,
  CHART_MARGIN,
  Chip,
  Label,
  POLICY_ORDER,
  POLICY_TOKENS,
  Playhead,
  TXT,
  TimeChart,
  bandProps,
  bandRows,
  lineProps,
  seriesRows,
  type XY,
} from './charts';
import { SeedList } from './SeedList';

function summaryPoints(run: RunDetail | null, envId: string | null): XY[] | null {
  if (run === null || envId === null) return null;
  const summary = run.summary;
  const env = summary?.envs[envId];
  const z = env?.curves.iqmZ;
  if (summary === null || summary === undefined || env === undefined || z === undefined) return null;
  const span = env.ceiling - env.floor;
  return seriesRows(
    summary.curveSteps,
    z.map((value) => (value === null ? null : env.floor + value * span)),
  );
}

const SPARSE_SAMPLES = 4;

function sparseOrigin(points: readonly { x: number }[], horizon: number): number {
  if (points.length === 0 || points.length >= SPARSE_SAMPLES) return 0;
  const first = points[0]?.x ?? 0;
  return first >= horizon ? 0 : first;
}

export function RewardView({
  runs,
  active,
}: {
  runs: Record<Policy, RunDetail | null>;
  active: RunDetail | null;
}) {
  const sel = useSelection();
  const lanes = useMemo(
    () => (active === null || sel.envId === null ? null : envLanes(active.meta, sel.envId)),
    [active, sel.envId],
  );
  const spec = useMemo<(DeriveSpec & { mode: 'windowedRates' }) | null>(
    () => (lanes === null || active === null ? null : { mode: 'windowedRates', lanes: lanes.lanes, win: retentionWindow(active.meta.steps) }),
    [lanes, active],
  );
  const rates = useTapeSnapshot(active?.id ?? null, sel.envId, 'windowedRates', spec);

  const hero = heroSeed(sel.selection);
  const heroIdx = hero === null || lanes === null ? -1 : lanes.seeds.indexOf(hero);
  const horizon = active?.meta.steps ?? runs.oak?.meta.steps ?? 0;
  const tone = POLICY_TOKENS[sel.policy];

  const avgPts = useMemo(() => (rates === null ? [] : seriesRows(rates.x, rates.iqm)), [rates]);
  const band = useMemo(() => (rates === null ? [] : bandRows(rates.x, rates.p10, rates.p90)), [rates]);
  const heroPts = useMemo(
    () => (rates === null || heroIdx < 0 ? [] : seriesRows(rates.x, rates.perLane[heroIdx] ?? [])),
    [rates, heroIdx],
  );
  const comparisons = useMemo(
    () =>
      POLICY_ORDER.filter((policy) => policy !== sel.policy).map((policy) => ({
        policy,
        run: runs[policy],
        pts: summaryPoints(runs[policy], sel.envId),
      })),
    [runs, sel.policy, sel.envId],
  );

  const stale = comparisons
    .filter(({ run, pts }) => run !== null && pts === null)
    .map(({ policy }) => STR.policy[policy]);

  const dragging = useRef(false);
  const stepAt = (event: React.PointerEvent<HTMLDivElement>): number => {
    const rect = event.currentTarget.getBoundingClientRect();
    const inner = Math.max(1, rect.width - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH);
    const frac = Math.min(1, Math.max(0, (event.clientX - rect.left - CHART_MARGIN.left - AXIS_WIDTH) / inner));
    return Math.round(frac * horizon);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="px-2">
        <Label hint={STR.reward.titleHint(formatStep(retentionWindow(horizon)))}>{STR.reward.title}</Label>
      </div>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-2">
        <Chip swatch={tone} label={STR.reward.average(STR.policy[sel.policy])} />
        <Chip swatch="var(--color-seed)" label={STR.reward.hero} />
        <Chip swatch={tone} label={STR.reward.range} block />
        {comparisons.map(
          ({ policy, pts }) =>
            pts !== null && (
              <Chip key={policy} swatch={POLICY_TOKENS[policy]} label={STR.reward.average(STR.policy[policy])} />
            ),
        )}
      </div>
      <div
        className="relative cursor-ew-resize touch-none select-none rounded-md transition-colors hover:bg-white/5"
        onPointerDown={(event) => {
          dragging.current = true;
          try {
            event.currentTarget.setPointerCapture(event.pointerId);
          } catch {}
          scrubToStep(stepAt(event));
        }}
        onPointerMove={(event) => {
          if (dragging.current) scrubToStep(stepAt(event));
        }}
        onPointerUp={(event) => {
          if (!dragging.current) return;
          dragging.current = false;
          const step = stepAt(event);
          scrubToStep(step);
          navigate({ at: step });
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
      >
        <TimeChart
          label={STR.reward.title}
          height={188}
          xMin={sparseOrigin(avgPts, horizon)}
          xMax={Math.max(1, horizon)}
          xLabel={STR.axis.steps}
          yLabel={STR.axis.rewardRate}
          overlay={<Playhead total={Math.max(1, horizon)} from={sparseOrigin(avgPts, horizon)} />}
        >
          <Area {...bandProps(tone)} data={band} />
          <Line {...lineProps(tone, 1.6)} name={STR.reward.average(STR.policy[sel.policy])} data={avgPts} />
          <Line {...lineProps('var(--color-seed)', 1.2)} name={STR.reward.hero} data={heroPts} />
          {comparisons.map(
            ({ policy, pts }) =>
              pts !== null && (
                <Line
                  key={policy}
                  {...lineProps(POLICY_TOKENS[policy], 1.2)}
                  name={STR.reward.average(STR.policy[policy])}
                  data={pts}
                />
              ),
          )}
        </TimeChart>
        {rates === null && (
          <div className={clsx('absolute inset-0 flex items-center justify-center', TXT.bodyDim)}>
            {active === null ? STR.reward.waiting : STR.reward.reading}
          </div>
        )}
      </div>
      {stale.length > 0 && (
        <div className={clsx('px-2', TXT.bodyDim)}>{STR.reward.stale(stale.join(' and '))}</div>
      )}
      <SeedList active={active} />
    </div>
  );
}
