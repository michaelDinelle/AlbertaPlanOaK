import { useMemo, useRef } from 'react';
import { Area, Line } from 'recharts';
import type { RunDetail } from '@shared/types';
import { envLanes } from '../lib/run';
import { heroSeed } from '../lib/selection';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { scrubToStep } from '../theater/controller';
import { navigate } from '../url/useUrlState';
import {
  AXIS_WIDTH,
  CHART_MARGIN,
  Chip,
  Playhead,
  TimeChart,
  bandProps,
  bandRows,
  lineProps,
  seriesRows,
  type XY,
} from './charts';
import { SeedList } from './SeedList';

function baselinePoints(run: RunDetail | null, envId: string | null): XY[] | null {
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

export function RewardView({
  detail,
  ppo,
  random,
  active,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
  active: RunDetail | null;
}) {
  const sel = useSelection();
  const lanes = useMemo(
    () => (detail === null || sel.envId === null ? null : envLanes(detail.meta, sel.envId)),
    [detail, sel.envId],
  );
  const spec = useMemo<(DeriveSpec & { mode: 'rates' }) | null>(
    () => (lanes === null ? null : { mode: 'rates', lanes: lanes.lanes }),
    [lanes],
  );
  const rates = useTapeSnapshot(sel.runId, sel.envId, 'rates', spec);

  const hero = heroSeed(sel.selection);
  const heroIdx = hero === null || lanes === null ? -1 : lanes.seeds.indexOf(hero);
  const horizon = detail?.meta.steps ?? 0;

  const oakPts = useMemo(() => (rates === null ? [] : seriesRows(rates.x, rates.iqm)), [rates]);
  const band = useMemo(() => (rates === null ? [] : bandRows(rates.x, rates.p10, rates.p90)), [rates]);
  const heroPts = useMemo(
    () => (rates === null || heroIdx < 0 ? [] : seriesRows(rates.x, rates.perLane[heroIdx] ?? [])),
    [rates, heroIdx],
  );
  const ppoPts = useMemo(() => baselinePoints(ppo, sel.envId), [ppo, sel.envId]);
  const randomPts = useMemo(() => baselinePoints(random, sel.envId), [random, sel.envId]);

  const stale = [
    ppo !== null && ppoPts === null ? STR.policy.ppo : null,
    random !== null && randomPts === null ? STR.policy.random : null,
  ].filter((name): name is string => name !== null);

  const dragging = useRef(false);
  const stepAt = (event: React.PointerEvent<HTMLDivElement>): number => {
    const rect = event.currentTarget.getBoundingClientRect();
    const inner = Math.max(1, rect.width - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH);
    const frac = Math.min(1, Math.max(0, (event.clientX - rect.left - CHART_MARGIN.left - AXIS_WIDTH) / inner));
    return Math.round(frac * horizon);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-2">
        <span className="text-2xs uppercase tracking-label text-ink-faint">{STR.reward.title}</span>
        <Chip swatch="var(--color-oak)" label={STR.reward.oak} />
        <Chip swatch="var(--color-seed)" label={STR.reward.hero} />
        <span className="flex items-center gap-1 font-mono text-2xs text-ink-dim">
          <span className="h-2 w-3 rounded-sm border border-oak/40 bg-oak/15" />
          {STR.reward.range}
        </span>
        {ppoPts !== null && <Chip swatch="var(--color-ppo)" label={STR.reward.average(STR.policy.ppo)} />}
        {randomPts !== null && <Chip swatch="var(--color-random)" label={STR.reward.average(STR.policy.random)} />}
      </div>
      <div
        className="relative cursor-ew-resize touch-none select-none rounded-md transition-colors hover:bg-white/5"
        title={STR.bar.scrub}
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
          xMax={Math.max(1, horizon)}
          overlay={<Playhead total={Math.max(1, horizon)} />}
        >
          <Area {...bandProps('var(--color-oak)')} data={band} />
          <Line {...lineProps('var(--color-oak)', 1.6)} data={oakPts} />
          <Line {...lineProps('var(--color-seed)', 1.2)} data={heroPts} />
          {ppoPts !== null && <Line {...lineProps('var(--color-ppo)', 1.2)} data={ppoPts} />}
          {randomPts !== null && <Line {...lineProps('var(--color-random)', 1.2)} data={randomPts} />}
        </TimeChart>
        {rates === null && (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-2xs text-ink-faint">
            {detail === null ? STR.reward.waiting : STR.reward.reading}
          </div>
        )}
      </div>
      {stale.length > 0 && (
        <div className="px-2 text-2xs text-ink-faint">{STR.reward.stale(stale.join(' and '))}</div>
      )}
      <SeedList detail={detail} active={active} />
    </div>
  );
}
