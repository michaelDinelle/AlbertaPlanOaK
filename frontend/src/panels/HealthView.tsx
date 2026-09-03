import { useMemo } from 'react';
import { clsx } from 'clsx';
import { Area, Line } from 'recharts';
import type { Policy, RunDetail } from '@shared/types';
import { formatCount, formatPercent, formatRate, formatStep } from '../lib/format';
import { envLanes } from '../lib/run';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { Chip, Label, TXT, TimeChart, bandProps, bandRows, lineProps, seriesRows } from './charts';

function Stat({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col" title={title}>
      <span className={TXT.data}>{value}</span>
      <span className={TXT.bodyDim}>{label}</span>
    </div>
  );
}

interface HeadChip {
  swatch: string;
  label: string;
  dashed?: boolean;
  block?: boolean;
}

function Head({ label, hint, chips }: { label: string; hint: string; chips?: HeadChip[] }) {
  return (
    <div className="flex items-center gap-2.5">
      <Label hint={hint}>{label}</Label>
      {chips?.map((chip) => (
        <Chip key={chip.label} {...chip} />
      ))}
    </div>
  );
}

export function HealthView({
  runs,
  active,
}: {
  runs: Record<Policy, RunDetail | null>;
  active: RunDetail | null;
}) {
  const sel = useSelection();
  const summary = active?.summary ?? null;
  const oakSummary = runs.oak?.summary ?? null;
  const lanes = useMemo(
    () => (active === null || sel.envId === null ? null : envLanes(active.meta, sel.envId)),
    [active, sel.envId],
  );
  const ratesSpec = useMemo<(DeriveSpec & { mode: 'rates' }) | null>(
    () => (lanes === null ? null : { mode: 'rates', lanes: lanes.lanes, win: 1 }),
    [lanes],
  );
  const rhoSpec = useMemo<(DeriveSpec & { mode: 'rho' }) | null>(
    () => (lanes === null ? null : { mode: 'rho', lanes: lanes.lanes }),
    [lanes],
  );
  const survivalSpec = useMemo<(DeriveSpec & { mode: 'survival' }) | null>(
    () => (lanes === null ? null : { mode: 'survival', lanes: lanes.lanes }),
    [lanes],
  );
  const rates = useTapeSnapshot(active?.id ?? null, sel.envId, 'rates', ratesSpec);
  const rho = useTapeSnapshot(active?.id ?? null, sel.envId, 'rho', rhoSpec);
  const survival = useTapeSnapshot(active?.id ?? null, sel.envId, 'survival', survivalSpec);

  const fade = useMemo(() => {
    if (rates === null) return [];
    let peak = -Infinity;
    const values = rates.iqm.map((v) => {
      if (v === null || !Number.isFinite(v)) return null;
      if (v > peak) peak = v;
      return peak - v;
    });
    return seriesRows(rates.x, values);
  }, [rates]);

  const capacity = useMemo(() => {
    const steps = oakSummary?.plasticitySteps ?? [];
    if (steps.length === 0 || sel.envId === null) return null;
    const pull = (
      summaryOf: typeof oakSummary,
      field: string,
    ): (number | null)[] | null => {
      const track = summaryOf?.envs[sel.envId ?? '']?.plasticity?.[field];
      return track !== undefined && track.iqm.length === steps.length ? track.iqm : null;
    };
    const oakDormant = pull(oakSummary, 'dormantFraction');
    const ppoDormant = pull(runs.ppo?.summary ?? null, 'dormantFraction');
    const oakErank = pull(oakSummary, 'erank');
    const ppoErank = pull(runs.ppo?.summary ?? null, 'erank');
    const tracks = [oakDormant, ppoDormant, oakErank, ppoErank];
    if (tracks.every((t) => t === null)) return null;
    const alive = tracks.some((t) => t !== null && t.some((v) => v !== null && Number.isFinite(v)));
    return {
      alive,
      oakDormant: oakDormant === null ? [] : seriesRows(steps, oakDormant),
      ppoDormant: ppoDormant === null ? [] : seriesRows(steps, ppoDormant),
      oakErank: oakErank === null ? [] : seriesRows(steps, oakErank),
      ppoErank: ppoErank === null ? [] : seriesRows(steps, ppoErank),
    };
  }, [oakSummary, runs.ppo, sel.envId]);

  const know = useMemo(() => {
    if (rates === null || rho === null) return null;
    const n = Math.min(rho.x.length, rates.iqm.length);
    return {
      measured: seriesRows(rates.x.slice(0, n), rates.iqm.slice(0, n)),
      believed: seriesRows(rho.x.slice(0, n), rho.iqm.slice(0, n)),
    };
  }, [rates, rho]);

  const kmBand = useMemo(() => {
    const km = sel.envId === null ? undefined : summary?.envs[sel.envId]?.divergence.km;
    if (km === undefined || km.steps.length === 0) return null;
    return bandRows(km.steps, km.ci95[0], km.ci95[1]);
  }, [summary, sel.envId]);

  const horizon = Math.max(1, active?.meta.steps ?? 1);
  const lanesTotal = summary === null ? 0 : summary.seedsPerEnv * Object.keys(summary.envs).length;
  const divergence = summary?.aggregate.divergence ?? null;
  const forgetting = sel.envId === null ? null : summary?.envs[sel.envId]?.forgetting ?? null;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto px-2 pb-1">
      <div className="flex gap-3">
        <Stat
          label={STR.health.diverged}
          value={divergence === null ? '—' : `${formatCount(divergence.diverged)} / ${formatCount(lanesTotal)}`}
          title={divergence === null ? undefined : STR.health.divergedTitle(formatPercent(divergence.rate, 2), formatPercent(divergence.rateCi95[0], 2), formatPercent(divergence.rateCi95[1], 2))}
        />
        <Stat
          label={STR.health.alive}
          value={divergence === null ? '—' : formatStep(divergence.expectedAliveSteps)}
          title={divergence === null ? undefined : STR.health.aliveTitle(formatStep(divergence.expectedAliveCi95[0]), formatStep(divergence.expectedAliveCi95[1]))}
        />
        <Stat
          label={STR.health.faded}
          value={forgetting === null ? '—' : formatRate(forgetting.iqm)}
          title={forgetting === null ? undefined : STR.health.fadedTitle(formatRate(forgetting.ci95[0]), formatRate(forgetting.ci95[1]))}
        />
      </div>
      <Head
        label={STR.health.survival}
        hint={STR.health.survivalHint}
        chips={[
          { swatch: 'var(--color-good)', label: STR.health.surviving },
          { swatch: 'var(--color-good)', label: STR.health.kmBand, block: true },
          { swatch: 'var(--color-ramp-2)', label: STR.health.positive },
        ]}
      />
      <TimeChart label={STR.health.survival} height={100} xMax={horizon} yFormat={(v) => formatPercent(v, 0)} xLabel={STR.axis.steps} yLabel={STR.axis.seedShare}>
        {kmBand !== null && <Area {...bandProps('var(--color-good)')} type="stepAfter" data={kmBand} />}
        {survival !== null && (
          <Line
            {...lineProps('var(--color-good)', 1.5)}
            name={STR.health.surviving}
            data={seriesRows(survival.x, survival.surviving)}
          />
        )}
        {survival !== null && (
          <Line
            {...lineProps('var(--color-ramp-2)', 1)}
            name={STR.health.positive}
            data={seriesRows(survival.x, survival.positive)}
          />
        )}
      </TimeChart>
      <Head
        label={STR.health.retention}
        hint={STR.health.retentionHint}
        chips={[{ swatch: 'var(--color-crit)', label: STR.health.faded }]}
      />
      <TimeChart label={STR.health.retention} height={100} xMax={horizon} xLabel={STR.axis.steps} yLabel={STR.axis.phi}>
        <Line {...lineProps('var(--color-crit)', 1.4)} name={STR.health.faded} data={fade} />
      </TimeChart>
      <Head
        label={STR.health.capacity}
        hint={STR.health.capacityHint}
        chips={[
          { swatch: 'var(--color-oak)', label: STR.policy.oak },
          { swatch: 'var(--color-ppo)', label: STR.policy.ppo },
          { swatch: 'var(--color-ink-dim)', label: STR.health.erank, dashed: true },
        ]}
      />
      {capacity === null || !capacity.alive ? (
        <div className={clsx('py-3 text-center', TXT.bodyDim)}>{STR.health.noPlasticity}</div>
      ) : (
        <TimeChart label={STR.health.capacity} height={100} xMax={horizon} yFormat={(v) => formatPercent(v, 0)} rightFormat={(v) => formatRate(v, 2)} xLabel={STR.axis.steps} yLabel={STR.axis.dormant} rightLabel={STR.axis.erank}>
          <Line
            {...lineProps('var(--color-oak)', 1.5)}
            name={`${STR.policy.oak} ${STR.health.dormant}`}
            data={capacity.oakDormant}
          />
          <Line
            {...lineProps('var(--color-ppo)', 1.5)}
            name={`${STR.policy.ppo} ${STR.health.dormant}`}
            data={capacity.ppoDormant}
          />
          <Line
            {...lineProps('var(--color-oak)', 1, '4 3', 'r')}
            name={`${STR.policy.oak} ${STR.health.erank}`}
            data={capacity.oakErank}
          />
          <Line
            {...lineProps('var(--color-ppo)', 1, '4 3', 'r')}
            name={`${STR.policy.ppo} ${STR.health.erank}`}
            data={capacity.ppoErank}
          />
        </TimeChart>
      )}
      <Head
        label={STR.health.know}
        hint={STR.health.knowHint}
        chips={[
          { swatch: 'var(--color-oak)', label: STR.health.measured },
          { swatch: 'var(--color-ramp-4)', label: STR.health.believed, dashed: true },
        ]}
      />
      <TimeChart label={STR.health.know} height={100} xMax={horizon} xLabel={STR.axis.steps} yLabel={STR.axis.rewardRate}>
        {know !== null && <Line {...lineProps('var(--color-oak)', 1.5)} name={STR.health.measured} data={know.measured} />}
        {know !== null && <Line {...lineProps('var(--color-ramp-4)', 1.1, '4 3')} name={STR.health.believed} data={know.believed} />}
      </TimeChart>
    </div>
  );
}
