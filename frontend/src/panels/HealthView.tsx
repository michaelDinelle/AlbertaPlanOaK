import { useMemo } from 'react';
import { Line } from 'recharts';
import type { RunDetail } from '@shared/types';
import { formatCount, formatPercent, formatRate, formatStep } from '../lib/format';
import { envLanes } from '../lib/run';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { Chip, TimeChart, lineProps, seriesRows } from './charts';

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <span className="font-mono text-sm text-ink">{value}</span>
      <span className="text-2xs text-ink-dim">{label}</span>
    </div>
  );
}

function Head({ label, chips }: { label: string; chips?: { swatch: string; label: string; dashed?: boolean }[] }) {
  return (
    <div className="flex items-center gap-2.5 pt-1">
      <span className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{label}</span>
      {chips?.map((chip) => <Chip key={chip.label} {...chip} />)}
    </div>
  );
}

export function HealthView({ detail, ppo }: { detail: RunDetail | null; ppo: RunDetail | null }) {
  const sel = useSelection();
  const keeper = detail?.summary ?? null;
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
  const survivalSpec = useMemo<(DeriveSpec & { mode: 'survival' }) | null>(
    () => (lanes === null ? null : { mode: 'survival', lanes: lanes.lanes }),
    [lanes],
  );
  const rates = useTapeSnapshot(sel.runId, sel.envId, 'rates', ratesSpec);
  const rho = useTapeSnapshot(sel.runId, sel.envId, 'rho', rhoSpec);
  const survival = useTapeSnapshot(sel.runId, sel.envId, 'survival', survivalSpec);

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
    const steps = keeper?.plasticitySteps ?? [];
    if (steps.length === 0 || sel.envId === null) return null;
    const pull = (
      summary: typeof keeper,
      field: string,
    ): (number | null)[] | null => {
      const track = summary?.envs[sel.envId ?? '']?.plasticity?.[field];
      return track !== undefined && track.iqm.length === steps.length ? track.iqm : null;
    };
    const oakDormant = pull(keeper, 'dormantFraction');
    const ppoDormant = pull(ppo?.summary ?? null, 'dormantFraction');
    const oakErank = pull(keeper, 'erank');
    const ppoErank = pull(ppo?.summary ?? null, 'erank');
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
  }, [keeper, ppo, sel.envId]);

  const know = useMemo(() => {
    if (rates === null || rho === null) return null;
    const n = Math.min(rho.x.length, rates.iqm.length);
    return {
      measured: seriesRows(rates.x.slice(0, n), rates.iqm.slice(0, n)),
      believed: seriesRows(rho.x.slice(0, n), rho.iqm.slice(0, n)),
    };
  }, [rates, rho]);

  const horizon = Math.max(1, detail?.meta.steps ?? 1);
  const lanesTotal = keeper === null ? 0 : keeper.seedsPerEnv * Object.keys(keeper.envs).length;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2">
      <div className="flex gap-3">
        <Stat
          label={STR.health.diverged}
          value={keeper === null ? '—' : `${formatCount(keeper.aggregate.divergence.diverged)} / ${formatCount(lanesTotal)}`}
        />
        <Stat
          label={STR.health.alive}
          value={keeper === null ? '—' : formatStep(keeper.aggregate.divergence.expectedAliveSteps)}
        />
        <Stat label={STR.health.faded} value={keeper === null ? '—' : formatRate(keeper.aggregate.forgetting.phiIQM)} />
      </div>
      <Head
        label={STR.health.survival}
        chips={[
          { swatch: 'var(--color-good)', label: STR.health.survival },
          { swatch: 'var(--color-ramp-2)', label: STR.health.positive },
        ]}
      />
      <TimeChart height={110} xMax={horizon} yFormat={(v) => formatPercent(v, 0)}>
        {survival !== null && <Line {...lineProps('var(--color-good)', 1.5)} data={seriesRows(survival.x, survival.surviving)} />}
        {survival !== null && <Line {...lineProps('var(--color-ramp-2)', 1)} data={seriesRows(survival.x, survival.positive)} />}
      </TimeChart>
      <Head label={STR.health.fade} chips={[{ swatch: 'var(--color-crit)', label: STR.health.faded }]} />
      <TimeChart height={110} xMax={horizon}>
        <Line {...lineProps('var(--color-crit)', 1.4)} data={fade} />
      </TimeChart>
      <Head
        label={STR.health.capacity}
        chips={[
          { swatch: 'var(--color-oak)', label: STR.policy.oak },
          { swatch: 'var(--color-ppo)', label: STR.policy.ppo },
          { swatch: 'var(--color-ink-dim)', label: STR.health.erank, dashed: true },
        ]}
      />
      {capacity === null ? (
        <div className="py-3 text-center text-2xs text-ink-faint">{STR.health.noPlasticity}</div>
      ) : (
        <TimeChart height={110} xMax={horizon} yFormat={(v) => formatPercent(v, 0)} rightFormat={(v) => formatRate(v, 2)}>
          <Line {...lineProps('var(--color-oak)', 1.5)} data={capacity.oakDormant} />
          <Line {...lineProps('var(--color-ppo)', 1.5)} data={capacity.ppoDormant} />
          <Line {...lineProps('var(--color-oak)', 1, '4 3', 'r')} data={capacity.oakErank} />
          <Line {...lineProps('var(--color-ppo)', 1, '4 3', 'r')} data={capacity.ppoErank} />
        </TimeChart>
      )}
      <Head
        label={STR.health.know}
        chips={[
          { swatch: 'var(--color-oak)', label: STR.health.measured },
          { swatch: 'var(--color-ramp-4)', label: STR.health.believed, dashed: true },
        ]}
      />
      <TimeChart height={110} xMax={horizon}>
        {know !== null && <Line {...lineProps('var(--color-oak)', 1.5)} data={know.measured} />}
        {know !== null && <Line {...lineProps('var(--color-ramp-4)', 1.1, '4 3')} data={know.believed} />}
      </TimeChart>
    </div>
  );
}
