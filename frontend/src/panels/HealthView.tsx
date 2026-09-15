import { memo, useDeferredValue } from 'react';
import { Line } from 'recharts';
import type { EnvId } from '../data/envs';
import type { RunDetail, Tape } from '../data/oak';
import { cached, envLanes, rhoRates, survival, windowedRates } from '../derive';
import { formatPercent } from '../format';
import { store } from '../store';
import { Key, TimeChart, lineProps, seriesRows } from './charts';

function Head({ label, hint, keys }: { label: string; hint: string; keys: { tone: string; label: string; dashed?: boolean }[] }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-2xs uppercase tracking-label" title={hint}>{label}</span>
      {keys.map((key) => <Key key={key.label} {...key} />)}
    </div>
  );
}

const seriesFor = (tape: Tape, detail: RunDetail, envId: EnvId) =>
  cached(tape, `health/${envId}`, () => {
    const lanes = envLanes(detail.meta, envId).lanes;
    const rates = windowedRates(tape, lanes, 1);
    const rho = rhoRates(tape, lanes);
    const alive = survival(tape, lanes);
    const n = Math.min(rho.x.length, rates.iqm.length);
    let peak = -Infinity;
    return {
      surviving: seriesRows(alive.x, alive.surviving),
      positive: seriesRows(alive.x, alive.positive),
      fade: seriesRows(rates.x, rates.iqm.map((v) => {
        if (v === null) return null;
        if (v > peak) peak = v;
        return peak - v;
      })),
      measured: seriesRows(rates.x.slice(0, n), rates.iqm.slice(0, n)),
      believed: seriesRows(rho.x.slice(0, n), rho.iqm.slice(0, n)),
    };
  });

export function HealthView(props: { runId: string; envId: EnvId }) {
  return <HealthPanel runId={useDeferredValue(props.runId)} envId={useDeferredValue(props.envId)} />;
}

const HealthPanel = memo(function HealthPanel({ runId, envId }: { runId: string; envId: EnvId }) {
  const details = store.use((s) => s.details);
  const tapes = store.use((s) => s.tapes);
  const active = details[runId]!;
  const { surviving, positive, fade, measured, believed } = seriesFor(tapes[runId]!, active, envId);
  const horizon = Math.max(1, active.meta.steps);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto px-2 pb-1">
      <Head label="Survival" hint="share of seeds still alive" keys={[{ tone: 'var(--color-good)', label: 'surviving' }, { tone: 'var(--color-ramp-2)', label: 'cumulative reward > 0' }]} />
      <TimeChart label="Survival" height={100} xMax={horizon} yFormat={(v) => formatPercent(v, 0)} yLabel="% of seeds">
        <Line {...lineProps('var(--color-good)', 1.5)} name="surviving" data={surviving} />
        <Line {...lineProps('var(--color-ramp-2)', 1)} name="cumulative reward > 0" data={positive} />
      </TimeChart>
      <Head label="Retention" hint="Φ = IQM reward rate below its running peak" keys={[{ tone: 'var(--color-crit)', label: 'Φ — reward rate lost from peak' }]} />
      <TimeChart label="Retention" height={100} xMax={horizon} yLabel="Φ">
        <Line {...lineProps('var(--color-crit)', 1.4)} name="Φ — reward rate lost from peak" data={fade} />
      </TimeChart>
      <Head label="ρ̄ vs measured" hint="ρ̄ = the agent's own running average-reward estimate" keys={[{ tone: 'var(--color-oak)', label: 'measured' }, { tone: 'var(--color-ramp-4)', label: 'ρ̄', dashed: true }]} />
      <TimeChart label="ρ̄ vs measured" height={100} xMax={horizon} yLabel="reward / step">
        <Line {...lineProps('var(--color-oak)', 1.5)} name="measured" data={measured} />
        <Line {...lineProps('var(--color-ramp-4)', 1.1, '4 3')} name="ρ̄" data={believed} />
      </TimeChart>
    </div>
  );
});
