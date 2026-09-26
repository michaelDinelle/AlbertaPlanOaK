import type { Recording } from '../data/layout';
import { cached, envLanes, survival, windowedRates } from '../derive';
import { formatPercent, formatRate } from '../format';
import { store } from '../store';
import { Key } from './charts';
import { Plot } from './Plot';

function Head({ label, hint, keys }: { label: string; hint: string; keys: { tone: string; label: string; dashed?: boolean }[] }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-2xs uppercase tracking-label" title={hint}>{label}</span>
      {keys.map((key) => <Key key={key.label} {...key} />)}
    </div>
  );
}

const seriesFor = (rec: Recording, envId: string) =>
  cached(rec, `health/${envId}`, () => {
    const lanes = envLanes(rec.meta, envId).lanes;
    const rates = windowedRates(rec, lanes, 1);
    const alive = survival(rec, lanes);
    let peak = -Infinity;
    const fade = rates.iqm.map((v) => {
      if (v === null) return null;
      if (v > peak) peak = v;
      return peak - v;
    });
    const horizon = Math.max(1, rec.meta.steps);
    const percent = (v: number): string => formatPercent(v, 0);
    return {
      survival: { xMax: horizon, yFormat: percent, lines: [
        { x: alive.x, y: alive.surviving, tone: 'var(--color-good)', width: 1.5, label: 'surviving' },
        { x: alive.x, y: alive.positive, tone: 'var(--color-ramp-2)', width: 1, label: 'cumulative reward > 0' },
      ] },
      retention: { xMax: horizon, yFormat: formatRate, lines: [{ x: rates.x, y: fade, tone: 'var(--color-crit)', width: 1.4, label: 'Φ — reward rate lost from peak' }] },
    };
  });

export function HealthView({ runId, envId }: { runId: string; envId: string }) {
  const recordings = store.use((s) => s.recordings);
  const { survival: alive, retention } = seriesFor(recordings[runId]!, envId);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto px-2 pb-1">
      <Head label="Survival" hint="share of seeds still alive" keys={[{ tone: 'var(--color-good)', label: 'surviving' }, { tone: 'var(--color-ramp-2)', label: 'cumulative reward > 0' }]} />
      <Plot label="Survival" height={100} spec={alive} yLabel="% of seeds" />
      <Head label="Retention" hint="Φ = IQM reward rate below its running peak" keys={[{ tone: 'var(--color-crit)', label: 'Φ — reward rate lost from peak' }]} />
      <Plot label="Retention" height={100} spec={retention} yLabel="Φ" />
    </div>
  );
}
