import { useMemo, type ReactNode } from 'react';
import type { RunDetail, RunSummary } from '@shared/types';
import { Card } from '../../components/ui/card';
import { formatCount, formatPercent, formatRate, formatStep } from '../../lib/format';

const MINI_TOKENS = ['--color-ppo', '--color-random'] as const;

function Tile({
  label,
  hint,
  value,
  sub,
  minis,
}: {
  label: string;
  hint: string;
  value: ReactNode;
  sub: ReactNode;
  minis: (string | null)[];
}) {
  return (
    <Card>
      <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint" title={hint}>{label}</div>
      <div className="mt-1.5 font-mono text-lg leading-6 text-ink">{value}</div>
      <div className="mt-0.5 text-2xs text-ink-dim">{sub}</div>
      <div className="mt-2 flex items-center gap-3 border-t border-line/60 pt-1.5 font-mono text-2xs text-ink-faint">
        {minis.map((text, i) => (
          <span key={MINI_TOKENS[i] ?? i} className="flex items-center gap-1">
            <span className="size-1.5 rounded-full" style={{ background: `var(${MINI_TOKENS[i] ?? '--color-ink-faint'})` }} />
            {text ?? '—'}
          </span>
        ))}
      </div>
    </Card>
  );
}

export function DiagnosticsStrip({
  detail,
  ppo,
  random,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
}) {
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );
  const keeper = summaries[0] ?? null;
  const baselines = summaries.slice(1) as (RunSummary | null)[];
  const lanesOf = (s: RunSummary): number => s.seedsPerEnv * Object.keys(s.envs).length;

  return (
    <>
      <Tile
        label="Divergence"
        hint="Lanes that died numerically before the horizon, out of every lane in the fleet"
        value={keeper === null ? '—' : `${formatCount(keeper.aggregate.divergence.diverged)} / ${formatCount(lanesOf(keeper))}`}
        sub={keeper === null ? 'needs summary v2' : `${formatPercent(keeper.aggregate.divergence.rate, 2)} of lanes`}
        minis={baselines.map((s) => (s === null ? null : formatCount(s.aggregate.divergence.diverged)))}
      />
      <Tile
        label="Expected alive-steps"
        hint="Mean steps a lane survives before numerical death, of the full horizon"
        value={keeper === null ? '—' : formatStep(keeper.aggregate.divergence.expectedAliveSteps)}
        sub={keeper === null ? 'needs summary v2' : `of ${formatStep(keeper.horizon)} horizon`}
        minis={baselines.map((s) => (s === null ? null : formatStep(s.aggregate.divergence.expectedAliveSteps)))}
      />
      <Tile
        label="Forgetting Φ"
        hint="Peak windowed rate minus final rate, z units, IQM across lanes — higher means more was lost"
        value={keeper === null ? '—' : formatRate(keeper.aggregate.forgetting.phiIQM)}
        sub={keeper === null
          ? 'needs summary v2'
          : `95% CI [${formatRate(keeper.aggregate.forgetting.ci95[0])}, ${formatRate(keeper.aggregate.forgetting.ci95[1])}]`}
        minis={baselines.map((s) => (s === null ? null : formatRate(s.aggregate.forgetting.phiIQM)))}
      />
      <Tile
        label="IPR-90"
        hint="Spread between the 5th and 95th percentile lanes' final z — run-to-run variability, not uncertainty"
        value={keeper === null ? '—' : formatRate(keeper.aggregate.iprZ90)}
        sub={keeper === null ? 'needs summary v2' : 'z spread, 5th–95th pct of lanes'}
        minis={baselines.map((s) => (s === null ? null : formatRate(s.aggregate.iprZ90)))}
      />
    </>
  );
}
