import { useMemo } from 'react';
import { clsx } from 'clsx';
import { Line } from 'recharts';
import { isEnvId } from '@shared/envs';
import { nameOf } from '../scenes/table';
import type { Policy, RunDetail } from '@shared/types';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { Chip, Label, POLICY_ORDER, POLICY_TOKENS, TXT, TimeChart, lineProps, seriesRows } from './charts';

function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value < 10 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`;
}

export function ComputeView({ runs }: { runs: Record<Policy, RunDetail | null> }) {
  const selection = useSelection();
  const traces = useMemo(() => {
    const envId = selection.envId;
    if (envId === null) return null;
    let min = Infinity;
    let max = 1;
    const lines = POLICY_ORDER.flatMap((policy) => {
      const summary = runs[policy]?.summary;
      const iqm = summary?.envs[envId]?.curves.iqm;
      if (summary == null || iqm === undefined) return [];
      const elapsed = summary.curveSeconds;
      const first = iqm.findIndex((value) => value !== null);
      const firstSeconds = first < 0 ? undefined : elapsed[first];
      if (firstSeconds !== undefined && firstSeconds > 0 && firstSeconds < min) min = firstSeconds;
      const last = elapsed[elapsed.length - 1];
      if (last !== undefined && last > max) max = last;
      return [{
        policy,
        token: POLICY_TOKENS[policy],
        line: seriesRows(elapsed, iqm),
      }];
    });
    return Number.isFinite(min) && lines.length > 0 ? { envId, min, max, lines } : null;
  }, [runs, selection.envId]);

  if (runs.oak?.summary === null || runs.oak === null) {
    return <div className={clsx('px-2 py-6 text-center', TXT.bodyDim)}>{STR.compute.needsSummary}</div>;
  }
  if (traces === null) {
    return <div className={clsx('px-2 py-6 text-center', TXT.bodyDim)}>{STR.compute.chooseEnvironment}</div>;
  }

  const name = isEnvId(traces.envId) ? nameOf(traces.envId) : traces.envId;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2 pb-1">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <Label hint={STR.compute.titleHint}>{STR.compute.efficiencyEnv(name)}</Label>
        {traces.lines.map((trace) => (
          <Chip key={trace.policy} swatch={trace.token} label={STR.policy[trace.policy]} />
        ))}
      </div>
      <TimeChart
        label={STR.compute.efficiencyEnv(name)}
        height={240}
        xScale="log"
        xMin={traces.min}
        xMax={traces.max}
        xFormat={seconds}
        xLabel={STR.axis.seconds}
        yLabel={STR.axis.rewardRate}
      >
        {traces.lines.map((trace) => (
          <Line
            key={trace.policy}
            {...lineProps(trace.token, 1.4)}
            name={STR.policy[trace.policy]}
            data={trace.line}
            connectNulls
          />
        ))}
      </TimeChart>
    </div>
  );
}
