import { useMemo } from 'react';
import { clsx } from 'clsx';
import { Line } from 'recharts';
import { isEnvId } from '@shared/envs';
import { nameOf } from '../scenes/table';
import type { Policy, RunDetail } from '@shared/types';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { Chip, Label, POLICY_ORDER, POLICY_TOKENS, TXT, TimeChart, lineProps, seriesRows } from './charts';

function exp(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const e = Math.log10(value);
  return Number.isInteger(e) ? `1e${e}` : value.toExponential(1);
}

export function ComputeView({ runs }: { runs: Record<Policy, RunDetail | null> }) {
  const sel = useSelection();
  const summaries = useMemo(() => POLICY_ORDER.map((policy) => runs[policy]?.summary ?? null), [runs]);
  const keeper = summaries[0] ?? null;

  const frontier = useMemo(() => {
    let min = Infinity;
    let max = 1;
    const traces = POLICY_ORDER.flatMap((policy, i) => {
      const f = summaries[i]?.aggregate.frontier;
      if (f === undefined || f.cumFlopsPerLane.length === 0) return [];
      const firstIdx = f.zIQM.findIndex((v) => v !== null);
      if (firstIdx >= 0) {
        const firstX = f.cumFlopsPerLane[firstIdx];
        if (firstX !== undefined && firstX > 0 && firstX < min) min = firstX;
      }
      const lastX = f.cumFlopsPerLane[f.cumFlopsPerLane.length - 1];
      if (lastX !== undefined && lastX > max) max = lastX;
      return [{ key: policy, token: POLICY_TOKENS[policy], line: seriesRows(f.cumFlopsPerLane, f.zIQM) }];
    });
    return Number.isFinite(min) && traces.length > 0 ? { min, max, traces } : null;
  }, [summaries]);

  const envFrontier = useMemo(() => {
    const envId = sel.envId;
    if (envId === null) return null;
    let min = Infinity;
    let max = 1;
    const traces = POLICY_ORDER.flatMap((policy, i) => {
      const summary = summaries[i] ?? null;
      const z = summary?.envs[envId]?.curves.iqmZ;
      const perStep = summary?.compute.envFlopsPerStep[envId];
      if (summary === null || z === undefined || perStep === undefined) return [];
      const flops = summary.curveSteps.map((step) => perStep * step);
      const firstIdx = z.findIndex((v) => v !== null);
      const firstX = firstIdx < 0 ? undefined : flops[firstIdx];
      if (firstX !== undefined && firstX > 0 && firstX < min) min = firstX;
      const lastX = flops[flops.length - 1];
      if (lastX !== undefined && lastX > max) max = lastX;
      return [{ key: policy, token: POLICY_TOKENS[policy], line: seriesRows(flops, z) }];
    });
    return Number.isFinite(min) && traces.length > 0 ? { min, max, traces } : null;
  }, [summaries, sel.envId]);

  if (keeper === null) {
    return <div className={clsx('px-2 py-6 text-center', TXT.bodyDim)}>{STR.compute.needsSummary}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2 pb-1">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <Label hint={STR.compute.titleHint}>{STR.compute.efficiency}</Label>
        {frontier?.traces.map((trace) => (
          <Chip key={trace.key} swatch={trace.token} label={STR.policy[trace.key]} />
        ))}
      </div>
      {frontier !== null && (
        <TimeChart
          label={STR.compute.efficiency}
          height={160}
          xScale="log"
          xMin={frontier.min}
          xMax={frontier.max}
          xFormat={exp}
          yFormat={(v) => v.toFixed(2)}
          xLabel={STR.axis.flops}
          yLabel={STR.axis.z}
        >
          {frontier.traces.map((trace) => (
            <Line
              key={trace.key}
              {...lineProps(trace.token, 1.4)}
              name={STR.policy[trace.key]}
              data={trace.line}
              connectNulls
            />
          ))}
        </TimeChart>
      )}
      {envFrontier !== null && sel.envId !== null && (
        <>
          <div className="pt-1">
            <Label hint={STR.compute.titleHint}>
              {STR.compute.efficiencyEnv(isEnvId(sel.envId) ? nameOf(sel.envId) : sel.envId)}
            </Label>
          </div>
          <TimeChart
            label={STR.compute.efficiencyEnv(sel.envId)}
            height={160}
            xScale="log"
            xMin={envFrontier.min}
            xMax={envFrontier.max}
            xFormat={exp}
            yFormat={(v) => v.toFixed(2)}
            xLabel={STR.axis.flops}
            yLabel={STR.axis.z}
          >
            {envFrontier.traces.map((trace) => (
              <Line
                key={trace.key}
                {...lineProps(trace.token, 1.4)}
                name={STR.policy[trace.key]}
                data={trace.line}
                connectNulls
              />
            ))}
          </TimeChart>
        </>
      )}
    </div>
  );
}
