import { useMemo } from 'react';
import uPlot from 'uplot';
import type { RunDetail } from '@shared/types';
import { Card, CardHead, ChartNote, LegendChip } from '../../components/ui/card';
import { cssVar, monoFont, tooltipPlugin, withAlpha } from '../../charts/plugins';
import { frontierOptions, type SeriesLine } from '../../charts/specs';
import { useUPlot } from '../../charts/useUPlot';

const POLICY_TOKENS = [
  { name: 'OaK', token: '--color-oak' },
  { name: 'PPO', token: '--color-ppo' },
  { name: 'random', token: '--color-random' },
] as const;

function formatFlops(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toExponential(1);
}

interface PolicyPoint {
  name: string;
  color: string;
  flopsPerStep: number;
  z: number;
}

function alignTables(tables: { x: number[]; y: (number | null)[] }[]): uPlot.AlignedData {
  const union = [...new Set(tables.flatMap((t) => t.x))].sort((a, b) => a - b);
  const columns = tables.map((table) => {
    const at = new Map<number, number | null>();
    table.x.forEach((x, i) => at.set(x, table.y[i] ?? null));
    return union.map((x) => at.get(x) ?? null);
  });
  return [union, ...columns] as unknown as uPlot.AlignedData;
}

function pointsPlugin(getPoints: () => PolicyPoint[]): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const pr = devicePixelRatio;
        const ctx = u.ctx;
        ctx.save();
        ctx.font = monoFont(10 * pr);
        ctx.textBaseline = 'middle';
        for (const point of getPoints()) {
          const x = u.valToPos(point.flopsPerStep, 'x', true);
          const y = u.valToPos(point.z, 'y', true);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          ctx.fillStyle = point.color;
          ctx.beginPath();
          ctx.arc(x, y, 4 * pr, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = withAlpha(point.color, 0.35);
          ctx.lineWidth = 3 * pr;
          ctx.stroke();
          const flip = x > u.bbox.left + u.bbox.width - 64 * pr;
          ctx.textAlign = flip ? 'right' : 'left';
          ctx.fillText(point.name, x + (flip ? -8 : 8) * pr, y);
        }
        ctx.restore();
      },
    },
  };
}

function isoOmegaPlugin(): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const sx = u.scales['x'];
        const sy = u.scales['y'];
        if (!sx || !sy) return;
        const x0 = sx.min ?? 0;
        const x1 = sx.max ?? 0;
        const y1 = sy.max ?? 0;
        if (x0 <= 0 || x1 <= x0 || y1 <= 0) return;
        const yFloor = y1 * 0.02;
        const kLo = Math.ceil(Math.log10(yFloor / x1));
        const kHi = Math.floor(Math.log10(y1 / x0));
        const ctx = u.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
        ctx.clip();
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.font = monoFont(9 * devicePixelRatio);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        for (let k = kLo; k <= kHi; k++) {
          const omega = 10 ** k;
          ctx.strokeStyle = withAlpha(cssVar('--color-ink'), 0.14);
          ctx.fillStyle = withAlpha(cssVar('--color-ink'), 0.4);
          ctx.beginPath();
          let labelAt: [number, number] | null = null;
          for (let t = 0; t <= 64; t++) {
            const x = x0 * (x1 / x0) ** (t / 64);
            const y = omega * x;
            const px = u.valToPos(x, 'x', true);
            const py = u.valToPos(y, 'y', true);
            if (t === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
            if (y <= y1) labelAt = [px, py];
          }
          ctx.stroke();
          if (labelAt !== null && labelAt[1] > u.bbox.top + 12 * devicePixelRatio) {
            ctx.fillText(`Ω=1e${k}`, Math.min(labelAt[0], u.bbox.left + u.bbox.width) - 3, labelAt[1] - 2);
          }
        }
        ctx.restore();
      },
    },
  };
}

export function Frontier({
  detail,
  ppo,
  random,
  className,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
  className?: string;
}) {
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );
  const keeper = summaries[0] ?? null;

  const points = useMemo<PolicyPoint[]>(
    () =>
      summaries.flatMap((summary, i) => {
        const meta = POLICY_TOKENS[i];
        if (summary === null || meta === undefined) return [];
        return [{
          name: meta.name,
          color: cssVar(meta.token),
          flopsPerStep: summary.compute.flopsPerStep,
          z: summary.aggregate.finalZ.iqm,
        }];
      }),
    [summaries],
  );

  const scatter = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map((p) => p.flopsPerStep);
    const zs = points.map((p) => p.z);
    const xRange: [number, number] = [Math.min(...xs) / 2.5, Math.max(...xs) * 2.5];
    const yTop = Math.max(1, ...zs) * 1.1;
    const data = alignTables(points.map((p) => ({ x: [p.flopsPerStep], y: [p.z] })));
    const lines: SeriesLine[] = points.map((p) => ({ label: p.name, stroke: p.color, width: 0 }));
    return { data, lines, xRange, yTop };
  }, [points]);

  const scatterBuild = useMemo(
    () =>
      frontierOptions({
        lines: scatter?.lines ?? [],
        yFormat: (v) => v.toFixed(2),
        xRange: () => scatter?.xRange ?? [1, 10],
        yRange: () => [0, scatter?.yTop ?? 1],
        plugins: [isoOmegaPlugin(), pointsPlugin(() => points)],
      }),
    [scatter, points],
  );
  const scatterChart = useUPlot({
    build: scatterBuild,
    data: scatter?.data ?? ([[]] as unknown as uPlot.AlignedData),
    deps: [points.map((p) => p.name).join(','), scatter?.xRange[0], scatter?.xRange[1]],
  });

  const curves = useMemo(() => {
    const tables: { x: number[]; y: (number | null)[] }[] = [];
    const lines: SeriesLine[] = [];
    summaries.forEach((summary, i) => {
      const meta = POLICY_TOKENS[i];
      if (summary === null || meta === undefined) return;
      const frontier = summary.aggregate.frontier;
      if (frontier.cumFlopsPerLane.length === 0) return;
      tables.push({ x: frontier.cumFlopsPerLane, y: frontier.zIQM });
      lines.push({ label: meta.name, stroke: cssVar(meta.token), width: 1.4, span: true });
    });
    if (tables.length === 0) return null;
    return { data: alignTables(tables), lines };
  }, [summaries]);

  const curvesBuild = useMemo(
    () =>
      frontierOptions({
        lines: curves?.lines ?? [],
        yFormat: (v) => v.toFixed(2),
        plugins: [tooltipPlugin({ yFormat: (v) => v.toFixed(3), xFormat: (v) => formatFlops(v), xUnit: 'FLOPs/lane' })],
      }),
    [curves],
  );
  const curvesChart = useUPlot({
    build: curvesBuild,
    data: curves?.data ?? ([[]] as unknown as uPlot.AlignedData),
    deps: [curves?.lines.map((l) => l.label).join(',')],
  });

  const ratios = keeper?.vsBaselines?.ppo?.computeRatioAt;
  const ratioEntries = ratios === undefined
    ? []
    : Object.entries(ratios).sort(([a], [b]) => Number(a) - Number(b));

  return (
    <Card className={className}>
      <CardHead title={<span title="Reward per compute: Ω = aggregate final-rate IQM (z) divided by per-step FLOPs">Compute</span>} />
      {keeper === null ? (
        <ChartNote>statistics need summary v2 on this run</ChartNote>
      ) : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="font-mono text-xl leading-6 text-ink">{keeper.compute.omega.toExponential(2)}</span>
            <span className="text-2xs text-ink-dim">Ω — normalized reward per FLOP-step</span>
            <span className="ml-auto flex items-center gap-2 font-mono text-2xs text-ink-faint">
              {summaries.slice(1).map((summary, i) => {
                const meta = POLICY_TOKENS[i + 1];
                if (meta === undefined) return null;
                return (
                  <span key={meta.name} className="flex items-center gap-1">
                    <span className="size-1.5 rounded-full" style={{ background: `var(${meta.token})` }} />
                    {summary === null ? '—' : summary.compute.omega.toExponential(2)}
                  </span>
                );
              })}
            </span>
          </div>
          {ratioEntries.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {ratioEntries.map(([threshold, ratio]) => {
                const n = ratio === null || ratio === 0 ? null : 1 / ratio;
                return (
                  <span
                    key={threshold}
                    title={n === null
                      ? `OaK did not reach z* = ${threshold} within the horizon`
                      : `To reach z* = ${threshold}, OaK spends ${n.toFixed(1)}× the FLOPs PPO spends`}
                    className="rounded-sm border border-line bg-sunken/60 px-1.5 py-0.5 font-mono text-2xs text-ink-dim"
                  >
                    z*={threshold} · {n === null ? '—' : `${n >= 10 ? n.toFixed(0) : n.toFixed(1)}× PPO`}
                  </span>
                );
              })}
              <span className="text-2xs text-ink-faint">compute to threshold, OaK ÷ PPO</span>
            </div>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-2xs text-ink-faint">
            <span className="uppercase tracking-[0.14em]">Fπ</span>
            {summaries.map((summary, i) => {
              const meta = POLICY_TOKENS[i];
              if (meta === undefined) return null;
              return (
                <span key={meta.name} className="flex items-center gap-1">
                  <span className="size-1.5 rounded-full" style={{ background: `var(${meta.token})` }} />
                  {formatFlops(summary?.compute.flopsPerStep)}
                </span>
              );
            })}
            <span>FLOPs/step</span>
          </div>
          <div className="relative mt-2 h-36">
            <div ref={scatterChart.ref} className="absolute inset-0" />
          </div>
          <div className="text-center text-2xs text-ink-faint">final z vs FLOPs/step · iso-Ω guides</div>
          <div className="relative mt-1 h-32">
            <div ref={curvesChart.ref} className="absolute inset-0" />
          </div>
          <div className="text-center text-2xs text-ink-faint">frontier — z-IQM vs cumulative FLOPs per lane</div>
        </>
      )}
    </Card>
  );
}
