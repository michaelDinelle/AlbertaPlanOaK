import { useMemo } from 'react';
import uPlot from 'uplot';
import type { RunDetail } from '@shared/types';
import { Card, CardHead, ChartNote, LegendChip } from '../../components/ui/card';
import { cssVar, monoFont, tooltipPlugin, withAlpha } from '../../charts/plugins';
import { profileOptions, type SeriesLine } from '../../charts/specs';
import { useUPlot } from '../../charts/useUPlot';
import { formatPercent } from '../../lib/format';

const POLICY_TOKENS = [
  { name: 'OaK', token: '--color-oak' },
  { name: 'PPO', token: '--color-ppo' },
  { name: 'random', token: '--color-random' },
] as const;

function tauGuides(): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = withAlpha(cssVar('--color-ink'), 0.22);
        ctx.fillStyle = withAlpha(cssVar('--color-ink'), 0.45);
        ctx.font = monoFont(9 * devicePixelRatio);
        ctx.textAlign = 'center';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        for (const [tau, label] of [[0, 'random'], [1, 'ceiling']] as const) {
          const x = Math.round(u.valToPos(tau, 'x', true)) + 0.5;
          if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) continue;
          ctx.beginPath();
          ctx.moveTo(x, u.bbox.top);
          ctx.lineTo(x, u.bbox.top + u.bbox.height);
          ctx.stroke();
          ctx.fillText(label, x, u.bbox.top + 10 * devicePixelRatio);
        }
        ctx.restore();
      },
    },
  };
}

export function Profiles({
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

  const { lines, bands, data } = useMemo(() => {
    const base = summaries.find((s) => s !== null) ?? null;
    if (base === null) {
      return { lines: [] as SeriesLine[], bands: [] as uPlot.Band[], data: [[]] as unknown as uPlot.AlignedData };
    }
    const x = base.aggregate.profile.tau;
    const lines: SeriesLine[] = [];
    const bands: uPlot.Band[] = [];
    const columns: (number | null)[][] = [];
    summaries.forEach((summary, i) => {
      const meta = POLICY_TOKENS[i];
      const profile = summary?.aggregate.profile;
      if (meta === undefined || profile === undefined || profile.tau.length !== x.length) return;
      const color = cssVar(meta.token);
      const ghost = withAlpha(color, 0);
      const at = lines.length + 1;
      lines.push({ label: `${meta.name} hi`, stroke: ghost, width: 0 });
      lines.push({ label: `${meta.name} lo`, stroke: ghost, width: 0 });
      lines.push({ label: meta.name, stroke: color, width: 1.6 });
      bands.push({ series: [at, at + 1], fill: withAlpha(color, 0.1) });
      columns.push(profile.ci95[1], profile.ci95[0], profile.fraction);
    });
    return { lines, bands, data: [x, ...columns] as unknown as uPlot.AlignedData };
  }, [summaries]);

  const build = useMemo(
    () =>
      profileOptions({
        lines,
        bands,
        xFormat: (v) => v.toFixed(2),
        plugins: [tauGuides(), tooltipPlugin({ yFormat: (v) => formatPercent(v, 0), xFormat: (v) => `τ ${v.toFixed(2)}` })],
      }),
    [lines, bands],
  );
  const chart = useUPlot({ build, data, deps: [lines.map((l) => l.label).join(',')] });

  return (
    <Card className={className}>
      <CardHead title={<span title="Fraction of all runs (16 envs × 64 seeds) whose final z is at least τ — higher curves dominate; bands are 95% CIs">Performance profiles</span>}>
        {POLICY_TOKENS.map((p) => (
          <LegendChip key={p.name} swatch={`var(${p.token})`} label={p.name} />
        ))}
      </CardHead>
      {lines.length === 0 ? (
        <ChartNote>statistics need summary v2 on this run</ChartNote>
      ) : (
        <div className="relative h-48">
          <div ref={chart.ref} className="absolute inset-0" />
        </div>
      )}
    </Card>
  );
}
