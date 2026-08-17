import { useEffect, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import type uPlot from 'uplot';
import type { RunDetail } from '@shared/types';
import { Card, CardHead, LegendChip } from '../../components/ui/card';
import { playheadPlugin, seekPlugin, tooltipPlugin, zeroLinePlugin, type BaselineBars } from '../../charts/plugins';
import { populationOptions } from '../../charts/specs';
import { ENV_SYNC_KEY } from '../../charts/sync';
import { useUPlot } from '../../charts/useUPlot';
import { envLanes } from '../../lib/run';
import { heroSeed } from '../../lib/selection';
import { useSelection } from '../../state/selectionStore';
import { useTapeSnapshot, type DeriveSpec } from '../../tape/store';
import { scrubToStep } from '../../theater/controller';
import { navigate } from '../../url/useUrlState';

type RewardMode = 'measured' | 'rho';

const MODE_LABEL: Record<RewardMode, string> = { measured: 'measured', rho: 'agent ρ̄' };

export function RewardChart({
  detail,
  ppo,
  random,
  height,
  className,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
  height: number;
  className?: string;
}) {
  const sel = useSelection();
  const [mode, setMode] = useState<RewardMode>('measured');

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
  const rates = useTapeSnapshot(sel.runId, sel.envId, 'rates', ratesSpec);
  const rho = useTapeSnapshot(sel.runId, sel.envId, 'rho', rhoSpec);
  const src = mode === 'measured' ? rates : rho;

  const hero = heroSeed(sel.selection);
  const heroIdx = hero === null || lanes === null ? -1 : lanes.seeds.indexOf(hero);

  const envId = sel.envId;
  const ppoVal = mode === 'measured' && envId !== null
    ? ppo?.summary?.envs[envId]?.finalRate.iqm ?? null
    : null;
  const randomVal = mode === 'measured' && envId !== null
    ? random?.summary?.envs[envId]?.finalRate.iqm ?? null
    : null;

  const barsRef = useRef<BaselineBars>({ ppo: null, random: null, active: 'oak' });
  barsRef.current = { ppo: ppoVal, random: randomVal, active: sel.policy };

  const build = useMemo(
    () =>
      populationOptions({
        getBars: () => barsRef.current,
        plugins: [
          tooltipPlugin({ xUnit: 'steps' }),
          zeroLinePlugin(),
          seekPlugin((x, phase) => {
            scrubToStep(x);
            if (phase !== 'drag') navigate({ at: Math.max(0, Math.floor(x)) });
          }),
          playheadPlugin(),
        ],
      }),
    [],
  );

  const data = useMemo<uPlot.AlignedData>(() => {
    if (src === null) return [[], [], [], [], []] as unknown as uPlot.AlignedData;
    const heroLine = heroIdx >= 0 ? src.perLane[heroIdx] ?? [] : src.x.map(() => null);
    return [src.x, src.p90, src.p10, src.iqm, heroLine] as unknown as uPlot.AlignedData;
  }, [src, heroIdx]);

  const chart = useUPlot({
    build,
    data,
    deps: [sel.runId, sel.envId, mode, ppoVal, randomVal],
    syncKey: ENV_SYNC_KEY,
  });

  useEffect(() => {
    chart.plot()?.redraw(false, false);
  }, [sel.policy]);

  return (
    <Card className={className}>
      <CardHead title={`Reward rate — ${MODE_LABEL[mode]}`}>
        <LegendChip swatch="var(--color-oak)" label="IQM" />
        <LegendChip swatch="var(--color-seed)" label="current seed" />
        <span className="flex items-center gap-1 font-mono text-2xs text-ink-dim">
          <span className="h-2 w-3 rounded-xs border border-oak/40 bg-oak/15" />
          P10–P90
        </span>
        {mode === 'measured' && (
          <>
            <LegendChip swatch="var(--color-ppo)" label="PPO final IQM" dim={sel.policy === 'random'} />
            <LegendChip swatch="var(--color-random)" label="random final IQM" dim={sel.policy === 'ppo'} />
          </>
        )}
        <div className="flex overflow-hidden rounded-sm border border-line">
          {(['measured', 'rho'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              title={m === 'measured'
                ? 'Reward rate measured over each tape window'
                : "The agent's own tracked reward-rate estimate ρ̄"}
              className={clsx(
                'px-2 py-0.5 font-mono text-2xs transition-colors',
                mode === m ? 'bg-accent/15 text-accent' : 'text-ink-dim hover:bg-hover hover:text-ink',
              )}
            >
              {MODE_LABEL[m]}
            </button>
          ))}
        </div>
      </CardHead>
      <div className="relative" style={{ height }}>
        <div ref={chart.ref} className="absolute inset-0" />
        {src === null && (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-2xs text-ink-faint">
            {detail === null ? 'waiting for the run' : 'reading the tape…'}
          </div>
        )}
      </div>
    </Card>
  );
}
