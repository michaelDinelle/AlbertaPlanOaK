import { memo, useEffect, useMemo, useRef } from 'react';
import { clsx } from 'clsx';
import { ArrowUpDown } from 'lucide-react';
import type { RunDetail } from '../data/types';
import { formatCount, formatSigned } from '../lib/format';
import { heroSeed, sortPermutation } from '../lib/selection';
import { cycleSortMode, promote, unpromote, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import type { SeedRow } from '../tape/derive';
import { Hotkey, Label, TXT, cssVar, withAlpha } from './charts';
import { useStepScrub } from './useStepScrub';

function drawSpark(
  canvas: HTMLCanvasElement,
  width: number,
  values: Float64Array,
  xs: Float64Array,
  stroke: string,
): void {
  const height = 16;
  if (width <= 1) return;
  const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
  const bw = Math.round(width * dpr);
  const bh = Math.round(height * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  const ctx = canvas.getContext('2d');
  if (ctx === null) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  let min = 0;
  let max = 0;
  let finiteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    finiteCount++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (finiteCount < 1) return;
  if (min === max) {
    min -= Math.abs(min) * 0.1 + 1e-6;
    max += Math.abs(max) * 0.1 + 1e-6;
  }
  const span = max - min;
  const pad = 3;
  const yAt = (v: number): number => height - pad - ((v - min) / span) * (height - pad * 2);
  const zeroLine = min < 0 && max > 0;

  const xMax = Math.max(1, xs[xs.length - 1] ?? 0);
  const line = new Path2D();
  const y0 = yAt(0);
  line.moveTo(0, y0);
  let lastY = y0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    const x = Math.min(Math.max((xs[i] ?? 0) / xMax, 0), 1) * width;
    lastY = yAt(v);
    line.lineTo(x, lastY);
  }
  line.lineTo(width, lastY);

  if (zeroLine) {
    ctx.save();
    ctx.strokeStyle = withAlpha(cssVar('--color-ink'), 0.22);
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    const y = Math.round(y0) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.restore();
  }

  const fill = new Path2D(line);
  const baseY = zeroLine ? y0 : height - pad;
  fill.lineTo(width, baseY);
  fill.lineTo(0, baseY);
  fill.closePath();
  ctx.fillStyle = withAlpha(stroke, 0.13);
  ctx.fill(fill);

  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.25;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke(line);
}

function Spark({
  values,
  xs,
  stroke,
  onStart,
}: {
  values: Float64Array;
  xs: Float64Array;
  stroke: string;
  onStart: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const holderRef = useRef<HTMLSpanElement | null>(null);
  const state = useRef({ width: 0, values, xs, stroke, schedule: () => {} });

  useEffect(() => {
    Object.assign(state.current, { values, xs, stroke });
    const canvas = canvasRef.current;
    const holder = holderRef.current;
    if (canvas === null || holder === null) return;
    let frame = 0;
    const draw = (): void => {
      frame = 0;
      const st = state.current;
      drawSpark(canvas, st.width, st.values, st.xs, st.stroke);
    };
    state.current.schedule = () => {
      if (frame === 0) frame = requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver((entries) => {
      state.current.width = Math.floor(entries[0]?.contentRect.width ?? 0);
      state.current.schedule();
    });
    observer.observe(holder);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    Object.assign(state.current, { values, xs, stroke });
    state.current.schedule();
  }, [values, xs, stroke]);

  const scrub = useStepScrub<HTMLSpanElement>(
    () => ({ min: 0, max: state.current.xs[state.current.xs.length - 1] ?? 0 }),
    undefined,
    onStart,
  );

  return (
    <span
      ref={holderRef}
      className="block min-w-0 flex-1 cursor-ew-resize touch-none select-none"
      title={STR.bar.scrub}
      {...scrub}
      onClick={(event) => event.stopPropagation()}
    >
      <canvas ref={canvasRef} className="block" />
    </span>
  );
}

const Row = memo(function Row({
  row,
  rank,
  isHero,
  isPromoted,
  tone,
}: {
  row: SeedRow;
  rank: number;
  isHero: boolean;
  isPromoted: boolean;
  tone: string;
}) {
  const dead = row.divergedAt !== null;
  const stroke = dead ? cssVar('--color-crit') : isHero ? cssVar('--color-seed') : tone;
  return (
    <button
      type="button"
      title={isPromoted ? STR.seeds.release(row.seed) : STR.seeds.promote(row.seed)}
      onClick={() => {
        if (isPromoted) unpromote();
        else promote(row.seed);
      }}
      className={clsx(
        'flex h-7 w-full items-center gap-2 rounded-md px-2 text-left transition-colors',
        isHero ? 'bg-seed/10 ring-1 ring-inset ring-seed/40' : 'hover:bg-white/5',
      )}
    >
      <span className={clsx('w-6 text-right', TXT.dataDim)}>{rank}</span>
      <span className={clsx('w-10 font-mono text-xs', isHero && 'text-seed')}>{row.seed}</span>
      <span
        className={clsx(
          'w-14 text-right font-mono text-xs tabular-nums',
          row.rate === null ? 'text-ink-faint' : row.rate > 0 ? 'text-good' : 'text-bad',
        )}
      >
        {row.rate === null ? '—' : formatSigned(row.rate, 3)}
      </span>
      <Spark values={row.spark} xs={row.sparkXs} stroke={stroke} onStart={() => promote(row.seed)} />
      {dead && (
        <span
          className="rounded-sm border border-crit/40 bg-crit/10 px-1 font-mono text-2xs leading-4 text-crit"
          title={STR.seeds.divergedAt(formatCount(row.divergedAt))}
        >
          {STR.seeds.diverged}
        </span>
      )}
    </button>
  );
});

export function SeedList({ active }: { active: RunDetail | null }) {
  const sel = useSelection();
  const spec = useMemo<(DeriveSpec & { mode: 'rows' }) | null>(() => {
    if (active === null || sel.envId === null) return null;
    return {
      mode: 'rows',
      envId: sel.envId,
      envSchedule: active.meta.envSchedule,
      seeds: active.meta.seeds,
      steps: active.meta.steps,
    };
  }, [active, sel.envId]);
  const rows = useTapeSnapshot(active?.id ?? null, sel.envId, 'rows', spec);
  const order = rows === null ? [] : sortPermutation(rows, sel.sort);
  const tone = cssVar(`--color-${sel.policy}`);
  const hero = heroSeed(sel.selection);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 px-2">
        <Label hint={STR.seeds.titleHint}>{STR.seeds.title}</Label>
        <span className={TXT.dataDim}>
          {rows !== null && rows.length > 0 ? formatCount(rows.length) : ''}
        </span>
        <span className="ml-auto flex">
          <Hotkey label={STR.seeds.sort} keyName="s" align="end">
            <button
              type="button"
              onClick={cycleSortMode}
              className={clsx('flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-0.5 hover:border-white/25 hover:text-ink', TXT.bodyDim)}
            >
              <ArrowUpDown className="size-2.5" />
              {STR.seeds.sortBy[sel.sort]}
            </button>
          </Hotkey>
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {rows === null && <div className={clsx('px-2 py-3', TXT.bodyDim)}>{STR.reward.reading}</div>}
        {rows !== null && rows.length === 0 && (
          <div className={clsx('px-2 py-3', TXT.bodyDim)}>{STR.seeds.empty}</div>
        )}
        {rows !== null &&
          order.map((index, rank) => {
            const row = rows[index];
            if (row === undefined) return null;
            return (
              <Row
                key={row.seed}
                row={row}
                rank={rank + 1}
                isHero={row.seed === hero}
                isPromoted={row.seed === sel.selection.promoted}
                tone={tone}
              />
            );
          })}
      </div>
    </div>
  );
}
