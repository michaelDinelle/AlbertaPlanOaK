import { useEffect, useMemo, useRef } from 'react';
import { clsx } from 'clsx';
import { ArrowUpDown } from 'lucide-react';
import type { RunDetail } from '@shared/types';
import { formatCount, formatSigned } from '../lib/format';
import { heroSeed, sortPermutation } from '../lib/selection';
import { cycleSortMode, promote, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import type { SeedRow } from '../tape/derive';
import { scrubToStep, stageMetaSnapshot } from '../theater/controller';
import { seedsFor } from '../theater/loader';
import { navigate } from '../url/useUrlState';
import { cssVar, withAlpha } from './charts';

function drawSpark(
  canvas: HTMLCanvasElement,
  width: number,
  values: readonly (number | null)[],
  xs: readonly number[],
  stroke: string,
): void {
  const height = 16;
  if (width <= 1) return;
  const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  if (ctx === null) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) return;
  const pad = 3;
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) {
    min -= Math.abs(min) * 0.1 + 1e-6;
    max += Math.abs(max) * 0.1 + 1e-6;
  }
  const span = max - min;
  const xAt = (i: number): number => Math.min(Math.max(xs[i] ?? 0, 0), 1) * width;
  const yAt = (v: number): number => height - pad - ((v - min) / span) * (height - pad * 2);

  if (min < 0 && max > 0) {
    ctx.save();
    ctx.strokeStyle = withAlpha(cssVar('--color-ink'), 0.22);
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    const y = Math.round(yAt(0)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.restore();
  }

  const points: [number, number][] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    points.push([xAt(i), yAt(v)]);
  }
  if (points.length < 2) return;

  const baseY = min < 0 && max > 0 ? yAt(0) : height - pad;
  ctx.beginPath();
  ctx.moveTo(points[0]![0], baseY);
  for (const [x, y] of points) ctx.lineTo(x, y);
  ctx.lineTo(points[points.length - 1]![0], baseY);
  ctx.closePath();
  ctx.fillStyle = withAlpha(stroke, 0.13);
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(points[0]![0], points[0]![1]);
  for (const [x, y] of points.slice(1)) ctx.lineTo(x, y);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.25;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();
}

function Spark({
  values,
  xs,
  stroke,
  draggedRef,
}: {
  values: readonly (number | null)[];
  xs: readonly number[];
  stroke: string;
  draggedRef: { current: boolean };
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const holderRef = useRef<HTMLSpanElement | null>(null);
  const down = useRef(false);
  const startX = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const holder = holderRef.current;
    if (canvas === null || holder === null) return;
    let frame = 0;
    const observer = new ResizeObserver((entries) => {
      const width = Math.floor(entries[0]?.contentRect.width ?? 0);
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => drawSpark(canvas, width, values, xs, stroke));
    });
    observer.observe(holder);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [values, xs, stroke]);

  const fracStep = (event: React.PointerEvent): number => {
    const holder = holderRef.current;
    if (holder === null) return 0;
    const rect = holder.getBoundingClientRect();
    const frac = Math.min(Math.max((event.clientX - rect.left) / Math.max(1, rect.width), 0), 1);
    return Math.round(frac * stageMetaSnapshot().totalSteps);
  };

  return (
    <span
      ref={holderRef}
      className="block min-w-0 flex-1 cursor-ew-resize"
      title={STR.bar.scrub}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        down.current = true;
        startX.current = event.clientX;
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {}
      }}
      onPointerMove={(event) => {
        if (!down.current) return;
        if (Math.abs(event.clientX - startX.current) > 3) draggedRef.current = true;
        if (draggedRef.current) scrubToStep(fracStep(event));
      }}
      onPointerUp={(event) => {
        if (!down.current) return;
        down.current = false;
        if (draggedRef.current) {
          const step = fracStep(event);
          scrubToStep(step);
          navigate({ at: step });
        }
      }}
      onPointerCancel={() => {
        down.current = false;
      }}
    >
      <canvas ref={canvasRef} className="block" />
    </span>
  );
}

function Row({ row, rank, isHero, playable }: { row: SeedRow; rank: number; isHero: boolean; playable: boolean }) {
  const draggedRef = useRef(false);
  const dead = row.divergedAt !== null;
  const stroke = dead ? cssVar('--color-crit') : isHero ? cssVar('--color-seed') : cssVar('--color-oak');
  return (
    <button
      type="button"
      title={playable ? STR.seeds.promote(row.seed) : STR.seeds.unplayable(row.seed)}
      onClick={() => {
        if (draggedRef.current) {
          draggedRef.current = false;
          return;
        }
        promote(row.seed, () => playable);
      }}
      className={clsx(
        'flex h-7 w-full items-center gap-2 rounded-md px-2 text-left transition-colors',
        isHero ? 'bg-seed/10 ring-1 ring-inset ring-seed/40' : 'hover:bg-white/5',
        !playable && 'opacity-45',
      )}
    >
      <span className="w-6 text-right font-mono text-2xs text-ink-faint">{rank}</span>
      <span className={clsx('w-10 font-mono text-xs', isHero && 'text-seed')}>{row.seed}</span>
      <span
        className={clsx(
          'w-14 text-right font-mono text-2xs tabular-nums',
          row.rate === null ? 'text-ink-faint' : row.rate > 0 ? 'text-good' : 'text-bad',
        )}
      >
        {row.rate === null ? '—' : formatSigned(row.rate, 3)}
      </span>
      <Spark
        values={[0, ...row.spark, row.spark[row.spark.length - 1] ?? null]}
        xs={[0, ...row.sparkXs, 1]}
        stroke={stroke}
        draggedRef={draggedRef}
      />
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
}

export function SeedList({ detail, active }: { detail: RunDetail | null; active: RunDetail | null }) {
  const sel = useSelection();
  const spec = useMemo<(DeriveSpec & { mode: 'rows' }) | null>(() => {
    if (detail === null || sel.envId === null) return null;
    return {
      mode: 'rows',
      envId: sel.envId,
      envSchedule: detail.meta.envSchedule,
      seeds: detail.meta.seeds,
      steps: detail.meta.steps,
      recorded: seedsFor(detail.recordings, sel.envId),
    };
  }, [detail, sel.envId]);
  const rows = useTapeSnapshot(sel.runId, sel.envId, 'rows', spec);
  const playableSeeds = useMemo(
    () => new Set(sel.envId === null ? [] : seedsFor(active?.recordings ?? [], sel.envId)),
    [active, sel.envId],
  );
  const order = rows === null ? [] : sortPermutation(rows, sel.sort);
  const hero = heroSeed(sel.selection);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 px-2 pb-1.5">
        <span className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.seeds.title}</span>
        <span className="font-mono text-2xs text-ink-faint">
          {rows !== null && rows.length > 0 ? formatCount(rows.length) : ''}
        </span>
        <button
          type="button"
          onClick={cycleSortMode}
          title={STR.seeds.sort}
          className="ml-auto flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-0.5 font-mono text-2xs text-ink-dim hover:border-white/25 hover:text-ink"
        >
          <ArrowUpDown className="size-2.5" />
          {STR.seeds.sortBy[sel.sort]}
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {rows === null && <div className="px-2 py-3 font-mono text-2xs text-ink-faint">{STR.reward.reading}</div>}
        {rows !== null && rows.length === 0 && (
          <div className="px-2 py-3 text-2xs text-ink-faint">{STR.seeds.empty}</div>
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
                playable={playableSeeds.has(row.seed)}
              />
            );
          })}
      </div>
    </div>
  );
}
