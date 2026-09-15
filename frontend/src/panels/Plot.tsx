import { useEffect, useRef, type PointerEvent } from 'react';
import { formatStep } from '../format';
import { frameListeners, type Frame } from '../theater/stage';
import { PLOT_INSET, drawPlot, valueAt, type PlotSpec } from './plotter';

export function Plot({ label, height, spec, yLabel, playhead = false, seek }: {
  label: string;
  height: number;
  spec: PlotSpec;
  yLabel: string;
  playhead?: boolean;
  seek?: (step: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const stepRef = useRef<HTMLDivElement | null>(null);
  const valueRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const specRef = useRef(spec);
  specRef.current = spec;
  const marks = useRef({ playhead: null as number | null, cursor: null as number | null });
  const raf = useRef(0);

  const draw = (): void => {
    raf.current = 0;
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const current = specRef.current;
    drawPlot(canvas, current, marks.current);
    const card = cardRef.current;
    const cursor = marks.current.cursor;
    if (card === null) return;
    if (cursor === null) {
      card.hidden = true;
      return;
    }
    if (stepRef.current !== null) stepRef.current.textContent = formatStep(cursor);
    current.lines.forEach((line, i) => {
      const el = valueRefs.current[i];
      if (el === null || el === undefined) return;
      const v = valueAt(line, cursor);
      el.textContent = v === null ? '—' : current.yFormat(v);
    });
    const width = canvas.clientWidth;
    const x = PLOT_INSET.left + (cursor / Math.max(1, current.xMax)) * (width - PLOT_INSET.left - PLOT_INSET.right);
    card.hidden = false;
    const cardWidth = card.offsetWidth;
    card.style.left = `${Math.max(0, Math.min(width - cardWidth, x + 12 + cardWidth <= width ? x + 12 : x - 12 - cardWidth)).toFixed(1)}px`;
  };
  const schedule = (): void => {
    if (raf.current === 0) raf.current = requestAnimationFrame(draw);
  };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const observer = new ResizeObserver(schedule);
    observer.observe(canvas);
    const listen = (frame: Frame): void => {
      marks.current.playhead = playhead ? Math.max(0, frame.step) : null;
      if (playhead) schedule();
    };
    frameListeners.add(listen);
    schedule();
    return () => {
      observer.disconnect();
      frameListeners.delete(listen);
      cancelAnimationFrame(raf.current);
      raf.current = 0;
    };
  }, [playhead]);

  useEffect(schedule, [spec]);

  const stepAt = (event: PointerEvent<HTMLElement>): number | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const f = (event.clientX - rect.left - PLOT_INSET.left) / Math.max(1, rect.width - PLOT_INSET.left - PLOT_INSET.right);
    if (f < 0 || f > 1) return null;
    return Math.round(f * specRef.current.xMax);
  };

  return (
    <div
      role="img"
      aria-label={label}
      className={`relative shrink-0 ${seek === undefined ? '' : 'cursor-ew-resize touch-none select-none rounded-md hover:bg-white/5'}`}
      onPointerDown={(event) => {
        if (seek === undefined || event.button !== 0 || !event.isPrimary) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        const step = stepAt(event);
        if (step !== null) seek(step);
      }}
      onPointerMove={(event) => {
        const step = stepAt(event);
        marks.current.cursor = step;
        if (seek !== undefined && step !== null && event.currentTarget.hasPointerCapture(event.pointerId)) seek(step);
        schedule();
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerLeave={() => {
        marks.current.cursor = null;
        schedule();
      }}
    >
      <span className="pointer-events-none absolute left-0.5 top-0 font-mono text-2xs text-ink-dim">{yLabel}</span>
      <canvas ref={canvasRef} className="block w-full" style={{ height }} />
      <div ref={cardRef} hidden className="glass pointer-events-none absolute top-6 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs text-ink">
        <div ref={stepRef} className="font-mono tabular-nums text-ink-dim" />
        {spec.lines.map((line, i) => (
          <div key={line.label} className="mt-0.5 flex items-center gap-1.5">
            <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: line.tone }} />
            <span ref={(el) => { valueRefs.current[i] = el; }} className="font-mono tabular-nums" />
            <span className="text-ink-dim">{line.label}</span>
          </div>
        ))}
      </div>
      <div className="pr-1 text-right font-mono text-2xs text-ink-dim">steps</div>
    </div>
  );
}
