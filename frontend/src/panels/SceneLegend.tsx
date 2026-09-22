import { useEffect, useRef, useState, type RefObject } from 'react';
import type { LegendSpec } from '@kit/types';
import { frameListeners, type Stage } from '../theater/stage';

const css = (tint: number): string => `#${tint.toString(16).padStart(6, '0')}`;
const VALUE = 'font-semibold tabular-nums';
const tone = (value: string): string => value.startsWith('+') ? 'text-good' : value.startsWith('−') ? 'text-bad' : 'text-ink';

export function SceneLegend({ stage }: { stage: RefObject<Stage | null> }) {
  const [spec, setSpec] = useState<LegendSpec | null>(null);
  const values = useRef(new Map<string, HTMLSpanElement>());
  const note = useRef<HTMLSpanElement | null>(null);
  const bar = useRef<HTMLDivElement | null>(null);
  const lastAt = useRef(0);

  useEffect(() => {
    const listen = (): void => {
      const next = stage.current?.legend() ?? null;
      setSpec(held => (held === next ? held : next));
      const now = performance.now();
      if (next === null || now - lastAt.current < 150) return;
      lastAt.current = now;
      for (const row of next.rows) {
        const element = values.current.get(row.key);
        if (element === undefined) continue;
        const value = row.value();
        if (element.textContent !== value) { element.textContent = value; element.className = `${VALUE} ${tone(value)}`; }
      }
      if (note.current !== null && next.note) note.current.textContent = next.note();
      if (bar.current !== null && next.bar) bar.current.style.transform = `scaleX(${Math.min(1, Math.max(0, next.bar())).toFixed(4)})`;
    };
    frameListeners.add(listen);
    return () => { frameListeners.delete(listen); };
  }, [stage]);

  if (spec === null) return null;
  return (
    <div className="glass flex flex-col gap-1.5 rounded-lg px-3 py-2 text-xs">
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-semibold tracking-[0.14em] text-ink-dim">{spec.title}</span>
        {spec.note !== undefined && <span ref={note} className="tabular-nums text-ink-faint" />}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {spec.rows.map(row => (
          <span key={row.key} className="flex items-center gap-1.5">
            <span className="size-2 rounded-full" style={{ backgroundColor: css(row.tint), boxShadow: `0 0 5px ${css(row.tint)}` }} />
            <span className="text-ink-dim">{row.label}</span>
            <span ref={element => { if (element !== null) values.current.set(row.key, element); else values.current.delete(row.key); }} className={`${VALUE} text-ink`} />
          </span>
        ))}
      </div>
      {spec.bar !== undefined && (
        <div className="h-[3px] overflow-hidden rounded-full bg-white/10">
          <div ref={bar} className="h-full origin-left rounded-full bg-accent/70" style={{ transform: 'scaleX(0)', willChange: 'transform' }} />
        </div>
      )}
    </div>
  );
}
