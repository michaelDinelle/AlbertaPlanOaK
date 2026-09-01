import { useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import {
  jbwPolarity,
  JBW_COLOR,
  JBW_JELLY,
  JBW_ONION,
  JBW_BANANA,
  JBW_TRUFFLE,
  JBW_POLARITY_PERIOD,
} from '@sim/envs/jellybeanworld';
import { Panel } from '../glass/Panel';
import { formatCount } from '../lib/format';
import { STR } from '../strings';
import { subscribeStageFrame } from '../theater/controller';

function cssColor(item: number): string {
  const c = JBW_COLOR[item] ?? [1, 1, 1];
  const to = (v: number): number => Math.round(Math.min(1, Math.max(0, v)) ** (1 / 2.2) * 255);
  return `rgb(${to(c[0])} ${to(c[1])} ${to(c[2])})`;
}

function Chip({ item, label, value }: { item: number; label: string; value: string }) {
  const positive = value.startsWith('+');
  const negative = value.startsWith('−');
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="size-2 rounded-full"
        style={{ backgroundColor: cssColor(item), boxShadow: `0 0 5px ${cssColor(item)}` }}
      />
      <span className="text-ink-dim">{label}</span>
      <span
        className={clsx(
          'font-semibold tabular-nums',
          positive && 'text-good',
          negative && 'text-bad',
          !positive && !negative && 'text-ink-faint',
        )}
      >
        {value}
      </span>
    </span>
  );
}

export function JbwLegend() {
  const [polarity, setPolarity] = useState<1 | -1>(1);
  const flipRef = useRef<HTMLSpanElement | null>(null);
  const cycleRef = useRef<HTMLDivElement | null>(null);
  const lastAt = useRef(0);

  useEffect(
    () =>
      subscribeStageFrame((frame) => {
        const step = Math.max(0, frame.step);
        const sign = jbwPolarity(step);
        setPolarity((held) => (held === sign ? held : sign));
        const now = performance.now();
        if (now - lastAt.current < 150) return;
        lastAt.current = now;
        const envStep = step + 1;
        const into = envStep % JBW_POLARITY_PERIOD;
        if (flipRef.current !== null) {
          flipRef.current.textContent = STR.jbw.flip(formatCount(JBW_POLARITY_PERIOD - into));
        }
        if (cycleRef.current !== null) {
          cycleRef.current.style.transform = `scaleX(${(into / JBW_POLARITY_PERIOD).toFixed(4)})`;
        }
      }),
    [],
  );

  return (
    <Panel className="pointer-events-auto flex flex-col gap-1.5 px-3 py-2 text-xs">
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-semibold tracking-[0.14em] text-ink-dim">{STR.jbw.reward}</span>
        <span ref={flipRef} className="tabular-nums text-ink-faint" />
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Chip item={JBW_JELLY} label={STR.jbw.jelly} value={polarity > 0 ? '+1' : '−1'} />
        <Chip item={JBW_ONION} label={STR.jbw.onion} value={polarity > 0 ? '−1' : '+1'} />
        <Chip item={JBW_BANANA} label={STR.jbw.banana} value={STR.jbw.zero} />
        <Chip item={JBW_TRUFFLE} label={STR.jbw.truffle} value={STR.jbw.beacon} />
      </div>
      <div className="h-[3px] overflow-hidden rounded-full bg-white/10">
        <div
          ref={cycleRef}
          className="h-full origin-left rounded-full bg-accent/70"
          style={{ transform: 'scaleX(0)', willChange: 'transform' }}
        />
      </div>
    </Panel>
  );
}
