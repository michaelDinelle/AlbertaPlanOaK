import { clsx } from 'clsx';
import type { CSSProperties, ReactNode } from 'react';

export function Card({
  className,
  style,
  children,
}: {
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}) {
  return (
    <div className={clsx('flex min-w-0 flex-col rounded-md border border-line bg-raised/60 px-4 py-3', className)} style={style}>
      {children}
    </div>
  );
}

export function CardHead({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-2 flex min-h-5 items-center gap-2">
      <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{title}</div>
      {children !== undefined && <div className="ml-auto flex items-center gap-2">{children}</div>}
    </div>
  );
}

export function ChartNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center py-5 text-center font-mono text-2xs text-ink-faint">
      {children}
    </div>
  );
}

export function LegendChip({
  swatch,
  label,
  dim,
  dashed,
}: {
  swatch: string;
  label: string;
  dim?: boolean;
  dashed?: boolean;
}) {
  return (
    <span className={clsx('flex items-center gap-1 font-mono text-2xs', dim === true ? 'text-ink-faint' : 'text-ink-dim')}>
      <span
        className="h-0.5 w-3 rounded-full"
        style={dashed === true
          ? { backgroundImage: `repeating-linear-gradient(90deg, ${swatch} 0 3px, transparent 3px 5px)` }
          : { background: swatch }}
      />
      {label}
    </span>
  );
}
