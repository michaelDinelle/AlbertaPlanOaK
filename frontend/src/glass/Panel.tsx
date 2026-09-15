import { clsx } from 'clsx';
import type { CSSProperties, ReactNode } from 'react';
import { useLiquidGlass } from './useLiquidGlass';

const CONTAIN: CSSProperties = { contain: 'layout style paint' };

const LENS_OPTICS: GlassOptions = { saturate: 1 };

export interface PanelProps {
  className?: string;
  children: ReactNode;
}

export function Panel({ className, children }: PanelProps) {
  const ref = useLiquidGlass<HTMLDivElement>(LENS_OPTICS);
  return (
    <div ref={ref} className={clsx('glass pointer-events-auto rounded-lg motion-safe:animate-panel-enter', className)} style={CONTAIN}>
      {children}
    </div>
  );
}
