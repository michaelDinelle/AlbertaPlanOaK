import { clsx } from 'clsx';
import { motion } from 'motion/react';
import type { CSSProperties, ReactNode } from 'react';
import { useLiquidGlass } from './useLiquidGlass';

const CONTAIN: CSSProperties = { contain: 'layout style paint' };

const ENTER = {
  initial: { transform: 'translateY(14px) scale(0.98)' },
  animate: { transform: 'translateY(0px) scale(1)' },
  transition: { type: 'spring', stiffness: 420, damping: 34 },
} as const;

const LENS_OPTICS: GlassOptions = { blur: 10, saturate: 1.35 };

export interface PanelProps {
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

export function Lens({ className, style, children, glass }: PanelProps & { glass?: GlassOptions }) {
  const ref = useLiquidGlass<HTMLDivElement>({ ...LENS_OPTICS, ...glass });
  return (
    <motion.div
      ref={ref}
      {...ENTER}
      className={clsx('glass pointer-events-auto rounded-lg', className)}
      style={{ ...CONTAIN, ...style }}
    >
      {children}
    </motion.div>
  );
}

export function Frost({ className, style, children }: PanelProps) {
  return (
    <motion.div
      {...ENTER}
      className={clsx('glass glass-frost pointer-events-auto rounded-lg', className)}
      style={{ ...CONTAIN, ...style }}
    >
      {children}
    </motion.div>
  );
}
