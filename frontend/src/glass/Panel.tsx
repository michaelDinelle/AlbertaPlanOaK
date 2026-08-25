import { clsx } from 'clsx';
import { motion, useReducedMotion } from 'motion/react';
import type { CSSProperties, ReactNode } from 'react';
import { useLiquidGlass } from './useLiquidGlass';

const CONTAIN: CSSProperties = { contain: 'layout style paint' };

const ENTER = {
  initial: { transform: 'translateY(14px) scale(0.98)' },
  animate: { transform: 'translateY(0px) scale(1)' },
  transition: { type: 'spring', stiffness: 420, damping: 34 },
} as const;

const LENS_OPTICS: GlassOptions = { saturate: 1 };

export interface PanelProps {
  className?: string;
  children: ReactNode;
}

export function Lens({ className, children }: PanelProps) {
  const ref = useLiquidGlass<HTMLDivElement>(LENS_OPTICS);
  const enter = useReducedMotion() === true ? undefined : ENTER;
  return (
    <motion.div
      ref={ref}
      {...enter}
      className={clsx('glass pointer-events-auto rounded-lg', className)}
      style={CONTAIN}
    >
      {children}
    </motion.div>
  );
}

export function Frost({ className, children }: PanelProps) {
  const enter = useReducedMotion() === true ? undefined : ENTER;
  return (
    <motion.div
      {...enter}
      className={clsx('glass glass-frost pointer-events-auto rounded-lg', className)}
      style={CONTAIN}
    >
      {children}
    </motion.div>
  );
}
