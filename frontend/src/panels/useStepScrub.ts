import { useRef, type PointerEvent } from 'react';
import { scrubToStep } from '../theater/controller';
import { navigate } from '../url/useUrlState';

export function useStepScrub<T extends HTMLElement>(
  range: () => { min: number; max: number },
  inset = { left: 0, right: 0 },
  onStart?: () => void,
) {
  const stepAt = useRef<(x: number) => number>(() => 0);
  const seek = (event: PointerEvent<T>): void => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const step = stepAt.current(event.clientX);
    scrubToStep(step);
    if (event.type === 'pointerup') {
      event.currentTarget.releasePointerCapture(event.pointerId);
      navigate({ at: step });
    }
  };
  return {
    onPointerDown(event: PointerEvent<T>) {
      if (event.button !== 0 || !event.isPrimary) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const { min, max } = range();
      const left = rect.left + inset.left;
      const width = Math.max(1, rect.width - inset.left - inset.right);
      stepAt.current = (x) => Math.round(min + Math.max(0, Math.min(1, (x - left) / width)) * (max - min));
      event.currentTarget.setPointerCapture(event.pointerId);
      onStart?.();
      seek(event);
    },
    onPointerMove: seek,
    onPointerUp: seek,
  };
}
