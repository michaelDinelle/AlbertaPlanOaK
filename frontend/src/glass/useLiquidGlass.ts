import { useEffect, useRef, type RefObject } from 'react';
import './vendor.js';

const SETTLE_MS = 160;

export function useLiquidGlass<T extends HTMLElement>(options?: GlassOptions): RefObject<T | null> {
  const ref = useRef<T | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    const glass = window.liquidGlass(el, optionsRef.current);
    const refractive = el.style.backdropFilter;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let armed = false;
    const observer = new ResizeObserver(() => {
      if (!armed) {
        armed = true;
        return;
      }
      if (glass.supported) el.style.backdropFilter = '';
      clearTimeout(timer);
      timer = setTimeout(() => {
        el.style.backdropFilter = refractive;
        glass.refresh();
      }, SETTLE_MS);
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
      glass.destroy();
    };
  }, []);

  return ref;
}
