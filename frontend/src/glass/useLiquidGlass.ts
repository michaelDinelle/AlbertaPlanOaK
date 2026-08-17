import { useEffect, useRef, type RefObject } from 'react';
import './vendor.js';

const MAX_LENS_SIDE = 800;
const SETTLE_MS = 160;

function frostFilter(): string {
  return getComputedStyle(document.documentElement).getPropertyValue('--glass-frost-filter').trim();
}

export function useLiquidGlass<T extends HTMLElement>(options?: GlassOptions): RefObject<T | null> {
  const ref = useRef<T | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    if (el.offsetWidth > MAX_LENS_SIDE || el.offsetHeight > MAX_LENS_SIDE) {
      el.style.backdropFilter = frostFilter();
      return () => {
        el.style.backdropFilter = '';
      };
    }
    const glass = window.liquidGlass(el, optionsRef.current);
    const refractive = el.style.backdropFilter;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let armed = false;
    const observer = new ResizeObserver(() => {
      if (!armed) {
        armed = true;
        return;
      }
      if (glass.supported) el.style.backdropFilter = frostFilter();
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
