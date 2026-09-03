import { useEffect, useRef, type RefObject } from 'react';
import './vendor.js';

export function useLiquidGlass<T extends HTMLElement>(options?: GlassOptions): RefObject<T | null> {
  const ref = useRef<T | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    const glass = window.liquidGlass(el, optionsRef.current);
    return () => glass.destroy();
  }, []);

  return ref;
}
