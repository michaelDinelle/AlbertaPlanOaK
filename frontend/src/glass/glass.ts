import './vendor.js';

declare global {
  interface Window {
    liquidGlass(el: Element, opts?: { saturate?: number }): { destroy(): void };
  }
}

export function glass(el: Element | null): (() => void) | undefined {
  if (el === null) return undefined;
  const handle = window.liquidGlass(el, { saturate: 1 });
  return () => handle.destroy();
}
