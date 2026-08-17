import { useEffect, useRef, type RefObject } from 'react';
import uPlot from 'uplot';
import type { OptionsBuilder, Size } from './specs';
import { cursorSync } from './sync';

export interface UPlotConfig {
  build: OptionsBuilder;
  data: uPlot.AlignedData;
  deps: readonly unknown[];
  syncKey?: string;
}

export interface UPlotHandle {
  ref: RefObject<HTMLDivElement | null>;
  plot(): uPlot | null;
}

export function useUPlot(config: UPlotConfig): UPlotHandle {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const configRef = useRef(config);
  configRef.current = config;

  useEffect(() => {
    const host = hostRef.current;
    if (host === null) return;
    let plot: uPlot | null = null;
    let size: Size = { width: 0, height: 0 };
    let frame = 0;

    const build = (): void => {
      plot?.destroy();
      plot = null;
      plotRef.current = null;
      if (size.width < 8 || size.height < 8) return;
      const current = configRef.current;
      const options = current.build(size);
      if (current.syncKey !== undefined) {
        options.cursor = { ...options.cursor, sync: cursorSync(current.syncKey) };
      }
      plot = new uPlot(options, current.data, host);
      plotRef.current = plot;
    };

    const measure = (): void => {
      const width = Math.floor(host.clientWidth);
      const height = Math.floor(host.clientHeight);
      if (width === size.width && height === size.height) return;
      size = { width, height };
      if (plot === null) build();
      else if (size.width >= 8 && size.height >= 8) plot.setSize(size);
      else build();
    };

    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    });
    observer.observe(host);
    measure();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      plot?.destroy();
      plot = null;
      plotRef.current = null;
    };
  }, config.deps);

  useEffect(() => {
    plotRef.current?.setData(config.data);
  }, [config.data]);

  return { ref: hostRef, plot: () => plotRef.current };
}
