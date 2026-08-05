
import { el } from './table';

export interface SparklineSpec {
  values: readonly (number | null)[];
  xs?: readonly number[];
  height?: number;
  width?: number;
  stroke?: string;
  fill?: string;
  zeroLine?: boolean;
  endDot?: boolean;
  ariaLabel?: string;
}

export interface SparklineHandle {
  root: HTMLElement;
  update(values: readonly (number | null)[]): void;
  destroy(): void;
}

const DEFAULT_STROKE = '#2b98d9';
const DEFAULT_FILL = 'rgba(43, 152, 217, 0.14)';
const SURFACE = '#121722';

export function sparkline(spec: SparklineSpec): SparklineHandle {
  const height = spec.height ?? 32;
  let values = [...spec.values];

  const canvas = el('canvas', { class: 'spark-canvas' });
  const root = el('div', {
    class: 'spark',
    ...(spec.ariaLabel ? { role: 'img', ariaLabel: spec.ariaLabel } : {}),
  }, [canvas]);
  root.style.height = `${height}px`;
  if (spec.width) root.style.width = `${spec.width}px`;

  let frame = 0;
  let destroyed = false;

  function draw(): void {
    if (destroyed) return;
    const width = Math.max(1, Math.floor(spec.width ?? root.clientWidth));
    if (width <= 1) return;
    const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
    if (finite.length < 2) return;

    const pad = 3;
    let min = Math.min(...finite);
    let max = Math.max(...finite);
    if (min === max) {
      min -= Math.abs(min) * 0.1 + 1e-6;
      max += Math.abs(max) * 0.1 + 1e-6;
    }
    const span = max - min;
    const n = values.length;
    const xs = spec.xs;
    const xAt = (i: number): number => (
      xs !== undefined
        ? Math.min(Math.max(xs[i] ?? 0, 0), 1) * width
        : pad + (i / Math.max(1, n - 1)) * (width - pad * 2)
    );
    const yAt = (v: number): number => height - pad - ((v - min) / span) * (height - pad * 2);

    if ((spec.zeroLine ?? true) && min < 0 && max > 0) {
      ctx.save();
      ctx.strokeStyle = 'rgba(215, 223, 235, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      const y = Math.round(yAt(0)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.restore();
    }

    const points: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      points.push([xAt(i), yAt(v)]);
    }
    if (points.length < 2) return;

    const baseY = min < 0 && max > 0 ? yAt(0) : height - pad;
    ctx.beginPath();
    ctx.moveTo(points[0]![0], baseY);
    for (const [x, y] of points) ctx.lineTo(x, y);
    ctx.lineTo(points[points.length - 1]![0], baseY);
    ctx.closePath();
    if (spec.fill) {
      ctx.fillStyle = spec.fill;
    } else {
      const gradient = ctx.createLinearGradient(0, Math.min(...points.map((p) => p[1])), 0, baseY);
      gradient.addColorStop(0, DEFAULT_FILL);
      gradient.addColorStop(1, 'rgba(43, 152, 217, 0)');
      ctx.fillStyle = gradient;
    }
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(points[0]![0], points[0]![1]);
    for (const [x, y] of points.slice(1)) ctx.lineTo(x, y);
    ctx.strokeStyle = spec.stroke ?? DEFAULT_STROKE;
    ctx.lineWidth = 1.25;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();

    if (spec.endDot ?? true) {
      const last = points[points.length - 1]!;
      ctx.beginPath();
      ctx.arc(last[0], last[1], 3.4, 0, Math.PI * 2);
      ctx.fillStyle = SURFACE;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(last[0], last[1], 2.1, 0, Math.PI * 2);
      ctx.fillStyle = spec.stroke ?? DEFAULT_STROKE;
      ctx.fill();
    }
  }

  const observer = spec.width || typeof ResizeObserver === 'undefined'
    ? null
    : new ResizeObserver(() => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(draw);
      });
  observer?.observe(root);
  queueMicrotask(draw);

  return {
    root,
    update(next) {
      values = [...next];
      draw();
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      root.remove();
    },
  };
}
