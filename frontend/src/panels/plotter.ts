import { formatStep } from '../format';

export const PLOT_INSET = { left: 62, right: 30, top: 20, bottom: 18 } as const;

export interface PlotLine {
  x: readonly number[];
  y: readonly (number | null)[];
  tone: string;
  width: number;
  dash?: readonly number[];
  label: string;
}

export interface PlotBand {
  x: readonly number[];
  lo: readonly (number | null)[];
  hi: readonly (number | null)[];
  tone: string;
}

export interface PlotSpec {
  xMax: number;
  lines: readonly PlotLine[];
  band?: PlotBand;
  yFormat: (value: number) => string;
}

export interface PlotMarks {
  playhead: number | null;
  cursor: number | null;
}

const X_STEPS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 250000, 500000, 1000000, 2000000, 5000000];

const colors = new Map<string, string>();

export function resolveTone(el: Element, tone: string): string {
  const key = tone.startsWith('var(') ? tone.slice(4, -1).trim() : tone;
  let color = colors.get(key);
  if (color === undefined) {
    color = key.startsWith('--') ? getComputedStyle(el).getPropertyValue(key).trim() : key;
    if (color === '') throw new Error(`unknown color ${tone}`);
    colors.set(key, color);
  }
  return color;
}

function niceStep(span: number, divisions: number): number {
  const raw = span / divisions;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / magnitude;
  return (unit < 1.5 ? 1 : unit < 3.5 ? 2 : unit < 7.5 ? 5 : 10) * magnitude;
}

export function valueAt(line: { x: readonly number[]; y: readonly (number | null)[] }, step: number): number | null {
  let lo = 0;
  let hi = line.x.length - 1;
  if (hi < 0) return null;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((line.x[mid] ?? 0) < step) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs((line.x[lo - 1] ?? 0) - step) <= Math.abs((line.x[lo] ?? 0) - step)) lo--;
  const v = line.y[lo];
  return v === null || v === undefined || !Number.isFinite(v) ? null : v;
}

export function drawPlot(canvas: HTMLCanvasElement, spec: PlotSpec, marks: PlotMarks): void {
  const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width <= 0 || height <= 0) return;
  const bw = Math.round(width * dpr);
  const bh = Math.round(height * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const xMax = Math.max(1, spec.xMax);
  const x0 = PLOT_INSET.left;
  const x1 = width - PLOT_INSET.right;
  const y0 = PLOT_INSET.top;
  const y1 = height - PLOT_INSET.bottom;
  let min = Infinity;
  let max = -Infinity;
  const scan = (x: readonly number[], y: readonly (number | null)[]): void => {
    for (let i = 0; i < x.length && (x[i] ?? 0) <= xMax; i++) {
      const v = y[i];
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  };
  for (const line of spec.lines) scan(line.x, line.y);
  if (spec.band !== undefined) {
    scan(spec.band.x, spec.band.lo);
    scan(spec.band.x, spec.band.hi);
  }
  if (!Number.isFinite(min)) {
    min = 0;
    max = 1;
  }
  if (min === max) {
    min -= Math.abs(min) * 0.1 + 1e-6;
    max += Math.abs(max) * 0.1 + 1e-6;
  }
  const pad = (max - min) * 0.06;
  min -= pad;
  max += pad;
  const toX = (step: number): number => x0 + (step / xMax) * (x1 - x0);
  const toY = (v: number): number => y1 - ((v - min) / (max - min)) * (y1 - y0);
  const ink = resolveTone(canvas, '--color-ink-dim');
  const grid = resolveTone(canvas, '--color-line');
  const mono = getComputedStyle(canvas).getPropertyValue('--font-mono').trim() || 'monospace';
  ctx.font = `11px ${mono}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  const yStep = niceStep(max - min, 4);
  ctx.lineWidth = 1;
  for (let v = Math.ceil(min / yStep) * yStep; v <= max; v += yStep) {
    const y = Math.round(toY(v)) + 0.5;
    ctx.strokeStyle = grid;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink;
    ctx.fillText(spec.yFormat(Math.abs(v) < yStep / 1000 ? 0 : v), x0 - 8, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xStep = X_STEPS.find((s) => (s / xMax) * (x1 - x0) >= 44) ?? X_STEPS[X_STEPS.length - 1]!;
  for (let s = 0; s <= xMax; s += xStep) ctx.fillText(formatStep(s), toX(s), y1 + 5);
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.clip();
  const trace = (x: readonly number[], y: readonly (number | null)[], forward: boolean, continued: boolean): boolean => {
    let open = continued;
    const n = x.length;
    for (let k = 0; k < n; k++) {
      const i = forward ? k : n - 1 - k;
      if ((x[i] ?? 0) > xMax) continue;
      const v = y[i];
      if (v === null || v === undefined || !Number.isFinite(v)) {
        open = false;
        continue;
      }
      if (open) ctx.lineTo(toX(x[i] ?? 0), toY(v));
      else ctx.moveTo(toX(x[i] ?? 0), toY(v));
      open = true;
    }
    return open;
  };
  if (spec.band !== undefined) {
    ctx.beginPath();
    trace(spec.band.x, spec.band.hi, true, false);
    trace(spec.band.x, spec.band.lo, false, true);
    ctx.closePath();
    ctx.fillStyle = resolveTone(canvas, spec.band.tone);
    ctx.globalAlpha = 0.12;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const line of spec.lines) {
    ctx.beginPath();
    trace(line.x, line.y, true, false);
    ctx.strokeStyle = resolveTone(canvas, line.tone);
    ctx.lineWidth = line.width;
    ctx.setLineDash(line.dash === undefined ? [] : [...line.dash]);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  const rule = (step: number, tone: string, alpha: number, dash: readonly number[]): void => {
    const x = Math.round(toX(step)) + 0.5;
    ctx.strokeStyle = resolveTone(canvas, tone);
    ctx.globalAlpha = alpha;
    ctx.setLineDash([...dash]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  };
  if (marks.playhead !== null && marks.playhead <= xMax) rule(marks.playhead, '--color-seed', 0.7, []);
  if (marks.cursor !== null && marks.cursor <= xMax) rule(marks.cursor, '--color-ink-faint', 0.6, [3, 3]);
  ctx.restore();
}
