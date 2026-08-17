import uPlot from 'uplot';
import { formatRate, formatStep } from '../lib/format';
import { currentPosition, subscribePosition, throttled } from '../playback/store';

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function withAlpha(color: string, alpha: number): string {
  if (color.startsWith('#') && color.length === 7) {
    return `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
  }
  if (color.endsWith(')') && !color.includes('/') && !color.includes(',')) {
    return `${color.slice(0, -1)} / ${alpha})`;
  }
  return color;
}

export function monoFont(px: number): string {
  const family = cssVar('--font-mono') || 'ui-monospace, Menlo, monospace';
  return `${px}px ${family}`;
}

export interface TooltipParams {
  yFormat?: (value: number) => string;
  xFormat?: (value: number) => string;
  xUnit?: string;
}

export function tooltipPlugin(params: TooltipParams = {}): uPlot.Plugin {
  const yFormat = params.yFormat ?? ((v: number) => formatRate(v));
  const xFormat = params.xFormat ?? formatStep;
  let tip: HTMLDivElement | null = null;
  const hide = (): void => {
    if (tip !== null) tip.hidden = true;
  };
  return {
    hooks: {
      init: (u) => {
        tip = document.createElement('div');
        tip.className = 'chart-tip';
        tip.hidden = true;
        u.root.appendChild(tip);
        u.root.addEventListener('mouseleave', hide);
      },
      destroy: () => {
        tip?.remove();
        tip = null;
      },
      setCursor: (u) => {
        if (tip === null) return;
        const { idx, left, top } = u.cursor;
        if (idx === null || idx === undefined || left === undefined || left < 0) {
          tip.hidden = true;
          return;
        }
        const xs = u.data[0];
        const x = xs?.[idx];
        if (x === undefined || x === null) {
          tip.hidden = true;
          return;
        }
        const rows: { value: number; node: HTMLElement }[] = [];
        for (let i = 1; i < u.series.length; i++) {
          const series = u.series[i];
          if (!series || series.show === false || series.width === 0) continue;
          const value = u.data[i]?.[idx];
          if (value === null || value === undefined) continue;
          const row = document.createElement('div');
          row.className = 'tip-row';
          const swatch = document.createElement('span');
          swatch.className = 'tip-swatch';
          swatch.style.background = typeof series.stroke === 'string' ? series.stroke : cssVar('--color-oak');
          const name = document.createElement('span');
          name.className = 'tip-name';
          name.textContent = typeof series.label === 'string' ? series.label : `s${i}`;
          const amount = document.createElement('span');
          amount.className = 'tip-value';
          amount.textContent = yFormat(Number(value));
          row.append(swatch, name, amount);
          rows.push({ value: Number(value), node: row });
        }
        if (rows.length === 0) {
          tip.hidden = true;
          return;
        }
        rows.sort((a, b) => b.value - a.value);
        const head = document.createElement('div');
        head.className = 'tip-head';
        head.textContent = `${xFormat(Number(x))}${params.xUnit ? ` ${params.xUnit}` : ''}`;
        tip.replaceChildren(head, ...rows.map((row) => row.node));
        tip.hidden = false;
        const width = tip.offsetWidth;
        const flip = left + width + 24 > u.width;
        tip.style.transform = `translate(${Math.round(flip ? left - width - 14 : left + 14)}px, ${Math.round(Math.max(4, (top ?? 0) - 12))}px)`;
      },
    },
  };
}

export function zeroLinePlugin(): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const scale = u.scales['y'];
        if (!scale || scale.min === undefined || scale.max === undefined) return;
        if (scale.min > 0 || scale.max < 0) return;
        const y = Math.round(u.valToPos(0, 'y', true)) + 0.5;
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = withAlpha(cssVar('--color-ink'), 0.28);
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(u.bbox.left, y);
        ctx.lineTo(u.bbox.left + u.bbox.width, y);
        ctx.stroke();
        ctx.restore();
      },
    },
  };
}

export type SeekPhase = 'drag' | 'end' | 'tap';

export function seekPlugin(onSeek: (x: number, phase: SeekPhase) => void): uPlot.Plugin {
  return {
    hooks: {
      init: (u) => {
        const over = u.over;
        over.style.cursor = 'crosshair';
        let down = false;
        let moved = false;
        let startX = 0;
        const xAt = (ev: PointerEvent): number | null => {
          const rect = over.getBoundingClientRect();
          const x = u.posToVal(ev.clientX - rect.left, 'x');
          return Number.isFinite(x) ? x : null;
        };
        over.addEventListener('pointerdown', (ev) => {
          if (ev.button !== 0) return;
          down = true;
          moved = false;
          startX = ev.clientX;
          try { over.setPointerCapture(ev.pointerId); } catch {  }
        });
        over.addEventListener('pointermove', (ev) => {
          if (!down) return;
          if (Math.abs(ev.clientX - startX) > 3) moved = true;
          if (moved) {
            const x = xAt(ev);
            if (x !== null) onSeek(x, 'drag');
          }
        });
        const finish = (ev: PointerEvent): void => {
          if (!down) return;
          down = false;
          const x = xAt(ev);
          if (x !== null) onSeek(x, moved ? 'end' : 'tap');
        };
        over.addEventListener('pointerup', finish);
        over.addEventListener('pointercancel', () => { down = false; });
      },
    },
  };
}

export function playheadPlugin(): uPlot.Plugin {
  let unsubscribe: (() => void) | null = null;
  let step = currentPosition();
  return {
    hooks: {
      init: (u) => {
        unsubscribe = subscribePosition(throttled((next: number) => {
          step = next;
          u.redraw(false, false);
        }));
      },
      destroy: () => {
        unsubscribe?.();
        unsubscribe = null;
      },
      draw: (u) => {
        const x = u.valToPos(step, 'x', true);
        const { top, height, left, width } = u.bbox;
        if (!Number.isFinite(x) || x < left || x > left + width) return;
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = withAlpha(cssVar('--color-seed'), 0.9);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + height);
        ctx.stroke();
        ctx.restore();
      },
    },
  };
}

export interface BaselineBars {
  ppo: number | null;
  random: number | null;
  active: 'oak' | 'ppo' | 'random';
}

export function baselineGutterPlugin(getBars: () => BaselineBars): uPlot.Plugin {
  const bar = (
    u: uPlot,
    value: number | null,
    offset: number,
    color: string,
    capWidth: number,
    alpha: number,
    label: string,
  ): void => {
    if (value === null || !Number.isFinite(value)) return;
    const scale = u.scales['y'];
    if (!scale || scale.min === undefined || scale.max === undefined || scale.min === null || scale.max === null) return;
    const pr = devicePixelRatio;
    const x = u.bbox.left + u.bbox.width + offset * pr;
    const barWidth = 12 * pr;
    const base = u.valToPos(Math.max(0, scale.min), 'y', true);
    const top = u.valToPos(value, 'y', true);
    if (!Number.isFinite(base) || !Number.isFinite(top)) return;
    const ctx = u.ctx;
    ctx.save();
    ctx.globalAlpha = alpha * 0.55;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.min(base, top), barWidth, Math.max(1, Math.abs(base - top)));
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = capWidth * pr;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x + barWidth, top);
    ctx.stroke();
    ctx.font = monoFont(9 * pr);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = color;
    ctx.fillText(label, x + barWidth / 2, Math.max(u.bbox.top + 9 * pr, top - 2 * pr));
    ctx.restore();
  };
  return {
    hooks: {
      draw: (u) => {
        const bars = getBars();
        const alphaOf = (p: 'ppo' | 'random'): number => (bars.active === 'oak' || bars.active === p ? 1 : 0.45);
        bar(u, bars.ppo, 8, cssVar('--color-ppo'), 2, alphaOf('ppo'), 'ppo');
        bar(u, bars.random, 26, cssVar('--color-random'), 1, alphaOf('random'), 'rand');
      },
    },
  };
}
