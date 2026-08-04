// SVG categorical bars: tier fractions, per-env positivity, plasticity quartiles.
//
// Mark spec (dataviz): thin bars, 4px rounded DATA end only (the baseline end
// stays square), a 2px surface gap between adjacent bars, and a direct value
// label on every row — there are never enough rows here for a legend to beat
// labelling each one. Magnitude bars take one hue (or an ordinal ramp);
// zero-crossing bars take the validated cyan/red diverging pair.

import { el } from './table';

const SVG_NS = 'http://www.w3.org/2000/svg';

const COLORS = {
  pos: '#2b98d9',
  neg: '#e0614f',
  track: 'rgba(132, 148, 171, 0.10)',
  tick: 'rgba(215, 223, 235, 0.55)',
  ramp: ['#22587a', '#2a7cae', '#2f9ade', '#79c9f2'] as const,
};

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key of Object.keys(attrs)) node.setAttribute(key, String(attrs[key]));
  return node;
}

/** Rounded data-end, square baseline-end. `dir` 1 = grows right, -1 = grows left. */
function barPath(x: number, y: number, width: number, height: number, radius: number, dir: 1 | -1): string {
  const w = Math.abs(width);
  if (w < 1) return `M${x - (dir < 0 ? 1 : 0)},${y} h1 v${height} h-1 Z`;
  const r = Math.min(radius, w, height / 2);
  if (dir === 1) {
    return `M${x},${y} H${x + w - r} A${r},${r} 0 0 1 ${x + w},${y + r} V${y + height - r} A${r},${r} 0 0 1 ${x + w - r},${y + height} H${x} Z`;
  }
  return `M${x},${y} H${x - w + r} A${r},${r} 0 0 0 ${x - w},${y + r} V${y + height - r} A${r},${r} 0 0 0 ${x - w + r},${y + height} H${x} Z`;
}

export interface BarItem {
  label: string;
  value: number;
  /** overrides the formatted value on the right */
  display?: string;
  hint?: string;
  color?: string;
  onClick?(): void;
}

export interface BarsSpec {
  items: readonly BarItem[];
  /** explicit domain; defaults to [min(0, …), max(…)] */
  domain?: [number, number];
  format?(value: number): string;
  /** tint each bar by its rank using the ordinal ramp (ordered categories only) */
  ramp?: boolean;
  rowHeight?: number;
  barHeight?: number;
  labelWidth?: number;
  valueWidth?: number;
}

export interface BarsHandle {
  root: HTMLElement;
  update(items: readonly BarItem[]): void;
  destroy(): void;
}

export interface RangeBarsHandle {
  root: HTMLElement;
  update(rows: readonly RangeRow[]): void;
  destroy(): void;
}

export function bars(spec: BarsSpec): BarsHandle {
  const rowHeight = spec.rowHeight ?? 24;
  const barHeight = spec.barHeight ?? 10;
  const valueWidth = spec.valueWidth ?? 76;
  const format = spec.format ?? ((v: number) => String(v));
  let items = [...spec.items];

  const root = el('div', { class: 'bars' });
  let frame = 0;
  let destroyed = false;

  function render(): void {
    if (destroyed) return;
    const width = Math.max(240, Math.floor(root.clientWidth || 480));
    const labelWidth = spec.labelWidth ?? Math.min(180, Math.max(96, Math.round(width * 0.3)));
    const trackX = labelWidth + 12;
    const trackWidth = Math.max(40, width - trackX - valueWidth - 8);
    const height = Math.max(rowHeight, items.length * rowHeight);

    const values = items.map((item) => item.value);
    const domain = spec.domain ?? [Math.min(0, ...values), Math.max(0, ...values, Number.EPSILON)];
    const [lo, hi] = domain;
    const span = hi - lo || 1;
    const zeroX = trackX + ((Math.min(Math.max(0, lo), hi) - lo) / span) * trackWidth;

    const node = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'bars-svg' });

    items.forEach((item, index) => {
      const y = index * rowHeight;
      const barY = y + (rowHeight - barHeight) / 2 + 1; // +1 keeps the 2px inter-bar gap

      const group = svg('g', { class: item.onClick ? 'bar-row is-clickable' : 'bar-row' });
      if (item.onClick) {
        group.addEventListener('click', () => item.onClick?.());
        group.setAttribute('tabindex', '0');
        group.setAttribute('role', 'button');
        group.addEventListener('keydown', (event) => {
          if ((event as KeyboardEvent).key === 'Enter') item.onClick?.();
        });
      }

      const label = svg('text', { x: labelWidth, y: y + rowHeight / 2 + 3.5, class: 'bar-label', 'text-anchor': 'end' });
      label.textContent = item.label;
      group.append(label);

      const track = svg('rect', {
        x: trackX, y: barY, width: trackWidth, height: barHeight, rx: 2, fill: COLORS.track,
      });
      group.append(track);

      const clamped = Math.max(lo, Math.min(hi, item.value));
      const magnitude = (Math.abs(clamped - Math.max(lo, Math.min(hi, 0))) / span) * trackWidth;
      const negative = clamped < 0;
      const color = item.color ?? (negative ? COLORS.neg : spec.ramp ? COLORS.ramp[Math.min(COLORS.ramp.length - 1, index)] ?? COLORS.pos : COLORS.pos);
      const bar = svg('path', {
        d: barPath(zeroX, barY, magnitude, barHeight, 4, negative ? -1 : 1),
        fill: color,
        class: 'bar-mark',
      });
      group.append(bar);

      if (lo < 0 && hi > 0) {
        group.append(svg('line', { x1: zeroX, y1: barY - 2, x2: zeroX, y2: barY + barHeight + 2, stroke: COLORS.tick, 'stroke-width': 1 }));
      }

      const value = svg('text', { x: width - 2, y: y + rowHeight / 2 + 3.5, class: 'bar-value', 'text-anchor': 'end' });
      value.textContent = item.display ?? format(item.value);
      group.append(value);

      if (item.hint) {
        const title = svg('title');
        title.textContent = `${item.label} — ${item.hint}`;
        group.append(title);
      }
      node.append(group);
    });

    root.replaceChildren(node);
  }

  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(render);
  });
  observer?.observe(root);
  queueMicrotask(render);

  return {
    root,
    update(next) {
      items = [...next];
      render();
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      root.remove();
    },
  };
}

// ---------- quartile ranges ----------

export interface RangeRow {
  label: string;
  q1: number;
  q2: number;
  q3: number;
  /** each row carries its OWN domain — these metrics do not share a scale */
  domain: [number, number];
  format?(value: number): string;
  hint?: string;
}

export interface RangeBarsSpec {
  rows: readonly RangeRow[];
  rowHeight?: number;
  labelWidth?: number;
}

/**
 * Quartile strips (q25 – q75 with a median tick). Every row is normalized to its
 * own domain and prints all three numbers, because zenith step, zenith rate and
 * decay ratio have nothing in common but their source — sharing one axis would
 * be the dual-axis mistake in bar form.
 */
export function rangeBars(spec: RangeBarsSpec): RangeBarsHandle {
  const rowHeight = spec.rowHeight ?? 34;
  let rows = [...spec.rows];
  const root = el('div', { class: 'bars bars-range' });
  let frame = 0;
  let destroyed = false;

  function render(): void {
    if (destroyed) return;
    const width = Math.max(240, Math.floor(root.clientWidth || 480));
    const labelWidth = spec.labelWidth ?? Math.min(170, Math.max(96, Math.round(width * 0.28)));
    const trackX = labelWidth + 12;
    const trackWidth = Math.max(40, width - trackX - 8);
    const height = Math.max(rowHeight, rows.length * rowHeight);
    const node = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'bars-svg' });

    rows.forEach((row, index) => {
      const y = index * rowHeight;
      const barY = y + 9;
      const barHeight = 8;
      const [lo, hi] = row.domain;
      const span = hi - lo || 1;
      const pos = (v: number) => trackX + ((Math.max(lo, Math.min(hi, v)) - lo) / span) * trackWidth;
      const format = row.format ?? ((v: number) => String(v));

      const group = svg('g', { class: 'bar-row' });
      const label = svg('text', { x: labelWidth, y: y + 15, class: 'bar-label', 'text-anchor': 'end' });
      label.textContent = row.label;
      group.append(label);

      group.append(svg('rect', { x: trackX, y: barY, width: trackWidth, height: barHeight, rx: 2, fill: COLORS.track }));

      const x1 = pos(Math.min(row.q1, row.q3));
      const x2 = pos(Math.max(row.q1, row.q3));
      group.append(svg('rect', {
        x: x1, y: barY, width: Math.max(2, x2 - x1), height: barHeight, rx: 4, fill: 'rgba(43,152,217,0.42)',
      }));
      const median = pos(row.q2);
      group.append(svg('line', { x1: median, y1: barY - 3, x2: median, y2: barY + barHeight + 3, stroke: COLORS.pos, 'stroke-width': 2, 'stroke-linecap': 'round' }));

      const readout = svg('text', { x: trackX, y: y + rowHeight - 5, class: 'bar-readout' });
      readout.textContent = `${format(row.q1)}  ·  ${format(row.q2)}  ·  ${format(row.q3)}`;
      group.append(readout);

      if (row.hint) {
        const title = svg('title');
        title.textContent = row.hint;
        group.append(title);
      }
      node.append(group);
    });

    root.replaceChildren(node);
  }

  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(render);
  });
  observer?.observe(root);
  queueMicrotask(render);

  return {
    root,
    update(next) {
      rows = [...next];
      render();
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      root.remove();
    },
  };
}

/** "0.8474 / 1 / 1" -> [0.8474, 1, 1]; returns null when the shape is unexpected. */
export function parseQuartiles(text: unknown): [number, number, number] | null {
  if (typeof text !== 'string') return null;
  const parts = text.split('/').map((p) => Number(p.trim()));
  if (parts.length !== 3) return null;
  const [a, b, c] = parts;
  if (a === undefined || b === undefined || c === undefined) return null;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return null;
  return [a, b, c];
}
