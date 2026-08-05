// DOM primitives shared by the app chrome, plus the sortable data table.
// (The primitives live here rather than in a fifth module so the client track
// stays inside its file ownership; they are deliberately tiny.)

import { TOKENS } from '@kit/tokens';
import { ENVS, type EnvId } from '@shared/envs';
import type { RunIndexEntry } from '@shared/types';
import { compareValues, envIdsOf, envName, formatCount, type EnvShare, type SortDirection } from '../state';

// ---------- element helpers ----------

export interface ElAttrs {
  class?: string;
  text?: string;
  html?: string;
  title?: string;
  href?: string;
  type?: string;
  value?: string;
  placeholder?: string;
  role?: string;
  ariaLabel?: string;
  ariaPressed?: string;
  ariaCurrent?: string;
  tabIndex?: number;
  dataset?: Record<string, string>;
  style?: Partial<CSSStyleDeclaration>;
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: ElAttrs = {},
  children: readonly (Node | string | null | undefined)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (attrs.class) node.className = attrs.class;
  if (attrs.text !== undefined) node.textContent = attrs.text;
  if (attrs.html !== undefined) node.innerHTML = attrs.html;
  if (attrs.title) node.title = attrs.title;
  if (attrs.role) node.setAttribute('role', attrs.role);
  if (attrs.ariaLabel) node.setAttribute('aria-label', attrs.ariaLabel);
  if (attrs.ariaPressed) node.setAttribute('aria-pressed', attrs.ariaPressed);
  if (attrs.ariaCurrent) node.setAttribute('aria-current', attrs.ariaCurrent);
  if (attrs.tabIndex !== undefined) node.tabIndex = attrs.tabIndex;
  if (attrs.href !== undefined && node instanceof HTMLAnchorElement) node.href = attrs.href;
  if (attrs.type && (node instanceof HTMLInputElement || node instanceof HTMLButtonElement)) node.type = attrs.type;
  if (attrs.placeholder && (node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement)) {
    node.placeholder = attrs.placeholder;
  }
  // options and selects carry values too — setting it only on inputs silently
  // produced valueless <option>s that no select could ever select
  if (attrs.value !== undefined && 'value' in node) (node as { value: string }).value = attrs.value;
  if (attrs.dataset) for (const key of Object.keys(attrs.dataset)) node.dataset[key] = attrs.dataset[key];
  if (attrs.style) Object.assign(node.style, attrs.style);
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

/** micro-caps section label with a leading hairline — the app's structural device */
export function eyebrow(label: string, trailing?: Node | string | null): HTMLElement {
  return el('div', { class: 'eyebrow' }, [
    el('span', { class: 'eyebrow-rule' }),
    el('span', { class: 'eyebrow-text', text: label }),
    trailing ?? null,
  ]);
}

export function statTile(
  label: string,
  value: string,
  options: { hint?: string; tone?: 'neutral' | 'pos' | 'neg' | 'dim'; mono?: boolean } = {},
): HTMLElement {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'stat-label', text: label }),
    el('div', { class: `stat-value tone-${options.tone ?? 'neutral'}`, text: value }),
    options.hint ? el('div', { class: 'stat-hint', text: options.hint }) : null,
  ]);
}

export function badge(text: string, kind = 'neutral', title?: string): HTMLElement {
  return el('span', { class: `badge badge-${kind}`, text, ...(title ? { title } : {}) });
}

/**
 * An anchor that reads as a button — the action shape empty states and view
 * headers use. Default is the hairline `.btn`; `quiet` swaps the border for a
 * raised fill, which only reads on the page background (inside a `.run-head`,
 * whose surface IS `--bg-raised`, a quiet button is invisible).
 */
export function linkButton(label: string, href: string, options: { quiet?: boolean; title?: string } = {}): HTMLElement {
  return el('a', {
    class: `btn${options.quiet ? ' btn-quiet' : ''}`,
    href,
    text: label,
    ...(options.title ? { title: options.title } : {}),
  });
}

export function actionButton(label: string, onClick: () => void, options: { quiet?: boolean } = {}): HTMLElement {
  const button = el('button', { class: `btn${options.quiet ? ' btn-quiet' : ''}`, text: label, type: 'button' });
  button.addEventListener('click', onClick);
  return button;
}

type Actions = Node | readonly (Node | null | undefined)[] | null | undefined;

/**
 * One or many action nodes get the same footer row, so a state with a single
 * "Try again" and one with three links share a baseline. Returns null when
 * nothing survives, which keeps an empty `.empty-actions` (a 0-height flex row
 * that still eats the container's gap) out of the DOM.
 */
function actionRow(actions: Actions): HTMLElement | null {
  const list = (actions === null || actions === undefined
    ? []
    : Array.isArray(actions)
      ? actions
      : [actions as Node]
  ).filter((node): node is Node => node !== null && node !== undefined);
  if (list.length === 0) return null;
  return el('div', { class: 'empty-actions' }, list);
}

export function emptyState(title: string, detail?: string, action?: Actions): HTMLElement {
  return el('div', { class: 'empty' }, [
    el('div', { class: 'empty-title', text: title }),
    detail ? el('div', { class: 'empty-detail', text: detail }) : null,
    actionRow(action),
  ]);
}

export interface ErrorStateOptions {
  /** overrides the default "Could not load" headline */
  title?: string;
  /** rendered after the retry button */
  actions?: Actions;
}

/**
 * The failure twin of `emptyState`. `retry` is optional on purpose: a 404 is a
 * settled answer, and offering "Try again" on one trains people to mash a button
 * that can only fail again — those callers pass a title, a plain-English detail
 * and a way out instead.
 */
export function errorState(message: string, retry?: () => void, options: ErrorStateOptions = {}): HTMLElement {
  const extra = options.actions === undefined ? [] : Array.isArray(options.actions) ? options.actions : [options.actions];
  return el('div', { class: 'empty empty-error' }, [
    el('div', { class: 'empty-title', text: options.title ?? 'Could not load' }),
    el('div', { class: 'empty-detail', text: message }),
    actionRow([retry ? actionButton('Try again', retry) : null, ...(extra as readonly (Node | null)[])]),
  ]);
}

export function spinner(label = 'Loading'): HTMLElement {
  return el('div', { class: 'loading' }, [el('span', { class: 'loading-pip' }), el('span', { text: label })]);
}

// ---------- app header ----------

/**
 * The one persistent chrome element, ONE bar: identity on the left (the brand
 * is the way home), and whatever the view needs on the right — the wall puts
 * its run picker there, the live view its pip. There is no nav: this is a
 * one-page app and a "Wall" link on the wall was a button to nowhere.
 */
export function appHeader(trailing?: Node | null): HTMLElement {
  return el('header', { class: 'app-header' }, [
    el('a', { class: 'brand', href: '#/', title: 'OaK Preserve — the wall' }, [
      el('span', { class: 'brand-mark', ariaLabel: 'OaK Preserve' }),
      el('span', { class: 'brand-name', text: 'OaK' }),
      el('span', { class: 'brand-sub', text: 'Preserve' }),
    ]),
    el('div', { class: 'header-trailing' }, [trailing ?? null]),
  ]);
}

export function viewShell(trailing?: Node | null): { root: HTMLElement; body: HTMLElement } {
  const body = el('main', { class: 'view-body' });
  const root = el('div', { class: 'view' }, [appHeader(trailing), body]);
  return { root, body };
}

export interface Crumb {
  label: string;
  /** omit on the trailing crumb — it is where you already are */
  href?: string;
}

/**
 * The second navigation device, under the header: where this page sits.
 *
 * Every deep view in the app is reachable from a shallower one and every deep
 * URL carries its parents (`#/run/:id`, `#/theater/:id/:envId/:seed`), so a
 * crumb trail is always reconstructible from the route alone — no history stack,
 * no view-to-view state, and a pasted URL keeps its way back. Separators are real
 * elements rather than `::after` content so the trail reads correctly aloud and
 * survives copy-paste.
 *
 * `trailing: true` prints the final separator and nothing after it — the shape
 * every view here uses, because the page's own `h1` IS the last crumb.
 */
export function breadcrumb(items: readonly Crumb[], options: { trailing?: boolean } = {}): HTMLElement {
  const nav = el('nav', { class: 'crumbs', ariaLabel: 'Breadcrumb' });
  items.forEach((item, index) => {
    if (index > 0) nav.append(el('span', { class: 'crumb-sep', text: '/' }));
    if (item.href === undefined) {
      nav.append(el('span', { class: 'crumb crumb-here', text: item.label, ariaCurrent: 'page' }));
    } else {
      nav.append(el('a', { class: 'crumb', href: item.href, text: item.label }));
    }
  });
  if (options.trailing && items.length > 0) nav.append(el('span', { class: 'crumb-sep', text: '/' }));
  return nav;
}

// ---------- sortable data table ----------

export interface TableColumn<R> {
  key: string;
  label: string;
  /** sortable value (numbers sort numerically, strings naturally) */
  value(row: R): unknown;
  /** rendered cell; defaults to String(value) */
  render?(row: R): Node | string;
  align?: 'left' | 'right';
  /** monospace + tabular figures (the default for numeric columns) */
  mono?: boolean;
  width?: string;
  title?: string;
}

export interface TableOptions<R> {
  columns: readonly TableColumn<R>[];
  rows: readonly R[];
  sortKey?: string;
  sortDirection?: SortDirection;
  rowKey?(row: R): string;
  onRowClick?(row: R): void;
  onRowHover?(row: R | null): void;
  emptyLabel?: string;
  /** rows rendered before the "show all" control appears */
  pageSize?: number;
}

export interface TableHandle<R> {
  root: HTMLElement;
  setRows(rows: readonly R[]): void;
  /** paint one row as focused (drives lane-chart <-> table cross-highlight) */
  highlight(key: string | null): void;
  destroy(): void;
}

export function dataTable<R>(options: TableOptions<R>): TableHandle<R> {
  let rows = [...options.rows];
  let sortKey = options.sortKey ?? options.columns[0]?.key ?? '';
  let direction: SortDirection = options.sortDirection ?? 'desc';
  let expanded = false;
  const pageSize = options.pageSize ?? 0;

  const head = el('thead');
  const bodyEl = el('tbody');
  const table = el('table', { class: 'table' }, [head, bodyEl]);
  const more = el('button', { class: 'btn btn-quiet table-more' });
  const root = el('div', { class: 'table-wrap' }, [table, more]);
  const rowNodes = new Map<string, HTMLElement>();

  const headerRow = el('tr');
  for (const column of options.columns) {
    const th = el('th', {
      class: `${column.align === 'right' || column.mono !== false ? 'align-right' : ''}`,
      ...(column.title ? { title: column.title } : {}),
    });
    if (column.width) th.style.width = column.width;
    if (column.align === 'left') th.className = '';
    const button = el('button', { class: 'th-sort', text: column.label }, [el('span', { class: 'th-caret' })]);
    button.addEventListener('click', () => {
      if (sortKey === column.key) direction = direction === 'asc' ? 'desc' : 'asc';
      else {
        sortKey = column.key;
        direction = 'desc';
      }
      paint();
    });
    th.append(button);
    headerRow.append(th);
  }
  head.append(headerRow);

  more.addEventListener('click', () => {
    expanded = !expanded;
    paint();
  });

  function paint(): void {
    const column = options.columns.find((c) => c.key === sortKey);
    const sorted = column ? [...rows].sort((a, b) => compareValues(column.value(a), column.value(b), direction)) : rows;
    const limit = pageSize > 0 && !expanded ? Math.min(pageSize, sorted.length) : sorted.length;

    for (const th of head.querySelectorAll('th')) th.removeAttribute('aria-sort');
    const index = options.columns.findIndex((c) => c.key === sortKey);
    if (index >= 0) {
      head.querySelectorAll('th')[index]?.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
    }

    rowNodes.clear();
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < limit; i++) {
      const row = sorted[i];
      if (row === undefined) continue;
      const key = options.rowKey?.(row) ?? String(i);
      const tr = el('tr', { dataset: { key } });
      for (const col of options.columns) {
        const rendered = col.render ? col.render(row) : String(col.value(row) ?? '');
        const td = el('td', { class: col.align === 'left' ? '' : 'align-right' });
        if (col.mono !== false && col.align !== 'left') td.classList.add('num');
        td.append(typeof rendered === 'string' ? document.createTextNode(rendered) : rendered);
        tr.append(td);
      }
      if (options.onRowClick) {
        tr.classList.add('is-clickable');
        tr.addEventListener('click', () => options.onRowClick?.(row));
      }
      if (options.onRowHover) {
        tr.addEventListener('mouseenter', () => options.onRowHover?.(row));
        tr.addEventListener('mouseleave', () => options.onRowHover?.(null));
      }
      rowNodes.set(key, tr);
      fragment.append(tr);
    }
    bodyEl.replaceChildren(fragment);

    if (sorted.length === 0) {
      bodyEl.append(
        el('tr', {}, [
          el('td', { class: 'table-empty', text: options.emptyLabel ?? 'No rows' }),
        ]),
      );
      const onlyCell = bodyEl.querySelector('.table-empty');
      onlyCell?.setAttribute('colspan', String(options.columns.length));
    }

    if (pageSize > 0 && sorted.length > pageSize) {
      more.hidden = false;
      more.textContent = expanded ? `Show top ${pageSize}` : `Show all ${sorted.length}`;
    } else {
      more.hidden = true;
    }
  }

  paint();

  return {
    root,
    setRows(next) {
      rows = [...next];
      paint();
    },
    highlight(key) {
      for (const [rowKey, node] of rowNodes) node.classList.toggle('is-highlight', rowKey === key);
      if (key !== null) rowNodes.get(key)?.scrollIntoView({ block: 'nearest' });
    },
    destroy() {
      rowNodes.clear();
      root.remove();
    },
  };
}

// ---------- env identity (moved here when the gallery page retired) ----------

/** The accent hex the 3D scenes use for this env slug (kit/tokens.ts). */
export function accentCss(slug: string): string {
  const value = TOKENS.accents[slug];
  if (value === undefined) return '#5c6a80';
  return `#${value.toString(16).padStart(6, '0')}`;
}

/**
 * The env spectrum strip: one accent cell per env, ordered by the canonical env
 * table. Two inputs, deliberately unequal in what they claim:
 *   * `EnvShare[]` — TRUE apportionment from `envShares(meta)`: cell widths are
 *     the real seed fractions and each cell names its seed count.
 *   * a run-index entry — IDENTITY ONLY: equal-width cells, no seed numbers,
 *     because the index never carries the split (inventing `batch / envIds.length`
 *     printed confident fiction on every mixed run).
 */
export function envSpectrum(source: Pick<RunIndexEntry, 'envIds'> | readonly EnvShare[]): HTMLElement {
  let shares: readonly EnvShare[] | null = null;
  let ids: EnvId[];
  if ('envIds' in source) {
    ids = envIdsOf(source);
  } else {
    shares = source;
    ids = source.map((share) => share.envId);
  }
  const strip = el('div', {
    class: 'spectrum',
    ariaLabel: ids.length > 0 ? `Environments: ${ids.map((id) => envName(id)).join(', ')}` : 'No environments',
    role: 'img',
  });
  if (ids.length === 0) {
    strip.append(el('span', { class: 'spectrum-cell', style: { flexGrow: '1', background: '#232c3d' } }));
    return strip;
  }
  ids.forEach((envId, index) => {
    const info = ENVS[envId];
    const share = shares?.[index];
    strip.append(el('span', {
      class: 'spectrum-cell',
      title: share ? `${info.name} — ${formatCount(share.lanes)} seeds` : info.name,
      style: {
        flexGrow: String(share ? Math.max(0.05, share.fraction) : 1),
        background: accentCss(info.slug),
      },
    }));
  });
  return strip;
}
