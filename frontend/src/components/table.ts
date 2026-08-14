
import { TOKENS } from '@kit/tokens';


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
  if (attrs.value !== undefined && 'value' in node) (node as { value: string }).value = attrs.value;
  if (attrs.dataset) for (const key of Object.keys(attrs.dataset)) node.dataset[key] = attrs.dataset[key];
  if (attrs.style) Object.assign(node.style, attrs.style);
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

export function eyebrow(label: string, trailing?: Node | string | null): HTMLElement {
  return el('div', { class: 'eyebrow' }, [
    el('span', { class: 'eyebrow-rule' }),
    el('span', { class: 'eyebrow-text', text: label }),
    trailing ?? null,
  ]);
}

export function badge(text: string, kind = 'neutral', title?: string): HTMLElement {
  return el('span', { class: `badge badge-${kind}`, text, ...(title ? { title } : {}) });
}

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
  title?: string;
  actions?: Actions;
}

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

export function accentCss(slug: string): string {
  const value = TOKENS.accents[slug];
  if (value === undefined) return '#5c6a80';
  return `#${value.toString(16).padStart(6, '0')}`;
}
