
export interface ViewHandle { destroy(): void }
export type ViewMount = (el: HTMLElement, params: Record<string, string>) => ViewHandle | Promise<ViewHandle>;

export interface Route {
  pattern: string;
  view: () => Promise<{ mount: ViewMount }>;
}

function matchRoute(pattern: string, hash: string): Record<string, string> | null {
  const p = pattern.split('/').filter(Boolean);
  const h = hash.split('?')[0]!.split('/').filter(Boolean);
  if (p.length !== h.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < p.length; i++) {
    const seg = p[i]!;
    const val = h[i]!;
    if (seg.startsWith(':')) params[seg.slice(1)] = decodeURIComponent(val);
    else if (seg !== val) return null;
  }
  return params;
}

export function startRouter(el: HTMLElement, routes: Route[], fallback: string): void {
  let current: ViewHandle | null = null;
  let generation = 0;

  async function navigate(): Promise<void> {
    const gen = ++generation;
    const hash = window.location.hash || fallback;
    for (const route of routes) {
      const params = matchRoute(route.pattern, hash);
      if (!params) continue;
      const mod = await route.view();
      if (gen !== generation) return;
      current?.destroy();
      current = null;
      el.replaceChildren();
      const handle = await mod.mount(el, params);
      if (gen !== generation) {
        handle.destroy();
        return;
      }
      current = handle;
      return;
    }
    window.location.hash = fallback;
  }

  window.addEventListener('hashchange', () => void navigate());
  void navigate();
}
