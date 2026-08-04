// Cross-origin policy for the local API.
//
// This server reads a developer's machine (bench roots, recorded lanes) and binds to
// 127.0.0.1, but "local" is not a security boundary: under `access-control-allow-origin: *`
// ANY page the browser happens to have open can read every endpoint. So the allow-list is
// exactly the two dev-server origins the Vite client is served from, echoed back only when
// the request actually carries one of them.
//
// In production the client is served by THIS server (same-origin) and needs no grant at
// all, so no CORS headers are emitted — and with nothing origin-dependent to emit, no
// `vary: origin` either. In development the headers do vary by Origin, so `vary` is part
// of the answer, including for the requests that get no grant.

export const DEV_ORIGINS: readonly string[] = [
  'http://localhost:5173',
  'http://127.0.0.1:5173',
] as const;

const DEV_BASE: Record<string, string> = {
  'access-control-allow-headers': 'range, if-none-match',
  'access-control-expose-headers': 'content-range, accept-ranges, content-length, etag',
  'access-control-allow-methods': 'GET, HEAD, OPTIONS',
  vary: 'origin',
};

/** Headers merged into every response. Empty in production (same-origin). */
export function corsHeaders(req: Request, production: boolean): Record<string, string> {
  if (production) return {};
  const origin = req.headers.get('origin');
  if (origin !== null && DEV_ORIGINS.includes(origin)) {
    return { ...DEV_BASE, 'access-control-allow-origin': origin };
  }
  return { ...DEV_BASE };
}
