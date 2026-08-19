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

export function corsHeaders(req: Request, production: boolean): Record<string, string> {
  if (production) return {};
  const origin = req.headers.get('origin');
  if (origin !== null && DEV_ORIGINS.includes(origin)) {
    return { ...DEV_BASE, 'access-control-allow-origin': origin };
  }
  return { ...DEV_BASE };
}
