export function formatRate(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-4) return value.toExponential(1);
  const decimals = Math.max(0, sig - 1 - Math.floor(Math.log10(magnitude)));
  return value.toFixed(Math.min(decimals, 6));
}

export function formatSigned(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const body = formatRate(value, sig);
  return value > 0 ? `+${body}` : body;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return Math.round(value).toLocaleString('en-US');
}

export function formatStep(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${trimZero(value / 1e9)}B`;
  if (abs >= 1e6) return `${trimZero(value / 1e6)}M`;
  if (abs >= 1e3) return `${trimZero(value / 1e3)}k`;
  return String(Math.round(value));
}

function trimZero(value: number): string {
  const text = value.toFixed(value < 10 ? 1 : 0);
  return text.endsWith('.0') ? text.slice(0, -2) : text;
}

export function formatPercent(fraction: number | null | undefined, decimals = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(decimals)}%`;
}

export function formatAgo(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return 'undated';
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return 'undated';
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 90) return 'just now';
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)}m ago`;
  const hours = minutes / 60;
  if (hours < 36) return `${Math.round(hours)}h ago`;
  const days = hours / 24;
  if (days < 30) return `${Math.round(days)}d ago`;
  const months = days / 30.44;
  if (months < 18) return `${Math.round(months)}mo ago`;
  return `${Math.round(days / 365.25)}y ago`;
}
