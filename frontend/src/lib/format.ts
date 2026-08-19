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
