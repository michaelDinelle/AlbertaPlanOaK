const finite = (value: number | null | undefined): value is number => value !== null && value !== undefined && Number.isFinite(value);

export function formatRate(value: number | null | undefined, sig = 3): string {
  if (!finite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-4) return value.toExponential(1);
  return value.toFixed(Math.min(6, Math.max(0, sig - 1 - Math.floor(Math.log10(magnitude)))));
}

export function formatSigned(value: number | null | undefined): string {
  return finite(value) && value > 0 ? `+${formatRate(value)}` : formatRate(value);
}

export function formatCount(value: number | null | undefined): string {
  return finite(value) ? Math.round(value).toLocaleString('en-US') : '—';
}

const trimZero = (value: number): string => value.toFixed(value < 10 ? 1 : 0).replace(/\.0$/, '');

export function formatStep(value: number | null | undefined): string {
  if (!finite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${trimZero(value / 1e9)}B`;
  if (abs >= 1e6) return `${trimZero(value / 1e6)}M`;
  if (abs >= 1e3) return `${trimZero(value / 1e3)}k`;
  return String(Math.round(value));
}

export function formatPercent(fraction: number | null | undefined, decimals = 1): string {
  return finite(fraction) ? `${(fraction * 100).toFixed(decimals)}%` : '—';
}

export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
  const whole = Math.floor(seconds);
  const tenths = Math.floor((seconds - whole) * 10);
  const s = String(whole % 60).padStart(2, '0');
  const m = Math.floor(whole / 60) % 60;
  const h = Math.floor(whole / 3600);
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}.${tenths}` : `${m}:${s}.${tenths}`;
}

export function formatDate(iso: string): string {
  const date = new Date(Date.parse(iso));
  const monthDay = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return date.getFullYear() === new Date().getFullYear() ? monthDay : `${monthDay} ${date.getFullYear()}`;
}
