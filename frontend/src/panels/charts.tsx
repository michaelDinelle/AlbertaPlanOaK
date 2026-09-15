import type { Policy } from '../data/oak';

export const POLICY_ORDER = ['oak', 'ppo', 'random'] as const;
export const POLICY_NAMES: Record<Policy, string> = { oak: 'OaK', ppo: 'PPO', random: 'Random' };
export const POLICY_TOKENS: Record<Policy, string> = { oak: 'var(--color-oak)', ppo: 'var(--color-ppo)', random: 'var(--color-random)' };
export const BEST_TOKEN = 'var(--color-best)';

export function Key({ tone, label, dashed, block }: { tone: string; label: string; dashed?: boolean; block?: boolean }) {
  return (
    <span className="flex items-center gap-1 text-xs text-ink-dim">
      <span
        className={block === true ? 'h-2 w-3 rounded-sm opacity-40' : 'h-0.5 w-3 rounded-full'}
        style={dashed === true ? { backgroundImage: `repeating-linear-gradient(90deg, ${tone} 0 3px, transparent 3px 5px)` } : { background: tone }}
      />
      {label}
    </span>
  );
}
