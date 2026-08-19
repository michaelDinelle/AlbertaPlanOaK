import { clsx } from 'clsx';
import { PanelRightClose } from 'lucide-react';
import type { Policy, RunDetail } from '@shared/types';
import { Frost } from '../glass/Panel';
import { STR } from '../strings';
import { ComputeView } from './ComputeView';
import { HealthView } from './HealthView';
import { RewardView } from './RewardView';
import { StandingsView } from './StandingsView';

export type PanelView = keyof typeof STR.panels;

export function StatsPanel({
  view,
  setView,
  runs,
  active,
  onHide,
}: {
  view: PanelView;
  setView: (view: PanelView) => void;
  runs: Record<Policy, RunDetail | null>;
  active: RunDetail | null;
  onHide: () => void;
}) {
  return (
    <Frost className="flex max-h-full w-full flex-col overflow-hidden py-2.5">
      <div className="flex items-center gap-1 px-2 pb-2">
        {(Object.keys(STR.panels) as PanelView[]).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setView(id)}
            title={STR.palette.show(STR.panels[id])}
            className={clsx(
              'rounded-md px-2.5 py-1 text-xs transition-colors active:bg-white/15',
              view === id ? 'bg-accent/20 text-accent' : 'text-ink-dim hover:bg-white/10 hover:text-ink',
            )}
          >
            {STR.panels[id]}
          </button>
        ))}
        <button
          type="button"
          onClick={onHide}
          title={STR.bar.hidePanel}
          className="ml-auto flex size-6 shrink-0 items-center justify-center rounded-md text-ink-faint transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
        >
          <PanelRightClose className="size-3.5" />
        </button>
      </div>
      {view === 'reward' && <RewardView runs={runs} active={active} />}
      {view === 'standings' && <StandingsView runs={runs} />}
      {view === 'compute' && <ComputeView runs={runs} />}
      {view === 'health' && <HealthView runs={runs} active={active} />}
    </Frost>
  );
}
