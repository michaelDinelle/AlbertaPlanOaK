import { clsx } from 'clsx';
import type { RunDetail } from '@shared/types';
import { Frost } from '../glass/Panel';
import { STR } from '../strings';
import { ComputeView } from './ComputeView';
import { HealthView } from './HealthView';
import { RewardView } from './RewardView';
import { StandingsView } from './StandingsView';
import type { PanelView } from './views';

export function StatsPanel({
  view,
  setView,
  detail,
  ppo,
  random,
  active,
}: {
  view: PanelView;
  setView: (view: PanelView) => void;
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
  active: RunDetail | null;
}) {
  return (
    <Frost className="flex h-full w-full flex-col overflow-hidden py-2.5">
      <div className="flex items-center gap-1 px-2 pb-2">
        {(Object.keys(STR.panels) as PanelView[]).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setView(id)}
            title={STR.palette.show(STR.panels[id])}
            className={clsx(
              'rounded-md px-2.5 py-1 text-xs transition-colors',
              view === id ? 'bg-accent/20 text-accent' : 'text-ink-dim hover:bg-white/10 hover:text-ink',
            )}
          >
            {STR.panels[id]}
          </button>
        ))}
      </div>
      {view === 'reward' && <RewardView detail={detail} ppo={ppo} random={random} active={active} />}
      {view === 'standings' && <StandingsView detail={detail} ppo={ppo} random={random} />}
      {view === 'compute' && <ComputeView detail={detail} ppo={ppo} random={random} />}
      {view === 'health' && <HealthView detail={detail} ppo={ppo} />}
    </Frost>
  );
}
