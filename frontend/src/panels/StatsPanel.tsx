import { clsx } from 'clsx';
import { Info, PanelRightClose } from 'lucide-react';
import type { Policy, RunDetail } from '@shared/types';
import { Panel } from '../glass/Panel';
import { Hotkey } from './charts';
import { STR } from '../strings';
import { ComputeView } from './ComputeView';
import { HealthView } from './HealthView';
import { RewardView } from './RewardView';
import { StandingsView } from './StandingsView';

export type PanelView = keyof typeof STR.panels;

export const PANEL_VIEWS = Object.keys(STR.panels) as PanelView[];

export function StatsPanel({
  view,
  setView,
  runs,
  best,
  active,
  onHide,
  onAbout,
}: {
  view: PanelView;
  setView: (view: PanelView) => void;
  runs: Record<Policy, RunDetail | null>;
  best: RunDetail | null;
  active: RunDetail | null;
  onHide: () => void;
  onAbout: () => void;
}) {
  return (
    <Panel className="flex max-h-full w-full flex-col overflow-hidden py-2.5">
      <div className="flex items-center gap-1 px-2 pb-2">
        {PANEL_VIEWS.map((id, index) => (
          <Hotkey key={id} label={STR.palette.show(STR.panels[id])} keyName={String(index + 1)}>
            <button
              type="button"
              onClick={() => setView(id)}
              className={clsx(
                'rounded-md px-2.5 py-1 text-xs transition-colors active:bg-white/15',
                view === id ? 'bg-accent/20 text-accent' : 'text-ink-dim hover:bg-white/10 hover:text-ink',
              )}
            >
              {STR.panels[id]}
            </button>
          </Hotkey>
        ))}
        <span className="ml-auto flex items-center gap-1">
          <Hotkey label={STR.about.title} keyName="?" align="end">
            <button
              type="button"
              onClick={onAbout}
              aria-label={STR.about.open}
              className="flex size-6 shrink-0 items-center justify-center rounded-md text-ink-faint transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
            >
              <Info className="size-3.5" />
            </button>
          </Hotkey>
          <Hotkey label={STR.bar.hidePanel} keyName="c" align="end">
            <button
              type="button"
              onClick={onHide}
              aria-label={STR.bar.hidePanel}
              className="flex size-6 shrink-0 items-center justify-center rounded-md text-ink-faint transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
            >
              <PanelRightClose className="size-3.5" />
            </button>
          </Hotkey>
        </span>
      </div>
      {view === 'reward' && <RewardView runs={runs} best={best} active={active} />}
      {view === 'standings' && <StandingsView runs={runs} />}
      {view === 'compute' && <ComputeView runs={runs} />}
      {view === 'health' && <HealthView runs={runs} active={active} />}
    </Panel>
  );
}
