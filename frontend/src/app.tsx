import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { ChevronDown, PanelRightOpen, RotateCcw } from 'lucide-react';
import { ENV_IDS } from '@shared/envs';
import type { Policy, RunDetail, RunIndexEntry } from '@shared/types';
import { AboutOaK } from './about/AboutOaK';
import { aboutSeen, markAboutSeen } from './about/seen';
import { baselineDetailQuery, runDetailQuery, runsQuery } from './data/queries';
import { Panel } from './glass/Panel';
import { formatStep } from './lib/format';
import { cycleEnv, keyAction } from './lib/keymap';
import { envShares, resolveFeaturedRun, runDateLabel, runEnvs, runLabel, runLabelFor, runShortId, sortRuns } from './lib/run';
import { CommandPalette } from './panels/CommandPalette';
import { Hotkey, POLICY_TOKENS } from './panels/charts';
import { JbwLegend } from './panels/JbwLegend';
import { PlaybackBar, PolicySwitch } from './panels/PlaybackBar';
import { PANEL_VIEWS, StatsPanel, type PanelView } from './panels/StatsPanel';
import {
  adoptDefaultEnv,
  adoptFromUrl,
  cycleSortMode,
  electLeader,
  resolveRun,
  selectEnv,
  selectPolicy,
  selectRun,
  selectionSnapshot,
  toggleDebug,
  unpromote,
  useSelection,
} from './state/selectionStore';
import { STR } from './strings';
import { setLive, useTapeProgress } from './tape/store';
import { setTransport, transportSnapshot } from './playback/store';
import { resetCamera, setTheaterInputs, stepBySteps, useStageMeta } from './theater/controller';
import { seedsFor } from './theater/loader';
import { TheaterStage } from './theater/TheaterStage';
import { sceneRegistry } from './scenes/table';
import { EnvTileButton } from './wall/EnvTileButton';
import { EnvWall } from './wall/EnvWall';
import { watchRuns } from './toasts';
import { urlState } from './url/useUrlState';

function Notice({ title, detail, pulse }: { title: string; detail: string | null; pulse?: boolean }) {
  return (
    <Panel className="flex max-w-md flex-col gap-1.5 px-5 py-4">
      <div className="font-mono text-sm text-ink">{title}</div>
      {detail !== null && <div className="text-xs text-ink-dim">{detail}</div>}
      {pulse === true && <span className="mt-1 size-1.5 motion-safe:animate-pulse rounded-full bg-accent" />}
    </Panel>
  );
}

function RunChip({
  entries,
  active,
}: {
  entries: readonly RunIndexEntry[] | undefined;
  active: RunDetail | null;
}) {
  const [open, setOpen] = useState(false);
  if (active === null) return null;
  const entry = entries?.find((run) => run.id === active.id);
  const runs = sortRuns(entries ?? []).slice(0, 16);
  return (
    <div className="pointer-events-auto relative" onKeyDown={(event) => { if (event.key === 'Escape') setOpen(false); }}>
      {open && <button type="button" aria-hidden className="fixed inset-0 cursor-default" onClick={() => setOpen(false)} />}
      <Panel className="relative px-3 py-1.5">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          title={STR.run.commit(active.id)}
          className="flex items-center gap-2.5 text-left"
        >
          <div className="flex flex-col">
            <span className="font-mono text-xs text-ink">{runLabelFor(active.id, entries, active)}</span>
            <span className="font-mono text-2xs text-ink-dim">
              {STR.run.facts(envShares(active.meta).length, formatStep(active.meta.steps), active.meta.preset)}
            </span>
          </div>
          {entry?.inFlight === true && (
            <span className="size-1.5 motion-safe:animate-pulse rounded-full bg-good" title={STR.bar.live} />
          )}
          <ChevronDown className={clsx('size-3 text-ink-faint transition-transform', open && 'rotate-180')} />
        </button>
      </Panel>
      {open && (
        <Panel className="absolute left-0 top-full mt-1.5 flex min-w-56 flex-col gap-0.5 p-1.5">
          {runs.map((run) => (
            <button
              key={run.id}
              type="button"
              onClick={() => {
                selectRun(run.id);
                setOpen(false);
              }}
              title={STR.run.switchTo(runLabel(run))}
              className={clsx(
                'flex flex-col gap-0.5 whitespace-nowrap rounded-md px-2.5 py-1.5 text-left transition-colors',
                run.id === active.id ? 'bg-accent/15' : 'hover:bg-white/10',
              )}
            >
              <span className="flex items-center gap-2 text-xs text-ink">
                {runDateLabel(run.startedAt)}
                {run.role === 'best' && <span className="text-best">{STR.best}</span>}
                {run.policy !== 'oak' && <span style={{ color: POLICY_TOKENS[run.policy] }}>{STR.policy[run.policy]}</span>}
                {run.ablationLabel !== 'none' && <span className="text-ink-dim">{run.ablationLabel}</span>}
                {run.inFlight && <span className="size-1.5 motion-safe:animate-pulse rounded-full bg-good" title={STR.bar.live} />}
              </span>
              <span className="font-mono text-2xs tabular-nums text-ink-dim">
                {runShortId(run.id)} · {STR.run.facts(run.envIds.length, formatStep(run.steps), run.preset)}
              </span>
            </button>
          ))}
        </Panel>
      )}
    </div>
  );
}

export function App() {
  const sel = useSelection();
  const meta = useStageMeta();
  const runsQ = useQuery(runsQuery());
  const entries = runsQ.data;
  const entry = useMemo(() => entries?.find((run) => run.id === sel.runId) ?? null, [entries, sel.runId]);

  const detailQ = useQuery({
    ...runDetailQuery(sel.runId ?? '', entry?.inFlight === true),
    enabled: sel.runId !== null,
  });
  const detail = detailQ.data ?? null;

  const bestId = useMemo(() => entries?.find((run) => run.role === 'best')?.id, [entries]);
  const bestQ = useQuery({ ...baselineDetailQuery(bestId ?? ''), enabled: bestId !== undefined && bestId !== sel.runId });
  const best = bestId !== undefined && bestId !== sel.runId ? bestQ.data ?? null : null;
  const ppoId = detail?.baselines.ppo;
  const randomId = detail?.baselines.random;
  const ppoQ = useQuery({ ...baselineDetailQuery(ppoId ?? ''), enabled: ppoId !== undefined });
  const randomQ = useQuery({ ...baselineDetailQuery(randomId ?? ''), enabled: randomId !== undefined });
  const ppoDetail = ppoQ.data ?? null;
  const randomDetail = randomQ.data ?? null;
  const runs = useMemo<Record<Policy, RunDetail | null>>(
    () => ({ oak: detail, ppo: ppoDetail, random: randomDetail }),
    [detail, ppoDetail, randomDetail],
  );

  const baselineId = sel.policy === 'oak' ? undefined : detail?.baselines[sel.policy];
  const active = sel.policy === 'oak'
    ? detail
    : baselineId === undefined ? null : sel.policy === 'ppo' ? ppoDetail : randomDetail;
  const baselinePending =
    sel.policy !== 'oak'
    && (detail === null
      ? detailQ.isPending
      : baselineId !== undefined && (sel.policy === 'ppo' ? ppoQ : randomQ).isPending);

  const envList = useMemo(() => (detail === null ? [] : runEnvs(detail.meta, detail.recordings)), [detail]);
  const wallEnvs = envList.length > 0 ? envList : ENV_IDS;
  const progress = useTapeProgress(sel.runId);
  const stride = Math.max(1, progress?.windowSteps ?? 1);
  const live = entry?.inFlight === true;

  const [view, setView] = useState<PanelView>('reward');
  const [wallOpen, setWallOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(() => !aboutSeen());
  const [panelHidden, setPanelHidden] = useState(false);
  const [panelSettled, setPanelSettled] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const closeAbout = (): void => {
    setAboutOpen(false);
    markAboutSeen();
  };

  useEffect(() => {
    if (!panelHidden) {
      setPanelSettled(false);
      return;
    }
    const timer = window.setTimeout(() => setPanelSettled(true), 320);
    return () => window.clearTimeout(timer);
  }, [panelHidden]);

  useEffect(() => {
    adoptFromUrl(urlState());
    const onPop = (): void => adoptFromUrl(urlState());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => watchRuns(entries), [entries]);

  useEffect(() => {
    if (sel.runId !== null || entries === undefined) return;
    const pick = resolveFeaturedRun(entries);
    if (pick.runId !== null) resolveRun(pick.runId);
  }, [entries, sel.runId]);

  useEffect(() => {
    if (detail === null || sel.envId !== null) return;
    const first = envList.find((id) => seedsFor(detail.recordings, id).length > 0) ?? envList[0];
    if (first !== undefined) adoptDefaultEnv(first);
  }, [detail, envList, sel.envId]);

  useEffect(() => {
    if (active === null || sel.envId === null || baselinePending) return;
    electLeader(seedsFor(active.recordings, sel.envId));
  }, [active, sel.envId, baselinePending]);

  useEffect(() => {
    setTheaterInputs({
      active,
      policy: sel.policy,
      baselineId,
      baselinePending,
      live: sel.policy === 'oak' && live,
    });
  }, [active, sel.policy, baselineId, baselinePending, live, setTheaterInputs]);

  useEffect(() => {
    const runId = sel.runId;
    if (runId === null) return;
    setLive(runId, live);
    return () => setLive(runId, false);
  }, [sel.runId, live]);

  useEffect(() => {
    if (sel.envId !== null) void sceneRegistry[sel.envId]?.();
  }, [sel.envId]);

  const policies = useMemo<Policy[]>(
    () => ['oak', ...(['ppo', 'random'] as const).filter((p) => detail?.baselines[p] !== undefined)],
    [detail],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if ((event.metaKey || event.ctrlKey) && (event.key === 'k' || event.key === 'K')) {
        event.preventDefault();
        setPaletteOpen((open) => !open);
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey || paletteOpen) return;
      const node = event.target;
      if (
        node instanceof HTMLInputElement
        || node instanceof HTMLSelectElement
        || node instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const action = keyAction(event.key, event.shiftKey);
      if (action === null) return;
      event.preventDefault();
      switch (action.kind) {
        case 'toggle':
          setTransport({ playing: !transportSnapshot().playing });
          break;
        case 'step':
          stepBySteps(action.frames * stride);
          break;
        case 'env': {
          const next = cycleEnv(envList, selectionSnapshot().envId, action.direction);
          if (next !== null) selectEnv(next);
          break;
        }
        case 'sort':
          cycleSortMode();
          break;
        case 'debug':
          toggleDebug();
          break;
        case 'policy': {
          const order = policies;
          const at = order.indexOf(selectionSnapshot().policy);
          const next = order[(at + 1) % order.length];
          if (next !== undefined) selectPolicy(next);
          break;
        }
        case 'fullscreen':
          if (document.fullscreenElement !== null) void document.exitFullscreen();
          else rootRef.current?.requestFullscreen().catch(() => {});
          break;
        case 'panel':
          setPanelHidden((hidden) => !hidden);
          break;
        case 'view': {
          const next = PANEL_VIEWS[action.index];
          if (next !== undefined) {
            setView(next);
            setPanelHidden(false);
          }
          break;
        }
        case 'about':
          if (aboutOpen) closeAbout();
          else setAboutOpen(true);
          break;
        case 'escape':
          if (aboutOpen) break;
          if (wallOpen) setWallOpen(false);
          else unpromote();
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [envList, stride, wallOpen, paletteOpen, aboutOpen, policies]);

  const notice = runsQ.isError
    ? {
        title: STR.notices.apiDown,
        detail: `${runsQ.error instanceof Error ? runsQ.error.message : String(runsQ.error)} — ${STR.notices.apiHint}`,
      }
    : sel.runId !== null && detailQ.isError
      ? {
          title: STR.notices.loadFailed(sel.runId),
          detail: detailQ.error instanceof Error ? detailQ.error.message : String(detailQ.error),
        }
      : entries !== undefined && sel.runId === null
        ? { title: STR.notices.noRun, detail: STR.notices.noRunHint }
        : meta.note !== null
          ? { title: meta.note, detail: meta.noteDetail, pulse: meta.status === 'loading' }
          : null;

  return (
    <div ref={rootRef} className="fixed inset-0 overflow-hidden bg-ground">
      <TheaterStage />
      {notice !== null && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center p-6">
          <Notice title={notice.title} detail={notice.detail ?? null} pulse={notice.pulse} />
        </div>
      )}
      <div className="pointer-events-none absolute left-4 top-4 z-20 flex items-start gap-2">
        <RunChip entries={entries} active={active} />
        <Panel>
          <button type="button" onClick={resetCamera} aria-label={STR.bar.resetCamera} title={STR.bar.resetCamera} className="pointer-events-auto flex h-9 items-center gap-2 rounded-lg px-3 text-xs text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15">
            <RotateCcw className="size-3.5" /><span className="hidden sm:inline">{STR.bar.resetCamera}</span>
          </button>
        </Panel>
      </div>
      <div
        className={clsx(
          'pointer-events-none absolute bottom-20 right-(--gutter) top-(--gutter) z-20 w-(--dock-width) max-w-[calc(100vw-2*var(--gutter))] transition-[transform,opacity] duration-300 ease-out motion-reduce:transition-none',
          panelHidden ? 'translate-x-[calc(100%+1rem)] opacity-0' : 'translate-x-0 opacity-100',
        )}
      >
        {!(panelHidden && panelSettled) && (
          <StatsPanel
            view={view}
            setView={setView}
            runs={runs}
            best={best}
            active={active}
            onHide={() => setPanelHidden(true)}
            onAbout={() => setAboutOpen(true)}
          />
        )}
      </div>
      {panelHidden && panelSettled && (
        <div className="absolute right-4 top-4 z-20">
          <Panel>
            <Hotkey label={STR.bar.showPanel} keyName="c" align="end">
              <button
                type="button"
                aria-label={STR.bar.showPanel}
                onClick={() => setPanelHidden(false)}
                className="flex size-9 items-center justify-center rounded-lg text-ink-dim transition-colors hover:bg-white/10 hover:text-ink active:bg-white/15"
              >
                <PanelRightOpen className="size-4" />
              </button>
            </Hotkey>
          </Panel>
        </div>
      )}
      <div className="pointer-events-none absolute inset-x-0 bottom-4 z-30 grid grid-cols-[1fr_minmax(0,760px)_1fr] items-end gap-3 px-4">
        <div className="flex flex-col items-start gap-3 justify-self-start">
          {sel.envId === 'JellyBeanWorld-cont-v0' && <JbwLegend />}
          <EnvTileButton onOpen={() => setWallOpen(true)} />
        </div>
        <PlaybackBar stride={stride} live={live} fullscreenTarget={rootRef} />
        <div className="justify-self-end">
          <PolicySwitch runs={runs} entries={entries} />
        </div>
      </div>
      {wallOpen && (
        <EnvWall detail={detail} envList={wallEnvs} onClose={() => setWallOpen(false)} />
      )}
      {paletteOpen && (
        <CommandPalette
          entries={entries}
          detail={detail}
          envList={wallEnvs}
          setView={setView}
          openWall={() => setWallOpen(true)}
          openAbout={() => setAboutOpen(true)}
          panelHidden={panelHidden}
          togglePanel={() => setPanelHidden((hidden) => !hidden)}
          fullscreenTarget={rootRef}
          onClose={() => setPaletteOpen(false)}
        />
      )}
      {aboutOpen && <AboutOaK onClose={closeAbout} />}
    </div>
  );
}
