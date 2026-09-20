import { useEffect, useMemo, useRef } from 'react';
import { entryFor, parseCatalog } from '@sim/catalog';
import { POLICIES, type Policy, type Recording, type RunIndexEntry } from './data/layout';
import { fetchRecording, fetchRuns } from './data/oak';
import { envLanes, leaderSeed, peakStep, rateWindow, runEnvs } from './derive';
import { formatDate, formatStep } from './format';
import { BEST_TOKEN, POLICY_NAMES, POLICY_ORDER, POLICY_TOKENS } from './panels/charts';
import { EnvPicker } from './panels/EnvPicker';
import { HealthView } from './panels/HealthView';
import { JbwLegend } from './panels/JbwLegend';
import { AUTHOR_LINKS } from './panels/links';
import { PlaybackBar, PolicySwitch } from './panels/PlaybackBar';
import { RewardView, type Comparison } from './panels/RewardView';
import { StandingsView } from './panels/StandingsView';
import { addRecording, cycleSpeed, fail, patch, selectEnv, selectRun, store, type Panel } from './store';
import { mountStage, type Alternate, type Stage } from './theater/stage';

const PANELS: Record<Panel, string> = { reward: 'Reward', standings: 'Standings', health: 'Health' };
const loading = new Set<string>();

async function ensure(id: string, refresh = false): Promise<void> {
  if (loading.has(id) || (!refresh && store.get().recordings[id] !== undefined)) return;
  loading.add(id);
  try {
    addRecording(await fetchRecording(id));
  } catch (error) {
    fail(error);
  } finally {
    loading.delete(id);
  }
}

export const baselineId = (run: Recording | null, policy: Policy, envId: string | null): string | null =>
  policy === 'oak' ? run?.id ?? null : envId === null ? null : run?.baselines[policy]?.[envId] ?? null;

const runLabel = (run: RunIndexEntry): string =>
  [formatDate(run.startedAt), run.role === 'best' ? 'best' : null, run.inFlight ? 'live' : null, run.id.slice(0, 7), `${run.envIds.length} envs · ${formatStep(run.steps)} steps · ${run.preset}`]
    .filter((part) => part !== null)
    .join(' · ');

export function App() {
  const s = store.use();
  const host = useRef<HTMLDivElement | null>(null);
  const stage = useRef<Stage | null>(null);
  const entry = s.runs?.find((run) => run.id === s.runId) ?? null;
  const run = s.runId === null ? null : s.recordings[s.runId] ?? null;
  const bestId = s.runs?.find((candidate) => candidate.role === 'best')?.id;
  const activeId = baselineId(run, s.policy, s.envId);
  const active = activeId === null ? null : s.recordings[activeId] ?? null;
  const live = s.policy === 'oak' && entry?.inFlight === true;
  const envs = useMemo(() => (run === null ? [] : runEnvs(run.meta)), [run]);
  const referenced = useMemo(() => (run === null ? [] : [...new Set(Object.values(run.baselines).flatMap((byEnv) => Object.values(byEnv)))]), [run]);
  const available = useMemo(() => Object.fromEntries(POLICIES.map((policy) => [policy, baselineId(run, policy, s.envId) !== null])) as Record<Policy, boolean>, [run, s.envId]);
  const lanes = useMemo(() => (active === null || s.envId === null ? null : envLanes(active.meta, s.envId)), [active, s.envId]);
  const seed = useMemo(() => {
    if (lanes === null || active === null || lanes.seeds.length === 0) return null;
    if (s.seed !== null && lanes.seeds.includes(s.seed)) return s.seed;
    return leaderSeed(active, lanes);
  }, [lanes, active, s.seed]);
  const startStep = useMemo(
    () => (active === null || lanes === null || seed === null ? null : peakStep(active, lanes.lanes[lanes.seeds.indexOf(seed)]!, rateWindow(active.meta.steps))),
    [active, lanes, seed],
  );
  const alternates = useMemo<Alternate[]>(() => {
    const envId = s.envId;
    if (active === null || envId === null || seed === null) return [];
    return POLICY_ORDER.flatMap((policy) => {
      const id = baselineId(run, policy, envId);
      const other = id === null || id === active.id ? undefined : s.recordings[id];
      if (other === undefined) return [];
      const laneSet = envLanes(other.meta, envId);
      const otherSeed = s.seed === null ? leaderSeed(other, laneSet) : seed;
      const lane = otherSeed === null ? undefined : laneSet.lanes[laneSet.seeds.indexOf(otherSeed)];
      return lane === undefined || otherSeed === null ? [] : [{ runId: other.id, lane, seed: otherSeed, entry: entryFor(parseCatalog(other.meta.envCatalog), envId), horizon: other.meta.steps }];
    });
  }, [active, s.envId, seed, s.seed, run, s.recordings]);
  const compare = useMemo<Comparison[]>(() => {
    const out: Comparison[] = [];
    for (const policy of POLICY_ORDER) {
      const id = baselineId(run, policy, s.envId);
      if (policy !== s.policy && id !== null && s.recordings[id] !== undefined) out.push({ key: policy, label: POLICY_NAMES[policy], tone: POLICY_TOKENS[policy], runId: id });
    }
    if (bestId !== undefined && bestId !== s.runId && s.recordings[bestId] !== undefined) out.push({ key: 'best', label: 'Best', tone: BEST_TOKEN, runId: bestId });
    return out;
  }, [run, s.runId, s.policy, s.envId, s.recordings, bestId]);
  const stride = active?.meta.batch ?? 1;

  useEffect(() => {
    const poll = (): Promise<void> => fetchRuns().then((runs) => patch({ runs, error: null }), fail);
    void poll();
    const timer = setInterval(() => {
      const runs = store.get().runs;
      if (runs === null || runs.some((candidate) => candidate.inFlight)) void poll();
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (s.runId !== null || s.runs === null) return;
    const pick = s.runs.find((candidate) => candidate.role === 'best') ?? s.runs.find((candidate) => candidate.policy === 'oak');
    if (pick !== undefined) selectRun(pick.id);
  }, [s.runs, s.runId]);

  useEffect(() => {
    for (const id of [s.runId, bestId, ...referenced]) if (id !== null && id !== undefined) void ensure(id);
  }, [s.runs, s.runId, bestId, referenced]);

  useEffect(() => {
    if (s.runId === null || entry?.inFlight !== true || s.recordings[s.runId] === undefined) return;
    void ensure(s.runId, true);
    stage.current?.pull();
  }, [s.runs]);

  useEffect(() => {
    const first = envs[0];
    if (first !== undefined && (s.envId === null || !envs.includes(s.envId))) selectEnv(first);
  }, [envs, s.envId]);

  useEffect(() => {
    const mounted = mountStage(host.current!);
    stage.current = mounted;
    return () => {
      mounted.destroy();
      stage.current = null;
    };
  }, []);

  useEffect(() => { stage.current?.[s.playing ? 'play' : 'pause'](); }, [s.playing]);
  useEffect(() => { stage.current?.setSpeed(s.speed); }, [s.speed]);
  useEffect(() => { stage.current?.setDebug(s.debug); }, [s.debug]);

  useEffect(() => {
    if (active === null || s.envId === null || seed === null || lanes === null) {
      stage.current?.setTarget(null);
      return;
    }
    try {
      stage.current?.setTarget({
        runId: active.id,
        envId: s.envId,
        seed,
        lane: lanes.lanes[lanes.seeds.indexOf(seed)]!,
        entry: entryFor(parseCatalog(active.meta.envCatalog), s.envId),
        horizon: active.meta.steps,
        live,
        startStep,
        alternates,
      });
    } catch (error) {
      fail(error);
    }
  }, [active, s.envId, seed, lanes, live, startStep, alternates]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const state = store.get();
      const cycleEnv = (direction: 1 | -1): void => {
        const at = state.envId === null ? -1 : envs.indexOf(state.envId);
        const next = envs[(at + direction + envs.length) % envs.length];
        if (next !== undefined) selectEnv(next);
      };
      const cyclePolicy = (): void => {
        const open = POLICY_ORDER.filter((p) => available[p]);
        const next = open[(open.indexOf(state.policy) + 1) % open.length];
        if (next !== undefined) patch({ policy: next });
      };
      const actions: Record<string, () => void> = {
        ' ': () => patch({ playing: !state.playing }),
        k: () => patch({ playing: !state.playing }),
        ArrowLeft: () => { patch({ playing: false }); stage.current?.stepBy(-20 * stride * (event.shiftKey ? 10 : 1)); },
        ArrowRight: () => { patch({ playing: false }); stage.current?.stepBy(20 * stride * (event.shiftKey ? 10 : 1)); },
        '[': () => cycleEnv(-1),
        ']': () => cycleEnv(1),
        g: () => patch({ debug: !state.debug }),
        x: cycleSpeed,
        p: cyclePolicy,
        c: () => patch({ panelOpen: !state.panelOpen }),
        '1': () => patch({ panel: 'reward', panelOpen: true }),
        '2': () => patch({ panel: 'standings', panelOpen: true }),
        '3': () => patch({ panel: 'health', panelOpen: true }),
        f: () => { void (document.fullscreenElement === null ? document.documentElement.requestFullscreen() : document.exitFullscreen()); },
      };
      const action = actions[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      if (action === undefined) return;
      event.preventDefault();
      action();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stride, envs, available]);

  const notice = s.error
    ?? s.stage.note
    ?? (s.runs !== null && s.runId === null ? 'No run to show yet' : null)
    ?? (s.policy !== 'oak' && run !== null && activeId === null ? `No recorded ${POLICY_NAMES[s.policy]} baseline for this environment` : null);

  return (
    <div className="fixed inset-0 overflow-hidden bg-ground">
      <div ref={host} className="absolute inset-0" />
      {notice !== null && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center p-6">
          <div className="glass max-w-md rounded-lg px-5 py-4 font-mono text-sm text-ink">{notice}</div>
        </div>
      )}
      <div className="absolute left-4 top-4 z-20 flex items-center gap-2">
        <select value={s.runId ?? ''} onChange={(event) => selectRun(event.target.value)} className="glass h-9 rounded-lg px-3 font-mono text-xs text-ink">
          {s.runs?.map((run) => <option key={run.id} value={run.id}>{runLabel(run)}</option>)}
        </select>
        <button type="button" onClick={() => stage.current?.resetCamera()} title="Reset camera" className="glass flex h-9 items-center gap-2 rounded-lg px-3 text-xs text-ink-dim hover:bg-white/10 hover:text-ink">⟲<span className="hidden sm:inline">Reset camera</span></button>
      </div>
      {s.panelOpen ? (
        <aside className="glass absolute bottom-20 right-4 top-4 z-20 flex w-[400px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-lg py-2.5">
          <header className="mx-3 mb-2 flex shrink-0 items-center gap-3 border-b border-white/10 pb-2">
            <span className="min-w-0 text-xs text-ink-dim">Michael Dinelle</span>
            <nav aria-label="Michael Dinelle’s links" className="ml-auto flex items-center gap-1">
              {AUTHOR_LINKS.map(({ href, label, Mark }) => (
                <a key={href} href={href} target="_blank" rel="noreferrer" aria-label={`Michael Dinelle on ${label}`} title={`Michael Dinelle on ${label}`} className="flex size-8 items-center justify-center rounded-md text-ink-dim hover:bg-white/10 hover:text-accent">
                  <Mark className="size-3.5" />
                </a>
              ))}
            </nav>
          </header>
          <div className="flex items-center gap-1 px-2 pb-2">
            {(Object.keys(PANELS) as Panel[]).map((panel) => (
              <button key={panel} type="button" onClick={() => patch({ panel })} title={`${PANELS[panel]} (${(Object.keys(PANELS) as Panel[]).indexOf(panel) + 1})`} className={`rounded-md px-2.5 py-1 text-xs ${s.panel === panel ? 'bg-accent/20 text-accent' : 'text-ink-dim hover:bg-white/10 hover:text-ink'}`}>{PANELS[panel]}</button>
            ))}
            <button type="button" onClick={() => patch({ panelOpen: false })} aria-label="Hide the side panel" title="Hide the side panel (c)" className="ml-auto flex size-8 items-center justify-center rounded-md text-ink-dim hover:bg-white/10 hover:text-ink"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M15 3v18" /></svg></button>
          </div>
          {s.panel === 'reward' && active !== null && s.envId !== null && seed !== null && <RewardView stage={stage} runId={active.id} envId={s.envId} seed={seed} compare={compare} />}
          {s.panel === 'standings' && <StandingsView />}
          {s.panel === 'health' && active !== null && s.envId !== null && <HealthView runId={active.id} envId={s.envId} />}
        </aside>
      ) : (
        <button type="button" onClick={() => patch({ panelOpen: true })} aria-label="Show the side panel" title="Show the side panel (c)" className="glass absolute right-4 top-4 z-20 flex size-9 items-center justify-center rounded-lg text-ink-dim hover:bg-white/10 hover:text-ink"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M15 3v18" /></svg></button>
      )}
      <div className="absolute inset-x-0 bottom-4 z-30 grid grid-cols-[1fr_minmax(0,760px)_1fr] items-end gap-3 px-4">
        <div className="flex flex-col items-start gap-3">
          {s.envId === 'JellyBeanWorld-cont-v0' && <JbwLegend />}
          <EnvPicker envs={envs} run={run} seed={seed} />
        </div>
        <PlaybackBar stage={stage} stride={stride} live={live} />
        <div className="justify-self-end"><PolicySwitch available={available} /></div>
      </div>
    </div>
  );
}
