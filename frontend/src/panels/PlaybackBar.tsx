import { useEffect, useRef, type MouseEvent, type ReactNode, type RefObject } from 'react';
import type { Policy } from '../data/layout';
import { formatClock, formatCount, formatStep } from '../format';
import { cycleSpeed, patch, selectPolicy, store } from '../store';
import { frameListeners, type Frame, type Stage } from '../theater/stage';
import { POLICY_NAMES, POLICY_ORDER } from './charts';

function BarButton({ on, disabled, title, onClick, children }: { on?: boolean; disabled?: boolean; title: string; onClick: (event: MouseEvent<HTMLButtonElement>) => void; children: ReactNode }) {
  return (
    <button
      type="button"
      disabled={disabled}
      aria-label={title}
      title={title}
      onClick={onClick}
      className={`flex h-7 min-w-7 items-center justify-center rounded-md px-1.5 font-mono text-xs active:bg-white/15 ${disabled === true ? 'text-ink-faint' : on === true ? 'bg-accent/20 text-accent' : 'text-ink hover:bg-white/10'}`}
    >
      {children}
    </button>
  );
}

const POLICY_ON = { oak: 'bg-oak/25 ring-oak/50', ppo: 'bg-ppo/25 ring-ppo/50', random: 'bg-random/25 ring-random/50', oracle: 'bg-oracle/25 ring-oracle/50' } as const;

export function PolicySwitch({ available }: { available: Record<Policy, boolean> }) {
  const { policy } = store.use();
  return (
    <div className="glass flex h-12 items-center gap-1 rounded-lg px-2">
      <span className="px-1.5 text-2xs uppercase tracking-label">policy</span>
      {POLICY_ORDER.map((p) => {
        const absent = !available[p];
        return (
          <button
            key={p}
            type="button"
            aria-disabled={absent}
            aria-pressed={policy === p}
            title={absent ? `No recorded ${POLICY_NAMES[p]} baseline for this environment` : `Watch the ${POLICY_NAMES[p]} fleet (p cycles)`}
            onClick={absent ? undefined : () => selectPolicy(p)}
            className={`flex h-8 items-center rounded-md px-3 text-xs font-medium ${absent ? 'cursor-default text-ink-faint' : policy === p ? `text-ink ring-1 ring-inset ${POLICY_ON[p]}` : 'text-ink-dim hover:bg-white/10 hover:text-ink'}`}
          >
            {POLICY_NAMES[p]}
          </button>
        );
      })}
    </div>
  );
}

export function PlaybackBar({ stage, live }: { stage: RefObject<Stage | null>; live: boolean }) {
  const { playing, speed, debug, stage: meta } = store.use();
  const scrubbing = useRef(false);
  const lastReadoutAt = useRef(0);
  const fillRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const readoutRef = useRef<HTMLDivElement | null>(null);
  const totalRef = useRef(meta.totalSteps);
  totalRef.current = meta.totalSteps;

  useEffect(() => {
    const listen = (frame: Frame): void => {
      const step = Math.max(0, frame.step);
      const ratio = totalRef.current > 0 ? Math.min(1, step / totalRef.current) : 0;
      if (fillRef.current !== null) fillRef.current.style.transform = `translateY(-50%) scaleX(${ratio.toFixed(5)})`;
      if (!scrubbing.current && inputRef.current !== null) inputRef.current.value = String(frame.requestedStep);
      const now = performance.now();
      if (now - lastReadoutAt.current < 150 || readoutRef.current === null) return;
      lastReadoutAt.current = now;
      readoutRef.current.textContent = `step ${frame.step < 0 ? 'start' : formatCount(frame.step)} of ${formatStep(totalRef.current)} · ${formatClock(frame.simTime)}${frame.requestedStep !== frame.step ? ` · waiting for ${formatStep(frame.requestedStep)}` : ''}`;
    };
    frameListeners.add(listen);
    return () => { frameListeners.delete(listen); };
  }, []);

  const idle = meta.status === 'empty' || meta.status === 'unsupported' || meta.status === 'error';
  const jump = 20;
  const ticks = meta.ticks;
  const busiest = ticks.reduce((max, tick) => Math.max(max, tick.count), 1);
  const step = (delta: number): void => {
    patch({ playing: false });
    stage.current?.stepBy(delta);
  };

  return (
    <div className={`glass flex h-12 w-[760px] max-w-full items-center gap-1 rounded-lg px-2 ${idle ? 'opacity-80' : ''}`}>
      <BarButton title={playing ? 'Pause (space)' : 'Play (space)'} disabled={idle} onClick={() => patch({ playing: !playing })}>{playing ? '❚❚' : '▶'}</BarButton>
      <BarButton title={`Back ${formatStep(jump)} steps (← · shift for ×10)`} disabled={idle} onClick={event => step(-jump * (event.shiftKey ? 10 : 1))}>‹</BarButton>
      <BarButton title={`Ahead ${formatStep(jump)} steps (→ · shift for ×10)`} disabled={idle} onClick={event => step(jump * (event.shiftKey ? 10 : 1))}>›</BarButton>
      <div className="relative mx-2 h-7 min-w-32 flex-1 rounded-md hover:bg-white/5 has-[:focus-visible]:ring-1 has-[:focus-visible]:ring-accent" title="Drag anywhere in the run">
        <div className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-white/15" />
        <div className="absolute inset-x-0 top-1/2 h-[5px] origin-left rounded-full bg-good/30" title={`Available through step ${formatCount(Math.max(0, meta.frameCount - 1))}`} style={{ transform: `translateY(-50%) scaleX(${meta.totalSteps ? (meta.frameCount - 1) / meta.totalSteps : 0})` }} />
        <div ref={fillRef} className="absolute inset-x-0 top-1/2 h-[3px] origin-left rounded-full bg-accent" style={{ transform: 'translateY(-50%) scaleX(0)', willChange: 'transform' }} />
        <div className="pointer-events-none absolute inset-0">
          {ticks.map((tick, i) => {
            const density = Math.sqrt(tick.count / busiest);
            const height = 6 + 8 * density;
            return (
              <span
                key={i}
                title={tick.label}
                className="absolute top-1/2 w-px -translate-x-1/2 bg-seed"
                style={{ left: `${(tick.position * 100).toFixed(3)}%`, opacity: (0.22 + 0.62 * density).toFixed(3), height: `${height.toFixed(1)}px`, marginTop: `${(-height / 2).toFixed(1)}px` }}
              />
            );
          })}
        </div>
        <input
          ref={inputRef}
          type="range"
          min="0"
          max={String(Math.max(1, meta.totalSteps))}
          step="1"
          defaultValue="0"
          disabled={idle}
          aria-label="Drag anywhere in the run"
          className="absolute inset-0 w-full cursor-pointer appearance-none bg-transparent"
          onPointerDown={() => { scrubbing.current = true; }}
          onPointerUp={() => { scrubbing.current = false; }}
          onPointerCancel={() => { scrubbing.current = false; }}
          onInput={(event) => stage.current?.seekStep(Number(event.currentTarget.value))}
        />
      </div>
      <div ref={readoutRef} className="min-w-[19ch] whitespace-nowrap font-mono text-2xs tabular-nums text-ink">{meta.status === 'loading' ? 'Loading scene and replay…' : idle ? meta.status : ''}</div>
      <div className="mx-0.5 h-5 w-px bg-white/10" />
      <BarButton title={`Playing at ${speed}× the world's own pace — click for the next speed (x)`} onClick={cycleSpeed}>
        <span className="inline-block w-9 text-center tabular-nums">×{speed}</span>
      </BarButton>
      <div className="mx-0.5 h-5 w-px bg-white/10" />
      <BarButton on={debug} title="Show what the world is really doing (g)" onClick={() => patch({ debug: !debug })}>◎</BarButton>
      {(meta.status === 'loading' || meta.status === 'buffering' || meta.identical === null && !idle) && <span className="mx-1 whitespace-nowrap font-mono text-2xs text-ink-dim">{meta.status === 'buffering' ? 'Buffering' : meta.status === 'loading' ? 'Loading' : 'Generating'} {meta.totalSteps ? Math.floor(Math.max(0, meta.frameCount - 1) / meta.totalSteps * 100) : 0}%</span>}
      {live && <span className="mx-1 size-1.5 motion-safe:animate-pulse rounded-full bg-good" title="Being recorded right now" />}
      {meta.identical !== null && (
        <span className={`mx-1 whitespace-nowrap rounded-sm border px-1 font-mono text-2xs leading-4 ${meta.identical ? 'border-good/40 bg-good/10 text-good' : 'border-crit/40 bg-crit/10 text-crit'}`} title={meta.identical ? 'The replayed lane reproduces the recorded reward stream bit for bit' : 'The replayed lane does not reproduce the recorded reward stream: the engine has changed since this recording'}>
          {meta.identical ? 'replay exact' : 'replay differs'}
        </span>
      )}
    </div>
  );
}
