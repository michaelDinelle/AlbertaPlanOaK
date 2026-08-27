import { Panel } from '../glass/Panel';
import { useOrchardReadout } from '../state/orchard';
import { STR } from '../strings';
import { TXT } from './charts';

const RING: readonly number[] = [7, 0, 1, 6, -1, 2, 5, 4, 3];
const PEAK = 2.04;
const SCENT_DOTS: readonly string[] = ['var(--color-seed)', 'var(--color-good)', 'var(--color-ppo)'];
const ITEMS: readonly string[] = [
  STR.orchard.banana,
  STR.orchard.onion,
  STR.orchard.jelly,
  STR.orchard.truffle,
];

export function OrchardReadout() {
  const readout = useOrchardReadout();
  if (!readout.live) return null;
  const paying = readout.polarity > 0 ? STR.orchard.jelly : STR.orchard.onion;
  const costing = readout.polarity > 0 ? STR.orchard.onion : STR.orchard.jelly;
  return (
    <Panel className="w-[232px] px-3 py-2.5">
      <div className={TXT.label}>{STR.orchard.senses}</div>
      <div className="mt-2 flex items-start gap-3">
        <div className="grid grid-cols-3 gap-0.5">
          {RING.map((slot, i) => {
            if (slot < 0) {
              return (
                <div
                  key={i}
                  className="size-5 rounded-xs border border-accent/60 bg-accent/25"
                  title={STR.orchard.agent}
                />
              );
            }
            const value = readout.sensed[slot] ?? 0;
            return (
              <div
                key={i}
                className="size-5 rounded-xs border border-white/10"
                style={{
                  background: `color-mix(in oklab, var(--color-env-jelly-bean-world) ${Math.round(
                    Math.min(100, (100 * value) / PEAK),
                  )}%, transparent)`,
                }}
                title={value.toFixed(2)}
              />
            );
          })}
        </div>
        <div className="flex-1">
          <div className={TXT.label}>{STR.orchard.scent}</div>
          <div className="mt-1 flex flex-col gap-0.5">
            {readout.scent.map((v, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <span
                  className="size-1.5 rounded-full"
                  style={{ background: SCENT_DOTS[i] ?? 'var(--color-ink-faint)' }}
                />
                <span className={TXT.data}>{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-2.5 border-t border-white/10 pt-2">
        <div className={TXT.label}>{STR.orchard.gathered}</div>
        <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5">
          {readout.inventory.map((n, i) => (
            <div key={i} className="flex items-baseline justify-between">
              <span className="text-2xs text-ink-dim">{ITEMS[i]}</span>
              <span className={TXT.data}>{n}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-2.5 border-t border-white/10 pt-2">
        <div className={TXT.label}>{STR.orchard.paying}</div>
        <div className="mt-1 flex items-baseline justify-between">
          <span className="text-xs text-good">{paying}</span>
          <span className="text-xs text-bad">{costing}</span>
        </div>
        <div className="mt-1.5 h-0.5 w-full rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-seed"
            style={{ width: `${Math.round(100 * Math.min(1, Math.max(0, readout.harvest)))}%` }}
            title={STR.orchard.harvest}
          />
        </div>
      </div>
    </Panel>
  );
}
