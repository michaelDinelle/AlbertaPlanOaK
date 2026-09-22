import { NO_EVENTS, type EnvDecoder } from '../types';

function accessBinomial(trials: number, successes: number, rate: number): number {
  let combinations = 1;
  for (let i = 1; i <= successes; i++) combinations = combinations * (trials - successes + i) / i;
  return combinations * rate ** successes * (1 - rate) ** (trials - successes);
}

function accessExpected(value: Float64Array, servers: number, free: number, rate: number): number {
  let expected = 0;
  for (let freed = 0; freed <= servers - free; freed++) expected += accessBinomial(servers - free, freed, rate) * value[free + freed]!;
  return expected;
}

export function accessPolicy(servers: number, rewards: readonly number[], rate: number, iterations: number, tolerance: number): Uint8Array<ArrayBuffer> {
  const value = new Float64Array(servers + 1), next = new Float64Array(servers + 1);
  for (let iteration = 0; iteration < iterations; iteration++) {
    for (let free = 0; free <= servers; free++) {
      const reject = accessExpected(value, servers, free, rate);
      let total = 0;
      for (const reward of rewards) { const accept = free > 0 ? reward + accessExpected(value, servers, free - 1, rate) : reject; total += Math.max(accept, reject) / rewards.length; }
      next[free] = total;
    }
    const reference = next[0]!;
    let delta = 0;
    for (let free = 0; free <= servers; free++) { const updated = next[free]! - reference; delta = Math.max(delta, Math.abs(updated - value[free]!)); value[free] = updated; }
    if (delta < tolerance) break;
  }
  const policy = new Uint8Array(rewards.length * (servers + 1));
  rewards.forEach((reward, p) => { for (let free = 1; free <= servers; free++) policy[p * (servers + 1) + free] = reward + accessExpected(value, servers, free - 1, rate) > accessExpected(value, servers, free, rate) ? 1 : 0; });
  return policy;
}

export function accessFreeRate(drift: boolean, step: number, period: number): number {
  return drift ? 0.04 + 0.02 * Math.sin(2 * Math.PI * (step % period) / period) : 0.06;
}


export const ACCESS_FREE = 0;
export const ACCESS_PRIORITY = 1;
export const ACCESS_ACTION = 2;

export const accessDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[ACCESS_FREE] = frame.words[0]!; out[ACCESS_PRIORITY] = frame.words[1]!; out[ACCESS_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
