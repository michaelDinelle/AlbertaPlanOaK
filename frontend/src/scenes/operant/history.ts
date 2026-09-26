import type { FrameUpdate } from '@kit/types';
import { operantLight, operantPress } from '@sim/envs/operant';
import { SIM_EVENT_TYPES, type EventStream } from '@sim/types';

const RING = 8192;
const GOAL = SIM_EVENT_TYPES.indexOf('goal');
const HARM = SIM_EVENT_TYPES.indexOf('harm');

export const PRESS = 1;
export const LIGHT = 2;
export const FOOD = 4;
export const COST = 8;

export interface Totals { presses: number; pellets: number }

export class History {
  private readonly flags = new Uint8Array(RING);
  private readonly rewards = new Float32Array(RING);
  private readonly held = new Int32Array(RING).fill(-1);
  private readonly pressesBefore: number[] = [0];
  private readonly pelletsBefore: number[] = [0];
  private frame: FrameUpdate | null = null;
  private stream: EventStream | null = null;

  bind(frame: FrameUpdate): void {
    this.frame = frame;
    if (frame.eventLog === this.stream) return;
    this.stream = frame.eventLog;
    this.held.fill(-1);
    this.pressesBefore.length = 1;
    this.pelletsBefore.length = 1;
  }

  at(s: number): number {
    if (s < 0 || this.frame === null) return -1;
    const slot = s % RING;
    if (this.held[slot] === s) return this.flags[slot]!;
    const f = this.frame.frameAt(s);
    if (f.step !== s) return -1;
    const press = operantPress(f);
    if ((press !== 0) !== (f.reward !== 0)) throw new Error(`Operant step ${s} pressed ${press} yet paid ${f.reward}; every press pays and nothing else does`);
    const bits = (press ? PRESS : 0) | (operantLight(f) ? LIGHT : 0) | (f.reward > 0 ? FOOD : f.reward < 0 ? COST : 0);
    this.flags[slot] = bits;
    this.rewards[slot] = f.reward;
    this.held[slot] = s;
    return bits;
  }

  has(s: number, bit: number): boolean {
    const bits = this.at(s);
    return bits >= 0 && (bits & bit) !== 0;
  }

  reward(s: number): number {
    return this.at(s) < 0 ? NaN : this.rewards[s % RING]!;
  }

  totals(stream: EventStream, k: number, out: Totals): Totals {
    out.presses = 0;
    out.pellets = 0;
    if (k < 0) return out;
    const size = stream.blockFrames, block = Math.floor(k / size);
    for (let b = this.pressesBefore.length - 1; b < block; b++) {
      const events = stream.blocks[b];
      if (!events) throw new Error(`Operant events for steps from ${b * size} were not derived before step ${k} was shown`);
      let presses = 0, pellets = 0;
      for (let e = 0; e < events.kinds.length; e++) {
        const kind = events.kinds[e];
        if (kind === GOAL) { presses++; pellets++; }
        else if (kind === HARM) presses++;
      }
      this.pressesBefore.push(this.pressesBefore[b]! + presses);
      this.pelletsBefore.push(this.pelletsBefore[b]! + pellets);
    }
    const events = stream.blocks[block];
    if (!events) throw new Error(`Operant events for step ${k} were not derived before it was shown`);
    out.presses = this.pressesBefore[block]!;
    out.pellets = this.pelletsBefore[block]!;
    for (let e = 0; e < events.steps.length && events.steps[e]! <= k; e++) {
      const kind = events.kinds[e];
      if (kind === GOAL) { out.presses++; out.pellets++; }
      else if (kind === HARM) out.presses++;
    }
    return out;
  }
}
