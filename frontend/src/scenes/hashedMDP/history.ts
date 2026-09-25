import type { EnvFrame } from '@sim/types';

export interface Edge {
  from: number;
  to: number;
  action: number;
  reward: number;
  weight: number;
  last: number;
}

export interface Loop {
  length: number;
  reward: number;
}

export class History {
  private readonly states: Int32Array;
  private readonly actions: Uint8Array;
  private readonly rewards: Float32Array;
  private readonly held: Int32Array;

  constructor(readonly size: number, private readonly actionCount: number) {
    this.states = new Int32Array(size);
    this.actions = new Uint8Array(size);
    this.rewards = new Float32Array(size);
    this.held = new Int32Array(size).fill(-1);
  }

  has(s: number): boolean {
    return s >= 0 && this.held[s % this.size] === s;
  }

  moved(s: number): boolean {
    return s >= 1 && this.has(s) && this.has(s - 1);
  }

  state(s: number): number {
    return this.states[s % this.size]!;
  }

  action(s: number): number {
    return this.actions[s % this.size]!;
  }

  reward(s: number): number {
    return this.rewards[s % this.size]!;
  }

  complete(step: number): boolean {
    for (let s = Math.max(0, step - this.size + 1); s <= step; s++) if (!this.has(s)) return false;
    return true;
  }

  fill(step: number, at: (s: number) => EnvFrame | null): boolean {
    let changed = false;
    for (let s = step; s >= Math.max(0, step - this.size + 1); s--) {
      const slot = s % this.size;
      if (this.held[slot] === s) continue;
      const frame = at(s);
      if (!frame) break;
      this.states[slot] = frame.words[0]!;
      this.actions[slot] = (frame.actions[0]! >>> 0) % this.actionCount;
      this.rewards[slot] = frame.reward;
      this.held[slot] = s;
      changed = true;
    }
    return changed;
  }

  edges(step: number, span: number, tau: number, out: Map<number, Edge>): void {
    out.clear();
    for (let s = step; s > step - span && s >= 1; s--) {
      if (!this.moved(s)) continue;
      const from = this.state(s - 1), action = this.action(s), key = from * this.actionCount + action, weight = Math.exp(-(step - s) / tau);
      const edge = out.get(key);
      if (edge) edge.weight += weight;
      else out.set(key, { from, to: this.state(s), action, reward: this.reward(s), weight, last: s });
    }
  }

  loop(step: number, longest: number): Loop | null {
    for (let length = 1; length <= longest; length++) {
      let periodic = this.moved(step - 3 * length + 1);
      for (let i = 0; periodic && i < 2 * length; i++) {
        const a = step - i, b = a - length;
        periodic = this.moved(a) && this.moved(b) && this.state(a) === this.state(b) && this.action(a) === this.action(b);
      }
      if (!periodic) continue;
      let reward = 0;
      for (let i = 0; i < length; i++) reward += this.reward(step - i);
      return { length, reward };
    }
    return null;
  }
}
