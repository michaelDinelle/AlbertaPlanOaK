import type { FrameUpdate } from '@kit/types';
import { corridorKeyX, corridorKeyY, CORRIDOR_X, CORRIDOR_Y } from '@sim/envs/corridor';
import type { GridCell } from '@sim/types';

export const ACTIONS = ['↑ north', '↓ south', '← west', 'east →'] as const;

export function eventSteps(frame: FrameUpdate, out: number[]): readonly number[] {
  out.length = 0;
  if (frame.events.length === 0 || !frame.prev) return out;
  const log = frame.eventLog, after = frame.prev.step, upTo = frame.state.step;
  for (let b = Math.floor((after + 1) / log.blockFrames); b <= Math.floor(upTo / log.blockFrames); b++) {
    const block = log.blocks[b];
    if (block) for (let i = 0; i < block.steps.length; i++) if (block.steps[i]! > after && block.steps[i]! <= upTo) out.push(block.steps[i]!);
  }
  if (out.length !== frame.events.length) throw new Error('Frame events and the event log disagree');
  return out;
}

export function keyCell(pose: Float32Array, index: number, out: GridCell): GridCell {
  out.x = pose[corridorKeyX(index)]!;
  out.y = pose[corridorKeyY(index)]!;
  return out;
}

export function agentCell(pose: Float32Array, out: GridCell): GridCell {
  out.x = pose[CORRIDOR_X]!;
  out.y = pose[CORRIDOR_Y]!;
  return out;
}

export function accentNamed(accents: Record<string, number>, name: string): number {
  const accent = accents[name];
  if (accent === undefined) throw new Error(`${name} has no accent`);
  return accent;
}
