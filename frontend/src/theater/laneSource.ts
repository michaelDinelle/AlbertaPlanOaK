import type { EnvId } from '@shared/envs';
import type { PlaybackSource } from '@kit/playback';
import type { EnvEntry } from '@sim/catalog';
import { readLaneGeometry, type LaneGeometry } from '@sim/geometry';
import { NO_EVENTS } from '@sim/events';
import type { EnvFrame, EnvGeometry, GridCell, SimEvent } from '@sim/types';
import type { LaneStream } from './laneStream';

const POSE_LEN = 9;
const CTRL_LEN = 6;

function gridFrom(entry: EnvEntry, geometry: LaneGeometry): EnvGeometry {
  if (!entry.isGrid) return { kind: entry.ctrlBox === 6 ? 'cheetah' : 'classic' };
  const hallways: GridCell[] = [];
  for (let y = 0; y < geometry.h; y++) {
    for (let x = 0; x < geometry.w; x++) {
      if (geometry.walls[y * geometry.w + x] === 1) continue;
      const solidRow = x > 0 && x < geometry.w - 1
        && geometry.walls[y * geometry.w + x - 1] === 1
        && geometry.walls[y * geometry.w + x + 1] === 1;
      const solidCol = y > 0 && y < geometry.h - 1
        && geometry.walls[(y - 1) * geometry.w + x] === 1
        && geometry.walls[(y + 1) * geometry.w + x] === 1;
      if (solidRow || solidCol) hallways.push({ x, y });
    }
  }
  return {
    kind: 'grid',
    grid: {
      w: geometry.w,
      h: geometry.h,
      walls: geometry.walls,
      actionMask: geometry.actionMask,
      hallways,
    },
  };
}

function deriveEvents(entry: EnvEntry, prev: Float32Array, next: Float32Array, done: boolean): SimEvent[] {
  const out: SimEvent[] = [];
  const at = entry.slotAt;
  if (done) out.push({ type: 'goal' });

  const carry = at['carry'];
  if (carry !== undefined && (prev[carry] ?? 0) === 0 && (next[carry] ?? 0) !== 0) {
    out.push({ type: 'key', index: 0 });
  }
  const tokens = at['tokens'];
  if (tokens !== undefined) {
    const a = Math.round(prev[tokens] ?? 0);
    const b = Math.round(next[tokens] ?? 0);
    for (let i = 0; i < 4; i++) {
      if (((a >> i) & 1) !== ((b >> i) & 1)) out.push({ type: 'key', index: i });
    }
  }
  const gates = at['gates'];
  if (gates !== undefined) {
    const a = Math.round(prev[gates] ?? 0);
    const b = Math.round(next[gates] ?? 0);
    for (let i = 0; i < 4; i++) {
      const wasOpen = ((a >> i) & 1) === 1;
      const isOpen = ((b >> i) & 1) === 1;
      if (wasOpen !== isOpen) out.push({ type: 'door', index: i, open: isOpen });
    }
  }
  const cue = at['cue'];
  if (cue !== undefined && done) {
    out.push({ type: 'cue', value: Math.round(next[cue] ?? 0) === 1 ? 1 : 0 });
  }

  if (entry.isGrid) {
    const x = at['x'];
    const y = at['y'];
    const dx = x === undefined ? 0 : Math.abs((next[x] ?? 0) - (prev[x] ?? 0));
    const dy = y === undefined ? 0 : Math.abs((next[y] ?? 0) - (prev[y] ?? 0));
    if (dx > 1.5 || dy > 1.5) out.push(done ? { type: 'teleport' } : { type: 'resurface' });
  }
  if (done) out.push({ type: 'done' });
  return out.length === 0 ? NO_EVENTS : out;
}

export interface LaneSourceOptions {
  stream: LaneStream;
  lane: number;
  seed: number;
  totalSteps: number;
  geometryBytes: Uint8Array;
}

export function makeLaneSource(opts: LaneSourceOptions): PlaybackSource {
  const entry = opts.stream.entry;
  const dv = new DataView(
    opts.geometryBytes.buffer,
    opts.geometryBytes.byteOffset,
    opts.geometryBytes.byteLength,
  );
  const laneGeometry = readLaneGeometry(dv, 0, entry);
  const geometry = gridFrom(entry, laneGeometry);

  const pose = new Float32Array(POSE_LEN);
  const prevPose = new Float32Array(POSE_LEN);
  const ctrl = new Float32Array(CTRL_LEN);
  const frame: EnvFrame<unknown> = { step: -1, ctrl, done: 0, pose, state: null };

  return {
    envId: entry.envId as EnvId,
    seed: opts.seed,
    controlHz: entry.controlHz,
    discretePose: entry.isGrid,
    geometry,
    get frameCount() {
      return opts.totalSteps + 1;
    },
    get complete() {
      return true;
    },
    frameAt(i) {
      const step = Math.max(0, Math.min(opts.totalSteps, Math.floor(i))) - 1;
      if (step < 0) {
        pose.fill(0);
        frame.step = -1;
        frame.done = 0;
        return frame;
      }
      opts.stream.pose(opts.lane, step, pose);
      frame.step = step;
      frame.done = 0;
      return frame;
    },
    eventsAt(i) {
      const step = Math.floor(i) - 1;
      if (step <= 0) return NO_EVENTS;
      if (!opts.stream.pose(opts.lane, step - 1, prevPose)) return NO_EVENTS;
      if (!opts.stream.pose(opts.lane, step, pose)) return NO_EVENTS;
      return deriveEvents(entry, prevPose, pose, false);
    },
    eventEntries() {
      return new Map<number, SimEvent[]>().entries();
    },
  };
}
