
import { philoxDraw, STREAM_GRIDWORLD, uniformIndex } from '../philox';
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const MKC_W = 17;
export const MKC_H = 7;
export const MKC_ROOM_CELLS = 15;
export const MKC_ROOM_BASE: readonly number[] = Object.freeze([1, 5, 9, 13]);
export const MKC_DOORS: readonly GridCell[] = Object.freeze([
  { x: 4, y: 3 },
  { x: 8, y: 3 },
  { x: 12, y: 3 },
]);
export const MKC_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const MKC_KEYS: readonly GridCell[] = Object.freeze([
  { x: 3, y: 5 },
  { x: 7, y: 5 },
  { x: 11, y: 5 },
]);
export const MKC_GOAL: GridCell = Object.freeze({ x: 15, y: 1 });

export interface MultiKeyCorridorState {
  x: number;
  y: number;
  gx: number;
  gy: number;
  keys: GridCell[];
  hasKey: [boolean, boolean, boolean];
  doorOpen: [boolean, boolean, boolean];
  flags: number;
  respawnIdx: number;
  poseAgrees: boolean;
}

export interface MultiKeyCorridorStatic {
  seed: number;
  walls: Uint8Array;
}

export function mkcIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 16 || y <= 0 || y >= 6) return true;
  if ((x === 4 || x === 8 || x === 12) && y !== 3) return true;
  return false;
}

export function mkcWalls(): Uint8Array {
  const walls = new Uint8Array(MKC_W * MKC_H);
  for (let y = 0; y < MKC_H; y++) {
    for (let x = 0; x < MKC_W; x++) walls[y * MKC_W + x] = mkcIsWall(x, y) ? 1 : 0;
  }
  return walls;
}

export function mkcRoomCells(room: number): GridCell[] {
  const base = MKC_ROOM_BASE[room] ?? 1;
  const out: GridCell[] = [];
  for (let i = 0; i < MKC_ROOM_CELLS; i++) {
    out.push({ x: base + (i % 3), y: 1 + Math.floor(i / 3) });
  }
  return out;
}

function roomCell(room: number, idx: number): GridCell {
  return { x: (MKC_ROOM_BASE[room] ?? 1) + (idx % 3), y: 1 + Math.floor(idx / 3) };
}

const MOVES: readonly (readonly [number, number])[] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

interface Replay {
  x: number;
  y: number;
  keys: GridCell[];
  gx: number;
  gy: number;
  flags: number;
  respawnIdx: number;
}

export function mkcStep(seed: number, s: Replay, action: number): number {
  const mv = MOVES[action & 3] ?? MOVES[3]!;
  let nx = s.x + mv[0];
  let ny = s.y + mv[1];
  if (mkcIsWall(nx, ny)) {
    nx = s.x;
    ny = s.y;
  } else if (ny === 3 && (nx === 4 || nx === 8 || nx === 12)) {
    const j = (nx >> 2) - 1;
    if ((s.flags & (8 << j)) === 0) {
      if ((s.flags & (1 << j)) !== 0) s.flags |= 8 << j;
      else {
        nx = s.x;
        ny = s.y;
      }
    }
  }
  for (let j = 0; j < 3; j++) {
    const k = s.keys[j];
    if (k && (s.flags & (1 << j)) === 0 && nx === k.x && ny === k.y) s.flags |= 1 << j;
  }
  let done = 0;
  if (nx === s.gx && ny === s.gy) {
    done = 1;
    const ri = s.respawnIdx;
    const raw = philoxDraw(seed, STREAM_GRIDWORLD, ri, 0);
    const raw2 = philoxDraw(seed, STREAM_GRIDWORLD, ri, 1);
    const agent = roomCell(0, uniformIndex(raw[0], MKC_ROOM_CELLS));
    nx = agent.x;
    ny = agent.y;
    s.keys = [
      roomCell(0, uniformIndex(raw[1], MKC_ROOM_CELLS)),
      roomCell(1, uniformIndex(raw[2], MKC_ROOM_CELLS)),
      roomCell(2, uniformIndex(raw[3], MKC_ROOM_CELLS)),
    ];
    const goal = roomCell(3, uniformIndex(raw2[0], MKC_ROOM_CELLS));
    s.gx = goal.x;
    s.gy = goal.y;
    s.flags = 0;
    s.respawnIdx = ri + 1;
  }
  s.x = nx;
  s.y = ny;
  return done;
}

function bits3(flags: number, shift: number): [boolean, boolean, boolean] {
  return [
    ((flags >> shift) & 1) !== 0,
    ((flags >> (shift + 1)) & 1) !== 0,
    ((flags >> (shift + 2)) & 1) !== 0,
  ];
}

function snapshot(s: Replay, poseAgrees: boolean): MultiKeyCorridorState {
  return {
    x: s.x,
    y: s.y,
    gx: s.gx,
    gy: s.gy,
    keys: s.keys.map((k) => ({ ...k })),
    hasKey: bits3(s.flags, 0),
    doorOpen: bits3(s.flags, 3),
    flags: s.flags,
    respawnIdx: s.respawnIdx,
    poseAgrees,
  };
}

function toReplay(st: MultiKeyCorridorState): Replay {
  return {
    x: st.x,
    y: st.y,
    keys: st.keys.map((k) => ({ ...k })),
    gx: st.gx,
    gy: st.gy,
    flags: st.flags,
    respawnIdx: st.respawnIdx,
  };
}

export const multiKeyCorridorDecoder: EnvDecoder<MultiKeyCorridorState, MultiKeyCorridorStatic> = {
  envId: 'MultiKeyCorridor-cont-v0',
  needsReplay: true,

  initStatic(seed: number): MultiKeyCorridorStatic {
    return { seed, walls: mkcWalls() };
  },

  geometry(st: MultiKeyCorridorStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: MKC_W,
        h: MKC_H,
        walls: st.walls,
        goal: { ...MKC_GOAL },
        doors: MKC_DOORS.map((c) => ({ ...c })),
        keys: MKC_KEYS.map((c) => ({ ...c })),
        hallways: MKC_DOORS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame<MultiKeyCorridorState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = MKC_START.x;
    pose[1] = MKC_START.y;
    pose[2] = MKC_GOAL.x;
    pose[3] = MKC_GOAL.y;
    const s: Replay = {
      x: MKC_START.x,
      y: MKC_START.y,
      keys: MKC_KEYS.map((k) => ({ ...k })),
      gx: MKC_GOAL.x,
      gy: MKC_GOAL.y,
      flags: 0,
      respawnIdx: 0,
    };
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: snapshot(s, true),
    };
  },

  decodeFrame(
    i: number,
    raw16: Float32Array,
    st: MultiKeyCorridorStatic,
    prev: EnvFrame<MultiKeyCorridorState>,
  ): EnvFrame<MultiKeyCorridorState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    const ctrl = raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN);
    const s = toReplay(prev.state);
    mkcStep(st.seed, s, Math.trunc(ctrl[0] ?? 0));
    const agrees =
      (pose[0] ?? 0) === s.x &&
      (pose[1] ?? 0) === s.y &&
      (pose[2] ?? 0) === s.gx &&
      (pose[3] ?? 0) === s.gy &&
      (pose[4] ?? 0) === (s.flags & 7) &&
      (pose[5] ?? 0) === ((s.flags >> 3) & 7);
    return {
      step: i,
      ctrl,
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: snapshot(s, agrees),
    };
  },

  deriveEvents(
    prev: EnvFrame<MultiKeyCorridorState>,
    next: EnvFrame<MultiKeyCorridorState>,
  ): SimEvent[] {
    const out: SimEvent[] = [];
    const goal = next.done !== 0;
    if (goal) out.push({ type: 'goal' });
    for (let j = 0; j < 3; j++) {
      if (!prev.state.hasKey[j] && next.state.hasKey[j]) out.push({ type: 'key', index: j });
    }
    for (let j = 0; j < 3; j++) {
      if (prev.state.doorOpen[j] !== next.state.doorOpen[j]) {
        out.push({ type: 'door', index: j, open: next.state.doorOpen[j] ?? false });
      }
    }
    if (goal) {
      out.push({ type: 'teleport' });
      out.push({ type: 'done' });
    }
    return out.length === 0 ? NO_EVENTS : out;
  },
};
