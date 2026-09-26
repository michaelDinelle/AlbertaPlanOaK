import * as THREE from 'three/webgpu';

export const TRACE = 0.15;
export const BAND = 0.42;
export const WIRE = 0.05;
export const PAD = 0.24;
export const PIN = 0.12;
export const LED = 0.19;
export const LAMP = 0.3;
export const LOG = 100;
export const GATE = new THREE.Vector2(0.42, 0.3);
export const EQUAL = new THREE.Vector2(0.3, 0.24);
export const AGENT = new THREE.Vector2(0.8, 0.8);
export const RESISTOR = new THREE.Vector2(0.26, 0.1);

const TURN = 2;
const REACH = 12.4;
const CELL_MAX = 1.3;
const ROW_MAX = 16;

export type Point = readonly [number, number];

export interface Station { x: number; z: number; tx: number; tz: number }

export interface Layout {
  delay: number;
  cells: number;
  cell: number;
  radius: number;
  rows: number;
  run: number;
  last: number;
  x0: number;
  z0: number;
  lead: number;
  pad: THREE.Vector2;
  coin: THREE.Vector2;
  taps: readonly [THREE.Vector2, THREE.Vector2];
  gate: THREE.Vector2;
  truth: THREE.Vector2;
  equal: THREE.Vector2;
  call: THREE.Vector2;
  verdict: THREE.Vector2;
  agent: THREE.Vector2;
  resistor: THREE.Vector2;
  ground: THREE.Vector2;
  log: { left: number; right: number; z: number; pitch: number };
  wires: {
    feed: Point[];
    probe: Point[];
    output: Point[];
    stubs: readonly [Point[], Point[]];
    truth: Point[];
    judged: Point[];
    called: Point[];
    verdict: Point[];
  };
  board: THREE.Box2;
}

function shape(cells: number): { rows: number; run: number; last: number } {
  for (let rows = 1; ; rows += 2) {
    const run = Math.ceil((cells - TURN * (rows - 1)) / rows), last = cells - (rows - 1) * (run + TURN);
    if (last >= 3 && run + (rows > 1 ? (2 * TURN) / Math.PI : 0) <= ROW_MAX) return { rows, run, last };
  }
}

export function along(layout: Pick<Layout, 'cell' | 'radius' | 'rows' | 'run' | 'last' | 'x0' | 'z0'>, arc: number, out: Station): Station {
  const { cell, radius, rows, run, last, x0, z0 } = layout;
  const set = (x: number, z: number, tx: number, tz: number): Station => {
    out.x = x; out.z = z; out.tx = tx; out.tz = tz;
    return out;
  };
  if (arc < 0) return set(x0 + arc * cell, z0, 1, 0);
  let rest = arc;
  for (let j = 0; j < rows; j++) {
    const length = j < rows - 1 ? run : last, dir = j % 2 === 0 ? 1 : -1, start = j % 2 === 0 ? x0 : x0 + run * cell, z = z0 + j * 2 * radius;
    if (rest <= length || j === rows - 1) return set(start + dir * rest * cell, z, dir, 0);
    rest -= length;
    if (rest <= TURN) {
      const theta = (Math.PI * rest) / TURN, cx = start + dir * length * cell;
      return set(cx + dir * radius * Math.sin(theta), z + radius - radius * Math.cos(theta), dir * Math.cos(theta), Math.sin(theta));
    }
    rest -= TURN;
  }
  throw new Error('Parity delay line ran out of rows');
}

export function pathArcs(layout: Pick<Layout, 'rows' | 'run' | 'last' | 'cells' | 'cell' | 'lead'>, tail: number): number[] {
  const arcs = [-layout.lead / layout.cell];
  let at = 0;
  for (let j = 0; j < layout.rows; j++) {
    const length = j < layout.rows - 1 ? layout.run : layout.last;
    for (let k = 0; k < length * 4; k++) arcs.push(at + k / 4);
    at += length;
    if (j === layout.rows - 1) break;
    for (let k = 0; k < 32; k++) arcs.push(at + (TURN * k) / 32);
    at += TURN;
  }
  arcs.push(layout.cells + tail / layout.cell);
  return arcs;
}

export function layoutFor(delay: number): Layout {
  const cells = delay + 3, { rows, run, last } = shape(cells);
  const cell = Math.min(CELL_MAX, REACH / (run + (rows > 1 ? (2 * TURN) / Math.PI : 0)));
  const radius = (TURN * cell) / Math.PI, x0 = -(run * cell) / 2, z0 = 0;
  const bottom = z0 + (rows - 1) * 2 * radius, bulge = rows > 1 ? radius + TRACE : 0;
  const tapA = x0 + (last - 1.5) * cell, tapB = x0 + (last - 0.5) * cell, tap = x0 + (last - 1) * cell, end = x0 + last * cell;
  const pad = new THREE.Vector2(x0 - PAD, z0), coin = new THREE.Vector2(pad.x, z0 - 0.82);
  const splay = Math.max(tap - tapA - 0.18, 0.2), drop = bottom + 0.3;
  const gate = new THREE.Vector2(tap, drop + splay + 0.22 + PIN + GATE.y);
  const truth = new THREE.Vector2(gate.x + GATE.x + PIN + 0.5, gate.y);
  const equal = new THREE.Vector2(truth.x, gate.y + 1.05);
  const call = new THREE.Vector2(equal.x - 0.98, equal.y);
  const verdict = new THREE.Vector2(equal.x + 0.98, equal.y);
  const resistor = new THREE.Vector2(end + 0.1 + RESISTOR.x, bottom);
  const ground = new THREE.Vector2(resistor.x + RESISTOR.x + 0.42, bottom);
  const lane = Math.min(pad.x - 0.55, x0 - bulge - 0.38);
  const agent = new THREE.Vector2(lane + 0.75 + AGENT.x, equal.y);
  const logZ = equal.y + AGENT.y + PIN + 0.34, left = lane - 0.05, right = verdict.x;
  const pinOut = (at: THREE.Vector2, half: THREE.Vector2, side: number): number => at.x + side * (half.x + PIN);
  const wires: Layout['wires'] = {
    feed: [[coin.x, coin.y + 0.26], [pad.x, pad.y - PAD * 0.6]],
    probe: [[pad.x, pad.y], [lane, pad.y + (pad.x - lane)], [lane, agent.y], [pinOut(agent, AGENT, -1), agent.y]],
    output: [[pinOut(agent, AGENT, 1), agent.y], [call.x - LED * 0.7, call.y]],
    stubs: [
      [[tapA, bottom + TRACE * 0.7], [tapA, drop], [tap - 0.18, drop + (tap - 0.18 - tapA)], [tap - 0.18, gate.y - GATE.y - PIN]],
      [[tapB, bottom + TRACE * 0.7], [tapB, drop], [tap + 0.18, drop + (tapB - tap - 0.18)], [tap + 0.18, gate.y - GATE.y - PIN]],
    ],
    truth: [[pinOut(gate, GATE, 1), gate.y], [truth.x - LED * 0.7, truth.y]],
    judged: [[truth.x, truth.y + LED * 0.7], [equal.x, equal.y - EQUAL.y - PIN]],
    called: [[call.x + LED * 0.7, call.y], [pinOut(equal, EQUAL, -1), equal.y]],
    verdict: [[pinOut(equal, EQUAL, 1), equal.y], [verdict.x - LAMP * 0.7, verdict.y]],
  };
  const board = new THREE.Box2(new THREE.Vector2(left - 0.62, coin.y - 0.9), new THREE.Vector2(Math.max(ground.x + 0.5, verdict.x + LAMP, x0 + run * cell + bulge) + 0.62, logZ + 0.78));
  return {
    delay, cells, cell, radius, rows, run, last, x0, z0, lead: PAD, pad, coin,
    taps: [new THREE.Vector2(tapA, bottom), new THREE.Vector2(tapB, bottom)],
    gate, truth, equal, call, verdict, agent, resistor, ground,
    log: { left, right, z: logZ, pitch: (right - left) / LOG },
    wires, board,
  };
}
