/// <reference types="bun" />
// Gridworld STATIC geometry golden tests: wall layouts, hallway/door/key/gap
// cells, per-lane ProceduralMaze walls, and the DeepSea-family action masks.
//
// Oracle: tools/gridworld_ref.py (the module `Sources/Gridworld.metal` is
// bitwise-certified against by `OaK gridworld-probe`). Regenerate with:
//   /Users/studio/OaKMeta/.venv/bin/python frontend/test/fixtures/gen.py --only grid

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';
import { FOURROOMS_FREE_CELLS, fourRoomsDecoder, fourRoomsIsWall } from '@sim/envs/fourrooms';
import { DEEPSEA_N, deepSeaDecoder, deepSeaMask, deepSeaMaskWords } from '@sim/envs/deepsea';
import {
  LARGE_DEEPSEA_N,
  largeDeepSeaDecoder,
  largeDeepSeaMask,
  largeDeepSeaMaskWords,
} from '@sim/envs/largedeepsea';
import { keyCorridorDecoder, keyCorridorIsWall } from '@sim/envs/keycorridor';
import { mkcIsWall, mkcRoomCells, multiKeyCorridorDecoder } from '@sim/envs/multikeycorridor';
import { chainDecoder } from '@sim/envs/chain';
import { memoryChainDecoder } from '@sim/envs/memorychain';
import { delayedChainDecoder } from '@sim/envs/delayedchain';
import { procMazeGen, proceduralMazeDecoder, procMazeWalls } from '@sim/envs/proceduralmaze';
import { highDimMazeDecoder, highDimMazeIsWall } from '@sim/envs/highdimmaze';
import { poMazeDecoder, poMazeIsWall, poMazeView } from '@sim/envs/pomaze';

interface Layout {
  w: number;
  h: number;
  walls: number[];
  freeCells?: number;
  goal?: GridCell;
  start?: GridCell;
  hallways?: GridCell[];
  gaps?: GridCell[];
  doors?: GridCell[];
  keys?: GridCell[];
  free?: GridCell[];
  leftRoom?: GridCell[];
  rightRoom?: GridCell[];
  rooms?: GridCell[][];
  roomBase?: number[];
  views?: { x: number; y: number; bits: number[] }[];
  lastRow?: number;
  rightCost?: number;
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8')) as T;
}

const layouts = load<{ version: number; layouts: Record<string, Layout> }>('static_layouts.json');
const procmaze = load<{
  cases: { seed: number; words: number[]; walls: number[]; carvedPath: GridCell[]; freeCells: number }[];
}>('procmaze_layouts.json');
const dsMasks = load<{ n: number; cases: { seed: number; maskLo: number; maskHi: number; mask: number[] }[] }>(
  'deepsea_masks.json',
);
const ldsMasks = load<{ n: number; cases: { seed: number; words: number[]; mask: number[] }[] }>(
  'largedeepsea_masks.json',
);

function layoutOf(envId: string): Layout {
  const l = layouts.layouts[envId];
  if (!l) throw new Error(`no fixture layout for ${envId}`);
  return l;
}

function gridOf(g: EnvGeometry): GridGeometry {
  expect(g.kind).toBe('grid');
  if (!g.grid) throw new Error('geometry.grid missing');
  return g.grid;
}

/** every decoder's geometry against the fixture's w/h/walls/goal. */
function expectLayout(envId: string, grid: GridGeometry): void {
  const l = layoutOf(envId);
  expect(grid.w).toBe(l.w);
  expect(grid.h).toBe(l.h);
  expect(grid.walls.length).toBe(l.w * l.h);
  expect(Array.from(grid.walls)).toEqual(l.walls);
  if (l.goal) expect(grid.goal).toEqual(l.goal);
}

describe('static wall layouts', () => {
  test('FourRooms: walls, 68 free cells, 4 hallways', () => {
    const grid = gridOf(fourRoomsDecoder.geometry(fourRoomsDecoder.initStatic(8), 8));
    expectLayout('FourRooms-cont-v0', grid);
    const free = grid.walls.length - grid.walls.reduce((a, b) => a + b, 0);
    expect(free).toBe(FOURROOMS_FREE_CELLS);
    expect(free).toBe(68);
    expect(grid.hallways).toEqual(layoutOf('FourRooms-cont-v0').hallways);
    // the hallway cells are exactly the free cells on the two wall lines
    for (const c of grid.hallways ?? []) expect(fourRoomsIsWall(c.x, c.y)).toBe(false);
    expect(fourRoomsIsWall(5, 5)).toBe(true); // the centre is wall
  });

  test('FourRooms: free-cell enumeration matches the reference order', () => {
    const l = layoutOf('FourRooms-cont-v0');
    const free: GridCell[] = [];
    for (let y = 0; y < 11; y++) {
      for (let x = 0; x < 11; x++) if (!fourRoomsIsWall(x, y)) free.push({ x, y });
    }
    expect(free).toEqual(l.free ?? []);
  });

  test('KeyCorridor: 9x7, door cell (4,3) free', () => {
    const grid = gridOf(keyCorridorDecoder.geometry(keyCorridorDecoder.initStatic(24), 24));
    expectLayout('KeyCorridor-cont-v0', grid);
    expect(grid.doors).toEqual(layoutOf('KeyCorridor-cont-v0').doors);
    expect(grid.keys).toEqual(layoutOf('KeyCorridor-cont-v0').keys);
    expect(keyCorridorIsWall(4, 3)).toBe(false);
    expect(keyCorridorIsWall(4, 2)).toBe(true);
  });

  test('MultiKeyCorridor: 17x7, three doors, four rooms', () => {
    const grid = gridOf(
      multiKeyCorridorDecoder.geometry(multiKeyCorridorDecoder.initStatic(80), 80),
    );
    expectLayout('MultiKeyCorridor-cont-v0', grid);
    const l = layoutOf('MultiKeyCorridor-cont-v0');
    expect(grid.doors).toEqual(l.doors);
    expect(grid.keys).toEqual(l.keys);
    for (let r = 0; r < 4; r++) expect(mkcRoomCells(r)).toEqual((l.rooms ?? [])[r] ?? []);
    for (const d of [4, 8, 12]) {
      expect(mkcIsWall(d, 3)).toBe(false);
      expect(mkcIsWall(d, 2)).toBe(true);
    }
  });

  test('HighDimMaze: serpentine walls + the two gaps', () => {
    const grid = gridOf(highDimMazeDecoder.geometry(highDimMazeDecoder.initStatic(104), 104));
    expectLayout('HighDimMaze-cont-v0', grid);
    expect(grid.hallways).toEqual(layoutOf('HighDimMaze-cont-v0').gaps);
    expect(highDimMazeIsWall(3, 9)).toBe(false);
    expect(highDimMazeIsWall(7, 1)).toBe(false);
    expect(highDimMazeIsWall(3, 8)).toBe(true);
  });

  test('POMaze: banded walls + the two gaps', () => {
    const grid = gridOf(poMazeDecoder.geometry(poMazeDecoder.initStatic(112), 112));
    expectLayout('POMaze-cont-v0', grid);
    expect(grid.hallways).toEqual(layoutOf('POMaze-cont-v0').gaps);
    expect(poMazeIsWall(8, 3)).toBe(false);
    expect(poMazeIsWall(2, 7)).toBe(false);
    expect(poMazeIsWall(3, 3)).toBe(true);
  });

  test('POMaze: the 8-neighbour view matches the kernel channel order', () => {
    for (const v of layoutOf('POMaze-cont-v0').views ?? []) {
      expect(Array.from(poMazeView(v.x, v.y))).toEqual(v.bits);
    }
  });

  test('corridors are w x 1 grids with the goal at the far end', () => {
    const cases: [string, ReturnType<typeof gridOf>][] = [
      ['Chain-cont-v0', gridOf(chainDecoder.geometry(chainDecoder.initStatic(64), 64))],
      [
        'MemoryChain-cont-v0',
        gridOf(memoryChainDecoder.geometry(memoryChainDecoder.initStatic(88), 88)),
      ],
      [
        'DelayedChain-cont-v0',
        gridOf(delayedChainDecoder.geometry(delayedChainDecoder.initStatic(120), 120)),
      ],
    ];
    for (const [envId, grid] of cases) {
      expectLayout(envId, grid);
      expect(grid.h).toBe(1);
      expect(grid.walls.every((v) => v === 0)).toBe(true);
    }
  });
});

describe('ProceduralMaze per-lane generation', () => {
  test('wall words match gw_pmaze_gen bitwise for 16 lane seeds', () => {
    expect(procmaze.cases.length).toBe(16);
    for (const c of procmaze.cases) {
      const { words } = procMazeGen(c.seed);
      expect(Array.from(words)).toEqual(c.words);
      // w2 is masked to its low 17 bits (81 bits total)
      expect((words[2] ?? 0) >>> 17).toBe(0);
    }
  });

  test('11x11 bitmap and carved path match, and the path is free', () => {
    for (const c of procmaze.cases) {
      const { words, carvedPath } = procMazeGen(c.seed);
      expect(Array.from(procMazeWalls(words))).toEqual(c.walls);
      expect(carvedPath).toEqual(c.carvedPath);
      expect(carvedPath.length).toBe(17);
      for (const cell of carvedPath) expect(c.walls[cell.y * 11 + cell.x]).toBe(0);
      // the path is monotone: every step advances x or y by exactly one
      for (let i = 1; i < carvedPath.length; i++) {
        const a = carvedPath[i - 1]!;
        const b = carvedPath[i]!;
        expect(b.x - a.x + (b.y - a.y)).toBe(1);
      }
      expect(carvedPath[0]).toEqual({ x: 1, y: 1 });
      expect(carvedPath[16]).toEqual({ x: 9, y: 9 });
    }
  });

  test('geometry exposes the carved path as a debug overlay', () => {
    const first = procmaze.cases[0]!;
    const grid = gridOf(
      proceduralMazeDecoder.geometry(proceduralMazeDecoder.initStatic(first.seed), first.seed),
    );
    expect(grid.w).toBe(11);
    expect(grid.h).toBe(11);
    expect(Array.from(grid.walls)).toEqual(first.walls);
    expect(grid.carvedPath).toEqual(first.carvedPath);
    expect(grid.goal).toEqual({ x: 9, y: 9 });
  });
});

describe('DeepSea family action masks', () => {
  test('DeepSea: 64-bit mask matches deepsea_reset for 16 lane seeds', () => {
    expect(dsMasks.n).toBe(DEEPSEA_N);
    expect(dsMasks.cases.length).toBe(16);
    for (const c of dsMasks.cases) {
      const [lo, hi] = deepSeaMaskWords(c.seed);
      expect(lo).toBe(c.maskLo);
      expect(hi).toBe(c.maskHi);
      expect(Array.from(deepSeaMask(lo, hi))).toEqual(c.mask);
    }
  });

  test('DeepSea: geometry carries the mask, no walls, treasure at (7,7)', () => {
    const c = dsMasks.cases[0]!;
    const grid = gridOf(deepSeaDecoder.geometry(deepSeaDecoder.initStatic(c.seed), c.seed));
    expectLayout('DeepSea-cont-v0', grid);
    expect(Array.from(grid.actionMask ?? [])).toEqual(c.mask);
    expect(grid.goal).toEqual({ x: 7, y: 7 });
  });

  test('LargeDeepSea: 256-bit mask matches ldeepsea_reset for 16 lane seeds', () => {
    expect(ldsMasks.n).toBe(LARGE_DEEPSEA_N);
    expect(ldsMasks.cases.length).toBe(16);
    for (const c of ldsMasks.cases) {
      const words = largeDeepSeaMaskWords(c.seed);
      expect(Array.from(words)).toEqual(c.words);
      expect(Array.from(largeDeepSeaMask(words))).toEqual(c.mask);
      expect(c.mask.length).toBe(256);
    }
  });

  test('LargeDeepSea: geometry carries the mask, treasure at (15,15)', () => {
    const c = ldsMasks.cases[0]!;
    const grid = gridOf(
      largeDeepSeaDecoder.geometry(largeDeepSeaDecoder.initStatic(c.seed), c.seed),
    );
    expectLayout('LargeDeepSea-cont-v0', grid);
    expect(Array.from(grid.actionMask ?? [])).toEqual(c.mask);
    expect(grid.goal).toEqual({ x: 15, y: 15 });
  });
});

describe('decoder wiring', () => {
  test('only MultiKeyCorridor needs a ctrl replay', () => {
    const all = [
      fourRoomsDecoder,
      deepSeaDecoder,
      largeDeepSeaDecoder,
      keyCorridorDecoder,
      chainDecoder,
      memoryChainDecoder,
      delayedChainDecoder,
      proceduralMazeDecoder,
      highDimMazeDecoder,
      poMazeDecoder,
    ];
    for (const d of all) expect(d.needsReplay).toBe(false);
    expect(multiKeyCorridorDecoder.needsReplay).toBe(true);
  });

  test('every decoder reports its own envId', () => {
    const pairs: [string, string][] = [
      [fourRoomsDecoder.envId, 'FourRooms-cont-v0'],
      [deepSeaDecoder.envId, 'DeepSea-cont-v0'],
      [largeDeepSeaDecoder.envId, 'LargeDeepSea-cont-v0'],
      [keyCorridorDecoder.envId, 'KeyCorridor-cont-v0'],
      [multiKeyCorridorDecoder.envId, 'MultiKeyCorridor-cont-v0'],
      [chainDecoder.envId, 'Chain-cont-v0'],
      [memoryChainDecoder.envId, 'MemoryChain-cont-v0'],
      [delayedChainDecoder.envId, 'DelayedChain-cont-v0'],
      [proceduralMazeDecoder.envId, 'ProceduralMaze-cont-v0'],
      [highDimMazeDecoder.envId, 'HighDimMaze-cont-v0'],
      [poMazeDecoder.envId, 'POMaze-cont-v0'],
    ];
    for (const [got, want] of pairs) expect(got).toBe(want);
  });
});
