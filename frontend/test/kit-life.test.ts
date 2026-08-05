/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import type { SimEvent } from '@sim/types';
import * as THREE from 'three';

import {
  CELL,
  DoorGate,
  GoalBeacon,
  HopMotion,
  KeyToken,
  buildCellField,
  cellIndex,
  cellToWorld,
  cellToWorldVec,
  clampCell,
  forEachCell,
  gridExtent,
  gridRadius,
  inBounds,
  isWall,
  wallCellCount,
  worldToCell,
  type CellFieldStyle,
} from '@kit/grid';
import { Wisp, placeAtCell, yawFromGridDelta } from '@kit/wisp';
import {
  CelebrationPool,
  PULSE_SECONDS,
  burst,
  celebrationCapacity,
  celebrationPulse,
  celebrationsFor,
  disposeCelebrations,
  existingCelebrations,
  makePulseChannel,
  pulseChannel,
  ripple,
  sigh,
} from '@kit/celebrate';
import {
  DEFAULT_ORBIT_LIMITS,
  clampOrbit,
  damp,
  makeFrameRig,
  makeOrbitRig,
  makeTrackRig,
  orbitPosition,
} from '@kit/rigs';
import {
  bareTreeGeometry,
  broadleafTreeGeometry,
  coniferTreeGeometry,
  excludeAny,
  excludeBand,
  excludeRect,
  flowerGeometry,
  grassClumpGeometry,
  rockGeometry,
  scatterFlowers,
  scatterGrass,
  scatterPlacements,
  scatterRocks,
  scatterTrees,
  vegetationTierScale,
} from '@kit/vegetation';
import { PARTICLE_PRESETS, makeAmbientParticles, makeMistPlanes, particleTierScale } from '@kit/particles';
import { makeWater, waterHeightAt, waterSegmentScale } from '@kit/water';
import { DEBUG_LAYER, GroundTruthOverlay, applyDebugLayer, defaultFkToWorld, enableDebugLayer } from '@kit/debug';
import { MAX_CATCHUP_FRAMES, firstRecordingFor, makePlaybackClock, sampleFrame, type HarnessSource } from '@kit/harness';
import { TOKENS } from '@kit/tokens';
import type { FrameState } from '@kit/types';
import type { EnvFrame, EnvGeometry, GridGeometry } from '@sim/types';


const HAND_ROWS = [
  '#####',
  '#..#.',
  '#...#',
  '.#.#.',
] as const;

function handGrid(): GridGeometry {
  const h = HAND_ROWS.length;
  const w = (HAND_ROWS[0] as string).length;
  const walls = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = HAND_ROWS[y] as string;
    for (let x = 0; x < w; x++) walls[y * w + x] = row[x] === '#' ? 1 : 0;
  }
  return { w, h, walls };
}

function handWallCount(): number {
  let n = 0;
  for (const row of HAND_ROWS) for (const ch of row) if (ch === '#') n++;
  return n;
}

const BASIC_STYLE: CellFieldStyle = { floorColor: 0x556644, wallColor: 0x99aa88 };

function instanceTranslation(mesh: THREE.InstancedMesh, i: number): [number, number, number] {
  const m = new THREE.Matrix4();
  mesh.getMatrixAt(i, m);
  const e = m.elements;
  return [e[12] as number, e[13] as number, e[14] as number];
}

function makeState(pose: readonly number[], derived: unknown = null): FrameState {
  const p = new Float32Array(9);
  for (let i = 0; i < 9; i++) p[i] = pose[i] ?? 0;
  return { pose: p, ctrl: new Float32Array(6), done: false, step: 0, simTime: 0, derived };
}


describe('grid coordinates', () => {
  test('1 cell = 1 world unit and the grid is centered on the origin', () => {
    expect(CELL).toBe(TOKENS.cellSize);
    const odd = { w: 11, h: 11 };
    expect(cellToWorld(5, 5, odd)).toEqual([0, 0, 0]);
    expect(cellToWorld(0, 0, odd)).toEqual([-5, 0, -5]);
    expect(cellToWorld(10, 10, odd)).toEqual([5, 0, 5]);

    const even = { w: 8, h: 8 };
    expect(cellToWorld(0, 0, even)).toEqual([-3.5, 0, -3.5]);
    expect(cellToWorld(7, 7, even)).toEqual([3.5, 0, 3.5]);
  });

  test('+x is grid-x (east) and +z is grid-y (south); y is always 0', () => {
    const g = { w: 9, h: 7 };
    const a = cellToWorld(3, 3, g);
    const east = cellToWorld(4, 3, g);
    const south = cellToWorld(3, 4, g);
    expect(east[0] - a[0]).toBeCloseTo(1, 12);
    expect(east[2] - a[2]).toBeCloseTo(0, 12);
    expect(south[2] - a[2]).toBeCloseTo(1, 12);
    expect(south[0] - a[0]).toBeCloseTo(0, 12);
    expect(a[1]).toBe(0);
  });

  test('worldToCell inverts cellToWorld exactly for every cell of several shapes', () => {
    for (const g of [{ w: 11, h: 11 }, { w: 8, h: 8 }, { w: 17, h: 7 }, { w: 32, h: 1 }, { w: 1, h: 32 }]) {
      for (let y = 0; y < g.h; y++) {
        for (let x = 0; x < g.w; x++) {
          const [wx, , wz] = cellToWorld(x, y, g);
          const back = worldToCell(wx, wz, g);
          expect(back.x).toBe(x);
          expect(back.y).toBe(y);
        }
      }
    }
  });

  test('worldToCell rounds to the nearest cell and never clamps on its own', () => {
    const g = { w: 5, h: 5 };
    expect(worldToCell(0.4, -0.4, g)).toEqual({ x: 2, y: 2 });
    expect(worldToCell(-9, 9, g)).toEqual({ x: -7, y: 11 });
    expect(clampCell(worldToCell(-9, 9, g), g)).toEqual({ x: 0, y: 4 });
  });

  test('cellToWorldVec matches cellToWorld and is allocation-free', () => {
    const g = { w: 6, h: 4 };
    const out = new THREE.Vector3(9, 9, 9);
    const same = cellToWorldVec(2, 3, g, out);
    expect(same).toBe(out);
    const [x, y, z] = cellToWorld(2, 3, g);
    expect([out.x, out.y, out.z]).toEqual([x, y, z]);
  });

  test('bounds, index, extent and radius helpers', () => {
    const g = handGrid();
    expect(g.w).toBe(5);
    expect(g.h).toBe(4);
    expect(inBounds(0, 0, g)).toBe(true);
    expect(inBounds(5, 0, g)).toBe(false);
    expect(cellIndex(2, 1, g)).toBe(7);
    expect(cellIndex(-1, 0, g)).toBe(-1);
    expect(gridExtent(g)).toEqual({ width: 5, depth: 4 });
    expect(gridRadius(g)).toBeCloseTo(Math.sqrt(25 + 16) / 2, 12);
  });

  test('isWall reads the array strictly and treats out-of-bounds as wall', () => {
    const g = handGrid();
    expect(isWall(g, 0, 0)).toBe(true);
    expect(isWall(g, 1, 1)).toBe(false);
    expect(isWall(g, 3, 1)).toBe(true);
    expect(isWall(g, 0, 3)).toBe(false);
    expect(isWall(g, -1, 1)).toBe(true);
    expect(isWall(g, 5, 1)).toBe(true);
  });

  test('forEachCell visits every cell once in row-major order', () => {
    const g = handGrid();
    const seen: number[] = [];
    forEachCell(g, (_x, _y, _wall, index) => seen.push(index));
    expect(seen.length).toBe(g.w * g.h);
    expect(seen).toEqual([...Array(g.w * g.h).keys()]);
  });
});


describe('buildCellField', () => {
  test('places exactly one wall block per 1 in the walls array', () => {
    const g = handGrid();
    const expected = handWallCount();
    expect(wallCellCount(g)).toBe(expected);

    const field = buildCellField(g, BASIC_STYLE);
    expect(field.wallCount).toBe(expected);
    expect(field.floorCount).toBe(g.w * g.h - expected);
    expect(field.wallMeshes.length).toBe(1);
    expect(field.floorMeshes.length).toBe(1);
    expect((field.wallMeshes[0] as THREE.InstancedMesh).count).toBe(expected);
    expect((field.floorMeshes[0] as THREE.InstancedMesh).count).toBe(g.w * g.h - expected);
    field.dispose();
  });

  test('every wall instance sits exactly on its cell, and only on wall cells', () => {
    const g = handGrid();
    const field = buildCellField(g, BASIC_STYLE);
    const mesh = field.wallMeshes[0] as THREE.InstancedMesh;

    for (let y = 0; y < g.h; y++) {
      for (let x = 0; x < g.w; x++) {
        const id = field.wallInstanceAt(x, y);
        if (isWall(g, x, y)) {
          expect(id).toBeGreaterThanOrEqual(0);
          const [wx, , wz] = cellToWorld(x, y, g);
          const t = instanceTranslation(mesh, id);
          expect(t[0]).toBeCloseTo(wx, 12);
          expect(t[1]).toBeCloseTo(0, 12);
          expect(t[2]).toBeCloseTo(wz, 12);
        } else {
          expect(id).toBe(-1);
          expect(field.floorInstanceAt(x, y)).toBeGreaterThanOrEqual(0);
        }
      }
    }
    field.dispose();
  });

  test('floorUnderWalls and floorFilter change floor coverage, never wall coverage', () => {
    const g = handGrid();
    const full = buildCellField(g, { ...BASIC_STYLE, floorUnderWalls: true });
    expect(full.floorCount).toBe(g.w * g.h);
    expect(full.wallCount).toBe(handWallCount());
    full.dispose();

    const triangle = buildCellField(g, { ...BASIC_STYLE, floorFilter: (x, y) => x <= y });
    let expectedFloors = 0;
    forEachCell(g, (x, y, wall) => {
      if (!wall && x <= y) expectedFloors++;
    });
    expect(triangle.floorCount).toBe(expectedFloors);
    expect(triangle.wallCount).toBe(handWallCount());
    triangle.dispose();
  });

  test('an all-open grid places zero walls and no wall mesh at all', () => {
    const g: GridGeometry = { w: 4, h: 3, walls: new Uint8Array(12) };
    const field = buildCellField(g, BASIC_STYLE);
    expect(field.wallCount).toBe(0);
    expect(field.wallMeshes.length).toBe(0);
    expect(field.floorCount).toBe(12);
    expect(field.group.children.length).toBe(1);
    field.dispose();
  });

  test('every wall shape and tier still places the same instance count', () => {
    const g = handGrid();
    for (const shape of ['block', 'hedge', 'pillar'] as const) {
      for (const tier of ['hero', 'medium', 'thumb'] as const) {
        const field = buildCellField(g, { ...BASIC_STYLE, wallShape: shape, wind: 0.04 }, tier);
        expect(field.wallCount).toBe(handWallCount());
        field.dispose();
      }
    }
  });

  test('per-cell tint hooks are called for the exact cells that get instances', () => {
    const g = handGrid();
    const tinted: string[] = [];
    const field = buildCellField(g, {
      ...BASIC_STYLE,
      floorTint: (x, y) => {
        tinted.push(`f${x},${y}`);
        return 0xffffff;
      },
      wallTint: (x, y) => {
        tinted.push(`w${x},${y}`);
        return 0xffffff;
      },
    });
    expect(tinted.filter((s) => s.startsWith('w')).length).toBe(field.wallCount);
    expect(tinted.filter((s) => s.startsWith('f')).length).toBe(field.floorCount);
    field.dispose();
  });
});

describe('cell-placed components', () => {
  test('GoalBeacon / KeyToken / DoorGate land on the exact cell', () => {
    const g = { w: 9, h: 7 };
    const [wx, , wz] = cellToWorld(6, 2, g);

    const beacon = new GoalBeacon({ color: 0xffd166 });
    beacon.setCell(6, 2, g);
    expect(beacon.group.position.x).toBeCloseTo(wx, 12);
    expect(beacon.group.position.z).toBeCloseTo(wz, 12);
    beacon.update(0.016, 1.5);
    beacon.flare();
    beacon.update(0.016, 1.5);
    beacon.dispose();

    const token = new KeyToken({ color: 0xffc94d });
    token.setCell(6, 2, g);
    expect(token.group.position.x).toBeCloseTo(wx, 12);
    expect(token.group.position.z).toBeCloseTo(wz, 12);
    expect(token.group.position.y).toBeGreaterThan(0);
    token.dispose();

    const door = new DoorGate({ color: 0x88664a });
    door.setCell(6, 2, g);
    expect(door.group.position.x).toBeCloseTo(wx, 12);
    expect(door.group.position.z).toBeCloseTo(wz, 12);
    door.dispose();
  });

  test('DoorGate animates from the host clock and snaps immediately when asked', () => {
    const door = new DoorGate({ color: 0x88664a, duration: 0.5 });
    expect(door.open).toBe(false);
    expect(door.progress).toBe(0);

    door.setOpen(true);
    expect(door.open).toBe(true);
    expect(door.progress).toBe(0);
    door.update(0.25, 0);
    expect(door.progress).toBeCloseTo(0.5, 6);
    door.update(0.25, 0);
    expect(door.progress).toBe(1);
    door.update(0.25, 0);
    expect(door.progress).toBe(1);

    door.setOpen(false, true);
    expect(door.progress).toBe(0);

    door.setOpen(true);
    door.snap();
    expect(door.progress).toBe(1);
    door.dispose();
  });
});


describe('HopMotion', () => {
  const grid = { w: 11, h: 11 };

  test('a fresh motion is landed on cell (0, 0)', () => {
    const hop = new HopMotion(grid);
    expect(hop.airborne).toBe(false);
    expect(hop.progress).toBe(1);
    expect(hop.cellX).toBe(0);
    expect(hop.cellY).toBe(0);
    const [x, , z] = cellToWorld(0, 0, grid);
    expect(hop.position.x).toBeCloseTo(x, 12);
    expect(hop.position.z).toBeCloseTo(z, 12);
  });

  test('snap(x, y) teleports: exact position, progress 1, not airborne, update is a no-op', () => {
    const hop = new HopMotion(grid);
    hop.snap(3, 7);
    const [x, , z] = cellToWorld(3, 7, grid);
    expect(hop.position.x).toBe(x);
    expect(hop.position.y).toBe(0);
    expect(hop.position.z).toBe(z);
    expect(hop.progress).toBe(1);
    expect(hop.airborne).toBe(false);
    expect(hop.cellX).toBe(3);
    expect(hop.cellY).toBe(7);

    hop.update(99);
    expect(hop.position.x).toBe(x);
    expect(hop.position.z).toBe(z);
    expect(hop.airborne).toBe(false);
  });

  test('snap() with no arguments finishes an in-flight hop instantly', () => {
    const hop = new HopMotion(grid, { duration: 1 });
    hop.snap(2, 2);
    hop.hopTo(5, 2, 0);
    hop.update(0.4);
    expect(hop.airborne).toBe(true);
    expect(hop.position.y).toBeGreaterThan(0);

    hop.snap();
    const [x, , z] = cellToWorld(5, 2, grid);
    expect(hop.position.x).toBeCloseTo(x, 12);
    expect(hop.position.y).toBe(0);
    expect(hop.position.z).toBeCloseTo(z, 12);
    expect(hop.progress).toBe(1);
    expect(hop.airborne).toBe(false);
  });

  test('the hop is a STRAIGHT line in xz with a parabolic arc in y', () => {
    const hop = new HopMotion(grid, { duration: 1, height: 0.4 });
    hop.snap(1, 1);
    const from = cellToWorld(1, 1, grid);
    const to = cellToWorld(4, 5, grid);
    hop.hopTo(4, 5, 0);

    for (const t of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      hop.update(t);
      const px = hop.position.x - from[0];
      const pz = hop.position.z - from[2];
      const dx = to[0] - from[0];
      const dz = to[2] - from[2];
      expect(Math.abs(px * dz - pz * dx)).toBeLessThan(1e-9);
      expect(hop.position.y).toBeGreaterThan(0);
    }

    hop.update(0.5);
    expect(hop.position.x).toBeCloseTo((from[0] + to[0]) / 2, 9);
    expect(hop.position.z).toBeCloseTo((from[2] + to[2]) / 2, 9);
    expect(hop.position.y).toBeCloseTo(0.4, 9);

    hop.update(1);
    expect(hop.position.x).toBe(to[0]);
    expect(hop.position.y).toBe(0);
    expect(hop.position.z).toBe(to[2]);
    expect(hop.airborne).toBe(false);
  });

  test('hopping to the current cell while landed is a no-op', () => {
    const hop = new HopMotion(grid);
    hop.snap(4, 4);
    hop.hopTo(4, 4, 0);
    expect(hop.airborne).toBe(false);
    expect(hop.progress).toBe(1);
  });

  test('setGrid rebases an IN-FLIGHT hop into the new grid', () => {
    const small = { w: 11, h: 11 };
    const big = { w: 21, h: 21 };
    const hop = new HopMotion(small, { duration: 1 });
    hop.snap(1, 1);
    hop.hopTo(4, 5, 0);
    hop.update(0.5);
    const midX = hop.position.x;
    const midZ = hop.position.z;

    hop.setGrid(big);
    expect(hop.position.x).toBeCloseTo(midX + ((small.w - big.w) / 2) * CELL, 12);
    expect(hop.position.z).toBeCloseTo(midZ + ((small.h - big.h) / 2) * CELL, 12);

    hop.update(1);
    const [x, , z] = cellToWorld(4, 5, big);
    expect(hop.position.x).toBeCloseTo(x, 12);
    expect(hop.position.z).toBeCloseTo(z, 12);
    expect(hop.airborne).toBe(false);
  });

  test('setGrid moves a LANDED motion onto the same cell in the new grid', () => {
    const hop = new HopMotion({ w: 11, h: 11 });
    hop.snap(3, 7);
    hop.setGrid({ w: 5, h: 9 });
    const [x, , z] = cellToWorld(3, 7, { w: 5, h: 9 });
    expect(hop.position.x).toBeCloseTo(x, 12);
    expect(hop.position.y).toBe(0);
    expect(hop.position.z).toBeCloseTo(z, 12);
    expect(hop.progress).toBe(1);
    hop.setGrid({ w: 5, h: 9 });
    expect(hop.position.x).toBeCloseTo(x, 12);
  });

  test('a hop in flight is replaced, never queued', () => {
    const hop = new HopMotion(grid, { duration: 1 });
    hop.snap(0, 0);
    hop.hopTo(1, 0, 0);
    hop.update(0.5);
    hop.hopTo(0, 1, 0.5);
    expect(hop.cellX).toBe(0);
    expect(hop.cellY).toBe(1);
    hop.update(1.5);
    const [x, , z] = cellToWorld(0, 1, grid);
    expect(hop.position.x).toBeCloseTo(x, 12);
    expect(hop.position.z).toBeCloseTo(z, 12);
  });
});


describe('Wisp', () => {
  const grid = { w: 11, h: 11 };

  test('setCell puts the avatar exactly on the cell and the orb at wispHeight', () => {
    const wisp = new Wisp({ color: 0x8fd47a }, grid);
    wisp.setCell(7, 2, grid);
    const [x, , z] = cellToWorld(7, 2, grid);
    expect(wisp.position.x).toBeCloseTo(x, 12);
    expect(wisp.position.z).toBeCloseTo(z, 12);
    expect(wisp.cellX).toBe(7);
    expect(wisp.cellY).toBe(2);

    wisp.group.updateMatrixWorld(true);
    const world = new THREE.Vector3();
    wisp.core.getWorldPosition(world);
    expect(world.x).toBeCloseTo(x, 6);
    expect(world.y).toBeCloseTo(TOKENS.wispHeight, 6);
    expect(world.z).toBeCloseTo(z, 6);
    wisp.dispose();
  });

  test('the probe object NEVER moves off the exact height — 4 s of idle bob', () => {
    const wisp = new Wisp({ color: 0x8fd47a }, grid);
    wisp.setCell(4, 6, grid);
    const [x, , z] = cellToWorld(4, 6, grid);
    const world = new THREE.Vector3();
    let worstY = 0;
    let worstXZ = 0;
    for (let i = 0; i < 240; i++) {
      const time = i / 60;
      wisp.update(1 / 60, time);
      wisp.group.updateMatrixWorld(true);
      wisp.probeObject.getWorldPosition(world);
      worstY = Math.max(worstY, Math.abs(world.y - TOKENS.wispHeight));
      worstXZ = Math.max(worstXZ, Math.abs(world.x - x), Math.abs(world.z - z));
    }
    expect(worstY).toBeLessThan(1e-6);
    expect(worstXZ).toBeLessThan(1e-6);
    expect(wisp.probeObject).toBe(wisp.core);

    const halo = wisp.group.getObjectByName('oakWispHalo');
    expect(halo).toBeDefined();
    let haloSpread = 0;
    for (let i = 0; i < 120; i++) {
      wisp.update(1 / 60, i / 60);
      wisp.group.updateMatrixWorld(true);
      halo?.getWorldPosition(world);
      haloSpread = Math.max(haloSpread, Math.abs(world.y - TOKENS.wispHeight));
    }
    expect(haloSpread).toBeGreaterThan(1e-3);
    wisp.dispose();
  });

  test('the probe object stays exact mid-hop as well (arc apex is the avatar, not the core offset)', () => {
    const wisp = new Wisp({ color: 0x8fd47a }, grid);
    wisp.setCell(1, 1, grid);
    wisp.hopTo({ x: 5, y: 1 }, 0);
    const world = new THREE.Vector3();
    for (let i = 1; i <= 20; i++) {
      const time = (i / 20) * 0.16;
      wisp.update(1 / 60, time);
      wisp.group.updateMatrixWorld(true);
      wisp.probeObject.getWorldPosition(world);
      expect(world.y - wisp.position.y).toBeCloseTo(TOKENS.wispHeight, 9);
    }
    wisp.dispose();
  });

  test('snap() lands an in-flight hop with no easing left over', () => {
    const wisp = new Wisp({ color: 0x8fd47a }, grid);
    wisp.setCell(1, 1, grid);
    wisp.update(0.016, 0);
    wisp.hopTo({ x: 5, y: 1 }, 0);
    wisp.update(0.016, 0.05);
    expect(wisp.motion.airborne).toBe(true);

    wisp.snap();
    const [x, , z] = cellToWorld(5, 1, grid);
    expect(wisp.position.x).toBeCloseTo(x, 12);
    expect(wisp.position.z).toBeCloseTo(z, 12);
    expect(wisp.motion.airborne).toBe(false);
    wisp.dispose();
  });

  test('carry() attaches and releases a token, returning the previous one', () => {
    const wisp = new Wisp({ color: 0xffc94d }, grid);
    const seed = new THREE.Object3D();
    const pearl = new THREE.Object3D();

    expect(wisp.carry(seed)).toBe(null);
    expect(wisp.carrying).toBe(seed);
    expect(seed.parent).not.toBe(null);

    expect(wisp.carry(pearl)).toBe(seed);
    expect(seed.parent).toBe(null);
    expect(wisp.carrying).toBe(pearl);

    expect(wisp.carry(null)).toBe(pearl);
    expect(wisp.carrying).toBe(null);
    wisp.dispose();
  });

  test('face() accepts yaw radians or a grid delta', () => {
    expect(yawFromGridDelta(0, 0)).toBe(0);
    expect(yawFromGridDelta(0, 1)).toBeCloseTo(0, 12);
    expect(yawFromGridDelta(1, 0)).toBeCloseTo(Math.PI / 2, 12);
    expect(yawFromGridDelta(-1, 0)).toBeCloseTo(-Math.PI / 2, 12);

    const wisp = new Wisp({ color: 0x8fd47a }, grid);
    wisp.face({ dx: 1, dy: 0 });
    wisp.snap();
    wisp.update(0.016, 0);
    wisp.face(1.25);
    wisp.snap();
    wisp.update(0.016, 0);
    wisp.dispose();
  });

  test('dim / flare / update never throw and stay bounded', () => {
    const wisp = new Wisp({ color: 0x9ff0a8, tier: 'medium' }, grid);
    wisp.setCell(0, 0, grid);
    wisp.dim();
    for (let i = 0; i < 30; i++) wisp.update(1 / 60, i / 60);
    wisp.flare();
    wisp.undim();
    for (let i = 0; i < 60; i++) wisp.update(1 / 60, i / 60);
    expect(Number.isFinite(wisp.position.x)).toBe(true);
    wisp.dispose();
  });

  test('a thumb-tier wisp drops the skirt and ribbon but keeps the exact core', () => {
    const wisp = new Wisp({ color: 0x8fd47a, tier: 'thumb' }, grid);
    wisp.setCell(3, 3, grid);
    wisp.update(0.016, 0);
    wisp.group.updateMatrixWorld(true);
    const world = new THREE.Vector3();
    wisp.core.getWorldPosition(world);
    const [x, , z] = cellToWorld(3, 3, grid);
    expect(world.x).toBeCloseTo(x, 6);
    expect(world.z).toBeCloseTo(z, 6);
    wisp.dispose();
  });

  test('placeAtCell positions any object on a cell', () => {
    const o = new THREE.Object3D();
    placeAtCell(o, 2, 5, grid, 0.4);
    const [x, , z] = cellToWorld(2, 5, grid);
    expect(o.position.x).toBeCloseTo(x, 12);
    expect(o.position.y).toBe(0.4);
    expect(o.position.z).toBeCloseTo(z, 12);
  });
});


describe('celebration pool', () => {
  test('capacity is tier policy and thumb celebrates silently', () => {
    expect(celebrationCapacity('hero').particles).toBeGreaterThan(celebrationCapacity('medium').particles);
    expect(celebrationCapacity('thumb')).toEqual({ particles: 0, ripples: 0 });
  });

  test('400 celebrations do not grow ANY buffer, mesh or child list', () => {
    const pool = new CelebrationPool({ tier: 'hero', seed: 7 });
    const children = pool.group.children.length;
    const points = pool.group.children.find((c) => c.name === 'oakCelebrationBurst') as THREE.Points;
    const positions = points.geometry.getAttribute('position');
    const buffer = positions.array;
    const capacity = pool.stats.particles;

    let t = 0;
    for (let i = 0; i < 400; i++) {
      t += 0.05;
      pool.burst({ x: i % 5, y: 0.5, z: i % 3 }, 0xffd166, 1);
      pool.ripple({ x: i % 5, y: 0, z: i % 3 }, 0xffd166, 1);
      pool.update(0.05, t);
      expect(pool.activeRipples).toBeLessThanOrEqual(8);
    }

    expect(pool.group.children.length).toBe(children);
    expect(pool.stats.particles).toBe(capacity);
    expect(points.geometry.getAttribute('position')).toBe(positions);
    expect(positions.array).toBe(buffer);
    expect(positions.count).toBe(capacity);
    expect(pool.stats.bursts).toBe(400);
    expect(pool.stats.ripples).toBe(400);
    expect(pool.stats.cursor).toBeLessThan(capacity);
    pool.dispose();
  });

  test('the ring cursor wraps instead of overflowing', () => {
    const pool = new CelebrationPool({ particleCapacity: 64, rippleCapacity: 2, burstSize: 48 });
    for (let i = 0; i < 20; i++) pool.burst({ x: 0, y: 0, z: 0 }, 0xffffff, 1);
    expect(pool.stats.cursor).toBeGreaterThanOrEqual(0);
    expect(pool.stats.cursor).toBeLessThan(64);
    pool.dispose();
  });

  test('a thumb pool still accepts every call (and still pulses)', () => {
    const pool = new CelebrationPool({ tier: 'thumb' });
    pool.burst({ x: 0, y: 0, z: 0 }, 0xffffff, 1);
    pool.ripple({ x: 0, y: 0, z: 0 }, 0xffffff, 1);
    pool.sigh(1);
    pool.update(0.016, 0.016);
    expect(pool.stats.particles).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    pool.dispose();
  });

  test('clear() erases everything a scrub must not carry across', () => {
    const pool = new CelebrationPool({ tier: 'hero' });
    pool.burst({ x: 0, y: 1, z: 0 }, 0xffd166, 1);
    pool.ripple({ x: 0, y: 0, z: 0 }, 0xffd166, 1);
    pool.update(0.016, 0.1);
    expect(pool.activeRipples).toBe(1);
    pool.clear();
    expect(pool.activeRipples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);
    pool.dispose();
  });

  test('the pulse channel is a 300 ms decay the host reads', () => {
    const pulse = makePulseChannel();
    expect(pulse.bloom).toBe(0);
    pulse.fire();
    expect(pulse.bloom).toBe(1);
    pulse.update(PULSE_SECONDS / 2);
    expect(pulse.bloom).toBeCloseTo(0.5, 6);
    pulse.update(PULSE_SECONDS);
    expect(pulse.bloom).toBe(0);

    pulse.fireSigh(0.5);
    expect(pulse.sigh).toBe(0.5);
    pulse.reset();
    expect(pulse.sigh).toBe(0);

    pulse.fire(0.4);
    pulse.fire(0.9);
    expect(pulse.bloom).toBe(0.9);
    pulse.fire(2);
    expect(pulse.bloom).toBe(1);
  });

  test('two pools do not share a pulse channel', () => {
    const a = new CelebrationPool({ tier: 'medium' });
    const b = new CelebrationPool({ tier: 'medium' });
    expect(a.pulse).not.toBe(b.pulse);
    expect(a.pulse).not.toBe(pulseChannel);

    a.burst({ x: 0, y: 1, z: 0 }, 0xffd166, 1);
    expect(a.pulse.bloom).toBe(1);
    expect(b.pulse.bloom).toBe(0);
    expect(pulseChannel.bloom).toBe(0);

    const shared = makePulseChannel();
    const c = new CelebrationPool({ tier: 'medium', pulse: shared });
    const d = new CelebrationPool({ tier: 'medium', pulse: shared });
    c.sigh(1);
    expect(d.pulse.sigh).toBe(1);

    for (const pool of [a, b, c, d]) pool.dispose();
  });

  test('a host resolves the pulse per scene', () => {
    const scene = new THREE.Scene();
    expect(existingCelebrations(scene)).toBeNull();
    expect(celebrationPulse(scene)).toBe(pulseChannel);

    const pool = celebrationsFor(scene, { tier: 'medium' });
    expect(existingCelebrations(scene)).toBe(pool);
    expect(celebrationPulse(scene)).toBe(pool.pulse);
    burst(scene, { x: 0, y: 1, z: 0 }, 0xffd166, 1);
    expect(celebrationPulse(scene).bloom).toBe(1);
    expect(pulseChannel.bloom).toBe(0);
    disposeCelebrations(scene);
  });

  test('a burst fired before the first update() at a non-zero playhead is still alive', () => {
    const pool = new CelebrationPool({ tier: 'hero', seed: 3 });
    const points = pool.group.children.find((c) => c.name === 'oakCelebrationBurst') as THREE.Points;
    const starts = points.geometry.getAttribute('aStart');
    const life = points.geometry.getAttribute('aLife');

    const now = 1234.5;
    pool.burst({ x: 0, y: 1, z: 0 }, 0xffd166, 1, now);
    pool.ripple({ x: 0, y: 0, z: 0 }, 0xffd166, 1, now);
    expect(starts.getX(0)).toBeCloseTo(now, 3);

    pool.update(1 / 60, now + 1 / 60);
    const age = now + 1 / 60 - starts.getX(0);
    expect(age).toBeGreaterThanOrEqual(0);
    expect(age).toBeLessThan(life.getX(0));
    expect(pool.activeRipples).toBe(1);

    const other = new CelebrationPool({ tier: 'hero', seed: 3 });
    other.setTime(now);
    other.burst({ x: 0, y: 1, z: 0 }, 0xffd166, 1);
    const otherStarts = (other.group.children.find((c) => c.name === 'oakCelebrationBurst') as THREE.Points)
      .geometry.getAttribute('aStart');
    expect(otherStarts.getX(0)).toBeCloseTo(now, 3);

    const ordered = new CelebrationPool({ tier: 'hero', seed: 3 });
    ordered.update(1 / 60, now);
    ordered.burst({ x: 0, y: 1, z: 0 }, 0xffd166, 1);
    const orderedStarts = (ordered.group.children.find((c) => c.name === 'oakCelebrationBurst') as THREE.Points)
      .geometry.getAttribute('aStart');
    expect(orderedStarts.getX(0)).toBeCloseTo(now, 3);

    for (const p of [pool, other, ordered]) p.dispose();
  });

  test('scene-level burst/ripple/sigh reuse ONE pool per scene', () => {
    const scene = new THREE.Scene();
    const a = celebrationsFor(scene, { tier: 'medium' });
    burst(scene, { x: 0, y: 1, z: 0 }, 0xffd166, 1);
    ripple(scene, { x: 0, y: 0, z: 0 }, 0xffd166, 1);
    sigh(scene, 1);
    const b = celebrationsFor(scene);
    expect(b).toBe(a);
    expect(a.stats.bursts).toBe(1);
    expect(a.stats.ripples).toBe(1);
    expect(scene.children.filter((c) => c.name === 'oakCelebrations').length).toBe(1);
    disposeCelebrations(scene);
    expect(celebrationsFor(scene)).not.toBe(a);
    disposeCelebrations(scene);
  });
});


describe('rig math', () => {
  test('damp is frame-rate independent and degenerate-safe', () => {
    expect(damp(8, 0)).toBe(0);
    expect(damp(0, 0.016)).toBe(1);
    expect(damp(8, 1 / 60)).toBeCloseTo(1 - Math.exp(-8 / 60), 12);
    expect(damp(8, 1)).toBeLessThan(1);
    expect(damp(8, 1)).toBeGreaterThan(damp(8, 0.5));
  });

  test('clampOrbit clamps polar, azimuth and distance in place', () => {
    const limits = { ...DEFAULT_ORBIT_LIMITS, minAzimuth: -1, maxAzimuth: 1 };
    const s = { azimuth: 5, polar: -3, distance: 1000 };
    const same = clampOrbit(s, limits);
    expect(same).toBe(s);
    expect(s.azimuth).toBe(1);
    expect(s.polar).toBe(limits.minPolar);
    expect(s.distance).toBe(limits.maxDistance);

    const t = { azimuth: -5, polar: 99, distance: 0 };
    clampOrbit(t, limits);
    expect(t.azimuth).toBe(-1);
    expect(t.polar).toBe(limits.maxPolar);
    expect(t.distance).toBe(limits.minDistance);
  });

  test('orbitPosition is the spherical placement around the target', () => {
    const out = new THREE.Vector3();
    const target = new THREE.Vector3(1, 2, 3);
    orbitPosition({ azimuth: 0, polar: Math.PI / 2, distance: 10 }, target, out);
    expect(out.x).toBeCloseTo(1, 9);
    expect(out.y).toBeCloseTo(2, 9);
    expect(out.z).toBeCloseTo(13, 9);

    orbitPosition({ azimuth: Math.PI / 2, polar: Math.PI / 2, distance: 4 }, target, out);
    expect(out.x).toBeCloseTo(5, 9);
    expect(out.z).toBeCloseTo(3, 9);

    orbitPosition({ azimuth: 0, polar: 0, distance: 6 }, target, out);
    expect(out.y).toBeCloseTo(8, 9);
  });

  test('OrbitRig keeps the desired state inside its limits under any input', () => {
    const rig = makeOrbitRig({ distance: 12, polar: 0.85, limits: { minDistance: 4, maxDistance: 20 } });
    expect(rig.camera.fov).toBe(TOKENS.fovHero);

    rig.orbit(100, 100);
    expect(rig.desired.polar).toBeLessThanOrEqual(rig.limits.maxPolar);
    expect(rig.desired.azimuth).toBeLessThanOrEqual(rig.limits.maxAzimuth);
    rig.orbit(-1000, -1000);
    expect(rig.desired.polar).toBeGreaterThanOrEqual(rig.limits.minPolar);
    expect(rig.desired.azimuth).toBeGreaterThanOrEqual(rig.limits.minAzimuth);

    rig.zoom(1e6);
    expect(rig.desired.distance).toBe(20);
    rig.zoom(-1e6);
    expect(rig.desired.distance).toBe(4);

    rig.setDesired(0.4, 0.9, 11);
    rig.snap();
    expect(rig.state).toEqual(rig.desired);
    const expected = orbitPosition(rig.desired, rig.target, new THREE.Vector3());
    expect(rig.camera.position.distanceTo(expected)).toBeLessThan(1e-9);

    rig.orbit(50, 50);
    for (let i = 0; i < 200; i++) rig.update(1 / 60);
    expect(rig.state.polar).toBeLessThanOrEqual(rig.limits.maxPolar + 1e-12);
    expect(rig.state.distance).toBeLessThanOrEqual(rig.limits.maxDistance + 1e-12);
    rig.resize(2);
    expect(rig.camera.aspect).toBe(2);
    rig.dispose();
  });

  test('TrackRig snaps to exactly offset-from-target with zero velocity', () => {
    const rig = makeTrackRig({ offset: [0, 1.5, 5], lookOffset: [0, 0.5, 0], damping: 4 });
    rig.setTarget(10, 0.7, 0);
    rig.snap();
    expect(rig.camera.position.x).toBeCloseTo(10, 9);
    expect(rig.camera.position.y).toBeCloseTo(2.2, 9);
    expect(rig.camera.position.z).toBeCloseTo(5, 9);
    expect(rig.velocity.length()).toBe(0);

    for (let i = 1; i <= 30; i++) {
      rig.setTarget(10 + i * 0.1, 0.7, 0);
      rig.update(1 / 60);
    }
    expect(rig.velocity.x).toBeGreaterThan(0);
    expect(rig.camera.position.x).toBeGreaterThan(10);

    rig.setTarget(-40, 0.7, 0);
    rig.snap();
    expect(rig.velocity.length()).toBe(0);
    expect(rig.camera.position.x).toBeCloseTo(-40, 9);
    rig.dispose();
  });

  test('TrackRig look-ahead is capped', () => {
    const rig = makeTrackRig({ offset: [0, 0, 4], lookAhead: 10, maxLookAhead: 2, velocityDamping: 1e6 });
    rig.setTarget(0, 0, 0);
    rig.snap();
    for (let i = 1; i <= 10; i++) {
      rig.setTarget(i * 1, 0, 0);
      rig.update(1 / 60);
    }
    expect(rig.camera.position.x - rig.target.x).toBeLessThanOrEqual(2 + 1e-6);
    rig.dispose();
  });

  test('FrameRig holds its authored frame and returns to it', () => {
    const rig = makeFrameRig({ position: [0, 2, 8], lookAt: [0, 1, 0], parallax: 0.5, damping: 6 });
    rig.snap();
    expect(rig.camera.position.x).toBeCloseTo(0, 9);
    expect(rig.camera.position.y).toBeCloseTo(2, 9);
    expect(rig.camera.position.z).toBeCloseTo(8, 9);

    const maxTravel = 0.5 * Math.sqrt(1 + 0.5 * 0.5);
    rig.setPointer(1, 0);
    rig.snap();
    expect(Math.abs(rig.camera.position.x)).toBeCloseTo(0.5, 9);
    expect(rig.camera.position.distanceTo(rig.base)).toBeLessThanOrEqual(maxTravel + 1e-9);

    rig.setPointer(0, 0);
    for (let i = 0; i < 200; i++) rig.update(1 / 60);
    expect(rig.camera.position.distanceTo(rig.base)).toBeLessThan(1e-6);

    rig.setPointer(50, -50);
    rig.snap();
    expect(rig.camera.position.distanceTo(rig.base)).toBeCloseTo(maxTravel, 9);
    rig.dispose();
  });

  test('a straight-down FrameRig has a defined basis (no NaN)', () => {
    const rig = makeFrameRig({ position: [0, 10, 0], lookAt: [0, 0, 0], parallax: 1 });
    rig.setPointer(1, 1);
    rig.snap();
    expect(Number.isFinite(rig.camera.position.x)).toBe(true);
    expect(Number.isFinite(rig.camera.position.y)).toBe(true);
    expect(Number.isFinite(rig.camera.position.z)).toBe(true);
    rig.dispose();
  });
});


describe('vegetation', () => {
  const area = { halfX: 20, halfZ: 20 };

  test('tier scale is the STYLE.md contract (x1 / x0.5 / x0.15)', () => {
    expect(vegetationTierScale('hero')).toBe(1);
    expect(vegetationTierScale(undefined)).toBe(1);
    expect(vegetationTierScale('medium')).toBe(0.5);
    expect(vegetationTierScale('thumb')).toBe(0.15);
  });

  test('grass is ONE draw call and its count follows the tier', () => {
    const hero = scatterGrass({ count: 400, area, seed: 5, color: 0x88aa55, tier: 'hero' });
    const medium = scatterGrass({ count: 400, area, seed: 5, color: 0x88aa55, tier: 'medium' });
    const thumb = scatterGrass({ count: 400, area, seed: 5, color: 0x88aa55, tier: 'thumb' });
    expect(hero.meshes.length).toBe(1);
    expect(hero.count).toBe(400);
    expect(medium.count).toBe(200);
    expect(thumb.count).toBe(60);
    hero.dispose();
    medium.dispose();
    thumb.dispose();
  });

  test('placement is deterministic from the seed and differs between seeds', () => {
    const a = scatterPlacements({ count: 50, area, seed: 11, color: 0 }, 'grass');
    const b = scatterPlacements({ count: 50, area, seed: 11, color: 0 }, 'grass');
    const c = scatterPlacements({ count: 50, area, seed: 12, color: 0 }, 'grass');
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  test('NOTHING is ever placed where exclude() says no', () => {
    const playfield = excludeRect(0, 0, 5.5, 5.5, 0.5);
    const handle = scatterTrees({ count: 600, area, seed: 3, color: 0x557733, exclude: playfield });
    let checked = 0;
    for (const mesh of handle.meshes) {
      for (let i = 0; i < mesh.count; i++) {
        const [x, , z] = instanceTranslation(mesh, i);
        expect(playfield(x, z)).toBe(false);
        checked++;
      }
    }
    expect(checked).toBe(handle.count);
    expect(handle.count).toBeGreaterThan(0);
    handle.dispose();
  });

  test('exclusion helpers compose', () => {
    const rect = excludeRect(0, 0, 2, 2, 0);
    const band = excludeBand(6, 1, 0);
    const both = excludeAny(rect, band, undefined);
    expect(both(0, 0)).toBe(true);
    expect(both(0, 6)).toBe(true);
    expect(both(10, 10)).toBe(false);
    expect(rect(2, 2)).toBe(true);
    expect(rect(2.01, 0)).toBe(false);
  });

  test('trees ship three archetypes — one draw call each — and retint without remodelling', () => {
    const handle = scatterTrees({ count: 120, area, seed: 9, color: 0x557733 });
    expect(handle.meshes.length).toBe(3);
    expect(handle.materials.length).toBe(3);
    let total = 0;
    for (const mesh of handle.meshes) total += mesh.count;
    expect(total).toBe(handle.count);
    expect(handle.count).toBe(120);

    const geometries = handle.meshes.map((m) => m.geometry);
    handle.retint(0xff0000, 0x0000ff);
    expect(handle.meshes.map((m) => m.geometry)).toEqual(geometries);
    const first = handle.materials[0] as THREE.MeshStandardMaterial;
    const last = handle.materials[2] as THREE.MeshStandardMaterial;
    expect(first.color.r).toBeGreaterThan(first.color.b);
    expect(last.color.b).toBeGreaterThan(last.color.r);

    handle.retint(0x00ff00);
    for (const m of handle.materials) expect((m as THREE.MeshStandardMaterial).color.g).toBeGreaterThan(0.9);
    handle.setTime(1.5);
    handle.setWind(0.1, 2);
    handle.dispose();
  });

  test('rocks and flowers place their instances too', () => {
    const rocks = scatterRocks({ count: 40, area, seed: 2, color: 0x776655 });
    expect(rocks.count).toBe(40);
    expect(rocks.meshes.length).toBe(2);
    rocks.dispose();

    const flowers = scatterFlowers({ count: 60, area, seed: 2, color: 0xffaacc });
    expect(flowers.count).toBe(60);
    flowers.dispose();
  });

  test('heightAt lifts instances onto the terrain', () => {
    const heightAt = (x: number): number => Math.sin(x) * 0.5 + 1;
    const handle = scatterRocks({ count: 25, area, seed: 4, color: 0x776655, heightAt });
    for (const mesh of handle.meshes) {
      for (let i = 0; i < mesh.count; i++) {
        const [x, y] = instanceTranslation(mesh, i);
        expect(y).toBeCloseTo(heightAt(x), 5);
      }
    }
    handle.dispose();
  });

  test('archetype geometries are non-degenerate and carry vertex colors', () => {
    for (const g of [
      grassClumpGeometry(),
      broadleafTreeGeometry(),
      coniferTreeGeometry(),
      bareTreeGeometry(),
      rockGeometry(),
      flowerGeometry(),
    ]) {
      const pos = g.getAttribute('position');
      const col = g.getAttribute('color');
      expect(pos.count).toBeGreaterThan(2);
      expect(col).toBeDefined();
      expect(col.count).toBe(pos.count);
      for (let i = 0; i < pos.count; i++) {
        expect(Number.isFinite(pos.getX(i))).toBe(true);
        expect(Number.isFinite(pos.getY(i))).toBe(true);
        expect(Number.isFinite(pos.getZ(i))).toBe(true);
      }
      g.dispose();
    }
  });

  test('an area whose whole extent is excluded places nothing (and does not hang)', () => {
    const handle = scatterGrass({
      count: 200,
      area: { halfX: 2, halfZ: 2 },
      seed: 1,
      color: 0x88aa55,
      exclude: () => true,
    });
    expect(handle.count).toBe(0);
    handle.dispose();
  });
});


describe('particles', () => {
  test('tier scale turns particles OFF at thumb', () => {
    expect(particleTierScale('hero')).toBe(1);
    expect(particleTierScale('medium')).toBe(0.5);
    expect(particleTierScale('thumb')).toBe(0);
  });

  test('every preset builds a single Points with per-particle attributes', () => {
    for (const preset of Object.keys(PARTICLE_PRESETS) as (keyof typeof PARTICLE_PRESETS)[]) {
      const handle = makeAmbientParticles({
        preset,
        seed: 3,
        tier: 'hero',
        bounds: { halfX: 8, halfY: 2, halfZ: 8, y: 1 },
      });
      expect(handle.count).toBe(PARTICLE_PRESETS[preset].count);
      const geo = handle.points.geometry;
      expect(geo.getAttribute('position').count).toBe(handle.count);
      expect(geo.getAttribute('aSeed').count).toBe(handle.count);
      expect(geo.getAttribute('aPhase').count).toBe(handle.count);
      expect(geo.getAttribute('aSize').count).toBe(handle.count);
      expect(handle.points.visible).toBe(true);
      handle.setTime(2);
      expect((handle.material.uniforms.uTime as { value: number }).value).toBe(2);
      handle.setDensity(2);
      expect((handle.material.uniforms.uDensity as { value: number }).value).toBe(1);
      handle.setDensity(-1);
      expect((handle.material.uniforms.uDensity as { value: number }).value).toBe(0);
      handle.setColor(0x123456);
      handle.setIntensity(2);
      handle.setPixelRatio(2);
      handle.dispose();
    }
  });

  test('a thumb field is empty and invisible but still a usable handle', () => {
    const handle = makeAmbientParticles({
      preset: 'pollen',
      seed: 1,
      tier: 'thumb',
      bounds: { halfX: 4, halfY: 1, halfZ: 4 },
    });
    expect(handle.count).toBe(0);
    expect(handle.points.visible).toBe(false);
    handle.setTime(1);
    handle.dispose();
  });

  test('particle placement is deterministic from the seed and inside the bounds', () => {
    const bounds = { halfX: 5, halfY: 2, halfZ: 3 };
    const a = makeAmbientParticles({ preset: 'dust', seed: 42, bounds, count: 50 });
    const b = makeAmbientParticles({ preset: 'dust', seed: 42, bounds, count: 50 });
    const pa = a.points.geometry.getAttribute('position').array as Float32Array;
    const pb = b.points.geometry.getAttribute('position').array as Float32Array;
    expect(Array.from(pa)).toEqual(Array.from(pb));
    for (let i = 0; i < 50; i++) {
      expect(Math.abs(pa[i * 3] as number)).toBeLessThanOrEqual(bounds.halfX);
      expect(Math.abs(pa[i * 3 + 1] as number)).toBeLessThanOrEqual(bounds.halfY);
      expect(Math.abs(pa[i * 3 + 2] as number)).toBeLessThanOrEqual(bounds.halfZ);
    }
    a.dispose();
    b.dispose();
  });

  test('mist planes stack and switch off at thumb', () => {
    const hero = makeMistPlanes({ size: 20, color: 0xd9e8ee, seed: 8, tier: 'hero', layers: 5 });
    expect(hero.layers).toBe(5);
    expect(hero.group.children.length).toBe(5);
    hero.setTime(1);
    hero.setOpacity(0.5);
    hero.setColor(0xffffff);
    hero.dispose();

    const thumb = makeMistPlanes({ size: 20, color: 0xd9e8ee, seed: 8, tier: 'thumb', layers: 5 });
    expect(thumb.layers).toBe(0);
    expect(thumb.group.visible).toBe(false);
    thumb.dispose();
  });
});

describe('water', () => {
  test('segment scale is tier policy', () => {
    expect(waterSegmentScale('hero')).toBe(1);
    expect(waterSegmentScale('medium')).toBeCloseTo(0.55, 12);
    expect(waterSegmentScale('thumb')).toBeCloseTo(0.22, 12);
  });

  test('the CPU height sampler is deterministic, bounded and time-varying', () => {
    const amp = 0.05;
    const a = waterHeightAt(1.3, -2.7, 4.2, amp, 1, 1);
    const b = waterHeightAt(1.3, -2.7, 4.2, amp, 1, 1);
    expect(a).toBe(b);
    expect(Math.abs(a)).toBeLessThanOrEqual(amp * (0.55 + 0.3 + 0.11 + 0.06) + 1e-12);
    expect(waterHeightAt(1.3, -2.7, 9.9, amp, 1, 1)).not.toBe(a);
    expect(waterHeightAt(0, 0, 0, amp, 1, 1)).toBeCloseTo(0, 12);
  });

  test('makeWater exposes a surface whose sampleHeight tracks the mesh level', () => {
    const water = makeWater({ size: 12, palette: TOKENS.palettes.dawn, tier: 'medium', amplitude: 0.04, level: 1.5 });
    expect(water.mesh.position.y).toBe(1.5);
    const h = water.sampleHeight(2, 3, 1.25);
    expect(h).toBeCloseTo(1.5 + waterHeightAt(2, 3, 1.25, 0.04, 1, 1), 9);

    water.setLevel(0);
    expect(water.sampleHeight(2, 3, 1.25)).toBeCloseTo(waterHeightAt(2, 3, 1.25, 0.04, 1, 1), 9);
    water.setTime(3);
    expect(water.uniforms.uTime.value).toBe(3);
    water.setColors(0x112233, 0x001122);
    water.setSunDirection(0, 0, 0);
    expect(water.uniforms.uSunDir.value.length()).toBeCloseTo(1, 9);
    water.dispose();
  });
});


describe('GroundTruthOverlay', () => {
  const grid = handGrid();
  const geometry: EnvGeometry = { kind: 'grid', grid: { ...grid, goal: { x: 4, y: 3 }, doors: [{ x: 2, y: 2 }] } };

  test('everything lives on layer 31 and starts hidden', () => {
    const overlay = new GroundTruthOverlay(geometry);
    expect(overlay.visible).toBe(false);
    expect(overlay.group.visible).toBe(false);
    let count = 0;
    overlay.group.traverse((o) => {
      expect(o.layers.mask).toBe((1 << DEBUG_LAYER) >>> 0);
      count++;
    });
    expect(count).toBeGreaterThan(1);
    overlay.dispose();
  });

  test('the visible setter opts the given camera in and out of the layer', () => {
    const camera = new THREE.PerspectiveCamera();
    const overlay = new GroundTruthOverlay(geometry, { camera });
    const probe = new THREE.Object3D();
    probe.layers.set(DEBUG_LAYER);

    expect(camera.layers.test(probe.layers)).toBe(false);
    overlay.visible = true;
    expect(camera.layers.test(probe.layers)).toBe(true);
    expect(overlay.group.visible).toBe(true);
    overlay.visible = false;
    expect(camera.layers.test(probe.layers)).toBe(false);
    overlay.dispose();
  });

  test('the agent cross lands on the TRUE cell from the frame, not the art', () => {
    const overlay = new GroundTruthOverlay(geometry);
    overlay.setState(makeState([], { x: 3, y: 1 }));
    const cross = overlay.group.getObjectByName('oakDebugAgentCross') as THREE.LineSegments;
    const p = cross.geometry.getAttribute('position').array as Float32Array;
    const [wx, , wz] = cellToWorld(3, 1, grid);
    expect(p[12] as number).toBeCloseTo(wx, 9);
    expect(p[14] as number).toBeCloseTo(wz, 9);
    expect(p[16] as number).toBeGreaterThan(p[13] as number);
    expect(cross.visible).toBe(true);
    overlay.dispose();
  });

  test('the cell sniffers cover {x,y}, {r,c} and {pos} decoders', () => {
    const overlay = new GroundTruthOverlay(geometry);
    const cross = overlay.group.getObjectByName('oakDebugAgentCross') as THREE.LineSegments;
    const p = cross.geometry.getAttribute('position').array as Float32Array;

    overlay.setState(makeState([], { r: 2, c: 1, lastA: 0 }));
    let expected = cellToWorld(1, 2, grid);
    expect(p[12] as number).toBeCloseTo(expected[0], 9);
    expect(p[14] as number).toBeCloseTo(expected[2], 9);

    overlay.setState(makeState([], { pos: 3, len: 5 }));
    expected = cellToWorld(3, 0, grid);
    expect(p[12] as number).toBeCloseTo(expected[0], 9);
    expect(p[14] as number).toBeCloseTo(expected[2], 9);

    overlay.setState(makeState([2, 3]));
    expected = cellToWorld(2, 3, grid);
    expect(p[12] as number).toBeCloseTo(expected[0], 9);
    expect(p[14] as number).toBeCloseTo(expected[2], 9);
    overlay.dispose();
  });

  test('an explicit resolver always wins over the sniffers', () => {
    const overlay = new GroundTruthOverlay(geometry, { agentCell: () => ({ x: 0, y: 3 }) });
    overlay.setState(makeState([], { x: 9, y: 9 }));
    const cross = overlay.group.getObjectByName('oakDebugAgentCross') as THREE.LineSegments;
    const p = cross.geometry.getAttribute('position').array as Float32Array;
    const expected = cellToWorld(0, 3, grid);
    expect(p[12] as number).toBeCloseTo(expected[0], 9);
    expect(p[14] as number).toBeCloseTo(expected[2], 9);
    overlay.dispose();
  });

  test('the goal marker tracks the LIVE respawning goal', () => {
    const overlay = new GroundTruthOverlay(geometry);
    overlay.setState(makeState([], { x: 1, y: 1, gx: 4, gy: 2 }));
    const goal = overlay.group.getObjectByName('oakDebugGoal') as THREE.LineSegments;
    const p = goal.geometry.getAttribute('position').array as Float32Array;
    const [gx, , gz] = cellToWorld(4, 2, grid);
    expect(p[0] as number).toBeCloseTo(gx, 5);
    expect((p[2] as number) + 0.42).toBeCloseTo(gz, 5);
    expect(goal.visible).toBe(true);
    overlay.dispose();
  });

  test('DeepSea action-mask arrows and a carved path are drawn when present', () => {
    const mask = new Uint8Array(grid.w * grid.h);
    mask.fill(1);
    const withExtras: EnvGeometry = {
      kind: 'grid',
      grid: { ...grid, actionMask: mask, carvedPath: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }] },
    };
    const overlay = new GroundTruthOverlay(withExtras);
    expect(overlay.group.getObjectByName('oakDebugMaskArrows')).toBeDefined();
    expect(overlay.group.getObjectByName('oakDebugCarvedPath')).toBeDefined();
    overlay.dispose();

    const plain = new GroundTruthOverlay(geometry);
    expect(plain.group.getObjectByName('oakDebugMaskArrows')).toBeUndefined();
    expect(plain.group.getObjectByName('oakDebugCarvedPath')).toBeUndefined();
    plain.dispose();
  });

  test('classic geometry gets a terrain polyline plus one cross per anchor', () => {
    const classic: EnvGeometry = {
      kind: 'classic',
      classic: {
        heightAt: (x: number) => Math.sin(3 * x) * 0.45 + 0.55,
        anchors: {
          trackLeft: [-1.2, 0.05, 0],
          trackRight: [0.6, 0.62, 0],
          goalFlag: [0.5, 0.55, 0],
        },
      },
    };
    const overlay = new GroundTruthOverlay(classic, { agentPos: () => [0.1, 0.7, 0] });
    const profile = overlay.group.getObjectByName('oakDebugProfile') as THREE.LineSegments;
    expect(profile).toBeDefined();
    const p = profile.geometry.getAttribute('position').array as Float32Array;
    expect(p[0] as number).toBeCloseTo(-1.2, 5);
    expect(p[1] as number).toBeCloseTo(Math.sin(3 * -1.2) * 0.45 + 0.55, 5);
    expect(overlay.group.getObjectByName('oakDebugAnchors')).toBeDefined();

    const cross = overlay.group.getObjectByName('oakDebugClassicAgent') as THREE.LineSegments;
    expect(cross.visible).toBe(false);
    overlay.setState(makeState([0.1, 0]));
    expect(cross.visible).toBe(true);
    overlay.dispose();
  });

  test('cheetah geometry gets a stick skeleton driven by the real FK', () => {
    let calls = 0;
    const cheetah: EnvGeometry = {
      kind: 'cheetah',
      cheetah: {
        radius: 0.046,
        headRadius: 0.15,
        fk: (_qpos, out) => {
          calls++;
          const dst = out ?? new Float32Array(20);
          for (let i = 0; i < 20; i++) dst[i] = i * 0.1;
          return dst;
        },
      },
    };
    const overlay = new GroundTruthOverlay(cheetah);
    const skeleton = overlay.group.getObjectByName('oakDebugSkeleton') as THREE.LineSegments;
    expect(skeleton).toBeDefined();
    overlay.setState(makeState([0, 0, 0, 0, 0, 0, 0, 0, 0]));
    expect(calls).toBe(1);
    const p = skeleton.geometry.getAttribute('position').array as Float32Array;
    expect(p[2] as number).toBe(0);
    expect(Number.isFinite(p[0] as number)).toBe(true);
    expect(overlay.group.getObjectByName('oakDebugToes')).toBeDefined();
    overlay.dispose();
  });

  test('layer helpers', () => {
    const o = new THREE.Group();
    o.add(new THREE.Object3D());
    applyDebugLayer(o);
    o.traverse((c) => expect(c.layers.mask).toBe((1 << DEBUG_LAYER) >>> 0));

    const camera = new THREE.PerspectiveCamera();
    enableDebugLayer(camera);
    expect(camera.layers.test(o.layers)).toBe(true);

    const out: [number, number, number] = [0, 0, 0];
    expect(defaultFkToWorld(1.5, 2.5, out)).toEqual([1.5, 2.5, 0]);
  });
});


function fakeFrame(step: number, x: number): EnvFrame<{ x: number }> {
  const pose = new Float32Array(9);
  pose[0] = x;
  pose[1] = x * 2;
  return { step, ctrl: new Float32Array(6), done: step === 2 ? 1 : 0, pose, state: { x } };
}

function fakeSource(discretePose: boolean): HarnessSource {
  const frames = [fakeFrame(-1, 0), fakeFrame(0, 10), fakeFrame(1, 20), fakeFrame(2, 30)];
  const events: SimEvent[][] = [[], [{ type: 'goal' }], [], [{ type: 'done' }]];
  return {
    kind: 'fixture',
    envId: 'FourRooms-cont-v0',
    seed: 0,
    controlHz: 10,
    discretePose,
    geometry: { kind: 'grid', grid: handGrid() },
    frameCount: 4,
    frameAt: (i: number) => frames[Math.max(0, Math.min(3, Math.floor(i)))]!,
    eventsAt: (i: number) => events[Math.floor(i)] ?? [],
    eventEntries: () => new Map<number, SimEvent[]>([[1, [{ type: 'goal' }]], [3, [{ type: 'done' }]]]).entries(),
    label: 'test',
  };
}

describe('harness playback', () => {
  test('the playback clock maps seconds to frame positions at controlHz', () => {
    const clock = makePlaybackClock(11, 10);
    expect(clock.duration).toBeCloseTo(1, 12);
    expect(clock.position()).toBe(0);
    expect(clock.advance(0.5)).toBeCloseTo(5, 9);
    expect(clock.time()).toBeCloseTo(0.5, 9);

    clock.seek(3);
    expect(clock.time()).toBeCloseTo(0.3, 9);
    clock.seek(-5);
    expect(clock.time()).toBe(0);
    clock.seek(1e6);
    expect(clock.position()).toBeCloseTo(10, 9);

    clock.seek(2);
    clock.playing = false;
    expect(clock.advance(10)).toBeCloseTo(2, 9);
    clock.playing = true;
    clock.speed = 2;
    expect(clock.advance(0.1)).toBeCloseTo(4, 9);
  });

  test('playback wraps at the end instead of stalling', () => {
    const clock = makePlaybackClock(11, 10);
    clock.advance(0.99);
    clock.advance(0.5);
    expect(clock.time()).toBeGreaterThanOrEqual(0);
    expect(clock.time()).toBeLessThanOrEqual(clock.duration);
  });

  test('continuous envs interpolate the pose; discrete envs SNAP', () => {
    const out = makeState([]);

    sampleFrame(fakeSource(false), 1.5, out);
    expect(out.pose[0]).toBeCloseTo(15, 9);
    expect(out.pose[1]).toBeCloseTo(30, 9);

    sampleFrame(fakeSource(true), 1.5, out);
    expect(out.pose[0]).toBe(10);
    expect(out.pose[1]).toBe(20);
  });

  test('sampleFrame clamps to the recording and reports the record step', () => {
    const source = fakeSource(false);
    const out = makeState([]);

    sampleFrame(source, -10, out);
    expect(out.step).toBe(-1);
    expect(out.simTime).toBeCloseTo(-0.1, 9);
    expect(out.derived).toEqual({ x: 0 });

    sampleFrame(source, 1e6, out);
    expect(out.step).toBe(2);
    expect(out.done).toBe(true);
    expect(out.pose[0]).toBe(30);

    sampleFrame(source, 1, out);
    expect(out.step).toBe(0);
    expect(out.done).toBe(false);
    expect(out.simTime).toBeCloseTo(0, 9);
  });

  test('firstRecordingFor picks the lowest seed for the env, or null', () => {
    const recs = [
      { envId: 'DeepSea-cont-v0' as const, seed: 16, steps: 10 },
      { envId: 'FourRooms-cont-v0' as const, seed: 9, steps: 10 },
      { envId: 'FourRooms-cont-v0' as const, seed: 8, steps: 10 },
    ];
    expect(firstRecordingFor(recs, 'FourRooms-cont-v0')?.seed).toBe(8);
    expect(firstRecordingFor(recs, 'DeepSea-cont-v0')?.seed).toBe(16);
    expect(firstRecordingFor(recs, 'HalfCheetah-v4')).toBe(null);
    expect(firstRecordingFor([], 'HalfCheetah-v4')).toBe(null);
  });

  test('the catch-up budget is a positive frame count', () => {
    expect(MAX_CATCHUP_FRAMES).toBeGreaterThan(0);
    expect(Number.isInteger(MAX_CATCHUP_FRAMES)).toBe(true);
  });
});
