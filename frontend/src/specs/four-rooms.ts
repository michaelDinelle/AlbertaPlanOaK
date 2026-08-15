import * as THREE from 'three';
import { CELL, cellToWorldVec } from '@kit/grid';
import type { GridCell } from '@sim/types';
import { oakMaterial } from '@kit/materials';
import { TOKENS } from '@kit/tokens';
import { Wisp } from '@kit/wisp';
import { assemble } from '@kit/gen/assembly';
import { field } from '@kit/gen/field';
import { patch } from '@kit/gen/patch';
import { circleOutline, polygonOutline } from '@kit/gen/outline';
import { curvePath, revolve, sweep } from '@kit/gen/sweep';
import type { EnvSpec } from '../engine/spec';

const SEASON_TINT = [0xf7c9d4, 0x9fd47a, 0xe0a35e, 0xdfe9f2] as const;
const SEASON_SCATTER = [0xffd6e3, 0x7fb85a, 0xd98b3e, 0xffffff] as const;

function roomOf(x: number, y: number, w: number, h: number): number {
  return (x > (w - 1) / 2 ? 1 : 0) + (y > (h - 1) / 2 ? 2 : 0);
}

function moonGate(): THREE.BufferGeometry {
  const ring = sweep({
    path: curvePath([
      new THREE.Vector3(-0.42, 0, 0),
      new THREE.Vector3(-0.3, 0.52, 0),
      new THREE.Vector3(0, 0.7, 0),
      new THREE.Vector3(0.3, 0.52, 0),
      new THREE.Vector3(0.42, 0, 0),
    ]),
    rings: 22,
    shape: circleOutline(7),
    radius: 0.085,
  });
  return assemble(
    [
      { geometry: ring, color: 0x9aa7b2 },
      {
        geometry: revolve([[0.16, 0], [0.13, 0.16], [0, 0.2]], 8),
        color: 0x808e99,
        position: [-0.42, 0, 0],
      },
      {
        geometry: revolve([[0.16, 0], [0.13, 0.16], [0, 0.2]], 8),
        color: 0x808e99,
        position: [0.42, 0, 0],
      },
    ],
    'moonGate',
  );
}

function sundial(): THREE.BufferGeometry {
  return assemble(
    [
      { geometry: revolve([[0.42, 0], [0.4, 0.09], [0.3, 0.12], [0, 0.13]], 16), color: 0xb9c2c8 },
      {
        geometry: sweep({
          path: curvePath([new THREE.Vector3(0, 0.13, 0), new THREE.Vector3(0.14, 0.5, 0)]),
          rings: 4,
          shape: polygonOutline(3, 1),
          radius: 0.05,
        }),
        color: 0x8b949b,
      },
    ],
    'sundial',
  );
}

function lanternFlower(petals: number): THREE.BufferGeometry {
  const parts = [
    {
      geometry: sweep({
        path: curvePath([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.42, 0)]),
        rings: 6,
        shape: circleOutline(6),
        radius: (t: number) => 0.045 * (1 - 0.35 * t),
      }),
      color: 0x6f9a4e,
      position: [0, 0, 0] as const,
    },
  ];
  for (let i = 0; i < petals; i++) {
    const a = (i / petals) * Math.PI * 2;
    parts.push({
      geometry: patch({
        segmentsU: 6,
        segmentsV: 8,
        doubleSided: true,
        point(u, v, out) {
          const spread = Math.sin(Math.PI * v) * 0.16;
          out.set((u - 0.5) * 2 * spread, v * 0.34, Math.sin(v * 2.1) * 0.09);
        },
      }),
      color: 0xf7a8b8,
      position: [Math.cos(a) * 0.06, 0.4, Math.sin(a) * 0.06] as const,
      rotation: [0.55, -a, 0] as const,
    } as never);
  }
  return assemble(parts as never, 'lanternFlower');
}

export const fourRooms: EnvSpec = {
  envId: 'FourRooms-cont-v0',
  title: 'Four Seasons Courtyard',
  logline:
    'A noon walled garden where the four rooms are literally four seasons — blossom, lawn, '
    + 'leaf-fall and frost — joined by stone moon-gates around a sundial, with a lantern-flower '
    + 'burning on whichever cell the goal has respawned to.',
  palette: 'noon',
  accent: 'four-rooms',
  ground: {
    kind: 'cells',
    extent: 40,
    floorTint: (x, y) => SEASON_TINT[roomOf(x, y, 11, 11)] ?? 0x9fd47a,
    wallTint: () => 0xb8c2b0,
    height: (x, z) => {
      const r = Math.hypot(x, z);
      return r < 8 ? 0 : -0.02 * (r - 8) + Math.sin(x * 0.31) * Math.cos(z * 0.27) * 0.22;
    },
  },
  camera: { target: [0, 0.5, 0], azimuth: 0.42, polar: 0.75, distance: 17.8 },
  bloom: 0.45,
  fogExtent: 520,

  build(ctx) {
    const grid = ctx.geometry.grid;
    if (!grid) return;
    const palette = TOKENS.palettes[fourRooms.palette];
    const owned: { dispose(): void }[] = [];
    const geoms: THREE.BufferGeometry[] = [];

    const stone = oakMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.86,
      toonSteps: 3,
      rimColor: palette.sun,
      rimStrength: 0.5,
      tier: ctx.tier,
      name: 'courtyardStone',
    });

    const seasonAt = (x: number, y: number): number => roomOf(x, y, grid.w, grid.h);

    const blade = patch({
      segmentsU: 1,
      segmentsV: 3,
      doubleSided: true,
      point(u, v, out) {
        out.set((u - 0.5) * 0.055 * (1 - v), v * 0.34, Math.sin(v * 2.4) * 0.05);
      },
    });
    geoms.push(blade);

    for (let room = 0; room < 4; room++) {
      const halfW = ((grid.w - 1) / 2) * CELL * 0.5;
      const halfH = ((grid.h - 1) / 2) * CELL * 0.5;
      const cx = (room % 2 === 0 ? -1 : 1) * halfW;
      const cz = (room < 2 ? -1 : 1) * halfH;
      const scatter = field({
        source: { kind: 'rect', halfX: halfW * 0.82, halfZ: halfH * 0.82, centre: [cx, 0.02, cz] },
        count: Math.round(120 * ctx.detail.counts),
        seed: ctx.seed,
        stream: `season${room}`,
        variants: [blade],
        material: stone,
        scale: [0.7, 1.35],
        tilt: 0.16,
        color: (_i, out) => out.set(SEASON_SCATTER[room] ?? 0xffffff),
        reject: (x, _y, z) => {
          const gx = Math.round(x / CELL + (grid.w - 1) / 2);
          const gy = Math.round(z / CELL + (grid.h - 1) / 2);
          return grid.walls[gy * grid.w + gx] === 1;
        },
        name: `season${room}`,
      });
      ctx.group.add(scatter.group);
      owned.push(scatter);
    }

    const trunk = sweep({
      path: curvePath([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.05, 1.1, 0.03)]),
      rings: 5,
      shape: circleOutline(6),
      radius: (t: number) => 0.09 * (1 - 0.45 * t),
    });
    const crownRound = revolve([[0, 0], [0.62, 0.34], [0.5, 0.86], [0, 1.05]], 9);
    const crownCone = revolve([[0, 0], [0.5, 0.2], [0.26, 0.8], [0, 1.2]], 8);
    const treeRound = assemble(
      [
        { geometry: trunk, color: 0x6b543a },
        { geometry: crownRound, color: 0x4f7a3a, position: [0, 1.0, 0] },
      ],
      'treeRound',
    );
    const treeCone = assemble(
      [
        { geometry: trunk, color: 0x6b543a },
        { geometry: crownCone, color: 0x3c6633, position: [0, 0.95, 0] },
      ],
      'treeCone',
    );
    trunk.dispose();
    crownRound.dispose();
    crownCone.dispose();
    geoms.push(treeRound, treeCone);

    const ring = field({
      source: { kind: 'rect', halfX: 18, halfZ: 18 },
      count: Math.round(150 * ctx.detail.counts),
      seed: ctx.seed,
      stream: 'courtyardTrees',
      variants: [treeRound, treeCone],
      material: stone,
      scale: [0.85, 1.7],
      yaw: true,
      height: (x, z) => fourRooms.ground.height?.(x, z) ?? 0,
      color: (i, out) => out.setHSL(0.28 + ((i * 37) % 11) * 0.004, 0.34, 0.3),
      reject: (x, _y, z) => Math.abs(x) < 7.4 && Math.abs(z) < 7.4,
      name: 'courtyardTrees',
    });
    ctx.group.add(ring.group);
    owned.push(ring);

    const gateGeom = moonGate();
    geoms.push(gateGeom);
    for (const hall of grid.hallways ?? []) {
      const mesh = new THREE.Mesh(gateGeom, stone);
      cellToWorldVec(hall.x, hall.y, grid, mesh.position);
      mesh.rotation.y = hall.x === Math.round((grid.w - 1) / 2) ? 0 : Math.PI / 2;
      ctx.group.add(mesh);
    }

    const dialGeom = sundial();
    geoms.push(dialGeom);
    const dial = new THREE.Mesh(dialGeom, stone);
    ctx.group.add(dial);

    const flowerGeom = lanternFlower(6);
    geoms.push(flowerGeom);
    const flower = new THREE.Mesh(flowerGeom, stone);
    ctx.group.add(flower);

    const wisp = new Wisp(
      {
        color: TOKENS.accents['four-rooms'] ?? TOKENS.celebrationGold,
        tier: ctx.tier,
        radius: 0.21,
        hop: { duration: 0.11, height: 0.17 },
        name: 'courtyardWisp',
      },
      grid,
    );
    ctx.scene.add(wisp.group);
    owned.push(wisp);

    const agent: GridCell = { x: 0, y: 0 };
    const goal: GridCell = { x: 0, y: 0 };
    void roomOf;
    void SEASON_TINT;

    return {
      update(frame, dt, time) {
        void seasonAt;
        frame.cell('agent', agent);
        frame.cell('goal', goal);
        if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) wisp.hopTo(agent, time);
        wisp.update(dt, time);
        cellToWorldVec(goal.x, goal.y, grid, flower.position);
        flower.scale.setScalar(0.9 + 0.1 * Math.sin(time * 1.4));
        dial.rotation.y = time * 0.06;
      },
      dispose() {
        for (const o of owned) o.dispose();
        for (const g of geoms) g.dispose();
        stone.dispose();
      },
    };
  },
};

export default fourRooms;
