import * as THREE from 'three';
import { CELL, cellToWorldVec } from '@kit/grid';
import { oakMaterial } from '@kit/materials';
import { TOKENS } from '@kit/tokens';
import { assemble } from '@kit/gen/assembly';
import { field } from '@kit/gen/field';
import { patch } from '@kit/gen/patch';
import { circleOutline } from '@kit/gen/outline';
import { curvePath, revolve, sweep } from '@kit/gen/sweep';
import type { EnvSpec } from '../engine/spec';

const BIT_ROWS = 4;

function rock(): THREE.BufferGeometry {
  return revolve([[0, 0], [0.34, 0.12], [0.29, 0.3], [0.12, 0.42], [0, 0.45]], 6);
}

function reed(): THREE.BufferGeometry {
  return patch({
    segmentsU: 1,
    segmentsV: 3,
    doubleSided: true,
    point(u, v, out) {
      out.set((u - 0.5) * 0.05 * (1 - v), v * 0.5, Math.sin(v * 2.2) * 0.07);
    },
  });
}

export const highDimMaze: EnvSpec = {
  envId: 'HighDimMaze-cont-v0',
  title: 'The Slot Canyons',
  logline:
    'Three terracotta bands split by two fins with a single gap each. The canyon walls carry the '
    + "agent's own coordinates as a lit four-bit readout, because what makes this maze high-dimensional "
    + 'is not the maze — it is the observation.',
  palette: 'afternoon',
  accent: 'high-dim-maze',
  ground: {
    kind: 'cells',
    extent: 60,
    color: 0xc08a5e,
    height: (x, z) => Math.sin(x * 0.14) * 0.5 + Math.cos(z * 0.11) * 0.4,
    floorTint: () => 0x8a7059,
    wallTint: () => 0xc98a5e,
  },
  camera: { target: [0, 0.4, 0], azimuth: 0.5, polar: 0.72, distance: 19 },
  bloom: 0.4,
  fogExtent: 520,

  build(ctx) {
    const grid = ctx.geometry.grid;
    if (!grid) return;
    const owned: { dispose(): void }[] = [];
    const geoms: THREE.BufferGeometry[] = [];

    const stone = oakMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.92,
      toonSteps: 3,
      rimColor: TOKENS.palettes[highDimMaze.palette].sun,
      rimStrength: 0.45,
      tier: ctx.tier,
      name: 'canyonStone',
    });

    const rockGeo = rock();
    const reedGeo = reed();
    geoms.push(rockGeo, reedGeo);

    const scatter = field({
      source: { kind: 'rect', halfX: 26, halfZ: 26 },
      count: Math.round(260 * ctx.detail.counts),
      seed: ctx.seed,
      stream: 'canyonRocks',
      variants: [rockGeo, reedGeo],
      material: stone,
      scale: [0.5, 1.9],
      height: (x, z) => highDimMaze.ground.height?.(x, z) ?? 0,
      color: (i, out) => out.setHSL(0.03 + ((i * 29) % 7) * 0.004, 0.42, 0.34),
      reject: (x, _y, z) => Math.abs(x) < 6.2 && Math.abs(z) < 6.2,
      name: 'canyonScatter',
    });
    ctx.group.add(scatter.group);
    owned.push(scatter);

    const archGeo = assemble(
      [
        {
          geometry: sweep({
            path: curvePath([
              new THREE.Vector3(-0.55, 0, 0),
              new THREE.Vector3(-0.4, 0.8, 0),
              new THREE.Vector3(0, 1.05, 0),
              new THREE.Vector3(0.4, 0.8, 0),
              new THREE.Vector3(0.55, 0, 0),
            ]),
            rings: 20,
            shape: circleOutline(6),
            radius: 0.13,
          }),
          color: 0xd9a271,
        },
      ],
      'canyonArch',
    );
    geoms.push(archGeo);
    for (const hall of grid.hallways ?? []) {
      const mesh = new THREE.Mesh(archGeo, stone);
      cellToWorldVec(hall.x, hall.y, grid, mesh.position);
      mesh.rotation.y = Math.PI / 2;
      ctx.group.add(mesh);
    }

    const bitGeo = new THREE.BoxGeometry(0.16, 0.16, 0.06);
    geoms.push(bitGeo);
    const bitMaterial = oakMaterial({
      color: 0xffd166,
      emissive: 0xffd166,
      roughness: 0.4,
      toonSteps: 2,
      tier: ctx.tier,
      name: 'canyonBits',
    });
    const bits: THREE.Mesh[] = [];
    const half = ((grid.h - 1) / 2) * CELL;
    for (let axis = 0; axis < 2; axis++) {
      for (let b = 0; b < BIT_ROWS; b++) {
        const mesh = new THREE.Mesh(bitGeo, bitMaterial);
        mesh.position.set(
          (axis === 0 ? -1 : 1) * (half + 0.55),
          1.15 - b * 0.24,
          (b - (BIT_ROWS - 1) / 2) * 0.34,
        );
        ctx.group.add(mesh);
        bits.push(mesh);
      }
    }

    return {
      update(frame) {
        const x = Math.round(frame.get('x'));
        const y = Math.round(frame.get('y'));
        for (let axis = 0; axis < 2; axis++) {
          const value = axis === 0 ? x : y;
          for (let b = 0; b < BIT_ROWS; b++) {
            const mesh = bits[axis * BIT_ROWS + b];
            if (mesh === undefined) continue;
            const on = ((value >> b) & 1) === 1;
            mesh.scale.setScalar(on ? 1 : 0.34);
          }
        }
      },
      dispose() {
        for (const o of owned) o.dispose();
        for (const g of geoms) g.dispose();
        stone.dispose();
        bitMaterial.dispose();
      },
    };
  },
};

export default highDimMaze;
