import * as THREE from 'three';
import { CELL, cellToWorldVec } from '@kit/grid';
import { oakMaterial } from '@kit/materials';
import { makeWater, type WaterHandle } from '@kit/water';
import { TOKENS } from '@kit/tokens';
import { assemble } from '@kit/gen/assembly';
import { field } from '@kit/gen/field';
import { patch } from '@kit/gen/patch';
import { circleOutline, polygonOutline } from '@kit/gen/outline';
import { curvePath, revolve, sweep } from '@kit/gen/sweep';
import type { EnvSpec } from '../engine/spec';

function blade(height: number, width: number): THREE.BufferGeometry {
  return patch({
    segmentsU: 1,
    segmentsV: 3,
    doubleSided: true,
    point(u, v, out) {
      out.set((u - 0.5) * width * (1 - v), v * height, Math.sin(v * 2.3) * height * 0.16);
    },
  });
}

function blossomTree(): THREE.BufferGeometry {
  const trunk = sweep({
    path: curvePath([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.12, 0.7, 0.05),
      new THREE.Vector3(-0.05, 1.4, -0.08),
    ]),
    rings: 8,
    shape: circleOutline(6),
    radius: (t) => 0.14 * (1 - 0.6 * t),
  });
  const parts: { geometry: THREE.BufferGeometry; color: number; position?: readonly [number, number, number] }[] = [
    { geometry: trunk, color: 0x6a4f3a },
  ];
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    parts.push({
      geometry: revolve([[0, 0], [0.44, 0.2], [0.34, 0.5], [0, 0.6]], 7),
      color: 0xf7c9d4,
      position: [Math.cos(a) * 0.3, 1.25 + Math.sin(a * 1.7) * 0.16, Math.sin(a) * 0.3],
    });
  }
  return assemble(parts, 'blossomTree');
}

function fordLantern(): THREE.BufferGeometry {
  return assemble(
    [
      { geometry: revolve([[0.07, 0], [0.06, 0.9], [0.05, 1.0]], 6), color: 0x4a4038 },
      { geometry: revolve([[0, 0], [0.17, 0.08], [0.15, 0.3], [0, 0.38]], 8), color: 0xffd9a0, position: [0, 1.0, 0] },
    ],
    'fordLantern',
  );
}

export const chain: EnvSpec = {
  envId: 'Chain-cont-v0',
  title: 'Sixteen Stones',
  logline:
    'A slow green creek crossed by sixteen pentagonal stepping stones, with a blossom tree on the far '
    + 'bank and a lantern at the ford.',
  palette: 'morning',
  accent: 'chain',
  ground: {
    kind: 'profile',
    extent: 70,
    color: 0x86a05e,
    height: (x, z) => {
      const bank = Math.max(0, Math.abs(z) - 3.4);
      return -0.5 + Math.min(1.6, bank * 0.5) + Math.sin(x * 0.2) * 0.12;
    },
  },
  camera: { target: [0, 0.2, 0], azimuth: 0.34, polar: 0.72, distance: 17, follow: 2.2 },
  bloom: 0.4,
  fogExtent: 460,

  build(ctx) {
    const grid = ctx.geometry.grid;
    if (!grid) return;
    const owned: { dispose(): void }[] = [];
    const geoms: THREE.BufferGeometry[] = [];
    const height = chain.ground.height;

    const bankMaterial = oakMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.9,
      toonSteps: 3,
      rimColor: TOKENS.palettes[chain.palette].sun,
      rimStrength: 0.4,
      tier: ctx.tier,
      name: 'creekBank',
    });

    const water: WaterHandle = makeWater({
      size: [70, 9],
      palette: TOKENS.palettes[chain.palette],
      shallow: 0x6fc6cf,
      deep: 0x1f6f8f,
      opacity: 0.82,
      amplitude: 0.03,
      waveScale: 0.6,
      speed: 0.35,
      tier: ctx.tier,
      name: 'creek',
    });
    water.mesh.position.y = -0.16;
    ctx.group.add(water.mesh);
    owned.push(water);

    const stoneGeo = sweep({
      path: curvePath([new THREE.Vector3(0, -0.55, 0), new THREE.Vector3(0, 0, 0)]),
      rings: 4,
      shape: polygonOutline(5, 1, 0.4),
      radius: (t) => 0.3 + 0.09 * t,
    });
    geoms.push(stoneGeo);
    for (let i = 0; i < grid.w; i++) {
      const mesh = new THREE.Mesh(stoneGeo, bankMaterial);
      cellToWorldVec(i, 0, grid, mesh.position);
      mesh.position.y = -0.02 + (i % 2 === 0 ? 0.01 : -0.01);
      mesh.rotation.y = i * 0.4;
      ctx.group.add(mesh);
    }

    const reed = blade(0.7, 0.05);
    const tuft = blade(0.34, 0.08);
    geoms.push(reed, tuft);
    const banks = field({
      source: { kind: 'rect', halfX: 30, halfZ: 16 },
      count: Math.round(700 * ctx.detail.counts),
      seed: ctx.seed,
      stream: 'creekReeds',
      variants: [reed, tuft],
      material: bankMaterial,
      scale: [0.6, 1.7],
      tilt: 0.2,
      height: (x, z) => height?.(x, z) ?? 0,
      color: (i, out) => out.setHSL(0.24 + ((i * 17) % 9) * 0.005, 0.38, 0.33),
      reject: (_x, _y, z) => Math.abs(z) < 3.1,
      name: 'creekReeds',
    });
    ctx.group.add(banks.group);
    owned.push(banks);

    const treeGeo = blossomTree();
    geoms.push(treeGeo);
    const tree = new THREE.Mesh(treeGeo, bankMaterial);
    cellToWorldVec(grid.w - 1, 0, grid, tree.position);
    tree.position.x += 1.6;
    tree.position.z += 3.6;
    tree.position.y = height?.(tree.position.x, tree.position.z) ?? 0;
    ctx.group.add(tree);

    const lanternGeo = fordLantern();
    geoms.push(lanternGeo);
    const lantern = new THREE.Mesh(lanternGeo, bankMaterial);
    cellToWorldVec(0, 0, grid, lantern.position);
    lantern.position.x -= 1.4;
    lantern.position.z -= 3.8;
    lantern.position.y = height?.(lantern.position.x, lantern.position.z) ?? 0;
    ctx.group.add(lantern);

    const flyGeo = revolve([[0, 0], [0.035, 0.02], [0, 0.05]], 5);
    geoms.push(flyGeo);
    const flies = field({
      source: { kind: 'volume', halfX: 8 * CELL, halfY: 0.5, halfZ: 2.4, centre: [0, 0.7, 0] },
      count: Math.round(40 * ctx.detail.counts),
      seed: ctx.seed ^ 0x51ed,
      stream: 'dragonflies',
      variants: [flyGeo],
      material: bankMaterial,
      scale: [0.7, 1.4],
      color: (_i, out) => out.set(0x8fd4e8),
      name: 'dragonflies',
    });
    ctx.group.add(flies.group);
    owned.push(flies);

    return {
      update(_frame, _dt, time) {
        water.setTime(time);
        flies.group.position.y = Math.sin(time * 0.9) * 0.12;
        tree.rotation.z = Math.sin(time * 0.35) * 0.02;
      },
      dispose() {
        for (const o of owned) o.dispose();
        for (const g of geoms) g.dispose();
        bankMaterial.dispose();
      },
    };
  },
};

export default chain;
