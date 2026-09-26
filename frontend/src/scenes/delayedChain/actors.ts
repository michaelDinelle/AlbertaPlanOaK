import * as THREE from 'three/webgpu';
import { GoalBeacon, cellToWorld, type GridSize, type Vec3Tuple } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { blendTokens, clamp01, shade } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash1f } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import { makeWorld, type WorldCore } from '@kit/world';
import { LAMP_REACH, LANTERN_Z, type MoorSkin } from './world';

const POST_HEIGHT = 1.78;
export const LAMP_Y = 1.5;
const LAMP_HDR = 2.6;
const HALO_HDR = 0.75;
const POOL_HDR = 0.46;
const GLOW_SHARES = [1, 0.62, 0.8] as const;

const propMaterial = (palette: Palette, name: string) => oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.9, rimColor: palette.hemiSky, rimStrength: 0.3, name });

function lanternGeometry(skin: MoorSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [
    { geometry: new THREE.CylinderGeometry(0.075, 0.115, 0.11, 6).translate(0, 0.055, 0), color: skin.stoneDark },
    { geometry: new THREE.CylinderGeometry(0.03, 0.048, POST_HEIGHT, 6).translate(0, POST_HEIGHT * 0.5, 0), color: skin.iron },
    { geometry: new THREE.CylinderGeometry(0.052, 0.052, 0.055, 6).translate(0, 1.16, 0), color: skin.ironLight },
    { geometry: new THREE.BoxGeometry(0.038, 0.038, LAMP_REACH + 0.08).translate(0, POST_HEIGHT - 0.08, (LAMP_REACH + 0.08) * 0.5 - 0.04), color: skin.iron },
    { geometry: new THREE.BoxGeometry(0.028, 0.13, 0.028).translate(0, POST_HEIGHT - 0.15, LAMP_REACH), color: skin.iron },
    { geometry: new THREE.ConeGeometry(0.125, 0.1, 6).translate(0, LAMP_Y + 0.15, LAMP_REACH), color: skin.ironLight },
    { geometry: new THREE.CylinderGeometry(0.092, 0.074, 0.035, 6).translate(0, LAMP_Y - 0.11, LAMP_REACH), color: skin.ironLight },
  ];
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    parts.push({ geometry: new THREE.BoxGeometry(0.017, 0.23, 0.017).translate(Math.cos(a) * 0.075, LAMP_Y + 0.015, LAMP_REACH + Math.sin(a) * 0.075), color: skin.iron });
  }
  return mergeColoredParts(parts, 'lanternPost');
}

function cairnGeometry(skin: MoorSkin, seed: number): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const dark = shade(skin.stoneDark, 0.1);
  const rings = [
    { y: 0.16, r: 0.38, n: 5, s: 0.23 },
    { y: 0.42, r: 0.26, n: 4, s: 0.2 },
    { y: 0.64, r: 0.14, n: 3, s: 0.17 },
    { y: 0.8, r: 0, n: 1, s: 0.15 },
  ];
  let k = 0;
  for (const ring of rings) {
    for (let i = 0; i < ring.n; i++, k++) {
      const h1 = hash1f(k, seed ^ 0x51a3d7), h2 = hash1f(k, seed ^ 0x2ce19b);
      const a = (i / ring.n) * Math.PI * 2 + ring.y * 3.1 + h1 * 0.5;
      const rock = new THREE.IcosahedronGeometry(ring.s * (0.82 + 0.36 * h2), 0)
        .scale(1, 0.72 + 0.22 * h1, 0.88)
        .rotateY(h1 * 3.14)
        .rotateZ(h2 * 0.8 - 0.4)
        .translate(Math.cos(a) * ring.r, ring.y + 0.05 * h2, Math.sin(a) * ring.r * 0.8);
      parts.push({ geometry: rock, color: k % 3 === 1 ? dark : skin.stone });
    }
  }
  parts.push({ geometry: new THREE.CylinderGeometry(0.16, 0.25, 1.16, 5).scale(1, 1, 0.62).rotateZ(0.15).rotateY(-0.34).translate(-0.42, 0.57, 0.1), color: skin.stone });
  parts.push({ geometry: new THREE.IcosahedronGeometry(0.19, 0).scale(1, 0.6, 0.9).translate(-0.3, 0.09, -0.24), color: dark });
  return mergeColoredParts(parts, 'endCairn');
}

function milestoneGeometry(skin: MoorSkin): THREE.BufferGeometry {
  const kerb = shade(skin.stoneDark, 0.25);
  return mergeColoredParts([
    { geometry: new THREE.CylinderGeometry(0.1, 0.15, 0.52, 4).rotateY(Math.PI * 0.25 + 0.16).rotateZ(-0.12).scale(1, 1, 0.72).translate(0, 0.25, 0), color: shade(skin.stone, 0.42) },
    { geometry: new THREE.CylinderGeometry(0.085, 0.105, 0.05, 4).rotateY(Math.PI * 0.25 + 0.16).rotateZ(-0.12).scale(1, 1, 0.72).translate(0.033, 0.51, 0), color: shade(skin.moss, 0.3) },
    { geometry: new THREE.CylinderGeometry(0.1, 0.13, 0.08, 7).translate(0.17, 0.03, 0.12), color: kerb },
    { geometry: new THREE.CylinderGeometry(0.08, 0.1, 0.06, 7).translate(-0.14, 0.025, -0.15), color: kerb },
    { geometry: new THREE.CylinderGeometry(0.22, 0.26, 0.05, 7).translate(0, 0.02, 0), color: kerb },
  ], 'startMilestone');
}

export interface LanternRoad extends WorldCore {
  readonly endLamp: THREE.Vector3;
  readonly count: number;
  lanternX(cell: number): number;
  setGlow(cell: number, v01: number): void;
  commit(): void;
  flare(strength: number): void;
  update(dt: number, time: number): void;
}

export function buildLanternRoad(palette: Palette, skin: MoorSkin, grid: GridSize, goal: number, seed: number, aurora: THREE.ColorRepresentation): LanternRoad {
  const w = makeWorld('lanternRoad'), count = grid.w, scratch: Vec3Tuple = [0, 0, 0];
  const lanternX = (cell: number) => cellToWorld(cell, 0, grid, scratch)[0];
  const additive = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
  const lamps = (name: string, geometry: THREE.BufferGeometry, material: THREE.Material) => {
    const mesh = w.add(w.own(instancedMesh(w.own(geometry), w.own(material), count)));
    mesh.name = name;
    mesh.frustumCulled = false;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    return mesh;
  };
  const posts = lamps('lanternPosts', lanternGeometry(skin), propMaterial(palette, 'lanternIron'));
  posts.receiveShadow = true;
  const glass = lamps('lanternFlames', new THREE.IcosahedronGeometry(0.058, 1).translate(0, LAMP_Y, LAMP_REACH), emissiveMaterial(skin.flame, LAMP_HDR, { name: 'lanternFlame' }));
  const halo = lamps('lanternHalos', new THREE.IcosahedronGeometry(0.12, 2).translate(0, LAMP_Y, LAMP_REACH), emissiveMaterial(skin.flame, HALO_HDR, { ...additive, opacity: 0.5, name: 'lanternHalo' }));
  const pool = lamps('lanternPools', new THREE.CircleGeometry(0.86, 24).rotateX(-Math.PI / 2).translate(0, 0.014, LAMP_REACH),
    emissiveMaterial(blendTokens(skin.flame, palette.accentLight, 0.1), POOL_HDR, { ...additive, opacity: 0.9, map: w.own(radialSprite(64, 1.7)), name: 'lanternPool' }));

  const matrix = new THREE.Matrix4(), position = new THREE.Vector3(), quaternion = new THREE.Quaternion(), euler = new THREE.Euler(), scale = new THREE.Vector3(), weather = new THREE.Color();
  for (let cell = 0; cell < count; cell++) {
    const end = cell === goal, lean = end ? 0 : hash1f(cell, seed ^ 0x2f5a17) - 0.5, wear = hash1f(cell, seed ^ 0x7c1d93);
    quaternion.setFromEuler(euler.set(0, lean * 0.22, lean * 0.05, 'YXZ'));
    matrix.compose(position.set(lanternX(cell), 0, LANTERN_Z), quaternion, scale.setScalar(end ? 1 : 0.94 + 0.12 * wear));
    for (const mesh of [posts, glass, halo, pool]) mesh.setMatrixAt(cell, matrix);
    posts.setColorAt(cell, end ? weather.setScalar(1) : weather.setRGB(0.9 + 0.2 * wear, 0.9 + 0.2 * wear, 0.92 + 0.18 * wear));
  }

  const endX = lanternX(goal);
  const cairn = w.add(new THREE.Mesh(w.own(cairnGeometry(skin, seed)), w.own(oakMaterial({
    vertexColors: true, flatShading: true, roughness: 0.9, rimColor: blendTokens(palette.hemiSky, aurora, 0.45), rimStrength: 0.38,
    toonSteps: 3, toonSoftness: 0.3, toonFloor: 0.2, name: 'endCairnStone',
  }))));
  cairn.name = 'endCairn';
  cairn.position.set(endX + 0.74, 0, -0.14);
  cairn.rotation.y = 0.42;
  cairn.castShadow = cairn.receiveShadow = true;
  const beacon = w.own(new GoalBeacon({ color: blendTokens(palette.accentLight, aurora, 0.55), height: 0.2, radius: 0.085, intensity: 0.3, name: 'endBeacon' }));
  beacon.setCell(goal, 0, grid);
  w.add(beacon.group);

  const lit = [glass, halo, pool];
  const glows = lit.map((mesh) => mesh.instanceColor!.array as Float32Array);
  return w.seal<LanternRoad>({
    endLamp: new THREE.Vector3(endX, LAMP_Y, LANTERN_Z + LAMP_REACH),
    count,
    lanternX,
    setGlow(cell, v01) {
      const v = clamp01(v01);
      glows.forEach((glow, k) => glow.fill(v * GLOW_SHARES[k]!, cell * 3, cell * 3 + 3));
    },
    commit() {
      for (const mesh of lit) mesh.instanceColor!.needsUpdate = true;
    },
    flare: (strength) => beacon.flare(strength),
    update: (dt, time) => beacon.update(dt, time),
  });
}

export function buildMilestone(palette: Palette, skin: MoorSkin, grid: GridSize): WorldCore {
  const w = makeWorld('startMilestone');
  const mesh = w.add(new THREE.Mesh(w.own(milestoneGeometry(skin)), w.own(propMaterial(palette, 'milestoneStone'))));
  mesh.name = 'milestone';
  mesh.position.set(cellToWorld(0, 0, grid)[0] - 0.34, 0, 1.32);
  mesh.rotation.y = 0.62;
  mesh.castShadow = mesh.receiveShadow = true;
  return w.seal<WorldCore>({});
}

export interface WispGlow extends WorldCore {
  readonly mesh: THREE.Mesh;
  setGain(gain: number): void;
}

export function buildWispGlow(color: THREE.ColorRepresentation): WispGlow {
  const w = makeWorld('wispGlow'), intensity = 0.72;
  const material = w.own(emissiveMaterial(color, intensity, { map: w.own(radialSprite(64, 2.6)), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, name: 'wispGlowMaterial' }));
  const mesh = w.add(new THREE.Mesh(w.own(new THREE.PlaneGeometry(2.1, 2.1)), material));
  mesh.name = 'wispGlow';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  return w.seal<WispGlow>({ mesh, setGain: (gain) => setEmissiveIntensity(material, intensity * gain) });
}
