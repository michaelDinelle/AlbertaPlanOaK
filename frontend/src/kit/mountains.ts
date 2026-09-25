import * as THREE from 'three/webgpu';
import { oakMaterial } from './materials';
import { blendTokens, TOKEN_WHITE } from './mix';
import { hash2f } from './rng';
import { TOKENS, type Palette } from './tokens';
import type { WorldBuilder } from './world';

const smooth = THREE.MathUtils.smoothstep;

export function terrainNoise(x: number, z: number): number {
  const ix = Math.floor(x), iz = Math.floor(z), u = smooth(x - ix, 0, 1), v = smooth(z - iz, 0, 1);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(hash2f(ix, iz, 81), hash2f(ix + 1, iz, 81), u), THREE.MathUtils.lerp(hash2f(ix, iz + 1, 81), hash2f(ix + 1, iz + 1, 81), u), v);
}

export function mountainStyle(palette: Palette) {
  return {
    valley: blendTokens(palette.groundA, palette.hemiGround, 0.55),
    meadow: blendTokens(palette.groundA, TOKENS.palettes.noon.groundA, 0.5),
    snow: blendTokens(TOKEN_WHITE, palette.hemiSky, 0.3),
    rock: blendTokens(0x6e707a, palette.hemiSky, 0.18),
    material: { rimColor: palette.skyHorizon, rimStrength: 0.24, toonSteps: 3, toonSoftness: 0.32, toonFloor: 0.2 },
  };
}

export async function makeMountains(world: WorldBuilder, palette: Palette, region: 'banff-canmore' | 'moraine-lake' = 'banff-canmore') {
  const alpine = region === 'moraine-lake';
  const columns = alpine ? 384 : 256, rows = alpine ? 256 : 192, datum = alpine ? 1880 : 1300;
  const response = await fetch(`/terrain/${region}-elevation.bin`);
  if (!response.ok) throw new Error(`Rockies elevation unavailable: ${response.status}`);
  const elevations = new Float32Array(await response.arrayBuffer());
  if (elevations.length !== columns * rows) throw new Error('Rockies elevation grid is incomplete');
  const geometry = new THREE.PlaneGeometry(alpine ? 3200 : 3600, alpine ? 2400 : 2600, columns - 1, rows - 1);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position');
  for (let i = 0; i < positions.count; i++) {
    const column = i % columns, row = Math.floor(i / columns);
    const edge = smooth(Math.min(column, columns - 1 - column), 0, 12) * smooth(Math.min(row, rows - 1 - row), 0, 10);
    positions.setXYZ(i, positions.getX(i), (elevations[i]! - datum) * (alpine ? 0.27 : 0.21) * edge - 8, positions.getZ(i) - (alpine ? 0 : 2600));
  }
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal'), colors = new Float32Array(positions.count * 3);
  const { rock, snow, meadow, material } = mountainStyle(palette), color = new THREE.Color();
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), z = positions.getZ(i), elevation = elevations[i]!;
    const column = i % columns, row = Math.floor(i / columns);
    const basin = (elevations[row * columns + Math.max(0, column - 3)]! + elevations[row * columns + Math.min(columns - 1, column + 3)]! + elevations[Math.max(0, row - 3) * columns + column]! + elevations[Math.min(rows - 1, row + 3) * columns + column]!) * 0.25 - elevation;
    const snowCover = smooth(elevation + terrainNoise(x * 0.035, z * 0.035) * 130, alpine ? 2700 : 2310, alpine ? 2880 : 2490) * smooth(normals.getY(i), 0.3, 0.8);
    color.copy(meadow).lerp(rock, smooth(elevation, alpine ? 2090 : 1800, alpine ? 2470 : 2300)).lerp(snow, snowCover);
    color.multiplyScalar(1 - smooth(basin, 0, 24) * 0.28);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.userData.source = { name: 'Canadian Digital Elevation Model via Mapzen', bounds: alpine ? [-116.33203125, 51.23991016823824, -116.138671875, 51.333357371124656] : [-115.6640625, 51.01375465718819, -115.3125, 51.17934297928927], url: 'https://registry.opendata.aws/terrain-tiles/' };
  return world.add(new THREE.Mesh(world.own(geometry), world.own(oakMaterial({ ...material, vertexColors: true, flatShading: true, roughness: 0.95 }))));
}
