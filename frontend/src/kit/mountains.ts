import * as THREE from 'three';
import { hash2f } from './rng';

const smooth = THREE.MathUtils.smoothstep;

export function terrainNoise(x: number, z: number): number {
  const ix = Math.floor(x), iz = Math.floor(z), u = smooth(x - ix, 0, 1), v = smooth(z - iz, 0, 1);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(hash2f(ix, iz, 81), hash2f(ix + 1, iz, 81), u), THREE.MathUtils.lerp(hash2f(ix, iz + 1, 81), hash2f(ix + 1, iz + 1, 81), u), v);
}

export async function mountainGeometry(region: 'banff-canmore' | 'moraine-lake' = 'banff-canmore'): Promise<THREE.BufferGeometry> {
  const alpine = region === 'moraine-lake';
  const columns = alpine ? 384 : 256, rows = alpine ? 256 : 192, datum = alpine ? 1880 : 1300;
  const response = await fetch(`/terrain/${region}-elevation.bin`);
  if (!response.ok) throw new Error(`Rockies elevation unavailable: ${response.status}`);
  const elevations = new Float32Array(await response.arrayBuffer());
  if (elevations.length !== columns * rows) throw new Error('Rockies elevation grid is incomplete');
  const geometry = new THREE.PlaneGeometry(alpine ? 3200 : 3600, alpine ? 2400 : 2600, columns - 1, rows - 1);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position');
  const uv = geometry.getAttribute('uv');
  for (let i = 0; i < positions.count; i++) {
    const column = i % columns, row = Math.floor(i / columns);
    const edge = smooth(Math.min(column, columns - 1 - column), 0, 12) * smooth(Math.min(row, rows - 1 - row), 0, 10);
    positions.setXYZ(i, positions.getX(i), (elevations[i]! - datum) * (alpine ? 0.27 : 0.21) * edge - 8, positions.getZ(i) - (alpine ? 0 : 2600));
    if (alpine) {
      const x = positions.getX(i), z = positions.getZ(i);
      const relief = (Math.abs(terrainNoise(x * 0.032, z * 0.028) - 0.5) - 0.25) * 18 + (terrainNoise(x * 0.085, z * 0.08) - 0.5) * 2;
      positions.setY(i, positions.getY(i) + relief * smooth(elevations[i]!, 2050, 2400) * edge);
    }
    uv.setXY(i, positions.getX(i) * 0.025, positions.getY(i) * 0.04);
  }
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal'), colors = new Float32Array(positions.count * 3);
  const rock = new THREE.Color(alpine ? 0x8799a1 : 0xb7afa4), snow = new THREE.Color(alpine ? 0xe9f1f4 : 0xffeadd), meadow = new THREE.Color(alpine ? 0x3e584b : 0x98916e), color = new THREE.Color();
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), z = positions.getZ(i), elevation = elevations[i]!;
    const snowCover = smooth(elevation + terrainNoise(x * 0.035, z * 0.035) * 130, alpine ? 2700 : 2310, alpine ? 3050 : 2630) * smooth(normals.getY(i), 0.3, 0.8);
    color.copy(meadow).lerp(rock, smooth(elevation, alpine ? 2090 : 1800, alpine ? 2470 : 2300)).lerp(snow, snowCover);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.userData.source = { name: 'Canadian Digital Elevation Model via Mapzen', bounds: alpine ? [-116.33203125, 51.23991016823824, -116.138671875, 51.333357371124656] : [-115.6640625, 51.01375465718819, -115.3125, 51.17934297928927], url: 'https://registry.opendata.aws/terrain-tiles/' };
  return geometry;
}
