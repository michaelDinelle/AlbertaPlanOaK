import * as THREE from 'three/webgpu';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

export function hoodooGeometry(): THREE.BufferGeometry {
  const height = (x: number, z: number) => 1.65 * Math.exp(-x * x / 8 - z * z / 2.5)
    + 2.3 * Math.exp(-Math.pow((x + 1.1 + z * 0.22) / 0.9, 6) - Math.pow((z + 0.2) / 0.8, 6))
    + 1.3 * Math.exp(-Math.pow((x - 1.4) / 0.8, 4) - Math.pow((z - 0.15) / 0.7, 4));
  const bluff = new THREE.PlaneGeometry(9, 5, 36, 24);
  bluff.rotateX(-Math.PI / 2);
  const position = bluff.getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), z = position.getZ(i);
    const erosion = 0.06 * Math.sin(x * 15 + z * 3) * Math.sin(z * 11);
    position.setY(i, height(x, z) * (1 + erosion) - 0.16);
  }
  bluff.computeVertexNormals();
  const parts: THREE.BufferGeometry[] = [bluff.toNonIndexed()];
  bluff.dispose();
  for (const [x, z, width] of [[-1.1, -0.2, 1.9]] as const) {
    const cap = new THREE.IcosahedronGeometry(1, 1);
    const vertices = cap.getAttribute('position');
    for (let i = 0; i < vertices.count; i++) {
      const px = vertices.getX(i), py = vertices.getY(i), pz = vertices.getZ(i);
      vertices.setXYZ(i, px * width * 0.62 * (1 + 0.15 * Math.sin(pz * 5 + x)), py * 0.22 + px * 0.06 + pz * 0.045, pz * 0.88 * (1 + 0.13 * Math.cos(px * 4 + x)));
    }
    cap.computeVertexNormals();
    cap.translate(x, height(x, z) - 0.22, z);
    parts.push(cap);
  }
  for (const [part, geometry] of parts.entries()) {
    const vertices = geometry.getAttribute('position');
    const colors = new Float32Array(vertices.count * 3);
    const sand = new THREE.Color(0xdec9a9), shale = new THREE.Color(0x796656), color = new THREE.Color();
    for (let i = 0; i < vertices.count; i++) {
      const y = vertices.getY(i);
      color.copy(sand).lerp(shale, part > 0 ? 0.65 : y < 1.1 ? 0.48 + 0.14 * Math.sin(y * 18) : 0.08);
      colors.set([color.r, color.g, color.b], i * 3);
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  const merged = mergeGeometries(parts), geometry = mergeVertices(merged);
  merged.dispose();
  for (const part of parts) part.dispose();
  return geometry;
}
