import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export function vineStrand(points: readonly (readonly [number, number, number])[], radius = 0.02, leaves = 8): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), false, 'centripetal');
  const stem = new THREE.TubeGeometry(curve, Math.max(8, points.length * 6), radius, 5, false);
  const pieces: THREE.BufferGeometry[] = [stem];
  const stemColor = new THREE.Color(0x4a6b34), leafA = new THREE.Color(0x6c9a45), leafB = new THREE.Color(0xa8c35a), color = new THREE.Color();
  const tint = (geometry: THREE.BufferGeometry, c: THREE.Color) => {
    const count = geometry.getAttribute('position').count, colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) c.toArray(colors, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  };
  tint(stem, stemColor);
  for (let i = 0; i < leaves; i++) {
    const t = (i + 0.5) / leaves, at = curve.getPointAt(t), tangent = curve.getTangentAt(t);
    const leaf = new THREE.CircleGeometry(radius * 6, 5).scale(1, 1.6, 1);
    const yaw = i * 2.4, side = new THREE.Vector3().crossVectors(tangent, THREE.Object3D.DEFAULT_UP).normalize();
    leaf.rotateX(-Math.PI / 2 + 0.6).rotateY(yaw).translate(at.x + side.x * radius * 4, at.y + radius * 2, at.z + side.z * radius * 4);
    tint(leaf, color.copy(leafA).lerp(leafB, (i % 3) / 2));
    pieces.push(leaf);
  }
  const merged = mergeGeometries(pieces.map(p => p.index ? p.toNonIndexed() : p));
  pieces.forEach(p => p.dispose());
  merged.computeVertexNormals();
  merged.name = 'vineStrand';
  return merged;
}
