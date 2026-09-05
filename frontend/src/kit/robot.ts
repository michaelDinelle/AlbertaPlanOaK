import * as THREE from 'three';

export function robotMaterials() {
  return {
    armor: new THREE.MeshStandardMaterial({ color: 0xe7b83c, roughness: 0.62, metalness: 0.18 }),
    frame: new THREE.MeshStandardMaterial({ color: 0x303b46, roughness: 0.58, metalness: 0.45 }),
    alloy: new THREE.MeshStandardMaterial({ color: 0xa5b0b6, roughness: 0.64, metalness: 0.48 }),
    rubber: new THREE.MeshStandardMaterial({ color: 0x222a30, roughness: 0.92 }),
    hardware: new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.56, metalness: 0.5 }),
    status: new THREE.MeshStandardMaterial({ color: 0x8dd4d0, roughness: 0.3, emissive: 0x589c99, emissiveIntensity: 0.12 }),
  };
}

export function bevelBox(width: number, length: number, depth: number, bevel: number): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const w = width * 0.5;
  const h = length * 0.5;
  const outline = [[-w + bevel, -h], [w - bevel, -h], [w, -h + bevel], [w, h - bevel], [w - bevel, h], [-w + bevel, h], [-w, h - bevel], [-w, -h + bevel]];
  outline.forEach(([x = 0, y = 0], i) => i === 0 ? shape.moveTo(x, y) : shape.lineTo(x, y));
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel * 0.35, bevelSize: bevel * 0.35, bevelSegments: 1, steps: 1, curveSegments: 1 });
  geometry.translate(0, 0, -depth * 0.5);
  return geometry;
}
