import * as THREE from 'three';
import { bevelBox, robotMaterials } from '@kit/robot';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { makeWorld, type WorldCore } from '@kit/world';
import { AB_L1, AB_L2, AB_SUCCESS_HEIGHT } from '@sim/envs/acrobot';
import { lakeHeight } from './world';

export interface ArmHandle extends WorldCore {
  tipWorld: THREE.Vector3;
  setState(th1: number, th2: number): void;
}

export function buildArm(pivot: readonly [number, number, number]) {
  const world = makeWorld('alpineAcrobot');
  const root = world.add(new THREE.Group());
  root.position.fromArray(pivot);
  const materials = robotMaterials();
  Object.values(materials).forEach(material => world.own(material));
  const { frame, armor, alloy, rubber, status } = materials;
  const parts = new Map<THREE.Group, Map<THREE.Material, THREE.BufferGeometry[]>>();
  function add(parent: THREE.Group, geometry: THREE.BufferGeometry, material: THREE.Material) {
    if (!parts.has(parent)) parts.set(parent, new Map());
    const batch = parts.get(parent)!;
    if (!batch.has(material)) batch.set(material, []);
    batch.get(material)!.push(geometry.index ? geometry.toNonIndexed() : geometry);
    if (geometry.index) geometry.dispose();
  }
  function box(parent: THREE.Group, material: THREE.Material, x: number, y: number, z: number, w: number, h: number, d: number, radius = 0.018) {
    add(parent, bevelBox(w, h, d, Math.min(radius, w / 3, h / 3, d / 3)).translate(x, y, z), material);
  }
  function disc(parent: THREE.Group, material: THREE.Material, x: number, y: number, z: number, radius: number, depth: number, sides = 32) {
    add(parent, new THREE.CylinderGeometry(radius, radius, depth, sides).rotateX(Math.PI / 2).translate(x, y, z), material);
  }
  function strut(parent: THREE.Group, a: THREE.Vector3, b: THREE.Vector3, radius: number) {
    const geometry = new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 12);
    geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize()));
    add(parent, geometry.translate(...a.clone().add(b).multiplyScalar(0.5).toArray()), alloy);
  }
  for (const side of [-1, 1]) {
    const x = side * 2.6, bed = lakeHeight(x, -0.7) - pivot[1];
    box(root, frame, x, (bed + 0.6) / 2, -0.7, 0.19, 0.6 - bed, 0.24, 0.025);
    box(root, alloy, x, bed + 0.05, -0.7, 0.75, 0.18, 1.1, 0.04);
    box(root, armor, x, -1.1, -0.55, 0.23, 1.6, 0.04);
    box(root, rubber, x, 0.54, -0.7, 0.26, 0.12, 0.3);
    for (const z of [-1.12, -0.28]) for (const dx of [-0.27, 0.27]) disc(root, frame, x + dx, bed + 0.16, z, 0.035, 0.03, 6);
    strut(root, new THREE.Vector3(x, -0.7, -0.7), new THREE.Vector3(x - side * 0.85, 0.5, -0.7), 0.034);
    for (let j = 0; j < 4; j++) box(root, status, x, AB_SUCCESS_HEIGHT + (j - 1.5) * 0.065, -0.53, 0.12, 0.014, 0.01, 0.003);
    box(root, frame, x, 0.85, -0.7, 0.08, 0.65, 0.12);
  }
  box(root, frame, 0, 0.5, -0.7, 5.42, 0.2, 0.28, 0.045);
  box(root, alloy, 0, 0.615, -0.7, 5.25, 0.025, 0.2, 0.006);
  box(root, armor, 0, 0.24, -0.62, 0.46, 0.48, 0.16, 0.07);
  disc(root, frame, 0, 0, -0.35, 0.2, 0.55);
  disc(root, alloy, 0, 0, -0.065, 0.23, 0.06);
  const first = new THREE.Group(), second = new THREE.Group();
  root.add(first);
  first.add(second);
  second.position.y = -AB_L1;
  for (const [node, length] of [[first, AB_L1], [second, AB_L2]] as const) {
    box(node, frame, 0, -length / 2, 0, 0.18, length - 0.12, 0.18, 0.05);
    for (const sign of [-1, 1]) {
      box(node, armor, 0, -length / 2, sign * 0.112, 0.24, length - 0.31, 0.055, 0.026);
      box(node, rubber, 0, -length / 2, sign * 0.144, 0.082, length - 0.47, 0.013, 0.006);
      for (const y of [-0.24, -length + 0.24]) for (const x of [-0.081, 0.081]) disc(node, alloy, x, y, sign * 0.15, 0.017, 0.012, 6);
      for (let slot = 0; slot < 4; slot++) box(node, alloy, 0, -length * 0.4 - slot * 0.052, sign * 0.156, 0.04, 0.012, 0.009, 0.003);
      box(node, status, 0, -length * 0.29, sign * 0.156, 0.037, 0.075, 0.009, 0.006);
    }
    for (const x of [-0.117, 0.117]) strut(node, new THREE.Vector3(x, -0.15, 0), new THREE.Vector3(x, -length + 0.15, 0), 0.024);
    disc(node, rubber, 0, 0, 0, 0.175, 0.24);
    for (const sign of [-1, 1]) {
      disc(node, alloy, 0, 0, sign * 0.137, 0.156, 0.036);
      disc(node, frame, 0, 0, sign * 0.16, 0.115, 0.018);
      disc(node, armor, 0, 0, sign * 0.178, 0.07, 0.022);
      for (let bolt = 0; bolt < 8; bolt++) {
        const angle = bolt * Math.PI / 4;
        disc(node, frame, Math.cos(angle) * 0.135, Math.sin(angle) * 0.135, sign * 0.161, 0.013, 0.014, 6);
      }
    }
  }
  disc(second, rubber, 0, -AB_L2, 0, 0.15, 0.2);
  disc(second, alloy, 0, -AB_L2, 0.12, 0.128, 0.04);
  disc(second, status, 0, -AB_L2, 0.146, 0.063, 0.012);
  for (const [parent, batch] of parts) for (const [material, geometries] of batch) {
    const geometry = world.own(mergeGeometries(geometries));
    geometries.forEach(part => part.dispose());
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = mesh.receiveShadow = true;
    parent.add(mesh);
  }
  const tipWorld = new THREE.Vector3();
  return world.seal<ArmHandle>({
    tipWorld,
    setState(th1: number, th2: number) {
      first.rotation.z = th1;
      second.rotation.z = th2;
      tipWorld.set(pivot[0] + AB_L1 * Math.sin(th1) + AB_L2 * Math.sin(th1 + th2), pivot[1] - AB_L1 * Math.cos(th1) - AB_L2 * Math.cos(th1 + th2), pivot[2]);
    },
  });
}
