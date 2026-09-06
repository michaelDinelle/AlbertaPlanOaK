import * as THREE from 'three';
import { bevelBox, engravedPanel, modelParts, robotMaterials } from '@kit/robot';
import { CELL } from '@kit/grid';
import { makeWorld, type WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';
import { riverHeight } from './world';

export const deckHeight = (x: number) => 0.003 * x * x;
export const chainX = (position: number, grid: GridGeometry) => (position - (grid.w - 1) / 2) * CELL;

export function buildCrossing(grid: GridGeometry) {
  const world = makeWorld('suspensionCrossing'), root = world.group;
  const materials = robotMaterials();
  Object.values(materials).forEach(material => world.own(material));
  const { frame, armor, alloy, rubber, status } = materials;
  const parts = modelParts(world), { box, disc, rod } = parts;
  const half = grid.w * CELL / 2 + 0.8;
  const wood = world.own(new THREE.MeshStandardMaterial({ roughness: 0.88, color: 0x99794d }));
  const footing = world.own(new THREE.MeshStandardMaterial({ color: 0x86968b, roughness: 1 }));
  const cableHeight = (x: number) => 1.05 + 1.8 * (Math.cosh(x / half) - 1) / (Math.cosh(1) - 1);
  const plankCount = Math.ceil((half + 0.65) * 2 / 0.28), pitch = (half + 0.65) * 2 / plankCount;
  for (let i = 0; i < plankCount; i++) {
    const x = (i + 0.5 - plankCount / 2) * pitch;
    const plank = bevelBox(pitch - 0.012, 0.11, 1.72, 0.012);
    parts.add(root, plank.rotateZ(Math.atan(0.006 * x)).translate(x, deckHeight(x) - 0.055, 0), wood);
    for (const z of [-0.65, 0.65]) parts.add(root, new THREE.CylinderGeometry(0.012, 0.012, 0.008, 6).translate(x, deckHeight(x) + 0.007, z), alloy);
  }
  for (const z of [-0.88, 0.88]) {
    for (const elevation of [-0.14, 0.52, 0.87]) {
      const curve = new THREE.CatmullRomCurve3(Array.from({ length: 33 }, (_, i) => {
        const x = (i / 32 * 2 - 1) * half;
        return new THREE.Vector3(x, deckHeight(x) + elevation, z);
      }));
      parts.add(root, new THREE.TubeGeometry(curve, 48, elevation < 0 ? 0.045 : 0.014, 8, false), frame);
    }
    const suspension = new THREE.CatmullRomCurve3(Array.from({ length: 65 }, (_, i) => {
      const x = (i / 64 * 2 - 1) * half;
      return new THREE.Vector3(x, cableHeight(x), z);
    }));
    parts.add(root, new THREE.TubeGeometry(suspension, 96, 0.025, 8, false), alloy);
    for (let i = 0; i <= grid.w; i++) {
      const x = (i / grid.w * 2 - 1) * half;
      rod(root, alloy, new THREE.Vector3(x, deckHeight(x) - 0.12, z), new THREE.Vector3(x, cableHeight(x), z), 0.011);
      box(root, frame, x, deckHeight(x) - 0.11, z, 0.11, 0.19, 0.07, 0.006);
    }
    for (const sign of [-1, 1]) {
      const x = sign * half, bed = riverHeight(x, z) - 0.4;
      box(root, footing, x, bed + 0.3, z, 0.75, 0.65, 0.8, 0.05);
      box(root, frame, x, (bed + 3.02) / 2, z, 0.16, 3.02 - bed, 0.2, 0.018);
      box(root, armor, x, 0.85, z + Math.sign(z) * 0.115, 0.19, 0.62, 0.04);
      box(root, alloy, x, bed + 0.67, z, 0.48, 0.07, 0.52, 0.01);
      for (const dx of [-0.16, 0.16]) for (const dz of [-0.18, 0.18]) parts.add(root, new THREE.CylinderGeometry(0.025, 0.025, 0.032, 6).translate(x + dx, bed + 0.73, z + dz), frame);
      const anchorX = x + sign * 2.3, anchorY = riverHeight(anchorX, z * 1.4);
      box(root, footing, anchorX, anchorY - 0.08, z * 1.4, 0.75, 0.5, 0.7, 0.06);
      rod(root, alloy, new THREE.Vector3(x, cableHeight(x), z), new THREE.Vector3(anchorX, anchorY + 0.16, z * 1.4), 0.028);
      disc(root, rubber, x, 2.88, z, 0.115, 0.26);
      disc(root, alloy, x, 2.88, z + Math.sign(z) * 0.145, 0.072, 0.025);
    }
  }
  for (const sign of [-1, 1]) box(root, frame, sign * half, 3.01, 0, 0.17, 0.15, 2.03, 0.02);
  const destination = chainX(grid.goal!.x, grid), beacon = world.own(status.clone());
  for (const z of [-0.66, 0.66]) {
    box(root, frame, destination, deckHeight(destination) + 0.36, z, 0.14, 0.72, 0.12);
    box(root, armor, destination, deckHeight(destination) + 0.36, z + Math.sign(z) * 0.075, 0.17, 0.48, 0.04);
    box(root, beacon, destination, deckHeight(destination) + 0.69, z, 0.1, 0.025, 0.14, 0.005);
  }
  parts.build();
  const markers = world.own(new THREE.InstancedMesh(world.own(bevelBox(0.32, 0.015, 0.055, 0.005)), status, grid.w * 2));
  const dummy = new THREE.Object3D(), tint = new THREE.Color();
  for (let i = 0; i < grid.w; i++) for (let side = 0; side < 2; side++) {
    const x = chainX(i, grid);
    dummy.position.set(x, deckHeight(x) + 0.018, side ? 0.73 : -0.73);
    dummy.updateMatrix(); markers.setMatrixAt(i * 2 + side, dummy.matrix);
  }
  markers.computeBoundingSphere(); world.add(markers);
  let active = -1;
  return world.seal<WorldCore & { setState(position: number, glow: number): void }>({
    setState(position: number, glow: number) {
      if (active !== position) {
        active = position;
        for (let i = 0; i < grid.w; i++) {
          tint.set(i === position ? 0xc8ffec : i < position ? 0x76aa94 : 0x334340);
          markers.setColorAt(i * 2, tint); markers.setColorAt(i * 2 + 1, tint);
        }
        markers.instanceColor!.needsUpdate = true;
      }
      beacon.emissiveIntensity = 0.2 + glow * 1.4;
    },
  });
}

export function buildRover() {
  const world = makeWorld('chainSurveyRover'), root = world.add(new THREE.Group());
  const materials = robotMaterials();
  Object.values(materials).forEach(material => world.own(material));
  const { armor, accent, frame, alloy, rubber, status } = materials;
  const parts = modelParts(world), { box, disc, rod } = parts;
  box(root, frame, 0, 0.37, 0, 1.02, 0.24, 0.55, 0.05);
  parts.add(root, engravedPanel(0.62, 0.91, 0.22, 0.055, 0.28).rotateZ(-Math.PI / 2).rotateX(-Math.PI / 2).translate(0, 0.56, 0), armor);
  for (const sign of [-1, 1]) {
    box(root, armor, -0.06, 0.5, sign * 0.34, 0.69, 0.16, 0.1, 0.025);
    box(root, rubber, sign * 0.5, 0.39, 0, 0.08, 0.13, 0.61, 0.025);
    for (let i = 0; i < 4; i++) box(root, frame, -0.3 + i * 0.034, 0.54, sign * 0.396, 0.022, 0.075, 0.012, 0.003);
    box(root, status, 0.29, 0.56, sign * 0.365, 0.105, 0.028, 0.024, 0.006);
    for (const x of [-0.38, 0.38]) {
      disc(root, alloy, x, 0.55, sign * 0.4, 0.027, 0.016, 6);
      rod(root, alloy, new THREE.Vector3(x * 0.55, 0.36, sign * 0.22), new THREE.Vector3(x, 0.2, sign * 0.37), 0.028);
    }
  }
  box(root, frame, 0.36, 0.81, 0, 0.065, 0.29, 0.075, 0.01);
  box(root, armor, 0.4, 0.95, 0, 0.23, 0.18, 0.42, 0.035);
  box(root, rubber, 0.527, 0.95, 0, 0.032, 0.12, 0.34, 0.015);
  for (const z of [-0.11, 0.11]) {
    parts.add(root, new THREE.CylinderGeometry(0.052, 0.052, 0.048, 24).rotateZ(Math.PI / 2).translate(0.545, 0.956, z), accent);
    parts.add(root, new THREE.CylinderGeometry(0.036, 0.036, 0.052, 24).rotateZ(Math.PI / 2).translate(0.555, 0.956, z), status);
  }
  rod(root, frame, new THREE.Vector3(-0.34, 0.69, 0.2), new THREE.Vector3(-0.37, 1.09, 0.2), 0.009);
  parts.add(root, new THREE.SphereGeometry(0.023, 12, 8).translate(-0.37, 1.1, 0.2), status);
  const wheel = new THREE.Group();
  parts.add(wheel, new THREE.TorusGeometry(0.145, 0.052, 12, 32), rubber);
  disc(wheel, frame, 0, 0, 0, 0.151, 0.13);
  for (const sign of [-1, 1]) {
    disc(wheel, alloy, 0, 0, sign * 0.08, 0.112, 0.035);
    disc(wheel, frame, 0, 0, sign * 0.102, 0.082, 0.012);
    disc(wheel, accent, 0, 0, sign * 0.112, 0.045, 0.022);
    for (let i = 0; i < 6; i++) {
      const a = i * Math.PI / 3;
      disc(wheel, alloy, Math.cos(a) * 0.086, Math.sin(a) * 0.086, sign * 0.114, 0.012, 0.012, 6);
    }
  }
  for (let i = 0; i < 24; i++) {
    const a = i * Math.PI / 12;
    parts.add(wheel, bevelBox(0.053, 0.024, 0.145, 0.004).translate(0, 0.19, 0).rotateZ(a), rubber);
  }
  parts.build();
  const wheelPositions = [-0.37, 0, 0.37].flatMap(x => [-0.39, 0.39].map(z => new THREE.Vector3(x, 0.2, z)));
  const matrix = new THREE.Matrix4();
  const wheels = wheel.children.map(child => {
    const mesh = child as THREE.Mesh;
    const instances = world.own(new THREE.InstancedMesh(mesh.geometry, mesh.material, wheelPositions.length));
    wheelPositions.forEach((position, i) => instances.setMatrixAt(i, matrix.makeTranslation(position)));
    instances.castShadow = instances.receiveShadow = true;
    instances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    instances.computeBoundingSphere(); root.add(instances);
    return instances;
  });
  let previous = 0, spin = 0;
  return world.seal<WorldCore & { position: THREE.Vector3; setPosition(x: number, snap: boolean): void }>({
    position: world.group.position,
    setPosition(x: number, snap: boolean) {
      if (!snap) spin -= (x - previous) / 0.2;
      previous = x;
      world.group.position.set(x, deckHeight(x), 0);
      root.rotation.z = Math.atan(0.006 * x);
      wheelPositions.forEach((position, i) => {
        matrix.makeRotationZ(spin).setPosition(position);
        for (const wheel of wheels) wheel.setMatrixAt(i, matrix);
      });
      for (const wheel of wheels) wheel.instanceMatrix.needsUpdate = true;
    },
  });
}
