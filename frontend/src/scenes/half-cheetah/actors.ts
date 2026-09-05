import * as THREE from 'three';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { CHEETAH_BONES, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';

export interface CheetahHandle {
  group: THREE.Group;
  joints: THREE.Object3D[];
  setPose(fk: Float32Array): void;
  dispose(): void;
}

export function buildCheetah(cheetah: CheetahGeometry): CheetahHandle {
  const group = new THREE.Group();
  group.name = 'fieldRobot';
  const armor = new THREE.MeshStandardMaterial({ color: 0xe7b83c, roughness: 0.62, metalness: 0.18 });
  const frame = new THREE.MeshStandardMaterial({ color: 0x303b46, roughness: 0.58, metalness: 0.45 });
  const alloy = new THREE.MeshStandardMaterial({ color: 0xa5b0b6, roughness: 0.64, metalness: 0.48 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x222a30, roughness: 0.92 });
  const hardware = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.56, metalness: 0.5 });
  const status = new THREE.MeshStandardMaterial({ color: 0x8dd4d0, roughness: 0.3, emissive: 0x589c99, emissiveIntensity: 0.12 });
  const geometries: THREE.BufferGeometry[] = [];
  const bevelBox = (width: number, length: number, depth: number, bevel: number): THREE.BufferGeometry => {
    const shape = new THREE.Shape();
    const w = width * 0.5;
    const h = length * 0.5;
    const outline = [[-w + bevel, -h], [w - bevel, -h], [w, -h + bevel], [w, h - bevel], [w - bevel, h], [-w + bevel, h], [-w, h - bevel], [-w, -h + bevel]];
    outline.forEach(([x = 0, y = 0], i) => i === 0 ? shape.moveTo(x, y) : shape.lineTo(x, y));
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel * 0.35, bevelSize: bevel * 0.35, bevelSegments: 1, steps: 1, curveSegments: 1 });
    geometry.translate(0, 0, -depth * 0.5);
    return geometry;
  };
  const add = (root: THREE.Object3D, geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0): THREE.Mesh => {
    geometries.push(geometry);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    root.add(mesh);
    return mesh;
  };
  const gearGeometries = [0.087, 0.064, 0.055].map((radius) => {
    const parts: ColoredPart[] = [];
    const housing = new THREE.CylinderGeometry(radius, radius, 0.14, 14);
    housing.rotateX(Math.PI / 2);
    parts.push({ geometry: housing, color: 0x34424b });
    for (const side of [-1, 1]) {
      const cover = new THREE.CylinderGeometry(radius * 0.72, radius * 0.72, 0.012, 12);
      cover.rotateX(Math.PI / 2);
      cover.translate(0, 0, side * 0.076);
      parts.push({ geometry: cover, color: 0xb4bdc0 });
      const hub = new THREE.CylinderGeometry(radius * 0.31, radius * 0.31, 0.015, 6);
      hub.rotateX(Math.PI / 2);
      hub.translate(0, 0, side * 0.085);
      parts.push({ geometry: hub, color: 0x535d64 });
      for (let screw = 0; screw < 4; screw++) {
        const angle = screw * Math.PI * 0.5 + 0.35;
        const fastener = new THREE.CylinderGeometry(0.007, 0.007, 0.014, 5);
        fastener.rotateX(Math.PI / 2);
        fastener.translate(Math.cos(angle) * radius * 0.53, Math.sin(angle) * radius * 0.53, side * 0.083);
        parts.push({ geometry: fastener, color: 0x333d43 });
      }
    }
    const geometry = mergeColoredParts(parts, 'robotGearbox');
    geometries.push(geometry);
    return geometry;
  });
  const padGeometry = new THREE.SphereGeometry(cheetah.radius, 12, 6);
  geometries.push(padGeometry);
  const joints = Array.from({ length: 10 }, (_, i) => {
    const joint = new THREE.Object3D();
    joint.name = `robotJoint${i}`;
    if (i !== FK_POINT_INDEX.o1 && i !== FK_POINT_INDEX.head) {
      const toe = i === FK_POINT_INDEX.toe_b || i === FK_POINT_INDEX.toe_f;
      const gear = i === FK_POINT_INDEX.bt || i === FK_POINT_INDEX.ft ? 0 : i === FK_POINT_INDEX.bs || i === FK_POINT_INDEX.fs ? 1 : 2;
      const mesh = new THREE.Mesh(toe ? padGeometry : gearGeometries[gear], toe ? rubber : hardware);
      mesh.castShadow = true;
      joint.add(mesh);
    }
    group.add(joint);
    return joint;
  });
  const rest = cheetah.fk(new Float32Array(9));
  const lengths: number[] = [];
  const bones = CHEETAH_BONES.map((bone, i) => {
    const length = Math.hypot((rest[bone[1] * 2] ?? 0) - (rest[bone[0] * 2] ?? 0), (rest[bone[1] * 2 + 1] ?? 0) - (rest[bone[0] * 2 + 1] ?? 0));
    lengths.push(length);
    const link = new THREE.Group();
    link.name = i === 0 ? 'robotChassis' : `robotLink${i}`;
    if (i === 0) {
      add(link, bevelBox(0.22, length * 0.96, 0.28, 0.035), frame, 0.015);
      add(link, bevelBox(0.18, length * 0.87, 0.32, 0.035), armor, -0.067);
      add(link, bevelBox(0.11, length * 0.66, 0.27, 0.018), frame, 0.115);
      add(link, bevelBox(0.018, length * 0.53, 0.25, 0.004), alloy, -0.168);
      for (const side of [-1, 1]) {
        add(link, bevelBox(0.13, length * 0.72, 0.014, 0.014), armor, -0.025, 0, side * 0.171);
        add(link, bevelBox(0.026, 0.12, 0.008, 0.004), status, -0.058, 0.24, side * 0.184);
        for (let vent = 0; vent < 4; vent++) add(link, new THREE.BoxGeometry(0.05, 0.012, 0.008), frame, 0.018, -0.17 + vent * 0.034, side * 0.183);
      }
      for (const end of [-1, 1]) add(link, bevelBox(0.18, 0.065, 0.28, 0.014), rubber, 0.01, end * length * 0.46);
    } else {
      const upper = i === 1 || i === 4;
      const foot = i === 3 || i === 6;
      add(link, bevelBox(upper ? 0.095 : 0.058, Math.max(0.035, length - 0.04), upper ? 0.12 : 0.072, 0.011), foot ? rubber : frame);
      if (!foot) {
        for (const side of [-1, 1]) add(link, bevelBox(upper ? 0.075 : 0.045, length * 0.55, 0.013, 0.008), upper ? armor : alloy, 0, 0, side * (upper ? 0.069 : 0.043));
      }
    }
    group.add(link);
    return link;
  });
  return {
    group,
    joints,
    setPose(fk) {
      for (let i = 0; i < joints.length; i++) joints[i]?.position.set(fk[i * 2] ?? 0, fk[i * 2 + 1] ?? 0, 0);
      for (let i = 0; i < CHEETAH_BONES.length; i++) {
        const bone = CHEETAH_BONES[i];
        const mesh = bones[i];
        if (!bone || !mesh) continue;
        const ax = fk[bone[0] * 2] ?? 0;
        const ay = fk[bone[0] * 2 + 1] ?? 0;
        const bx = fk[bone[1] * 2] ?? 0;
        const by = fk[bone[1] * 2 + 1] ?? 0;
        mesh.position.set((ax + bx) * 0.5, (ay + by) * 0.5, 0);
        mesh.rotation.z = Math.atan2(ax - bx, by - ay);
        mesh.scale.y = Math.hypot(bx - ax, by - ay) / (lengths[i] ?? 1);
      }
    },
    dispose() {
      for (const geometry of geometries) geometry.dispose();
      for (const material of [armor, frame, alloy, rubber, hardware, status]) material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
