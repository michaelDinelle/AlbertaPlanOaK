import * as THREE from 'three';
import { bevelBox, engravedPanel, modelParts, robotMaterials } from '@kit/robot';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { makeWorld, type WorldCore } from '@kit/world';
import { CHEETAH_BONES, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';

export interface CheetahHandle extends WorldCore {
  joints: THREE.Object3D[];
  setPose(fk: Float32Array): void;
}

export function buildCheetah(cheetah: CheetahGeometry): CheetahHandle {
  const world = makeWorld('fieldRobot'), group = world.group, parts = modelParts(world);
  const materials = robotMaterials(), { armor, accent, frame, alloy, rubber, hardware, status } = materials;
  Object.values(materials).forEach(material => world.own(material));
  const add = (root: THREE.Group, geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0) => parts.add(root, geometry.translate(x, y, z), material);
  const disc = (radius: number, depth: number, sides: number) => new THREE.CylinderGeometry(radius, radius, depth, sides).rotateX(Math.PI / 2);
  const gearGeometries = [0.087, 0.064, 0.055].map((radius) => {
    const parts: ColoredPart[] = [];
    parts.push({ geometry: disc(radius, 0.14, 14), color: frame.color });
    for (const side of [-1, 1]) {
      parts.push({ geometry: disc(radius * 0.72, 0.012, 12).translate(0, 0, side * 0.076), color: alloy.color });
      parts.push({ geometry: disc(radius * 0.31, 0.015, 6).translate(0, 0, side * 0.085), color: accent.color });
      for (let screw = 0; screw < 4; screw++) {
        const angle = screw * Math.PI * 0.5 + 0.35;
        const fastener = disc(0.007, 0.014, 5).translate(Math.cos(angle) * radius * 0.53, Math.sin(angle) * radius * 0.53, side * 0.083);
        parts.push({ geometry: fastener, color: rubber.color });
      }
    }
    return world.own(mergeColoredParts(parts, 'robotGearbox'));
  });
  const padGeometry = world.own(new THREE.SphereGeometry(cheetah.radius, 12, 6));
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
      add(link, engravedPanel(0.32, length * 0.87, 0.18, 0.035, 0.24).rotateY(-Math.PI / 2), armor, -0.067);
      add(link, bevelBox(0.11, length * 0.66, 0.27, 0.018), frame, 0.115);
      for (const side of [-1, 1]) {
        add(link, bevelBox(0.13, length * 0.72, 0.014, 0.014), armor, -0.025, 0, side * 0.171);
        add(link, bevelBox(0.026, 0.12, 0.008, 0.004), status, -0.058, 0.24, side * 0.184);
        for (let vent = 0; vent < 4; vent++) add(link, new THREE.BoxGeometry(0.05, 0.012, 0.008), frame, 0.018, -0.285 + vent * 0.027, side * 0.183);
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
  parts.build();
  return world.seal<CheetahHandle>({
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
  });
}
