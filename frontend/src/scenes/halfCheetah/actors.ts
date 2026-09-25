import * as THREE from 'three/webgpu';
import { bevelBox, engravedPanel, modelParts, robotMaterials } from '@kit/robot';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { makeWorld, type WorldCore } from '@kit/world';
import { cheetahFrames, farEnd, type Capsule, type CheetahModel } from './body';

export interface CheetahHandle extends WorldCore {
  frames: Float32Array;
  setPose(pose: ArrayLike<number>): void;
  setEffort(effort: ArrayLike<number>): void;
}

export function buildCheetah(model: CheetahModel, glow: THREE.ColorRepresentation): CheetahHandle {
  const world = makeWorld('fieldRobot'), group = world.group, parts = modelParts(world);
  const materials = robotMaterials(), { armor, accent, frame, alloy, rubber, hardware, status } = materials;
  Object.values(materials).forEach(material => world.own(material));
  const bodies = model.parent.map((_, k) => {
    const body = new THREE.Group();
    body.name = k === 0 ? 'robotChassis' : `robotLink${k}`;
    group.add(body);
    return body;
  });
  const along = (geometry: THREE.BufferGeometry, x: number, z: number, ax: number, az: number) => geometry.rotateZ(Math.atan2(-ax, az)).translate(x, z, 0);
  const add = (body: THREE.Group, capsule: Capsule, sign: number, geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0) =>
    parts.add(body, along(geometry.translate(x, y, z), capsule.x, capsule.z, sign * capsule.ax, sign * capsule.az), material);
  const disc = (radius: number, depth: number, sides: number) => new THREE.CylinderGeometry(radius, radius, depth, sides).rotateX(Math.PI / 2);
  const gearGeometries = [0.087, 0.064, 0.055].map((radius) => {
    const gear: ColoredPart[] = [{ geometry: disc(radius, 0.14, 14), color: frame.color }];
    for (const side of [-1, 1]) {
      gear.push({ geometry: disc(radius * 0.72, 0.012, 12).translate(0, 0, side * 0.076), color: alloy.color });
      for (let screw = 0; screw < 4; screw++) {
        const angle = screw * Math.PI * 0.5 + 0.35;
        gear.push({ geometry: disc(0.007, 0.014, 5).translate(Math.cos(angle) * radius * 0.53, Math.sin(angle) * radius * 0.53, side * 0.083), color: rubber.color });
      }
    }
    return world.own(mergeColoredParts(gear, 'robotGearbox'));
  });
  const hubGeometries = [0.087, 0.064, 0.055].map(radius => world.own(mergeColoredParts([-1, 1].map(side => ({ geometry: disc(radius * 0.31, 0.015, 6).translate(0, 0, side * 0.085), color: 0xffffff })), 'robotGearHub')));
  const hubs = model.parent.slice(1).map((_, j) => {
    const hub = world.own(new THREE.MeshStandardNodeMaterial({ color: accent.color, roughness: accent.roughness, metalness: accent.metalness, emissive: glow, emissiveIntensity: 0 }));
    hub.name = `robotTorqueHub${j}`;
    return hub;
  });
  model.parent.forEach((parent, k) => {
    if (parent < 0) return;
    const level = model.parent[parent]! < 0 ? 0 : model.parent[model.parent[parent]!]! < 0 ? 1 : 2;
    for (const [geometry, material] of [[gearGeometries[level]!, hardware], [hubGeometries[level]!, hubs[k - 1]!]] as const) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      bodies[k]!.add(mesh);
    }
  });
  const pad = world.own(new THREE.SphereGeometry(model.radius, 12, 6));
  model.capsules.forEach((capsule, i) => {
    const body = bodies[capsule.body]!, length = 2 * capsule.half;
    if (i === 0) {
      const sign = capsule.ax < 0 ? -1 : 1;
      add(body, capsule, sign, bevelBox(0.22, length * 0.96, 0.28, 0.035), frame, 0.015);
      add(body, capsule, sign, engravedPanel(0.32, length * 0.87, 0.18, 0.035, 0.24).rotateY(-Math.PI / 2), armor, -0.067);
      add(body, capsule, sign, bevelBox(0.11, length * 0.66, 0.27, 0.018), frame, 0.115);
      for (const side of [-1, 1]) {
        add(body, capsule, sign, bevelBox(0.13, length * 0.72, 0.014, 0.014), armor, -0.025, 0, side * 0.171);
        add(body, capsule, sign, bevelBox(0.026, 0.12, 0.008, 0.004), status, -0.058, 0.24, side * 0.184);
        for (let vent = 0; vent < 4; vent++) add(body, capsule, sign, new THREE.BoxGeometry(0.05, 0.012, 0.008), frame, 0.018, -0.285 + vent * 0.027, side * 0.183);
      }
      for (const end of [-1, 1]) add(body, capsule, sign, bevelBox(0.18, 0.065, 0.28, 0.014), rubber, 0.01, end * length * 0.46);
      return;
    }
    if (capsule.body === 0) {
      const [tx, tz] = farEnd(capsule), pod = { ...capsule, x: tx, z: tz, ax: 0, az: 1 };
      add(body, capsule, 1, bevelBox(0.07, length * 0.8, 0.12, 0.012), frame);
      for (const side of [-1, 1]) add(body, capsule, 1, bevelBox(0.05, length * 0.5, 0.012, 0.006), alloy, 0, -0.02, side * 0.066);
      add(body, pod, 1, bevelBox(0.11, 0.09, 0.2, 0.03), armor);
      add(body, pod, 1, bevelBox(0.014, 0.034, 0.17, 0.005), status, 0.07, 0.006);
      return;
    }
    const children = model.parent.filter(p => p === capsule.body).length;
    const upper = model.parent[capsule.body] === 0, foot = children === 0;
    add(body, capsule, 1, bevelBox(upper ? 0.095 : 0.058, Math.max(0.035, length - 0.035), upper ? 0.12 : 0.072, 0.011), foot ? rubber : frame);
    if (!foot) for (const side of [-1, 1]) add(body, capsule, 1, bevelBox(upper ? 0.075 : 0.045, length * 0.55, 0.013, 0.008), upper ? armor : alloy, 0, 0, side * (upper ? 0.069 : 0.043));
    if (foot) {
      const [tx, tz] = farEnd(capsule), toe = new THREE.Mesh(pad, rubber);
      toe.position.set(tx, tz, 0);
      toe.castShadow = true;
      body.add(toe);
    }
  });
  parts.build();
  const frames = new Float32Array(model.parent.length * 3);
  return world.seal<CheetahHandle>({
    frames,
    setPose(pose) {
      cheetahFrames(model, pose, frames);
      bodies.forEach((body, k) => {
        body.position.set(frames[k * 3]!, frames[k * 3 + 1]!, 0);
        body.rotation.z = -frames[k * 3 + 2]!;
      });
    },
    setEffort(effort) {
      hubs.forEach((hub, j) => { hub.emissiveIntensity = 2.4 * Math.min(1, Math.abs(effort[j] ?? 0)); });
    },
  });
}
