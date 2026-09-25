import * as THREE from 'three/webgpu';
import { instancedMesh } from '../instances';
import { engravedPanel, modelParts, robotMaterials } from '../robot';
import { makeWorld, type WorldCore } from '../world';
import { vineStrand } from '../props/vine';

export interface CourierPose { x: number; y: number; heading: number }

export interface CourierOptions {
  name?: string;
  scale?: number;
  tint?: THREE.ColorRepresentation;
  overgrowth?: number;
}

export function headingYaw(heading: number): number {
  return Math.PI / 2 - (heading & 3) * Math.PI / 2;
}

const ease = (t: number): number => t * t * (3 - 2 * t);

export function buildCourier(opts: CourierOptions = {}) {
  const world = makeWorld(opts.name ?? 'courier'), root = world.group, materials = robotMaterials();
  root.scale.setScalar(opts.scale ?? 1.9);
  Object.values(materials).forEach(material => world.own(material));
  if (opts.tint !== undefined) materials.accent.color.set(opts.tint);
  const { armor, accent, frame, alloy, rubber, status } = materials, parts = modelParts(world), { box, disc, rod } = parts;

  box(root, frame, 0, 0.27, 0, 0.66, 0.18, 0.49, 0.06);
  parts.add(root, engravedPanel(0.49, 0.63, 0.14, 0.045, 0.24).rotateZ(-Math.PI / 2).rotateX(-Math.PI / 2).translate(0, 0.39, 0), armor);
  for (const z of [-0.29, 0.29]) {
    box(root, armor, -0.015, 0.28, z, 0.42, 0.12, 0.09, 0.025);
    box(root, accent, 0.06, 0.345, z, 0.2, 0.018, 0.08, 0.006);
    for (let i = 0; i < 3; i++) box(root, frame, -0.19 + i * 0.034, 0.295, z * 1.17, 0.018, 0.045, 0.014, 0.003);
    for (const x of [-0.23, 0.23]) rod(root, alloy, new THREE.Vector3(x * 0.5, 0.28, z * 0.7), new THREE.Vector3(x, 0.15, z), 0.02);
  }
  for (let i = 0; i < 8; i++) {
    const angle = i * Math.PI / 4, x = Math.cos(angle) * 0.33, z = Math.sin(angle) * 0.32;
    parts.add(root, new THREE.CylinderGeometry(0.033, 0.039, 0.027, 20).rotateX(Math.PI / 2).rotateY(Math.PI / 2 - angle).translate(x, 0.32, z), alloy);
    parts.add(root, new THREE.SphereGeometry(0.025, 16, 10).scale(1, 1, 0.32).rotateY(Math.PI / 2 - angle).translate(x * 1.05, 0.32, z * 1.05), status);
  }
  box(root, accent, 0.39, 0.29, 0, 0.11, 0.14, 0.30, 0.035);
  box(root, rubber, 0.448, 0.29, 0, 0.02, 0.09, 0.25, 0.022);
  for (let i = -4; i <= 4; i++) rod(root, alloy, new THREE.Vector3(0.462, 0.251, i * 0.025), new THREE.Vector3(0.462, 0.329, i * 0.025), 0.003);
  for (const z of [-0.125, 0.125]) rod(root, frame, new THREE.Vector3(0.35, 0.29, z), new THREE.Vector3(0.08, 0.29, z * 1.8), 0.033);
  for (const z of [-0.135, 0, 0.135]) {
    box(root, frame, -0.26, 0.43, z, 0.1, 0.055, 0.095, 0.012);
    for (const y of [0.47, 0.59]) parts.add(root, new THREE.CylinderGeometry(0.049,0.049,0.025,20).translate(-0.26,y,z), alloy);
    for (const side of [-1, 1]) rod(root, accent, new THREE.Vector3(-0.26,0.47,z+side*0.041), new THREE.Vector3(-0.26,0.59,z+side*0.041), 0.005);
  }
  const wheel = new THREE.Group();
  parts.add(wheel, new THREE.TorusGeometry(0.11, 0.036, 12, 28), rubber);
  disc(wheel, frame, 0, 0, 0, 0.105, 0.09);
  for (const side of [-1, 1]) {
    disc(wheel, alloy, 0, 0, side * 0.058, 0.075, 0.025);
    disc(wheel, accent, 0, 0, side * 0.075, 0.032, 0.02, 8);
    for (let i = 0; i < 6; i++) disc(wheel, frame, Math.cos(i * Math.PI / 3) * 0.054, Math.sin(i * Math.PI / 3) * 0.054, side * 0.075, 0.009, 0.013, 6);
  }
  for (let i = 0; i < 20; i++) parts.add(wheel, new THREE.BoxGeometry(0.026, 0.014, 0.067).translate(0, 0.144, 0).rotateZ(i * Math.PI / 10), rubber);
  const indicators = world.own(instancedMesh(world.own(new THREE.SphereGeometry(1, 8, 6)), world.own(new THREE.MeshBasicMaterial({ color: 0xffffff })), 11));
  const sensor = new THREE.Object3D(), color = new THREE.Color();
  for (let i = 0; i < 11; i++) {
    if (i < 3) { sensor.position.set(-0.26, 0.53, (i - 1) * 0.135); sensor.scale.set(0.04, 0.065, 0.04); }
    else { const angle = (i - 3) * Math.PI / 4; sensor.position.set(Math.cos(angle) * 0.353, 0.32, Math.sin(angle) * 0.343); sensor.scale.setScalar(0.012); }
    sensor.updateMatrix(); indicators.setMatrixAt(i, sensor.matrix); indicators.setColorAt(i, color.setScalar(0.04));
  }
  indicators.name = 'localScentAndNeighbourSensors'; indicators.computeBoundingSphere(); world.add(indicators);
  parts.build();
  const wheelPositions = [-0.23, 0.23].flatMap(x => [-0.29, 0.29].map(z => new THREE.Vector3(x, 0.145, z)));
  const matrix = new THREE.Matrix4(), spins = new Float64Array(4);
  const wheels = wheel.children.map(child => {
    const mesh = child as THREE.Mesh;
    const instances = world.own(instancedMesh(mesh.geometry, mesh.material, 4));
    wheelPositions.forEach((position, i) => instances.setMatrixAt(i, matrix.makeTranslation(position)));
    instances.castShadow = instances.receiveShadow = true;
    instances.instanceMatrix.setUsage(THREE.DynamicDrawUsage); instances.computeBoundingSphere(); world.add(instances);
    return instances;
  });
  const overgrowth = opts.overgrowth ?? 0.5;
  if (overgrowth > 0) {
    const vines = world.own(vineStrand([[-0.28, 0.24, -0.2], [-0.1, 0.42, -0.26], [0.14, 0.4, -0.18], [0.3, 0.3, 0.02], [0.2, 0.44, 0.24], [-0.05, 0.36, 0.28]], 0.012 * overgrowth, 7 + Math.round(overgrowth * 9)));
    const vineMaterial = world.own(new THREE.MeshStandardMaterial({ color: 0x4f7a3c, vertexColors: true, roughness: 0.95 }));
    const vineMesh = new THREE.Mesh(vines, vineMaterial);
    vineMesh.castShadow = true;
    world.add(vineMesh);
  }
  const yawFrom = { value: 0 }, last = new THREE.Vector3(), tilt = new THREE.Quaternion(), normal = new THREE.Vector3();
  const handle = {
    drive(from: CourierPose, to: CourierPose, alpha: number, ground: (x: number, z: number) => number, snap = false) {
      const turning = (from.heading & 3) !== (to.heading & 3);
      const a = headingYaw(from.heading), b = headingYaw(to.heading);
      const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
      const yaw = snap ? b : turning ? a + delta * ease(Math.min(1, alpha)) : b;
      const t = snap ? 1 : turning ? 0 : Math.min(1, alpha);
      const x = from.x + (to.x - from.x) * t, z = from.y + (to.y - from.y) * t, y = ground(x, z);
      const travel = snap ? 0 : Math.hypot(x - last.x, z - last.z);
      const turn = snap ? 0 : Math.atan2(Math.sin(yaw - yawFrom.value), Math.cos(yaw - yawFrom.value));
      last.set(x, y, z); yawFrom.value = yaw;
      root.position.set(x, y, z);
      root.rotation.set(0, yaw, 0);
      normal.set(ground(x - 0.25, z) - ground(x + 0.25, z), 0.5, ground(x, z - 0.25) - ground(x, z + 0.25)).normalize();
      root.quaternion.premultiply(tilt.setFromUnitVectors(THREE.Object3D.DEFAULT_UP, normal));
      handle.roll(travel, turn);
    },
    sense(values: readonly number[]) {
      for (let i = 0; i < 11; i++) indicators.setColorAt(i, color.set(i < 3 ? [0xff786c, 0xc3e26e, 0xad9fff][i]! : 0x6db9c1).multiplyScalar(0.045 + Math.log1p(values[i]!) * 0.6));
      indicators.instanceColor!.needsUpdate = true;
    },
    roll(distance: number, turn: number) {
      wheelPositions.forEach((position, i) => {
        spins[i]! -= (distance + turn * position.z) / (0.145 * root.scale.x);
        matrix.makeRotationZ(spins[i]!).setPosition(position);
        for (const wheel of wheels) wheel.setMatrixAt(i, matrix);
      });
      for (const wheel of wheels) wheel.instanceMatrix.needsUpdate = true;
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}

