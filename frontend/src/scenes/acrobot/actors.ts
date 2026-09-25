import * as THREE from 'three/webgpu';
import { oakMaterial } from '@kit/materials';
import { iceMaterial } from '@kit/props/ice';
import { bevelBox, engravedPanel, modelParts } from '@kit/robot';
import { makeWorld, type WorldCore } from '@kit/world';
import { AB_L1, AB_L2, AB_SUCCESS_HEIGHT } from '@sim/envs/acrobot';
import { DECK_Y, lakeHeight, PIER_X, PIER_Z } from './world';

export interface ArmHandle extends WorldCore {
  tipWorld: THREE.Vector3;
  elbowWorld: THREE.Vector3;
  setState(th1: number, th2: number): void;
  setStatus(strength: number, chill: number): void;
}

export interface ArmOptions {
  warm: THREE.ColorRepresentation;
  status: THREE.ColorRepresentation;
  frost: boolean;
  blind: boolean;
}

export function buildArm(pivot: readonly [number, number, number], opts: ArmOptions): ArmHandle {
  const world = makeWorld('alpineAcrobot');
  const root = world.add(new THREE.Group());
  root.position.fromArray(pivot);
  const rim = { rimColor: opts.warm, rimStrength: 0.55, rimPower: 2.6, toonSteps: 3, toonSoftness: 0.3, toonFloor: 0.14 };
  const frame = world.own(oakMaterial({ color: 0x687887, roughness: 0.58, metalness: 0.45, ...rim, name: 'armFrame' }));
  const armor = world.own(oakMaterial({ color: 0xe4e2da, vertexColors: true, roughness: 0.5, metalness: 0.16, ...rim, name: 'armArmor' }));
  const accent = world.own(oakMaterial({ color: 0xdab96c, roughness: 0.5, metalness: 0.45, ...rim, name: 'armAccent' }));
  const alloy = opts.frost ? iceMaterial(world, 0xd6ecfa, 'armFrost') : world.own(oakMaterial({ color: 0xbfcbd0, roughness: 0.55, metalness: 0.48, ...rim, name: 'armAlloy' }));
  const rubber = world.own(oakMaterial({ color: 0x303a43, roughness: 0.92, ...rim, rimStrength: 0.3, name: 'armRubber' }));
  const statusColor = new THREE.Color(opts.status), chillColor = new THREE.Color(0xbfefff);
  const status = world.own(new THREE.MeshStandardMaterial({ color: opts.blind ? 0x1a2128 : opts.status, roughness: 0.3, emissive: opts.status, emissiveIntensity: opts.blind ? 0 : 0.35 }));
  const parts = modelParts(world), { box, disc } = parts;
  const strut = (parent: THREE.Group, a: THREE.Vector3, b: THREE.Vector3, radius: number) => parts.rod(parent, alloy, a, b, radius);
  for (const side of [-1, 1]) {
    const x = side * PIER_X, bed = lakeHeight(x, PIER_Z) - pivot[1];
    box(root, frame, x, (bed + DECK_Y) / 2, PIER_Z, 0.19, DECK_Y - bed, 0.24, 0.025);
    box(root, alloy, x, bed + 0.05, PIER_Z, 0.75, 0.18, 1.1, 0.04);
    box(root, armor, x, -1.1, -0.55, 0.23, 1.6, 0.04);
    box(root, rubber, x, 0.54, PIER_Z, 0.26, 0.12, 0.3);
    for (const z of [-1.12, -0.28]) for (const dx of [-0.27, 0.27]) disc(root, frame, x + dx, bed + 0.16, z, 0.035, 0.03, 6);
    strut(root, new THREE.Vector3(x, PIER_Z, PIER_Z), new THREE.Vector3(x - side * 0.85, 0.5, PIER_Z), 0.034);
    for (let j = 0; j < 4; j++) box(root, status, x, AB_SUCCESS_HEIGHT + (j - 1.5) * 0.065, -0.53, 0.12, 0.014, 0.01, 0.003);
    box(root, frame, x, 0.85, PIER_Z, 0.08, 0.65, 0.12);
  }
  box(root, frame, 0, 0.5, PIER_Z, 5.42, 0.2, 0.28, 0.045);
  box(root, alloy, 0, 0.615, PIER_Z, 5.25, 0.025, 0.2, 0.006);
  parts.add(root, engravedPanel(0.46, 0.35, 0.16, 0.055, 0.28).translate(0, 0.39, -0.62), armor);
  disc(root, frame, 0, 0, -0.35, 0.2, 0.55);
  disc(root, alloy, 0, 0, -0.065, 0.23, 0.06);
  const first = new THREE.Group(), second = new THREE.Group();
  root.add(first);
  first.add(second);
  second.position.y = -AB_L1;
  for (const [node, length] of [[first, AB_L1], [second, AB_L2]] as const) {
    const upper = node === first, front = upper ? 0.146 : 0.126;
    const shell = bevelBox(upper ? 0.31 : 0.245, length - 0.24, upper ? 0.25 : 0.21, 0.04);
    parts.add(node, shell.translate(0, -length / 2, 0), armor);
    box(node, frame, 0, -length / 2, -front, 0.075, length - 0.35, 0.04, 0.015);
    for (const y of [-0.24, -length + 0.24]) for (const x of [-0.1, 0.1]) disc(node, alloy, x, y, front, 0.016, 0.014, 6);
    if (!upper) {
      box(node, rubber, 0, -length * 0.56, front, 0.09, length * 0.29, 0.015, 0.009);
      for (let slot = 0; slot < 5; slot++) box(node, alloy, 0, -length * 0.46 - slot * 0.045, front + 0.012, 0.05, 0.016, 0.009, 0.004);
    }
    box(node, status, 0, -length * 0.27, front + 0.006, 0.055, 0.052, 0.012, 0.008);
    for (const x of [-0.16, 0.16]) strut(node, new THREE.Vector3(x, -0.18, 0), new THREE.Vector3(x, -length + 0.18, 0), 0.021);
    disc(node, rubber, 0, 0, 0, 0.175, 0.24);
    for (const sign of [-1, 1]) {
      disc(node, alloy, 0, 0, sign * 0.137, 0.156, 0.036);
      disc(node, frame, 0, 0, sign * 0.16, 0.115, 0.018);
      disc(node, accent, 0, 0, sign * 0.178, 0.055, 0.022);
      for (let bolt = 0; bolt < 8; bolt++) {
        const angle = bolt * Math.PI / 4;
        disc(node, frame, Math.cos(angle) * 0.135, Math.sin(angle) * 0.135, sign * 0.161, 0.013, 0.014, 6);
      }
    }
  }
  disc(second, rubber, 0, -AB_L2, 0, 0.15, 0.2);
  disc(second, alloy, 0, -AB_L2, 0.12, 0.128, 0.04);
  disc(second, status, 0, -AB_L2, 0.146, 0.063, 0.012);
  parts.build();
  const tipWorld = new THREE.Vector3(), elbowWorld = new THREE.Vector3();
  return world.seal<ArmHandle>({
    tipWorld,
    elbowWorld,
    setState(th1: number, th2: number) {
      first.rotation.z = th1;
      second.rotation.z = th2;
      elbowWorld.set(pivot[0] + AB_L1 * Math.sin(th1), pivot[1] - AB_L1 * Math.cos(th1), pivot[2]);
      tipWorld.set(pivot[0] + AB_L1 * Math.sin(th1) + AB_L2 * Math.sin(th1 + th2), pivot[1] - AB_L1 * Math.cos(th1) - AB_L2 * Math.cos(th1 + th2), pivot[2]);
    },
    setStatus(strength: number, chill: number) {
      if (opts.blind) return;
      status.emissive.copy(statusColor).lerp(chillColor, chill);
      status.emissiveIntensity = 0.35 + strength * 1.4 + chill * 2;
    },
  });
}
