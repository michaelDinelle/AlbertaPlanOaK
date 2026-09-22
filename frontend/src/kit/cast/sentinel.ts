import * as THREE from 'three';
import { engravedPanel, modelParts } from '../robot';
import { makeWorld, type WorldCore } from '../world';
import { propMaterials } from '../props/materials';

export interface SentinelOptions {
  name?: string;
  scale?: number;
  tint?: THREE.ColorRepresentation;
}

export function buildSentinel(opts: SentinelOptions = {}) {
  const world = makeWorld(opts.name ?? 'sentinel'), root = world.group, materials = propMaterials(world);
  root.scale.setScalar(opts.scale ?? 1);
  const visor = world.own(new THREE.MeshStandardMaterial({ color: opts.tint ?? 0x5ec1ff, emissive: opts.tint ?? 0x5ec1ff, emissiveIntensity: 0.9, roughness: 0.3 }));
  const armor = world.own(new THREE.MeshStandardMaterial({ color: 0xe3d6b8, vertexColors: true, roughness: 0.85 }));
  const parts = modelParts(world), { box, disc, rod } = parts;
  const torso = new THREE.Group(), head = new THREE.Group();
  const legs = [-1, 1].map(() => ({ upper: new THREE.Group(), lower: new THREE.Group() }));
  const arms = [-1, 1].map(() => new THREE.Group());
  root.add(torso); torso.add(head);
  box(torso, materials.sandstone, 0, 0.98, 0, 0.5, 0.58, 0.32, 0.05);
  parts.add(torso, engravedPanel(0.3, 0.34, 0.06, 0.03, 0.17).translate(0, 1.0, 0.19), armor);
  box(torso, materials.brass, 0, 1.3, 0, 0.58, 0.08, 0.36, 0.02);
  for (const side of [-1, 1]) { disc(torso, materials.brass, side * 0.31, 1.24, 0, 0.09, 0.1, 12); box(torso, materials.sandstone, side * 0.2, 0.62, 0, 0.16, 0.16, 0.24, 0.03); }
  head.position.set(0, 1.5, 0);
  box(head, materials.sandstone, 0, 0.14, 0, 0.3, 0.3, 0.3, 0.05);
  box(head, visor, 0, 0.16, 0.14, 0.2, 0.05, 0.04, 0.01);
  box(head, materials.brass, 0, 0.31, 0, 0.34, 0.04, 0.34, 0.01);
  legs.forEach((leg, i) => {
    const side = i === 0 ? -1 : 1;
    leg.upper.position.set(side * 0.16, 0.66, 0); root.add(leg.upper);
    rod(leg.upper, materials.sandstone, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.32, 0), 0.06);
    leg.lower.position.set(0, -0.32, 0); leg.upper.add(leg.lower);
    rod(leg.lower, materials.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.3, 0), 0.045);
    box(leg.lower, materials.sandstone, 0, -0.33, 0.04, 0.12, 0.06, 0.22, 0.02);
  });
  arms.forEach((arm, i) => {
    const side = i === 0 ? -1 : 1;
    arm.position.set(side * 0.34, 1.22, 0); torso.add(arm);
    rod(arm, materials.sandstone, new THREE.Vector3(0, 0, 0), new THREE.Vector3(side * 0.04, -0.42, 0), 0.05);
    disc(arm, materials.brass, side * 0.04, -0.46, 0, 0.06, 0.06, 10);
  });
  parts.build();
  let phase = 0;
  const handle = {
    stride(speed: number, dt: number) {
      phase += speed * dt * 6;
      const swing = Math.sin(phase) * Math.min(0.7, speed * 0.6);
      legs.forEach((leg, i) => { const s = i === 0 ? 1 : -1; leg.upper.rotation.x = swing * s; leg.lower.rotation.x = Math.max(0, -swing * s) * 0.9; });
      arms.forEach((arm, i) => { const s = i === 0 ? -1 : 1; arm.rotation.x = swing * s * 0.6; });
      torso.position.y = Math.abs(Math.sin(phase)) * Math.min(0.04, speed * 0.03);
    },
    idle(time: number) {
      torso.rotation.z = Math.sin(time * 0.7) * 0.012;
      head.rotation.y = Math.sin(time * 0.35) * 0.18;
      torso.position.y = Math.sin(time * 1.4) * 0.008;
    },
    face(yaw: number) { root.rotation.y = yaw; },
    place(x: number, y: number, z: number) { root.position.set(x, y, z); },
    glow(strength: number) { visor.emissiveIntensity = 0.5 + strength * 2.2; },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
