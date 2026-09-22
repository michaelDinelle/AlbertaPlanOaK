import * as THREE from 'three';
import { modelParts } from '../robot';
import type { WorldBuilder } from '../world';
import { propMaterials, type PropMaterials } from './materials';

export function dial(world: WorldBuilder, radius: number, ticks: number, tint: THREE.ColorRepresentation, materials?: PropMaterials, majorEvery = 10) {
  const m = materials ?? propMaterials(world), parts = modelParts(world), group = new THREE.Group(), glass = m.glass(tint, 1.4);
  parts.add(group, new THREE.CylinderGeometry(radius * 1.08, radius * 1.16, 0.12, 48).translate(0, 0.06, 0), m.sandstone);
  parts.add(group, new THREE.TorusGeometry(radius, 0.03, 8, 64).rotateX(Math.PI / 2).translate(0, 0.13, 0), m.brass);
  for (let i = 0; i < ticks; i++) {
    const angle = i / ticks * Math.PI * 2, major = i % majorEvery === 0, length = major ? 0.16 : 0.07;
    parts.add(group, new THREE.BoxGeometry(0.02, 0.012, length).translate(0, 0.13, radius - 0.05 - length / 2).rotateY(-angle), major ? m.brass : m.shale);
  }
  const needle = new THREE.Group();
  needle.position.y = 0.15;
  group.add(needle);
  parts.add(needle, new THREE.BoxGeometry(0.035, 0.02, radius * 0.86).translate(0, 0, radius * 0.36), m.brass);
  parts.add(needle, new THREE.SphereGeometry(0.05, 10, 8).translate(0, 0.01, radius * 0.8), glass);
  parts.add(needle, new THREE.CylinderGeometry(0.09, 0.09, 0.05, 16), m.brass);
  parts.build();
  return { group, needle, glass, setAngle(angle: number) { needle.rotation.y = -angle; } };
}
