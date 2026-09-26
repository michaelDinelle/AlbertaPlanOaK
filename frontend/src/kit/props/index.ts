import * as THREE from 'three/webgpu';
import { modelParts } from '../robot';
import type { WorldBuilder } from '../world';
import { propMaterials, type PropMaterials } from './materials';
import { vineStrand } from './vine';

export { propMaterials, vineStrand };
export { hoodooGeometry } from './hoodoo';

const materialsOf = (world: WorldBuilder, given?: PropMaterials): PropMaterials => given ?? propMaterials(world);

export function relief(world: WorldBuilder, width: number, height: number, overgrowth = 0.6, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group();
  const courses = Math.max(2, Math.round(height / 0.36));
  for (let row = 0; row < courses; row++) {
    const offset = (row % 2) * 0.35, blocks = Math.ceil(width / 0.7) + 1;
    for (let i = 0; i < blocks; i++) {
      const x = -width / 2 + i * 0.7 + offset - 0.35, w = Math.min(0.68, width / 2 - x + 0.34);
      if (w <= 0.1 || x > width / 2) continue;
      const wear = 0.94 + ((row * 7 + i * 3) % 5) * 0.012;
      parts.box(group, row % 3 === 2 ? m.shale : m.sandstone, x + w / 2, 0.18 + row * 0.36, 0, w * wear, 0.34 * wear, 0.42, 0.03);
    }
  }
  parts.box(group, m.brass, 0, height + 0.02, 0, width + 0.1, 0.05, 0.5, 0.01);
  parts.build();
  const strands = Math.round(overgrowth * Math.max(1, width / 1.2));
  for (let s = 0; s < strands; s++) {
    const x = -width / 2 + (s + 0.5) * width / strands, points: [number, number, number][] = [];
    for (let k = 0; k < 5; k++) points.push([x + Math.sin(s * 3.1 + k * 1.7) * 0.25, height * (1 - k / 4.5), 0.24 + Math.sin(k * 2.1 + s) * 0.03]);
    const mesh = new THREE.Mesh(world.own(vineStrand(points, 0.018, 9)), m.vine);
    mesh.castShadow = true;
    group.add(mesh);
  }
  return group;
}

export function gate(world: WorldBuilder, width: number, height: number, tint: THREE.ColorRepresentation, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group(), glass = m.glass(tint, 0.7);
  for (const side of [-1, 1]) { parts.box(group, m.sandstone, side * (width / 2 + 0.2), height / 2, 0, 0.4, height, 0.5, 0.04); parts.box(group, m.brass, side * (width / 2 + 0.2), height + 0.03, 0, 0.5, 0.06, 0.6, 0.01); }
  parts.box(group, m.sandstone, 0, height + 0.25, 0, width + 0.8, 0.4, 0.55, 0.05);
  parts.box(group, m.brass, 0, 0.03, 0, width, 0.06, 0.6, 0.01);
  parts.box(group, glass, 0, 0.07, 0, width * 0.9, 0.03, 0.4, 0.01);
  const leaves = [-1, 1].map(side => { const leaf = new THREE.Group(); leaf.position.set(side * width / 2, 0, 0); group.add(leaf); return leaf; });
  leaves.forEach((leaf, i) => { const side = i === 0 ? 1 : -1; parts.box(leaf, m.brass, side * width / 4, height / 2, 0, width / 2 - 0.04, height - 0.1, 0.06, 0.02); parts.box(leaf, glass, side * width / 4, height * 0.62, 0.04, width / 2 - 0.24, 0.05, 0.02, 0.005); });
  parts.build();
  return {
    group, glass,
    open(t: number) { leaves.forEach((leaf, i) => { leaf.rotation.y = (i === 0 ? -1 : 1) * t * 1.35; }); },
    lit(strength: number) { glass.emissiveIntensity = 0.7 + strength * 2.4; },
  };
}

export function lamp(world: WorldBuilder, tint: THREE.ColorRepresentation, height = 1.6, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group(), glass = m.glass(tint, 1.2);
  parts.box(group, m.sandstone, 0, 0.1, 0, 0.3, 0.2, 0.3, 0.03);
  parts.rod(group, m.brass, new THREE.Vector3(0, 0.2, 0), new THREE.Vector3(0, height, 0), 0.03);
  parts.box(group, m.brass, 0, height + 0.02, 0, 0.26, 0.04, 0.26, 0.01);
  parts.box(group, glass, 0, height + 0.18, 0, 0.16, 0.26, 0.16, 0.02);
  parts.box(group, m.brass, 0, height + 0.34, 0, 0.22, 0.05, 0.22, 0.01);
  parts.build();
  const light = new THREE.PointLight(tint, 0, 6, 2);
  light.position.set(0, height + 0.18, 0);
  group.add(light);
  return { group, glass, light, lit(strength: number) { glass.emissiveIntensity = 0.4 + strength * 1.6; light.intensity = strength * 2.5; } };
}

export function brazier(world: WorldBuilder, tint: THREE.ColorRepresentation, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group(), glass = m.glass(tint, 1.5);
  parts.disc(group, m.sandstone, 0, 0.08, 0, 0.3, 0.16, 12);
  parts.rod(group, m.brass, new THREE.Vector3(0, 0.16, 0), new THREE.Vector3(0, 0.7, 0), 0.04);
  parts.add(group, new THREE.CylinderGeometry(0.32, 0.18, 0.22, 12, 1, true).translate(0, 0.8, 0), m.brass);
  parts.add(group, new THREE.SphereGeometry(0.2, 12, 8).scale(1, 0.6, 1).translate(0, 0.86, 0), glass);
  parts.build();
  return { group, glass, lit(strength: number) { glass.emissiveIntensity = 0.3 + strength * 2.2; } };
}

export function plinth(world: WorldBuilder, width: number, height: number, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group();
  parts.box(group, m.sandstone, 0, height * 0.15, 0, width * 1.2, height * 0.3, width * 1.2, 0.04);
  parts.box(group, m.sandstone, 0, height * 0.6, 0, width, height * 0.6, width, 0.04);
  parts.box(group, m.brass, 0, height * 0.92, 0, width * 1.1, height * 0.06, width * 1.1, 0.01);
  parts.box(group, m.sandstone, 0, height, 0, width * 1.15, height * 0.1, width * 1.15, 0.03);
  parts.build();
  return group;
}

export function banner(world: WorldBuilder, tint: THREE.ColorRepresentation, height = 2.4, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group();
  parts.rod(group, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, height, 0), 0.03);
  parts.rod(group, m.brass, new THREE.Vector3(-0.42, height - 0.1, 0), new THREE.Vector3(0.42, height - 0.1, 0), 0.02);
  parts.build();
  const cloth = new THREE.Mesh(world.own(new THREE.PlaneGeometry(0.76, height * 0.55, 4, 10).translate(0, height - 0.12 - height * 0.275, 0.02)), m.cloth(tint));
  cloth.castShadow = true;
  group.add(cloth);
  return group;
}

export function tokenRing(world: WorldBuilder, slots: number, radius: number, tint: THREE.ColorRepresentation, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group(), glass = m.glass(tint, 0.6);
  parts.add(group, new THREE.TorusGeometry(radius, 0.06, 8, 48).rotateX(Math.PI / 2).translate(0, 0.04, 0), m.brass);
  const tokens = Array.from({ length: slots }, (_, i) => {
    const angle = i / slots * Math.PI * 2, token = new THREE.Group();
    token.position.set(Math.cos(angle) * radius, 0.12, Math.sin(angle) * radius);
    group.add(token);
    parts.disc(token, m.sandstone, 0, 0, 0, 0.22, 0.08, 10);
    parts.disc(token, glass, 0, 0.05, 0, 0.14, 0.03, 10);
    return token;
  });
  parts.build();
  const lights = tokens.map(token => { const mesh = token.children[1] as THREE.Mesh | undefined; return mesh; });
  return {
    group, tokens, glass,
    lift(index: number, amount: number) { tokens.forEach((token, i) => { token.position.y = 0.12 + (i === index ? amount : 0); }); },
    lightCount: lights.length,
  };
}

export function signalPole(world: WorldBuilder, bits: number, tint: THREE.ColorRepresentation, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group();
  const height = 0.6 + bits * 0.22;
  parts.box(group, m.sandstone, 0, 0.1, 0, 0.36, 0.2, 0.36, 0.03);
  parts.rod(group, m.brass, new THREE.Vector3(0, 0.2, 0), new THREE.Vector3(0, height, 0), 0.03);
  parts.build();
  const lampGeometry = world.own(new THREE.SphereGeometry(0.07, 10, 8));
  const lamps = Array.from({ length: bits }, (_, i) => {
    const material = m.glass(tint, 0.1), mesh = new THREE.Mesh(lampGeometry, material);
    mesh.position.set(0, 0.5 + i * 0.22, 0.08);
    group.add(mesh);
    return material;
  });
  return { group, set(value: number) { lamps.forEach((material, i) => { material.emissiveIntensity = ((value >>> i) & 1) ? 1.6 : 0.08; }); } };
}

export function doorway(world: WorldBuilder, width: number, height: number, materials?: PropMaterials) {
  const m = materialsOf(world, materials), parts = modelParts(world), group = new THREE.Group();
  for (const side of [-1, 1]) parts.box(group, m.sandstone, side * (width / 2 + 0.16), height / 2, 0, 0.32, height, 0.45, 0.04);
  parts.box(group, m.sandstone, 0, height + 0.18, 0, width + 0.64, 0.36, 0.5, 0.05);
  parts.box(group, m.brass, 0, height + 0.38, 0, width + 0.72, 0.05, 0.56, 0.01);
  parts.build();
  return group;
}
