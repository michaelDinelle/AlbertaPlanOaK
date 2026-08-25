import * as THREE from 'three';
import { oakMaterial, type OakMaterial } from './materials';
import { blendTokens } from './mix';
import type { Palette } from './tokens';

export interface FigureBag {
  group: THREE.Group;
  material: OakMaterial;
  shadeMaterial: OakMaterial;
  joint(parent: THREE.Object3D, name: string, x?: number, y?: number, z?: number): THREE.Object3D;
  limb(parent: THREE.Object3D, length: number, radius: number, name: string, shaded?: boolean): THREE.Mesh;
  rod(parent: THREE.Object3D, length: number, radius: number, name: string, shaded?: boolean): THREE.Mesh;
  orb(parent: THREE.Object3D, radius: number, name: string, shaded?: boolean): THREE.Mesh;
  box(parent: THREE.Object3D, wx: number, wy: number, wz: number, name: string, shaded?: boolean): THREE.Mesh;
  dispose(): void;
}

export interface FigureOptions {
  palette: Palette;
  accent: number;
  name: string;
}

export function makeFigure(opts: FigureOptions): FigureBag {
  const { palette, accent, name } = opts;
  const group = new THREE.Group();
  group.name = name;
  const geometries: THREE.BufferGeometry[] = [];

  const skin = (color: THREE.Color, tag: string): OakMaterial =>
    oakMaterial({
      color,
      flatShading: true,
      roughness: 0.62,
      rimColor: palette.hemiSky,
      rimStrength: 0.55,
      toonSteps: 3,
      toonSoftness: 0.34,
      toonFloor: 0.12,
      name: `${name}${tag}`,
    });
  const material = skin(blendTokens(accent, palette.hemiGround, 0.18), 'Skin');
  const shadeMaterial = skin(blendTokens(accent, palette.hemiGround, 0.55), 'SkinShade');

  const mesh = (geometry: THREE.BufferGeometry, parent: THREE.Object3D, meshName: string, shaded: boolean): THREE.Mesh => {
    geometries.push(geometry);
    const m = new THREE.Mesh(geometry, shaded ? shadeMaterial : material);
    m.name = meshName;
    m.castShadow = true;
    parent.add(m);
    return m;
  };

  return {
    group,
    material,
    shadeMaterial,

    joint(parent, jointName, x = 0, y = 0, z = 0) {
      const node = new THREE.Object3D();
      node.name = jointName;
      node.position.set(x, y, z);
      parent.add(node);
      return node;
    },

    limb(parent, length, radius, limbName, shaded = false) {
      const g = new THREE.CapsuleGeometry(radius, length, 3, 7);
      g.translate(0, -length * 0.5, 0);
      return mesh(g, parent, limbName, shaded);
    },

    rod(parent, length, radius, rodName, shaded = false) {
      const g = new THREE.CapsuleGeometry(radius, length, 3, 7);
      g.rotateZ(Math.PI / 2);
      g.translate(length * 0.5, 0, 0);
      return mesh(g, parent, rodName, shaded);
    },

    orb(parent, radius, orbName, shaded = false) {
      const g = new THREE.IcosahedronGeometry(radius, 1);
      return mesh(g, parent, orbName, shaded);
    },

    box(parent, wx, wy, wz, boxName, shaded = false) {
      const g = new THREE.BoxGeometry(wx, wy, wz);
      return mesh(g, parent, boxName, shaded);
    },

    dispose() {
      for (const g of geometries) g.dispose();
      geometries.length = 0;
      material.dispose();
      shadeMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
