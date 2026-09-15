import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { SVGLoader } from 'three/addons/loaders/SVGLoader.js';
import type { WorldBuilder } from './world';
import { TOKENS } from './tokens';
import butterflySVG from '../../../butterfly_thick.svg' with { type: 'text' };

const emblem = new SVGLoader().parse(butterflySVG).paths[0]!;
const butterfly = emblem.toShapes();
const logoBounds = new THREE.Box2().setFromPoints(butterfly.flatMap(shape => shape.getPoints(12)));
const logoCenter = logoBounds.getCenter(new THREE.Vector2()), logoWidth = logoBounds.max.x - logoBounds.min.x;

export function robotMaterials() {
  return {
    armor: new THREE.MeshStandardMaterial({ color: 0xe4e2da, vertexColors: true, roughness: 0.5, metalness: 0.16 }),
    accent: new THREE.MeshStandardMaterial({ color: 0xdab96c, roughness: 0.5, metalness: 0.45 }),
    frame: new THREE.MeshStandardMaterial({ color: 0x687887, roughness: 0.58, metalness: 0.45 }),
    alloy: new THREE.MeshStandardMaterial({ color: 0xbfcbd0, roughness: 0.55, metalness: 0.48 }),
    rubber: new THREE.MeshStandardMaterial({ color: 0x303a43, roughness: 0.92 }),
    hardware: new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.56, metalness: 0.5 }),
    status: new THREE.MeshStandardMaterial({ color: TOKENS.uiCyan, roughness: 0.3, emissive: TOKENS.uiCyan, emissiveIntensity: 0.12 }),
  };
}

function panelOutline(width: number, length: number, bevel: number) {
  const shape = new THREE.Shape();
  const w = width * 0.5;
  const h = length * 0.5;
  const outline = [[-w + bevel, -h], [w - bevel, -h], [w, -h + bevel], [w, h - bevel], [w - bevel, h], [-w + bevel, h], [-w, h - bevel], [-w, -h + bevel]];
  outline.forEach(([x = 0, y = 0], i) => i === 0 ? shape.moveTo(x, y) : shape.lineTo(x, y));
  shape.closePath();
  return shape;
}

export function bevelBox(width: number, length: number, depth: number, bevel: number): THREE.BufferGeometry {
  const shape = panelOutline(width, length, bevel);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel * 0.35, bevelSize: bevel * 0.35, bevelSegments: 1, steps: 1, curveSegments: 1 });
  geometry.translate(0, 0, -depth * 0.5);
  return tint(geometry);
}

function tint(geometry: THREE.BufferGeometry, color: THREE.ColorRepresentation = 0xffffff, edgeColor = color) {
  const face = new THREE.Color(color), edge = new THREE.Color(edgeColor), normals = geometry.getAttribute('normal');
  const colors = new Float32Array(geometry.getAttribute('position').count * 3);
  for (let i = 0; i < colors.length; i += 3) (normals.getZ(i / 3) < 0.999 ? edge : face).toArray(colors, i);
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}

export function engravedPanel(width: number, height: number, depth: number, bevel: number, markWidth: number) {
  const outline = panelOutline(width, height, bevel), islands: THREE.Shape[] = [];
  const scaled = (path: THREE.Path) => path.getPoints(6).map(point => new THREE.Vector2((point.x - logoCenter.x) * markWidth / logoWidth, (logoCenter.y - point.y) * markWidth / logoWidth));
  const cuts = butterfly.map(shape => {
    const cut = new THREE.Shape(scaled(shape));
    outline.holes.push(new THREE.Path(cut.getPoints()));
    cut.holes = shape.holes.map(hole => {
      const points = scaled(hole);
      islands.push(new THREE.Shape(points));
      return new THREE.Path(points);
    });
    return cut;
  });
  const body = bevelBox(width, height, depth, bevel), normals = body.getAttribute('normal'), colors = body.getAttribute('color'), keep: number[] = [];
  for (let i = 0; i < normals.count; i += 3) if (normals.getZ(i) < 0.999) {
    keep.push(i, i + 1, i + 2);
    if (normals.getZ(i) > 0) for (let j = i; j < i + 3; j++) colors.setXYZ(j, emblem.color.r, emblem.color.g, emblem.color.b);
  }
  body.setIndex(keep);
  const recess = Math.min(depth * 0.3, markWidth * 0.055), floor = depth * 0.5 + bevel * 0.35 - recess;
  const lip = Math.min(recess * 0.2, markWidth * 0.004);
  const face = new THREE.ExtrudeGeometry([outline, ...islands], { depth: recess - lip, bevelEnabled: true, bevelThickness: lip, bevelSize: lip, bevelSegments: 1, steps: 1, curveSegments: 1 }).translate(0, 0, floor);
  const cavity = new THREE.ShapeGeometry(cuts, 1).translate(0, 0, floor);
  const pieces = [body.toNonIndexed(), tint(face, 0xffffff, emblem.color), tint(cavity.toNonIndexed(), emblem.color.clone().multiplyScalar(0.45))];
  const geometry = mergeGeometries(pieces);
  for (const part of [body, cavity, ...pieces]) part.dispose();
  geometry.name = 'butterflyEngravedArmor';
  return geometry;
}

export function modelParts(world: WorldBuilder) {
  const parts = new Map<THREE.Group, Map<THREE.Material, THREE.BufferGeometry[]>>();
  function add(parent: THREE.Group, geometry: THREE.BufferGeometry, material: THREE.Material) {
    if (!geometry.hasAttribute('color')) tint(geometry);
    if (!parts.has(parent)) parts.set(parent, new Map());
    const batch = parts.get(parent)!;
    if (!batch.has(material)) batch.set(material, []);
    batch.get(material)!.push(geometry.index ? geometry.toNonIndexed() : geometry);
    if (geometry.index) geometry.dispose();
  }
  return {
    add,
    box(parent: THREE.Group, material: THREE.Material, x: number, y: number, z: number, w: number, h: number, d: number, radius = 0.018) {
      add(parent, bevelBox(w, h, d, Math.min(radius, w / 3, h / 3, d / 3)).translate(x, y, z), material);
    },
    disc(parent: THREE.Group, material: THREE.Material, x: number, y: number, z: number, radius: number, depth: number, sides = 32) {
      add(parent, new THREE.CylinderGeometry(radius, radius, depth, sides).rotateX(Math.PI / 2).translate(x, y, z), material);
    },
    rod(parent: THREE.Group, material: THREE.Material, a: THREE.Vector3, b: THREE.Vector3, radius: number) {
      const geometry = new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 12);
      geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, b.clone().sub(a).normalize()));
      add(parent, geometry.translate(...a.clone().add(b).multiplyScalar(0.5).toArray()), material);
    },
    build() {
      for (const [parent, batch] of parts) for (const [material, geometries] of batch) {
        const geometry = world.own(mergeGeometries(geometries));
        geometries.forEach(part => part.dispose());
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = mesh.receiveShadow = true;
        parent.add(mesh);
      }
      parts.clear();
    },
  };
}
