import * as THREE from 'three';
import { blendTokens, shade } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { rockGeometry } from '@kit/vegetation';
import {
  JBW_BANANA,
  JBW_JELLY,
  JBW_ONION,
  JBW_TREE,
  JBW_TRUFFLE,
  JBW_WALL,
} from '@sim/envs/jellybeanworld';
import type { OrchardSkin } from './world';

export const ITEM_KINDS: readonly number[] = [
  JBW_BANANA,
  JBW_ONION,
  JBW_JELLY,
  JBW_WALL,
  JBW_TREE,
  JBW_TRUFFLE,
];

export const AGENT_SPAN = 2;

const ITEM_SPAN: Record<number, number> = {
  [JBW_BANANA]: AGENT_SPAN,
  [JBW_ONION]: AGENT_SPAN,
  [JBW_JELLY]: AGENT_SPAN,
  [JBW_WALL]: AGENT_SPAN * 0.62,
  [JBW_TREE]: AGENT_SPAN * 2.9,
  [JBW_TRUFFLE]: AGENT_SPAN,
};

export const ITEM_CAPACITY: Record<number, number> = {
  [JBW_BANANA]: 4096,
  [JBW_ONION]: 4096,
  [JBW_JELLY]: 4096,
  [JBW_WALL]: 16384,
  [JBW_TREE]: 4096,
  [JBW_TRUFFLE]: 1024,
};

function bananaGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const flesh = new THREE.TorusGeometry(0.2, 0.062, 4, 9, Math.PI * 1.15);
  flesh.rotateX(Math.PI / 2);
  flesh.rotateZ(0.42);
  flesh.translate(0, 0.02, 0);
  parts.push({ geometry: flesh, color: skin.banana });
  const tip = new THREE.ConeGeometry(0.05, 0.11, 5);
  tip.rotateZ(1.05);
  tip.translate(0.19, 0.07, 0.02);
  parts.push({ geometry: tip, color: shade(skin.banana, 0.34) });
  return mergeColoredParts(parts, 'orchardBanana');
}

function onionGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const bulb = new THREE.SphereGeometry(0.17, 7, 5);
  bulb.scale(1, 1.16, 1);
  parts.push({ geometry: bulb, color: skin.onion });
  const neck = new THREE.ConeGeometry(0.055, 0.19, 5);
  neck.translate(0, 0.22, 0);
  parts.push({ geometry: neck, color: blendTokens(skin.onion, skin.papery, 0.62) });
  const root = new THREE.ConeGeometry(0.04, 0.09, 4);
  root.rotateX(Math.PI);
  root.translate(0, -0.2, 0);
  parts.push({ geometry: root, color: skin.papery });
  return mergeColoredParts(parts, 'orchardOnion');
}

function jellyGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.SphereGeometry(0.155, 8, 6);
  body.scale(1.32, 0.86, 0.86);
  parts.push({ geometry: body, color: skin.jelly });
  const sheen = new THREE.SphereGeometry(0.062, 5, 4);
  sheen.translate(0.055, 0.085, 0.03);
  parts.push({ geometry: sheen, color: blendTokens(skin.jelly, skin.papery, 0.4) });
  return mergeColoredParts(parts, 'orchardJelly');
}

function truffleGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.DodecahedronGeometry(0.145, 0);
  const pos = body.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const k = 0.82 + hash2f(i, 17, 0x7a3) * 0.42;
    pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k * 0.82, pos.getZ(i) * k);
  }
  body.computeVertexNormals();
  parts.push({ geometry: body, color: skin.truffle });
  const crown = new THREE.TetrahedronGeometry(0.07, 0);
  crown.translate(0.03, 0.1, -0.02);
  parts.push({ geometry: crown, color: blendTokens(skin.truffle, skin.bloom, 0.5) });
  return mergeColoredParts(parts, 'orchardTruffle');
}

function wallGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const courses: readonly (readonly [number, number, number, number])[] = [
    [0, 0.15, 0.46, 1],
    [0.11, 0.4, 0.4, 0.72],
    [-0.09, 0.62, 0.33, 0.5],
  ];
  for (let i = 0; i < courses.length; i++) {
    const c = courses[i];
    if (!c) continue;
    const stone = rockGeometry(1.1 + i * 0.35, 0);
    stone.scale(c[2] * 2.1, c[3] * 0.5 + 0.2, c[2] * 1.5);
    stone.rotateY(0.4 + i * 1.1);
    stone.translate(c[0], c[1], i % 2 === 0 ? 0.03 : -0.04);
    parts.push({ geometry: stone, color: blendTokens(skin.stone, skin.moss, i * 0.22) });
  }
  const cap = rockGeometry(2.3, 0);
  cap.scale(0.7, 0.2, 0.5);
  cap.translate(0.02, 0.79, 0);
  parts.push({ geometry: cap, color: blendTokens(skin.stone, skin.papery, 0.3) });
  return mergeColoredParts(parts, 'orchardWall');
}

function treeGeometry(skin: OrchardSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const trunk = new THREE.CylinderGeometry(0.085, 0.15, 1.05, 6, 1);
  trunk.translate(0, 0.52, 0);
  trunk.rotateZ(0.04);
  parts.push({ geometry: trunk, color: skin.bark });
  const crowns: readonly (readonly [number, number, number, number])[] = [
    [0, 1.32, 0, 0.56],
    [0.26, 1.12, 0.16, 0.36],
    [-0.24, 1.18, -0.14, 0.33],
    [0.06, 1.66, -0.1, 0.31],
  ];
  for (let i = 0; i < crowns.length; i++) {
    const c = crowns[i];
    if (!c) continue;
    const lobe = new THREE.IcosahedronGeometry(c[3], 0);
    lobe.scale(1.12, 0.86, 1.12);
    lobe.translate(c[0], c[1], c[2]);
    parts.push({ geometry: lobe, color: blendTokens(skin.canopyLow, skin.canopyHigh, i / 3) });
  }
  return mergeColoredParts(parts, 'orchardTree');
}

function rawItemGeometry(item: number, skin: OrchardSkin): THREE.BufferGeometry {
  if (item === JBW_BANANA) return bananaGeometry(skin);
  if (item === JBW_ONION) return onionGeometry(skin);
  if (item === JBW_JELLY) return jellyGeometry(skin);
  if (item === JBW_WALL) return wallGeometry(skin);
  if (item === JBW_TREE) return treeGeometry(skin);
  if (item === JBW_TRUFFLE) return truffleGeometry(skin);
  throw new Error(`no orchard geometry for item ${item}`);
}

export function orchardItemGeometry(item: number, skin: OrchardSkin): THREE.BufferGeometry {
  const span = ITEM_SPAN[item];
  if (span === undefined) throw new Error(`no orchard span for item ${item}`);
  const geometry = rawItemGeometry(item, skin);
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box) throw new Error(`orchard geometry ${item} has no bounds`);
  const largest = Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
  if (!(largest > 0)) throw new Error(`orchard geometry ${item} is degenerate`);
  const k = span / largest;
  geometry.scale(k, k, k);
  geometry.computeBoundingBox();
  const sat = geometry.boundingBox;
  if (sat) geometry.translate(0, -sat.min.y, 0);
  return geometry;
}
