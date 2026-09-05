import * as THREE from 'three';
import { blendTokens, shade } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { rockGeometry } from '@kit/vegetation';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { radialSprite } from '@kit/textures';
import { Wisp } from '@kit/wisp';
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
const BEACON_HEIGHT = 22;
const BEACON_BASE = 0.74;
const BEACON_TOP = 0.3;
const BEACON_POOL = 6.4;
const BEACON_LIFT = 1.2;
const BEACON_INTENSITY = 0.7;
const BEACON_OPACITY = 0.3;
const BEACON_POOL_GLOW = 0.7;
const BEACON_POOL_OPACITY = 0.3;

export const ITEM_MOTION: Record<number, { hover: number; bob: number; spin: number }> = {
  [JBW_BANANA]: { hover: AGENT_SPAN * 0.52, bob: AGENT_SPAN * 0.11, spin: 1.4 },
  [JBW_ONION]: { hover: AGENT_SPAN * 0.52, bob: AGENT_SPAN * 0.11, spin: 1.25 },
  [JBW_JELLY]: { hover: AGENT_SPAN * 0.52, bob: AGENT_SPAN * 0.11, spin: 1.35 },
  [JBW_WALL]: { hover: 0, bob: 0, spin: 0 },
  [JBW_TREE]: { hover: 0, bob: 0, spin: 0 },
  [JBW_TRUFFLE]: { hover: AGENT_SPAN * 0.62, bob: AGENT_SPAN * 0.13, spin: 1.15 },
};

export const ITEM_GLOW = 0.16;
export const HALO_GLOW = 0.035;

export const HALO_WHITEN = 0.16;

const ITEM_SPAN: Record<number, number> = {
  [JBW_BANANA]: AGENT_SPAN,
  [JBW_ONION]: AGENT_SPAN,
  [JBW_JELLY]: AGENT_SPAN,
  [JBW_WALL]: AGENT_SPAN * 0.62,
  [JBW_TREE]: AGENT_SPAN * 2.9,
  [JBW_TRUFFLE]: AGENT_SPAN,
};

export const DRAW_RADIUS = 4096;

const CANDIDATES = 1 << 16;
const WALL_CELLS = CANDIDATES * 2;

export const ITEM_CAPACITY: Record<number, number> = {
  [JBW_BANANA]: CANDIDATES,
  [JBW_ONION]: CANDIDATES,
  [JBW_JELLY]: CANDIDATES,
  [JBW_WALL]: WALL_CELLS,
  [JBW_TREE]: CANDIDATES,
  [JBW_TRUFFLE]: CANDIDATES,
};

export const DECAL_CAPACITY = CANDIDATES;
export const DEBUG_CELL_CAPACITY = CANDIDATES + WALL_CELLS;

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


export interface BeaconHandle {
  mesh: THREE.InstancedMesh;
  pool: THREE.InstancedMesh;
  setTime(t: number): void;
  dispose(): void;
}

const BEAM_VERT = `
#include <common>
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	vAxial = uv.y;
	vec4 wp = modelMatrix * instanceMatrix * vec4( position, 1.0 );
	vBeamNormal = normalize( mat3( modelMatrix ) * mat3( instanceMatrix ) * normal );
	vBeamView = normalize( cameraPosition - wp.xyz );
	gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const BEAM_FRAG = `
#include <common>
uniform vec3 uColor;
uniform float uIntensity;
uniform float uOpacity;
uniform float uTime;
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	float facing = abs( dot( normalize( vBeamNormal ), normalize( vBeamView ) ) );
	float body = pow( facing, 1.35 );
	float fall = mix( 1.0, 0.45, vAxial );
	float capFade = ( 1.0 - smoothstep( 0.3, 1.0, vAxial ) ) * smoothstep( 0.0, 0.14, vAxial );
	float bands = 0.87 + 0.13 * sin( vAxial * 11.0 - uTime * 0.55 );
	float a = body * fall * capFade * bands * uOpacity;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( uColor * uIntensity, a );
	#include <colorspace_fragment>
}
`;

export function buildBeacon(color: THREE.ColorRepresentation, capacity: number): BeaconHandle {
  const geometry = new THREE.CylinderGeometry(
    BEACON_TOP,
    BEACON_BASE,
    BEACON_HEIGHT,
    9,
    1,
    true,
  );
  geometry.translate(0, BEACON_HEIGHT / 2 + BEACON_LIFT, 0);
  const uniforms = {
    uColor: { value: new THREE.Color(color) },
    uIntensity: { value: BEACON_INTENSITY },
    uOpacity: { value: BEACON_OPACITY },
    uTime: { value: 0 },
  };
  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: BEAM_VERT,
    fragmentShader: BEAM_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
    name: 'orchardBeacon',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, capacity);
  mesh.name = 'orchardBeacons';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  mesh.count = 0;

  const poolGeometry = new THREE.PlaneGeometry(BEACON_POOL, BEACON_POOL);
  poolGeometry.rotateX(-Math.PI / 2);
  const poolMaterial = emissiveMaterial(color, BEACON_POOL_GLOW, {
    map: radialSprite(64, 2.1),
    transparent: true,
    opacity: BEACON_POOL_OPACITY,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'orchardBeaconPool',
  });
  const pool = new THREE.InstancedMesh(poolGeometry, poolMaterial, capacity);
  pool.name = 'orchardBeaconPools';
  pool.castShadow = false;
  pool.receiveShadow = false;
  pool.frustumCulled = false;
  pool.renderOrder = 2;
  pool.count = 0;

  return {
    mesh,
    pool,
    setTime(t) {
      uniforms.uTime.value = t;
      const breath = 1 + 0.09 * Math.sin(t * 0.73) + 0.05 * Math.sin(t * 1.9 + 1.1);
      uniforms.uIntensity.value = BEACON_INTENSITY * breath;
      setEmissiveIntensity(poolMaterial, BEACON_POOL_GLOW * breath);
    },
    dispose() {
      mesh.dispose();
      pool.dispose();
      geometry.dispose();
      poolGeometry.dispose();
      material.dispose();
      poolMaterial.dispose();
      if (poolMaterial.map) poolMaterial.map.dispose();
    },
  };
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

export function buildSeedpod(color: number): Wisp {
  const traveler = new Wisp({ color: blendTokens(color, 0xffedbb, 0.6), radius: AGENT_SPAN / 4, height: AGENT_SPAN * 0.42, skirt: false, ribbon: true, silhouette: false, hop: { duration: 0.11, height: 0.16 }, name: 'orchardSeedpod' }, { w: 1, h: 1 });
  const parts: ColoredPart[] = [];
  for (const side of [-1, 1]) {
    const shell = new THREE.SphereGeometry(0.62, 10, 8, 0.16, Math.PI - 0.32);
    shell.scale(0.86, 0.78, 0.66);
    shell.rotateY(side > 0 ? 0 : Math.PI);
    shell.rotateX(side * 0.16);
    shell.translate(0, -0.24, side * 0.25);
    parts.push({ geometry: shell, color: side > 0 ? 0xb59d65 : 0xddc991 });
  }
  const leaf = new THREE.SphereGeometry(0.2, 7, 5);
  leaf.scale(0.35, 1.1, 0.16);
  leaf.rotateZ(-0.5);
  leaf.translate(0.12, 0.68, 0);
  parts.push({ geometry: leaf, color: 0x8fa669 });
  const geometry = mergeColoredParts(parts, 'seedpodShell');
  const material = oakMaterial({ vertexColors: true, roughness: 0.75, side: THREE.DoubleSide, rimStrength: 0.25 });
  const shell = new THREE.Mesh(geometry, material);
  shell.castShadow = true;
  traveler.core.add(shell);
  const dispose = traveler.dispose.bind(traveler);
  traveler.dispose = () => {
    geometry.dispose();
    material.dispose();
    dispose();
  };
  return traveler;
}
