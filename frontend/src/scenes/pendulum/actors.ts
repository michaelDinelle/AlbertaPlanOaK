
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import {
  CAP_R,
  EYE_DROP,
  EYE_MAJOR,
  EYE_TUBE,
  FINIAL_H,
  LANTERN_H,
  LANTERN_R,
  RIB_HALF,
  RIB_INSET,
  RING_R,
  RING_TUBE,
  ROD_LEN,
  TASSEL_CORD_LEN,
  TASSEL_DROP,
  TASSEL_WEIGHT_H,
  TASSEL_WEIGHT_R,
} from './dims';
import { blendTokens, clampRange, mergeColoredParts, type ColoredPart } from './world';


const ROD_TOP = -0.05;
const ROD_BOTTOM = -(ROD_LEN - 0.15);
const PAPER_BASE_HDR = 2.7;

export interface LanternRigHandle {
  group: THREE.Group;
  swingRoot: THREE.Object3D;
  bobRoot: THREE.Object3D;
  readonly flameWorld: THREE.Vector3;
  setState(theta: number, thetaDot: number, torque: number): void;
  setGlow(serenity: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface LanternRigOptions {
  palette: Palette;
  accent: THREE.ColorRepresentation;
  pivot: readonly [number, number, number];
  maxTorque: number;
  maxSpeed: number;
}

export function buildLanternRig(opts: LanternRigOptions): LanternRigHandle {
  const { palette, accent, pivot, maxTorque, maxSpeed } = opts;
  const group = new THREE.Group();
  group.name = 'lanternRig';

  const iron = blendTokens(palette.hemiGround, palette.skyTop, 0.35);
  const ironLight = blendTokens(iron, palette.hemiSky, 0.35);
  const cord = blendTokens(palette.groundA, palette.hemiGround, 0.5);

  const swingRoot = new THREE.Object3D();
  swingRoot.name = 'swingRoot';
  swingRoot.position.set(pivot[0], pivot[1], pivot[2]);
  group.add(swingRoot);

  const rodParts: ColoredPart[] = [];
  {
    const eye = new THREE.TorusGeometry(EYE_MAJOR, EYE_TUBE, 5, 10);
    eye.translate(0, -EYE_DROP, 0);
    rodParts.push({ geometry: eye, color: ironLight });
    const len = ROD_TOP - ROD_BOTTOM;
    const shaft = new THREE.CylinderGeometry(0.024, 0.031, len, 6);
    shaft.translate(0, ROD_BOTTOM + len * 0.5, 0);
    rodParts.push({ geometry: shaft, color: iron });
    const yoke = new THREE.TorusGeometry(0.036, 0.011, 5, 10);
    yoke.translate(0, ROD_BOTTOM - 0.012, 0);
    rodParts.push({ geometry: yoke, color: ironLight });
    const hook = new THREE.CylinderGeometry(0.009, 0.009, 0.055, 5);
    hook.translate(0, ROD_BOTTOM - 0.042, 0);
    rodParts.push({ geometry: hook, color: iron });
  }
  const rodGeometry = mergeColoredParts(rodParts, 'lanternRod');
  const ironMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.68,
    metalness: 0.12,
    rimColor: blendTokens(palette.sun, palette.hemiSky, 0.35),
    rimStrength: 0.6,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.12,
    name: 'lanternIronMaterial',
  });
  const rodMesh = new THREE.Mesh(rodGeometry, ironMaterial);
  rodMesh.name = 'lanternRod';
  rodMesh.castShadow = true;
  swingRoot.add(rodMesh);

  const bobRoot = new THREE.Object3D();
  bobRoot.name = 'bobRoot';
  group.add(bobRoot);

  const bobTilt = new THREE.Object3D();
  bobTilt.name = 'bobTilt';
  bobRoot.add(bobTilt);

  const frameParts: ColoredPart[] = [];
  {
    const topCap = new THREE.CylinderGeometry(CAP_R * 0.62, CAP_R, 0.05, 6);
    topCap.translate(0, LANTERN_H * 0.5 + 0.024, 0);
    frameParts.push({ geometry: topCap, color: ironLight });
    const bottomCap = new THREE.CylinderGeometry(CAP_R * 0.94, CAP_R * 0.66, 0.042, 6);
    bottomCap.translate(0, -LANTERN_H * 0.5 - 0.02, 0);
    frameParts.push({ geometry: bottomCap, color: ironLight });
    const finial = new THREE.ConeGeometry(0.026, FINIAL_H, 6);
    finial.translate(0, LANTERN_H * 0.5 + 0.076, 0);
    frameParts.push({ geometry: finial, color: iron });
    const ring = new THREE.TorusGeometry(RING_R, RING_TUBE, 4, 9);
    ring.translate(0, LANTERN_H * 0.5 + 0.113, 0);
    frameParts.push({ geometry: ring, color: iron });
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const rib = new THREE.BoxGeometry(RIB_HALF * 2, LANTERN_H + 0.02, RIB_HALF * 2);
      rib.translate(Math.cos(a) * LANTERN_R * RIB_INSET, 0, Math.sin(a) * LANTERN_R * RIB_INSET);
      frameParts.push({ geometry: rib, color: iron });
    }
    const tasselCord = new THREE.CylinderGeometry(0.005, 0.005, TASSEL_CORD_LEN, 4);
    tasselCord.translate(0, -LANTERN_H * 0.5 - (TASSEL_DROP - TASSEL_WEIGHT_H * 0.92), 0);
    frameParts.push({ geometry: tasselCord, color: cord });
    const weight = new THREE.ConeGeometry(TASSEL_WEIGHT_R, TASSEL_WEIGHT_H, 6);
    weight.rotateZ(Math.PI);
    weight.translate(0, -LANTERN_H * 0.5 - TASSEL_DROP, 0);
    frameParts.push({ geometry: weight, color: cord });
  }
  const frameGeometry = mergeColoredParts(frameParts, 'lanternFrame');
  const frameMesh = new THREE.Mesh(frameGeometry, ironMaterial);
  frameMesh.name = 'lanternFrame';
  frameMesh.castShadow = true;
  bobTilt.add(frameMesh);

  const paperGeometry = makePaperGeometry();
  const paperMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(accent, palette.sun, 0.22),
    PAPER_BASE_HDR,
    { transparent: true, opacity: 0.95, side: THREE.DoubleSide, name: 'lanternPaper' },
  );
  const paperMesh = new THREE.Mesh(paperGeometry, paperMaterial);
  paperMesh.name = 'lanternPaper';
  bobTilt.add(paperMesh);

  const flameGeometry = new THREE.IcosahedronGeometry(0.05, 1);
  flameGeometry.scale(1, 1.45, 1);
  const flameMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.sun, accent, 0.62),
    PAPER_BASE_HDR * 1.5,
    { name: 'lanternFlameCore' },
  );
  const flameMesh = new THREE.Mesh(flameGeometry, flameMaterial);
  flameMesh.name = 'lanternFlame';
  flameMesh.position.y = -0.01;
  bobTilt.add(flameMesh);

  const curlRoot = new THREE.Object3D();
  curlRoot.name = 'torqueCurlRoot';
  curlRoot.position.set(0, -ROD_LEN, 0);
  swingRoot.add(curlRoot);

  const curlGeometry = makeCurlGeometry();
  const curlMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.hemiSky, accent, 0.55),
    2.3,
    {
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      name: 'torqueCurlMaterial',
    },
  );
  curlMaterial.vertexColors = true;
  curlMaterial.opacity = 0;
  const curlTier = true;
  const curlPos = new THREE.Mesh(curlGeometry, curlMaterial);
  curlPos.name = 'torqueCurlPositive';
  curlPos.renderOrder = 4;
  curlPos.visible = false;
  const curlNeg = new THREE.Mesh(curlGeometry, curlMaterial);
  curlNeg.name = 'torqueCurlNegative';
  curlNeg.renderOrder = 4;
  curlNeg.scale.x = -1;
  curlNeg.visible = false;
  curlRoot.add(curlPos, curlNeg);

  const flameWorld = new THREE.Vector3();
  let glow = 1;
  let breath = 0;
  let spin = 0;

  const applyFlame = (): void => {
    const flicker = 1 + 0.07 * Math.sin(breath * 3.1) + 0.04 * Math.sin(breath * 7.7 + 1.1);
    const level = glow * flicker;
    setEmissiveIntensity(paperMaterial, PAPER_BASE_HDR * (0.035 + 0.965 * level));
    setEmissiveIntensity(flameMaterial, PAPER_BASE_HDR * 1.5 * (0.02 + 0.98 * level));
    paperMaterial.opacity = 0.55 + 0.42 * clampRange(level, 0, 1);
  };

  return {
    group,
    swingRoot,
    bobRoot,
    flameWorld,

    setState(theta, thetaDot, torque) {
      const sin = Math.sin(theta);
      const cos = Math.cos(theta);
      bobRoot.position.set(pivot[0] + ROD_LEN * sin, pivot[1] - ROD_LEN * cos, pivot[2]);
      swingRoot.rotation.z = theta;

      const lag = clampRange(thetaDot / maxSpeed, -1, 1);
      bobTilt.rotation.z = theta + lag * 0.22;

      const effort = clampRange(Math.abs(torque) / maxTorque, 0, 1);
      const live = curlTier && effort > 0.02;
      curlPos.visible = live && torque > 0;
      curlNeg.visible = live && torque < 0;
      curlMaterial.opacity = live ? 0.08 + 0.76 * effort : 0;
      const scale = 0.72 + 0.5 * effort;
      curlPos.scale.set(scale, scale, scale);
      curlNeg.scale.set(-scale, scale, scale);

      flameWorld.set(bobRoot.position.x, bobRoot.position.y - 0.01, bobRoot.position.z);
    },

    setGlow(serenity) {
      glow = clampRange(serenity, 0, 1);
      applyFlame();
    },

    setTime(t) {
      breath = t;
      spin = t;
      const drift = 1 + 0.05 * Math.sin(spin * 2.3);
      curlRoot.rotation.z = 0.09 * Math.sin(spin * 1.7);
      curlRoot.scale.setScalar(drift);
      applyFlame();
    },

    dispose() {
      rodGeometry.dispose();
      frameGeometry.dispose();
      ironMaterial.dispose();
      paperGeometry.dispose();
      paperMaterial.dispose();
      flameGeometry.dispose();
      flameMaterial.dispose();
      curlGeometry.dispose();
      curlMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makePaperGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const rings: readonly (readonly [number, number, number])[] = [
    [-LANTERN_H * 0.5, -LANTERN_H * 0.18, CAP_R * 0.92],
    [-LANTERN_H * 0.18, LANTERN_H * 0.18, LANTERN_R],
    [LANTERN_H * 0.18, LANTERN_H * 0.5, LANTERN_R],
  ];
  const radii = [CAP_R * 0.92, LANTERN_R, LANTERN_R, CAP_R * 0.9];
  for (let i = 0; i < rings.length; i++) {
    const ring = rings[i] as readonly [number, number, number];
    const y0 = ring[0];
    const y1 = ring[1];
    const r0 = radii[i] as number;
    const r1 = radii[i + 1] as number;
    const g = new THREE.CylinderGeometry(r1, r0, y1 - y0, 6, 1, true);
    g.translate(0, (y0 + y1) * 0.5, 0);
    parts.push({ geometry: g, color: 0xffffff });
  }
  return mergeColoredParts(parts, 'lanternPaper');
}

function makeCurlGeometry(): THREE.BufferGeometry {
  interface Stroke {
    radius: number;
    radiusEnd: number;
    from: number;
    to: number;
    width: number;
    lift: number;
  }
  const strokes: readonly Stroke[] = [
    { radius: 0.21, radiusEnd: 0.42, from: 1.78, to: -0.38, width: 0.062, lift: 0.16 },
    { radius: 0.3, radiusEnd: 0.58, from: 1.42, to: -0.72, width: 0.078, lift: 0.19 },
    { radius: 0.25, radiusEnd: 0.47, from: 2.16, to: 0.18, width: 0.048, lift: 0.13 },
  ];
  const SEG = 26;
  const verts = strokes.length * (SEG + 1) * 2;
  const positions = new Float32Array(verts * 3);
  const colors = new Float32Array(verts * 4);
  const indices: number[] = [];
  let v = 0;
  for (const stroke of strokes) {
    const base = v;
    for (let i = 0; i <= SEG; i++) {
      const k = i / SEG;
      const a = stroke.from + (stroke.to - stroke.from) * k;
      const radius = stroke.radius + (stroke.radiusEnd - stroke.radius) * k * k;
      const cx = Math.cos(a) * radius;
      const cy = Math.sin(a) * radius;
      const taper = Math.sin(Math.PI * Math.pow(k, 0.72)) * (1 - 0.55 * k);
      const half = stroke.width * taper;
      const nx = Math.cos(a);
      const ny = Math.sin(a);
      positions[v * 3] = cx - nx * half;
      positions[v * 3 + 1] = cy - ny * half;
      positions[v * 3 + 2] = stroke.lift;
      positions[(v + 1) * 3] = cx + nx * half;
      positions[(v + 1) * 3 + 1] = cy + ny * half;
      positions[(v + 1) * 3 + 2] = stroke.lift;
      const alpha = Math.pow(taper, 0.85) * (0.35 + 0.65 * k);
      const lum = 0.6 + 0.4 * k;
      for (let e = 0; e < 2; e++) {
        const o = (v + e) * 4;
        colors[o] = lum;
        colors[o + 1] = lum;
        colors[o + 2] = lum;
        colors[o + 3] = alpha;
      }
      v += 2;
    }
    for (let i = 0; i < SEG; i++) {
      const a = base + i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 4));
  g.setIndex(indices);
  g.computeBoundingSphere();
  g.name = 'torqueCurl';
  return g;
}
