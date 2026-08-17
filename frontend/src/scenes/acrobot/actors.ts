
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clampRange, TOKEN_BLACK } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { streamFor } from '@kit/rng';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import { AB_L1, AB_L2 } from '@sim/envs/acrobot';
import { EXACT_POINT_FLOATS, acrobotPointsExact } from './world';


const VINE_R1 = 0.055;
const VINE_R2 = 0.042;
const ELBOW_R = 0.082;
const LANTERN_DROP = 0.135;
const LANTERN_R = 0.125;
const LANTERN_H = 0.225;
const LANTERN_BASE_HDR = 1.06;
const CURL_R = 0.22;
const CURL_SWEEP = 0.92;

const TRAIL_SAMPLES = 64;
const TRAIL_SAMPLE_DT = 0.022;
export const TRAIL_SPAN = 0.95;
const TRAIL_MAX_HALF_W = 0.046;

export interface ArmHandle {
  group: THREE.Group;
  pivotRoot: THREE.Object3D;
  elbowRoot: THREE.Object3D;
  tipRoot: THREE.Object3D;
  torqueHead: THREE.Object3D;
  readonly tipWorld: THREE.Vector3;
  readonly lanternWorld: THREE.Vector3;
  setState(th1: number, th2: number, torque: number, tipSpeed: number): void;
  setTime(t: number): void;
  setLanternGain(gain: number): void;
  pushTrail(t: number): void;
  refreshTrail(t: number): void;
  clearTrail(): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface ArmOptions {
  palette: Palette;
  accent: THREE.ColorRepresentation;
  gold: THREE.ColorRepresentation;
  seed: number;
  pivot: readonly [number, number, number];
}

export function buildArm(opts: ArmOptions): ArmHandle {
  const { palette, accent, gold, seed, pivot } = opts;
  const group = new THREE.Group();
  group.name = 'trapezeRig';

  const vineDark = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.45);
  const vineLit = blendTokens(palette.hemiGround, palette.groundB, 0.3);
  const leaf = blendTokens(palette.groundB, palette.sun, 0.3);
  const brass = blendTokens(palette.sun, palette.hemiGround, 0.4);

  const pivotRoot = new THREE.Object3D();
  pivotRoot.name = 'pivotRoot';
  pivotRoot.position.set(pivot[0], pivot[1], pivot[2]);
  group.add(pivotRoot);

  const link1Node = new THREE.Object3D();
  link1Node.name = 'link1Node';
  pivotRoot.add(link1Node);

  const elbowRoot = new THREE.Object3D();
  elbowRoot.name = 'elbowRoot';
  elbowRoot.position.set(0, -AB_L1, 0);
  link1Node.add(elbowRoot);

  const link2Node = new THREE.Object3D();
  link2Node.name = 'link2Node';
  elbowRoot.add(link2Node);

  const tipRoot = new THREE.Object3D();
  tipRoot.name = 'tipRoot';
  tipRoot.position.set(0, -AB_L2, 0);
  link2Node.add(tipRoot);

  const vineMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.85,
    rimColor: palette.sun,
    rimStrength: 1.05,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    name: 'trapezeVine',
  });

  const link1Geometry = makeVineGeometry(AB_L1, VINE_R1, seed ^ 0x11, vineDark, vineLit, leaf);
  const link1 = new THREE.Mesh(link1Geometry, vineMaterial);
  link1.name = 'link1';
  link1.castShadow = true;
  link1Node.add(link1);

  const link2Geometry = makeVineGeometry(AB_L2, VINE_R2, seed ^ 0x22, vineDark, vineLit, leaf);
  const link2 = new THREE.Mesh(link2Geometry, vineMaterial);
  link2.name = 'link2';
  link2.castShadow = true;
  link2Node.add(link2);

  const knotParts: ColoredPart[] = [];
  {
    const ball = new THREE.IcosahedronGeometry(ELBOW_R, 1);
    ball.scale(1, 0.92, 1);
    knotParts.push({ geometry: ball, color: vineDark });
    for (let i = 0; i < 3; i++) {
      const wrap = new THREE.TorusGeometry(ELBOW_R * 0.96, 0.014, 4, 10);
      wrap.rotateY((i / 3) * Math.PI);
      wrap.rotateX(Math.PI / 2 + (i - 1) * 0.34);
      knotParts.push({ geometry: wrap, color: i === 1 ? brass : vineLit });
    }
  }
  const knotGeometry = mergeColoredParts(knotParts, 'elbowKnot');
  const knot = new THREE.Mesh(knotGeometry, vineMaterial);
  knot.name = 'elbowKnot';
  knot.castShadow = true;
  elbowRoot.add(knot);

  const curl = new THREE.Object3D();
  curl.name = 'torqueCurl';
  curl.visible = false;
  elbowRoot.add(curl);

  const curlGeometry = makeCurlGeometry(CURL_R, CURL_SWEEP);
  const curlMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(gold, accent, 0.55), 1.1, {
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'torqueCurlMaterial',
  });
  curlMaterial.opacity = 0;
  const curlMesh = new THREE.Mesh(curlGeometry, curlMaterial);
  curlMesh.name = 'torqueCurlMesh';
  curlMesh.renderOrder = 4;
  curl.add(curlMesh);

  const torqueHead = new THREE.Object3D();
  torqueHead.name = 'torqueHead';
  torqueHead.position.set(Math.cos(CURL_SWEEP) * CURL_R, Math.sin(CURL_SWEEP) * CURL_R, 0);
  curl.add(torqueHead);

  const lantern = new THREE.Object3D();
  lantern.name = 'lantern';
  lantern.position.set(0, -LANTERN_DROP, 0);
  tipRoot.add(lantern);

  const frameParts: ColoredPart[] = [];
  {
    const yoke = new THREE.TorusGeometry(0.05, 0.011, 4, 10);
    yoke.rotateY(Math.PI / 2);
    yoke.translate(0, LANTERN_DROP - 0.02, 0);
    frameParts.push({ geometry: yoke, color: brass });
    const cap = new THREE.ConeGeometry(LANTERN_R * 1.18, 0.075, 8);
    cap.translate(0, LANTERN_H * 0.5 + 0.03, 0);
    frameParts.push({ geometry: cap, color: brass });
    const base = new THREE.CylinderGeometry(LANTERN_R * 1.05, LANTERN_R * 0.78, 0.038, 8);
    base.translate(0, -LANTERN_H * 0.5 - 0.014, 0);
    frameParts.push({ geometry: base, color: brass });
    const finial = new THREE.ConeGeometry(0.028, 0.07, 6);
    finial.rotateZ(Math.PI);
    finial.translate(0, -LANTERN_H * 0.5 - 0.06, 0);
    frameParts.push({ geometry: finial, color: blendTokens(brass, palette.hemiGround, 0.4) });
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + Math.PI / 8;
      const rib = new THREE.BoxGeometry(0.012, LANTERN_H, 0.012);
      rib.translate(Math.cos(a) * LANTERN_R * 0.97, 0, Math.sin(a) * LANTERN_R * 0.97);
      frameParts.push({ geometry: rib, color: brass });
    }
  }
  const lanternFrameGeometry = mergeColoredParts(frameParts, 'lanternFrame');
  const lanternFrame = new THREE.Mesh(lanternFrameGeometry, vineMaterial);
  lanternFrame.name = 'lanternFrame';
  lanternFrame.castShadow = true;
  lantern.add(lanternFrame);

  const paperGeometry = new THREE.CylinderGeometry(LANTERN_R, LANTERN_R * 0.92, LANTERN_H, 8, 1, true);
  const paperMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(accent, palette.accentLight, 0.3),
    LANTERN_BASE_HDR,
    { side: THREE.DoubleSide, name: 'lanternPaper' },
  );
  const paper = new THREE.Mesh(paperGeometry, paperMaterial);
  paper.name = 'lanternPaper';
  lantern.add(paper);

  const coreGeometry = new THREE.IcosahedronGeometry(0.042, 1);
  const coreMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(gold, accent, 0.3), 1.7, {
    name: 'lanternCore',
  });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  core.name = 'lanternCore';
  lantern.add(core);

  const trailX = new Float32Array(TRAIL_SAMPLES);
  const trailY = new Float32Array(TRAIL_SAMPLES);
  const trailT = new Float64Array(TRAIL_SAMPLES);
  const liveX = new Float32Array(TRAIL_SAMPLES);
  const liveY = new Float32Array(TRAIL_SAMPLES);
  const liveAge = new Float32Array(TRAIL_SAMPLES);
  let trailCount = 0;
  let trailHead = 0;

  const ribbonVerts = TRAIL_SAMPLES * 2;
  const ribbonPos = new Float32Array(ribbonVerts * 3);
  const ribbonCol = new Float32Array(ribbonVerts * 4);
  const ribbonIdx = new Uint16Array((TRAIL_SAMPLES - 1) * 6);
  for (let i = 0; i < TRAIL_SAMPLES - 1; i++) {
    const a = i * 2;
    ribbonIdx[i * 6] = a;
    ribbonIdx[i * 6 + 1] = a + 2;
    ribbonIdx[i * 6 + 2] = a + 1;
    ribbonIdx[i * 6 + 3] = a + 1;
    ribbonIdx[i * 6 + 4] = a + 2;
    ribbonIdx[i * 6 + 5] = a + 3;
  }
  const ribbonGeometry = new THREE.BufferGeometry();
  const ribbonPosAttr = new THREE.BufferAttribute(ribbonPos, 3);
  ribbonPosAttr.setUsage(THREE.DynamicDrawUsage);
  const ribbonColAttr = new THREE.BufferAttribute(ribbonCol, 4);
  ribbonColAttr.setUsage(THREE.DynamicDrawUsage);
  ribbonGeometry.setAttribute('position', ribbonPosAttr);
  ribbonGeometry.setAttribute('color', ribbonColAttr);
  ribbonGeometry.setIndex(new THREE.BufferAttribute(ribbonIdx, 1));
  ribbonGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(pivot[0], pivot[1], pivot[2]), AB_L1 + AB_L2 + 1);
  ribbonGeometry.setDrawRange(0, 0);
  ribbonGeometry.name = 'tipRibbon';
  const ribbonMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(accent, palette.sun, 0.2), 1.55, {
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'tipRibbonMaterial',
  });
  ribbonMaterial.vertexColors = true;
  const ribbon = new THREE.Mesh(ribbonGeometry, ribbonMaterial);
  ribbon.name = 'tipRibbon';
  ribbon.frustumCulled = false;
  ribbon.renderOrder = 3;
  ribbon.visible = false;
  group.add(ribbon);

  const points = new Float64Array(EXACT_POINT_FLOATS);
  const tipWorld = new THREE.Vector3();
  const lanternWorld = new THREE.Vector3();
  let speed01 = 0;
  let lanternGain = 1;
  let flicker = 1;

  const applyLantern = (): void => {
    setEmissiveIntensity(paperMaterial, LANTERN_BASE_HDR * lanternGain * flicker);
    setEmissiveIntensity(coreMaterial, 5.2 * lanternGain * flicker);
  };

  return {
    group,
    pivotRoot,
    elbowRoot,
    tipRoot,
    torqueHead,
    tipWorld,
    lanternWorld,

    setState(th1, th2, torque, tipSpeed) {
      link1Node.rotation.z = th1;
      link2Node.rotation.z = th2;

      acrobotPointsExact(th1, th2, points);
      tipWorld.set(pivot[0] + (points[2] ?? 0), pivot[1] + (points[3] ?? 0), pivot[2]);
      const sx = Math.sin(th1 + th2);
      const cy = -Math.cos(th1 + th2);
      lanternWorld.set(
        tipWorld.x + sx * LANTERN_DROP,
        tipWorld.y + cy * LANTERN_DROP,
        pivot[2],
      );

      const t = torque > 0 ? 1 : torque < 0 ? -1 : 0;
      curl.visible = t !== 0;
      if (t !== 0) {
        curl.scale.set(t, 1, 1);
        curlMaterial.opacity = 0.2 + 0.14 * flicker;
      }

      speed01 = clampRange(tipSpeed / 8, 0, 1);
      paper.scale.set(1, 1 + speed01 * 0.06, 1);
    },

    setTime(t) {
      flicker = 1 + 0.09 * Math.sin(t * 3.1) + 0.05 * Math.sin(t * 7.7 + 1.3);
      applyLantern();
    },

    setLanternGain(gain) {
      lanternGain = gain > 0 ? gain : 0;
      applyLantern();
    },

    pushTrail(t) {
      const x = tipWorld.x;
      const y = tipWorld.y;
      if (trailCount > 0) {
        const head = trailHead;
        const dt = t - (trailT[head] ?? 0);
        if (dt < 0) {
          trailCount = 0;
        } else if (dt < TRAIL_SAMPLE_DT) {
          trailX[head] = x;
          trailY[head] = y;
          trailT[head] = t;
          return;
        }
      }
      trailHead = trailCount === 0 ? 0 : (trailHead + 1) % TRAIL_SAMPLES;
      trailX[trailHead] = x;
      trailY[trailHead] = y;
      trailT[trailHead] = t;
      if (trailCount < TRAIL_SAMPLES) trailCount++;
    },

    refreshTrail(t) {
      let live = 0;
      for (let k = 0; k < trailCount; k++) {
        const i = (trailHead - k + TRAIL_SAMPLES * 2) % TRAIL_SAMPLES;
        const age = t - (trailT[i] ?? 0);
        if (age > TRAIL_SPAN) break;
        liveX[live] = trailX[i] ?? 0;
        liveY[live] = trailY[i] ?? 0;
        liveAge[live] = age > 0 ? age : 0;
        live++;
      }
      if (live < 2) {
        ribbon.visible = false;
        ribbonGeometry.setDrawRange(0, 0);
        return;
      }
      for (let k = 0; k < live; k++) {
        const ax = liveX[Math.max(0, k - 1)] ?? 0;
        const ay = liveY[Math.max(0, k - 1)] ?? 0;
        const bx = liveX[Math.min(live - 1, k + 1)] ?? 0;
        const by = liveY[Math.min(live - 1, k + 1)] ?? 0;
        let nx = 0;
        let ny = 1;
        const dx = bx - ax;
        const dy = by - ay;
        const len = Math.hypot(dx, dy);
        if (len > 1e-6) {
          nx = -dy / len;
          ny = dx / len;
        }
        const fade = 1 - (liveAge[k] ?? 0) / TRAIL_SPAN;
        const halfW = TRAIL_MAX_HALF_W * fade * fade * (0.3 + 0.7 * speed01);
        const x = liveX[k] ?? 0;
        const y = liveY[k] ?? 0;
        const a = k * 2;
        ribbonPos[a * 3] = x - nx * halfW;
        ribbonPos[a * 3 + 1] = y - ny * halfW;
        ribbonPos[a * 3 + 2] = pivot[2];
        ribbonPos[(a + 1) * 3] = x + nx * halfW;
        ribbonPos[(a + 1) * 3 + 1] = y + ny * halfW;
        ribbonPos[(a + 1) * 3 + 2] = pivot[2];
        const alpha = fade * fade * fade * (0.2 + 0.8 * speed01);
        const lum = 0.5 + 0.5 * fade;
        for (let e = 0; e < 2; e++) {
          const v = (a + e) * 4;
          ribbonCol[v] = lum;
          ribbonCol[v + 1] = lum;
          ribbonCol[v + 2] = lum;
          ribbonCol[v + 3] = alpha;
        }
      }
      ribbonPosAttr.needsUpdate = true;
      ribbonColAttr.needsUpdate = true;
      ribbonGeometry.setDrawRange(0, (live - 1) * 6);
      ribbon.visible = speed01 > 0.02;
    },

    clearTrail() {
      trailCount = 0;
      trailHead = 0;
      ribbon.visible = false;
      ribbonGeometry.setDrawRange(0, 0);
    },

    setPixelRatio() {
    },

    dispose() {
      link1Geometry.dispose();
      link2Geometry.dispose();
      knotGeometry.dispose();
      lanternFrameGeometry.dispose();
      vineMaterial.dispose();
      paperGeometry.dispose();
      paperMaterial.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      curlGeometry.dispose();
      curlMaterial.dispose();
      ribbonGeometry.dispose();
      ribbonMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


export function makeVineGeometry(
  length: number,
  radius: number,
  seed: number,
  dark: THREE.Color,
  lit: THREE.Color,
  leafColor: THREE.Color,
): THREE.BufferGeometry {
  const rng = streamFor(seed, 'vine');
  const parts: ColoredPart[] = [];
  const SEGS = 6;
  for (let i = 0; i < SEGS; i++) {
    const y0 = -(length * i) / SEGS;
    const y1 = -(length * (i + 1)) / SEGS;
    const bow = Math.sin((i / SEGS) * Math.PI) * radius * 0.55;
    const seg = new THREE.CylinderGeometry(radius * 0.94, radius, y0 - y1, 7, 1);
    seg.translate(bow * (i % 2 === 0 ? 1 : -1) * 0.5, (y0 + y1) * 0.5, 0);
    parts.push({ geometry: seg, color: i % 2 === 0 ? dark : lit });
  }
  for (let i = 1; i < SEGS; i += 2) {
    const y = -(length * i) / SEGS;
    const knot = new THREE.TorusGeometry(radius * 1.15, radius * 0.34, 4, 8);
    knot.rotateX(Math.PI / 2);
    knot.translate(0, y, 0);
    parts.push({ geometry: knot, color: lit });
  }
  for (let i = 0; i < 2; i++) {
    const y = -length * (0.3 + rng() * 0.45);
    const a = rng() * Math.PI * 2;
    const g = new THREE.ConeGeometry(radius * 0.9, radius * 4.4, 4);
    g.rotateZ(Math.PI / 2);
    g.rotateY(a);
    g.translate(Math.cos(a) * radius * 2.2, y, Math.sin(a) * radius * 2.2);
    parts.push({ geometry: g, color: leafColor });
  }
  return mergeColoredParts(parts, 'trapezeVine');
}

export function makeCurlGeometry(radius: number, sweep: number): THREE.BufferGeometry {
  const STEPS = 14;
  const tris: number[] = [];
  const push = (x: number, y: number): void => {
    tris.push(x, y, 0);
  };
  let prevInX = 0;
  let prevInY = 0;
  let prevOutX = 0;
  let prevOutY = 0;
  for (let i = 0; i <= STEPS; i++) {
    const t = i / STEPS;
    const a = -sweep + t * sweep * 2;
    const w = 0.008 + 0.026 * t * t;
    const inR = radius - w;
    const outR = radius + w;
    const cx = Math.cos(a);
    const cy = Math.sin(a);
    const ix = cx * inR;
    const iy = cy * inR;
    const ox = cx * outR;
    const oy = cy * outR;
    if (i > 0) {
      push(prevInX, prevInY);
      push(prevOutX, prevOutY);
      push(ox, oy);
      push(prevInX, prevInY);
      push(ox, oy);
      push(ix, iy);
    }
    prevInX = ix;
    prevInY = iy;
    prevOutX = ox;
    prevOutY = oy;
  }
  {
    const a = sweep;
    const cx = Math.cos(a);
    const cy = Math.sin(a);
    const tx = -cy;
    const ty = cx;
    const bx = cx * radius;
    const by = cy * radius;
    const h = 0.085;
    const wSide = 0.062;
    push(bx + cx * wSide, by + cy * wSide);
    push(bx - cx * wSide, by - cy * wSide);
    push(bx + tx * h, by + ty * h);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(tris), 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'torqueCurl';
  return g;
}
