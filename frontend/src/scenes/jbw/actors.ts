import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { WorldBuilder } from '@kit/world';
import { JBW_BANANA, JBW_ONION, JBW_JELLY, JBW_WALL, JBW_TREE, JBW_TRUFFLE, JBW_COLOR, jbwCollectible } from '@sim/envs/jbw';
import { aspenTree, fieldstones } from '@kit/landscape';

export interface ItemModel { geometry: THREE.BufferGeometry; material: THREE.Material; depth?: THREE.MeshDepthMaterial }

export function itemModels(world: WorldBuilder): Map<number, ItemModel> {
  const models = new Map<number, ItemModel>(), motion = { value: new THREE.Vector4() };
  const animate = (material: THREE.Material) => {
    const compile = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      compile.call(material, shader, renderer); shader.uniforms.uItemMotion = motion;
      shader.vertexShader = shader.vertexShader.replace('#include <common>', `#include <common>
uniform vec4 uItemMotion;
mat3 itemRotation() { return mat3(vec3(uItemMotion.x, 0.0, -uItemMotion.y), vec3(0.0, 1.0, 0.0), vec3(uItemMotion.y, 0.0, uItemMotion.x)); }
`).replace('#include <begin_vertex>', '#include <begin_vertex>\ntransformed = itemRotation() * transformed; transformed.y += dot(vec2(instanceMatrix[0].x, -instanceMatrix[0].z), uItemMotion.zw) * 0.16;')
        .replace('#include <beginnormal_vertex>', '#include <beginnormal_vertex>\nobjectNormal = itemRotation() * objectNormal;');
    };
  };
  world.time({ setTime(time) { motion.value.set(Math.cos(time * 0.48), Math.sin(time * 0.48), Math.sin(time * 1.6), Math.cos(time * 1.6)); } });
  const add = (kind: number, geometry: THREE.BufferGeometry, material: THREE.MeshStandardMaterial) => {
    world.own(geometry); world.own(material);
    if (jbwCollectible(kind)) {
      geometry.scale(2.1, 2.1, 2.1).translate(0, 1.05, 0);
      material.emissive.copy(material.color); material.emissiveIntensity = 0.16 / (material.color.r * 0.2126 + material.color.g * 0.7152 + material.color.b * 0.0722);
      material.onBeforeCompile = shader => {
        shader.fragmentShader = shader.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
float edgeGlow = pow(1.0 - max(dot(normal, normalize(vViewPosition)), 0.0), 3.0);
totalEmissiveRadiance += diffuseColor.rgb / max(dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.001) * edgeGlow * 0.9;
`);
      };
      const radius = kind === JBW_TRUFFLE ? 1.7 : 0.95;
      const ring = world.own(new THREE.PlaneGeometry(radius * 2, radius * 2).rotateX(-Math.PI / 2).translate(0, 0.035, 0));
      const rgba = new Float32Array(ring.getAttribute('position').count * 4);
      for (let i = 0; i < rgba.length; i += 4) rgba.set([1, 1, 1, 0.38], i);
      ring.setAttribute('color', new THREE.BufferAttribute(rgba, 4));
      let glow: THREE.BufferGeometry = ring;
      if (kind === JBW_TRUFFLE) {
        const plume = new THREE.CylinderGeometry(1.1, 0.45, 6, 12, 4, true).translate(0, 3.05, 0), positions = plume.getAttribute('position'), color = new Float32Array(positions.count * 4);
        for (let i = 0; i < positions.count; i++) color.set([1, 0.85, 0.68, Math.pow(Math.max(0, 1 - positions.getY(i) / 6.05), 1.5) * 0.045], i * 4);
        plume.setAttribute('color', new THREE.BufferAttribute(color, 4));
        glow = world.own(mergeGeometries([ring, plume])); plume.dispose();
      }
      const glowMaterial = world.own(new THREE.MeshBasicMaterial({ color: kind === JBW_TRUFFLE ? 0xffd394 : material.color, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      glowMaterial.customProgramCacheKey = () => `scent-source-${kind}`;
      glowMaterial.onBeforeCompile = shader => {
        shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 vGlowUV;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvGlowUV = uv;');
        shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vGlowUV;').replace('#include <color_fragment>', `#include <color_fragment>\n${kind === JBW_TRUFFLE ? 'if (vPlumeHeight < 0.05)' : ''} { float radius = length(vGlowUV * 2.0 - 1.0); if (radius >= 1.0) discard; diffuseColor.a *= (1.0 - radius) * (1.0 - radius); }`);
        if (kind !== JBW_TRUFFLE) return;
        shader.uniforms.uItemMotion = motion;
        shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nuniform vec4 uItemMotion; varying vec3 vPlumeNormal, vPlumeView; varying float vPlumeHeight;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvPlumeHeight = position.y; transformed.xz += sin(vec2(position.y * 1.7, position.y * 1.3) + uItemMotion.zw) * position.y * 0.045;')
          .replace('#include <project_vertex>', '#include <project_vertex>\nvPlumeNormal = normalMatrix * mat3(instanceMatrix) * normal; vPlumeView = -mvPosition.xyz;');
        shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vPlumeNormal, vPlumeView; varying float vPlumeHeight;')
          .replace('#include <color_fragment>', '#include <color_fragment>\nif (vPlumeHeight > 0.05) diffuseColor.a *= smoothstep(0.0, 0.7, abs(dot(normalize(vPlumeNormal), normalize(vPlumeView))));');
      };
      models.set(kind + 8, { geometry: glow, material: glowMaterial });
    }
    const depth = world.own(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking }));
    if (jbwCollectible(kind)) { animate(material); animate(depth); }
    models.set(kind, { geometry, material, depth });
  };
  const surface = (geometry: THREE.BufferGeometry, tint: (x: number, y: number, z: number) => THREE.Color) => {
    const p = geometry.getAttribute('position'), colors = new Float32Array(p.count * 3);
    for (let i = 0; i < p.count; i++) tint(p.getX(i), p.getY(i), p.getZ(i)).toArray(colors, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  };
  const fruit = (kind: number, roughness: number) => new THREE.MeshStandardMaterial({ color: new THREE.Color(...JBW_COLOR[kind]!).convertSRGBToLinear(), roughness, vertexColors: true });
  const banana = new THREE.LatheGeometry(Array.from({ length: 17 }, (_, i) => {
    const t = i / 16; return new THREE.Vector2(0.075 * Math.pow(Math.sin(t * Math.PI), 0.4) + 0.005, t * 0.72);
  }), 12);
  const bp = banana.getAttribute('position');
  for (let i = 0; i < bp.count; i++) bp.setX(i, bp.getX(i) + 0.22 * Math.sin(bp.getY(i) / 0.72 * Math.PI));
  surface(banana, (_x, y) => new THREE.Color(0xffffff).lerp(new THREE.Color(0x746634), 1 - THREE.MathUtils.smoothstep(Math.min(y, 0.72 - y), 0, 0.036)));
  banana.rotateZ(-0.9).translate(-0.34, 0.11, 0); banana.computeVertexNormals();
  add(JBW_BANANA, banana, fruit(JBW_BANANA, 0.38));
  const profile = new THREE.SplineCurve([[0, 0], [0.12, 0.04], [0.25, 0.16], [0.28, 0.29], [0.23, 0.43], [0.1, 0.51], [0.034, 0.61], [0.008, 0.73]].map(([x, y]) => new THREE.Vector2(x, y)));
  const onion = new THREE.LatheGeometry(profile.getPoints(28), 32);
  const op = onion.getAttribute('position');
  for (let i = 0; i < op.count; i++) {
    const angle = Math.atan2(op.getZ(i), op.getX(i)), rib = 1 + 0.025 * Math.cos(angle * 12);
    op.setXYZ(i, op.getX(i) * rib, op.getY(i), op.getZ(i) * rib);
  }
  surface(onion, (x, y, z) => new THREE.Color().setScalar(0.83 + 0.17 * Math.cos(Math.atan2(z, x) * 12) ** 2).lerp(new THREE.Color(0xddaaf2), THREE.MathUtils.smoothstep(y, 0.52, 0.72)));
  onion.computeVertexNormals(); add(JBW_ONION, onion, fruit(JBW_ONION, 0.34));
  const jelly = new THREE.SphereGeometry(1, 28, 18), jp = jelly.getAttribute('position');
  for (let i = 0; i < jp.count; i++) {
    const x = jp.getX(i), y = jp.getY(i), z = jp.getZ(i);
    jp.setXYZ(i, x * 0.38, y * 0.21 + 0.23 + 0.055 * x * x, z * 0.24 + 0.095 * x * x);
  }
  jelly.computeVertexNormals();
  add(JBW_JELLY, jelly, new THREE.MeshPhysicalMaterial({ color: new THREE.Color(...JBW_COLOR[JBW_JELLY]!).convertSRGBToLinear(), roughness: 0.24, clearcoat: 0.55, clearcoatRoughness: 0.26, sheen: 0.25, sheenColor: 0xffb8a4 }));
  const stone = fieldstones();
  const wall = (base: THREE.BufferGeometry) => {
    const pieces: THREE.BufferGeometry[] = [], centers: THREE.Vector3[] = [];
    for (let y = 0; y < 3; y++) for (const side of [-1, 1]) {
      const x = side * 0.235 + (y % 2 ? 0.015 : -0.015), z = y * 0.27 - 0.05;
      pieces.push(base.clone().scale(0.7, 1.3, 1.2).translate(x, z, 0)); centers.push(new THREE.Vector3(x, z + 0.29, 0));
    }
    for (const [part, geometry] of pieces.entries()) {
      const position = geometry.getAttribute('position'), indices = geometry.index!, visible: number[] = [];
      for (let i = 0; i < indices.count; i += 3) {
        const triangle = [indices.getX(i), indices.getX(i + 1), indices.getX(i + 2)];
        if (!centers.some((center, other) => other !== part && triangle.every(v => ((position.getX(v) - center.x) / 0.21) ** 2 + ((position.getY(v) - center.y) / 0.20) ** 2 + ((position.getZ(v) - center.z) / 0.30) ** 2 < 1))) visible.push(...triangle);
      }
      geometry.setIndex(visible);
    }
    for (const [y, sx, sy, sz] of [[1.09, 0.51, 0.18, 0.34], [0.025, 0.63, 0.12, 0.51]]) {
      const cap = new THREE.SphereGeometry(1, 24, 12).scale(sx!, sy!, sz!).translate(0, y!, 0);
      const count = cap.getAttribute('position').count, colors = new Float32Array(count * 3), snow = new THREE.Color(0xf5f7ff);
      for (let j = 0; j < count; j++) snow.toArray(colors, j * 3);
      cap.setAttribute('color', new THREE.BufferAttribute(colors, 3)); pieces.push(cap);
    }
    const geometry = mergeGeometries(pieces); pieces.forEach(part => part.dispose()); base.dispose();
    return geometry;
  };
  add(JBW_WALL, wall(stone.geometry), stone.material);
  const tree = aspenTree(world, []);
  add(JBW_TREE, tree.geometry, tree.material);
  const truffle = new THREE.SphereGeometry(0.22, 24, 16), tp = truffle.getAttribute('position');
  for (let i = 0; i < tp.count; i++) {
    const x = tp.getX(i), y = tp.getY(i), z = tp.getZ(i), bump = 1 + 0.055 * Math.sin(x * 83) * Math.sin(y * 71) * Math.sin(z * 79);
    tp.setXYZ(i, x * bump, y * bump * 0.8 + 0.15, z * bump);
  }
  surface(truffle, (x, y, z) => new THREE.Color().setScalar(0.72 + 0.28 * Math.sin(x * 83) ** 2 * Math.cos(y * 71 + z * 67) ** 2));
  truffle.computeVertexNormals();
  add(JBW_TRUFFLE, truffle, fruit(JBW_TRUFFLE, 0.92));
  return models;
}
