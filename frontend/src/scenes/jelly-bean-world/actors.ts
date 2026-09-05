import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { alpineTexture } from '@kit/alpine';
import { engravedPanel, modelParts, robotMaterials } from '@kit/robot';
import { makeWorld, type WorldBuilder, type WorldCore } from '@kit/world';
import { JBW_BANANA, JBW_ONION, JBW_JELLY, JBW_WALL, JBW_TREE, JBW_TRUFFLE, JBW_COLOR, jbwCollectible } from '@sim/envs/jellybeanworld';
import { aspenTree, fieldstones } from '../half-cheetah/world';

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
  const stone = fieldstones(world);
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
      const uv = cap.getAttribute('uv'), colors = new Float32Array(uv.count * 3), snow = new THREE.Color(0xf5f7ff);
      for (let j = 0; j < uv.count; j++) { uv.setXY(j, -1, -1); snow.toArray(colors, j * 3); }
      cap.setAttribute('color', new THREE.BufferAttribute(colors, 3)); pieces.push(cap);
    }
    const geometry = mergeGeometries(pieces); pieces.forEach(part => part.dispose()); base.dispose();
    return geometry;
  };
  stone.material.color.set(0xffffff);
  stone.material.onBeforeCompile = shader => { shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', 'if (vMapUv.x >= 0.0) {\n#include <map_fragment>\n}'); };
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
  const truffleMaterial = fruit(JBW_TRUFFLE, 0.92);
  truffleMaterial.map = truffleMaterial.bumpMap = alpineTexture(world, 'prairie-fieldstone'); truffleMaterial.bumpScale = 0.018;
  add(JBW_TRUFFLE, truffle, truffleMaterial);
  return models;
}

export function buildForager() {
  const world = makeWorld('parklandForager'), root = world.group, materials = robotMaterials();
  root.scale.setScalar(1.9);
  Object.values(materials).forEach(material => world.own(material));
  const { armor, accent, frame, alloy, rubber, status } = materials, parts = modelParts(world), { box, disc, rod } = parts;
  box(root, frame, 0, 0.27, 0, 0.66, 0.18, 0.49, 0.06);
  parts.add(root, engravedPanel(0.49, 0.63, 0.14, 0.045, 0.24).rotateZ(-Math.PI / 2).rotateX(-Math.PI / 2).translate(0, 0.39, 0), armor);
  for (const z of [-0.29, 0.29]) {
    box(root, armor, -0.015, 0.28, z, 0.42, 0.12, 0.09, 0.025);
    box(root, accent, 0.06, 0.345, z, 0.2, 0.018, 0.08, 0.006);
    for (let i = 0; i < 3; i++) box(root, frame, -0.19 + i * 0.034, 0.295, z * 1.17, 0.018, 0.045, 0.014, 0.003);
    for (const x of [-0.23, 0.23]) rod(root, alloy, new THREE.Vector3(x * 0.5, 0.28, z * 0.7), new THREE.Vector3(x, 0.15, z), 0.02);
  }
  for (let i = 0; i < 8; i++) {
    const angle = i * Math.PI / 4, x = Math.cos(angle) * 0.33, z = Math.sin(angle) * 0.32;
    parts.add(root, new THREE.CylinderGeometry(0.033, 0.039, 0.027, 20).rotateX(Math.PI / 2).rotateY(Math.PI / 2 - angle).translate(x, 0.32, z), alloy);
    parts.add(root, new THREE.SphereGeometry(0.025, 16, 10).scale(1, 1, 0.32).rotateY(Math.PI / 2 - angle).translate(x * 1.05, 0.32, z * 1.05), status);
  }
  box(root, accent, 0.39, 0.29, 0, 0.11, 0.14, 0.30, 0.035);
  box(root, rubber, 0.448, 0.29, 0, 0.02, 0.09, 0.25, 0.022);
  for (let i = -4; i <= 4; i++) rod(root, alloy, new THREE.Vector3(0.462, 0.251, i * 0.025), new THREE.Vector3(0.462, 0.329, i * 0.025), 0.003);
  for (const z of [-0.125, 0.125]) rod(root, frame, new THREE.Vector3(0.35, 0.29, z), new THREE.Vector3(0.08, 0.29, z * 1.8), 0.033);
  for (const z of [-0.135, 0, 0.135]) {
    box(root, frame, -0.26, 0.43, z, 0.1, 0.055, 0.095, 0.012);
    for (const y of [0.47, 0.59]) parts.add(root, new THREE.CylinderGeometry(0.049,0.049,0.025,20).translate(-0.26,y,z), alloy);
    for (const side of [-1, 1]) rod(root, accent, new THREE.Vector3(-0.26,0.47,z+side*0.041), new THREE.Vector3(-0.26,0.59,z+side*0.041), 0.005);
  }
  const wheel = new THREE.Group();
  parts.add(wheel, new THREE.TorusGeometry(0.11, 0.036, 12, 28), rubber);
  disc(wheel, frame, 0, 0, 0, 0.105, 0.09);
  for (const side of [-1, 1]) {
    disc(wheel, alloy, 0, 0, side * 0.058, 0.075, 0.025);
    disc(wheel, accent, 0, 0, side * 0.075, 0.032, 0.02, 8);
    for (let i = 0; i < 6; i++) disc(wheel, frame, Math.cos(i * Math.PI / 3) * 0.054, Math.sin(i * Math.PI / 3) * 0.054, side * 0.075, 0.009, 0.013, 6);
  }
  for (let i = 0; i < 20; i++) parts.add(wheel, new THREE.BoxGeometry(0.026, 0.014, 0.067).translate(0, 0.144, 0).rotateZ(i * Math.PI / 10), rubber);
  const indicators = world.own(new THREE.InstancedMesh(world.own(new THREE.SphereGeometry(1, 8, 6)), world.own(new THREE.MeshBasicMaterial({ color: 0xffffff })), 11));
  const sensor = new THREE.Object3D(), color = new THREE.Color();
  for (let i = 0; i < 11; i++) {
    if (i < 3) { sensor.position.set(-0.26, 0.53, (i - 1) * 0.135); sensor.scale.set(0.04, 0.065, 0.04); }
    else { const angle = (i - 3) * Math.PI / 4; sensor.position.set(Math.cos(angle) * 0.353, 0.32, Math.sin(angle) * 0.343); sensor.scale.setScalar(0.012); }
    sensor.updateMatrix(); indicators.setMatrixAt(i, sensor.matrix); indicators.setColorAt(i, color.setScalar(0.04));
  }
  indicators.name = 'localScentAndNeighbourSensors'; indicators.computeBoundingSphere(); world.add(indicators);
  parts.build();
  const wheelPositions = [-0.23, 0.23].flatMap(x => [-0.29, 0.29].map(z => new THREE.Vector3(x, 0.145, z)));
  const matrix = new THREE.Matrix4(), spins = new Float64Array(4);
  const wheels = wheel.children.map(child => {
    const mesh = child as THREE.Mesh;
    const instances = world.own(new THREE.InstancedMesh(mesh.geometry, mesh.material, 4));
    wheelPositions.forEach((position, i) => instances.setMatrixAt(i, matrix.makeTranslation(position)));
    instances.castShadow = instances.receiveShadow = true;
    instances.instanceMatrix.setUsage(THREE.DynamicDrawUsage); instances.computeBoundingSphere(); world.add(instances);
    return instances;
  });
  const handle = {
    sense(values: readonly number[]) {
      for (let i = 0; i < 11; i++) indicators.setColorAt(i, color.set(i < 3 ? [0xff786c, 0xc3e26e, 0xad9fff][i]! : 0x6db9c1).multiplyScalar(0.045 + Math.log1p(values[i]!) * 0.6));
      indicators.instanceColor!.needsUpdate = true;
    },
    roll(distance: number, turn: number) {
      wheelPositions.forEach((position, i) => {
        spins[i]! -= (distance + turn * position.z) / (0.145 * root.scale.x);
        matrix.makeRotationZ(spins[i]!).setPosition(position);
        for (const wheel of wheels) wheel.setMatrixAt(i, matrix);
      });
      for (const wheel of wheels) wheel.instanceMatrix.needsUpdate = true;
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
