import * as THREE from 'three';
import type { SpatialInstances } from './instances';
import type { WorldBuilder } from './world';

type Cache = { target: THREE.WebGLRenderTarget; matrix: THREE.Matrix4; revisions: Map<SpatialInstances, number>; transforms: Map<SpatialInstances, THREE.Matrix4>; visible: Map<SpatialInstances, boolean> };

export function cacheStaticShadows(world: WorldBuilder) {
  const sources = new Set<SpatialInstances>(), caches = new Map<THREE.Camera, Cache>();
  const group = new THREE.Group(), matrix = new THREE.Matrix4(), sphere = new THREE.Sphere(), strips: THREE.Vector4[] = [];
  const uniforms = { uDepth: { value: null as THREE.DepthTexture | null }, uShift: { value: new THREE.Vector2() } };
  const material = world.own(new THREE.ShaderMaterial({
    uniforms, depthWrite: true, colorWrite: false, side: THREE.DoubleSide,
    vertexShader: 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'uniform sampler2D uDepth; uniform vec2 uShift; void main() { ivec2 p = ivec2(gl_FragCoord.xy) - ivec2(uShift), size = textureSize(uDepth, 0); gl_FragDepth = any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size)) ? 1.0 : texelFetch(uDepth, p, 0).r; gl_FragColor = vec4(1.0); }',
  }));
  const helper = () => {
    const geometry = world.own(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3)));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true; mesh.frustumCulled = false; mesh.customDepthMaterial = material;
    mesh.onBeforeRender = () => geometry.setDrawRange(0, 0);
    return mesh;
  };
  const restore = helper(), save = helper();
  let active: Cache, valid = false, dirty = false;
  restore.onBeforeShadow = (renderer, _object, _camera, camera) => {
    const target = renderer.getRenderTarget()!;
    let cache = caches.get(camera);
    if (cache && (cache.target.width !== target.width || cache.target.height !== target.height)) { cache.target.dispose(); caches.delete(camera); cache = undefined; }
    if (!cache) {
      const copy = new THREE.WebGLRenderTarget(target.width, target.height, { depthTexture: new THREE.DepthTexture(target.width, target.height, THREE.UnsignedIntType) });
      renderer.initRenderTarget(copy); cache = { target: copy, matrix: new THREE.Matrix4(), revisions: new Map(), transforms: new Map(), visible: new Map() }; caches.set(camera, cache);
      renderer.setRenderTarget(target);
    }
    active = cache; matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const current = matrix.elements, previous = cache.matrix.elements, shift = uniforms.uShift.value;
    valid = sources.size === cache.revisions.size && [...sources].every(source => source.removed <= cache.revisions.get(source)! && source.group.matrixWorld.equals(cache.transforms.get(source)!) && source.group.visible === cache.visible.get(source));
    valid &&= current.every((value, i) => i === 12 || i === 13 || Math.abs(value - previous[i]!) < 1e-8);
    shift.set((current[12]! - previous[12]!) * target.width / 2, (current[13]! - previous[13]!) * target.height / 2);
    valid &&= Math.abs(shift.x - Math.round(shift.x)) < 0.001 && Math.abs(shift.y - Math.round(shift.y)) < 0.001 && Math.abs(shift.x) < target.width && Math.abs(shift.y) < target.height;
    shift.round(); strips.length = 0;
    if (valid) {
      const x = shift.x * 2 / target.width, y = shift.y * 2 / target.height;
      if (x) strips.push(new THREE.Vector4(x > 0 ? -1 : 1 + x, -1, x > 0 ? -1 + x : 1, 1));
      if (y) strips.push(new THREE.Vector4(-1, y > 0 ? -1 : 1 + y, 1, y > 0 ? -1 + y : 1));
    }
    dirty = !valid || shift.lengthSq() > 0 || [...sources].some(source => source.revision !== cache.revisions.get(source));
    uniforms.uDepth.value = cache.target.depthTexture;
    restore.geometry.setDrawRange(0, valid ? 3 : 0);
    for (const source of sources) for (const meshes of source.batches.values()) meshes.forEach(mesh => {
      if (!mesh.userData.shadowCaster) return;
      sphere.copy(mesh.boundingSphere!).applyMatrix4(mesh.matrixWorld).applyMatrix4(matrix);
      const update = !valid || mesh.userData.revision > active.revisions.get(source)! || strips.some(strip => sphere.center.x + sphere.radius >= strip.x && sphere.center.y + sphere.radius >= strip.y && sphere.center.x - sphere.radius <= strip.z && sphere.center.y - sphere.radius <= strip.w);
      mesh.castShadow = update;
    });
  };
  save.onBeforeShadow = renderer => {
    if (dirty) {
      const target = renderer.getRenderTarget()!;
      renderer.copyTextureToTexture(target.depthTexture!, active.target.depthTexture!);
      renderer.setRenderTarget(target); active.matrix.copy(matrix);
      for (const source of sources) { active.revisions.set(source, source.revision); active.transforms.set(source, source.group.matrixWorld.clone()); active.visible.set(source, source.group.visible); }
    }
    save.geometry.setDrawRange(0, 0);
    for (const source of sources) for (const meshes of source.batches.values()) meshes.forEach(mesh => { mesh.castShadow = mesh.userData.shadowCaster ?? false; });
  };
  group.add(restore, save); world.add(group);
  world.group.children.splice(world.group.children.indexOf(group), 1); world.group.children.unshift(group);
  world.own({ dispose() { for (const cache of caches.values()) cache.target.dispose(); } });
  return {
    add(source: SpatialInstances) {
      sources.add(source);
      group.add(source.group); group.remove(save); group.add(save);
    },
  };
}
