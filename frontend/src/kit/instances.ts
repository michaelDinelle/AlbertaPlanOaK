import * as THREE from 'three/webgpu';
import { Fn, cameraProjectionMatrix, dot, floor, fract, instanceIndex, instancedBufferAttribute, length, materialColor, max, mix, modelViewMatrix, normalLocal, positionGeometry, select, smoothstep, storage, transformNormal, uniform, varying, vec2, vec3, vec4 } from 'three/tsl';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { deformOf, glFragCoord, type NumberUniform } from './materials';

export interface InstanceModel {
  geometry: THREE.BufferGeometry;
  material: THREE.NodeMaterial;
  castShadow?: boolean;
  receiveShadow?: boolean;
  padding?: number;
}

type Detail = { geometry: THREE.BufferGeometry; error: number };
type Chunk = { matrices: Float32Array; colors?: Float32Array; born: number; box: THREE.Box3; center: THREE.Vector3; extent: THREE.Vector3; minScale: number; maxScale: number };
type Batch = { box: THREE.Box3; mesh: THREE.InstancedMesh; selected: Chunk[]; queued: Chunk[]; born: THREE.InstancedBufferAttribute; projection: NumberUniform; clock: NumberUniform; count: number; capacity: number };
type View = { camera: THREE.Camera; shadow: boolean; matrix: THREE.Matrix4; revision: number; height: number; batches: Batch[][] };
const sources = new WeakMap<THREE.Object3D, SpatialInstances>();
const detailCache = new WeakMap<THREE.BufferGeometry, Detail[]>();
const matrix = new THREE.Matrix4(), localView = new THREE.Matrix4(), frustum = new THREE.Frustum();

function details(source: THREE.BufferGeometry): Detail[] {
  let levels = detailCache.get(source);
  if (levels) return levels;
  levels = [{ geometry: source, error: 0 }]; detailCache.set(source, levels);
  if ((source.index?.count ?? source.getAttribute('position').count) < 96) return levels;
  const welded = mergeVertices(source), position = welded.getAttribute('position');
  const indices = Uint32Array.from(welded.index!.array), attributes = Object.entries(welded.attributes).filter(([name]) => name !== 'position');
  const stride = attributes.reduce((sum, [, value]) => sum + value.itemSize, 0), weights = attributes.flatMap(([name, value]) => Array(value.itemSize).fill(name === 'normal' ? 0.15 : name === 'uv' ? 0.1 : 1));
  const values = new Float32Array(position.count * stride);
  for (let i = 0; i < position.count; i++) { let at = i * stride; for (const [, value] of attributes) for (let j = 0; j < value.itemSize; j++) values[at++] = value.getComponent(i, j); }
  for (let count = Math.floor(indices.length / 12) * 3; count >= 12; count = Math.floor(count / 12) * 3) {
    const [reduced, error] = MeshoptSimplifier.simplifyWithAttributes(indices, position.array as Float32Array, 3, values, stride, weights, null, count, 1e6, ['ErrorAbsolute', 'Prune', 'Permissive']);
    if (reduced.length >= (levels.at(-1)!.geometry.index?.count ?? indices.length) * 0.8 || reduced.length < 3) break;
    const geometry = new THREE.BufferGeometry();
    for (const [name, attribute] of Object.entries(welded.attributes)) geometry.setAttribute(name, attribute);
    geometry.setIndex(new THREE.BufferAttribute(reduced, 1)); geometry.boundingBox = source.boundingBox; geometry.boundingSphere = source.boundingSphere;
    levels.push({ geometry, error: Math.max(error, levels.at(-1)!.error * 1.8, 1e-5) });
  }
  source.addEventListener('dispose', () => { for (const level of levels!.slice(1)) level.geometry.dispose(); welded.dispose(); detailCache.delete(source); });
  return levels;
}

export function instancedMesh(geometry: THREE.BufferGeometry, material: THREE.Material | THREE.Material[], count: number): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.instanceMatrix = new THREE.StorageInstancedBufferAttribute(mesh.instanceMatrix.array as Float32Array, 16);
  return mesh;
}

export class SpatialInstances {
  readonly group = new THREE.Group();
  readonly batches = new Map<string, THREE.InstancedMesh[]>();
  private readonly chunks = new Map<number | string, Chunk>();
  private readonly views = new Map<THREE.Camera, View>();
  private readonly levels: Detail[][] = [];
  private readonly retired: Batch[] = [];
  private readonly bounds = new THREE.Box3();
  revision = 0;
  removed = 0;

  constructor(readonly models: InstanceModel[]) {
    sources.set(this.group, this);
    for (const model of models) {
      model.geometry.computeBoundingBox(); model.geometry.computeBoundingSphere();
      this.bounds.union(model.geometry.boundingBox!.clone().expandByScalar(model.padding ?? 0));
      if (model.material.transparent && !model.material.depthWrite && model.material.blending === THREE.AdditiveBlending) model.material.forceSinglePass = true;
    }
  }

  private createBatch(model: InstanceModel, level: Detail, next: Detail | undefined, view: View, capacity: number): Batch {
    const projection = uniform(1), clock = uniform(0), error = uniform(new THREE.Vector2(level.error, next?.error ?? 1e20)), born = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    const source = model.material, material = source.clone(), mesh = instancedMesh(level.geometry, material, capacity), matrices = mesh.instanceMatrix;
    material.alphaTest = source.alphaTest;
    const transform = storage(matrices, 'mat4', capacity).element(instanceIndex), deformed = deformOf(source)?.(transform), age = clock.sub(instancedBufferAttribute(born, 'float'));
    const detail = varying(Fn(() => {
      const depth = max(modelViewMatrix.mul(vec4(transform.mul(vec4(0, 0, 0, 1)).xyz, 1)).z.negate(), 0.001).toVar();
      const scale = max(length(transform.mul(vec4(1, 0, 0, 0)).xyz), max(length(transform.mul(vec4(0, 1, 0, 0)).xyz), length(transform.mul(vec4(0, 0, 1, 0)).xyz)));
      const range = smoothstep(0.65, 1.05, error.mul(projection.mul(scale).div(select(cameraProjectionMatrix.mul(vec4(0, 0, 1, 0)).w.equal(-1), depth, 1))));
      return vec2(range.x, mix(range.x, range.y, smoothstep(0, 0.4, age)));
    })());
    const coverage = fract(fract(dot(floor(glFragCoord), vec2(0.06711056, 0.00583715))).mul(52.9829189));
    const keep = coverage.greaterThanEqual(detail.x).and(coverage.lessThan(detail.y));
    material.positionNode = Fn(() => {
      if (deformed?.normal) normalLocal.assign(transformNormal(deformed.normal, transform));
      const position = transform.mul(vec4(deformed?.position ?? positionGeometry, 1)).xyz.toVar();
      return select(detail.x.greaterThanEqual(detail.y), vec3(0), position);
    })();
    material.maskNode = source.maskNode ? (source.maskNode as THREE.Node<'bool'>).and(keep) : keep;
    if (source.maskShadowNode) material.maskShadowNode = (source.maskShadowNode as THREE.Node<'bool'>).and(keep);
    if (deformed?.tint) material.colorNode = vec4((source.colorNode ?? materialColor) as THREE.Node<'vec4'>).mul(vec4(varying(deformed.tint), 1));
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    mesh.castShadow = view.shadow;
    mesh.receiveShadow = model.receiveShadow ?? false; mesh.matrixAutoUpdate = false; mesh.boundingSphere = new THREE.Sphere();
    const batch: Batch = { box: new THREE.Box3(), mesh, selected: [], queued: [], born, projection, clock, count: 0, capacity };
    mesh.onBeforeRender = (_renderer, _scene, camera) => { mesh.count = camera === view.camera ? batch.count : 0; };
    this.group.add(mesh); return batch;
  }

  prepare(camera: THREE.Camera, height: number, shadow: boolean): void {
    if (!shadow) for (const batch of this.retired.splice(0)) this.dropBatch(batch);
    if (!this.levels.length && this.chunks.size) this.levels.push(...this.models.map(model => details(model.geometry)));
    matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(this.group.matrixWorld);
    let view = this.views.get(camera);
    if (!view) { view = { camera, shadow, matrix: new THREE.Matrix4(), revision: -1, height: 0, batches: this.models.map(() => []) }; this.views.set(camera, view); }
    const time = performance.now() / 1000;
    for (const batches of view.batches) for (const batch of batches) if (batch) batch.clock.value = time;
    if (view.revision === this.revision && view.height === height && view.matrix.equals(matrix)) return;
    view.matrix.copy(matrix); view.height = height; view.revision = this.revision;
    frustum.setFromProjectionMatrix(matrix); localView.multiplyMatrices(camera.matrixWorldInverse, this.group.matrixWorld);
    const p = localView.elements, pixels = Math.abs(camera.projectionMatrix.elements[5]!) * height / 2, perspective = camera.projectionMatrix.elements[15] === 0;
    for (const batches of view.batches) for (const batch of batches) if (batch) { batch.queued.length = 0; batch.box.makeEmpty(); }
    this.models.forEach((model, m) => {
      if (shadow && !model.castShadow) return;
      const levels = this.levels[m]!, batches = view!.batches[m]!;
      for (const chunk of this.chunks.values()) {
        if (!frustum.intersectsBox(chunk.box)) continue;
        const distance = -(p[2]! * chunk.center.x + p[6]! * chunk.center.y + p[10]! * chunk.center.z + p[14]!);
        const radius = Math.abs(p[2]!) * chunk.extent.x + Math.abs(p[6]!) * chunk.extent.y + Math.abs(p[10]!) * chunk.extent.z;
        const minMetric = pixels * chunk.minScale / (perspective ? Math.max(0.001, distance + radius) : 1), maxMetric = pixels * chunk.maxScale / (perspective ? Math.max(0.001, distance - radius) : 1);
        for (let l = 0; l < levels.length; l++) {
          if (levels[l]!.error * minMetric >= 1.05 || (levels[l+1]?.error ?? Infinity) * maxMetric <= 0.65) continue;
          const batch = batches[l] ??= this.createBatch(model, levels[l]!, levels[l+1], view!, 64);
          batch.box.union(chunk.box); batch.queued.push(chunk);
        }
      }
      for (let l = 0; l < batches.length; l++) {
        let batch = batches[l]; if (!batch) continue;
        const count = batch.queued.reduce((sum, chunk) => sum + chunk.matrices.length / 16, 0);
        if (batch.capacity < count) {
          const queued = batch.queued, box = batch.box; batch.mesh.removeFromParent(); this.retired.push(batch);
          batch = batches[l] = this.createBatch(model, levels[l]!, levels[l+1], view!, 2 ** Math.ceil(Math.log2(count))); batch.queued = queued; batch.box = box;
        }
        const mesh = batch.mesh; let first = Infinity, last = 0, offset = 0, previousOffset = 0;
        batch.queued.forEach((chunk, i) => {
          const count = chunk.matrices.length / 16;
          if (batch!.selected[i] !== chunk || offset !== previousOffset) {
            first = Math.min(first, offset); last = offset + count;
            mesh.instanceMatrix.array.set(chunk.matrices, offset * 16);
            if (chunk.colors) mesh.instanceColor!.array.set(chunk.colors, offset * 3);
            else mesh.instanceColor!.array.fill(1, offset * 3, (offset + count) * 3);
            batch!.born.array.fill(shadow ? 0 : chunk.born, offset, offset + count);
          }
          offset += count; previousOffset += (batch!.selected[i]?.matrices.length ?? 0) / 16;
        });
        batch.box.getBoundingSphere(mesh.boundingSphere!); mesh.matrixWorld.copy(this.group.matrixWorld);
        if (first < Infinity || batch.count !== count) {
          if (shadow) this.removed = ++this.revision;
          for (const attribute of [mesh.instanceMatrix, mesh.instanceColor!, batch.born]) if (first < Infinity) { if (attribute !== mesh.instanceColor) attribute.addUpdateRange(first * attribute.itemSize, (last-first) * attribute.itemSize); attribute.needsUpdate = true; }
        }
        batch.count = mesh.count = count;
        [batch.selected, batch.queued] = [batch.queued, batch.selected];
        batch.projection.value = pixels; batch.clock.value = time;
      }
    });
    this.batches.set(String(camera.id), view.batches.flat().filter(Boolean).map(batch => batch.mesh));
    view.revision = this.revision;
  }

  scatter(matrices: Float32Array, colors?: Float32Array): void {
    const bounds = new THREE.Box3(), point = new THREE.Vector3(), chunks = new Map<string, { matrices: number[]; colors: number[] }>();
    for (let i = 0; i < matrices.length; i += 16) bounds.expandByPoint(point.fromArray(matrices, i + 12));
    bounds.getSize(point);
    const size = Math.max(point.x, point.z, 1) / Math.max(1, Math.sqrt(matrices.length / 16 / 64));
    for (let i = 0; i < matrices.length; i += 16) {
      const key = `${Math.floor(matrices[i + 12]! / size)}:${Math.floor(matrices[i + 14]! / size)}`;
      let chunk = chunks.get(key);
      if (!chunk) { chunk = { matrices: [], colors: [] }; chunks.set(key, chunk); }
      chunk.matrices.push(...matrices.subarray(i, i + 16));
      if (colors) chunk.colors.push(...colors.subarray(i / 16 * 3, i / 16 * 3 + 3));
    }
    for (const key of this.chunks.keys()) if (!chunks.has(String(key))) this.delete(key);
    for (const [key, chunk] of chunks) this.set(key, Float32Array.from(chunk.matrices), colors ? Float32Array.from(chunk.colors) : undefined);
  }

  set(key: number | string, matrices: Float32Array, colors?: Float32Array): void {
    if (!matrices.length) { this.delete(key); return; }
    const previous = this.chunks.get(key);
    if (previous?.matrices.length === matrices.length && previous.matrices.every((v,i) => v === matrices[i]) && previous.colors?.length === colors?.length && (!colors || previous.colors!.every((v,i) => v === colors[i]))) return;
    const chunk: Chunk = { matrices: matrices.slice(), colors: colors?.slice(), born: previous?.born ?? performance.now() / 1000, box: new THREE.Box3(), center: new THREE.Vector3(), extent: new THREE.Vector3(), minScale: Infinity, maxScale: 0 };
    const box = new THREE.Box3(), origin = new THREE.Vector3();
    for (let offset = 0; offset < matrices.length; offset += 16) {
      matrix.fromArray(matrices, offset); const scale = matrix.getMaxScaleOnAxis();
      chunk.box.union(box.copy(this.bounds).applyMatrix4(matrix)).expandByPoint(origin.setFromMatrixPosition(matrix));
      chunk.minScale = Math.min(chunk.minScale, scale); chunk.maxScale = Math.max(chunk.maxScale, scale);
    }
    chunk.box.getCenter(chunk.center); chunk.box.getSize(chunk.extent).multiplyScalar(0.5);
    this.chunks.set(key, chunk); this.revision++; if (previous) this.removed = this.revision;
  }

  delete(key: number | string): void { if (this.chunks.delete(key)) this.removed = ++this.revision; }
  private dropBatch(batch: Batch): void { batch.mesh.removeFromParent(); (batch.mesh.material as THREE.Material).dispose(); batch.mesh.dispose(); }
  dispose(): void { for (const batch of this.retired.splice(0)) this.dropBatch(batch); for (const view of this.views.values()) for (const batch of view.batches.flat()) if (batch) this.dropBatch(batch); this.views.clear(); this.chunks.clear(); this.batches.clear(); sources.delete(this.group); this.group.removeFromParent(); }
}

export function prepareInstances(scene: THREE.Scene, camera: THREE.Camera, target: THREE.RenderTarget | null): void {
  const active: SpatialInstances[] = [];
  let shadow = false;
  scene.traverseVisible(object => { const source = sources.get(object); if (source) active.push(source); if ((object instanceof THREE.DirectionalLight || object instanceof THREE.SpotLight) && object.shadow.camera === camera) shadow = true; });
  for (const source of active) source.prepare(camera, target!.height, shadow);
}
