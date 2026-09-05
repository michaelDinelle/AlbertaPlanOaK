import * as THREE from 'three';
import type { WorldBuilder } from '@kit/world';
import { SpatialInstances, type InstanceModel } from '@kit/instances';
import { jbwCollectible, JBW_BLOCK, JBW_TREE, JBW_WALL } from '@sim/envs/jellybeanworld';
import { cellKey, TILE_CELLS, visitBlock } from './fieldmath';
import type { CollectionLog } from './collectionlog';
import { itemModels } from './actors';
import { biomeAt } from './ambience';
import type { ViewCoverage } from '@kit/view';

interface Tile { x: number; y: number; cells: number[]; matrices: Map<number, Float32Array>; colors: Float32Array; stamp: number }

export class ItemField {
  canopyHeight = 0;
  readonly staticInstances: SpatialInstances[] = [];
  private readonly models;
  private readonly batches = new Map<number, SpatialInstances>();
  private readonly dirty = new Set<number>();
  private readonly tiles = new Map<number, Tile>();
  private readonly postings = new Map<number, number[]>();
  private readonly frustum = new THREE.Frustum();
  private readonly projection = new THREE.Matrix4();
  private readonly sphere = new THREE.Sphere(new THREE.Vector3(), TILE_CELLS * 1.6);
  private readonly dummy = new THREE.Object3D();
  private pending: { x: number; y: number; distance: number }[] = [];
  private version = -1;
  private logged = 0;
  private stamp = '';

  constructor(private readonly world: WorldBuilder, private readonly seed: number, private readonly height: (x: number, y: number) => number) {
    this.models = itemModels(world);
    for (const [kind, model] of this.models) {
      model.geometry.computeBoundingBox();
      this.canopyHeight = Math.max(this.canopyHeight, model.geometry.boundingBox!.max.y * 1.18 * (kind === JBW_TREE ? 1.6 : 1) + 0.16);
      if (kind >= 8) continue;
      const layers: InstanceModel[] = [{ ...model, castShadow: true, receiveShadow: true, padding: 0.5 }];
      const companion = this.models.get(kind + 8);
      if (companion) layers.push({ ...companion, padding: 0.5 });
      const batch = world.own(new SpatialInstances(layers));
      this.batches.set(kind, batch); world.add(batch.group);
      if (kind === JBW_TREE || kind === JBW_WALL) this.staticInstances.push(batch);
    }
  }

  get materials(): THREE.Material[] { return [...this.models.values()].map(model => model.material); }

  private populate(tile: Tile, step: number, log: CollectionLog): void {
    const times = this.postings.get(cellKey(tile.x, tile.y)) ?? [];
    let lo = 0, hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (times[mid]! <= step) lo = mid + 1; else hi = mid; }
    if (tile.stamp === lo) return;
    tile.stamp = lo; this.dirty.add(cellKey(tile.x, tile.y));
    const groups = new Map<number, number[]>();
    for (let i = 0; i < tile.cells.length; i += 3) {
      const x = tile.cells[i]!, y = tile.cells[i + 1]!, kind = tile.cells[i + 2]!;
      if (jbwCollectible(kind) && log.depleted(x, y, step)) continue;
      const variation = Math.sin(x * 127.1 + y * 311.7) * 0.5 + 0.5;
      this.dummy.position.set(x, this.height(x, y), y);
      this.dummy.rotation.set(0, kind === JBW_WALL ? 0 : variation * Math.PI * 2, 0);
      if (jbwCollectible(kind)) this.dummy.rotation.y += x * 0.71 + y * 0.37;
      this.dummy.scale.setScalar(kind === JBW_TREE ? 0.82 + variation * 0.36 : 1);
      if (kind === JBW_TREE) this.dummy.scale.multiply(new THREE.Vector3(1.4, 1.6, 1.4));
      this.dummy.updateMatrix();
      if (!groups.has(kind)) groups.set(kind, []);
      groups.get(kind)!.push(...this.dummy.matrix.elements);
    }
    tile.matrices.clear();
    for (const [kind, matrices] of groups) tile.matrices.set(kind, Float32Array.from(matrices));
    const trees = tile.matrices.get(JBW_TREE), tint = new THREE.Color(), green = new THREE.Color(0xf4e6ff), amber = new THREE.Color(0xffffff), cool = new THREE.Color(0xd6e9ff);
    tile.colors = new Float32Array((trees?.length ?? 0) / 16 * 3);
    for (let i = 0; trees && i < trees.length; i += 16) {
      const [autumn, woodland] = biomeAt(trees[i + 12]!, trees[i + 14]!, this.seed);
      tint.copy(green).lerp(amber, autumn).lerp(cool, woodland).toArray(tile.colors, i / 16 * 3);
    }
  }

  update(coverage: ViewCoverage, camera: THREE.PerspectiveCamera, step: number, log: CollectionLog): void {
    if (this.version !== log.version) {
      this.version = log.version; this.logged = 0; this.postings.clear();
      for (const tile of this.tiles.values()) tile.stamp = -1;
    }
    for (let i = this.logged; i < log.log.count; i++) {
      const key = cellKey(Math.floor(log.log.xs[i]! / TILE_CELLS), Math.floor(log.log.ys[i]! / TILE_CELLS));
      if (!this.postings.has(key)) this.postings.set(key, []);
      this.postings.get(key)!.push(log.log.steps[i]!);
    }
    this.logged = log.log.count;
    const cx = Math.floor(coverage.x / TILE_CELLS), cy = Math.floor(coverage.z / TILE_CELLS);
    const radius = Math.ceil(Math.min(2048, Math.max(1024, camera.position.distanceTo(this.dummy.position.set(coverage.x, this.height(coverage.x, coverage.z), coverage.z)) * 2.5)) / TILE_CELLS);
    this.projection.copy(camera.projectionMatrix); this.projection.elements[0]! /= 1.18; this.projection.elements[5]! /= 1.18;
    this.frustum.setFromProjectionMatrix(this.projection.multiply(camera.matrixWorldInverse));
    const direction = camera.getWorldDirection(this.dummy.position);
    const stamp = `${cx}:${cy}:${radius}:${Math.round(Math.atan2(direction.x, direction.z) * 20)}:${Math.round(Math.asin(direction.y) * 20)}:${Math.round(camera.position.x / 8)}:${Math.round(camera.position.z / 8)}:${Math.round(camera.position.y / 8)}`;
    if (stamp !== this.stamp) {
      this.stamp = stamp; this.pending = [];
      for (const [key, tile] of this.tiles) if (Math.abs(tile.x - cx) > radius + 2 || Math.abs(tile.y - cy) > radius + 2) { this.tiles.delete(key); this.dirty.add(cellKey(tile.x, tile.y)); }
      for (let y = cy - radius; y <= cy + radius; y++) for (let x = cx - radius; x <= cx + radius; x++) {
        this.sphere.center.set((x + 0.5) * TILE_CELLS, 3, (y + 0.5) * TILE_CELLS); this.sphere.radius = TILE_CELLS * 1.6;
        if (!this.tiles.has(cellKey(x, y)) && this.frustum.intersectsSphere(this.sphere)) this.pending.push({ x, y, distance: this.sphere.center.distanceToSquared(camera.position) });
      }
      this.pending.sort((a, b) => b.distance - a.distance);
    }
    const start = performance.now();
    while (this.pending.length && performance.now() - start < 1.5) {
      const next = this.pending.pop()!, cells: number[] = [];
      for (let by = next.y * TILE_CELLS / JBW_BLOCK; by < (next.y + 1) * TILE_CELLS / JBW_BLOCK; by++) for (let bx = next.x * TILE_CELLS / JBW_BLOCK; bx < (next.x + 1) * TILE_CELLS / JBW_BLOCK; bx++) visitBlock(this.seed, bx, by, (x, y, kind) => cells.push(x, y, kind));
      const tile: Tile = { ...next, cells, matrices: new Map(), colors: new Float32Array(), stamp: -1 };
      this.tiles.set(cellKey(next.x, next.y), tile);
    }
    for (const tile of this.tiles.values()) this.populate(tile, step, log);
    for (const key of this.dirty) {
      const tile = this.tiles.get(key);
      for (const [kind, batch] of this.batches) {
        const matrices = tile?.matrices.get(kind);
        if (matrices) batch.set(key, matrices, kind === JBW_TREE ? tile!.colors : undefined);
        else batch.delete(key);
      }
    }

    this.dirty.clear();
  }
}
