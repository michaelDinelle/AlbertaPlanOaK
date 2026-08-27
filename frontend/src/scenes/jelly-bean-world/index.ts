import * as THREE from 'three';
import { applyDebugLayer, type GroundTruthOptions } from '@kit/debug';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import type { OrbitRigOptions } from '@kit/rigs';
import { skyFromPalette } from '@kit/sky';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  JBW_BANANA,
  JBW_EMPTY,
  JBW_HEADING,
  JBW_JELLY,
  JBW_ONION,
  JBW_TREE,
  JBW_TRUFFLE,
  JBW_WALL,
  JBW_X,
  JBW_Y,
  jbwDepleted,
  jbwItemAt,
} from '@sim/envs/jellybeanworld';

const colors = [0x000000, 0xf5d44b, 0x9b63d4, 0xee776b, 0x4a86b8, 0x4f9a64, 0x7a4a2d];
const scents = [
  [0, 0, 0], [1.92, 1.76, 0.4], [0.68, 0.01, 0.99], [1.64, 0.54, 0.4],
  [0, 0, 0], [0, 0.47, 0.06], [8.4, 4.8, 2.6],
] as const;
const scentColors = [0xffba68, 0x8c6ed8, 0x73c98b];

interface JellyWorld extends WorldCore {
  rig: LightRig;
  items: THREE.InstancedMesh[];
  agent: THREE.Group;
  sensed: THREE.Group;
  scentLights: THREE.PointLight[];
  scentLabel: THREE.Sprite;
  negative: THREE.Mesh;
}

function geometry(item: number): THREE.BufferGeometry {
  if (item === JBW_BANANA) {
    const g = new THREE.TorusGeometry(0.19, 0.055, 5, 10, Math.PI * 1.35);
    g.rotateX(Math.PI / 2);
    g.rotateZ(-0.65);
    return g;
  }
  if (item === JBW_ONION) return new THREE.IcosahedronGeometry(0.18, 1);
  if (item === JBW_JELLY) return new THREE.SphereGeometry(0.19, 8, 5);
  if (item === JBW_WALL) return new THREE.BoxGeometry(0.94, 0.62, 0.94);
  if (item === JBW_TREE) return new THREE.ConeGeometry(0.3, 0.82, 7);
  return new THREE.DodecahedronGeometry(0.16, 0);
}

function itemHeight(item: number): number {
  if (item === JBW_WALL) return 0.31;
  if (item === JBW_TREE) return 0.41;
  return 0.18;
}

function scentAt(seed: number, x: number, y: number, pose: ArrayLike<number>): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (dx === 0 && dy === 0) continue;
      const item = jbwItemAt(seed, x + dx, y + dy);
      if (item === JBW_EMPTY || item === JBW_WALL || jbwDepleted(pose, dx, dy)) continue;
      const response = Math.exp(-0.42 * Math.hypot(dx, dy));
      const source = scents[item] ?? scents[0];
      out[0] += (source[0] ?? 0) * response;
      out[1] += (source[1] ?? 0) * response;
      out[2] += (source[2] ?? 0) * response;
    }
  }
  return out;
}

function labelSprite(): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 64;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.userData['canvas'] = canvas;
  sprite.scale.set(3.8, 0.48, 1);
  sprite.position.set(0, 1.25, -2.9);
  return sprite;
}

function setLabel(sprite: THREE.Sprite, values: readonly number[]): void {
  const canvas = sprite.userData['canvas'] as HTMLCanvasElement;
  const context = canvas.getContext('2d');
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = 'rgba(7, 12, 18, 0.72)';
  context.roundRect(20, 7, 472, 50, 18);
  context.fill();
  context.font = '500 22px ui-monospace, SFMono-Regular, monospace';
  context.textAlign = 'center';
  context.fillStyle = '#eaf4ef';
  context.fillText(values.map((v) => v.toFixed(3)).join('    '), 256, 40);
  const material = sprite.material as THREE.SpriteMaterial;
  if (material.map) material.map.needsUpdate = true;
}

function buildWorld(ctx: SceneContext, scene: THREE.Scene): JellyWorld {
  const palette = ctx.tokens.palettes.afternoon;
  const w = makeWorld('jellyBeanWorld');
  const sky = w.own(skyFromPalette(palette, 'afternoon', { radius: 180, sunSize: 0.028 }));
  w.add(sky.mesh);
  w.time(sky);
  const rig = w.own(makeLightRig(palette, { pointCount: 4, pointDistance: 7 }));
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 5.2);
  w.add(rig.group);
  applyFog(scene, palette, 28);

  const groundGeometry = w.own(new THREE.PlaneGeometry(7, 7));
  const groundMaterial = w.own(oakMaterial({
    color: 0x56684f, roughness: 0.94, flatShading: true, toonSteps: 3,
    rimColor: palette.skyHorizon, rimStrength: 0.12, name: 'jellyGround',
  }));
  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  w.add(ground);
  const lattice = new THREE.GridHelper(7, 7, 0x7f9273, 0x65765e);
  lattice.position.y = 0.006;
  const latticeMaterials = Array.isArray(lattice.material) ? lattice.material : [lattice.material];
  for (const material of latticeMaterials) {
    material.transparent = true;
    material.opacity = 0.24;
    w.own(material);
  }
  w.own(lattice.geometry);
  w.add(lattice);

  const items: THREE.InstancedMesh[] = [];
  for (let item = JBW_BANANA; item <= JBW_TRUFFLE; item++) {
    const g = w.own(geometry(item));
    const m = w.own(oakMaterial({
      color: colors[item], roughness: item === JBW_JELLY ? 0.58 : 0.86,
      flatShading: true, toonSteps: 3, rimColor: palette.skyHorizon, rimStrength: 0.22,
      name: `jellyRole${item}`,
    }));
    const mesh = new THREE.InstancedMesh(g, m, 49);
    mesh.name = `jellyRole${item}`;
    mesh.castShadow = true;
    mesh.receiveShadow = item === JBW_WALL;
    mesh.frustumCulled = false;
    mesh.count = 0;
    items[item] = mesh;
    w.add(mesh);
  }

  const agent = new THREE.Group();
  agent.name = 'jellyExplorer';
  const bodyGeometry = w.own(new THREE.OctahedronGeometry(0.24, 1));
  const bodyMaterial = w.own(oakMaterial({ color: 0xf3eee1, roughness: 0.7, flatShading: true }));
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.y = 0.29;
  body.castShadow = true;
  agent.add(body);
  const noseGeometry = w.own(new THREE.ConeGeometry(0.09, 0.28, 5));
  noseGeometry.rotateX(-Math.PI / 2);
  noseGeometry.translate(0, 0.28, -0.25);
  const nose = new THREE.Mesh(noseGeometry, bodyMaterial);
  nose.castShadow = true;
  agent.add(nose);
  w.add(agent);

  const sensed = new THREE.Group();
  sensed.name = 'jellySensedCells';
  const ringGeometry = w.own(new THREE.RingGeometry(0.36, 0.43, 24));
  const ringMaterial = w.own(new THREE.MeshBasicMaterial({
    color: ctx.tokens.uiCyan, transparent: true, opacity: 0.72, side: THREE.DoubleSide,
    depthWrite: false, toneMapped: false,
  }));
  for (let i = 0; i < 8; i++) {
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.025;
    sensed.add(ring);
  }
  applyDebugLayer(sensed);
  sensed.visible = false;
  w.add(sensed);

  const scentLights = rig.points.slice(0, 3);
  for (let i = 0; i < scentLights.length; i++) {
    const light = scentLights[i];
    if (!light) continue;
    light.color.setHex(scentColors[i] ?? 0xffffff);
    light.position.set((i - 1) * 1.25, 0.12, 0.6);
  }
  const scentLabel = labelSprite();
  applyDebugLayer(scentLabel);
  scentLabel.visible = false;
  w.add(scentLabel);

  const negativeGeometry = w.own(new THREE.RingGeometry(0.34, 0.48, 40));
  const negativeMaterial = w.own(new THREE.MeshBasicMaterial({
    color: 0x17202a, transparent: true, opacity: 0, side: THREE.DoubleSide,
    depthWrite: false, toneMapped: false,
  }));
  const negative = new THREE.Mesh(negativeGeometry, negativeMaterial);
  negative.rotation.x = -Math.PI / 2;
  negative.position.y = 0.018;
  negative.visible = false;
  w.add(negative);

  return w.seal<JellyWorld>({ rig, items, agent, sensed, scentLights, scentLabel, negative });
}

class JellyBeanWorldScene extends StageScene<JellyWorld> {
  protected readonly stageName = 'jellyBeanWorld';
  private seed = 0;
  private debug = false;
  private negativeAt = -Infinity;
  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly rotation = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3(1, 1, 1);

  protected buildWorld(ctx: SceneContext): JellyWorld {
    this.seed = ctx.seed;
    return buildWorld(ctx, this.scene);
  }
  protected buildActors(): void {}
  protected rigOptions(): OrbitRigOptions {
    return { target: [0, 0.2, 0], distance: 8.8, azimuth: 0.72, polar: 0.72, follow: 10 };
  }
  protected overlayOptions(): GroundTruthOptions { return {}; }
  protected start(_ctx: SceneContext, world: JellyWorld): void {
    world.setTime(0);
  }
  protected poolOptions() { return { burstSize: 36 }; }

  override setDebug(on: boolean): void {
    super.setDebug(on);
    this.debug = on;
    if (this.world) {
      this.world.sensed.visible = on;
      this.world.scentLabel.visible = on;
    }
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<JellyWorld>): void {
    const pose = frame.state.pose;
    const x = Math.trunc(pose[JBW_X] ?? 0);
    const y = Math.trunc(pose[JBW_Y] ?? 0);
    const counts = new Uint8Array(7);
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        if (dx === 0 && dy === 0) continue;
        const item = jbwItemAt(this.seed, x + dx, y + dy);
        if (item === JBW_EMPTY || (item !== JBW_WALL && item !== JBW_TREE && jbwDepleted(pose, dx, dy))) continue;
        const mesh = world.items[item];
        if (!mesh) continue;
        this.position.set(dx, itemHeight(item), dy);
        this.rotation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), ((x + dx) * 0.71 + (y + dy) * 1.13));
        this.matrix.compose(this.position, this.rotation, this.scale);
        mesh.setMatrixAt(counts[item] ?? 0, this.matrix);
        counts[item] = (counts[item] ?? 0) + 1;
      }
    }
    for (let item = JBW_BANANA; item <= JBW_TRUFFLE; item++) {
      const mesh = world.items[item];
      if (!mesh) continue;
      mesh.count = counts[item] ?? 0;
      mesh.instanceMatrix.needsUpdate = true;
    }

    world.agent.rotation.y = -(pose[JBW_HEADING] ?? 0) * Math.PI * 0.5;
    world.agent.position.y = 0.025 + Math.sin(time * 2.2) * 0.018;
    const scent = scentAt(this.seed, x, y, pose);
    for (let i = 0; i < world.scentLights.length; i++) {
      const light = world.scentLights[i];
      if (light) light.intensity = 0.08 * Math.log1p(scent[i] ?? 0);
    }
    if (this.debug) setLabel(world.scentLabel, scent);

    const sensed = world.sensed.children;
    let k = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      sensed[k++]?.position.set(dx, 0.025, dy);
    }

    if (frame.scrubbed) {
      pool.clear();
      this.negativeAt = -Infinity;
      world.negative.visible = false;
      rig.snap();
    }
    for (const event of frame.events) {
      if (event.type !== 'collection') continue;
      if (event.reward > 0) {
        pool.ripple(this.position.set(0, 0.02, 0), 0xffc46b, 0.72, time);
        pool.pulse.fire(0.72);
      } else if (event.reward < 0) {
        this.negativeAt = time;
        world.negative.visible = true;
      }
    }
    const negativeAge = time - this.negativeAt;
    if (negativeAge >= 0 && negativeAge < 0.8) {
      const fade = 1 - negativeAge / 0.8;
      world.negative.visible = true;
      world.negative.scale.setScalar(0.28 + 1.45 * fade);
      (world.negative.material as THREE.MeshBasicMaterial).opacity = 0.62 * fade;
    } else world.negative.visible = false;

    pool.update(frame.dtReal, time);
    world.setTime(time);
    rig.follow(0, 0.2, 0);
    rig.update(frame.dtReal);
  }
  protected drop(): void {}
}

const create: SceneFactory = () => new JellyBeanWorldScene();
export default create;
