
import * as THREE from 'three';
import type { GridCell } from '@sim/types';
import { HopMotion, cellToWorldVec, type GridSize, type HopMotionOptions } from './grid';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity, wispMaterial } from './materials';
import type { EmissiveMaterial, OakMaterial, WispMaterial } from './materials';
import { radialSprite } from './textures';
import { TOKENS } from './tokens';
import type { QualityTier } from './types';

export type FaceDir = number | { dx: number; dy: number };

export function yawFromGridDelta(dx: number, dy: number): number {
  if (dx === 0 && dy === 0) return 0;
  return Math.atan2(dx, dy);
}

export interface WispOptions {
  color: THREE.ColorRepresentation;
  tier?: QualityTier;
  height?: number;
  radius?: number;
  skirt?: boolean;
  ribbon?: boolean;
  ribbonLength?: number;
  silhouette?: boolean;
  contactShade?: boolean;
  hop?: HopMotionOptions;
  name?: string;
}

const RIBBON_MIN = 6;

export const AGENT_RENDER_ORDER = 3;

const SHADE_BLEND = 0.93;
const SILHOUETTE_SIZE = 1.4;
const SHADE_SIZE = 5.6;
const SHADE_LIFT = 0.014;
const SILHOUETTE_OPACITY = 0.8;
const SHADE_OPACITY = 0.5;
const SHADE_FALLOFF = 1;

export class Wisp {
  readonly group = new THREE.Group();
  readonly avatar = new THREE.Group();
  readonly core: THREE.Mesh;
  readonly motion: HopMotion;

  private readonly body = new THREE.Group();
  private readonly decor = new THREE.Group();
  private readonly skirt: THREE.Mesh | null;
  private readonly carryAnchor = new THREE.Group();
  private readonly ribbon: THREE.Line | null;
  private readonly ribbonPositions: Float32Array | null;
  private readonly ribbonAttr: THREE.BufferAttribute | null;
  private readonly ribbonCount: number;
  private ribbonFilled = false;

  private readonly coreMaterial: WispMaterial;
  private readonly skirtMaterial: OakMaterial | null;
  private readonly ribbonMaterial: THREE.LineBasicMaterial | null;
  private readonly haloMaterial: EmissiveMaterial;
  private readonly halo: THREE.Mesh;
  private readonly geometries: THREE.BufferGeometry[] = [];

  private readonly silhouette: THREE.Mesh | null;
  private readonly silhouetteMaterial: EmissiveMaterial | null;
  private readonly shade: THREE.Mesh | null;
  private readonly shadeMaterial: EmissiveMaterial | null;
  private readonly shadeSprite: THREE.Texture | null;
  private readonly shadeSize: number;

  private readonly height: number;
  private time = 0;
  private targetYaw = 0;
  private yaw = 0;
  private brightness = 1;
  private targetBrightness = 1;
  private flareAmount = 0;
  private carried: THREE.Object3D | null = null;
  private readonly baseIntensity: number;
  private shadeLift = 1;

  constructor(opts: WispOptions, grid: GridSize = { w: 1, h: 1 }) {
    const tier = opts.tier;
    const hero = tier !== 'thumb';
    const detail = tier === 'thumb' ? 0.35 : tier === 'medium' ? 0.65 : 1;
    this.height = opts.height ?? TOKENS.wispHeight;
    const radius = opts.radius ?? 0.16;
    this.group.name = opts.name ?? 'oakWisp';
    this.motion = new HopMotion(grid, opts.hop);

    const coreGeo = new THREE.IcosahedronGeometry(radius, tier === 'thumb' ? 1 : 2);
    const haloGeo = new THREE.IcosahedronGeometry(radius * 2.4, tier === 'thumb' ? 0 : 1);
    this.geometries.push(coreGeo, haloGeo);

    this.coreMaterial = wispMaterial(opts.color, { intensity: 1.9, rimPower: 2.1, tier, name: 'oakWispCore' });
    this.baseIntensity = 0.7;
    this.haloMaterial = emissiveMaterial(opts.color, this.baseIntensity, {
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'oakWispHalo',
    });

    this.decor.name = 'oakWispDecor';
    this.core = new THREE.Mesh(coreGeo, this.coreMaterial);
    this.core.name = 'oakWispCore';
    this.core.renderOrder = AGENT_RENDER_ORDER;
    this.halo = new THREE.Mesh(haloGeo, this.haloMaterial);
    this.halo.name = 'oakWispHalo';
    this.halo.renderOrder = AGENT_RENDER_ORDER;
    this.body.add(this.core, this.decor);
    this.decor.add(this.halo);

    const wantsSilhouette = opts.silhouette ?? true;
    const wantsShade = opts.contactShade ?? true;
    this.shadeSprite = wantsShade ? radialSprite(tier === 'thumb' ? 32 : 64, SHADE_FALLOFF) : null;
    const shadeColor = new THREE.Color(opts.color).lerp(new THREE.Color(TOKENS.palettes.night.skyTop), SHADE_BLEND);
    this.shadeSize = radius * SHADE_SIZE;

    if (wantsSilhouette) {
      this.silhouetteMaterial = emissiveMaterial(shadeColor, 1, {
        transparent: true,
        opacity: SILHOUETTE_OPACITY,
        depthWrite: false,
        name: 'oakWispSilhouette',
      });
      this.silhouette = new THREE.Mesh(coreGeo, this.silhouetteMaterial);
      this.silhouette.name = 'oakWispSilhouette';
      this.silhouette.scale.setScalar(SILHOUETTE_SIZE);
      this.silhouette.renderOrder = -1;
      this.body.add(this.silhouette);
    } else {
      this.silhouette = null;
      this.silhouetteMaterial = null;
    }

    if (wantsShade && this.shadeSprite) {
      const geo = new THREE.PlaneGeometry(1, 1);
      geo.rotateX(-Math.PI / 2);
      geo.name = 'oakWispShade';
      this.geometries.push(geo);
      this.shadeMaterial = emissiveMaterial(shadeColor, 1, {
        transparent: true,
        opacity: SHADE_OPACITY,
        depthWrite: false,
        map: this.shadeSprite,
        name: 'oakWispShade',
      });
      this.shade = new THREE.Mesh(geo, this.shadeMaterial);
      this.shade.name = 'oakWispShade';
      this.shade.renderOrder = -1;
      this.shade.position.y = SHADE_LIFT;
      this.shade.scale.setScalar(this.shadeSize);
      this.group.add(this.shade);
    } else {
      this.shade = null;
      this.shadeMaterial = null;
    }

    const wantsSkirt = (opts.skirt ?? true) && hero;
    if (wantsSkirt) {
      const petals = Math.max(3, Math.round(6 * detail));
      const skirtGeo = buildSkirt(petals, radius);
      this.geometries.push(skirtGeo);
      this.skirtMaterial = oakMaterial({
        color: opts.color,
        roughness: 0.7,
        flatShading: true,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthWrite: false,
        vertexColors: true,
        toonSteps: 2,
        rimColor: opts.color,
        rimStrength: 0.5,
        tier,
        name: 'oakWispSkirt',
      });
      this.skirt = new THREE.Mesh(skirtGeo, this.skirtMaterial);
      this.skirt.name = 'oakWispSkirt';
      this.skirt.renderOrder = AGENT_RENDER_ORDER;
      this.skirt.position.y = -radius * 0.7;
      this.decor.add(this.skirt);
    } else {
      this.skirt = null;
      this.skirtMaterial = null;
    }

    this.carryAnchor.name = 'oakWispCarry';
    this.carryAnchor.position.set(0, radius * 0.2, radius * 1.5);
    this.decor.add(this.carryAnchor);

    this.body.position.y = this.height;
    this.avatar.name = 'oakWispAvatar';
    this.avatar.add(this.body);
    this.group.add(this.avatar);

    const wantsRibbon = (opts.ribbon ?? true) && hero;
    if (wantsRibbon) {
      this.ribbonCount = Math.max(RIBBON_MIN, Math.round((opts.ribbonLength ?? 24) * detail));
      this.ribbonPositions = new Float32Array(this.ribbonCount * 3);
      const geo = new THREE.BufferGeometry();
      this.ribbonAttr = new THREE.BufferAttribute(this.ribbonPositions, 3);
      this.ribbonAttr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('position', this.ribbonAttr);
      geo.setDrawRange(0, this.ribbonCount);
      this.geometries.push(geo);
      this.ribbonMaterial = new THREE.LineBasicMaterial({
        color: new THREE.Color(opts.color),
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        fog: false,
      });
      this.ribbonMaterial.name = 'oakWispRibbon';
      this.ribbon = new THREE.Line(geo, this.ribbonMaterial);
      this.ribbon.name = 'oakWispRibbon';
      this.ribbon.renderOrder = AGENT_RENDER_ORDER;
      this.ribbon.frustumCulled = false;
      this.group.add(this.ribbon);
    } else {
      this.ribbon = null;
      this.ribbonPositions = null;
      this.ribbonAttr = null;
      this.ribbonMaterial = null;
      this.ribbonCount = 0;
    }

    this.avatar.position.copy(this.motion.position);
    this.resetRibbon();
    this.syncShade();
  }

  get position(): THREE.Vector3 {
    return this.avatar.position;
  }

  get cellX(): number {
    return this.motion.cellX;
  }

  get cellY(): number {
    return this.motion.cellY;
  }

  get probeObject(): THREE.Object3D {
    return this.core;
  }

  setCell(x: number, y: number, grid: GridSize): void {
    this.motion.setGrid(grid);
    this.motion.snap(x, y);
    this.avatar.position.copy(this.motion.position);
    this.resetRibbon();
    this.syncShade();
  }

  hopTo(cell: GridCell | { x: number; y: number }, now?: number): void {
    this.motion.hopTo(cell.x, cell.y, now ?? this.time);
    this.avatar.position.copy(this.motion.position);
    this.syncShade();
  }

  face(dir: FaceDir): void {
    this.targetYaw = typeof dir === 'number' ? dir : yawFromGridDelta(dir.dx, dir.dy);
  }

  carry(token: THREE.Object3D | null): THREE.Object3D | null {
    const previous = this.carried;
    if (previous) {
      this.carryAnchor.remove(previous);
      previous.position.set(0, 0, 0);
    }
    this.carried = token;
    if (token) {
      token.position.set(0, 0, 0);
      this.carryAnchor.add(token);
    }
    return previous;
  }

  get carrying(): THREE.Object3D | null {
    return this.carried;
  }

  dim(amount = 0.45): void {
    this.targetBrightness = Math.max(0, amount);
  }

  undim(): void {
    this.targetBrightness = 1;
  }

  flare(strength = 1): void {
    this.flareAmount = Math.max(this.flareAmount, strength);
  }

  snap(): void {
    this.motion.snap();
    this.avatar.position.copy(this.motion.position);
    this.yaw = this.targetYaw;
    this.body.rotation.y = this.yaw;
    this.brightness = this.targetBrightness;
    this.flareAmount = 0;
    this.resetRibbon();
    this.syncShade();
    this.applyBrightness();
  }

  update(dt: number, time: number): void {
    this.time = time;
    this.motion.update(time);
    this.avatar.position.copy(this.motion.position);

    const k = 1 - Math.exp(-12 * Math.max(0, dt));
    let d = this.targetYaw - this.yaw;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    this.yaw += d * k;
    this.body.rotation.y = this.yaw;

    this.decor.position.y = Math.sin(time * 1.9) * 0.018;
    const breath = 1 + Math.sin(time * 1.9) * 0.03;
    this.core.scale.setScalar(breath);
    this.silhouette?.scale.setScalar(SILHOUETTE_SIZE * breath);
    if (this.skirt) {
      this.skirt.rotation.y = -time * 0.9;
      this.skirt.scale.setScalar(1 + Math.sin(time * 2.4) * 0.05);
    }
    this.carryAnchor.rotation.y = time * 1.2;

    if (this.flareAmount > 0) this.flareAmount = Math.max(0, this.flareAmount - dt / 0.5);
    this.brightness += (this.targetBrightness - this.brightness) * (1 - Math.exp(-6 * Math.max(0, dt)));
    this.applyBrightness();
    this.coreMaterial.oak.setTime(time);
    this.syncShade();
    this.pushRibbon();
  }

  dispose(): void {
    this.carry(null);
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.coreMaterial.dispose();
    this.haloMaterial.dispose();
    this.skirtMaterial?.dispose();
    this.ribbonMaterial?.dispose();
    this.silhouetteMaterial?.dispose();
    this.shadeMaterial?.dispose();
    this.shadeSprite?.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }


  private applyBrightness(): void {
    const gain = this.brightness * (1 + 1.8 * this.flareAmount);
    this.coreMaterial.oak.setRim(1.9 * gain);
    this.coreMaterial.pulse.value = this.flareAmount;
    setEmissiveIntensity(this.haloMaterial, this.baseIntensity * gain);
    if (this.ribbonMaterial) this.ribbonMaterial.opacity = 0.4 * this.brightness;
    if (this.skirtMaterial) this.skirtMaterial.opacity = 0.85 * this.brightness;
    if (this.silhouetteMaterial) this.silhouetteMaterial.opacity = SILHOUETTE_OPACITY * this.brightness;
    if (this.shadeMaterial) this.shadeMaterial.opacity = SHADE_OPACITY * this.brightness * this.shadeLift;
  }

  private syncShade(): void {
    const shade = this.shade;
    if (!shade) return;
    const lift = Math.max(0, this.avatar.position.y);
    const k = 1 / (1 + 2.6 * lift);
    shade.position.x = this.avatar.position.x;
    shade.position.z = this.avatar.position.z;
    shade.scale.setScalar(this.shadeSize * k);
    this.shadeLift = k;
    if (this.shadeMaterial) this.shadeMaterial.opacity = SHADE_OPACITY * this.brightness * k;
  }

  private resetRibbon(): void {
    const p = this.ribbonPositions;
    if (!p || !this.ribbonAttr) return;
    const x = this.avatar.position.x;
    const y = this.avatar.position.y + this.height;
    const z = this.avatar.position.z;
    for (let i = 0; i < this.ribbonCount; i++) {
      p[i * 3] = x;
      p[i * 3 + 1] = y;
      p[i * 3 + 2] = z;
    }
    this.ribbonAttr.needsUpdate = true;
    this.ribbonFilled = true;
  }

  private pushRibbon(): void {
    const p = this.ribbonPositions;
    if (!p || !this.ribbonAttr) return;
    if (!this.ribbonFilled) this.resetRibbon();
    p.copyWithin(3, 0, (this.ribbonCount - 1) * 3);
    p[0] = this.avatar.position.x;
    p[1] = this.avatar.position.y + this.height;
    p[2] = this.avatar.position.z;
    this.ribbonAttr.needsUpdate = true;
  }
}

function buildSkirt(petals: number, radius: number): THREE.BufferGeometry {
  const positions = new Float32Array(petals * 3 * 3);
  const colors = new Float32Array(petals * 3 * 3);
  const inner = radius * 0.55;
  const outer = radius * 1.9;
  for (let i = 0; i < petals; i++) {
    const a0 = (i / petals) * Math.PI * 2;
    const a1 = ((i + 0.62) / petals) * Math.PI * 2;
    const mid = (a0 + a1) * 0.5;
    const o = i * 9;
    positions[o] = Math.cos(a0) * inner;
    positions[o + 1] = 0;
    positions[o + 2] = Math.sin(a0) * inner;
    positions[o + 3] = Math.cos(a1) * inner;
    positions[o + 4] = 0;
    positions[o + 5] = Math.sin(a1) * inner;
    positions[o + 6] = Math.cos(mid) * outer;
    positions[o + 7] = -radius * 0.5;
    positions[o + 8] = Math.sin(mid) * outer;
    for (let v = 0; v < 3; v++) {
      const k = v === 2 ? 0.35 : 1;
      colors[o + v * 3] = k;
      colors[o + v * 3 + 1] = k;
      colors[o + v * 3 + 2] = k;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.name = 'oakWispSkirt';
  return g;
}

export function placeAtCell(object: THREE.Object3D, x: number, y: number, grid: GridSize, height = 0): void {
  cellToWorldVec(x, y, grid, object.position);
  object.position.y = height;
}
