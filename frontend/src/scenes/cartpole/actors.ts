import * as THREE from 'three';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { blendTokens, clampRange } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import type { PropMaterials } from '@kit/props/materials';
import { iceMaterial } from '@kit/props/ice';
import { engravedPanel, modelParts } from '@kit/robot';
import type { Palette } from '@kit/tokens';
import { makeWorld, type WorldCore } from '@kit/world';
import { RAIL_Y } from './world';

const WHEEL_R = 0.075;
const BODY_W = 0.64;
const BODY_H = 0.2;
const BODY_Y = RAIL_Y + WHEEL_R + 0.02;

export interface CartOptions {
  palette: Palette;
  accent: THREE.ColorRepresentation;
  materials: PropMaterials;
  poleLength: number;
  seed: number;
  slip: boolean;
}

export function buildCart(opts: CartOptions) {
  const { palette, accent, materials, poleLength } = opts;
  const world = makeWorld('brassCart'), root = world.group, parts = modelParts(world), { box, disc, rod } = parts;
  const brass = materials.brass, armor = world.own(oakMaterial({ color: 0xe8dcc0, vertexColors: true, roughness: 0.55, metalness: 0.12, rimColor: palette.sun, rimStrength: 0.55, toonSteps: 3, name: 'cartArmor' }));
  const iron = world.own(oakMaterial({ color: 0x5c6470, roughness: 0.6, metalness: 0.35, rimColor: palette.sun, rimStrength: 0.4, toonSteps: 3, name: 'cartIron' }));
  const body = new THREE.Group();
  root.add(body);
  box(body, brass, 0, BODY_Y, 0, BODY_W, BODY_H, 0.3, 0.03);
  box(body, iron, 0, BODY_Y - BODY_H / 2 - 0.01, 0, BODY_W * 0.8, 0.03, 0.34, 0.008);
  parts.add(body, engravedPanel(0.26, 0.14, 0.03, 0.015, 0.14).translate(0, BODY_Y, 0.165), armor);
  parts.add(body, engravedPanel(0.26, 0.14, 0.03, 0.015, 0.14).rotateY(Math.PI).translate(0, BODY_Y, -0.165), armor);
  for (const side of [-1, 1]) {
    box(body, iron, side * (BODY_W / 2 + 0.02), BODY_Y, 0, 0.05, BODY_H * 0.7, 0.2, 0.01);
    for (const z of [-0.12, 0.12]) rod(body, iron, new THREE.Vector3(side * 0.2, BODY_Y - BODY_H / 2, z), new THREE.Vector3(side * 0.2, RAIL_Y + WHEEL_R, z), 0.014);
  }
  disc(body, brass, 0, BODY_Y + BODY_H / 2 + 0.01, 0, 0.07, 0.03, 16);
  parts.build();
  const wheelGeometry = world.own(new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.036, 18).rotateX(Math.PI / 2));
  const wheelMaterial = opts.slip ? iceMaterial(world, 0xd8eefc, 'cartWheelIce') : iron;
  const wheels = [-0.2, 0.2].flatMap(x => [-0.13, 0.13].map(z => {
    const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheel.position.set(x, RAIL_Y + WHEEL_R, z);
    wheel.castShadow = true;
    root.add(wheel);
    return wheel;
  }));
  const thrust = [-1, 1].map(side => {
    const material = emissiveMaterial(blendTokens(accent, palette.sun, 0.3), 0, { transparent: true, opacity: 0.85, name: 'cartThrust' });
    world.own(material);
    const glow = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.045, 10, 8).scale(1.6, 1, 1)), material);
    glow.position.set(side * (BODY_W / 2 + 0.06), BODY_Y, 0);
    root.add(glow);
    return material;
  });
  const mast = new THREE.Group();
  mast.position.set(0, BODY_Y + BODY_H / 2 + 0.03, 0);
  root.add(mast);
  rod(mast, brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, poleLength, 0), 0.026);
  disc(mast, iron, 0, 0, 0, 0.05, 0.06, 12);
  parts.add(mast, new THREE.ConeGeometry(0.045, 0.09, 8).translate(0, poleLength + 0.04, 0), brass);
  rod(mast, brass, new THREE.Vector3(0, poleLength - 0.04, 0), new THREE.Vector3(0.46, poleLength - 0.04, 0), 0.011);
  parts.build();
  const clothMaterial = world.own(oakMaterial({ color: accent, emissive: accent, emissiveIntensity: 0, roughness: 1, side: THREE.DoubleSide, rimColor: palette.sun, rimStrength: 0.35, wind: { amplitude: 0.045, frequency: 2.6, height: 0.3 }, toonSteps: 3, name: 'pennantCloth' }));
  const cloth = new THREE.Mesh(world.own(new THREE.PlaneGeometry(0.44, 0.26, 5, 3).translate(0.22 + 0.03, poleLength - 0.18, 0)), clothMaterial);
  cloth.castShadow = true;
  mast.add(cloth);
  const skid = makeAmbientParticles({ preset: 'dust', count: 90, bounds: { halfX: 0.2, halfY: 0.05, halfZ: 0.14 }, seed: opts.seed ^ 0x3d91, color: opts.slip ? 0xd8f0ff : blendTokens(palette.groundB, palette.sun, 0.3), size: 0.09, intensity: 1.1, opacity: 0.6, rise: 0.08, swirl: 0.03, name: 'cartSkid' });
  skid.setDensity(0);
  skid.points.position.y = RAIL_Y + 0.02;
  root.add(skid.points);
  world.own(skid); world.time(skid); world.size(skid);
  const tipWorld = new THREE.Vector3();
  const handle = {
    tipWorld,
    setState(x: number, theta: number, force: number, upright: number, skidAmount: number) {
      root.position.x = x;
      mast.rotation.z = -theta;
      const roll = -x / WHEEL_R;
      for (const wheel of wheels) wheel.rotation.z = roll;
      thrust.forEach((material, i) => setEmissiveIntensity(material, force !== 0 && (i === 0) === (force > 0) ? 2.6 : 0));
      clothMaterial.emissiveIntensity = 0.75 * clampRange(upright, 0, 1);
      skid.setDensity(clampRange(skidAmount, 0, 1));
      tipWorld.set(x + Math.sin(theta) * poleLength, mast.position.y + Math.cos(theta) * poleLength, 0);
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
