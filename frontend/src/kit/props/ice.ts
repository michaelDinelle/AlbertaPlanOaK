import type * as THREE from 'three';
import { philoxDraw } from '@sim/philox';
import { oakMaterial } from '../materials';
import type { WorldBuilder } from '../world';

export const ENV_RNG_STREAM = 3;

export const wordUnit = (word: number): number => (word >>> 8) * 2 ** -24;

export function slipDraw(seed: number, step: number, slip: number): [number, number, number, number] | null {
  if (!(slip > 0) || step <= 0) return null;
  const draw = philoxDraw(seed, ENV_RNG_STREAM, step);
  return wordUnit(draw[0]) < slip ? draw : null;
}

export function iceMaterial(world: WorldBuilder, tint: THREE.ColorRepresentation = 0xdff3ff, name = 'iceSheen') {
  return world.own(oakMaterial({ color: tint, roughness: 0.14, metalness: 0.08, rimColor: 0xcdf4ff, rimStrength: 0.9, rimPower: 2.2, toonSteps: 2, toonSoftness: 0.5, toonFloor: 0.4, name }));
}
