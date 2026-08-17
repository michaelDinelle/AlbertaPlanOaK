import * as THREE from 'three';
import { TOKENS } from './tokens';

export const TOKEN_WHITE = TOKENS.palettes.noon.sun;
export const TOKEN_BLACK = TOKENS.palettes.night.skyTop;

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function clampRange(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

export function blendTokens(a: THREE.ColorRepresentation, b: THREE.ColorRepresentation, k: number): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}
