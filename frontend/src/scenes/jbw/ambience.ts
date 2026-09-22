import * as THREE from 'three';
import { makeAmbientParticles } from '@kit/particles';
import { valueNoise2 } from '@kit/textures';
import type { WorldBuilder } from '@kit/world';

export function biomeAt(x: number, z: number, seed: number): [number, number] {
  return [THREE.MathUtils.smoothstep(valueNoise2(x / 150, z / 150, 8192, seed ^ 0x682f), 0.28, 0.72), THREE.MathUtils.smoothstep(valueNoise2(x / 390, z / 390, 8192, seed ^ 0xa179), 0.58, 0.82)];
}

export function winterWeather(world: WorldBuilder, seed: number) {
  const snow = world.own(world.size(world.time(makeAmbientParticles({ preset: 'snow', seed, size: 1.1, opacity: 0.72, bounds: { halfX: 36, halfY: 14, halfZ: 36 }, name: 'driftingSnow' }))));
  world.add(snow.points);
  snow.material.uniforms.uTwinkle!.value = 0;
  return { setView(x: number, y: number, z: number) { snow.points.position.set(x, y + 8, z); } };
}
