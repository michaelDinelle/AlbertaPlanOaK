import * as THREE from 'three/webgpu';
import { makeAmbientParticles } from '@kit/particles';
import type { WorldBuilder } from '@kit/world';

export function winterWeather(world: WorldBuilder, seed: number) {
  const snow = world.own(world.time(makeAmbientParticles({ preset: 'snow', seed, size: 1.1, opacity: 0.72, twinkle: 0, bounds: { halfX: 36, halfY: 14, halfZ: 36 }, name: 'driftingSnow' })));
  world.add(snow.points);
  return { setView(x: number, y: number, z: number) { snow.points.position.set(x, y + 8, z); } };
}
