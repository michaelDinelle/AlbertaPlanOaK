import * as THREE from 'three/webgpu';
import { abs, asin, atan, clamp, color, exp, max, mix, pow, sin, smoothstep, uniform, vec3 } from 'three/tsl';
import type { SkyHandle } from '@kit/sky';

export interface AuroraHandle {
  setGain(gain: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildAurora(sky: SkyHandle, tint: THREE.ColorRepresentation, azimuth = 0): AuroraHandle {
  const gain = uniform(1);
  const base = color(new THREE.Color(tint));
  sky.add((dir, time) => {
    const longitude = atan(dir.z, dir.x).add(azimuth);
    const elevation = asin(clamp(dir.y, -1, 1));
    let aurora: THREE.Node<'vec3'> = vec3(0);
    for (let band = 0; band < 3; band++) {
      const fold = sin(longitude.mul(2).add(band * 2.1).add(time.mul(0.019)));
      const edge = fold.mul(0.057).add(0.072 + band * 0.048).add(sin(longitude.mul(5).sub(time.mul(0.013))).mul(0.018));
      const above = elevation.sub(edge);
      const hem = exp(abs(above).mul(-115));
      const veil = smoothstep(-0.015, 0.025, above).mul(exp(max(above, 0).mul(-28)));
      const rays = sin(longitude.mul(82).add(fold.mul(9)).add(time.mul(0.11))).mul(0.25).add(0.58).add(sin(longitude.mul(157).sub(time.mul(0.06))).mul(0.17));
      const reach = pow(sin(longitude.mul(2).add(band * 0.9)).mul(0.5).add(0.5), 1.6);
      const shade = mix(base, vec3(0.64, 0.3, 0.61), smoothstep(0.03, 0.13, above));
      aurora = aurora.add(shade.mul(hem.mul(0.28).add(veil.mul(rays).mul(0.5))).mul(reach));
    }
    return aurora.mul(gain).mul(smoothstep(0.008, 0.028, elevation));
  });
  return {
    setGain(value) { gain.value = value; },
    setTime(t) { sky.uniforms.uTime.value = t; },
    dispose() { gain.value = 0; },
  };
}
