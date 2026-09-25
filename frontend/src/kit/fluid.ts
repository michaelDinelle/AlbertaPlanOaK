import * as THREE from 'three/webgpu';
import { abs, clamp, dot, exp, float, max, min, screenCoordinate, screenUV, select, sin, smoothstep, texture, textureLoad, uniform, vec2, vec4 } from 'three/tsl';
import { flowWater, type WaterHandle } from './water';

type N<T extends string> = THREE.Node<T>;

export function simulateWater(renderer: THREE.WebGPURenderer, water: WaterHandle, bedHeight: (x: number, z: number) => number, bounds: THREE.Vector4, flow = new THREE.Vector2()) {
  const resolution = 128, gravity = 9.81;
  const dx = bounds.z / resolution, dz = bounds.w / resolution;
  const data = new Float32Array(resolution * resolution * 4);
  let maximumDepth = 0;
  for (let z = 0; z < resolution; z++) for (let x = 0; x < resolution; x++) {
    const px = bounds.x + (x + 0.5) * dx, pz = bounds.y + (z + 0.5) * dz;
    const depth = Math.max(0, water.mesh.position.y - bedHeight(px, pz));
    maximumDepth = Math.max(maximumDepth, depth);
    const index = (x + z * resolution) * 4;
    const height = depth > 0 ? (Math.sin(px * 0.7 + pz * 0.28) + Math.sin(px * 0.32 - pz * 0.5)) * 0.009 * Math.min(depth, 1) : 0;
    data.set([height, height, depth], index);
  }
  const initial = new THREE.DataTexture(data, resolution, resolution, THREE.RGBAFormat, THREE.FloatType);
  initial.needsUpdate = true;
  const dt = Math.min(1 / 60, 0.45 / Math.sqrt(gravity * maximumDepth * (1 / dx ** 2 + 1 / dz ** 2)));
  const step = uniform(dt), clock = uniform(0), cell = uniform(new THREE.Vector2(dx, dz)), area = uniform(bounds), current = vec2(flow.x, flow.y);
  const target = () => new THREE.RenderTarget(resolution, resolution, { type: THREE.HalfFloatType, depthBuffer: false });
  let read = target(), write = target();
  const surface = target(), state = texture(read.texture, screenUV, float(0)), texel = 1 / resolution;

  const advected = (uv: N<'vec2'>) => {
    const moved = state.sample(uv.sub(current.mul(step).div(area.zw)));
    return vec4(moved.r, moved.g, state.sample(uv).b, moved.a);
  };
  const c = advected(screenUV), l = advected(screenUV.sub(vec2(texel, 0))), r = advected(screenUV.add(vec2(texel, 0)));
  const b = advected(screenUV.sub(vec2(0, texel))), t = advected(screenUV.add(vec2(0, texel)));
  const flux = min(r.b, c.b).mul(r.r.sub(c.r)).add(min(l.b, c.b).mul(l.r.sub(c.r))).div(cell.x.mul(cell.x))
    .add(min(t.b, c.b).mul(t.r.sub(c.r)).add(min(b.b, c.b).mul(b.r.sub(c.r))).div(cell.y.mul(cell.y)));
  const p = area.xy.add(screenUV.mul(area.zw));
  const wind = sin(dot(p, vec2(0.65, 0.23)).sub(clock.mul(2))).add(sin(dot(p, vec2(-0.29, 0.51)).sub(clock.mul(1.6))).mul(0.5));
  const damping = exp(step.negate().mul(float(0.15).div(c.b.add(0.2)).add(0.09)));
  const next = c.r.add(c.r.sub(c.g).mul(damping)).add(flux.mul(gravity).add(wind.mul(0.09).mul(smoothstep(0, 0.8, c.b))).mul(step).mul(step));
  const foam = max(c.a.mul(exp(step.negate().mul(0.9))), abs(next.sub(c.r)).div(step).mul(smoothstep(0.1, 1, c.b).oneMinus()));
  const slope = vec2(
    state.sample(screenUV.add(vec2(texel, 0))).r.sub(state.sample(screenUV.sub(vec2(texel, 0))).r),
    state.sample(screenUV.add(vec2(0, texel))).r.sub(state.sample(screenUV.sub(vec2(0, texel))).r),
  ).div(cell.mul(2));

  const copy = Object.assign(new THREE.NodeMaterial(), { fragmentNode: textureLoad(initial, screenCoordinate) });
  const simulate = Object.assign(new THREE.NodeMaterial(), { fragmentNode: select(c.b.lessThanEqual(0), vec4(0), vec4(next, c.r, c.b, foam)) });
  const normals = Object.assign(new THREE.NodeMaterial(), { fragmentNode: vec4(state.r, slope, state.a) });
  const quad = new THREE.QuadMesh();
  const pass = (material: THREE.NodeMaterial, output: THREE.RenderTarget) => {
    const previous = renderer.getRenderTarget();
    state.value = read.texture;
    quad.material = material;
    renderer.setRenderTarget(output);
    quad.render(renderer);
    renderer.setRenderTarget(previous);
  };
  pass(copy, read);
  pass(normals, surface);
  flowWater(water, { sample: (at) => texture(surface.texture, clamp(at.sub(area.xy).div(area.zw), 0.5 / resolution, 1 - 0.5 / resolution), float(0)), flow: current });

  let previous: number | null = null, accumulated = 0, elapsed = 0;
  return {
    setTime(time: number) {
      accumulated += previous === null ? 0 : Math.min(Math.max(time - previous, 0), 0.1);
      previous = time;
      const changed = accumulated >= dt;
      while (accumulated >= dt) {
        elapsed += dt;
        clock.value = elapsed;
        pass(simulate, write);
        [read, write] = [write, read];
        accumulated -= dt;
      }
      if (changed) pass(normals, surface);
    },
    dispose() {
      for (const item of [copy, simulate, normals, initial, read, write, surface]) item.dispose();
    },
  };
}
