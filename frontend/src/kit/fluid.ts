import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import type { WaterHandle } from './water';

export function simulateWater(renderer: THREE.WebGLRenderer, water: WaterHandle, bedHeight: (x: number, z: number) => number, bounds: THREE.Vector4, flow = new THREE.Vector2()) {
  const resolution = 256, gravity = 9.81;
  const dx = bounds.z / resolution, dz = bounds.w / resolution;
  const gpu = new GPUComputationRenderer(resolution, resolution, renderer);
  gpu.setDataType(THREE.HalfFloatType);
  const initial = gpu.createTexture(), data = initial.image.data!;
  let maximumDepth = 0;
  for (let z = 0; z < resolution; z++) for (let x = 0; x < resolution; x++) {
    const px = bounds.x + (x + 0.5) * dx, pz = bounds.y + (z + 0.5) * dz;
    const depth = Math.max(0, water.mesh.position.y - bedHeight(px, pz));
    maximumDepth = Math.max(maximumDepth, depth);
    const index = (x + z * resolution) * 4;
    data[index] = depth > 0 ? (Math.sin(px * 0.7 + pz * 0.28) + Math.sin(px * 0.32 - pz * 0.5)) * 0.009 * Math.min(depth, 1) : 0;
    data[index + 1] = data[index];
    data[index + 2] = depth;
    data[index + 3] = 0;
  }
  const dt = Math.min(1 / 60, 0.45 / Math.sqrt(gravity * maximumDepth * (1 / dx ** 2 + 1 / dz ** 2)));
  const uniforms = { uStep: { value: dt }, uTime: { value: 0 }, uCell: { value: new THREE.Vector2(dx, dz) }, uBounds: { value: bounds } };
  const current = `vec2(${flow.x.toFixed(6)}, ${flow.y.toFixed(6)})`;
  const state = gpu.addVariable('fluidState', `
uniform float uStep;
uniform float uTime;
uniform vec2 uCell;
uniform vec4 uBounds;
vec4 advectedState(vec2 uv) {
  vec4 state = texture2D(fluidState, uv - ${current} * uStep / uBounds.zw);
  state.b = texture2D(fluidState, uv).b;
  return state;
}
void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec2 texel = 1.0 / resolution.xy;
  vec4 c = advectedState(uv);
  if (c.b <= 0.0) { gl_FragColor = vec4(0.0); return; }
  vec4 l = advectedState(uv - vec2(texel.x, 0.0));
  vec4 r = advectedState(uv + vec2(texel.x, 0.0));
  vec4 b = advectedState(uv - vec2(0.0, texel.y));
  vec4 t = advectedState(uv + vec2(0.0, texel.y));
  float flux = (min(r.b, c.b) * (r.r - c.r) + min(l.b, c.b) * (l.r - c.r)) / (uCell.x * uCell.x)
             + (min(t.b, c.b) * (t.r - c.r) + min(b.b, c.b) * (b.r - c.r)) / (uCell.y * uCell.y);
  vec2 p = uBounds.xy + uv * uBounds.zw;
  float wind = sin(dot(p, vec2(0.65, 0.23)) - uTime * 2.0) + 0.5 * sin(dot(p, vec2(-0.29, 0.51)) - uTime * 1.6);
  float damping = exp(-uStep * (0.09 + 0.15 / (c.b + 0.2)));
  float next = c.r + (c.r - c.g) * damping + (${gravity.toFixed(2)} * flux + wind * 0.09 * smoothstep(0.0, 0.8, c.b)) * uStep * uStep;
  float foam = max(c.a * exp(-uStep * 0.9), abs(next - c.r) / uStep * (1.0 - smoothstep(0.1, 1.0, c.b)));
  gl_FragColor = vec4(next, c.r, c.b, foam);
}`, initial);
  Object.assign(state.material.uniforms, uniforms);
  state.minFilter = state.magFilter = THREE.LinearFilter;
  gpu.setVariableDependencies(state, [state]);
  const error = gpu.init();
  if (error) { gpu.dispose(); throw new Error(`Water simulation unavailable: ${error}`); }
  const surface = new THREE.WebGLRenderTarget(resolution, resolution, { type: THREE.HalfFloatType, depthBuffer: false });
  const input = { value: gpu.getCurrentRenderTarget(state).texture };
  const normals = gpu.createShaderMaterial(`
uniform sampler2D uState;
uniform vec2 uCell;
void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy, texel = 1.0 / resolution.xy;
  vec4 c = texture2D(uState, uv);
  vec2 slope = vec2(texture2D(uState, uv + vec2(texel.x, 0.0)).r - texture2D(uState, uv - vec2(texel.x, 0.0)).r,
                    texture2D(uState, uv + vec2(0.0, texel.y)).r - texture2D(uState, uv - vec2(0.0, texel.y)).r) / (2.0 * uCell);
  gl_FragColor = vec4(c.r, slope, c.a);
}`, { uState: input, uCell: uniforms.uCell });
  gpu.doRenderTarget(normals, surface);
  Object.assign(water.material.uniforms, { uFluid: { value: surface.texture }, uFluidBounds: { value: bounds } });
  const sample = `
uniform sampler2D uFluid;
uniform vec4 uFluidBounds;
vec4 fluidSample(vec2 p) {
  return texture2D(uFluid, clamp((p - uFluidBounds.xy) / uFluidBounds.zw, vec2(${0.5 / resolution}), vec2(${1 - 0.5 / resolution})));
}
`;
  water.material.vertexShader = sample + water.material.vertexShader
    .replace('oakWaveHeight( wp, uTime, uWaveScale, uSpeed ) * uAmplitude', 'fluidSample(wp).r');
  water.material.fragmentShader = sample + water.material.fragmentShader
    .replace('vec2 slope = oakWaveSlope(vWorld.xz, uTime, uWaveScale, uSpeed) * uAmplitude;', 'vec4 fluid = fluidSample(vWorld.xz); vec2 slope = fluid.gb;')
    .replace('oakWaveSlope(vWorld.xz, uTime, uWaveScale * 5.0', `oakWaveSlope(vWorld.xz - ${current} * uTime, uTime, uWaveScale * 5.0`)
    .replace('float alpha = coverage *', 'col += vec3(0.4, 0.48, 0.46) * fluid.a * 0.15;\nfloat alpha = coverage *');
  water.material.needsUpdate = true;
  let previous: number | null = null, accumulated = 0, elapsed = 0;
  return {
    setTime(time: number) {
      accumulated += previous === null ? 0 : Math.min(Math.max(time - previous, 0), 0.1);
      previous = time;
      const changed = accumulated >= dt;
      while (accumulated >= dt) {
        elapsed += dt;
        uniforms.uTime.value = elapsed;
        gpu.compute();
        accumulated -= dt;
      }
      if (changed) {
        input.value = gpu.getCurrentRenderTarget(state).texture;
        gpu.doRenderTarget(normals, surface);
      }
    },
    dispose() { normals.dispose(); surface.dispose(); gpu.dispose(); },
  };
}
