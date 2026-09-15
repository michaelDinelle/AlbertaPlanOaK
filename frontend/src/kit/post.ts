import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneSize } from './types';
import { TOKENS } from './tokens';
import type { FogRecipe } from '../scenes/table';
import { prepareInstances } from './instances';
export const MSAA_SAMPLES = 4;

const BLOOM_THRESHOLD = 0.85;

export interface ScenePost {
  bloom: number;
  fog: FogRecipe;
}

function resolveFog(recipe: FogRecipe): number {
  const color = new THREE.Color(TOKENS.palettes[recipe.from[0]][recipe.from[1]]);
  for (const [token, k] of recipe.blend ?? []) {
    color.lerp(new THREE.Color(TOKENS.palettes[token[0]][token[1]]), k);
  }
  return color.getHex();
}

export interface ComposerHandle {
  setBloomPulse(t01: number): void;
  render(deltaTime: number): void;
  setGlass(rects: readonly GlassRect[]): void;
  resize(width: number, height: number, dpr: number): void;
  dispose(): void;
}

export const BLOOM_RESOLUTION_SCALE = 0.5;

export function pulsedBloomStrength(base: number, t01: number): number {
  const t = t01 < 0 ? 0 : t01 > 1 ? 1 : t01;
  return base * (1 + 1.2 * t);
}

export interface GlassRect {
  x: number;
  y: number;
  width: number;
  height: number;
  radius: number;
}

export const GLASS_RECTS = 16;

const GLASS_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uResolution: { value: new THREE.Vector2() },
    uCount: { value: 0 },
    uRects: { value: Array.from({ length: GLASS_RECTS }, () => new THREE.Vector4()) },
    uRadii: { value: new Float32Array(GLASS_RECTS) },
  },
  vertexShader: `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
uniform sampler2D tDiffuse;
uniform vec2 uResolution;
uniform int uCount;
uniform vec4 uRects[${GLASS_RECTS}];
uniform float uRadii[${GLASS_RECTS}];
varying vec2 vUv;
const float SCALE = -112.0;
const float CHROMA = 6.0;
const float BORDER = 0.07;
const float MAP_BLUR = 12.0;
const float BLUR = 3.0;
const float SATURATE = 1.5;
float coverage(vec2 p, vec4 rect, float radius) {
  vec2 half_ = rect.zw * 0.5;
  float inset = BORDER * min(rect.z, rect.w);
  float r = max(radius - inset, 2.0);
  vec2 q = abs(p - rect.xy - half_) - (half_ - inset - r);
  float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
  return 0.93 * (1.0 - smoothstep(-MAP_BLUR, MAP_BLUR, sd));
}
vec3 blurred(vec2 px) {
  vec2 t = 1.0 / uResolution;
  vec2 uv = px * t;
  return (texture2D(tDiffuse, uv + vec2(BLUR, 0.0) * t).rgb + texture2D(tDiffuse, uv - vec2(BLUR, 0.0) * t).rgb
        + texture2D(tDiffuse, uv + vec2(0.0, BLUR) * t).rgb + texture2D(tDiffuse, uv - vec2(0.0, BLUR) * t).rgb) * 0.25;
}
void main() {
  vec2 px = vUv * uResolution;
  vec4 rect = vec4(0.0);
  float radius = 0.0;
  float best = 1e30;
  for (int i = 0; i < ${GLASS_RECTS}; i++) {
    if (i >= uCount) break;
    vec4 r = uRects[i];
    if (px.x < r.x || px.y < r.y || px.x > r.x + r.z || px.y > r.y + r.w) continue;
    float area = r.z * r.w;
    if (area < best) { best = area; rect = r; radius = uRadii[i]; }
  }
  if (best == 1e30) { gl_FragColor = texture2D(tDiffuse, vUv); return; }
  vec2 centred = (px - rect.xy) / rect.zw - 0.5;
  float open = 1.0 - coverage(px, rect, radius);
  vec3 col;
  col.r = blurred(px + SCALE * open * centred).r;
  col.g = blurred(px + (SCALE + CHROMA) * open * centred).g;
  col.b = blurred(px + (SCALE + 2.0 * CHROMA) * open * centred).b;
  float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
  gl_FragColor = vec4(mix(vec3(luma), col, SATURATE), 1.0);
}`,
};

export function buildComposer(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  post: ScenePost,
  size: SceneSize,
): ComposerHandle {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = TOKENS.exposure;

  const width = Math.max(1, Math.floor(size.width));
  const height = Math.max(1, Math.floor(size.height));
  const dpr = size.dpr > 0 ? size.dpr : 1;
  const bloomStrength = post.bloom;

  const target = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    samples: MSAA_SAMPLES,
    depthBuffer: true,
    stencilBuffer: false,
  });
  target.texture.name = 'oakComposer.rt';

  const composer = new EffectComposer(renderer, target);
  composer.setPixelRatio(dpr);
  composer.setSize(width, height);

  const renderPass = new RenderPass(scene, camera);
  const beforeRender = scene.onBeforeRender;
  scene.onBeforeRender = (...args) => { beforeRender.apply(scene, args); prepareInstances(scene, renderer, args[2]); };
  composer.addPass(renderPass);

  const bloom = new UnrealBloomPass(
    new THREE.Vector2(
      Math.max(1, Math.round(width * dpr * BLOOM_RESOLUTION_SCALE)),
      Math.max(1, Math.round(height * dpr * BLOOM_RESOLUTION_SCALE)),
    ),
    bloomStrength,
    0.6,
    BLOOM_THRESHOLD,
  );
  composer.addPass(bloom);

  const outputPass = new OutputPass();
  outputPass.uniforms.uTint = { value: new THREE.Color(resolveFog(post.fog)) };
  outputPass.material.fragmentShader = outputPass.material.fragmentShader
    .replace('uniform sampler2D tDiffuse;', 'uniform sampler2D tDiffuse; uniform vec3 uTint;')
    .replace('gl_FragColor = texture2D( tDiffuse, vUv );', `
vec3 col = max(texture2D(tDiffuse, vUv).rgb, 0.0);
col += vec3(0.004) * (1.0 - smoothstep(0.0, 0.35, col));
col = mix(col, col * uTint, 0.14);
col *= 1.0 - 0.34 * smoothstep(0.34, 0.86, distance(vUv, vec2(0.5)));
gl_FragColor = vec4(max(col, 0.0), 1.0);
`);
  composer.addPass(outputPass);

  const glass = new ShaderPass(GLASS_SHADER);
  glass.uniforms.uResolution!.value.set(width * dpr, height * dpr);
  composer.addPass(glass);

  const resizeBloom = (w: number, h: number, ratio: number) => {
    bloom.setSize(
      Math.max(1, Math.round(w * ratio * BLOOM_RESOLUTION_SCALE)),
      Math.max(1, Math.round(h * ratio * BLOOM_RESOLUTION_SCALE)),
    );
  };
  resizeBloom(width, height, dpr);

  return {
    setBloomPulse(t01: number) {
      bloom.strength = pulsedBloomStrength(bloomStrength, t01);
    },
    render(deltaTime: number) {
      composer.render(deltaTime);
    },
    setGlass(rects) {
      const count = Math.min(GLASS_RECTS, rects.length);
      glass.uniforms.uCount!.value = count;
      for (let i = 0; i < count; i++) {
        const r = rects[i]!;
        (glass.uniforms.uRects!.value as THREE.Vector4[])[i]!.set(r.x, r.y, r.width, r.height);
        (glass.uniforms.uRadii!.value as Float32Array)[i] = r.radius;
      }
    },
    resize(w: number, h: number, ratio: number) {
      const rw = Math.max(1, Math.floor(w));
      const rh = Math.max(1, Math.floor(h));
      composer.setPixelRatio(ratio);
      composer.setSize(rw, rh);
      resizeBloom(rw, rh, ratio);
      glass.uniforms.uResolution!.value.set(rw * ratio, rh * ratio);
    },
    dispose() {
      composer.removePass(renderPass);
      composer.removePass(bloom);
      composer.removePass(outputPass);
      composer.removePass(glass);
      renderPass.dispose(); scene.onBeforeRender = beforeRender;
      bloom.dispose();
      outputPass.dispose();
      glass.dispose();
      composer.dispose();
      target.dispose();
    },
  };
}
