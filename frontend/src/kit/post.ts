import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
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
  resize(width: number, height: number, dpr: number): void;
  dispose(): void;
}

export const BLOOM_RESOLUTION_SCALE = 0.5;

export function pulsedBloomStrength(base: number, t01: number): number {
  const t = t01 < 0 ? 0 : t01 > 1 ? 1 : t01;
  return base * (1 + 1.2 * t);
}

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
    resize(w: number, h: number, ratio: number) {
      const rw = Math.max(1, Math.floor(w));
      const rh = Math.max(1, Math.floor(h));
      composer.setPixelRatio(ratio);
      composer.setSize(rw, rh);
      resizeBloom(rw, rh, ratio);
    },
    dispose() {
      composer.removePass(renderPass);
      composer.removePass(bloom);
      composer.removePass(outputPass);
      renderPass.dispose(); scene.onBeforeRender = beforeRender;
      bloom.dispose();
      outputPass.dispose();
      composer.dispose();
      target.dispose();
    },
  };
}
