import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneSize } from './types';
import { TOKENS } from './tokens';
import type { FogRecipe } from '../scenes/table';

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

export interface GradeUniforms {
  tDiffuse: { value: THREE.Texture | null };
  uLift: { value: THREE.Vector3 };
  uTint: { value: THREE.Color };
  uTintStrength: { value: number };
  uGamma: { value: number };
  uVignette: { value: number };
}

export interface ComposerHandle {
  setBloomPulse(t01: number): void;
  render(deltaTime: number): void;
  resize(width: number, height: number, dpr: number): void;
  dispose(): void;
}

export const BLOOM_RESOLUTION_SCALE = 0.5;
export const MSAA_SAMPLES = 4;

export function pulsedBloomStrength(base: number, t01: number): number {
  const t = t01 < 0 ? 0 : t01 > 1 ? 1 : t01;
  return base * (1 + 1.2 * t);
}

const GRADE_VERT =  `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const GRADE_FRAG =  `
uniform sampler2D tDiffuse;
uniform vec3 uLift;
uniform vec3 uTint;
uniform float uTintStrength;
uniform float uGamma;
uniform float uVignette;
varying vec2 vUv;

void main() {
	vec4 texel = texture2D( tDiffuse, vUv );
	vec3 col = max( texel.rgb, 0.0 );

	col += uLift * ( 1.0 - smoothstep( 0.0, 0.35, col ) );

	col = mix( col, col * uTint, uTintStrength );

	col = pow( col, vec3( uGamma ) );

	float d = distance( vUv, vec2( 0.5 ) );
	col *= 1.0 - uVignette * smoothstep( 0.34, 0.86, d );

	gl_FragColor = vec4( max( col, 0.0 ), texel.a );
}
`;

export const OakGradeShader = {
  name: 'OakGradeShader',
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uLift: { value: new THREE.Vector3(0.004, 0.004, 0.004) },
    uTint: { value: new THREE.Color(0xffffff) },
    uTintStrength: { value: 0.14 },
    uGamma: { value: 1.0 },
    uVignette: { value: 0.34 },
  },
  vertexShader: GRADE_VERT,
  fragmentShader: GRADE_FRAG,
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

  const gradePass = new ShaderPass(OakGradeShader);
  const gradeUniforms = gradePass.uniforms as unknown as GradeUniforms;
  gradeUniforms.uTint.value.set(resolveFog(post.fog));
  composer.addPass(gradePass);

  const outputPass = new OutputPass();
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
      composer.removePass(gradePass);
      composer.removePass(outputPass);
      renderPass.dispose();
      bloom.dispose();
      gradePass.dispose();
      outputPass.dispose();
      composer.dispose();
      target.dispose();
    },
  };
}
