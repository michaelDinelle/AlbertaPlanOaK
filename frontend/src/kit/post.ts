
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneSize } from './types';
import { TOKENS } from './tokens';

export interface PostConfig {
  bloomStrength: number;
  bloomThreshold: number;
  fogTint: number;
}

export interface GradeUniforms {
  tDiffuse: { value: THREE.Texture | null };
  uLift: { value: THREE.Vector3 };
  uTint: { value: THREE.Color };
  uTintStrength: { value: number };
  uGamma: { value: number };
  uVignette: { value: number };
  uGrain: { value: number };
  uResolution: { value: THREE.Vector2 };
}

export interface ComposerHandle {
  setBloomPulse(t01: number): void;
  render(deltaTime?: number): void;
  resize(width: number, height: number, dpr?: number): void;
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
uniform float uGrain;
uniform vec2 uResolution;
varying vec2 vUv;

float oakBayer2( vec2 a ) {
	a = floor( a );
	return fract( a.x * 0.5 + a.y * a.y * 0.75 );
}
float oakBayer8( vec2 a ) {
	return oakBayer2( 0.25 * a ) * 0.0625 + oakBayer2( 0.5 * a ) * 0.25 + oakBayer2( a );
}

void main() {
	vec4 texel = texture2D( tDiffuse, vUv );
	vec3 col = max( texel.rgb, 0.0 );

	// lift the shadows only — highlights and HDR bloom stay untouched
	col += uLift * ( 1.0 - smoothstep( 0.0, 0.35, col ) );

	// token tint (the scene's fogTint) pulls the whole frame into one palette
	col = mix( col, col * uTint, uTintStrength );

	col = pow( col, vec3( uGamma ) );

	float d = distance( vUv, vec2( 0.5 ) );
	col *= 1.0 - uVignette * smoothstep( 0.34, 0.86, d );

	col += ( oakBayer8( gl_FragCoord.xy ) - 0.5 ) * uGrain;

	gl_FragColor = vec4( max( col, 0.0 ), texel.a );
}
`;

export const OakGradeShader = {
  name: 'OakGradeShader',
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uLift: { value: new THREE.Vector3(0.012, 0.012, 0.012) },
    uTint: { value: new THREE.Color(0xffffff) },
    uTintStrength: { value: 0.14 },
    uGamma: { value: 0.97 },
    uVignette: { value: 0.34 },
    uGrain: { value: 0.02 },
    uResolution: { value: new THREE.Vector2(1, 1) },
  },
  vertexShader: GRADE_VERT,
  fragmentShader: GRADE_FRAG,
};


export function buildComposer(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  config: PostConfig,
  size: SceneSize,
): ComposerHandle {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = TOKENS.exposure;

  const width = Math.max(1, Math.floor(size.width));
  const height = Math.max(1, Math.floor(size.height));
  const dpr = size.dpr > 0 ? size.dpr : 1;
  const bloomStrength = config.bloomStrength;

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

  const scale = BLOOM_RESOLUTION_SCALE;
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(Math.max(1, Math.round(width * dpr * scale)), Math.max(1, Math.round(height * dpr * scale))),
    bloomStrength,
    0.6,
    config.bloomThreshold,
  );
  composer.addPass(bloom);

  const gradePass = new ShaderPass(OakGradeShader);
  const gradeUniforms = gradePass.uniforms as unknown as GradeUniforms;
  gradeUniforms.uLift.value.setScalar(0.004);
  gradeUniforms.uTint.value.set(config.fogTint);
  gradeUniforms.uTintStrength.value = 0.14;
  gradeUniforms.uGamma.value = 1.0;
  gradeUniforms.uVignette.value = 0.34;
  gradeUniforms.uGrain.value = 0.02;
  gradeUniforms.uResolution.value.set(width * dpr, height * dpr);
  composer.addPass(gradePass);

  const outputPass = new OutputPass();
  composer.addPass(outputPass);

  const resizeBloom = (w: number, h: number, ratio: number) => {
    bloom.setSize(Math.max(1, Math.round(w * ratio * scale)), Math.max(1, Math.round(h * ratio * scale)));
  };
  resizeBloom(width, height, dpr);

  let currentDpr = dpr;

  return {
    setBloomPulse(t01: number) {
      bloom.strength = pulsedBloomStrength(bloomStrength, t01);
    },
    render(deltaTime?: number) {
      composer.render(deltaTime);
    },
    resize(w: number, h: number, ratio?: number) {
      const rw = Math.max(1, Math.floor(w));
      const rh = Math.max(1, Math.floor(h));
      currentDpr = ratio !== undefined && ratio > 0 ? ratio : currentDpr;
      composer.setPixelRatio(currentDpr);
      composer.setSize(rw, rh);
      resizeBloom(rw, rh, currentDpr);
      gradeUniforms.uResolution.value.set(rw * currentDpr, rh * currentDpr);
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
