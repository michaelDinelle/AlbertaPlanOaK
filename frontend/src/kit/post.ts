
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { QualityTier, SceneSize } from './types';
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
  composer: EffectComposer | null;
  bloom: UnrealBloomPass | null;
  gradeUniforms: GradeUniforms | null;
  bloomStrength: number;
  setBloomPulse(t01: number): void;
  render(deltaTime?: number): void;
  resize(width: number, height: number, dpr?: number): void;
  dispose(): void;
}


export function bloomResolutionScale(tier: QualityTier): number {
  return tier === 'hero' ? 0.5 : tier === 'medium' ? 0.25 : 0;
}

export function msaaSamples(tier: QualityTier): number {
  return tier === 'hero' ? 4 : 0;
}

export function usesPost(tier: QualityTier): boolean {
  return tier !== 'thumb';
}

export function pulsedBloomStrength(base: number, t01: number): number {
  const t = t01 < 0 ? 0 : t01 > 1 ? 1 : t01;
  return base * (1 + 1.2 * t);
}

export interface GradeSettings {
  lift: number;
  gamma: number;
  tint: number;
  tintStrength: number;
  vignette: number;
  grain: number;
}

export function gradeDefaults(config: PostConfig, tier: QualityTier): GradeSettings {
  const hero = tier === 'hero';
  return {
    lift: hero ? 0.004 : 0.003,
    gamma: 1.0,
    tint: config.fogTint,
    tintStrength: 0.14,
    vignette: hero ? 0.34 : 0.26,
    grain: hero ? 0.02 : 0.012,
  };
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
  tier: QualityTier,
  size: SceneSize,
): ComposerHandle {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = TOKENS.exposure;

  const width = Math.max(1, Math.floor(size.width));
  const height = Math.max(1, Math.floor(size.height));
  const dpr = size.dpr > 0 ? size.dpr : 1;
  const bloomStrength = config.bloomStrength;

  if (!usesPost(tier)) {
    return {
      composer: null,
      bloom: null,
      gradeUniforms: null,
      bloomStrength,
      setBloomPulse() {
      },
      render() {
        renderer.render(scene, camera);
      },
      resize() {
      },
      dispose() {
      },
    };
  }

  const samples = msaaSamples(tier);
  const target = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    samples,
    depthBuffer: true,
    stencilBuffer: false,
  });
  target.texture.name = 'oakComposer.rt';

  const composer = new EffectComposer(renderer, target);
  composer.setPixelRatio(dpr);
  composer.setSize(width, height);

  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const scale = bloomResolutionScale(tier);
  let bloom: UnrealBloomPass | null = null;
  if (scale > 0) {
    bloom = new UnrealBloomPass(
      new THREE.Vector2(Math.max(1, Math.round(width * dpr * scale)), Math.max(1, Math.round(height * dpr * scale))),
      bloomStrength,
      0.6,
      config.bloomThreshold,
    );
    composer.addPass(bloom);
  }

  const gradePass = new ShaderPass(OakGradeShader);
  const gradeUniforms = gradePass.uniforms as unknown as GradeUniforms;
  const g = gradeDefaults(config, tier);
  gradeUniforms.uLift.value.setScalar(g.lift);
  gradeUniforms.uTint.value.set(g.tint);
  gradeUniforms.uTintStrength.value = g.tintStrength;
  gradeUniforms.uGamma.value = g.gamma;
  gradeUniforms.uVignette.value = g.vignette;
  gradeUniforms.uGrain.value = g.grain;
  gradeUniforms.uResolution.value.set(width * dpr, height * dpr);
  composer.addPass(gradePass);

  const outputPass = new OutputPass();
  composer.addPass(outputPass);

  const resizeBloom = (w: number, h: number, ratio: number) => {
    if (!bloom) return;
    bloom.setSize(Math.max(1, Math.round(w * ratio * scale)), Math.max(1, Math.round(h * ratio * scale)));
  };
  resizeBloom(width, height, dpr);

  let currentDpr = dpr;

  return {
    composer,
    bloom,
    gradeUniforms,
    bloomStrength,
    setBloomPulse(t01: number) {
      if (bloom) bloom.strength = pulsedBloomStrength(bloomStrength, t01);
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
      if (bloom) composer.removePass(bloom);
      composer.removePass(gradePass);
      composer.removePass(outputPass);
      renderPass.dispose();
      bloom?.dispose();
      gradePass.dispose();
      outputPass.dispose();
      composer.dispose();
      target.dispose();
    },
  };
}
