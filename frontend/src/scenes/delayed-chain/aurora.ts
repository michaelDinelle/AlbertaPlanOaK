
import * as THREE from 'three';
import { TOKENS, type Palette } from '@kit/tokens';
import { blendTokens } from './world';

export interface AuroraOptions {
  palette: Palette;
  tokens: typeof TOKENS;
  center: number;
  radius: number;
  name?: string;
}

export interface AuroraHandle {
  group: THREE.Group;
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  readonly nearestX: number;
  setGain(g: number): void;
  setTime(t: number): void;
  dispose(): void;
}

const ARC = 0.6;
const BASE_Y = 0.5;
const TOP_Y = 17;
const RIBBONS = 4;

const VERT =  `
varying vec2 vAur;
void main() {
	vAur = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const FRAG =  `
#include <common>
uniform float uTime;
uniform float uGain;
uniform float uIntensity;
uniform vec3 uBase;
uniform vec3 uCrown;
uniform vec3 uEmber;
varying vec2 vAur;

float aurHash( float p ) {
	p = fract( p * 0.1031 );
	p *= p + 33.33;
	p *= p + p;
	return fract( p );
}

float aurNoise( float x ) {
	float i = floor( x );
	float f = fract( x );
	f = f * f * ( 3.0 - 2.0 * f );
	return mix( aurHash( i ), aurHash( i + 1.0 ), f );
}

void main() {
	float u = vAur.x;
	float v = vAur.y;
	float t = uTime;

	float acc = 0.0;
	vec3 col = vec3( 0.0 );

	for ( int i = 0; i < ${RIBBONS}; i ++ ) {
		float fi = float( i );
		float phase = fi * 2.3994;
		// each curtain drifts on its own slow beat and leans with height
		float c = 0.5 + 0.29 * sin( t * 0.043 + phase ) + 0.11 * sin( t * 0.107 + phase * 2.1 );
		c += 0.055 * sin( v * 3.1 + t * 0.13 + phase ) + 0.028 * sin( v * 7.3 - t * 0.08 + phase * 1.7 );
		float w = 0.026 + 0.055 * v + 0.012 * sin( t * 0.061 + phase );
		float d = ( u - c ) / max( w, 0.001 );
		float band = exp( - d * d );
		// bright lower edge, dissolving upward; nothing at the very hem
		float vert = smoothstep( 1.0, 0.18, v ) * smoothstep( 0.0, 0.09, v );
		// vertical striations: the thing that makes a gradient read as a curtain
		float streak = 0.62 + 0.38 * aurNoise( u * 128.0 + fi * 21.0 + t * 0.04 );
		float a = band * vert * streak;
		col += mix( uBase, uCrown, clamp( v * 1.25, 0.0, 1.0 ) ) * a;
		col += uEmber * a * 0.22 * smoothstep( 0.3, 0.0, v );
		acc += a;
	}

	float edge = smoothstep( 0.0, 0.19, u ) * smoothstep( 1.0, 0.81, u );
	float alpha = clamp( acc * edge, 0.0, 1.0 );
	if ( alpha <= 0.003 ) discard;
	gl_FragColor = vec4( col * ( uIntensity * uGain * edge ), alpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

function auroraGeometry(segments: number, rows: number): THREE.BufferGeometry {
  const nu = segments + 1;
  const nv = rows + 1;
  const positions = new Float32Array(nu * nv * 3);
  const uvs = new Float32Array(nu * nv * 2);
  const indices: number[] = [];
  for (let j = 0; j < nv; j++) {
    const v = j / rows;
    const y = BASE_Y + (TOP_Y - BASE_Y) * v;
    for (let i = 0; i < nu; i++) {
      const u = i / segments;
      const a = (u * 2 - 1) * ARC;
      const o = (j * nu + i) * 3;
      positions[o] = Math.cos(a);
      positions[o + 1] = y;
      positions[o + 2] = Math.sin(a);
      const p = (j * nu + i) * 2;
      uvs[p] = u;
      uvs[p + 1] = v;
    }
  }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < segments; i++) {
      const a = j * nu + i;
      const b = a + 1;
      const c = a + nu;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  g.name = 'auroraCurtain';
  return g;
}

export function buildAurora(opts: AuroraOptions): AuroraHandle {
  const detail = 1;
  const segments = Math.max(16, Math.round(112 * detail));
  const rows = Math.max(2, Math.round(9 * detail));

  const geometry = auroraGeometry(segments, rows);
  const group = new THREE.Group();
  group.name = opts.name ?? 'aurora';
  group.position.set(opts.center, 0, 0);

  const palette = opts.palette;
  const teal = opts.tokens.accents['delayed-chain'] ?? opts.tokens.uiCyan;
  const base = blendTokens(teal, palette.hemiSky, 0.12);
  const crown = blendTokens(opts.tokens.cueMagenta, palette.skyHorizon, 0.46);
  const ember = blendTokens(opts.tokens.celebrationGold, palette.sun, 0.24);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uGain: { value: 1 },
      uIntensity: { value: 0.62 },
      uBase: { value: base },
      uCrown: { value: crown },
      uEmber: { value: ember },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  material.name = 'auroraMaterial';

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'auroraCurtain';
  mesh.scale.set(opts.radius, 1, opts.radius);
  mesh.frustumCulled = false;
  group.add(mesh);

  const nearestX = opts.center + opts.radius * Math.cos(ARC);

  return {
    group,
    mesh,
    material,
    nearestX,
    setGain(g) {
      (material.uniforms.uGain as { value: number }).value = g > 0 ? g : 0;
    },
    setTime(t) {
      (material.uniforms.uTime as { value: number }).value = t;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
