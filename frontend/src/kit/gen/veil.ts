import * as THREE from 'three';

export interface VeilOptions {
  layers: number;
  width: number;
  height: number;
  spacing: number;
  color: THREE.ColorRepresentation;
  edge: THREE.ColorRepresentation;
  drift: readonly [number, number];
  scale: number;
  sharpness: number;
  gain: number;
  taper: number;
  billboard?: boolean;
  name?: string;
}

export interface VeilHandle {
  group: THREE.Group;
  material: THREE.ShaderMaterial;
  setTime(t: number): void;
  setGain(g: number): void;
  dispose(): void;
}

const VERT = `
varying vec2 vVeilUv;
varying float vVeilLayer;
attribute float aLayer;
uniform float uBillboard;
void main() {
	vVeilUv = uv;
	vVeilLayer = aLayer;
	vec4 mv = modelViewMatrix * vec4( position, 1.0 );
	if ( uBillboard > 0.5 ) {
		mv = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );
		mv.xy += position.xy;
	}
	gl_Position = projectionMatrix * mv;
}
`;

const FRAG = `
precision highp float;
varying vec2 vVeilUv;
varying float vVeilLayer;
uniform vec3 uColor;
uniform vec3 uEdge;
uniform vec2 uDrift;
uniform float uTime;
uniform float uScale;
uniform float uSharpness;
uniform float uGain;
uniform float uTaper;

float oakHash( vec2 p ) {
	return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 );
}

float oakNoise( vec2 p ) {
	vec2 i = floor( p );
	vec2 f = fract( p );
	f = f * f * ( 3.0 - 2.0 * f );
	return mix(
		mix( oakHash( i ), oakHash( i + vec2( 1.0, 0.0 ) ), f.x ),
		mix( oakHash( i + vec2( 0.0, 1.0 ) ), oakHash( i + vec2( 1.0, 1.0 ) ), f.x ),
		f.y );
}

float oakFbm( vec2 p ) {
	float a = 0.5;
	float s = 0.0;
	for ( int i = 0; i < 4; i++ ) {
		s += a * oakNoise( p );
		p *= 2.03;
		a *= 0.5;
	}
	return s;
}

void main() {
	vec2 p = vVeilUv * uScale + uDrift * uTime + vec2( vVeilLayer * 7.3, vVeilLayer * 3.1 );
	float n = oakFbm( p );
	n = pow( clamp( n, 0.0, 1.0 ), uSharpness );
	float fade = pow( clamp( 1.0 - abs( vVeilUv.y * 2.0 - 1.0 ), 0.0, 1.0 ), uTaper );
	float edgeMix = clamp( vVeilUv.y, 0.0, 1.0 );
	float alpha = n * fade * uGain;
	if ( alpha < 0.002 ) discard;
	gl_FragColor = vec4( mix( uColor, uEdge, edgeMix ), alpha );
}
`;

export function veil(opts: VeilOptions): VeilHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'oakVeil';

  const layers = Math.max(1, Math.floor(opts.layers));
  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {
      uColor: { value: new THREE.Color(opts.color) },
      uEdge: { value: new THREE.Color(opts.edge) },
      uDrift: { value: new THREE.Vector2(opts.drift[0], opts.drift[1]) },
      uTime: { value: 0 },
      uScale: { value: opts.scale },
      uSharpness: { value: opts.sharpness },
      uGain: { value: opts.gain },
      uTaper: { value: opts.taper },
      uBillboard: { value: opts.billboard ? 1 : 0 },
    },
  });
  material.name = `${group.name}:material`;

  const geometries: THREE.BufferGeometry[] = [];
  for (let i = 0; i < layers; i++) {
    const g = new THREE.PlaneGeometry(opts.width, opts.height, 1, 1);
    const n = g.getAttribute('position').count;
    const layer = new Float32Array(n).fill(i / Math.max(1, layers - 1));
    g.setAttribute('aLayer', new THREE.BufferAttribute(layer, 1));
    const mesh = new THREE.Mesh(g, material);
    mesh.position.z = (i - (layers - 1) / 2) * opts.spacing;
    mesh.frustumCulled = false;
    mesh.renderOrder = 4;
    group.add(mesh);
    geometries.push(g);
  }

  return {
    group,
    material,
    setTime(t) {
      material.uniforms['uTime']!.value = t;
    },
    setGain(g) {
      material.uniforms['uGain']!.value = g;
    },
    dispose() {
      for (const g of geometries) g.dispose();
      geometries.length = 0;
      material.dispose();
      group.clear();
      group.removeFromParent();
    },
  };
}
