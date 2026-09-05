import * as THREE from 'three';
import { Reflector } from 'three/addons/objects/Reflector.js';
import type { Palette } from './tokens';

const REFLECTION_DISTORTION = 0.055;

export interface WaterOptions {
  size: number | readonly [number, number];
  palette: Palette;
  segments?: number;
  shallow?: THREE.ColorRepresentation;
  deep?: THREE.ColorRepresentation;
  sky?: THREE.ColorRepresentation;
  caustic?: THREE.ColorRepresentation;
  amplitude?: number;
  waveScale?: number;
  speed?: number;
  fresnel?: number;
  caustics?: number;
  depthFalloff?: number;
  opacity?: number;
  sunDir?: THREE.Vector3 | readonly [number, number, number];
  level?: number;
  name?: string;
  bedHeight?: (x: number, z: number) => number;
}

export type WaterHandle = ReturnType<typeof makeWater>;

const WAVES: readonly (readonly [number, number, number, number])[] = [
  [1.0, 0.15, 0.9, 0.55],
  [-0.35, 1.0, 1.35, 0.42],
  [0.7, -0.7, 2.6, 0.95],
  [-0.9, -0.45, 4.1, 1.35],
];
const WAVE_AMPS: readonly number[] = [0.55, 0.3, 0.11, 0.06];

export function waterHeightAt(x: number, z: number, time: number, amplitude: number, waveScale: number, speed: number): number {
  let h = 0;
  for (let i = 0; i < WAVES.length; i++) {
    const w = WAVES[i] as readonly [number, number, number, number];
    const amp = WAVE_AMPS[i] as number;
    h += amp * Math.sin((x * w[0] + z * w[1]) * w[2] * waveScale + time * w[3] * speed);
  }
  return h * amplitude;
}

const WAVE_GLSL =  `
const int OAK_WAVES = ${WAVES.length};
const vec4 OAK_WAVE[ OAK_WAVES ] = vec4[ OAK_WAVES ](${WAVES.map(w => `vec4(${w.map(n => n.toFixed(2)).join(',')})`).join(',')});
const float OAK_WAVE_AMP[ OAK_WAVES ] = float[ OAK_WAVES ](${WAVE_AMPS.join(',')});

float oakWaveHeight( vec2 p, float t, float scale, float speed ) {
	float h = 0.0;
	for ( int i = 0; i < OAK_WAVES; i ++ ) {
		vec4 w = OAK_WAVE[ i ];
		h += OAK_WAVE_AMP[ i ] * sin( dot( p, w.xy ) * w.z * scale + t * w.w * speed );
	}
	return h;
}

vec2 oakWaveSlope( vec2 p, float t, float scale, float speed ) {
	vec2 g = vec2( 0.0 );
	for ( int i = 0; i < OAK_WAVES; i ++ ) {
		vec4 w = OAK_WAVE[ i ];
		float k = w.z * scale;
		float c = cos( dot( p, w.xy ) * k + t * w.w * speed );
		g += OAK_WAVE_AMP[ i ] * c * k * w.xy;
	}
	return g;
}
`;

const WATER_VERT =  `
#include <common>
#include <fog_pars_vertex>
uniform float uTime;
uniform float uAmplitude;
uniform float uWaveScale;
uniform float uSpeed;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vHeight;
varying float vDepth;
${WAVE_GLSL}
void main() {
	vec3 p = position;
	vec4 world4 = modelMatrix * vec4( p, 1.0 );
	vec2 wp = world4.xz;
	float h = oakWaveHeight( wp, uTime, uWaveScale, uSpeed ) * uAmplitude;
	world4.y += h;
	vec2 slope = oakWaveSlope( wp, uTime, uWaveScale, uSpeed ) * uAmplitude;
	vNormalW = normalize( vec3( -slope.x, 1.0, -slope.y ) );
	vWorld = world4.xyz;
	vHeight = h;
	vec4 mv = viewMatrix * world4;
	vDepth = -mv.z;
	gl_Position = projectionMatrix * mv;
	#ifdef USE_FOG
		vFogDepth = -mv.z;
	#endif
}
`;

const WATER_FRAG =  `
#include <common>
#include <fog_pars_fragment>
uniform vec3 uShallow;
uniform vec3 uDeep;
uniform vec3 uSky;
uniform vec3 uCaustic;
uniform float uTime;
uniform float uFresnel;
uniform float uCausticStrength;
uniform float uDepthFalloff;
uniform float uOpacity;
uniform float uWaveScale;
uniform float uSpeed;
uniform vec3 uSunDir;
uniform float uSpecular;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vHeight;
varying float vDepth;

float oakCaustic( vec2 p, float t ) {
	vec2 a = p * 1.7 + vec2( t * 0.09, t * 0.06 );
	vec2 b = p * 2.3 - vec2( t * 0.07, t * 0.11 );
	float ca = abs( fract( a.x + a.y ) - 0.5 ) + abs( fract( a.x - a.y ) - 0.5 );
	float cb = abs( fract( b.x + b.y ) - 0.5 ) + abs( fract( b.x - b.y ) - 0.5 );
	float c = 1.0 - clamp( min( ca, cb ) * 2.4, 0.0, 1.0 );
	return pow( c, 3.0 );
}

void main() {
	vec3 n = normalize( vNormalW );
	vec3 viewDir = normalize( cameraPosition - vWorld );

	float byDepth = 1.0 - exp( -vDepth / max( uDepthFalloff, 0.001 ) );
	float byTrough = clamp( 0.5 - vHeight * 6.0, 0.0, 1.0 );
	vec3 col = mix( uShallow, uDeep, clamp( byDepth * 0.75 + byTrough * 0.35, 0.0, 1.0 ) );

	float caustic = oakCaustic( vWorld.xz * uWaveScale, uTime * uSpeed );
	col += uCaustic * ( caustic * uCausticStrength * ( 1.0 - byDepth * 0.7 ) );

	float f = pow( clamp( 1.0 - dot( n, viewDir ), 0.0, 1.0 ), 3.0 );
	col = mix( col, uSky, f * uFresnel );

	vec3 halfV = normalize( normalize( uSunDir ) + viewDir );
	float spec = pow( clamp( dot( n, halfV ), 0.0, 1.0 ), 96.0 );
	col += uSky * ( spec * uSpecular );

	float alpha = clamp( uOpacity + f * 0.35, 0.0, 1.0 );
	gl_FragColor = vec4( col, alpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}
`;

export function makeWater(opts: WaterOptions) {
  const sizeX = typeof opts.size === 'number' ? opts.size : opts.size[0];
  const sizeZ = typeof opts.size === 'number' ? opts.size : opts.size[1];
  const seg = Math.max(2, Math.round(opts.segments ?? 96));
  const p = opts.palette;

  const amplitude = opts.amplitude ?? 0.035;
  const waveScale = opts.waveScale ?? 1;
  const speed = opts.speed ?? 1;

  const sun = opts.sunDir instanceof THREE.Vector3 ? opts.sunDir.clone() : new THREE.Vector3(...(opts.sunDir ?? [0, 1, 0]));
  if (sun.lengthSq() < 1e-12) sun.set(0, 1, 0);
  sun.normalize();

  const uniforms = {
    uTime: { value: 0 },
    uShallow: { value: new THREE.Color(opts.shallow ?? p.skyHorizon).lerp(new THREE.Color(p.groundB), 0.35) },
    uDeep: { value: new THREE.Color(opts.deep ?? p.fog).lerp(new THREE.Color(p.skyBottom), 0.55) },
    uSky: { value: new THREE.Color(opts.sky ?? p.skyHorizon) },
    uCaustic: { value: new THREE.Color(opts.caustic ?? p.sun) },
    uAmplitude: { value: amplitude },
    uWaveScale: { value: waveScale },
    uSpeed: { value: speed },
    uFresnel: { value: opts.fresnel ?? 0.55 },
    uCausticStrength: { value: opts.caustics ?? 0.35 },
    uDepthFalloff: { value: opts.depthFalloff ?? 9 },
    uOpacity: { value: opts.opacity ?? 0.86 },
    uSunDir: { value: sun },
    uSpecular: { value: 1.6 },
  };

  const shader = {
    uniforms: { ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog), ...uniforms },
    vertexShader: WATER_VERT,
    fragmentShader: WATER_FRAG,
  };
  const geometry = new THREE.PlaneGeometry(sizeX, sizeZ, seg, seg);
  if (!opts.bedHeight) geometry.rotateX(-Math.PI / 2);
  geometry.name = 'oakWater';
  let mesh: THREE.Mesh;
  if (opts.bedHeight) {
    const positions = geometry.getAttribute('position'), depths = new Float32Array(positions.count);
    const stretch = (p: number, size: number) => Math.sign(p) * (4 * Math.abs(p * 2 / size) + (size / 2 - 4) * Math.abs(p * 2 / size) ** 3);
    for (let i = 0; i < positions.count; i++) {
      const x = stretch(positions.getX(i), sizeX), z = -stretch(positions.getY(i), sizeZ);
      positions.setXY(i, x, -z);
      depths[i] = (opts.level ?? 0) - opts.bedHeight(x, z);
    }
    geometry.setAttribute('waterDepth', new THREE.BufferAttribute(depths, 1));
    const indices: number[] = [], source = geometry.index!.array;
    for (let i = 0; i < source.length; i += 3) {
      const a = source[i]!, b = source[i + 1]!, c = source[i + 2]!;
      if (Math.max(depths[a]!, depths[b]!, depths[c]!) > -0.3) indices.push(a, b, c);
    }
    geometry.setIndex(indices);
    shader.vertexShader = shader.vertexShader.replace('void main() {', `
uniform mat4 textureMatrix;
attribute float waterDepth;
varying float vWaterDepth;
varying vec4 vReflection;
void main() {
  vReflection = textureMatrix * vec4(position, 1.0);
  vWaterDepth = waterDepth;
`);
    shader.fragmentShader = shader.fragmentShader.replace('void main() {', `
uniform sampler2D tDiffuse;
uniform float uAmplitude;
varying float vWaterDepth;
varying vec4 vReflection;
${WAVE_GLSL}
void main() {`).replace('vec3 n = normalize( vNormalW );', `
  float depth = vWaterDepth + vHeight;
  if (depth <= 0.0) discard;
  float footprint = length(fwidth(vWorld.xz));
  vec2 slope = oakWaveSlope(vWorld.xz, uTime, uWaveScale, uSpeed) * uAmplitude;
  slope += oakWaveSlope(vWorld.xz, uTime, uWaveScale * 5.0, uSpeed * 1.3) * uAmplitude * 0.035 / (1.0 + footprint * 8.0);
  vec3 n = normalize(vec3(-slope.x, 1.0, -slope.y));
`).replace('-vDepth / max( uDepthFalloff, 0.001 )', '-depth / max(uDepthFalloff, 0.001)')
      .replace('col = mix( col, uSky, f * uFresnel );', `
  f = 0.025 + 0.975 * pow(1.0 - max(dot(n, viewDir), 0.0), 5.0);
  vec2 uv = vReflection.xy / vReflection.w + n.xz * ${REFLECTION_DISTORTION} * smoothstep(0.0, 0.8, depth);
  vec3 reflected = texture2D(tDiffuse, clamp(uv, 0.002, 0.998)).rgb;
  float absorption = 1.0 - exp(-depth / max(uDepthFalloff, 0.001));
  float coverage = f + absorption * (1.0 - f);
  col = (reflected * f + col * absorption * (1.0 - f)) / max(coverage, 0.001);
`).replace('float alpha = clamp( uOpacity + f * 0.35, 0.0, 1.0 );', 'float alpha = coverage * smoothstep(0.0, 0.035, depth);');
    const reflection = new Reflector(geometry, { textureWidth: 1, textureHeight: 1, clipBias: 0.002, shader: {
      ...shader, uniforms: { ...shader.uniforms, tDiffuse: { value: null }, color: { value: null }, textureMatrix: { value: new THREE.Matrix4() } },
    } });
    reflection.rotation.x = -Math.PI / 2;
    const capture = reflection.onBeforeRender, size = new THREE.Vector2(), normal = new THREE.Vector3(), target = reflection.getRenderTarget();
    reflection.onBeforeRender = function (renderer, scene, camera, geometry, material, group) {
      renderer.getDrawingBufferSize(size);
      if (target.width !== size.x || target.height !== size.y) target.setSize(size.x, size.y);
      normal.set(0, 1, 0).transformDirection(camera.matrixWorldInverse);
      const projection = camera.projectionMatrix.elements;
      const horizon = (normal.z + Math.abs(normal.x / projection[0]!)) * projection[5]! / normal.y;
      target.scissor.set(0, 0, size.x, Math.ceil(size.y * THREE.MathUtils.clamp((horizon + 1) / 2 + REFLECTION_DISTORTION, 0, 1)));
      target.scissorTest = true;
      capture.call(this, renderer, scene, camera, geometry, material, group);
    };
    mesh = reflection;
  } else mesh = new THREE.Mesh(geometry, new THREE.ShaderMaterial(shader));
  const material = mesh.material as THREE.ShaderMaterial;
  Object.assign(material.uniforms, uniforms);
  material.name = opts.name ?? 'oakWater';
  material.transparent = material.fog = true;
  material.depthWrite = false;
  mesh.name = opts.name ?? 'oakWater';
  mesh.position.y = opts.level ?? 0;
  mesh.receiveShadow = false;
  mesh.castShadow = false;
  mesh.renderOrder = 1;

  return {
    mesh,
    material,
    uniforms,
    setTime(t: number) {
      uniforms.uTime.value = t;
    },
    setLevel(y: number) {
      mesh.position.y = y;
    },
    setColors(shallow: THREE.ColorRepresentation, deep: THREE.ColorRepresentation) {
      uniforms.uShallow.value.set(shallow);
      uniforms.uDeep.value.set(deep);
    },
    setSunDirection(x: number, y: number, z: number) {
      uniforms.uSunDir.value.set(x, y, z);
      if (uniforms.uSunDir.value.lengthSq() < 1e-12) uniforms.uSunDir.value.set(0, 1, 0);
      uniforms.uSunDir.value.normalize();
    },
    sampleHeight(x: number, z: number, time: number) {
      return mesh.position.y + waterHeightAt(x, z, time, amplitude, waveScale, speed);
    },
    dispose() {
      geometry.dispose();
      if (mesh instanceof Reflector) mesh.dispose();
      else material.dispose();
      mesh.removeFromParent();
    },
  };
}
