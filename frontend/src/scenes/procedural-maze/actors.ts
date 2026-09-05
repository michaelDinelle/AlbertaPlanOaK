import { sculptureMaterial } from '../key-corridor/world';
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import { clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import type { RuinSkin } from './world';

export const BEAM_LENGTH = 4.4;
export const BEAM_BASE_RADIUS = 0.4;
export const BEAM_TOP_RADIUS = 0.62;
export const BEAM_HDR = 0.86;
export const POOL_HDR = 0.62;
export const HALO_HDR = 0.34;
const LIGHT_HEIGHT = 0.55;

export interface SunShaftOptions {
  palette: Palette;
  skin: RuinSkin;
  sunDir: readonly [number, number, number];
  name?: string;
}

export interface SunShaftHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly lightHeight: number;
  setState(grow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

const BEAM_VERT =  `
#include <common>
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	vAxial = uv.y;
	vec4 wp = modelMatrix * vec4( position, 1.0 );
	vBeamNormal = normalize( mat3( modelMatrix ) * normal );
	vBeamView = normalize( cameraPosition - wp.xyz );
	gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const BEAM_FRAG =  `
#include <common>
uniform vec3 uColor;
uniform float uIntensity;
uniform float uOpacity;
uniform float uTime;
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	float facing = abs( dot( normalize( vBeamNormal ), normalize( vBeamView ) ) );
	float body = pow( facing, 1.35 );
	float fall = mix( 0.6, 1.0, vAxial );
	float capFade = 1.0 - smoothstep( 0.5, 1.0, vAxial );
	float bands = 0.87 + 0.13 * sin( vAxial * 11.0 - uTime * 0.55 );
	float a = body * fall * capFade * bands * uOpacity;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( uColor * uIntensity, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export function buildSunShaft(opts: SunShaftOptions): SunShaftHandle {
  const { palette, skin, sunDir } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sunShaft';

  const root = new THREE.Object3D();
  root.name = 'sunShaftRoot';
  group.add(root);

  const decor = new THREE.Group();
  decor.name = 'sunShaftDecor';
  root.add(decor);

  const radial = 22;
  const beamGeometry = new THREE.CylinderGeometry(
    BEAM_TOP_RADIUS,
    BEAM_BASE_RADIUS,
    BEAM_LENGTH,
    radial,
    1,
    true,
  );
  beamGeometry.translate(0, BEAM_LENGTH / 2, 0);
  beamGeometry.name = 'sunShaftBeam';

  const beamMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(skin.sunWarm) },
      uIntensity: { value: BEAM_HDR },
      uOpacity: { value: 0.26 },
      uTime: { value: 0 },
    },
    vertexShader: BEAM_VERT,
    fragmentShader: BEAM_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  beamMaterial.name = 'sunShaftBeamMaterial';

  const beam = new THREE.Mesh(beamGeometry, beamMaterial);
  beam.name = 'sunShaftBeam';
  beam.frustumCulled = false;
  const axis = new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]).normalize();
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  decor.add(beam);

  const sprite = radialSprite(64, 2.1);
  sprite.name = 'sunShaftPoolSprite';
  const poolGeometry = new THREE.PlaneGeometry(1.15, 1.15);
  poolGeometry.rotateX(-Math.PI / 2);
  poolGeometry.name = 'sunShaftPool';
  const poolMaterial: EmissiveMaterial = emissiveMaterial(skin.sunWarm, POOL_HDR, {
    map: sprite,
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'sunShaftPoolMaterial',
  });
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'sunShaftPool';
  pool.position.y = 0.022;
  pool.renderOrder = 2;
  decor.add(pool);

  const haloGeometry = new THREE.PlaneGeometry(2.5, 2.5);
  haloGeometry.rotateX(-Math.PI / 2);
  haloGeometry.name = 'sunShaftHalo';
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.sunWarm, HALO_HDR, {
    map: sprite,
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'sunShaftHaloMaterial',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'sunShaftHalo';
  halo.position.y = 0.014;
  halo.renderOrder = 1;
  decor.add(halo);

  const stubGeometry = atriumGeometry(skin);
  const stubMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.94,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.28,
    toonFloor: 0.42,
    name: 'sunShaftStubMaterial',
  });
  const stubs = new THREE.Mesh(stubGeometry, stubMaterial);
  stubs.name = 'sunShaftStubs';
  stubs.castShadow = true;
  stubs.receiveShadow = true;
  decor.add(stubs);

  let flareAmount = 0;
  let grow = 1;
  let breath = 1;

  const applyIntensity = (): void => {
    const gain = breath * (1 + 1.9 * flareAmount) * (0.25 + 0.75 * grow);
    (beamMaterial.uniforms.uIntensity as { value: number }).value = BEAM_HDR * gain;
    setEmissiveIntensity(poolMaterial, POOL_HDR * gain);
    setEmissiveIntensity(haloMaterial, HALO_HDR * gain);
  };
  applyIntensity();

  return {
    group,
    root,
    lightHeight: LIGHT_HEIGHT,

    setState(grow01) {
      grow = clamp01(grow01);
      decor.scale.setScalar(grow);
      applyIntensity();
    },

    flare(strength = 1) {
      flareAmount = Math.max(flareAmount, strength);
      applyIntensity();
    },

    setTime(t) {
      (beamMaterial.uniforms.uTime as { value: number }).value = t;
      breath = 1 + 0.09 * Math.sin(t * 0.73);
      applyIntensity();
      oakUniforms(stubMaterial)?.setTime(t);
    },

    update(dt) {
      if (flareAmount > 0) {
        flareAmount = Math.max(0, flareAmount - Math.max(0, dt) / 0.6);
        applyIntensity();
      }
    },

    dispose() {
      beamGeometry.dispose();
      beamMaterial.dispose();
      poolGeometry.dispose();
      poolMaterial.dispose();
      haloGeometry.dispose();
      haloMaterial.dispose();
      sprite.dispose();
      stubGeometry.dispose();
      stubMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function atriumGeometry(skin: RuinSkin): THREE.BufferGeometry {
  const radial = 10;
  const parts: ColoredPart[] = [];

  const corners: readonly (readonly [number, number, number])[] = [
    [-0.33, -0.33, 0.21],
    [0.33, -0.33, 0.13],
    [0.33, 0.33, 0.25],
    [-0.33, 0.33, 0.16],
  ];
  for (let i = 0; i < corners.length; i++) {
    const c = corners[i] as readonly [number, number, number];
    const stub = new THREE.CylinderGeometry(0.085, 0.105, c[2], radial, 1);
    stub.translate(0, c[2] * 0.5, 0);
    stub.rotateZ((i % 2 === 0 ? 1 : -1) * 0.08);
    stub.translate(c[0], 0.03, c[1]);
    parts.push({ geometry: stub, color: i % 2 === 0 ? skin.stoneLight : skin.stoneDark });
  }

  const ring = new THREE.RingGeometry(0.3, 0.42, 24);
  ring.rotateX(-Math.PI / 2);
  ring.translate(0, 0.028, 0);
  parts.push({ geometry: ring, color: skin.stoneLight });

  const slab = new THREE.BoxGeometry(0.62, 0.055, 0.62);
  slab.translate(0, 0.028, 0);
  parts.push({ geometry: slab, color: skin.flag });

  return mergeColoredParts(parts, 'sunShaftAtrium');
}
