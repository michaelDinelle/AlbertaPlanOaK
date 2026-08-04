// The Overgrown Ruins — the sunbeam atrium (the goal).
//
// ACCURACY: `root` is the probe object. Its world position is EXACTLY
// `cellToWorld(gx, gy)` with y = 0 and nothing else — no hover, no lean, no easing. The
// beam, the pool of light on the flagstone and the four broken column stubs all hang off
// `decor`, a child that carries the grow-in scale. Delete every decoration and what
// remains is a correctly-placed empty node on the live goal cell.
//
// THE BEAM IS THIS SCENE'S ONE BESPOKE SHADER (STYLE.md allows exactly one). It is a
// hollow cone drawn additively, and the trick that makes a shell read as a volume is
// `abs(dot(normal, viewDir))`: the shell is brightest where it faces the camera — through
// the middle of the beam — and fades to nothing at its own silhouette, which is the
// opposite of a fresnel rim and the reason it does not look like a paper cone. Its axis
// is the SCENE's sun direction, shared with the sky dome and the key light, so the beam,
// the sun disc and the shadows can never disagree about where the sun is.
//
// HONESTY: this marks the goal cell and only the goal cell. It says nothing about whether
// the goal can be reached — ProceduralMaze really does respawn goals into sealed pockets
// — so there is no trail, no arrow, no glow along the way, and nothing here ever reads
// `carvedPath`.
//
// IDEMPOTENCE: `setState(grow)` and `setTime(t)` are pure. Nothing integrates, remembers
// a position or accumulates, so calling them twice with the same arguments leaves the
// same world, and a scrub that snaps grow to 1 lands exactly on the frame a long-settled
// atrium would show.

import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { clamp01, mergeColoredParts, type ColoredPart, type RuinSkin } from './world';

/**
 * How far up the beam reaches along the sun direction. At 5.6 it leaned 2.7 units clear
 * of the ruin and ran out of the top of the composed frame, which reads as a light leak
 * rather than as a column of air; a shaft has to END somewhere you can see.
 */
export const BEAM_LENGTH = 4.4;
/**
 * Where it lands, and how much it spreads. A sunbeam through a hole in a roof is very
 * nearly PARALLEL — the sun is half a degree wide. The first pass opened to 1.12 at the
 * top and the result read as a theatre spotlight aimed at the floor, which is a completely
 * different (and much cheaper) idea than daylight finding its way in.
 */
export const BEAM_BASE_RADIUS = 0.4;
export const BEAM_TOP_RADIUS = 0.62;
/**
 * The beam's gain. It sits DELIBERATELY under the 0.85 bloom threshold: at 1.22 (the first
 * pass) an additive cone five units tall, stacked on its own pool and halo, bloomed into a
 * white column that ate four cells of maze and cropped out of the frame. A shaft of
 * sunlight is bright AIR. The only thing here allowed over the threshold is the small hot
 * pool where it actually lands.
 */
export const BEAM_HDR = 0.86;
/** the warm pool where it lands — the one bloom-feeder of the atrium */
export const POOL_HDR = 0.62;
/**
 * The wide soft halo the pool sits inside — a suggestion of light, never a second lamp.
 * It is declared at the value the halo actually RENDERS at: `applyIntensity()` re-derives
 * every emissive from its constant on the first frame, so a constant that disagreed with
 * that line (this one read 0.14 while the halo drew at 0.34) was documentation of a value
 * nothing ever showed.
 */
export const HALO_HDR = 0.34;
/** local height the scene lifts its point light to */
const LIGHT_HEIGHT = 0.55;

export interface SunShaftOptions {
  tier: QualityTier;
  palette: Palette;
  skin: RuinSkin;
  /** unit vector pointing TOWARD the sun — the world's, never a second opinion */
  sunDir: readonly [number, number, number];
  name?: string;
}

export interface SunShaftHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(gx, gy), y = 0, always */
  root: THREE.Object3D;
  /** local height of the light source the scene should put its point light at */
  readonly lightHeight: number;
  /** the whole visual state: how far the atrium has grown in since it moved */
  setState(grow01: number): void;
  /** a bright kindling beat when the sun finds a new opening */
  flare(strength?: number): void;
  /** ctx.time() — the dust banding and the slow breath */
  setTime(t: number): void;
  /** flare decay (dt is wall seconds, time the playback clock) */
  update(dt: number, time: number): void;
  dispose(): void;
}

const BEAM_VERT = /* glsl */ `
#include <common>
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	// uv.y runs 0 at the base to 1 at the top of a CylinderGeometry
	vAxial = uv.y;
	vec4 wp = modelMatrix * vec4( position, 1.0 );
	vBeamNormal = normalize( mat3( modelMatrix ) * normal );
	vBeamView = normalize( cameraPosition - wp.xyz );
	gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const BEAM_FRAG = /* glsl */ `
#include <common>
uniform vec3 uColor;
uniform float uIntensity;
uniform float uOpacity;
uniform float uTime;
varying vec3 vBeamNormal;
varying vec3 vBeamView;
varying float vAxial;
void main() {
	// A SHELL THAT READS AS A VOLUME: brightest where the cone faces the camera (through
	// the middle of the beam), gone at its own silhouette. The inverse of a fresnel.
	float facing = abs( dot( normalize( vBeamNormal ), normalize( vBeamView ) ) );
	float body = pow( facing, 1.35 );
	// Brighter toward the SUN, not toward the floor: the pool below is already a hot
	// sprite, and piling the beam's own maximum on top of it is what blew the first pass
	// out. Multiplied by the top fade, the beam peaks in its upper middle — bright air
	// hanging over the atrium — and arrives at the flagstones as a soft edge.
	float fall = mix( 0.6, 1.0, vAxial );
	float capFade = 1.0 - smoothstep( 0.5, 1.0, vAxial );
	// dust drifting up the shaft, on the playback clock: a paused frame holds it
	float bands = 0.87 + 0.13 * sin( vAxial * 11.0 - uTime * 0.55 );
	float a = body * fall * capFade * bands * uOpacity;
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( uColor * uIntensity, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export function buildSunShaft(opts: SunShaftOptions): SunShaftHandle {
  const { tier, palette, skin, sunDir } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sunShaft';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'sunShaftRoot';
  group.add(root);

  // ---- everything decorative rides the grow-in scale ----
  const decor = new THREE.Group();
  decor.name = 'sunShaftDecor';
  root.add(decor);

  // ---- the beam ----
  const radial = Math.max(8, Math.round(22 * detail));
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
  // point the cone's +y axis at the sun: the beam comes FROM the sky, along the same
  // vector the key light and the sun disc use
  const axis = new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]).normalize();
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  decor.add(beam);

  // ---- the warm pool on the flagstone ----
  const sprite = radialSprite(64, 2.1);
  sprite.name = 'sunShaftPoolSprite';
  // ONE CELL OF LIGHT, and no more: the pool is where the beam LANDS, and a pool that
  // spilled over its neighbours would be claiming ground the goal does not occupy. The
  // quad is 1.15 so the sprite's own falloff — (1 − r)^2.1, exactly 0 at the quad's
  // inscribed circle — has somewhere to land: at the cell edge it is already 0.013.
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

  // ---- the atrium: four broken column stubs around the opening ----
  const stubGeometry = atriumGeometry(skin, tier);
  const stubMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.94,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 3,
    toonSoftness: 0.28,
    toonFloor: 0.42,
    tier,
    name: 'sunShaftStubMaterial',
  });
  const stubs = new THREE.Mesh(stubGeometry, stubMaterial);
  stubs.name = 'sunShaftStubs';
  stubs.castShadow = tier !== 'thumb';
  stubs.receiveShadow = tier !== 'thumb';
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
      // growth is a SCALE on the decoration node — the root never moves, so an atrium
      // that is 3% grown is still exactly on its cell
      decor.scale.setScalar(grow);
      applyIntensity();
    },

    flare(strength = 1) {
      flareAmount = Math.max(flareAmount, strength);
      applyIntensity();
    },

    setTime(t) {
      (beamMaterial.uniforms.uTime as { value: number }).value = t;
      // the beam breathes as the air moves through it — pure ctx.time()
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

/**
 * The atrium floor: four broken column stubs at the corners of the cell plus a worn
 * threshold ring. The widest thing here is the tallest stub's sheared top corner at 0.455
 * from the cell centre (0.33 offset + 0.105 radius, leaned 0.08 rad); the foundation face
 * of the block next door stands at 0.52, so an atrium that respawns hard against a wall
 * still never pushes into the masonry.
 */
function atriumGeometry(skin: RuinSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(5, Math.round(10 * detail));
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
    // each stub sheared a different way, so four of them never read as a table
    stub.rotateZ((i % 2 === 0 ? 1 : -1) * 0.08);
    stub.translate(c[0], 0.03, c[1]);
    parts.push({ geometry: stub, color: i % 2 === 0 ? skin.stoneLight : skin.stoneDark });
  }

  const ring = new THREE.RingGeometry(0.3, 0.42, Math.max(10, Math.round(24 * detail)));
  ring.rotateX(-Math.PI / 2);
  ring.translate(0, 0.028, 0);
  parts.push({ geometry: ring, color: skin.stoneLight });

  const slab = new THREE.BoxGeometry(0.62, 0.055, 0.62);
  slab.translate(0, 0.028, 0);
  parts.push({ geometry: slab, color: skin.flag });

  return mergeColoredParts(parts, 'sunShaftAtrium');
}
