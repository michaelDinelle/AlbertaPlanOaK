import * as THREE from 'three';
import { CELL, buildCellField, type CellFieldHandle } from '@kit/grid';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import type { SceneContext } from '@kit/types';
import { flowerGeometry, rockGeometry } from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import type { GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import { isReef, reefDepthY, reefSkin, shelfTint, terraceShelves, SHELF_THICKNESS, TERRACE_DESCENT, type ReefSkin } from './reef';

export function causticLighting(material: THREE.MeshStandardMaterial, strength = 0.075): { setTime(t: number): void } {
  const clock = { value: 0 };
  const compile = material.onBeforeCompile;
  const cacheKey = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    compile.call(material, shader, renderer);
    shader.uniforms.uReefTime = clock;
    shader.vertexShader = `varying vec3 vReefWorld;\n${shader.vertexShader}`.replace(
      '#include <project_vertex>',
      `vec4 reefWorld = vec4(transformed, 1.0);
#ifdef USE_INSTANCING
reefWorld = instanceMatrix * reefWorld;
#endif
vReefWorld = (modelMatrix * reefWorld).xyz;
#include <project_vertex>`,
    );
    shader.fragmentShader = `uniform float uReefTime;\nvarying vec3 vReefWorld;\n${shader.fragmentShader}`.replace(
      '#include <emissivemap_fragment>',
      `#include <emissivemap_fragment>
vec2 causticP = vReefWorld.xz * 5.8;
float causticA = sin(causticP.x + sin(causticP.y * 1.2 + uReefTime * .28));
float causticB = sin(causticP.y - sin(causticP.x * .8 - uReefTime * .22));
float caustic = pow(max(0., 1. - abs(causticA + causticB) * 2.4), 5.);
caustic = mix(.045, caustic, 1. - smoothstep(.3, 1.2, length(fwidth(causticP))));
totalEmissiveRadiance += vec3(.42,.91,1.) * caustic * ${strength.toFixed(3)} * (.7 + .3 * exp(-abs(vReefWorld.y)*.08));`,
    );
  };
  material.customProgramCacheKey = () => `${cacheKey}:caustic:${strength}`;
  return { setTime(t) { clock.value = t; } };
}

export function shelfLandscape(grid: GridGeometry, depth: number): { height: string; detail: string; heightAt(x: number, z: number): number } {
  const span = (grid.h - 1) * CELL;
  const half = grid.w * CELL / 2;
  const drop = TERRACE_DESCENT / (grid.h - 1) * 0.5 + 0.08;
  const smooth = (a: number, b: number, v: number): number => {
    const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  return {
    height: `float row = clamp((p.y+${(span / 2).toFixed(4)})/${CELL.toFixed(4)},0.,${(grid.h - 1).toFixed(1)});
float shelf = -${(TERRACE_DESCENT / (grid.h - 1)).toFixed(4)}*(floor(row)+smoothstep(.25,.75,fract(row)))-${drop.toFixed(4)};
float coast = 1.-smoothstep(${(CELL * 0.7).toFixed(4)},${(CELL * 3.7).toFixed(4)},p.x-p.y-2.2-sin((p.x+p.y)*.25)*1.2-sin((p.x+p.y)*.57)*.7);
float outer = smoothstep(${half.toFixed(4)},${(half + 12).toFixed(4)},max(abs(p.x),abs(p.y)));
shelf += outer*sin(p.x*.09)*sin(p.y*.07)*1.2;
float bed = -${depth.toFixed(1)} + sin(p.x*.035)*cos(p.y*.03)*1.5;
float bank = smoothstep(${(half + 5).toFixed(4)},${(half + 28).toFixed(4)},p.x);
return mix(mix(bed,shelf,coast),shelf+1.+sin(p.y*.06)*4.,bank);`,
    detail: `float row = clamp((p.z+${(span / 2).toFixed(4)})/${span.toFixed(4)},0.,1.); color = mix(uLandA,uLandB,row); color *= .94 + .06*sin(p.y*3.1+sin(p.x*.3+p.z*.2));`,
    heightAt(x, z) {
      const row = Math.min(grid.h - 1, Math.max(0, (z + span / 2) / CELL));
      const outer = smooth(half, half + 12, Math.max(Math.abs(x), Math.abs(z)));
      const shelf = -TERRACE_DESCENT / (grid.h - 1) * (Math.floor(row) + smooth(0.25, 0.75, row % 1)) - drop + outer * Math.sin(x * 0.09) * Math.sin(z * 0.07) * 1.2;
      const bed = -depth + Math.sin(x * 0.035) * Math.cos(z * 0.03) * 1.5;
      const coast = 1 - smooth(CELL * 0.7, CELL * 3.7, x - z - 2.2 - Math.sin((x + z) * 0.25) * 1.2 - Math.sin((x + z) * 0.57) * 0.7);
      const bank = smooth(half + 5, half + 28, x);
      return (bed + (shelf - bed) * coast) * (1 - bank) + (shelf + 1 + Math.sin(z * 0.06) * 4) * bank;
    },
  };
}

export function coralGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < 7; i++) {
    const angle = -1.25 + i * 0.42;
    const direction = new THREE.Vector3(Math.sin(angle) * 0.65, 0.7 + Math.cos(angle) * 0.2, Math.sin(i * 2.4) * 0.15).normalize();
    const stem = new THREE.CylinderGeometry(0.012, 0.037, 0.6, 5);
    stem.translate(0, 0.3, 0);
    stem.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, direction));
    parts.push({ geometry: stem, color: 0xffffff });
    for (const side of [-1, 1]) {
      const branch = new THREE.CylinderGeometry(0.006, 0.016, 0.23, 4);
      branch.rotateZ(side * 0.55);
      branch.translate(direction.x * 0.35 + side * 0.05, direction.y * 0.35 + 0.1, direction.z * 0.35);
      parts.push({ geometry: branch, color: 0xffffff });
    }
  }
  return mergeColoredParts(parts, 'staghornCoral');
}

export function makeKelpGeometry(): THREE.BufferGeometry {
  const detail = 1;
  const blades = Math.max(3, Math.round(5 * detail));
  const segments = Math.max(2, Math.round(4 * detail));
  const quads = blades * segments;
  const positions = new Float32Array(quads * 6 * 3);
  const colors = new Float32Array(quads * 6 * 3);
  let o = 0;

  const push = (x: number, y: number, z: number, k: number): void => {
    positions[o * 3] = x;
    positions[o * 3 + 1] = y;
    positions[o * 3 + 2] = z;
    colors[o * 3] = k;
    colors[o * 3 + 1] = k;
    colors[o * 3 + 2] = k;
    o++;
  };

  for (let b = 0; b < blades; b++) {
    const yaw = (b / blades) * Math.PI * 2 + ((b * 7) % 5) * 0.19;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const height = .74 * (0.72 + 0.28 * (((b * 3) % 4) / 3));
    const offset = 0.06 + 0.05 * (((b * 5) % 3) / 2);
    for (let s = 0; s < segments; s++) {
      const t0 = s / segments;
      const t1 = (s + 1) / segments;
      const y0 = t0 * height;
      const y1 = t1 * height;
      const w0 = 0.055 * (1 - 0.78 * t0);
      const w1 = 0.055 * (1 - 0.78 * t1);
      const s0 = Math.sin(t0 * 2.4) * 0.055;
      const s1 = Math.sin(t1 * 2.4) * 0.055;
      const k0 = 0.42 + 0.7 * t0;
      const k1 = 0.42 + 0.7 * t1;
      const ax = cos * (offset + s0);
      const az = sin * (offset + s0);
      const bx = cos * (offset + s1);
      const bz = sin * (offset + s1);
      const nx = -sin;
      const nz = cos;
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(ax + nx * w0, y0, az + nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(ax - nx * w0, y0, az - nz * w0, k0);
      push(bx + nx * w1, y1, bz + nz * w1, k1);
      push(bx - nx * w1, y1, bz - nz * w1, k1);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'reefKelpClump';
  return g;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  skin: ReefSkin;
  silt: AmbientParticlesHandle;
  rush: AmbientParticlesHandle;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, reach: Uint8Array): WorldHandle {
  const palette = ctx.tokens.palettes.underwater;
  const skin = reefSkin(palette, ctx.tokens);
  const w = makeWorld('sunlitDescent');
  const sky = w.time(w.own(skyFromPalette(palette, 'underwater', { topSpread: 0.8, bottomSpread: 0.7 })));
  sky.setColors(0x91ece1, 0x217fa2, 0x05294c);
  sky.material.fragmentShader = sky.material.fragmentShader.replace(
    'gl_FragColor = vec4( col, 1.0 );',
    `float overhead = pow(max(dir.y, 0.), 8.);
col += vec3(.66,1.,.87) * overhead * (1.25 + .16 * sin(dir.x * 30. + dir.z * 27. + uTime * .17));
gl_FragColor = vec4(col, 1.);`,
  );
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: [-0.22, 0.95, 0.22], pointCount: 3, pointDistance: 9 }));
  rig.hemi.intensity *= 1.25;
  rig.setShadowFocus({ x: 0, y: -1.75, z: 0 }, 12);
  w.add(rig.group);
  scene.fog = new THREE.FogExp2(0x247b96, 0.012);
  const tint = new THREE.Color();
  const field = w.own(buildCellField(grid, {
    floorColor: 0xffffff,
    wallColor: 0xffffff,
    floorFilter: (x, y) => isReef(reach, grid, x, y),
    floorTint: (x, y) => shelfTint(skin, grid, x, y, tint).clone(),
    floorInset: 0.006,
    floorThickness: SHELF_THICKNESS,
    jitter: 0.018,
    rimColor: 0xacefe0,
    rimStrength: 0.24,
    seed: ctx.seed,
    name: 'reefShelves',
  }));
  terraceShelves(field, grid);
  w.time(field);
  w.time(causticLighting(field.floorMaterial));
  w.add(field.group);
  const half = grid.w * CELL * 0.5;
  const shelf = shelfLandscape(grid, 6.5);
  const heightAt = shelf.heightAt;
  const floor = w.time(w.own(makeContinuousSurface({
    name: 'reefEscarpment', colorA: skin.shelfShallow, colorB: skin.shelfDeep, segments: 112,
    height: shelf.height,
    detail: shelf.detail,
  })));
  w.time(causticLighting(floor.material, 0.018));
  w.add(floor.mesh);
  const outcropGeometry = w.own(rockGeometry(2.7, 1));
  const rockMaterial = w.own(oakMaterial({ color: 0xffffff, flatShading: true, roughness: 0.9, rimStrength: 0.2 }));
  w.time(causticLighting(rockMaterial, 0.028));
  const rocks = w.own(makeSceneryBank({
    name: 'reefOutcrops', seed: ctx.seed, tileSize: 18, perTile: 2,
    geometry: outcropGeometry, material: rockMaterial, colorA: 0x325e62, colorB: 0x63938d,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 2; i++) {
        const x = (tx + rng()) * 18;
        const z = (tz + rng()) * 18;
        if (Math.abs(x) < half + 7 && Math.abs(z) < half + 7) continue;
        write(x, heightAt(x, z) - 0.15, z, rng() * 6.28, 0.8 + rng() * 1.6, 0.8 + rng() * 0.6, rng());
      }
    },
  }));
  w.add(rocks.mesh);
  const gardenMaterials: THREE.Material[] = [];
  const gardens = [
    { geometry: makeKelpGeometry(), color: skin.kelp, count: 72, scale: 0.85, name: 'reefKelp' },
    { geometry: flowerGeometry(5), color: skin.anemone, count: 48, scale: 1, name: 'reefAnemones' },
  ].map((spec, index) => {
    const geometry = w.own(spec.geometry);
    const material = w.own(oakMaterial({ color: spec.color, vertexColors: true, flatShading: true, roughness: 0.8, side: THREE.DoubleSide, wind: { amplitude: 0.025, frequency: 0.65, height: 0.74 } }));
    gardenMaterials.push(material);
    const bank = w.own(makeSceneryBank({
      name: spec.name, seed: ctx.seed ^ (0x7ee1 + index), tileSize: 12, perTile: spec.count,
      geometry, material, colorA: 0x9ac2b0, colorB: 0xd5dfbd,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < spec.count; i++) {
          const x = (tx + rng()) * 12;
          const z = (tz + rng()) * 12;
          const column = Math.round(x / CELL + (grid.w - 1) / 2);
          const row = Math.round(z / CELL + (grid.h - 1) / 2);
          const onReef = isReef(reach, grid, column, row);
          const centerX = (column - (grid.w - 1) / 2) * CELL;
          const centerZ = (row - (grid.h - 1) / 2) * CELL;
          if (onReef && Math.hypot(x - centerX, z - centerZ) < (column === grid.w - 1 && row === grid.h - 1 ? 0.8 : 0.24)) continue;
          write(x, onReef ? reefDepthY(row, grid) : heightAt(x, z), z, rng() * 6.28, spec.scale * (0.8 + rng() * 0.6), 0.8 + rng() * 0.4, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const silt = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 64, bounds: { halfX: 0.35, halfY: 0.2, halfZ: 0.35 },
    seed: ctx.seed, color: skin.silt, size: 0.9, intensity: 1.1, rise: -0.07, name: 'reefSilt',
  }))));
  const rush = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'bubbles', count: 100, bounds: { halfX: 0.26, halfY: 3.6, halfZ: 0.26 },
    seed: ctx.seed ^ 0x2c58, color: skin.bubble, size: 1.5, intensity: 2, rise: 2.4, name: 'resurfaceBubbles',
  }))));
  silt.setDensity(0);
  rush.setDensity(0);
  w.add(silt.points);
  w.add(rush.points);
  return w.seal({
    sky, rig, field, skin, silt, rush,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      floor.setView(coverage);
      rocks.setView(coverage);
      for (const bank of gardens) bank.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.9);
    },
  }, { time(t) { for (const material of gardenMaterials) oakUniforms(material)?.setTime(t); } });
}

export function treasureCell(grid: GridGeometry): { c: number; r: number } {
  return { c: grid.goal?.x ?? grid.w - 1, r: grid.goal?.y ?? grid.h - 1 };
}
