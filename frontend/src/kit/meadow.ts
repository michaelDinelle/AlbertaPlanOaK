import * as THREE from 'three/webgpu';
import { abs, color, exp, max, mix, modelWorldMatrix, sin, smoothstep, step, vec3, vec4, vertexColor } from 'three/tsl';
import { makeHeightField } from '@kit/heightfield';
import { deformOf, oakMaterial, setDeform } from '@kit/materials';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import type { SceneContext } from '@kit/types';
import { bareTreeGeometry, broadleafTreeGeometry, coniferTreeGeometry, grassClumpGeometry, type TreeArchetype } from '@kit/vegetation';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import type { WorldBuilder } from '@kit/world';

export interface MeadowOptions {
  name: string;
  halfX: number;
  halfZ: number;
  ground: THREE.ColorRepresentation;
  grass: THREE.ColorRepresentation;
  trees: readonly THREE.ColorRepresentation[];
  archetypes?: readonly TreeArchetype[];
  treeKind?: (x: number, z: number) => TreeArchetype;
  grounds?: readonly THREE.ColorRepresentation[];
  amplitude?: number;
}

export function makeMeadow(w: WorldBuilder, ctx: SceneContext, opts: MeadowOptions) {
  const amplitude = opts.amplitude ?? 2.6;
  const field = makeHeightField({ base: -0.075, court: { halfX: opts.halfX, halfZ: opts.halfZ, rise: 0.34 }, octaves: [{ wavelength: 31, amplitude: amplitude * 0.55, seed: ctx.seed }, { wavelength: 13, amplitude: amplitude * 0.25, seed: ctx.seed ^ 0x51 }] });
  const heightAt = field.heightAt;
  const grounds = opts.grounds?.map((value) => color(new THREE.Color(value)));
  const surface = w.time(w.own(makeContinuousSurface({
    name: `${opts.name}Terrain`,
    seed: ctx.seed,
    colorA: opts.ground,
    colorB: opts.grass,
    height: field.height,
    follow: false,
    detail({ p, color: ground, noise }) {
      const grass = sin(p.x.mul(1.7).add(sin(p.z.mul(1.1)))).mul(sin(p.z.mul(2.3)));
      if (grounds) {
        const seasonX = smoothstep(-4, 4, p.x.add(noise(p.xz.mul(0.08).add(11)).sub(0.5).mul(7)));
        const seasonZ = smoothstep(-4, 4, p.z.add(noise(p.xz.mul(0.07).add(23)).sub(0.5).mul(7)));
        const season = mix(mix(grounds[0]!, grounds[1]!, seasonX), mix(grounds[2]!, grounds[3]!, seasonX), seasonZ);
        const garden = exp(max(0, max(abs(p.x).sub(opts.halfX), abs(p.z).sub(opts.halfZ))).negate().mul(0.07));
        ground = mix(ground, season, garden.mul(0.62));
      }
      return ground.mul(grass.mul(0.045).add(0.96));
    },
  })));
  w.add(surface.mesh);
  const material = w.own(oakMaterial({ vertexColors: true, roughness: 0.9, flatShading: true,
    rimColor: opts.ground, rimStrength: 0.3, wind: { amplitude: 0.03, frequency: 0.8, height: 1.7 }, name: `${opts.name}Trees` }));
  const wind = deformOf(material)!, grove = (i: number) => color(new THREE.Color(opts.trees[i % opts.trees.length] as THREE.ColorRepresentation));
  setDeform(material, (matrix) => {
    const world = modelWorldMatrix.mul(matrix ? matrix.mul(vec4(0, 0, 0, 1)) : vec4(0, 0, 0, 1)).xz, tone = vertexColor();
    const season = smoothstep(-6, 6, world.add(sin(world.yx.mul(0.17)).mul(2.5)));
    return { ...wind(matrix), tint: mix(vec3(1), mix(mix(grove(0), grove(1), season.x), mix(grove(2), grove(3), season.x), season.y), step(tone.r, tone.g.mul(1.35))) };
  });
  const kinds = opts.archetypes ?? ['broadleaf'];
  const treeGeometry = (kind: TreeArchetype, detail: number) => {
    const geometry = kind === 'conifer' ? coniferTreeGeometry(detail) : kind === 'bare' ? bareTreeGeometry(detail) : broadleafTreeGeometry(detail);
    const colors = geometry.getAttribute('color');
    const trunk = kind === 'bare' ? colors.count : (kind === 'conifer' ? Math.max(5, 5 + detail * 3) : Math.max(4, 4 + detail * 2)) * 12;
    for (let i = 0; i < trunk; i++) colors.setXYZ(i, 0.28, 0.15, 0.075);
    return geometry;
  };
  const banks = kinds.map((kind, variant) => {
    const geometry = w.own(treeGeometry(kind, 1));
    const bank = w.own(makeSceneryBank({
      name: `${opts.name}${kind}`, geometry, material, tileSize: 12, perTile: 16,
      seed: ctx.seed ^ (variant * 997), colorA: 0xffffff, colorB: 0xbdcbb2, receiveShadow: true,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < 16; i++) {
          const x = (tx + (i % 4 + rng()) / 4) * 12;
          const z = (tz + (Math.floor(i / 4) + rng()) / 4) * 12;
          if (Math.abs(x) < opts.halfX + 0.7 && Math.abs(z) < opts.halfZ + 0.7) continue;
          if (opts.treeKind ? opts.treeKind(x, z) !== kind : rng() > 1 / kinds.length) continue;
          const grove = 0.5 + 0.5 * Math.sin(x * 0.24 + Math.cos(z * 0.21) * 1.8);
          if (rng() > 0.3 + grove * 0.7) continue;
          write(x, heightAt(x, z), z, rng() * Math.PI * 2, 0.7 + rng() * 0.65, 0.9 + rng() * 0.45, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const massParts: ColoredPart[] = [];
  for (let i = 0; i < 18; i++) {
    const kind = kinds[i % kinds.length] as TreeArchetype;
    const crown = treeGeometry(kind, 0);
    const x = Math.sin(i * 13.37) * 5.2;
    const z = Math.cos(i * 7.81) * 5.2;
    const size = 0.75 + hash2f(i, 31, ctx.seed) * 0.55;
    crown.scale(size, size * (0.9 + hash2f(i, 14, ctx.seed) * 0.45), size);
    crown.rotateY(i * 2.4);
    crown.translate(x, 0, z);
    massParts.push({ geometry: crown, color: new THREE.Color().setScalar(0.78 + (i % 4) * 0.06) });
  }
  const massColors = massParts.flatMap(({ geometry }) => Array.from(geometry.getAttribute('color').array));
  const massGeometry = w.own(mergeColoredParts(massParts, `${opts.name}Woodland`));
  massGeometry.setAttribute('color', new THREE.Float32BufferAttribute(massColors, 3));
  const masses = w.own(makeSceneryBank({
    name: `${opts.name}DistantGroves`, geometry: massGeometry, material,
    tileSize: 96, perTile: 8, seed: ctx.seed, colorA: 0xffffff, colorB: 0xc9d3d6,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 8; i++) {
        const x = (tx + rng()) * 96;
        const z = (tz + rng()) * 96;
        if (Math.abs(x) < opts.halfX + 70 && Math.abs(z) < opts.halfZ + 70) continue;
        write(x, heightAt(x, z) - 0.05, z, rng() * Math.PI * 2, 1, 1, rng());
      }
    },
  }));
  w.add(masses.mesh);
  const grassMaterial = w.own(oakMaterial({ color: opts.grass, vertexColors: true, roughness: 0.94,
    side: THREE.DoubleSide, wind: { amplitude: 0.04, frequency: 1.1, height: 0.5 }, name: `${opts.name}Understory` }));
  const grass = w.own(makeSceneryBank({
    name: `${opts.name}Meadow`, geometry: w.own(grassClumpGeometry(3)), material: grassMaterial,
    tileSize: 12, perTile: 256, seed: ctx.seed ^ 0x54ad, colorA: 0xe6efc4, colorB: 0xabb998,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 256; i++) {
        const x = (tx + rng()) * 12;
        const z = (tz + rng()) * 12;
        const outer = Math.max(Math.abs(x) - opts.halfX, Math.abs(z) - opts.halfZ);
        if (outer < 0.02) continue;
        const patch = 0.5 + 0.5 * Math.sin(x * 0.8 + Math.sin(z * 0.55) * 2.1);
        if (rng() > 0.24 + Math.max(Math.exp(-outer * 0.6), patch * 0.65) * 0.76) continue;
        write(x, heightAt(x, z), z, rng() * Math.PI * 2, 0.32 + rng() * 0.36, 0.8 + rng() * 0.7, rng());
      }
    },
  }));
  w.add(grass.mesh);
  w.time({ setTime(t) { material.oak.setTime(t); grassMaterial.oak.setTime(t); } });
  return {
    heightAt,
    setView(coverage: ViewCoverage) {
      surface.setView(coverage);
      for (const bank of banks) bank.setView(coverage);
      grass.setView(coverage);
      masses.setView(coverage);
    },
  };
}


