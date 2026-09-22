import * as THREE from 'three';
import { makeHeightField } from '@kit/heightfield';
import { oakMaterial } from '@kit/materials';
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
  const grounds = opts.grounds?.map((value) => {
    const c = new THREE.Color(value);
    return `vec3(${c.r.toFixed(5)}, ${c.g.toFixed(5)}, ${c.b.toFixed(5)})`;
  });
  const groundDetail = grounds ? `
    float seasonX = smoothstep(-4.0, 4.0, p.x + (landNoise(p.xz * 0.08 + 11.0) - 0.5) * 7.0);
    float seasonZ = smoothstep(-4.0, 4.0, p.z + (landNoise(p.xz * 0.07 + 23.0) - 0.5) * 7.0);
    vec3 season = mix(mix(${grounds[0]}, ${grounds[1]}, seasonX), mix(${grounds[2]}, ${grounds[3]}, seasonX), seasonZ);
    float garden = exp(-max(0.0, max(abs(p.x) - ${opts.halfX.toFixed(4)}, abs(p.z) - ${opts.halfZ.toFixed(4)})) * 0.07);
    color = mix(color, season, garden * 0.62);` : '';
  const surface = w.time(w.own(makeContinuousSurface({
    name: `${opts.name}Terrain`,
    seed: ctx.seed,
    colorA: opts.ground,
    colorB: opts.grass,
    height: field.glsl,
    follow: false,
    detail: `
      float grass = sin(p.x * 1.7 + sin(p.z * 1.1)) * sin(p.z * 2.3);
      ${groundDetail}
      color *= 0.96 + 0.045 * grass;
    `,
  })));
  w.add(surface.mesh);
  const material = w.own(oakMaterial({ vertexColors: true, roughness: 0.9, flatShading: true,
    rimColor: opts.ground, rimStrength: 0.3, wind: { amplitude: 0.03, frequency: 0.8, height: 1.7 }, name: `${opts.name}Trees` }));
  material.customProgramCacheKey = () => `${opts.name}|grove-colors`;
  const compile = material.onBeforeCompile;
  material.onBeforeCompile = (shader, renderer) => {
    compile.call(material, shader, renderer);
    shader.uniforms.uGroveColors = { value: Array.from({ length: 4 }, (_, i) => new THREE.Color(opts.trees[i % opts.trees.length] as THREE.ColorRepresentation)) };
    shader.vertexShader = 'varying vec2 vGroveWorld; varying float vGroveCanopy;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vGroveWorld = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xz;
      vGroveCanopy = step(color.r, color.g * 1.35);
    `);
    shader.fragmentShader = 'uniform vec3 uGroveColors[4]; varying vec2 vGroveWorld; varying float vGroveCanopy;\n' + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      vec2 season = smoothstep(vec2(-6.0), vec2(6.0), vGroveWorld + sin(vGroveWorld.yx * 0.17) * 2.5);
      vec3 grove = mix(mix(uGroveColors[0], uGroveColors[1], season.x), mix(uGroveColors[2], uGroveColors[3], season.x), season.y);
      diffuseColor.rgb *= mix(vec3(1.0), grove, vGroveCanopy);
    `);
  };
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


