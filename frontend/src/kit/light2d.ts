import * as THREE from 'three/webgpu';
import { Break, Fn, If, Loop, clamp, cos, float, floor, length, max, mix, mod, screenCoordinate, screenUV, select, sin, texture, uniform, uvec2, vec2, vec3, vec4 } from 'three/tsl';
import type { SceneSize } from './types';

type N<T extends string> = THREE.Node<T>;

export const LIGHT_LAYER = 30;
const LIGHT_SCALE = 0.5;
const PROBE = 2;
const MARCH = 32;
const FAR = -16384;
const NEIGHBOURS = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] as const;

export interface Light2D {
  lit(albedo: N<'vec3'>): N<'vec3'>;
  emitter(geometry: THREE.BufferGeometry, tint: N<'vec3'>): THREE.Mesh;
  occluder(geometry: THREE.BufferGeometry): THREE.Mesh;
  setEnabled(on: boolean): void;
  resize(size: SceneSize): void;
  render(): void;
  dispose(): void;
}

const target = (width: number, height: number, filter: THREE.MagnificationTextureFilter) =>
  new THREE.RenderTarget(width, height, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: filter, magFilter: filter });

const intervalStart = (cascade: number) => PROBE * (4 ** cascade - 1) / 3;

export function makeLight2D(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, ambient: THREE.Color): Light2D {
  const eye = new THREE.OrthographicCamera();
  const quad = new THREE.QuadMesh();
  let width = 1, height = 1, cascades = 1, enabled = true;
  let field = target(1, 1, THREE.NearestFilter), seedRead = target(1, 1, THREE.NearestFilter), seedWrite = target(1, 1, THREE.NearestFilter);
  let upper = target(1, 1, THREE.NearestFilter), lower = target(1, 1, THREE.NearestFilter), fluence = target(1, 1, THREE.LinearFilter);

  const size = uniform(new THREE.Vector2(1, 1)), span = uniform(new THREE.Vector2(1, 1));
  const stride = uniform(1), block = uniform(PROBE), start = uniform(0), reach = uniform(PROBE), top = uniform(1), shading = uniform(1);
  const glow = uniform(ambient.clone());
  const fieldTexture = texture(field.texture), seedTexture = texture(seedRead.texture), cascadeTexture = texture(upper.texture), irradiance = texture(fluence.texture);

  const inside = (p: N<'vec2'>) => p.x.greaterThanEqual(0).and(p.y.greaterThanEqual(0)).and(p.x.lessThan(size.x)).and(p.y.lessThan(size.y));
  const texelOf = (p: N<'vec2'>) => uvec2(clamp(floor(p), vec2(0), size.sub(1)));

  const seed = new THREE.NodeMaterial();
  seed.fragmentNode = Fn(() => {
    const cell = floor(screenCoordinate.xy);
    return select(fieldTexture.load(uvec2(cell)).a.greaterThan(0.5), vec4(cell.add(0.5), 0, 1), vec4(FAR, FAR, 0, 1));
  })();

  const jump = new THREE.NodeMaterial();
  jump.fragmentNode = Fn(() => {
    const cell = floor(screenCoordinate.xy), centre = cell.add(0.5);
    const best = vec2(FAR, FAR).toVar(), nearest = float(1e9).toVar();
    for (const [x, y] of NEIGHBOURS) {
      const at = cell.add(vec2(x, y).mul(stride)), found = seedTexture.load(texelOf(at)).xy, distance = length(found.sub(centre));
      If(inside(at).and(distance.lessThan(nearest)), () => {
        nearest.assign(distance);
        best.assign(found);
      });
    }
    return vec4(best, 0, 1);
  })();

  const gather = (probe: N<'vec2'>, direction: N<'float'>, edge: N<'float'>) => {
    let sum: N<'vec3'> = vec3(0);
    for (let k = 0; k < 4; k++) {
      const child = direction.mul(4).add(k);
      sum = sum.add(cascadeTexture.load(uvec2(probe.mul(edge).add(vec2(mod(child, edge), floor(child.div(edge)))))).rgb);
    }
    return sum.mul(0.25);
  };

  const bilinear = (position: N<'vec2'>, edge: N<'float'>, direction: (probe: N<'vec2'>) => N<'vec3'>) => {
    const grid = position.div(edge).sub(0.5), base = floor(grid), f = grid.sub(base), last = span.div(edge).sub(1);
    const at = (x: number, y: number) => direction(clamp(base.add(vec2(x, y)), vec2(0), last));
    return mix(mix(at(0, 0), at(1, 0), f.x), mix(at(0, 1), at(1, 1), f.x), f.y);
  };

  const cascade = new THREE.NodeMaterial();
  cascade.fragmentNode = Fn(() => {
    const cell = floor(screenCoordinate.xy), probe = floor(cell.div(block)), local = cell.sub(probe.mul(block));
    const direction = local.y.mul(block).add(local.x), angle = direction.add(0.5).div(block.mul(block)).mul(2 * Math.PI);
    const heading = vec2(cos(angle), sin(angle)), origin = probe.add(0.5).mul(block);
    const radiance = vec3(0).toVar(), open = float(1).toVar(), t = start.toVar();
    Loop(MARCH, () => {
      If(t.greaterThanEqual(start.add(reach)), () => { Break(); });
      const p = origin.add(heading.mul(t));
      If(inside(p).not(), () => { Break(); });
      const texel = texelOf(p), distance = length(seedTexture.load(texel).xy.sub(p));
      If(distance.lessThan(0.75), () => {
        radiance.assign(fieldTexture.load(texel).rgb);
        open.assign(0);
        Break();
      });
      t.addAssign(max(distance.sub(0.5), 0.5));
    });
    If(top.lessThan(0.5).and(open.greaterThan(0)), () => {
      const edge = block.mul(2);
      radiance.addAssign(bilinear(origin, edge, (upperProbe) => gather(upperProbe, direction, edge)));
    });
    return vec4(radiance, 1);
  })();

  const resolve = new THREE.NodeMaterial();
  resolve.fragmentNode = Fn(() => {
    const centre = floor(screenCoordinate.xy).add(0.5), edge = float(PROBE);
    return vec4(bilinear(centre, edge, (probe) => {
      let sum: N<'vec3'> = vec3(0);
      for (let k = 0; k < 4; k++) sum = sum.add(cascadeTexture.load(uvec2(probe.mul(edge).add(vec2(k % 2, Math.floor(k / 2))))).rgb);
      return sum.mul(0.25);
    }), 1);
  })();

  const pass = (material: THREE.NodeMaterial, output: THREE.RenderTarget) => {
    quad.material = material;
    renderer.setRenderTarget(output);
    quad.render(renderer);
  };

  const proxy = (geometry: THREE.BufferGeometry, color: N<'vec3'>) => {
    const material = new THREE.MeshBasicNodeMaterial({ fog: false });
    material.colorNode = color;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.layers.set(LIGHT_LAYER);
    return mesh;
  };

  return {
    lit: (albedo) => albedo.mul(mix(vec3(1), irradiance.sample(screenUV).rgb.add(glow), shading)),
    emitter: (geometry, tint) => proxy(geometry, tint),
    occluder: (geometry) => proxy(geometry, vec3(0)),
    setEnabled(on) {
      enabled = on;
      shading.value = on ? 1 : 0;
    },
    resize({ width: w, height: h, dpr }) {
      width = Math.max(1, Math.round(w * dpr * LIGHT_SCALE));
      height = Math.max(1, Math.round(h * dpr * LIGHT_SCALE));
      cascades = Math.max(1, Math.ceil(Math.log(3 * Math.hypot(width, height) / PROBE + 1) / Math.log(4)));
      const tile = PROBE * 2 ** (cascades - 1), paddedWidth = Math.ceil(width / tile) * tile, paddedHeight = Math.ceil(height / tile) * tile;
      for (const item of [field, seedRead, seedWrite, upper, lower, fluence]) item.dispose();
      field = target(width, height, THREE.NearestFilter);
      seedRead = target(width, height, THREE.NearestFilter);
      seedWrite = target(width, height, THREE.NearestFilter);
      upper = target(paddedWidth, paddedHeight, THREE.NearestFilter);
      lower = target(paddedWidth, paddedHeight, THREE.NearestFilter);
      fluence = target(width, height, THREE.LinearFilter);
      fieldTexture.value = field.texture;
      irradiance.value = fluence.texture;
      size.value.set(width, height);
      span.value.set(paddedWidth, paddedHeight);
    },
    render() {
      if (!enabled) return;
      const previous = renderer.getRenderTarget(), clear = renderer.getClearColor(new THREE.Color()), alpha = renderer.getClearAlpha(), background = scene.background;
      eye.copy(camera as THREE.OrthographicCamera);
      eye.layers.set(LIGHT_LAYER);
      scene.background = null;
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(field);
      renderer.clear();
      renderer.render(scene, eye);
      scene.background = background;
      pass(seed, seedWrite);
      [seedRead, seedWrite] = [seedWrite, seedRead];
      for (let step = 2 ** Math.floor(Math.log2(Math.max(width, height))); step >= 1; step /= 2) {
        stride.value = step;
        seedTexture.value = seedRead.texture;
        pass(jump, seedWrite);
        [seedRead, seedWrite] = [seedWrite, seedRead];
      }
      seedTexture.value = seedRead.texture;
      for (let i = cascades - 1; i >= 0; i--) {
        block.value = PROBE * 2 ** i;
        start.value = intervalStart(i);
        reach.value = PROBE * 4 ** i;
        top.value = i === cascades - 1 ? 1 : 0;
        cascadeTexture.value = upper.texture;
        pass(cascade, lower);
        [upper, lower] = [lower, upper];
      }
      cascadeTexture.value = upper.texture;
      pass(resolve, fluence);
      renderer.setClearColor(clear, alpha);
      renderer.setRenderTarget(previous);
    },
    dispose() {
      for (const item of [field, seedRead, seedWrite, upper, lower, fluence, seed, jump, cascade, resolve]) item.dispose();
    },
  };
}
