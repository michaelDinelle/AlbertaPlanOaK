import * as THREE from 'three/webgpu';
import { Fn, abs, atan, clamp, dot, exp, float, floor, fwidth, length, max, min, mix, positionWorld, round, sin, smoothstep, texture, uniform, vec2, vec3 } from 'three/tsl';
import { fractal, noise, rgb, sdBox, sdCircle } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;

export const AMBER = 0xe8822e;
export const VIOLET = 0x8a6ae8;
export const HALO = 0xffcf5c;
export const LANTERN = 0xffe0a0;
export const HEDGEHOG = 0x8a5a3a;

export interface Floor {
  material: THREE.MeshBasicNodeMaterial;
  map: THREE.DataTexture;
  data: Uint8Array;
  t: { value: number };
  agent: { value: THREE.Vector2 };
  dir: { value: THREE.Vector2 };
  good: { value: number };
  time: { value: number };
}

const ellipse = (p: N<'vec2'>, r: N<'vec2'>): N<'float'> => length(p.div(r)).sub(1).mul(min(r.x, r.y));

export function floorMaterial(side: number, fov: number, margin: number, scale: number): Floor {
  const data = new Uint8Array(side * side * 4);
  const map = new THREE.DataTexture(data, side, side, THREE.RGBAFormat, THREE.UnsignedByteType);
  map.magFilter = THREE.NearestFilter;
  map.minFilter = THREE.NearestFilter;
  map.generateMipmaps = false;
  map.needsUpdate = true;
  const t = uniform(0), agent = uniform(new THREE.Vector2()), dir = uniform(new THREE.Vector2(0, 1)), good = uniform(1), time = uniform(0);
  const S = float(side), half = (side - 1) / 2;

  const color = Fn(() => {
    const c = positionWorld.xz.add(half);
    const fw = max(length(fwidth(c)).mul(0.75), 1e-4);
    const aa = (d: N<'float'>) => clamp(float(0.5).sub(d.div(fw)), 0, 1);
    const index = floor(c.add(0.5)), l = c.sub(index);
    const wrapped = index.sub(S.mul(floor(index.div(S))));
    const cell = texture(map, wrapped.add(0.5).div(S)).mul(255 / 64).round();
    const edge = max(abs(c.x.sub(half)), abs(c.y.sub(half))).sub(side / 2), inside = aa(edge);

    const n = fractal(c.mul(0.55), 4), speck = noise(c.mul(3.1));
    let col: N<'vec3'> = mix(rgb(0x2b3b25), rgb(0x46592f), n);
    col = mix(col, rgb(0x3a2f22), smoothstep(0.62, 0.8, speck).mul(0.55));
    const border = float(0.5).sub(max(abs(l.x), abs(l.y)));
    col = col.mul(float(1).sub(clamp(float(1).sub(border.div(fw)), 0, 1).mul(0.22)));

    const o0 = c.sub(agent), o = o0.sub(S.mul(round(o0.div(S))));
    const wd = max(abs(o.x), abs(o.y)).sub(fov + 0.5);
    const window = aa(wd);
    col = col.mul(mix(float(0.72), float(1.3), window)).add(rgb(LANTERN).mul(window.mul(0.07)));
    col = col.add(rgb(LANTERN).mul(exp(max(wd, 0).mul(-1.6)).mul(0.05)));

    const itemA = cell.x, itemB = cell.y, modeA = cell.z, modeB = cell.w;
    const eaten = float(1).sub(smoothstep(0.45, 0.78, t)), wilt = float(1).sub(smoothstep(0.0, 0.85, t));
    const sA = itemA.greaterThan(0.5).select(modeA.lessThan(0.5).select(float(1), modeA.lessThan(1.5).select(eaten, wilt)), float(0));
    const sB = modeB.greaterThan(0.5).select(smoothstep(0.3, 1.0, t), float(0));
    const useA = sA.greaterThanEqual(sB);
    const item = useA.select(itemA, itemB), s = max(useA.select(sA, sB), 0.001);
    const withered = useA.and(modeA.greaterThan(1.5)).select(float(1).sub(sA), float(0));
    const amber = item.lessThan(1.5);
    const paying = amber.select(good, float(1).sub(good)).mul(s.greaterThan(0.002).select(float(1), float(0)));
    const z = s.div(0.8), m = vec2(l.x, l.y.negate()).div(z);

    col = col.add(rgb(HALO).mul(float(1).sub(smoothstep(0.05, 0.48, length(l))).mul(paying).mul(s).mul(0.75)));
    const shade = aa(ellipse(l.sub(vec2(0.03, 0.26)).div(s), vec2(0.3, 0.1)).mul(s)).mul(0.45);
    col = mix(col, col.mul(0.35), shade.mul(item.greaterThan(0.5).select(float(1), float(0))));

    const capA = max(sdCircle(m.sub(vec2(0, -0.04)), 0.3), m.y.add(0.04).negate());
    const stemA = sdBox(m.sub(vec2(0, -0.2)), vec2(0.085, 0.14), 0.04);
    const spots = min(min(sdCircle(m.sub(vec2(-0.13, 0.07)), 0.055), sdCircle(m.sub(vec2(0.11, 0.12)), 0.05)), sdCircle(m.sub(vec2(0.0, 0.2)), 0.04));
    const capB = max(ellipse(m.sub(vec2(0, 0.04)), vec2(0.4, 0.2)), m.y.sub(0.0).negate());
    const stemB = sdBox(m.sub(vec2(0, -0.15)), vec2(0.055, 0.18), 0.03);
    const rimB = ellipse(m.sub(vec2(0, 0.04)), vec2(0.4, 0.2)).abs().sub(0.025);
    const cap = amber.select(capA, capB), stem = amber.select(stemA, stemB);
    const capTint = amber.select(rgb(AMBER), rgb(VIOLET)).mul(float(0.88).add(m.y.mul(0.5)));
    const present = item.greaterThan(0.5).select(float(1), float(0));
    const stemTint = mix(rgb(0xefe2c8), rgb(0xb8a88c), smoothstep(-0.3, 0.1, m.x));
    const mushroom = mix(col, stemTint, aa(stem.mul(z)).mul(present));
    let body = mix(mushroom, capTint, aa(cap.mul(z)).mul(present));
    body = mix(body, amber.select(rgb(0xfff1d6), rgb(0xc9b8ff)), aa(amber.select(max(spots, capA), max(rimB, capB)).mul(z)).mul(present));
    col = mix(body, mix(body, rgb(0x4a4236), 0.6), withered.mul(present));

    const rim = clamp(float(1).sub(abs(wd).div(fw.mul(1.3))), 0, 1);
    col = mix(col, rgb(LANTERN), rim.mul(0.55));

    const k = o.div(scale), perp = vec2(dir.y.negate(), dir.x);
    const q = vec2(dot(k, perp), dot(k, dir));
    col = mix(col, col.mul(0.3), aa(ellipse(q.sub(vec2(0.04, -0.06)), vec2(0.4, 0.46)).mul(scale)).mul(0.5));
    const angle = atan(q.x, q.y.negate());
    const bristle = sin(angle.mul(15).add(time.mul(0.5))).mul(0.035).mul(smoothstep(-0.2, 0.25, q.y.negate()));
    const back = ellipse(q.sub(vec2(0, -0.08)), vec2(0.33, 0.37)).sub(bristle);
    const face = ellipse(q.sub(vec2(0, 0.22)), vec2(0.17, 0.22));
    const nose = sdCircle(q.sub(vec2(0, 0.43)), 0.05);
    const eyes = min(sdCircle(q.sub(vec2(-0.075, 0.27)), 0.03), sdCircle(q.sub(vec2(0.075, 0.27)), 0.03));
    const stripes = sin(angle.mul(15)).mul(0.5).add(0.5);
    col = col.add(rgb(LANTERN).mul(exp(length(o).div(scale).mul(-1.7)).mul(0.2)));
    const coat = mix(rgb(0x6b4630), rgb(HEDGEHOG), stripes.mul(0.6).add(length(q).mul(0.8)));
    col = mix(col, coat, aa(back.mul(scale)));
    col = mix(col, mix(rgb(0xd9b98c), rgb(0xf2dcb4), smoothstep(0.0, 0.4, q.y)), aa(face.mul(scale)));
    col = mix(col, rgb(0x1a120c), aa(min(nose, eyes).mul(scale)));

    const grey = dot(col, vec3(0.3, 0.55, 0.15));
    const ghost = mix(vec3(grey, grey, grey), col, 0.45).mul(0.5);
    col = mix(ghost, col, inside);
    const frame = clamp(float(1).sub(abs(edge.sub(0.04)).div(fw.mul(1.2))), 0, 1);
    col = mix(col, rgb(0x6d5a3e), frame.mul(0.85));
    const outer = max(abs(c.x.sub(half)), abs(c.y.sub(half))).sub(side / 2 + margin);
    return col.mul(float(1).sub(smoothstep(-0.9, 0, outer).mul(0.6)));
  });

  const material = new THREE.MeshBasicNodeMaterial();
  material.colorNode = color();
  return { material, map, data, t, agent, dir, good, time };
}
