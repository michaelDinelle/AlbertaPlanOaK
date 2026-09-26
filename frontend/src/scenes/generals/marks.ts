import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, fract, fwidth, max, min, mix, positionWorld, select, sin, uniform, vec2, vec3 } from 'three/tsl';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { coverOf, flat } from './board';

type N<T extends string> = THREE.Node<T>;
type Level = THREE.UniformNode<'float', number>;

export const SHOCK_STEPS = 14;
export const MARCH_STEPS = 6;

export interface Marks {
  mesh: THREE.Mesh;
  cursor: THREE.UniformNode<'vec2', THREE.Vector2>;
  mode: Level;
  order: THREE.UniformNode<'vec4', THREE.Vector4>;
  ordered: Level;
  march: THREE.UniformNode<'vec4', THREE.Vector4>;
  marched: Level;
  held: Level;
  jump: THREE.UniformNode<'vec4', THREE.Vector4>;
  jumped: Level;
  home: THREE.UniformNode<'vec2', THREE.Vector2>;
  homed: Level;
  fallen: THREE.UniformNode<'vec2', THREE.Vector2>;
  fall: Level;
  front: Level;
  shock: THREE.Color;
  time: Level;
  dispose(): void;
}

const heading = (dir: N<'float'>): N<'vec2'> =>
  select(dir.lessThan(1.5), vec2(0, select(dir.lessThan(0.5), float(1), float(-1))), vec2(select(dir.lessThan(2.5), float(-1), float(1)), 0));

function arrow(p: N<'vec2'>, from: N<'vec2'>, dir: N<'vec2'>, reach: N<'float'> | number, width: number): N<'float'> {
  const tip = from.add(dir.mul(reach)), back = tip.sub(dir.mul(0.26)), across = vec2(dir.y.negate(), dir.x);
  const edge = (a: N<'vec2'>, b: N<'vec2'>) => {
    const e = b.sub(a);
    return p.sub(a).dot(vec2(e.y, e.x.negate()).div(max(e.length(), 1e-4)));
  };
  const b = back.add(across.mul(0.2)), c = back.sub(across.mul(0.2));
  return min(sdSegment(p, from.add(dir.mul(0.14)), back, width), max(max(edge(tip, b), edge(b, c)), edge(c, tip)));
}

export function marksMaterial(side: number, accent: number): Marks {
  const cursor = uniform(new THREE.Vector2()), mode = uniform(0);
  const order = uniform(new THREE.Vector4()), ordered = uniform(0);
  const march = uniform(new THREE.Vector4()), marched = uniform(-1), held = uniform(0);
  const jump = uniform(new THREE.Vector4()), jumped = uniform(-1);
  const home = uniform(new THREE.Vector2()), homed = uniform(0);
  const fallen = uniform(new THREE.Vector2()), fall = uniform(-1), front = uniform(-1e3), time = uniform(0);
  const shock = new THREE.Color(), shockTint = uniform(shock) as unknown as N<'vec3'>;
  const hue = rgb(accent), white = vec3(1.25, 1.22, 1.15), ink = rgb(0x0d0f14), glow = hue.mul(1.35);

  const p = vec2(positionWorld.x, positionWorld.z.negate()), cover = coverOf(fwidth(positionWorld.x));
  const onBoard = cover(max(abs(p.x), abs(p.y)).sub(side / 2));
  const point = (v: N<'vec2'>) => vec2(v.x, v.y.negate());
  const q = p.sub(point(cursor));

  const plus = min(sdBox(q, vec2(1.5, 0.5), 0.04), sdBox(q, vec2(0.5, 1.5), 0.04));
  const looking = mode.lessThan(0.5), halving = select(mode.greaterThan(1.5), float(1), float(0));
  const frame = abs(sdBox(q, vec2(0.53), 0.08)).sub(0.04);
  const reticle = select(looking, select(abs(q.x).greaterThan(0.24).and(abs(q.y).greaterThan(0.24)), frame, float(9)), frame.sub(0.018));
  const half = (at: N<'vec2'>, radius: number) => {
    const d = sdCircle(p.sub(at), radius);
    return { disc: d, split: max(d, p.x.sub(at.x)) };
  };
  const badge = half(point(cursor).add(vec2(0.44, 0.44)), 0.18);

  const source = point(vec2(order.x, order.y)), dir = heading(order.z), split = select(order.w.greaterThan(0.5), float(1), float(0)).mul(ordered);
  const pending = arrow(p, source, dir, 0.92, 0.06), pendingHalf = half(source.add(dir.mul(0.52)), 0.13);

  const mFrom = point(vec2(march.x, march.y)), mDir = point(vec2(march.z, march.w)).sub(mFrom), t = clamp(marched.div(MARCH_STEPS), 0, 1);
  const stride = select(held.greaterThan(0.5), sin(t.mul(Math.PI)).mul(0.22), t.mul(0.55));
  const marching = arrow(p, mFrom.add(mDir.mul(stride)), mDir, 0.92, 0.06);
  const marchAlpha = select(marched.greaterThanEqual(0), t.oneMinus().pow(1.5), float(0));
  const mid = mFrom.add(mDir.mul(0.5));
  const cross = min(sdSegment(p, mid.add(vec2(-0.13, -0.13)), mid.add(vec2(0.13, 0.13)), 0.04), sdSegment(p, mid.add(vec2(-0.13, 0.13)), mid.add(vec2(0.13, -0.13)), 0.04));

  const trail = sdSegment(p, point(vec2(jump.x, jump.y)), point(vec2(jump.z, jump.w)), 0.05);
  const jumpAlpha = select(jumped.greaterThanEqual(0), exp(jumped.mul(-0.9)), float(0));

  const pulse = fract(time.mul(0.55));
  const ring = abs(p.sub(point(home)).length().sub(pulse.mul(0.9).add(0.55))).sub(0.03);

  const f = p.sub(point(fallen)), crack = clamp(fall.div(SHOCK_STEPS), 0, 1), spread = crack.oneMinus().pow(2).oneMinus();
  const wave = min(abs(f.length().sub(spread.mul(3).add(0.3))).sub(crack.oneMinus().mul(0.11).add(0.02)), abs(f.length().sub(spread.mul(1.7).add(0.2))).sub(crack.oneMinus().mul(0.05).add(0.01)));
  const shown = select(fall.greaterThanEqual(0), crack.oneMinus().pow(1.2), float(0)).mul(cover(front.sub(p.x.sub(p.y).add(side))));

  let color: N<'vec3'> = vec3(0);
  let alpha: N<'float'> = float(0);
  const blend = (strength: N<'float'>, paint: N<'vec3'>) => {
    color = mix(color, paint, strength);
    alpha = max(alpha, strength);
  };
  const layer = (shape: N<'float'>, paint: N<'vec3'>, strength: N<'float'>, outline = 0.03) => {
    blend(cover(shape.sub(outline)).mul(strength).mul(0.8), ink);
    blend(cover(shape).mul(strength), paint);
  };
  blend(cover(plus).mul(onBoard).mul(0.1), white);
  blend(cover(abs(plus).sub(0.022)).mul(onBoard).mul(0.75), vec3(0.85, 0.93, 1.05));
  layer(ring, glow, homed.mul(pulse.oneMinus().pow(2)).mul(0.7), 0.02);
  layer(sdBox(f, vec2(0.46), 0.08), shockTint.mul(1.5), shown.mul(clamp(float(1).sub(crack.mul(3)), 0, 1)).mul(0.55));
  layer(wave, shockTint.mul(1.6), shown, 0.025);
  layer(trail, white, jumpAlpha.mul(0.6), 0.02);
  layer(marching, select(held.greaterThan(0.5), rgb(0x9aa3b5), glow), marchAlpha);
  layer(cross, rgb(0xe8ecf4), marchAlpha.mul(held));
  layer(pending, glow, ordered);
  layer(pendingHalf.disc, hue.mul(0.3), split, 0.025);
  blend(cover(pendingHalf.split).mul(split), white);
  layer(reticle, select(looking, white, glow), float(1));
  layer(badge.disc, hue.mul(0.3), halving, 0.025);
  blend(cover(badge.split).mul(halving), white);

  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(side + 8, side + 8), material);
  mesh.renderOrder = 6;
  mesh.frustumCulled = false;
  return {
    mesh, cursor, mode, order, ordered, march, marched, held, jump, jumped, home, homed, fallen, fall, front, shock, time,
    dispose() {
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}
