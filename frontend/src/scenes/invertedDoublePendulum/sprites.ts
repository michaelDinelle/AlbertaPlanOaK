import * as THREE from 'three/webgpu';
import { abs, attribute, exp, float, floor, fract, length, max, min, mix, positionGeometry, positionWorld, round, select, smoothstep, uv, vec2 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, fractal, grain, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Paint = (p: N<'vec2'>) => readonly [N<'vec3'>, N<'float'>];
export type Level = THREE.UniformNode<'float', number>;
export type Point = THREE.UniformNode<'vec2', THREE.Vector2>;

export const PALETTE = {
  void: 0x040409,
  board: 0x15141d,
  minor: 0x2e2c40,
  major: 0x46425f,
  steel: 0xb4bac8,
  gleam: 0xf2f4fa,
  alloy: 0x6a7080,
  anodized: 0x2a2c37,
  deep: 0x0c0d13,
  rubber: 0x4a4352,
  tube: 0x8d93a4,
  weave: 0xa9aebd,
  amber: 0xffc27a,
  push: 0x5ec1ff,
  ember: 0xff6a4f,
  spark: 0xffd9c4,
  warn: 0xf2b134,
} as const;

export const ROD = 0.016;
export const CARRIAGE = { half: 0.19, y: 0.004, height: 0.066, radius: 0.022 } as const;
export const BEAM = { top: -0.104, bottom: -0.16 } as const;
export const BLOCK = { width: 0.075, top: 0.054 } as const;
export const PAD = { width: 0.1, half: 0.036 } as const;
export const LINK = 0.019;
export const HUB = { pivot: 0.046, elbow: 0.038, tip: 0.028 } as const;
export const LED = { elbow: 0.014, tip: 0.017, glow: 0.034 } as const;
const LEG = { at: 0.62, half: 0.018, drop: 3 } as const;
const PUSH = { gap: 0.03, reach: 0.36, head: 0.055, half: 0.012 } as const;

export interface RailSpan { range: number; face: number; outer: number; end: number }

export const flat = (width: number, height: number): THREE.PlaneGeometry => new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);

export function sprite(x0: number, y0: number, x1: number, y1: number, order: number, paint: Paint): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const [color, alpha] = paint(vec2(positionGeometry.x, positionGeometry.z.negate()));
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(x1 - x0, y1 - y0).translate((x0 + x1) / 2, 0, -(y0 + y1) / 2), material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

const band = (value: N<'float'>, from: number, to: number): N<'float'> => coverage(max(value.sub(to), float(from).sub(value)));

export function board(lit: Lit, reach: number, accent: number): Paint {
  return () => {
    const q = vec2(positionWorld.x, positionWorld.z.negate());
    const lines = (spacing: number): N<'float'> => {
      const g = abs(fract(q.div(spacing).add(0.5)).sub(0.5)).mul(spacing);
      return min(g.x, g.y);
    };
    const mottle = fractal(q.mul(1.6), 4).mul(0.16).add(1), speck = grain(floor(q.mul(700))).x.mul(0.14).add(0.93);
    let color: N<'vec3'> = rgb(PALETTE.board).mul(mottle.mul(speck));
    color = mix(color, rgb(PALETTE.minor), coverage(lines(0.1).sub(0.0015)));
    color = mix(color, rgb(PALETTE.major), coverage(lines(0.5).sub(0.0024)));
    const dash = coverage(abs(fract(q.x.div(0.05)).sub(0.5)).sub(0.2).mul(0.05));
    color = mix(color, rgb(accent).mul(0.5), coverage(abs(q.y.sub(reach)).sub(0.002)).mul(dash));
    return [lit(color), float(1)];
  };
}

export function rail(lit: Lit, span: RailSpan, accent: number): Paint {
  const { range, outer, end } = span, beamY = (BEAM.top + BEAM.bottom) / 2, beamH = (BEAM.top - BEAM.bottom) / 2;
  const blockX = outer + BLOCK.width / 2, blockY = (BLOCK.top + BEAM.bottom) / 2, blockH = (BLOCK.top - BEAM.bottom) / 2;
  return (p) => {
    const m = vec2(abs(p.x), p.y);
    const rod = sdSegment(p, vec2(-outer, 0), vec2(outer, 0), ROD);
    const beam = sdBox(p.sub(vec2(0, beamY)), vec2(end, beamH), 0.006);
    const block = sdBox(m.sub(vec2(blockX, blockY)), vec2(BLOCK.width / 2, blockH), 0.008);
    const leg = sdBox(m.sub(vec2(range * LEG.at, BEAM.bottom - LEG.drop / 2)), vec2(LEG.half, LEG.drop / 2), 0.004);
    const bolts = min(sdCircle(m.sub(vec2(blockX, BLOCK.top - 0.022)), 0.008), sdCircle(m.sub(vec2(blockX, beamY)), 0.008));
    const collar = sdBox(m.sub(vec2(outer - 0.008, 0)), vec2(0.008, ROD + 0.006), 0.003);
    const minor = abs(p.x.sub(round(p.x.div(0.1)).mul(0.1))), major = abs(p.x.sub(round(p.x.div(0.5)).mul(0.5)));
    const within = coverage(abs(p.x).sub(range + 0.002));
    const ticks = max(coverage(minor.sub(0.0014)).mul(band(p.y, BEAM.top - 0.014, BEAM.top)), coverage(major.sub(0.002)).mul(band(p.y, BEAM.top - 0.026, BEAM.top))).mul(within);
    const stops = coverage(abs(m.x.sub(range)).sub(0.0026)).mul(band(p.y, BEAM.bottom + 0.008, BEAM.top));
    const fall = smoothstep(-1.4, -0.2, p.y);
    const tone = smoothstep(-ROD, ROD, p.y), shine = coverage(abs(p.y.sub(ROD * 0.45)).sub(0.0024));
    let color: N<'vec3'> = rgb(PALETTE.anodized).mul(fall.mul(0.8).add(0.2));
    color = mix(color, rgb(PALETTE.anodized).mul(smoothstep(BEAM.bottom, BEAM.top, p.y).mul(0.5).add(0.75)), coverage(beam));
    color = mix(color, rgb(PALETTE.alloy).mul(0.9), coverage(abs(p.y.sub(BEAM.top - 0.004)).sub(0.0016)).mul(coverage(beam)));
    color = mix(color, rgb(PALETTE.steel).mul(0.85), ticks);
    color = mix(color, rgb(PALETTE.warn), stops);
    color = mix(color, rgb(PALETTE.alloy).mul(smoothstep(BEAM.bottom, BLOCK.top, p.y).mul(0.4).add(0.6)), coverage(block));
    color = mix(color, rgb(PALETTE.steel), coverage(bolts));
    color = mix(color, rgb(PALETTE.deep), coverage(abs(bolts).sub(0.0012)).mul(coverage(block)));
    color = mix(color, mix(rgb(PALETTE.steel).mul(0.4), rgb(PALETTE.steel), tone).add(rgb(PALETTE.gleam).mul(shine.mul(0.8))), coverage(rod));
    color = mix(color, rgb(accent).mul(0.7), coverage(collar));
    return [lit(color), coverage(min(min(min(rod, beam), min(block, leg)), collar))];
  };
}

export function bumper(lit: Lit, span: RailSpan, side: number, squeeze: Level, flare: Level): Paint {
  return (p) => {
    const x = p.x.mul(side), inner = squeeze.add(span.face), centre = inner.add(span.outer).mul(0.5), half = float(span.outer).sub(inner).mul(0.5);
    const pad = sdBox(vec2(x.sub(centre), p.y), vec2(half, PAD.half), 0.012);
    const rib = coverage(abs(fract(x.sub(inner).div(half.mul(0.5))).sub(0.5)).sub(0.36).mul(half.mul(0.5)));
    const face = smoothstep(0.03, 0, x.sub(inner));
    const shade = smoothstep(-PAD.half, PAD.half, p.y).mul(0.6).add(0.6).mul(rib.mul(-0.3).add(1));
    const color = lit(rgb(PALETTE.rubber).mul(shade)).add(rgb(PALETTE.ember).mul(flare.mul(face).mul(4)));
    return [color, coverage(pad)];
  };
}

export function carriage(lit: Lit, accent: number): Paint {
  const { half, y, height, radius } = CARRIAGE, bottom = y - height;
  return (p) => {
    const m = vec2(abs(p.x), p.y);
    const body = sdBox(p.sub(vec2(0, y)), vec2(half, height), radius);
    const plate = sdBox(p.sub(vec2(0, y + height - 0.015)), vec2(half - 0.016, 0.011), 0.005);
    const housing = sdBox(m.sub(vec2(half - 0.043, y - 0.006)), vec2(0.03, height - 0.022), 0.01);
    const bolts = sdCircle(vec2(m.x.sub(half - 0.043), abs(p.y.sub(y - 0.006)).sub(0.026)), 0.0072);
    const needle = sdSegment(p, vec2(0, bottom + 0.004), vec2(0, BEAM.top + 0.003), 0.0032);
    const nib = sdBox(p.sub(vec2(0, bottom - 0.004)), vec2(0.012, 0.007), 0.003);
    const rise = smoothstep(bottom, y + height, p.y);
    let color: N<'vec3'> = mix(rgb(PALETTE.alloy).mul(0.55), rgb(PALETTE.alloy).mul(1.2), rise);
    color = mix(color, rgb(PALETTE.anodized).mul(1.4), coverage(housing));
    color = mix(color, rgb(PALETTE.deep), coverage(abs(housing).sub(0.0012)));
    color = mix(color, rgb(PALETTE.steel), coverage(bolts));
    color = mix(color, rgb(accent).mul(mix(float(0.75), float(1.15), rise)), coverage(plate));
    color = mix(color, rgb(PALETTE.deep), coverage(abs(body).sub(0.0014)).mul(coverage(body)));
    color = mix(color, rgb(accent), coverage(min(needle, nib)).mul(coverage(body).oneMinus()));
    return [lit(color), coverage(min(body, min(needle, nib)))];
  };
}

export function link(lit: Lit, span: number): Paint {
  return (p) => {
    const tube = sdSegment(p, vec2(0, 0), vec2(span, 0), LINK);
    const sleeve = min(sdBox(p.sub(vec2(0.06, 0)), vec2(0.028, LINK + 0.004), 0.004), sdBox(p.sub(vec2(span - 0.06, 0)), vec2(0.028, LINK + 0.004), 0.004));
    const brush = grain(floor(vec2(p.x.mul(900), p.y.mul(60)))).x.mul(0.1).add(0.95);
    const shade = smoothstep(-LINK, LINK, p.y);
    let color: N<'vec3'> = mix(rgb(PALETTE.tube).mul(0.55), rgb(PALETTE.weave), shade).mul(brush);
    color = color.add(rgb(PALETTE.gleam).mul(coverage(abs(p.y.sub(LINK * 0.5)).sub(0.0022)).mul(0.07)));
    color = mix(color, rgb(PALETTE.steel).mul(shade.mul(0.6).add(0.45)), coverage(sleeve));
    return [lit(color), coverage(min(tube, sleeve))];
  };
}

export function hub(lit: Lit, radius: number, cap: number): Paint {
  return (p) => {
    const r = length(p);
    let color: N<'vec3'> = mix(rgb(PALETTE.steel).mul(0.5), rgb(PALETTE.steel), smoothstep(-radius, radius, p.y));
    color = mix(color, rgb(PALETTE.deep), coverage(abs(r.sub(radius * 0.68)).sub(0.0018)));
    color = mix(color, rgb(cap), coverage(r.sub(radius * 0.5)));
    color = mix(color, rgb(PALETTE.deep), coverage(abs(r.sub(radius)).sub(0.0016)));
    return [lit(color), coverage(r.sub(radius))];
  };
}

export function led(radius: number, tint: number): Paint {
  return (p) => {
    const r = length(p), hot = smoothstep(radius, 0, r);
    const color = mix(rgb(tint), rgb(PALETTE.gleam), hot.mul(hot).mul(0.8)).mul(2.8);
    return [color, coverage(r.sub(radius))];
  };
}

export function push(force: Level, flow: Level): Paint {
  return (p) => {
    const dir = select(force.lessThan(0), float(-1), float(1)), size = abs(force).mul(PUSH.reach);
    const q = vec2(p.x.mul(dir).sub(CARRIAGE.half + PUSH.gap), p.y.sub(CARRIAGE.y)), head = min(size, PUSH.head), stem = size.sub(head);
    const shaft = sdBox(q.sub(vec2(stem.mul(0.5), 0)), vec2(stem.mul(0.5), PUSH.half), 0.004);
    const wedge = max(stem.sub(q.x), abs(q.y).mul(head).add(q.x.sub(size).mul(PUSH.half * 2.4)).div(length(vec2(head, PUSH.half * 2.4))));
    const arrow = min(shaft, wedge);
    const pulse = fract(q.x.div(0.05).sub(flow.mul(3))).mul(0.45).add(0.75);
    return [rgb(PALETTE.push).mul(pulse.mul(1.6)), coverage(arrow).mul(smoothstep(0.004, 0.02, size))];
  };
}

export function ghost(elbow: Point, tip: Point, fade: Level): Paint {
  return (p) => {
    const d = min(sdSegment(p, vec2(0, 0), elbow, LINK), sdSegment(p, elbow, tip, LINK));
    const ring = sdCircle(p.sub(tip), HUB.tip);
    const shape = min(d, ring);
    return [rgb(PALETTE.ember).mul(1.1), max(coverage(abs(shape).sub(0.0018)).mul(0.75), coverage(shape).mul(0.08)).mul(fade)];
  };
}

export interface Ribbon {
  mesh: THREE.Mesh;
  positions: Float32Array;
  ages: Float32Array;
  quads: Uint32Array;
  commit(indices: number): void;
}

export function ribbon(capacity: number, tint: number, gain: number, layer: number): Ribbon {
  const positions = new Float32Array(capacity * 6), side = new Float32Array(capacity * 2), ages = new Float32Array(capacity * 2), indices = new Uint32Array(Math.max(1, capacity - 1) * 6);
  for (let i = 0; i < capacity; i++) {
    side[2 * i] = 1;
    side[2 * i + 1] = -1;
  }
  const geometry = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage), age = new THREE.BufferAttribute(ages, 1).setUsage(THREE.DynamicDrawUsage), index = new THREE.BufferAttribute(indices, 1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', position);
  geometry.setAttribute('side', new THREE.BufferAttribute(side, 1));
  geometry.setAttribute('age', age);
  geometry.setIndex(index);
  geometry.setDrawRange(0, 0);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.MaxEquation });
  const t = attribute<'float'>('age', 'float'), s = abs(attribute<'float'>('side', 'float'));
  const fade = exp(t.mul(-4.2)).sub(exp(float(-4.2)).mul(t));
  material.colorNode = mix(rgb(tint), rgb(PALETTE.gleam), smoothstep(0.4, 0, s).mul(fade).mul(0.3)).mul(smoothstep(1, 0.25, s).mul(fade).mul(gain));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = layer;
  mesh.frustumCulled = false;
  return {
    mesh, positions, ages, quads: indices,
    commit(count) {
      geometry.setDrawRange(0, count);
      position.needsUpdate = true;
      age.needsUpdate = true;
      index.needsUpdate = true;
    },
  };
}

export interface Sparks {
  mesh: THREE.InstancedMesh;
  level: THREE.InstancedBufferAttribute;
}

export function sparks(capacity: number, size: number, layer: number): Sparks {
  const geometry = flat(size, size), level = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aSpark', level);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });
  const r = length(uv().sub(0.5)).mul(2), a = attribute<'float'>('aSpark', 'float');
  material.colorNode = mix(rgb(PALETTE.ember), rgb(PALETTE.spark), smoothstep(0.3, 0, r)).mul(2);
  material.opacityNode = a.mul(max(exp(r.mul(r).mul(-14)), coverage(r.sub(0.2))));
  const mesh = instancedMesh(geometry, material, capacity);
  mesh.count = 0;
  mesh.renderOrder = layer;
  mesh.frustumCulled = false;
  return { mesh, level };
}
