import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, smoothstep, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
export type Scalar = THREE.UniformNode<'float', number>;

export const REACH = 7.5;
export const RISE = 5.4;
export const GAP = 0.45;
export const STRIP = 1.4;
export const STRIP_MID = RISE + GAP + STRIP / 2;
export const PANEL = 5.8;
export const PANEL_MID = -REACH - GAP - PANEL / 2;
export const BIN = 0.25;
export const BINS = Math.round((2 * REACH) / BIN);
export const SPAN = 6;
export const TICK = 0.2;
export const TICKS = Math.round((2 * SPAN) / TICK) + 1;
export const HEADER = 0.8;
export const ROW_TOP = RISE - HEADER;
export const ROW_FLOOR = -RISE + 0.2;
export const TRACK_MID = -0.05;
export const TRACK_HALF = 1.62;
export const METER_MID = 2.18;
export const METER_HALF = 0.46;
export const LABEL_MID = -2.3;
export const METER_LIFT = 0.3;
export const ROW_BASE = 0.78;
export const ROW_AREA = 0.6;

export const CLASS = [0x6fa8ff, 0xffb25e] as const;
export const MISS = 0xff4d64;
export const GOLD = 0xffd166;
const INPUT = 0xa9bfd0;
const FELT = 0x080c12;
const GLASS = 0x0f151c;
const GRID = 0x223140;
const BEZEL = 0x2b3a48;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const unit = (): N<'vec2'> => uv().sub(0.5).mul(2);
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(fwidth(positionWorld.x), 1e-6))), 0, 1);
const ring = (p: N<'vec2'>, radius: number, half: number): N<'float'> => abs(length(p).sub(radius)).sub(half);

export function sprite(color: N<'vec3'>, opacity: N<'float'>, order: number, geometry: THREE.BufferGeometry = flat(1, 1)): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function frame(q: N<'vec2'>, centre: THREE.Vector2, half: THREE.Vector2, radius: number) {
  const p = q.sub(vec2(centre.x, centre.y)), edge = sdBox(p, vec2(half.x, half.y), radius);
  const bevel = crisp(abs(edge.add(0.05)).sub(0.022)).mul(smoothstep(-2, 2, p.y.sub(p.x.mul(0.3))).mul(0.6).add(0.4));
  return { p, edge, bevel };
}

const PLATE = { centre: new THREE.Vector2(0, 0), half: new THREE.Vector2(REACH, RISE) };
const BAND = { centre: new THREE.Vector2(0, -STRIP_MID), half: new THREE.Vector2(REACH, STRIP / 2) };
const TABLE = { centre: new THREE.Vector2(PANEL_MID, 0), half: new THREE.Vector2(PANEL / 2, RISE) };

export function backdrop(lit: Lit): THREE.Mesh {
  const q = chart();
  const near = [PLATE, BAND, TABLE].map(({ centre, half }) => sdBox(q.sub(vec2(centre.x + 0.08, centre.y - 0.12)), vec2(half.x, half.y), 0.24)).reduce((a, b) => min(a, b));
  const felt = rgb(FELT).mul(noise(q.mul(1.3)).mul(0.1).add(1));
  const shade = smoothstep(-0.2, 0.8, near).mul(0.65).add(0.35);
  return sprite(lit(felt.mul(shade)), float(1), 0, flat(120, 80));
}

export function plate(lit: Lit, rule: THREE.Color): THREE.Mesh {
  const q = chart(), { edge, bevel } = frame(q, PLATE.centre, PLATE.half, 0.24);
  const base = rgb(GLASS).mul(noise(q.mul(3.1)).mul(0.06).add(1));
  const cell = abs(fract(q.add(0.5)).sub(0.5)), half = abs(fract(q.mul(0.5).add(0.5)).sub(0.5)).mul(2);
  const inside = crisp(edge.add(0.12));
  let color: N<'vec3'> = base;
  color = mix(color, rgb(GRID).mul(0.7), crisp(min(cell.x, cell.y).sub(0.004)).mul(inside));
  color = mix(color, rgb(GRID), crisp(min(half.x, half.y).sub(0.006)).mul(inside));
  color = mix(color, rgb(BEZEL), bevel);
  const key = crisp(max(abs(abs(q.y).sub(RISE - 0.16)).sub(0.012), abs(q.x).sub(REACH - 0.3))).mul(smoothstep(0.1, 1.2, abs(q.x)));
  color = mix(color, mix(rgb(CLASS[1]), rgb(CLASS[0]), crisp(q.x)).mul(0.55), key);
  const d = abs(q.x), line = crisp(d.sub(0.022)).mul(crisp(edge.add(0.1)));
  const halo = exp(d.negate().div(0.09)).mul(0.22).add(exp(d.negate().div(0.6)).mul(0.035)).mul(inside);
  const shaded = lit(color).add(rgb(rule).mul(halo));
  return sprite(mix(shaded, rgb(rule).mul(1.55), line), crisp(edge), 1, flat(2 * REACH + 0.2, 2 * RISE + 0.2));
}

export function band(lit: Lit, rule: THREE.Color, counts: THREE.DataTexture, marker: Scalar, shown: Scalar): THREE.Mesh {
  const q = chart(), { p, edge, bevel } = frame(q, BAND.centre, BAND.half, 0.18);
  const floorY = -STRIP / 2 + 0.2, tall = STRIP - 0.4, y = p.y.sub(floorY);
  const k = clamp(floor(q.x.add(REACH).div(BIN)), 0, BINS - 1), bin = texture(counts).load(uvec2(vec2(k, 0)));
  const across = abs(fract(q.x.add(REACH).div(BIN)).sub(0.5)).mul(BIN).sub(BIN * 0.36);
  const bar = crisp(max(max(across, y.sub(bin.g.mul(tall))), y.negate())).mul(select(bin.g.greaterThan(0), float(1), float(0)));
  const right = crisp(y.sub(bin.r.mul(tall))), side = mix(rgb(CLASS[1]), rgb(CLASS[0]), crisp(q.x));
  const inside = crisp(edge.add(0.1));
  let color: N<'vec3'> = lit(mix(rgb(GLASS), rgb(BEZEL), bevel));
  color = mix(color, rgb(GRID), crisp(abs(y).sub(0.006)).mul(inside));
  color = mix(color, mix(rgb(MISS), side, right).mul(0.85), bar.mul(inside));
  color = mix(color, rgb(rule).mul(1.55), crisp(abs(q.x).sub(0.022)).mul(crisp(edge.add(0.08))));
  const tick = crisp(abs(q.x.sub(marker)).sub(0.018)).mul(crisp(abs(p.y.sub(floorY + tall / 2)).sub(tall / 2 + 0.08))).mul(shown);
  color = mix(color, vec3(1), tick);
  return sprite(color, crisp(edge), 1, flat(2 * REACH + 0.2, STRIP + 0.2).translate(0, 0, STRIP_MID));
}

export function table(lit: Lit, data: THREE.DataTexture, rows: number): THREE.Mesh {
  const q = chart(), { p, edge, bevel } = frame(q, TABLE.centre, TABLE.half, 0.24);
  const pitch = (ROW_TOP - ROW_FLOOR) / rows, depth = float(ROW_TOP).sub(p.y).div(pitch), area = pitch * ROW_AREA;
  const r = clamp(floor(depth), 0, rows - 1), baseline = float(ROW_TOP).sub(r.add(ROW_BASE).mul(pitch));
  const y = p.y.sub(baseline), store = texture(data), info = store.load(uvec2(vec2(TICKS, r)));
  const scale = TRACK_HALF / SPAN, x = p.x.sub(TRACK_MID);
  const b = clamp(x.div(scale).add(SPAN).div(TICK), 0, TICKS - 1), k0 = floor(b), k1 = min(k0.add(1), TICKS - 1);
  const density = mix(store.load(uvec2(vec2(k0, r))).r, store.load(uvec2(vec2(k1, r))).r, fract(b)).mul(area);
  const track = crisp(abs(x).sub(TRACK_HALF)), inside = crisp(edge.add(0.1));
  const block = crisp(max(depth.negate(), depth.sub(rows)).mul(pitch).sub(0.01));
  let color: N<'vec3'> = lit(mix(rgb(GLASS), rgb(BEZEL), bevel));
  color = mix(color, rgb(GRID), crisp(abs(fract(depth.add(0.5)).sub(0.5)).mul(pitch).sub(0.005)).mul(block).mul(inside));
  color = mix(color, rgb(GRID).mul(1.4), crisp(abs(y).sub(0.006)).mul(track));
  color = mix(color, rgb(GRID).mul(1.6), crisp(max(abs(x).sub(0.008), max(y.sub(area), y.negate().sub(0.06)))));
  color = mix(color, rgb(INPUT).mul(0.34), crisp(max(y.sub(density), y.negate())).mul(track));
  color = mix(color, rgb(INPUT).mul(0.95), crisp(abs(y.sub(density)).sub(0.012)).mul(track).mul(crisp(y.negate().add(0.004))));
  const at = clamp(info.r, -SPAN, SPAN).mul(scale);
  const needle = crisp(max(abs(x.sub(at)).sub(0.014), max(y.sub(area + 0.06), y.negate().sub(0.04))));
  color = mix(color, vec3(0.8), max(needle, crisp(sdCircle(vec2(x.sub(at), y), 0.055))));
  const meter = vec2(p.x.sub(METER_MID), p.y.sub(baseline.add(METER_LIFT)));
  const reach = info.g.mul(METER_HALF), bar = sdBox(vec2(meter.x.sub(reach.mul(0.5)), meter.y), vec2(abs(reach).mul(0.5).add(0.001), 0.07), 0.02);
  color = mix(color, rgb(0x070b10), crisp(sdBox(meter, vec2(METER_HALF + 0.06, 0.11), 0.11)));
  color = mix(color, rgb(GRID).mul(1.6), crisp(abs(sdBox(meter, vec2(METER_HALF + 0.06, 0.11), 0.11)).sub(0.006)));
  color = mix(color, select(info.g.greaterThan(0), rgb(CLASS[1]), rgb(CLASS[0])).mul(0.95), crisp(bar));
  color = mix(color, rgb(GRID).mul(2.4), crisp(max(abs(meter.x).sub(0.008), abs(meter.y).sub(0.15))));
  return sprite(color, crisp(edge), 1, flat(PANEL + 0.2, 2 * RISE + 0.2).translate(PANEL_MID, 0, 0));
}

export function dot(): THREE.MeshBasicNodeMaterial {
  const p = unit(), material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = mix(vec3(0.12), vec3(1), coverage(sdCircle(p, 0.62)));
  material.opacityNode = coverage(sdCircle(p, 0.84));
  return material;
}

export function cross(): THREE.MeshBasicNodeMaterial {
  const p = unit(), a = 0.55, material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const d = min(sdSegment(p, vec2(-a, -a), vec2(a, a), 0.13), sdSegment(p, vec2(-a, a), vec2(a, -a), 0.13));
  material.colorNode = mix(vec3(0.05), vec3(1), coverage(d));
  material.opacityNode = coverage(d.sub(0.16));
  return material;
}

export function disc(tint: Tint, radius = 0.8): THREE.Mesh {
  return sprite(tint, coverage(sdCircle(unit(), radius)), 7);
}

export function halo(tint: Tint, fade: Scalar, radius: number, half: number, order: number): THREE.Mesh {
  return sprite(tint, coverage(ring(unit(), radius, half)).mul(fade), order);
}

export function dashes(tint: Tint, fade: Scalar): THREE.Mesh {
  const along = positionWorld.x.mul(1 / 0.16);
  const gap = coverage(abs(fract(along).sub(0.5)).sub(0.25));
  return sprite(tint, gap.mul(fade).mul(coverage(abs(uv().y.sub(0.5)).sub(0.25))), 5, flat(1, 0.06));
}

export function sparkle(tint: Tint, fade: Scalar): THREE.Mesh {
  const p = unit(), arms = min(sdBox(p, vec2(0.95, 0.05), 0.05), sdBox(p, vec2(0.05, 0.95), 0.05));
  const glow = exp(length(p).negate().mul(5)).mul(0.9);
  return sprite(tint, max(coverage(arms).mul(float(1).sub(length(p).mul(0.6))), glow).mul(fade), 8);
}

export function crosshair(tint: Tint, fade: Scalar): THREE.Mesh {
  const p = unit(), gap = sdCircle(p, 0.32).negate();
  const bars = min(sdBox(p, vec2(0.9, 0.035), 0), sdBox(p, vec2(0.035, 0.9), 0));
  return sprite(tint, max(coverage(max(bars, gap)), coverage(ring(p, 0.22, 0.03))).mul(fade), 6);
}

export function label(text: string, height: number, tint: number, weight = 600): { mesh: THREE.Mesh; width: number; texture: THREE.DataTexture } {
  const px = 96, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Covariate labels need a 2D canvas');
  const font = `${weight} ${px}px ${FONT}`;
  context.font = font;
  const metrics = context.measureText(text), w = Math.ceil(metrics.width + px * 0.3), h = Math.ceil(px * 1.4);
  canvas.width = w;
  canvas.height = h;
  context.font = font;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, w / 2, h / 2);
  const pixels = context.getImageData(0, 0, w, h).data, alpha = new Uint8Array(w * h);
  for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) alpha[(h - 1 - row) * w + column] = pixels[(row * w + column) * 4 + 3]!;
  const map = new THREE.DataTexture(alpha, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  const tall = height * 1.4, wide = (tall * w) / h;
  return { mesh: sprite(rgb(tint), texture(map, uv()).r, 4, flat(wide, tall)), width: wide, texture: map };
}
