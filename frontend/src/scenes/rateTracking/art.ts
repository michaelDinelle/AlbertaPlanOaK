import * as THREE from 'three/webgpu';
import { Fn, abs, atan, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, sin, smoothstep, sqrt, texture, uniform, uvec2, vec2, vec3, vec4 } from 'three/tsl';
import { grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
type Layer = (tint: N<'vec3'>, cover: N<'float'>) => void;

export const HISTORY = 100;
export const SLOT = 0.15;
export const UNIT = 0.22;
export const NOW_X = 2.4;
export const LANE_X = NOW_X + 0.46;
export const SCALE_X = NOW_X + 0.9;
export const MID_Y = 0.3;
export const PANEL_X = NOW_X + 2.05;
export const GAUGE = { x: 6.5, y: 1.35, r: 0.9 } as const;
export const METER = { x: 6.5, base: -3.1, top: -1.1, gap: 0.62, half: 0.17 } as const;
export const KEY = { x: [5.45, 6.75], y: [-0.28, -0.6] } as const;
export const SWEEP = 1.5 * Math.PI;
export const START = 1.25 * Math.PI;

export const GOLD = 0xffc85e;
export const CORAL = 0xff6f61;
export const SEA = 0x57b7c9;
export const INK = 0xc9d3de;
export const STEEL = 0xaab6c6;

const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const LIGHT = vec2(-0.55, 0.83);

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);
export const level = (value = 0): Level => uniform(value);

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const pixel = (): N<'float'> => max(fwidth(positionWorld.x), fwidth(positionWorld.z));
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel())), 0, 1);
const flag = (condition: N<'bool'>): N<'float'> => select(condition, float(1), float(0));
const diamond = (p: N<'vec2'>, r: N<'float'> | number): N<'float'> => abs(p.x).add(abs(p.y)).sub(r).mul(Math.SQRT1_2);

function sprite(width: number, depth: number, x: number, y: number, order: number, draw: (layer: Layer) => void): THREE.Mesh {
  const shade = Fn(() => {
    const color = vec3(0).toVar(), alpha = float(0).toVar();
    draw((tint, cover) => {
      const c = clamp(cover, 0, 1).toVar(), a = c.add(alpha.mul(c.oneMinus())).toVar();
      color.assign(tint.mul(c).add(color.mul(alpha).mul(c.oneMinus())).div(max(a, 1e-4)));
      alpha.assign(a);
    });
    return vec4(color, alpha);
  })();
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = shade.xyz;
  material.opacityNode = shade.w;
  const mesh = new THREE.Mesh(flat(width, depth), material);
  mesh.position.set(x, 0, -y);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export interface SeaLevels { scroll: Level; center: Level; five: Level; flash: Level; time: Level }

export function sea(store: THREE.DataTexture, u: SeaLevels, drift: number, reach: number, accent: number, lit: Lit): THREE.Mesh {
  const data = texture(store);
  const at = (c: N<'float'>, row: number): N<'vec4'> => data.load(uvec2(vec2(c.add(1), row)));
  const screen = (value: N<'float'>): N<'float'> => float(MID_Y).add(value.sub(u.center).mul(UNIT));
  const columnX = (c: N<'float'>): N<'float'> => float(NOW_X).sub(c.add(u.scroll).mul(SLOT));
  const ageFade = (c: N<'float'>): N<'float'> => smoothstep(0.3, 1, c.add(u.scroll).div(HISTORY)).oneMinus();
  return sprite(60, 60, 0, 0, 0, (layer) => {
    const q = chart(), px = pixel();
    const m = float(NOW_X).sub(q.x).div(SLOT).sub(u.scroll).toVar();
    const c0 = clamp(floor(m), -1, HISTORY - 1).toVar(), c1 = min(c0.add(1), HISTORY - 1), f = clamp(m.sub(c0), 0, 1).toVar();
    const a0 = at(c0, 0).toVar(), a1 = at(c1, 0).toVar();
    const surface = screen(mix(a0.g, a1.g, f)).toVar();
    const slope = select(m.lessThan(-1), float(0), a1.g.sub(a0.g).mul(UNIT / SLOT));
    const ds = q.y.sub(surface).div(sqrt(slope.mul(slope).add(1))).toVar();
    const known = mix(flag(a0.a.greaterThan(3.5)), flag(a1.a.greaterThan(3.5)), f).toVar();
    const wet = crisp(ds).mul(known).toVar();

    const cs = clamp(floor(m.add(0.5)), -1, HISTORY - 1);
    const storm = clamp(at(cs, 1).r.div(drift), 0, 1).toVar();
    const height = max(q.y.sub(surface), 0), depth = max(surface.sub(q.y), 0).toVar();
    const air = mix(mix(rgb(0x0f1c2e), rgb(0x03060c), exp(height.mul(-0.32)).oneMinus()), rgb(0x1c2634), storm.mul(0.5));
    const shallow = exp(depth.mul(-0.55)).toVar();
    const caustic = noise(vec2(q.x.mul(1.3), q.y.mul(4.2)).add(vec2(u.time.mul(0.05), u.time.mul(-0.1)))).mul(0.05).mul(shallow);
    const water = mix(rgb(0x020d16), rgb(0x16566a), shallow).add(caustic).add(rgb(SEA).mul(exp(depth.div(-0.07)).mul(0.16)));
    layer(lit(mix(air, water, wet)), float(1));

    const tilt = 0.26, cosT = Math.cos(tilt), sinT = Math.sin(tilt), cell = vec2(0.09, 0.95);
    const r = vec2(q.x.mul(cosT).add(q.y.mul(sinT)), q.y.mul(cosT).sub(q.x.mul(sinT)).add(u.time.mul(6.5))).toVar();
    const slot = floor(r.div(cell)), drop = grain(slot).toVar(), more = grain(slot.add(vec2(31.7, 11.3))).toVar(), local = fract(r.div(cell)).toVar();
    const span = more.y.mul(0.3).add(0.3), along = local.y.sub(more.x.mul(span.oneMinus())).div(span).toVar();
    const body = smoothstep(0, 0.12, along).mul(smoothstep(0.55, 1, along).oneMinus()).mul(mix(float(1), float(0.25), clamp(along, 0, 1)));
    const streak = crisp(abs(local.x.sub(drop.y.mul(0.6).add(0.2))).mul(cell.x).sub(0.003)).mul(body);
    layer(rgb(0xa9bcd2), streak.mul(flag(drop.x.lessThan(storm.mul(0.4)))).mul(storm).mul(wet.oneMinus()).mul(0.36));

    const z = u.center.add(q.y.sub(MID_Y).div(UNIT)).toVar();
    const minor = abs(fract(z.add(0.5)).sub(0.5)).mul(UNIT).toVar();
    const major = abs(fract(z.add(u.five).div(5).add(0.5)).sub(0.5)).mul(5 * UNIT).toVar();
    const chartFade = smoothstep(0.35, 1, clamp(m.div(HISTORY), 0, 1)).oneMinus().mul(crisp(q.x.sub(NOW_X))).toVar();
    const toScale = crisp(q.x.sub(SCALE_X - 0.1));
    layer(rgb(0x9fb8d0), crisp(minor.sub(px.mul(0.5))).mul(0.05).mul(chartFade));
    layer(rgb(0x9fb8d0), crisp(major.sub(px.mul(0.6))).mul(max(chartFade.mul(0.14), toScale.mul(0.07))));

    const cn = clamp(floor(m.add(0.5)), 0, HISTORY - 1).toVar(), xc = columnX(cn).toVar(), col = at(cn, 0).toVar(), meta = at(cn, 1).toVar(), fade = ageFade(cn).toVar();
    layer(rgb(0x9fb8d0), crisp(abs(q.x.sub(xc)).sub(px.mul(0.5))).mul(meta.a).mul(fade).mul(0.07));

    const front = at(c0, 1).b.mul(ageFade(c0)), fx = columnX(c0).sub(SLOT / 2);
    layer(rgb(0xdcecf8), crisp(abs(q.x.sub(fx)).sub(0.007)).mul(flag(fract(q.y.mul(4)).lessThan(0.55))).mul(front).mul(0.55));

    layer(rgb(SEA), exp(abs(ds).div(-0.05)).mul(0.2).mul(known));
    layer(rgb(SEA).mul(1.2), crisp(abs(ds).sub(0.009)).mul(known));

    layer(lit(mix(rgb(STEEL).mul(0.85), rgb(0x2a5a69), wet.mul(0.6))), crisp(abs(q.x.sub(NOW_X)).sub(0.017)));
    const tx = q.x.sub(NOW_X).toVar();
    const minorTick = crisp(minor.sub(0.0035)).mul(crisp(tx.sub(0.075))).mul(crisp(float(0.012).sub(tx)));
    const majorTick = crisp(major.sub(0.006)).mul(crisp(tx.sub(0.14))).mul(crisp(float(0.012).sub(tx)));
    layer(lit(rgb(STEEL)), max(minorTick.mul(0.7), majorTick));

    const reading = flag(fract(col.a.mul(0.5)).greaterThan(0.25)).toVar(), forecast = flag(fract(col.a.mul(0.25)).greaterThan(0.375)).toVar();
    const yr = screen(col.r).toVar(), yf = screen(col.b).toVar(), gap = abs(yf.sub(yr));
    const now = select(cn.lessThan(0.5), u.flash, float(0)).toVar();
    const severity = clamp(meta.g.div(reach), 0, 1);
    const hair = sdBox(q.sub(vec2(xc, yf.add(yr).mul(0.5))), vec2(float(0.0045).add(now.mul(0.004)), gap.mul(0.5).add(0.001)), 0).toVar();
    const both = reading.mul(forecast).mul(fade).toVar();
    layer(rgb(CORAL), exp(max(hair, 0).div(-0.045)).mul(now).mul(both).mul(0.7));
    layer(rgb(CORAL).mul(now.mul(0.8).add(1)), crisp(hair).mul(both).mul(min(severity.mul(0.5).add(0.35).add(now), 1)));

    const tick = sdBox(q.sub(vec2(xc, yf)), vec2(0.055, 0.011), 0.008).toVar();
    layer(rgb(0x061018), crisp(tick.sub(0.009)).mul(forecast).mul(fade).mul(0.75));
    layer(rgb(accent), crisp(tick).mul(forecast).mul(fade));

    const bead = sdCircle(q.sub(vec2(xc, yr)), 0.04).toVar();
    layer(rgb(0x140b02), crisp(bead.sub(0.013)).mul(reading).mul(fade).mul(0.85));
    layer(rgb(GOLD), crisp(bead).mul(reading).mul(fade));
  });
}

export function buoy(y: Level, lit: Lit): THREE.Mesh {
  return sprite(0.3, 0.3, NOW_X, 0, 7, (layer) => {
    const p = chart().sub(vec2(NOW_X, y)), r = length(p).toVar(), facing = dot(p.div(max(r, 1e-4)), LIGHT).toVar();
    layer(rgb(0x120a02), crisp(abs(r.sub(0.058)).sub(0.035)).mul(0.85));
    layer(lit(mix(rgb(GOLD).mul(0.62), rgb(GOLD).mul(1.3), facing.mul(0.5).add(0.5))), crisp(abs(r.sub(0.058)).sub(0.024)));
    layer(rgb(0xfff4dc), crisp(abs(r.sub(0.058)).sub(0.006)).mul(facing.mul(0.5).add(0.3).max(0)).mul(0.6));
  });
}

export interface LanternLevels {
  railY: Level;
  railA: Level;
  hangY: Level;
  hangA: Level;
  capHigh: Level;
  capLow: Level;
  landX: Level;
  landY: Level;
  landA: Level;
  landS: Level;
}

function lanternShape(layer: Layer, p: N<'vec2'>, size: N<'float'>, alpha: N<'float'>, accent: number): void {
  const d = diamond(p, float(0.075).mul(size)).toVar();
  layer(rgb(accent).mul(0.9), exp(max(d, 0).div(-0.05)).mul(0.45).mul(alpha));
  layer(rgb(0x061018), crisp(d.sub(0.012)).mul(alpha).mul(0.8));
  layer(rgb(accent), crisp(d).mul(alpha));
  layer(rgb(0xf4fbff), crisp(diamond(p, float(0.028).mul(size))).mul(alpha));
}

export function lantern(u: LanternLevels, half: number, accent: number): THREE.Mesh {
  const left = NOW_X - 0.2, right = LANE_X + 0.3;
  return sprite(right - left, 16, (left + right) / 2, 0, 8, (layer) => {
    const q = chart(), cy = q.y.sub(u.railY);
    const dots = crisp(length(vec2(fract(q.x.div(0.045)).sub(0.5).mul(0.045), cy)).sub(0.0065));
    layer(rgb(GOLD).mul(0.8), dots.mul(crisp(float(NOW_X + 0.16).sub(q.x))).mul(crisp(q.x.sub(LANE_X - 0.04))).mul(u.railA).mul(0.55));
    layer(rgb(accent).mul(0.75), crisp(sdBox(q.sub(vec2(LANE_X, u.railY)), vec2(0.0045, half), 0)).mul(u.railA).mul(0.75));
    for (const [sign, pinned] of [[1, u.capHigh], [-1, u.capLow]] as const) {
      const cap = sdBox(q.sub(vec2(LANE_X, u.railY.add(sign * half))), vec2(0.05, 0.007), 0.004).toVar();
      layer(rgb(accent), exp(max(cap, 0).div(-0.05)).mul(pinned).mul(u.railA).mul(0.7));
      layer(rgb(accent).mul(pinned.mul(0.6).add(0.8)), crisp(cap).mul(u.railA));
    }
    lanternShape(layer, q.sub(vec2(u.landX, u.landY)), u.landS, u.landA, accent);
    lanternShape(layer, q.sub(vec2(LANE_X, u.hangY)), float(1), u.hangA, accent);
  });
}

export interface GaugeLevels { needle: Level; period: Level; agent: Level; oracle: Level }

export function gauge(u: GaugeLevels, ticks: number, accent: number, lit: Lit): THREE.Mesh {
  const R = GAUGE.r;
  return sprite(2.9, 9.4, GAUGE.x, -0.4, 6, (layer) => {
    const q = chart(), p = q.sub(vec2(GAUGE.x, GAUGE.y)).toVar(), r = length(p).toVar();
    layer(rgb(0x000000), crisp(r.sub(R + 0.16)).mul(0.35));
    layer(lit(rgb(0x0b121b).mul(dot(p.div(max(r, 1e-4)), LIGHT).mul(0.12).add(1))), crisp(r.sub(R)));
    layer(lit(rgb(STEEL).mul(0.8)), crisp(abs(r.sub(R)).sub(0.014)));
    const swept = fract(float(START).sub(atan(p.y, p.x)).div(2 * Math.PI)).mul(2 * Math.PI).toVar();
    const along = clamp(swept.div(SWEEP), 0, 1);
    layer(mix(rgb(accent).mul(0.55), rgb(CORAL).mul(0.85), along), crisp(abs(r.sub(R * 0.82)).sub(0.028)).mul(flag(swept.lessThan(SWEEP))).mul(0.8));
    for (let i = 0; i <= ticks; i++) {
      const angle = START - (SWEEP * i) / ticks, dir = vec2(Math.cos(angle), Math.sin(angle)), major = i % 2 === 0;
      layer(lit(rgb(INK)), crisp(sdSegment(p, dir.mul(R * (major ? 0.62 : 0.68)), dir.mul(R * 0.76), major ? 0.011 : 0.007)));
    }
    const heading = vec2(cos(u.needle), sin(u.needle)).toVar();
    const needle = sdSegment(p, heading.mul(-0.14), heading.mul(R * 0.78), 0.017).toVar();
    layer(rgb(0x000000), crisp(needle.sub(0.012)).mul(0.6));
    layer(rgb(0xf2f6fa), crisp(needle));
    layer(rgb(accent), crisp(sdSegment(p, heading.mul(R * 0.56), heading.mul(R * 0.78), 0.017)));
    layer(lit(rgb(STEEL)), crisp(sdCircle(p, 0.075)));
    layer(rgb(0x0b121b), crisp(sdCircle(p, 0.03)));
    const ring = crisp(abs(r.sub(R + 0.1)).sub(0.018)).toVar();
    layer(rgb(0x2a3646), ring);
    layer(rgb(accent).mul(0.9), ring.mul(flag(fract(atan(p.x, p.y).div(2 * Math.PI)).lessThan(u.period))));
    const tip = u.period.mul(2 * Math.PI);
    layer(rgb(0xf4fbff), crisp(sdCircle(p.sub(vec2(sin(tip), cos(tip)).mul(R + 0.1)), 0.028)));

    const [left, right] = KEY.x, [upper, lower] = KEY.y;
    layer(rgb(0x140b02), crisp(sdCircle(q.sub(vec2(left, upper)), 0.053)).mul(0.85));
    layer(rgb(GOLD), crisp(sdCircle(q.sub(vec2(left, upper)), 0.04)));
    const tick = sdBox(q.sub(vec2(right, upper)), vec2(0.055, 0.011), 0.008).toVar();
    layer(rgb(0x061018), crisp(tick.sub(0.009)).mul(0.75));
    layer(rgb(accent), crisp(tick));
    layer(rgb(CORAL), crisp(sdBox(q.sub(vec2(left, lower)), vec2(0.0045, 0.075), 0)));
    layer(rgb(SEA).mul(0.35), crisp(sdBox(q.sub(vec2(right, lower - 0.05)), vec2(0.07, 0.05), 0)));
    layer(rgb(SEA).mul(1.2), crisp(sdBox(q.sub(vec2(right, lower)), vec2(0.07, 0.009), 0)));

    for (const [x, value, tint] of [[METER.x - METER.gap / 2, u.agent, accent], [METER.x + METER.gap / 2, u.oracle, SEA]] as const) {
      const top = float(METER.base).add(value.mul(UNIT)).toVar();
      const box = sdBox(q.sub(vec2(x, top.add(METER.base).mul(0.5))), vec2(METER.half, top.sub(METER.base).mul(0.5).max(0.001)), 0.02).toVar();
      layer(rgb(tint).mul(0.25), crisp(box));
      layer(rgb(tint), crisp(abs(box).sub(0.012)));
      layer(rgb(tint).mul(1.2), crisp(sdBox(q.sub(vec2(x, top)), vec2(METER.half + 0.03, 0.013), 0.006)));
    }
    layer(lit(rgb(STEEL).mul(0.7)), crisp(sdBox(q.sub(vec2(METER.x, METER.base)), vec2(METER.gap / 2 + METER.half + 0.14, 0.008), 0)));
  });
}

export function panel(lit: Lit): THREE.Mesh {
  return sprite(8, 16, PANEL_X + 3.6, 0, 5, (layer) => {
    const q = chart(), edge = q.x.sub(PANEL_X).toVar();
    const brushed = noise(vec2(q.x.mul(0.8), q.y.mul(22))).mul(0.04).add(noise(q.mul(1.7)).mul(0.05)).add(1);
    layer(rgb(0x000000), exp(edge.negate().max(0).div(-0.22)).mul(crisp(edge)).mul(0.55));
    layer(lit(rgb(0x0b1119).mul(brushed)), crisp(edge.negate()));
    layer(lit(rgb(STEEL).mul(0.55)), crisp(abs(edge.sub(0.02)).sub(0.012)));
    layer(rgb(0x000000), crisp(abs(edge.sub(0.045)).sub(0.01)).mul(0.5));
  });
}

export class Text {
  readonly mesh: THREE.Mesh;
  readonly width: number;
  private readonly context: CanvasRenderingContext2D;
  private readonly map: THREE.DataTexture;
  private readonly pixels: Uint8Array;
  private value = '';

  constructor(sample: string, height: number, tint: number, private readonly align: -1 | 0 | 1, weight = 600, spacing = 0) {
    const px = 96, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('RateTracking labels need a 2D canvas');
    const style = (): void => {
      context.font = `${weight} ${px}px ${FONT}`;
      context.letterSpacing = `${spacing * px}px`;
      context.fillStyle = '#ffffff';
      context.textBaseline = 'middle';
      context.textAlign = align === -1 ? 'left' : align === 1 ? 'right' : 'center';
    };
    style();
    canvas.width = Math.ceil(context.measureText(sample).width + px * 0.5);
    canvas.height = Math.ceil(px * 1.4);
    style();
    this.context = context;
    this.pixels = new Uint8Array(canvas.width * canvas.height);
    this.map = new THREE.DataTexture(this.pixels, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
    this.map.minFilter = THREE.LinearMipmapLinearFilter;
    this.map.magFilter = THREE.LinearFilter;
    this.map.generateMipmaps = true;
    const tall = height * 1.4;
    this.width = (tall * canvas.width) / canvas.height;
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = rgb(tint);
    material.opacityNode = texture(this.map).r;
    this.mesh = new THREE.Mesh(flat(this.width, tall), material);
    this.mesh.renderOrder = 10;
    this.mesh.frustumCulled = false;
  }

  set(value: string): this {
    if (value === this.value) return this;
    this.value = value;
    const { canvas } = this.context, w = canvas.width, h = canvas.height, pad = h * 0.15;
    this.context.clearRect(0, 0, w, h);
    this.context.fillText(value, this.align === -1 ? pad : this.align === 1 ? w - pad : w / 2, h / 2);
    const rgba = this.context.getImageData(0, 0, w, h).data;
    for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) this.pixels[(h - 1 - row) * w + column] = rgba[(row * w + column) * 4 + 3]!;
    this.map.needsUpdate = true;
    return this;
  }

  place(x: number, y: number): this {
    const { canvas } = this.context, pad = (this.width * canvas.height * 0.15) / canvas.width;
    this.mesh.position.set(x + this.align * (pad - this.width / 2), 0, -y);
    return this;
  }

  dispose(): void {
    this.map.dispose();
  }
}
