import * as THREE from 'three/webgpu';
import { abs, cos, dot, exp, float, floor, instancedDynamicBufferAttribute, length, max, min, mix, positionWorld, select, sin, smoothstep, texture, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, rgb, sdBox, sdCircle } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const DISC = 0.41;
export const SPRITE = 1.3;
export const FRAME = 0.62;
export const LIP = 0.05;
export const CORNER = 0.3;
export const HINT = 0.5;
export const BRACKET = 1.34;
export const PLAQUE_W = 2.7;
export const PLAQUE_H = 5.2;
export const WELL_Y = 0.5;
export const WELL_R = 0.84;

export const EBONY = 0x17181b;
export const IVORY = 0xebe7dd;
const EBONY_RIM = 0x44464c;
const IVORY_RIM = 0xa7a195;
const FELT = 0x1c7447;
const LINE = 0x08170f;
const WOOD_DARK = 0x2a170c;
const WOOD_LIGHT = 0x5c3a20;
const TABLE_LOW = 0x141619;
const TABLE_HIGH = 0x22262b;
const CARD_LOW = 0x121417;
const CARD_HIGH = 0x1f2327;
export const LAST_DOT = 0xff5540;
export const INLAY = 0xd9c498;
const LIGHT = [-0.55, 0.83] as const;
const SANS = '"Avenir Next", Avenir, "Helvetica Neue", "Segoe UI", system-ui, sans-serif';
const PX = 160;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const toward = (v: N<'vec2'>) => v.div(max(length(v), 1e-5));

export function roundedRing(outer: number, radius: number, inner: number): THREE.BufferGeometry {
  const shape = new THREE.Shape(), a = outer - radius;
  shape.moveTo(-a, -outer);
  shape.lineTo(a, -outer);
  shape.absarc(a, -a, radius, -Math.PI / 2, 0, false);
  shape.lineTo(outer, a);
  shape.absarc(a, a, radius, 0, Math.PI / 2, false);
  shape.lineTo(-a, outer);
  shape.absarc(-a, a, radius, Math.PI / 2, Math.PI, false);
  shape.lineTo(-outer, -a);
  shape.absarc(-a, -a, radius, Math.PI, 1.5 * Math.PI, false);
  const hole = new THREE.Path();
  hole.moveTo(-inner, -inner);
  hole.lineTo(-inner, inner);
  hole.lineTo(inner, inner);
  hole.lineTo(inner, -inner);
  hole.lineTo(-inner, -inner);
  shape.holes.push(hole);
  return new THREE.ShapeGeometry(shape, 12).rotateX(-Math.PI / 2);
}

export function tableMaterial(half: number, seatX: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = vec2(positionWorld.x, positionWorld.z);
  const mottle = fractal(p.mul(0.45), 4).mul(0.5).add(0.5), fibre = fractal(p.mul(vec2(1.1, 15)), 2);
  const base = mix(rgb(TABLE_LOW), rgb(TABLE_HIGH), mottle).mul(fibre.mul(0.04).add(1));
  const pool = exp(length(p).div(half * 2.2).pow(2).negate()).mul(0.5).add(0.5);
  const cast = vec2(0.16, 0.24);
  const board = sdBox(p.sub(cast), vec2(half + FRAME), CORNER + 0.15);
  const seat = (x: number) => sdBox(p.sub(vec2(x, 0)).sub(cast), vec2(PLAQUE_W / 2, PLAQUE_H / 2), 0.3);
  const shadow = max(smoothstep(0.8, -0.3, board).mul(0.64), smoothstep(0.6, -0.25, min(seat(-seatX), seat(seatX))).mul(0.5));
  material.colorNode = lit(base.mul(pool).mul(shadow.oneMinus()));
  return material;
}

export function boardMaterial(size: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const half = size / 2, outer = half + FRAME, star = half - 2, well = half + LIP;
  const p = vec2(positionWorld.x, positionWorld.z);
  const edgeAt = (q: N<'vec2'>) => sdBox(q, vec2(outer), CORNER);
  const wellAt = (q: N<'vec2'>) => sdBox(q, vec2(well), 0.04);
  const e = 0.02, dx = vec2(e, 0), dz = vec2(0, e);
  const edge = edgeAt(p), inner = wellAt(p);
  const outward = toward(vec2(edgeAt(p.add(dx)).sub(edgeAt(p.sub(dx))), edgeAt(p.add(dz)).sub(edgeAt(p.sub(dz)))));
  const inward = toward(vec2(wellAt(p.sub(dx)).sub(wellAt(p.add(dx))), wellAt(p.sub(dz)).sub(wellAt(p.add(dz)))));
  const towardLight = vec2(LIGHT[0], -LIGHT[1]);

  const side = abs(p.x).greaterThan(abs(p.y));
  const along = select(side, p.y, p.x), across = select(side, p.x, p.y);
  const grain = fractal(vec2(along.mul(0.42), across.mul(6.5)), 4).mul(0.5).add(0.5);
  const figure = sin(across.mul(38).add(grain.mul(7))).mul(0.5).add(0.5);
  let wood: N<'vec3'> = mix(rgb(WOOD_DARK), rgb(WOOD_LIGHT), grain.mul(0.75).add(figure.mul(0.18)).add(0.04));
  const rim = smoothstep(-0.16, -0.01, edge).mul(dot(outward, towardLight)).mul(0.5);
  const lipLight = smoothstep(0.12, 0, inner).mul(dot(inward, towardLight)).mul(0.4);
  const miter = coverage(abs(abs(p.x).sub(abs(p.y))).mul(Math.SQRT1_2).sub(0.005)).mul(0.55);
  const sheen = exp(abs(p.x.mul(0.6).add(p.y).add(outer * 0.9)).mul(-1.1)).mul(0.12);
  wood = wood.mul(float(1).add(rim).add(lipLight).add(sheen)).mul(miter.oneMinus());

  const fibre = fractal(p.mul(26), 2).mul(0.03).add(fractal(p.mul(1.7), 4).mul(0.075));
  const pool = float(1.1).sub(smoothstep(0, half * 1.5, length(p)).mul(0.24));
  const shade = max(smoothstep(0.3, 0, p.y.add(well)), smoothstep(0.3, 0, p.x.add(well))).mul(0.34);
  let felt: N<'vec3'> = rgb(FELT).mul(fibre.add(1)).mul(pool).mul(shade.oneMinus());
  const q = p.add(half), g = abs(q.sub(floor(q.add(0.5))));
  const lines = coverage(min(g.x, g.y).sub(0.018)).mul(coverage(sdBox(p, vec2(half + 0.02), 0)));
  felt = mix(felt, rgb(LINE), lines);
  if (star > 0) felt = mix(felt, rgb(LINE), coverage(sdCircle(abs(p).sub(vec2(star)), 0.075)));

  material.colorNode = lit(mix(wood, felt, coverage(inner)));
  material.opacityNode = coverage(edge);
  return material;
}

export function discMaterial(look: THREE.InstancedBufferAttribute, fx: THREE.InstancedBufferAttribute, lit: Lit, glowTint: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const v = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, f = instancedDynamicBufferAttribute(fx, 'vec4') as N<'vec4'>;
  const white = v.x, squash = max(v.y, 0.05), alpha = v.z, glow = v.w, angle = f.x, lift = f.y, marker = f.z;
  const c = cos(angle), s = sin(angle);
  const turn = (q: N<'vec2'>) => vec2(q.x.mul(c).add(q.y.mul(s)), q.y.mul(c).sub(q.x.mul(s)));
  const back = (q: N<'vec2'>) => vec2(q.x.mul(c).sub(q.y.mul(s)), q.y.mul(c).add(q.x.mul(s)));
  const p = uv().sub(0.5).mul(SPRITE), axes = vec2(squash.mul(DISC), DISC);
  const n = turn(p).div(axes), r = length(n), minor = squash.mul(DISC);
  const body = r.sub(1).mul(minor);
  const shift = vec2(0.045, -0.06).mul(lift.mul(3.5).add(1));
  const cast = length(turn(p.sub(shift)).div(axes)).sub(1).mul(minor);
  const soft = lift.mul(0.1).add(0.035);
  const shadow = smoothstep(soft, soft.negate(), cast).mul(lift.mul(-0.22).add(0.55));
  const facing = back(n), dome = dot(facing, vec2(...LIGHT));
  const bevel = smoothstep(0.6, 1, r).mul(dome).mul(mix(float(0.45), float(0.22), white));
  const groove = exp(abs(r.sub(0.72)).mul(-42)).mul(0.09);
  const tone = float(1).add(bevel).sub(groove).add(dome.mul(0.04)).mul(squash.mul(0.28).add(0.72));
  const face = mix(rgb(EBONY), rgb(IVORY), white).mul(tone);
  const albedo = mix(face, mix(rgb(EBONY_RIM), rgb(IVORY_RIM), white), smoothstep(0.87, 0.97, r));
  const gloss = exp(length(facing.sub(vec2(-0.3, 0.36))).pow(2).mul(-9)).mul(mix(float(0.42), float(0.16), white)).mul(squash);
  const spot = coverage(length(p).sub(0.085)).mul(marker);
  let color: N<'vec3'> = lit(albedo).add(vec3(gloss));
  color = mix(color, rgb(LAST_DOT).mul(1.2), spot);
  color = color.add(rgb(glowTint).mul(glow).mul(r.mul(-0.45).add(1)).mul(white.mul(-0.7).add(1)));
  const cover = coverage(body), total = cover.add(shadow.mul(cover.oneMinus()));
  material.colorNode = color.mul(cover).div(max(total, 1e-4));
  material.opacityNode = total.mul(alpha);
  return material;
}

export function hintMaterial(alpha: THREE.InstancedBufferAttribute, tint: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const a = instancedDynamicBufferAttribute(alpha, 'float') as N<'float'>;
  const p = uv().sub(0.5).mul(HINT), r = length(p);
  const ring = coverage(abs(r.sub(0.12)).sub(0.017)), core = coverage(r.sub(0.032));
  material.colorNode = rgb(tint).mul(1.15);
  material.opacityNode = max(ring.mul(0.75), core.mul(0.6)).mul(a);
  return material;
}

export function bracketMaterial(tint: N<'vec3'>, alpha: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(BRACKET);
  const ring = abs(sdBox(p, vec2(0.47), 0.07)).sub(0.024);
  const corner = max(ring, float(0.19).sub(min(abs(p.x), abs(p.y))));
  const halo = exp(max(corner, 0).mul(-24)).mul(0.3);
  material.colorNode = tint.mul(1.3);
  material.opacityNode = min(coverage(corner).add(halo), 1).mul(alpha);
  return material;
}

export function plaqueMaterial(tint: N<'vec3'>, glow: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(vec2(PLAQUE_W + 0.3, PLAQUE_H + 0.3));
  const card = sdBox(p, vec2(PLAQUE_W / 2, PLAQUE_H / 2), 0.22);
  const height = p.y.div(PLAQUE_H).add(0.5);
  let face: N<'vec3'> = mix(rgb(CARD_LOW), rgb(CARD_HIGH), height.mul(0.8).add(0.1)).mul(fractal(p.mul(6), 2).mul(0.035).add(1));
  const edgeLight = smoothstep(-0.08, 0, card).mul(dot(toward(p), vec2(...LIGHT))).mul(0.5);
  face = face.mul(edgeLight.add(1));
  const q = p.sub(vec2(0, WELL_Y)), well = length(q).sub(WELL_R);
  const recess = smoothstep(0.05, -0.02, well).mul(q.y.div(WELL_R).mul(0.25).add(0.3));
  face = face.mul(recess.oneMinus());
  const inset = coverage(abs(card.add(0.1)).sub(0.011));
  const color = lit(face).add(tint.mul(inset.mul(glow.mul(0.9).add(0.25))));
  material.colorNode = color;
  material.opacityNode = coverage(card);
  return material;
}

export function bannerMaterial(width: number, height: number, tint: N<'vec3'>, alpha: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(vec2(width + 0.6, height + 0.6));
  const box = sdBox(p, vec2(width / 2, height / 2), 0.2);
  const border = coverage(abs(box.add(0.08)).sub(0.014));
  const halo = exp(max(box, 0).mul(-9)).mul(smoothstep(0, 0.01, box)).mul(0.35);
  const glass = mix(rgb(0x0a0c0e), rgb(0x15191c), p.y.div(height).add(0.5));
  const cover = coverage(box);
  const color = mix(mix(tint.mul(1.2), glass, cover), tint.mul(1.5), border);
  material.colorNode = color;
  material.opacityNode = min(cover.mul(0.9).add(border).add(halo), 1).mul(alpha);
  return material;
}

export function textMaterial(caption: Caption, tint: N<'vec3'>, alpha: Level, lit: Lit | null): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  material.colorNode = lit ? lit(tint) : tint;
  material.opacityNode = texture(caption.texture, uv()).r.mul(alpha);
  return material;
}

export type Paint = (context: CanvasRenderingContext2D, px: number) => void;

export class Caption {
  readonly texture: THREE.DataTexture;
  private readonly context: CanvasRenderingContext2D;
  private readonly alpha: Uint8Array;
  private shown: string | null = null;

  constructor(width: number, height: number, readonly px = PX) {
    const canvas = document.createElement('canvas');
    canvas.width = 4 * Math.ceil(width * px / 4);
    canvas.height = 4 * Math.ceil(height * px / 4);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Reversi captions need a 2D canvas');
    this.context = context;
    this.alpha = new Uint8Array(canvas.width * canvas.height);
    this.texture = new THREE.DataTexture(this.alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
    this.texture.minFilter = THREE.LinearMipmapLinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = true;
    this.texture.needsUpdate = true;
  }

  draw(key: string, paint: Paint): void {
    if (key === this.shown) return;
    this.shown = key;
    const context = this.context, { width, height } = context.canvas;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#ffffff';
    paint(context, this.px);
    const pixels = context.getImageData(0, 0, width, height).data;
    for (let y = 0; y < height; y++) {
      const to = (height - 1 - y) * width, from = y * width;
      for (let x = 0; x < width; x++) this.alpha[to + x] = pixels[4 * (from + x) + 3]!;
    }
    this.texture.needsUpdate = true;
  }

  write(text: string, size: number, weight: number, tracking: number): void {
    this.draw(`${text}|${size}|${weight}|${tracking}`, (context, px) => {
      const { width, height } = context.canvas;
      const fit = (scale: number) => {
        context.font = `${weight} ${size * px * scale}px ${SANS}`;
        context.letterSpacing = `${tracking * size * px * scale}px`;
      };
      fit(1);
      const room = 0.96 * width, measured = context.measureText(text).width;
      if (measured > room) fit(room / measured);
      const metrics = context.measureText(text);
      context.textAlign = 'center';
      context.textBaseline = 'alphabetic';
      context.fillText(text, width / 2 + tracking * size * px / 2, (height + metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2);
    });
  }

  dispose(): void {
    this.texture.dispose();
  }
}

export function glyph(context: CanvasRenderingContext2D, px: number, text: string, x: number, y: number, size: number, weight: number): void {
    context.font = `${weight} ${size * px}px ${SANS}`;
    context.letterSpacing = '0px';
    const metrics = context.measureText(text);
    context.textAlign = 'center';
    context.textBaseline = 'alphabetic';
    context.fillText(text, x * px, y * px + (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2);
}
