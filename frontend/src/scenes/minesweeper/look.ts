import * as THREE from 'three/webgpu';
import { abs, dot, exp, float, floor, fract, fwidth, max, min, mix, mod, normalize, positionWorld, select, sign, smoothstep, step, texture, uv, vec2 } from 'three/tsl';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Point = readonly [number, number];

export const RING = 1;
export const BAR = 1.1;
export const DIGIT = 0.5;
export const LED_HEIGHT = 0.74;

export const INK = [0x000000, 0x1d5bd6, 0x1b8a3a, 0xd3372b, 0x2b2c9c, 0x8e2020, 0x12808a, 0x26272c, 0x6b707a] as const;
export const PING = 0x5ee0ff;
export const LAMP = 0xffe0b4;
export const BLAST = 0xffa45c;

const PLATE = 0xe7ebef;
const PLATE_EDGE = 0xa2abb6;
const WELL = 0x4a525d;
const CAP = 0xa9b2be;
const CAP_LIGHT = 0xf3f6f9;
const CAP_DARK = 0x535d69;
const STRUCK = 0xd8352a;
const MINE = 0x1b1f25;
const CLOTH = 0xe0392c;
const POLE = 0x262a31;
const CROSS = 0x16181d;
const STEEL = 0x8f99a7;
const STEEL_LIGHT = 0xdce3eb;
const STEEL_DARK = 0x444c57;
const GLASS = 0x1c0d0c;
const SEGMENT_ON = 0xff4a2a;
const SEGMENT_OFF = 0x3a1611;
const DESK_A = 0x0d1116;
const DESK_B = 0x171d25;
const LIGHT = vec2(-Math.SQRT1_2, Math.SQRT1_2);
const FONT = '"Avenir Next", "Helvetica Neue", system-ui, sans-serif';
const GLYPH_PX = 128;
const SEGMENTS: readonly (readonly [number, number, number, number])[] = [
  [-0.13, 0.37, 0.13, 0.37], [0.175, 0.32, 0.175, 0.05], [0.175, -0.05, 0.175, -0.32], [-0.13, -0.37, 0.13, -0.37], [-0.175, -0.05, -0.175, -0.32], [-0.175, 0.32, -0.175, 0.05], [-0.13, 0, 0.13, 0],
];
export const SEVEN: Record<string, number> = { '0': 63, '1': 6, '2': 91, '3': 79, '4': 102, '5': 109, '6': 125, '7': 7, '8': 127, '9': 111, '-': 64 };

export const ledLayout = (w: number, h: number, digits: number) => {
  const half = digits * DIGIT / 2 + 0.14;
  return { x: w / 2 - half, y: h / 2 + RING + BAR / 2 - 0.04, width: 2 * half };
};

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = (blending: THREE.Blending = THREE.NormalBlending) => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending });

const polygon = (p: N<'vec2'>, points: readonly Point[]): N<'float'> => {
  let d: N<'float'> | null = null;
  points.forEach(([ax, ay], k) => {
    const [bx, by] = points[(k + 1) % points.length]!, length = Math.hypot(bx - ax, by - ay);
    const edge = dot(p.sub(vec2(ax, ay)), vec2((ay - by) / length, (bx - ax) / length));
    d = d === null ? edge : max(d, edge);
  });
  return d!;
};

const boxNormal = (p: N<'vec2'>, half: N<'vec2'>, radius: number): N<'vec2'> => {
  const q = abs(p).sub(half.sub(radius));
  const outward = select(max(q.x, q.y).greaterThan(0), normalize(max(q, vec2(0)).add(vec2(1e-5))), select(q.x.greaterThan(q.y), vec2(1, 0), vec2(0, 1)));
  return outward.mul(sign(p));
};

const spikes = (p: N<'vec2'>, reach: number, radius: number): N<'float'> => {
  const diagonal = reach * Math.SQRT1_2;
  return min(min(sdSegment(p, vec2(-reach, 0), vec2(reach, 0), radius), sdSegment(p, vec2(0, -reach), vec2(0, reach), radius)),
    min(sdSegment(p, vec2(-diagonal, -diagonal), vec2(diagonal, diagonal), radius), sdSegment(p, vec2(-diagonal, diagonal), vec2(diagonal, -diagonal), radius)));
};

export function numerals(): THREE.DataTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 8 * GLYPH_PX;
  canvas.height = GLYPH_PX;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Minesweeper numerals need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.font = `800 100px ${FONT}`;
  const probe = context.measureText('8');
  context.font = `800 ${100 * 0.62 * GLYPH_PX / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent)}px ${FONT}`;
  for (let n = 1; n <= 8; n++) {
    const fit = context.measureText(String(n));
    context.fillText(String(n), (n - 0.5) * GLYPH_PX, GLYPH_PX / 2 + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2);
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const atlas = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  atlas.minFilter = THREE.LinearMipmapLinearFilter;
  atlas.magFilter = THREE.LinearFilter;
  atlas.wrapS = atlas.wrapT = THREE.ClampToEdgeWrapping;
  atlas.generateMipmaps = true;
  atlas.needsUpdate = true;
  return atlas;
}

export function deskMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), p = positionWorld.xz;
  const broad = fractal(p.mul(0.07), 4).mul(0.5).add(0.5).clamp(0, 1), fine = fractal(p.mul(2.3), 2).mul(0.5).add(0.5).clamp(0, 1);
  material.colorNode = lit(mix(rgb(DESK_A), rgb(DESK_B), broad).mul(fine.mul(0.1).add(0.95)));
  return material;
}

export function cellMaterial(atlas: THREE.Texture, shape: N<'vec4'>, mark: N<'vec4'>, ink: N<'vec4'>, heat: N<'float'>, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), p = uv().sub(0.5);
  const raise = shape.x, flag = shape.y, mine = shape.z, count = shape.w, struck = mark.x, cross = mark.y, fresh = mark.w, ping = ink.w;

  const edge = float(0.5).sub(max(abs(p.x), abs(p.y)));
  const sink = min(smoothstep(0.3, 0.5, p.y).add(smoothstep(0.3, 0.5, p.x.negate())), 1).mul(0.13);
  const floorTone = mix(mix(rgb(PLATE), rgb(PLATE_EDGE), coverage(edge.sub(0.022))).mul(sink.oneMinus()), rgb(STRUCK), struck);
  const glyph = vec2(count.sub(1).add(p.x.add(0.5)).div(8), p.y.negate().add(0.5));
  const letter = texture(atlas, glyph).r.mul(step(0.5, count));
  const mq = p.div(max(mine, 1e-3)), present = step(1e-3, mine);
  const body = coverage(min(sdCircle(mq, 0.19), spikes(mq, 0.32, 0.03))).mul(present);
  const shine = coverage(sdCircle(mq.sub(vec2(-0.065, 0.065)), 0.048)).mul(present);
  const plate = mix(mix(mix(floorTone, ink.xyz, letter), rgb(MINE), body), rgb(0xf4f6f8), shine.mul(0.9));

  const size = mix(float(0.72), float(1), raise), kq = p.div(size);
  const key = sdBox(kq, vec2(0.44), 0.08);
  const keyCover = coverage(key).mul(smoothstep(0.05, 0.95, raise));
  const rim = smoothstep(-0.115, -0.03, key), facing = dot(normalize(kq.add(vec2(1e-4, 0))), LIGHT).mul(0.5).add(0.5);
  const cap = mix(rgb(CAP).mul(kq.y.sub(kq.x).mul(0.12).add(1)), mix(rgb(CAP_DARK), rgb(CAP_LIGHT), facing), rim);
  const shadow = smoothstep(-0.02, 0.08, sdBox(kq.sub(vec2(0.03, -0.04)), vec2(0.44), 0.08)).oneMinus().mul(0.45).mul(raise);
  const base = mix(plate, rgb(WELL).mul(shadow.oneMinus()), smoothstep(0.45, 1, raise));
  const capped = mix(base, cap, keyCover);

  const foot = vec2(0.04, -0.27), fq = p.sub(foot).div(max(flag, 1e-3)).add(foot), planted = step(1e-3, flag);
  const cloth = coverage(polygon(fq, [[0.08, 0.35], [0.08, -0.01], [-0.31, 0.17]])).mul(planted);
  const stand = min(sdBox(fq.sub(foot), vec2(0.2, 0.04), 0.025), sdBox(fq.sub(vec2(0.04, -0.205)), vec2(0.1, 0.035), 0.02));
  const staff = coverage(min(sdSegment(fq, vec2(0.05, -0.22), vec2(0.05, 0.35), 0.031), stand)).mul(planted);
  const flagged = mix(mix(capped, rgb(CLOTH).mul(fq.x.mul(0.7).add(1)), cloth), rgb(POLE), staff);
  const crossed = coverage(min(sdSegment(p, vec2(-0.29, -0.29), vec2(0.29, 0.29), 0.045), sdSegment(p, vec2(-0.29, 0.29), vec2(0.29, -0.29), 0.045))).mul(cross);
  const albedo = mix(flagged, rgb(CROSS), crossed);

  const emissive = rgb(PING).mul(exp(edge.mul(-16)).mul(ping).mul(1.8))
    .add(rgb(STRUCK).mul(struck.mul(heat).mul(body.oneMinus()).mul(2.2)))
    .add(rgb(0xe6f6ff).mul(exp(edge.mul(-16)).mul(fresh).mul(0.4)));
  material.colorNode = lit(albedo).add(emissive);
  material.opacityNode = mark.z;
  return material;
}

export interface FrameSpec { w: number; h: number; digits: number }

export function frameMaterial({ w, h, digits }: FrameSpec, width: number, depth: number, centre: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet(), s = uv().sub(0.5).mul(vec2(width, depth)).add(vec2(0, centre));
  const half = vec2(w / 2 + RING, h / 2 + RING + BAR / 2), middle = vec2(0, BAR / 2), round = 0.3;
  const outer = sdBox(s.sub(middle), half, round), cover = coverage(outer);
  const drop = smoothstep(-0.25, 0.75, sdBox(s.sub(middle).sub(vec2(0.14, -0.26)), half, round)).oneMinus().mul(0.62);
  const brushed = fractal(s.mul(vec2(0.45, 30)), 3).mul(0.5).add(0.5).clamp(0, 1);
  const steel = rgb(STEEL).mul(brushed.mul(0.12).add(0.92)).mul(s.y.sub(centre).div(depth).mul(0.16).add(1));
  const bevel = smoothstep(-0.17, -0.02, outer), facing = dot(boxNormal(s.sub(middle), half, round), LIGHT).mul(0.5).add(0.5);
  const body = mix(steel, mix(rgb(STEEL_DARK), rgb(STEEL_LIGHT), facing), bevel);

  const board = sdBox(s, vec2(w / 2, h / 2), 0.04), lip = smoothstep(0, 0.16, board).oneMinus().mul(step(0, board));
  const inward = dot(boxNormal(s, vec2(w / 2, h / 2), 0.04), LIGHT).negate().mul(0.5).add(0.5);
  const recessed = mix(body, mix(rgb(STEEL_DARK), rgb(STEEL_LIGHT).mul(0.8), inward), lip.mul(0.85));

  const led = ledLayout(w, h, digits), ledY = led.y, ledHalf = vec2(led.width / 2, LED_HEIGHT / 2), ledX = led.x;
  let glass: N<'float'> | null = null;
  for (const x of [-ledX, ledX]) {
    const d = sdBox(s.sub(vec2(x, ledY)), ledHalf, 0.08);
    glass = glass === null ? d : min(glass, d);
  }
  const windowCover = coverage(glass!), windowRim = smoothstep(0, 0.1, glass!).oneMinus().mul(step(0, glass!));
  const inset = mix(recessed, rgb(STEEL_DARK).mul(0.7), windowRim.mul(0.8));
  const glassTone = rgb(GLASS).mul(smoothstep(-0.2, 0, glass!).mul(-0.5).add(1));
  let albedo = mix(inset, glassTone, windowCover);
  for (const [x, y] of [[-w / 2 - RING / 2, -h / 2 - RING / 2], [w / 2 + RING / 2, -h / 2 - RING / 2], [-w / 2 - RING / 2, ledY], [w / 2 + RING / 2, ledY]] as const) {
    const at = s.sub(vec2(x, y)), head = coverage(sdCircle(at, 0.13));
    const tone = mix(rgb(STEEL_LIGHT).mul(0.9), rgb(STEEL_DARK), smoothstep(-0.12, 0.12, at.x.sub(at.y).mul(0.7)));
    albedo = mix(mix(albedo, tone, head), rgb(STEEL_DARK).mul(0.55), coverage(sdSegment(at, vec2(-0.08, -0.05), vec2(0.08, 0.05), 0.018)).mul(head));
  }
  const mark = s.sub(vec2(0, ledY));
  albedo = mix(albedo, rgb(STEEL_DARK).mul(0.85), coverage(min(sdCircle(mark, 0.16), spikes(mark, 0.27, 0.028))).mul(0.7));

  const alpha = cover.add(cover.oneMinus().mul(drop));
  material.colorNode = lit(albedo).mul(cover).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  return material;
}

export function ledMaterial(masks: readonly N<'float'>[], width: number): THREE.MeshBasicNodeMaterial {
  const material = sheet(), s = uv().sub(0.5).mul(vec2(width, LED_HEIGHT));
  const digits = masks.length, index = floor(s.x.div(DIGIT).add(digits / 2)).clamp(0, digits - 1);
  let mask: N<'float'> = masks[digits - 1]!;
  for (let k = digits - 2; k >= 0; k--) mask = select(index.equal(k), masks[k]!, mask);
  const cell = s.sub(vec2(index.add(0.5).mul(DIGIT).sub(digits * DIGIT / 2), 0)).div(0.84);
  const slant = vec2(cell.x.sub(cell.y.mul(0.1)), cell.y), pixel = fwidth(s.x).div(0.84);
  let lit: N<'float'> = float(0), all: N<'float'> = float(0);
  SEGMENTS.forEach(([ax, ay, bx, by], bit) => {
    const piece = sdSegment(slant, vec2(ax, ay), vec2(bx, by), 0.045).div(pixel).negate().add(0.5).clamp(0, 1);
    all = max(all, piece);
    lit = max(lit, piece.mul(mod(floor(mask.div(2 ** bit)), 2)));
  });
  const glow = exp(abs(s.y).mul(-4)).mul(0.08);
  material.colorNode = mix(rgb(SEGMENT_OFF), rgb(SEGMENT_ON).mul(2.6), step(1e-3, lit)).add(rgb(SEGMENT_ON).mul(glow));
  material.opacityNode = max(max(lit, all.mul(0.55)), glow);
  return material;
}

export function cursorMaterial(pinch: N<'float'>, tone: N<'vec3'>, span: number): THREE.MeshBasicNodeMaterial {
  const material = sheet(), s = uv().sub(0.5).mul(span);
  const reach = float(0.54).sub(pinch.mul(0.08));
  const ring = abs(sdBox(s, vec2(reach), 0.12)).sub(0.04);
  const arms = max(ring, reach.sub(0.22).sub(min(abs(s.x), abs(s.y))));
  const bracket = coverage(arms), halo = coverage(arms.sub(0.05));
  const frame = coverage(abs(sdBox(s, vec2(1.5), 0.16)).sub(0.02)), frameHalo = coverage(abs(sdBox(s, vec2(1.5), 0.16)).sub(0.045));
  const lines = max(bracket, frame.mul(0.75));
  const alpha = max(lines, max(halo.mul(0.85), frameHalo.mul(0.55)));
  material.colorNode = mix(rgb(0x0b0e12), tone, lines.div(max(alpha, 1e-4)));
  material.opacityNode = alpha;
  return material;
}

export function linkMaterial(strength: N<'float'>, length: N<'float'>): THREE.MeshBasicNodeMaterial {
  const material = sheet(THREE.AdditiveBlending), t = uv();
  const phase = fract(t.x.mul(length).mul(2.2)), dash = smoothstep(0.05, 0.12, phase).mul(smoothstep(0.55, 0.62, phase).oneMinus());
  const across = smoothstep(0.15, 0.5, abs(t.y.sub(0.5))).oneMinus();
  material.colorNode = rgb(PING).mul(1.3);
  material.opacityNode = dash.mul(across).mul(strength);
  return material;
}
