import * as THREE from 'three/webgpu';
import { abs, clamp, dot, exp, float, floor, fract, instancedBufferAttribute, instancedDynamicBufferAttribute, length, max, min, mix, mod, normalGeometry, normalWorld, positionWorld, screenCoordinate, select, sign, sin, smoothstep, step, texture, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Level = THREE.UniformNode<'float', number>;

export const PAINT = [0xff7a1a, 0xd31f2c, 0xffd21c, 0xecece6, 0x1d5fe0, 0x12a84a] as const;
export const GOLD = 0xffd166;
export const LAPSE = 0x7a95c4;
const PLASTIC = 0x141518;
const INK = 0x1b1d22;
const CREAM = 0xe9e3d4;
const AMBER = 0xffa12e;
const DESK = [0x1a120c, 0x2c1f15] as const;
const FELT = [0x222c35, 0x2b3742] as const;
const THREAD = 0x8f9daa;
const PRINT = 0xa6b3bf;
const STEEL = 0xb8c0c8;
const FONT = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const GLYPH_PX = 128;
const COLUMNS = 4;
const STICKER = 0.415;
const STICKER_ROUND = 0.085;

export const KEY = new THREE.Vector2(0.36, 0.3);
export const KEY_QUAD = new THREE.Vector2(1.1, 1);
export const TILE = 0.19;
export const TILE_QUAD = 0.62;

export interface Atlas { texture: THREE.DataTexture; columns: number; rows: number; index: ReadonlyMap<string, number> }
export interface Hexagon { x: number; z: number; radius: number }
export interface Console { x: number; z: number; half: THREE.Vector2; led: THREE.Vector2; ledZ: number; cells: Level[] }

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const hexagonGeometry = (radius: number) => new THREE.CircleGeometry(radius, 6).rotateZ(Math.PI / 6).rotateX(-Math.PI / 2);

const basic = (additive = false) => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, ...(additive ? { blending: THREE.AdditiveBlending } : {}) });
const up = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const hash = (p: N<'vec2'>): N<'float'> => fract(sin(dot(p, vec2(12.9898, 78.233))).mul(43758.5453));

const hexagon = (p: N<'vec2'>, radius: number): N<'float'> => {
  const inner = float(radius).mul(0.866025404), k = vec2(-0.866025404, 0.5);
  const a = abs(vec2(p.y, p.x)), b = a.sub(k.mul(min(dot(k, a), 0).mul(2)));
  const c = b.sub(vec2(clamp(b.x, inner.mul(-0.577350269), inner.mul(0.577350269)), inner));
  return length(c).mul(sign(c.y));
};

const composite = (color: N<'vec3'>, face: N<'float'>, under: N<'vec3'>, alpha: N<'float'>) => {
  const total = face.add(face.oneMinus().mul(alpha));
  return { color: color.mul(face).add(under.mul(face.oneMinus().mul(alpha))).div(max(total, 1e-4)), alpha: total };
};

export function glyphAtlas(labels: readonly string[]): Atlas {
  const rows = Math.ceil(labels.length / COLUMNS), canvas = document.createElement('canvas');
  canvas.width = COLUMNS * GLYPH_PX;
  canvas.height = rows * GLYPH_PX;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Pocket cube glyphs need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  labels.forEach((label, i) => {
    context.font = `700 100px ${FONT}`;
    const probe = context.measureText(label);
    const size = 100 * Math.min(0.56 * GLYPH_PX / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent), 0.84 * GLYPH_PX / probe.width);
    context.font = `700 ${size}px ${FONT}`;
    const fit = context.measureText('U');
    context.fillText(label, (i % COLUMNS + 0.5) * GLYPH_PX, (Math.floor(i / COLUMNS) + 0.5) * GLYPH_PX + fit.actualBoundingBoxAscent / 2);
  });
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) alpha[(canvas.height - 1 - y) * canvas.width + x] = pixels[4 * (y * canvas.width + x) + 3]!;
  const out = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  out.minFilter = THREE.LinearMipmapLinearFilter;
  out.magFilter = THREE.LinearFilter;
  out.generateMipmaps = true;
  out.needsUpdate = true;
  return { texture: out, columns: COLUMNS, rows, index: new Map(labels.map((label, i) => [label, i])) };
}

const glyphAt = (atlas: Atlas, glyph: N<'float'>, local: N<'vec2'>): N<'float'> => {
  const col = mod(glyph, atlas.columns), row = floor(glyph.div(atlas.columns));
  const inside = step(0, local.x).mul(step(local.x, 1)).mul(step(0, local.y)).mul(step(local.y, 1)).mul(step(0, glyph));
  const at = vec2(col.add(clamp(local.x, 0.002, 0.998)), float(atlas.rows - 1).sub(row).add(clamp(local.y, 0.002, 0.998))).div(vec2(atlas.columns, atlas.rows));
  return texture(atlas.texture, at).r.mul(inside);
};

const paintOf = (face: N<'float'>): N<'vec3'> =>
  select(face.lessThan(0.5), rgb(PAINT[0]), select(face.lessThan(1.5), rgb(PAINT[1]), select(face.lessThan(2.5), rgb(PAINT[2]), select(face.lessThan(3.5), rgb(PAINT[3]), select(face.lessThan(4.5), rgb(PAINT[4]), rgb(PAINT[5]))))));

export function matMaterial(light: Light2D, half: THREE.Vector2, cubes: readonly Hexagon[], console: Console): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = up();
  const grain = fractal(vec2(p.x.mul(0.16), p.y.mul(2.8)), 5).mul(0.5).add(0.5), knots = fractal(p.mul(0.7), 3).mul(0.5).add(0.5);
  const desk = mix(rgb(DESK[0]), rgb(DESK[1]), grain).mul(knots.mul(0.2).add(0.9));
  const d = sdBox(p, vec2(half.x, half.y), 0.5), inside = coverage(d);
  const weave = fractal(p.mul(11), 3).mul(0.5).add(0.5), cloud = fractal(p.mul(0.45), 4).mul(0.5).add(0.5);
  const felt = mix(rgb(FELT[0]), rgb(FELT[1]), cloud).mul(weave.mul(0.1).add(0.95));
  const q = abs(p).div(vec2(half.x, half.y)), along = select(q.x.greaterThan(q.y), p.y, p.x);
  const stitch = coverage(abs(d.add(0.2)).sub(0.014)).mul(step(0.45, fract(along.mul(4.2))));
  const edge = smoothstep(-0.07, 0, d).mul(0.35);
  let shadow: N<'float'> = smoothstep(0.55, 0, d).mul(inside.oneMinus()).mul(0.55);
  for (const cube of cubes) {
    const at = p.sub(vec2(cube.x + 0.1 * cube.radius, -cube.z - 0.16 * cube.radius));
    shadow = max(shadow, smoothstep(0.38 * cube.radius, -0.2 * cube.radius, hexagon(at, cube.radius * 0.93)).mul(0.62));
  }
  const box = sdBox(p.sub(vec2(console.x + 0.08, -console.z - 0.14)), vec2(console.half.x, console.half.y), 0.28);
  shadow = max(shadow, smoothstep(0.36, -0.12, box).mul(0.6));
  const cloth = mix(felt.mul(edge.oneMinus()), rgb(THREAD), stitch.mul(0.5));
  const surface = mix(desk, cloth, inside).mul(shadow.oneMinus());
  material.colorNode = light.lit(surface);
  return material;
}

export interface CubieAttributes { stick: THREE.InstancedBufferAttribute; side: THREE.InstancedBufferAttribute; seat: THREE.InstancedBufferAttribute; fx: THREE.InstancedBufferAttribute }

export function cubieMaterial(light: Light2D, attributes: CubieAttributes, toward: THREE.Vector3): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  const stick = instancedDynamicBufferAttribute(attributes.stick, 'vec3') as N<'vec3'>;
  const side = instancedBufferAttribute(attributes.side, 'vec3') as N<'vec3'>;
  const seat = instancedBufferAttribute(attributes.seat, 'vec3') as N<'vec3'>;
  const fx = instancedDynamicBufferAttribute(attributes.fx, 'vec4') as N<'vec4'>;
  const n = normalGeometry, outward = step(0.5, dot(n, side)), paint = paintOf(dot(abs(n), stick));
  const p = uv().sub(0.5), d = sdBox(p, vec2(STICKER), STICKER_ROUND), sticker = coverage(d).mul(outward);
  const edge = max(abs(p.x), abs(p.y));
  const plastic = rgb(PLASTIC).add(vec3(smoothstep(0.43, 0.5, edge).mul(0.045)));
  const molded = paint.mul(smoothstep(-0.04, 0, d).mul(-0.2).add(1)).mul(smoothstep(0.2, -0.4, p.x.add(p.y)).mul(0.06).add(0.97));
  const shade = max(dot(normalWorld, vec3(toward.x, toward.y, toward.z)), 0).mul(0.45).add(0.55);
  const body = mix(plastic, molded, sticker).mul(shade);
  const glow = fx.x, dim = fx.y;
  const grey = vec3(dot(body, vec3(0.2126, 0.7152, 0.0722))).mul(rgb(0xa9bcdc));
  const muted = mix(body, grey, dim.mul(0.75)).mul(dim.mul(-0.3).add(1));
  const emission = paint.mul(sticker.mul(glow.mul(0.22))).add(rgb(GOLD).mul(glow.mul(0.05)));
  material.colorNode = light.lit(muted).add(emission);
  const sweep = clamp(positionWorld.z.sub(seat.y.sub(seat.z)).div(seat.z.mul(2)), 0, 1);
  const h = mix(hash(floor(screenCoordinate.xy.div(3))), sweep, 0.62);
  material.maskNode = h.greaterThanEqual(fx.z).and(h.lessThan(fx.w));
  return material;
}

export function arrowMaterial(alpha: Level, turn: Level, accent: number): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false });
  const raw = uv().sub(0.5).mul(2), p = vec2(raw.x.mul(turn), raw.y);
  const aperture = 2.2, radius = 0.5, thick = 0.07, head = 0.24, back = 0.02, wide = 0.2;
  const sc = vec2(Math.sin(aperture), Math.cos(aperture)), q = vec2(abs(p.x), p.y);
  const arc = select(q.x.mul(sc.y).greaterThan(q.y.mul(sc.x)), length(q.sub(sc.mul(radius))), abs(length(q).sub(radius))).sub(thick);
  const end = vec2(Math.sin(aperture) * radius, Math.cos(aperture) * radius), tangent = vec2(Math.cos(aperture), -Math.sin(aperture)), normal = vec2(Math.sin(aperture), Math.cos(aperture));
  const rel = p.sub(end), a = dot(rel, tangent), c = abs(dot(rel, normal));
  const slant = c.mul(head + back).add(a.mul(wide)).sub(wide * head).div(Math.hypot(head + back, wide));
  const tip = max(a.negate().sub(back), slant), shape = min(arc, tip);
  const fill = coverage(shape), rim = coverage(shape.sub(0.05));
  material.colorNode = mix(rgb(INK), mix(vec3(1), rgb(accent), 0.35).mul(1.05), fill.div(max(rim, 1e-4)));
  material.opacityNode = rim.mul(alpha);
  return material;
}

export function keyMaterial(light: Light2D, atlas: Atlas, attribute: THREE.InstancedBufferAttribute, accent: number): THREE.MeshBasicNodeMaterial {
  const material = basic(), k = instancedDynamicBufferAttribute(attribute, 'vec4') as N<'vec4'>, glyph = k.x, press = k.y;
  const t = uv().sub(0.5).mul(vec2(KEY_QUAD.x, KEY_QUAD.y)), c = t.add(vec2(0, press.mul(0.025)));
  const capD = sdBox(c, vec2(KEY.x, KEY.y), 0.09), cap = coverage(capD);
  const skirt = sdBox(t.sub(vec2(0, -0.04)), vec2(KEY.x, KEY.y), 0.09);
  const shadow = smoothstep(0.1, -0.05, skirt).mul(press.mul(-0.5).add(0.75));
  const top = mix(rgb(0x2a2e36), rgb(0x3b404a), smoothstep(-KEY.y, KEY.y, c.y)).mul(smoothstep(-0.05, 0, capD).mul(c.y.div(KEY.y).mul(0.25)).add(1));
  const ink = glyphAt(atlas, glyph, c.div(KEY.y * 1.7).add(0.5));
  const inked = mix(top.mul(press.mul(0.35).add(1)), mix(rgb(0xd9dee6), rgb(accent).mul(1.7), press), ink);
  const halo = exp(max(capD, 0).mul(-15)).mul(press).mul(smoothstep(0.22, 0.02, capD));
  const out = composite(light.lit(inked), cap, mix(vec3(0), rgb(accent).mul(1.4), halo.div(max(max(halo, shadow), 1e-4))), max(shadow, halo));
  material.colorNode = out.color;
  material.opacityNode = out.alpha;
  return material;
}

export function tileMaterial(light: Light2D, atlas: Atlas, attribute: THREE.InstancedBufferAttribute, accent: number): THREE.MeshBasicNodeMaterial {
  const material = basic(), k = instancedDynamicBufferAttribute(attribute, 'vec4') as N<'vec4'>, glyph = k.x, fill = k.y, fresh = k.z, mood = k.w;
  const t = uv().sub(0.5).mul(TILE_QUAD), size = fill.mul(TILE);
  const socket = coverage(abs(sdBox(t, vec2(TILE), 0.05)).sub(0.012)).mul(0.5);
  const d = sdBox(t, vec2(size), size.mul(0.26)), tile = coverage(d).mul(step(0.02, fill));
  const cream = mix(rgb(CREAM).mul(0.82), rgb(CREAM), smoothstep(-TILE, TILE, t.y));
  const ink = glyphAt(atlas, glyph, t.div(max(size, 0.01).mul(1.75)).add(0.5));
  const gold = max(mood, 0), blue = max(mood.negate(), 0);
  const tone = mix(mix(cream, rgb(GOLD).mul(1.2), gold.mul(0.55)), rgb(LAPSE), blue.mul(0.6));
  const face = mix(tone, rgb(INK), ink);
  const ring = coverage(abs(d.sub(0.045)).sub(0.018)).mul(fresh);
  const color = mix(mix(rgb(PRINT).mul(0.5), light.lit(face), tile), rgb(accent).mul(1.6), ring);
  material.colorNode = color;
  material.opacityNode = max(max(tile, ring), socket.mul(tile.oneMinus()));
  return material;
}

export function labelMaterial(atlas: Atlas, attribute: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
  const material = basic(), k = instancedDynamicBufferAttribute(attribute, 'vec4') as N<'vec4'>;
  const ink = glyphAt(atlas, k.x, uv());
  material.colorNode = rgb(PRINT);
  material.opacityNode = ink.mul(k.y);
  return material;
}

export function pinMaterial(light: Light2D): THREE.MeshBasicNodeMaterial {
  const material = basic(), t = uv().sub(0.5);
  const head = sdCircle(t, 0.3), body = coverage(head), rim = smoothstep(-0.08, 0, head);
  const dome = smoothstep(0.3, 0, length(t.sub(vec2(-0.09, 0.09)))).mul(0.55).add(0.55);
  const shadow = smoothstep(0.1, -0.05, sdCircle(t.sub(vec2(0.05, -0.06)), 0.3)).mul(0.6);
  const steel = rgb(STEEL).mul(dome).mul(rim.mul(-0.45).add(1));
  const out = composite(light.lit(steel), body, vec3(0), shadow);
  material.colorNode = out.color;
  material.opacityNode = out.alpha;
  return material;
}

const SEGMENTS = [[-1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, -1], [-1, -1, 1, -1], [-1, 0, -1, -1], [-1, 1, -1, 0], [-1, 0, 1, 0]] as const;
const DIGITS = [63, 6, 91, 79, 102, 109, 125, 7, 127, 111] as const;
export const SLASH = DIGITS.length;

const maskOf = (value: N<'float'>): N<'float'> => {
  let mask: N<'float'> = float(0);
  for (let digit = DIGITS.length - 1; digit >= 0; digit--) mask = select(value.lessThan(digit + 0.5).and(value.greaterThan(digit - 0.5)), float(DIGITS[digit]!), mask);
  return mask;
};

export function consoleMaterial(light: Light2D, console: Console, keys: readonly THREE.Vector2[]): THREE.MeshBasicNodeMaterial {
  const material = basic(), p = up().sub(vec2(console.x, -console.z));
  const bodyD = sdBox(p, vec2(console.half.x, console.half.y), 0.28), body = coverage(bodyD);
  const shell = mix(rgb(0x121317), rgb(0x1d1f25), smoothstep(-console.half.y, console.half.y, p.y)).mul(smoothstep(-0.06, 0, bodyD).mul(0.6).add(1));
  const ledAt = p.sub(vec2(0, -(console.ledZ - console.z))), windowD = sdBox(ledAt, vec2(console.led.x, console.led.y), 0.08), glass = coverage(windowD);
  const bezel = coverage(abs(windowD.add(0.012)).sub(0.014));
  let bed: N<'float'> = float(0);
  for (const key of keys) bed = max(bed, coverage(sdBox(p.sub(vec2(key.x - console.x, -(key.y - console.z))), vec2(KEY.x + 0.06, KEY.y + 0.06), 0.13)));
  const cells = console.cells.length, pitch = (2 * console.led.x - 0.24) / cells, w = pitch * 0.27, h = console.led.y * 0.62;
  let lit: N<'float'> = float(0), ghost: N<'float'> = float(0);
  console.cells.forEach((cell, i) => {
    const centre = -console.led.x + 0.12 + (i + 0.5) * pitch, local = ledAt.sub(vec2(centre, 0)), slanted = vec2(local.x.sub(local.y.mul(0.12)), local.y);
    const mask = maskOf(cell);
    SEGMENTS.forEach(([x0, y0, x1, y1], bit) => {
      const gap = 0.16, ax = x0 + (x1 - x0) * gap, ay = y0 + (y1 - y0) * gap, bx = x1 - (x1 - x0) * gap, by = y1 - (y1 - y0) * gap;
      const segment = coverage(sdSegment(slanted, vec2(ax * w, ay * h), vec2(bx * w, by * h), w * 0.2));
      lit = max(lit, segment.mul(mod(floor(mask.div(2 ** bit)), 2)));
      ghost = max(ghost, segment);
    });
    const slash = coverage(sdSegment(slanted, vec2(-w * 0.8, -h), vec2(w * 0.8, h), w * 0.2)).mul(step(SLASH - 0.5, cell));
    lit = max(lit, slash);
  });
  const screen = mix(rgb(0x1a0904), rgb(0x2a0f06), ghost.mul(0.8));
  const shown = mix(mix(mix(shell, rgb(0x08090b), bed.mul(0.9)), mix(screen, rgb(AMBER).mul(2.4), lit), glass), rgb(0x3a3d44), bezel);
  material.colorNode = light.lit(shown).add(rgb(AMBER).mul(lit.mul(glass).mul(0.9)));
  material.opacityNode = body;
  return material;
}
