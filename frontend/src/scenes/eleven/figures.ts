import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Level = THREE.UniformNode<'float', number>;

export const TOKEN = 1.3;
export const SPRITE = 9;
export const BALL = 0.5;
export const LIFT = 0.55;
export const CURSOR = 10;
export const BALL_SPRITE = 2 * BALL + 0.3;
export const SHADOW_SPRITE = 3.2;
export const TETHER = 0.6;
export const STATURE = 1.8;
export const MAST = 34;

export const KITS = [0xff9a3c, 0xa98bff, 0x3f73e0, 0xf0d24a] as const;
const INKS = [0x2b1606, 0x1c1233, 0xd9e0ee, 0x2a2206] as const;
const FLAG = [0xffd23c, 0xe8412f] as const;
const COLS = 4;
const CELL_PX = 128;
const SANS = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const rotate = (p: N<'vec2'>, angle: N<'float'>): N<'vec2'> => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
};

const pick = (index: N<'float'>, colors: readonly number[]): N<'vec3'> =>
  colors.slice(0, -1).reduceRight<N<'vec3'>>((rest, color, i) => select(index.lessThan(i + 0.5), rgb(color), rest), rgb(colors.at(-1)!));

export interface Layer { color: N<'vec3'>; alpha: N<'float'> }

export const over = (below: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(below.color.mul(alpha.oneMinus())),
  alpha: alpha.add(below.alpha.mul(alpha.oneMinus())),
});

export const finish = (material: THREE.MeshBasicNodeMaterial, layer: Layer) => {
  material.colorNode = layer.color.div(max(layer.alpha, 1e-4));
  material.opacityNode = layer.alpha;
  return material;
};

export interface Numerals { texture: THREE.DataTexture; rows: number }

export function numerals(count: number): Numerals {
  const rows = Math.ceil(count / COLS), canvas = document.createElement('canvas');
  canvas.width = COLS * CELL_PX;
  canvas.height = rows * CELL_PX;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Eleven shirt numbers need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  for (let i = 0; i < count; i++) {
    const text = String(i + 1), font = `800 100px ${SANS}`;
    context.font = font;
    const probe = context.measureText(text);
    const size = 100 * Math.min(0.78 * CELL_PX / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent), 0.86 * CELL_PX / probe.width);
    context.font = font.replace('100px', `${size}px`);
    const fit = context.measureText(text);
    context.fillText(text, (i % COLS + 0.5) * CELL_PX, (Math.floor(i / COLS) + 0.5) * CELL_PX + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2);
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const map = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  return { texture: map, rows };
}

const glyph = (atlas: Numerals, index: N<'float'>, local: N<'vec2'>): N<'float'> => {
  const inside = step(0, local.x).mul(step(local.x, 1)).mul(step(0, local.y)).mul(step(local.y, 1));
  const cell = vec2(mod(index, COLS), floor(index.div(COLS)));
  const at = vec2(cell.x.add(clamp(local.x, 0.004, 0.996)), cell.y.add(clamp(local.y.oneMinus(), 0.004, 0.996)));
  return texture(atlas.texture, at.div(vec2(COLS, atlas.rows))).r.mul(inside);
};

const glovesAt = (q: N<'vec2'>): N<'float'> =>
  min(sdCircle(q.sub(vec2(1.72 * TOKEN, 0.44 * TOKEN)), 0.3 * TOKEN), sdCircle(q.sub(vec2(1.72 * TOKEN, -0.44 * TOKEN)), 0.3 * TOKEN));

const bodyAt = (point: N<'vec2'>, pose: N<'float'>, heading: N<'float'>): N<'float'> => {
  const q = rotate(point, heading);
  const upright = sdCircle(point, TOKEN);
  const lying = sdSegment(q, vec2(-0.95 * TOKEN, 0), vec2(0.95 * TOKEN, 0), 0.52 * TOKEN);
  const stretch = min(sdSegment(q, vec2(-0.6 * TOKEN, 0), vec2(1.2 * TOKEN, 0), 0.5 * TOKEN), glovesAt(q));
  return select(pose.lessThan(0.5), upright, select(pose.lessThan(1.5), lying, stretch));
};

export function tokenShadowMaterial(towers: readonly THREE.Vector2[]): THREE.MeshBasicNodeMaterial {
  const material = sheet(), look = attribute<'vec4'>('aLook', 'vec4');
  const p = uv().sub(0.5).mul(SPRITE), centre = vec2(positionWorld.x, positionWorld.z.negate()).sub(p);
  const height = select(look.z.lessThan(0.5), float(1), float(0.3));
  const cast = towers.map((tower) => {
    const away = centre.sub(vec2(tower.x, tower.y)), reach = length(away);
    const along = away.div(max(reach, 1e-3)), extent = min(reach.mul(1.25 * TOKEN / (MAST - STATURE)), SPRITE / 2 - 0.9).mul(height);
    const fade = clamp(p.x.mul(along.x).add(p.y.mul(along.y)).div(max(extent, 1e-3)), 0, 1);
    return smoothstep(0.3, -0.25, sdSegment(p, vec2(0), along.mul(extent), 0.46 * TOKEN)).mul(smoothstep(1, 0.6, fade)).mul(0.17);
  }).reduce((a, b) => a.add(b));
  const contact = smoothstep(0.35, -0.3, bodyAt(p, look.z, look.w).sub(0.08)).mul(0.3);
  material.colorNode = vec3(0);
  material.opacityNode = min(cast.add(contact), 0.62);
  return material;
}

export function tokenMaterial(atlas: Numerals): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const look = attribute<'vec4'>('aLook', 'vec4'), motion = attribute<'vec4'>('aMotion', 'vec4');
  const kit = look.x, label = look.y, pose = look.z, heading = look.w;
  const standing = step(pose, 0.5), lying = step(0.5, pose).mul(step(pose, 1.5)), diving = step(1.5, pose);
  const p = uv().sub(0.5).mul(SPRITE);
  const body = bodyAt(p, pose, heading);
  const shirt = pick(kit, KITS), ink = pick(kit, INKS);
  const r = length(p).div(TOKEN);
  const dome = float(1.12).sub(r.mul(r).mul(0.32)).add(p.x.mul(-0.06)).add(p.y.mul(0.09));
  const worn = mix(shirt.mul(dome), mix(shirt, rgb(0x6f8f5a), 0.35).mul(0.9), lying.add(diving.mul(0.2)));
  const rim = coverage(abs(body.add(0.09)).sub(0.09));
  const gloves = coverage(glovesAt(rotate(p, heading)).add(0.04)).mul(diving);
  const kitted = mix(mix(worn, rgb(0xe9edf2), gloves), shirt.mul(0.42), rim);
  const numeral = glyph(atlas, label, p.div(1.36 * TOKEN).add(0.5)).mul(standing);
  const facing = rotate(p, motion.x);
  const wedge = max(abs(facing.y).sub(float(TOKEN * 1.5).sub(facing.x).mul(0.72)), float(TOKEN * 1.1).sub(facing.x));
  const staffX = 0.78 * TOKEN, staffY = 1.62 * TOKEN, cloth = 0.36 * TOKEN;
  const pole = sdSegment(p, vec2(staffX, 0.55 * TOKEN), vec2(staffX, 1.9 * TOKEN), 0.06 * TOKEN);
  const banner = sdBox(p.sub(vec2(staffX + cloth, staffY)), vec2(cloth, 0.26 * TOKEN), 0.03);
  const quarter = abs(step(staffX + cloth, p.x).sub(step(staffY, p.y)));
  const flagged = step(0.5, motion.z);
  let layer: Layer = { color: vec3(0), alpha: float(0) };
  layer = over(layer, shirt.mul(1.25), coverage(wedge).mul(clamp(motion.y, 0, 1)).mul(standing).mul(0.85));
  layer = over(layer, mix(kitted, ink, numeral), coverage(body));
  layer = over(layer, rgb(0x2a2a2a), coverage(pole).mul(flagged));
  layer = over(layer, mix(rgb(FLAG[0]), rgb(FLAG[1]), quarter), coverage(banner).mul(flagged));
  return finish(material, layer);
}

export function tokenRimMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet(), look = attribute<'vec4'>('aLook', 'vec4');
  const p = uv().sub(0.5).mul(SPRITE), body = bodyAt(p, look.z, look.w), shirt = pick(look.x, KITS);
  const edge = abs(body.add(0.06)).sub(0.06);
  material.colorNode = shirt.mul(1.15);
  material.opacityNode = coverage(edge).mul(0.95);
  return material;
}

export interface CursorLevels { stamina: Level; ping: Level; aimX: Level; aimY: Level; glow: Level }

export function cursorMaterial(accent: number): { material: THREE.MeshBasicNodeMaterial; levels: CursorLevels } {
  const material = sheet();
  const levels: CursorLevels = { stamina: uniform(1), ping: uniform(0), aimX: uniform(0), aimY: uniform(0), glow: uniform(1) };
  const p = uv().sub(0.5).mul(CURSOR), radius = length(p);
  const ring = abs(radius.sub(TOKEN + 0.4)).sub(0.1);
  const halo = exp(max(ring, 0).mul(-4)).mul(0.28);
  const turn = fract(atan(p.x, p.y).div(2 * Math.PI).add(1));
  const filled = step(turn, levels.stamina);
  const tone = mix(rgb(0xff5a3c), mix(rgb(0xffc24a), rgb(accent), smoothstep(0.45, 0.8, levels.stamina)), smoothstep(0.15, 0.45, levels.stamina)).mul(levels.glow);
  const aim = vec2(levels.aimX, levels.aimY), reach = length(aim), heading = atan(aim.y, aim.x);
  const facing = rotate(p, heading);
  const notch = max(abs(facing.y).sub(float(TOKEN + 1.3).sub(facing.x).mul(0.85)), float(TOKEN + 0.72).sub(facing.x));
  const ping = abs(radius.sub(mix(float(TOKEN + 0.4), float(TOKEN + 3.4), levels.ping))).sub(0.1);
  let layer: Layer = { color: tone.mul(halo), alpha: halo };
  layer = over(layer, rgb(0x0b120e), coverage(ring.sub(0.05)).mul(0.5));
  layer = over(layer, tone.mul(1.9), coverage(ring).mul(filled.mul(0.7).add(0.3)));
  layer = over(layer, tone.mul(1.7), coverage(notch).mul(smoothstep(0.02, 0.4, reach)));
  layer = over(layer, tone.mul(1.6), coverage(ping).mul(levels.ping).mul(0.9));
  return { material: finish(material, layer), levels };
}

export function ballMaterial(spin: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(BALL_SPRITE), d = sdCircle(p, BALL);
  const q = rotate(p, spin);
  const patch = (angle: number, distance: number) => sdCircle(q.sub(vec2(Math.cos(angle) * distance, Math.sin(angle) * distance)), 0.1);
  const panels = min(min(patch(0.3, 0.2), patch(2.4, 0.22)), min(patch(4.3, 0.21), sdCircle(q, 0.07)));
  const shine = smoothstep(BALL, 0, length(p.sub(vec2(-0.12, 0.14))));
  const leather = rgb(0xf6f7f2).mul(shine.mul(0.35).add(0.85));
  const face = mix(leather, rgb(0x20242a), coverage(panels).mul(0.85));
  const rim = coverage(abs(d.add(0.035)).sub(0.035));
  return finish(material, { color: mix(face, rgb(0x121417), rim), alpha: coverage(d) });
}

export function ballShadowMaterial(height: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5).mul(SHADOW_SPRITE), spread = height.mul(0.1).add(1);
  const d = length(p.div(vec2(1.1, 0.8)).div(spread)).sub(BALL);
  material.colorNode = vec3(0);
  material.opacityNode = smoothstep(0.22, -0.18, d).mul(0.5).div(height.mul(0.3).add(1));
  return material;
}

export function trailMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aTrail', 'vec4');
  const span = state.x, width = state.y;
  const p = uv().sub(0.5).mul(vec2(span.add(width), width));
  const d = sdSegment(p, vec2(span.mul(-0.5), 0), vec2(span.mul(0.5), 0), width.mul(0.5));
  const along = clamp(p.x.div(max(span, 1e-3)).add(0.5), 0, 1);
  material.colorNode = rgb(0xf4f8ff);
  material.opacityNode = coverage(d).mul(mix(state.z, state.w, along)).mul(smoothstep(width.mul(0.5), 0, abs(p.y)).mul(0.6).add(0.4));
  return material;
}

export function tetherMaterial(accent: number, span: Level, flow: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const along = uv().x.mul(span), across = uv().y.sub(0.5).mul(TETHER);
  const dash = abs(fract(along.sub(flow).div(1.1)).sub(0.5)).sub(0.22).mul(1.1);
  const line = max(abs(across).sub(0.05), dash);
  const ends = smoothstep(0, 1.2, along).mul(smoothstep(span, span.sub(1.2), along));
  material.colorNode = rgb(accent).mul(1.3);
  material.opacityNode = coverage(line).mul(ends).mul(0.6);
  return material;
}

