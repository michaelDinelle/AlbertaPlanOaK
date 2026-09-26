import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, mod, positionWorld, pow, sin, smoothstep, step, texture, uniform, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Scalar = N<'float'> | number;

export const FIREWORK = [0xff5148, 0xffcf3a, 0x46d27e, 0xeef2ff, 0x4c9bff] as const;
export const GOLD = 0xffd98a;
export const PAPER = 0xffb45c;
export const MISFIRE = 0xff4a3a;

export const CARD_W = 1;
export const CARD_H = 1.4;
export const CARD_QW = 1.56;
export const CARD_QH = 2.08;
export const CHIP_W = 0.66;
export const CHIP_H = 0.32;
export const TOKEN_Q = 0.72;
export const LANTERN_Q = 0.98;
export const LANTERN_R = 0.27;
export const RING_R = 0.375;
export const CELL_W = 0.44;
export const CELL_H = 0.5;
export const CELL_Q = 0.3;
export const BURST_Q = 2.6;
export const BADGE_Q = 0.72;
export const PLATE_W = 0.8;
export const PLATE_H = 0.4;

const CORNER = 0.1;
const GLYPHS = 10;
const FONT = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';

export interface Atlas { texture: THREE.DataTexture; fill: number }

export function digitAtlas(): Atlas {
  const px = 128, canvas = document.createElement('canvas');
  canvas.width = GLYPHS * px;
  canvas.height = px;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Hanabi numerals need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.font = `800 ${0.8 * px}px ${FONT}`;
  const probe = context.measureText('8'), cap = probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent;
  for (let g = 0; g < GLYPHS; g++) context.fillText(String(g), (g + 0.5) * px, px / 2 + cap / 2 - probe.actualBoundingBoxDescent);
  const pixels = context.getImageData(0, 0, canvas.width, px).data, alpha = new Uint8Array(canvas.width * px);
  for (let y = 0; y < px; y++) for (let x = 0; x < canvas.width; x++) alpha[(px - 1 - y) * canvas.width + x] = pixels[4 * (y * canvas.width + x) + 3]!;
  const out = new THREE.DataTexture(alpha, canvas.width, px, THREE.RedFormat, THREE.UnsignedByteType);
  out.minFilter = THREE.LinearMipmapLinearFilter;
  out.magFilter = THREE.LinearFilter;
  out.generateMipmaps = true;
  out.needsUpdate = true;
  return { texture: out, fill: cap / px };
}

const scalar = (value: Scalar): N<'float'> => typeof value === 'number' ? float(value) : value;

const glyph = (atlas: Atlas, local: N<'vec2'>, centre: N<'vec2'>, height: Scalar, index: Scalar): N<'float'> => {
  const q = local.sub(centre).div(scalar(height).div(atlas.fill)).add(0.5);
  const inside = step(0, q.x).mul(step(q.x, 1)).mul(step(0, q.y)).mul(step(q.y, 1));
  const c = clamp(q, vec2(0), vec2(1));
  return texture(atlas.texture, vec2(scalar(index).add(c.x).div(GLYPHS), c.y)).r.mul(inside);
};

const bit = (mask: N<'float'>, k: number): N<'float'> => mod(floor(mask.div(2 ** k)), 2);

const inks = uniformArray(FIREWORK.map((hex) => new THREE.Color(hex)), 'color');
export const inkOf = (index: N<'float'>): N<'vec3'> => inks.element(clamp(index, 0, FIREWORK.length - 1).toInt()) as unknown as N<'vec3'>;

interface Layer { color: N<'vec3'>; alpha: N<'float'> }
const over = (under: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => {
  const a = alpha.add(under.alpha.mul(alpha.oneMinus()));
  return { color: color.mul(alpha).add(under.color.mul(under.alpha).mul(alpha.oneMinus())).div(max(a, 1e-4)), alpha: a };
};

const chevron = (p: N<'vec2'>, tip: N<'vec2'>, span: number, depth: number, width: number, up: N<'float'>): N<'float'> => {
  const back = up.mul(2).sub(1).mul(-depth);
  const left = tip.add(vec2(-span, back)), right = tip.add(vec2(span, back));
  return min(sdSegment(p, left, tip, width), sdSegment(p, right, tip, width));
};

const surface = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

export interface TableZone { x: number; z: number; w: number; h: number }

export function tableMaterial(light: Light2D, halfW: number, halfH: number, zones: readonly TableZone[]): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = positionWorld.xz, edge = sdBox(p, vec2(halfW, halfH), 0.55);
  const reach = length(p.mul(vec2(0.85 / halfW, 1.05 / halfH)));
  const lacquer = mix(rgb(0x1c2146), rgb(0x0a0c1c), smoothstep(0.1, 1.1, reach));
  const streak = fractal(p.mul(vec2(0.28, 2.4)), 4).mul(0.5).add(0.5);
  const dust = smoothstep(0.77, 0.83, fractal(p.mul(13), 2).mul(0.5).add(0.5)).mul(smoothstep(0.55, 0.85, fractal(p.mul(0.9).add(7.1), 3).mul(0.5).add(0.5)));
  let top = lacquer.mul(streak.mul(0.16).add(0.92)).add(rgb(GOLD).mul(dust.mul(0.12)));
  for (const zone of zones) {
    const line = coverage(abs(sdBox(p.sub(vec2(zone.x, zone.z)), vec2(zone.w / 2, zone.h / 2), 0.28)).sub(0.011));
    top = mix(top, rgb(0xc9a45c), line.mul(0.32));
  }
  const rim = smoothstep(-0.36, -0.33, edge);
  const lip = exp(abs(edge.add(0.07)).mul(-40)).mul(0.35);
  const vermilion = mix(rgb(0x6b150e), rgb(0xa8321f), smoothstep(-0.34, -0.05, edge)).add(rgb(0xffb07a).mul(lip));
  const inlay = coverage(abs(edge.add(0.4)).sub(0.014));
  top = mix(mix(top, vermilion, rim), rgb(0xe6be70), inlay.mul(0.8));
  const floorTone = rgb(0x040509).mul(fractal(p.mul(0.6), 3).mul(0.2).add(0.9));
  material.colorNode = light.lit(mix(top, floorTone, coverage(edge.negate())));
  return material;
}

export interface CardBuffers { card: THREE.InstancedBufferAttribute; look: THREE.InstancedBufferAttribute }

export function cardMaterial(light: Light2D, atlas: Atlas, buffers: CardBuffers, accent: number): THREE.MeshBasicNodeMaterial {
  const card = instancedDynamicBufferAttribute(buffers.card, 'vec4') as N<'vec4'>, look = instancedDynamicBufferAttribute(buffers.look, 'vec4') as N<'vec4'>;
  const colour = card.x, rank = card.y, kind = card.z, bits = card.w;
  const flags = look.x, glow = look.y, alpha = look.z, lift = look.w;
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CARD_QW, CARD_QH)), half = vec2(CARD_W / 2, CARD_H / 2);
  const d = sdBox(t, half, CORNER), body = coverage(d);
  const shade = t.y.div(CARD_H).add(0.5);
  const knownColour = step(colour, 4.5), knownRank = step(rank, 4.5);
  const face = step(kind, 0.5), back = step(0.5, kind).mul(step(kind, 2.5)), notes = step(0.5, kind).mul(step(kind, 1.5)), slot = step(2.5, kind);
  const clued = bit(flags, 0), marked = bit(flags, 1), touched = bit(flags, 2), below = bit(flags, 3);
  const colours = mod(bits, 32), ranks = floor(bits.div(32));
  const narrowColours = step(colours, 30.5), narrowRanks = step(ranks, 30.5);
  const ink = inkOf(colour), pink = rgb(accent);

  const rel = t.sub(vec2(0, 0.1)), r = length(rel), angle = atan(rel.y, rel.x);
  const spokes = pow(abs(cos(angle.mul(9))), float(26));
  const burst = spokes.mul(smoothstep(0.13, 0.22, r)).mul(smoothstep(0.46, 0.28, r)).mul(0.5)
    .add(exp(abs(r.sub(0.43)).mul(-55)).mul(pow(abs(cos(angle.mul(9))), float(70))).mul(0.9));
  const frameLine = coverage(abs(d.add(0.075)).sub(0.013));
  const big = glyph(atlas, t, vec2(0, 0.08), 0.5, rank.add(1));
  const bigShadow = glyph(atlas, t.sub(vec2(0.022, -0.028)), vec2(0, 0.08), 0.5, rank.add(1));
  const corner = glyph(atlas, t, vec2(-0.33, 0.52), 0.19, rank.add(1));
  const night = mix(rgb(0x0a0c24), rgb(0x1a2052), shade);
  const band = max(narrowColours, narrowRanks).mul(smoothstep(-0.3, -0.34, t.y));
  let faceColor = night.add(ink.mul(burst.mul(0.42)));
  faceColor = mix(faceColor, rgb(0x04050d), bigShadow.mul(0.7));
  faceColor = mix(faceColor, ink, frameLine.mul(0.95));
  faceColor = mix(faceColor, mix(ink, vec3(1), 0.2), max(big, corner));
  faceColor = mix(faceColor, rgb(0x05060f), band.mul(0.72));

  const pipRow = (y: number, radius: number, strength: N<'float'>, base: N<'vec3'>): N<'vec3'> => {
    let out = base;
    for (let k = 0; k < 5; k++) {
      const q = t.sub(vec2(-0.32 + 0.16 * k, y)), on = bit(colours, k);
      const fill = coverage(sdCircle(q, radius)).mul(on), ring = coverage(abs(sdCircle(q, radius * 0.8)).sub(0.009)).mul(on.oneMinus());
      out = mix(out, rgb(FIREWORK[k]!), fill.mul(strength));
      out = mix(out, rgb(0x5a5f80), ring.mul(strength).mul(0.8));
    }
    return out;
  };
  const numeralRow = (y: number, height: number, strength: N<'float'>, base: N<'vec3'>, tone: N<'vec3'>): N<'vec3'> => {
    let out = base;
    for (let k = 0; k < 5; k++) {
      const g = glyph(atlas, t, vec2(-0.32 + 0.16 * k, y), height, k + 1);
      out = mix(out, mix(rgb(0x3c4064), tone, bit(ranks, k)), g.mul(strength));
    }
    return out;
  };
  faceColor = pipRow(-0.43, 0.047, narrowColours, faceColor);
  faceColor = numeralRow(-0.58, 0.12, narrowRanks, faceColor, vec3(0.95));

  const backInk = mix(pink, ink, knownColour.mul(notes));
  const petals = pow(abs(cos(angle.mul(6))), float(5)).mul(smoothstep(0.08, 0.14, r)).mul(smoothstep(0.34, 0.24, r));
  const halo = exp(abs(r.sub(0.36)).mul(-70)).mul(0.8).add(exp(abs(r.sub(0.43)).mul(-90)).mul(pow(abs(cos(angle.mul(12))), float(40))));
  const hush = knownRank.mul(notes).mul(0.75).oneMinus();
  let backColor = mix(rgb(0x120f2c), rgb(0x251c4e), shade);
  backColor = backColor.add(backInk.mul(petals.mul(0.3).add(halo.mul(0.45)).mul(hush)));
  backColor = mix(backColor, backInk, frameLine.mul(0.9));
  const known = glyph(atlas, t, vec2(0, 0.06), 0.48, rank.add(1)).mul(knownRank).mul(notes);
  const knownShadow = glyph(atlas, t.sub(vec2(0.022, -0.028)), vec2(0, 0.06), 0.48, rank.add(1)).mul(knownRank).mul(notes);
  backColor = mix(backColor, rgb(0x05040d), knownShadow.mul(0.7));
  backColor = mix(backColor, mix(vec3(0.96), ink, knownColour), known);
  backColor = pipRow(0.5, 0.052, notes, backColor);
  backColor = numeralRow(-0.52, 0.13, notes, backColor, vec3(1));

  const dash = step(0.5, fract(t.x.add(t.y).mul(4.5)));
  const slotLine = coverage(abs(d.add(0.03)).sub(0.014)).mul(dash);
  const slotTint = mix(vec3(0.55), ink, knownColour);
  const slotDot = coverage(sdCircle(t.sub(vec2(0, 0.08)), 0.09)).mul(knownColour);

  const cardColor = mix(mix(backColor, faceColor, face), slotTint, slot);
  const inkMask = max(max(big, corner).mul(face), max(frameLine, known.mul(0.6)).mul(max(face, back)));
  const lit = min(light.lit(cardColor), cardColor.mul(1.2)).add(cardColor.mul(inkMask.mul(0.35)).mul(dot(ink, vec3(0.2126, 0.7152, 0.0722)).mul(-0.75).add(1)));

  const shadowD = sdBox(t.sub(vec2(0.05, lift.mul(-0.07).sub(0.06))), half, CORNER);
  const shadow = smoothstep(lift.mul(0.1).add(0.08), -0.06, shadowD).mul(lift.mul(0.2).add(0.42)).mul(slot.oneMinus());
  const glowTone = mix(vec3(1, 0.93, 0.85), ink, knownColour.mul(0.7));
  const haloA = exp(max(d, 0).mul(-13)).mul(glow).mul(step(0, d)).mul(smoothstep(0.21, 0.1, d));
  const ringA = coverage(abs(d.sub(0.048)).sub(0.02)).mul(clued).mul(slot.oneMinus());
  const tip = vec2(0, float(1).sub(below.mul(2)).mul(CARD_H / 2 + 0.08));
  const pointer = coverage(chevron(t, tip, 0.1, 0.09, 0.024, below)).mul(touched);
  const badgeAt = t.sub(vec2(CARD_W / 2 - 0.02, CARD_H / 2 - 0.02));
  const badge = coverage(sdCircle(badgeAt, 0.12)).mul(marked).mul(max(face, notes));
  const arrow = coverage(chevron(badgeAt, vec2(0, 0.045), 0.055, 0.06, 0.018, float(1))).mul(badge);

  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, glowTone.mul(1.4), haloA);
  layer = over(layer, rgb(GOLD).mul(1.1), ringA);
  layer = over(layer, lit, body.mul(slot.oneMinus()));
  layer = over(layer, light.lit(slotTint).mul(1.3), max(slotLine, slotDot).mul(slot).mul(0.8));
  layer = over(layer, mix(rgb(0x2fae62), rgb(0xf4fff6), arrow), badge);
  layer = over(layer, vec3(1, 0.97, 0.92), pointer);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export function chipMaterial(atlas: Atlas, chip: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(chip, 'vec4') as N<'vec4'>, colour = v.x, rank = v.y, alpha = v.z;
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CHIP_W + 0.08, CHIP_H + 0.08)), d = sdBox(t, vec2(CHIP_W / 2, CHIP_H / 2), 0.07);
  const ink = inkOf(colour);
  const swatch = coverage(sdCircle(t.sub(vec2(-0.14, 0)), 0.082));
  const numeral = glyph(atlas, t, vec2(0.12, 0), 0.19, rank.add(1));
  const edge = coverage(abs(d.add(0.012)).sub(0.01));
  let color: N<'vec3'> = rgb(0x0b0c1c);
  color = mix(color, ink.mul(0.55), edge);
  color = mix(color, ink, swatch);
  color = mix(color, mix(ink, vec3(1), 0.35), numeral);
  material.colorNode = color;
  material.opacityNode = coverage(d).mul(alpha);
  return material;
}

export function burstMaterial(burst: THREE.InstancedBufferAttribute, clock: N<'float'>): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(burst, 'vec4') as N<'vec4'>, colour = v.x, height = v.y, flare = v.z, alpha = v.w, seed = colour.mul(0.37).add(0.11);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: THREE.AdditiveBlending });
  const t = uv().sub(0.5).mul(BURST_Q), r = length(t), angle = atan(t.y, t.x);
  const level = height.div(5), reach = height.mul(0.15).add(0.3).mul(flare.mul(0.12).add(1));
  const u = r.div(reach), spoke = angle.mul(28).div(2 * Math.PI).add(seed), lane = floor(spoke), across = fract(spoke).sub(0.5);
  const jitter = fract(sin(lane.mul(12.9898).add(seed.mul(78.233))).mul(43758.5453));
  const twinkle = sin(clock.mul(jitter.mul(5).add(4)).add(lane.mul(1.7))).mul(0.3).add(0.7);
  const trail = exp(abs(across).mul(r).mul(-95)).mul(smoothstep(0.08, 0.3, u)).mul(smoothstep(1.02, 0.55, u)).mul(u);
  const star = exp(length(vec2(across.mul(r).mul(2 * Math.PI / 28), r.sub(reach.mul(jitter.mul(0.14).add(0.84))))).mul(-60)).mul(twinkle);
  const inner = exp(abs(across).mul(r).mul(-120)).mul(exp(abs(u.sub(0.5)).mul(-18))).mul(step(0.5, fract(lane.mul(0.5))));
  const crown = exp(abs(u.sub(1.12)).mul(-40)).mul(pow(abs(cos(angle.mul(20).add(clock.mul(0.6)))), float(30))).mul(step(4.5, height));
  const core = exp(r.mul(-7)).mul(0.28);
  const power = trail.mul(0.8).add(star.mul(1.5)).add(inner.mul(0.6)).add(core).mul(level.mul(0.7).add(0.3)).mul(flare.mul(2.2).add(1)).mul(step(0.5, height));
  const ink = inkOf(colour);
  material.colorNode = mix(ink, vec3(1), core.add(star.mul(0.35))).mul(2.2).add(rgb(GOLD).mul(crown.mul(1.6)));
  material.opacityNode = clamp(power.add(crown.mul(0.8)), 0, 1).mul(smoothstep(1.3, 1.0, r)).mul(alpha);
  return material;
}

export function ledgerMaterial(light: Light2D, atlas: Atlas, cell: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(cell, 'vec4') as N<'vec4'>, colour = v.x, rank = v.y, discarded = v.z, flags = v.w;
  const gone = bit(flags, 0), played = bit(flags, 1), critical = bit(flags, 2), fresh = bit(flags, 3), copies = floor(flags.div(16));
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CELL_W + CELL_Q, CELL_H + CELL_Q)), d = sdBox(t, vec2(CELL_W / 2, CELL_H / 2), 0.06);
  const ink = inkOf(colour);
  let color: N<'vec3'> = mix(rgb(0x0b0d20), ink.mul(0.2), played);
  color = mix(color, ink.mul(0.6), coverage(abs(d.add(0.01)).sub(0.007)).mul(0.8));
  const numeral = glyph(atlas, t, vec2(0, 0.07), 0.2, rank.add(1));
  color = mix(color, mix(ink, vec3(1), played.mul(0.5)), numeral.mul(played.mul(0.45).add(0.4)));
  for (let k = 0; k < 3; k++) {
    const offset = float(k).sub(copies.sub(1).mul(0.5)).mul(0.11);
    const tick = coverage(sdBox(t.sub(vec2(offset, -0.155)), vec2(0.04, 0.045), 0.015)).mul(step(k + 0.5, copies));
    const filled = step(k + 0.5, discarded);
    color = mix(color, mix(rgb(0x2a2e4c), ink.mul(1.1), filled), tick);
  }
  const cross = min(sdSegment(t, vec2(-0.16, -0.19), vec2(0.16, 0.19), 0.016), sdSegment(t, vec2(-0.16, 0.19), vec2(0.16, -0.19), 0.016));
  color = mix(color, rgb(MISFIRE), coverage(cross).mul(gone).mul(0.9));
  const warn = coverage(sdCircle(t.sub(vec2(0.15, 0.18)), 0.035)).mul(critical);
  color = mix(color, rgb(0xffa43a), warn);
  const flash = exp(max(d, 0).mul(-20)).mul(fresh).mul(step(0, d)).mul(smoothstep(CELL_Q / 2, CELL_Q / 4, d));
  material.colorNode = mix(light.lit(color).add(color.mul(numeral.mul(0.3))), ink.mul(1.6), flash.mul(step(0, d)));
  material.opacityNode = max(coverage(d), flash.mul(0.8));
  return material;
}

export function tokenMaterial(light: Light2D, token: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(token, 'vec4') as N<'vec4'>, kind = v.x, lit = v.y, flash = v.z, alpha = v.w;
  const material = surface();
  const t = uv().sub(0.5).mul(TOKEN_Q), r = length(t);
  const fuse = step(0.5, kind).mul(step(kind, 1.5)), pip = step(1.5, kind);
  const radius = mix(float(0.26), float(0.12), pip), d = r.sub(radius);
  const dome = t.y.mul(0.9).sub(t.x.mul(0.4)).div(radius).mul(0.18).add(1);
  const blue = mix(rgb(0x2556c9), rgb(0x78b6ff), smoothstep(-0.26, 0.26, t.y)).mul(dome);
  const glint = exp(abs(t.x).mul(-40)).mul(exp(abs(t.y).mul(-9))).add(exp(abs(t.y).mul(-40)).mul(exp(abs(t.x).mul(-9)))).mul(smoothstep(0.17, 0.08, r));
  const spentBlue = rgb(0x151a38).add(vec3(glint.mul(0.08)));
  const rim = coverage(abs(d.add(0.03)).sub(0.012));
  const clueColor = mix(mix(spentBlue, blue.add(vec3(glint.mul(0.9))), lit), mix(rgb(0x39406a), vec3(0.95), lit), rim);
  const flameD = min(sdCircle(t.sub(vec2(0, -0.04)), 0.085), sdSegment(t, vec2(0, -0.02), vec2(0, 0.13), 0.035));
  const flame = coverage(flameD), core = coverage(sdCircle(t.sub(vec2(0, -0.045)), 0.04));
  const ember = mix(rgb(0xb3281a), rgb(0xff6a3d), smoothstep(-0.26, 0.26, t.y)).mul(dome);
  const crack = coverage(min(min(sdSegment(t, vec2(-0.16, 0.12), vec2(-0.03, 0.02), 0.012), sdSegment(t, vec2(-0.03, 0.02), vec2(0.04, -0.06), 0.012)), sdSegment(t, vec2(0.04, -0.06), vec2(0.15, -0.14), 0.012)));
  const lostFuse = mix(rgb(0x1d1716), rgb(0x8a2a1c), crack.mul(0.8));
  const liveFuse = mix(mix(ember, rgb(0xffd08a), flame), rgb(0xfff4d6), core);
  const fuseColor = mix(mix(lostFuse, liveFuse, lit), mix(rgb(0x3a2522), rgb(0xffc3a0), lit), rim);
  const pipColor = mix(rgb(0x3a2a14), rgb(PAPER).mul(1.4), lit);
  const color = mix(mix(clueColor, fuseColor, fuse), pipColor, pip);
  const tone = mix(rgb(0x9fc4ff), rgb(MISFIRE), fuse);
  const glow = exp(max(d, 0).mul(-14)).mul(flash).mul(step(0, d)).mul(smoothstep(TOKEN_Q / 2, TOKEN_Q / 2 - 0.08, r));
  material.colorNode = mix(light.lit(color).add(color.mul(lit.mul(0.25))), tone.mul(1.8), glow);
  material.opacityNode = max(coverage(d), glow).mul(alpha);
  return material;
}

export function lanternMaterial(atlas: Atlas, lantern: THREE.InstancedBufferAttribute, extra: THREE.InstancedBufferAttribute, accent: number, clock: N<'float'>): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(lantern, 'vec4') as N<'vec4'>, w = instancedDynamicBufferAttribute(extra, 'vec4') as N<'vec4'>;
  const agent = v.x, active = v.y, progress = v.z, ticks = v.w, numeral = w.x, alpha = w.y;
  const material = surface();
  const t = uv().sub(0.5).mul(LANTERN_Q), r = length(t), angle = atan(t.y, t.x);
  const d = r.sub(LANTERN_R);
  const paper = mix(rgb(PAPER), rgb(accent), agent);
  const glowing = active.mul(0.7).add(0.3);
  const ribs = pow(abs(cos(angle.mul(8))), float(90)).mul(smoothstep(0.1, 0.17, r)).mul(0.35);
  const bands = exp(abs(r.sub(0.19)).mul(-95)).mul(0.25);
  const cap = coverage(abs(r.sub(0.145)).sub(0.011));
  const warm = mix(paper.mul(0.45), mix(paper, vec3(1, 0.97, 0.9), smoothstep(LANTERN_R, 0.02, r).mul(0.7)), glowing);
  let color = warm.mul(ribs.add(bands).oneMinus());
  color = mix(color, rgb(0x5a2e1a), cap.mul(0.6));
  const seat = glyph(atlas, t, vec2(0, 0), 0.17, numeral).mul(step(0.5, agent.oneMinus()));
  const flower = coverage(r.sub(cos(angle.mul(5)).mul(0.027).add(0.068))).mul(agent);
  color = mix(color, rgb(0x3a1c10), max(seat, flower).mul(0.85));
  const turn = fract(float(0.25).sub(angle.div(2 * Math.PI))), slice = floor(turn.mul(ticks));
  const gap = abs(fract(turn.mul(ticks)).sub(0.5)).sub(0.5).negate().mul(r.mul(2 * Math.PI).div(max(ticks, 1))).sub(0.022);
  const ring = coverage(abs(r.sub(RING_R)).sub(0.026)).mul(coverage(gap.negate())).mul(step(0.5, ticks));
  const done = step(slice, progress.sub(0.5));
  const ringColor = mix(rgb(0x3b3350), rgb(PAPER).mul(1.6), done);
  const halo = exp(max(d, 0).mul(-12)).mul(active).mul(step(0, d)).mul(smoothstep(LANTERN_Q / 2, RING_R, r)).mul(sin(clock.mul(3)).mul(0.08).add(0.5));
  let layer: Layer = { color: paper.mul(1.5), alpha: halo };
  layer = over(layer, ringColor, ring.mul(done.mul(0.5).add(0.5)));
  layer = over(layer, color.mul(glowing.mul(0.5).add(0.75)), coverage(d));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export interface CursorLook { alpha: N<'float'>; flash: N<'float'>; tint: THREE.Color; clock: N<'float'> }

export function cursorMaterial(accent: number, look: CursorLook): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CARD_QW, CARD_QH)), d = sdBox(t, vec2(CARD_W / 2, CARD_H / 2), CORNER);
  const breathe = sin(look.clock.mul(4.2)).mul(0.12).add(0.88);
  const ring = coverage(abs(d.sub(0.1)).sub(0.026));
  const halo = exp(abs(d.sub(0.1)).mul(-18)).mul(0.5).mul(step(0, d)).mul(smoothstep(0.22, 0.14, d));
  const flashTone = uniform(look.tint) as unknown as N<'vec3'>;
  const tone = mix(rgb(accent).mul(1.6), flashTone.mul(1.3), look.flash);
  material.colorNode = tone;
  material.opacityNode = clamp(ring.add(halo.mul(breathe)), 0, 1).mul(look.alpha);
  return material;
}

export interface BadgeLook { alpha: N<'float'>; rank: N<'float'>; colour: N<'float'> }

export function badgeMaterial(atlas: Atlas, look: BadgeLook): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = uv().sub(0.5).mul(BADGE_Q), r = length(t), d = r.sub(0.24);
  const byRank = step(-0.5, look.rank);
  const ink = inkOf(look.colour);
  const numeral = glyph(atlas, t, vec2(0, 0), 0.24, look.rank.add(1)).mul(byRank);
  const fill = mix(ink, rgb(0xfdf6e8), byRank);
  const rim = coverage(abs(d.add(0.025)).sub(0.014));
  let color = mix(fill, vec3(1), rim.mul(0.8));
  color = mix(color, rgb(0x14162e), numeral);
  const halo = exp(max(d, 0).mul(-16)).mul(step(0, d)).mul(smoothstep(0.35, 0.27, r)).mul(0.55);
  material.colorNode = mix(fill.mul(1.4), color, coverage(d));
  material.opacityNode = max(coverage(d), halo).mul(look.alpha);
  return material;
}

export interface PlateLook { value: N<'float'>; alpha: N<'float'> }

export function plateMaterial(light: Light2D, atlas: Atlas, look: PlateLook, accent: number): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(PLATE_W, PLATE_H)), d = sdBox(t, vec2(PLATE_W / 2 - 0.03, PLATE_H / 2 - 0.03), 0.1);
  const tens = floor(look.value.div(10)), ones = mod(look.value, 10), wide = step(9.5, look.value);
  const tensGlyph = glyph(atlas, t, vec2(-0.09, 0), 0.2, tens).mul(wide);
  const onesGlyph = glyph(atlas, t, vec2(wide.mul(0.09), 0), 0.2, ones);
  let color = mix(rgb(0x120f26), rgb(0x241c46), t.y.div(PLATE_H).add(0.5));
  color = mix(color, rgb(accent).mul(0.8), coverage(abs(d.add(0.015)).sub(0.01)));
  color = mix(color, rgb(0xfdf0f4), max(tensGlyph, onesGlyph));
  material.colorNode = light.lit(color).add(color.mul(max(tensGlyph, onesGlyph).mul(0.4)));
  material.opacityNode = coverage(d).mul(look.alpha);
  return material;
}
