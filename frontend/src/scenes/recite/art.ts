import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, floor, fwidth, length, max, min, mix, positionWorld, select, sign, smoothstep, step, texture, uniform, vec2 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { fractal, noise, rgb, sdBox } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;

export const PITCH = 1.3;
export const HEARD = 0;
export const SAID = 1.7;
export const RECORD_Z = 3.05;
export const GAP = 0.34;
export const ROWS = 6;
export const TEXELS = 3 + ROWS;
export const SYM = [0xff7a59, 0x3fd0a0, 0xf2c14e, 0x9b8cff] as const;
export const NAMES = ['ring', 'peak', 'kite', 'cross'] as const;
export const GOLD = TOKENS.celebrationGold;
export const MISS = TOKENS.failureBlue;
export const IVORY = 0xf1ead8;

const SLATE = 0x14292e;
const BENCH = 0x1b373c;
const WELL_INK = 0x0a171a;
const RIM = 0x9cbdb4;
const WELL = 0.46;
const GLYPH = 0.25;

export const columnX = (length: number, j: number): number => (j + 0.5 - length / 2) * PITCH;

const pick = <T extends string>(i: N<'float'>, a: N<T>, b: N<T>, c: N<T>, d: N<T>): N<T> =>
  mix(mix(a as never, b as never, step(0.5, i)) as never, mix(c as never, d as never, step(2.5, i)) as never, step(1.5, i)) as unknown as N<T>;

const triangle = (p: N<'vec2'>, r: number): N<'float'> => {
  const k = Math.sqrt(3);
  const px = abs(p.x).sub(r), py = p.y.add(r / k);
  const fold = px.add(py.mul(k)).greaterThan(0);
  const qx = select(fold, px.sub(py.mul(k)).mul(0.5), px);
  const qy = select(fold, px.mul(-k).sub(py).mul(0.5), py);
  const cx = qx.sub(clamp(qx, -2 * r, 0));
  return length(vec2(cx, qy)).negate().mul(sign(qy));
};

const glyph = (p: N<'vec2'>, symbol: N<'float'>): N<'float'> => {
  const s = GLYPH;
  const ring = abs(length(p).sub(s * 0.74)).sub(s * 0.24);
  const peak = triangle(p.add(vec2(0, s * 0.08)), s * 1.02);
  const kite = sdBox(vec2(p.x.add(p.y), p.y.sub(p.x)).mul(Math.SQRT1_2), vec2(s * 0.7, s * 0.7), 0.03);
  const cross = min(sdBox(p, vec2(s, s * 0.3), 0.03), sdBox(p, vec2(s * 0.3, s), 0.03));
  return pick<'float'>(symbol, ring, peak, kite, cross);
};

const tint = (symbol: N<'float'>): N<'vec3'> => pick<'vec3'>(symbol, rgb(SYM[0]), rgb(SYM[1]), rgb(SYM[2]), rgb(SYM[3]));

export type Board = ReturnType<typeof boardMaterial>;

export function boardMaterial(length: number, lit: Light2D['lit']) {
  const values = new Float32Array(length * TEXELS * 4), halves = new Uint16Array(values.length);
  const data = new THREE.DataTexture(halves, length, TEXELS, THREE.RGBAFormat, THREE.HalfFloatType);
  data.minFilter = data.magFilter = THREE.NearestFilter;
  data.generateMipmaps = false;
  data.needsUpdate = true;
  const cursor = uniform(new THREE.Vector2()), fade = uniform(1), enter = uniform(1);

  const q = vec2(positionWorld.x, positionWorld.z);
  const px = max(fwidth(q.x), 1e-5);
  const aa = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
  const col = clamp(floor(q.x.div(PITCH).add(length / 2)), 0, length - 1);
  const xc = col.add(0.5 - length / 2).mul(PITCH);
  const u = col.add(0.5).div(length);
  const texel = (row: N<'float'>) => texture(data, vec2(u, row.add(0.5).div(TEXELS)));
  const local = (z: number) => vec2(q.x.sub(xc), q.y.sub(z).negate());

  const grain = fractal(q.mul(0.55), 3).mul(0.07).add(noise(q.mul(5.1)).mul(0.03)).add(1);
  const half = vec2(length * PITCH / 2 + 0.5, 0.74);
  const bench = min(sdBox(q.sub(vec2(0, HEARD)), half, 0.6), sdBox(q.sub(vec2(0, SAID)), half, 0.6));
  const centre = vec2(0, (HEARD + RECORD_Z + ROWS * GAP) / 2);
  let matte: N<'vec3'> = rgb(SLATE).mul(grain).mul(float(1).sub(smoothstep(3, 9 + length * 0.55, q.sub(centre).length()).mul(0.4)));
  const lane = select(q.y.lessThan((HEARD + SAID) / 2), float(HEARD), float(SAID));
  const active = float(1).sub(clamp(abs(lane.sub(cursor.y)).div(SAID - HEARD), 0, 1));
  matte = mix(matte, rgb(BENCH).mul(grain).mul(active.mul(0.28).add(0.9)), aa(bench));
  matte = mix(matte, rgb(RIM).mul(active.mul(0.3).add(0.26)), aa(abs(bench).sub(0.012)));

  const hp = local(HEARD), sp = local(SAID), hr = hp.length(), sr = sp.length();
  const hw = hr.sub(WELL), sw = sr.sub(WELL);
  for (const w of [hw, sw]) {
    matte = mix(matte, rgb(WELL_INK).mul(float(0.8).add(w.mul(0.6))), aa(w));
    matte = mix(matte, rgb(RIM).mul(0.55), aa(abs(w).sub(0.02)));
  }
  let color: N<'vec3'> = lit(matte);

  const heard = texel(float(0)), said = texel(float(1));
  const hs = heard.x, shown = heard.y.mul(fade), now = heard.z, reveal = heard.w;
  const glow = max(max(heard.y.mul(0.36), now), reveal.mul(0.74));
  color = color.add(tint(hs).mul(exp(hr.mul(-4.2)).mul(now).mul(0.34).mul(aa(hw)).mul(fade)));
  color = mix(color, tint(hs).mul(glow), aa(glyph(hp, hs)).mul(shown));

  const call = said.x, stamp = said.y, hit = max(said.z, 0).mul(fade), miss = max(said.z.negate(), 0).mul(fade);
  const inked = aa(sr.sub(stamp.mul(0.38))).mul(step(0.001, stamp)).mul(fade);
  color = mix(color, tint(call).mul(mix(float(0.66), float(0.36), miss)), inked);
  color = mix(color, rgb(IVORY).mul(mix(float(0.86), float(0.5), miss)), aa(glyph(sp.div(max(stamp, 0.01)), call).mul(stamp)).mul(inked));
  color = mix(color, rgb(GOLD).mul(0.8), aa(abs(sr.sub(0.43)).sub(0.03)).mul(hit));
  const slash = sdBox(vec2(sp.x.add(sp.y), sp.y.sub(sp.x)).mul(Math.SQRT1_2), vec2(0.035, 0.36), 0.02);
  color = mix(color, rgb(MISS).mul(0.78), aa(slash).mul(miss));
  const thread = sdBox(q.sub(vec2(xc, (HEARD + SAID) / 2)), vec2(0.028, (SAID - HEARD) / 2 - WELL - 0.03), 0.02);
  color = mix(color, rgb(GOLD).mul(0.72), aa(thread).mul(hit));
  color = mix(color, rgb(GOLD).mul(0.78), aa(abs(hw).sub(0.03)).mul(hit));

  const cd = q.sub(cursor).length().sub(0.6);
  color = color.add(rgb(IVORY).mul(exp(max(cd, 0).mul(-7)).mul(0.07)));
  color = mix(color, rgb(IVORY).mul(0.72), aa(abs(cd).sub(0.026)));

  const along = q.y.sub(RECORD_Z).div(GAP).add(float(1).sub(enter));
  const row = clamp(floor(along.add(0.5)), 0, ROWS);
  const record = texel(row.add(2));
  const presence = mix(float(1), enter, step(row, 0.5)).mul(mix(float(1), float(1).sub(enter), step(ROWS - 0.5, row)));
  const pip = vec2(q.x.sub(xc), along.sub(row).mul(GAP)).length().sub(0.1);
  color = mix(color, rgb(RIM).mul(0.3), aa(abs(pip).sub(0.012)).mul(float(1).sub(record.x)).mul(presence).mul(step(along, ROWS - 0.5)));
  color = mix(color, mix(rgb(MISS).mul(0.5), rgb(GOLD).mul(0.8), record.y), aa(pip).mul(record.x).mul(presence));

  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  material.colorNode = color;
  const commit = () => {
    for (let i = 0; i < values.length; i++) halves[i] = THREE.DataUtils.toHalfFloat(values[i]!);
    data.needsUpdate = true;
  };
  return { material, data, values, commit, cursor, fade, enter };
}
