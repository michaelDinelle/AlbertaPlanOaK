import * as THREE from 'three/webgpu';
import { Fn, abs, clamp, dot, float, floor, fwidth, max, min, mix, mod, normalize, positionWorld, select, sin, texture, uniform, uvec2, vec2, vec3, vec4 } from 'three/tsl';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { MATCHES, MATCH_STEPS, MAX_ENERGY, UNITS } from '@sim/envs/relicfog';
import { hull, triangle } from './fleet';
import { numeral, type Glyphs } from './glyphs';
import { coverOf, flat, type Level, type N } from './sector';

export const DECK_WIDTH = 6.6;
export const DECK_HEIGHT = 24;
const ROW = 0.9;
const ROSTER = 5.05;
const KEYPAD = ROSTER + UNITS * ROW + 0.55;
const CARD = 0x121528;
const CARD_EDGE = 0x262b4d;
const MUTED = 0x8a90b8;
const INK = 0x06070f;
const GOLD = 0xffd166;

export interface Deck {
  mesh: THREE.Mesh;
  rows: THREE.DataTexture;
  ours: Level;
  theirs: Level;
  ourWins: Level;
  theirWins: Level;
  clock: Level;
  cursor: Level;
  mode: Level;
  key: Level;
  verdict: Level;
  time: Level;
  dispose(): void;
}

const hexagon = (p: N<'vec2'>, r: number): N<'float'> => {
  const q = abs(p), k = vec2(-0.866025404, 0.5);
  const folded = q.sub(k.mul(min(dot(k, q), 0).mul(2)));
  const limit = r * 0.577350269, rim = folded.sub(vec2(clamp(folded.x, -limit, limit), r));
  return rim.length().mul(select(rim.y.greaterThan(0), float(1), float(-1)));
};

const arrowShape = (p: N<'vec2'>, dir: readonly [number, number], size: number): N<'float'> => {
  const [dx, dy] = dir, along = dot(p, vec2(dx, dy)), across = dot(p, vec2(dy, -dx)), a = vec2(across, along).div(size);
  return min(sdSegment(a, vec2(0, -0.34), vec2(0, 0.05), 0.09), triangle(a, [0, 0.42], [-0.3, 0.02], [0.3, 0.02]).sub(0.02)).mul(size);
};

export function deckMaterial(set: Glyphs, accent: number, foe: number, origin: THREE.Vector2): Deck {
  const rows = new THREE.DataTexture(new Float32Array(UNITS * 4), UNITS, 1, THREE.RGBAFormat, THREE.FloatType);
  rows.minFilter = rows.magFilter = THREE.NearestFilter;
  rows.needsUpdate = true;
  const table = texture(rows);
  const ours = uniform(0), theirs = uniform(0), ourWins = uniform(0), theirWins = uniform(0), clock = uniform(0);
  const cursor = uniform(-1), mode = uniform(0), key = uniform(-1), verdict = uniform(0), time = uniform(0);
  const hue = rgb(accent), rival = rgb(foe);

  const p = vec2(positionWorld.x.sub(origin.x), positionWorld.z.sub(origin.y));
  const cover = coverOf(fwidth(p.x));
  const flip = (q: N<'vec2'>) => vec2(q.x, q.y.negate());
  const shade = Fn(() => {
    const color = vec3(0).toVar(), alpha = float(0).toVar();
    const paint = (shape: N<'float'>, tint: N<'vec3'>, strength: N<'float'> | number = 1) => {
      const c = cover(shape).mul(strength);
      color.assign(mix(color, tint, c));
      alpha.assign(max(alpha, c));
    };
    const outline = (shape: N<'float'>, width: number, strength: N<'float'> | number = 1) => paint(shape.sub(width), rgb(INK), typeof strength === 'number' ? strength * 0.85 : strength.mul(0.85));
    const card = (top: number, bottom: number) => sdBox(p.sub(vec2(DECK_WIDTH / 2, (top + bottom) / 2)), vec2(DECK_WIDTH / 2, (bottom - top) / 2), 0.22);
    const when = (condition: N<'bool'>) => select(condition, float(1), float(0));

    for (const [top, bottom] of [[0, 4.55], [ROSTER - 0.25, KEYPAD - 0.25], [KEYPAD, DECK_HEIGHT]] as const) {
      outline(card(top, bottom), 0.05);
      paint(card(top, bottom), rgb(CARD), 0.94);
      paint(abs(card(top, bottom)).sub(0.02), rgb(CARD_EDGE));
    }

    for (let k = 0; k < MATCHES; k++) {
      const c = p.sub(vec2(0.95 + k * 1.175, 0.95)), shape = hexagon(c, 0.44);
      const won = float(k).lessThan(ourWins), lost = float(MATCHES - 1 - k).lessThan(theirWins), current = abs(ourWins.sub(k)).lessThan(0.5).and(lost.not());
      const pulse = sin(time.mul(4)).mul(0.5).add(0.5);
      outline(shape, 0.05);
      paint(shape, select(won, hue.mul(1.15), select(lost, rival.mul(1.05), rgb(0x1b1f3a))));
      paint(abs(shape).sub(0.035), select(current, hue.mul(pulse.mul(0.6).add(0.9)), rgb(CARD_EDGE).mul(1.6)), select(current, float(1), float(0.6)));
      paint(sdCircle(c, 0.1), rgb(0xfff4dc).mul(0.85), when(won.or(lost)).mul(0.9));
    }
    paint(abs(sdBox(p.sub(vec2(DECK_WIDTH / 2, 0.95)), vec2(DECK_WIDTH / 2 - 0.12, 0.64), 0.2)).sub(0.03), select(verdict.greaterThan(0), rgb(GOLD), rival), abs(verdict));

    const digits = (value: N<'float'>, at: readonly [number, number], em: number, anchor: -1 | 0 | 1, tint: N<'vec3'>) => {
      const shape = numeral(set, flip(p.sub(vec2(at[0], at[1]))), value, em, anchor);
      outline(shape, 0.06);
      paint(shape, tint);
    };
    digits(ours, [0.35, 2.55], 1.05, -1, hue.mul(1.25).add(0.05));
    digits(theirs, [DECK_WIDTH - 0.35, 2.55], 1.05, 1, rival.mul(1.1).add(0.05));
    const total = max(ours.add(theirs), 1), share = select(ours.add(theirs).greaterThan(0.5), ours.div(total), float(0.5));
    const bar = sdBox(p.sub(vec2(DECK_WIDTH / 2, 3.45)), vec2(DECK_WIDTH / 2 - 0.35, 0.09), 0.09);
    paint(bar, mix(rival.mul(0.85), hue, when(p.x.sub(0.35).div(DECK_WIDTH - 0.7).lessThan(share))));
    const clockBar = sdBox(p.sub(vec2(DECK_WIDTH / 2, 4.0)), vec2(DECK_WIDTH / 2 - 0.35, 0.05), 0.05);
    const along = clamp(p.x.sub(0.35).div(DECK_WIDTH - 0.7), 0, 1);
    paint(clockBar, select(along.lessThan(clock.div(MATCH_STEPS)), rgb(0xd8dcff), rgb(0x2a2f55)));
    paint(sdSegment(p, vec2(0.35 + (DECK_WIDTH - 0.7) * 0.25, 3.88), vec2(0.35 + (DECK_WIDTH - 0.7) * 0.25, 4.12), 0.012).min(sdSegment(p, vec2(DECK_WIDTH / 2, 3.88), vec2(DECK_WIDTH / 2, 4.12), 0.012)).min(sdSegment(p, vec2(0.35 + (DECK_WIDTH - 0.7) * 0.75, 3.88), vec2(0.35 + (DECK_WIDTH - 0.7) * 0.75, 4.12), 0.012)), rgb(MUTED), 0.6);

    const slot = clamp(floor(p.y.sub(ROSTER).div(ROW)), 0, UNITS - 1), listed = when(p.y.greaterThanEqual(ROSTER).and(p.y.lessThan(ROSTER + UNITS * ROW)));
    const row = table.load(uvec2(vec2(slot, 0))), flags = row.x, energy = row.y, kind = row.z, dir = vec2(row.w.sub(floor(row.w.div(16)).mul(16)).sub(8), floor(row.w.div(16)).sub(8));
    const alive = mod(flags, 2).greaterThan(0.5), ordered = mod(floor(flags.div(2)), 2).greaterThan(0.5), stalled = mod(floor(flags.div(4)), 2).greaterThan(0.5);
    const local = p.sub(vec2(0, slot.mul(ROW).add(ROSTER + ROW / 2)));
    const here = abs(cursor.sub(slot)).lessThan(0.5), fade = select(alive, float(1), float(0.3)).mul(listed);
    const strip = sdBox(local.sub(vec2(DECK_WIDTH / 2, 0)), vec2(DECK_WIDTH / 2 - 0.12, ROW / 2 - 0.05), 0.12);
    paint(strip, mix(rgb(0x191d38), hue.mul(0.32), when(here)), select(here, float(0.95), float(0.5)).mul(listed));
    paint(abs(strip).sub(0.03), hue.mul(1.4), when(here).mul(sin(time.mul(6)).mul(0.25).add(0.75)).mul(listed));
    const icon = hull(flip(local.sub(vec2(0.55, 0))).div(1.3)).mul(1.3);
    paint(icon.sub(0.03), rgb(INK), fade);
    paint(select(alive, icon, abs(icon).sub(0.015)), select(stalled, rgb(0x5d6384), hue.mul(1.1)), fade);
    const gauge = sdBox(local.sub(vec2(2.3, 0)), vec2(1.3, 0.13), 0.13);
    const level = clamp(energy.div(MAX_ENERGY), 0, 1), inGauge = local.x.sub(1.0).div(2.6);
    paint(gauge, rgb(0x0c0e1d), fade);
    paint(gauge, mix(hue.mul(0.75), rgb(0x3fe0cf), 0.35), when(inGauge.lessThan(level).and(alive)).mul(listed));
    paint(abs(gauge).sub(0.012), rgb(CARD_EDGE).mul(1.8), fade);
    const value = numeral(set, flip(local.sub(vec2(5.05, 0))), energy, 0.52, 1), living = when(alive).mul(listed);
    outline(value, 0.05, living);
    paint(value, select(energy.lessThan(0), rgb(0xff5a5a), vec3(0.8, 0.79, 0.86)), living);
    const glyphAt = local.sub(vec2(DECK_WIDTH - 0.62, 0));
    const toward = normalize(select(dir.length().greaterThan(0.01), vec2(dir.x, dir.y.negate()), vec2(0, 1)));
    const along2 = dot(flip(glyphAt), toward), across2 = dot(flip(glyphAt), vec2(toward.y, toward.x.negate()));
    const arrow = min(sdSegment(vec2(across2, along2), vec2(0, -0.16), vec2(0, 0.04), 0.045), triangle(vec2(across2, along2), [0, 0.22], [-0.15, 0.02], [0.15, 0.02]).sub(0.01));
    const hold = abs(glyphAt.length().sub(0.11)).sub(0.035);
    const cross = min(abs(glyphAt.length().sub(0.14)).sub(0.028), min(sdBox(glyphAt, vec2(0.24, 0.022)), sdBox(glyphAt, vec2(0.022, 0.24))));
    const waiting = sdCircle(glyphAt, 0.055);
    const mark = select(ordered, select(kind.lessThan(0.5), hold, select(kind.lessThan(4.5), arrow, cross)), waiting);
    paint(mark.sub(0.04), rgb(INK), fade.mul(0.9));
    paint(mark, select(ordered, hue.mul(1.5).add(0.1), rgb(MUTED)), fade.mul(select(ordered, float(1), float(0.6))));

    const middle = KEYPAD + (DECK_HEIGHT - KEYPAD) / 2;
    const keys: readonly (readonly [number, number])[] = [[1.95, middle], [1.95, middle - 1.02], [2.97, middle], [1.95, middle + 1.02], [0.93, middle], [4.95, middle]];
    const arrows: readonly (readonly [number, number])[] = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    keys.forEach(([x, y], k) => {
      const local = p.sub(vec2(x, y)), wide = k === 5 ? 0.95 : 0.44, shape = sdBox(local, vec2(wide, 0.44), 0.14);
      const hot = when(abs(key.sub(k)).lessThan(0.5)), armed = when(mode.greaterThan(1.5)).mul(k === 5 ? 1 : 0);
      const lit = max(hot, armed.mul(0.55));
      outline(shape, 0.04);
      paint(shape, mix(rgb(0x20254a), hue.mul(0.85), lit));
      paint(sdBox(local.sub(vec2(0, -0.34)), vec2(wide - 0.1, 0.05), 0.04), rgb(0x0b0d1a), lit.oneMinus().mul(0.5));
      const ink = mix(rgb(0xb6bde0), vec3(1.2, 1.15, 1.25), lit);
      if (k === 0) paint(abs(sdCircle(local, 0.15)).sub(0.04), ink);
      else if (k < 5) paint(arrowShape(flip(local), arrows[k - 1]!, 0.5), ink);
      else paint(min(abs(sdCircle(local, 0.2)).sub(0.035), min(sdBox(local, vec2(0.34, 0.03)), sdBox(local, vec2(0.03, 0.34)))), ink);
    });
    return vec4(color, clamp(alpha, 0, 1));
  })();

  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = shade.xyz;
  material.opacityNode = shade.w;
  const mesh = new THREE.Mesh(flat(DECK_WIDTH + 0.6, DECK_HEIGHT + 0.6), material);
  mesh.position.set(origin.x + DECK_WIDTH / 2, 0, origin.y + DECK_HEIGHT / 2);
  mesh.renderOrder = 12;
  mesh.frustumCulled = false;
  return {
    mesh, rows, ours, theirs, ourWins, theirWins, clock, cursor, mode, key, verdict, time,
    dispose() {
      rows.dispose();
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}
