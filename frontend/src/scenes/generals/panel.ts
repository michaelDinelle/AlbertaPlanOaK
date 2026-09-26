import * as THREE from 'three/webgpu';
import { Fn, abs, atan, clamp, dot, exp, float, floor, fwidth, max, min, mix, mod, normalize, positionWorld, select, texture, uniform, uniformArray, uvec2, vec2, vec3, vec4 } from 'three/tsl';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { NOBODY } from '@sim/envs/generals';
import { coverOf, crown, flat } from './board';
import { numeral, type Numerals } from './glyphs';

type N<T extends string> = THREE.Node<T>;
type Level = THREE.UniformNode<'float', number>;

export const PANEL_WIDTH = 8;
const DIAL = 2.7;
const HEAD = 0.7;
const ROW = 1.24;
const CARD = 1.06;
const PAD = 0.45;
const KEY = 0.92;
const KEYS_HEIGHT = 2.2;
const INK = 0x0c0f15;
const CARD_TINT = 0x1a2130;
const CAP = 0xf4c95e;
const MUTED = 0x9aa6bd;
const SCAN = 0x5ec1ff;

export const panelHeight = (players: number) => DIAL + HEAD + players * ROW + PAD + KEYS_HEIGHT;
export const rowTop = (rank: number) => DIAL + HEAD + rank * ROW;

export interface Panel {
  mesh: THREE.Mesh;
  rows: THREE.DataTexture;
  palette: THREE.Color[];
  phase: Level;
  round: Level;
  turn: Level;
  key: Level;
  pressed: Level;
  mode: Level;
  sweep: Level;
  dispose(): void;
}

const edge = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>): N<'float'> => {
  const e = b.sub(a);
  return dot(p.sub(a), normalize(vec2(e.y, e.x.negate())));
};

const triangle = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, c: N<'vec2'>): N<'float'> => max(max(edge(p, a, b), edge(p, b, c)), edge(p, c, a));

const pointing = (p: N<'vec2'>, dir: readonly [number, number], reach: number, back: number, width: number): N<'float'> => {
  const [dx, dy] = dir, tip = vec2(dx * reach, dy * reach);
  const left = vec2(-dx * back - dy * width, -dy * back + dx * width), right = vec2(-dx * back + dy * width, -dy * back - dx * width);
  const d1 = triangle(p, tip, left, right), d2 = triangle(p, tip, right, left);
  return min(d1, d2);
};

export function panelMaterial(players: number, substeps: number, set: Numerals, accent: number, origin: THREE.Vector2, scale: number): Panel {
  const rows = new THREE.DataTexture(new Float32Array(players * 4), players, 1, THREE.RGBAFormat, THREE.FloatType);
  rows.minFilter = rows.magFilter = THREE.NearestFilter;
  rows.needsUpdate = true;
  const palette = Array.from({ length: NOBODY }, () => new THREE.Color());
  const inks = uniformArray(palette, 'color');
  const phase = uniform(0), round = uniform(0), turn = uniform(0), key = uniform(-1), pressed = uniform(9), mode = uniform(0), sweep = uniform(-1);
  const table = texture(rows), hue = rgb(accent);

  const p = vec2(positionWorld.x.sub(origin.x), positionWorld.z.sub(origin.y)).div(scale);
  const cover = coverOf(fwidth(p.x));
  const shade = Fn(() => {
    const color = vec3(0).toVar(), alpha = float(0).toVar();
    const paint = (shape: N<'float'>, tint: N<'vec3'>, strength: N<'float'> | number = 1) => {
      const c = cover(shape).mul(strength);
      color.assign(mix(color, tint, c));
      alpha.assign(max(alpha, c));
    };
    const outline = (shape: N<'float'>, width: number, strength: N<'float'> | number = 1) => paint(shape.sub(width), rgb(INK), typeof strength === 'number' ? strength * 0.85 : strength.mul(0.85));
    const flip = (q: N<'vec2'>) => vec2(q.x, q.y.negate());
    const when = (condition: N<'bool'>) => select(condition, float(1), float(0));

    const d = p.sub(vec2(DIAL / 2, DIAL / 2)), r = d.length();
    const turnOf = mod(atan(d.x, d.y.negate()).div(2 * Math.PI).add(1), 1), slice = turnOf.mul(substeps), index = floor(slice);
    const gap = abs(slice.sub(index).sub(0.5)).mul(2 * Math.PI / substeps).mul(r);
    const notch = max(abs(r.sub(1.02)).sub(0.14), gap.sub(2 * Math.PI / substeps * 1.02 * 0.36));
    const passed = index.lessThan(floor(phase).add(0.5)), current = abs(index.sub(floor(phase))).lessThan(0.5);
    paint(sdCircle(d, 1.3), rgb(CARD_TINT), 0.92);
    paint(notch, select(current, vec3(1.3, 1.25, 1.15), select(passed, hue.mul(1.1), rgb(0x3a4458))));
    paint(abs(r.sub(0.64)).sub(0.07), select(turnOf.lessThan(round), rgb(CAP), rgb(0x2b3344)));
    paint(sdCircle(d, 0.44), rgb(0x10141c));
    paint(crown(flip(d).div(0.62)).mul(0.62), rgb(CAP).mul(0.85));
    const turnDigits = numeral(set, flip(p.sub(vec2(PANEL_WIDTH - 0.15, DIAL / 2))), turn, 1.3, PANEL_WIDTH - DIAL - 0.4, 1);
    outline(turnDigits, 0.07);
    paint(turnDigits, vec3(1.05, 1.0, 0.92));

    const head = DIAL + HEAD * 0.5;
    paint(min(sdSegment(p, vec2(4.62, head + 0.22), vec2(5.02, head - 0.2), 0.05), sdSegment(p, vec2(4.66, head - 0.04), vec2(4.9, head + 0.2), 0.04)), rgb(MUTED));
    paint(abs(sdBox(p.sub(vec2(7.3, head)), vec2(0.19), 0.03)).sub(0.035), rgb(MUTED));

    for (let i = 0; i < players; i++) {
      const row = table.load(uvec2(i, 0)), army = row.x, land = row.y, flags = row.z, top = row.w;
      const owner = mod(flags, 8), alive = mod(floor(flags.div(8)), 2).greaterThan(0.5), mine = floor(flags.div(16)).greaterThan(0.5);
      const tint = inks.element(owner.toInt()) as unknown as N<'vec3'>;
      const local = p.sub(vec2(0, top.add(CARD / 2)));
      const card = sdBox(local.sub(vec2(PANEL_WIDTH / 2, 0)), vec2(PANEL_WIDTH / 2, CARD / 2), 0.14);
      const fade = select(alive, float(1), float(0.42));
      outline(card, 0.04);
      paint(card, mix(rgb(CARD_TINT), tint, 0.18).mul(fade));
      paint(sdBox(local.sub(vec2(0.62, 0)), vec2(0.5, CARD / 2 - 0.08), 0.1), tint.mul(fade));
      const emblem = crown(flip(local.sub(vec2(0.62, 0))).div(0.8)).mul(0.8);
      paint(emblem.sub(0.03), rgb(INK), when(alive).mul(0.8));
      paint(emblem, rgb(CAP), when(alive));
      paint(min(sdSegment(local, vec2(0.4, -0.22), vec2(0.84, 0.22), 0.06), sdSegment(local, vec2(0.4, 0.22), vec2(0.84, -0.22), 0.06)), rgb(0x12151c), when(alive.not()));
      paint(abs(card).sub(0.05), hue.mul(1.3), when(mine));
      paint(pointing(local.sub(vec2(-0.3, 0)), [1, 0], 0.2, 0.16, 0.2), hue.mul(1.3), when(mine));
      const seen = when(abs(owner.sub(sweep)).lessThan(0.5));
      outline(sdBox(local.sub(vec2(PANEL_WIDTH + 0.34, 0)), vec2(0.13), 0.13), 0.04, seen);
      paint(sdBox(local.sub(vec2(PANEL_WIDTH + 0.34, 0)), vec2(0.13), 0.13), rgb(SCAN), seen);
      const digits = min(numeral(set, flip(local.sub(vec2(5.35, 0))), army, 0.72, 3.8, 1), numeral(set, flip(local.sub(vec2(7.65, 0))), land, 0.72, 1.9, 1));
      outline(digits, 0.05, fade);
      paint(digits, vec3(1.0, 0.98, 0.94).mul(fade));
    }

    const middle = rowTop(players) + PAD + KEYS_HEIGHT / 2, flash = exp(pressed.mul(-2.2));
    const keys: readonly (readonly [number, number])[] = [[1.95, middle - 0.5], [1.95, middle + 0.52], [0.93, middle + 0.52], [2.97, middle + 0.52], [4.9, middle], [6.6, middle]];
    const arrows: readonly (readonly [number, number])[] = [[0, -1], [0, 1], [-1, 0], [1, 0]];
    keys.forEach(([x, y], k) => {
      const local = p.sub(vec2(x, y)), wide = k >= 4 ? 0.72 : KEY / 2, shape = sdBox(local, vec2(wide, KEY / 2), 0.14);
      const hot = when(abs(key.sub(k)).lessThan(0.5)).mul(flash);
      outline(shape, 0.04);
      paint(shape, mix(rgb(0x252d3d), hue.mul(0.9), hot));
      paint(sdBox(local.sub(vec2(0, 0.36)), vec2(wide - 0.1, 0.05), 0.04), rgb(0x0e1118), hot.oneMinus().mul(0.5));
      const ink = mix(rgb(0xb8c2d6), vec3(1.3, 1.25, 1.15), hot);
      if (k < 4) paint(pointing(local, arrows[k]!, 0.22, 0.14, 0.2), ink);
      else if (k === 4) {
        const eye = max(sdCircle(local.sub(vec2(0, 0.26)), 0.36), sdCircle(local.add(vec2(0, 0.26)), 0.36)), disc = sdCircle(local, 0.24);
        paint(min(abs(eye).sub(0.035), sdCircle(local, 0.08)), ink, when(mode.lessThan(0.5)));
        paint(disc, ink, when(mode.greaterThan(0.5).and(mode.lessThan(1.5))));
        paint(min(abs(disc).sub(0.035), max(disc, local.x)), ink, when(mode.greaterThan(1.5)));
      } else {
        const chevron = (dx: number) => sdSegment(vec2(local.x.sub(dx), abs(local.y)), vec2(-0.1, 0.2), vec2(0.1, 0), 0.05);
        paint(min(chevron(-0.12), chevron(0.12)), ink);
      }
    });
    return vec4(color, clamp(alpha, 0, 1));
  })();

  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = shade.xyz;
  material.opacityNode = shade.w;
  const left = -1, right = PANEL_WIDTH + 0.8, height = panelHeight(players) + 1;
  const mesh = new THREE.Mesh(flat((right - left) * scale, height * scale), material);
  mesh.position.set(origin.x + (left + right) / 2 * scale, 0, origin.y + (height / 2 - 0.5) * scale);
  mesh.renderOrder = 5;
  mesh.frustumCulled = false;
  return {
    mesh, rows, palette, phase, round, turn, key, pressed, mode, sweep,
    dispose() {
      rows.dispose();
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}
