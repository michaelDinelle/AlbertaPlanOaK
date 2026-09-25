import * as THREE from 'three/webgpu';
import { CHESS_BISHOP, CHESS_KING, CHESS_KNIGHT, CHESS_PAWN, CHESS_QUEEN, CHESS_ROOK } from '@sim/envs/chess';

export const PIECE_KINDS = 6;
const PIECE_PAD = 14;
export const PIECE_SPAN = 100 + 2 * PIECE_PAD;
export const PIECE_TEXELS = 320;
const CELL_PX = PIECE_TEXELS;
const SCALE = CELL_PX / PIECE_SPAN;
const LINE = 2.3;
const TAU = 2 * Math.PI;
const INF = 1e20;

type Pen = CanvasRenderingContext2D;
type Seg = readonly number[];
type Shape = (pen: Pen) => void;

interface Sketch {
  solid(shape: Shape): void;
  band(shape: Shape): void;
  cut(shape: Shape): void;
  line(shape: Shape): void;
  dot(shape: Shape): void;
}

function lathe(pen: Pen, x0: number, y0: number, segs: readonly Seg[]): void {
  const points: [number, number][] = [[x0, y0]];
  pen.moveTo(x0, y0);
  for (const s of segs) {
    if (s.length === 2) pen.lineTo(s[0]!, s[1]!);
    else if (s.length === 4) pen.quadraticCurveTo(s[0]!, s[1]!, s[2]!, s[3]!);
    else pen.bezierCurveTo(s[0]!, s[1]!, s[2]!, s[3]!, s[4]!, s[5]!);
    points.push([s[s.length - 2]!, s[s.length - 1]!]);
  }
  for (let i = segs.length - 1; i >= 0; i--) {
    const s = segs[i]!, [x, y] = points[i]!;
    if (s.length === 2) pen.lineTo(100 - x, y);
    else if (s.length === 4) pen.quadraticCurveTo(100 - s[0]!, s[1]!, 100 - x, y);
    else pen.bezierCurveTo(100 - s[2]!, s[3]!, 100 - s[0]!, s[1]!, 100 - x, y);
  }
  pen.closePath();
}

const plinth = (s: Sketch, ring: number, foot: number): void => {
  s.band((p) => p.roundRect(50 - ring, 76.5, 2 * ring, 6, 2));
  s.solid((p) => p.roundRect(50 - foot, 82.5, 2 * foot, 9.5, 3.2));
  s.line((p) => { p.moveTo(50 - foot, 82.5); p.lineTo(50 + foot, 82.5); });
};

const skirt = (s: Sketch, top: number, neck: number, flare: number): void => {
  s.solid((p) => lathe(p, 50, top, [[50 + neck, top], [50 + neck - 0.5, top + (77.5 - top) * 0.45, 50 + flare - 6, 72, 50 + flare, 77.5], [50, 77.5]]));
};

const SKETCHES: Record<number, (s: Sketch) => void> = {
  [CHESS_PAWN]: (s) => {
    s.solid((p) => p.arc(50, 37.5, 10, 0, TAU));
    skirt(s, 49, 7.2, 17.5);
    s.band((p) => p.roundRect(36.5, 47, 27, 6, 3));
    plinth(s, 19, 22);
  },
  [CHESS_ROOK]: (s) => {
    s.solid((p) => p.roundRect(32.5, 19, 35, 14.5, [1.5, 1.5, 3, 3]));
    s.cut((p) => { p.rect(40.3, 16, 5.6, 8.5); p.rect(54.1, 16, 5.6, 8.5); });
    s.solid((p) => lathe(p, 50, 33, [[66.5, 33], [62.5, 38.5], [50, 38.5]]));
    s.solid((p) => lathe(p, 50, 38, [[62.5, 38], [62.2, 52, 63.5, 66, 64.5, 72], [50, 72]]));
    s.line((p) => { p.moveTo(33, 33.2); p.lineTo(67, 33.2); });
    s.band((p) => p.roundRect(34, 70.5, 32, 6, 2));
    plinth(s, 20, 23);
  },
  [CHESS_KNIGHT]: (s) => {
    s.solid((p) => {
      p.moveTo(66.5, 77);
      p.bezierCurveTo(72.5, 63, 73, 49, 67.5, 37);
      p.bezierCurveTo(64.5, 30, 60.5, 23.5, 56.5, 18.5);
      p.lineTo(57.5, 8.5);
      p.lineTo(51.5, 15);
      p.bezierCurveTo(46, 15.5, 41.5, 18.5, 37.5, 22.5);
      p.bezierCurveTo(32.5, 27.5, 27, 35.5, 24, 43.5);
      p.bezierCurveTo(22.8, 47, 24.5, 50.8, 28.5, 51.2);
      p.lineTo(33.5, 50.6);
      p.bezierCurveTo(37.5, 50.2, 41, 49.6, 44.5, 51.5);
      p.bezierCurveTo(40.5, 56.5, 36.5, 63, 35.2, 70);
      p.bezierCurveTo(34.8, 73, 34.3, 75.5, 33.8, 77.5);
      p.closePath();
    });
    s.line((p) => { p.moveTo(57.5, 20.5); p.bezierCurveTo(63.5, 28, 67.8, 38, 68.6, 52); });
    s.line((p) => { p.moveTo(28.8, 51); p.bezierCurveTo(31.5, 49.2, 33.5, 48.4, 35.8, 48.2); });
    s.line((p) => { p.moveTo(52.5, 44); p.bezierCurveTo(50.5, 48, 47.5, 50.2, 44.5, 51.5); });
    s.dot((p) => p.ellipse(43.8, 29.5, 2.6, 2.1, -0.5, 0, TAU));
    s.dot((p) => p.ellipse(28.6, 44.2, 1.5, 1.2, -0.6, 0, TAU));
    plinth(s, 19.5, 23);
  },
  [CHESS_BISHOP]: (s) => {
    s.solid((p) => p.arc(50, 15.2, 4.4, 0, TAU));
    s.solid((p) => lathe(p, 50, 18.8, [[57.5, 24, 63.8, 33, 62.3, 41.2], [61.3, 46.2, 57, 48.8, 50, 48.8]]));
    s.line((p) => { p.moveTo(56.8, 28.4); p.lineTo(47.6, 38.6); });
    skirt(s, 52, 6.3, 16.5);
    s.band((p) => p.roundRect(36.5, 47.5, 27, 6, 3));
    plinth(s, 19.5, 23);
  },
  [CHESS_QUEEN]: (s) => {
    s.solid((p) => lathe(p, 50, 17, [[53.8, 30.5], [60.2, 19.5], [63.3, 31.8], [70.3, 23.8], [64.8, 41.5], [50, 41.5]]));
    s.solid((p) => { p.arc(50, 14.3, 3.9, 0, TAU); });
    s.solid((p) => { p.arc(60.4, 17.2, 3.4, 0, TAU); p.moveTo(43, 17.2); p.arc(39.6, 17.2, 3.4, 0, TAU); });
    s.solid((p) => { p.arc(70.8, 21.8, 3.4, 0, TAU); p.moveTo(32.6, 21.8); p.arc(29.2, 21.8, 3.4, 0, TAU); });
    s.band((p) => p.roundRect(35, 39.5, 30, 5.5, 2));
    skirt(s, 44.5, 7.6, 17.5);
    s.band((p) => p.roundRect(36, 48.5, 28, 5.5, 2.75));
    plinth(s, 20, 23.5);
  },
  [CHESS_KING]: (s) => {
    s.solid((p) => { p.roundRect(46.8, 3, 6.4, 17.7, 1.2); p.roundRect(41.6, 7.3, 16.8, 6, 1.2); });
    s.solid((p) => lathe(p, 50, 19.2, [[58.5, 19.2, 64.5, 25.5, 64.3, 33.8], [50, 33.8]]));
    s.band((p) => p.roundRect(34.5, 33, 31, 6.5, 2));
    s.solid((p) => lathe(p, 50, 39, [[58, 39], [56.2, 42, 56.5, 45.5, 57.8, 48], [50, 48]]));
    skirt(s, 50, 7.6, 17.5);
    s.band((p) => p.roundRect(35.5, 46.5, 29, 5.5, 2.75));
    plinth(s, 20, 23.5);
  },
};

function sketchInto(pen: Pen, kind: number, detail: boolean): void {
  const draw = SKETCHES[kind];
  if (!draw) throw new Error(`Chess has no piece of kind ${kind}`);
  pen.save();
  pen.fillStyle = pen.strokeStyle = '#ffffff';
  pen.lineWidth = LINE;
  pen.lineCap = pen.lineJoin = 'round';
  const run = (shape: Shape, action: 'fill' | 'stroke' | 'cut' | null) => {
    if (!action) return;
    pen.beginPath();
    shape(pen);
    if (action === 'stroke') pen.stroke();
    else {
      pen.globalCompositeOperation = action === 'cut' ? 'destination-out' : 'source-over';
      pen.fill();
      pen.globalCompositeOperation = 'source-over';
    }
  };
  draw({
    solid: (shape) => run(shape, detail ? null : 'fill'),
    band: (shape) => run(shape, detail ? 'stroke' : 'fill'),
    cut: (shape) => run(shape, detail ? null : 'cut'),
    line: (shape) => run(shape, detail ? 'stroke' : null),
    dot: (shape) => run(shape, detail ? 'fill' : null),
  });
  pen.restore();
}

function edt1d(grid: Float64Array, offset: number, stride: number, length: number, f: Float64Array, v: Uint16Array, z: Float64Array): void {
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  f[0] = grid[offset]!;
  for (let q = 1, k = 0, s = 0; q < length; q++) {
    f[q] = grid[offset + q * stride]!;
    const q2 = q * q;
    do {
      const r = v[k]!;
      s = (f[q]! - f[r]! + q2 - r * r) / (q - r) / 2;
    } while (s <= z[k]! && --k > -1);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  for (let q = 0, k = 0; q < length; q++) {
    while (z[k + 1]! < q) k++;
    const r = v[k]!, qr = q - r;
    grid[offset + q * stride] = f[r]! + qr * qr;
  }
}

function edt(grid: Float64Array, width: number, height: number): void {
  const size = Math.max(width, height), f = new Float64Array(size), v = new Uint16Array(size), z = new Float64Array(size + 1);
  for (let x = 0; x < width; x++) edt1d(grid, x, width, height, f, v, z);
  for (let y = 0; y < height; y++) edt1d(grid, y * width, 1, width, f, v, z);
}

function signedField(alpha: Uint8ClampedArray, width: number, height: number): Float64Array {
  const count = width * height, outer = new Float64Array(count), inner = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const a = alpha[4 * i + 3]! / 255;
    outer[i] = a >= 1 ? 0 : a <= 0 ? INF : Math.max(0, 0.5 - a) ** 2;
    inner[i] = a >= 1 ? INF : a <= 0 ? 0 : Math.max(0, a - 0.5) ** 2;
  }
  edt(outer, width, height);
  edt(inner, width, height);
  for (let i = 0; i < count; i++) outer[i] = Math.sqrt(outer[i]!) - Math.sqrt(inner[i]!);
  return outer;
}

export function pieceAtlas(): THREE.DataTexture {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = CELL_PX;
  const pen = canvas.getContext('2d', { willReadFrequently: true });
  if (!pen) throw new Error('Chess pieces need a 2D canvas');
  const width = PIECE_KINDS * CELL_PX, data = new Uint16Array(width * CELL_PX * 4), one = THREE.DataUtils.toHalfFloat(1);
  const layer = (kind: number, detail: boolean): Float64Array => {
    pen.setTransform(1, 0, 0, 1, 0, 0);
    pen.clearRect(0, 0, CELL_PX, CELL_PX);
    pen.setTransform(SCALE, 0, 0, SCALE, PIECE_PAD * SCALE, PIECE_PAD * SCALE);
    sketchInto(pen, kind, detail);
    return signedField(pen.getImageData(0, 0, CELL_PX, CELL_PX).data, CELL_PX, CELL_PX);
  };
  for (let kind = 1; kind <= PIECE_KINDS; kind++) {
    const body = layer(kind, false), detail = layer(kind, true);
    for (let y = 0; y < CELL_PX; y++) for (let x = 0; x < CELL_PX; x++) {
      const from = y * CELL_PX + x, to = 4 * ((CELL_PX - 1 - y) * width + (kind - 1) * CELL_PX + x);
      data[to] = THREE.DataUtils.toHalfFloat(body[from]! / CELL_PX);
      data[to + 1] = THREE.DataUtils.toHalfFloat(detail[from]! / CELL_PX);
      data[to + 2] = 0;
      data[to + 3] = one;
    }
  }
  const texture = new THREE.DataTexture(data, width, CELL_PX, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}
