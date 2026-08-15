export type Outline = Float32Array;

export function outline(points: readonly (readonly [number, number])[]): Outline {
  const out = new Float32Array(points.length * 2);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p === undefined) continue;
    out[i * 2] = p[0];
    out[i * 2 + 1] = p[1];
  }
  return out;
}

export function circleOutline(segments: number, radius = 1): Outline {
  const n = Math.max(3, Math.floor(segments));
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    out[i * 2] = Math.cos(a) * radius;
    out[i * 2 + 1] = Math.sin(a) * radius;
  }
  return out;
}

export function polygonOutline(sides: number, radius = 1, turn = 0): Outline {
  const n = Math.max(3, Math.floor(sides));
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const a = turn + (i / n) * Math.PI * 2;
    out[i * 2] = Math.cos(a) * radius;
    out[i * 2 + 1] = Math.sin(a) * radius;
  }
  return out;
}

export function starOutline(points: number, inner: number, outer: number): Outline {
  const n = Math.max(3, Math.floor(points)) * 2;
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const r = i % 2 === 0 ? outer : inner;
    out[i * 2] = Math.cos(a) * r;
    out[i * 2 + 1] = Math.sin(a) * r;
  }
  return out;
}

export function outlineCount(shape: Outline): number {
  return shape.length >> 1;
}

export function outlineAt(shape: Outline, i: number, out: [number, number]): [number, number] {
  const n = outlineCount(shape);
  const k = ((i % n) + n) % n;
  out[0] = shape[k * 2] ?? 0;
  out[1] = shape[k * 2 + 1] ?? 0;
  return out;
}
