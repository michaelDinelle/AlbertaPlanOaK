
const M0 = 0xd2511f53;
const M1 = 0xcd9e8d57;
const W0 = 0x9e3779b9;
const W1 = 0xbb67ae85;

export const STREAM_RESET = 0;

export function mulhi(a: number, b: number): number {
  const ah = a >>> 16;
  const al = a & 0xffff;
  const bh = b >>> 16;
  const bl = b & 0xffff;
  const albl = al * bl;
  const albh = al * bh;
  const ahbl = ah * bl;
  const ahbh = ah * bh;
  const mid = (albl >>> 16) + (albh & 0xffff) + (ahbl & 0xffff);
  return (ahbh + (albh >>> 16) + (ahbl >>> 16) + (mid >>> 16)) >>> 0;
}

export function philox4x32_10(
  k0: number,
  k1: number,
  c0: number,
  c1: number,
  c2: number,
  c3: number,
): [number, number, number, number] {
  let key0 = k0 >>> 0;
  let key1 = k1 >>> 0;
  let x0 = c0 >>> 0;
  let x1 = c1 >>> 0;
  let x2 = c2 >>> 0;
  let x3 = c3 >>> 0;
  for (let r = 0; r < 10; r++) {
    if (r > 0) {
      key0 = (key0 + W0) >>> 0;
      key1 = (key1 + W1) >>> 0;
    }
    const hi0 = mulhi(M0, x0);
    const lo0 = Math.imul(M0, x0) >>> 0;
    const hi1 = mulhi(M1, x2);
    const lo1 = Math.imul(M1, x2) >>> 0;
    const n0 = (hi1 ^ x1 ^ key0) >>> 0;
    const n2 = (hi0 ^ x3 ^ key1) >>> 0;
    x0 = n0;
    x1 = lo1;
    x2 = n2;
    x3 = lo0;
  }
  return [x0, x1, x2, x3];
}

export function philoxDraw(
  laneSeed: number,
  stream: number,
  idx: number,
  block = 0,
): [number, number, number, number] {
  return philox4x32_10(laneSeed, stream, idx, block, 0, 0);
}
