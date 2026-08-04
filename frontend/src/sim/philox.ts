// Philox4x32-10 — the OaK randomness primitive, ported bit-exactly.
//
// Authoritative sources:
//   Sources/OakPhysics.metal L1256-1268  (hc_philox4x32, hc_u01, hc_u01open, hc_uniform)
//   tools/gridworld_ref.py   L83-100     (_philox4x32_10, _u01)
//   tools/physics_parity.py              (the cross-checked mirror)
//
// Every value here is computed in exact uint32 / float32 semantics:
//   * 32x32 -> 64 multiplies are split into 16-bit halves (Math.imul for the low
//     word, an exact-double 4-partial-product sum for the high word). No BigInt.
//   * u01 is `float(x >> 8) * 2^-24` evaluated in float32 — reproduced with
//     Math.fround. `x >>> 8` is at most 2^24-1 so it is exact in f32, and the
//     scale is a power of two, so the fround is a formality that pins the type.

/** Philox round multipliers (Salmon et al. 2011, as used by OakPhysics.metal). */
const M0 = 0xd2511f53;
const M1 = 0xcd9e8d57;
/** Golden-ratio / sqrt(3)-1 Weyl key bumps. */
const W0 = 0x9e3779b9;
const W1 = 0xbb67ae85;

/** OaK RNG stream ids: key = (laneSeed, stream). */
export const STREAM_RESET = 0;
export const STREAM_CTRL = 1;
export const STREAM_ACT = 2;
export const STREAM_GRIDWORLD = 3;

/** 2^-24, the u01 quantum. */
const TWO_POW_NEG24 = 5.9604644775390625e-8;

/**
 * High 32 bits of the unsigned 32x32 product (Metal `mulhi`).
 * Exact: every partial product is < 2^32 and the accumulation stays < 2^53.
 */
export function mulhi(a: number, b: number): number {
  const ah = a >>> 16;
  const al = a & 0xffff;
  const bh = b >>> 16;
  const bl = b & 0xffff;
  const albl = al * bl;
  const albh = al * bh;
  const ahbl = ah * bl;
  const ahbh = ah * bh;
  // carry out of the middle 16-bit column
  const mid = (albl >>> 16) + (albh & 0xffff) + (ahbl & 0xffff);
  return (ahbh + (albh >>> 16) + (ahbl >>> 16) + (mid >>> 16)) >>> 0;
}

/**
 * Philox4x32-10 with key `(k0, k1)` and counter `(c0, c1, c2, c3)`.
 * Returns the 4 output words as unsigned 32-bit numbers.
 */
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

/**
 * Philox in the OaK calling convention: key = (laneSeed, stream),
 * counter = (idx, block, 0, 0). This is the shape every OaK draw uses.
 */
export function philoxDraw(
  laneSeed: number,
  stream: number,
  idx: number,
  block = 0,
): [number, number, number, number] {
  return philox4x32_10(laneSeed, stream, idx, block, 0, 0);
}

/** `hc_u01`: float32(x >> 8) * 2^-24, in [0, 1). */
export function u01(x: number): number {
  return Math.fround((x >>> 8) * TWO_POW_NEG24);
}

/** `hc_u01open`: (float32(x >> 8) + 1) * 2^-24, in (0, 1]. */
export function u01open(x: number): number {
  return Math.fround(((x >>> 8) + 1) * TWO_POW_NEG24);
}

/**
 * `hc_uniform`: lo + (hi - lo) * u01(x), evaluated in float32 op order.
 * Caveat: the GPU may contract the multiply-add into an FMA, so this agrees to
 * <= 1 float32 ulp rather than bitwise. Only the cheetah reset noise (the
 * synthesized frame -1) depends on it, and only visually.
 */
export function uniform(x: number, lo: number, hi: number): number {
  const span = Math.fround(Math.fround(hi) - Math.fround(lo));
  return Math.fround(Math.fround(lo) + Math.fround(span * u01(x)));
}

/**
 * Uniform free-index draw: `trunc(u01(word) * n)` in float32, the exact
 * gridworld respawn convention (`int(_u01(w) * F32(n))`).
 */
export function uniformIndex(x: number, n: number): number {
  return Math.trunc(Math.fround(u01(x) * Math.fround(n)));
}

/**
 * Box-Muller pair used by `hc_rng_reset_noise` (float32 throughout):
 * rad = sqrt(-2 ln u01open(a)), ang = 2*pi*u01(b) -> (rad*cos, rad*sin).
 * Note: JS Math.log/cos/sin are double-precision libm calls, so these agree
 * with the GPU to ~1 float32 ulp rather than bitwise. The cheetah reset noise
 * is a visual init only; nothing downstream depends on bit equality here.
 */
export function boxMuller(a: number, b: number): [number, number] {
  const u1 = u01open(a);
  const u2 = u01(b);
  const rad = Math.fround(Math.sqrt(Math.fround(-2 * Math.log(u1))));
  const ang = Math.fround(Math.fround(2 * Math.PI) * u2);
  return [Math.fround(rad * Math.cos(ang)), Math.fround(rad * Math.sin(ang))];
}

/**
 * Fill `out` with `count` consecutive u32 words from the OaK stream
 * (blocks advance by 1, 4 words per block), starting at `blockStart`.
 */
export function philoxWords(
  laneSeed: number,
  stream: number,
  idx: number,
  count: number,
  blockStart = 0,
  out: Uint32Array = new Uint32Array(count),
): Uint32Array {
  for (let w = 0; w < count; w += 4) {
    const [a, b, c, d] = philoxDraw(laneSeed, stream, idx, blockStart + (w >>> 2));
    out[w] = a;
    if (w + 1 < count) out[w + 1] = b;
    if (w + 2 < count) out[w + 2] = c;
    if (w + 3 < count) out[w + 3] = d;
  }
  return out;
}
