#ifndef OAK_PORTABLE_H
#define OAK_PORTABLE_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define OAK_PI ((float)M_PI)
#define oak_popcount(x) __builtin_popcount(x)
#define OAK_LANES 1u
static inline float wordFloat(uint32_t w) { union { uint32_t u; float f; } v; v.u = w; return v.f; }
static inline uint32_t floatWord(float f) { union { uint32_t u; float f; } v; v.f = f; return v.u; }

typedef float float4 __attribute__((ext_vector_type(4)));
typedef uint32_t lane4 __attribute__((ext_vector_type(4)));
typedef int32_t int4 __attribute__((ext_vector_type(4)));
static inline float4 splat4(float x) { float4 v = x; return v; }
static inline const void *opaque(const void *p) { __asm__("" : "+r"(p)); return p; }
static inline float4 flag4(lane4 set) { return (float4)((lane4)(set != 0u) & 0x3F800000u); }
static inline float4 select4(lane4 mask, float4 yes, float4 no) { return (float4)(((lane4)yes & mask) | ((lane4)no & ~mask)); }

static inline float oak_pow2i(int n) { return wordFloat((uint32_t)(n + 127) << 23); }

static inline float oak_exp(float x) {
    float n = floorf(fmaf(x, 1.44269504f, 0.5f));
    float f = fmaf(-n, 0.693145751953125f, x);
    f = fmaf(-n, 1.428606765330187e-06f, f);
    float p = 1.0f / 5040.0f;
    p = fmaf(p, f, 1.0f / 720.0f);
    p = fmaf(p, f, 1.0f / 120.0f);
    p = fmaf(p, f, 1.0f / 24.0f);
    p = fmaf(p, f, 1.0f / 6.0f);
    p = fmaf(p, f, 0.5f);
    p = fmaf(p, f, 1.0f);
    p = fmaf(p, f, 1.0f);
    float bounded = fminf(fmaxf(n, -149.0f), 127.0f);
    float r = n > 127.0f ? (p * 2.0f) * oak_pow2i(127) : p * oak_pow2i((int)bounded);
    r = x < -87.33654f ? 0.0f : r;
    r = x > 88.72283f ? INFINITY : r;
    return x != x ? x : r;
}

static inline float oak_log(float x) {
    if (x != x) return x;
    if (x < 0.0f) return NAN;
    if (x == 0.0f) return -INFINITY;
    if (x == INFINITY) return INFINITY;
    uint32_t bits = floatWord(x);
    int e = (int)((bits >> 23) & 255u) - 127;
    float m = wordFloat((bits & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421356f) {
        m *= 0.5f;
        e += 1;
    }
    float s = (m - 1.0f) / (m + 1.0f);
    float s2 = s * s;
    float p = 1.0f / 11.0f;
    p = fmaf(p, s2, 1.0f / 9.0f);
    p = fmaf(p, s2, 1.0f / 7.0f);
    p = fmaf(p, s2, 1.0f / 5.0f);
    p = fmaf(p, s2, 1.0f / 3.0f);
    p = fmaf(p, s2, 1.0f);
    return fmaf((float)e, 0.693147180559945f, 2.0f * s * p);
}

static inline float oak_tanh(float x) {
    float a = fabsf(x);
    float a2 = a * a;
    float small = a * fmaf(a2, fmaf(a2, 2.0f / 15.0f, -1.0f / 3.0f), 1.0f);
    float e = oak_exp(2.0f * a);
    float wide = 1.0f - 2.0f / (e + 1.0f);
    float t = a < 0.04f ? small : a > 9.0f ? 1.0f : wide;
    t = x < 0.0f ? -t : t;
    return x != x ? x : t;
}

static inline void oak_sincosQuarter(float x, float j, float *s, float *c) {
    float r = fmaf(-j, 1.57079601287841796875f, x);
    r = fmaf(-j, 3.1391647e-07f, r);
    float r2 = r * r;
    float sp = 1.0f / 362880.0f;
    sp = fmaf(sp, r2, -1.0f / 5040.0f);
    sp = fmaf(sp, r2, 1.0f / 120.0f);
    sp = fmaf(sp, r2, -1.0f / 6.0f);
    sp = fmaf(sp, r2, 1.0f);
    float cp = -1.0f / 3628800.0f;
    cp = fmaf(cp, r2, 1.0f / 40320.0f);
    cp = fmaf(cp, r2, -1.0f / 720.0f);
    cp = fmaf(cp, r2, 1.0f / 24.0f);
    cp = fmaf(cp, r2, -0.5f);
    int q = ((int)j) & 3;
    float sSign = q & 2 ? -1.0f : 1.0f, cSign = (q + 1) & 2 ? -1.0f : 1.0f;
    *s = q & 1 ? fmaf(cp, r2 * sSign, sSign) : r * sSign * sp;
    *c = q & 1 ? r * cSign * sp : fmaf(cp, r2 * cSign, cSign);
}

static inline void oak_sincos(float x, float *s, float *c) {
    if (x != x || x == INFINITY || x == -INFINITY) {
        *s = NAN;
        *c = NAN;
        return;
    }
    oak_sincosQuarter(x, floorf(fmaf(x, 0.636619772f, 0.5f)), s, c);
}

static inline float oak_sin(float x) { float s, c; oak_sincos(x, &s, &c); return s; }
static inline float oak_cos(float x) { float s, c; oak_sincos(x, &s, &c); return c; }
static inline float oak_rsqrt(float x) { return 1.0f / sqrtf(x); }
static inline float oak_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
static inline float finiteOrZero(float x) { return isfinite(x) ? x : 0.0f; }
static inline float keepFinite(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }

#endif
