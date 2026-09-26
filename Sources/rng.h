#ifndef OAK_RNG_H
#define OAK_RNG_H
#include <arm_neon.h>
#include "portable.h"

#define RNG_STREAM_RESET 0u
#define RNG_STREAM_ACT 2u
#define ENV_RNG_STREAM 3u
#define RNG_STREAM_RANDOM_ACT 4u
#define RNG_STREAM_PPO_INIT 5u
#define RNG_STREAM_PPO_ACT 6u
#define RNG_STREAM_PPO_SHUFFLE 7u
#define LEAGUE_RNG_STREAM 8u

typedef struct { uint32_t x, y, z, w; } u4;

static const uint32_t PHILOX4_WEYL[8] = { 0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u, 0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u };

static inline u4 philox4x32(uint32_t key0, uint32_t key1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
    const uint32_t *weyl = (const uint32_t *)opaque(PHILOX4_WEYL);
    uint32_t weyl0 = weyl[0], weyl1 = weyl[4];
    u4 ctr = { c0, c1, c2, c3 };
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            key0 += weyl0;
            key1 += weyl1;
        }
        uint64_t product0 = (uint64_t)0xD2511F53u * ctr.x, product1 = (uint64_t)0xCD9E8D57u * ctr.z;
        u4 next = { (uint32_t)(product1 >> 32) ^ ctr.y ^ key0, (uint32_t)product1, (uint32_t)(product0 >> 32) ^ ctr.w ^ key1, (uint32_t)product0 };
        ctr = next;
    }
    return ctr;
}

typedef struct { lane4 x, y, z, w; } u44;

static inline u44 philox4(uint32_t key0, uint32_t key1, lane4 c0, lane4 c1, lane4 c2, lane4 c3) {
    const uint32_t *weyl = (const uint32_t *)opaque(PHILOX4_WEYL);
    uint32x4_t x = (uint32x4_t)c0, y = (uint32x4_t)c1, z = (uint32x4_t)c2, w = (uint32x4_t)c3, k0 = vdupq_n_u32(key0), k1 = vdupq_n_u32(key1);
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            k0 = vaddq_u32(k0, vld1q_u32(weyl));
            k1 = vaddq_u32(k1, vld1q_u32(weyl + 4));
        }
        uint32x4_t a0 = vreinterpretq_u32_u64(vmull_n_u32(vget_low_u32(x), 0xD2511F53u)), b0 = vreinterpretq_u32_u64(vmull_high_n_u32(x, 0xD2511F53u));
        uint32x4_t a1 = vreinterpretq_u32_u64(vmull_n_u32(vget_low_u32(z), 0xCD9E8D57u)), b1 = vreinterpretq_u32_u64(vmull_high_n_u32(z, 0xCD9E8D57u));
        x = veor3q_u32(vuzp2q_u32(a1, b1), y, k0);
        y = vuzp1q_u32(a1, b1);
        z = veor3q_u32(vuzp2q_u32(a0, b0), w, k1);
        w = vuzp1q_u32(a0, b0);
    }
    return (u44){ (lane4)x, (lane4)y, (lane4)z, (lane4)w };
}

static inline float u01(uint32_t x) { return (float)(x >> 8) * 0x1p-24f; }
static inline uint32_t u01Bound(float p) { return (uint32_t)ceilf(p * 0x1p24f) << 8u; }
static inline float u01open(uint32_t x) { return ((float)(x >> 8) + 1.0f) * 0x1p-24f; }
static inline float uniformRange(uint32_t x, float lo, float hi) { return lo + (hi - lo) * u01(x); }

static inline float4 replicate4(const float *p) { return (float4)vld1q_dup_f32(p); }

static const float NORMAL4_TERMS[18] = { -4.0f / 11.0f, -4.0f / 9.0f, -4.0f / 7.0f, -4.0f / 5.0f, -4.0f / 3.0f, -2.0f * 0.693147180559945f, 2.0f * OAK_PI,
                                         0.636619772f, 1.57079601287841796875f, 3.1391647e-07f, 1.0f / 362880.0f, -1.0f / 5040.0f, 1.0f / 120.0f, -1.0f / 6.0f,
                                         -1.0f / 3628800.0f, 1.0f / 40320.0f, -1.0f / 720.0f, 1.0f / 24.0f };

static inline float4 normal4(lane4 w0, lane4 w1) {
    const float *k = (const float *)opaque(NORMAL4_TERMS);
    lane4 t = (lane4)vcvtq_n_f32_u32((uint32x4_t)((w0 >> 8u) + 1u), 24) - 0x3F3504F4u;
    float4 m = (float4)((t & 0x007FFFFFu) + 0x3F3504F4u);
    float4 s = (m - 1.0f) / (m + 1.0f), s2 = s * s;
    float4 p = replicate4(k);
    p = __builtin_elementwise_fma(p, s2, replicate4(k + 1));
    p = __builtin_elementwise_fma(p, s2, replicate4(k + 2));
    p = __builtin_elementwise_fma(p, s2, replicate4(k + 3));
    p = __builtin_elementwise_fma(p, s2, replicate4(k + 4));
    p = __builtin_elementwise_fma(p, s2, splat4(-4.0f));
    float4 radius = __builtin_elementwise_sqrt(__builtin_elementwise_fma(__builtin_convertvector((int4)t >> 23, float4), replicate4(k + 5), s * p));
    float4 x = replicate4(k + 6) * (float4)vcvtq_n_f32_u32((uint32x4_t)(w1 >> 8u), 24);
    float4 j = __builtin_elementwise_floor(__builtin_elementwise_fma(x, replicate4(k + 7), splat4(0.5f)));
    float4 r = __builtin_elementwise_fma(-j, replicate4(k + 8), x);
    r = __builtin_elementwise_fma(-j, replicate4(k + 9), r);
    float4 r2 = r * r;
    float4 sp = replicate4(k + 10);
    sp = __builtin_elementwise_fma(sp, r2, replicate4(k + 11));
    sp = __builtin_elementwise_fma(sp, r2, replicate4(k + 12));
    sp = __builtin_elementwise_fma(sp, r2, replicate4(k + 13));
    sp = __builtin_elementwise_fma(sp, r2, splat4(1.0f));
    float4 sr = r * sp;
    float4 cp = replicate4(k + 14);
    cp = __builtin_elementwise_fma(cp, r2, replicate4(k + 15));
    cp = __builtin_elementwise_fma(cp, r2, replicate4(k + 16));
    cp = __builtin_elementwise_fma(cp, r2, replicate4(k + 17));
    cp = __builtin_elementwise_fma(cp, r2, splat4(-0.5f));
    float4 cr = __builtin_elementwise_fma(cp, r2, splat4(1.0f));
    lane4 q = (lane4)__builtin_convertvector(j, int4);
    float4 c = select4((lane4)((q & 1u) != 0u), sr, cr);
    return radius * (float4)((lane4)c ^ ((q + 1u) << 30 & 0x80000000u));
}

static inline float4 normalFrom4(u4 a, u4 b) {
    lane4 va = { a.x, a.y, a.z, a.w }, vb = { b.x, b.y, b.z, b.w };
    return normal4(__builtin_shufflevector(va, vb, 0, 2, 4, 6), __builtin_shufflevector(va, vb, 1, 3, 5, 7));
}

static inline float normalFrom(uint32_t w0, uint32_t w1) { return normalFrom4((u4){ w0, w1, 0u, 0u }, (u4){ 0u, 0u, 0u, 0u }).x; }

static inline uint32_t baselineWord(uint32_t seed, uint32_t stream, uint32_t stepIdx, uint32_t i) {
    u4 r = philox4x32(seed, stream, stepIdx, i >> 2, 0u, 0u);
    uint32_t words[4] = { r.x, r.y, r.z, r.w };
    return words[i & 3u];
}

#endif
