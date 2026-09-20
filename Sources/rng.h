#ifndef OAK_RNG_H
#define OAK_RNG_H
#include "portable.h"

#define RNG_STREAM_RESET 0u
#define RNG_STREAM_ACT 2u
#define ENV_RNG_STREAM 3u
#define RNG_STREAM_RANDOM_ACT 4u
#define RNG_STREAM_PPO_INIT 5u
#define RNG_STREAM_PPO_ACT 6u
#define RNG_STREAM_PPO_SHUFFLE 7u

typedef struct { uint32_t x, y, z, w; } u4;

static inline u4 philox4x32(uint32_t key0, uint32_t key1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
    u4 ctr = { c0, c1, c2, c3 };
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            key0 += 0x9E3779B9u;
            key1 += 0xBB67AE85u;
        }
        uint32_t hi0 = mulhi(0xD2511F53u, ctr.x), lo0 = 0xD2511F53u * ctr.x;
        uint32_t hi1 = mulhi(0xCD9E8D57u, ctr.z), lo1 = 0xCD9E8D57u * ctr.z;
        u4 next = { hi1 ^ ctr.y ^ key0, lo1, hi0 ^ ctr.w ^ key1, lo0 };
        ctr = next;
    }
    return ctr;
}

static inline float u01(uint32_t x) { return (float)(x >> 8) * 0x1p-24f; }
static inline float u01open(uint32_t x) { return ((float)(x >> 8) + 1.0f) * 0x1p-24f; }
static inline float uniformRange(uint32_t x, float lo, float hi) { return lo + (hi - lo) * u01(x); }

static inline float normalFrom(uint32_t w0, uint32_t w1) {
    float s, c;
    oak_sincos(2.0f * OAK_PI * u01(w1), &s, &c);
    return sqrtf(-2.0f * oak_log(u01open(w0))) * c;
}

static inline uint32_t baselineWord(uint32_t seed, uint32_t stream, uint32_t stepIdx, uint32_t i) {
    u4 r = philox4x32(seed, stream, stepIdx, i >> 2, 0u, 0u);
    uint32_t words[4] = { r.x, r.y, r.z, r.w };
    return words[i & 3u];
}

#endif
