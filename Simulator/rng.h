#ifndef RNG_H
#define RNG_H
#include <stdint.h>

#define PHILOX_ROUNDS 10
#define PHILOX_MULTIPLIER_0 0xD2511F53u
#define PHILOX_MULTIPLIER_1 0xCD9E8D57u
#define PHILOX_WEYL_0 0x9E3779B9u
#define PHILOX_WEYL_1 0xBB67AE85u

#define RNG_KEY_AGENT 2u
#define RNG_KEY_ENVIRONMENT 3u

typedef struct { uint32_t x, y, z, w; } u4;

static inline u4 philox4x32(uint32_t key0, uint32_t key1, u4 counter){
    for (int round = 0; round < PHILOX_ROUNDS; ++round){
        if (round > 0){
            key0 += PHILOX_WEYL_0;
            key1 += PHILOX_WEYL_1;
        }
        uint32_t high0 = (uint32_t)(((uint64_t)PHILOX_MULTIPLIER_0 * counter.x) >> 32);
        uint32_t low0  = PHILOX_MULTIPLIER_0 * counter.x;
        uint32_t high1 = (uint32_t)(((uint64_t)PHILOX_MULTIPLIER_1 * counter.z) >> 32);
        uint32_t low1  = PHILOX_MULTIPLIER_1 * counter.z;
        u4 next = { high1 ^ counter.y ^ key0, low1, high0 ^ counter.w ^ key1, low0 };
        counter = next;
    }
    return counter;
}

#endif
