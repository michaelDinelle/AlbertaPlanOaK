#include "../env.h"

typedef struct { uint32_t inputs, subset; } subsetparityKnobs;

static const subsetparityKnobs SUBSETPARITY_KNOBS[4] = {
    { 8u, 1u },
    { 16u, 1u },
    { 8u, 2u },
    { 16u, 2u }
};


uint32_t subsetparity_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static uint32_t subsetparity_bits(const subsetparityKnobs *k, uint32_t seed, uint32_t index) { return philox4x32(seed, ENV_RNG_STREAM, index, 0u, 0u, 0u).x & ((1u << k->inputs) - 1u); }

static void subsetparity_draw(const subsetparityKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 bits = r.x & ((1u << k->inputs) - 1u);
    __builtin_memcpy(draws, &bits, sizeof bits);
}

static void subsetparity_obs(const subsetparityKnobs *k, uint32_t bits, float *obs) {
    for (uint32_t b = 0u; b < k->inputs; b += 4u) {
        float4 row = flag4(bits & ((lane4){ 1u, 2u, 4u, 8u } << b));
        __builtin_memcpy(obs + b, &row, sizeof row);
    }
    obs[k->inputs] = 1.0f;
}

static uint32_t subsetparity_target(uint32_t mask, uint32_t bits) {
    uint32_t x = mask & bits;
    x ^= x >> 16u; x ^= x >> 8u; x ^= x >> 4u; x ^= x >> 2u; x ^= x >> 1u;
    return x & 1u;
}

static inline __attribute__((always_inline)) float subsetparity_advance(const subsetparityKnobs *k, const float *ctrl, uint32_t *words, uint32_t bits, float *obs) {
    float reward = ((uint32_t)ctrl[0] & 1u) == subsetparity_target(words[0], words[1]) ? 1.0f : 0.0f;
    words[1] = bits;
    subsetparity_obs(k, bits, obs);
    return reward;
}

static inline __attribute__((always_inline)) float subsetparity_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t bits, float *obs, uint32_t *done) {
    *done = 0u;
    switch (instance) {
        case 0u: return subsetparity_advance(&SUBSETPARITY_KNOBS[0], ctrl, words, bits, obs);
        case 1u: return subsetparity_advance(&SUBSETPARITY_KNOBS[1], ctrl, words, bits, obs);
        case 2u: return subsetparity_advance(&SUBSETPARITY_KNOBS[2], ctrl, words, bits, obs);
        default: return subsetparity_advance(&SUBSETPARITY_KNOBS[3], ctrl, words, bits, obs);
    }
}

static __attribute__((noinline)) float subsetparity_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    subsetparity_draw(&SUBSETPARITY_KNOBS[instance], seed, step, extra + 1);
    return subsetparity_play(instance, ctrl, words, extra[1], obs, done);
}

void subsetparity_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const subsetparityKnobs *k = &SUBSETPARITY_KNOBS[instance];
    uint32_t mask = 0u;
    for (uint32_t i = 0u; (uint32_t)oak_popcount(mask) < k->subset; i++) mask |= 1u << (philox4x32(seed, RNG_STREAM_RESET, i, 0u, 0u, 0u).x % k->inputs);
    words[0] = mask;
    words[1] = subsetparity_bits(k, seed, 0u);
    batchReset(&extra[0]);
}

float subsetparity_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return subsetparity_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return subsetparity_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void subsetparity_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    subsetparity_obs(&SUBSETPARITY_KNOBS[instance], words[1], obs);
}

void subsetparity_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)subsetparity_target(words[0], words[1]);
}
