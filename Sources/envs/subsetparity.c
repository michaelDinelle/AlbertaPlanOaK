#include "../env.h"

typedef struct { uint32_t inputs, subset; } subsetparityKnobs;

static const subsetparityKnobs SUBSETPARITY_KNOBS[4] = {
    { 8u, 1u },
    { 16u, 1u },
    { 8u, 2u },
    { 16u, 2u }
};


static uint32_t subsetparity_bits(const subsetparityKnobs *k, uint32_t seed, uint32_t index) { return philox4x32(seed, ENV_RNG_STREAM, index, 0u, 0u, 0u).x & ((1u << k->inputs) - 1u); }

static void subsetparity_obs(const subsetparityKnobs *k, uint32_t bits, float *obs) {
    for (uint32_t b = 0u; b < k->inputs; b++) obs[b] = (float)((bits >> b) & 1u);
    obs[k->inputs] = 1.0f;
}

static uint32_t subsetparity_target(uint32_t mask, uint32_t bits) { return oak_popcount(mask & bits) & 1u; }

void subsetparity_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const subsetparityKnobs *k = &SUBSETPARITY_KNOBS[instance];
    uint32_t mask = 0u;
    for (uint32_t i = 0u; (uint32_t)oak_popcount(mask) < k->subset; i++) mask |= 1u << (philox4x32(seed, RNG_STREAM_RESET, i, 0u, 0u, 0u).x % k->inputs);
    words[0] = mask;
    words[1] = subsetparity_bits(k, seed, 0u);
}

float subsetparity_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const subsetparityKnobs *k = &SUBSETPARITY_KNOBS[instance];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == subsetparity_target(words[0], words[1]) ? 1.0f : 0.0f;
    words[1] = subsetparity_bits(k, seed, step);
    subsetparity_obs(k, words[1], obs);
    return reward;
}

void subsetparity_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    subsetparity_obs(&SUBSETPARITY_KNOBS[instance], words[1], obs);
}

void subsetparity_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)subsetparity_target(words[0], words[1]);
}
