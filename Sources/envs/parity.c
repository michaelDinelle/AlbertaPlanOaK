#include "../env.h"

typedef struct { uint32_t delay; } parityKnobs;

static const parityKnobs PARITY_KNOBS[2] = {
    { 4u },
    { 64u }
};


static uint32_t parity_bitAt(const uint32_t *words, uint32_t age) { return (words[age >> 5u] >> (age & 31u)) & 1u; }

static void parity_push(uint32_t *words, uint32_t bit) {
    for (uint32_t w = 2u; w > 0u; w--) words[w] = (words[w] << 1u) | (words[w - 1u] >> 31u);
    words[0] = (words[0] << 1u) | bit;
}

static uint32_t parity_target(const parityKnobs *k, const uint32_t *words) { return parity_bitAt(words, k->delay) ^ parity_bitAt(words, k->delay + 1u); }

void parity_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    u4 r = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    words[0] = r.x; words[1] = r.y; words[2] = r.z;
}

float parity_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const parityKnobs *k = &PARITY_KNOBS[instance];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == parity_target(k, words) ? 1.0f : 0.0f;
    parity_push(words, philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x & 1u);
    obs[0] = (float)parity_bitAt(words, 0u);
    obs[1] = 1.0f;
    return reward;
}

void parity_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    obs[0] = (float)parity_bitAt(words, 0u);
    obs[1] = 1.0f;
}

void parity_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)parity_target(&PARITY_KNOBS[instance], words);
}
