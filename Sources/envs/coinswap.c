#include "../env.h"

typedef struct { float swapRate; } coinswapKnobs;

OAK_CONSTANT coinswapKnobs COINSWAP_KNOBS[2] = {
    { 0.0001f },
    { 0.01f }
};


void coinswap_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
}

float coinswap_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const coinswapKnobs *k = &COINSWAP_KNOBS[instance];
    *done = 0u;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float bias = ((uint32_t)ctrl[0] & 1u) == words[0] ? 0.9f : 0.1f;
    float reward = u01(r.x) < bias ? 1.0f : 0.0f;
    if (u01(r.y) < k->swapRate) words[0] ^= 1u;
    obs[0] = 1.0f;
    return reward;
}

void coinswap_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void coinswap_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
