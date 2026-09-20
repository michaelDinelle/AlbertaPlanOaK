#include "../env.h"

#define CELLULAR_CELLS 16u
#define CELLULAR_RULE 110u

static uint32_t cellular_next(uint32_t cells) {
    uint32_t next = 0u;
    for (uint32_t i = 0u; i < CELLULAR_CELLS; i++) {
        uint32_t left = (cells >> ((i + CELLULAR_CELLS - 1u) % CELLULAR_CELLS)) & 1u;
        uint32_t mid = (cells >> i) & 1u;
        uint32_t right = (cells >> ((i + 1u) % CELLULAR_CELLS)) & 1u;
        next |= ((CELLULAR_RULE >> ((left << 2u) | (mid << 1u) | right)) & 1u) << i;
    }
    return next;
}

static void cellular_obs(uint32_t cells, device float *obs) {
    for (uint32_t i = 0u; i < CELLULAR_CELLS; i++) obs[i] = (float)((cells >> i) & 1u);
    obs[CELLULAR_CELLS] = 1.0f;
}

void cellular_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & ((1u << CELLULAR_CELLS) - 1u);
}

float cellular_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    *done = 0u;
    uint32_t cells = words[0] ^ ((uint32_t)ctrl[0] & 1u);
    cells = cellular_next(cells);
    if ((step % 64u) == 0u) cells ^= philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x & 0x8000u;
    words[0] = cells;
    cellular_obs(cells, obs);
    return (float)((cells >> 1u) & 1u);
}

void cellular_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    cellular_obs(words[0], obs);
}

void cellular_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t keep = (cellular_next(words[0]) >> 1u) & 1u, flip = (cellular_next(words[0] ^ 1u) >> 1u) & 1u;
    ctrl[0] = flip > keep ? 1.0f : 0.0f;
}
