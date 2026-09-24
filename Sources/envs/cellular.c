#include "../env.h"

#define CELLULAR_CELLS 16u
#define CELLULAR_RULE 110u

_Static_assert(CELLULAR_RULE == 110u, "");

static uint32_t cellular_next(uint32_t cells) {
    uint32_t mid = cells & ((1u << CELLULAR_CELLS) - 1u), ring = mid | mid << CELLULAR_CELLS, left = ring << 1u | ring >> 31u, right = ring >> 1u | ring << 31u;
    return ((mid ^ right) | (mid & ~left)) & ((1u << CELLULAR_CELLS) - 1u);
}

static void cellular_obs(uint32_t cells, float *obs) {
    for (uint32_t i = 0u; i < CELLULAR_CELLS; i += 4u) {
        float4 row = flag4(cells & ((lane4){ 1u, 2u, 4u, 8u } << i));
        __builtin_memcpy(obs + i, &row, sizeof row);
    }
    obs[CELLULAR_CELLS] = 1.0f;
}

uint32_t cellular_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 2u; }

static void cellular_draw(uint32_t seed, uint32_t step, uint32_t *extra) {
    extra[0] = step;
    extra[1] = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x & 0x8000u;
}

void cellular_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & ((1u << CELLULAR_CELLS) - 1u);
    cellular_draw(seed, 64u, extra);
}

static inline __attribute__((always_inline)) float cellular_finish(uint32_t cells, uint32_t *words, float *obs) {
    words[0] = cells;
    cellular_obs(cells, obs);
    return (float)((cells >> 1u) & 1u);
}

static __attribute__((noinline)) float cellular_flip(uint32_t seed, uint32_t step, uint32_t cells, uint32_t *words, uint32_t *extra, float *obs) {
    if (extra[0] != step) __builtin_trap();
    cells ^= extra[1];
    cellular_draw(seed, step + 64u, extra);
    return cellular_finish(cells, words, obs);
}

float cellular_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance;
    *done = 0u;
    uint32_t cells = cellular_next(words[0] ^ ((uint32_t)ctrl[0] & 1u));
    if ((step % 64u) == 0u) return cellular_flip(seed, step, cells, words, extra, obs);
    return cellular_finish(cells, words, obs);
}

void cellular_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    cellular_obs(words[0], obs);
}

void cellular_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t keep = (cellular_next(words[0]) >> 1u) & 1u, flip = (cellular_next(words[0] ^ 1u) >> 1u) & 1u;
    ctrl[0] = flip > keep ? 1.0f : 0.0f;
}
