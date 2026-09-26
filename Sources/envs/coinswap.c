#include "../env.h"

typedef struct { float swapRate; } coinswapKnobs;

static const coinswapKnobs COINSWAP_KNOBS[2] = {
    { 0.0001f },
    { 0.01f }
};


uint32_t coinswap_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static void coinswap_draw(const coinswapKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 drawn = ((lane4)(r.x < u01Bound(0.1f)) & 1u) | ((lane4)(r.x < u01Bound(0.9f)) & 2u) | ((lane4)(r.y < u01Bound(k->swapRate)) & 4u);
    __builtin_memcpy(draws, &drawn, sizeof drawn);
}

static inline __attribute__((always_inline)) float coinswap_play(const float *ctrl, uint32_t *words, uint32_t drawn, float *obs, uint32_t *done) {
    *done = 0u;
    uint32_t match = ((uint32_t)ctrl[0] & 1u) == words[0];
    float reward = (drawn >> match) & 1u ? 1.0f : 0.0f;
    words[0] ^= (drawn >> 2u) & 1u;
    obs[0] = 1.0f;
    return reward;
}

static __attribute__((noinline)) float coinswap_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    coinswap_draw(&COINSWAP_KNOBS[instance], seed, step, extra + 1);
    return coinswap_play(ctrl, words, extra[1], obs, done);
}

void coinswap_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
    batchReset(&extra[0]);
}

float coinswap_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return coinswap_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return coinswap_play(ctrl, words, extra[1u + slot], obs, done);
}

void coinswap_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void coinswap_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
