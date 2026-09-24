#include "../env.h"

typedef struct { float switchRate; } hiddenmodeKnobs;

static const hiddenmodeKnobs HIDDENMODE_KNOBS[2] = {
    { 0.001f },
    { 0.01f }
};


uint32_t hiddenmode_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static void hiddenmode_draw(const hiddenmodeKnobs *k, uint32_t seed, uint32_t step, uint32_t *flips) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 flip = (lane4)(r.x < u01Bound(k->switchRate)) & 1u;
    __builtin_memcpy(flips, &flip, sizeof flip);
}

static inline __attribute__((always_inline)) float hiddenmode_play(const float *ctrl, uint32_t *words, uint32_t flip, float *obs, uint32_t *done) {
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == words[0] ? 1.0f : 0.0f;
    words[0] ^= flip;
    obs[0] = 1.0f;
    return reward;
}

static __attribute__((noinline)) float hiddenmode_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    hiddenmode_draw(&HIDDENMODE_KNOBS[instance], seed, step, extra + 1);
    return hiddenmode_play(ctrl, words, extra[1], obs, done);
}

void hiddenmode_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
    batchReset(&extra[0]);
}

float hiddenmode_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return hiddenmode_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return hiddenmode_play(ctrl, words, extra[1u + slot], obs, done);
}

void hiddenmode_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void hiddenmode_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
