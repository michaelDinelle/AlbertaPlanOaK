#include "../env.h"

#define COVARIATE_INPUTS 4u
#define COVARIATE_PERIOD 100000u

typedef struct { float shift; } covariateKnobs;

static const covariateKnobs COVARIATE_KNOBS[2] = {
    { 0.0f },
    { 2.0f }
};

enum { COVARIATE_TAG = 0, COVARIATE_TARGETS = 1, COVARIATE_DRAWS = COVARIATE_TARGETS + ENV_BATCH, COVARIATE_EXTRA = COVARIATE_DRAWS + COVARIATE_INPUTS * ENV_BATCH };

static inline __attribute__((always_inline)) void covariate_fill(const covariateKnobs *k, uint32_t seed, uint32_t first, const uint32_t *words, uint32_t *extra) {
    lane4 index = first + (lane4){ 0u, 1u, 2u, 3u };
    u44 r = philox4(seed, ENV_RNG_STREAM, index, 1u, 0u, 0u), s = philox4(seed, ENV_RNG_STREAM, index, 2u, 0u, 0u);
    float4 offset = select4((lane4)(((index / COVARIATE_PERIOD) & 1u) != 0u), splat4(k->shift), splat4(-k->shift));
    float32x4x4_t x = { { normal4(r.x, r.y) + offset, normal4(r.z, r.w), normal4(s.x, s.y) - offset, normal4(s.z, s.w) } };
    float4 y = splat4(wordFloat(words[COVARIATE_INPUTS]));
    for (uint32_t j = 0u; j < COVARIATE_INPUTS; j++) y = __builtin_elementwise_fma(splat4(wordFloat(words[j])), (float4)x.val[j], y);
    lane4 targets = (lane4)(y > 0.0f) & 1u;
    __builtin_memcpy(extra + COVARIATE_TARGETS, &targets, sizeof targets);
    vst4q_f32((float *)extra + COVARIATE_DRAWS, x);
}

static void covariate_show(uint32_t slot, const uint32_t *extra, float *obs) {
    __builtin_memcpy(obs, extra + COVARIATE_DRAWS + COVARIATE_INPUTS * slot, COVARIATE_INPUTS * sizeof(float));
    obs[COVARIATE_INPUTS] = 1.0f;
}

static float covariate_reward(uint32_t slot, const float *ctrl, const uint32_t *extra) {
    return ((uint32_t)ctrl[0] & 1u) == extra[COVARIATE_TARGETS + slot - 1u] ? 1.0f : 0.0f;
}

static __attribute__((noinline)) float covariate_refill(const covariateKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, const uint32_t *words, uint32_t *extra, float *obs) {
    float reward = covariate_reward(ENV_BATCH, ctrl, extra);
    covariate_fill(k, seed, step, words, extra);
    covariate_show(0u, extra, obs);
    return reward;
}

uint32_t covariate_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return COVARIATE_EXTRA;
}

void covariate_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    for (uint32_t j = 0u; j <= COVARIATE_INPUTS; j++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, j, 0u, 0u, 0u);
        words[j] = floatWord(normalFrom(r.x, r.y));
    }
    batchReset(extra + COVARIATE_TAG);
    covariate_fill(&COVARIATE_KNOBS[instance], seed, extra[COVARIATE_TAG], words, extra);
}

float covariate_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    uint32_t slot = batchSlot(extra + COVARIATE_TAG, step);
    if (slot == ENV_BATCH) return covariate_refill(&COVARIATE_KNOBS[instance], seed, step, ctrl, words, extra, obs);
    covariate_show(slot, extra, obs);
    return covariate_reward(slot, ctrl, extra);
}

void covariate_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words;
    uint32_t slot = step - extra[COVARIATE_TAG];
    if (slot >= ENV_BATCH) __builtin_trap();
    covariate_show(slot, extra, obs);
}

void covariate_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)obs;
    uint32_t slot = step - extra[COVARIATE_TAG];
    if (slot >= ENV_BATCH) __builtin_trap();
    ctrl[0] = (float)extra[COVARIATE_TARGETS + slot];
}
