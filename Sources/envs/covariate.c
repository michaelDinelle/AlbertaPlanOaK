#include "../env.h"

#define COVARIATE_INPUTS 4u
#define COVARIATE_PERIOD 100000u

typedef struct { float shift; } covariateKnobs;

static const covariateKnobs COVARIATE_KNOBS[2] = {
    { 0.0f },
    { 2.0f }
};


static void covariate_input(const covariateKnobs *k, uint32_t seed, uint32_t index, float *obs) {
    float offset = ((index / COVARIATE_PERIOD) & 1u) ? k->shift : -k->shift;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 1u, 0u, 0u);
    u4 s = philox4x32(seed, ENV_RNG_STREAM, index, 2u, 0u, 0u);
    obs[0] = normalFrom(r.x, r.y) + offset;
    obs[1] = normalFrom(r.z, r.w);
    obs[2] = normalFrom(s.x, s.y) - offset;
    obs[3] = normalFrom(s.z, s.w);
    obs[COVARIATE_INPUTS] = 1.0f;
}

static uint32_t covariate_sign(const uint32_t *words, const float *x) {
    float y = wordFloat(words[COVARIATE_INPUTS]);
    for (uint32_t j = 0u; j < COVARIATE_INPUTS; j++) y = fmaf(wordFloat(words[j]), x[j], y);
    return y > 0.0f ? 1u : 0u;
}

void covariate_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    for (uint32_t j = 0u; j <= COVARIATE_INPUTS; j++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, j, 0u, 0u, 0u);
        words[j] = floatWord(normalFrom(r.x, r.y));
    }
}

float covariate_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const covariateKnobs *k = &COVARIATE_KNOBS[instance];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == covariate_sign(words, obs) ? 1.0f : 0.0f;
    covariate_input(k, seed, step, obs);
    return reward;
}

void covariate_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)words; (void)extra;
    covariate_input(&COVARIATE_KNOBS[instance], seed, step, obs);
}

void covariate_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra;
    ctrl[0] = (float)covariate_sign(words, obs);
}
