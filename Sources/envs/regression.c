#include "../env.h"

#define REGRESSION_BINS 5u

typedef struct { uint32_t inputs; float drift; } regressionKnobs;

static const regressionKnobs REGRESSION_KNOBS[2] = {
    { 8u, 0.001f },
    { 16u, 0.001f }
};

static const float REGRESSION_EDGE[REGRESSION_BINS - 1u] = { -1.0f, -0.3f, 0.3f, 1.0f };

static void regression_input(const regressionKnobs *k, uint32_t seed, uint32_t index, float *obs) {
    for (uint32_t j = 0u; j < k->inputs; j += 2u) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 256u + j, 0u, 0u);
        obs[j] = normalFrom(r.x, r.y);
        if (j + 1u < k->inputs) obs[j + 1u] = normalFrom(r.z, r.w);
    }
    obs[k->inputs] = 1.0f;
}

static uint32_t regression_bin(const regressionKnobs *k, const uint32_t *words, const float *x) {
    float y = 0.0f;
    for (uint32_t j = 0u; j < k->inputs; j++) y = fmaf(wordFloat(words[j]), x[j], y);
    uint32_t bin = 0u;
    while (bin < REGRESSION_BINS - 1u && y > REGRESSION_EDGE[bin]) bin++;
    return bin;
}

void regression_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const regressionKnobs *k = &REGRESSION_KNOBS[instance];
    float scale = oak_rsqrt((float)k->inputs);
    for (uint32_t j = 0u; j < k->inputs; j++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, j, 0u, 0u, 0u);
        words[j] = floatWord(scale * normalFrom(r.x, r.y));
    }
}

float regression_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const regressionKnobs *k = &REGRESSION_KNOBS[instance];
    *done = 0u;
    float reward = (uint32_t)ctrl[0] % REGRESSION_BINS == regression_bin(k, words, obs) ? 1.0f : 0.0f;
    for (uint32_t j = 0u; j < k->inputs; j += 2u) {
        u4 d = philox4x32(seed, ENV_RNG_STREAM, step, 1u + j, 0u, 0u);
        words[j] = floatWord(wordFloat(words[j]) + k->drift * normalFrom(d.x, d.y));
        if (j + 1u < k->inputs) words[j + 1u] = floatWord(wordFloat(words[j + 1u]) + k->drift * normalFrom(d.z, d.w));
    }
    regression_input(k, seed, step, obs);
    return reward;
}

void regression_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)words; (void)extra;
    regression_input(&REGRESSION_KNOBS[instance], seed, step, obs);
}

void regression_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    ctrl[0] = (float)regression_bin(&REGRESSION_KNOBS[instance], words, obs);
}
