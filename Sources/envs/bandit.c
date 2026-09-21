#include "../env.h"

typedef struct { uint32_t arms; float drift; } banditKnobs;

static const banditKnobs BANDIT_KNOBS[4] = {
    { 3u, 0.0f },
    { 6u, 0.0f },
    { 3u, 0.001f },
    { 6u, 0.001f }
};


void bandit_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    for (uint32_t i = 0u; i < k->arms; i++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, i, 0u, 0u, 0u);
        words[i] = floatWord(normalFrom(r.x, r.y));
    }
}

float bandit_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    *done = 0u;
    uint32_t arm = (uint32_t)ctrl[0] % k->arms;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float reward = wordFloat(words[arm]) + 0.5f * normalFrom(r.x, r.y);
    for (uint32_t i = 0u; i < k->arms; i++) {
        u4 d = philox4x32(seed, ENV_RNG_STREAM, step, 1u + i, 0u, 0u);
        words[i] = floatWord(wordFloat(words[i]) + k->drift * normalFrom(d.x, d.y));
    }
    obs[0] = 1.0f;
    return reward;
}

void bandit_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void bandit_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    uint32_t best = 0u;
    for (uint32_t i = 1u; i < k->arms; i++) if (wordFloat(words[i]) > wordFloat(words[best])) best = i;
    ctrl[0] = (float)best;
}
