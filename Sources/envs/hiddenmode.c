#include "../env.h"

typedef struct { float switchRate; } hiddenmodeKnobs;

static const hiddenmodeKnobs HIDDENMODE_KNOBS[2] = {
    { 0.001f },
    { 0.01f }
};


void hiddenmode_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
}

float hiddenmode_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const hiddenmodeKnobs *k = &HIDDENMODE_KNOBS[instance];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == words[0] ? 1.0f : 0.0f;
    if (u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) < k->switchRate) words[0] ^= 1u;
    obs[0] = 1.0f;
    return reward;
}

void hiddenmode_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void hiddenmode_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
