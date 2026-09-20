#include "../env.h"

typedef struct { float cueRate; } operantKnobs;

OAK_CONSTANT operantKnobs OPERANT_KNOBS[2] = {
    { 1.0f / 16.0f },
    { 1.0f / 64.0f }
};


void operant_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float operant_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const operantKnobs *k = &OPERANT_KNOBS[instance];
    *done = 0u;
    uint32_t press = (uint32_t)ctrl[0] & 1u;
    float reward = words[0] != 0u ? (press ? 1.0f : 0.0f) : (press ? -0.1f : 0.0f);
    words[0] = u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) < k->cueRate ? 1u : 0u;
    obs[0] = (float)words[0];
    obs[1] = 1.0f;
    return reward;
}

void operant_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    obs[0] = (float)words[0];
    obs[1] = 1.0f;
}

void operant_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
