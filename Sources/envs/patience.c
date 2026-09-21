#include "../env.h"

typedef struct { uint32_t length; float cashOut; } patienceKnobs;

static const patienceKnobs PATIENCE_KNOBS[3] = {
    { 5u, 0.1f },
    { 50u, 0.1f },
    { 500u, 0.1f }
};


static void patience_obs(const patienceKnobs *k, uint32_t position, float *obs) {
    obs[0] = (float)position / (float)k->length;
    obs[1] = 1.0f;
}

void patience_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float patience_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    const patienceKnobs *k = &PATIENCE_KNOBS[instance];
    uint32_t position = words[0];
    float reward = 0.0f;
    *done = 0u;
    if (((uint32_t)ctrl[0] & 1u) != 0u) {
        position += 1u;
        if (position >= k->length) { reward = (float)k->length; position = 0u; *done = 1u; }
    } else {
        reward = k->cashOut;
        position = 0u;
        *done = 1u;
    }
    words[0] = position;
    patience_obs(k, position, obs);
    return reward;
}

void patience_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    patience_obs(&PATIENCE_KNOBS[instance], words[0], obs);
}

void patience_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
