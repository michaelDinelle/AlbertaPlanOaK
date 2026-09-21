#include "../env.h"

#define PASSWORD_BASE 4u

typedef struct { uint32_t length; } passwordKnobs;

static const passwordKnobs PASSWORD_KNOBS[3] = {
    { 2u },
    { 4u },
    { 6u }
};


static uint32_t password_digit(uint32_t seed, uint32_t index) { return philox4x32(seed, RNG_STREAM_RESET, index, 0u, 0u, 0u).x % PASSWORD_BASE; }

static void password_obs(const passwordKnobs *k, uint32_t phase, float *obs) {
    obs[0] = (float)phase / (float)k->length;
    obs[1] = 1.0f;
}

void password_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float password_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)step; (void)extra;
    const passwordKnobs *k = &PASSWORD_KNOBS[instance];
    uint32_t phase = words[0];
    float reward = 0.0f;
    *done = 0u;
    if ((uint32_t)ctrl[0] % PASSWORD_BASE == password_digit(seed, phase)) {
        phase += 1u;
        if (phase >= k->length) { reward = 1.0f; phase = 0u; *done = 1u; }
    } else {
        phase = 0u;
    }
    words[0] = phase;
    password_obs(k, phase, obs);
    return reward;
}

void password_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    password_obs(&PASSWORD_KNOBS[instance], words[0], obs);
}

void password_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)password_digit(seed, words[0]);
}
