#include "../env.h"

typedef struct { uint32_t delay; } keydoorKnobs;

static const keydoorKnobs KEYDOOR_KNOBS[3] = {
    { 8u },
    { 64u },
    { 512u }
};


static void keydoor_obs(const keydoorKnobs *k, uint32_t phase, uint32_t hasKey, float *obs) {
    obs[0] = phase == 0u ? 1.0f : 0.0f;
    obs[1] = (float)hasKey;
    obs[2] = (float)phase / (float)k->delay;
    obs[3] = 1.0f;
}

void keydoor_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float keydoor_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const keydoorKnobs *k = &KEYDOOR_KNOBS[instance];
    uint32_t phase = words[0], hasKey = words[1];
    float reward;
    *done = 0u;
    if (phase == 0u) hasKey = (uint32_t)ctrl[0] & 1u;
    if (phase + 1u >= k->delay) {
        reward = hasKey ? 1.0f : 0.0f;
        phase = 0u;
        hasKey = 0u;
        *done = 1u;
    } else {
        reward = 0.1f * (u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) - 0.5f);
        phase += 1u;
    }
    words[0] = phase; words[1] = hasKey;
    keydoor_obs(k, phase, hasKey, obs);
    return reward;
}

void keydoor_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    keydoor_obs(&KEYDOOR_KNOBS[instance], words[0], words[1], obs);
}

void keydoor_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 0u ? 1.0f : 0.0f;
}
