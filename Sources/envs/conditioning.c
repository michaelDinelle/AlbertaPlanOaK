#include "../env.h"

typedef struct { uint32_t interval, pattern; } conditioningKnobs;

OAK_CONSTANT conditioningKnobs CONDITIONING_KNOBS[3] = {
    { 10u, 0u },
    { 100u, 0u },
    { 10u, 1u }
};


static uint32_t conditioning_predicts(constant const conditioningKnobs *k, uint32_t cues) { return k->pattern ? (cues & 3u) == 3u && (cues & 4u) == 0u : cues != 0u; }

static void conditioning_obs(constant const conditioningKnobs *k, uint32_t cues, uint32_t countdown, device float *obs) {
    obs[0] = (float)(cues & 1u);
    obs[1] = k->pattern ? (float)((cues >> 1u) & 1u) : 0.0f;
    obs[2] = k->pattern ? (float)((cues >> 2u) & 1u) : 0.0f;
    obs[3] = countdown == 1u ? 1.0f : 0.0f;
    obs[4] = 1.0f;
}

void conditioning_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float conditioning_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const conditioningKnobs *k = &CONDITIONING_KNOBS[instance];
    uint32_t countdown = words[0];
    *done = 0u;
    uint32_t blink = (uint32_t)ctrl[0] & 1u;
    float reward = countdown == 1u ? (blink ? 1.0f : 0.0f) : (blink ? -0.05f : 0.0f);
    if (countdown > 0u) countdown -= 1u;
    uint32_t cues = 0u;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    if (countdown == 0u && u01(r.x) < 1.0f / (float)(2u * k->interval)) {
        cues = k->pattern ? 1u + (r.y % 7u) : 1u;
        if (conditioning_predicts(k, cues)) countdown = k->interval + 1u;
    }
    words[0] = countdown; words[1] = cues;
    conditioning_obs(k, cues, countdown, obs);
    return reward;
}

void conditioning_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    conditioning_obs(&CONDITIONING_KNOBS[instance], words[1], words[0], obs);
}

void conditioning_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 1u ? 1.0f : 0.0f;
}
