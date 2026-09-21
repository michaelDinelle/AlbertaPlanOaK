#include "../env.h"

#define NEXTING_PERIOD 100u

typedef struct { uint32_t horizon; } nextingKnobs;

static const nextingKnobs NEXTING_KNOBS[2] = {
    { 1u },
    { 10u }
};


static uint32_t nexting_pulseWithin(const nextingKnobs *k, uint32_t phase) { return (phase + k->horizon) % NEXTING_PERIOD < k->horizon; }

static void nexting_obs(uint32_t seed, uint32_t step, uint32_t phase, float *obs) {
    float s, c;
    oak_sincos(2.0f * OAK_PI * (float)phase / (float)NEXTING_PERIOD, &s, &c);
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    obs[0] = s + 0.3f * normalFrom(r.x, r.y);
    obs[1] = c + 0.3f * normalFrom(r.z, r.w);
    obs[2] = phase == 0u ? 1.0f : 0.0f;
    obs[3] = 1.0f;
}

void nexting_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % NEXTING_PERIOD;
}

float nexting_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const nextingKnobs *k = &NEXTING_KNOBS[instance];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == nexting_pulseWithin(k, words[0]) ? 1.0f : 0.0f;
    words[0] = (words[0] + 1u) % NEXTING_PERIOD;
    nexting_obs(seed, step, words[0], obs);
    return reward;
}

void nexting_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    nexting_obs(seed, step, words[0], obs);
}

void nexting_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)nexting_pulseWithin(&NEXTING_KNOBS[instance], words[0]);
}
