#include "../env.h"

typedef struct { float eta, innovation; } trackingKnobs;

static const trackingKnobs TRACKING_KNOBS[2] = {
    { 0.99f, 0.14106736f },
    { 0.999f, 0.044710178f }
};


void tracking_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float tracking_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const trackingKnobs *k = &TRACKING_KNOBS[instance];
    *done = 0u;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float next = k->eta * wordFloat(words[0]) + k->innovation * normalFrom(r.x, r.y);
    float miss = 3.0f * ctrl[0] - next;
    words[0] = floatWord(next);
    obs[0] = next + 0.5f * normalFrom(r.z, r.w);
    obs[1] = 1.0f;
    return -miss * miss;
}

void tracking_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    obs[0] = wordFloat(words[0]) + 0.5f * normalFrom(r.z, r.w);
    obs[1] = 1.0f;
}

void tracking_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = oak_clamp((&TRACKING_KNOBS[instance])->eta * wordFloat(words[0]) / 3.0f, -1.0f, 1.0f);
}
