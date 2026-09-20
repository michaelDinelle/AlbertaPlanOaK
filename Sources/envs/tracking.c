#include "../env.h"

typedef struct { float eta, innovation; } trackingKnobs;

OAK_CONSTANT trackingKnobs TRACKING_KNOBS[2] = {
    { 0.99f, 0.14106736f },
    { 0.999f, 0.044710178f }
};


void tracking_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float tracking_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const trackingKnobs *k = &TRACKING_KNOBS[instance];
    *done = 0u;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float next = k->eta * wordFloat(words[0]) + k->innovation * normalFrom(r.x, r.y);
    float miss = 3.0f * ctrl[0] - next;
    words[0] = floatWord(next);
    obs[0] = next + 0.5f * normalFrom(r.z, r.w);
    obs[1] = 1.0f;
    return -miss * miss;
}

void tracking_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    obs[0] = wordFloat(words[0]) + 0.5f * normalFrom(r.z, r.w);
    obs[1] = 1.0f;
}

void tracking_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = oak_clamp((&TRACKING_KNOBS[instance])->eta * wordFloat(words[0]) / 3.0f, -1.0f, 1.0f);
}
