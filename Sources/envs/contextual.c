#include "../env.h"

#define CONTEXTUAL_ARMS 4u

typedef struct { uint32_t inputs; float drift; } contextualKnobs;

OAK_CONSTANT contextualKnobs CONTEXTUAL_KNOBS[4] = {
    { 8u, 0.0f },
    { 16u, 0.0f },
    { 8u, 0.001f },
    { 16u, 0.001f }
};


static void contextual_context(constant const contextualKnobs *k, uint32_t seed, uint32_t index, device float *obs) {
    for (uint32_t j = 0u; j < k->inputs; j += 2u) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 256u + j, 0u, 0u);
        obs[j] = normalFrom(r.x, r.y);
        if (j + 1u < k->inputs) obs[j + 1u] = normalFrom(r.z, r.w);
    }
    obs[k->inputs] = 1.0f;
}

uint32_t contextual_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return CONTEXTUAL_ARMS * CONTEXTUAL_KNOBS[instance].inputs;
}

void contextual_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)words;
    constant const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    device float *weights = (device float *)extra;
    float scale = oak_rsqrt((float)k->inputs);
    for (uint32_t i = 0u; i < CONTEXTUAL_ARMS * k->inputs; i++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, i, 0u, 0u, 0u);
        weights[i] = scale * normalFrom(r.x, r.y);
    }
}

float contextual_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)words;
    constant const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    device float *weights = (device float *)extra;
    *done = 0u;
    uint32_t arm = (uint32_t)ctrl[0] % CONTEXTUAL_ARMS;
    float value = 0.0f;
    for (uint32_t j = 0u; j < k->inputs; j++) value = fmaf(weights[arm * k->inputs + j], obs[j], value);
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float reward = value + 0.1f * normalFrom(r.x, r.y);
    if (k->drift != 0.0f)
        for (uint32_t i = 0u; i < CONTEXTUAL_ARMS * k->inputs; i += 2u) {
            u4 d = philox4x32(seed, ENV_RNG_STREAM, step, 1u + i, 0u, 0u);
            weights[i] += k->drift * normalFrom(d.x, d.y);
            if (i + 1u < CONTEXTUAL_ARMS * k->inputs) weights[i + 1u] += k->drift * normalFrom(d.z, d.w);
        }
    contextual_context(k, seed, step, obs);
    return reward;
}

void contextual_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)words; (void)extra;
    contextual_context(&CONTEXTUAL_KNOBS[instance], seed, step, obs);
}

void contextual_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)words;
    constant const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    device const float *weights = (device const float *)extra;
    uint32_t best = 0u;
    float bestValue = -FLT_MAX;
    for (uint32_t arm = 0u; arm < CONTEXTUAL_ARMS; arm++) {
        float value = 0.0f;
        for (uint32_t j = 0u; j < k->inputs; j++) value = fmaf(weights[arm * k->inputs + j], obs[j], value);
        if (value > bestValue) { bestValue = value; best = arm; }
    }
    ctrl[0] = (float)best;
}
