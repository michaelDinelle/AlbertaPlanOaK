#include "../env.h"

typedef struct { uint32_t length; float cashOut; } patienceKnobs;

static const patienceKnobs PATIENCE_KNOBS[3] = {
    { 5u, 0.1f },
    { 50u, 0.1f },
    { 500u, 0.1f }
};


static uint32_t patience_offset(uint32_t instance) {
    uint32_t offset = 0u;
    for (uint32_t i = 0u; i < instance; i++) offset += PATIENCE_KNOBS[i].length;
    return offset;
}

uint64_t patience_shared(float *table) {
    uint32_t instances = sizeof PATIENCE_KNOBS / sizeof PATIENCE_KNOBS[0];
    if (table)
        for (uint32_t i = 0u; i < instances; i++)
            for (uint32_t position = 0u; position < PATIENCE_KNOBS[i].length; position++) table[patience_offset(i) + position] = (float)position / (float)PATIENCE_KNOBS[i].length;
    return patience_offset(instances);
}

static void patience_obs(uint32_t instance, uint32_t position, float *obs, const float *shared) {
    obs[0] = shared[patience_offset(instance) + position];
    obs[1] = 1.0f;
}

void patience_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

static inline __attribute__((always_inline)) float patience_advance(uint32_t instance, const float *ctrl, uint32_t *words, float *obs, uint32_t *done, const float *shared) {
    const patienceKnobs *k = &PATIENCE_KNOBS[instance];
    uint32_t wait = (uint32_t)ctrl[0] & 1u, stay = words[0] < k->length - 1u ? words[0] + 1u : 0u, position = wait ? stay : 0u, end = position == 0u;
    float full = end ? (float)k->length : 0.0f, reward = wait ? full : k->cashOut;
    *done = end;
    words[0] = position;
    patience_obs(instance, position, obs, shared);
    return reward;
}

float patience_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)seed; (void)step; (void)extra;
    switch (instance) {
        case 0u: return patience_advance(0u, ctrl, words, obs, done, shared);
        case 1u: return patience_advance(1u, ctrl, words, obs, done, shared);
        default: return patience_advance(2u, ctrl, words, obs, done, shared);
    }
}

void patience_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step;
    patience_obs(instance, words[0], obs, shared);
}

void patience_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
