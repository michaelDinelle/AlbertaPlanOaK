#include "../env.h"

#define PASSWORD_BASE 4u

typedef struct { uint32_t length; } passwordKnobs;

static const passwordKnobs PASSWORD_KNOBS[3] = {
    { 2u },
    { 4u },
    { 6u }
};


static uint32_t password_offset(uint32_t instance) {
    uint32_t offset = 0u;
    for (uint32_t i = 0u; i < instance; i++) offset += PASSWORD_KNOBS[i].length;
    return offset;
}

uint64_t password_shared(float *table) {
    uint32_t instances = sizeof PASSWORD_KNOBS / sizeof PASSWORD_KNOBS[0];
    if (table)
        for (uint32_t i = 0u; i < instances; i++)
            for (uint32_t phase = 0u; phase < PASSWORD_KNOBS[i].length; phase++) table[password_offset(i) + phase] = (float)phase / (float)PASSWORD_KNOBS[i].length;
    return password_offset(instances);
}

uint32_t password_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return PASSWORD_BASE; }

static uint32_t password_digit(const uint32_t *extra, uint32_t index) {
    uint32_t digit = 0u;
    for (uint32_t d = 0u; d < PASSWORD_BASE; d++) digit = (extra[d] >> index) & 1u ? d : digit;
    return digit;
}

static void password_obs(uint32_t instance, uint32_t phase, float *obs, const float *shared) {
    obs[0] = shared[password_offset(instance) + phase];
    obs[1] = 1.0f;
}

static inline __attribute__((always_inline)) float password_advance(uint32_t instance, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    const passwordKnobs *k = &PASSWORD_KNOBS[instance];
    uint32_t phase = words[0], matches = extra[(uint32_t)ctrl[0] % PASSWORD_BASE], last = k->length - 1u;
    uint32_t open = phase == last && (matches >> last) & 1u;
    phase = (matches & ((1u << last) - 1u)) >> phase & 1u ? phase + 1u : 0u;
    *done = open;
    words[0] = phase;
    password_obs(instance, phase, obs, shared);
    return open ? 1.0f : 0.0f;
}

void password_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)words;
    const passwordKnobs *k = &PASSWORD_KNOBS[instance];
    for (uint32_t d = 0u; d < PASSWORD_BASE; d++) extra[d] = 0u;
    for (uint32_t index = 0u; index < k->length; index++) extra[philox4x32(seed, RNG_STREAM_RESET, index, 0u, 0u, 0u).x % PASSWORD_BASE] |= 1u << index;
}

float password_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)seed; (void)step;
    switch (instance) {
        case 0u: return password_advance(0u, ctrl, words, extra, obs, done, shared);
        case 1u: return password_advance(1u, ctrl, words, extra, obs, done, shared);
        default: return password_advance(2u, ctrl, words, extra, obs, done, shared);
    }
}

void password_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step;
    password_obs(instance, words[0], obs, shared);
}

void password_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    ctrl[0] = (float)password_digit(extra, words[0]);
}

uint32_t password_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    uint32_t count = PASSWORD_KNOBS[instance].length;
    if (!out) return count;
    for (uint32_t index = 0u; index < count; index++) out[index] = password_digit(extra, index);
    return count;
}
