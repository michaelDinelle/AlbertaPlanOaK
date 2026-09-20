#include "../env.h"

#define RECITE_SYMBOLS 4u

typedef struct { uint32_t length; } reciteKnobs;

OAK_CONSTANT reciteKnobs RECITE_KNOBS[2] = {
    { 4u },
    { 16u }
};


static uint32_t recite_symbol(uint32_t seed, uint32_t sequence, uint32_t index) { return philox4x32(seed, ENV_RNG_STREAM, sequence, index, 0u, 0u).x & 3u; }

static void recite_obs(constant const reciteKnobs *k, uint32_t seed, uint32_t sequence, uint32_t phase, device float *obs) {
    uint32_t watching = phase < k->length;
    for (uint32_t s = 0u; s < RECITE_SYMBOLS; s++) obs[s] = watching && recite_symbol(seed, sequence, phase) == s ? 1.0f : 0.0f;
    obs[RECITE_SYMBOLS] = watching ? 0.0f : 1.0f;
    obs[RECITE_SYMBOLS + 1u] = (float)(phase % k->length) / (float)k->length;
    obs[RECITE_SYMBOLS + 2u] = 1.0f;
}

void recite_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float recite_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)step; (void)extra;
    constant const reciteKnobs *k = &RECITE_KNOBS[instance];
    uint32_t sequence = words[0], phase = words[1];
    *done = 0u;
    float reward = 0.0f;
    if (phase >= k->length && (uint32_t)ctrl[0] % RECITE_SYMBOLS == recite_symbol(seed, sequence, phase - k->length)) reward = 1.0f;
    phase += 1u;
    if (phase >= 2u * k->length) { phase = 0u; sequence += 1u; *done = 1u; }
    words[0] = sequence; words[1] = phase;
    recite_obs(k, seed, sequence, phase, obs);
    return reward;
}

void recite_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)extra; (void)step;
    recite_obs(&RECITE_KNOBS[instance], seed, words[0], words[1], obs);
}

void recite_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)step; (void)extra; (void)obs;
    constant const reciteKnobs *k = &RECITE_KNOBS[instance];
    ctrl[0] = words[1] >= k->length ? (float)recite_symbol(seed, words[0], words[1] - k->length) : 0.0f;
}
