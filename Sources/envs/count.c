#include "../env.h"

#define COUNT_MODULUS 4u

typedef struct { uint32_t period; } countKnobs;

OAK_CONSTANT countKnobs COUNT_KNOBS[2] = {
    { 8u },
    { 32u }
};


static void count_obs(uint32_t event, uint32_t query, device float *obs) {
    obs[0] = (float)event;
    obs[1] = (float)query;
    obs[2] = 1.0f;
}

void count_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float count_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const countKnobs *k = &COUNT_KNOBS[instance];
    uint32_t count = words[0], phase = words[1], event = words[2], query = words[3];
    *done = 0u;
    float reward = 0.0f;
    if (query) { reward = (uint32_t)ctrl[0] % COUNT_MODULUS == count % COUNT_MODULUS ? 1.0f : 0.0f; count = 0u; }
    if (event) count += 1u;
    phase = (phase + 1u) % k->period;
    event = u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) < 0.3f ? 1u : 0u;
    query = phase == 0u ? 1u : 0u;
    words[0] = count; words[1] = phase; words[2] = event; words[3] = query;
    count_obs(event, query, obs);
    return reward;
}

void count_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    count_obs(words[2], words[3], obs);
}

void count_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)(words[0] % COUNT_MODULUS);
}
