#include "../env.h"

#define PERMUTED_STATES 8u
#define PERMUTED_ACTIONS 4u

typedef struct { uint32_t recodePeriod; } permutedKnobs;

OAK_CONSTANT permutedKnobs PERMUTED_KNOBS[2] = {
    { 0u },
    { 100000u }
};


static uint32_t permuted_epoch(constant const permutedKnobs *k, uint32_t step) { return k->recodePeriod == 0u ? 0u : step / k->recodePeriod; }

static uint32_t permuted_code(uint32_t seed, uint32_t epoch, uint32_t state) {
    uint32_t order[PERMUTED_STATES];
    for (uint32_t i = 0u; i < PERMUTED_STATES; i++) order[i] = i;
    for (uint32_t i = PERMUTED_STATES - 1u; i > 0u; i--) {
        uint32_t j = philox4x32(seed, RNG_STREAM_RESET, epoch, i, 0u, 0u).x % (i + 1u);
        uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    return order[state];
}

static uint32_t permuted_lookup(uint32_t seed, uint32_t state) { return philox4x32(seed, RNG_STREAM_RESET, 0xffffu, state, 0u, 0u).x % PERMUTED_ACTIONS; }

static void permuted_obs(constant const permutedKnobs *k, uint32_t seed, uint32_t step, uint32_t state, device float *obs) {
    uint32_t code = permuted_code(seed, permuted_epoch(k, step), state);
    for (uint32_t i = 0u; i < PERMUTED_STATES; i++) obs[i] = i == code ? 1.0f : 0.0f;
    obs[PERMUTED_STATES] = 1.0f;
}

void permuted_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, ENV_RNG_STREAM, 0u, 0u, 0u, 0u).x % PERMUTED_STATES;
}

float permuted_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    *done = 0u;
    float reward = (uint32_t)ctrl[0] % PERMUTED_ACTIONS == permuted_lookup(seed, words[0]) ? 1.0f : 0.0f;
    words[0] = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x % PERMUTED_STATES;
    permuted_obs(&PERMUTED_KNOBS[instance], seed, step, words[0], obs);
    return reward;
}

void permuted_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)extra;
    permuted_obs(&PERMUTED_KNOBS[instance], seed, step, words[0], obs);
}

void permuted_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)permuted_lookup(seed, words[0]);
}
