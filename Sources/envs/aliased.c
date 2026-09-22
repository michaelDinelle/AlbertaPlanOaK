#include "../env.h"

#define ALIASED_CLAIM_REWARD 1.0f
#define ALIASED_FALSE_CLAIM -0.1f

typedef struct { uint32_t cells; } aliasedKnobs;

static const aliasedKnobs ALIASED_KNOBS[3] = {
    { 8u },
    { 32u },
    { 128u }
};


static uint32_t aliased_colour(uint32_t seed, uint32_t cell) { return philox4x32(seed, RNG_STREAM_RESET, cell, 0u, 0u, 0u).x & 1u; }

static void aliased_obs(uint32_t seed, uint32_t position, float *obs) {
    obs[0] = (float)aliased_colour(seed, position);
    obs[1] = 1.0f;
}

static uint32_t aliased_landing(const aliasedKnobs *k, uint32_t seed, uint32_t index, uint32_t target) {
    uint32_t cell = philox4x32(seed, ENV_RNG_STREAM, index, 0u, 0u, 0u).x % k->cells;
    return cell == target ? (cell + 1u) % k->cells : cell;
}

void aliased_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const aliasedKnobs *k = &ALIASED_KNOBS[instance];
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0xfffeu, 0u, 0u, 0u).x % k->cells;
    words[0] = aliased_landing(k, seed, 0u, words[1]);
}

float aliased_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const aliasedKnobs *k = &ALIASED_KNOBS[instance];
    *done = 0u;
    float reward = 0.0f;
    if (((uint32_t)ctrl[0] & 1u) == 0u) words[0] = (words[0] + 1u) % k->cells;
    else if (words[0] == words[1]) { reward = ALIASED_CLAIM_REWARD; words[0] = aliased_landing(k, seed, step, words[1]); *done = 1u; }
    else reward = ALIASED_FALSE_CLAIM;
    aliased_obs(seed, words[0], obs);
    return reward;
}

void aliased_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)extra; (void)step;
    aliased_obs(seed, words[0], obs);
}

void aliased_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == words[1] ? 1.0f : 0.0f;
}
