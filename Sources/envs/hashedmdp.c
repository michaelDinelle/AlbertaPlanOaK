#include "../env.h"

#define HASHED_ACTIONS 4u

typedef struct { uint32_t bits, local; } hashedKnobs;

static const hashedKnobs HASHED_KNOBS[4] = {
    { 8u, 1u },
    { 16u, 1u },
    { 8u, 0u },
    { 16u, 0u }
};


static uint32_t hashed_next(const hashedKnobs *k, uint32_t seed, uint32_t state, uint32_t action) {
    u4 r = philox4x32(seed, RNG_STREAM_RESET, state, action, 1u, 0u);
    return k->local ? state ^ (1u << (r.x % k->bits)) : r.x & ((1u << k->bits) - 1u);
}

static float hashed_reward(uint32_t seed, uint32_t state, uint32_t action) {
    u4 r = philox4x32(seed, RNG_STREAM_RESET, state, action, 2u, 0u);
    return (r.x % 20u) == 0u ? normalFrom(r.y, r.z) : 0.0f;
}

static void hashed_obs(const hashedKnobs *k, uint32_t state, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b++) obs[b] = (float)((state >> b) & 1u);
    obs[k->bits] = 1.0f;
}

void hashed_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0xffffu, 0u, 0u, 0u).x & ((1u << (&HASHED_KNOBS[instance])->bits) - 1u);
}

float hashed_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)step; (void)extra;
    const hashedKnobs *k = &HASHED_KNOBS[instance];
    *done = 0u;
    uint32_t action = (uint32_t)ctrl[0] % HASHED_ACTIONS;
    float reward = hashed_reward(seed, words[0], action);
    words[0] = hashed_next(k, seed, words[0], action);
    hashed_obs(k, words[0], obs);
    return reward;
}

void hashed_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    hashed_obs(&HASHED_KNOBS[instance], words[0], obs);
}

void hashed_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)step; (void)extra; (void)obs;
    const hashedKnobs *k = &HASHED_KNOBS[instance];
    uint32_t best = 0u;
    float bestValue = -FLT_MAX;
    for (uint32_t a = 0u; a < HASHED_ACTIONS; a++) {
        uint32_t next = hashed_next(k, seed, words[0], a);
        float future = -FLT_MAX;
        for (uint32_t b = 0u; b < HASHED_ACTIONS; b++) { float r = hashed_reward(seed, next, b); if (r > future) future = r; }
        float value = hashed_reward(seed, words[0], a) + future;
        if (value > bestValue) { bestValue = value; best = a; }
    }
    ctrl[0] = (float)best;
}
