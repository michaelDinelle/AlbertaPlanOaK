#include "../env.h"

#define HASHED_ACTIONS 4u
#define HASHED_REWARD_ODDS 20u
#define HASHED_DISCOUNT 0.999
#define HASHED_SWEEPS 2000u
#define HASHED_TOLERANCE 1e-9

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
    return (r.x % HASHED_REWARD_ODDS) == 0u ? normalFrom(r.y, r.z) : 0.0f;
}

static void hashed_obs(const hashedKnobs *k, uint32_t state, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b++) obs[b] = (float)((state >> b) & 1u);
    obs[k->bits] = 1.0f;
}

uint32_t hashed_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return (1u << HASHED_KNOBS[instance].bits) / 16u;
}

static uint32_t hashed_policyAt(const uint32_t *policy, uint32_t state) { return (policy[state >> 4u] >> (2u * (state & 15u))) & 3u; }

static void hashed_solve(const hashedKnobs *k, uint32_t seed, uint32_t *policy) {
    uint32_t states = 1u << k->bits, pairs = states * HASHED_ACTIONS;
    uint32_t *next = malloc(pairs * sizeof *next);
    float *reward = malloc(pairs * sizeof *reward);
    double *value = calloc(states, sizeof *value);
    for (uint32_t s = 0u; s < states; s++)
        for (uint32_t a = 0u; a < HASHED_ACTIONS; a++) { next[s * HASHED_ACTIONS + a] = hashed_next(k, seed, s, a); reward[s * HASHED_ACTIONS + a] = hashed_reward(seed, s, a); }
    for (uint32_t sweep = 0u; sweep < HASHED_SWEEPS; sweep++) {
        double delta = 0.0;
        for (uint32_t s = 0u; s < states; s++) {
            double best = -DBL_MAX;
            for (uint32_t a = 0u; a < HASHED_ACTIONS; a++) {
                double q = (double)reward[s * HASHED_ACTIONS + a] + HASHED_DISCOUNT * value[next[s * HASHED_ACTIONS + a]];
                if (q > best) best = q;
            }
            double change = fabs(best - value[s]);
            if (change > delta) delta = change;
            value[s] = best;
        }
        if (delta < HASHED_TOLERANCE) break;
    }
    for (uint32_t w = 0u; w < states / 16u; w++) policy[w] = 0u;
    for (uint32_t s = 0u; s < states; s++) {
        uint32_t best = 0u;
        double bestQ = -DBL_MAX;
        for (uint32_t a = 0u; a < HASHED_ACTIONS; a++) {
            double q = (double)reward[s * HASHED_ACTIONS + a] + HASHED_DISCOUNT * value[next[s * HASHED_ACTIONS + a]];
            if (q > bestQ) { bestQ = q; best = a; }
        }
        policy[s >> 4u] |= best << (2u * (s & 15u));
    }
    free(next); free(reward); free(value);
}

void hashed_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const hashedKnobs *k = &HASHED_KNOBS[instance];
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0xffffu, 0u, 0u, 0u).x & ((1u << k->bits) - 1u);
    hashed_solve(k, seed, extra);
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
    (void)instance; (void)seed; (void)step; (void)obs;
    ctrl[0] = (float)hashed_policyAt(extra, words[0]);
}
