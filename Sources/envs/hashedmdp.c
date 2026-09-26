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


static inline __attribute__((always_inline)) uint32_t hashed_next(const hashedKnobs *k, uint32_t seed, uint32_t state, uint32_t action) {
    u4 r = philox4x32(seed, RNG_STREAM_RESET, state, action, 1u, 0u);
    return k->local ? state ^ (1u << (r.x % k->bits)) : r.x & ((1u << k->bits) - 1u);
}

static inline __attribute__((always_inline)) float hashed_reward(uint32_t seed, uint32_t state, uint32_t action) {
    u4 r = philox4x32(seed, RNG_STREAM_RESET, state, action, 2u, 0u);
    return (r.x % HASHED_REWARD_ODDS) == 0u ? normalFrom(r.y, r.z) : 0.0f;
}

static void hashed_obs(const hashedKnobs *k, uint32_t state, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b += 4u) {
        float4 bit = flag4((lane4)state & ((lane4){ 1u, 2u, 4u, 8u } << b));
        __builtin_memcpy(obs + b, &bit, sizeof bit);
    }
    obs[k->bits] = 1.0f;
}

static uint32_t hashed_pairs(const hashedKnobs *k) { return (1u << k->bits) * HASHED_ACTIONS; }

static int hashed_tabled(const hashedKnobs *k) { return k->bits <= 8u; }

static uint32_t hashed_tableWords(const hashedKnobs *k) { return hashed_tabled(k) ? hashed_pairs(k) / 4u + hashed_pairs(k) : 0u; }

static uint8_t *hashed_successors(uint32_t *extra) { return (uint8_t *)extra; }

static float *hashed_gains(const hashedKnobs *k, uint32_t *extra) { return (float *)(extra + hashed_pairs(k) / 4u); }

static uint32_t *hashed_solved(const hashedKnobs *k, uint32_t *extra) { return extra + hashed_tableWords(k); }

static uint32_t *hashed_policy(const hashedKnobs *k, uint32_t *extra) { return hashed_solved(k, extra) + 1u; }

uint32_t hashed_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const hashedKnobs *k = &HASHED_KNOBS[instance];
    return hashed_tableWords(k) + 1u + (1u << k->bits) / 16u;
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
            const uint32_t *to = next + (size_t)s * HASHED_ACTIONS;
            const float *gain = reward + (size_t)s * HASHED_ACTIONS;
            double best = -DBL_MAX;
            for (uint32_t a = 0u; a < HASHED_ACTIONS; a++) {
                double q = (double)gain[a] + HASHED_DISCOUNT * value[to[a]];
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
    *hashed_solved(k, extra) = 0u;
    if (!hashed_tabled(k)) return;
    float *gain = hashed_gains(k, extra);
    uint8_t *successor = hashed_successors(extra);
    for (uint32_t pair = 0u; pair < hashed_pairs(k); pair++) {
        gain[pair] = hashed_reward(seed, pair / HASHED_ACTIONS, pair % HASHED_ACTIONS);
        successor[pair] = (uint8_t)hashed_next(k, seed, pair / HASHED_ACTIONS, pair % HASHED_ACTIONS);
    }
}

static inline __attribute__((always_inline)) float hashed_move(const hashedKnobs *k, uint32_t seed, uint32_t action, uint32_t *words, uint32_t *extra, float *obs) {
    uint32_t state = words[0], pair = state * HASHED_ACTIONS + action;
    float reward = hashed_tabled(k) ? hashed_gains(k, extra)[pair] : hashed_reward(seed, state, action);
    state = hashed_tabled(k) ? hashed_successors(extra)[pair] : hashed_next(k, seed, state, action);
    words[0] = state;
    hashed_obs(k, state, obs);
    return reward;
}

float hashed_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)step;
    *done = 0u;
    uint32_t action = (uint32_t)ctrl[0] % HASHED_ACTIONS;
    switch (instance) {
        case 0u: return hashed_move(&HASHED_KNOBS[0], seed, action, words, extra, obs);
        case 1u: return hashed_move(&HASHED_KNOBS[1], seed, action, words, extra, obs);
        case 2u: return hashed_move(&HASHED_KNOBS[2], seed, action, words, extra, obs);
        default: return hashed_move(&HASHED_KNOBS[3], seed, action, words, extra, obs);
    }
}

void hashed_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    hashed_obs(&HASHED_KNOBS[instance], words[0], obs);
}

void hashed_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)step; (void)obs;
    const hashedKnobs *k = &HASHED_KNOBS[instance];
    if (*hashed_solved(k, extra) == 0u) { hashed_solve(k, seed, hashed_policy(k, extra)); *hashed_solved(k, extra) = 1u; }
    ctrl[0] = (float)hashed_policyAt(hashed_policy(k, extra), words[0]);
}
