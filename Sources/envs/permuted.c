#include "../env.h"

#define PERMUTED_STATES 8u
#define PERMUTED_ACTIONS 4u

typedef struct { uint32_t recodePeriod; } permutedKnobs;

static const permutedKnobs PERMUTED_KNOBS[2] = {
    { 0u },
    { 100000u }
};

enum { PERMUTED_ROWS = 1, PERMUTED_ROW = 16, PERMUTED_LOOKUP = PERMUTED_ROWS + PERMUTED_STATES * PERMUTED_ROW, PERMUTED_DRAWS = PERMUTED_LOOKUP + PERMUTED_STATES };

static uint32_t permuted_epoch(const permutedKnobs *k, uint32_t step) { return k->recodePeriod == 0u ? 0u : step / k->recodePeriod; }

static void permuted_code(uint32_t seed, uint32_t epoch, uint32_t *extra) {
    uint32_t order[PERMUTED_STATES];
    for (uint32_t i = 0u; i < PERMUTED_STATES; i++) order[i] = i;
    for (uint32_t i = PERMUTED_STATES - 1u; i > 0u; i--) {
        uint32_t j = philox4x32(seed, RNG_STREAM_RESET, epoch, i, 0u, 0u).x % (i + 1u);
        uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (uint32_t state = 0u; state < PERMUTED_STATES; state++) {
        float row[PERMUTED_ROW] = { 0.0f };
        for (uint32_t i = 0u; i < PERMUTED_STATES; i++) row[i] = i == order[state] ? 1.0f : 0.0f;
        row[PERMUTED_STATES] = 1.0f;
        __builtin_memcpy(extra + PERMUTED_ROWS + state * PERMUTED_ROW, row, sizeof row);
    }
    extra[0] = epoch;
}

static uint32_t permuted_lookup(uint32_t seed, uint32_t state) { return philox4x32(seed, RNG_STREAM_RESET, 0xffffu, state, 0u, 0u).x % PERMUTED_ACTIONS; }

static lane4 permuted_draw(uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 state = r.x % PERMUTED_STATES;
    __builtin_memcpy(draws + 1, &state, sizeof state);
    return state;
}

static void permuted_obs(const uint32_t *extra, uint32_t state, float *obs) { __builtin_memcpy(obs, extra + PERMUTED_ROWS + state * PERMUTED_ROW, (PERMUTED_STATES + 1u) * sizeof(float)); }

uint32_t permuted_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return PERMUTED_DRAWS + 1u + ENV_BATCH;
}

void permuted_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    permuted_code(seed, 0u, extra);
    for (uint32_t state = 0u; state < PERMUTED_STATES; state++) extra[PERMUTED_LOOKUP + state] = permuted_lookup(seed, state);
    batchReset(extra + PERMUTED_DRAWS);
    words[0] = philox4x32(seed, ENV_RNG_STREAM, 0u, 0u, 0u, 0u).x % PERMUTED_STATES;
}

static inline __attribute__((always_inline)) float permuted_move(uint32_t state, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    *done = 0u;
    float reward = (uint32_t)ctrl[0] % PERMUTED_ACTIONS == extra[PERMUTED_LOOKUP + words[0]] ? 1.0f : 0.0f;
    words[0] = state;
    permuted_obs(extra, state, obs);
    return reward;
}

static __attribute__((noinline)) float permuted_refill(uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    return permuted_move(permuted_draw(seed, step, extra + PERMUTED_DRAWS)[0], ctrl, words, extra, obs, done);
}

static float permuted_recode(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done);

static inline __attribute__((always_inline)) float permuted_shift(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    if (extra[0] != permuted_epoch(&PERMUTED_KNOBS[instance], step)) return permuted_recode(instance, seed, step, ctrl, words, extra, obs, done);
    uint32_t slot = batchSlot(extra + PERMUTED_DRAWS, step);
    if (slot == ENV_BATCH) return permuted_refill(seed, step, ctrl, words, extra, obs, done);
    return permuted_move(extra[PERMUTED_DRAWS + 1u + slot], ctrl, words, extra, obs, done);
}

static __attribute__((noinline)) float permuted_recode(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: permuted_code(seed, permuted_epoch(&PERMUTED_KNOBS[0], step), extra); return permuted_shift(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: permuted_code(seed, permuted_epoch(&PERMUTED_KNOBS[1], step), extra); return permuted_shift(1u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

float permuted_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return permuted_shift(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return permuted_shift(1u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void permuted_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed;
    if (extra[0] != permuted_epoch(&PERMUTED_KNOBS[instance], step)) __builtin_trap();
    permuted_obs(extra, words[0], obs);
}

void permuted_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    ctrl[0] = (float)extra[PERMUTED_LOOKUP + words[0]];
}
