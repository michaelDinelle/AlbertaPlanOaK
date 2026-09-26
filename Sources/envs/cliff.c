#include "../env.h"

#define CLIFF_WIDTH 12u
#define CLIFF_HEIGHT 4u
#define CLIFF_STEP_COST -1.0f
#define CLIFF_FALL -100.0f
#define CLIFF_GOAL_REWARD 100.0f

typedef struct { float slip; } cliffKnobs;

static const cliffKnobs CLIFF_KNOBS[2] = {
    { 0.0f },
    { 0.1f }
};

enum { CLIFF_CELLS = 16 * CLIFF_HEIGHT, CLIFF_MOVES = 4 * CLIFF_CELLS, CLIFF_CODES = 2 * CLIFF_MOVES, CLIFF_ROWS = CLIFF_CODES + CLIFF_MOVES / 4, CLIFF_ROW = 8, CLIFF_DRAWS = CLIFF_ROWS + CLIFF_CELLS * CLIFF_ROW, CLIFF_FELL = 64, CLIFF_REACHED = 128 };

static void cliff_obs(uint32_t x, uint32_t y, float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) obs[b] = (float)((x >> b) & 1u);
    obs[4] = (float)(y & 1u); obs[5] = (float)((y >> 1u) & 1u);
    obs[6] = 1.0f;
}

static uint8_t cliff_move(uint32_t x0, uint32_t y0, uint32_t action) {
    int dx, dy;
    gridDelta(action, &dx, &dy);
    int x = (int)x0 + dx, y = (int)y0 - dy;
    if (x < 0) x = 0; if (x >= (int)CLIFF_WIDTH) x = (int)CLIFF_WIDTH - 1;
    if (y < 0) y = 0; if (y >= (int)CLIFF_HEIGHT) y = (int)CLIFF_HEIGHT - 1;
    if (y == 0 && x > 0 && x < (int)CLIFF_WIDTH - 1) return CLIFF_FELL;
    if (y == 0 && x == (int)CLIFF_WIDTH - 1) return CLIFF_REACHED;
    return (uint8_t)(x | y << 4);
}

static lane4 cliff_draw(const cliffKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 slip = (lane4)(r.x < u01Bound(k->slip)), steer = ((r.y & 3u) << 2u & slip) | (3u & ~slip);
    __builtin_memcpy(draws + 1, &steer, sizeof steer);
    return steer;
}

uint32_t cliff_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return CLIFF_DRAWS + (CLIFF_KNOBS[instance].slip > 0.0f ? 1u + ENV_BATCH : 0u);
}

void cliff_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)seed;
    uint8_t *next = (uint8_t *)(extra + CLIFF_CODES);
    for (uint32_t y = 0u; y < CLIFF_HEIGHT; y++)
        for (uint32_t x = 0u; x < CLIFF_WIDTH; x++) {
            float row[CLIFF_ROW] = { 0.0f };
            cliff_obs(x, y, row);
            __builtin_memcpy(extra + CLIFF_ROWS + (x | y << 4) * CLIFF_ROW, row, sizeof row);
            for (uint32_t action = 0u; action < 4u; action++) {
                uint32_t entry = action * CLIFF_CELLS + (x | y << 4), code = cliff_move(x, y, action);
                next[entry] = (uint8_t)code;
                extra[2u * entry] = code & 15u;
                extra[2u * entry + 1u] = (code >> 4) & 3u;
            }
        }
    if (CLIFF_KNOBS[instance].slip > 0.0f) batchReset(extra + CLIFF_DRAWS);
    words[0] = 0u; words[1] = 0u;
}

static inline __attribute__((always_inline)) float cliff_walk(uint32_t action, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t move = action * CLIFF_CELLS, cell = words[0] | words[1] << 4, code = ((const uint8_t *)(extra + CLIFF_CODES) + move)[cell];
    __builtin_memcpy(words, extra + 2u * move + 2u * (size_t)cell, 2u * sizeof(uint32_t));
    *done = code >= CLIFF_FELL;
    __builtin_memcpy(obs, extra + CLIFF_ROWS + (code % CLIFF_FELL) * CLIFF_ROW, 7u * sizeof(float));
    static const float REWARDS[3] = { CLIFF_STEP_COST, CLIFF_FALL, CLIFF_GOAL_REWARD };
    return REWARDS[code / CLIFF_FELL];
}

static uint32_t cliff_action(const float *ctrl, uint32_t steer) { return ((uint32_t)ctrl[0] & steer & 3u) | steer >> 2u; }

static __attribute__((noinline)) float cliff_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    return cliff_walk(cliff_action(ctrl, cliff_draw(&CLIFF_KNOBS[instance], seed, step, extra + CLIFF_DRAWS)[0]), words, extra, obs, done);
}

static inline __attribute__((always_inline)) float cliff_slide(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    if (CLIFF_KNOBS[instance].slip > 0.0f) {
        uint32_t slot = batchSlot(extra + CLIFF_DRAWS, step);
        if (slot == ENV_BATCH) return cliff_refill(instance, seed, step, ctrl, words, extra, obs, done);
        return cliff_walk(cliff_action(ctrl, extra[CLIFF_DRAWS + 1u + slot]), words, extra, obs, done);
    }
    return cliff_walk((uint32_t)ctrl[0] & 3u, words, extra, obs, done);
}

float cliff_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return cliff_slide(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return cliff_slide(1u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void cliff_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step;
    __builtin_memcpy(obs, extra + CLIFF_ROWS + (words[0] | words[1] << 4) * CLIFF_ROW, 7u * sizeof(float));
}

void cliff_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t safeRow = (&CLIFF_KNOBS[instance])->slip > 0.0f ? CLIFF_HEIGHT - 1u : 1u;
    uint32_t x = words[0], y = words[1];
    if (x == CLIFF_WIDTH - 1u) ctrl[0] = 1.0f;
    else if (y < safeRow) ctrl[0] = 0.0f;
    else ctrl[0] = 3.0f;
}
