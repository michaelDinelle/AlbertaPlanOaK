#include "../env.h"

typedef struct { int extent; uint32_t mapWords; float moveCost, slip; } deepseaKnobs;

static const deepseaKnobs DEEPSEA_KNOBS[4] = {
    { 7, 2u, -0.00125f, 0.0f },
    { 15, 8u, -0.000625f, 0.0f },
    { 31, 32u, -0.0003125f, 0.0f },
    { 15, 8u, -0.000625f, 0.05f }
};


static void deepsea_obs(const deepseaKnobs *k, uint32_t row, uint32_t column, uint32_t lastAction, float *obs) {
    float gridExtent = (float)k->extent;
    obs[0] = (float)row / gridExtent;
    obs[1] = (float)column / gridExtent;
    obs[2] = (float)(k->extent - (int)column) / gridExtent;
    obs[3] = (float)(k->extent - (int)row) / gridExtent;
    obs[4] = (column == (uint32_t)k->extent) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = (float)lastAction;
    obs[7] = 1.0f;
}

static uint32_t deepsea_rightward(const deepseaKnobs *k, const uint32_t *words) {
    uint32_t mapBitIndex = words[0] * (uint32_t)(k->extent + 1) + words[1];
    return (words[3u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) & 1u;
}

static uint32_t deepsea_rows(const deepseaKnobs *k) { return 16u * (uint32_t)(k->extent + 1); }
static uint32_t deepsea_rewards(const deepseaKnobs *k) { return deepsea_rows(k) + 4u * (uint32_t)k->extent; }
static uint32_t deepsea_draws(const deepseaKnobs *k) { return deepsea_rewards(k) + 4u; }

static lane4 deepsea_draw(const deepseaKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 flip = (lane4)(r.x < u01Bound(k->slip)) & 1u;
    __builtin_memcpy(draws + 1, &flip, sizeof flip);
    return flip;
}

uint32_t deepsea_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    return deepsea_draws(k) + (k->slip > 0.0f ? 1u + ENV_BATCH : 0u);
}

void deepsea_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    for (uint32_t column = 0u; column <= (uint32_t)k->extent; column++)
        for (uint32_t action = 0u; action < 2u; action++) {
            float row[8];
            deepsea_obs(k, 0u, column, action, row);
            row[0] = 0.0f; row[3] = 0.0f;
            __builtin_memcpy(extra + (column * 2u + action) * 8u, row, sizeof row);
        }
    for (uint32_t r = 0u; r < (uint32_t)k->extent; r++) {
        float row[8];
        deepsea_obs(k, r, 0u, 0u, row);
        row[1] = 0.0f; row[2] = 0.0f;
        __builtin_memcpy(extra + deepsea_rows(k) + 4u * r, row, 4u * sizeof(float));
    }
    for (uint32_t action = 0u; action < 2u; action++)
        for (uint32_t hit = 0u; hit < 2u; hit++) {
            float reward = action != 0u ? k->moveCost : 0.0f;
            if (hit) reward = reward + 1.0f;
            __builtin_memcpy(extra + deepsea_rewards(k) + action * 2u + hit, &reward, sizeof reward);
        }
    if (k->slip > 0.0f) batchReset(extra + deepsea_draws(k));
    words[0] = 0u; words[1] = 0u; words[2] = 0u;
    for (uint32_t block = 0u; block * 4u < k->mapWords; block++) {
        u4 rm = philox4x32(seed, RNG_STREAM_RESET, block, 0u, 0u, 0u);
        uint32_t w[4] = { rm.x, rm.y, rm.z, rm.w };
        for (uint32_t i = 0u; i < 4u && block * 4u + i < k->mapWords; i++) words[3u + block * 4u + i] = w[i];
    }
}

static inline __attribute__((always_inline)) float deepsea_move(const deepseaKnobs *k, uint32_t step, uint32_t flip, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t extent = (uint32_t)k->extent, row = (step - 1u) % extent;
    if (words[0] != row) __builtin_trap();
    uint32_t action = ((uint32_t)ctrl[0] ^ flip) & 1u, mapBitIndex = row * (extent + 1u);
    uint32_t matching = (words[3u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) ^ (action - 1u);
    uint32_t column = words[1], up = column < extent ? column + 1u : extent, down = column != 0u ? column - 1u : 0u;
    uint32_t moved = (matching >> column) & 1u ? up : down, wrap = row + 1u >= extent;
    uint32_t nextRow = wrap ? 0u : row + 1u, nextColumn = wrap ? 0u : moved;
    words[0] = nextRow; words[1] = nextColumn; words[2] = action;
    *done = 0u;
    float4 head, tail, fraction;
    __builtin_memcpy(&head, extra + (nextColumn * 2u + action) * 8u, sizeof head);
    __builtin_memcpy(&tail, extra + (nextColumn * 2u + action) * 8u + 4u, sizeof tail);
    __builtin_memcpy(&fraction, extra + deepsea_rows(k) + 4u * nextRow, sizeof fraction);
    head = (float4)((lane4)head | (lane4)fraction);
    __builtin_memcpy(obs, &head, sizeof head);
    __builtin_memcpy(obs + 4, &tail, sizeof tail);
    float reward;
    __builtin_memcpy(&reward, extra + deepsea_rewards(k) + action * 2u + (wrap & (moved == extent)), sizeof reward);
    return reward;
}

static inline __attribute__((always_inline)) float deepsea_redraw(const deepseaKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    return deepsea_move(k, step, deepsea_draw(k, seed, step, extra + deepsea_draws(k))[0], ctrl, words, extra, obs, done);
}

static __attribute__((noinline)) float deepsea_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: return deepsea_redraw(&DEEPSEA_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
        case 1u: return deepsea_redraw(&DEEPSEA_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
        case 2u: return deepsea_redraw(&DEEPSEA_KNOBS[2], seed, step, ctrl, words, extra, obs, done);
        case 3u: return deepsea_redraw(&DEEPSEA_KNOBS[3], seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float deepsea_walk(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    if (k->slip > 0.0f) {
        uint32_t slot = batchSlot(extra + deepsea_draws(k), step);
        if (slot == ENV_BATCH) return deepsea_refill(instance, seed, step, ctrl, words, extra, obs, done);
        return deepsea_move(k, step, extra[deepsea_draws(k) + 1u + slot], ctrl, words, extra, obs, done);
    }
    return deepsea_move(k, step, 0u, ctrl, words, extra, obs, done);
}

float deepsea_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return deepsea_walk(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return deepsea_walk(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return deepsea_walk(2u, seed, step, ctrl, words, extra, obs, done);
        case 3u: return deepsea_walk(3u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void deepsea_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    deepsea_obs(&DEEPSEA_KNOBS[instance], words[0], words[1], words[2], obs);
}

void deepsea_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)deepsea_rightward(&DEEPSEA_KNOBS[instance], words);
}
