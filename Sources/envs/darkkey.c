#include "../env.h"

typedef struct { uint32_t side, horizon, period; } darkkeyKnobs;

static const darkkeyKnobs DARKKEY_KNOBS[1] = {
    { 9u, 50u, 1000u }
};

static uint32_t darkkey_cell(const darkkeyKnobs *k, uint32_t draw) {
    uint32_t index = (uint32_t)(u01(draw) * (float)(k->side * k->side));
    return index % k->side | index / k->side << 4;
}

static void darkkey_begin(const darkkeyKnobs *k, uint32_t seed, uint32_t step, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    uint32_t start = darkkey_cell(k, r.x);
    if (step % (k->horizon * k->period) == 0u) words[2] = darkkey_cell(k, r.y) | darkkey_cell(k, r.z) << 8;
    words[0] = start & 15u;
    words[1] = start >> 4;
    words[3] = 0u;
}

static void darkkey_obs(uint32_t x, uint32_t y, float *obs) {
    obs[0] = (float)x;
    obs[1] = (float)y;
}

static uint32_t darkkey_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 5.0f ? (uint32_t)c : 0u;
}

void darkkey_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)extra; (void)steps;
    darkkey_begin(&DARKKEY_KNOBS[instance], seed, 0u, words);
}

static __attribute__((noinline, cold)) float darkkey_event(const darkkeyKnobs *k, uint32_t seed, uint32_t step, uint32_t x, uint32_t y, uint32_t *words, float *obs, uint32_t *done) {
    uint32_t cell = x | y << 4, task = words[2], found = words[3];
    float reward = 0.0f;
    if (found == 0u && cell == (task & 255u)) { found = 1u; reward += 1.0f; }
    if (found == 1u && cell == task >> 8) { found = 2u; reward += 1.0f; }
    words[0] = x; words[1] = y; words[3] = found;
    *done = step % k->horizon == 0u;
    if (*done) darkkey_begin(k, seed, step, words);
    darkkey_obs(words[0], words[1], obs);
    return reward;
}

static inline __attribute__((always_inline)) float darkkey_move(const darkkeyKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    uint32_t action = darkkey_action(ctrl), x = words[0], y = words[1], task = words[2], found = words[3];
    int dx, dy;
    gridDelta(action - 1u, &dx, &dy);
    if (action == 0u) { dx = 0; dy = 0; }
    uint32_t nx = x + (uint32_t)dx, ny = y + (uint32_t)dy;
    if (nx >= k->side || ny >= k->side) { nx = x; ny = y; }
    uint32_t sought = found == 0u ? task & 255u : task >> 8;
    if (__builtin_expect(((found < 2u) & ((nx | ny << 4) == sought)) | (step % k->horizon == 0u), 0)) return darkkey_event(k, seed, step, nx, ny, words, obs, done);
    words[0] = nx; words[1] = ny;
    *done = 0u;
    darkkey_obs(nx, ny, obs);
    return 0.0f;
}

float darkkey_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)extra; (void)shared;
    switch (instance) {
        case 0u: return darkkey_move(&DARKKEY_KNOBS[0], seed, step, ctrl, words, obs, done);
        default: __builtin_trap();
    }
}

void darkkey_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    darkkey_obs(words[0], words[1], obs);
}

void darkkey_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs; (void)shared;
    uint32_t x = words[0], y = words[1], found = words[3], sought = found == 0u ? words[2] & 255u : words[2] >> 8, tx = sought & 15u, ty = sought >> 4;
    uint32_t action = found == 2u || (x == tx && y == ty) ? 0u : x != tx ? (x < tx ? 4u : 3u) : (y < ty ? 2u : 1u);
    ctrl[0] = (float)action;
}
