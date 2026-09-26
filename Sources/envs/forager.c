#include "../env.h"

#define FORAGER_SWITCH_PERIOD 100000u

typedef struct { uint32_t side, fov, switching; } foragerKnobs;

static const foragerKnobs FORAGER_KNOBS[4] = {
    { 16u, 1u, 0u },
    { 64u, 1u, 0u },
    { 16u, 2u, 0u },
    { 16u, 1u, 1u }
};


uint32_t forager_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return FORAGER_KNOBS[instance].side * FORAGER_KNOBS[instance].side + 1u + ENV_BATCH;
}

static void forager_spawn(const foragerKnobs *k, uint32_t seed, uint32_t index, uint32_t slot, uint32_t *map) {
    for (uint32_t probe = 0u;; probe++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 64u + slot, probe, 0u);
        uint32_t cell = r.x % (k->side * k->side);
        if (map[cell] == 0u) { map[cell] = 1u + (r.y & 1u); return; }
    }
}

static uint32_t forager_good(const foragerKnobs *k, uint32_t step) { return k->switching && ((step / FORAGER_SWITCH_PERIOD) & 1u) ? 2u : 1u; }

static const float FORAGER_VALUES[2][3] = { { 0.0f, 1.0f, -1.0f }, { 0.0f, -1.0f, 1.0f } };

static const float *forager_values(const foragerKnobs *k, uint32_t step) {
    return FORAGER_VALUES[forager_good(k, step) - 1u];
}

static inline __attribute__((always_inline)) lane4 forager_draw(const foragerKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 63u, 0u, 0u);
    lane4 clock = (r.x % (k->side * k->side)) << 2u | (r.y & 1u) << 1u | ((lane4)(r.z < u01Bound(0.25f)) & 1u);
    __builtin_memcpy(draws + 1, &clock, sizeof clock);
    return clock;
}

static inline __attribute__((always_inline)) void forager_obs(const foragerKnobs *k, uint32_t step, uint32_t x0, uint32_t y0, const uint32_t *map, float *obs) {
    const float *value = forager_values(k, step);
    uint32_t span = 2u * k->fov + 1u, i = 0u;
    for (uint32_t dy = 0u; dy < span; dy++) {
        const uint32_t *row = map + (y0 + k->side + dy - k->fov) % k->side * k->side;
        for (uint32_t dx = 0u; dx < span; dx++) obs[i++] = value[row[(x0 + k->side + dx - k->fov) % k->side]];
    }
    obs[i] = 1.0f;
}

void forager_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    uint32_t *map = extra;
    words[0] = k->side / 2u; words[1] = k->side / 2u;
    words[2] = k->side * k->side / 16u;
    for (uint32_t slot = 0u; slot < words[2]; slot++) forager_spawn(k, seed, 0u, slot, map);
    batchReset(extra + k->side * k->side);
}

void forager_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed;
    switch (instance) {
        case 0u: forager_obs(&FORAGER_KNOBS[0], step, words[0], words[1], extra, obs); return;
        case 1u: forager_obs(&FORAGER_KNOBS[1], step, words[0], words[1], extra, obs); return;
        case 2u: forager_obs(&FORAGER_KNOBS[2], step, words[0], words[1], extra, obs); return;
        case 3u: forager_obs(&FORAGER_KNOBS[3], step, words[0], words[1], extra, obs); return;
        default: __builtin_trap();
    }
}

static float forager_reward(const foragerKnobs *k, uint32_t step, uint32_t item) { return forager_values(k, step)[item]; }

static __attribute__((noinline)) float forager_grow(uint32_t instance, uint32_t seed, uint32_t step, uint32_t item, uint32_t *words, uint32_t *extra, float *obs) {
    forager_spawn(&FORAGER_KNOBS[instance], seed, step, 0u, extra);
    words[2] += 1u;
    forager_observe(instance, seed, words, extra, step, obs, 0);
    return forager_reward(&FORAGER_KNOBS[instance], step, item);
}

static inline __attribute__((always_inline)) float forager_move(uint32_t instance, uint32_t seed, uint32_t step, uint32_t clock, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    uint32_t cells = k->side * k->side, *map = extra;
    int dx, dy;
    gridDelta((uint32_t)ctrl[0] & 3u, &dx, &dy);
    uint32_t x = (words[0] + k->side + (uint32_t)dx) % k->side, y = (words[1] + k->side + (uint32_t)dy) % k->side, count = words[2];
    uint32_t cell = y * k->side + x, item = map[cell];
    map[cell] = 0u;
    count -= item != 0u;
    uint32_t decaying = clock >> 2u, decays = (map[decaying] != 0u) & (clock >> 1u) & 1u;
    map[decaying] = decays ? 0u : map[decaying];
    count -= decays;
    words[0] = x; words[1] = y; words[2] = count;
    *done = 0u;
    if (__builtin_unpredictable((count < cells / 16u) & clock)) return forager_grow(instance, seed, step, item, words, extra, obs);
    forager_obs(k, step, x, y, map, obs);
    return forager_reward(k, step, item);
}

static inline __attribute__((always_inline)) float forager_redraw(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    return forager_move(instance, seed, step, forager_draw(k, seed, step, extra + k->side * k->side)[0], ctrl, words, extra, obs, done);
}

static __attribute__((noinline)) float forager_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: return forager_redraw(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return forager_redraw(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return forager_redraw(2u, seed, step, ctrl, words, extra, obs, done);
        case 3u: return forager_redraw(3u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float forager_walk(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    uint32_t *draws = extra + k->side * k->side, slot = batchSlot(draws, step);
    if (slot == ENV_BATCH) return forager_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return forager_move(instance, seed, step, draws[1u + slot], ctrl, words, extra, obs, done);
}

float forager_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return forager_walk(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return forager_walk(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return forager_walk(2u, seed, step, ctrl, words, extra, obs, done);
        case 3u: return forager_walk(3u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) void forager_seek(const foragerKnobs *k, uint32_t step, const uint32_t *words, const uint32_t *map, float *ctrl) {
    uint32_t good = forager_good(k, step + 1u), cells = k->side * k->side;
    uint8_t firstStep[64 * 64];
    uint16_t queue[64 * 64];
    for (uint32_t i = 0u; i < cells; i++) firstStep[i] = 255u;
    uint32_t start = words[1] * k->side + words[0];
    firstStep[start] = 254u;
    uint32_t head = 0u, tail = 0u;
    queue[tail++] = (uint16_t)start;
    while (head < tail) {
        uint32_t cell = queue[head++], cx = cell % k->side, cy = cell / k->side;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            uint32_t nx = (cx + k->side + (uint32_t)dx) % k->side, ny = (cy + k->side + (uint32_t)dy) % k->side, next = ny * k->side + nx;
            if (firstStep[next] != 255u || (map[next] != 0u && map[next] != good)) continue;
            firstStep[next] = firstStep[cell] == 254u ? (uint8_t)d : firstStep[cell];
            if (map[next] == good) { ctrl[0] = (float)firstStep[next]; return; }
            queue[tail++] = (uint16_t)next;
        }
    }
    ctrl[0] = 0.0f;
}

void forager_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)obs;
    switch (instance) {
        case 0u: forager_seek(&FORAGER_KNOBS[0], step, words, extra, ctrl); return;
        case 1u: forager_seek(&FORAGER_KNOBS[1], step, words, extra, ctrl); return;
        case 2u: forager_seek(&FORAGER_KNOBS[2], step, words, extra, ctrl); return;
        case 3u: forager_seek(&FORAGER_KNOBS[3], step, words, extra, ctrl); return;
        default: __builtin_trap();
    }
}

uint32_t forager_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    uint32_t cells = FORAGER_KNOBS[instance].side * FORAGER_KNOBS[instance].side, count = cells / 16u;
    if (!out) return count;
    for (uint32_t i = 0u; i < count; i++) out[i] = 0u;
    for (uint32_t cell = 0u; cell < cells; cell++) out[cell / 16u] |= (extra[cell] & 3u) << (2u * (cell % 16u));
    return count;
}
