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
    return FORAGER_KNOBS[instance].side * FORAGER_KNOBS[instance].side;
}

static void forager_spawn(const foragerKnobs *k, uint32_t seed, uint32_t index, uint32_t slot, uint32_t *map) {
    for (uint32_t probe = 0u;; probe++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 64u + slot, probe, 0u);
        uint32_t cell = r.x % (k->side * k->side);
        if (map[cell] == 0u) { map[cell] = 1u + (r.y & 1u); return; }
    }
}

static uint32_t forager_good(const foragerKnobs *k, uint32_t step) { return k->switching && ((step / FORAGER_SWITCH_PERIOD) & 1u) ? 2u : 1u; }

static void forager_obs(const foragerKnobs *k, uint32_t step, const uint32_t *words, const uint32_t *map, float *obs) {
    uint32_t good = forager_good(k, step), span = 2u * k->fov + 1u, i = 0u;
    for (uint32_t dy = 0u; dy < span; dy++)
        for (uint32_t dx = 0u; dx < span; dx++) {
            uint32_t x = (words[0] + k->side + dx - k->fov) % k->side, y = (words[1] + k->side + dy - k->fov) % k->side;
            uint32_t item = map[y * k->side + x];
            obs[i++] = item == 0u ? 0.0f : item == good ? 1.0f : -1.0f;
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
}

float forager_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    uint32_t *map = extra;
    int dx, dy;
    gridDelta((uint32_t)ctrl[0] & 3u, &dx, &dy);
    words[0] = (words[0] + k->side + (uint32_t)dx) % k->side;
    words[1] = (words[1] + k->side + (uint32_t)dy) % k->side;
    *done = 0u;
    float reward = 0.0f;
    uint32_t cell = words[1] * k->side + words[0];
    if (map[cell] != 0u) {
        reward = map[cell] == forager_good(k, step) ? 1.0f : -1.0f;
        map[cell] = 0u;
        words[2] -= 1u;
    }
    u4 clock = philox4x32(seed, ENV_RNG_STREAM, step, 63u, 0u, 0u);
    uint32_t decaying = clock.x % (k->side * k->side);
    if (map[decaying] != 0u && (clock.y & 1u) != 0u) { map[decaying] = 0u; words[2] -= 1u; }
    if (words[2] < k->side * k->side / 16u && u01(clock.z) < 0.25f) {
        forager_spawn(k, seed, step, 0u, map);
        words[2] += 1u;
    }
    forager_obs(k, step, words, map, obs);
    return reward;
}

void forager_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed;
    forager_obs(&FORAGER_KNOBS[instance], step, words, extra, obs);
}

void forager_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)obs;
    const foragerKnobs *k = &FORAGER_KNOBS[instance];
    const uint32_t *map = extra;
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
