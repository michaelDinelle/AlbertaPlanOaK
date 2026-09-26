#include "../env.h"

typedef struct { uint32_t doors, cues; } corridorKnobs;

static const corridorKnobs CORRIDOR_KNOBS[2] = {
    { 1u, 1u },
    { 3u, 0u }
};

enum { CORRIDOR_CELLS = 256, CORRIDOR_PAIRS = 2 * CORRIDOR_CELLS, CORRIDOR_ORIGIN = CORRIDOR_PAIRS + 2 * CORRIDOR_CELLS, CORRIDOR_FLAGS = CORRIDOR_PAIRS + 4 * CORRIDOR_CELLS, CORRIDOR_FLAG = 8 };

static int corridor_wall(const corridorKnobs *k, int x, int y) {
    return x <= 0 || x >= 4 * (int)(k->doors + 1u) || y <= 0 || y >= 6 || (x % 4 == 0 && y != 3);
}

static float corridor_width(const corridorKnobs *k) { return (float)(4u * (k->doors + 1u)); }

static uint32_t corridor_cell(uint32_t x, uint32_t y) { return x | y << 5; }

static void corridor_place(const corridorKnobs *k, uint32_t seed, uint32_t resample, uint32_t *words) {
    uint32_t drawn[8];
    for (uint32_t block = 0u; 4u * block < k->doors + 2u; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, resample, block, 0u, 0u);
        drawn[4u * block] = r.x; drawn[4u * block + 1u] = r.y; drawn[4u * block + 2u] = r.z; drawn[4u * block + 3u] = r.w;
    }
    uint32_t cell = (uint32_t)(u01(drawn[0]) * 15.0f);
    words[0] = 1u + cell % 3u;
    words[1] = 1u + cell / 3u;
    for (uint32_t i = 0u; i < k->doors; i++) {
        cell = (uint32_t)(u01(drawn[1u + i]) * 15.0f);
        words[7u + i] = (4u * i + 1u + cell % 3u) | ((1u + cell / 3u) << 4u);
    }
    cell = (uint32_t)(u01(drawn[1u + k->doors]) * 15.0f);
    words[2] = 4u * k->doors + 1u + cell % 3u;
    words[3] = 1u + cell / 3u;
}

uint32_t corridor_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return CORRIDOR_FLAGS + (CORRIDOR_FLAG << (2u * CORRIDOR_KNOBS[instance].doors));
}

void corridor_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)seed; (void)steps;
    const corridorKnobs *k = &CORRIDOR_KNOBS[instance];
    float width = corridor_width(k);
    uint16_t *moves = (uint16_t *)extra;
    for (uint32_t cell = 0u; cell < CORRIDOR_CELLS; cell++)
        for (uint32_t action = 0u; action < 4u; action++) {
            int x = (int)(cell & 31u), y = (int)(cell >> 5), dx, dy;
            gridDelta(action, &dx, &dy);
            int nx = x + dx, ny = y + dy;
            uint32_t need = 0u;
            if (corridor_wall(k, nx, ny)) { nx = x; ny = y; }
            else if (ny == 3 && nx % 4 == 0) need = 1u << ((uint32_t)nx / 4u - 1u);
            moves[4u * cell + action] = (uint16_t)(corridor_cell((uint32_t)nx, (uint32_t)ny) | need << 8);
        }
    for (int dy = -7; dy <= 7; dy++)
        for (int dx = -15; dx <= 15; dx++) {
            float pair[2] = { (float)dx / width, (float)dy / 6.0f };
            __builtin_memcpy(extra + CORRIDOR_ORIGIN + 2 * (dx + 32 * dy), pair, sizeof pair);
        }
    for (uint32_t bits = 0u; bits < 1u << (2u * k->doors); bits++) {
        float flags[CORRIDOR_FLAG] = { 0.0f };
        if (k->cues) {
            int held = (bits & 1u) != 0u;
            flags[0] = held ? 1.0f : 0.0f;
            flags[1] = (bits & 2u) != 0u ? 1.0f : 0.0f;
            flags[2] = held ? 0.0f : 1.0f;
        } else
            for (uint32_t i = 0u; i < 2u * k->doors; i++) flags[i] = (bits & (1u << i)) != 0u ? 1.0f : 0.0f;
        __builtin_memcpy(extra + CORRIDOR_FLAGS + CORRIDOR_FLAG * bits, flags, sizeof flags);
    }
    words[0] = 1u; words[1] = 1u;
    words[2] = 4u * k->doors + 3u; words[3] = 1u;
    for (uint32_t i = 0u; i < k->doors; i++) words[7u + i] = (4u * i + 3u) | (5u << 4u);
}

static int corridor_sought(uint32_t bits, uint32_t key, int offset) { return offset & ((int)(bits >> key & 1u) - 1); }

static inline __attribute__((always_inline)) void corridor_view(const corridorKnobs *k, uint32_t cell, uint32_t goal, uint32_t bits, uint32_t clock, const int *key, const uint32_t *extra, float *obs) {
    const uint32_t *origin = extra + CORRIDOR_ORIGIN, *flags = extra + CORRIDOR_FLAGS + CORRIDOR_FLAG * (bits & ((1u << (2u * k->doors)) - 1u));
    __builtin_memcpy(obs, origin + 2L * (int)cell, 2u * sizeof(float));
    if (k->cues) {
        __builtin_memcpy(obs + 2, flags, 4u * sizeof(float));
        __builtin_memcpy(obs + 5, origin + 2L * corridor_sought(bits, 0u, key[0]), 2u * sizeof(float));
        __builtin_memcpy(obs + 7, origin + 2L * ((int)corridor_cell(4u, 3u) - (int)cell), 2u * sizeof(float));
        __builtin_memcpy(obs + 9, origin + 2L * ((int)goal - (int)cell), 2u * sizeof(float));
        obs[11] = cell == corridor_cell(4u, 3u) ? 1.0f : 0.0f;
        obs[12] = fminf((float)clock / 256.0f, 1.0f);
        obs[13] = 1.0f;
        return;
    }
    __builtin_memcpy(obs + 2, flags, 2u * k->doors * sizeof(float));
    for (uint32_t i = 0u; i < k->doors; i++) __builtin_memcpy(obs + 2u + 2u * k->doors + 2u * i, origin + 2L * corridor_sought(bits, i, key[i]), 2u * sizeof(float));
    __builtin_memcpy(obs + 2u + 4u * k->doors, origin + 2L * ((int)goal - (int)cell), 2u * sizeof(float));
    obs[4u + 4u * k->doors] = 1.0f;
}

static inline __attribute__((always_inline)) void corridor_keys(const corridorKnobs *k, const uint32_t *words, uint32_t cell, int *key) {
    for (uint32_t i = 0u; i < k->doors; i++) key[i] = (int)(words[7u + i] + (words[7u + i] & 0xf0u)) - (int)cell;
}

static inline __attribute__((always_inline)) void corridor_show(const corridorKnobs *k, const uint32_t *words, const uint32_t *extra, float *obs) {
    uint32_t cell = corridor_cell(words[0], words[1]);
    int key[3];
    corridor_keys(k, words, cell, key);
    corridor_view(k, cell, corridor_cell(words[2], words[3]), words[4], words[5], key, extra, obs);
}

static void corridor_see(uint32_t instance, const uint32_t *words, const uint32_t *extra, float *obs) {
    switch (instance) {
        case 0u: corridor_show(&CORRIDOR_KNOBS[0], words, extra, obs); return;
        case 1u: corridor_show(&CORRIDOR_KNOBS[1], words, extra, obs); return;
        default: __builtin_trap();
    }
}

static __attribute__((noinline, cold)) float corridor_arrive(uint32_t instance, uint32_t seed, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    float reward = 0.0f;
    *done = 0u;
    if (words[0] == words[2] && words[1] == words[3]) {
        reward = 1.0f;
        *done = 1u;
        corridor_place(&CORRIDOR_KNOBS[instance], seed, words[6], words);
        words[4] = 0u; words[5] = 0u; words[6] += 1u;
    }
    corridor_see(instance, words, extra, obs);
    return reward;
}

static inline __attribute__((always_inline)) float corridor_move(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    const corridorKnobs *k = &CORRIDOR_KNOBS[instance];
    uint32_t x = words[0], y = words[1], goalX = words[2], goalY = words[3], bits = words[4], clock = words[5] + 1u;
    uint32_t code = ((const uint16_t *)extra)[4u * corridor_cell(x, y) + ((uint32_t)ctrl[0] & 3u)], need = code >> 8, nx = code & 31u, ny = code >> 5 & 7u;
    if (__builtin_expect(need != 0u, 0) && (bits & need << k->doors) == 0u) {
        if ((bits & need) != 0u) bits |= need << k->doors;
        else { nx = x; ny = y; }
    }
    uint32_t cell = corridor_cell(nx, ny);
    int key[3];
    corridor_keys(k, words, cell, key);
    for (uint32_t i = 0u; i < k->doors; i++) bits |= (uint32_t)(key[i] == 0) << i;
    words[0] = nx; words[1] = ny; words[4] = bits; words[5] = clock;
    if (__builtin_expect(nx == goalX && ny == goalY, 0)) return corridor_arrive(instance, seed, words, extra, obs, done);
    *done = 0u;
    corridor_view(k, cell, corridor_cell(goalX, goalY), bits, clock, key, extra, obs);
    return 0.0f;
}

float corridor_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return corridor_move(0u, seed, ctrl, words, extra, obs, done);
        case 1u: return corridor_move(1u, seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void corridor_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    corridor_see(instance, words, extra, obs);
}

void corridor_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)extra; (void)obs; (void)shared;
    const corridorKnobs *k = &CORRIDOR_KNOBS[instance];
    int x = (int)words[0], y = (int)words[1], room = x / 4, tx, ty;
    uint32_t bits = words[4];
    if ((uint32_t)room == k->doors) { tx = (int)words[2]; ty = (int)words[3]; }
    else if ((bits & (1u << room)) == 0u) { tx = (int)(words[7u + (uint32_t)room] & 15u); ty = (int)(words[7u + (uint32_t)room] >> 4u); }
    else { tx = 4 * (room + 1); ty = 3; }
    uint32_t action;
    if (x == tx && y == ty) action = y == 1 ? 0u : y == 5 ? 1u : x % 4 == 3 ? 3u : (x % 4 == 1 && !(y == 3 && room > 0)) ? 2u : 0u;
    else if (tx % 4 == 0 && x % 4 != 0 && y != ty) action = y > ty ? 0u : 1u;
    else if (x != tx) action = x > tx ? 2u : 3u;
    else action = y > ty ? 0u : 1u;
    ctrl[0] = (float)action;
}
