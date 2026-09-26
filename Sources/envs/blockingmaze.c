#include "../env.h"

#define BLOCKING_WIDTH 9
#define BLOCKING_HEIGHT 6
#define BLOCKING_PERIOD 100000u

typedef struct { GridWalls phase[2]; } blockingKnobs;

static const blockingKnobs BLOCKING_KNOBS[2] = {
    { { { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 0 }, { 3 }, { 7 }, { 3 } }, { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 8 }, { 3 } } } },
    { { { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 8 }, { 3 } }, { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 7 }, { 3 } } } }
};

enum { BLOCKING_CELLS = 16 * BLOCKING_HEIGHT, BLOCKING_MOVES = 8 * BLOCKING_CELLS, BLOCKING_CODES = 2 * BLOCKING_MOVES, BLOCKING_ROWS = BLOCKING_CODES + BLOCKING_MOVES / 4, BLOCKING_ROW = 8, BLOCKING_GOAL = 128, BLOCKING_ROUTES = BLOCKING_ROWS + BLOCKING_CELLS * BLOCKING_ROW, BLOCKING_SPAN = 256 };

static void blocking_obs(uint32_t x, uint32_t y, float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) obs[b] = (float)((x >> b) & 1u);
    for (uint32_t b = 0u; b < 3u; b++) obs[4u + b] = (float)((y >> b) & 1u);
    obs[7] = 1.0f;
}

static uint8_t blocking_move(const GridWalls *walls, int x0, int y0, uint32_t action) {
    int dx, dy;
    gridDelta(action, &dx, &dy);
    int x = x0 + dx, y = y0 + dy;
    if (gridWall(walls, x, y)) { x = x0; y = y0; }
    if (x == 8 && y == 0) return (uint8_t)(3 | 5 << 4 | BLOCKING_GOAL);
    return (uint8_t)(x | y << 4);
}

uint32_t blocking_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return BLOCKING_ROUTES + 1u + 2u * BLOCKING_SPAN / 4u;
}

void blocking_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)seed;
    uint8_t *next = (uint8_t *)(extra + BLOCKING_CODES);
    for (uint32_t y = 0u; y < BLOCKING_HEIGHT; y++)
        for (uint32_t x = 0u; x < BLOCKING_WIDTH; x++) {
            float row[BLOCKING_ROW];
            blocking_obs(x, y, row);
            __builtin_memcpy(extra + BLOCKING_ROWS + (x | y << 4) * BLOCKING_ROW, row, sizeof row);
            for (uint32_t move = 0u; move < 8u; move++) {
                uint32_t entry = move * BLOCKING_CELLS + (x | y << 4), code = blocking_move(&BLOCKING_KNOBS[instance].phase[move / 4u], (int)x, (int)y, move % 4u);
                next[entry] = (uint8_t)code;
                extra[2u * entry] = code & 15u;
                extra[2u * entry + 1u] = (code >> 4) & 7u;
            }
        }
    words[0] = 3u; words[1] = 5u;
}

float blocking_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance; (void)seed;
    uint32_t move = (((step / BLOCKING_PERIOD) & 1u) * 4u + ((uint32_t)ctrl[0] & 3u)) * BLOCKING_CELLS, cell = words[0] | words[1] << 4, code = ((const uint8_t *)(extra + BLOCKING_CODES) + move)[cell];
    __builtin_memcpy(words, extra + 2u * move + 2u * (size_t)cell, 2u * sizeof(uint32_t));
    *done = code / BLOCKING_GOAL;
    __builtin_memcpy(obs, extra + BLOCKING_ROWS + (code % BLOCKING_GOAL) * BLOCKING_ROW, BLOCKING_ROW * sizeof(float));
    return (float)(code / BLOCKING_GOAL);
}

void blocking_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step;
    __builtin_memcpy(obs, extra + BLOCKING_ROWS + (words[0] | words[1] << 4) * BLOCKING_ROW, BLOCKING_ROW * sizeof(float));
}

void blocking_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)obs;
    uint8_t *routes = (uint8_t *)(extra + BLOCKING_ROUTES + 1);
    if (extra[BLOCKING_ROUTES] == 0u) {
        for (uint32_t phase = 0u; phase < 2u; phase++) gridDistance(gridWallBlocked, &BLOCKING_KNOBS[instance].phase[phase], 8, 0, routes + phase * BLOCKING_SPAN);
        extra[BLOCKING_ROUTES] = 1u;
    }
    int d = gridDescend(routes + (((step + 1u) / BLOCKING_PERIOD) & 1u) * BLOCKING_SPAN, (int)words[0], (int)words[1]);
    ctrl[0] = (float)(d < 0 ? 0 : d);
}
