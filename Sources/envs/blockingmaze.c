#include "../env.h"

#define BLOCKING_WIDTH 9
#define BLOCKING_HEIGHT 6
#define BLOCKING_PERIOD 100000u

typedef struct { GridWalls phase[2]; } blockingKnobs;

static const blockingKnobs BLOCKING_KNOBS[2] = {
    { { { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 0 }, { 3 }, { 7 }, { 3 } }, { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 8 }, { 3 } } } },
    { { { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 8 }, { 3 } }, { BLOCKING_WIDTH, BLOCKING_HEIGHT, 0, 1, { 1 }, { 3 }, { 7 }, { 3 } } } }
};

static void blocking_obs(uint32_t x, uint32_t y, float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) obs[b] = (float)((x >> b) & 1u);
    for (uint32_t b = 0u; b < 3u; b++) obs[4u + b] = (float)((y >> b) & 1u);
    obs[7] = 1.0f;
}

void blocking_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed; (void)extra;
    words[0] = 3u; words[1] = 5u;
}

float blocking_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)extra;
    const GridWalls *walls = &BLOCKING_KNOBS[instance].phase[(step / BLOCKING_PERIOD) & 1u];
    int dx, dy;
    gridDelta((uint32_t)ctrl[0] & 3u, &dx, &dy);
    int x = (int)words[0] + dx, y = (int)words[1] + dy;
    if (gridWall(walls, x, y)) { x = (int)words[0]; y = (int)words[1]; }
    float reward = 0.0f;
    *done = 0u;
    if (x == 8 && y == 0) { reward = 1.0f; x = 3; y = 5; *done = 1u; }
    words[0] = (uint32_t)x; words[1] = (uint32_t)y;
    blocking_obs(words[0], words[1], obs);
    return reward;
}

void blocking_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    blocking_obs(words[0], words[1], obs);
}

void blocking_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)obs;
    int d = gridFirstStep(&BLOCKING_KNOBS[instance].phase[((step + 1u) / BLOCKING_PERIOD) & 1u], (int)words[0], (int)words[1], 8, 0);
    ctrl[0] = (float)(d < 0 ? 0 : d);
}
