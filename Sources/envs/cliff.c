#include "../env.h"

#define CLIFF_WIDTH 12u
#define CLIFF_HEIGHT 4u

typedef struct { float slip; } cliffKnobs;

OAK_CONSTANT cliffKnobs CLIFF_KNOBS[2] = {
    { 0.0f },
    { 0.1f }
};


static void cliff_obs(uint32_t x, uint32_t y, device float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) obs[b] = (float)((x >> b) & 1u);
    obs[4] = (float)(y & 1u); obs[5] = (float)((y >> 1u) & 1u);
    obs[6] = 1.0f;
}

void cliff_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)extra; words[0] = 0u; words[1] = 0u; }

float cliff_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const cliffKnobs *k = &CLIFF_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0] & 3u;
    if (k->slip > 0.0f) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
        if (u01(r.x) < k->slip) action = r.y & 3u;
    }
    int dx, dy;
    gridDelta(action, &dx, &dy);
    int x = (int)words[0] + dx, y = (int)words[1] - dy;
    if (x < 0) x = 0; if (x >= (int)CLIFF_WIDTH) x = (int)CLIFF_WIDTH - 1;
    if (y < 0) y = 0; if (y >= (int)CLIFF_HEIGHT) y = (int)CLIFF_HEIGHT - 1;
    float reward = -1.0f;
    *done = 0u;
    if (y == 0 && x > 0 && x < (int)CLIFF_WIDTH - 1) { reward = -100.0f; x = 0; y = 0; *done = 1u; }
    else if (y == 0 && x == (int)CLIFF_WIDTH - 1) { reward = 0.0f; x = 0; y = 0; *done = 1u; }
    words[0] = (uint32_t)x; words[1] = (uint32_t)y;
    cliff_obs(words[0], words[1], obs);
    return reward;
}

void cliff_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    cliff_obs(words[0], words[1], obs);
}

void cliff_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t safeRow = (&CLIFF_KNOBS[instance])->slip > 0.0f ? CLIFF_HEIGHT - 1u : 1u;
    uint32_t x = words[0], y = words[1];
    if (x == CLIFF_WIDTH - 1u) ctrl[0] = 1.0f;
    else if (y < safeRow) ctrl[0] = 0.0f;
    else ctrl[0] = 3.0f;
}
