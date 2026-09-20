#include "../env.h"

typedef struct { uint32_t goalPeriod; } hallKnobs;

OAK_CONSTANT hallKnobs HALL_KNOBS[2] = {
    { 0u },
    { 10000u }
};


OAK_CONSTANT GridWalls HALL_WALLS = { 11, 11, 1, 6, { 5, 5, 5, 1, 3, 9 }, { 1, 3, 9, 5, 5, 5 }, { 5, 5, 5, 1, 7, 9 }, { 1, 7, 9, 5, 5, 5 } };

static int hall_isWall(int x, int y) { return gridWall(&HALL_WALLS, x, y); }

static void hall_obs(uint32_t x, uint32_t y, device float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) { obs[b] = (float)((x >> b) & 1u); obs[4u + b] = (float)((y >> b) & 1u); }
    obs[8] = hall_isWall((int)x, (int)y - 1) ? 1.0f : 0.0f;
    obs[9] = hall_isWall((int)x, (int)y + 1) ? 1.0f : 0.0f;
    obs[10] = hall_isWall((int)x - 1, (int)y) ? 1.0f : 0.0f;
    obs[11] = hall_isWall((int)x + 1, (int)y) ? 1.0f : 0.0f;
    obs[12] = (x == 5u || y == 5u) ? 1.0f : 0.0f;
    obs[13] = 1.0f;
}

static void hall_goal(uint32_t seed, uint32_t epoch, device uint32_t *gx, device uint32_t *gy) {
    int x, y;
    sampleFreeCell(seed, 0xf0000u + epoch, 0u, &HALL_WALLS, -1, -1, &x, &y);
    *gx = (uint32_t)x; *gy = (uint32_t)y;
}

void hall_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = 1u; words[1] = 1u;
    hall_goal(seed, 0u, &words[2], &words[3]);
    words[4] = 0u;
}

float hall_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const hallKnobs *k = &HALL_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0] & 3u;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    if (u01(r.x) < 0.1f) action = r.y & 3u;
    int dx, dy;
    gridDelta(action, &dx, &dy);
    int x = (int)words[0] + dx, y = (int)words[1] + dy;
    if (hall_isWall(x, y)) { x = (int)words[0]; y = (int)words[1]; }
    float reward = 0.0f;
    *done = 0u;
    if (k->goalPeriod != 0u && step / k->goalPeriod != words[4]) { words[4] = step / k->goalPeriod; hall_goal(seed, words[4], &words[2], &words[3]); }
    if ((uint32_t)x == words[2] && (uint32_t)y == words[3]) {
        reward = 1.0f;
        sampleFreeCell(seed, step, 0u, &HALL_WALLS, (int)words[2], (int)words[3], &x, &y);
        *done = 1u;
    }
    words[0] = (uint32_t)x; words[1] = (uint32_t)y;
    hall_obs(words[0], words[1], obs);
    return reward;
}

void hall_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    hall_obs(words[0], words[1], obs);
}

void hall_pose(uint32_t instance, device const uint32_t *words, thread const float *ctrl, thread float *pose) {
    (void)instance; (void)ctrl;
    poseZero(pose);
    for (int i = 0; i < 4; i++) pose[i] = (float)words[i];
}

void hall_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    int d = gridFirstStep(&HALL_WALLS, (int)words[0], (int)words[1], (int)words[2], (int)words[3]);
    ctrl[0] = (float)(d < 0 ? 0 : d);
}
