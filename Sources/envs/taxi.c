#include "../env.h"

#define TAXI_SIDE 5
#define TAXI_DEPOTS 4u

OAK_CONSTANT int TAXI_DEPOT_X[TAXI_DEPOTS] = { 0, 4, 0, 3 };
OAK_CONSTANT int TAXI_DEPOT_Y[TAXI_DEPOTS] = { 0, 0, 4, 4 };

static int taxi_blocked(int x, int y, int nx, int ny) {
    if (nx < 0 || ny < 0 || nx >= TAXI_SIDE || ny >= TAXI_SIDE) return 1;
    if (ny != y) return 0;
    int lo = x < nx ? x : nx;
    if (y <= 1 && lo == 1) return 1;
    if (y >= 3 && (lo == 0 || lo == 2)) return 1;
    return 0;
}

static void taxi_obs(const device uint32_t *words, device float *obs) {
    for (uint32_t b = 0u; b < 3u; b++) { obs[b] = (float)((words[0] >> b) & 1u); obs[3u + b] = (float)((words[1] >> b) & 1u); }
    for (uint32_t d = 0u; d <= TAXI_DEPOTS; d++) obs[6u + d] = words[2] == d ? 1.0f : 0.0f;
    for (uint32_t d = 0u; d < TAXI_DEPOTS; d++) obs[11u + d] = words[3] == d ? 1.0f : 0.0f;
    obs[15] = 1.0f;
}

static void taxi_task(uint32_t seed, uint32_t index, device uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 5u, 0u, 0u);
    words[2] = r.x % TAXI_DEPOTS;
    words[3] = (words[2] + 1u + r.y % (TAXI_DEPOTS - 1u)) % TAXI_DEPOTS;
}

void taxi_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = 2u; words[1] = 2u;
    taxi_task(seed, 0u, words);
}

float taxi_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    uint32_t action = (uint32_t)ctrl[0] % 6u;
    int x = (int)words[0], y = (int)words[1];
    float reward = -1.0f;
    *done = 0u;
    if (action < 4u) {
        int dx, dy;
        gridDelta(action, &dx, &dy);
        if (!taxi_blocked(x, y, x + dx, y + dy)) { x += dx; y += dy; }
    } else if (action == 4u) {
        if (words[2] < TAXI_DEPOTS && TAXI_DEPOT_X[words[2]] == x && TAXI_DEPOT_Y[words[2]] == y) words[2] = TAXI_DEPOTS;
        else reward = -10.0f;
    } else {
        if (words[2] == TAXI_DEPOTS && TAXI_DEPOT_X[words[3]] == x && TAXI_DEPOT_Y[words[3]] == y) { reward = 20.0f; taxi_task(seed, step, words); *done = 1u; }
        else reward = -10.0f;
    }
    words[0] = (uint32_t)x; words[1] = (uint32_t)y;
    taxi_obs(words, obs);
    return reward;
}

void taxi_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    taxi_obs(words, obs);
}

void taxi_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    int x = (int)words[0], y = (int)words[1];
    uint32_t target = words[2] < TAXI_DEPOTS ? words[2] : words[3];
    int tx = TAXI_DEPOT_X[target], ty = TAXI_DEPOT_Y[target];
    if (x == tx && y == ty) { ctrl[0] = words[2] < TAXI_DEPOTS ? 4.0f : 5.0f; return; }
    uint8_t firstStep[TAXI_SIDE * TAXI_SIDE];
    uint8_t queue[TAXI_SIDE * TAXI_SIDE];
    for (int i = 0; i < TAXI_SIDE * TAXI_SIDE; i++) firstStep[i] = 255u;
    firstStep[y * TAXI_SIDE + x] = 254u;
    int head = 0, tail = 0;
    queue[tail++] = (uint8_t)(y * TAXI_SIDE + x);
    while (head < tail) {
        int cell = queue[head++], cx = cell % TAXI_SIDE, cy = cell / TAXI_SIDE;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            int nx = cx + dx, ny = cy + dy;
            if (taxi_blocked(cx, cy, nx, ny) || firstStep[ny * TAXI_SIDE + nx] != 255u) continue;
            firstStep[ny * TAXI_SIDE + nx] = firstStep[cell] == 254u ? (uint8_t)d : firstStep[cell];
            if (nx == tx && ny == ty) { ctrl[0] = (float)firstStep[ny * TAXI_SIDE + nx]; return; }
            queue[tail++] = (uint8_t)(ny * TAXI_SIDE + nx);
        }
    }
    ctrl[0] = 0.0f;
}
