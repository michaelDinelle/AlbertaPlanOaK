#include "../env.h"

#define TAXI_SIDE 5
#define TAXI_DEPOTS 4u

static const int TAXI_DEPOT_X[TAXI_DEPOTS] = { 0, 4, 0, 3 };
static const int TAXI_DEPOT_Y[TAXI_DEPOTS] = { 0, 0, 4, 4 };

enum { TAXI_CELLS = 8 * TAXI_SIDE, TAXI_ROWS = 6 * TAXI_CELLS / 4, TAXI_ROW = 8, TAXI_TASK = TAXI_ROWS + TAXI_CELLS * TAXI_ROW, TAXI_TASK_ROW = 16, TAXI_ROUTES = TAXI_TASK + (TAXI_DEPOTS + 1u) * TAXI_DEPOTS * TAXI_TASK_ROW };

static int taxi_blocked(int x, int y, int nx, int ny) {
    if (nx < 0 || ny < 0 || nx >= TAXI_SIDE || ny >= TAXI_SIDE) return 1;
    if (ny != y) return 0;
    int lo = x < nx ? x : nx;
    if (y <= 1 && lo == 1) return 1;
    if (y >= 3 && (lo == 0 || lo == 2)) return 1;
    return 0;
}

static void taxi_obs(const uint32_t *words, float *obs) {
    for (uint32_t b = 0u; b < 3u; b++) { obs[b] = (float)((words[0] >> b) & 1u); obs[3u + b] = (float)((words[1] >> b) & 1u); }
    for (uint32_t d = 0u; d <= TAXI_DEPOTS; d++) obs[6u + d] = words[2] == d ? 1.0f : 0.0f;
    for (uint32_t d = 0u; d < TAXI_DEPOTS; d++) obs[11u + d] = words[3] == d ? 1.0f : 0.0f;
    obs[15] = 1.0f;
}

static void taxi_show(const uint32_t *extra, uint32_t cell, uint32_t passenger, uint32_t destination, float *obs) {
    __builtin_memcpy(obs, extra + TAXI_ROWS + cell * TAXI_ROW, 6u * sizeof(float));
    __builtin_memcpy(obs + 6, extra + TAXI_TASK + (passenger * TAXI_DEPOTS + destination) * TAXI_TASK_ROW, 10u * sizeof(float));
}

static void taxi_task(uint32_t seed, uint32_t index, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 5u, 0u, 0u);
    words[2] = r.x % TAXI_DEPOTS;
    words[3] = (words[2] + 1u + r.y % (TAXI_DEPOTS - 1u)) % TAXI_DEPOTS;
}

uint32_t taxi_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return TAXI_ROUTES + 1u + TAXI_DEPOTS * TAXI_CELLS / 4u;
}

void taxi_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    uint8_t *next = (uint8_t *)extra;
    for (uint32_t passenger = 0u; passenger <= TAXI_DEPOTS; passenger++)
        for (uint32_t destination = 0u; destination < TAXI_DEPOTS; destination++) {
            uint32_t task[4] = { 0u, 0u, passenger, destination };
            float row[16];
            taxi_obs(task, row);
            __builtin_memcpy(extra + TAXI_TASK + (passenger * TAXI_DEPOTS + destination) * TAXI_TASK_ROW, row + 6, 10u * sizeof(float));
        }
    for (int y = 0; y < TAXI_SIDE; y++)
        for (int x = 0; x < TAXI_SIDE; x++) {
            uint32_t cell[4] = { (uint32_t)x, (uint32_t)y, 0u, 0u };
            float row[16];
            taxi_obs(cell, row);
            __builtin_memcpy(extra + TAXI_ROWS + (uint32_t)(x | y << 3) * TAXI_ROW, row, 6u * sizeof(float));
            for (uint32_t action = 0u; action < 6u; action++) {
                int dx = 0, dy = 0;
                if (action < 4u) gridDelta(action, &dx, &dy);
                if (taxi_blocked(x, y, x + dx, y + dy)) { dx = 0; dy = 0; }
                next[action * TAXI_CELLS + (uint32_t)(x | y << 3)] = (uint8_t)((x + dx) | (y + dy) << 3);
            }
        }
    words[0] = 2u; words[1] = 2u;
    taxi_task(seed, 0u, words);
}

static __attribute__((noinline)) float taxi_deliver(uint32_t seed, uint32_t step, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    taxi_task(seed, step, words);
    *done = 1u;
    taxi_show(extra, words[0] | words[1] << 3, words[2], words[3], obs);
    return 20.0f;
}

float taxi_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance;
    uint32_t action = (uint32_t)ctrl[0] % 6u;
    uint32_t x = words[0], y = words[1], passenger = words[2], destination = words[3], target = passenger < TAXI_DEPOTS ? passenger : destination;
    uint32_t there = (TAXI_DEPOT_X[target] == (int)x) & (TAXI_DEPOT_Y[target] == (int)y);
    uint32_t pickup = (action == 4u) & (passenger < TAXI_DEPOTS) & there;
    if ((action == 5u) & (passenger == TAXI_DEPOTS) & there) return taxi_deliver(seed, step, words, extra, obs, done);
    const uint8_t *next = (const uint8_t *)extra + action * TAXI_CELLS;
    uint32_t cell = next[x | y << 3];
    passenger = pickup ? TAXI_DEPOTS : passenger;
    words[0] = cell & 7u; words[1] = cell >> 3; words[2] = passenger;
    *done = 0u;
    taxi_show(extra, cell, passenger, destination, obs);
    return action < 4u || pickup ? -1.0f : -10.0f;
}

void taxi_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step;
    taxi_show(extra, words[0] | words[1] << 3, words[2], words[3], obs);
}

void taxi_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    uint8_t *routes = (uint8_t *)(extra + TAXI_ROUTES + 1);
    if (extra[TAXI_ROUTES] == 0u) {
        for (uint32_t depot = 0u; depot < TAXI_DEPOTS; depot++) {
            uint8_t *distance = routes + depot * TAXI_CELLS, queue[TAXI_CELLS];
            __builtin_memset(distance, 255, TAXI_CELLS);
            int head = 0, tail = 0;
            distance[TAXI_DEPOT_X[depot] | TAXI_DEPOT_Y[depot] << 3] = 0u;
            queue[tail++] = (uint8_t)(TAXI_DEPOT_X[depot] | TAXI_DEPOT_Y[depot] << 3);
            while (head < tail) {
                int cell = queue[head++], cx = cell & 7, cy = cell >> 3;
                for (uint32_t d = 0u; d < 4u; d++) {
                    int dx, dy;
                    gridDelta(d, &dx, &dy);
                    int nx = cx + dx, ny = cy + dy;
                    if (taxi_blocked(cx, cy, nx, ny) || distance[nx | ny << 3] != 255u) continue;
                    distance[nx | ny << 3] = (uint8_t)(distance[cell] + 1u);
                    queue[tail++] = (uint8_t)(nx | ny << 3);
                }
            }
        }
        extra[TAXI_ROUTES] = 1u;
    }
    int x = (int)words[0], y = (int)words[1];
    uint32_t target = words[2] < TAXI_DEPOTS ? words[2] : words[3];
    int tx = TAXI_DEPOT_X[target], ty = TAXI_DEPOT_Y[target];
    if (x == tx && y == ty) { ctrl[0] = words[2] < TAXI_DEPOTS ? 4.0f : 5.0f; return; }
    const uint8_t *distance = routes + target * TAXI_CELLS;
    int best = 255, d = -1;
    for (int a = 0; a < 4; a++) {
        int dx, dy;
        gridDelta((uint32_t)a, &dx, &dy);
        if (taxi_blocked(x, y, x + dx, y + dy)) continue;
        int far = distance[(x + dx) | (y + dy) << 3];
        if (far < best) { best = far; d = a; }
    }
    ctrl[0] = (float)(d < 0 ? 0 : d);
}
