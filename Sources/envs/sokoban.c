#include "../env.h"

#define SOKOBAN_SIDE 5
#define SOKOBAN_CELLS (SOKOBAN_SIDE * SOKOBAN_SIDE)
#define SOKOBAN_PATIENCE 200u

static uint32_t sokoban_pack(int ax, int ay, int bx, int by) { return (uint32_t)(ax + SOKOBAN_SIDE * ay) * SOKOBAN_CELLS + (uint32_t)(bx + SOKOBAN_SIDE * by); }

static int sokoban_move(thread int *ax, thread int *ay, thread int *bx, thread int *by, uint32_t direction) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    int nx = *ax + dx, ny = *ay + dy;
    if (nx < 0 || ny < 0 || nx >= SOKOBAN_SIDE || ny >= SOKOBAN_SIDE) return 0;
    if (nx == *bx && ny == *by) {
        int px = *bx + dx, py = *by + dy;
        if (px < 0 || py < 0 || px >= SOKOBAN_SIDE || py >= SOKOBAN_SIDE) return 0;
        *bx = px; *by = py;
    }
    *ax = nx; *ay = ny;
    return 1;
}

static void sokoban_puzzle(uint32_t seed, uint32_t index, device uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 7u, 0u, 0u);
    uint32_t box = 6u + (r.x % 3u) + SOKOBAN_SIDE * (r.y % 3u);
    uint32_t target = r.z % SOKOBAN_CELLS;
    while (target == box) target = (target + 1u) % SOKOBAN_CELLS;
    uint32_t agent = r.w % SOKOBAN_CELLS;
    while (agent == box || agent == target) agent = (agent + 1u) % SOKOBAN_CELLS;
    words[0] = agent % SOKOBAN_SIDE; words[1] = agent / SOKOBAN_SIDE;
    words[2] = box % SOKOBAN_SIDE; words[3] = box / SOKOBAN_SIDE;
    words[4] = target % SOKOBAN_SIDE; words[5] = target / SOKOBAN_SIDE;
    words[6] = 0u;
}

static void sokoban_obs(const device uint32_t *words, device float *obs) {
    for (uint32_t i = 0u; i < 6u; i++) obs[i] = (float)words[i] / (float)(SOKOBAN_SIDE - 1);
    obs[6] = 1.0f;
}

void sokoban_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    sokoban_puzzle(seed, 0u, words);
}

float sokoban_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    int ax = (int)words[0], ay = (int)words[1], bx = (int)words[2], by = (int)words[3];
    sokoban_move(&ax, &ay, &bx, &by, (uint32_t)ctrl[0] & 3u);
    words[0] = (uint32_t)ax; words[1] = (uint32_t)ay; words[2] = (uint32_t)bx; words[3] = (uint32_t)by;
    words[6] += 1u;
    float reward = 0.0f;
    *done = 0u;
    if (words[2] == words[4] && words[3] == words[5]) { reward = 1.0f; sokoban_puzzle(seed, step, words); *done = 1u; }
    else if (words[6] >= SOKOBAN_PATIENCE) sokoban_puzzle(seed, step, words);
    sokoban_obs(words, obs);
    return reward;
}

void sokoban_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    sokoban_obs(words, obs);
}

void sokoban_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    enum { STATES = SOKOBAN_CELLS * SOKOBAN_CELLS };
    uint8_t firstStep[STATES];
    uint16_t queue[STATES];
    for (int i = 0; i < STATES; i++) firstStep[i] = 255u;
    uint32_t start = sokoban_pack((int)words[0], (int)words[1], (int)words[2], (int)words[3]);
    firstStep[start] = 254u;
    int head = 0, tail = 0;
    queue[tail++] = (uint16_t)start;
    int tx = (int)words[4], ty = (int)words[5];
    while (head < tail) {
        uint32_t state = queue[head++];
        int ax = (int)(state / SOKOBAN_CELLS) % SOKOBAN_SIDE, ay = (int)(state / SOKOBAN_CELLS) / SOKOBAN_SIDE;
        int bx = (int)(state % SOKOBAN_CELLS) % SOKOBAN_SIDE, by = (int)(state % SOKOBAN_CELLS) / SOKOBAN_SIDE;
        for (uint32_t d = 0u; d < 4u; d++) {
            int nax = ax, nay = ay, nbx = bx, nby = by;
            if (!sokoban_move(&nax, &nay, &nbx, &nby, d)) continue;
            uint32_t next = sokoban_pack(nax, nay, nbx, nby);
            if (firstStep[next] != 255u) continue;
            firstStep[next] = firstStep[state] == 254u ? (uint8_t)d : firstStep[state];
            if (nbx == tx && nby == ty) { ctrl[0] = (float)firstStep[next]; return; }
            queue[tail++] = (uint16_t)next;
        }
    }
    ctrl[0] = 0.0f;
}
