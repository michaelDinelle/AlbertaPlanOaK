#include "../env.h"

typedef struct { uint32_t cells, layout, view, dash; } mazeKnobs;

static const mazeKnobs MAZE_KNOBS[4] = {
    { 5u, 0u, 0u, 0u },
    { 15u, 0u, 0u, 0u },
    { 5u, 1u, 1u, 0u },
    { 5u, 2u, 2u, 1u }
};

enum { MAZE_CELLS_MAX = 15, MAZE_SIDE_MAX = 2 * MAZE_CELLS_MAX - 1, MAZE_STATES_MAX = MAZE_SIDE_MAX * MAZE_SIDE_MAX, MAZE_ROW = 16, MAZE_GOAL = 4, MAZE_PAGE = 1024 };

typedef struct { const mazeKnobs *k; const uint32_t *words; } mazeGrid;

static uint32_t maze_side(const mazeKnobs *k) { return 2u * k->cells - 1u; }
static uint32_t maze_shift(const mazeKnobs *k) { return 32u - (uint32_t)__builtin_clz(maze_side(k) + 1u); }
static uint32_t maze_codes(const mazeKnobs *k) { return (maze_side(k) + 2u) << maze_shift(k); }
static uint32_t maze_slots(const mazeKnobs *k) { return k->dash ? 16u : 4u; }
static uint32_t maze_states(const mazeKnobs *k) { return maze_side(k) * maze_side(k) * (k->dash ? 4u : 1u); }
static uint32_t maze_stride(const mazeKnobs *k) { return 2u << maze_shift(k); }
static uint32_t maze_reach(const mazeKnobs *k) { return MAZE_GOAL * (maze_stride(k) + 1u) * (maze_side(k) - 1u); }
static uint32_t maze_goals(const mazeKnobs *k) { return (maze_codes(k) * maze_slots(k) / 2u + maze_reach(k) + MAZE_PAGE - 1u) / MAZE_PAGE * MAZE_PAGE; }
static uint32_t maze_rows(const mazeKnobs *k) { return maze_goals(k) + maze_reach(k) + MAZE_ROW; }
static uint32_t maze_route(const mazeKnobs *k) { return maze_rows(k) + MAZE_ROW * maze_codes(k); }
static uint32_t maze_goalSlot(const mazeKnobs *k) { return k->view == 1u ? 10u : k->view == 2u ? 6u : 2u; }

static int maze_wall(const mazeKnobs *k, const uint32_t *words, int x, int y) {
    int side = (int)maze_side(k);
    if (x <= 0 || x > side || y <= 0 || y > side) return 1;
    if (k->layout == 1u) return (y == 3 && x != 8) || (y == 7 && x != 2);
    if (k->layout == 2u) return (x == 3 && y != 9) || (x == 7 && y != 1);
    uint32_t idx = (uint32_t)(x - 1) + (uint32_t)side * (uint32_t)(y - 1);
    return (words[7u + (idx >> 5u)] >> (idx & 31u)) & 1u;
}

static int maze_blocked(const void *grid, int x, int y) {
    const mazeGrid *g = (const mazeGrid *)grid;
    return maze_wall(g->k, g->words, x, y);
}

static void maze_carve(const mazeKnobs *k, uint32_t seed, uint32_t *walls) {
    uint32_t n = k->cells, side = maze_side(k), bits = side * side;
    for (uint32_t i = 0u; i < (bits + 31u) / 32u; i++) walls[i] = i + 1u < (bits + 31u) / 32u || bits % 32u == 0u ? 0xffffffffu : (1u << (bits % 32u)) - 1u;
    for (uint32_t j = 0u; j < n; j++)
        for (uint32_t i = 0u; i < n; i++) {
            uint32_t idx = 2u * i + side * 2u * j;
            walls[idx >> 5u] &= ~(1u << (idx & 31u));
        }
    uint32_t visited[(MAZE_CELLS_MAX * MAZE_CELLS_MAX + 31) / 32] = {1u};
    uint8_t stack[MAZE_CELLS_MAX * MAZE_CELLS_MAX];
    uint32_t stackSize = 1u, counter = 0u, wordIndex = 0u;
    stack[0] = 0u;
    u4 block = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    static const int DX[4] = { 1, 0, -1, 0 }, DY[4] = { 0, 1, 0, -1 };
    while (stackSize > 0u) {
        uint32_t current = stack[stackSize - 1u];
        int ci = (int)(current % n), cj = (int)(current / n);
        uint32_t choices[4], count = 0u;
        for (uint32_t d = 0u; d < 4u; d++) {
            int ni = ci + DX[d], nj = cj + DY[d];
            if (ni < 0 || ni >= (int)n || nj < 0 || nj >= (int)n) continue;
            uint32_t cell = (uint32_t)(ni + (int)n * nj);
            if (((visited[cell >> 5u] >> (cell & 31u)) & 1u) == 0u) choices[count++] = cell;
        }
        if (count == 0u) { stackSize--; continue; }
        if (wordIndex == 4u) {
            block = philox4x32(seed, RNG_STREAM_RESET, ++counter, 0u, 0u, 0u);
            wordIndex = 0u;
        }
        uint32_t drawn[4] = { block.x, block.y, block.z, block.w };
        uint32_t next = choices[drawn[wordIndex++] % count];
        uint32_t idx = ((uint32_t)ci + next % n) + side * ((uint32_t)cj + next / n);
        walls[idx >> 5u] &= ~(1u << (idx & 31u));
        visited[next >> 5u] |= 1u << (next & 31u);
        stack[stackSize++] = (uint8_t)next;
    }
}

static void maze_obs(const mazeKnobs *k, const uint32_t *words, float *obs) {
    int x = (int)words[0], y = (int)words[1], goalX = (int)words[2], goalY = (int)words[3];
    if (k->view == 1u) {
        static const int NX[8] = { 0, 1, 1, 1, 0, -1, -1, -1 }, NY[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
        for (uint32_t i = 0u; i < 8u; i++) obs[i] = maze_wall(k, words, x + NX[i], y + NY[i]) ? 1.0f : 0.0f;
        obs[8] = (float)x / 10.0f;
        obs[9] = (float)y / 10.0f;
        obs[10] = goalX > x ? 1.0f : 0.0f;
        obs[11] = goalX < x ? 1.0f : 0.0f;
        obs[12] = goalY > y ? 1.0f : 0.0f;
        obs[13] = goalY < y ? 1.0f : 0.0f;
        obs[14] = 1.0f;
        return;
    }
    if (k->view == 2u) {
        obs[0] = (float)x / 16.0f;
        obs[1] = (float)y / 16.0f;
        obs[2] = maze_wall(k, words, x, y - 1) ? 1.0f : 0.0f;
        obs[3] = maze_wall(k, words, x, y + 1) ? 1.0f : 0.0f;
        obs[4] = maze_wall(k, words, x - 1, y) ? 1.0f : 0.0f;
        obs[5] = maze_wall(k, words, x + 1, y) ? 1.0f : 0.0f;
        obs[6] = (float)(goalX - x) / 16.0f;
        obs[7] = (float)(goalY - y) / 16.0f;
        for (uint32_t bit = 0u; bit < 4u; bit++) {
            obs[8u + bit] = (float)(((uint32_t)x >> bit) & 1u);
            obs[12u + bit] = (float)(((uint32_t)y >> bit) & 1u);
        }
        return;
    }
    float span = (float)(maze_side(k) + 1u);
    obs[0] = (float)x / span;
    obs[1] = (float)y / span;
    obs[2] = (float)(goalX - x) / span;
    obs[3] = (float)(goalY - y) / span;
    obs[4] = (float)(abs(goalX - x) + abs(goalY - y)) / (float)(2u * (maze_side(k) - 1u));
    obs[5] = maze_wall(k, words, x, y - 1) ? 1.0f : 0.0f;
    obs[6] = maze_wall(k, words, x, y + 1) ? 1.0f : 0.0f;
    obs[7] = maze_wall(k, words, x - 1, y) ? 1.0f : 0.0f;
    obs[8] = maze_wall(k, words, x + 1, y) ? 1.0f : 0.0f;
    obs[9] = maze_wall(k, words, x + 1, y - 1) ? 1.0f : 0.0f;
    obs[10] = maze_wall(k, words, x - 1, y - 1) ? 1.0f : 0.0f;
    obs[11] = maze_wall(k, words, x + 1, y + 1) ? 1.0f : 0.0f;
    obs[12] = maze_wall(k, words, x - 1, y + 1) ? 1.0f : 0.0f;
    obs[13] = (4.0f - obs[5] - obs[6] - obs[7] - obs[8]) / 4.0f;
    obs[14] = fminf((float)words[4] / 256.0f, 1.0f);
    obs[15] = (x == goalX || y == goalY) ? 1.0f : 0.0f;
    obs[16] = 1.0f;
}

static int maze_move(const mazeKnobs *k, const uint32_t *words, int *x, int *y, uint32_t direction) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    if (maze_wall(k, words, *x + dx, *y + dy)) return 0;
    *x += dx; *y += dy;
    return 1;
}

void maze_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const mazeKnobs *k = &MAZE_KNOBS[instance];
    if (k->layout == 0u) maze_carve(k, seed, words + 7);
    words[0] = 1u; words[1] = 1u;
    words[2] = maze_side(k); words[3] = maze_side(k);
    words[6] = 1u;
    uint32_t shift = maze_shift(k), side = maze_side(k), probe[OAK_ENV_WORDS_MAX];
    uint16_t *moves = (uint16_t *)extra;
    __builtin_memcpy(probe, words, sizeof probe);
    for (uint32_t y = 0u; y <= side + 1u; y++)
        for (uint32_t x = 0u; x <= side + 1u; x++) {
            uint32_t cell = x | y << shift;
            float row[OAK_ENV_OBS_MAX] = { 0.0f };
            probe[0] = x; probe[1] = y;
            maze_obs(k, probe, row);
            __builtin_memcpy(extra + maze_rows(k) + MAZE_ROW * cell, row, MAZE_ROW * sizeof(float));
            for (uint32_t slot = 0u; slot < maze_slots(k); slot++) {
                int nx = (int)x, ny = (int)y;
                if (slot < 4u) maze_move(k, words, &nx, &ny, slot);
                else if (slot > 4u && slot < 9u && maze_move(k, words, &nx, &ny, slot - 5u)) maze_move(k, words, &nx, &ny, slot - 5u);
                moves[maze_slots(k) * cell + slot] = (uint16_t)((uint32_t)nx | (uint32_t)ny << shift);
            }
        }
    for (int dy = 1 - (int)side; dy < (int)side; dy++)
        for (int dx = 1 - (int)side; dx < (int)side; dx++) {
            float obs[OAK_ENV_OBS_MAX] = { 0.0f };
            probe[0] = side; probe[1] = side; probe[2] = side + (uint32_t)dx; probe[3] = side + (uint32_t)dy;
            maze_obs(k, probe, obs);
            __builtin_memcpy(extra + maze_goals(k) + MAZE_GOAL * (dx + (int)maze_stride(k) * dy), obs + maze_goalSlot(k), MAZE_GOAL * sizeof(float));
        }
}

static inline __attribute__((always_inline)) void maze_view(const mazeKnobs *k, uint32_t x, uint32_t y, uint32_t goalX, uint32_t goalY, uint32_t clock, const uint32_t *extra, float *obs) {
    const uint32_t *row = extra + maze_rows(k) + MAZE_ROW * (x | y << maze_shift(k));
    const uint32_t *goal = extra + maze_goals(k) + MAZE_GOAL * ((long)goalX - (long)x + (long)maze_stride(k) * ((long)goalY - (long)y));
    if (k->view == 1u) {
        __builtin_memcpy(obs, row, 12u * sizeof(float));
        __builtin_memcpy(obs + 10, goal, 4u * sizeof(float));
        obs[14] = 1.0f;
        return;
    }
    __builtin_memcpy(obs, row, MAZE_ROW * sizeof(float));
    if (k->view == 2u) {
        __builtin_memcpy(obs + 6, goal, 2u * sizeof(float));
        return;
    }
    __builtin_memcpy(obs + 2, goal, 3u * sizeof(float));
    obs[14] = fminf((float)clock / 256.0f, 1.0f);
    obs[15] = (x == goalX || y == goalY) ? 1.0f : 0.0f;
    obs[16] = 1.0f;
}

static void maze_see(uint32_t instance, const uint32_t *words, const uint32_t *extra, float *obs) {
    switch (instance) {
        case 0u: maze_view(&MAZE_KNOBS[0], words[0], words[1], words[2], words[3], words[4], extra, obs); return;
        case 1u: maze_view(&MAZE_KNOBS[1], words[0], words[1], words[2], words[3], words[4], extra, obs); return;
        case 2u: maze_view(&MAZE_KNOBS[2], words[0], words[1], words[2], words[3], words[4], extra, obs); return;
        case 3u: maze_view(&MAZE_KNOBS[3], words[0], words[1], words[2], words[3], words[4], extra, obs); return;
        default: __builtin_trap();
    }
}

static __attribute__((noinline, cold)) float maze_arrive(uint32_t instance, uint32_t seed, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    const mazeKnobs *k = &MAZE_KNOBS[instance];
    float reward = 0.0f;
    *done = 0u;
    if (words[0] == words[2] && words[1] == words[3]) {
        reward = 1.0f;
        *done = 1u;
        mazeGrid grid = { k, words };
        int x, y, gx, gy;
        sampleFreeCell(seed, words[5], 0u, maze_blocked, &grid, maze_side(k), 0, 0, &x, &y);
        sampleFreeCell(seed, words[5], 1u, maze_blocked, &grid, maze_side(k), x, y, &gx, &gy);
        words[0] = (uint32_t)x; words[1] = (uint32_t)y;
        words[2] = (uint32_t)gx; words[3] = (uint32_t)gy;
        words[4] = 0u; words[5] += 1u;
    }
    maze_see(instance, words, extra, obs);
    return reward;
}

static inline __attribute__((always_inline)) float maze_walk(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    const mazeKnobs *k = &MAZE_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0], shift = maze_shift(k), heading = words[6], slot = action & 3u;
    if (k->dash) {
        uint32_t chosen = action;
        if (__builtin_expect(chosen >= 6u, 0)) chosen %= 6u;
        slot = chosen == 5u ? 5u + (heading & 3u) : chosen;
        heading = chosen < 4u ? chosen : heading;
    }
    uint32_t next = ((const uint16_t *)extra)[maze_slots(k) * (words[0] | words[1] << shift) + slot];
    uint32_t x = next & ((1u << shift) - 1u), y = next >> shift, goalX = words[2], goalY = words[3], clock = words[4] + 1u;
    words[0] = x; words[1] = y; words[4] = clock;
    if (k->dash) words[6] = heading;
    if (__builtin_expect(x == goalX && y == goalY, 0)) return maze_arrive(instance, seed, words, extra, obs, done);
    *done = 0u;
    maze_view(k, x, y, goalX, goalY, clock, extra, obs);
    return 0.0f;
}

float maze_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return maze_walk(0u, seed, ctrl, words, extra, obs, done);
        case 1u: return maze_walk(1u, seed, ctrl, words, extra, obs, done);
        case 2u: return maze_walk(2u, seed, ctrl, words, extra, obs, done);
        case 3u: return maze_walk(3u, seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void maze_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    maze_see(instance, words, extra, obs);
}

static uint32_t maze_next(const mazeKnobs *k, const uint32_t *words, uint32_t state, uint32_t action) {
    uint32_t side = maze_side(k), headings = k->dash ? 4u : 1u, cell = state / headings, heading = state % headings;
    int x = 1 + (int)(cell % side), y = 1 + (int)(cell / side);
    if (action < 4u) { heading = k->dash ? action : 0u; maze_move(k, words, &x, &y, action); }
    else if (action == 5u && maze_move(k, words, &x, &y, heading)) maze_move(k, words, &x, &y, heading);
    return ((uint32_t)(x - 1) + side * (uint32_t)(y - 1)) * headings + heading;
}

uint32_t maze_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const mazeKnobs *k = &MAZE_KNOBS[instance];
    return maze_route(k) + 1u + (maze_states(k) + 1u) / 2u;
}

void maze_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const mazeKnobs *k = &MAZE_KNOBS[instance];
    uint32_t side = maze_side(k), headings = k->dash ? 4u : 1u, states = side * side * headings, actions = k->dash ? 6u : 4u;
    uint32_t *route = extra + maze_route(k);
    uint16_t *distance = (uint16_t *)(route + 1);
    if (route[0] != words[5] + 1u) {
        uint16_t start[MAZE_STATES_MAX + 1] = {0}, from[MAZE_STATES_MAX * 4], queue[MAZE_STATES_MAX], fill[MAZE_STATES_MAX];
        for (uint32_t s = 0u; s < states; s++)
            for (uint32_t a = 0u; a < actions; a++) start[maze_next(k, words, s, a) + 1u]++;
        for (uint32_t s = 0u; s < states; s++) start[s + 1u] += start[s];
        for (uint32_t s = 0u; s < states; s++) fill[s] = start[s];
        for (uint32_t s = 0u; s < states; s++)
            for (uint32_t a = 0u; a < actions; a++) from[fill[maze_next(k, words, s, a)]++] = (uint16_t)s;
        uint32_t head = 0u, tail = 0u, goal = (words[2] - 1u) + side * (words[3] - 1u);
        for (uint32_t s = 0u; s < states; s++) distance[s] = 0xffffu;
        for (uint32_t h = 0u; h < headings; h++) { distance[goal * headings + h] = 0u; queue[tail++] = (uint16_t)(goal * headings + h); }
        while (head < tail) {
            uint32_t t = queue[head++];
            for (uint32_t e = start[t]; e < start[t + 1u]; e++)
                if (distance[from[e]] == 0xffffu) { distance[from[e]] = (uint16_t)(distance[t] + 1u); queue[tail++] = from[e]; }
        }
        route[0] = words[5] + 1u;
    }
    uint32_t state = ((words[0] - 1u) + side * (words[1] - 1u)) * headings + (k->dash ? words[6] & 3u : 0u), best = 0u;
    for (uint32_t a = 1u; a < actions; a++)
        if (distance[maze_next(k, words, state, a)] < distance[maze_next(k, words, state, best)]) best = a;
    ctrl[0] = (float)best;
}
