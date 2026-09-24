#include "../env.h"

typedef struct { uint32_t goalPeriod, rooms; } hallKnobs;

static const hallKnobs HALL_KNOBS[3] = {
    { 0u, 0u },
    { 10000u, 0u },
    { 0u, 1u }
};


static const GridWalls HALL_WALLS = { 11, 11, 1, 6, { 5, 5, 5, 1, 3, 9 }, { 1, 3, 9, 5, 5, 5 }, { 5, 5, 5, 1, 7, 9 }, { 1, 7, 9, 5, 5, 5 } };

enum { HALL_CELLS = 16 * 11, HALL_MOVES = 4 * HALL_CELLS, HALL_CODES = 2 * HALL_MOVES, HALL_ROWS = HALL_CODES + HALL_MOVES / 4, HALL_ROW = 16, HALL_WALL_BITS = HALL_ROWS + HALL_CELLS * HALL_ROW, HALL_DRAWS = HALL_WALL_BITS + (HALL_CELLS + 31) / 32, HALL_ROUTE = HALL_DRAWS + 1 + ENV_BATCH, HALL_SPAN = 256 };

static int hall_isWall(int x, int y) { return gridWall(&HALL_WALLS, x, y); }

static int hall_blocked(const void *grid, int x, int y) {
    uint32_t cell = (uint32_t)x | (uint32_t)y << 4;
    return (uint32_t)x >= 16u || (uint32_t)y >= 11u || ((const uint32_t *)grid)[cell >> 5] >> (cell & 31u) & 1u;
}

static void hall_obs(uint32_t x, uint32_t y, float *obs) {
    for (uint32_t b = 0u; b < 4u; b++) { obs[b] = (float)((x >> b) & 1u); obs[4u + b] = (float)((y >> b) & 1u); }
    obs[8] = hall_isWall((int)x, (int)y - 1) ? 1.0f : 0.0f;
    obs[9] = hall_isWall((int)x, (int)y + 1) ? 1.0f : 0.0f;
    obs[10] = hall_isWall((int)x - 1, (int)y) ? 1.0f : 0.0f;
    obs[11] = hall_isWall((int)x + 1, (int)y) ? 1.0f : 0.0f;
    obs[12] = (x == 5u || y == 5u) ? 1.0f : 0.0f;
    obs[13] = 1.0f;
}

static const uint8_t HALL_FREE_X[68] = { 1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9, 2, 8,
                                         1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9 };
static const uint8_t HALL_FREE_Y[68] = { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
                                         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 };

static int hall_room(int x, int y) {
    if (x < 5 && y < 5) return 0;
    if (x > 5 && y < 5) return 1;
    if (x < 5 && y > 5) return 2;
    if (x > 5 && y > 5) return 3;
    return -1;
}

static void hall_roomsObs(const uint32_t *words, float *obs) {
    int x = (int)words[0], y = (int)words[1], goalX = (int)words[2], goalY = (int)words[3], room = hall_room(x, y);
    obs[0] = (float)x / 10.0f;
    obs[1] = (float)y / 10.0f;
    obs[2] = (float)(goalX - x) / 10.0f;
    obs[3] = (float)(goalY - y) / 10.0f;
    obs[4] = (float)(abs(goalX - x) + abs(goalY - y)) / 20.0f;
    for (int r = 0; r < 4; r++) obs[5 + r] = room == r ? 1.0f : 0.0f;
    obs[9] = hall_isWall(x, y - 1) ? 1.0f : 0.0f;
    obs[10] = hall_isWall(x, y + 1) ? 1.0f : 0.0f;
    obs[11] = hall_isWall(x - 1, y) ? 1.0f : 0.0f;
    obs[12] = hall_isWall(x + 1, y) ? 1.0f : 0.0f;
    obs[13] = room == -1 ? 1.0f : 0.0f;
    obs[14] = (hall_room(goalX, goalY) == room && room != -1) ? 1.0f : 0.0f;
    obs[15] = fminf((float)words[4] / 256.0f, 1.0f);
    obs[16] = 1.0f;
}

static void hall_roomsGoal(uint32_t seed, uint32_t resample, uint32_t *words) {
    for (uint32_t block = 0u;; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, resample, block, 0u, 0u);
        uint32_t drawn[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t i = 0u; i < 4u; i++) {
            uint32_t cell = (uint32_t)(u01(drawn[i]) * 68.0f);
            if (HALL_FREE_X[cell] != words[0] || HALL_FREE_Y[cell] != words[1]) {
                words[2] = HALL_FREE_X[cell];
                words[3] = HALL_FREE_Y[cell];
                return;
            }
        }
    }
}

enum { HALL_ROOMS_CELLS = HALL_MOVES / 4, HALL_ROOMS_GOALS = HALL_ROOMS_CELLS + HALL_CELLS / 4, HALL_ROOMS_ORIGIN = HALL_ROOMS_GOALS + 4 * 8 * 18, HALL_ROOMS_END = HALL_ROOMS_ORIGIN + 4 * (8 * 18 + 1) };
_Static_assert(HALL_ROOMS_END <= HALL_CODES, "");

static __attribute__((noinline, cold)) void hall_roomsTables(uint32_t *extra) {
    const uint8_t *codes = (const uint8_t *)(extra + HALL_CODES);
    uint8_t *moves = (uint8_t *)extra, *rooms = (uint8_t *)(extra + HALL_ROOMS_CELLS);
    uint32_t probe[6] = { 0u };
    for (uint32_t y = 0u; y < 11u; y++)
        for (uint32_t x = 0u; x < 11u; x++) {
            float row[17];
            probe[0] = x; probe[1] = y;
            hall_roomsObs(probe, row);
            __builtin_memcpy(extra + HALL_ROWS + (x | y << 4) * HALL_ROW, row, HALL_ROW * sizeof(float));
            rooms[x | y << 4] = (uint8_t)(hall_room((int)x, (int)y) + 1);
            for (uint32_t action = 0u; action < 4u; action++) moves[(x | y << 4) << 2 | action] = codes[action * HALL_CELLS + (x | y << 4)];
        }
    for (int dy = -8; dy <= 8; dy++)
        for (int dx = -8; dx <= 8; dx++) {
            float obs[17];
            uint32_t goal[6] = { 8u, 8u, (uint32_t)(8 + dx), (uint32_t)(8 + dy), 0u, 0u };
            hall_roomsObs(goal, obs);
            __builtin_memcpy(extra + HALL_ROOMS_ORIGIN + 4 * (dx + 17 * dy), obs + 2, 4u * sizeof(float));
        }
}

static inline __attribute__((always_inline)) void hall_roomsView(uint32_t x, uint32_t y, uint32_t goalX, uint32_t goalY, uint32_t clock, const uint32_t *extra, float *obs) {
    const uint8_t *rooms = (const uint8_t *)(extra + HALL_ROOMS_CELLS);
    uint32_t cell = x | y << 4, room = rooms[cell], shared = room == rooms[goalX | goalY << 4] && room != 0u;
    int dx = (int)goalX - (int)x, dy = (int)goalY - (int)y;
    __builtin_memcpy(obs, extra + HALL_ROWS + cell * HALL_ROW, HALL_ROW * sizeof(float));
    __builtin_memcpy(obs + 2, extra + HALL_ROOMS_ORIGIN + 4L * (dx + 17 * dy), 3u * sizeof(float));
    obs[14] = shared ? 1.0f : 0.0f;
    obs[15] = fminf((float)clock / 256.0f, 1.0f);
    obs[16] = 1.0f;
}

static __attribute__((noinline, cold)) float hall_roomsArrive(uint32_t seed, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    float reward = 0.0f;
    if (words[0] == words[2] && words[1] == words[3]) {
        reward = 1.0f;
        words[4] = 0u;
        hall_roomsGoal(seed, words[5]++, words);
    }
    *done = 0u;
    hall_roomsView(words[0], words[1], words[2], words[3], words[4], extra, obs);
    return reward;
}

static __attribute__((noinline)) float hall_rooms(uint32_t seed, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t code = ((const uint8_t *)extra)[(words[0] | words[1] << 4) << 2 | ((uint32_t)ctrl[0] & 3u)];
    uint32_t x = code & 15u, y = code >> 4, goalX = words[2], goalY = words[3], clock = words[4] + 1u;
    words[0] = x; words[1] = y; words[4] = clock;
    if (__builtin_expect(x == goalX && y == goalY, 0)) return hall_roomsArrive(seed, words, extra, obs, done);
    *done = 0u;
    hall_roomsView(x, y, goalX, goalY, clock, extra, obs);
    return 0.0f;
}

static void hall_goal(uint32_t seed, uint32_t epoch, const uint32_t *walls, uint32_t *gx, uint32_t *gy) {
    int x, y;
    sampleFreeCell(seed, 0xf0000u + epoch, 0u, hall_blocked, walls, 9u, -1, -1, &x, &y);
    *gx = (uint32_t)x; *gy = (uint32_t)y;
}

static lane4 hall_draw(uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 slip = (lane4)(r.x < u01Bound(0.1f)), steer = ((r.y & 3u) << 2u & slip) | (3u & ~slip);
    __builtin_memcpy(draws + 1, &steer, sizeof steer);
    return steer;
}

uint32_t hall_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return HALL_ROUTE + 1u + HALL_SPAN / 4u;
}

void hall_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    uint8_t *next = (uint8_t *)(extra + HALL_CODES);
    for (uint32_t y = 0u; y < 11u; y++)
        for (uint32_t x = 0u; x < 16u; x++) {
            if (hall_isWall((int)x, (int)y)) extra[HALL_WALL_BITS + ((x | y << 4) >> 5)] |= 1u << ((x | y << 4) & 31u);
            if (x >= 11u) continue;
            float row[HALL_ROW] = { 0.0f };
            hall_obs(x, y, row);
            __builtin_memcpy(extra + HALL_ROWS + (x | y << 4) * HALL_ROW, row, sizeof row);
            for (uint32_t action = 0u; action < 4u; action++) {
                int dx, dy;
                gridDelta(action, &dx, &dy);
                int nx = (int)x + dx, ny = (int)y + dy;
                if (hall_isWall(nx, ny)) { nx = (int)x; ny = (int)y; }
                uint32_t entry = action * HALL_CELLS + (x | y << 4);
                next[entry] = (uint8_t)(nx | ny << 4);
                extra[2u * entry] = (uint32_t)nx;
                extra[2u * entry + 1u] = (uint32_t)ny;
            }
        }
    batchReset(extra + HALL_DRAWS);
    words[0] = 1u; words[1] = 1u;
    if (HALL_KNOBS[instance].rooms) { hall_roomsTables(extra); words[2] = 9u; words[3] = 9u; return; }
    hall_goal(seed, 0u, extra + HALL_WALL_BITS, &words[2], &words[3]);
    words[4] = 0u;
}

static __attribute__((noinline)) float hall_arrive(const hallKnobs *k, uint32_t seed, uint32_t step, uint32_t cell, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    int x = (int)(cell & 15u), y = (int)(cell >> 4);
    float reward = 0.0f;
    *done = 0u;
    if (k->goalPeriod != 0u && step / k->goalPeriod != words[4]) { words[4] = step / k->goalPeriod; hall_goal(seed, words[4], extra + HALL_WALL_BITS, &words[2], &words[3]); }
    if ((uint32_t)x == words[2] && (uint32_t)y == words[3]) {
        reward = 1.0f;
        sampleFreeCell(seed, step, 0u, hall_blocked, extra + HALL_WALL_BITS, 9u, (int)words[2], (int)words[3], &x, &y);
        *done = 1u;
    }
    words[0] = (uint32_t)x; words[1] = (uint32_t)y;
    __builtin_memcpy(obs, extra + HALL_ROWS + (words[0] | words[1] << 4) * HALL_ROW, 14u * sizeof(float));
    return reward;
}

static inline __attribute__((always_inline)) float hall_walk(const hallKnobs *k, uint32_t seed, uint32_t step, uint32_t action, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t move = action * HALL_CELLS, cell = words[0] | words[1] << 4, code = ((const uint8_t *)(extra + HALL_CODES) + move)[cell];
    if ((k->goalPeriod != 0u && step / k->goalPeriod != words[4]) || code == (words[2] | words[3] << 4)) return hall_arrive(k, seed, step, code, words, extra, obs, done);
    __builtin_memcpy(words, extra + 2u * move + 2u * (size_t)cell, 2u * sizeof(uint32_t));
    *done = 0u;
    __builtin_memcpy(obs, extra + HALL_ROWS + code * HALL_ROW, 14u * sizeof(float));
    return 0.0f;
}

static uint32_t hall_action(const float *ctrl, uint32_t steer) { return ((uint32_t)ctrl[0] & steer & 3u) | steer >> 2u; }

static __attribute__((noinline)) float hall_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t action = hall_action(ctrl, hall_draw(seed, step, extra + HALL_DRAWS)[0]);
    switch (instance) {
        case 0u: return hall_walk(&HALL_KNOBS[0], seed, step, action, words, extra, obs, done);
        case 1u: return hall_walk(&HALL_KNOBS[1], seed, step, action, words, extra, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float hall_slip(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t slot = batchSlot(extra + HALL_DRAWS, step);
    if (slot == ENV_BATCH) return hall_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return hall_walk(&HALL_KNOBS[instance], seed, step, hall_action(ctrl, extra[HALL_DRAWS + 1u + slot]), words, extra, obs, done);
}

float hall_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return hall_slip(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return hall_slip(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return hall_rooms(seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void hall_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)step;
    if (HALL_KNOBS[instance].rooms) { hall_roomsView(words[0], words[1], words[2], words[3], words[4], extra, obs); return; }
    __builtin_memcpy(obs, extra + HALL_ROWS + (words[0] | words[1] << 4) * HALL_ROW, 14u * sizeof(float));
}

void hall_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    uint32_t goal = (words[2] + 1u) | (words[3] + 1u) << 4;
    uint8_t *distance = (uint8_t *)(extra + HALL_ROUTE + 1);
    if (extra[HALL_ROUTE] != goal) {
        gridDistance(hall_blocked, extra + HALL_WALL_BITS, (int)words[2], (int)words[3], distance);
        extra[HALL_ROUTE] = goal;
    }
    int d = gridDescend(distance, (int)words[0], (int)words[1]);
    ctrl[0] = (float)(d < 0 ? 0 : d);
}
