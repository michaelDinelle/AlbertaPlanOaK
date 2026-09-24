#ifndef OAK_ENV_H
#define OAK_ENV_H
#include "rng.h"

#define OAK_ENV_OBS_MAX 64u
#define OAK_ENV_ACT_MAX 6u
#define OAK_ENV_WORDS_MAX 40u
#define OAK_ENV_TABLE_MAX 256u
#define OAK_ENV_VIEW_MAX 1024u
#define OAK_ENV_SEEDS 32u

static inline float wrapPi(float angle) {
    return angle - 2.0f * OAK_PI * floorf((angle + OAK_PI) / (2.0f * OAK_PI));
}

static inline void gridDelta(uint32_t direction, int *dx, int *dy) {
    switch (direction & 3u) {
        case 0u: *dx = 0; *dy = -1; return;
        case 1u: *dx = 0; *dy = 1; return;
        case 2u: *dx = -1; *dy = 0; return;
        default: *dx = 1; *dy = 0; return;
    }
}

#define GRID_SEGMENTS 8

typedef struct {
    int width, height, border, segments;
    int x0[GRID_SEGMENTS], y0[GRID_SEGMENTS], x1[GRID_SEGMENTS], y1[GRID_SEGMENTS];
} GridWalls;

typedef int (*GridBlocked)(const void *grid, int x, int y);

static inline int gridWall(const GridWalls *g, int x, int y) {
    if (x < 0 || y < 0 || x >= g->width || y >= g->height) return 1;
    if (g->border && (x == 0 || y == 0 || x == g->width - 1 || y == g->height - 1)) return 1;
    for (int i = 0; i < g->segments; i++)
        if (x >= g->x0[i] && x <= g->x1[i] && y >= g->y0[i] && y <= g->y1[i]) return 1;
    return 0;
}

static inline int gridWallBlocked(const void *grid, int x, int y) { return gridWall((const GridWalls *)grid, x, y); }

static inline void sampleFreeCell(uint32_t laneSeed, uint32_t resampleIndex, uint32_t sampleBlock, GridBlocked blocked, const void *grid,
                                  uint32_t interior, int excludeX, int excludeY, int *outX, int *outY) {
    for (;;) {
        u4 raw = philox4x32(laneSeed, ENV_RNG_STREAM, resampleIndex, sampleBlock, 0u, 0u);
        uint32_t philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
        for (int wi = 0; wi < 4; wi++) {
            uint32_t idx = (uint32_t)(u01(philoxWords[wi]) * (float)(interior * interior));
            int candidateX = 1 + (int)(idx % interior), candidateY = 1 + (int)(idx / interior);
            if (!blocked(grid, candidateX, candidateY) && (candidateX != excludeX || candidateY != excludeY)) {
                *outX = candidateX;
                *outY = candidateY;
                return;
            }
        }
        sampleBlock += 2u;
    }
}

static inline void gridDistance(GridBlocked blocked, const void *grid, int goalX, int goalY, uint8_t *distance) {
    uint8_t queue[256];
    int head = 0, tail = 0, goal = (goalX + 1) | (goalY + 1) << 4;
    for (int i = 0; i < 256; i++) distance[i] = 255u;
    distance[goal] = 0u;
    queue[tail++] = (uint8_t)goal;
    while (head < tail) {
        int cell = queue[head++], x = (cell & 15) - 1, y = (cell >> 4) - 1;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            int next = (x + dx + 1) | (y + dy + 1) << 4;
            if (blocked(grid, x + dx, y + dy) || distance[next] != 255u) continue;
            distance[next] = (uint8_t)(distance[cell] + 1u);
            queue[tail++] = (uint8_t)next;
        }
    }
}

static inline int gridDescend(const uint8_t *distance, int x, int y) {
    if (distance[(x + 1) | (y + 1) << 4] == 0u) return -1;
    int best = 255, direction = -1;
    for (int d = 0; d < 4; d++) {
        int dx, dy;
        gridDelta((uint32_t)d, &dx, &dy);
        int far = distance[(x + dx + 1) | (y + dy + 1) << 4];
        if (far < best) { best = far; direction = d; }
    }
    return direction;
}

enum { ENV_BATCH = 4 };

static inline void batchReset(uint32_t *tag) { *tag = 1u - ENV_BATCH; }

static inline uint32_t batchSlot(uint32_t *tag, uint32_t step) {
    uint32_t slot = step - *tag;
    if (slot < ENV_BATCH) return slot;
    if (slot != ENV_BATCH) __builtin_trap();
    *tag = step;
    return ENV_BATCH;
}

enum { LEAGUE_MEMBERS = 8 };

static inline uint32_t leaguePick(const float *winRate, uint32_t word) {
    float weight[LEAGUE_MEMBERS], total = 0.0f;
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) {
        weight[i] = (1.0f - winRate[i]) * (1.0f - winRate[i]);
        total += weight[i];
    }
    float target = u01(word) * total;
    for (uint32_t i = 0u; i + 1u < LEAGUE_MEMBERS; i++) {
        if (target < weight[i]) return i;
        target -= weight[i];
    }
    return LEAGUE_MEMBERS - 1u;
}

static inline void leagueScore(float *winRate, uint32_t member, float outcome) { winRate[member] += 0.125f * (outcome - winRate[member]); }

static inline void leagueRenew(float *winRate, float *theta, uint32_t dims, const float *low, const float *high, uint32_t seed, uint32_t game) {
    uint32_t weakest = 0u, strongest = 0u;
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++) {
        if (winRate[i] > winRate[weakest]) weakest = i;
        if (winRate[i] < winRate[strongest]) strongest = i;
    }
    if (weakest == strongest) return;
    for (uint32_t d = 0u; d < dims; d += 4u) {
        u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)game, d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
        float4 n = normal4(r.x, r.y);
        for (uint32_t i = 0u; i < 4u && d + i < dims; i++) {
            float value = theta[strongest * dims + d + i] + 0.1f * (high[d + i] - low[d + i]) * n[i];
            theta[weakest * dims + d + i] = fminf(fmaxf(value, low[d + i]), high[d + i]);
        }
    }
    winRate[weakest] = winRate[strongest];
}

void env_reset(uint32_t env, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps);
float env_step(uint32_t env, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared);
void env_observe(uint32_t env, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared);
void env_oracle(uint32_t env, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared);
uint32_t env_extraWords(uint32_t env, uint32_t steps);
uint32_t env_obs(uint32_t env);
uint32_t env_actions(uint32_t env);
uint32_t env_box(uint32_t env);
uint64_t env_sharedWords(void);
uint64_t env_sharedOffset(uint32_t env);
uint32_t env_view(uint32_t env, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);

typedef struct {
    const char *name, *family, *category;
    uint32_t instance;
    uint32_t id, obs, actions, box, seedStart, words;
} EnvSpec;
extern const uint32_t OAK_ENV_TABLE_COUNT;
const EnvSpec *env_at(uint32_t index);
const EnvSpec *env_named(const char *name);
void env_buildShared(float *shared, const EnvSpec *const *specs, uint32_t count);
void env_checkTable(void);

#endif
