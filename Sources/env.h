#ifndef OAK_ENV_H
#define OAK_ENV_H
#include "rng.h"

#define OAK_ENV_OBS_MAX 64u
#define OAK_ENV_ACT_MAX 6u
#define OAK_ENV_WORDS_MAX 40u
#define OAK_ENV_TABLE_MAX 256u
#define OAK_POSE_LEN 9u

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

static inline int gridWall(const GridWalls *g, int x, int y) {
    if (x < 0 || y < 0 || x >= g->width || y >= g->height) return 1;
    if (g->border && (x == 0 || y == 0 || x == g->width - 1 || y == g->height - 1)) return 1;
    for (int i = 0; i < g->segments; i++)
        if (x >= g->x0[i] && x <= g->x1[i] && y >= g->y0[i] && y <= g->y1[i]) return 1;
    return 0;
}

static inline void sampleFreeCell(uint32_t laneSeed, uint32_t resampleIndex, uint32_t sampleBlock, const GridWalls *g,
                                  int excludeX, int excludeY, int *outX, int *outY) {
    for (;;) {
        u4 raw = philox4x32(laneSeed, ENV_RNG_STREAM, resampleIndex, sampleBlock, 0u, 0u);
        uint32_t philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
        for (int wi = 0; wi < 4; wi++) {
            uint32_t idx = (uint32_t)(u01(philoxWords[wi]) * 81.0f);
            int candidateX = 1 + (int)(idx % 9u), candidateY = 1 + (int)(idx / 9u);
            if (!gridWall(g, candidateX, candidateY) && (candidateX != excludeX || candidateY != excludeY)) {
                *outX = candidateX;
                *outY = candidateY;
                return;
            }
        }
        sampleBlock += 2u;
    }
}

#define GRID_BFS_SIDE 32

static inline int gridFirstStep(const GridWalls *g, int x, int y, int goalX, int goalY) {
    if (x == goalX && y == goalY) return -1;
    uint8_t firstStep[GRID_BFS_SIDE * GRID_BFS_SIDE];
    uint16_t queue[GRID_BFS_SIDE * GRID_BFS_SIDE];
    for (int i = 0; i < GRID_BFS_SIDE * GRID_BFS_SIDE; i++) firstStep[i] = 255u;
    int head = 0, tail = 0;
    firstStep[y * GRID_BFS_SIDE + x] = 254u;
    queue[tail++] = (uint16_t)(y * GRID_BFS_SIDE + x);
    while (head < tail) {
        int cell = queue[head++], cx = cell % GRID_BFS_SIDE, cy = cell / GRID_BFS_SIDE;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            int nx = cx + dx, ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= GRID_BFS_SIDE || ny >= GRID_BFS_SIDE) continue;
            int next = ny * GRID_BFS_SIDE + nx;
            if (firstStep[next] != 255u || gridWall(g, nx, ny)) continue;
            firstStep[next] = firstStep[cell] == 254u ? (uint8_t)d : firstStep[cell];
            if (nx == goalX && ny == goalY) return firstStep[next];
            queue[tail++] = (uint16_t)next;
        }
    }
    return -1;
}

void env_reset(uint32_t env, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps);
float env_step(uint32_t env, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared);
void env_observe(uint32_t env, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared);
void env_pose(uint32_t env, const uint32_t *words, const float *ctrl, float *pose);
void env_oracle(uint32_t env, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared);
uint32_t env_extraWords(uint32_t env, uint32_t steps);
uint32_t env_obs(uint32_t env);
uint32_t env_actions(uint32_t env);
uint32_t env_box(uint32_t env);
uint32_t jbw_sharedWords(void);
void jbw_buildShared(float *shared);

typedef struct {
    const char *name, *family;
    uint32_t instance;
    uint32_t id, obs, actions, box, seedStart, seeds, poseLen;
} EnvSpec;
extern const uint32_t OAK_ENV_TABLE_COUNT;
const EnvSpec *env_at(uint32_t index);
const EnvSpec *env_named(const char *name);
void env_checkTable(void);

static inline void poseZero(float *pose) { for (uint32_t i = 0u; i < OAK_POSE_LEN; i++) pose[i] = 0.0f; }

#endif
