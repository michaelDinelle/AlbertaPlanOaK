#include <dispatch/dispatch.h>
#include <string.h>
#include "../env.h"

#define SOKOBAN_SIDE 5
#define SOKOBAN_CELLS (SOKOBAN_SIDE * SOKOBAN_SIDE)
#define SOKOBAN_PATIENCE 200u

typedef struct { uint32_t side, boxes, maxSteps, pool; } sokobanKnobs;

static const sokobanKnobs SOKOBAN_KNOBS[3] = {
    { 5u, 1u, 200u, 0u },
    { 7u, 2u, 200u, 4096u },
    { 10u, 4u, 120u, 4096u }
};

enum { SOKOBAN_STATES = SOKOBAN_CELLS * SOKOBAN_CELLS };
enum { SOKOBAN_FLOOR, SOKOBAN_TARGET, SOKOBAN_WALL = 4 };
enum { SOKOBAN_PLAYER, SOKOBAN_BOXES, SOKOBAN_PLACED, SOKOBAN_CLOCK, SOKOBAN_LEVEL };
enum { SOKOBAN_ROW = 16, SOKOBAN_TTL = 300, SOKOBAN_EXPLORED = 300000, SOKOBAN_BATCH = 16, SOKOBAN_BUDGET = 1 << 21 };

static const float SOKOBAN_TURN = 0.35f;
static const uint64_t SOKOBAN_WEST = 0x0101010101010101ull, SOKOBAN_EAST = 0x8080808080808080ull;

static uint32_t sokoban_pack(int ax, int ay, int bx, int by) { return (uint32_t)(ax + SOKOBAN_SIDE * ay) * SOKOBAN_CELLS + (uint32_t)(bx + SOKOBAN_SIDE * by); }

static int sokoban_inside(int x, int y) { return ((uint32_t)x < SOKOBAN_SIDE) & ((uint32_t)y < SOKOBAN_SIDE); }

static int sokoban_move(int *ax, int *ay, int *bx, int *by, uint32_t direction) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    int nx = *ax + dx, ny = *ay + dy, px = *bx + dx, py = *by + dy;
    int push = (nx == *bx) & (ny == *by), moved = sokoban_inside(nx, ny) & (!push | sokoban_inside(px, py));
    int agent = -moved, box = -(moved & push);
    *bx = (*bx & ~box) | (px & box); *by = (*by & ~box) | (py & box);
    *ax = (*ax & ~agent) | (nx & agent); *ay = (*ay & ~agent) | (ny & agent);
    return moved;
}

static void sokoban_puzzle(uint32_t seed, uint32_t index, uint32_t *words) {
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

static void sokoban_obs(const uint32_t *words, float *obs) {
    for (uint32_t i = 0u; i < 6u; i++) obs[i] = (float)words[i] / (float)(SOKOBAN_SIDE - 1);
    obs[6] = 1.0f;
}

static __attribute__((noinline)) float sokoban_finish(uint32_t seed, uint32_t step, uint32_t *words, float *obs, uint32_t *done) {
    float reward = 0.0f;
    *done = 0u;
    if (words[2] == words[4] && words[3] == words[5]) { reward = 1.0f; sokoban_puzzle(seed, step, words); *done = 1u; }
    else if (words[6] >= SOKOBAN_PATIENCE) sokoban_puzzle(seed, step, words);
    sokoban_obs(words, obs);
    return reward;
}

static __attribute__((noinline)) float sokoban_walk(uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    int ax = (int)words[0], ay = (int)words[1], bx = (int)words[2], by = (int)words[3];
    sokoban_move(&ax, &ay, &bx, &by, (uint32_t)ctrl[0] & 3u);
    uint32_t state[7] = { (uint32_t)ax, (uint32_t)ay, (uint32_t)bx, (uint32_t)by, words[4], words[5], words[6] + 1u };
    __builtin_memcpy(words, state, sizeof state);
    if ((state[2] == state[4] && state[3] == state[5]) || state[6] >= SOKOBAN_PATIENCE) return sokoban_finish(seed, step, words, obs, done);
    *done = 0u;
    sokoban_obs(state, obs);
    return 0.0f;
}

static void sokoban_route(int tx, int ty, uint8_t *distance) {
    uint16_t queue[SOKOBAN_STATES];
    __builtin_memset(distance, 255, SOKOBAN_STATES);
    int head = 0, tail = 0;
    for (int ay = 0; ay < SOKOBAN_SIDE; ay++)
        for (int ax = 0; ax < SOKOBAN_SIDE; ax++)
            if (ax != tx || ay != ty) {
                distance[sokoban_pack(ax, ay, tx, ty)] = 0u;
                queue[tail++] = (uint16_t)sokoban_pack(ax, ay, tx, ty);
            }
    while (head < tail) {
        uint32_t state = queue[head++];
        int ax = (int)(state / SOKOBAN_CELLS) % SOKOBAN_SIDE, ay = (int)(state / SOKOBAN_CELLS) / SOKOBAN_SIDE;
        int bx = (int)(state % SOKOBAN_CELLS) % SOKOBAN_SIDE, by = (int)(state % SOKOBAN_CELLS) / SOKOBAN_SIDE;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            int px = ax - dx, py = ay - dy;
            if (!sokoban_inside(px, py)) continue;
            uint32_t before[2] = { sokoban_pack(px, py, bx, by), sokoban_pack(px, py, ax, ay) };
            int valid[2] = { px != bx || py != by, bx == ax + dx && by == ay + dy };
            for (int k = 0; k < 2; k++) {
                if (!valid[k] || distance[before[k]] != 255u) continue;
                distance[before[k]] = (uint8_t)(distance[state] + 1u);
                queue[tail++] = (uint16_t)before[k];
            }
        }
    }
}

static __attribute__((noinline)) void sokoban_guide(const uint32_t *words, uint32_t *extra, float *ctrl) {
    uint8_t *distance = (uint8_t *)(extra + 1);
    uint32_t target = 1u + words[4] + SOKOBAN_SIDE * words[5];
    if (extra[0] != target) {
        sokoban_route((int)words[4], (int)words[5], distance);
        extra[0] = target;
    }
    int best = 255, choice = -1;
    for (uint32_t d = 0u; d < 4u; d++) {
        int ax = (int)words[0], ay = (int)words[1], bx = (int)words[2], by = (int)words[3];
        if (!sokoban_move(&ax, &ay, &bx, &by, d)) continue;
        int far = distance[sokoban_pack(ax, ay, bx, by)];
        if (far < best) { best = far; choice = (int)d; }
    }
    ctrl[0] = (float)(choice < 0 ? 0 : choice);
}

static int sokoban_offset(uint32_t direction, int row) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    return dy * row + dx;
}

static uint32_t sokoban_gridWords(const sokobanKnobs *k) { return (k->side + 4u) * SOKOBAN_ROW / 4u; }

static uint32_t sokoban_entryWords(const sokobanKnobs *k) { return sokoban_gridWords(k) + 3u + (2u * k->maxSteps + 31u) / 32u; }

static uint64_t sokoban_poolOffset(uint32_t instance) {
    uint64_t offset = 0u;
    for (uint32_t i = 0u; i < instance; i++) offset += (uint64_t)SOKOBAN_KNOBS[i].pool * sokoban_entryWords(&SOKOBAN_KNOBS[i]);
    return offset;
}

static const uint32_t *sokoban_level(uint32_t instance, const float *shared, uint32_t index) {
    return (const uint32_t *)shared + sokoban_poolOffset(instance) + (uint64_t)index * sokoban_entryWords(&SOKOBAN_KNOBS[instance]);
}

static uint32_t sokoban_inner(uint32_t cell) { return ((cell >> 4) - 3u) * 8u + (cell & 15u) - 3u; }

static uint32_t sokoban_outer(uint32_t inner) { return inner + (inner & 56u) + 51u; }

static uint32_t sokoban_match(uint32_t boxes, uint32_t cell) {
    uint32_t v = boxes ^ cell * 0x01010101u;
    return ~(((v & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | v | 0x7F7F7F7Fu);
}

static inline __attribute__((always_inline)) void sokoban_look(const sokobanKnobs *k, const uint8_t *grid, uint32_t player, uint32_t boxes, float *obs) {
    static const uint8_t SPAN[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    static const int8_t CODES[8] = { 0, 1, 2, 3, -2, 0, 4, 5 };
    uint32_t rows = k->side - 2u;
    uint8x8_t span = vld1_u8(SPAN), codes = vreinterpret_u8_s8(vld1_s8(CODES)), two = vdup_n_u8(2u), six = vdup_n_u8(6u);
    for (uint32_t r = 0u; r < rows; r++) {
        uint32_t base = (r + 3u) * SOKOBAN_ROW + 3u;
        uint8x8_t cells = vadd_u8(vdup_n_u8((uint8_t)base), span);
        uint8x8_t held = vorr_u8(vorr_u8(vceq_u8(cells, vdup_n_u8((uint8_t)boxes)), vceq_u8(cells, vdup_n_u8((uint8_t)(boxes >> 8)))),
                                 vorr_u8(vceq_u8(cells, vdup_n_u8((uint8_t)(boxes >> 16))), vceq_u8(cells, vdup_n_u8((uint8_t)(boxes >> 24)))));
        uint8x8_t mark = vorr_u8(vand_u8(held, two), vand_u8(vceq_u8(cells, vdup_n_u8((uint8_t)player)), six));
        int16x8_t value = vmovl_s8(vreinterpret_s8_u8(vtbl1_u8(codes, vadd_u8(vld1_u8(grid + base), mark))));
        float row[8];
        vst1q_f32(row, vcvtq_n_f32_s32(vmovl_s16(vget_low_s16(value)), 1));
        vst1q_f32(row + 4, vcvtq_n_f32_s32(vmovl_high_s16(value), 1));
        __builtin_memcpy(obs + rows * r, row, rows * sizeof(float));
    }
}

static uint32_t sokoban_draw(const sokobanKnobs *k, uint32_t seed, uint32_t step) { return (uint32_t)((uint64_t)philox4x32(seed, ENV_RNG_STREAM, step, 8u, 0u, 0u).x * k->pool >> 32); }

static __attribute__((noinline)) float sokoban_next(uint32_t instance, uint32_t seed, uint32_t step, uint32_t *words, float *obs, uint32_t *done, const float *shared, float reward) {
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    uint32_t index = sokoban_draw(k, seed, step), start = sokoban_gridWords(k);
    const uint32_t *level = sokoban_level(instance, shared, index);
    uint32_t state[5] = { level[start], level[start + 1u], 0u, 0u, index };
    __builtin_memcpy(words, state, sizeof state);
    *done = 1u;
    sokoban_look(k, (const uint8_t *)level, state[SOKOBAN_PLAYER], state[SOKOBAN_BOXES], obs);
    return reward;
}

static inline __attribute__((always_inline)) float sokoban_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done, const float *shared) {
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    const uint32_t *level = sokoban_level(instance, shared, words[SOKOBAN_LEVEL]);
    const uint8_t *grid = (const uint8_t *)level;
    uint32_t start = sokoban_gridWords(k), player = words[SOKOBAN_PLAYER], boxes = words[SOKOBAN_BOXES];
    if (player == 0u) { player = level[start]; boxes = level[start + 1u]; }
    float c = ctrl[0];
    uint32_t action = c >= 0.0f && c < 5.0f ? (uint32_t)c : 0u, delta = (uint32_t)sokoban_offset(action - 1u, SOKOBAN_ROW) & (0u - (action != 0u));
    uint32_t next = player + delta, beyond = next + delta, hit = sokoban_match(boxes, next);
    uint32_t push = (hit != 0u) & (grid[beyond] != SOKOBAN_WALL) & (sokoban_match(boxes, beyond) == 0u);
    uint32_t walk = (grid[next] != SOKOBAN_WALL) & ((hit == 0u) | push);
    int gain = (int)push * ((grid[beyond] == SOKOBAN_TARGET) - (grid[next] == SOKOBAN_TARGET));
    uint32_t placed = words[SOKOBAN_PLACED] + (uint32_t)gain, clock = words[SOKOBAN_CLOCK] + 1u;
    boxes ^= (push ? (hit >> 7) * 255u : 0u) & (next ^ beyond) * 0x01010101u;
    player = walk ? next : player;
    float reward = -0.1f + (float)gain + (placed == k->boxes ? 10.0f : 0.0f);
    if (placed == k->boxes || clock >= k->maxSteps) return sokoban_next(instance, seed, step, words, obs, done, shared, reward);
    uint32_t state[5] = { player, boxes, placed, clock, words[SOKOBAN_LEVEL] };
    __builtin_memcpy(words, state, sizeof state);
    *done = 0u;
    sokoban_look(k, grid, player, boxes, obs);
    return reward;
}

static __attribute__((noinline)) float sokoban_pool(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done, const float *shared) {
    switch (instance) {
        case 1u: return sokoban_play(1u, seed, step, ctrl, words, obs, done, shared);
        default: return sokoban_play(2u, seed, step, ctrl, words, obs, done, shared);
    }
}

typedef struct { uint32_t index, instance, block, used, bits[4]; } sokobanDice;

typedef struct { uint64_t boxes, moved; uint32_t key; uint8_t player, action, pulled, box[4]; uint16_t swaps, spread; } sokobanFrame;

typedef struct { uint64_t boxes; uint32_t parent, next, slot; uint16_t cost; uint8_t player, push; } sokobanNode;

typedef struct {
    uint32_t *explored, bits, count, mask, used, best, reached, choose[4][64];
    sokobanNode *nodes;
    uint64_t *index;
    sokobanFrame stack[SOKOBAN_TTL];
} sokobanForge;

static uint32_t sokoban_roll(sokobanDice *d) {
    if (d->used == 4u) {
        u4 r = philox4x32(d->index, ENV_RNG_STREAM, d->block++, d->instance, 0u, 1u);
        d->bits[0] = r.x; d->bits[1] = r.y; d->bits[2] = r.z; d->bits[3] = r.w;
        d->used = 0u;
    }
    return d->bits[d->used++];
}

static uint32_t sokoban_below(sokobanDice *d, uint32_t n) {
    for (;;) {
        uint64_t m = (uint64_t)sokoban_roll(d) * n;
        if ((uint32_t)m >= (0u - n) % n) return (uint32_t)(m >> 32);
    }
}

static uint32_t sokoban_carve(const sokobanKnobs *k, sokobanDice *d, uint8_t *grid, uint8_t *cells) {
    static const int8_t TURNS[4][2] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };
    static const uint16_t MASKS[5] = { 0x038u, 0x092u, 0x098u, 0x0D8u, 0x0B0u };
    int side = (int)k->side, steps = 17 * 2 * side / 10;
    memset(grid, SOKOBAN_WALL, (k->side + 4u) * SOKOBAN_ROW);
    uint32_t turn = sokoban_below(d, 4u);
    int row = 1 + (int)sokoban_below(d, k->side - 1u), col = 1 + (int)sokoban_below(d, k->side - 1u);
    for (int s = 0; s < steps; s++) {
        if (sokoban_roll(d) < u01Bound(SOKOBAN_TURN)) turn = sokoban_below(d, 4u);
        row += TURNS[turn][0]; col += TURNS[turn][1];
        row = row < 1 ? 1 : row > side - 2 ? side - 2 : row;
        col = col < 1 ? 1 : col > side - 2 ? side - 2 : col;
        uint32_t mask = MASKS[sokoban_below(d, 5u)];
        for (int b = 0; b < 9; b++)
            if (mask >> b & 1u) grid[(row + b / 3 + 1) * SOKOBAN_ROW + col + b % 3 + 1] = SOKOBAN_FLOOR;
    }
    uint32_t count = 0u;
    for (int r = 0; r < side; r++)
        for (int c = 0; c < side; c++) {
            uint32_t cell = (uint32_t)((r + 2) * SOKOBAN_ROW + c + 2);
            if (r == 0 || c == 0 || r == side - 1 || c == side - 1) grid[cell] = SOKOBAN_WALL;
            if (grid[cell] == SOKOBAN_FLOOR) cells[count++] = (uint8_t)cell;
        }
    return count;
}

static uint32_t sokoban_take(sokobanDice *d, uint8_t *cells, uint32_t count) {
    uint32_t pick = sokoban_below(d, count), cell = cells[pick];
    memmove(cells + pick, cells + pick + 1u, count - pick - 1u);
    return cell;
}

static uint32_t sokoban_distance(uint32_t a, uint32_t b) {
    int dy = (int)(a >> 4) - (int)(b >> 4), dx = (int)(a & 15u) - (int)(b & 15u);
    return (uint32_t)(abs(dx) + abs(dy));
}

static uint32_t sokoban_rank(const sokobanForge *f, uint64_t boxes) {
    uint32_t rank = 0u;
    for (uint32_t order = 0u; boxes; order++, boxes &= boxes - 1u) rank += f->choose[order][__builtin_ctzll(boxes)];
    return rank;
}

static uint32_t sokoban_key(const sokobanForge *forge, const sokobanFrame *f, uint32_t boxes) {
    return sokoban_rank(forge, f->moved) | (~(uint32_t)f->pulled & ((1u << boxes) - 1u)) << 20 | sokoban_inner(f->player) << 24;
}

static int sokoban_claim(uint32_t *table, uint32_t bits, uint32_t key) {
    uint32_t mask = (1u << bits) - 1u;
    for (uint32_t slot = (key * 0x9E3779B1u) >> (32u - bits);; slot = (slot + 1u) & mask) {
        if (table[slot] == key) return 0;
        if (table[slot] == 0xFFFFFFFFu) { table[slot] = key; return 1; }
    }
}

static uint32_t sokoban_score(const uint8_t *grid, uint64_t goal, const sokobanFrame *f, uint32_t boxes) {
    if (f->pulled != (1u << boxes) - 1u || grid[f->player] == SOKOBAN_TARGET || (f->boxes & goal)) return 0u;
    return (uint32_t)f->swaps * f->spread;
}

static int sokoban_pull(const uint8_t *grid, const uint8_t *targets, const sokobanFrame *f, uint32_t action, sokobanFrame *child) {
    uint32_t step = (uint32_t)sokoban_offset(action, SOKOBAN_ROW), next = f->player + step, behind = f->player - step;
    if (grid[next] == SOKOBAN_WALL || f->boxes >> sokoban_inner(next) & 1u) return 0;
    uint32_t held = grid[behind] != SOKOBAN_WALL && f->boxes >> sokoban_inner(behind) & 1u;
    if (action >= 4u && !held) return 0;
    *child = *f;
    child->player = (uint8_t)next;
    child->action = 0u;
    if (action < 4u && held) {
        uint32_t j = 0u;
        while (f->box[j] != behind) j++;
        uint64_t from = 1ull << sokoban_inner(behind), to = 1ull << sokoban_inner(f->player);
        child->box[j] = f->player;
        child->boxes ^= from | to;
        child->moved = (child->moved & ~from) | to;
        child->pulled |= (uint8_t)(1u << j);
        child->swaps++;
        child->spread = (uint16_t)(f->spread + sokoban_distance(f->player, targets[j]) - sokoban_distance(behind, targets[j]));
    }
    return 1;
}

static uint32_t sokoban_reverse(const uint8_t *grid, const uint8_t *targets, uint32_t boxes, uint32_t player, sokobanForge *f, sokobanFrame *best) {
    memset(f->explored, 0xFF, sizeof(uint32_t) << f->bits);
    sokobanFrame *stack = f->stack;
    uint64_t goal = 0u;
    stack[0] = (sokobanFrame){ .player = (uint8_t)player };
    for (uint32_t j = 0u; j < boxes; j++) {
        stack[0].box[j] = targets[j];
        goal |= 1ull << sokoban_inner(targets[j]);
    }
    stack[0].boxes = goal;
    stack[0].key = sokoban_key(f, &stack[0], boxes);
    sokoban_claim(f->explored, f->bits, stack[0].key);
    *best = stack[0];
    uint32_t explored = 1u, top = 0u, bestScore = 0u;
    for (;;) {
        sokobanFrame *frame = &stack[top];
        if (frame->action == 8u || top + 2u >= SOKOBAN_TTL) {
            if (top-- == 0u) break;
            continue;
        }
        sokobanFrame child;
        if (!sokoban_pull(grid, targets, frame, frame->action++, &child)) continue;
        if (explored >= SOKOBAN_EXPLORED) break;
        child.key = sokoban_key(f, &child, boxes);
        if ((top && child.key == stack[top - 1u].key) || !sokoban_claim(f->explored, f->bits, child.key)) continue;
        explored++;
        uint32_t score = sokoban_score(grid, goal, &child, boxes);
        if (score > bestScore) { bestScore = score; *best = child; }
        stack[++top] = child;
    }
    return bestScore;
}

static uint64_t sokoban_expand(uint64_t frontier) { return frontier << 8 | frontier >> 8 | (frontier << 1 & ~SOKOBAN_WEST) | (frontier >> 1 & ~SOKOBAN_EAST); }

static uint64_t sokoban_pushable(uint64_t boxes, uint64_t reach, uint64_t open, uint32_t direction) {
    switch (direction) {
        case 0u: return boxes & reach >> 8 & open << 8;
        case 1u: return boxes & reach << 8 & open >> 8;
        case 2u: return boxes & (reach >> 1 & ~SOKOBAN_EAST) & (open << 1 & ~SOKOBAN_WEST);
        default: return boxes & (reach << 1 & ~SOKOBAN_WEST) & (open >> 1 & ~SOKOBAN_EAST);
    }
}

static uint32_t sokoban_find(const sokobanForge *f, uint32_t key, uint32_t *slot) {
    for (*slot = (key * 0x9E3779B1u) >> __builtin_clz(f->mask);; *slot = (*slot + 1u) & f->mask) {
        if (f->index[*slot] == 0u) return ~0u;
        if ((uint32_t)(f->index[*slot] >> 32) == key + 1u) return (uint32_t)f->index[*slot];
    }
}

static void sokoban_file(sokobanForge *f, uint32_t key, uint32_t slot, uint32_t id) {
    if (f->index[slot] == 0u && 2u * ++f->used > f->mask) {
        uint64_t *outgoing = f->index;
        uint32_t size = f->mask + 1u;
        f->mask = 2u * size - 1u;
        f->index = calloc(2u * size, sizeof(uint64_t));
        if (!f->index) __builtin_trap();
        for (uint32_t i = 0u; i < size; i++)
            if (outgoing[i]) {
                uint32_t at;
                sokoban_find(f, (uint32_t)(outgoing[i] >> 32) - 1u, &at);
                f->index[at] = outgoing[i];
                f->nodes[(uint32_t)outgoing[i]].slot = at;
            }
        free(outgoing);
        sokoban_find(f, key, &slot);
    }
    f->index[slot] = (uint64_t)(key + 1u) << 32 | id;
    f->nodes[id].slot = slot;
}

static void sokoban_reach(uint64_t open, uint32_t from, uint8_t *distance, uint64_t *reach) {
    uint64_t seen = 1ull << from, frontier = seen;
    distance[from] = 0u;
    for (uint32_t far = 1u; frontier; far++) {
        frontier = sokoban_expand(frontier) & open & ~seen;
        seen |= frontier;
        for (uint64_t f = frontier; f; f &= f - 1u) distance[__builtin_ctzll(f)] = (uint8_t)far;
    }
    *reach = seen;
}

static int sokoban_blocked(const void *open, int x, int y) { return (uint32_t)x >= 8u || (uint32_t)y >= 8u || !(*(const uint64_t *)open >> (y * 8 + x) & 1u); }

static uint32_t sokoban_assign(const uint8_t (*far)[256], uint64_t boxes, uint32_t count, uint32_t need) {
    uint32_t best[16], least = 0xFFFFu;
    for (uint32_t m = 1u; m < 16u; m++) best[m] = 0xFFFFu;
    best[0] = 0u;
    for (uint32_t i = 0u; i < count; i++, boxes &= boxes - 1u) {
        uint32_t cell = sokoban_outer((uint32_t)__builtin_ctzll(boxes));
        for (uint32_t m = 1u << count; m-- > 0u;) {
            uint32_t held = (uint32_t)__builtin_popcount(m);
            if (held > i || held + count < need + i) continue;
            for (uint32_t t = 0u; t < count; t++)
                if (!(m >> t & 1u) && best[m] + far[t][cell] < best[m | 1u << t]) best[m | 1u << t] = best[m] + far[t][cell];
        }
    }
    for (uint32_t m = 0u; m < 1u << count; m++)
        if ((uint32_t)__builtin_popcount(m) == need && best[m] < least) least = best[m];
    return least;
}

static uint32_t sokoban_search(const sokobanKnobs *k, uint64_t ground, uint64_t goal, uint64_t live, const uint8_t (*far)[256], uint64_t boxes, uint32_t player, uint32_t need, sokobanForge *f) {
    uint8_t distance[64];
    uint32_t buckets[256], slot, guess = sokoban_assign(far, boxes, k->boxes, need);
    uint64_t room = need == k->boxes ? live : ground;
    memset(f->index, 0, ((size_t)f->mask + 1u) * sizeof(uint64_t));
    f->used = 0u;
    uint32_t key = sokoban_rank(f, boxes) | player << 20;
    f->nodes[0] = (sokobanNode){ .boxes = boxes, .parent = ~0u, .next = ~0u, .player = (uint8_t)player };
    sokoban_find(f, key, &slot);
    sokoban_file(f, key, slot, 0u);
    f->count = 1u;
    f->best = 0u;
    f->reached = (uint32_t)__builtin_popcountll(boxes & goal);
    if (guess > k->maxSteps) return ~0u;
    memset(buckets, 0xFF, sizeof buckets);
    buckets[guess] = 0u;
    for (uint32_t bound = guess; bound <= k->maxSteps; bound++)
        while (buckets[bound] != ~0u) {
            uint32_t id = buckets[bound];
            const sokobanNode *n = &f->nodes[id];
            buckets[bound] = n->next;
            if ((uint32_t)f->index[n->slot] != id) continue;
            if ((uint32_t)__builtin_popcountll(n->boxes & goal) >= need) return id;
            uint64_t open = ground & ~n->boxes, reach;
            sokoban_reach(open, n->player, distance, &reach);
            for (uint32_t d = 0u; d < 4u; d++)
                for (uint64_t can = sokoban_pushable(n->boxes, reach, open & room, d); can; can &= can - 1u) {
                    uint32_t box = (uint32_t)__builtin_ctzll(can), to = box + (uint32_t)sokoban_offset(d, 8);
                    uint32_t cost = n->cost + distance[box - (uint32_t)sokoban_offset(d, 8)] + 1u;
                    uint64_t moved = n->boxes ^ (1ull << box | 1ull << to);
                    uint32_t estimate = sokoban_assign(far, moved, k->boxes, need);
                    if (cost + estimate > k->maxSteps) continue;
                    uint32_t known = sokoban_find(f, key = sokoban_rank(f, moved) | box << 20, &slot);
                    if (known != ~0u && f->nodes[known].cost <= cost) continue;
                    if (f->count == SOKOBAN_BUDGET) return ~0u;
                    f->nodes[f->count] = (sokobanNode){ .boxes = moved, .parent = id, .next = buckets[cost + estimate], .cost = (uint16_t)cost, .player = (uint8_t)box, .push = (uint8_t)(box << 2 | d) };
                    uint32_t placed = (uint32_t)__builtin_popcountll(moved & goal);
                    if (placed > f->reached) { f->reached = placed; f->best = f->count; }
                    sokoban_file(f, key, slot, f->count);
                    buckets[cost + estimate] = f->count++;
                }
        }
    return ~0u;
}

static uint32_t sokoban_solve(const sokobanKnobs *k, const uint8_t *grid, uint32_t player, const uint8_t *start, sokobanForge *f, uint32_t *moves) {
    uint8_t far[4][256], queue[256];
    uint64_t ground = 0u, goal = 0u, boxes = 0u, live = 0u;
    uint32_t length = 0u, targets = 0u;
    memset(far, 255, sizeof far);
    memset(moves, 0, (2u * k->maxSteps + 31u) / 32u * sizeof(uint32_t));
    for (uint32_t i = 0u; i < 64u; i++) {
        uint32_t cell = sokoban_outer(i);
        if (grid[cell] != SOKOBAN_WALL) ground |= 1ull << i;
        if (grid[cell] != SOKOBAN_TARGET) continue;
        uint32_t head = 0u, tail = 0u;
        goal |= 1ull << i;
        far[targets][cell] = 0u;
        queue[tail++] = (uint8_t)cell;
        while (head < tail) {
            uint32_t at = queue[head++];
            for (uint32_t d = 0u; d < 4u; d++) {
                uint32_t step = (uint32_t)sokoban_offset(d, SOKOBAN_ROW), from = at - step, stand = from - step;
                if (grid[from] == SOKOBAN_WALL || grid[stand] == SOKOBAN_WALL || far[targets][from] != 255u) continue;
                far[targets][from] = (uint8_t)(far[targets][at] + 1u);
                queue[tail++] = (uint8_t)from;
            }
        }
        targets++;
    }
    for (uint32_t i = 0u; i < 64u; i++)
        for (uint32_t t = 0u; t < targets; t++)
            if (far[t][sokoban_outer(i)] != 255u) live |= 1ull << i;
    for (uint32_t j = 0u; j < k->boxes; j++) boxes |= 1ull << sokoban_inner(start[j]);
    f->mask = 4095u;
    f->index = malloc((f->mask + 1u) * sizeof(uint64_t));
    if (!f->index) __builtin_trap();
    uint32_t id = ~0u;
    for (uint32_t need = k->boxes; id == ~0u; need--) {
        id = sokoban_search(k, ground, goal, live, (const uint8_t (*)[256])far, boxes, sokoban_inner(player), need, f);
        if (id == ~0u && f->reached + 1u >= need) id = f->best;
    }
    uint32_t chain[256], pushes = 0u, stand = sokoban_inner(player);
    for (uint32_t at = id; f->nodes[at].parent != ~0u; at = f->nodes[at].parent) chain[pushes++] = at;
    while (pushes--) {
        const sokobanNode *p = &f->nodes[chain[pushes]];
        uint32_t box = p->push >> 2, d = p->push & 3u;
        uint8_t distance[256];
        uint64_t open = ground & ~boxes;
        uint32_t behind = box - (uint32_t)sokoban_offset(d, 8);
        int x = (int)(stand & 7u), y = (int)(stand >> 3), toward;
        gridDistance(sokoban_blocked, &open, (int)(behind & 7u), (int)(behind >> 3), distance);
        if (distance[(x + 1) | (y + 1) << 4] == 255u) __builtin_trap();
        for (; (toward = gridDescend(distance, x, y)) >= 0; length++) {
            int dx, dy;
            gridDelta((uint32_t)toward, &dx, &dy);
            x += dx; y += dy;
            moves[length / 16u] |= (uint32_t)toward << (2u * (length % 16u));
        }
        moves[length / 16u] |= d << (2u * (length % 16u));
        length++;
        boxes ^= 1ull << box | 1ull << (box + (uint32_t)sokoban_offset(d, 8));
        stand = box;
    }
    if (length != f->nodes[id].cost) __builtin_trap();
    free(f->index);
    return length;
}

static void sokoban_forge(uint32_t instance, uint32_t index, uint32_t *entry, sokobanForge *f) {
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    sokobanDice dice = { index, instance, 0u, 4u, { 0u, 0u, 0u, 0u } };
    uint8_t *grid = (uint8_t *)entry, cells[64], targets[4] = { 0u, 0u, 0u, 0u };
    sokobanFrame best;
    for (;;) {
        uint32_t count = sokoban_carve(k, &dice, grid, cells);
        if (count <= k->boxes + 1u) continue;
        uint32_t player = sokoban_take(&dice, cells, count--);
        for (uint32_t j = 0u; j < k->boxes; j++) grid[targets[j] = (uint8_t)sokoban_take(&dice, cells, count--)] = SOKOBAN_TARGET;
        if (sokoban_reverse(grid, targets, k->boxes, player, f, &best) > 0u) break;
    }
    uint32_t start = sokoban_gridWords(k), boxes = 0u;
    for (uint32_t j = 0u; j < k->boxes; j++) boxes |= (uint32_t)best.box[j] << (8u * j);
    entry[start] = best.player;
    entry[start + 1u] = boxes;
    entry[start + 2u] = sokoban_solve(k, grid, best.player, best.box, f, entry + start + 3u);
}

static void sokoban_forgeBatch(void *context, size_t batch) {
    uint32_t instance = batch < SOKOBAN_KNOBS[2].pool / SOKOBAN_BATCH ? 2u : 1u;
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    uint32_t first = (uint32_t)(instance == 2u ? batch : batch - SOKOBAN_KNOBS[2].pool / SOKOBAN_BATCH) * SOKOBAN_BATCH, inner = (k->side - 2u) * (k->side - 2u);
    uint64_t states = inner;
    for (uint32_t b = 0u; b < k->boxes && states < SOKOBAN_EXPLORED; b++) states *= 2u * inner;
    sokobanForge f;
    f.bits = 64u - (uint32_t)__builtin_clzll(states < SOKOBAN_EXPLORED ? states : SOKOBAN_EXPLORED);
    for (uint32_t c = 0u; c < 64u; c++)
        for (uint32_t order = 0u, term = c; order < 4u; order++, term = term * (c - order) / (order + 1u)) f.choose[order][c] = term;
    f.explored = malloc(sizeof(uint32_t) << f.bits);
    f.nodes = malloc(SOKOBAN_BUDGET * sizeof(sokobanNode));
    if (!f.explored || !f.nodes) __builtin_trap();
    for (uint32_t i = first; i < first + SOKOBAN_BATCH; i++) sokoban_forge(instance, i, (uint32_t *)context + sokoban_poolOffset(instance) + (uint64_t)i * sokoban_entryWords(k), &f);
    free(f.explored);
    free(f.nodes);
}

uint64_t sokoban_shared(float *table) {
    if (table) dispatch_apply_f((SOKOBAN_KNOBS[1].pool + SOKOBAN_KNOBS[2].pool) / SOKOBAN_BATCH, DISPATCH_APPLY_AUTO, table, sokoban_forgeBatch);
    return sokoban_poolOffset(3u);
}

void sokoban_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)extra;
    if (instance == 0u) { sokoban_puzzle(seed, 0u, words); return; }
    uint32_t state[5] = { 0u, 0u, 0u, 0u, sokoban_draw(&SOKOBAN_KNOBS[instance], seed, 0u) };
    __builtin_memcpy(words, state, sizeof state);
}

uint32_t sokoban_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return instance == 0u ? 1u + (SOKOBAN_STATES + 3u) / 4u : 0u;
}

float sokoban_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)extra;
    return instance == 0u ? sokoban_walk(seed, step, ctrl, words, obs, done) : sokoban_pool(instance, seed, step, ctrl, words, obs, done, shared);
}

void sokoban_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step;
    if (instance == 0u) { sokoban_obs(words, obs); return; }
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    const uint32_t *level = sokoban_level(instance, shared, words[SOKOBAN_LEVEL]);
    uint32_t start = sokoban_gridWords(k), player = words[SOKOBAN_PLAYER], boxes = words[SOKOBAN_BOXES];
    if (player == 0u) { player = level[start]; boxes = level[start + 1u]; }
    sokoban_look(k, (const uint8_t *)level, player, boxes, obs);
}

static inline __attribute__((always_inline)) void sokoban_solution(uint32_t instance, const uint32_t *words, float *ctrl, const float *shared) {
    const uint32_t *level = sokoban_level(instance, shared, words[SOKOBAN_LEVEL]) + sokoban_gridWords(&SOKOBAN_KNOBS[instance]);
    uint32_t at = words[SOKOBAN_CLOCK];
    ctrl[0] = at < level[2] ? (float)(1u + (level[3u + at / 16u] >> (2u * (at % 16u)) & 3u)) : 0.0f;
}

static __attribute__((noinline)) void sokoban_replay(uint32_t instance, const uint32_t *words, float *ctrl, const float *shared) {
    switch (instance) {
        case 1u: sokoban_solution(1u, words, ctrl, shared); return;
        default: sokoban_solution(2u, words, ctrl, shared); return;
    }
}

void sokoban_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs;
    if (instance == 0u) sokoban_guide(words, extra, ctrl);
    else sokoban_replay(instance, words, ctrl, shared);
}

uint32_t sokoban_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)extra;
    if (instance == 0u) return 0u;
    const sokobanKnobs *k = &SOKOBAN_KNOBS[instance];
    uint32_t rows = k->side - 2u, count = (rows + 6u) / 2u;
    if (!out) return count;
    const uint32_t *level = sokoban_level(instance, shared, words[SOKOBAN_LEVEL]);
    const uint8_t *grid = (const uint8_t *)level;
    uint16_t field[16] = { 0u };
    uint32_t targets = 0u, placed = 0u, start = sokoban_gridWords(k);
    for (uint32_t r = 0u; r < rows; r++)
        for (uint32_t c = 0u; c < SOKOBAN_ROW; c++) {
            uint32_t cell = (r + 3u) * SOKOBAN_ROW + c;
            if (grid[cell] != SOKOBAN_WALL) field[r] |= (uint16_t)(1u << c);
            if (grid[cell] == SOKOBAN_TARGET) targets |= cell << (8u * placed++);
        }
    field[rows] = (uint16_t)targets; field[rows + 1u] = (uint16_t)(targets >> 16);
    field[rows + 2u] = (uint16_t)level[start + 1u]; field[rows + 3u] = (uint16_t)(level[start + 1u] >> 16);
    field[rows + 4u] = (uint16_t)level[start];
    for (uint32_t i = 0u; i < count; i++) out[i] = field[2u * i] | (uint32_t)field[2u * i + 1u] << 16;
    return count;
}
