#include "../env.h"

#define JBW_EMPTY 0u
#define JBW_BANANA 1u
#define JBW_ONION 2u
#define JBW_JELLY 3u
#define JBW_WALL 4u
#define JBW_TREE 5u
#define JBW_TRUFFLE 6u
#define JBW_RESPONSE_RADIUS_MAX 17u
#define JBW_RESPONSE_CELLS_MAX 153u
#define JBW_DESTRUCTION_LIFETIME 2000u
#define JBW_RESPONSE_WORDS (JBW_RESPONSE_CELLS_MAX * JBW_DESTRUCTION_LIFETIME)
#define JBW_BLOCK_ITEMS 4u

typedef struct { uint32_t radius, cells, slot; } jbwKnobs;

static const jbwKnobs JBW_KNOBS[2] = {
    { 17u, 153u, 0u },
    { 8u, 36u, 1u }
};


static const int JBW_DIRECTIONS[8][2] = { {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1} };
static const int JBW_FORWARD[4][2] = { {0, -1}, {1, 0}, {0, 1}, {-1, 0} };

static int jbw_floorDiv(int x, int d) {
    int q = x / d;
    return q - ((x < 0 && x % d != 0) ? 1 : 0);
}

static void jbw_candidate(u4 block, int bx, int by, uint32_t slot, int *cx, int *cy) {
    uint32_t bits = (block.x >> (8u * slot)) & 255u;
    *cx = bx * 16 + (int)(bits & 15u);
    *cy = by * 16 + (int)((bits >> 4u) & 15u);
}

static uint32_t jbw_typeOf(u4 block, u4 ecology, uint32_t slot) {
    uint32_t roll = (block.y >> (8u * slot)) & 255u;
    if ((ecology.y & 3u) == 0u) return (roll & 31u) == 0u ? JBW_TRUFFLE : JBW_TREE;
    uint32_t roll24 = roll | (((block.z >> (8u * slot)) & 255u) << 8u) | (((block.w >> (8u * slot)) & 255u) << 16u);
    uint32_t pick = roll24 % 3u;
    return pick == 0u ? JBW_BANANA : pick == 1u ? JBW_JELLY : JBW_ONION;
}

static int jbw_wallHit(u4 w, int bx, int by, int x, int y) {
    if ((w.z & 7u) != 0u) return 0;
    return (w.w & 1u) == 0u ? y == by * 16 + (int)(w.y & 15u) : x == bx * 16 + (int)(w.x & 15u);
}

static uint64_t jbw_key(int x, int y) { return ((uint64_t)(uint32_t)x << 32u) | (uint64_t)(uint32_t)y; }

static uint32_t jbw_slot(uint64_t key, uint32_t mask) {
    key ^= key >> 30u;
    key *= 0xbf58476d1ce4e5b9ull;
    key ^= key >> 27u;
    key *= 0x94d049bb133111ebull;
    key ^= key >> 31u;
    return (uint32_t)key & mask;
}

static uint32_t jbw_destroyedAt(const uint32_t *history, uint32_t mask, int x, int y) {
    uint64_t key = jbw_key(x, y);
    uint32_t slot = jbw_slot(key, mask);
    for (uint32_t probe = 0u; probe <= mask; probe++, slot = (slot + 1u) & mask) {
        const uint32_t *entry = history + 3u * slot;
        if (entry[2] == 0u) return 0u;
        if (entry[0] == (uint32_t)(key >> 32u) && entry[1] == (uint32_t)key) return entry[2];
    }
    return 0u;
}

static int jbw_destroy(uint32_t *history, uint32_t mask, int x, int y, uint32_t step) {
    uint64_t key = jbw_key(x, y);
    uint32_t slot = jbw_slot(key, mask);
    for (uint32_t probe = 0u; probe <= mask; probe++, slot = (slot + 1u) & mask) {
        uint32_t *entry = history + 3u * slot;
        if (entry[2] == 0u) {
            entry[0] = (uint32_t)(key >> 32u);
            entry[1] = (uint32_t)key;
            entry[2] = step + 1u;
            return 1;
        }
        if (entry[0] == (uint32_t)(key >> 32u) && entry[1] == (uint32_t)key) return 1;
    }
    return 0;
}

static int jbw_blocks(uint32_t item) { return item == JBW_WALL || item == JBW_TREE; }
static int jbw_collectible(uint32_t item) { return item == JBW_BANANA || item == JBW_ONION || item == JBW_JELLY || item == JBW_TRUFFLE; }

static float jbw_energy(uint32_t item) {
    switch (item) {
        case JBW_BANANA: return 2.04f;
        case JBW_ONION: return 1.68f;
        case JBW_JELLY: return 1.29f;
        case JBW_WALL: return 1.34f;
        case JBW_TREE: return 0.53f;
        case JBW_TRUFFLE: return 0.79f;
        default: return 0.0f;
    }
}

static void jbw_scent(uint32_t item, float *s) {
    switch (item) {
        case JBW_BANANA: s[0] = 1.92f; s[1] = 1.76f; s[2] = 0.40f; return;
        case JBW_ONION: s[0] = 0.68f; s[1] = 0.01f; s[2] = 0.99f; return;
        case JBW_JELLY: s[0] = 1.64f; s[1] = 0.54f; s[2] = 0.40f; return;
        case JBW_TREE: s[0] = 0.00f; s[1] = 0.47f; s[2] = 0.06f; return;
        case JBW_TRUFFLE: s[0] = 8.40f; s[1] = 4.80f; s[2] = 2.60f; return;
        default: s[0] = 0.0f; s[1] = 0.0f; s[2] = 0.0f; return;
    }
}

static uint32_t jbw_responseIndex(uint32_t cells, uint32_t time, int dx, int dy) {
    uint32_t x = (uint32_t)abs(dx), y = (uint32_t)abs(dy);
    if (x < y) { uint32_t t = x; x = y; y = t; }
    return time * cells + x * (x + 1u) / 2u + y;
}

static void jbw_buildResponse(const jbwKnobs *k, float *response) {
    uint32_t radius = k->radius, cells = k->cells;
    for (uint32_t i = 0u; i < cells * JBW_DESTRUCTION_LIFETIME; i++) response[i] = 0.0f;
    response[0] = 1.0f;
    for (uint32_t time = 1u; time < JBW_DESTRUCTION_LIFETIME; time++) {
        uint32_t previous = (time - 1u) * cells;
        uint32_t current = time * cells;
        for (uint32_t i = 0u; i < cells; i++) response[current + i] = 0.4f * response[previous + i];
        response[current] += 1.0f;
        response[current + cells - 1u] += 0.28f * response[previous + cells - 2u];
        for (int y = 0; y < (int)radius - 1; y++) {
            response[current + cells - radius + (uint32_t)y] += 0.14f * (
                response[jbw_responseIndex(cells, time - 1u, (int)radius - 2, y)]
                + response[jbw_responseIndex(cells, time - 1u, (int)radius - 1, y + 1)]
                + response[jbw_responseIndex(cells, time - 1u, (int)radius - 1, y - 1)]);
        }
        for (int x = 0; x < (int)radius - 1; x++) {
            for (int y = 0; y <= x; y++) {
                uint32_t index = (uint32_t)(x * (x + 1) / 2 + y);
                response[current + index] += 0.14f * (
                    response[jbw_responseIndex(cells, time - 1u, x + 1, y)]
                    + response[jbw_responseIndex(cells, time - 1u, x - 1, y)]
                    + response[jbw_responseIndex(cells, time - 1u, x, y + 1)]
                    + response[jbw_responseIndex(cells, time - 1u, x, y - 1)]);
            }
        }
    }
}

enum { JBW_TORUS = 64, JBW_STEADY_WORDS = JBW_TORUS * JBW_TORUS, JBW_GONE = 8, JBW_CODES = 64, JBW_STALE = 1 << 16 };

typedef struct {
    float4 scent[JBW_BLOCK_ITEMS];
    uint32_t key[JBW_BLOCK_ITEMS], stamp[JBW_BLOCK_ITEMS];
    int x[JBW_BLOCK_ITEMS], y[JBW_BLOCK_ITEMS];
} jbwBlock;

typedef struct {
    float4 scent, share[9];
    jbwBlock blocks[16];
    float near[16], energy[JBW_CODES];
    uint32_t latest[16], watch[9];
    int bx[16], by[16];
    int x, y;
    uint32_t step, live;
    uint64_t solid[JBW_TORUS], jelly[JBW_TORUS], onion[JBW_TORUS];
    uint8_t map[JBW_TORUS * JBW_TORUS];
} jbwWindow;

enum { JBW_WINDOW_WORDS = sizeof(jbwWindow) / sizeof(uint32_t) };

static jbwWindow *jbw_window(uint32_t *extra) { return (jbwWindow *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static const jbwWindow *jbw_view(const uint32_t *extra) { return (const jbwWindow *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }

static uint32_t jbw_torus(int bx, int by) { return ((uint32_t)by & 3u) << 2u | ((uint32_t)bx & 3u); }
static uint32_t jbw_cell(int x, int y) { return ((uint32_t)y & (JBW_TORUS - 1u)) * JBW_TORUS | ((uint32_t)x & (JBW_TORUS - 1u)); }
static uint32_t jbw_offset(int x, int y) { return (uint32_t)y * JBW_TORUS + (uint32_t)x; }

static void jbw_buildSteady(const jbwKnobs *k, const float *response, float *steady) {
    for (int dy = -JBW_TORUS / 2; dy < JBW_TORUS / 2; dy++)
        for (int dx = -JBW_TORUS / 2; dx < JBW_TORUS / 2; dx++)
            steady[JBW_STEADY_WORDS / 2 + JBW_TORUS / 2 + (int)jbw_offset(dx, dy)] = abs(dx) < (int)k->radius && abs(dy) < (int)k->radius
                ? response[jbw_responseIndex(k->cells, JBW_DESTRUCTION_LIFETIME - 1u, dx, dy)] : 0.0f;
}

static uint32_t jbw_settleAge(const jbwKnobs *k, const float *response) {
    uint32_t settle = 0u;
    for (uint32_t i = 0u; i < k->cells; i++) {
        uint32_t age = JBW_DESTRUCTION_LIFETIME - 1u;
        while (age > 0u && response[(age - 1u) * k->cells + i] == response[(JBW_DESTRUCTION_LIFETIME - 1u) * k->cells + i]) age--;
        if (age > settle) settle = age;
    }
    return settle;
}

static const float *jbw_table(const jbwKnobs *k, const float *shared) { return shared + k->slot * JBW_RESPONSE_WORDS; }
static const float *jbw_steady(const jbwKnobs *k, const float *shared) { return shared + 2u * JBW_RESPONSE_WORDS + k->slot * JBW_STEADY_WORDS + JBW_STEADY_WORDS / 2 + JBW_TORUS / 2; }
static uint32_t jbw_settle(const jbwKnobs *k, const float *shared) { return (uint32_t)shared[2u * (JBW_RESPONSE_WORDS + JBW_STEADY_WORDS) + k->slot]; }

static uint32_t jbw_age(uint32_t stamp, uint32_t step) { return step - (stamp - 1u); }
static int jbw_live(uint32_t stamp, uint32_t step, uint32_t settle) { return stamp != 0u && jbw_age(stamp, step) <= settle; }
static uint32_t jbw_recent(uint32_t stamp, uint32_t other, uint32_t step) { return other != 0u && (stamp == 0u || jbw_age(other, step) < jbw_age(stamp, step)) ? other : stamp; }

static void jbw_mark(jbwWindow *w, int x, int y, uint32_t code) {
    uint32_t row = (uint32_t)y & (JBW_TORUS - 1u);
    uint64_t bit = 1ull << ((uint32_t)x & (JBW_TORUS - 1u));
    w->map[jbw_cell(x, y)] = (uint8_t)code;
    w->solid[row] = jbw_blocks(code & 7u) ? w->solid[row] | bit : w->solid[row] & ~bit;
    w->jelly[row] = (code & 15u) == JBW_JELLY ? w->jelly[row] | bit : w->jelly[row] & ~bit;
    w->onion[row] = (code & 15u) == JBW_ONION ? w->onion[row] | bit : w->onion[row] & ~bit;
}

static void jbw_load(jbwWindow *w, uint32_t seed, uint32_t step, const uint32_t *history, uint32_t historyMask, int bx, int by) {
    uint32_t t = jbw_torus(bx, by), latest = 0u;
    jbwBlock *b = &w->blocks[t];
    u44 f = philox4(seed, ENV_RNG_STREAM, (lane4){ (uint32_t)bx, (uint32_t)bx, (uint32_t)jbw_floorDiv(bx * 16, 64), 0u },
                    (lane4){ (uint32_t)by, (uint32_t)by, (uint32_t)jbw_floorDiv(by * 16, 64), 0u }, (lane4){ 16u, 80u, 64u, 0u }, (lane4)0u);
    u4 block = { f.x[0], f.y[0], f.z[0], f.w[0] }, wall = { f.x[1], f.y[1], f.z[1], f.w[1] }, ecology = { f.x[2], f.y[2], f.z[2], f.w[2] };
    w->bx[t] = bx;
    w->by[t] = by;
    for (int row = 0; row < 16; row++) {
        uint8_t *line = w->map + jbw_cell(bx * 16, by * 16 + row);
        uint32_t r = (uint32_t)(by * 16 + row) & (JBW_TORUS - 1u), shift = (uint32_t)(bx * 16) & (JBW_TORUS - 1u);
        uint64_t hits = 0u;
        for (int col = 0; col < 16; col++) {
            int hit = jbw_wallHit(wall, bx, by, bx * 16 + col, by * 16 + row);
            line[col] = hit ? (uint8_t)JBW_WALL : (uint8_t)JBW_EMPTY;
            hits |= (uint64_t)hit << col;
        }
        w->solid[r] = (w->solid[r] & ~(0xFFFFull << shift)) | hits << shift;
        w->jelly[r] &= ~(0xFFFFull << shift);
        w->onion[r] &= ~(0xFFFFull << shift);
    }
    for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
        int px, py;
        jbw_candidate(block, bx, by, slot, &px, &py);
        int duplicate = 0;
        for (uint32_t prior = 0u; prior < slot; prior++) {
            int qx, qy;
            jbw_candidate(block, bx, by, prior, &qx, &qy);
            if (qx == px && qy == py) { duplicate = 1; break; }
        }
        uint32_t kind = duplicate || jbw_wallHit(wall, bx, by, px, py) ? JBW_EMPTY : jbw_typeOf(block, ecology, slot);
        float s[3];
        jbw_scent(kind, s);
        b->scent[slot] = (float4){ s[0], s[1], s[2], 0.0f };
        b->key[slot] = jbw_offset(px, py);
        b->stamp[slot] = jbw_collectible(kind) ? jbw_destroyedAt(history, historyMask, px, py) : 0u;
        b->x[slot] = px;
        b->y[slot] = py;
        latest = jbw_recent(latest, b->stamp[slot], step);
        if (kind != JBW_EMPTY) jbw_mark(w, px, py, kind | (b->stamp[slot] != 0u ? JBW_GONE : 0u) | slot << 4u);
    }
    w->latest[t] = latest;
}

static void jbw_ensure(jbwWindow *w, uint32_t seed, uint32_t step, const uint32_t *history, uint32_t historyMask, int bx, int by) {
    uint32_t t = jbw_torus(bx, by);
    if (w->bx[t] != bx || w->by[t] != by) jbw_load(w, seed, step, history, historyMask, bx, by);
}

static void jbw_near(jbwWindow *w, int x, int y) {
    float e[8];
    for (uint32_t i = 0u; i < 8u; i++) e[i] = w->energy[w->map[jbw_cell(x + JBW_DIRECTIONS[i][0], y + JBW_DIRECTIONS[i][1])]];
    float4 ahead = { e[0], e[1], e[2], e[3] }, behind = { e[4], e[5], e[6], e[7] };
    __builtin_memcpy(w->near, &ahead, sizeof ahead);
    __builtin_memcpy(w->near + 4, &behind, sizeof behind);
    __builtin_memcpy(w->near + 8, &ahead, sizeof ahead);
    __builtin_memcpy(w->near + 12, &behind, sizeof behind);
}

static void jbw_move(jbwWindow *w, uint32_t seed, uint32_t step, const uint32_t *history, uint32_t historyMask, int x, int y) {
    int bx = x >> 4, by = y >> 4, sx = bx - (w->x >> 4), sy = by - (w->y >> 4);
    w->x = x;
    w->y = y;
    if (sx != 0 || sy != 0)
        for (int i = -1; i <= 1; i++) jbw_ensure(w, seed, step, history, historyMask, sx != 0 ? bx + sx : bx + i, sy != 0 ? by + sy : by + i);
    jbw_near(w, x, y);
}

static void jbw_gone(jbwWindow *w, int x, int y, uint32_t stamp) {
    uint32_t t = jbw_torus(x >> 4, y >> 4), code = w->map[jbw_cell(x, y)];
    w->blocks[t].stamp[code >> 4u] = stamp;
    w->latest[t] = stamp;
    jbw_mark(w, x, y, code | JBW_GONE);
}

static float4 jbw_sum(const jbwBlock *b, float v0, float v1, float v2, float v3) {
    return ((b->scent[0] * v0 + b->scent[1] * v1) + b->scent[2] * v2) + b->scent[3] * v3;
}

static float4 jbw_steadyShare(const jbwBlock *b, const float *steady, uint32_t agent) {
    return jbw_sum(b, steady[(int)(b->key[0] - agent)], steady[(int)(b->key[1] - agent)], steady[(int)(b->key[2] - agent)], steady[(int)(b->key[3] - agent)]);
}

static float jbw_aged(float value, uint32_t age, uint32_t far, uint32_t close, uint32_t settle, const jbwKnobs *k, const float *table) {
    uint32_t edge = k->radius - 1u, hi = far < edge ? far : edge, lo = close < hi ? close : hi;
    float gone = table[(age < settle ? age : settle) * k->cells + hi * (hi + 1u) / 2u + lo];
    return far < k->radius ? value - gone : value;
}

static inline __attribute__((always_inline)) float4 jbw_agedShare(const jbwBlock *b, int x, int y, uint32_t agent, uint32_t step, uint32_t settle, const jbwKnobs *k, const float *table, const float *steady, uint32_t *watch) {
    float value[JBW_BLOCK_ITEMS];
    uint32_t watched = 0u, youngest = ~0u;
    for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
        value[slot] = steady[(int)(b->key[slot] - agent)];
        if (b->stamp[slot] != 0u) {
            uint32_t age = jbw_age(b->stamp[slot], step), ax = (uint32_t)abs(b->x[slot] - x), ay = (uint32_t)abs(b->y[slot] - y);
            uint32_t far = ax > ay ? ax : ay, close = ax > ay ? ay : ax;
            value[slot] = jbw_aged(value[slot], age, far, close, settle, k, table);
            int take = (far < k->radius) & (age < youngest);
            watched = take ? b->stamp[slot] : watched;
            youngest = take ? age : youngest;
        }
    }
    *watch = watched;
    return jbw_sum(b, value[0], value[1], value[2], value[3]);
}

static int jbw_span(const jbwKnobs *k) { return ((2 * ((int)k->radius - 1) + 15) >> 4) + 1; }
static uint32_t jbw_visit(const jbwKnobs *k, int x, int y, int p) {
    int reach = (int)k->radius - 1;
    return jbw_torus(((x - reach) >> 4) + p % jbw_span(k), ((y - reach) >> 4) + p / jbw_span(k));
}

static inline __attribute__((always_inline)) float4 jbw_total(const float4 *share, const jbwKnobs *k) {
    float4 scent = share[0];
    for (int p = 1; p < jbw_span(k) * jbw_span(k); p++) scent += share[p];
    return scent;
}

static inline __attribute__((always_inline)) float4 jbw_sweep(const jbwWindow *w, float4 *share, uint32_t *aged, const jbwKnobs *k, const float *shared) {
    const float *steady = jbw_steady(k, shared);
    int x = w->x, y = w->y;
    uint32_t agent = jbw_offset(x, y), mask = 0u;
    float4 scent = 0.0f;
    for (int p = 0; p < jbw_span(k) * jbw_span(k); p++) {
        uint32_t t = jbw_visit(k, x, y, p);
        float4 part = jbw_steadyShare(&w->blocks[t], steady, agent);
        mask |= (uint32_t)(w->latest[t] != 0u) << p;
        share[p] = part;
        scent = p == 0 ? part : scent + part;
    }
    *aged = mask;
    return scent;
}

static inline __attribute__((always_inline)) uint32_t jbw_ages(const jbwWindow *w, float4 *share, uint32_t *watch, uint32_t *settled, uint32_t aged, uint32_t step, const jbwKnobs *k, const float *shared) {
    const float *table = jbw_table(k, shared), *steady = jbw_steady(k, shared);
    int x = w->x, y = w->y;
    uint32_t settle = jbw_settle(k, shared), live = 0u, still = 0u, agent = jbw_offset(x, y);
    for (; aged != 0u; aged &= aged - 1u) {
        int p = __builtin_ctz(aged);
        uint32_t t = jbw_visit(k, x, y, p);
        share[p] = jbw_agedShare(&w->blocks[t], x, y, agent, step, settle, k, table, steady, &watch[p]);
        live |= (uint32_t)jbw_live(watch[p], step, settle) << p;
        still |= (uint32_t)!jbw_live(w->latest[t], step, settle) << p;
    }
    *settled = still;
    return live;
}

static inline __attribute__((always_inline)) float4 jbw_ripen(jbwWindow *w, uint32_t aged, uint32_t step, const jbwKnobs *k, const float *shared) {
    int x = w->x, y = w->y;
    uint32_t settled, live = jbw_ages(w, w->share, w->watch, &settled, aged, step, k, shared);
    for (; settled != 0u; settled &= settled - 1u) {
        uint32_t t = jbw_visit(k, x, y, __builtin_ctz(settled));
        for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; slot++)
            if (w->blocks[t].stamp[slot] != 0u) {
                w->blocks[t].stamp[slot] = 0u;
                w->blocks[t].scent[slot] = 0.0f;
            }
        w->latest[t] = 0u;
    }
    w->live = live;
    w->scent = jbw_total(w->share, k);
    return w->scent;
}

static inline __attribute__((always_inline)) float4 jbw_full(jbwWindow *w, uint32_t step, const jbwKnobs *k, const float *shared) {
    uint32_t aged;
    float4 scent = jbw_sweep(w, w->share, &aged, k, shared);
    if (aged != 0u) return jbw_ripen(w, aged, step, k, shared);
    w->live = 0u;
    w->scent = scent;
    return scent;
}

static inline __attribute__((always_inline)) float4 jbw_refresh(jbwWindow *w, uint32_t step, const jbwKnobs *k, const float *shared) {
    const float *table = jbw_table(k, shared), *steady = jbw_steady(k, shared);
    int x = w->x, y = w->y;
    uint32_t settle = jbw_settle(k, shared), agent = jbw_offset(x, y), still = w->live, watched;
    for (uint32_t live = still; live != 0u; live &= live - 1u) {
        int p = __builtin_ctz(live);
        if (jbw_live(w->watch[p], step, settle)) w->share[p] = jbw_agedShare(&w->blocks[jbw_visit(k, x, y, p)], x, y, agent, step, settle, k, table, steady, &watched);
        else still &= ~(1u << p);
    }
    w->live = still;
    w->scent = jbw_total(w->share, k);
    return w->scent;
}

static void jbw_emit(const jbwWindow *w, const uint32_t *words, float4 scent, float *obs) {
    __builtin_memcpy(obs, w->near + 2u * (words[2] & 3u), 8u * sizeof(float));
    __builtin_memcpy(obs + 8, &scent, sizeof scent);
    lane4 held;
    __builtin_memcpy(&held, words + 4, sizeof held);
    float4 counts = __builtin_convertvector(held, float4);
    __builtin_memcpy(obs + 12, &counts, sizeof counts);
}

static float jbw_show(const jbwWindow *w, const uint32_t *words, float4 scent, float *obs) {
    jbw_emit(w, words, scent, obs);
    if (words[8] != 0u) obs[0] = NAN;
    return 0.0f;
}

static uint32_t jbw_slots(uint32_t steps) {
    uint32_t slots = 2u;
    while (slots < steps + 2u) slots <<= 1;
    return slots;
}

void jbw_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance;
    extra[0] = jbw_slots(steps) - 1u;
    jbwWindow *w = jbw_window(extra);
    for (uint32_t code = 0u; code < JBW_CODES; code++) w->energy[code] = (code & JBW_GONE) != 0u ? 0.0f : jbw_energy(code & 7u);
    w->step = 0u;
    w->x = (int)words[0];
    w->y = (int)words[1];
    for (int by = (w->y >> 4) - 1; by <= (w->y >> 4) + 2; by++)
        for (int bx = (w->x >> 4) - 1; bx <= (w->x >> 4) + 2; bx++) jbw_load(w, seed, 0u, extra + 4u + JBW_WINDOW_WORDS, extra[0], bx, by);
    jbw_near(w, w->x, w->y);
    w->live = JBW_STALE;
}

static __attribute__((noinline)) float jbw_fresh(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)seed;
    (void)ctrl;
    (void)done;
    jbwWindow *w = jbw_window(extra);
    float4 scent = jbw_full(w, step, &JBW_KNOBS[instance], shared);
    scent.w = (float)words[3];
    return jbw_show(w, words, scent, obs);
}

static __attribute__((noinline)) float jbw_refreshed(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    jbwWindow *w = jbw_window(extra);
    if (w->live == JBW_STALE) __attribute__((musttail)) return jbw_fresh(instance, seed, step, ctrl, words, extra, obs, done, shared);
    return jbw_show(w, words, instance == 0u ? jbw_refresh(w, step, &JBW_KNOBS[0], shared) : jbw_refresh(w, step, &JBW_KNOBS[1], shared), obs);
}

static __attribute__((noinline)) float jbw_ripened(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)seed;
    (void)ctrl;
    (void)done;
    jbwWindow *w = jbw_window(extra);
    float4 scent = instance == 0u ? jbw_ripen(w, w->live, step, &JBW_KNOBS[0], shared) : jbw_ripen(w, w->live, step, &JBW_KNOBS[1], shared);
    scent.w = 1.0f;
    return jbw_show(w, words, scent, obs);
}

static __attribute__((noinline)) float jbw_arrive(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    jbwWindow *w = jbw_window(extra);
    uint32_t *history = extra + 4u + JBW_WINDOW_WORDS;
    uint32_t historyMask = extra[0];
    int px = (int)words[0], py = (int)words[1];
    uint32_t code = w->map[jbw_cell(px, py)], item = code & 7u;
    int stamped = 0;
    if (jbw_collectible(item) && (code & JBW_GONE) == 0u) {
        uint32_t inventory = item == JBW_BANANA ? 4u : item == JBW_ONION ? 5u : item == JBW_JELLY ? 6u : 7u;
        words[inventory]++;
        int inserted = jbw_destroy(history, historyMask, px, py, step);
        if (!inserted) words[8] = 1u;
        else if (step + 1u != 0u) jbw_gone(w, px, py, step + 1u);
        stamped = inserted || step + 1u == 0u;
    }
    jbw_move(w, seed, step, history, historyMask, px, py);
    float reward = 0.0f;
    if ((uint32_t)ctrl[0] == 0u && stamped) {
        float sign = (step / 100000u & 1u) == 0u ? 1.0f : -1.0f;
        if (item == JBW_JELLY) reward = sign;
        else if (item == JBW_ONION) reward = -sign;
    }
    jbw_fresh(instance, seed, step, ctrl, words, extra, obs, done, shared);
    return reward;
}

static __attribute__((noinline)) float jbw_advance(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    jbwWindow *w = jbw_window(extra);
    int x = w->x, y = w->y, px = x + JBW_FORWARD[words[2] & 3u][0], py = y + JBW_FORWARD[words[2] & 3u][1];
    uint32_t code = w->map[jbw_cell(px, py)];
    if (jbw_blocks(code & 7u)) {
        if (w->live != 0u) __attribute__((musttail)) return jbw_refreshed(instance, seed, step, ctrl, words, extra, obs, done, shared);
        return jbw_show(w, words, w->scent, obs);
    }
    words[0] = (uint32_t)px;
    words[1] = (uint32_t)py;
    words[3] = 1u;
    if ((jbw_collectible(code & 7u) && (code & JBW_GONE) == 0u) || ((uint32_t)(px ^ x) | (uint32_t)(py ^ y)) >= 16u)
        __attribute__((musttail)) return jbw_arrive(instance, seed, step, ctrl, words, extra, obs, done, shared);
    w->x = px;
    w->y = py;
    jbw_near(w, px, py);
    uint32_t aged;
    float4 scent = instance == 0u ? jbw_sweep(w, w->share, &aged, &JBW_KNOBS[0], shared) : jbw_sweep(w, w->share, &aged, &JBW_KNOBS[1], shared);
    w->live = aged;
    if (aged != 0u) __attribute__((musttail)) return jbw_ripened(instance, seed, step, ctrl, words, extra, obs, done, shared);
    w->scent = scent;
    scent.w = 1.0f;
    return jbw_show(w, words, scent, obs);
}

float jbw_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    jbwWindow *w = jbw_window(extra);
    if ((step != w->step + 1u) | (((uint64_t)words[1] << 32 | words[0]) != ((uint64_t)(uint32_t)w->y << 32 | (uint32_t)w->x))) __builtin_trap();
    w->step = step;
    uint32_t turn = floatWord(ctrl[0]) - 0x3F800000u;
    *done = 0u;
    words[3] = 0u;
    if (turn >= 0x00C00000u) __attribute__((musttail)) return jbw_advance(instance, seed, step, ctrl, words, extra, obs, done, shared);
    words[2] = (words[2] + (turn < 0x00800000u ? 3u : 1u)) & 3u;
    if (w->live != 0u) __attribute__((musttail)) return jbw_refreshed(instance, seed, step, ctrl, words, extra, obs, done, shared);
    return jbw_show(w, words, w->scent, obs);
}

void jbw_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed;
    const jbwWindow *w = jbw_view(extra);
    if (step != w->step || (int)words[0] != w->x || (int)words[1] != w->y) __builtin_trap();
    const jbwKnobs *k = &JBW_KNOBS[instance];
    float4 share[9];
    uint32_t aged, watch[9], settled;
    float4 scent = jbw_sweep(w, share, &aged, k, shared);
    if (aged != 0u) {
        jbw_ages(w, share, watch, &settled, aged, step, k, shared);
        scent = jbw_total(share, k);
    }
    scent.w = (float)words[3];
    jbw_emit(w, words, scent, obs);
}

uint64_t jbw_shared(float *table) {
    if (table)
        for (uint32_t i = 0u; i < 2u; i++) {
            const jbwKnobs *k = &JBW_KNOBS[i];
            float *response = table + k->slot * JBW_RESPONSE_WORDS;
            jbw_buildResponse(k, response);
            jbw_buildSteady(k, response, table + 2u * JBW_RESPONSE_WORDS + k->slot * JBW_STEADY_WORDS);
            table[2u * (JBW_RESPONSE_WORDS + JBW_STEADY_WORDS) + k->slot] = (float)jbw_settleAge(k, response);
        }
    return 2u * (JBW_RESPONSE_WORDS + JBW_STEADY_WORDS + 1u);
}

uint32_t jbw_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance;
    return 4u + JBW_WINDOW_WORDS + 3u * jbw_slots(steps);
}

#define JBW_ORACLE_REACH 12

static uint64_t jbw_rotate(uint64_t bits, uint32_t shift) { return bits >> shift | bits << ((0u - shift) & (JBW_TORUS - 1u)); }

static int jbw_seek(const uint32_t *open, const uint32_t *goal) {
    enum { SIDE = 2 * JBW_ORACLE_REACH + 1 };
    const uint32_t box = (1u << SIDE) - 1u;
    lane4 wave[2][SIDE + 2], none = 0u;
    uint32_t seen[SIDE];
    for (int r = 0; r < SIDE + 2; r++) wave[0][r] = wave[1][r] = none;
    for (int r = 0; r < SIDE; r++) seen[r] = 0u;
    seen[JBW_ORACLE_REACH] = 1u << JBW_ORACLE_REACH;
    for (uint32_t d = 0u; d < 4u; d++) {
        int r = JBW_ORACLE_REACH + JBW_FORWARD[d][1];
        uint32_t bit = 1u << (JBW_ORACLE_REACH + JBW_FORWARD[d][0]);
        if (goal[r] & bit) return (int)d;
        if (open[r] & bit) { wave[0][r + 1][d] = bit; seen[r] |= bit; }
    }
    for (int depth = 1, now = 0;; depth++, now ^= 1) {
        int lo = JBW_ORACLE_REACH - depth - 1 < 0 ? 0 : JBW_ORACLE_REACH - depth - 1, hi = JBW_ORACLE_REACH + depth + 1 >= SIDE ? SIDE - 1 : JBW_ORACLE_REACH + depth + 1;
        lane4 hit = none, grown = none;
        for (int r = lo; r <= hi; r++) {
            lane4 at = wave[now][r + 1];
            lane4 reach = ((at << 1u) | (at >> 1u) | wave[now][r] | wave[now][r + 2]) & box;
            lane4 before = reach | __builtin_shufflevector(reach, none, 4, 0, 1, 2);
            before |= __builtin_shufflevector(before, none, 4, 4, 0, 1);
            lane4 claim = reach & ~(__builtin_shufflevector(before, none, 4, 0, 1, 2) | seen[r]);
            hit |= claim & goal[r];
            wave[now ^ 1][r + 1] = claim & open[r];
            grown |= claim & open[r];
            seen[r] |= before.w;
        }
        for (uint32_t d = 0u; d < 4u; d++) if (hit[d] != 0u) return (int)d;
        if ((grown.x | grown.y | grown.z | grown.w) == 0u) return -1;
    }
}

void jbw_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)obs;
    const jbwWindow *w = jbw_view(extra);
    int x = (int)words[0], y = (int)words[1];
    if (step != w->step || x != w->x || y != w->y) __builtin_trap();
    uint32_t heading = words[2] & 3u;
    uint32_t good = (((step + 1u) / 100000u) & 1u) == 0u ? JBW_JELLY : JBW_ONION;
    const uint64_t *want = good == JBW_JELLY ? w->jelly : w->onion, *avoid = good == JBW_JELLY ? w->onion : w->jelly;
    enum { SIDE = 2 * JBW_ORACLE_REACH + 1 };
    uint32_t open[SIDE], goal[SIDE], any = 0u, shift = (uint32_t)(x - JBW_ORACLE_REACH) & (JBW_TORUS - 1u);
    for (int r = 0; r < SIDE; r++) {
        uint32_t row = (uint32_t)(y - JBW_ORACLE_REACH + r) & (JBW_TORUS - 1u);
        goal[r] = (uint32_t)jbw_rotate(want[row], shift) & ((1u << SIDE) - 1u);
        open[r] = ~(uint32_t)jbw_rotate(w->solid[row] | want[row] | avoid[row], shift) & ((1u << SIDE) - 1u);
        any |= goal[r];
    }
    int found = any != 0u ? jbw_seek(open, goal) : -1;
    uint32_t move;
    if (found >= 0) {
        move = (uint32_t)found;
    } else {
        uint32_t right = (heading + 1u) & 3u, left = (heading + 3u) & 3u;
        u4 wander = philox4x32(seed, ENV_RNG_STREAM, step, 0xa11u, 0u, 0u);
        if ((wander.x & 15u) == 0u) move = (wander.y & 1u) ? right : left;
        else if (!jbw_blocks(w->map[jbw_cell(x + JBW_FORWARD[heading][0], y + JBW_FORWARD[heading][1])] & 7u)) move = heading;
        else if (!jbw_blocks(w->map[jbw_cell(x + JBW_FORWARD[right][0], y + JBW_FORWARD[right][1])] & 7u)) move = right;
        else move = left;
    }
    ctrl[0] = move == heading ? 0.0f : move == ((heading + 1u) & 3u) ? 2.0f : 1.0f;
    ctrl[1] = 0.0f; ctrl[2] = 0.0f;
}
