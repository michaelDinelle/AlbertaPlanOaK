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

static u4 jbw_field(uint32_t seed, int x, int y, uint32_t stream) {
    return philox4x32(seed, ENV_RNG_STREAM, (uint32_t)x, (uint32_t)y, stream, 0u);
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

static int jbw_wallAt(uint32_t seed, int x, int y) {
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    return jbw_wallHit(jbw_field(seed, bx, by, 80u), bx, by, x, y);
}

static uint32_t jbw_itemAt(uint32_t seed, int x, int y) {
    if (jbw_wallAt(seed, x, y)) return JBW_WALL;
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    u4 block = jbw_field(seed, bx, by, 16u);
    for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
        int cx, cy;
        jbw_candidate(block, bx, by, slot, &cx, &cy);
        if (cx != x || cy != y) continue;
        u4 ecology = jbw_field(seed, jbw_floorDiv(x, 64), jbw_floorDiv(y, 64), 64u);
        return jbw_typeOf(block, ecology, slot);
    }
    return JBW_EMPTY;
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

static void jbw_scentAt(const jbwKnobs *knobs, uint32_t seed, int x, int y, uint32_t step, const uint32_t *history, uint32_t historyMask, float *scent, const float *shared) {
    const float *table = shared + knobs->slot * JBW_RESPONSE_WORDS;
    scent[0] = 0.0f; scent[1] = 0.0f; scent[2] = 0.0f;
    int reach = (int)knobs->radius - 1;
    int bx0 = jbw_floorDiv(x - reach, 16), bx1 = jbw_floorDiv(x + reach, 16);
    int by0 = jbw_floorDiv(y - reach, 16), by1 = jbw_floorDiv(y + reach, 16);
    uint32_t span = (uint32_t)(bx1 - bx0 + 1);
    uint32_t blocks = span * (uint32_t)(by1 - by0 + 1);
    for (uint32_t k = 0u; k < blocks; k++) {
        int by = by0 + (int)(k / span), bx = bx0 + (int)(k % span);
        u4 block = jbw_field(seed, bx, by, 16u);
        u4 wall = jbw_field(seed, bx, by, 80u);
        u4 ecology = jbw_field(seed, jbw_floorDiv(bx * 16, 64), jbw_floorDiv(by * 16, 64), 64u);
        float share[3] = { 0.0f, 0.0f, 0.0f };
        uint32_t carries = 0u;
        for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
            int px, py;
            jbw_candidate(block, bx, by, slot, &px, &py);
            int duplicate = 0;
            for (uint32_t prior = 0u; prior < slot; prior++) {
                int qx, qy;
                jbw_candidate(block, bx, by, prior, &qx, &qy);
                if (qx == px && qy == py) { duplicate = 1; break; }
            }
            if (duplicate) continue;
            int dx = px - x, dy = py - y;
            if (abs(dx) >= (int)knobs->radius || abs(dy) >= (int)knobs->radius) continue;
            if (jbw_wallHit(wall, bx, by, px, py)) continue;
            uint32_t item = jbw_typeOf(block, ecology, slot);
            uint32_t deletion = jbw_collectible(item) ? jbw_destroyedAt(history, historyMask, px, py) : 0u;
            float value = table[jbw_responseIndex(knobs->cells, JBW_DESTRUCTION_LIFETIME - 1u, dx, dy)];
            if (deletion != 0u) {
                uint32_t age = step - (deletion - 1u);
                if (age >= JBW_DESTRUCTION_LIFETIME) value = 0.0f;
                else value -= table[jbw_responseIndex(knobs->cells, age, dx, dy)];
            }
            float s[3];
            jbw_scent(item, s);
            share[0] += s[0] * value; share[1] += s[1] * value; share[2] += s[2] * value;
            carries = 1u;
        }
        if (carries) { scent[0] += share[0]; scent[1] += share[1]; scent[2] += share[2]; }
    }
}

static void jbw_observeWords(const jbwKnobs *k, uint32_t seed, const uint32_t *words, const uint32_t *history, uint32_t historyMask, uint32_t step, float *obs, const float *shared) {
    int x = (int)words[0], y = (int)words[1];
    uint32_t heading = words[2] & 3u;
    for (uint32_t i = 0u; i < 8u; i++) {
        const int *d = JBW_DIRECTIONS[(2u * heading + i) & 7u];
        uint32_t item = jbw_itemAt(seed, x + d[0], y + d[1]);
        int absent = !jbw_blocks(item) && jbw_destroyedAt(history, historyMask, x + d[0], y + d[1]) != 0u;
        obs[i] = absent ? 0.0f : jbw_energy(item);
    }
    float scent[3];
    jbw_scentAt(k, seed, x, y, step, history, historyMask, scent, shared);
    obs[8] = scent[0]; obs[9] = scent[1]; obs[10] = scent[2];
    obs[11] = (float)words[3];
    for (uint32_t i = 0u; i < 4u; i++) obs[12u + i] = (float)words[4u + i];
}

static uint32_t jbw_slots(uint32_t steps) {
    uint32_t slots = 2u;
    while (slots < steps + 2u) slots <<= 1;
    return slots;
}

void jbw_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)seed; (void)words;
    extra[0] = jbw_slots(steps) - 1u;
}

float jbw_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance;
    uint32_t *history = extra + 1;
    uint32_t historyMask = extra[0];
    uint32_t action = (uint32_t)ctrl[0];
    *done = 0u;
    float reward = 0.0f;
    words[3] = 0u;
    if (action == 1u) words[2] = (words[2] + 3u) & 3u;
    else if (action == 2u) words[2] = (words[2] + 1u) & 3u;
    else {
        int px = (int)words[0] + JBW_FORWARD[words[2] & 3u][0], py = (int)words[1] + JBW_FORWARD[words[2] & 3u][1];
        uint32_t item = jbw_itemAt(seed, px, py);
        if (!jbw_blocks(item)) {
            words[0] = (uint32_t)px; words[1] = (uint32_t)py; words[3] = 1u;
            if (jbw_collectible(item) && jbw_destroyedAt(history, historyMask, px, py) == 0u) {
                uint32_t inventory = item == JBW_BANANA ? 4u : item == JBW_ONION ? 5u : item == JBW_JELLY ? 6u : 7u;
                words[inventory]++;
                if (!jbw_destroy(history, historyMask, px, py, step)) words[8] = 1u;
            }
        }
    }
    if (action == 0u && words[3] != 0u) {
        int x = (int)words[0], y = (int)words[1];
        uint32_t item = jbw_itemAt(seed, x, y);
        if (jbw_destroyedAt(history, historyMask, x, y) == step + 1u) {
            float sign = (step / 100000u & 1u) == 0u ? 1.0f : -1.0f;
            if (item == JBW_JELLY) reward = sign;
            else if (item == JBW_ONION) reward = -sign;
        }
    }
    jbw_observeWords(&JBW_KNOBS[instance], seed, words, history, historyMask, step, obs, shared);
    if (words[8] != 0u) obs[0] = NAN;
    return reward;
}

void jbw_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance;
    jbw_observeWords(&JBW_KNOBS[instance], seed, words, extra + 1, extra[0], step, obs, shared);
}

uint32_t jbw_sharedWords(void) { return 2u * JBW_RESPONSE_WORDS; }

void jbw_buildShared(float *shared) {
    for (uint32_t i = 0u; i < 2u; i++) jbw_buildResponse(&JBW_KNOBS[i], shared + JBW_KNOBS[i].slot * JBW_RESPONSE_WORDS);
}

uint32_t jbw_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance;
    return 1u + 3u * jbw_slots(steps);
}

#define JBW_ORACLE_REACH 12

void jbw_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)obs;
    const uint32_t *history = extra + 1;
    uint32_t historyMask = extra[0];
    int x = (int)words[0], y = (int)words[1];
    uint32_t heading = words[2] & 3u;
    uint32_t good = (((step + 1u) / 100000u) & 1u) == 0u ? JBW_JELLY : JBW_ONION;
    uint32_t bad = good == JBW_JELLY ? JBW_ONION : JBW_JELLY;
    enum { SIDE = 2 * JBW_ORACLE_REACH + 1, CELLS = SIDE * SIDE };
    uint8_t firstStep[CELLS];
    uint16_t queue[CELLS];
    for (int i = 0; i < CELLS; i++) firstStep[i] = 255u;
    int origin = JBW_ORACLE_REACH * SIDE + JBW_ORACLE_REACH;
    firstStep[origin] = 254u;
    int head = 0, tail = 0, found = -1;
    queue[tail++] = (uint16_t)origin;
    while (head < tail && found < 0) {
        int cell = queue[head++], cx = cell % SIDE - JBW_ORACLE_REACH, cy = cell / SIDE - JBW_ORACLE_REACH;
        for (uint32_t d = 0u; d < 4u && found < 0; d++) {
            int nx = cx + JBW_FORWARD[d][0], ny = cy + JBW_FORWARD[d][1];
            if (nx < -JBW_ORACLE_REACH || ny < -JBW_ORACLE_REACH || nx > JBW_ORACLE_REACH || ny > JBW_ORACLE_REACH) continue;
            int next = (ny + JBW_ORACLE_REACH) * SIDE + (nx + JBW_ORACLE_REACH);
            if (firstStep[next] != 255u) continue;
            uint32_t item = jbw_itemAt(seed, x + nx, y + ny);
            int present = jbw_collectible(item) && jbw_destroyedAt(history, historyMask, x + nx, y + ny) == 0u;
            if (jbw_blocks(item) || (present && item == bad)) continue;
            firstStep[next] = firstStep[cell] == 254u ? (uint8_t)d : firstStep[cell];
            if (present && item == good) found = firstStep[next];
            else queue[tail++] = (uint16_t)next;
        }
    }
    uint32_t want;
    if (found >= 0) {
        want = (uint32_t)found;
    } else {
        uint32_t right = (heading + 1u) & 3u, left = (heading + 3u) & 3u;
        u4 wander = philox4x32(seed, ENV_RNG_STREAM, step, 0xa11u, 0u, 0u);
        if ((wander.x & 15u) == 0u) want = (wander.y & 1u) ? right : left;
        else if (!jbw_blocks(jbw_itemAt(seed, x + JBW_FORWARD[heading][0], y + JBW_FORWARD[heading][1]))) want = heading;
        else if (!jbw_blocks(jbw_itemAt(seed, x + JBW_FORWARD[right][0], y + JBW_FORWARD[right][1]))) want = right;
        else want = left;
    }
    ctrl[0] = want == heading ? 0.0f : want == ((heading + 1u) & 3u) ? 2.0f : 1.0f;
    ctrl[1] = 0.0f; ctrl[2] = 0.0f;
}
