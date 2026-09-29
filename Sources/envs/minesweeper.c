#include "../env.h"

typedef struct { uint32_t width, height, mines; } minesweeperKnobs;

static const minesweeperKnobs MINESWEEPER_KNOBS[3] = {
    { 9u, 9u, 10u },
    { 16u, 16u, 40u },
    { 30u, 16u, 99u }
};

enum { MINESWEEPER_X, MINESWEEPER_Y, MINESWEEPER_GAME, MINESWEEPER_OPENED, MINESWEEPER_FLAGS, MINESWEEPER_FEED, MINESWEEPER_IDLE, MINESWEEPER_FORCED };
enum { MINESWEEPER_HIDDEN, MINESWEEPER_FLAGGED, MINESWEEPER_OUTSIDE, MINESWEEPER_SHOWN };
enum { MINESWEEPER_STRIDE = 32, MINESWEEPER_ROW = MINESWEEPER_STRIDE / 4 };

static uint32_t minesweeper_mines(const minesweeperKnobs *k) { return (k->height + 2u) * MINESWEEPER_ROW; }
static uint32_t minesweeper_opens(const minesweeperKnobs *k) { return minesweeper_mines(k) + k->height + 2u; }
static uint32_t minesweeper_pending(const minesweeperKnobs *k) { return minesweeper_opens(k) + k->height; }
static uint32_t minesweeper_queued(const minesweeperKnobs *k) { return minesweeper_pending(k) + k->height; }
static uint32_t minesweeper_patience(const minesweeperKnobs *k) { return 2u * (k->width + k->height); }

uint32_t minesweeper_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return minesweeper_queued(&MINESWEEPER_KNOBS[instance]) + 1u;
}

static inline __attribute__((always_inline)) void minesweeper_board(const minesweeperKnobs *k, uint32_t seed, uint32_t *words, uint32_t *extra) {
    uint8_t *cell = (uint8_t *)extra;
    uint32_t *mine = extra + minesweeper_mines(k), *open = extra + minesweeper_opens(k), *pending = extra + minesweeper_pending(k);
    for (uint32_t y = 0u; y < k->height; y++) {
        __builtin_memset(cell + (y + 1u) * MINESWEEPER_STRIDE + 1u, MINESWEEPER_HIDDEN, k->width);
        mine[y + 1u] = 0u; open[y] = 0u; pending[y] = 0u;
    }
    extra[minesweeper_queued(k)] = 0u;
    uint32_t placed = 0u;
    for (uint32_t block = 0u; placed < k->mines; block += 4u) {
        u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)words[MINESWEEPER_GAME], block + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
        lane4 draws[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t lane = 0u; lane < 4u; lane++)
            for (uint32_t part = 0u; part < 4u && placed < k->mines; part++) {
                uint32_t draw = draws[part][lane], bit = 2u << ((draw & 0x7FFFu) % k->width), *row = mine + 1u + (draw >> 17) % k->height;
                if (*row & bit) continue;
                *row |= bit;
                placed++;
            }
    }
    words[MINESWEEPER_OPENED] = 0u;
    words[MINESWEEPER_FLAGS] = 0u;
    words[MINESWEEPER_IDLE] = 0u;
}

static __attribute__((noinline)) void minesweeper_deal(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra) {
    switch (instance) {
        case 0u: minesweeper_board(&MINESWEEPER_KNOBS[0], seed, words, extra); return;
        case 1u: minesweeper_board(&MINESWEEPER_KNOBS[1], seed, words, extra); return;
        case 2u: minesweeper_board(&MINESWEEPER_KNOBS[2], seed, words, extra); return;
        default: __builtin_trap();
    }
}

static uint32_t minesweeper_around(uint32_t at, uint32_t d) {
    static const int AROUND[8] = { -MINESWEEPER_STRIDE - 1, -MINESWEEPER_STRIDE, -MINESWEEPER_STRIDE + 1, -1, 1, MINESWEEPER_STRIDE - 1, MINESWEEPER_STRIDE, MINESWEEPER_STRIDE + 1 };
    return at + (uint32_t)AROUND[d];
}

static uint32_t minesweeper_mined(const uint32_t *mine, uint32_t at) { return mine[at / MINESWEEPER_STRIDE] >> (at & (MINESWEEPER_STRIDE - 1u)) & 1u; }

typedef struct { uint8_t *cell; const uint32_t *mine; uint32_t *open, *pending, *queued, x, y; } minesweeperReveal;

static inline __attribute__((always_inline)) uint32_t minesweeper_show(const minesweeperReveal *r, uint32_t at) {
    uint32_t x = (at & (MINESWEEPER_STRIDE - 1u)) - 1u, y = at / MINESWEEPER_STRIDE - 1u;
    uint32_t n = (uint32_t)__builtin_popcount((r->mine[y] >> x & 7u) | (r->mine[y + 1u] >> x & 7u) << 3 | (r->mine[y + 2u] >> x & 7u) << 6);
    r->cell[at] = (uint8_t)(MINESWEEPER_SHOWN + n);
    r->open[y] |= 2u << x;
    if (x - r->x + 1u > 2u || y - r->y + 1u > 2u) {
        r->pending[y] |= 1u << x;
        *r->queued |= 1u << y;
    }
    return n;
}

static __attribute__((noinline)) void minesweeper_open(const minesweeperKnobs *k, uint32_t *words, uint32_t *extra, uint32_t at) {
    minesweeperReveal r = { (uint8_t *)extra, extra + minesweeper_mines(k), extra + minesweeper_opens(k), extra + minesweeper_pending(k), extra + minesweeper_queued(k), words[MINESWEEPER_X], words[MINESWEEPER_Y] };
    uint16_t stack[MINESWEEPER_STRIDE * MINESWEEPER_STRIDE];
    uint32_t top = 0u, opened = 1u;
    if (minesweeper_show(&r, at) == 0u) stack[top++] = (uint16_t)at;
    while (top) {
        uint32_t zero = stack[--top];
        for (uint32_t d = 0u; d < 8u; d++) {
            uint32_t next = minesweeper_around(zero, d);
            if (r.cell[next] != MINESWEEPER_HIDDEN) continue;
            opened++;
            if (minesweeper_show(&r, next) == 0u) stack[top++] = (uint16_t)next;
        }
    }
    words[MINESWEEPER_OPENED] += opened;
}

static float minesweeper_end(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t *done, float reward) {
    words[MINESWEEPER_GAME]++;
    minesweeper_deal(instance, seed, words, extra);
    *done = 1u;
    return reward;
}

static float minesweeper_uncover(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t *done, uint32_t at) {
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    uint32_t *mine = extra + minesweeper_mines(k), shifted = minesweeper_mined(mine, at);
    if (shifted) {
        if (words[MINESWEEPER_OPENED]) return minesweeper_end(instance, seed, words, extra, done, -1.0f);
        uint32_t row = 1u, interior = (2u << k->width) - 2u;
        while (!(~mine[row] & interior)) row++;
        uint32_t free = ~mine[row] & interior;
        mine[row] |= free & (0u - free);
        mine[at / MINESWEEPER_STRIDE] &= ~(1u << (at & (MINESWEEPER_STRIDE - 1u)));
    }
    minesweeper_open(k, words, extra, at);
    return !shifted && words[MINESWEEPER_OPENED] == k->width * k->height - k->mines ? minesweeper_end(instance, seed, words, extra, done, 1.0f) : 0.0f;
}

static __attribute__((noinline)) float minesweeper_press(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t *done) {
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    const uint8_t *cell = (const uint8_t *)extra;
    const uint32_t *mine = extra + minesweeper_mines(k);
    uint32_t at = (words[MINESWEEPER_Y] + 1u) * MINESWEEPER_STRIDE + words[MINESWEEPER_X] + 1u, code = cell[at];
    uint32_t safe = k->width * k->height - k->mines, opened = words[MINESWEEPER_OPENED];
    if (code == MINESWEEPER_HIDDEN) return minesweeper_uncover(instance, seed, words, extra, done, at);
    if (code < MINESWEEPER_SHOWN) return 0.0f;
    uint32_t flags = 0u;
    for (uint32_t d = 0u; d < 8u; d++) flags += cell[minesweeper_around(at, d)] == MINESWEEPER_FLAGGED;
    if (flags != code - MINESWEEPER_SHOWN) return 0.0f;
    for (uint32_t d = 0u; d < 8u; d++) {
        uint32_t next = minesweeper_around(at, d);
        if (cell[next] == MINESWEEPER_HIDDEN && minesweeper_mined(mine, next)) return minesweeper_end(instance, seed, words, extra, done, -1.0f);
    }
    for (uint32_t d = 0u; d < 8u; d++) {
        uint32_t next = minesweeper_around(at, d);
        if (cell[next] == MINESWEEPER_HIDDEN) minesweeper_open(k, words, extra, next);
    }
    return words[MINESWEEPER_OPENED] != opened && words[MINESWEEPER_OPENED] == safe ? minesweeper_end(instance, seed, words, extra, done, 1.0f) : 0.0f;
}

static __attribute__((noinline)) float minesweeper_force(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t *done) {
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    uint8_t *cell = (uint8_t *)extra;
    uint32_t x = words[MINESWEEPER_X], y = words[MINESWEEPER_Y], best = 0xFFFFFFFFu, target = 0u;
    for (uint32_t row = 0u; row < k->height; row++)
        for (uint32_t column = 0u; column < k->width; column++) {
            uint32_t at = (row + 1u) * MINESWEEPER_STRIDE + column + 1u;
            uint32_t rank = (uint32_t)(cell[at] == MINESWEEPER_FLAGGED) << 16 | ((column > x ? column - x : x - column) + (row > y ? row - y : y - row));
            if (cell[at] < MINESWEEPER_OUTSIDE && rank < best) { best = rank; target = at; }
        }
    words[MINESWEEPER_X] = (target & (MINESWEEPER_STRIDE - 1u)) - 1u;
    words[MINESWEEPER_Y] = target / MINESWEEPER_STRIDE - 1u;
    if (cell[target] == MINESWEEPER_FLAGGED) {
        cell[target] = MINESWEEPER_HIDDEN;
        words[MINESWEEPER_FLAGS]--;
    }
    return minesweeper_uncover(instance, seed, words, extra, done, target);
}

static void minesweeper_obs(const minesweeperKnobs *k, const uint32_t *words, const uint32_t *extra, float *obs) {
    static const float CODE[MINESWEEPER_SHOWN + 9] = { 0.0f, -0.5f, -1.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
    const uint8_t *cell = (const uint8_t *)extra, *window = cell + words[MINESWEEPER_Y] * MINESWEEPER_STRIDE + words[MINESWEEPER_X];
    for (uint32_t row = 0u; row < 3u; row++)
        for (uint32_t column = 0u; column < 3u; column++) obs[row * 3u + column] = CODE[window[row * MINESWEEPER_STRIDE + column]];
    uint32_t feed = words[MINESWEEPER_FEED];
    float across = (float)(k->width - 1u), down = (float)(k->height - 1u);
    obs[9] = (float)words[MINESWEEPER_X] / across;
    obs[10] = (float)words[MINESWEEPER_Y] / down;
    obs[11] = feed ? (float)((feed & (MINESWEEPER_STRIDE - 1u)) - 1u) / across : -1.0f;
    obs[12] = feed ? (float)(feed / MINESWEEPER_STRIDE - 1u) / down : -1.0f;
    obs[13] = feed ? CODE[cell[feed]] : -1.0f;
    obs[14] = (float)((int)k->mines - (int)words[MINESWEEPER_FLAGS]) / (float)k->mines;
    obs[15] = 1.0f;
    obs[16] = (float)words[MINESWEEPER_IDLE] / (float)minesweeper_patience(k);
}

static uint32_t minesweeper_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static inline __attribute__((always_inline)) float minesweeper_play(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    uint32_t action = minesweeper_action(ctrl), x = words[MINESWEEPER_X], y = words[MINESWEEPER_Y], opened = words[MINESWEEPER_OPENED], flags = words[MINESWEEPER_FLAGS];
    float reward = 0.0f;
    *done = 0u;
    if (action < 4u) {
        int dx, dy;
        gridDelta(action, &dx, &dy);
        words[MINESWEEPER_X] = x + (uint32_t)dx < k->width ? x + (uint32_t)dx : x;
        words[MINESWEEPER_Y] = y + (uint32_t)dy < k->height ? y + (uint32_t)dy : y;
    } else if (action == 4u) reward = minesweeper_press(instance, seed, words, extra, done);
    else {
        uint8_t *cell = (uint8_t *)extra + (y + 1u) * MINESWEEPER_STRIDE + x + 1u;
        uint32_t code = *cell;
        if (code < MINESWEEPER_OUTSIDE) {
            *cell = (uint8_t)(code ^ 1u);
            words[MINESWEEPER_FLAGS] = code ? words[MINESWEEPER_FLAGS] - 1u : words[MINESWEEPER_FLAGS] + 1u;
        }
    }
    uint32_t idle = *done || words[MINESWEEPER_OPENED] != opened ? 0u : words[MINESWEEPER_IDLE] + (words[MINESWEEPER_FLAGS] <= flags), forced = idle == minesweeper_patience(k);
    words[MINESWEEPER_IDLE] = forced ? 0u : idle;
    words[MINESWEEPER_FORCED] = forced;
    if (forced) reward = minesweeper_force(instance, seed, words, extra, done);
    uint32_t *queued = extra + minesweeper_queued(k), rows = *queued, feed = 0u;
    if (rows) {
        uint32_t row = (uint32_t)__builtin_ctz(rows), *pending = extra + minesweeper_pending(k) + row, bits = *pending;
        feed = (row + 1u) * MINESWEEPER_STRIDE + (uint32_t)__builtin_ctz(bits) + 1u;
        *pending = bits & (bits - 1u);
        if (!*pending) *queued = rows & (rows - 1u);
    }
    words[MINESWEEPER_FEED] = feed;
    minesweeper_obs(k, words, extra, obs);
    return reward;
}

float minesweeper_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return minesweeper_play(0u, seed, ctrl, words, extra, obs, done);
        case 1u: return minesweeper_play(1u, seed, ctrl, words, extra, obs, done);
        case 2u: return minesweeper_play(2u, seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void minesweeper_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    __builtin_memset(extra, MINESWEEPER_OUTSIDE, (k->height + 2u) * MINESWEEPER_STRIDE);
    extra[minesweeper_mines(k)] = 0u;
    extra[minesweeper_mines(k) + k->height + 1u] = 0u;
    words[MINESWEEPER_X] = (k->width - 1u) / 2u;
    words[MINESWEEPER_Y] = (k->height - 1u) / 2u;
    words[MINESWEEPER_GAME] = 0u;
    words[MINESWEEPER_FEED] = 0u;
    words[MINESWEEPER_FORCED] = 0u;
    minesweeper_deal(instance, seed, words, extra);
}

void minesweeper_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    minesweeper_obs(&MINESWEEPER_KNOBS[instance], words, extra, obs);
}

uint32_t minesweeper_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    uint32_t cells = k->width * k->height, count = (3u * cells + 31u) / 32u;
    if (!out) return count;
    const uint8_t *cell = (const uint8_t *)extra;
    const uint32_t *mine = extra + minesweeper_mines(k);
    for (uint32_t i = 0u; i < count; i++) out[i] = 0u;
    for (uint32_t y = 0u; y < k->height; y++)
        for (uint32_t x = 0u; x < k->width; x++) {
            uint32_t at = (y + 1u) * MINESWEEPER_STRIDE + x + 1u, i = y * k->width + x;
            uint32_t planes = minesweeper_mined(mine, at) | (uint32_t)(cell[at] >= MINESWEEPER_SHOWN) << 1 | (uint32_t)(cell[at] == MINESWEEPER_FLAGGED) << 2;
            for (uint32_t p = 0u; p < 3u; p++) {
                uint32_t bit = p * cells + i;
                out[bit / 32u] |= (planes >> p & 1u) << (bit & 31u);
            }
        }
    return count;
}

typedef struct { const uint8_t *cell; const uint32_t *covered; uint32_t *mined, *safe; } minesweeperBoard;

static uint32_t minesweeper_count(const uint32_t *rows) { return (uint32_t)(__builtin_popcount(rows[0]) + __builtin_popcount(rows[1]) + __builtin_popcount(rows[2])); }

static uint32_t minesweeper_clue(const minesweeperBoard *b, uint32_t x, uint32_t y, uint32_t *unknown) {
    uint32_t window = 7u << x, known = 0u;
    for (uint32_t i = 0u; i < 3u; i++) {
        unknown[i] = b->covered[y + i] & ~b->mined[y + i] & window;
        known += (uint32_t)__builtin_popcount(b->mined[y + i] & window);
    }
    return b->cell[(y + 1u) * MINESWEEPER_STRIDE + x + 1u] - MINESWEEPER_SHOWN - known;
}

static uint32_t minesweeper_mark(const minesweeperBoard *b, uint32_t mines, uint32_t y, const uint32_t *rows) {
    uint32_t *into = mines ? b->mined : b->safe;
    for (uint32_t i = 0u; i < 3u; i++) into[y + i] |= rows[i];
    return mines ? 1u : 2u;
}

static uint32_t minesweeper_single(const minesweeperBoard *b, uint32_t x, uint32_t y) {
    uint32_t unknown[3], left = minesweeper_clue(b, x, y, unknown), count = minesweeper_count(unknown);
    return !count || (left && left != count) ? 0u : minesweeper_mark(b, left != 0u, y, unknown);
}

static uint32_t minesweeper_pair(const minesweeperBoard *b, uint32_t x, uint32_t y, uint32_t bx, uint32_t by) {
    uint32_t a[3], rest[3], shared = 0u;
    int left = (int)minesweeper_clue(b, x, y, a), right = (int)minesweeper_clue(b, bx, by, rest), sizeA = (int)minesweeper_count(a), sizeB = (int)minesweeper_count(rest);
    for (uint32_t i = 0u; i < 3u; i++) {
        uint32_t j = by + i - y, overlap = j < 3u ? rest[i] & a[j] : 0u;
        shared += (uint32_t)__builtin_popcount(overlap);
        rest[i] &= ~overlap;
    }
    int both = (int)shared, onlyA = sizeA - both, onlyB = sizeB - both;
    int least = left - onlyA > right - onlyB ? left - onlyA : right - onlyB, most = both < left ? both : left;
    least = least > 0 ? least : 0;
    most = most < right ? most : right;
    return !onlyB ? 0u : right - most == onlyB ? minesweeper_mark(b, 1u, by, rest) : right == least ? minesweeper_mark(b, 0u, by, rest) : 0u;
}

static uint32_t minesweeper_deduce(const minesweeperKnobs *k, const minesweeperBoard *b, const uint32_t *open) {
    uint32_t edge[MINESWEEPER_STRIDE], found = 1u;
    for (uint32_t y = 0u; y < k->height; y++) {
        uint32_t near = b->covered[y] | b->covered[y + 1u] | b->covered[y + 2u];
        edge[y] = open[y] & (near | near << 1 | near >> 1);
    }
    while (found == 1u) {
        found = 0u;
        for (uint32_t y = 0u; y < k->height && found < 2u; y++)
            for (uint32_t bits = edge[y]; bits && found < 2u; bits &= bits - 1u) found |= minesweeper_single(b, (uint32_t)__builtin_ctz(bits) - 1u, y);
        for (uint32_t y = 0u; y < k->height && !found; y++)
            for (uint32_t bits = edge[y]; bits && !found; bits &= bits - 1u) {
                uint32_t x = (uint32_t)__builtin_ctz(bits) - 1u;
                for (uint32_t by = y < 2u ? 0u : y - 2u; by < k->height && by <= y + 2u && !found; by++)
                    for (uint32_t near = edge[by] & (0x1Fu << x) >> 1 & ~(by == y ? bits & (0u - bits) : 0u); near && !found; near &= near - 1u) found = minesweeper_pair(b, x, y, (uint32_t)__builtin_ctz(near) - 1u, by);
            }
    }
    return found;
}

static void minesweeper_guess(const minesweeperKnobs *k, const minesweeperBoard *b, uint32_t *pick) {
    uint32_t known = 0u, unknown = 0u, bestRisk = 1u, bestOf = 0u;
    for (uint32_t y = 1u; y <= k->height; y++) {
        known += (uint32_t)__builtin_popcount(b->mined[y]);
        unknown += (uint32_t)__builtin_popcount(b->covered[y] & ~b->mined[y]);
    }
    for (uint32_t y = 0u; y < k->height; y++)
        for (uint32_t bits = b->covered[y + 1u] & ~b->mined[y + 1u]; bits; bits &= bits - 1u) {
            uint32_t at = (y + 1u) * MINESWEEPER_STRIDE + (uint32_t)__builtin_ctz(bits), risk = 0u, of = 1u;
            for (uint32_t d = 0u; d < 8u; d++) {
                uint32_t next = minesweeper_around(at, d), around[3];
                if (b->cell[next] <= MINESWEEPER_SHOWN) continue;
                uint32_t left = minesweeper_clue(b, (next & (MINESWEEPER_STRIDE - 1u)) - 1u, next / MINESWEEPER_STRIDE - 1u, around), among = minesweeper_count(around);
                if (left * of > risk * among) { risk = left; of = among; }
            }
            if (!risk) { risk = k->mines - known; of = unknown; }
            if (risk * bestOf < bestRisk * of) {
                for (uint32_t r = 0u; r <= y; r++) pick[r] = 0u;
                bestRisk = risk; bestOf = of;
            }
            if (risk * bestOf == bestRisk * of) pick[y] |= bits & (0u - bits);
        }
}

void minesweeper_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const minesweeperKnobs *k = &MINESWEEPER_KNOBS[instance];
    const uint8_t *cell = (const uint8_t *)extra;
    const uint32_t *open = extra + minesweeper_opens(k);
    uint32_t covered[MINESWEEPER_STRIDE] = { 0u }, mined[MINESWEEPER_STRIDE] = { 0u }, safe[MINESWEEPER_STRIDE] = { 0u }, pick[MINESWEEPER_STRIDE] = { 0u }, interior = (2u << k->width) - 2u;
    for (uint32_t y = 0u; y < k->height; y++) covered[y + 1u] = interior & ~open[y];
    minesweeperBoard board = { cell, covered, mined, safe };
    const uint32_t *target = safe + 1u;
    if (!minesweeper_deduce(k, &board, open)) {
        minesweeper_guess(k, &board, pick);
        target = pick;
    }
    uint32_t x = words[MINESWEEPER_X], y = words[MINESWEEPER_Y], place = x + 1u, best = 0xFFFFFFFFu, tx = x, ty = y;
    for (uint32_t r = 0u; r < k->height; r++) {
        uint32_t bits = target[r];
        if (!bits) continue;
        uint32_t left = bits & ((2u << place) - 1u), right = bits & ~((1u << place) - 1u), column = 0u, reach = 0xFFFFFFFFu;
        if (left) { column = 31u - (uint32_t)__builtin_clz(left); reach = place - column; }
        if (right && (uint32_t)__builtin_ctz(right) - place < reach) { column = (uint32_t)__builtin_ctz(right); reach = column - place; }
        uint32_t distance = reach + (r > y ? r - y : y - r);
        if (distance < best) { best = distance; tx = column - 1u; ty = r; }
    }
    uint32_t action = tx < x ? 2u : tx > x ? 3u : ty < y ? 0u : ty > y ? 1u : cell[(y + 1u) * MINESWEEPER_STRIDE + x + 1u] == MINESWEEPER_FLAGGED ? 5u : 4u;
    ctrl[0] = (float)action;
}
