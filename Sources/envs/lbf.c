#include "../env.h"

typedef struct { uint32_t field, players, foods, sight, levels, coop, steps, partner; float penalty; } lbfKnobs;

static const lbfKnobs LBF_KNOBS[4] = {
    { 8u, 2u, 2u, 8u, 2u, 0u, 50u, 1u, 0.0f },
    { 8u, 2u, 2u, 8u, 2u, 1u, 50u, 4u, 0.0f },
    { 8u, 2u, 2u, 2u, 2u, 1u, 50u, 4u, 0.0f },
    { 10u, 3u, 3u, 10u, 2u, 0u, 50u, 0u, 0.1f }
};

enum { LBF_NONE, LBF_NORTH, LBF_SOUTH, LBF_WEST, LBF_EAST, LBF_LOAD, LBF_ACTIONS };
enum { LBF_H1 = 1, LBF_H2, LBF_H3, LBF_H4 };
enum { LBF_PLAYER = 0, LBF_FOOD = 3, LBF_CLOCK = 6, LBF_SPAWNED, LBF_EPISODE, LBF_KINDS, LBF_LAST, LBF_WORDS };
enum { LBF_SPAWN = 1, LBF_CHOICE = 2, LBF_MAX = 3, LBF_ROW = 16, LBF_CELLS = 10 * LBF_ROW, LBF_STRIDE = 8, LBF_MASK = 63 };
enum { LBF_TAG = LBF_CELLS / 4, LBF_FILLED, LBF_DRAWS, LBF_EXTRA = LBF_DRAWS + 4 * ENV_BATCH, LBF_LINE = 11 };

typedef struct { uint32_t seed, counter, tag, block, used, count, bits[4 * ENV_BATCH]; } lbfStream;
typedef __uint128_t lbfWide;

static int lbf_row(uint32_t entity) { return (int)(entity >> 4 & 15u); }

static int lbf_col(uint32_t entity) { return (int)(entity & 15u); }

static int lbf_level(uint32_t entity) { return (int)(entity >> 8); }

static int lbf_far(int dr, int dc) { return (dr < 0 ? -dr : dr) + (dc < 0 ? -dc : dc); }

uint32_t lbf_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return LBF_EXTRA;
}

static __attribute__((noinline)) void lbf_refill(lbfStream *s) {
    u4 r = philox4x32(s->seed, ENV_RNG_STREAM, s->counter, s->tag, s->block++, 0u);
    s->bits[0] = r.x;
    s->bits[1] = r.y;
    s->bits[2] = r.z;
    s->bits[3] = r.w;
    s->used = 0u;
    s->count = 4u;
}

static inline __attribute__((always_inline)) uint32_t lbf_word(lbfStream *s) {
    if (s->used == s->count) lbf_refill(s);
    return s->bits[s->used++];
}

static void lbf_draw(lbfStream *s, uint32_t seed, uint32_t counter, uint32_t tag) {
    u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)counter, (lane4)tag, (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u);
    uint32x4x4_t lanes = { { (uint32x4_t)r.x, (uint32x4_t)r.y, (uint32x4_t)r.z, (uint32x4_t)r.w } };
    vst4q_u32(s->bits, lanes);
    s->seed = seed;
    s->counter = counter;
    s->tag = tag;
    s->block = ENV_BATCH;
    s->used = 0u;
    s->count = 4u * ENV_BATCH;
}

static inline __attribute__((always_inline)) uint32_t lbf_below(lbfStream *s, uint32_t n) {
    uint64_t m = (uint64_t)lbf_word(s) * n;
    if ((uint32_t)m < n) {
        uint32_t floor = (0u - n) % n;
        while ((uint32_t)m < floor) m = (uint64_t)lbf_word(s) * n;
    }
    return (uint32_t)(m >> 32);
}

static uint32_t lbf_select(lbfWide mask, uint32_t index) {
    uint64_t half = (uint64_t)mask;
    uint32_t base = 0u, low = (uint32_t)__builtin_popcountll(half);
    if (index >= low) {
        index -= low;
        half = (uint64_t)(mask >> 64);
        base = 64u;
    }
    uint64_t prefix = (uint64_t)vcnt_u8(vcreate_u8(half)) * 0x0101010101010101ull;
    uint64_t above = ((prefix | 0x8080808080808080ull) - 0x0101010101010101ull * (index + 1u)) & 0x8080808080808080ull;
    uint32_t byte = (uint32_t)__builtin_popcountll(~above & 0x8080808080808080ull);
    uint32_t bits = (uint32_t)(half >> (8u * byte)) & 255u;
    for (index -= byte ? (uint32_t)(prefix >> (8u * byte - 8u)) & 255u : 0u; index > 0u; index--) bits &= bits - 1u;
    return base + 8u * byte + (uint32_t)__builtin_ctz(bits);
}

static uint64_t lbf_zone(int r, int c, int span) {
    uint64_t columns = ((uint64_t)1 << span) - 1u, one = (uint64_t)1 << c;
    uint64_t three = ((uint64_t)7 << c >> 1) & columns, five = ((uint64_t)0x11 << c >> 2) & columns;
    uint64_t zone = (three | five) << (LBF_STRIDE * r);
    if (r > 0) zone |= three << (LBF_STRIDE * (r - 1));
    if (r + 1 < span) zone |= three << (LBF_STRIDE * (r + 1));
    if (r > 1) zone |= one << (LBF_STRIDE * (r - 2));
    if (r + 2 < span) zone |= one << (LBF_STRIDE * (r + 2));
    return zone;
}

static void lbf_mark(uint8_t *code, uint32_t cell, uint32_t food, uint32_t alive) {
    uint32_t load = 1u << LBF_LOAD | food << 6;
    if (alive) {
        code[cell - LBF_ROW] = (uint8_t)((code[cell - LBF_ROW] & ~(1u << LBF_SOUTH)) | load);
        code[cell + LBF_ROW] = (uint8_t)((code[cell + LBF_ROW] & ~(1u << LBF_NORTH)) | load);
        code[cell - 1u] = (uint8_t)((code[cell - 1u] & ~(1u << LBF_EAST)) | load);
        code[cell + 1u] = (uint8_t)((code[cell + 1u] & ~(1u << LBF_WEST)) | load);
        return;
    }
    code[cell - LBF_ROW] = (uint8_t)((code[cell - LBF_ROW] & 31u) | 1u << LBF_SOUTH);
    code[cell + LBF_ROW] = (uint8_t)((code[cell + LBF_ROW] & 31u) | 1u << LBF_NORTH);
    code[cell - 1u] = (uint8_t)((code[cell - 1u] & 31u) | 1u << LBF_EAST);
    code[cell + 1u] = (uint8_t)((code[cell + 1u] & 31u) | 1u << LBF_WEST);
}

static inline __attribute__((always_inline)) void lbf_codes(const lbfKnobs *k, const uint32_t *words, uint8_t *code) {
    int side = (int)k->field;
    uint8_t row[LBF_ROW];
    for (int c = 0; c < LBF_ROW; c++)
        row[c] = (uint8_t)((uint32_t)(c < side) * (1u << LBF_NONE | 1u << LBF_NORTH | 1u << LBF_SOUTH | (uint32_t)(c > 0) << LBF_WEST | (uint32_t)(c + 1 < side) << LBF_EAST));
    uint8x16_t middle = vld1q_u8(row);
    vst1q_u8(code, vbicq_u8(middle, vdupq_n_u8(1u << LBF_NORTH)));
    for (int r = 1; r + 1 < side; r++) vst1q_u8(code + r * LBF_ROW, middle);
    vst1q_u8(code + (side - 1) * LBF_ROW, vbicq_u8(middle, vdupq_n_u8(1u << LBF_SOUTH)));
    for (uint32_t j = 0u; j < k->foods; j++)
        if (lbf_level(words[LBF_FOOD + j])) lbf_mark(code, words[LBF_FOOD + j] & 255u, j, 1u);
}

static inline __attribute__((always_inline)) void lbf_spawn(const lbfKnobs *k, uint32_t seed, uint32_t *words, uint8_t *code, uint32_t stale) {
    lbfStream s;
    lbf_draw(&s, seed, words[LBF_EPISODE], LBF_SPAWN);
    int side = (int)k->field, span = side - 2, sum = 0;
    lbfWide board = ((lbfWide)1 << (side * side)) - 1u;
    for (uint32_t i = 0u; i < k->players; i++) {
        lbfWide blocked = 0u;
        for (uint32_t j = 0u; j < k->players; j++)
            if (j < i || stale) blocked |= (lbfWide)1 << (lbf_row(words[LBF_PLAYER + j]) * side + lbf_col(words[LBF_PLAYER + j]));
        lbfWide open = board & ~blocked;
        int cell = (int)lbf_select(open, lbf_below(&s, (uint32_t)(__builtin_popcountll((uint64_t)open) + __builtin_popcountll((uint64_t)(open >> 64)))));
        int level = 1 + (int)lbf_below(&s, k->levels);
        sum += level;
        words[LBF_PLAYER + i] = (uint32_t)((cell / side) << 4 | cell % side) | (uint32_t)level << 8;
    }
    uint64_t interior = 0u, taken = 0u;
    for (int r = 0; r < span; r++) interior |= (((uint64_t)1 << span) - 1u) << (LBF_STRIDE * r);
    for (uint32_t i = 0u; i < k->players; i++) {
        int r = lbf_row(words[LBF_PLAYER + i]) - 1, c = lbf_col(words[LBF_PLAYER + i]) - 1;
        if (r >= 0 && r < span && c >= 0 && c < span) taken |= (uint64_t)1 << (LBF_STRIDE * r + c);
    }
    uint32_t food[LBF_MAX], placed = 0u, spawned = 0u;
    for (uint32_t j = 0u; j < k->foods; j++) {
        uint64_t open = interior & ~taken;
        if (!open) break;
        uint32_t bit = lbf_select(open, lbf_below(&s, (uint32_t)__builtin_popcountll(open)));
        int r = (int)(bit / LBF_STRIDE), c = (int)(bit % LBF_STRIDE);
        int level = k->coop ? sum : 1 + (int)lbf_below(&s, (uint32_t)sum);
        taken |= lbf_zone(r, c, span);
        spawned += (uint32_t)level;
        uint32_t entity = (uint32_t)((r + 1) << 4 | (c + 1)) | (uint32_t)level << 8, at = placed++;
        while (at > 0u && (food[at - 1u] & 255u) > (entity & 255u)) {
            food[at] = food[at - 1u];
            at--;
        }
        food[at] = entity;
    }
    for (uint32_t j = 0u; j < k->foods; j++) words[LBF_FOOD + j] = j < placed ? food[j] : 0u;
    uint32_t kinds = 0u;
    for (uint32_t i = 1u; i < k->players; i++) kinds |= (k->partner ? k->partner : 1u + lbf_below(&s, 4u)) << (4u * i);
    words[LBF_KINDS] = kinds;
    words[LBF_SPAWNED] = spawned;
    words[LBF_CLOCK] = 0u;
    lbf_codes(k, words, code);
}

static uint32_t lbf_nth(uint32_t valid, uint32_t index) {
    for (; index > 0u; index--) valid &= valid - 1u;
    return (uint32_t)__builtin_ctz(valid);
}

static inline __attribute__((always_inline)) uint32_t lbf_pick(uint32_t valid, lbfStream *s) { return lbf_nth(valid, lbf_below(s, (uint32_t)__builtin_popcount(valid))); }

static int lbf_round(int sum, int count) { return count == 1 ? sum : count == 2 ? (sum + (sum >> 1 & 1)) >> 1 : (sum + 1) / 3; }

static inline __attribute__((always_inline)) uint32_t lbf_intent(const lbfKnobs *k, const uint32_t *words, uint32_t i, uint32_t kind, uint32_t valid, int *target) {
    int sight = (int)k->sight, side = (int)k->field, full = sight + 1 >= side;
    uint32_t self = words[LBF_PLAYER + i];
    int r = lbf_row(self), c = lbf_col(self);
    int top = full || r <= sight ? 0 : r - sight, left = full || c <= sight ? 0 : c - sight;
    int bottom = full || r + sight >= side ? side - 1 : r + sight, right = full || c + sight >= side ? side - 1 : c + sight;
    int refR = r - top, refC = c - left, levels = lbf_level(self);
    if (kind == LBF_H2 || kind == LBF_H4) {
        int count = 0, sumR = 0, sumC = 0;
        levels = 0;
        for (uint32_t j = 0u; j < k->players; j++) {
            uint32_t p = words[LBF_PLAYER + j];
            int pr = lbf_row(p) - top, pc = lbf_col(p) - left;
            if (!full && j != i && (pr < 0 || pr > 2 * sight || pc < 0 || pc > 2 * sight)) continue;
            count++;
            sumR += pr;
            sumC += pc;
            levels += lbf_level(p);
        }
        refR = lbf_round(sumR, count);
        refC = lbf_round(sumC, count);
    }
    int best = -1, bestDistance = 0;
    for (uint32_t j = 0u; j < k->foods; j++) {
        uint32_t f = words[LBF_FOOD + j];
        int level = lbf_level(f), fr = lbf_row(f), fc = lbf_col(f);
        if (!level || (!full && (fr < top || fr > bottom || fc < left || fc > right))) continue;
        if ((kind == LBF_H3 && level > lbf_level(self)) || (kind == LBF_H4 && level > levels)) continue;
        int dr = fr - top - refR, dc = fc - left - refC, distance = dr * dr + dc * dc;
        if (best < 0 || distance < bestDistance) {
            best = (int)j;
            bestDistance = distance;
        }
    }
    *target = best;
    if (best < 0) return LBF_ACTIONS;
    int dr = lbf_row(words[LBF_FOOD + best]) - r, dc = lbf_col(words[LBF_FOOD + best]) - c;
    if (lbf_far(dr, dc) == 1) return LBF_LOAD;
    if (dr < 0 && (valid >> LBF_NORTH & 1u)) return LBF_NORTH;
    if (dr > 0 && (valid >> LBF_SOUTH & 1u)) return LBF_SOUTH;
    if (dc > 0 && (valid >> LBF_EAST & 1u)) return LBF_EAST;
    if (dc < 0 && (valid >> LBF_WEST & 1u)) return LBF_WEST;
    return LBF_ACTIONS;
}

static inline __attribute__((always_inline)) uint32_t lbf_partner(const lbfKnobs *k, const uint32_t *words, uint32_t i, uint32_t valid, int *target) {
    if (k->partner) return lbf_intent(k, words, i, k->partner, valid, target);
    switch (words[LBF_KINDS] >> (4u * i) & 15u) {
        case LBF_H1: return lbf_intent(k, words, i, LBF_H1, valid, target);
        case LBF_H2: return lbf_intent(k, words, i, LBF_H2, valid, target);
        case LBF_H3: return lbf_intent(k, words, i, LBF_H3, valid, target);
        case LBF_H4: return lbf_intent(k, words, i, LBF_H4, valid, target);
        default: __builtin_trap();
    }
}

static __attribute__((noinline)) void lbf_fill(uint32_t seed, uint32_t *extra) {
    u44 r = philox4(seed, ENV_RNG_STREAM, extra[LBF_TAG] + (lane4){ 0u, 1u, 2u, 3u }, (lane4)LBF_CHOICE, (lane4)0u, (lane4)0u);
    __builtin_memcpy(extra + LBF_DRAWS, &r, sizeof r);
    extra[LBF_FILLED] = 1u;
}

static __attribute__((noinline)) uint32_t lbf_rare(uint32_t seed, uint32_t step, const uint32_t *extra, uint32_t slot, uint32_t first, uint32_t second) {
    const uint32_t *d = extra + LBF_DRAWS;
    lbfStream s = { seed, step, LBF_CHOICE, 1u, 0u, 4u, { d[slot], d[ENV_BATCH + slot], d[2 * ENV_BATCH + slot], d[3 * ENV_BATCH + slot] } };
    uint32_t picks = first ? lbf_pick(first, &s) : 0u;
    return picks | (second ? lbf_pick(second, &s) : 0u) << 4;
}

static inline __attribute__((always_inline)) uint32_t lbf_chance(uint32_t seed, uint32_t step, uint32_t *extra, uint32_t slot, uint32_t first, uint32_t second) {
    if (!extra[LBF_FILLED]) lbf_fill(seed, extra);
    const uint32_t *d = extra + LBF_DRAWS;
    uint32_t picks = 0u, word = d[slot];
    if (first) {
        uint32_t n = (uint32_t)__builtin_popcount(first);
        uint64_t m = (uint64_t)word * n;
        if ((uint32_t)m < n) return lbf_rare(seed, step, extra, slot, first, second);
        picks = lbf_nth(first, (uint32_t)(m >> 32));
        word = d[ENV_BATCH + slot];
    }
    if (second) {
        uint32_t n = (uint32_t)__builtin_popcount(second);
        uint64_t m = (uint64_t)word * n;
        if ((uint32_t)m < n) return lbf_rare(seed, step, extra, slot, first, second);
        picks |= lbf_nth(second, (uint32_t)(m >> 32)) << 4;
    }
    return picks;
}

static inline __attribute__((always_inline)) void lbf_decide(const lbfKnobs *k, const uint32_t *words, uint32_t i, const uint32_t *valid, uint32_t *act, uint32_t *random) {
    int target;
    act[i] = lbf_partner(k, words, i, valid[i], &target);
    random[i] = act[i] == LBF_ACTIONS ? valid[i] & LBF_MASK : 0u;
}

static uint32_t lbf_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)LBF_ACTIONS ? (uint32_t)c : LBF_NONE;
}

static inline __attribute__((always_inline)) void lbf_triple(float *obs, int r, int c, int level) {
    obs[0] = (float)r;
    obs[1] = (float)c;
    obs[2] = (float)level;
}

static inline __attribute__((always_inline)) void lbf_obs(const lbfKnobs *k, const uint32_t *words, float *obs) {
    int sight = (int)k->sight, side = (int)k->field, full = sight + 1 >= side;
    uint32_t self = words[LBF_PLAYER];
    int r = lbf_row(self), c = lbf_col(self);
    int top = full || r <= sight ? 0 : r - sight, left = full || c <= sight ? 0 : c - sight;
    int bottom = full || r + sight >= side ? side - 1 : r + sight, right = full || c + sight >= side ? side - 1 : c + sight;
    uint32_t at = 0u;
    for (uint32_t j = 0u; j < k->foods; j++) {
        uint32_t f = words[LBF_FOOD + j];
        int fr = lbf_row(f), fc = lbf_col(f);
        if (!lbf_level(f) || (!full && (fr < top || fr > bottom || fc < left || fc > right))) continue;
        lbf_triple(obs + at, fr - top, fc - left, lbf_level(f));
        at += 3u;
    }
    for (; at < 3u * k->foods; at += 3u) lbf_triple(obs + at, -1, -1, 0);
    lbf_triple(obs + at, r - top, c - left, lbf_level(self));
    at += 3u;
    for (uint32_t i = 1u; i < k->players; i++) {
        uint32_t p = words[LBF_PLAYER + i];
        int pr = lbf_row(p) - top, pc = lbf_col(p) - left;
        if (!full && (pr < 0 || pr > 2 * sight || pc < 0 || pc > 2 * sight)) continue;
        lbf_triple(obs + at, pr, pc, lbf_level(p));
        at += 3u;
    }
    for (; at < 3u * (k->foods + k->players); at += 3u) lbf_triple(obs + at, -1, -1, 0);
}

static __attribute__((noinline)) void lbf_respawn(uint32_t instance, uint32_t seed, uint32_t *words, uint8_t *code, uint32_t stale) {
    switch (instance) {
        case 0u: lbf_spawn(&LBF_KNOBS[0], seed, words, code, stale); return;
        case 1u: lbf_spawn(&LBF_KNOBS[1], seed, words, code, stale); return;
        case 2u: lbf_spawn(&LBF_KNOBS[2], seed, words, code, stale); return;
        case 3u: lbf_spawn(&LBF_KNOBS[3], seed, words, code, stale); return;
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float lbf_resolve(const lbfKnobs *k, uint32_t *words, uint8_t *code, const uint32_t *valid, uint32_t *act, uint32_t *done) {
    uint32_t cell[LBF_MAX], target[LBF_MAX], last = 0u;
    for (uint32_t i = 0u; i < k->players; i++) {
        uint32_t a = valid[i] >> act[i] & 1u ? act[i] : LBF_NONE;
        int dx = 0, dy = 0;
        if (a >= LBF_NORTH && a <= LBF_EAST) gridDelta(a - LBF_NORTH, &dx, &dy);
        cell[i] = words[LBF_PLAYER + i] & 255u;
        target[i] = cell[i] + (uint32_t)(dy * LBF_ROW + dx);
        act[i] = a;
        last |= a << (4u * i);
    }
    for (uint32_t i = 0u; i < k->players; i++) {
        uint32_t clash = 0u;
        for (uint32_t j = 0u; j < k->players; j++) clash |= (uint32_t)(j != i) & (uint32_t)(target[j] == target[i]);
        if (!clash) words[LBF_PLAYER + i] = (words[LBF_PLAYER + i] & ~255u) | target[i];
    }
    uint32_t sums = 0u, groups = 0u;
    for (uint32_t i = 0u; i < k->players; i++) {
        uint32_t loads = act[i] == LBF_LOAD, shift = 8u * (valid[i] >> 6);
        sums += loads ? (words[LBF_PLAYER + i] >> 8) << shift : 0u;
        groups |= loads << (shift + i);
    }
    float reward = 0.0f;
    uint32_t alive = 0u;
    for (uint32_t j = 0u; j < k->foods; j++) {
        uint32_t f = words[LBF_FOOD + j], group = groups >> (8u * j) & 7u, sum = sums >> (8u * j) & 255u, level = f >> 8;
        if (group && sum >= level) {
            if (group & 1u) reward = (float)((words[LBF_PLAYER] >> 8) * level) / (float)(sum * words[LBF_SPAWNED]);
            words[LBF_FOOD + j] = f & 255u;
            lbf_mark(code, f & 255u, j, 0u);
            continue;
        }
        if (group & 1u) reward = 0.0f - k->penalty;
        alive += level != 0u;
    }
    words[LBF_LAST] = last;
    uint32_t clock = words[LBF_CLOCK] + 1u;
    words[LBF_CLOCK] = clock;
    *done = alive == 0u || clock >= k->steps;
    return reward;
}

static inline __attribute__((always_inline)) float lbf_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const lbfKnobs *k = &LBF_KNOBS[instance];
    uint8_t *code = (uint8_t *)extra;
    uint32_t valid[LBF_MAX], act[LBF_MAX], slot = batchSlot(extra + LBF_TAG, step), random[LBF_MAX] = { 0u, 0u, 0u };
    if (slot == ENV_BATCH) {
        extra[LBF_FILLED] = 0u;
        slot = 0u;
    }
    for (uint32_t i = 0u; i < k->players; i++) valid[i] = code[words[LBF_PLAYER + i] & 255u];
    act[0] = lbf_action(ctrl);
    lbf_decide(k, words, 1u, valid, act, random);
    if (k->players > 2u) lbf_decide(k, words, 2u, valid, act, random);
    if (random[1] | random[2]) {
        uint32_t picks = lbf_chance(seed, step, extra, slot, random[1], random[2]);
        if (random[1]) act[1] = picks & 15u;
        if (random[2]) act[2] = picks >> 4;
    }
    float reward = lbf_resolve(k, words, code, valid, act, done);
    if (*done) {
        words[LBF_EPISODE]++;
        lbf_respawn(instance, seed, words, code, 1u);
    }
    lbf_obs(k, words, obs);
    return reward;
}

void lbf_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    for (uint32_t i = 0u; i < LBF_WORDS; i++) words[i] = 0u;
    batchReset(extra + LBF_TAG);
    extra[LBF_FILLED] = 0u;
    lbf_respawn(instance, seed, words, (uint8_t *)extra, 0u);
}

float lbf_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return lbf_play(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return lbf_play(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return lbf_play(2u, seed, step, ctrl, words, extra, obs, done);
        case 3u: return lbf_play(3u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void lbf_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    switch (instance) {
        case 0u: lbf_obs(&LBF_KNOBS[0], words, obs); return;
        case 1u: lbf_obs(&LBF_KNOBS[1], words, obs); return;
        case 2u: lbf_obs(&LBF_KNOBS[2], words, obs); return;
        case 3u: lbf_obs(&LBF_KNOBS[3], words, obs); return;
        default: __builtin_trap();
    }
}

static lbfWide lbf_cell(int r, int c) { return (lbfWide)1 << (r * LBF_LINE + c); }

static lbfWide lbf_grow(lbfWide b) { return b | b << 1 | b >> 1 | b << LBF_LINE | b >> LBF_LINE; }

static int lbf_route(lbfWide open, lbfWide goal, lbfWide from, lbfWide *before) {
    lbfWide seen = goal, layer = goal, last = 0;
    int distance = 0;
    while (!(layer & from)) {
        last = layer;
        layer = lbf_grow(layer) & open & ~seen;
        if (!layer) return -1;
        seen |= layer;
        distance++;
    }
    *before = last;
    return distance;
}

static int lbf_walk(const lbfKnobs *k, const uint32_t *words, uint32_t i, int j, int *endR, int *endC) {
    int r = lbf_row(words[LBF_PLAYER + i]), c = lbf_col(words[LBF_PLAYER + i]), fr = lbf_row(words[LBF_FOOD + j]), fc = lbf_col(words[LBF_FOOD + j]);
    for (int steps = 0; steps < 4 * (int)k->field; steps++) {
        int dr = fr - r, dc = fc - c;
        if (lbf_far(dr, dc) == 1) {
            *endR = r;
            *endC = c;
            return steps;
        }
        int nr = r + (dr < 0 ? -1 : dr > 0 ? 1 : 0), nc = c;
        uint32_t blocked = nr == r;
        for (uint32_t f = 0u; f < k->foods && !blocked; f++) blocked = lbf_level(words[LBF_FOOD + f]) && lbf_row(words[LBF_FOOD + f]) == nr && lbf_col(words[LBF_FOOD + f]) == nc;
        if (blocked) {
            nr = r;
            nc = c + (dc < 0 ? -1 : dc > 0 ? 1 : 0);
            blocked = nc == c;
            for (uint32_t f = 0u; f < k->foods && !blocked; f++) blocked = lbf_level(words[LBF_FOOD + f]) && lbf_row(words[LBF_FOOD + f]) == nr && lbf_col(words[LBF_FOOD + f]) == nc;
            if (blocked) return -1;
        }
        r = nr;
        c = nc;
    }
    return -1;
}

static inline __attribute__((always_inline)) void lbf_plan(const lbfKnobs *k, const uint32_t *words, const uint8_t *code, float *ctrl) {
    int side = (int)k->field, r = lbf_row(words[LBF_PLAYER]), c = lbf_col(words[LBF_PLAYER]), mine = lbf_level(words[LBF_PLAYER]);
    uint32_t valid[LBF_MAX], intent[LBF_MAX] = { LBF_NONE, LBF_NONE, LBF_NONE };
    int target[LBF_MAX] = { -1, -1, -1 }, arrive[LBF_MAX] = { 0, 0, 0 }, nextR[LBF_MAX] = { 0, 0, 0 }, nextC[LBF_MAX] = { 0, 0, 0 };
    lbfWide grid = 0, foods = 0, stay = 0, reserved[LBF_MAX] = { 0, 0, 0 };
    for (int row = 0; row < side; row++) grid |= (((lbfWide)1 << side) - 1) << (row * LBF_LINE);
    for (uint32_t j = 0u; j < k->foods; j++)
        if (lbf_level(words[LBF_FOOD + j])) foods |= lbf_cell(lbf_row(words[LBF_FOOD + j]), lbf_col(words[LBF_FOOD + j]));
    for (uint32_t i = 0u; i < k->players; i++) valid[i] = code[words[LBF_PLAYER + i] & 255u];
    for (uint32_t i = 1u; i < k->players; i++) {
        int pr = lbf_row(words[LBF_PLAYER + i]), pc = lbf_col(words[LBF_PLAYER + i]);
        intent[i] = lbf_partner(k, words, i, valid[i], &target[i]);
        int dx = 0, dy = 0;
        if (intent[i] >= LBF_NORTH && intent[i] <= LBF_EAST) gridDelta(intent[i] - LBF_NORTH, &dx, &dy);
        nextR[i] = pr + dy;
        nextC[i] = pc + dx;
        if (intent[i] == LBF_LOAD || intent[i] == LBF_NONE) stay |= lbf_cell(pr, pc);
        if (target[i] < 0) continue;
        int er, ec;
        arrive[i] = lbf_walk(k, words, i, target[i], &er, &ec);
        if (arrive[i] < 0) arrive[i] = lbf_far(pr - lbf_row(words[LBF_FOOD + target[i]]), pc - lbf_col(words[LBF_FOOD + target[i]])) + 1;
        else reserved[target[i]] |= lbf_cell(er, ec);
    }
    lbfWide open = grid & ~foods & ~stay, me = lbf_cell(r, c), before = 0;
    int best = -1, bestDistance = 0;
    float bestRate = 0.0f;
    for (uint32_t j = 0u; j < k->foods; j++) {
        int level = lbf_level(words[LBF_FOOD + j]), fr = lbf_row(words[LBF_FOOD + j]), fc = lbf_col(words[LBF_FOOD + j]);
        if (!level) continue;
        lbfWide ahead, around = lbf_grow(lbf_cell(fr, fc)) & (open | me), goal = around & ~reserved[j];
        int distance = lbf_route(open | me, goal ? goal : around, me, &ahead);
        if (distance < 0) continue;
        int heads = 0, order[LBF_MAX];
        for (uint32_t i = 1u; i < k->players; i++) {
            if (target[i] != (int)j) continue;
            int at = heads++;
            while (at > 0 && arrive[order[at - 1]] > arrive[i]) {
                order[at] = order[at - 1];
                at--;
            }
            order[at] = (int)i;
        }
        int finish = -1, sum = mine;
        if (mine >= level) finish = distance + 1;
        else
            for (int h = 0; h < heads; h++) {
                sum += lbf_level(words[LBF_PLAYER + order[h]]);
                if (sum >= level) {
                    finish = (distance > arrive[order[h]] ? distance : arrive[order[h]]) + 1;
                    break;
                }
            }
        if (finish < 0) continue;
        int alone = 0, group = mine;
        for (int h = 0; h < heads && finish >= 0; h++) {
            alone += lbf_level(words[LBF_PLAYER + order[h]]);
            if (alone >= level && arrive[order[h]] + 1 < finish) finish = -1;
        }
        if (finish < 0) continue;
        for (int h = 0; h < heads; h++) group += arrive[order[h]] + 1 <= finish ? lbf_level(words[LBF_PLAYER + order[h]]) : 0;
        float rate = (float)(mine * level) / ((float)(group * (int)words[LBF_SPAWNED]) * (float)finish);
        if (best < 0 || rate > bestRate) {
            best = (int)j;
            bestRate = rate;
            bestDistance = distance;
            before = ahead;
        }
    }
    if (best < 0 && (int)k->sight + 1 < side) {
        for (uint32_t i = 1u; i < k->players; i++) {
            if (target[i] >= 0) continue;
            int sight = (int)k->sight, pr = lbf_row(words[LBF_PLAYER + i]), pc = lbf_col(words[LBF_PLAYER + i]);
            int top = pr > sight ? pr - sight : 0, left = pc > sight ? pc - sight : 0;
            int bottom = pr + sight < side ? pr + sight : side - 1, right = pc + sight < side ? pc + sight : side - 1;
            uint32_t seen = 0u;
            for (uint32_t j = 0u; j < k->foods; j++) {
                int fr = lbf_row(words[LBF_FOOD + j]), fc = lbf_col(words[LBF_FOOD + j]);
                seen |= (uint32_t)(lbf_level(words[LBF_FOOD + j]) != 0 && fr >= top && fr <= bottom && fc >= left && fc <= right);
            }
            if (!seen) continue;
            lbfWide window = 0;
            for (int row = top; row <= top + 2 * sight && row < side; row++)
                window |= ((((lbfWide)1 << (2 * sight + 1)) - 1) << left) << (row * LBF_LINE);
            window &= grid;
            lbfWide ahead;
            if (lbf_route(open | me, window & (open | me), me, &ahead) <= 0) continue;
            uint32_t move = LBF_NONE;
            for (uint32_t a = LBF_EAST; a >= LBF_NORTH; a--) {
                int dx, dy;
                gridDelta(a - LBF_NORTH, &dx, &dy);
                if ((valid[0] >> a & 1u) && (ahead & lbf_cell(r + dy, c + dx))) move = a;
            }
            ctrl[0] = (float)move;
            return;
        }
    }
    if (best < 0) {
        int nearest = 1 << 30;
        for (uint32_t j = 0u; j < k->foods; j++) {
            if (!lbf_level(words[LBF_FOOD + j])) continue;
            int fr = lbf_row(words[LBF_FOOD + j]), fc = lbf_col(words[LBF_FOOD + j]), far = 1 << 20, near = 0;
            for (uint32_t i = 1u; i < k->players; i++) {
                int m = lbf_far(lbf_row(words[LBF_PLAYER + i]) - fr, lbf_col(words[LBF_PLAYER + i]) - fc);
                if (m < far) {
                    far = m;
                    near = (int)i;
                }
            }
            lbfWide ahead, facing = 0;
            int closest = 1 << 20;
            for (uint32_t a = LBF_NORTH; a <= LBF_EAST; a++) {
                int dx, dy;
                gridDelta(a - LBF_NORTH, &dx, &dy);
                lbfWide spot = lbf_cell(fr + dy, fc + dx) & (open | me);
                int m = lbf_far(lbf_row(words[LBF_PLAYER + near]) - fr - dy, lbf_col(words[LBF_PLAYER + near]) - fc - dx);
                if (spot && m < closest) {
                    closest = m;
                    facing = spot;
                }
            }
            int distance = lbf_route(open | me, facing ? facing : lbf_grow(lbf_cell(fr, fc)) & (open | me), me, &ahead);
            if (distance < 0) continue;
            if (distance + far < nearest) {
                nearest = distance + far;
                best = (int)j;
                bestDistance = distance;
                before = ahead;
            }
        }
    }
    if (best < 0) {
        ctrl[0] = (float)LBF_NONE;
        return;
    }
    if (bestDistance == 0) {
        int level = lbf_level(words[LBF_FOOD + best]), sum = mine, fr = lbf_row(words[LBF_FOOD + best]), fc = lbf_col(words[LBF_FOOD + best]);
        for (uint32_t i = 1u; i < k->players; i++)
            if (intent[i] == LBF_LOAD && lbf_far(lbf_row(words[LBF_PLAYER + i]) - fr, lbf_col(words[LBF_PLAYER + i]) - fc) == 1) sum += lbf_level(words[LBF_PLAYER + i]);
        ctrl[0] = (float)(sum >= level || k->penalty == 0.0f ? LBF_LOAD : LBF_NONE);
        return;
    }
    uint32_t fallback = LBF_NONE;
    for (uint32_t a = LBF_NORTH; a <= LBF_EAST; a++) {
        int dx, dy;
        gridDelta(a - LBF_NORTH, &dx, &dy);
        int tr = r + dy, tc = c + dx;
        if (!(valid[0] >> a & 1u) || !(before & lbf_cell(tr, tc))) continue;
        uint32_t clash = 0u;
        for (uint32_t i = 1u; i < k->players; i++) {
            if (intent[i] == LBF_ACTIONS) clash |= (uint32_t)(lbf_row(words[LBF_PLAYER + i]) == tr && lbf_col(words[LBF_PLAYER + i]) == tc);
            else clash |= (uint32_t)(nextR[i] == tr && nextC[i] == tc);
        }
        if (!clash) {
            ctrl[0] = (float)a;
            return;
        }
        if (fallback == LBF_NONE) fallback = a;
    }
    ctrl[0] = (float)fallback;
}

void lbf_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    switch (instance) {
        case 0u: lbf_plan(&LBF_KNOBS[0], words, (const uint8_t *)extra, ctrl); return;
        case 1u: lbf_plan(&LBF_KNOBS[1], words, (const uint8_t *)extra, ctrl); return;
        case 2u: lbf_plan(&LBF_KNOBS[2], words, (const uint8_t *)extra, ctrl); return;
        case 3u: lbf_plan(&LBF_KNOBS[3], words, (const uint8_t *)extra, ctrl); return;
        default: __builtin_trap();
    }
}
