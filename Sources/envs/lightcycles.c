#include "../env.h"

typedef struct { uint32_t width, height, startLeft, startRight, startTop, startBottom, bot; } lightcyclesKnobs;

static const lightcyclesKnobs LIGHTCYCLES_KNOBS[3] = {
    { 16u, 16u, 2u, 5u, 2u, 13u, 0u },
    { 16u, 16u, 2u, 5u, 2u, 13u, 1u },
    { 16u, 16u, 2u, 5u, 2u, 13u, 2u }
};

enum { LIGHTCYCLES_RAMBLER, LIGHTCYCLES_HOARDER, LIGHTCYCLES_TACTICIAN };
enum { LIGHTCYCLES_LEFT, LIGHTCYCLES_STRAIGHT, LIGHTCYCLES_RIGHT };
enum { LIGHTCYCLES_MINE = 0, LIGHTCYCLES_THEIRS = 8, LIGHTCYCLES_AGENT = 16, LIGHTCYCLES_BOT, LIGHTCYCLES_GAME, LIGHTCYCLES_TICK, LIGHTCYCLES_LAST, LIGHTCYCLES_WORDS };
enum { LIGHTCYCLES_SENSES = 16, LIGHTCYCLES_WIN = 1024 };

static const uint8_t LIGHTCYCLES_TURN[4][3] = { { 2u, 0u, 3u }, { 3u, 1u, 2u }, { 1u, 2u, 0u }, { 0u, 3u, 1u } };
static const uint16_t LIGHTCYCLES_LANES[16] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u };

typedef struct { uint16x8_t lo, hi; } lightcyclesBoard;

typedef struct {
    float senses[LIGHTCYCLES_SENSES];
    uint16_t rows[16], cols[16];
} lightcyclesLane;

static lightcyclesLane *lightcycles_lane(uint32_t *extra) { return (lightcyclesLane *)extra; }
static const lightcyclesLane *lightcycles_peek(const uint32_t *extra) { return (const lightcyclesLane *)extra; }

uint32_t lightcycles_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)(sizeof(lightcyclesLane) / sizeof(uint32_t));
}

static inline __attribute__((always_inline)) lightcyclesBoard lightcycles_or(lightcyclesBoard a, lightcyclesBoard b) { return (lightcyclesBoard){ vorrq_u16(a.lo, b.lo), vorrq_u16(a.hi, b.hi) }; }
static inline __attribute__((always_inline)) lightcyclesBoard lightcycles_without(lightcyclesBoard a, lightcyclesBoard b) { return (lightcyclesBoard){ vbicq_u16(a.lo, b.lo), vbicq_u16(a.hi, b.hi) }; }
static inline __attribute__((always_inline)) uint32_t lightcycles_any(lightcyclesBoard a) { return vmaxvq_u32(vreinterpretq_u32_u16(vorrq_u16(a.lo, a.hi))) != 0u; }
static inline __attribute__((always_inline)) int32_t lightcycles_count(lightcyclesBoard a) { return (int32_t)vaddlvq_u8(vaddq_u8(vcntq_u8(vreinterpretq_u8_u16(a.lo)), vcntq_u8(vreinterpretq_u8_u16(a.hi)))); }

static inline __attribute__((always_inline)) lightcyclesBoard lightcycles_spread(lightcyclesBoard s) {
    uint8x16_t zero = vdupq_n_u8(0), lo = vreinterpretq_u8_u16(s.lo), hi = vreinterpretq_u8_u16(s.hi);
    uint16x8_t sideLo = vorrq_u16(vshlq_n_u16(s.lo, 1), vshrq_n_u16(s.lo, 1)), sideHi = vorrq_u16(vshlq_n_u16(s.hi, 1), vshrq_n_u16(s.hi, 1));
    uint16x8_t upLo = vreinterpretq_u16_u8(vorrq_u8(vextq_u8(zero, lo, 14), vextq_u8(lo, hi, 2)));
    uint16x8_t upHi = vreinterpretq_u16_u8(vorrq_u8(vextq_u8(lo, hi, 14), vextq_u8(hi, zero, 2)));
    return (lightcyclesBoard){ vorrq_u16(sideLo, upLo), vorrq_u16(sideHi, upHi) };
}

static inline __attribute__((always_inline)) lightcyclesBoard lightcycles_cell(uint32_t x, uint32_t y) {
    uint16x8_t row = vdupq_n_u16((uint16_t)y), bit = vdupq_n_u16((uint16_t)(1u << x));
    return (lightcyclesBoard){ vandq_u16(vceqq_u16(vld1q_u16(LIGHTCYCLES_LANES), row), bit), vandq_u16(vceqq_u16(vld1q_u16(LIGHTCYCLES_LANES + 8), row), bit) };
}

static inline __attribute__((always_inline)) lightcyclesBoard lightcycles_board(const lightcyclesKnobs *k, const uint16_t *rows) {
    uint16x8_t height = vdupq_n_u16((uint16_t)k->height), beyond = vdupq_n_u16((uint16_t)~((1u << k->width) - 1u));
    uint16x8_t lo = vorrq_u16(vorrq_u16(vld1q_u16(rows), beyond), vcgeq_u16(vld1q_u16(LIGHTCYCLES_LANES), height));
    uint16x8_t hi = vorrq_u16(vorrq_u16(vld1q_u16(rows + 8), beyond), vcgeq_u16(vld1q_u16(LIGHTCYCLES_LANES + 8), height));
    return (lightcyclesBoard){ lo, hi };
}

static inline __attribute__((always_inline)) uint32_t lightcycles_layer(lightcyclesBoard *claimed, lightcyclesBoard *fa, lightcyclesBoard *fb, lightcyclesBoard *ra, lightcyclesBoard *rb) {
    lightcyclesBoard na = lightcycles_without(lightcycles_spread(*fa), *claimed), nb = lightcycles_without(lightcycles_spread(*fb), *claimed), both = lightcycles_or(na, nb);
    *claimed = lightcycles_or(*claimed, both);
    *ra = lightcycles_or(*ra, na);
    *rb = lightcycles_or(*rb, nb);
    *fa = na;
    *fb = nb;
    return lightcycles_any(both);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_blocked(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t x, uint32_t y) {
    return x >= k->width || y >= k->height || (rows[y] >> x & 1u);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_ahead(uint32_t head, uint32_t action, uint32_t *x, uint32_t *y) {
    uint32_t heading = LIGHTCYCLES_TURN[head >> 8 & 3u][action];
    int dx, dy;
    gridDelta(heading, &dx, &dy);
    *x = (head & 15u) + (uint32_t)dx;
    *y = (head >> 4 & 15u) + (uint32_t)dy;
    return heading;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_open(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t head) {
    uint32_t open = 0u, x, y;
    for (uint32_t a = 0u; a < 3u; a++) {
        lightcycles_ahead(head, a, &x, &y);
        open |= (uint32_t)!lightcycles_blocked(k, rows, x, y) << a;
    }
    return open;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_room(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t head, uint32_t action) {
    uint32_t x, y;
    lightcycles_ahead(head, action, &x, &y);
    uint32_t row = ~(uint32_t)rows[y] & ((1u << k->width) - 1u), up = y ? ~(uint32_t)rows[y ? y - 1u : 0u] : 0u, down = y + 1u < k->height ? ~(uint32_t)rows[y + 1u < k->height ? y + 1u : y] : 0u;
    return ((row << 1) >> x & 1u) + (row >> x >> 1 & 1u) + (up >> x & 1u) + (down >> x & 1u);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_hugging(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t head, uint32_t set) {
    uint32_t fewest = 5u, tied = 0u;
    for (uint32_t a = 0u; a < 3u; a++) {
        if (!(set >> a & 1u)) continue;
        uint32_t room = lightcycles_room(k, rows, head, a);
        if (room < fewest) { fewest = room; tied = 0u; }
        if (room == fewest) tied |= 1u << a;
    }
    return tied;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_draw(uint32_t seed, const uint32_t *words, uint32_t set) {
    uint32_t n = (uint32_t)__builtin_popcount(set);
    if (n > 1u) {
        uint32_t w = philox4x32(seed, ENV_RNG_STREAM, words[LIGHTCYCLES_GAME], words[LIGHTCYCLES_TICK], 1u, 0u).x;
        for (uint32_t i = (uint32_t)(((uint64_t)w * n) >> 32); i > 0u; i--) set &= set - 1u;
    }
    return (uint32_t)__builtin_ctz(set);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_pick(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t seed, const uint32_t *words, uint32_t set) {
    return lightcycles_draw(seed, words, lightcycles_hugging(k, rows, words[LIGHTCYCLES_BOT], set));
}

static inline __attribute__((always_inline)) uint32_t lightcycles_joined(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t head, uint32_t open) {
    uint32_t heading = head >> 8 & 3u, x = head & 15u, y = head >> 4 & 15u;
    int fx, fy, lx, ly;
    gridDelta(heading, &fx, &fy);
    gridDelta(LIGHTCYCLES_TURN[heading][LIGHTCYCLES_LEFT], &lx, &ly);
    uint32_t cx = x + (uint32_t)fx, cy = y + (uint32_t)fy;
    uint32_t leftLink = !(open & 1u) || !lightcycles_blocked(k, rows, cx + (uint32_t)lx, cy + (uint32_t)ly);
    uint32_t rightLink = !(open & 4u) || !lightcycles_blocked(k, rows, cx - (uint32_t)lx, cy - (uint32_t)ly);
    return (open & 2u) && leftLink && rightLink;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_hoard(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t seed, const uint32_t *words, uint32_t open) {
    lightcyclesBoard wall = lightcycles_board(k, rows);
    int32_t area[3] = { 0, 0, 0 };
    for (uint32_t flood = 0u, rest = open; rest; flood++) {
        uint32_t first = (uint32_t)__builtin_ctz(rest), x, y;
        lightcycles_ahead(words[LIGHTCYCLES_BOT], first, &x, &y);
        lightcyclesBoard front = lightcycles_cell(x, y), claimed = lightcycles_or(wall, front), targets = { vdupq_n_u16(0), vdupq_n_u16(0) };
        for (uint32_t a = first + 1u; a < 3u; a++) {
            if (!(rest >> a & 1u)) continue;
            lightcycles_ahead(words[LIGHTCYCLES_BOT], a, &x, &y);
            targets = lightcycles_or(targets, lightcycles_cell(x, y));
        }
        do {
            front = lightcycles_without(lightcycles_spread(front), claimed);
            claimed = lightcycles_or(claimed, front);
            if (!flood && !lightcycles_any(lightcycles_without(targets, claimed))) return lightcycles_pick(k, rows, seed, words, open);
        } while (lightcycles_any(front));
        int32_t size = lightcycles_count(lightcycles_without(claimed, wall));
        for (uint32_t a = 0u; a < 3u; a++) {
            if (!(rest >> a & 1u)) continue;
            lightcycles_ahead(words[LIGHTCYCLES_BOT], a, &x, &y);
            if (lightcycles_any(lightcycles_without(lightcycles_cell(x, y), claimed))) continue;
            area[a] = size;
            rest &= ~(1u << a);
        }
    }
    int32_t top = -1;
    uint32_t set = 0u;
    for (uint32_t a = 0u; a < 3u; a++) {
        if (!(open >> a & 1u)) continue;
        if (area[a] > top) { top = area[a]; set = 0u; }
        if (area[a] == top) set |= 1u << a;
    }
    return lightcycles_pick(k, rows, seed, words, set);
}

static int32_t lightcycles_territory(lightcyclesBoard wall, uint32_t ux, uint32_t uy, uint32_t vx, uint32_t vy) {
    lightcyclesBoard fu = lightcycles_cell(ux, uy), fv = lightcycles_cell(vx, vy), claimed = lightcycles_or(wall, lightcycles_or(fu, fv));
    lightcyclesBoard ru = { vdupq_n_u16(0), vdupq_n_u16(0) }, rv = ru;
    while (lightcycles_layer(&claimed, &fu, &fv, &ru, &rv)) {}
    return lightcycles_count(ru) - lightcycles_count(rv);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_minimax(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t us, uint32_t them, uint32_t open) {
    lightcyclesBoard wall = lightcycles_board(k, rows);
    int32_t best = INT32_MIN;
    uint32_t set = 0u;
    for (uint32_t move = 0u; move < 3u; move++) {
        uint32_t ux, uy;
        if (!(open >> move & 1u)) continue;
        lightcycles_ahead(us, move, &ux, &uy);
        int32_t worst = INT32_MAX;
        for (uint32_t reply = 0u; reply < 3u && worst >= best; reply++) {
            uint32_t vx, vy;
            lightcycles_ahead(them, reply, &vx, &vy);
            int32_t value = lightcycles_blocked(k, rows, vx, vy) ? LIGHTCYCLES_WIN : vx == ux && vy == uy ? 0 : lightcycles_territory(wall, ux, uy, vx, vy);
            worst = value < worst ? value : worst;
        }
        if (worst > best) { best = worst; set = 0u; }
        if (worst == best) set |= 1u << move;
    }
    return set;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_bot(const lightcyclesKnobs *k, const uint16_t *rows, uint32_t seed, const uint32_t *words) {
    uint32_t bot = words[LIGHTCYCLES_BOT], open = lightcycles_open(k, rows, bot);
    if (!(open & (open - 1u))) return open ? (uint32_t)__builtin_ctz(open) : LIGHTCYCLES_STRAIGHT;
    if (k->bot == LIGHTCYCLES_RAMBLER) return lightcycles_draw(seed, words, open);
    if (k->bot == LIGHTCYCLES_HOARDER) return lightcycles_joined(k, rows, bot, open) ? lightcycles_pick(k, rows, seed, words, open) : lightcycles_hoard(k, rows, seed, words, open);
    return lightcycles_pick(k, rows, seed, words, lightcycles_minimax(k, rows, bot, words[LIGHTCYCLES_AGENT], open));
}

static inline __attribute__((always_inline)) void lightcycles_mark(lightcyclesLane *l, uint32_t *words, uint32_t side, uint32_t x, uint32_t y) {
    words[side + (y >> 1)] |= 1u << (x + ((y & 1u) << 4));
    l->rows[y] = (uint16_t)(l->rows[y] | 1u << x);
    l->cols[x] = (uint16_t)(l->cols[x] | 1u << y);
}

static void lightcycles_start(const lightcyclesKnobs *k, lightcyclesLane *l, uint32_t seed, uint32_t *words) {
    u4 draw = philox4x32(seed, ENV_RNG_STREAM, words[LIGHTCYCLES_GAME], 0u, 0u, 0u);
    uint32_t x = k->startLeft + (uint32_t)(((uint64_t)draw.x * (k->startRight - k->startLeft + 1u)) >> 32);
    uint32_t y = k->startTop + (uint32_t)(((uint64_t)draw.y * (k->startBottom - k->startTop + 1u)) >> 32);
    uint32_t heading = draw.z >> 30, bx = k->width - 1u - x, by = k->height - 1u - y;
    for (uint32_t i = 0u; i < LIGHTCYCLES_AGENT; i++) words[i] = 0u;
    for (uint32_t i = 0u; i < 16u; i++) l->rows[i] = l->cols[i] = 0u;
    lightcycles_mark(l, words, LIGHTCYCLES_MINE, x, y);
    lightcycles_mark(l, words, LIGHTCYCLES_THEIRS, bx, by);
    words[LIGHTCYCLES_AGENT] = x | y << 4 | heading << 8;
    words[LIGHTCYCLES_BOT] = bx | by << 4 | (heading ^ 1u) << 8;
    words[LIGHTCYCLES_TICK] = 0u;
}

static inline __attribute__((always_inline)) float lightcycles_tick(const lightcyclesKnobs *k, lightcyclesLane *l, uint32_t seed, uint32_t *words, uint32_t action, uint32_t move, uint32_t *done) {
    uint32_t ax, ay, bx, by;
    uint32_t ah = lightcycles_ahead(words[LIGHTCYCLES_AGENT], action, &ax, &ay), bh = lightcycles_ahead(words[LIGHTCYCLES_BOT], move, &bx, &by);
    uint32_t meet = ax == bx && ay == by;
    uint32_t agentCrash = meet | lightcycles_blocked(k, l->rows, ax, ay), botCrash = meet | lightcycles_blocked(k, l->rows, bx, by);
    if (!(agentCrash | botCrash)) {
        lightcycles_mark(l, words, LIGHTCYCLES_MINE, ax, ay);
        lightcycles_mark(l, words, LIGHTCYCLES_THEIRS, bx, by);
        words[LIGHTCYCLES_AGENT] = ax | ay << 4 | ah << 8;
        words[LIGHTCYCLES_BOT] = bx | by << 4 | bh << 8;
        words[LIGHTCYCLES_TICK]++;
        return 0.0f;
    }
    uint32_t result = agentCrash ? (botCrash ? 3u : 2u) : 1u;
    words[LIGHTCYCLES_LAST] = result | ((ax + 1u) & 31u) << 2 | ((ay + 1u) & 31u) << 7 | ((bx + 1u) & 31u) << 12 | ((by + 1u) & 31u) << 17;
    words[LIGHTCYCLES_GAME]++;
    lightcycles_start(k, l, seed, words);
    *done = 1u;
    return result == 1u ? 1.0f : result == 2u ? -1.0f : 0.0f;
}

static inline __attribute__((always_inline)) uint32_t lightcycles_rays(const lightcyclesKnobs *k, const lightcyclesLane *l, uint32_t head) {
    uint32_t x = head & 15u, y = head >> 4 & 15u, row = l->rows[y], col = l->cols[x];
    uint32_t north = (uint32_t)__builtin_clz(((col << 1) | 1u) << (31u - y)), south = (uint32_t)__builtin_ctz((col | 1u << k->height) >> (y + 1u));
    uint32_t west = (uint32_t)__builtin_clz(((row << 1) | 1u) << (31u - x)), east = (uint32_t)__builtin_ctz((row | 1u << k->width) >> (x + 1u));
    return north | south << 8 | west << 16 | east << 24;
}

static inline __attribute__((always_inline)) float lightcycles_ray(uint32_t rays, uint32_t heading, float scale) { return (float)(rays >> (heading << 3) & 255u) * scale; }

static inline __attribute__((always_inline)) void lightcycles_sense(const lightcyclesKnobs *k, const uint32_t *words, lightcyclesLane *l) {
    float *o = l->senses, scale = 1.0f / (float)((k->width > k->height ? k->width : k->height) - 1u);
    uint32_t agent = words[LIGHTCYCLES_AGENT], bot = words[LIGHTCYCLES_BOT], ah = agent >> 8 & 3u, bh = bot >> 8 & 3u;
    uint32_t mine = lightcycles_rays(k, l, agent), theirs = lightcycles_rays(k, l, bot);
    int hx, hy, lx, ly, gx, gy;
    gridDelta(ah, &hx, &hy);
    gridDelta(LIGHTCYCLES_TURN[ah][LIGHTCYCLES_LEFT], &lx, &ly);
    gridDelta(bh, &gx, &gy);
    int ax = (int)(agent & 15u), ay = (int)(agent >> 4 & 15u), dx = (int)(bot & 15u) - ax, dy = (int)(bot >> 4 & 15u) - ay;
    o[0] = lightcycles_ray(mine, ah, scale);
    o[1] = lightcycles_ray(mine, LIGHTCYCLES_TURN[ah][LIGHTCYCLES_LEFT], scale);
    o[2] = lightcycles_ray(mine, LIGHTCYCLES_TURN[ah][LIGHTCYCLES_RIGHT], scale);
    o[3] = (float)lightcycles_blocked(k, l->rows, (uint32_t)(ax + hx + lx), (uint32_t)(ay + hy + ly));
    o[4] = (float)lightcycles_blocked(k, l->rows, (uint32_t)(ax + hx - lx), (uint32_t)(ay + hy - ly));
    o[5] = (float)ax * scale;
    o[6] = (float)ay * scale;
    o[7] = (float)hx;
    o[8] = (float)hy;
    o[9] = (float)(dx * hx + dy * hy) * scale;
    o[10] = (float)(dy * hx - dx * hy) * scale;
    o[11] = (float)(gx * hx + gy * hy);
    o[12] = (float)(gy * hx - gx * hy);
    o[13] = lightcycles_ray(theirs, bh, scale);
    o[14] = lightcycles_ray(theirs, LIGHTCYCLES_TURN[bh][LIGHTCYCLES_LEFT], scale);
    o[15] = lightcycles_ray(theirs, LIGHTCYCLES_TURN[bh][LIGHTCYCLES_RIGHT], scale);
}

static inline __attribute__((always_inline)) void lightcycles_emit(const lightcyclesLane *l, float *obs) { __builtin_memcpy(obs, l->senses, sizeof l->senses); }

static inline __attribute__((always_inline)) uint32_t lightcycles_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 3.0f ? (uint32_t)c : LIGHTCYCLES_STRAIGHT;
}

static inline __attribute__((always_inline)) float lightcycles_play(const lightcyclesKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    lightcyclesLane *l = lightcycles_lane(extra);
    *done = 0u;
    float reward = lightcycles_tick(k, l, seed, words, lightcycles_action(ctrl), lightcycles_bot(k, l->rows, seed, words), done);
    lightcycles_sense(k, words, l);
    lightcycles_emit(l, obs);
    return reward;
}

static void lightcycles_setup(const lightcyclesKnobs *k, uint32_t seed, uint32_t *words, uint32_t *extra) {
    lightcyclesLane *l = lightcycles_lane(extra);
    for (uint32_t i = 0u; i < LIGHTCYCLES_WORDS; i++) words[i] = 0u;
    lightcycles_start(k, l, seed, words);
    lightcycles_sense(k, words, l);
}

void lightcycles_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    switch (instance) {
        case 0u: lightcycles_setup(&LIGHTCYCLES_KNOBS[0], seed, words, extra); return;
        case 1u: lightcycles_setup(&LIGHTCYCLES_KNOBS[1], seed, words, extra); return;
        case 2u: lightcycles_setup(&LIGHTCYCLES_KNOBS[2], seed, words, extra); return;
        default: __builtin_trap();
    }
}

float lightcycles_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return lightcycles_play(&LIGHTCYCLES_KNOBS[0], seed, ctrl, words, extra, obs, done);
        case 1u: return lightcycles_play(&LIGHTCYCLES_KNOBS[1], seed, ctrl, words, extra, obs, done);
        case 2u: return lightcycles_play(&LIGHTCYCLES_KNOBS[2], seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void lightcycles_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)words; (void)step; (void)shared;
    lightcycles_emit(lightcycles_peek(extra), obs);
}

static inline __attribute__((always_inline)) uint32_t lightcycles_counsel(const lightcyclesKnobs *k, const uint16_t *rows, const uint32_t *words) {
    uint32_t agent = words[LIGHTCYCLES_AGENT], open = lightcycles_open(k, rows, agent);
    if (!(open & (open - 1u))) return open ? (uint32_t)__builtin_ctz(open) : LIGHTCYCLES_STRAIGHT;
    return (uint32_t)__builtin_ctz(lightcycles_hugging(k, rows, agent, lightcycles_minimax(k, rows, agent, words[LIGHTCYCLES_BOT], open)));
}

void lightcycles_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const uint16_t *rows = lightcycles_peek(extra)->rows;
    switch (instance) {
        case 0u: ctrl[0] = (float)lightcycles_counsel(&LIGHTCYCLES_KNOBS[0], rows, words); return;
        case 1u: ctrl[0] = (float)lightcycles_counsel(&LIGHTCYCLES_KNOBS[1], rows, words); return;
        case 2u: ctrl[0] = (float)lightcycles_counsel(&LIGHTCYCLES_KNOBS[2], rows, words); return;
        default: __builtin_trap();
    }
}
