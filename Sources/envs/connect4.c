#include "../env.h"

typedef struct { uint32_t opponent, nodesPerMove, jitter, oracleNodes; } connect4Knobs;

static const connect4Knobs CONNECT4_KNOBS[4] = {
    { 0u, 0u, 0u, 64u },
    { 1u, 0u, 0u, 64u },
    { 2u, 512u, 2u, 8192u },
    { 3u, 512u, 2u, 8192u }
};

static const float CONNECT4_THETA[16] = { 4.0f, 14.0f, 4.0f, 60.0f, 10.0f, 6.0f, 2.0f, 0.0f, -4.0f, -14.0f, -4.0f, 0.0f, -8.0f, -7.0f, -2.0f, -1.0f };
static const float CONNECT4_THETA_LOW[16] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -4.0f, -8.0f, -24.0f, -16.0f, -120.0f, -40.0f, -12.0f, -6.0f, -4.0f };
static const float CONNECT4_THETA_HIGH[16] = { 8.0f, 24.0f, 16.0f, 120.0f, 40.0f, 12.0f, 6.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f };

enum { CONNECT4_RANDOM, CONNECT4_GREEDY, CONNECT4_SEARCH, CONNECT4_LEAGUE };
enum { CONNECT4_PLAY, CONNECT4_NEXT };
enum { CONNECT4_MINE = 0, CONNECT4_THEIRS = 2, CONNECT4_PROPOSALS = 4, CONNECT4_SHOWN, CONNECT4_GAME, CONNECT4_MEMBER, CONNECT4_LAST, CONNECT4_WORDS };
enum { CONNECT4_COLUMNS = 7, CONNECT4_ROWS = 6, CONNECT4_CELLS = 42, CONNECT4_FEATURES = 16 };
enum { CONNECT4_PICK, CONNECT4_PONDER, CONNECT4_SHUFFLE };
enum { CONNECT4_WIN = 1 << 20, CONNECT4_DECIDED = CONNECT4_WIN - 64, CONNECT4_INFINITY = 1 << 22 };

static const uint64_t CONNECT4_BOTTOM = 0x40810204081ull;
static const uint64_t CONNECT4_BOARD = 0xFDFBF7EFDFBFull;
static const uint64_t CONNECT4_ODD = 0x54A952A54A95ull;
static const uint64_t CONNECT4_EVEN = 0xA952A54A952Aull;
static const uint64_t CONNECT4_VERTICAL = 0x1C3870E1C387ull;
static const uint64_t CONNECT4_HORIZONTAL = 0x7EFDFBFull;
static const uint64_t CONNECT4_RISING = 0xE1C387ull;
static const uint64_t CONNECT4_FALLING = 0x70E1C38ull;
static const uint64_t CONNECT4_CENTRE = 0x7E00000ull;
static const uint64_t CONNECT4_INNER = 0x3F00FC000ull;
static const uint64_t CONNECT4_OUTER = 0x1F800001F80ull;
static const uint32_t CONNECT4_ORDER = 0x6051423u;

typedef uint64_t connect4Pair __attribute__((ext_vector_type(2)));

typedef struct {
    uint64_t own, all;
    int32_t alpha, beta, best;
    uint8_t next, allowed;
} connect4Frame;

typedef struct {
    int8x16_t weight[8];
    connect4Pair good;
    connect4Frame frame[CONNECT4_CELLS + 1];
    uint32_t sp, depth, horizon, ply, nodes, budget, done, moves, current, leader, choice, guard;
    uint8_t order[8], rank[8], jitter[8];
} connect4Search;

typedef struct {
    connect4Search search;
    float winRate[LEAGUE_MEMBERS], theta[LEAGUE_MEMBERS * CONNECT4_FEATURES];
} connect4Lane;

typedef struct {
    connect4Search search;
    uint32_t key, column;
} connect4Oracle;

static connect4Lane *connect4_lane(uint32_t *extra) { return (connect4Lane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static connect4Oracle *connect4_oracleOf(connect4Lane *l) { return (connect4Oracle *)(l + 1); }

uint32_t connect4_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)((sizeof(connect4Lane) + sizeof(connect4Oracle)) / sizeof(uint32_t)) + 4u;
}

static uint64_t connect4_board(const uint32_t *words, uint32_t at) { return (uint64_t)words[at] | (uint64_t)words[at + 1u] << 32; }

static void connect4_put(uint32_t *words, uint32_t at, uint64_t board) {
    words[at] = (uint32_t)board;
    words[at + 1u] = (uint32_t)(board >> 32);
}

static uint64_t connect4_drop(uint64_t all, uint32_t column) { return (all + (1ull << 7u * column)) & (0x3Full << 7u * column); }

static uint32_t connect4_column(uint64_t cells) { return (uint32_t)__builtin_ctzll(cells) / 7u; }

static uint32_t connect4_columns(uint64_t cells) {
    uint32_t columns = 0u;
    for (uint32_t c = 0u; c < CONNECT4_COLUMNS; c++) columns |= (uint32_t)((cells >> 7u * c & 0x3Fu) != 0u) << c;
    return columns;
}

static inline uint64_t connect4_fours(uint64_t p) {
    uint64_t h = p & p >> 7, v = p & p >> 1, d = p & p >> 8, a = p & p >> 6;
    return (h & h >> 14) | (v & v >> 2) | (d & d >> 16) | (a & a >> 12);
}

static inline __attribute__((always_inline)) connect4Pair connect4_line(connect4Pair s, uint32_t d) {
    connect4Pair up = s << d, down = s >> d;
    return (up & s << 2u * d & (s << 3u * d | down)) | (down & s >> 2u * d & (up | s >> 3u * d));
}

static inline __attribute__((always_inline)) connect4Pair connect4_threats(connect4Pair s, uint64_t all) {
    connect4Pair r = (s << 1) & (s << 2) & (s << 3);
    r |= connect4_line(s, 7u) | connect4_line(s, 6u) | connect4_line(s, 8u);
    return r & (CONNECT4_BOARD ^ all);
}

static inline __attribute__((always_inline)) uint8x16_t connect4_pairs(connect4Pair s, uint32_t d, uint64_t valid) {
    connect4Pair s1 = s ^ s >> d, c1 = s & s >> d;
    connect4Pair low = s1 ^ s1 >> 2u * d, mid = c1 ^ c1 >> 2u * d ^ (s1 & s1 >> 2u * d);
    return vcntq_u8((uint8x16_t)(mid & ~low & ~(low | mid).yx & valid));
}

static inline __attribute__((always_inline)) int8x16_t connect4_count(connect4Pair cells) { return (int8x16_t)vcntq_u8((uint8x16_t)cells); }

static inline __attribute__((always_inline)) int32_t connect4_eval(uint64_t x, uint64_t y, connect4Pair good, const int8x16_t *w) {
    connect4Pair s = { x, y };
    uint64_t all = x | y;
    connect4Pair t = connect4_threats(s, all), open = (all + CONNECT4_BOTTOM) & CONNECT4_BOARD;
    uint8x16_t pairs = connect4_pairs(s, 1u, CONNECT4_VERTICAL) + connect4_pairs(s, 7u, CONNECT4_HORIZONTAL) + connect4_pairs(s, 8u, CONNECT4_RISING) + connect4_pairs(s, 6u, CONNECT4_FALLING);
    int32x4_t a = vdotq_s32(vdupq_n_s32(0), (int8x16_t)pairs, w[0]);
    int32x4_t b = vdotq_s32(vdupq_n_s32(0), connect4_count(t & good), w[1]);
    a = vdotq_s32(a, connect4_count(t & ~good), w[2]);
    b = vdotq_s32(b, connect4_count(t & open), w[3]);
    a = vdotq_s32(a, connect4_count(t & t >> 1), w[4]);
    b = vdotq_s32(b, connect4_count(s & CONNECT4_CENTRE), w[5]);
    a = vdotq_s32(a, connect4_count(s & CONNECT4_INNER), w[6]);
    b = vdotq_s32(b, connect4_count(s & CONNECT4_OUTER), w[7]);
    return vaddvq_s32(a + b);
}

static void connect4_weigh(int8x16_t *w, const float *theta) {
    for (uint32_t f = 0u; f < CONNECT4_FEATURES / 2u; f++)
        w[f] = vcombine_s8(vdup_n_s8((int8_t)lrintf(theta[f])), vdup_n_s8((int8_t)lrintf(theta[CONNECT4_FEATURES / 2u + f])));
}

static void connect4_iterate(connect4Search *s) {
    connect4Frame *root = &s->frame[0];
    root->alpha = root->best = -CONNECT4_INFINITY;
    root->next = 0u;
    s->sp = 0u;
    s->good = (s->ply + s->depth) & 1u ? (connect4Pair){ CONNECT4_EVEN, CONNECT4_ODD } : (connect4Pair){ CONNECT4_ODD, CONNECT4_EVEN };
}

static uint32_t connect4_safe(uint64_t own, uint64_t all) {
    uint64_t open = (all + CONNECT4_BOTTOM) & CONNECT4_BOARD;
    connect4Pair t = connect4_threats((connect4Pair){ own, all ^ own }, all);
    uint64_t win = t.x & open, forced = t.y & open, safe = open & ~(t.y >> 1);
    return connect4_columns(win ? win : forced ? forced : safe ? safe : open);
}

static void connect4_begin(connect4Search *s, uint64_t own, uint64_t all, uint32_t allowed, uint32_t budget, u4 draw, uint32_t jitter) {
    s->moves = 0u;
    for (uint32_t i = 0u; i < CONNECT4_COLUMNS; i++) {
        uint32_t column = CONNECT4_ORDER >> 4u * i & 15u;
        if (allowed >> column & 1u) s->order[s->moves++] = (uint8_t)column;
    }
    for (uint32_t c = 0u; c < 4u; c++) {
        s->rank[c] = (uint8_t)(draw.x >> 8u * c);
        s->rank[4u + c] = (uint8_t)(draw.y >> 8u * c);
    }
    for (uint32_t c = 0u; c < 8u; c++) s->jitter[c] = (uint8_t)(((draw.z >> 4u * c & 15u) * (jitter + 1u)) >> 4);
    s->ply = (uint32_t)__builtin_popcountll(all);
    s->horizon = CONNECT4_CELLS - s->ply;
    s->depth = 1u;
    s->nodes = s->done = 0u;
    s->budget = budget;
    s->choice = s->leader = s->order[0];
    s->frame[0].own = own;
    s->frame[0].all = all;
    s->frame[0].beta = CONNECT4_INFINITY;
    s->frame[0].allowed = 0x7Fu;
    connect4_iterate(s);
}

static inline __attribute__((always_inline)) void connect4_score(connect4Search *s, connect4Frame *f, int32_t value, uint32_t column) {
    if (s->sp) {
        f->best = value > f->best ? value : f->best;
        f->alpha = value > f->alpha ? value : f->alpha;
        return;
    }
    int32_t noisy = value + (value < CONNECT4_DECIDED && value > -CONNECT4_DECIDED ? s->jitter[column] : 0);
    if (noisy > f->alpha || (noisy == f->alpha && s->rank[column] > s->rank[s->leader])) {
        f->alpha = noisy;
        f->best = value;
        s->leader = column;
    }
}

static void connect4_deepen(connect4Search *s) {
    uint32_t i = 0u;
    while (s->order[i] != s->leader) i++;
    for (; i > 0u; i--) s->order[i] = s->order[i - 1u];
    s->order[0] = (uint8_t)s->leader;
    s->depth++;
    connect4_iterate(s);
}

static void connect4_think(connect4Search *s) {
    while (!s->done) {
        connect4Frame *f = &s->frame[s->sp];
        if (f->next >= (s->sp ? (uint32_t)CONNECT4_COLUMNS : s->moves) || f->alpha >= f->beta) {
            if (s->sp) {
                int32_t value = -f->best;
                s->sp--;
                connect4_score(s, f - 1, value, s->current);
                continue;
            }
            s->choice = s->leader;
            if (s->nodes >= s->budget || s->moves == 1u || s->depth >= s->horizon || f->best >= CONNECT4_DECIDED || f->best <= -CONNECT4_DECIDED) s->done = 1u;
            else connect4_deepen(s);
            continue;
        }
        if (s->nodes >= s->budget) {
            if (s->frame[0].best > -CONNECT4_INFINITY) s->choice = s->leader;
            s->done = 1u;
            continue;
        }
        uint32_t column = s->sp ? CONNECT4_ORDER >> 4u * f->next & 15u : s->order[f->next];
        f->next++;
        uint64_t bit = connect4_drop(f->all, column);
        if (!bit || !(f->allowed >> column & 1u)) continue;
        s->nodes++;
        uint64_t mover = f->own | bit, all = f->all | bit;
        int32_t value;
        if (connect4_fours(mover)) value = CONNECT4_WIN - (int32_t)s->sp - 1;
        else if (all == CONNECT4_BOARD) value = 0;
        else if (s->sp + 1u < s->depth) {
            connect4Frame *c = f + 1;
            c->own = f->all ^ f->own;
            c->all = all;
            c->alpha = -f->beta;
            c->beta = s->sp ? -f->alpha : 1 + s->jitter[column] - f->alpha;
            c->best = -CONNECT4_INFINITY;
            c->next = 0u;
            c->allowed = (uint8_t)(s->guard ? connect4_safe(c->own, all) : 0x7Fu);
            if (!s->sp) s->current = column;
            s->sp++;
            continue;
        } else value = -connect4_eval(f->all ^ f->own, mover, s->good, s->weight);
        connect4_score(s, f, value, column);
    }
}

static uint32_t connect4_proposed(const uint32_t *words) { return words[CONNECT4_PROPOSALS] >> 4u * words[CONNECT4_SHOWN] & 15u; }

static uint32_t connect4_offered(const uint32_t *words) { return words[CONNECT4_PROPOSALS] >> 28; }

static void connect4_obs(const uint32_t *words, float *obs) {
    uint64_t theirs = connect4_board(words, CONNECT4_THEIRS), all = connect4_board(words, CONNECT4_MINE) | theirs;
    uint64_t mine = (all ^ theirs) | connect4_drop(all, connect4_proposed(words));
    for (uint32_t row = 0u; row < CONNECT4_ROWS; row++)
        for (uint32_t column = 0u; column < CONNECT4_COLUMNS; column++) {
            uint32_t bit = 7u * column + row;
            obs[CONNECT4_COLUMNS * row + column] = (float)(mine >> bit & 1u) - (float)(theirs >> bit & 1u);
        }
}

static void connect4_offer(uint32_t *words, uint32_t seed) {
    uint64_t all = connect4_board(words, CONNECT4_MINE) | connect4_board(words, CONNECT4_THEIRS);
    uint64_t draw = philox4x32(seed, ENV_RNG_STREAM, words[CONNECT4_GAME], (uint32_t)__builtin_popcountll(all), CONNECT4_SHUFFLE, 0u).x;
    uint32_t column[CONNECT4_COLUMNS], n = 0u;
    for (uint64_t open = (all + CONNECT4_BOTTOM) & CONNECT4_BOARD; open; open &= open - 1u) column[n++] = connect4_column(open);
    for (uint32_t i = n - 1u; i > 0u; i--) {
        draw *= i + 1u;
        uint32_t j = (uint32_t)(draw >> 32), held = column[i];
        draw &= 0xFFFFFFFFull;
        column[i] = column[j];
        column[j] = held;
    }
    uint32_t packed = n << 28;
    for (uint32_t i = 0u; i < n; i++) packed |= column[i] << 4u * i;
    words[CONNECT4_PROPOSALS] = packed;
    words[CONNECT4_SHOWN] = 0u;
}

static uint32_t connect4_place(uint32_t *words, uint32_t who, uint32_t column) {
    uint32_t at = who ? CONNECT4_THEIRS : CONNECT4_MINE;
    uint64_t all = connect4_board(words, CONNECT4_MINE) | connect4_board(words, CONNECT4_THEIRS);
    uint64_t bit = connect4_drop(all, column), side = connect4_board(words, at) | bit;
    connect4_put(words, at, side);
    return connect4_fours(side) ? 1u + who : (all | bit) == CONNECT4_BOARD ? 3u : 0u;
}

static uint32_t connect4_bot(const connect4Knobs *k, uint32_t seed, const uint32_t *words) {
    uint64_t mine = connect4_board(words, CONNECT4_MINE), theirs = connect4_board(words, CONNECT4_THEIRS), all = mine | theirs;
    uint64_t open = (all + CONNECT4_BOTTOM) & CONNECT4_BOARD, pick = open;
    if (k->opponent == CONNECT4_GREEDY) {
        connect4Pair t = connect4_threats((connect4Pair){ theirs, mine }, all) & open;
        pick = t.x ? t.x : t.y ? t.y : open;
    }
    uint32_t draw = philox4x32(seed, ENV_RNG_STREAM, words[CONNECT4_GAME], (uint32_t)__builtin_popcountll(all), CONNECT4_PICK, 0u).x;
    for (uint32_t i = (uint32_t)(((uint64_t)draw * (uint64_t)__builtin_popcountll(pick)) >> 32); i > 0u; i--) pick &= pick - 1u;
    return connect4_column(pick);
}

static uint32_t connect4_reply(connect4Lane *l, const connect4Knobs *k, uint32_t seed, const uint32_t *words) {
    if (k->opponent < CONNECT4_SEARCH) return connect4_bot(k, seed, words);
    uint64_t theirs = connect4_board(words, CONNECT4_THEIRS), all = connect4_board(words, CONNECT4_MINE) | theirs;
    u4 draw = philox4x32(seed, ENV_RNG_STREAM, words[CONNECT4_GAME], (uint32_t)__builtin_popcountll(all), CONNECT4_PONDER, 0u);
    connect4_begin(&l->search, theirs, all, connect4_columns((all + CONNECT4_BOTTOM) & CONNECT4_BOARD), k->nodesPerMove, draw, k->jitter);
    connect4_think(&l->search);
    return l->search.choice;
}

static void connect4_start(connect4Lane *l, uint32_t *words, const connect4Knobs *k, uint32_t seed) {
    connect4_put(words, CONNECT4_MINE, 0u);
    connect4_put(words, CONNECT4_THEIRS, 0u);
    if (k->opponent == CONNECT4_LEAGUE) {
        words[CONNECT4_MEMBER] = leaguePick(l->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, words[CONNECT4_GAME], 0x5049434Bu, 0u, 0u).x);
        connect4_weigh(l->search.weight, l->theta + CONNECT4_FEATURES * words[CONNECT4_MEMBER]);
    }
    if (words[CONNECT4_GAME] & 1u) connect4_place(words, 1u, connect4_reply(l, k, seed, words));
    connect4_offer(words, seed);
}

static void connect4_league(connect4Lane *l, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->winRate[i] = 0.5f;
    for (uint32_t d = 0u; d < CONNECT4_FEATURES; d++) l->theta[d] = CONNECT4_THETA[d];
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < CONNECT4_FEATURES; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = CONNECT4_THETA[d + j] + 0.1f * (CONNECT4_THETA_HIGH[d + j] - CONNECT4_THETA_LOW[d + j]) * n[j];
                l->theta[CONNECT4_FEATURES * i + d + j] = fminf(fmaxf(value, CONNECT4_THETA_LOW[d + j]), CONNECT4_THETA_HIGH[d + j]);
            }
        }
}

static __attribute__((noinline)) void connect4_finish(connect4Lane *l, uint32_t *words, const connect4Knobs *k, uint32_t seed, float reward) {
    if (k->opponent == CONNECT4_LEAGUE) {
        leagueScore(l->winRate, words[CONNECT4_MEMBER], reward > 0.0f ? 1.0f : reward < 0.0f ? 0.0f : 0.5f);
        leagueRenew(l->winRate, l->theta, CONNECT4_FEATURES, CONNECT4_THETA_LOW, CONNECT4_THETA_HIGH, seed, words[CONNECT4_GAME]);
    }
    words[CONNECT4_GAME]++;
    connect4_start(l, words, k, seed);
}

static float connect4_turn(connect4Lane *l, uint32_t *words, const connect4Knobs *k, uint32_t seed, uint32_t *done) {
    uint32_t column = connect4_proposed(words), result = connect4_place(words, 0u, column), last = column + 1u;
    if (!result) {
        uint32_t reply = connect4_reply(l, k, seed, words);
        result = connect4_place(words, 1u, reply);
        last |= (reply + 1u) << 3;
    }
    words[CONNECT4_LAST] = last | result << 6;
    if (!result) {
        connect4_offer(words, seed);
        return 0.0f;
    }
    float reward = result == 1u ? 1.0f : result == 2u ? -1.0f : 0.0f;
    *done = 1u;
    connect4_finish(l, words, k, seed, reward);
    return reward;
}

static uint32_t connect4_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 2.0f ? (uint32_t)c : CONNECT4_NEXT;
}

void connect4_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const connect4Knobs *k = &CONNECT4_KNOBS[instance];
    connect4Lane *l = connect4_lane(extra);
    connect4Oracle *o = connect4_oracleOf(l);
    for (uint32_t i = 0u; i < CONNECT4_WORDS; i++) words[i] = 0u;
    connect4_weigh(l->search.weight, CONNECT4_THETA);
    connect4_weigh(o->search.weight, CONNECT4_THETA);
    l->search.guard = 0u;
    o->search.guard = 1u;
    o->key = 0xFFFFFFFFu;
    if (k->opponent == CONNECT4_LEAGUE) connect4_league(l, seed);
    connect4_start(l, words, k, seed);
}

float connect4_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    uint32_t shown = words[CONNECT4_SHOWN];
    float reward = 0.0f;
    *done = 0u;
    if (connect4_action(ctrl) == CONNECT4_NEXT && shown + 1u < connect4_offered(words)) words[CONNECT4_SHOWN] = shown + 1u;
    else reward = connect4_turn(connect4_lane(extra), words, &CONNECT4_KNOBS[instance], seed, done);
    connect4_obs(words, obs);
    return reward;
}

void connect4_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    connect4_obs(words, obs);
}

static uint32_t connect4_decide(connect4Oracle *o, const connect4Knobs *k, uint64_t mine, uint64_t theirs) {
    uint64_t all = mine | theirs, open = (all + CONNECT4_BOTTOM) & CONNECT4_BOARD;
    connect4Pair t = connect4_threats((connect4Pair){ mine, theirs }, all);
    uint64_t win = t.x & open, loss = t.y & open, safe = open & ~(t.y >> 1);
    if (win) return connect4_column(win);
    if (loss) return connect4_column(loss);
    connect4_begin(&o->search, mine, all, connect4_columns(safe ? safe : open), k->oracleNodes, (u4){ 0u, 0u, 0u, 0u }, 0u);
    connect4_think(&o->search);
    return o->search.choice;
}

void connect4_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    connect4Oracle *o = connect4_oracleOf(connect4_lane(extra));
    uint64_t mine = connect4_board(words, CONNECT4_MINE), theirs = connect4_board(words, CONNECT4_THEIRS);
    uint32_t key = words[CONNECT4_GAME] << 6 | (uint32_t)__builtin_popcountll(mine | theirs);
    if (o->key != key) {
        o->column = connect4_decide(o, &CONNECT4_KNOBS[instance], mine, theirs);
        o->key = key;
    }
    ctrl[0] = (float)(connect4_proposed(words) == o->column ? CONNECT4_PLAY : CONNECT4_NEXT);
}
