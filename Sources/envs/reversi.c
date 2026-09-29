#include "../env.h"

typedef struct { uint32_t size, opponent, oracleExact, oracleNodes; } reversiKnobs;

static const reversiKnobs REVERSI_KNOBS[4] = {
    { 6u, 0u, 12u, 4096u },
    { 8u, 1u, 12u, 4096u },
    { 8u, 2u, 14u, 16384u },
    { 8u, 3u, 12u, 4096u }
};

static const float REVERSI_SWH[10] = { 1.0f, -0.25f, 0.1f, 0.05f, -0.25f, 0.01f, 0.01f, 0.05f, 0.02f, 0.01f };
static const float REVERSI_THETA_LOW[10] = { 0.75f, -0.5f, -0.15f, -0.2f, -0.5f, -0.24f, -0.24f, -0.2f, -0.23f, -0.24f };
static const float REVERSI_THETA_HIGH[10] = { 1.25f, 0.0f, 0.35f, 0.3f, 0.0f, 0.26f, 0.26f, 0.3f, 0.27f, 0.26f };

enum { REVERSI_GREEDY, REVERSI_WPC, REVERSI_ALPHABETA, REVERSI_LEAGUE };
enum { REVERSI_PLAY, REVERSI_NEXT, REVERSI_ACTIONS };
enum { REVERSI_OWN = 0, REVERSI_OPP = 2, REVERSI_LEGAL = 4, REVERSI_PROPOSAL = 6, REVERSI_SHOWN, REVERSI_GAME, REVERSI_LAST, REVERSI_MEMBER, REVERSI_WORDS };
enum { REVERSI_CLASSES = 10, REVERSI_PASS = 65, REVERSI_FINAL = 1024, REVERSI_INFINITY = 1 << 30, REVERSI_PICK = 0x5049434B, REVERSI_SHUFFLE = 0x53485546 };

static const uint64_t REVERSI_INNER = 0x7E7E7E7E7E7E7E7Eull;

typedef struct {
    int8x16_t weight[4];
    float winRate[LEAGUE_MEMBERS], theta[LEAGUE_MEMBERS * REVERSI_CLASSES];
    uint8_t order[64];
} reversiLane;

typedef struct { uint32_t key, square, nodes, budget; } reversiOracle;

static reversiLane *reversi_lane(uint32_t *extra) { return (reversiLane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static reversiOracle *reversi_oracleOf(reversiLane *l) { return (reversiOracle *)(l + 1); }

uint32_t reversi_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)((sizeof(reversiLane) + sizeof(reversiOracle)) / sizeof(uint32_t)) + 4u;
}

static uint64_t reversi_get(const uint32_t *words, uint32_t at) { return (uint64_t)words[at] | (uint64_t)words[at + 1u] << 32; }

static void reversi_put(uint32_t *words, uint32_t at, uint64_t board) {
    words[at] = (uint32_t)board;
    words[at + 1u] = (uint32_t)(board >> 32);
}

static inline uint64_t reversi_board(uint32_t size) { return size == 8u ? ~0ull : 0x3F3F3F3F3F3Full; }

static inline uint32_t reversi_count(uint64_t b) { return (uint32_t)__builtin_popcountll(b); }

static inline __attribute__((always_inline)) uint64_t reversi_line(uint64_t me, uint64_t them, uint64_t empty, uint32_t d) {
    uint64_t pl = them & (them << d), pr = pl >> d;
    uint64_t l = them & (me << d), r = them & (me >> d);
    l |= them & (l << d);
    r |= them & (r >> d);
    l |= pl & (l << 2u * d);
    r |= pr & (r >> 2u * d);
    l |= pl & (l << 2u * d);
    r |= pr & (r >> 2u * d);
    return empty & ((l << d) | (r >> d));
}

static inline __attribute__((always_inline)) uint64_t reversi_moves(uint64_t me, uint64_t them, uint64_t empty) {
    uint64_t inner = them & REVERSI_INNER;
    return reversi_line(me, inner, empty, 1u) | reversi_line(me, them, empty, 8u) | reversi_line(me, inner, empty, 7u) | reversi_line(me, inner, empty, 9u);
}

static inline __attribute__((always_inline)) uint64_t reversi_rise(uint64_t ray, uint64_t me, uint64_t them) {
    uint64_t stop = ray & ~them, first = stop & (0u - stop);
    return first & me ? ray & (first - 1u) : 0u;
}

static inline __attribute__((always_inline)) uint64_t reversi_fall(uint64_t ray, uint64_t me, uint64_t them) {
    uint64_t stop = ray & ~them, first = (0x8000000000000000ull >> __builtin_clzll(stop | 1u)) & stop;
    return first & me ? ray & (0u - (first << 1)) : 0u;
}

static inline __attribute__((always_inline)) uint64_t reversi_flips(uint64_t me, uint64_t them, uint32_t square) {
    uint32_t x = square & 7u, y = square >> 3;
    int32_t across = (int32_t)x - (int32_t)y, down = (int32_t)(x + y) - 7;
    uint64_t above = ~1ull << square, below = (1ull << square) - 1u;
    uint64_t row = 0xFFull << 8u * y, file = 0x0101010101010101ull << x;
    uint64_t diagonal = across >= 0 ? 0x8040201008040201ull >> 8u * (uint32_t)across : 0x8040201008040201ull << 8u * (uint32_t)-across;
    uint64_t anti = down >= 0 ? 0x0102040810204080ull << 8u * (uint32_t)down : 0x0102040810204080ull >> 8u * (uint32_t)-down;
    return reversi_rise(row & above, me, them) | reversi_rise(file & above, me, them) | reversi_rise(diagonal & above, me, them) | reversi_rise(anti & above, me, them) |
           reversi_fall(row & below, me, them) | reversi_fall(file & below, me, them) | reversi_fall(diagonal & below, me, them) | reversi_fall(anti & below, me, them);
}

static inline __attribute__((always_inline)) int32_t reversi_wpc(uint64_t me, uint64_t them, const int8x16_t *w) {
    const uint8x16_t bits = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u };
    const uint8x16_t pick = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
    uint8x16_t table = vcombine_u8(vcreate_u8(me), vcreate_u8(them));
    int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
    for (uint32_t k = 0u; k < 4u; k++) {
        uint8x16_t at = vaddq_u8(pick, vdupq_n_u8((uint8_t)(2u * k)));
        int8x16_t mine = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(table, at), bits));
        int8x16_t theirs = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(table, vaddq_u8(at, vdupq_n_u8(8u))), bits));
        if (k & 1u) b = vdotq_s32(b, vsubq_s8(theirs, mine), w[k]);
        else a = vdotq_s32(a, vsubq_s8(theirs, mine), w[k]);
    }
    return vaddvq_s32(vaddq_s32(a, b));
}

static void reversi_weigh(int8x16_t *w, const float *theta) {
    static const uint64_t CLASS[REVERSI_CLASSES] = { 0x8100000000000081ull, 0x4281000000008142ull, 0x2400810000810024ull, 0x1800008181000018ull, 0x0042000000004200ull,
                                                     0x0024420000422400ull, 0x0018004242001800ull, 0x0000240000240000ull, 0x0000182424180000ull, 0x0000001818000000ull };
    int8_t square[64];
    for (uint32_t c = 0u; c < REVERSI_CLASSES; c++)
        for (uint64_t rest = CLASS[c]; rest; rest &= rest - 1u) square[__builtin_ctzll(rest)] = (int8_t)lrintf(100.0f * theta[c]);
    for (uint32_t k = 0u; k < 4u; k++) w[k] = vld1q_s8(square + 16u * k);
}

static uint32_t reversi_pick(uint64_t ties, uint32_t draw) {
    for (uint32_t i = (uint32_t)(((uint64_t)draw * reversi_count(ties)) >> 32); i > 0u; i--) ties &= ties - 1u;
    return (uint32_t)__builtin_ctzll(ties);
}

static uint32_t reversi_break(uint64_t ties, uint32_t seed, uint32_t game, uint64_t all) {
    if (!(ties & (ties - 1u))) return (uint32_t)__builtin_ctzll(ties);
    return reversi_pick(ties, philox4x32(seed, ENV_RNG_STREAM, game, reversi_count(all), 0u, 0u).x);
}

static uint32_t reversi_next(uint64_t moves) {
    static const uint64_t GROUP[5] = { 0x8100000000000081ull, 0x2400810000810024ull, 0x1800248181240018ull, 0x0000182424180000ull, 0x003C425A5A423C00ull };
    for (uint32_t g = 0u; g < 5u; g++)
        if (moves & GROUP[g]) return (uint32_t)__builtin_ctzll(moves & GROUP[g]);
    return (uint32_t)__builtin_ctzll(moves);
}

static int32_t reversi_final(uint64_t me, uint64_t them) { return ((int32_t)reversi_count(me) - (int32_t)reversi_count(them)) * REVERSI_FINAL; }

static uint32_t reversi_think(const reversiLane *l, uint32_t seed, uint32_t game, uint64_t bot, uint64_t agent, uint64_t board, uint64_t moves) {
    int32_t alpha = -REVERSI_INFINITY;
    uint64_t ties = moves & (moves - 1u) ? 0u : moves;
    for (uint64_t rest = ties ? 0u : moves; rest;) {
        uint32_t move = reversi_next(rest);
        rest &= ~(1ull << move);
        uint64_t f = reversi_flips(bot, agent, move), b = bot | f | 1ull << move, a = agent ^ f, empty = board & ~(a | b), replies = reversi_moves(a, b, empty);
        int32_t low = replies ? REVERSI_INFINITY : reversi_moves(b, a, empty) ? reversi_wpc(b, a, l->weight) : reversi_final(b, a);
        while (replies && low >= alpha) {
            uint32_t reply = reversi_next(replies);
            replies &= ~(1ull << reply);
            uint64_t g = reversi_flips(a, b, reply), mine = b ^ g, theirs = a | g | 1ull << reply;
            int32_t value = (mine | theirs) == board ? reversi_final(mine, theirs) : reversi_wpc(mine, theirs, l->weight);
            low = value < low ? value : low;
        }
        if (low > alpha) {
            alpha = low;
            ties = 0u;
        }
        if (low == alpha) ties |= 1ull << move;
    }
    return reversi_break(ties, seed, game, bot | agent);
}

static uint32_t reversi_reply(const reversiLane *l, const reversiKnobs *k, uint32_t seed, uint32_t game, uint64_t bot, uint64_t agent, uint64_t moves) {
    if (k->opponent == REVERSI_ALPHABETA) return reversi_think(l, seed, game, bot, agent, reversi_board(k->size), moves);
    int32_t best = -REVERSI_INFINITY;
    uint64_t ties = 0u;
    for (uint64_t rest = moves; rest; rest &= rest - 1u) {
        uint32_t square = (uint32_t)__builtin_ctzll(rest);
        uint64_t f = reversi_flips(bot, agent, square);
        int32_t value = k->opponent == REVERSI_GREEDY ? (int32_t)reversi_count(f) : reversi_wpc(bot | f | 1ull << square, agent ^ f, l->weight);
        if (value > best) {
            best = value;
            ties = 0u;
        }
        if (value == best) ties |= 1ull << square;
    }
    return reversi_break(ties, seed, game, bot | agent);
}

static void reversi_league(reversiLane *l, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->winRate[i] = 0.5f;
    for (uint32_t d = 0u; d < REVERSI_CLASSES; d++) l->theta[d] = REVERSI_SWH[d];
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < REVERSI_CLASSES; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u && d + j < REVERSI_CLASSES; j++) {
                float value = REVERSI_SWH[d + j] + 0.1f * (REVERSI_THETA_HIGH[d + j] - REVERSI_THETA_LOW[d + j]) * n[j];
                l->theta[REVERSI_CLASSES * i + d + j] = fminf(fmaxf(value, REVERSI_THETA_LOW[d + j]), REVERSI_THETA_HIGH[d + j]);
            }
        }
}

static void reversi_propose(reversiLane *l, uint32_t *words, uint32_t seed, uint64_t own, uint64_t opp, uint64_t legal) {
    uint32_t n = 0u, discs = reversi_count(own | opp);
    for (uint64_t rest = legal; rest; rest &= rest - 1u) l->order[n++] = (uint8_t)__builtin_ctzll(rest);
    u4 r = { 0u, 0u, 0u, 0u };
    for (uint32_t i = n - 1u; i > 0u; i--) {
        if (i == n - 1u || (i & 3u) == 3u) r = philox4x32(seed, ENV_RNG_STREAM, words[REVERSI_GAME], discs, REVERSI_SHUFFLE, i >> 2);
        uint32_t draws[4] = { r.x, r.y, r.z, r.w }, j = (uint32_t)(((uint64_t)draws[i & 3u] * (i + 1u)) >> 32);
        uint8_t square = l->order[i];
        l->order[i] = l->order[j];
        l->order[j] = square;
    }
    reversi_put(words, REVERSI_OWN, own);
    reversi_put(words, REVERSI_OPP, opp);
    reversi_put(words, REVERSI_LEGAL, legal);
    words[REVERSI_PROPOSAL] = l->order[0];
    words[REVERSI_SHOWN] = 0u;
}

static void reversi_start(reversiLane *l, uint32_t *words, const reversiKnobs *k, uint32_t seed) {
    uint32_t game = words[REVERSI_GAME], c = k->size / 2u;
    uint64_t board = reversi_board(k->size), white = 1ull << 9u * (c - 1u) | 1ull << 9u * c, black = 1ull << (9u * c - 8u) | 1ull << (9u * c - 1u);
    uint64_t own = game & 1u ? white : black, opp = game & 1u ? black : white;
    words[REVERSI_LAST] = 0u;
    if (k->opponent == REVERSI_LEAGUE) {
        words[REVERSI_MEMBER] = leaguePick(l->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, game, REVERSI_PICK, 0u, 0u).x);
        reversi_weigh(l->weight, l->theta + REVERSI_CLASSES * words[REVERSI_MEMBER]);
    }
    if (game & 1u) {
        uint32_t square = reversi_reply(l, k, seed, game, opp, own, reversi_moves(opp, own, board & ~(own | opp)));
        uint64_t f = reversi_flips(opp, own, square);
        opp |= f | 1ull << square;
        own ^= f;
        words[REVERSI_LAST] = square + 1u;
    }
    reversi_propose(l, words, seed, own, opp, reversi_moves(own, opp, board & ~(own | opp)));
}

static __attribute__((noinline)) float reversi_over(reversiLane *l, uint32_t *words, const reversiKnobs *k, uint32_t seed, uint32_t *done, uint64_t own, uint64_t opp) {
    int32_t diff = (int32_t)reversi_count(own) - (int32_t)reversi_count(opp);
    float reward = diff > 0 ? 1.0f : diff < 0 ? -1.0f : 0.0f;
    *done = 1u;
    if (k->opponent == REVERSI_LEAGUE) {
        leagueScore(l->winRate, words[REVERSI_MEMBER], reward > 0.0f ? 1.0f : reward < 0.0f ? 0.0f : 0.5f);
        leagueRenew(l->winRate, l->theta, REVERSI_CLASSES, REVERSI_THETA_LOW, REVERSI_THETA_HIGH, seed, words[REVERSI_GAME]);
    }
    words[REVERSI_GAME]++;
    reversi_start(l, words, k, seed);
    return reward;
}

static __attribute__((noinline)) float reversi_place(reversiLane *l, uint32_t *words, const reversiKnobs *k, uint32_t seed, uint32_t *done) {
    uint64_t own = reversi_get(words, REVERSI_OWN), opp = reversi_get(words, REVERSI_OPP), board = reversi_board(k->size), legal;
    uint32_t square = words[REVERSI_PROPOSAL];
    uint64_t f = reversi_flips(own, opp, square);
    own |= f | 1ull << square;
    opp ^= f;
    uint64_t moves = reversi_moves(opp, own, board & ~(own | opp));
    words[REVERSI_LAST] = REVERSI_PASS;
    do {
        if (moves) {
            uint32_t reply = reversi_reply(l, k, seed, words[REVERSI_GAME], opp, own, moves);
            uint64_t g = reversi_flips(opp, own, reply);
            opp |= g | 1ull << reply;
            own ^= g;
            words[REVERSI_LAST] = reply + 1u;
        }
        uint64_t empty = board & ~(own | opp);
        legal = reversi_moves(own, opp, empty);
        moves = legal || !moves ? 0u : reversi_moves(opp, own, empty);
    } while (moves);
    if (!legal) return reversi_over(l, words, k, seed, done, own, opp);
    reversi_propose(l, words, seed, own, opp, legal);
    return 0.0f;
}

static inline __attribute__((always_inline)) void reversi_obs(uint32_t size, const uint32_t *words, float *obs) {
    uint64_t own = reversi_get(words, REVERSI_OWN), opp = reversi_get(words, REVERSI_OPP);
    uint32_t square = words[REVERSI_PROPOSAL];
    uint64_t f = reversi_flips(own, opp, square);
    own |= f | 1ull << square;
    opp ^= f;
    const lane4 low = { 0x1u, 0x2u, 0x4u, 0x8u }, high = { 0x10u, 0x20u, 0x40u, 0x80u };
    for (uint32_t y = 0u; y < size; y++) {
        lane4 mine = (lane4)(uint32_t)(own >> 8u * y), theirs = (lane4)(uint32_t)(opp >> 8u * y);
        float4 a = flag4(mine & low) - flag4(theirs & low), b = flag4(mine & high) - flag4(theirs & high);
        __builtin_memcpy(obs + size * y, &a, sizeof a);
        __builtin_memcpy(obs + size * y + 4u, &b, (size - 4u) * sizeof(float));
    }
}

static inline __attribute__((always_inline)) uint32_t reversi_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)REVERSI_ACTIONS ? (uint32_t)c : REVERSI_PLAY;
}

static inline __attribute__((always_inline)) float reversi_advance(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const reversiKnobs *k = &REVERSI_KNOBS[instance];
    reversiLane *l = reversi_lane(extra);
    uint32_t shown = words[REVERSI_SHOWN] + 1u;
    float reward = 0.0f;
    *done = 0u;
    if (reversi_action(ctrl) == REVERSI_PLAY || shown == reversi_count(reversi_get(words, REVERSI_LEGAL))) reward = reversi_place(l, words, k, seed, done);
    else {
        words[REVERSI_SHOWN] = shown;
        words[REVERSI_PROPOSAL] = l->order[shown];
    }
    reversi_obs(k->size, words, obs);
    return reward;
}

void reversi_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const reversiKnobs *k = &REVERSI_KNOBS[instance];
    reversiLane *l = reversi_lane(extra);
    for (uint32_t i = 0u; i < REVERSI_WORDS; i++) words[i] = 0u;
    reversi_weigh(l->weight, REVERSI_SWH);
    reversi_oracleOf(l)->key = 0xFFFFFFFFu;
    if (k->opponent == REVERSI_LEAGUE) reversi_league(l, seed);
    reversi_start(l, words, k, seed);
}

float reversi_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return reversi_advance(0u, seed, ctrl, words, extra, obs, done);
        case 1u: return reversi_advance(1u, seed, ctrl, words, extra, obs, done);
        case 2u: return reversi_advance(2u, seed, ctrl, words, extra, obs, done);
        default: return reversi_advance(3u, seed, ctrl, words, extra, obs, done);
    }
}

void reversi_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    reversi_obs(REVERSI_KNOBS[instance].size, words, obs);
}

static int32_t reversi_judge(uint64_t me, uint64_t them, uint32_t size) {
    uint64_t board = reversi_board(size), empty = board & ~(me | them), mine = reversi_moves(me, them, empty), theirs = reversi_moves(them, me, empty);
    uint64_t corners = 1ull | 1ull << (size - 1u) | 1ull << 8u * (size - 1u) | 1ull << 9u * (size - 1u);
    if (!(mine | theirs)) {
        int32_t diff = (int32_t)reversi_count(me) - (int32_t)reversi_count(them);
        return diff > 0 ? REVERSI_INFINITY / 2 : diff < 0 ? -REVERSI_INFINITY / 2 : -REVERSI_FINAL;
    }
    return 16 * ((int32_t)reversi_count(me & corners) - (int32_t)reversi_count(them & corners)) + 2 * ((int32_t)reversi_count(mine) - (int32_t)reversi_count(theirs));
}

static int32_t reversi_solve(reversiOracle *o, uint64_t me, uint64_t them, uint64_t board, int32_t alpha, int32_t beta) {
    uint64_t empty = board & ~(me | them), moves = reversi_moves(me, them, empty);
    if (!moves) {
        if (reversi_moves(them, me, empty)) return -reversi_solve(o, them, me, board, -beta, -alpha);
        int32_t diff = (int32_t)reversi_count(me) - (int32_t)reversi_count(them);
        return (diff > 0) - (diff < 0);
    }
    for (uint64_t rest = moves; rest && alpha < beta;) {
        if (++o->nodes > o->budget) return alpha;
        uint32_t square = reversi_next(rest);
        rest &= ~(1ull << square);
        uint64_t f = reversi_flips(me, them, square);
        int32_t value = -reversi_solve(o, them ^ f, me | f | 1ull << square, board, -beta, -alpha);
        alpha = value > alpha ? value : alpha;
    }
    return alpha;
}

static uint32_t reversi_decide(reversiOracle *o, const reversiKnobs *k, uint64_t me, uint64_t them) {
    uint64_t board = reversi_board(k->size), empty = board & ~(me | them), moves = reversi_moves(me, them, empty);
    uint32_t choice = (uint32_t)__builtin_ctzll(moves);
    if (!(moves & (moves - 1u))) return choice;
    if (reversi_count(empty) <= k->oracleExact) {
        o->nodes = 0u;
        o->budget = k->oracleNodes;
        int32_t best = -2;
        for (uint64_t rest = moves; rest && best < 1 && o->nodes <= o->budget;) {
            uint32_t square = reversi_next(rest);
            rest &= ~(1ull << square);
            uint64_t f = reversi_flips(me, them, square);
            int32_t value = -reversi_solve(o, them ^ f, me | f | 1ull << square, board, -1, -best);
            if (value > best) {
                best = value;
                choice = square;
            }
        }
        if (o->nodes <= o->budget && best >= 0) return choice;
    }
    int32_t top = -REVERSI_INFINITY;
    for (uint64_t rest = moves; rest; rest &= rest - 1u) {
        uint32_t square = (uint32_t)__builtin_ctzll(rest);
        uint64_t f = reversi_flips(me, them, square), m = me | f | 1ull << square, t = them ^ f, theirs = reversi_moves(t, m, board & ~(m | t));
        int32_t value = theirs ? REVERSI_INFINITY : reversi_judge(m, t, k->size);
        for (uint64_t replies = theirs; replies && value > top; replies &= replies - 1u) {
            uint32_t reply = (uint32_t)__builtin_ctzll(replies);
            uint64_t g = reversi_flips(t, m, reply);
            int32_t v = reversi_judge(m ^ g, t | g | 1ull << reply, k->size);
            value = v < value ? v : value;
        }
        if (value > top) {
            top = value;
            choice = square;
        }
    }
    return choice;
}

void reversi_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    reversiOracle *o = reversi_oracleOf(reversi_lane(extra));
    uint64_t own = reversi_get(words, REVERSI_OWN), opp = reversi_get(words, REVERSI_OPP);
    uint32_t key = words[REVERSI_GAME] << 7 | reversi_count(own | opp);
    if (o->key != key) {
        o->square = reversi_decide(o, &REVERSI_KNOBS[instance], own, opp);
        o->key = key;
    }
    ctrl[0] = (float)(words[REVERSI_PROPOSAL] == o->square ? REVERSI_PLAY : REVERSI_NEXT);
}
