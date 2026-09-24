#include "../env.h"

typedef struct { uint32_t material, opponent, nodes, depth, quiescence, period, oracleNodes; } chessKnobs;

static const chessKnobs CHESS_KNOBS[4] = {
    { 1u, 0u, 0u, 0u, 0u, 1u, 0u },
    { 2u, 0u, 0u, 0u, 0u, 1u, 0u },
    { 0u, 1u, 0u, 0u, 0u, 1u, 80u },
    { 0u, 2u, 512u, 2u, 4u, 2u, 4000u }
};

enum { CHESS_FULL, CHESS_KQK, CHESS_KRK };
enum { CHESS_TABLE, CHESS_RANDOM, CHESS_SEARCH };
enum { CHESS_UP, CHESS_DOWN, CHESS_LEFT, CHESS_RIGHT, CHESS_SELECT, CHESS_CANCEL, CHESS_ACTIONS };
enum { CHESS_PAWN = 1, CHESS_KNIGHT, CHESS_BISHOP, CHESS_ROOK, CHESS_QUEEN, CHESS_KING };
enum { CHESS_PLAIN, CHESS_DOUBLE, CHESS_PASSANT, CHESS_CASTLE };
enum { CHESS_ONGOING, CHESS_CHECKMATE, CHESS_INSUFFICIENT, CHESS_STALEMATE, CHESS_FIFTY, CHESS_THREEFOLD };
enum { CHESS_CURSOR = 8, CHESS_STATE, CHESS_GAME, CHESS_LAST, CHESS_THINK, CHESS_PLY, CHESS_WORDS };
enum { CHESS_CAPACITY = 256, CHESS_HISTORY = 104, CHESS_FRAMES = 16, CHESS_OPENING = 2, CHESS_CEILING = 9, CHESS_QUIESCENCE = 6, CHESS_OBS = 18 };
enum { CHESS_ENTRIES = 1 << 18, CHESS_LOST = 0x80, CHESS_HELD = 0x40, CHESS_RECORD = 9 };
enum { CHESS_MATE = 100000, CHESS_INFINITY = 200000, CHESS_DRAWN = 1 << 23, CHESS_CONTEMPT = 50, CHESS_MARGIN = 200 };
enum { CHESS_START = 1, CHESS_PICK, CHESS_TIE, CHESS_SHUFFLE };

static const uint32_t CHESS_UNGENERATED = 0xFFFFFFFFu;
static const uint64_t CHESS_FILE = 0x0101010101010101ull;
static const uint64_t CHESS_FILE_H = 0x8080808080808080ull;
static const uint64_t CHESS_DARK = 0xAA55AA55AA55AA55ull;

static const uint64_t CHESS_KNIGHT_ATTACKS[64] = {
    0x0000000000020400ull, 0x0000000000050800ull, 0x00000000000A1100ull, 0x0000000000142200ull,
    0x0000000000284400ull, 0x0000000000508800ull, 0x0000000000A01000ull, 0x0000000000402000ull,
    0x0000000002040004ull, 0x0000000005080008ull, 0x000000000A110011ull, 0x0000000014220022ull,
    0x0000000028440044ull, 0x0000000050880088ull, 0x00000000A0100010ull, 0x0000000040200020ull,
    0x0000000204000402ull, 0x0000000508000805ull, 0x0000000A1100110Aull, 0x0000001422002214ull,
    0x0000002844004428ull, 0x0000005088008850ull, 0x000000A0100010A0ull, 0x0000004020002040ull,
    0x0000020400040200ull, 0x0000050800080500ull, 0x00000A1100110A00ull, 0x0000142200221400ull,
    0x0000284400442800ull, 0x0000508800885000ull, 0x0000A0100010A000ull, 0x0000402000204000ull,
    0x0002040004020000ull, 0x0005080008050000ull, 0x000A1100110A0000ull, 0x0014220022140000ull,
    0x0028440044280000ull, 0x0050880088500000ull, 0x00A0100010A00000ull, 0x0040200020400000ull,
    0x0204000402000000ull, 0x0508000805000000ull, 0x0A1100110A000000ull, 0x1422002214000000ull,
    0x2844004428000000ull, 0x5088008850000000ull, 0xA0100010A0000000ull, 0x4020002040000000ull,
    0x0400040200000000ull, 0x0800080500000000ull, 0x1100110A00000000ull, 0x2200221400000000ull,
    0x4400442800000000ull, 0x8800885000000000ull, 0x100010A000000000ull, 0x2000204000000000ull,
    0x0004020000000000ull, 0x0008050000000000ull, 0x00110A0000000000ull, 0x0022140000000000ull,
    0x0044280000000000ull, 0x0088500000000000ull, 0x0010A00000000000ull, 0x0020400000000000ull
};

static const uint64_t CHESS_KING_ATTACKS[64] = {
    0x0000000000000302ull, 0x0000000000000705ull, 0x0000000000000E0Aull, 0x0000000000001C14ull,
    0x0000000000003828ull, 0x0000000000007050ull, 0x000000000000E0A0ull, 0x000000000000C040ull,
    0x0000000000030203ull, 0x0000000000070507ull, 0x00000000000E0A0Eull, 0x00000000001C141Cull,
    0x0000000000382838ull, 0x0000000000705070ull, 0x0000000000E0A0E0ull, 0x0000000000C040C0ull,
    0x0000000003020300ull, 0x0000000007050700ull, 0x000000000E0A0E00ull, 0x000000001C141C00ull,
    0x0000000038283800ull, 0x0000000070507000ull, 0x00000000E0A0E000ull, 0x00000000C040C000ull,
    0x0000000302030000ull, 0x0000000705070000ull, 0x0000000E0A0E0000ull, 0x0000001C141C0000ull,
    0x0000003828380000ull, 0x0000007050700000ull, 0x000000E0A0E00000ull, 0x000000C040C00000ull,
    0x0000030203000000ull, 0x0000070507000000ull, 0x00000E0A0E000000ull, 0x00001C141C000000ull,
    0x0000382838000000ull, 0x0000705070000000ull, 0x0000E0A0E0000000ull, 0x0000C040C0000000ull,
    0x0003020300000000ull, 0x0007050700000000ull, 0x000E0A0E00000000ull, 0x001C141C00000000ull,
    0x0038283800000000ull, 0x0070507000000000ull, 0x00E0A0E000000000ull, 0x00C040C000000000ull,
    0x0302030000000000ull, 0x0705070000000000ull, 0x0E0A0E0000000000ull, 0x1C141C0000000000ull,
    0x3828380000000000ull, 0x7050700000000000ull, 0xE0A0E00000000000ull, 0xC040C00000000000ull,
    0x0203000000000000ull, 0x0507000000000000ull, 0x0A0E000000000000ull, 0x141C000000000000ull,
    0x2838000000000000ull, 0x5070000000000000ull, 0xA0E0000000000000ull, 0x40C0000000000000ull
};

static const uint64_t CHESS_DIAGONALS[15] = {
    0x0100000000000000ull, 0x0201000000000000ull, 0x0402010000000000ull, 0x0804020100000000ull, 0x1008040201000000ull, 0x2010080402010000ull, 0x4020100804020100ull, 0x8040201008040201ull,
    0x0080402010080402ull, 0x0000804020100804ull, 0x0000008040201008ull, 0x0000000080402010ull, 0x0000000000804020ull, 0x0000000000008040ull, 0x0000000000000080ull
};

static const uint64_t CHESS_ANTIDIAGONALS[15] = {
    0x0000000000000001ull, 0x0000000000000102ull, 0x0000000000010204ull, 0x0000000001020408ull, 0x0000000102040810ull, 0x0000010204081020ull, 0x0001020408102040ull, 0x0102040810204080ull,
    0x0204081020408000ull, 0x0408102040800000ull, 0x0810204080000000ull, 0x1020408000000000ull, 0x2040800000000000ull, 0x4080000000000000ull, 0x8000000000000000ull
};

static const int16_t CHESS_VALUE[8] = { 0, 100, 320, 330, 500, 900, 0, 0 };

static const int16_t CHESS_SQUARES[8][64] = {
    { -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
      -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
      0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0 },
    { -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
      -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50 },
    { -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
      -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0 },
    { -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5,
      0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20 },
    { 0 },
    { -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30,
      -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30, -30, -50 }
};

static const uint8_t CHESS_PROMOTE[4] = { CHESS_QUEEN, CHESS_ROOK, CHESS_BISHOP, CHESS_KNIGHT };

typedef struct {
    uint64_t bb[8];
    uint8_t board[64];
    uint8_t side, castle, ep, clock;
    int32_t score;
} chessPos;

typedef struct {
    chessPos pos;
    int32_t alpha, beta, best;
    uint32_t count, next, depth, move;
    uint32_t list[CHESS_CAPACITY];
} chessFrame;

typedef struct {
    uint32_t sp, nodes, budget, done, choice, leader, horizon, ceiling, quiescence, careful;
    int32_t floor;
    chessFrame frame[CHESS_FRAMES];
} chessSearch;

typedef struct {
    float obs[20];
    float pad[104];
    chessPos pos;
    uint64_t movable, targets;
    uint32_t count, captures, check, seen, twice;
    uint32_t list[CHESS_CAPACITY];
    uint32_t hash[CHESS_HISTORY];
    uint32_t record[CHESS_HISTORY][CHESS_RECORD];
    uint8_t repeated[CHESS_HISTORY];
    chessSearch search;
} chessLane;

typedef struct {
    uint32_t game, ply, move;
    chessSearch search;
} chessOracle;

static chessLane *chess_lane(uint32_t *extra) { return (chessLane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const chessLane *chess_peek(const uint32_t *extra) { return (const chessLane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static chessOracle *chess_oracleOf(chessLane *l) { return (chessOracle *)(l + 1); }

uint32_t chess_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)((sizeof(chessLane) + sizeof(chessOracle)) / sizeof(uint32_t)) + 4u;
}

static inline uint32_t chess_low(uint64_t b) { return (uint32_t)__builtin_ctzll(b); }

static inline uint64_t chess_ray(uint64_t occ, uint64_t line, uint64_t bit) {
    uint64_t o = occ & line;
    return ((o - bit) ^ __builtin_bitreverse64(__builtin_bitreverse64(o) - __builtin_bitreverse64(bit))) & line;
}

static inline uint64_t chess_rook(uint32_t s, uint64_t occ) {
    uint64_t bit = 1ull << s;
    return chess_ray(occ, (CHESS_FILE << (s & 7u)) ^ bit, bit) | chess_ray(occ, (0xFFull << (s & 56u)) ^ bit, bit);
}

static inline uint64_t chess_bishop(uint32_t s, uint64_t occ) {
    uint64_t bit = 1ull << s;
    return chess_ray(occ, CHESS_DIAGONALS[7u + (s & 7u) - (s >> 3)] ^ bit, bit) | chess_ray(occ, CHESS_ANTIDIAGONALS[(s & 7u) + (s >> 3)] ^ bit, bit);
}

static inline uint64_t chess_reach(uint32_t piece, uint32_t s, uint64_t occ) {
    return piece == CHESS_QUEEN ? chess_rook(s, occ) | chess_bishop(s, occ) : chess_rook(s, occ);
}

static inline uint64_t chess_pawnSet(uint64_t pawns, uint32_t color) {
    return color ? ((pawns >> 9) & ~CHESS_FILE_H) | ((pawns >> 7) & ~CHESS_FILE) : ((pawns << 7) & ~CHESS_FILE_H) | ((pawns << 9) & ~CHESS_FILE);
}

static inline uint64_t chess_side(const chessPos *p, uint32_t color) { return p->bb[7u * color]; }

static uint64_t chess_attacks(const chessPos *p, uint32_t by, uint64_t occ) {
    uint64_t side = chess_side(p, by), a = chess_pawnSet(p->bb[CHESS_PAWN] & side, by) | CHESS_KING_ATTACKS[chess_low(p->bb[CHESS_KING] & side)];
    for (uint64_t n = p->bb[CHESS_KNIGHT] & side; n; n &= n - 1u) a |= CHESS_KNIGHT_ATTACKS[chess_low(n)];
    for (uint64_t d = (p->bb[CHESS_BISHOP] | p->bb[CHESS_QUEEN]) & side; d; d &= d - 1u) a |= chess_bishop(chess_low(d), occ);
    for (uint64_t o = (p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN]) & side; o; o &= o - 1u) a |= chess_rook(chess_low(o), occ);
    return a;
}

static inline uint64_t chess_attackers(const chessPos *p, uint32_t s, uint32_t by, uint64_t occ) {
    uint64_t bit = 1ull << s;
    return ((CHESS_KNIGHT_ATTACKS[s] & p->bb[CHESS_KNIGHT]) | (CHESS_KING_ATTACKS[s] & p->bb[CHESS_KING]) | (chess_pawnSet(bit, by ^ 1u) & p->bb[CHESS_PAWN]) |
            (chess_bishop(s, occ) & (p->bb[CHESS_BISHOP] | p->bb[CHESS_QUEEN])) | (chess_rook(s, occ) & (p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN]))) & chess_side(p, by);
}

static uint64_t chess_between(uint32_t a, uint32_t b) {
    uint64_t ends = (1ull << a) | (1ull << b);
    if (((a ^ b) & 7u) == 0u || ((a ^ b) >> 3) == 0u) return chess_rook(a, ends) & chess_rook(b, ends);
    return chess_bishop(a, ends) & chess_bishop(b, ends);
}

static uint64_t chess_line(uint32_t a, uint32_t b) {
    if (((a ^ b) & 7u) == 0u) return CHESS_FILE << (a & 7u);
    if (((a ^ b) >> 3) == 0u) return 0xFFull << (a & 56u);
    uint64_t diagonal = CHESS_DIAGONALS[7u + (a & 7u) - (a >> 3)];
    return diagonal >> b & 1u ? diagonal : CHESS_ANTIDIAGONALS[(a & 7u) + (a >> 3)];
}

static inline uint32_t chess_word(uint32_t from, uint32_t to, uint32_t piece, uint32_t taken, uint32_t promo, uint32_t special) {
    return from | to << 6 | piece << 12 | taken << 15 | promo << 18 | special << 21 | (taken ? (64u | taken << 3 | (7u - piece)) << 24 : 0u);
}

static inline void chess_spread(const chessPos *p, uint32_t *list, uint32_t *nc, uint32_t *nq, uint32_t from, uint32_t piece, uint64_t targets, uint64_t enemy) {
    for (uint64_t t = targets & enemy; t; t &= t - 1u) {
        uint32_t to = chess_low(t);
        list[(*nc)++] = chess_word(from, to, piece, p->board[to] & 7u, 0u, 0u);
    }
    for (uint64_t t = targets & ~enemy; t; t &= t - 1u) list[CHESS_CAPACITY - ++(*nq)] = chess_word(from, chess_low(t), piece, 0u, 0u, 0u);
}

static inline void chess_promote(uint32_t *list, uint32_t *n, uint32_t reverse, uint32_t from, uint32_t to, uint32_t taken) {
    for (uint32_t i = 0u; i < 4u; i++) {
        uint32_t move = chess_word(from, to, CHESS_PAWN, taken, CHESS_PROMOTE[i], CHESS_PLAIN);
        if (reverse) list[CHESS_CAPACITY - ++(*n)] = move;
        else list[(*n)++] = move;
    }
}

static inline void chess_pawns(const chessPos *p, uint32_t *list, uint32_t *nc, uint32_t *nq, uint64_t pawns, uint64_t capture, uint64_t move, uint64_t empty) {
    uint32_t us = p->side;
    uint64_t last = us ? 0xFFull : 0xFF00000000000000ull, third = us ? 0x0000FF0000000000ull : 0x0000000000FF0000ull;
    uint64_t single = (us ? pawns >> 8 : pawns << 8) & empty, twice = (us ? (single & third) >> 8 : (single & third) << 8) & empty & move;
    uint64_t west = (us ? (pawns & ~CHESS_FILE) >> 9 : (pawns & ~CHESS_FILE) << 7) & capture;
    uint64_t east = (us ? (pawns & ~CHESS_FILE_H) >> 7 : (pawns & ~CHESS_FILE_H) << 9) & capture;
    single &= move;
    for (uint64_t t = west; t; t &= t - 1u) {
        uint32_t to = chess_low(t), from = us ? to + 9u : to - 7u;
        if ((1ull << to) & last) chess_promote(list, nc, 0u, from, to, p->board[to] & 7u);
        else list[(*nc)++] = chess_word(from, to, CHESS_PAWN, p->board[to] & 7u, 0u, CHESS_PLAIN);
    }
    for (uint64_t t = east; t; t &= t - 1u) {
        uint32_t to = chess_low(t), from = us ? to + 7u : to - 9u;
        if ((1ull << to) & last) chess_promote(list, nc, 0u, from, to, p->board[to] & 7u);
        else list[(*nc)++] = chess_word(from, to, CHESS_PAWN, p->board[to] & 7u, 0u, CHESS_PLAIN);
    }
    for (uint64_t t = single; t; t &= t - 1u) {
        uint32_t to = chess_low(t), from = us ? to + 8u : to - 8u;
        if ((1ull << to) & last) chess_promote(list, nq, 1u, from, to, 0u);
        else list[CHESS_CAPACITY - ++(*nq)] = chess_word(from, to, CHESS_PAWN, 0u, 0u, CHESS_PLAIN);
    }
    for (uint64_t t = twice; t; t &= t - 1u) {
        uint32_t to = chess_low(t);
        list[CHESS_CAPACITY - ++(*nq)] = chess_word(us ? to + 16u : to - 16u, to, CHESS_PAWN, 0u, 0u, CHESS_DOUBLE);
    }
}

static uint32_t chess_generate(const chessPos *p, uint32_t *list, uint32_t quiet, uint32_t *captures, uint64_t *checking) {
    uint32_t us = p->side, them = us ^ 1u, nc = 0u, nq = 0u;
    uint64_t own = chess_side(p, us), enemy = chess_side(p, them), occ = own | enemy, empty = ~occ;
    uint64_t kingBit = p->bb[CHESS_KING] & own, orthogonal = (p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN]) & enemy, diagonal = (p->bb[CHESS_BISHOP] | p->bb[CHESS_QUEEN]) & enemy;
    uint32_t king = chess_low(kingBit);
    uint64_t checkers = chess_attackers(p, king, them, occ), danger = 0u, steps = CHESS_KING_ATTACKS[king] & (quiet ? ~own : enemy);
    if (quiet) {
        danger = chess_attacks(p, them, occ ^ kingBit);
        steps &= ~danger;
    } else
        for (uint64_t t = steps; t; t &= t - 1u)
            if (chess_attackers(p, chess_low(t), them, occ ^ kingBit)) steps ^= t & -t;
    chess_spread(p, list, &nc, &nq, king, CHESS_KING, steps, enemy);
    if (!(checkers & (checkers - 1u))) {
        uint64_t mask = checkers ? ((checkers & (orthogonal | diagonal)) ? chess_between(king, chess_low(checkers)) | checkers : checkers) : ~0ull;
        uint64_t capture = enemy & mask, move = (quiet ? empty : 0u) & mask, pinned = 0u;
        uint64_t lines = (CHESS_FILE << (king & 7u)) | (0xFFull << (king & 56u)), cross = CHESS_DIAGONALS[7u + (king & 7u) - (king >> 3)] | CHESS_ANTIDIAGONALS[(king & 7u) + (king >> 3)];
        if ((orthogonal & lines) | (diagonal & cross)) {
            uint64_t snipers = (chess_rook(king, enemy) & orthogonal) | (chess_bishop(king, enemy) & diagonal);
            for (; snipers; snipers &= snipers - 1u) {
                uint64_t between = chess_between(king, chess_low(snipers)) & occ;
                if (between && !(between & (between - 1u))) pinned |= between & own;
            }
        }
        for (uint64_t n = p->bb[CHESS_KNIGHT] & own & ~pinned; n; n &= n - 1u) {
            uint32_t s = chess_low(n);
            chess_spread(p, list, &nc, &nq, s, CHESS_KNIGHT, CHESS_KNIGHT_ATTACKS[s] & (capture | move), enemy);
        }
        for (uint64_t d = (p->bb[CHESS_BISHOP] | p->bb[CHESS_QUEEN]) & own; d; d &= d - 1u) {
            uint32_t s = chess_low(d);
            uint64_t t = chess_bishop(s, occ) & (capture | move);
            if (pinned >> s & 1u) t &= chess_line(king, s);
            chess_spread(p, list, &nc, &nq, s, p->board[s] & 7u, t, enemy);
        }
        for (uint64_t o = (p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN]) & own; o; o &= o - 1u) {
            uint32_t s = chess_low(o);
            uint64_t t = chess_rook(s, occ) & (capture | move);
            if (pinned >> s & 1u) t &= chess_line(king, s);
            chess_spread(p, list, &nc, &nq, s, p->board[s] & 7u, t, enemy);
        }
        uint64_t pawns = p->bb[CHESS_PAWN] & own;
        if (pawns) {
            chess_pawns(p, list, &nc, &nq, pawns & ~pinned, capture, move, empty);
            for (uint64_t b = pawns & pinned; b; b &= b - 1u) {
                uint64_t line = chess_line(king, chess_low(b));
                chess_pawns(p, list, &nc, &nq, b & -b, capture & line, move & line, empty);
            }
            if (p->ep) {
                uint32_t target = p->ep, victim = target ^ 8u;
                for (uint64_t c = chess_pawnSet(1ull << target, them) & pawns; c; c &= c - 1u) {
                    uint32_t from = chess_low(c);
                    uint64_t after = (occ ^ (1ull << from) ^ (1ull << victim)) | (1ull << target), foes = enemy ^ (1ull << victim);
                    uint64_t hit = ((chess_rook(king, after) & orthogonal) | (chess_bishop(king, after) & diagonal) | (CHESS_KNIGHT_ATTACKS[king] & p->bb[CHESS_KNIGHT]) |
                                    (chess_pawnSet(kingBit, us) & p->bb[CHESS_PAWN])) & foes;
                    if (!hit) list[nc++] = chess_word(from, target, CHESS_PAWN, CHESS_PAWN, 0u, CHESS_PASSANT);
                }
            }
        }
        if (quiet && !checkers && p->castle) {
            uint32_t rights = p->castle >> (2u * us), home = us ? 56u : 0u;
            if ((rights & 1u) && !((occ | danger) & (0x60ull << home))) list[CHESS_CAPACITY - ++nq] = chess_word(home + 4u, home + 6u, CHESS_KING, 0u, 0u, CHESS_CASTLE);
            if ((rights & 2u) && !(occ & (0x0Eull << home)) && !(danger & (0x0Cull << home))) list[CHESS_CAPACITY - ++nq] = chess_word(home + 4u, home + 2u, CHESS_KING, 0u, 0u, CHESS_CASTLE);
        }
    }
    __builtin_memmove(list + nc, list + CHESS_CAPACITY - nq, nq * sizeof(uint32_t));
    *captures = nc;
    *checking = checkers;
    return nc + nq;
}

static inline int32_t chess_psq(uint32_t piece, uint32_t color, uint32_t s) { return CHESS_VALUE[piece] + CHESS_SQUARES[piece][color ? s : s ^ 56u]; }

static uint32_t chess_castleLoss(uint32_t s) { return s == 0u ? 2u : s == 4u ? 3u : s == 7u ? 1u : s == 56u ? 8u : s == 60u ? 12u : s == 63u ? 4u : 0u; }

static void chess_make(chessPos *p, uint32_t move) {
    uint32_t from = move & 63u, to = move >> 6 & 63u, piece = move >> 12 & 7u, taken = move >> 15 & 7u, promo = move >> 18 & 7u, special = move >> 21 & 3u;
    uint32_t us = p->side, them = us ^ 1u;
    uint64_t toBit = 1ull << to, moved = (1ull << from) | toBit;
    int32_t delta = chess_psq(piece, us, to) - chess_psq(piece, us, from);
    p->bb[piece] ^= moved;
    p->bb[7u * us] ^= moved;
    p->board[from] = 0u;
    p->board[to] = (uint8_t)(piece | us << 3);
    if (taken) {
        uint32_t at = special == CHESS_PASSANT ? to ^ 8u : to;
        uint64_t atBit = 1ull << at;
        p->bb[taken] ^= atBit;
        p->bb[7u * them] ^= atBit;
        if (at != to) p->board[at] = 0u;
        delta += chess_psq(taken, them, at);
    }
    if (promo) {
        p->bb[CHESS_PAWN] ^= toBit;
        p->bb[promo] ^= toBit;
        p->board[to] = (uint8_t)(promo | us << 3);
        delta += chess_psq(promo, us, to) - chess_psq(CHESS_PAWN, us, to);
    }
    if (special == CHESS_CASTLE) {
        uint32_t rookFrom = to > from ? to + 1u : to - 2u, rookTo = to > from ? to - 1u : to + 1u;
        uint64_t rookMoved = (1ull << rookFrom) | (1ull << rookTo);
        p->bb[CHESS_ROOK] ^= rookMoved;
        p->bb[7u * us] ^= rookMoved;
        p->board[rookFrom] = 0u;
        p->board[rookTo] = (uint8_t)(CHESS_ROOK | us << 3);
        delta += chess_psq(CHESS_ROOK, us, rookTo) - chess_psq(CHESS_ROOK, us, rookFrom);
    }
    p->score += us ? -delta : delta;
    p->castle &= (uint8_t)~(chess_castleLoss(from) | chess_castleLoss(to));
    p->ep = (uint8_t)(special == CHESS_DOUBLE ? (from + to) >> 1 : 0u);
    p->clock = (uint8_t)(piece == CHESS_PAWN || taken ? 0u : p->clock + 1u);
    p->side = (uint8_t)them;
}

static void chess_rebuild(chessPos *p) {
    for (uint32_t i = 0u; i < 8u; i++) p->bb[i] = 0u;
    p->score = 0;
    for (uint32_t s = 0u; s < 64u; s++) {
        uint32_t code = p->board[s];
        if (!code) continue;
        p->bb[code & 7u] |= 1ull << s;
        p->bb[7u * (code >> 3)] |= 1ull << s;
        p->score += (code >> 3) ? -chess_psq(code & 7u, 1u, s) : chess_psq(code & 7u, 0u, s);
    }
}

static uint32_t chess_ending(const chessPos *p, uint32_t color) {
    uint64_t side = chess_side(p, color);
    if (!(p->bb[CHESS_QUEEN] & side)) return 1u;
    return __builtin_popcountll((p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN]) & side) == 1 && __builtin_popcountll((p->bb[CHESS_KNIGHT] | p->bb[CHESS_BISHOP]) & side) <= 1;
}

static int32_t chess_eval(const chessPos *p) {
    uint32_t table = chess_ending(p, 0u) && chess_ending(p, 1u) ? 7u : 0u;
    int32_t s = p->score + CHESS_SQUARES[table][chess_low(p->bb[CHESS_KING] & p->bb[0]) ^ 56u] - CHESS_SQUARES[table][chess_low(p->bb[CHESS_KING] & p->bb[7])];
    return p->side ? -s : s;
}

static uint32_t chess_lacking(const chessPos *p, uint32_t color) {
    uint64_t side = chess_side(p, color), other = chess_side(p, color ^ 1u);
    if (side & (p->bb[CHESS_PAWN] | p->bb[CHESS_ROOK] | p->bb[CHESS_QUEEN])) return 0u;
    if (side & p->bb[CHESS_KNIGHT]) return __builtin_popcountll(side) <= 2 && !(other & ~p->bb[CHESS_KING] & ~p->bb[CHESS_QUEEN]);
    if (side & p->bb[CHESS_BISHOP]) return (!(p->bb[CHESS_BISHOP] & CHESS_DARK) || !(p->bb[CHESS_BISHOP] & ~CHESS_DARK)) && !p->bb[CHESS_PAWN] && !p->bb[CHESS_KNIGHT];
    return 1u;
}

static void chess_pack(const chessPos *p, uint32_t *out) {
    for (uint32_t i = 0u; i < 4u; i++) {
        uint16x8_t v = vreinterpretq_u16_u8(vld1q_u8(p->board + 16u * i));
        vst1_u8((uint8_t *)(out + 2u * i), vmovn_u16(vsraq_n_u16(v, v, 4)));
    }
}

static uint32_t chess_hash(const uint32_t *record) {
    uint64_t w[4];
    __builtin_memcpy(w, record, sizeof w);
    uint32_t h = __builtin_arm_crc32cd(0u, w[0]);
    h = __builtin_arm_crc32cd(h, w[1]);
    h = __builtin_arm_crc32cd(h, w[2]);
    h = __builtin_arm_crc32cd(h, w[3]);
    return __builtin_arm_crc32cw(h, record[8]);
}

static uint32_t chess_occurrences(const chessLane *l, const uint32_t *record, uint32_t hash) {
    uint32_t n = 0u;
    for (uint32_t i = 0u; i < l->seen; i++) n += l->hash[i] == hash && !__builtin_memcmp(l->record[i], record, sizeof l->record[i]);
    return n;
}

static uint32_t chess_quietMove(uint32_t move) { return (move >> 12 & 7u) != CHESS_PAWN && !(move >> 15 & 7u); }

static uint32_t chess_fiftyClaim(const chessLane *l) {
    uint32_t scratch[CHESS_CAPACITY], captures;
    uint64_t checkers;
    for (uint32_t i = 0u; i < l->count; i++) {
        if (!chess_quietMove(l->list[i])) continue;
        chessPos next = l->pos;
        chess_make(&next, l->list[i]);
        if (chess_generate(&next, scratch, 1u, &captures, &checkers)) return 1u;
    }
    return 0u;
}

static uint32_t chess_reachable(const chessLane *l, const uint32_t *record, const uint32_t *list, uint32_t count) {
    for (uint32_t r = 0u; r < l->twice; r++) {
        const uint32_t *t = l->record[l->repeated[r]];
        if (t[8] != ((record[8] & 31u) ^ 1u)) continue;
        uint32_t from = 64u, to = 64u, diff = 0u;
        for (uint32_t w = 0u; w < 8u && diff <= 2u; w++)
            for (uint32_t x = record[w] ^ t[w], s = 8u * w; x; x >>= 4, s++) {
                if (!(x & 15u)) continue;
                diff++;
                if ((t[w] >> 4u * (s & 7u) & 15u) == 0u) from = s;
                else to = s;
            }
        if (diff != 2u || from == 64u || to == 64u) continue;
        for (uint32_t i = 0u; i < count; i++)
            if ((list[i] & 4095u) == (from | to << 6) && chess_quietMove(list[i])) return 1u;
    }
    return 0u;
}

static uint32_t chess_settle(chessLane *l, uint32_t *words, uint32_t agent, uint32_t irreversible) {
    chessPos *p = &l->pos;
    uint64_t checkers;
    l->count = chess_generate(p, l->list, 1u, &l->captures, &checkers);
    l->check = checkers != 0u;
    uint32_t record[CHESS_RECORD], passant = 0u;
    chess_pack(p, record);
    __builtin_memcpy(words, record, 8u * sizeof(uint32_t));
    for (uint32_t i = 0u; p->ep && i < l->captures; i++) passant |= (l->list[i] >> 21 & 3u) == CHESS_PASSANT;
    record[8] = p->side | (uint32_t)p->castle << 1 | (passant ? (p->ep & 7u) + 1u : 0u) << 5;
    if (irreversible) l->seen = l->twice = 0u;
    if (l->seen == CHESS_HISTORY) __builtin_trap();
    uint32_t hash = chess_hash(record), seen = chess_occurrences(l, record, hash) + 1u;
    l->hash[l->seen] = hash;
    __builtin_memcpy(l->record[l->seen], record, sizeof record);
    if (seen == 2u) l->repeated[l->twice++] = (uint8_t)l->seen;
    l->seen++;
    words[CHESS_STATE] = p->side | (uint32_t)p->castle << 1 | (uint32_t)p->ep << 8 | (uint32_t)p->clock << 16;
    words[CHESS_LAST] = (words[CHESS_LAST] & ~0x10000u) | (p->side == agent && l->check) << 16;
    if (!l->count && checkers) return CHESS_CHECKMATE;
    if (chess_lacking(p, 0u) && chess_lacking(p, 1u)) return CHESS_INSUFFICIENT;
    if (!l->count) return CHESS_STALEMATE;
    if (p->clock >= 100u || (p->clock == 99u && chess_fiftyClaim(l))) return CHESS_FIFTY;
    if (seen >= 3u || chess_reachable(l, record, l->list, l->count)) return CHESS_THREEFOLD;
    return CHESS_ONGOING;
}

static uint32_t chess_agent(const chessKnobs *k, const uint32_t *words) { return k->material == CHESS_FULL ? words[CHESS_GAME] & 1u : 0u; }

static uint32_t chess_cell(uint32_t square) { return (square >> 3) * 10u + (square & 7u) + 11u; }

static float chess_code(uint32_t code, uint32_t agent) {
    static const float sixths[8] = { 0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f, 1.0f, 0.0f };
    float value = sixths[code & 7u];
    return !code || (code >> 3) == agent ? value : -value;
}

static void chess_paint(chessLane *l, uint32_t agent) {
    for (uint32_t i = 0u; i < 104u; i++) l->pad[i] = 0.0f;
    for (uint32_t s = 0u; s < 64u; s++) l->pad[chess_cell(s ^ (agent ? 63u : 0u))] = chess_code(l->pos.board[s], agent);
}

static void chess_touch(chessLane *l, uint32_t agent, uint64_t squares) {
    for (; squares; squares &= squares - 1u) {
        uint32_t s = chess_low(squares);
        l->pad[chess_cell(s ^ (agent ? 63u : 0u))] = chess_code(l->pos.board[s], agent);
    }
}

static void chess_frame(chessLane *l, uint32_t cw, uint32_t agent) {
    static const float sevenths[8] = { 0.0f, 1.0f / 7.0f, 2.0f / 7.0f, 3.0f / 7.0f, 4.0f / 7.0f, 5.0f / 7.0f, 6.0f / 7.0f, 1.0f };
    float *o = l->obs;
    uint32_t cursor = cw & 63u, file = cursor & 7u, rank = cursor >> 3, sel = cw >> 8 & 127u;
    const float *row = l->pad + rank * 10u + file;
    vst1q_f32(o, vld1q_f32(row + 20u));
    vst1q_f32(o + 3, vld1q_f32(row + 10u));
    vst1q_f32(o + 6, vld1q_f32(row));
    o[9] = sevenths[file];
    o[10] = sevenths[rank];
    o[12] = (cw >> 16 & 127u) ? 1.0f : (float)(((sel ? l->targets : l->movable) >> (cursor ^ (agent ? 63u : 0u))) & 1u);
}

static void chess_status(chessLane *l, const uint32_t *words, uint32_t agent) {
    float *o = l->obs;
    uint32_t flip = agent ? 63u : 0u, cw = words[CHESS_CURSOR], sel = cw >> 8 & 127u, promo = cw >> 16 & 127u, cand = cw >> 24 & 3u;
    uint32_t last = words[CHESS_LAST], from = last & 127u, to = last >> 8 & 127u;
    o[11] = promo ? (float)CHESS_PROMOTE[cand] / 6.0f : sel ? (float)(l->pos.board[sel - 1u] & 7u) / 6.0f : 0.0f;
    o[13] = from ? (float)((from - 1u) ^ flip) / 63.0f : -1.0f;
    o[14] = to ? (float)((to - 1u) ^ flip) / 63.0f : -1.0f;
    o[15] = (float)(last >> 16 & 1u);
    o[16] = words[CHESS_THINK] ? 1.0f : 0.0f;
    o[17] = to ? l->pad[chess_cell((to - 1u) ^ flip)] : 0.0f;
}

static void chess_view(chessLane *l, const uint32_t *words, uint32_t agent) {
    chess_frame(l, words[CHESS_CURSOR], agent);
    chess_status(l, words, agent);
}

static uint64_t chess_movable(const chessLane *l) {
    uint64_t m = 0u;
    for (uint32_t i = 0u; i < l->count; i++) m |= 1ull << (l->list[i] & 63u);
    return m;
}

static uint64_t chess_targets(const chessLane *l, uint32_t from) {
    uint64_t m = 0u;
    for (uint32_t i = 0u; i < l->count; i++) m |= (uint64_t)((l->list[i] & 63u) == from) << (l->list[i] >> 6 & 63u);
    return m;
}

static uint32_t chess_find(const chessLane *l, uint32_t from, uint32_t to, uint32_t promo) {
    for (uint32_t i = 0u; i < l->count; i++) {
        uint32_t m = l->list[i];
        if ((m & 4095u) == (from | to << 6) && (!promo || (m >> 18 & 7u) == promo)) return m;
    }
    __builtin_trap();
}

static void chess_standard(chessPos *p) {
    static const uint8_t back[8] = { CHESS_ROOK, CHESS_KNIGHT, CHESS_BISHOP, CHESS_QUEEN, CHESS_KING, CHESS_BISHOP, CHESS_KNIGHT, CHESS_ROOK };
    __builtin_memset(p->board, 0, sizeof p->board);
    for (uint32_t f = 0u; f < 8u; f++) {
        p->board[f] = back[f];
        p->board[8u + f] = CHESS_PAWN;
        p->board[48u + f] = CHESS_PAWN | 8u;
        p->board[56u + f] = (uint8_t)(back[f] | 8u);
    }
    p->side = 0u;
    p->castle = 15u;
    p->ep = 0u;
    p->clock = 0u;
    chess_rebuild(p);
}

static void chess_scatter(chessPos *p, uint32_t piece, uint32_t seed, uint32_t game) {
    for (uint32_t block = 0u;; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, game, block, CHESS_START, 0u);
        uint32_t draws[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t i = 0u; i < 4u; i++) {
            uint32_t king = draws[i] & 63u, man = draws[i] >> 6 & 63u, foe = draws[i] >> 12 & 63u;
            uint64_t occ = (1ull << king) | (1ull << man) | (1ull << foe);
            if (king == man || king == foe || man == foe || (CHESS_KING_ATTACKS[king] >> foe & 1u) || (chess_reach(piece, man, occ) >> foe & 1u)) continue;
            __builtin_memset(p->board, 0, sizeof p->board);
            p->board[king] = CHESS_KING;
            p->board[man] = (uint8_t)piece;
            p->board[foe] = CHESS_KING | 8u;
            p->side = 0u;
            p->castle = 0u;
            p->ep = 0u;
            p->clock = 0u;
            chess_rebuild(p);
            return;
        }
    }
}

static uint32_t chess_draw(uint32_t seed, const uint32_t *words, uint32_t tag, uint32_t bound) {
    uint32_t floor = (0u - bound) % bound;
    for (uint32_t block = 0u;; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, words[CHESS_GAME], words[CHESS_PLY], tag, block);
        uint32_t draws[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t i = 0u; i < 4u; i++) {
            uint64_t scaled = (uint64_t)draws[i] * bound;
            if ((uint32_t)scaled >= floor) return (uint32_t)(scaled >> 32);
        }
    }
}

static const uint8_t *chess_tables(const float *shared, uint32_t piece) { return (const uint8_t *)shared + (piece == CHESS_QUEEN ? 0u : 2u * CHESS_ENTRIES); }

static uint32_t chess_bare(const chessPos *p, uint32_t *strong) {
    uint64_t occ = p->bb[0] | p->bb[7];
    if (__builtin_popcountll(occ) != 3) return 0u;
    uint64_t man = occ & ~p->bb[CHESS_KING];
    if (!(man & (p->bb[CHESS_QUEEN] | p->bb[CHESS_ROOK]))) return 0u;
    *strong = (man & p->bb[7]) ? 1u : 0u;
    return p->board[chess_low(man)] & 7u;
}

static uint32_t chess_rank(const uint8_t *tables, const chessPos *p, uint32_t strong, uint32_t move) {
    uint32_t from = move & 63u, to = move >> 6 & 63u;
    uint64_t kings = p->bb[CHESS_KING], man = (p->bb[0] | p->bb[7]) & ~kings;
    uint32_t king = chess_low(kings & chess_side(p, strong)), piece = chess_low(man), foe = chess_low(kings & chess_side(p, strong ^ 1u));
    if (p->side == strong) {
        king = from == king ? to : king;
        piece = from == piece ? to : piece;
        uint8_t v = tables[CHESS_ENTRIES + (king | piece << 6 | foe << 12)];
        return v & CHESS_LOST ? v & 63u : 255u;
    }
    if (to == piece) return 0u;
    uint8_t v = tables[king | piece << 6 | to << 12];
    return v ? 64u - v : 0u;
}

static uint32_t chess_probe(const float *shared, const chessPos *p, const uint32_t *list, uint32_t count, uint32_t *pick) {
    uint32_t strong, piece = chess_bare(p, &strong);
    if (!piece) return 0u;
    const uint8_t *tables = chess_tables(shared, piece);
    uint32_t best = 256u;
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t r = chess_rank(tables, p, strong, list[i]);
        if (r < best) { best = r; *pick = list[i]; }
    }
    return 1u;
}

static uint32_t chess_defend(const chessLane *l, const uint32_t *words, uint32_t seed, const float *shared) {
    uint32_t strong, piece = chess_bare(&l->pos, &strong), best = 256u, ties = 0u, tied[8] = { 0u };
    const uint8_t *tables = chess_tables(shared, piece);
    if (!piece || l->count > 8u) __builtin_trap();
    for (uint32_t i = 0u; i < l->count; i++) {
        uint32_t r = chess_rank(tables, &l->pos, strong, l->list[i]);
        if (r < best) { best = r; ties = 0u; }
        if (r == best) tied[ties++] = l->list[i];
    }
    return tied[ties == 1u ? 0u : chess_draw(seed, words, CHESS_TIE, ties)];
}

static void chess_order(uint32_t *list, uint32_t count) {
    for (uint32_t i = 1u; i < count; i++) {
        uint32_t m = list[i], j = i;
        for (; j > 0u && (list[j - 1u] >> 24) < (m >> 24); j--) list[j] = list[j - 1u];
        list[j] = m;
    }
}

static void chess_start(chessSearch *s, uint32_t count, uint32_t budget, uint32_t horizon, uint32_t ceiling, uint32_t quiescence, uint32_t careful) {
    chessFrame *root = &s->frame[0];
    s->sp = s->nodes = s->done = 0u;
    s->careful = careful;
    s->floor = chess_eval(&root->pos) - CHESS_MARGIN;
    s->budget = budget;
    s->horizon = horizon;
    s->ceiling = ceiling;
    s->quiescence = quiescence;
    s->choice = s->leader = root->list[0];
    root->count = count;
    root->next = 0u;
    root->alpha = root->best = -CHESS_INFINITY;
    root->beta = CHESS_INFINITY;
    root->depth = horizon;
}

static void chess_expand(chessSearch *s, chessFrame *f) {
    const chessPos *p = &f->pos;
    uint32_t captures = 0u, last = s->sp >= s->horizon + s->quiescence;
    uint64_t checkers = f->depth ? 1u : s->careful ? chess_attackers(p, chess_low(p->bb[CHESS_KING] & chess_side(p, p->side)), p->side ^ 1u, p->bb[0] | p->bb[7]) : 0u;
    f->next = 0u;
    if (checkers) {
        f->count = chess_generate(p, f->list, 1u, &captures, &checkers);
        if (!f->count) f->best = checkers ? (int32_t)s->sp - CHESS_MATE : 0;
        else if (!f->depth && last) {
            f->best = chess_eval(p);
            f->count = 0u;
        }
    } else {
        int32_t stand = chess_eval(p);
        f->best = stand;
        f->alpha = stand > f->alpha ? stand : f->alpha;
        f->count = stand < f->beta && !last ? chess_generate(p, f->list, 0u, &captures, &checkers) : 0u;
    }
    chess_order(f->list, captures);
}

static inline void chess_backup(chessSearch *s, chessFrame *f, int32_t value, uint32_t move) {
    if ((move & CHESS_DRAWN) && value > -CHESS_CONTEMPT) value = -CHESS_CONTEMPT;
    if (value > f->best) {
        f->best = value;
        if (f == s->frame) s->leader = move;
    }
    if (value > f->alpha) f->alpha = value;
}

static void chess_iterate(chessSearch *s) {
    chessFrame *root = s->frame;
    s->choice = s->leader;
    if (s->horizon >= s->ceiling || s->nodes >= s->budget || root->count < 2u || root->best >= CHESS_MATE - 1000 || root->best <= 1000 - CHESS_MATE) {
        s->done = 1u;
        return;
    }
    uint32_t i = 0u;
    while (root->list[i] != s->leader) i++;
    for (; i > 0u; i--) root->list[i] = root->list[i - 1u];
    root->list[0] = s->leader;
    root->next = 0u;
    root->alpha = root->best = -CHESS_INFINITY;
    root->depth = ++s->horizon;
}

static uint32_t chess_think(chessSearch *s, uint32_t quota) {
    while (!s->done) {
        chessFrame *f = &s->frame[s->sp];
        if (f->count == CHESS_UNGENERATED) {
            if (!quota) return 0u;
            quota--;
            chess_expand(s, f);
            continue;
        }
        if (f->next >= f->count || f->alpha >= f->beta) {
            if (!s->sp) {
                chess_iterate(s);
                continue;
            }
            s->sp--;
            chess_backup(s, f - 1, -f->best, f->move);
            continue;
        }
        if (s->nodes >= s->budget && (!s->careful || s->frame[0].best >= s->floor || s->horizon > CHESS_OPENING)) {
            if (s->frame[0].best > -CHESS_INFINITY) s->choice = s->leader;
            s->done = 1u;
            continue;
        }
        uint32_t move = f->list[f->next];
        if (!quota) return 0u;
        quota--;
        f->next++;
        s->nodes++;
        chessFrame *c = f + 1;
        c->pos = f->pos;
        chess_make(&c->pos, move);
        c->alpha = -f->beta;
        c->beta = -f->alpha;
        c->best = -CHESS_INFINITY;
        c->count = CHESS_UNGENERATED;
        c->depth = f->depth ? f->depth - 1u : 0u;
        c->move = move;
        s->sp++;
    }
    return 1u;
}

static void chess_prepare(chessLane *l, const uint32_t *words, const chessKnobs *k, uint32_t seed) {
    chessSearch *s = &l->search;
    chessFrame *root = s->frame;
    uint32_t n = l->count;
    root->pos = l->pos;
    __builtin_memcpy(root->list, l->list, n * sizeof(uint32_t));
    u4 r = { 0u, 0u, 0u, 0u };
    for (uint32_t i = n - 1u; i > 0u; i--) {
        if (i == n - 1u || (i & 3u) == 3u) r = philox4x32(seed, ENV_RNG_STREAM, words[CHESS_GAME], words[CHESS_PLY], CHESS_SHUFFLE, i >> 2);
        uint32_t draws[4] = { r.x, r.y, r.z, r.w }, j = (uint32_t)(((uint64_t)draws[i & 3u] * (i + 1u)) >> 32), m = root->list[i];
        root->list[i] = root->list[j];
        root->list[j] = m;
    }
    chess_order(root->list, n);
    chess_start(s, n, k->nodes, k->depth, k->depth, k->quiescence, 0u);
}

static void chess_begin(chessLane *l, uint32_t *words, const chessKnobs *k, uint32_t seed) {
    chessPos *p = &l->pos;
    if (k->material == CHESS_FULL) chess_standard(p);
    else chess_scatter(p, k->material == CHESS_KQK ? CHESS_QUEEN : CHESS_ROOK, seed, words[CHESS_GAME]);
    uint32_t agent = chess_agent(k, words);
    words[CHESS_PLY] = 0u;
    words[CHESS_CURSOR] &= 63u;
    words[CHESS_THINK] = 0u;
    chess_paint(l, agent);
    chess_settle(l, words, agent, 1u);
    words[CHESS_THINK] = p->side != agent;
    l->movable = p->side == agent ? chess_movable(l) : 0u;
    l->targets = 0u;
    chess_view(l, words, agent);
}

static float chess_play(chessLane *l, uint32_t *words, const chessKnobs *k, uint32_t seed, uint32_t move, uint32_t *done) {
    chessPos *p = &l->pos;
    uint32_t mover = p->side, agent = chess_agent(k, words), castle = p->castle, to = move >> 6 & 63u;
    uint64_t before = p->bb[0] | p->bb[7];
    chess_make(p, move);
    chess_touch(l, agent, (before ^ (p->bb[0] | p->bb[7])) | (1ull << to));
    words[CHESS_PLY]++;
    words[CHESS_CURSOR] &= 63u;
    if (mover != agent) words[CHESS_LAST] = (words[CHESS_LAST] & ~0x7F7Fu) | ((move & 63u) + 1u) | (to + 1u) << 8;
    uint32_t outcome = chess_settle(l, words, agent, (move >> 12 & 7u) == CHESS_PAWN || (move >> 15 & 7u) || castle != p->castle);
    if (outcome) {
        float reward = outcome == CHESS_CHECKMATE ? (mover == agent ? 1.0f : -1.0f) : 0.0f;
        words[CHESS_LAST] = outcome << 20 | (reward > 0.0f ? 1u : reward < 0.0f ? 2u : 3u) << 24;
        words[CHESS_GAME]++;
        *done = 1u;
        chess_begin(l, words, k, seed);
        return reward;
    }
    words[CHESS_THINK] = p->side != agent;
    l->movable = p->side == agent ? chess_movable(l) : 0u;
    l->targets = 0u;
    chess_view(l, words, agent);
    return 0.0f;
}

static float chess_reply(chessLane *l, uint32_t *words, const chessKnobs *k, uint32_t seed, const float *shared, uint32_t *done) {
    if (k->opponent == CHESS_TABLE) return chess_play(l, words, k, seed, chess_defend(l, words, seed, shared), done);
    if (k->opponent == CHESS_RANDOM) return chess_play(l, words, k, seed, l->list[chess_draw(seed, words, CHESS_PICK, l->count)], done);
    uint32_t t = words[CHESS_THINK];
    if (t == 1u) chess_prepare(l, words, k, seed);
    else if (chess_think(&l->search, 1u)) return chess_play(l, words, k, seed, l->search.choice, done);
    words[CHESS_THINK] = t + 1u;
    return 0.0f;
}

static float chess_act(chessLane *l, uint32_t *words, const chessKnobs *k, uint32_t seed, uint32_t action, uint32_t *done) {
    uint32_t agent = chess_agent(k, words), cw = words[CHESS_CURSOR], cursor = cw & 63u, sel = cw >> 8 & 127u, promo = cw >> 16 & 127u, cand = cw >> 24 & 3u;
    if (!promo && action < CHESS_SELECT) {
        int dx, dy;
        gridDelta(action, &dx, &dy);
        int file = (int)(cursor & 7u) + dx, rank = (int)(cursor >> 3) - dy;
        if (file < 0 || file > 7 || rank < 0 || rank > 7) return 0.0f;
        cw = (cw & ~63u) | (uint32_t)(rank * 8 + file);
        words[CHESS_CURSOR] = cw;
        chess_frame(l, cw, agent);
        return 0.0f;
    }
    if (action == CHESS_CANCEL && !promo && !sel) return 0.0f;
    if (promo) {
        if (action == CHESS_SELECT) return chess_play(l, words, k, seed, chess_find(l, sel - 1u, promo - 1u, CHESS_PROMOTE[cand]), done);
        if (action == CHESS_LEFT) cand = (cand + 3u) & 3u;
        if (action == CHESS_RIGHT) cand = (cand + 1u) & 3u;
        if (action == CHESS_CANCEL) promo = cand = 0u;
    } else if (action == CHESS_CANCEL) sel = 0u;
    else {
        uint32_t here = cursor ^ (agent ? 63u : 0u);
        if (sel && (l->targets >> here & 1u)) {
            uint32_t move = chess_find(l, sel - 1u, here, 0u);
            if (!(move >> 18 & 7u)) return chess_play(l, words, k, seed, move, done);
            promo = here + 1u;
            cand = 0u;
        } else if ((l->movable >> here & 1u) && sel != here + 1u) {
            sel = here + 1u;
            l->targets = chess_targets(l, here);
        } else return 0.0f;
    }
    words[CHESS_CURSOR] = cursor | sel << 8 | promo << 16 | cand << 24;
    chess_view(l, words, agent);
    return 0.0f;
}

static uint32_t chess_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)CHESS_ACTIONS ? (uint32_t)c : 0u;
}

void chess_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    chessLane *l = chess_lane(extra);
    for (uint32_t i = 0u; i < CHESS_WORDS; i++) words[i] = 0u;
    words[CHESS_CURSOR] = 4u;
    l->seen = l->twice = 0u;
    chess_oracleOf(l)->game = 0xFFFFFFFFu;
    chess_begin(l, words, &CHESS_KNOBS[instance], seed);
}

static inline __attribute__((always_inline)) float chess_run(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    const chessKnobs *k = &CHESS_KNOBS[instance];
    chessLane *l = chess_lane(extra);
    uint32_t t = words[CHESS_THINK];
    float reward = 0.0f;
    *done = 0u;
    if (t && k->opponent == CHESS_SEARCH && (t - 1u) % k->period) words[CHESS_THINK] = t + 1u;
    else reward = t ? chess_reply(l, words, k, seed, shared, done) : chess_act(l, words, k, seed, chess_action(ctrl), done);
    __builtin_memcpy(obs, l->obs, CHESS_OBS * sizeof(float));
    return reward;
}

float chess_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step;
    return instance == 3u ? chess_run(3u, seed, ctrl, words, extra, obs, done, shared) : chess_run(instance, seed, ctrl, words, extra, obs, done, shared);
}

void chess_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)words; (void)step; (void)shared;
    __builtin_memcpy(obs, chess_peek(extra)->obs, CHESS_OBS * sizeof(float));
}

static void chess_avoid(const chessLane *l, uint32_t *list, uint32_t count) {
    const uint32_t *now = l->record[l->seen - 1u];
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t m = list[i], from = m & 63u, to = m >> 6 & 63u;
        if (!chess_quietMove(m) || (chess_castleLoss(from) & l->pos.castle)) continue;
        uint32_t next[CHESS_RECORD];
        __builtin_memcpy(next, now, sizeof next);
        next[from >> 3] &= ~(15u << 4u * (from & 7u));
        next[to >> 3] |= (uint32_t)l->pos.board[from] << 4u * (to & 7u);
        next[8] = ((now[8] & 31u) ^ 1u);
        if (chess_occurrences(l, next, chess_hash(next))) {
            list[i] |= CHESS_DRAWN;
            continue;
        }
        if (!l->twice) continue;
        chessPos after = l->pos;
        uint32_t replies[CHESS_CAPACITY], captures;
        uint64_t checkers;
        chess_make(&after, m);
        if (chess_reachable(l, next, replies, chess_generate(&after, replies, 1u, &captures, &checkers))) list[i] |= CHESS_DRAWN;
    }
}

static uint32_t chess_decide(chessLane *l, chessOracle *o, const chessKnobs *k, const float *shared) {
    uint32_t pick = l->list[0];
    if (chess_probe(shared, &l->pos, l->list, l->count, &pick)) return pick;
    chessSearch *s = &o->search;
    chessFrame *root = s->frame;
    root->pos = l->pos;
    __builtin_memcpy(root->list, l->list, l->count * sizeof(uint32_t));
    chess_order(root->list, l->captures);
    chess_avoid(l, root->list, l->count);
    chess_start(s, l->count, k->oracleNodes, CHESS_OPENING, CHESS_CEILING, CHESS_QUIESCENCE, 1u);
    chess_think(s, 0xFFFFFFFFu);
    return s->choice & ~(uint32_t)CHESS_DRAWN;
}

static uint32_t chess_steer(const uint32_t *words, uint32_t move, uint32_t agent) {
    uint32_t cw = words[CHESS_CURSOR], cursor = cw & 63u, sel = cw >> 8 & 127u, promo = cw >> 16 & 127u, cand = cw >> 24 & 3u;
    uint32_t from = move & 63u, to = move >> 6 & 63u, piece = move >> 18 & 7u;
    if (promo) {
        if (promo != to + 1u || sel != from + 1u) return CHESS_CANCEL;
        uint32_t want = 0u;
        while (CHESS_PROMOTE[want] != piece) want++;
        if (want == cand) return CHESS_SELECT;
        return ((want - cand) & 3u) <= 2u ? CHESS_RIGHT : CHESS_LEFT;
    }
    uint32_t target = (sel == from + 1u ? to : from) ^ (agent ? 63u : 0u);
    if (cursor == target) return CHESS_SELECT;
    if ((target >> 3) != (cursor >> 3)) return (target >> 3) > (cursor >> 3) ? CHESS_UP : CHESS_DOWN;
    return (target & 7u) > (cursor & 7u) ? CHESS_RIGHT : CHESS_LEFT;
}

void chess_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs;
    if (words[CHESS_THINK]) {
        ctrl[0] = (float)CHESS_CANCEL;
        return;
    }
    const chessKnobs *k = &CHESS_KNOBS[instance];
    chessLane *l = chess_lane(extra);
    chessOracle *o = chess_oracleOf(l);
    if (o->game != words[CHESS_GAME] || o->ply != words[CHESS_PLY]) {
        o->move = chess_decide(l, o, k, shared);
        o->game = words[CHESS_GAME];
        o->ply = words[CHESS_PLY];
    }
    ctrl[0] = (float)chess_steer(words, o->move, chess_agent(k, words));
}

static void chess_retrograde(uint8_t *wtm, uint32_t piece) {
    uint8_t *btm = wtm + CHESS_ENTRIES;
    __builtin_memset(wtm, 0, 2u * CHESS_ENTRIES);
    for (uint32_t i = 0u; i < CHESS_ENTRIES; i++) {
        uint32_t king = i & 63u, man = i >> 6 & 63u, foe = i >> 12;
        uint64_t kb = 1ull << king, mb = 1ull << man, fb = 1ull << foe;
        if (king == man || king == foe || man == foe || (CHESS_KING_ATTACKS[king] & fb)) continue;
        uint64_t flights = CHESS_KING_ATTACKS[foe] & ~(CHESS_KING_ATTACKS[king] | chess_reach(piece, man, kb | mb));
        if (flights & mb) {
            btm[i] = CHESS_HELD;
            continue;
        }
        uint32_t n = (uint32_t)__builtin_popcountll(flights);
        btm[i] = (uint8_t)(n ? n : (chess_reach(piece, man, kb | mb | fb) & fb) ? CHESS_LOST : CHESS_HELD);
    }
    for (uint32_t d = 0u;; d++) {
        uint32_t found = 0u;
        for (uint32_t i = 0u; i < CHESS_ENTRIES; i++) {
            if (btm[i] != (CHESS_LOST | d)) continue;
            uint32_t king = i & 63u, man = i >> 6 & 63u, foe = i >> 12;
            uint64_t kb = 1ull << king, mb = 1ull << man, fb = 1ull << foe;
            for (uint64_t from = CHESS_KING_ATTACKS[king] & ~(mb | fb | CHESS_KING_ATTACKS[foe]); from; from &= from - 1u) {
                uint32_t s = chess_low(from), j = s | man << 6 | foe << 12;
                if ((chess_reach(piece, man, (1ull << s) | mb | fb) & fb) || wtm[j]) continue;
                wtm[j] = (uint8_t)(d + 1u);
                found = 1u;
            }
            for (uint64_t from = chess_reach(piece, man, kb | fb) & ~(kb | fb); from; from &= from - 1u) {
                uint32_t s = chess_low(from), j = king | s << 6 | foe << 12;
                if ((chess_reach(piece, s, kb | (1ull << s) | fb) & fb) || wtm[j]) continue;
                wtm[j] = (uint8_t)(d + 1u);
                found = 1u;
            }
        }
        if (!found) break;
        for (uint32_t i = 0u; i < CHESS_ENTRIES; i++) {
            if (wtm[i] != d + 1u) continue;
            uint32_t king = i & 63u, man = i >> 6 & 63u, foe = i >> 12;
            for (uint64_t from = CHESS_KING_ATTACKS[foe] & ~((1ull << king) | (1ull << man) | CHESS_KING_ATTACKS[king]); from; from &= from - 1u) {
                uint32_t j = king | man << 6 | chess_low(from) << 12, v = btm[j];
                if (v >= 1u && v <= 8u) btm[j] = (uint8_t)(v == 1u ? CHESS_LOST | (d + 1u) : v - 1u);
            }
        }
    }
    for (uint32_t i = 0u; i < CHESS_ENTRIES; i++) btm[i] = btm[i] & CHESS_LOST ? btm[i] : 0u;
}

uint64_t chess_shared(float *table) {
    if (table) {
        chess_retrograde((uint8_t *)table, CHESS_QUEEN);
        chess_retrograde((uint8_t *)table + 2u * CHESS_ENTRIES, CHESS_ROOK);
    }
    return CHESS_ENTRIES;
}
