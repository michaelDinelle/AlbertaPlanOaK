#include <string.h>
#include "../env.h"

#define POMMERMAN_BOMB_LIFE 9u
#define POMMERMAN_FLAME_LIFE 3u
#define POMMERMAN_START_AMMO 1u
#define POMMERMAN_START_BLAST 2u
#define POMMERMAN_MAX_AMMO 10u
#define POMMERMAN_SHUT_IN 4u
#define POMMERMAN_BREATHING 2u
#define POMMERMAN_HORIZON 14u
#define POMMERMAN_SIGHT 3u

typedef struct {
    uint32_t side, seats, rigid, wood, items, team, fog, league, tickLimit, collapseStart, collapseEvery, collapseRings, maxBlast;
    float flee, strike, loot, chase, dig, wander, woodBomb, enemyBomb;
} pommermanKnobs;

static const pommermanKnobs POMMERMAN_KNOBS[5] = {
    { 8u, 2u, 16u, 16u, 10u, 0u, 0u, 0u, 800u, 0u, 0u, 0u, 10u, 6.0f, 5.0f, 3.0f, 10.0f, 8.0f, 0.25f, 0.75f, 1.0f },
    { 11u, 4u, 36u, 36u, 20u, 0u, 0u, 0u, 800u, 0u, 0u, 0u, 10u, 6.0f, 5.0f, 3.0f, 10.0f, 8.0f, 0.25f, 0.75f, 1.0f },
    { 11u, 4u, 36u, 36u, 20u, 1u, 4u, 0u, 800u, 0u, 0u, 0u, 4u, 6.0f, 5.0f, 3.0f, 10.0f, 8.0f, 0.25f, 0.75f, 1.0f },
    { 11u, 4u, 36u, 36u, 20u, 0u, 0u, 0u, 800u, 500u, 75u, 4u, 10u, 6.0f, 5.0f, 3.0f, 10.0f, 8.0f, 0.25f, 0.75f, 1.0f },
    { 11u, 4u, 36u, 36u, 20u, 0u, 0u, 1u, 800u, 0u, 0u, 0u, 10u, 6.0f, 5.0f, 3.0f, 10.0f, 8.0f, 0.25f, 0.75f, 1.0f }
};

static const float POMMERMAN_THETA_LOW[8] = { 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float POMMERMAN_THETA_HIGH[8] = { 10.0f, 9.0f, 6.0f, 12.0f, 10.0f, 0.5f, 1.0f, 1.0f };

enum { POM_STOP, POM_UP, POM_DOWN, POM_LEFT, POM_RIGHT, POM_BOMB, POM_ACTIONS };
enum { POM_EXTRA, POM_RANGE, POM_KICK, POM_KINDS };
enum { POM_SEATS = 4, POM_BOMBS = 32, POM_THETA = 8, POM_NONE = 255, POM_PAIRS = 64, POM_LAYERS = 16 };
enum { PW_TICK, PW_GAME, PW_SEAT, PW_ALIVE, PW_RESULT, PW_CELL, PW_AMMO = PW_CELL + POM_SEATS, PW_BLAST = PW_AMMO + POM_SEATS, PW_KICK = PW_BLAST + POM_SEATS, PW_ACT = PW_KICK + POM_SEATS, PW_WORDS = PW_ACT + POM_SEATS };
enum { POM_WON = 1u, POM_LOST, POM_TIED };
enum { POM_SPAN = 2u * POMMERMAN_SIGHT + 1u, POM_GRID = POM_SPAN * POM_SPAN, POM_OTHERS = POM_GRID + 4 };
enum { POM_DRAW_BOTS, POM_DRAW_BOARD, POM_DRAW_ITEMS, POM_DRAW_SEAT };

typedef unsigned __int128 pomBits;

typedef struct { pomBits board, notFirst, notLast, column; uint32_t side; } pomGeo;

typedef struct { uint8_t cell, life, blast, owner, dir, pad[3]; } pomBomb;

typedef struct { uint32_t flee, strike, loot, chase, dig, pad; uint64_t wander, woodBomb, enemyBomb; } pomStyle;

typedef struct {
    float theta[LEAGUE_MEMBERS * POM_THETA], winRate[LEAGUE_MEMBERS];
    uint32_t member[POM_SEATS];
    pomStyle style[LEAGUE_MEMBERS];
} pomLeague;

typedef struct { pomBits flame[POMMERMAN_HORIZON + 1u], open[POMMERMAN_HORIZON + 1u], safe[POMMERMAN_HORIZON + 1u], roam; } pomForecast;

typedef struct {
    pomBits rigid, wood, bomb, flame[POMMERMAN_FLAME_LIFE], item[POM_KINDS], hidden[POM_KINDS];
    pomBits reach[POM_BOMBS];
    pomBomb bombs[POM_BOMBS];
    uint32_t bombCount, death[POM_SEATS];
    uint8_t bombAt[128];
} pomBoard;

typedef struct {
    pomBoard board, belief, sim;
    pomBits laneWood;
    uint32_t pairs, lean;
    uint8_t pair[POM_PAIRS];
    pomLeague league;
    pomForecast plain, probe;
} pomWorld;

static pomWorld *pom_world(uint32_t *extra) { return (pomWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const pomWorld *pom_read(const uint32_t *extra) { return (const pomWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }

uint32_t pommerman_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)(sizeof(pomWorld) / sizeof(uint32_t)) + 4u;
}

static inline pomBits pom_wide(uint64_t high, uint64_t low) { return (pomBits)high << 64 | (pomBits)low; }

static inline __attribute__((always_inline)) pomGeo pom_geo(uint32_t side) {
    pomGeo g;
    g.side = side;
    if (side == 8u) {
        g.board = pom_wide(0u, 0xffffffffffffffffull);
        g.notFirst = pom_wide(0u, 0xfefefefefefefefeull);
        g.notLast = pom_wide(0u, 0x7f7f7f7f7f7f7f7full);
        g.column = pom_wide(0u, 0x0101010101010101ull);
    } else {
        g.board = pom_wide(0x1ffffffffffffffull, 0xffffffffffffffffull);
        g.notFirst = pom_wide(0x1ffbff7feffdffbull, 0xff7feffdffbff7feull);
        g.notLast = pom_wide(0xffdffbff7feffdull, 0xffbff7feffdffbffull);
        g.column = pom_wide(0x400801002004ull, 0x80100200400801ull);
    }
    return g;
}

static inline pomBits pom_bit(uint32_t cell) { return (pomBits)1u << cell; }

static inline uint32_t pom_low(pomBits m) {
    uint64_t low = (uint64_t)m;
    return low ? (uint32_t)__builtin_ctzll(low) : 64u + (uint32_t)__builtin_ctzll((uint64_t)(m >> 64));
}

static inline uint32_t pom_high(pomBits m) {
    uint64_t high = (uint64_t)(m >> 64);
    return high ? 127u - (uint32_t)__builtin_clzll(high) : 63u - (uint32_t)__builtin_clzll((uint64_t)m);
}

static inline uint32_t pom_end(pomBits m, uint32_t word, uint32_t bit) { return word >> bit & 1u ? pom_high(m) : pom_low(m); }

static inline uint32_t pom_count(pomBits m) { return (uint32_t)__builtin_popcountll((uint64_t)m) + (uint32_t)__builtin_popcountll((uint64_t)(m >> 64)); }

static inline pomBits pom_spread(const pomGeo *g, pomBits x) {
    return x | ((x << 1) & g->notFirst) | ((x >> 1) & g->notLast) | (x << g->side) | (x >> g->side);
}

static inline uint32_t pom_below(uint32_t word, uint32_t n) { return (uint32_t)(((uint64_t)word * n) >> 32); }

static inline uint64_t pom_odds(float p) { return p >= 1.0f ? 1ull << 32 : (uint64_t)u01Bound(p); }

static inline pomStyle pom_style(const float *theta) {
    return (pomStyle){ (uint32_t)theta[0], (uint32_t)theta[1], (uint32_t)theta[2], (uint32_t)theta[3], (uint32_t)theta[4], 0u, pom_odds(theta[5]), pom_odds(theta[6]), pom_odds(theta[7]) };
}

static inline uint32_t pom_step(uint32_t side, uint32_t cell, uint32_t action) {
    int dx, dy;
    gridDelta(action - 1u, &dx, &dy);
    int x = (int)(cell % side) + dx, y = (int)(cell / side) + dy;
    return x < 0 || y < 0 || x >= (int)side || y >= (int)side ? POM_NONE : (uint32_t)(y * (int)side + x);
}

static inline uint32_t pom_toward(uint32_t side, uint32_t cell, uint32_t next) {
    return next + side == cell ? POM_UP : next == cell + side ? POM_DOWN : next + 1u == cell ? POM_LEFT : POM_RIGHT;
}

static inline uint32_t pom_pick(pomBits mask, uint32_t word) {
    for (uint32_t skip = pom_below(word, pom_count(mask)); skip > 0u; skip--) mask &= mask - 1u;
    return pom_low(mask);
}

static inline __attribute__((always_inline)) pomBits pom_cross(const pomGeo *g, uint32_t cell, uint32_t reach, pomBits stop, pomBits burn) {
    pomBits origin = pom_bit(cell), open = g->board & ~stop, pass = ~burn, cross = origin;
    pomBits east = (origin << 1) & g->notFirst & open, west = (origin >> 1) & g->notLast & open;
    pomBits south = (origin << g->side) & open, north = (origin >> g->side) & open;
    for (uint32_t n = 1u; n <= reach && (east | west | south | north); n++) {
        cross |= east | west | south | north;
        east = ((east & pass) << 1) & g->notFirst & open;
        west = ((west & pass) >> 1) & g->notLast & open;
        south = ((south & pass) << g->side) & open;
        north = ((north & pass) >> g->side) & open;
    }
    return cross;
}

static pomBits pom_window(const pomGeo *g, uint32_t cell, uint32_t radius) {
    uint32_t side = g->side, r = cell / side, c = cell % side;
    uint32_t top = r > radius ? r - radius : 0u, bottom = r + radius < side ? r + radius : side - 1u;
    uint32_t left = c > radius ? c - radius : 0u, right = c + radius < side ? c + radius : side - 1u;
    pomBits rows = (pom_bit((bottom + 1u) * side) - 1u) & ~(pom_bit(top * side) - 1u);
    pomBits columns = (g->column * (pomBits)((1u << (right - left + 1u)) - 1u)) << left;
    return rows & columns & g->board;
}

static pomBits pom_ring(const pomGeo *g, uint32_t ring) {
    uint32_t side = g->side;
    pomBits ring_ = 0;
    for (uint32_t i = ring; i < side - ring; i++) {
        ring_ |= pom_bit(ring * side + i) | pom_bit((side - 1u - ring) * side + i);
        ring_ |= pom_bit(i * side + ring) | pom_bit(i * side + side - 1u - ring);
    }
    return ring_;
}

static uint32_t pom_corner(uint32_t side, uint32_t corner) {
    uint32_t low = 1u, high = side - 2u;
    return corner == 0u ? low * side + low : corner == 1u ? high * side + low : corner == 2u ? high * side + high : low * side + high;
}

static uint32_t pom_seatCorner(const pommermanKnobs *k, uint32_t seat) { return k->seats == 2u ? 2u * seat : seat; }

static uint32_t pom_partner(const pommermanKnobs *k, uint32_t seat) { return k->team ? (seat + 2u) & 3u : POM_NONE; }

static uint32_t pom_rival(const pommermanKnobs *k, uint32_t seat, uint32_t other) { return other != seat && other != pom_partner(k, seat); }

static void pom_lanes(const pommermanKnobs *k, pomBits *laneWood, pomBits *reserved) {
    uint32_t side = k->side, low = 1u, seated = 0u;
    *laneWood = 0;
    *reserved = 0;
    for (uint32_t s = 0u; s < k->seats; s++) {
        uint32_t corner = pom_seatCorner(k, s), cell = pom_corner(side, corner), r = cell / side, c = cell % side;
        seated |= 1u << corner;
        *reserved |= pom_bit(cell);
        for (uint32_t d = 1u; d <= POMMERMAN_BREATHING; d++)
            *reserved |= pom_bit(r * side + (c == low ? c + d : c - d)) | pom_bit((r == low ? r + d : r - d) * side + c);
    }
    for (uint32_t e = 0u; e < 4u; e++) {
        uint32_t a = e == 2u ? 1u : e == 1u ? 3u : 0u, b = e == 3u ? 1u : e == 0u ? 3u : 2u;
        if (!(seated >> a & seated >> b & 1u)) continue;
        uint32_t from = pom_corner(side, a), to = pom_corner(side, b), stride = to - from >= side ? side : 1u;
        for (uint32_t cell = from + stride; cell < to; cell += stride)
            if (!(*reserved & pom_bit(cell))) *laneWood |= pom_bit(cell);
    }
    *reserved |= *laneWood;
}

static pomBits pom_flood(const pomGeo *g, pomBits from, pomBits through) {
    pomBits reached = from;
    for (;;) {
        pomBits next = pom_spread(g, reached) & through;
        if (next == reached) return reached;
        reached = next;
    }
}

static void pom_words(uint32_t *out, uint32_t count, uint32_t seed, uint32_t game, uint32_t block, uint32_t tag) {
    for (uint32_t i = 0u; i < count; i += 16u) {
        u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)game, (lane4)(block + i / 16u), (lane4){ 0u, 1u, 2u, 3u }, (lane4)tag);
        for (uint32_t j = 0u; j < 4u; j++) {
            out[i + j] = r.x[j];
            out[i + 4u + j] = r.y[j];
            out[i + 8u + j] = r.z[j];
            out[i + 12u + j] = r.w[j];
        }
    }
}

static void pom_layout(const pommermanKnobs *k, pomWorld *w) {
    uint32_t side = k->side;
    pomBits reserved;
    pom_lanes(k, &w->laneWood, &reserved);
    w->pairs = 0u;
    for (uint32_t r = 0u; r < side; r++)
        for (uint32_t c = r + 1u; c < side; c++)
            if (!(reserved & pom_bit(r * side + c))) w->pair[w->pairs++] = (uint8_t)(r * side + c);
}

static void pom_generate(const pommermanKnobs *k, const pomGeo *g, pomWorld *world, uint32_t seed, uint32_t game) {
    pomBoard *w = &world->board;
    uint32_t side = k->side, pairs = world->pairs, rigidPairs = k->rigid / 2u, draw[64];
    pomBits laneWood = world->laneWood;
    uint32_t woodPairs = (k->wood - pom_count(laneWood)) / 2u, chosen = rigidPairs + woodPairs;
    pomBits home = pom_bit(pom_corner(side, pom_seatCorner(k, k->seats - 1u))), seated = 0;
    for (uint32_t s = 0u; s < k->seats; s++) seated |= pom_bit(pom_corner(side, pom_seatCorner(k, s)));
    for (uint32_t attempt = 0u;; attempt++) {
        uint8_t order[POM_PAIRS];
        memcpy(order, world->pair, pairs);
        pom_words(draw, chosen, seed, game, 16u * attempt, POM_DRAW_BOARD);
        pomBits rigid = 0, wood = laneWood;
        for (uint32_t i = 0u; i < chosen; i++) {
            uint32_t j = i + pom_below(draw[i], pairs - i), cell = order[j];
            order[j] = order[i];
            order[i] = (uint8_t)cell;
            pomBits both = pom_bit(cell) | pom_bit(cell % side * side + cell / side);
            if (i < rigidPairs) rigid |= both;
            else wood |= both;
        }
        pomBits open = g->board & ~rigid;
        if (pom_count(open & ~wood & ~seated & ~pom_flood(g, home, open)) > POMMERMAN_SHUT_IN) continue;
        w->rigid = rigid;
        w->wood = wood;
        break;
    }
    uint32_t woods[64], count = 0u;
    for (pomBits m = w->wood; m; m &= m - 1u) woods[count++] = pom_low(m);
    pom_words(draw, 2u * k->items, seed, game, 0u, POM_DRAW_ITEMS);
    for (uint32_t t = 0u; t < POM_KINDS; t++) w->hidden[t] = w->item[t] = 0;
    for (uint32_t i = 0u; i < k->items; i++) {
        uint32_t j = i + pom_below(draw[i], count - i), cell = woods[j];
        woods[j] = woods[i];
        woods[i] = cell;
        w->hidden[pom_below(draw[k->items + i], POM_KINDS)] |= pom_bit(cell);
    }
}

static void pom_leagueSeat(const pommermanKnobs *k, pomWorld *w, uint32_t seed, uint32_t game, uint32_t me) {
    float busy[LEAGUE_MEMBERS];
    memcpy(busy, w->league.winRate, sizeof busy);
    u4 r = philox4x32(seed, LEAGUE_RNG_STREAM, game, 0x5345u, 0u, 0u);
    uint32_t words4[4] = { r.x, r.y, r.z, r.w }, used = 0u, n = 0u;
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (s == me) { w->league.member[s] = POM_NONE; continue; }
        uint32_t m = leaguePick(busy, words4[n++]);
        while (used >> m & 1u) m = (m + 1u) % LEAGUE_MEMBERS;
        used |= 1u << m;
        busy[m] = 1.0f;
        w->league.member[s] = m;
    }
}

static void pom_begin(const pommermanKnobs *k, const pomGeo *g, pomWorld *w, uint32_t *words, uint32_t seed, uint32_t game) {
    pom_generate(k, g, w, seed, game);
    w->board.bomb = w->board.flame[0] = w->board.flame[1] = w->board.flame[2] = 0;
    w->board.bombCount = 0u;
    memset(w->board.bombAt, 0, sizeof w->board.bombAt);
    u4 start = philox4x32(seed, ENV_RNG_STREAM, game, 0u, 0u, POM_DRAW_SEAT);
    uint32_t me = pom_below(start.x, k->seats);
    w->lean = start.y;
    words[PW_TICK] = 0u;
    words[PW_GAME] = game;
    words[PW_SEAT] = me;
    words[PW_ALIVE] = (1u << k->seats) - 1u;
    for (uint32_t s = 0u; s < POM_SEATS; s++) {
        uint32_t cell = s < k->seats ? pom_corner(k->side, pom_seatCorner(k, s)) : 0u;
        words[PW_CELL + s] = cell;
        words[PW_AMMO + s] = s < k->seats ? POMMERMAN_START_AMMO : 0u;
        words[PW_BLAST + s] = s < k->seats ? POMMERMAN_START_BLAST : 0u;
        words[PW_KICK + s] = 0u;
        words[PW_ACT + s] = POM_STOP;
        w->board.death[s] = 0xFFFFFFFFu;
    }
    if (k->league) pom_leagueSeat(k, w, seed, game, me);
}

static void pom_reachAll(const pomGeo *g, pomBoard *w, pomBits changed) {
    for (uint32_t b = 0u; b < w->bombCount; b++)
        if (w->reach[b] & changed) w->reach[b] = pom_cross(g, w->bombs[b].cell, w->bombs[b].blast - 1u, w->rigid, w->wood);
}

static inline __attribute__((always_inline)) void pom_place(const pomGeo *g, pomBoard *w, uint32_t *words, uint32_t seat) {
    uint32_t cell = words[PW_CELL + seat];
    if (!words[PW_AMMO + seat] || (w->bomb & pom_bit(cell))) return;
    uint32_t b = w->bombCount++;
    w->bombs[b] = (pomBomb){ (uint8_t)cell, (uint8_t)(POMMERMAN_BOMB_LIFE + 1u), (uint8_t)words[PW_BLAST + seat], (uint8_t)seat, 0u, { 0u, 0u, 0u } };
    w->reach[b] = pom_cross(g, cell, words[PW_BLAST + seat] - 1u, w->rigid, w->wood);
    w->bombAt[cell] = (uint8_t)(b + 1u);
    w->bomb |= pom_bit(cell);
    words[PW_AMMO + seat]--;
}

typedef struct { uint8_t from, to, index; } pomMover;

static inline __attribute__((always_inline)) void pom_walk(const pomMover *m, uint32_t n, uint32_t *words) {
    uint32_t moving = 0u, clash = 0u, touch = 0u;
    for (uint32_t i = 0u; i < n; i++) moving |= (uint32_t)(m[i].to != m[i].from) << i;
    if (!moving) return;
    for (uint32_t i = 0u; i < n; i++)
        for (uint32_t j = 0u; j < i; j++) touch |= (m[i].to == m[j].to) | (m[i].to == m[j].from) | (m[j].to == m[i].from);
    if (!touch) {
        for (uint32_t i = 0u; i < n; i++) words[PW_CELL + m[i].index] = m[i].to;
        return;
    }
    for (uint32_t i = 0u; i < n; i++)
        for (uint32_t j = 0u; j < n; j++)
            if (i != j && (moving >> i & moving >> j & 1u) && m[i].to == m[j].to) clash |= 1u << i;
    moving &= ~clash;
    clash = 0u;
    for (uint32_t i = 0u; i < n; i++)
        for (uint32_t j = 0u; j < n; j++)
            if (i != j && (moving >> i & moving >> j & 1u) && m[i].to == m[j].from && m[j].to == m[i].from) clash |= 1u << i;
    moving &= ~clash;
    for (uint32_t changed = 1u; changed;) {
        changed = 0u;
        for (uint32_t i = 0u; i < n; i++) {
            if (!(moving >> i & 1u)) continue;
            for (uint32_t j = 0u; j < n; j++)
                if (!(moving >> j & 1u) && m[j].from == m[i].to) {
                    moving &= ~(1u << i);
                    changed = 1u;
                    break;
                }
        }
    }
    for (uint32_t i = 0u; i < n; i++)
        if (moving >> i & 1u) words[PW_CELL + m[i].index] = m[i].to;
}

static __attribute__((noinline)) void pom_shove(const pommermanKnobs *k, const pomGeo *g, pomBoard *w, uint32_t *words, const uint32_t *act) {
    uint32_t side = k->side, alive = words[PW_ALIVE], n = w->bombCount, seats = k->seats, later = 0u, shifted = 0u;
    uint8_t ac[128], bc[128], ap[POM_SEATS], at[POM_SEATS], bp[POM_BOMBS], bt[POM_BOMBS], kicker[POM_BOMBS], dest[POM_BOMBS];
    uint64_t delayed = 0u;
    pomBits walls = w->rigid | w->wood, blocked = walls | w->item[POM_EXTRA] | w->item[POM_RANGE] | w->item[POM_KICK];
    memset(ac, 0, sizeof ac);
    memset(bc, 0, sizeof bc);
    for (uint32_t s = 0u; s < seats; s++) {
        uint32_t from = words[PW_CELL + s], to = from;
        if ((alive >> s & 1u) && act[s] >= POM_UP && act[s] <= POM_RIGHT) {
            uint32_t t = pom_step(side, from, act[s]);
            if (t != POM_NONE && !(walls & pom_bit(t))) to = t;
        }
        ap[s] = (uint8_t)from;
        at[s] = (uint8_t)to;
    }
    for (uint32_t b = 0u; b < n; b++) {
        uint32_t from = w->bombs[b].cell, to = from;
        if (w->bombs[b].dir) {
            uint32_t t = pom_step(side, from, w->bombs[b].dir);
            if (t != POM_NONE && !(blocked & pom_bit(t))) to = t;
        }
        bp[b] = (uint8_t)from;
        bt[b] = (uint8_t)to;
        kicker[b] = 0u;
    }
    for (uint32_t x = 0u; x < seats; x++)
        for (uint32_t y = x + 1u; y < seats; y++)
            if ((alive >> x & alive >> y & 1u) && at[x] != ap[x] && at[x] == ap[y] && at[y] == ap[x]) {
                at[x] = ap[x];
                at[y] = ap[y];
            }
    for (uint32_t b = 0u; b < n; b++) {
        if (bt[b] == bp[b]) continue;
        uint32_t hit = 0u;
        for (uint32_t s = 0u; s < seats; s++) hit |= (alive >> s & 1u) && at[s] != ap[s] && at[s] == bp[b] && ap[s] == bt[b];
        for (uint32_t o = 0u; o < n; o++)
            if (o != b && bt[o] != bp[o] && bt[o] == bp[b] && bp[o] == bt[b]) {
                bt[o] = bp[o];
                hit = 1u;
            }
        if (hit) bt[b] = bp[b];
    }
    for (uint32_t s = 0u; s < seats; s++) ac[at[s]] += (uint8_t)(alive >> s & 1u);
    for (uint32_t b = 0u; b < n; b++) bc[bt[b]]++;
    for (uint32_t changed = 1u; changed;) {
        changed = 0u;
        for (uint32_t s = 0u; s < seats; s++)
            if ((alive >> s & 1u) && at[s] != ap[s] && (ac[at[s]] > 1u || bc[at[s]] > 1u)) {
                at[s] = ap[s];
                ac[ap[s]]++;
                changed = 1u;
            }
        for (uint32_t b = 0u; b < n; b++)
            if (bt[b] != bp[b] && (bc[bt[b]] > 1u || ac[bt[b]] > 1u)) {
                bt[b] = bp[b];
                bc[bp[b]]++;
                changed = 1u;
            }
    }
    for (uint32_t b = 0u; b < n; b++) {
        uint32_t t = bt[b], s = 0u;
        if (!ac[t]) continue;
        while (s < seats && !((alive >> s & 1u) && at[s] == t)) s++;
        if (s == seats) continue;
        if (t == ap[s]) {
            if (t != bp[b]) {
                dest[b] = bp[b];
                delayed |= 1ull << b;
            }
            continue;
        }
        uint32_t q = words[PW_KICK + s] ? pom_step(side, t, act[s]) : POM_NONE;
        if (q != POM_NONE && !ac[q] && !bc[q] && !(blocked & pom_bit(q))) {
            bc[t] = 0u;
            dest[b] = (uint8_t)q;
            kicker[b] = (uint8_t)(s + 1u);
            w->bombs[b].dir = (uint8_t)act[s];
        } else {
            dest[b] = bp[b];
            later |= 1u << s;
        }
        delayed |= 1ull << b;
    }
    for (uint64_t m = delayed; m; m &= m - 1u) {
        uint32_t b = (uint32_t)__builtin_ctzll(m);
        bt[b] = dest[b];
        bc[dest[b]]++;
    }
    for (uint32_t s = 0u; s < seats; s++)
        if (later >> s & 1u) {
            at[s] = ap[s];
            ac[ap[s]]++;
        }
    for (uint32_t changed = 1u; changed;) {
        changed = 0u;
        for (uint32_t s = 0u; s < seats; s++) {
            if (!(alive >> s & 1u) || at[s] == ap[s] || (ac[at[s]] < 2u && !bc[at[s]])) continue;
            for (uint32_t b = 0u; b < n; b++)
                if (kicker[b] == s + 1u) {
                    bt[b] = bp[b];
                    bc[bp[b]]++;
                    kicker[b] = 0u;
                }
            at[s] = ap[s];
            ac[ap[s]]++;
            changed = 1u;
        }
        for (uint32_t b = 0u; b < n; b++) {
            if ((bt[b] == bp[b] && !kicker[b]) || (bc[bt[b]] < 2u && !ac[bt[b]])) continue;
            bt[b] = bp[b];
            bc[bp[b]]++;
            if (kicker[b]) {
                uint32_t s = kicker[b] - 1u;
                at[s] = ap[s];
                ac[ap[s]]++;
                kicker[b] = 0u;
            }
            changed = 1u;
        }
    }
    for (uint32_t s = 0u; s < seats; s++)
        if (alive >> s & 1u) words[PW_CELL + s] = at[s];
    for (uint32_t b = 0u; b < n; b++) {
        if (bt[b] == bp[b] && !kicker[b]) w->bombs[b].dir = 0u;
        else if (bt[b] != bp[b]) {
            w->bombs[b].cell = bt[b];
            shifted |= 1u << b;
        }
    }
    if (!shifted) return;
    w->bomb = 0;
    memset(w->bombAt, 0, sizeof w->bombAt);
    for (uint32_t b = 0u; b < n; b++) {
        w->bomb |= pom_bit(w->bombs[b].cell);
        w->bombAt[w->bombs[b].cell] = (uint8_t)(b + 1u);
        if (shifted >> b & 1u) w->reach[b] = pom_cross(g, w->bombs[b].cell, w->bombs[b].blast - 1u, w->rigid, w->wood);
    }
}

static inline __attribute__((always_inline)) void pom_move(const pommermanKnobs *k, const pomGeo *g, pomBoard *w, uint32_t *words, const uint32_t *act) {
    uint32_t side = k->side, alive = words[PW_ALIVE], count = 0u, push = 0u;
    pomMover m[POM_SEATS];
    pomBits walls = w->rigid | w->wood;
    for (uint32_t b = 0u; b < w->bombCount; b++) push |= w->bombs[b].dir;
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (!(alive >> s & 1u)) continue;
        uint32_t from = words[PW_CELL + s], to = from, a = act[s];
        if (a >= POM_UP && a <= POM_RIGHT) {
            uint32_t t = pom_step(side, from, a);
            if (t != POM_NONE && !(walls & pom_bit(t))) {
                if (!(w->bomb & pom_bit(t))) to = t;
                else push |= words[PW_KICK + s];
            }
        }
        m[count++] = (pomMover){ (uint8_t)from, (uint8_t)to, (uint8_t)s };
    }
    if (push) pom_shove(k, g, w, words, act);
    else pom_walk(m, count, words);
}

static pomBits pom_bodies(const pommermanKnobs *k, const uint32_t *words) {
    pomBits bodies = 0;
    for (uint32_t s = 0u; s < k->seats; s++)
        if (words[PW_ALIVE] >> s & 1u) bodies |= pom_bit(words[PW_CELL + s]);
    return bodies;
}

static void pom_explode(const pomGeo *g, pomBoard *w, uint32_t *words, uint32_t fire) {
    pomBits burst = 0;
    uint32_t spent = 0u;
    while (fire) {
        uint32_t b = (uint32_t)__builtin_ctz(fire);
        fire &= fire - 1u;
        spent |= 1u << b;
        uint32_t *ammo = &words[PW_AMMO + w->bombs[b].owner];
        *ammo = *ammo < POMMERMAN_MAX_AMMO ? *ammo + 1u : POMMERMAN_MAX_AMMO;
        pomBits cross = w->reach[b];
        burst |= cross;
        for (pomBits hit = cross & w->bomb; hit; hit &= hit - 1u) {
            uint32_t slot = w->bombAt[pom_low(hit)] - 1u;
            if (!((spent | fire) >> slot & 1u)) fire |= 1u << slot;
        }
    }
    uint32_t kept = 0u;
    for (uint32_t b = 0u; b < w->bombCount; b++) {
        w->bombAt[w->bombs[b].cell] = 0u;
        if (spent >> b & 1u) continue;
        w->bombs[kept] = w->bombs[b];
        w->reach[kept] = w->reach[b];
        kept++;
    }
    w->bombCount = kept;
    w->bomb = 0;
    for (uint32_t b = 0u; b < kept; b++) {
        w->bomb |= pom_bit(w->bombs[b].cell);
        w->bombAt[w->bombs[b].cell] = (uint8_t)(b + 1u);
    }
    pomBits burned = w->wood & burst;
    w->wood &= ~burst;
    for (uint32_t t = 0u; t < POM_KINDS; t++) w->item[t] &= ~burst;
    w->flame[2] = burst;
    if (burned) pom_reachAll(g, w, burned);
}

static void pom_collapse(const pommermanKnobs *k, const pomGeo *g, pomBoard *w, uint32_t *words, uint32_t ring) {
    pomBits cells = pom_ring(g, ring);
    for (uint32_t s = 0u; s < k->seats; s++)
        if ((words[PW_ALIVE] >> s & 1u) && (cells & pom_bit(words[PW_CELL + s]))) words[PW_ALIVE] &= ~(1u << s);
    uint32_t kept = 0u;
    for (uint32_t b = 0u; b < w->bombCount; b++) {
        w->bombAt[w->bombs[b].cell] = 0u;
        if (cells & pom_bit(w->bombs[b].cell)) {
            uint32_t *ammo = &words[PW_AMMO + w->bombs[b].owner];
            *ammo = *ammo < POMMERMAN_MAX_AMMO ? *ammo + 1u : POMMERMAN_MAX_AMMO;
            continue;
        }
        w->bombs[kept] = w->bombs[b];
        w->reach[kept] = w->reach[b];
        kept++;
    }
    w->bombCount = kept;
    w->bomb = 0;
    for (uint32_t b = 0u; b < kept; b++) {
        w->bomb |= pom_bit(w->bombs[b].cell);
        w->bombAt[w->bombs[b].cell] = (uint8_t)(b + 1u);
    }
    for (uint32_t t = 0u; t < POMMERMAN_FLAME_LIFE; t++) w->flame[t] &= ~cells;
    w->wood &= ~cells;
    for (uint32_t t = 0u; t < POM_KINDS; t++) {
        w->item[t] &= ~cells;
        w->hidden[t] &= ~cells;
    }
    w->rigid |= cells;
    pom_reachAll(g, w, cells);
}

static inline uint32_t pom_collapsing(const pommermanKnobs *k, uint32_t tick) {
    return k->collapseRings && tick >= k->collapseStart && (tick - k->collapseStart) % k->collapseEvery == 0u && (tick - k->collapseStart) / k->collapseEvery < k->collapseRings;
}

static inline __attribute__((always_inline)) void pom_tick(const pommermanKnobs *k, const pomGeo *g, pomBoard *w, uint32_t *words, const uint32_t *act) {
    uint32_t prior = words[PW_TICK], before = words[PW_ALIVE];
    if (pom_collapsing(k, prior)) pom_collapse(k, g, w, words, (prior - k->collapseStart) / k->collapseEvery);
    pomBits dying = w->flame[0];
    w->flame[0] = w->flame[1];
    w->flame[1] = w->flame[2];
    w->flame[2] = 0;
    pomBits still = w->flame[0] | w->flame[1];
    if (dying & (w->hidden[POM_EXTRA] | w->hidden[POM_RANGE] | w->hidden[POM_KICK]))
        for (uint32_t t = 0u; t < POM_KINDS; t++) {
            pomBits shown = w->hidden[t] & dying;
            w->item[t] |= shown & ~still;
            w->hidden[t] &= ~shown;
        }
    uint32_t alive = words[PW_ALIVE];
    for (uint32_t s = 0u; s < POM_SEATS; s++) words[PW_ACT + s] = s < k->seats && (alive >> s & 1u) ? act[s] : POM_STOP;
    for (uint32_t s = 0u; s < k->seats; s++)
        if ((alive >> s & 1u) && act[s] == POM_BOMB) pom_place(g, w, words, s);
    pom_move(k, g, w, words, act);
    pomBits bodies = pom_bodies(k, words);
    if ((w->item[POM_EXTRA] | w->item[POM_RANGE] | w->item[POM_KICK]) & bodies)
        for (uint32_t s = 0u; s < k->seats; s++) {
            if (!(alive >> s & 1u)) continue;
            pomBits at = pom_bit(words[PW_CELL + s]);
            if (w->item[POM_EXTRA] & at) {
                words[PW_AMMO + s] = words[PW_AMMO + s] < POMMERMAN_MAX_AMMO ? words[PW_AMMO + s] + 1u : POMMERMAN_MAX_AMMO;
                w->item[POM_EXTRA] &= ~at;
            } else if (w->item[POM_RANGE] & at) {
                words[PW_BLAST + s] = words[PW_BLAST + s] < k->maxBlast ? words[PW_BLAST + s] + 1u : k->maxBlast;
                w->item[POM_RANGE] &= ~at;
            } else if (w->item[POM_KICK] & at) {
                words[PW_KICK + s] = 1u;
                w->item[POM_KICK] &= ~at;
            }
        }
    uint32_t fire = 0u, lit = (still & w->bomb) != 0;
    for (uint32_t b = 0u; b < w->bombCount; b++) {
        w->bombs[b].life--;
        if (!w->bombs[b].life || (lit && (still & pom_bit(w->bombs[b].cell)))) fire |= 1u << b;
    }
    if (fire) pom_explode(g, w, words, fire);
    pomBits flames = still | w->flame[2];
    uint32_t tick = prior + 1u;
    if (flames & bodies)
        for (uint32_t s = 0u; s < k->seats; s++)
            if ((alive >> s & 1u) && (flames & pom_bit(words[PW_CELL + s]))) words[PW_ALIVE] &= ~(1u << s);
    for (uint32_t s = 0u; s < k->seats; s++)
        if ((before & ~words[PW_ALIVE]) >> s & 1u) w->death[s] = tick;
    words[PW_TICK] = tick;
}

static float pom_judge(const pommermanKnobs *k, const uint32_t *words, uint32_t *done) {
    uint32_t me = words[PW_SEAT], alive = words[PW_ALIVE], mine = 1u << me;
    uint32_t over = words[PW_TICK] >= k->tickLimit;
    if (k->team) {
        uint32_t ours = mine | 1u << pom_partner(k, me), theirs = 15u & ~ours;
        if (!(alive & theirs) && (alive & ours)) { *done = POM_WON; return 1.0f; }
        if (!(alive & ours) && (alive & theirs)) { *done = POM_LOST; return -1.0f; }
        if (!alive || over) { *done = POM_TIED; return -1.0f; }
        return 0.0f;
    }
    if (alive && !(alive & (alive - 1u))) {
        *done = alive == mine ? POM_WON : POM_LOST;
        return alive == mine ? 1.0f : -1.0f;
    }
    if (over || !alive) {
        *done = POM_TIED;
        return over || k->seats > 2u ? -1.0f : 0.0f;
    }
    if (!(alive & mine)) { *done = POM_LOST; return -1.0f; }
    return 0.0f;
}

static inline __attribute__((always_inline)) uint32_t pom_route(const pomGeo *g, uint32_t cell, pomBits through, pomBits goal, uint32_t depth, uint32_t lean, uint32_t word) {
    pomBits layer[POM_LAYERS], front = pom_bit(cell), open = through & ~front;
    depth = depth < POM_LAYERS - 1u ? depth : POM_LAYERS - 1u;
    for (uint32_t d = 1u; d <= depth; d++) {
        front = pom_spread(g, front) & open;
        if (!front) return POM_NONE;
        layer[d] = front;
        open &= ~front;
        pomBits hit = front & goal;
        if (!hit) continue;
        uint32_t at = pom_end(hit, lean, 0u);
        for (uint32_t back = d - 1u; back > 0u; back--) at = pom_end(pom_spread(g, pom_bit(at)) & layer[back], word, back);
        return at;
    }
    return POM_NONE;
}

static inline __attribute__((always_inline)) uint32_t pom_seek(const pomGeo *g, uint32_t cell, pomBits through, pomBits loot, uint32_t lootDepth, pomBits hunt, uint32_t huntDepth, pomBits dig, uint32_t digDepth, uint32_t lean, uint32_t word) {
    pomBits layer[POM_LAYERS], front = pom_bit(cell), open = through & ~front;
    uint32_t depth = lootDepth > huntDepth ? lootDepth : huntDepth, digAt = 0u, digCell = 0u;
    depth = digDepth > depth ? digDepth : depth;
    depth = depth < POM_LAYERS - 1u ? depth : POM_LAYERS - 1u;
    uint32_t goal = POM_NONE, goalDepth = 0u;
    for (uint32_t d = 1u; d <= depth; d++) {
        front = pom_spread(g, front) & open;
        if (!front) break;
        layer[d] = front;
        open &= ~front;
        if (d <= lootDepth && (front & loot)) { goal = pom_end(front & loot, lean, 0u); goalDepth = d; break; }
        if (d <= huntDepth && (front & hunt) && d > lootDepth) { goal = pom_end(front & hunt, lean, 0u); goalDepth = d; break; }
        if (d <= huntDepth && (front & hunt) && goal == POM_NONE) { goal = pom_end(front & hunt, lean, 0u); goalDepth = d; }
        if (!digAt && d <= digDepth && (front & dig)) { digAt = d; digCell = pom_end(front & dig, lean, 0u); }
        if (d >= lootDepth && goal != POM_NONE) break;
        if (d >= lootDepth && d >= huntDepth && digAt) break;
    }
    if (goal == POM_NONE && digAt) { goal = digCell; goalDepth = digAt; }
    if (goal == POM_NONE) return POM_NONE;
    for (uint32_t back = goalDepth - 1u; back > 0u; back--) goal = pom_end(pom_spread(g, pom_bit(goal)) & layer[back], word, back);
    return goal;
}

typedef struct { pomBits walk, threat, rigid, wood, bomb, loot, rivals, friends; } pomSight;

static inline __attribute__((always_inline)) uint32_t pom_sapper(const pomGeo *g, const pomSight *s, uint32_t cell, uint32_t ammo, uint32_t blast, const pomStyle *st, uint32_t lean, uint32_t wander, uint32_t bet, uint32_t choice) {
    uint32_t side = g->side;
    pomBits here = pom_bit(cell);
    if (s->threat & here) {
        uint32_t next = pom_route(g, cell, s->walk, s->walk & ~s->threat, st->flee, lean, choice);
        if (next != POM_NONE) return pom_toward(side, cell, next);
        pomBits out = pom_spread(g, here) & s->walk & ~here;
        return out ? pom_toward(side, cell, pom_pick(out, choice)) : POM_STOP;
    }
    if (ammo && !(s->bomb & here)) {
        pomBits cross = pom_cross(g, cell, blast - 1u, s->rigid, s->wood);
        uint32_t want = ((cross & s->rivals) && bet < st->enemyBomb) || ((cross & s->wood) && bet < st->woodBomb);
        if (want && !(cross & s->friends) && pom_route(g, cell, s->walk, s->walk & ~(s->threat | cross), st->strike, lean, choice) != POM_NONE) return POM_BOMB;
    }
    pomBits safe = s->walk & ~s->threat, out = pom_spread(g, here) & safe & ~here;
    if (wander >= st->wander) {
        pomBits dig = ammo ? pom_spread(g, s->wood) & safe : 0;
        uint32_t next = pom_seek(g, cell, safe, s->loot, st->loot, pom_spread(g, s->rivals) & safe, st->chase, dig, st->dig, lean, choice);
        if (next != POM_NONE) return pom_toward(side, cell, next);
    }
    return out ? pom_toward(side, cell, pom_pick(out, choice)) : POM_STOP;
}

static inline __attribute__((always_inline)) pomSight pom_common(const pomGeo *g, const pomBoard *w) {
    pomBits flames = w->flame[1] | w->flame[2], danger = flames;
    for (uint32_t b = 0u; b < w->bombCount; b++) danger |= w->reach[b];
    return (pomSight){ g->board & ~(w->rigid | w->wood | w->bomb | flames), danger, w->rigid, w->wood, w->bomb, w->item[0] | w->item[1] | w->item[2], 0, 0 };
}

static inline __attribute__((always_inline)) pomSight pom_sight(const pommermanKnobs *k, const pomGeo *g, const pomBoard *w, const uint32_t *words, const pomSight *all, uint32_t s) {
    uint32_t alive = words[PW_ALIVE];
    pomSight sight = *all;
    if (!k->fog && !k->team) {
        sight.rivals = pom_bodies(k, words) & ~pom_bit(words[PW_CELL + s]);
        return sight;
    }
    pomBits view = g->board;
    if (k->fog) {
        view = pom_window(g, words[PW_CELL + s], k->fog);
        sight.threat = w->flame[1] | w->flame[2];
        for (uint32_t b = 0u; b < w->bombCount; b++)
            if (view & pom_bit(w->bombs[b].cell)) sight.threat |= w->reach[b];
        sight.walk &= view;
        sight.threat &= view;
        sight.rigid &= view;
        sight.wood &= view;
        sight.bomb &= view;
        sight.loot &= view;
    }
    for (uint32_t o = 0u; o < k->seats; o++) {
        if (!(alive >> o & 1u) || o == s) continue;
        pomBits at = pom_bit(words[PW_CELL + o]) & view;
        if (pom_rival(k, s, o)) sight.rivals |= at;
        else sight.friends |= at;
    }
    return sight;
}

static inline __attribute__((always_inline)) void pom_bots(const pommermanKnobs *k, const pomGeo *g, const pomWorld *w, const uint32_t *words, uint32_t seed, uint32_t *act) {
    uint32_t me = words[PW_SEAT], alive = words[PW_ALIVE];
    for (uint32_t s = 0u; s < POM_SEATS; s++) act[s] = POM_STOP;
    u44 draw = philox4(seed, ENV_RNG_STREAM, (lane4)words[PW_GAME], (lane4)words[PW_TICK], (lane4){ 0u, 1u, 2u, 3u }, (lane4)POM_DRAW_BOTS);
    pomSight all = pom_common(g, &w->board);
    const float knob[POM_THETA] = { k->flee, k->strike, k->loot, k->chase, k->dig, k->wander, k->woodBomb, k->enemyBomb };
    const pomStyle fixed = pom_style(knob);
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (s == me || !(alive >> s & 1u)) continue;
        pomSight sight = pom_sight(k, g, &w->board, words, &all, s);
        const pomStyle *st = k->league ? &w->league.style[w->league.member[s]] : &fixed;
        act[s] = pom_sapper(g, &sight, words[PW_CELL + s], words[PW_AMMO + s], words[PW_BLAST + s], st, w->lean & 1u, draw.x[s], draw.y[s], draw.z[s]);
    }
}

static inline float pom_look(const pomBoard *w, pomBits rivals, pomBits friends, uint32_t side, int x, int y) {
    uint32_t inside = x >= 0 && y >= 0 && x < (int)side && y < (int)side, cell = inside ? (uint32_t)y * side + (uint32_t)x : 0u;
    pomBits bit = pom_bit(cell);
    if (!inside || (w->rigid & bit)) return 0.8f;
    if (w->wood & bit) return 0.6f;
    if (w->bomb & bit) return -0.55f + 0.2f * (float)(w->bombs[w->bombAt[cell] - 1u].life - 1u) / (float)(POMMERMAN_BOMB_LIFE - 1u);
    for (uint32_t t = POMMERMAN_FLAME_LIFE; t-- > 0u;)
        if (w->flame[t] & bit) return -0.65f - 0.05f * (float)(t + 1u);
    if (rivals & bit) return -1.0f;
    if (friends & bit) return 1.0f;
    if (w->item[POM_EXTRA] & bit) return 0.15f;
    if (w->item[POM_RANGE] & bit) return 0.3f;
    if (w->item[POM_KICK] & bit) return 0.45f;
    return 0.0f;
}

static void pom_observeWorld(const pommermanKnobs *k, const pomGeo *g, const pomBoard *w, const uint32_t *words, float *obs) {
    uint32_t me = words[PW_SEAT], alive = words[PW_ALIVE], side = k->side, cell = words[PW_CELL + me];
    int r = (int)(cell / side), c = (int)(cell % side), sight = (int)POMMERMAN_SIGHT;
    pomBits view = k->fog ? pom_window(g, cell, k->fog) : g->board, rivals = 0, friends = 0;
    float inv = 1.0f / (float)side, *other = obs + POM_OTHERS;
    for (uint32_t j = 1u; j < k->seats; j++, other += 3) {
        uint32_t s = (me + j) % k->seats, at = words[PW_CELL + s];
        other[0] = other[1] = other[2] = 0.0f;
        if (!(alive >> s & 1u) || !(view & pom_bit(at))) continue;
        uint32_t rival = pom_rival(k, me, s);
        if (rival) rivals |= pom_bit(at);
        else friends |= pom_bit(at);
        other[0] = rival ? -1.0f : 1.0f;
        other[1] = (float)((int)(at % side) - c) * inv;
        other[2] = (float)((int)(at / side) - r) * inv;
    }
    for (int y = r - sight; y <= r + sight; y++)
        for (int x = c - sight; x <= c + sight; x++)
            *obs++ = pom_look(w, rivals, friends, side, x, y);
    obs[0] = (float)words[PW_AMMO + me] / (float)POMMERMAN_MAX_AMMO;
    obs[1] = (float)words[PW_BLAST + me] / (float)k->maxBlast;
    obs[2] = (float)words[PW_KICK + me];
    obs[3] = (float)words[PW_TICK] / (float)k->tickLimit;
}

static void pom_leagueInit(const pommermanKnobs *k, pomLeague *l, uint32_t seed) {
    const float base[POM_THETA] = { k->flee, k->strike, k->loot, k->chase, k->dig, k->wander, k->woodBomb, k->enemyBomb };
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->winRate[i] = 0.5f;
    for (uint32_t d = 0u; d < POM_THETA; d++) l->theta[d] = base[d];
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < POM_THETA; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = base[d + j] + 0.1f * (POMMERMAN_THETA_HIGH[d + j] - POMMERMAN_THETA_LOW[d + j]) * n[j];
                l->theta[POM_THETA * i + d + j] = fminf(fmaxf(value, POMMERMAN_THETA_LOW[d + j]), POMMERMAN_THETA_HIGH[d + j]);
            }
        }
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->style[i] = pom_style(l->theta + POM_THETA * i);
}

static void pom_leagueEnd(const pommermanKnobs *k, pomWorld *w, const uint32_t *words, uint32_t seed) {
    uint32_t me = words[PW_SEAT], mine = w->board.death[me];
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (s == me) continue;
        uint32_t theirs = w->board.death[s];
        leagueScore(w->league.winRate, w->league.member[s], mine > theirs ? 1.0f : mine < theirs ? 0.0f : 0.5f);
    }
    leagueRenew(w->league.winRate, w->league.theta, POM_THETA, POMMERMAN_THETA_LOW, POMMERMAN_THETA_HIGH, seed, words[PW_GAME]);
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) w->league.style[i] = pom_style(w->league.theta + POM_THETA * i);
}

static uint32_t pom_action(const float *ctrl) {
    float a = ctrl[0];
    return a >= 0.0f && a < (float)POM_ACTIONS ? (uint32_t)a : POM_STOP;
}

static inline __attribute__((always_inline)) float pom_advance(const pommermanKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const pomGeo g = pom_geo(k->side);
    pomWorld *w = pom_world(extra);
    uint32_t act[POM_SEATS], result = 0u, me = words[PW_SEAT];
    float reward;
    do {
        pom_bots(k, &g, w, words, seed, act);
        act[me] = pom_action(ctrl);
        pom_tick(k, &g, &w->board, words, act);
        reward = pom_judge(k, words, &result);
    } while (!result && !(words[PW_ALIVE] >> me & 1u));
    *done = result != 0u;
    if (result) {
        words[PW_RESULT] = result;
        if (k->league) pom_leagueEnd(k, w, words, seed);
        pom_begin(k, &g, w, words, seed, words[PW_GAME] + 1u);
    }
    pom_observeWorld(k, &g, &w->board, words, obs);
    return reward;
}

void pommerman_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const pommermanKnobs *k = &POMMERMAN_KNOBS[instance];
    const pomGeo g = pom_geo(k->side);
    pomWorld *w = pom_world(extra);
    memset(w, 0, sizeof *w);
    for (uint32_t i = 0u; i < PW_WORDS; i++) words[i] = 0u;
    if (k->league) pom_leagueInit(k, &w->league, seed);
    pom_layout(k, w);
    pom_begin(k, &g, w, words, seed, 0u);
}

static __attribute__((noinline)) float pom_duel(uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) { return pom_advance(&POMMERMAN_KNOBS[0], seed, ctrl, words, extra, obs, done); }
static __attribute__((noinline)) float pom_melee(uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) { return pom_advance(&POMMERMAN_KNOBS[1], seed, ctrl, words, extra, obs, done); }
static __attribute__((noinline)) float pom_teams(uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) { return pom_advance(&POMMERMAN_KNOBS[2], seed, ctrl, words, extra, obs, done); }
static __attribute__((noinline)) float pom_shrink(uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) { return pom_advance(&POMMERMAN_KNOBS[3], seed, ctrl, words, extra, obs, done); }
static __attribute__((noinline)) float pom_rivalry(uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) { return pom_advance(&POMMERMAN_KNOBS[4], seed, ctrl, words, extra, obs, done); }

float pommerman_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return pom_duel(seed, ctrl, words, extra, obs, done);
        case 1u: return pom_melee(seed, ctrl, words, extra, obs, done);
        case 2u: return pom_teams(seed, ctrl, words, extra, obs, done);
        case 3u: return pom_shrink(seed, ctrl, words, extra, obs, done);
        default: return pom_rivalry(seed, ctrl, words, extra, obs, done);
    }
}

void pommerman_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    const pommermanKnobs *k = &POMMERMAN_KNOBS[instance];
    const pomGeo g = pom_geo(k->side);
    pom_observeWorld(k, &g, &pom_read(extra)->board, words, obs);
}

typedef struct { uint32_t cell, blast; } pomArm;

static void pom_forecast(const pommermanKnobs *k, const pomGeo *g, const pomBoard *w, uint32_t tick, pomForecast *f, const pomArm *arm, uint32_t arms, pomBits bodies) {
    uint32_t n = w->bombCount, cell[POM_BOMBS + 2u], when[POM_BOMBS + 2u];
    pomBits cross[POM_BOMBS + 2u];
    for (uint32_t i = 0u; i < n; i++) {
        cell[i] = w->bombs[i].cell;
        when[i] = w->bombs[i].life;
        cross[i] = w->reach[i];
    }
    for (uint32_t a = 0u; a < arms; a++, n++) {
        cell[n] = arm[a].cell;
        when[n] = POMMERMAN_BOMB_LIFE + 1u;
        cross[n] = pom_cross(g, arm[a].cell, arm[a].blast - 1u, w->rigid, w->wood);
    }
    for (uint32_t changed = 1u; changed;) {
        changed = 0u;
        for (uint32_t i = 0u; i < n; i++)
            for (uint32_t j = 0u; j < n; j++)
                if (when[i] < when[j] && (cross[i] & pom_bit(cell[j]))) {
                    when[j] = when[i];
                    changed = 1u;
                }
    }
    pomBits ground = g->board & ~(w->rigid | w->wood | bodies);
    for (uint32_t t = 0u; t <= POMMERMAN_HORIZON; t++) {
        f->flame[t] = 0;
        f->open[t] = ground;
    }
    f->flame[1] = w->flame[1] | w->flame[2];
    f->flame[2] = w->flame[2];
    for (uint32_t i = 0u; i < n; i++) {
        for (uint32_t t = when[i]; t < when[i] + POMMERMAN_FLAME_LIFE; t++) f->flame[t] |= cross[i];
        for (uint32_t t = 0u; t < when[i]; t++) f->open[t] &= ~pom_bit(cell[i]);
    }
    for (uint32_t t = 1u; k->collapseRings && t <= POMMERMAN_HORIZON; t++) {
        uint32_t at = tick + t;
        if (!pom_collapsing(k, at)) continue;
        pomBits ring = pom_ring(g, (at - k->collapseStart) / k->collapseEvery);
        f->flame[t] |= ring;
        for (uint32_t u = t; u <= POMMERMAN_HORIZON; u++) f->open[u] &= ~ring;
    }
    f->safe[POMMERMAN_HORIZON] = f->open[POMMERMAN_HORIZON] & ~f->flame[POMMERMAN_HORIZON];
    for (uint32_t t = POMMERMAN_HORIZON - 1u; t > 2u; t--) f->safe[t] = f->open[t] & ~f->flame[t] & pom_spread(g, f->safe[t + 1u]);
    pomBits loose = f->open[2] & ~f->flame[2] & pom_spread(g, f->safe[3]);
    f->roam = f->open[1] & ~f->flame[1] & pom_spread(g, loose);
    f->safe[2] = loose;
    f->safe[1] = f->open[1] & ~pom_spread(g, bodies) & ~f->flame[1] & pom_spread(g, f->safe[2]);
}

static uint32_t pom_lives(const pomGeo *g, const pomForecast *f, uint32_t cell) {
    pomBits at = pom_bit(cell);
    return (!(f->flame[1] & at) && (pom_spread(g, f->safe[2]) & at)) || (pom_spread(g, at) & ~at & f->safe[1]) != 0;
}

enum { POM_AIM_LOOT, POM_AIM_HUNT, POM_AIM_DIG, POM_AIMS };

static void pom_arrive(const pomGeo *g, const pomForecast *f, uint32_t cell, const pomBits *aim, uint32_t depth, uint32_t *arrival) {
    pomBits reach = pom_bit(cell);
    for (uint32_t a = 0u; a < POM_AIMS; a++) arrival[a] = (aim[a] & reach) ? 1u : 255u;
    for (uint32_t t = 2u; t <= depth; t++) {
        reach = pom_spread(g, reach) & f->safe[t < POMMERMAN_HORIZON ? t : POMMERMAN_HORIZON];
        if (!reach) return;
        for (uint32_t a = 0u; a < POM_AIMS; a++)
            if (arrival[a] == 255u && (aim[a] & reach)) arrival[a] = t;
    }
}

static void pom_believe(const pommermanKnobs *k, const pomGeo *g, pomWorld *w, const uint32_t *words, uint32_t *known) {
    const pomBoard *truth = &w->board;
    pomBoard *b = &w->belief;
    uint32_t me = words[PW_SEAT], kept = 0u;
    memcpy(known, words, PW_WORDS * sizeof *known);
    *b = *truth;
    for (uint32_t t = 0u; t < POM_KINDS; t++) b->hidden[t] = 0;
    if (!k->fog) return;
    pomBits view = pom_window(g, words[PW_CELL + me], k->fog);
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (s == me) continue;
        if (!(view & pom_bit(words[PW_CELL + s]))) known[PW_ALIVE] &= ~(1u << s);
        if (!pom_rival(k, me, s)) continue;
        known[PW_AMMO + s] = 1u;
        known[PW_BLAST + s] = k->maxBlast;
        known[PW_KICK + s] = 1u;
    }
    b->rigid &= view;
    b->wood &= view;
    for (uint32_t t = 0u; t < POMMERMAN_FLAME_LIFE; t++) b->flame[t] &= view;
    for (uint32_t t = 0u; t < POM_KINDS; t++) b->item[t] &= view;
    b->bomb &= view;
    memset(b->bombAt, 0, sizeof b->bombAt);
    for (uint32_t i = 0u; i < truth->bombCount; i++) {
        uint32_t cell = truth->bombs[i].cell;
        if (!(view & pom_bit(cell))) continue;
        b->bombs[kept] = truth->bombs[i];
        b->reach[kept] = pom_cross(g, cell, truth->bombs[i].blast - 1u, b->rigid, b->wood);
        b->bombAt[cell] = (uint8_t)(kept + 1u);
        kept++;
    }
    b->bombCount = kept;
}

static inline __attribute__((always_inline)) int32_t pom_score(const pommermanKnobs *k, const pomGeo *g, pomWorld *w, const uint32_t *words, uint32_t *act, uint32_t action) {
    uint32_t me = words[PW_SEAT], side = k->side, sim[PW_WORDS];
    memcpy(sim, words, sizeof sim);
    w->sim = w->belief;
    act[me] = action;
    pom_tick(k, g, &w->sim, sim, act);
    uint32_t alive = sim[PW_ALIVE];
    if (!(alive >> me & 1u)) return INT32_MIN + 1;
    const pomBoard *b = &w->sim;
    uint32_t cell = sim[PW_CELL + me], r = cell / side, c = cell % side, blast = sim[PW_BLAST + me];
    pomBits here = pom_bit(cell), bodies = pom_bodies(k, sim) & ~here;
    pom_forecast(k, g, b, sim[PW_TICK], &w->plain, 0, 0u, bodies);
    int32_t score = 0;
    if (pom_lives(g, &w->plain, cell)) {
        score = 3;
        for (uint32_t s = 0u; s < k->seats && score == 3; s++) {
            if (!(alive >> s & 1u) || !pom_rival(k, me, s) || !sim[PW_AMMO + s]) continue;
            uint32_t at = sim[PW_CELL + s], sr = at / side, sc = at % side;
            if ((sr > r ? sr - r : r - sr) + (sc > c ? sc - c : c - sc) > 3u || (b->bomb & pom_bit(at))) continue;
            pomArm arm = { at, sim[PW_BLAST + s] };
            pom_forecast(k, g, b, sim[PW_TICK], &w->probe, &arm, 1u, bodies);
            if (!pom_lives(g, &w->probe, cell)) score = 2;
        }
    } else score = 1;
    score *= 100000000;
    pomBits rivals = 0, cross = pom_cross(g, cell, blast - 1u, b->rigid, b->wood);
    for (uint32_t s = 0u; s < k->seats; s++) {
        if (s == me || !(alive >> s & 1u)) continue;
        uint32_t at = sim[PW_CELL + s];
        uint32_t trapped = !(pom_spread(g, pom_bit(at)) & w->plain.roam);
        if (pom_rival(k, me, s)) {
            rivals |= pom_bit(at);
            score += trapped ? 10000000 : 0;
        } else score -= trapped ? 10000000 : 0;
    }
    score += ((w->belief.item[POM_EXTRA] | w->belief.item[POM_RANGE] | w->belief.item[POM_KICK]) & here) ? 1000000 : 0;
    pomBits aim[POM_AIMS] = { b->item[0] | b->item[1] | b->item[2], 0, 0 };
    for (pomBits m = rivals; m; m &= m - 1u) {
        uint32_t at = pom_low(m);
        aim[POM_AIM_HUNT] |= pom_cross(g, at, blast - 1u, b->rigid, b->wood) & ~pom_bit(at);
    }
    if (sim[PW_AMMO + me]) aim[POM_AIM_DIG] = pom_spread(g, b->wood) & ~b->wood;
    uint32_t arrival[POM_AIMS];
    pom_arrive(g, &w->plain, cell, aim, 16u, arrival);
    if (arrival[POM_AIM_LOOT] <= 8u) score += (int32_t)(9u - arrival[POM_AIM_LOOT]) * 100000;
    if (action == POM_BOMB) score += (cross & rivals) ? 8000 : (cross & b->wood) ? 3000 : -5000;
    if (arrival[POM_AIM_HUNT] != 255u) score += (int32_t)(50u - arrival[POM_AIM_HUNT]) * 100;
    if (arrival[POM_AIM_DIG] != 255u) score += (int32_t)(50u - arrival[POM_AIM_DIG]) * 10;
    return score;
}

static inline __attribute__((always_inline)) uint32_t pom_decide(const pommermanKnobs *k, const pomGeo *g, pomWorld *w, const uint32_t *words) {
    uint32_t me = words[PW_SEAT], side = k->side;
    uint32_t known[PW_WORDS], act[POM_SEATS] = { POM_STOP, POM_STOP, POM_STOP, POM_STOP }, cell = words[PW_CELL + me], choice = POM_STOP;
    pom_believe(k, g, w, words, known);
    const pomBoard *b = &w->belief;
    int32_t best = INT32_MIN;
    for (uint32_t a = POM_STOP; a < POM_ACTIONS; a++) {
        if (a == POM_BOMB && (!words[PW_AMMO + me] || (b->bomb & pom_bit(cell)))) continue;
        if (a >= POM_UP && a <= POM_RIGHT) {
            uint32_t t = pom_step(side, cell, a);
            if (t == POM_NONE || ((b->rigid | b->wood) & pom_bit(t)) || ((b->bomb & pom_bit(t)) && !words[PW_KICK + me])) continue;
        }
        int32_t score = pom_score(k, g, w, known, act, a);
        if (score > best) {
            best = score;
            choice = a;
        }
    }
    return choice;
}

static inline __attribute__((always_inline)) uint32_t pom_consult(const pommermanKnobs *k, pomWorld *w, const uint32_t *words) {
    const pomGeo g = pom_geo(k->side);
    return pom_decide(k, &g, w, words);
}

void pommerman_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    pomWorld *w = pom_world(extra);
    switch (instance) {
        case 0u: ctrl[0] = (float)pom_consult(&POMMERMAN_KNOBS[0], w, words); return;
        case 1u: ctrl[0] = (float)pom_consult(&POMMERMAN_KNOBS[1], w, words); return;
        case 2u: ctrl[0] = (float)pom_consult(&POMMERMAN_KNOBS[2], w, words); return;
        case 3u: ctrl[0] = (float)pom_consult(&POMMERMAN_KNOBS[3], w, words); return;
        default: ctrl[0] = (float)pom_consult(&POMMERMAN_KNOBS[4], w, words); return;
    }
}

uint32_t pommerman_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    const pommermanKnobs *k = &POMMERMAN_KNOBS[instance];
    uint32_t cells = k->side * k->side, size = (cells + 1u) / 2u + POM_BOMBS / 2u;
    if (!out) return size;
    const pomBoard *w = &pom_read(extra)->board;
    for (uint32_t i = 0u; i < size; i++) out[i] = 0u;
    for (uint32_t cell = 0u; cell < cells; cell++) {
        pomBits bit = pom_bit(cell);
        uint32_t terrain = (w->rigid & bit) ? 1u : (w->wood & bit) ? 2u : 0u, item = 0u, hidden = 0u;
        for (uint32_t t = 0u; t < POM_KINDS; t++) {
            if (w->item[t] & bit) item = t + 1u;
            if (w->hidden[t] & bit) { item = t + 1u; hidden = 1u; }
        }
        uint32_t flame = (w->flame[2] & bit) ? 3u : (w->flame[1] & bit) ? 2u : (w->flame[0] & bit) ? 1u : 0u, field = terrain | item << 2 | hidden << 4 | flame << 5;
        if (w->bomb & bit) {
            const pomBomb *b = &w->bombs[w->bombAt[cell] - 1u];
            field |= (uint32_t)b->life << 7 | (uint32_t)b->dir << 11 | (uint32_t)b->owner << 14;
        }
        out[cell >> 1] |= field << (16u * (cell & 1u));
    }
    uint32_t base = (cells + 1u) / 2u;
    for (uint32_t b = 0u; b < w->bombCount; b++) {
        uint32_t field = (uint32_t)w->bombs[b].cell | (uint32_t)w->bombs[b].blast << 7;
        out[base + (b >> 1)] |= field << (16u * (b & 1u));
    }
    return size;
}
