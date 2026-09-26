#include <string.h>
#include "../env.h"

#define GENERALS_SUBSTEPS 16u
#define GENERALS_TICK_LIMIT 50000u
#define GENERALS_AFK_TICKS 2000u
#define GENERALS_TURN 2u
#define GENERALS_ROUND 50u
#define GENERALS_CITY_LOW 40u
#define GENERALS_CITY_SPAN 11u
#define GENERALS_CITY_REFILL 40
#define GENERALS_JUMPS 8u
#define GENERALS_SURVEY_ROWS 2u
#define GENERALS_SURVEY_CITIES 2u
#define GENERALS_SURVEY_STEPS 24u
#define GENERALS_HUNT 4u
#define GENERALS_REPLAN 8u
#define GENERALS_SCOUT 8u
#define GENERALS_ORACLE_WATCH 24u

typedef struct { uint32_t side, players, cities, spacing, mountains, opponent, league; } generalsKnobs;

static const generalsKnobs GENERALS_KNOBS[5] = {
    { 12u, 2u, 3u, 8u, 51u, 0u, 0u },
    { 23u, 2u, 11u, 15u, 51u, 1u, 0u },
    { 28u, 4u, 16u, 9u, 51u, 2u, 0u },
    { 32u, 8u, 20u, 9u, 51u, 3u, 0u },
    { 32u, 8u, 20u, 9u, 51u, 3u, 1u }
};

static const float GENERALS_STYLES[40] = {
    8.0f, 1.5f, 1.3f, 3.0f, 6.0f, 0.8f, 0.5f, 0.0f,
    4.0f, 2.5f, 1.1f, 8.0f, 3.0f, 2.0f, 0.1f, 0.0f,
    16.0f, 1.8f, 2.5f, 3.0f, 10.0f, 0.6f, 0.9f, 1.0f,
    10.0f, 1.1f, 1.8f, 4.0f, 6.0f, 1.0f, 0.5f, 1.0f,
    10.0f, 1.5f, 1.6f, 5.0f, 6.0f, 1.0f, 0.5f, 0.0f
};

static const float GENERALS_THETA_LOW[8] = { 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 0.5f, 0.0f, 0.0f };
static const float GENERALS_THETA_HIGH[8] = { 24.0f, 3.0f, 3.0f, 8.0f, 12.0f, 2.0f, 1.0f, 1.0f };

enum { GENERALS_SETTLER, GENERALS_RAIDER, GENERALS_BALANCED, GENERALS_STYLED };
enum { GEN_STYLE_RAIDER, GEN_STYLE_RUSHER, GEN_STYLE_TURTLE, GEN_STYLE_ECONOMIST, GEN_STYLE_BALANCED, GEN_STYLES };
enum { GEN_SIDE = 32, GEN_CELLS = GEN_SIDE * GEN_SIDE, GEN_ROWS = GEN_SIDE + 4, GEN_PLAYERS = 8, GEN_NOBODY = 8, GEN_THETA = 8, GEN_CITIES = 20, GEN_FAR = 255 };
enum { GW_CURSOR, GW_MODE, GW_PHASE, GW_TICK, GW_PENDING, GW_SEAT, GW_SWEEP, GW_GAME, GW_QUIET, GW_WORDS };
enum { GEN_LOOK, GEN_ALL, GEN_HALF };
enum { GEN_PLAIN, GEN_PEAK, GEN_CITY, GEN_CAPITAL, GEN_KINDS };
enum { GEN_EXPAND, GEN_GREED, GEN_STRIKE, GEN_GATHER, GEN_RADIUS, GEN_ALARM, GEN_SPLIT, GEN_TARGET };
enum { GEN_HOLD, GEN_ASSAULT, GEN_RETREAT, GEN_SIEGE, GEN_SPREAD, GEN_FEED };
enum { GEN_MOVE = 1u << 13, GEN_SPLIT_BIT = 1u << 12 };
enum { GEN_TAG_MOUNTAIN = 1u, GEN_TAG_GENERAL, GEN_TAG_PATH, GEN_TAG_CITY, GEN_TAG_SEAT, GEN_TAG_STYLE, GEN_TAG_SETTLER, GEN_TAG_MARSHAL, GEN_TAG_LEAGUE };
enum { GEN_SURVEY_MOUNTAINS, GEN_SURVEY_GENERALS, GEN_SURVEY_PATHS, GEN_SURVEY_CITIES, GEN_SURVEY_BOARD, GEN_SURVEY_DONE };

typedef struct {
    uint32_t reached[GEN_ROWS], front[GEN_ROWS], next[GEN_ROWS];
    uint32_t layer, low, high;
} generalsFlood;

typedef struct {
    uint32_t mountain[GEN_ROWS], city[GEN_ROWS], path[GEN_ROWS], open[GEN_ROWS];
    int32_t army[GEN_CELLS];
    uint8_t kind[GEN_CELLS];
    uint16_t general[GEN_PLAYERS], cityCell[GEN_CITIES];
    uint8_t cityArmy[GEN_CITIES];
    uint32_t game, stage, index, left, at, step;
} generalsSurvey;

typedef struct {
    float theta[GEN_THETA];
    uint32_t stack, aim, trail, mode, split, replans, member;
    uint32_t seen[GEN_PLAYERS], seenTick[GEN_PLAYERS];
    int32_t seenArmy[GEN_PLAYERS];
} generalsMarshal;

typedef struct {
    float theta[LEAGUE_MEMBERS * GEN_THETA], winRate[LEAGUE_MEMBERS];
} generalsLeague;

typedef struct {
    uint32_t game, tick, ready, start, count, jumps;
    uint8_t press[GENERALS_SUBSTEPS];
    uint32_t jump[GENERALS_JUMPS];
    generalsFlood flood, trail;
    uint8_t distance[GEN_CELLS], route[GEN_CELLS];
} generalsOracle;

typedef struct {
    int32_t army[GEN_CELLS];
    uint8_t owner[GEN_CELLS], kind[GEN_CELLS];
    uint32_t own[GEN_PLAYERS][GEN_ROWS];
    uint32_t board[GEN_ROWS], passable[GEN_ROWS], city[GEN_ROWS], sight[GEN_ROWS], occupied[GEN_ROWS];
    float face[2 * GEN_KINDS * (GEN_PLAYERS + 1)];
    int32_t land[GEN_PLAYERS], total[GEN_PLAYERS];
    uint32_t home[GEN_PLAYERS], act[GEN_PLAYERS];
    uint32_t alive, jumps, jumpReady, structures;
    uint16_t structure[GEN_PLAYERS + GEN_CITIES];
    uint32_t jump[GENERALS_JUMPS];
    generalsMarshal marshal[GEN_PLAYERS];
    generalsLeague league;
    generalsSurvey survey;
    generalsOracle oracle;
} generalsWorld;

typedef struct { uint64_t key; uint32_t player, source, dest, split, strike; } generalsOrder;

static generalsWorld *generals_world(uint32_t *extra) { return (generalsWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const generalsWorld *generals_read(const uint32_t *extra) { return (const generalsWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }

uint32_t generals_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)(sizeof(generalsWorld) / sizeof(uint32_t)) + 4u;
}

static uint32_t generals_mask(uint32_t side) { return side >= 32u ? 0xFFFFFFFFu : (1u << side) - 1u; }
static uint32_t generals_bit(const uint32_t *rows, uint32_t cell) { return rows[(cell >> 5) + 1u] >> (cell & 31u) & 1u; }
static void generals_set(uint32_t *rows, uint32_t cell) { rows[(cell >> 5) + 1u] |= 1u << (cell & 31u); }
static void generals_clear(uint32_t *rows, uint32_t cell) { rows[(cell >> 5) + 1u] &= ~(1u << (cell & 31u)); }
static uint32_t generals_cell(uint32_t y, uint32_t row) { return (y - 1u) << 5 | (uint32_t)__builtin_ctz(row); }
static uint32_t generals_pack(uint32_t cell, uint32_t dir, uint32_t split) { return GEN_MOVE | split << 12 | dir << 10 | cell; }
static uint32_t generals_below(uint32_t word, uint32_t n) { return (uint32_t)(((uint64_t)word * n) >> 32); }

static uint32_t generals_neighbor(uint32_t cell, uint32_t dir, uint32_t side) {
    int dx, dy;
    gridDelta(dir, &dx, &dy);
    uint32_t x = (cell & 31u) + (uint32_t)dx, y = (cell >> 5) + (uint32_t)dy;
    return x < side && y < side ? y << 5 | x : GEN_CELLS;
}

static uint32_t generals_span(uint32_t a, uint32_t b) {
    int dx = (int)(a & 31u) - (int)(b & 31u), dy = (int)(a >> 5) - (int)(b >> 5);
    return (uint32_t)(abs(dx) + abs(dy));
}

static void generals_look(const uint32_t *own, uint32_t side, uint32_t *out) {
    uint32_t mask = generals_mask(side);
    for (uint32_t y = 1u; y <= side; y++) {
        uint32_t r = own[y - 1u] | own[y] | own[y + 1u];
        out[y] = (r | r << 1 | r >> 1) & mask;
    }
}

static uint32_t generals_count(const uint32_t *rows, uint32_t side) {
    uint32_t n = 0u;
    for (uint32_t y = 1u; y <= side; y++) n += (uint32_t)oak_popcount(rows[y]);
    return n;
}

static uint32_t generals_select(const uint32_t *rows, uint32_t side, uint32_t index) {
    for (uint32_t y = 1u; y <= side; y++) {
        uint32_t row = rows[y], n = (uint32_t)oak_popcount(row);
        if (index >= n) { index -= n; continue; }
        while (index--) row &= row - 1u;
        return generals_cell(y, row);
    }
    __builtin_trap();
}

static void generals_floodStart(generalsFlood *f, uint8_t *distance, const uint32_t *sources, uint32_t side) {
    memset(distance, GEN_FAR, GEN_CELLS);
    memset(f, 0, sizeof *f);
    f->low = side + 1u;
    for (uint32_t y = 1u; y <= side; y++) {
        f->reached[y] = f->front[y] = sources[y];
        if (!sources[y]) continue;
        f->low = y < f->low ? y : f->low;
        f->high = y;
        for (uint32_t row = sources[y]; row; row &= row - 1u) distance[generals_cell(y, row)] = 0u;
    }
}

static uint32_t generals_floodLayer(generalsFlood *f, uint8_t *distance, const uint32_t *pass, uint32_t side) {
    if (f->low > f->high || f->layer >= GEN_FAR - 1u) return 0u;
    uint32_t top = f->low > 1u ? f->low - 1u : 1u, bottom = f->high < side ? f->high + 1u : side, low = side + 1u, high = 0u;
    for (uint32_t y = top; y <= bottom; y += 4u) {
        uint32x4_t mid = vld1q_u32(f->front + y);
        uint32x4_t n = vorrq_u32(vorrq_u32(vld1q_u32(f->front + y - 1u), vld1q_u32(f->front + y + 1u)), vorrq_u32(vshlq_n_u32(mid, 1), vshrq_n_u32(mid, 1)));
        vst1q_u32(f->next + y, vbicq_u32(vandq_u32(n, vld1q_u32(pass + y)), vld1q_u32(f->reached + y)));
    }
    uint8_t label = (uint8_t)(f->layer + 1u);
    for (uint32_t y = top - 1u; y <= bottom + 1u && y <= side; y++) f->front[y] = 0u;
    for (uint32_t y = top; y <= bottom; y++) {
        uint32_t row = f->next[y];
        if (!row) continue;
        low = y < low ? y : low;
        high = y;
        f->front[y] = row;
        f->reached[y] |= row;
        for (; row; row &= row - 1u) distance[generals_cell(y, row)] = label;
    }
    f->low = low;
    f->high = high;
    if (low > high) return 0u;
    f->layer++;
    return 1u;
}

static uint32_t generals_descend(const uint8_t *distance, uint32_t cell, uint32_t side) {
    uint32_t here = distance[cell];
    if (here == 0u || here == GEN_FAR) return 4u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t next = generals_neighbor(cell, d, side);
        if (next < GEN_CELLS && distance[next] + 1u == here) return d;
    }
    return 4u;
}

static uint32_t generals_mountainRow(uint32_t seed, uint32_t game, uint32_t y, uint32_t threshold) {
    u4 a = philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_MOUNTAIN, y, 0u), b = philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_MOUNTAIN, y, 1u);
    const uint8x16_t weight = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u };
    uint8x16_t limit = vdupq_n_u8((uint8_t)threshold);
    uint8x16_t low = vandq_u8(vcltq_u8(vreinterpretq_u8_u32(vld1q_u32(&a.x)), limit), weight);
    uint8x16_t high = vandq_u8(vcltq_u8(vreinterpretq_u8_u32(vld1q_u32(&b.x)), limit), weight);
    uint8x16_t sum = vpaddq_u8(low, high);
    sum = vpaddq_u8(sum, sum);
    sum = vpaddq_u8(sum, sum);
    return vgetq_lane_u32(vreinterpretq_u32_u8(sum), 0);
}

static void generals_placeGeneral(const generalsKnobs *k, generalsSurvey *s, uint32_t seed) {
    uint32_t side = k->side, i = s->index, reach = k->spacing - 1u;
    if (i == 0u)
        for (uint32_t y = 1u; y <= side; y++) s->open[y] = generals_mask(side);
    uint32_t count = generals_count(s->open, side);
    if (count == 0u) __builtin_trap();
    uint32_t cell = generals_select(s->open, side, generals_below(philox4x32(seed, ENV_RNG_STREAM, s->game, GEN_TAG_GENERAL, i, 0u).x, count));
    s->general[i] = (uint16_t)cell;
    generals_clear(s->mountain, cell);
    uint32_t gx = cell & 31u, gy = cell >> 5;
    for (uint32_t y = gy > reach ? gy - reach : 0u; y <= gy + reach && y < side; y++) {
        uint32_t r = reach - (y > gy ? y - gy : gy - y), low = gx > r ? gx - r : 0u, high = gx + r < side ? gx + r : side - 1u;
        s->open[y + 1u] &= ~(((2u << high) - 1u) & ~((1u << low) - 1u));
    }
}

static void generals_carve(generalsSurvey *s, uint32_t seed) {
    uint32_t to = s->general[0], cell = s->at == GEN_CELLS ? s->general[s->index] : s->at;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, s->game, GEN_TAG_PATH, s->index, 0u);
    uint32_t bits[4] = { r.x, r.y, r.z, r.w }, x = cell & 31u, y = cell >> 5, tx = to & 31u, ty = to >> 5, stop = s->step + GENERALS_SURVEY_STEPS;
    generals_set(s->path, cell);
    for (; (x != tx || y != ty) && s->step < stop; s->step++) {
        if (x != tx && (y == ty || (bits[s->step >> 5] >> (s->step & 31u) & 1u))) x = x < tx ? x + 1u : x - 1u;
        else y = y < ty ? y + 1u : y - 1u;
        generals_clear(s->mountain, y << 5 | x);
        generals_set(s->path, y << 5 | x);
    }
    s->at = x == tx && y == ty ? GEN_CELLS : y << 5 | x;
}

static void generals_placeCities(const generalsKnobs *k, generalsSurvey *s, uint32_t seed) {
    uint32_t side = k->side;
    for (uint32_t j = s->index; j < s->index + GENERALS_SURVEY_CITIES && j < k->cities; j++) {
        if (s->left == 0u) __builtin_trap();
        u4 r = philox4x32(seed, ENV_RNG_STREAM, s->game, GEN_TAG_CITY, j, 0u);
        uint32_t cell = generals_select(s->open, side, generals_below(r.x, s->left));
        generals_clear(s->open, cell);
        s->left--;
        generals_set(s->city, cell);
        s->cityCell[j] = (uint16_t)cell;
        s->cityArmy[j] = (uint8_t)(GENERALS_CITY_LOW + generals_below(r.y, GENERALS_CITY_SPAN));
    }
}

static void generals_boardRows(generalsSurvey *s, uint32_t from, uint32_t to) {
    const uint8x16_t pick[2] = { { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u }, { 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u } };
    const uint8x16_t bit = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u };
    for (uint32_t y = from; y < to; y++) {
        uint8x16_t row = vreinterpretq_u8_u32(vdupq_n_u32(s->mountain[y + 1u]));
        for (uint32_t h = 0u; h < 2u; h++) vst1q_u8(s->kind + (y << 5) + 16u * h, vandq_u8(vtstq_u8(vqtbl1q_u8(row, pick[h]), bit), vdupq_n_u8(GEN_PEAK)));
    }
}

static void generals_survey(const generalsKnobs *k, generalsSurvey *s, uint32_t seed) {
    uint32_t side = k->side;
    switch (s->stage) {
        case GEN_SURVEY_MOUNTAINS:
            if (s->index == 0u) {
                memset(s->mountain, 0, sizeof s->mountain);
                memset(s->city, 0, sizeof s->city);
                memset(s->path, 0, sizeof s->path);
            }
            for (uint32_t y = s->index; y < s->index + GENERALS_SURVEY_ROWS && y < side; y++) s->mountain[y + 1u] = generals_mountainRow(seed, s->game, y, k->mountains) & generals_mask(side);
            s->index += GENERALS_SURVEY_ROWS;
            if (s->index >= side) { s->stage = GEN_SURVEY_GENERALS; s->index = 0u; }
            return;
        case GEN_SURVEY_GENERALS:
            generals_placeGeneral(k, s, seed);
            if (++s->index == k->players) { s->stage = GEN_SURVEY_PATHS; s->index = 1u; }
            return;
        case GEN_SURVEY_PATHS:
            generals_carve(s, seed);
            if (s->at == GEN_CELLS) {
                s->step = 0u;
                if (++s->index == k->players) { s->stage = GEN_SURVEY_CITIES; s->index = 0u; }
            }
            return;
        case GEN_SURVEY_CITIES:
            if (s->step == 0u) {
                for (uint32_t y = 1u; y <= side; y++) s->open[y] = generals_mask(side) & ~s->mountain[y] & ~s->path[y];
                s->left = generals_count(s->open, side);
                s->step = 1u;
                return;
            }
            generals_placeCities(k, s, seed);
            s->index += GENERALS_SURVEY_CITIES;
            if (s->index >= k->cities) { s->stage = GEN_SURVEY_BOARD; s->index = 0u; }
            return;
        case GEN_SURVEY_BOARD:
            if (s->index == 0u) {
                memset(s->army, 0, sizeof s->army);
                generals_boardRows(s, 0u, GEN_SIDE / 2u);
                s->index = 1u;
                return;
            }
            generals_boardRows(s, GEN_SIDE / 2u, GEN_SIDE);
            for (uint32_t c = 0u; c < k->cities; c++) {
                s->kind[s->cityCell[c]] = GEN_CITY;
                s->army[s->cityCell[c]] = s->cityArmy[c];
            }
            for (uint32_t p = 0u; p < k->players; p++) {
                s->kind[s->general[p]] = GEN_CAPITAL;
                s->army[s->general[p]] = 1;
            }
            s->stage = GEN_SURVEY_DONE;
            return;
        default: return;
    }
}

static void generals_rival(const generalsKnobs *k, generalsWorld *w, uint32_t b, uint32_t seed, uint32_t game, uint32_t *picked, float *masked) {
    generalsMarshal *m = &w->marshal[b];
    memset(m, 0, sizeof *m);
    m->stack = w->home[b];
    m->aim = m->trail = GEN_CELLS;
    if (k->opponent == GENERALS_SETTLER) return;
    if (k->league) {
        uint32_t member = leaguePick(masked, philox4x32(seed, LEAGUE_RNG_STREAM, game, GEN_TAG_LEAGUE, b, 0u).x);
        if (*picked >> member & 1u)
            for (member = 0u; *picked >> member & 1u; member++) {}
        *picked |= 1u << member;
        masked[member] = 1.0f;
        m->member = member;
        memcpy(m->theta, w->league.theta + member * GEN_THETA, sizeof m->theta);
        return;
    }
    uint32_t style = k->opponent == GENERALS_RAIDER ? GEN_STYLE_RAIDER : k->opponent == GENERALS_BALANCED ? GEN_STYLE_BALANCED
                   : GEN_STYLE_RUSHER + generals_below(philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_STYLE, b, 0u).x, GEN_STYLES - 1u);
    memcpy(m->theta, GENERALS_STYLES + style * GEN_THETA, sizeof m->theta);
}

static void generals_begin(const generalsKnobs *k, generalsWorld *w, uint32_t *words, uint32_t seed, uint32_t game) {
    generalsSurvey *s = &w->survey;
    if (s->stage != GEN_SURVEY_DONE || s->game != game) __builtin_trap();
    uint32_t side = k->side, players = k->players, mask = generals_mask(side);
    memcpy(w->army, s->army, sizeof w->army);
    memcpy(w->kind, s->kind, sizeof w->kind);
    memset(w->owner, GEN_NOBODY, sizeof w->owner);
    memset(w->own, 0, sizeof w->own);
    memset(w->occupied, 0, sizeof w->occupied);
    for (uint32_t y = 0u; y < GEN_ROWS; y++) {
        uint32_t on = y >= 1u && y <= side ? mask : 0u;
        w->board[y] = on;
        w->passable[y] = on & ~s->mountain[y];
        w->city[y] = s->city[y];
    }
    w->structures = 0u;
    for (uint32_t p = 0u; p < players; p++) {
        uint32_t cell = s->general[p];
        w->home[p] = cell;
        w->owner[cell] = (uint8_t)p;
        generals_set(w->own[p], cell);
        generals_set(w->occupied, cell);
        w->land[p] = 1;
        w->total[p] = 1;
        w->act[p] = 0u;
        w->structure[w->structures++] = (uint16_t)cell;
    }
    for (uint32_t c = 0u; c < k->cities; c++) w->structure[w->structures++] = s->cityCell[c];
    w->alive = (1u << players) - 1u;
    w->jumpReady = 0u;
    uint32_t seat = generals_below(philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_SEAT, 0u, 0u).x, players);
    for (uint32_t kind = 0u; kind < GEN_KINDS; kind++)
        for (uint32_t owner = 0u; owner <= GEN_PLAYERS; owner++) {
            uint32_t rank = kind == GEN_CITY ? 1u : kind == GEN_CAPITAL ? 2u : 0u;
            uint32_t seen = kind == GEN_PEAK ? 2u : owner == GEN_NOBODY ? 3u + rank : owner == seat ? 5u + rank : 8u + 3u * owner + rank;
            w->face[kind * (GEN_PLAYERS + 1u) + owner] = (float)(kind == GEN_PEAK || kind == GEN_CITY) * 0.03125f;
            w->face[(GEN_KINDS + kind) * (GEN_PLAYERS + 1u) + owner] = (float)seen * 0.03125f;
        }
    words[GW_CURSOR] = w->home[seat];
    words[GW_MODE] = GEN_LOOK;
    words[GW_PHASE] = 0u;
    words[GW_TICK] = 0u;
    words[GW_PENDING] = 0u;
    words[GW_SEAT] = seat;
    words[GW_SWEEP] = 0u;
    words[GW_GAME] = game;
    words[GW_QUIET] = 0u;
    generals_look(w->own[seat], side, w->sight);
    uint32_t picked = 0u;
    float masked[LEAGUE_MEMBERS];
    memcpy(masked, w->league.winRate, sizeof masked);
    for (uint32_t b = 0u; b < players; b++)
        if (b != seat) generals_rival(k, w, b, seed, game, &picked, masked);
    s->game = game + 1u;
    s->stage = GEN_SURVEY_MOUNTAINS;
    s->index = s->left = s->step = 0u;
    s->at = GEN_CELLS;
}

static void generals_fall(generalsWorld *w, uint32_t side, uint32_t victim, uint32_t victor) {
    uint32_t cell = w->home[victim];
    w->kind[cell] = GEN_CITY;
    generals_set(w->city, cell);
    for (uint32_t y = 1u; y <= side; y++) {
        w->own[victor][y] |= w->own[victim][y];
        w->own[victim][y] = 0u;
    }
    uint8x16_t lost = vdupq_n_u8((uint8_t)victim), gain = vdupq_n_u8((uint8_t)victor);
    int32x4_t gained = vdupq_n_s32(0);
    for (uint32_t c = 0u; c < side << 5; c += 16u) {
        uint8x16_t owner = vld1q_u8(w->owner + c), taken = vceqq_u8(owner, lost);
        vst1q_u8(w->owner + c, vbslq_u8(taken, gain, owner));
        int16x8_t low = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(taken))), high = vmovl_high_s8(vreinterpretq_s8_u8(taken));
        int32x4_t masks[4] = { vmovl_s16(vget_low_s16(low)), vmovl_high_s16(low), vmovl_s16(vget_low_s16(high)), vmovl_high_s16(high) };
        for (uint32_t j = 0u; j < 4u; j++) {
            int32x4_t army = vld1q_s32(w->army + c + 4u * j), kept = vsubq_s32(army, vshrq_n_s32(army, 1));
            vst1q_s32(w->army + c + 4u * j, vbslq_s32(vreinterpretq_u32_s32(masks[j]), kept, army));
            gained = vaddq_s32(gained, vandq_s32(kept, masks[j]));
        }
    }
    w->total[victor] += vaddvq_s32(gained);
    w->land[victor] += w->land[victim];
    w->land[victim] = 0;
    w->total[victim] = 0;
    w->alive &= ~(1u << victim);
}

static uint32_t generals_execute(generalsWorld *w, uint32_t side, const generalsOrder *o) {
    uint32_t p = o->player, source = o->source, dest = o->dest;
    if (w->owner[source] != p) return 0u;
    int32_t a = w->army[source], moved = o->split ? a / 2 : a - 1;
    if (moved < 1) return 0u;
    w->army[source] = a - moved;
    uint32_t q = w->owner[dest];
    if (q == p) {
        w->army[dest] += moved;
        return 1u;
    }
    int32_t defend = w->army[dest];
    if (moved <= defend) {
        w->army[dest] = defend - moved;
        w->total[p] -= moved;
        if (q != GEN_NOBODY) w->total[q] -= moved;
        return 1u;
    }
    w->army[dest] = moved - defend;
    w->total[p] -= defend;
    w->owner[dest] = (uint8_t)p;
    generals_set(w->own[p], dest);
    generals_set(w->occupied, dest);
    w->land[p]++;
    if (q != GEN_NOBODY) {
        w->total[q] -= defend;
        w->land[q]--;
        generals_clear(w->own[q], dest);
        if (w->kind[dest] == GEN_CAPITAL) generals_fall(w, side, q, p);
    }
    return 1u;
}

static void generals_recount(generalsWorld *w, uint32_t side, uint32_t p) {
    int32_t land = 0, total = 0;
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = w->own[p][y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            land++;
            total += w->army[c];
            w->owner[c] = (uint8_t)p;
        }
    w->land[p] = land;
    w->total[p] = total;
}

static uint32_t generals_trade(generalsWorld *w, uint32_t side, const generalsOrder *a, const generalsOrder *b) {
    uint32_t p = a->player, q = b->player, gp = w->home[p], gq = w->home[q];
    if (a->dest != gq || b->dest != gp || w->owner[a->source] != p || w->owner[b->source] != q) return 0u;
    int32_t ap = w->army[a->source], aq = w->army[b->source];
    int32_t mp = a->split ? ap / 2 : ap - 1, mq = b->split ? aq / 2 : aq - 1;
    if (mp < 1 || mq < 1) return 0u;
    w->army[a->source] = ap - mp;
    w->army[b->source] = aq - mq;
    int32_t holdQ = w->army[gq], holdP = w->army[gp];
    if (mp <= holdQ || mq <= holdP) {
        w->army[a->source] = ap;
        w->army[b->source] = aq;
        return 0u;
    }
    for (uint32_t y = 1u; y <= side; y++) {
        uint32_t mine = w->own[p][y], theirs = w->own[q][y];
        for (uint32_t row = mine | theirs; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            w->army[c] -= w->army[c] / 2;
        }
        w->own[p][y] = theirs;
        w->own[q][y] = mine;
    }
    w->army[gq] = mp - holdQ;
    w->army[gp] = mq - holdP;
    w->home[p] = gq;
    w->home[q] = gp;
    generals_recount(w, side, p);
    generals_recount(w, side, q);
    return 1u;
}

static uint32_t generals_resolve(const generalsKnobs *k, generalsWorld *w, uint32_t time) {
    generalsOrder o[GEN_PLAYERS];
    uint32_t n = 0u, side = k->side, flip = time & 1u ? 7u : 0u, moved = 0u;
    for (uint32_t p = 0u; p < k->players; p++) {
        uint32_t code = w->act[p];
        w->act[p] = 0u;
        if (!code || !(w->alive >> p & 1u)) continue;
        uint32_t source = code & (GEN_CELLS - 1u), dest = generals_neighbor(source, code >> 10 & 3u, side);
        if (dest >= GEN_CELLS || w->kind[dest] == GEN_PEAK || w->owner[source] != p || w->army[source] < 2) continue;
        uint32_t strike = w->kind[dest] == GEN_CAPITAL && w->owner[dest] != p, at = n++;
        uint64_t key = (uint64_t)!strike << 62 | (uint64_t)(w->owner[dest] == p) << 61 | (uint64_t)(uint32_t)w->army[source] << 3 | (7u - (p ^ flip));
        for (; at > 0u && o[at - 1u].key < key; at--) o[at] = o[at - 1u];
        o[at] = (generalsOrder){ key, p, source, dest, code >> 12 & 1u, strike };
    }
    uint32_t calm = 0u, chased[GEN_PLAYERS], tangled = 0u;
    while (calm < n && !o[calm].strike) calm++;
    for (uint32_t i = 0u; i < calm; i++) {
        chased[i] = 0u;
        for (uint32_t j = 0u; j < calm; j++)
            if (j != i && o[j].dest == o[i].source) chased[i] |= 1u << j;
        tangled |= chased[i];
    }
    if (!tangled)
        for (uint32_t i = 0u; i < calm; i++) moved |= generals_execute(w, side, &o[i]);
    else
        for (uint32_t left = (1u << calm) - 1u; left;) {
            uint32_t pick = (uint32_t)__builtin_ctz(left);
            for (uint32_t rest = left; rest; rest &= rest - 1u) {
                uint32_t i = (uint32_t)__builtin_ctz(rest);
                if (!(chased[i] & left)) { pick = i; break; }
            }
            left &= ~(1u << pick);
            moved |= generals_execute(w, side, &o[pick]);
        }
    for (uint32_t strikes = ((1u << n) - 1u) & ~((1u << calm) - 1u); strikes;) {
        uint32_t best = (uint32_t)__builtin_ctz(strikes), traded = 0u;
        strikes &= ~(1u << best);
        for (uint32_t pool = strikes; pool && !traded; pool &= pool - 1u) {
            uint32_t j = (uint32_t)__builtin_ctz(pool);
            if (o[j].player == w->owner[o[best].dest] && generals_trade(w, side, &o[best], &o[j])) {
                strikes &= ~(1u << j);
                traded = 1u;
            }
        }
        moved |= traded ? 1u : generals_execute(w, side, &o[best]);
    }
    return moved;
}

static void generals_growth(const generalsKnobs *k, generalsWorld *w, uint32_t time) {
    if (time % GENERALS_TURN) return;
    for (uint32_t i = 0u; i < w->structures; i++) {
        uint32_t c = w->structure[i], o = w->owner[c];
        if (o != GEN_NOBODY) {
            w->army[c]++;
            w->total[o]++;
        } else if (w->army[c] < GENERALS_CITY_REFILL) w->army[c]++;
    }
    if (time % GENERALS_ROUND) return;
    uint8x16_t nobody = vdupq_n_u8(GEN_NOBODY);
    for (uint32_t c = 0u; c < k->side << 5; c += 16u) {
        int8x16_t owned = vreinterpretq_s8_u8(vmvnq_u8(vceqq_u8(vld1q_u8(w->owner + c), nobody)));
        int16x8_t low = vmovl_s8(vget_low_s8(owned)), high = vmovl_high_s8(owned);
        int32_t *army = w->army + c;
        vst1q_s32(army, vsubq_s32(vld1q_s32(army), vmovl_s16(vget_low_s16(low))));
        vst1q_s32(army + 4, vsubq_s32(vld1q_s32(army + 4), vmovl_high_s16(low)));
        vst1q_s32(army + 8, vsubq_s32(vld1q_s32(army + 8), vmovl_s16(vget_low_s16(high))));
        vst1q_s32(army + 12, vsubq_s32(vld1q_s32(army + 12), vmovl_high_s16(high)));
    }
    for (uint32_t p = 0u; p < k->players; p++) w->total[p] += w->land[p];
}

static uint32_t generals_rank(const generalsKnobs *k, const generalsWorld *w, uint32_t seat, uint32_t *list) {
    int32_t armies[GENERALS_JUMPS], floor = 1;
    uint32_t n = 0u;
    for (uint32_t y = 1u; y <= k->side; y++)
        for (uint32_t row = w->own[seat][y]; row; row &= row - 1u) {
            uint32_t cell = generals_cell(y, row);
            int32_t army = w->army[cell];
            if (army <= floor) continue;
            uint32_t at = n < GENERALS_JUMPS ? n++ : n - 1u;
            for (; at > 0u && armies[at - 1u] < army; at--) {
                list[at] = list[at - 1u];
                armies[at] = armies[at - 1u];
            }
            list[at] = cell;
            armies[at] = army;
            if (n == GENERALS_JUMPS) floor = armies[n - 1u];
        }
    return n;
}

static void generals_press(const generalsKnobs *k, generalsWorld *w, uint32_t *words, uint32_t action) {
    uint32_t cursor = words[GW_CURSOR], mode = words[GW_MODE];
    if (action < 4u) {
        if (mode == GEN_LOOK) {
            uint32_t next = generals_neighbor(cursor, action, k->side);
            if (next < GEN_CELLS) words[GW_CURSOR] = next;
            return;
        }
        if (w->owner[cursor] == words[GW_SEAT] && w->army[cursor] >= 2) words[GW_PENDING] = generals_pack(cursor, action, mode == GEN_HALF);
        words[GW_MODE] = GEN_LOOK;
        return;
    }
    if (action == 4u) {
        words[GW_MODE] = mode == GEN_HALF ? GEN_LOOK : mode + 1u;
        return;
    }
    if (!w->jumpReady) {
        w->jumps = generals_rank(k, w, words[GW_SEAT], w->jump);
        w->jumpReady = 1u;
    }
    if (w->jumps == 0u) return;
    uint32_t next = w->jump[0];
    for (uint32_t i = 0u; i < w->jumps; i++)
        if (w->jump[i] == cursor) next = w->jump[(i + 1u) % w->jumps];
    words[GW_CURSOR] = next;
}

static void generals_observeWorld(const generalsKnobs *k, const generalsWorld *w, const uint32_t *words, float *obs) {
    static const float MODES[3] = { 0.0f, 1.0f, 0.5f };
    uint32_t seat = words[GW_SEAT], cursor = words[GW_CURSOR], home = w->home[seat], x = cursor & 31u, y = cursor >> 5, side = k->side;
    uint32_t cells[5] = { cursor, cursor - GEN_SIDE, cursor + GEN_SIDE, cursor - 1u, cursor + 1u };
    uint32_t open[5] = { 1u, y > 0u, y + 1u < side, x > 0u, x + 1u < side };
    for (uint32_t i = 0u; i < 5u; i++) {
        uint32_t c = open[i] ? cells[i] : cursor, visible = generals_bit(w->sight, c);
        float code = w->face[(visible * GEN_KINDS + w->kind[c]) * (GEN_PLAYERS + 1u) + w->owner[c]];
        float army = (float)(w->army[c] & -(int32_t)visible);
        obs[2u * i] = open[i] ? code : 2.0f * 0.03125f;
        obs[2u * i + 1u] = open[i] ? army : 0.0f;
    }
    obs[10] = (float)((int32_t)x - (int32_t)(home & 31u));
    obs[11] = (float)((int32_t)y - (int32_t)(home >> 5));
    obs[12] = MODES[words[GW_MODE]];
    obs[13] = (float)words[GW_PHASE] * (1.0f / (float)GENERALS_SUBSTEPS);
    obs[14] = (float)((words[GW_TICK] / GENERALS_TURN) % (GENERALS_ROUND / GENERALS_TURN)) * ((float)GENERALS_TURN / (float)GENERALS_ROUND);
    uint32_t p = words[GW_SWEEP];
    obs[15] = p == seat ? 1.0f : ((w->alive >> p) & 1u ? 1.0f : -1.0f) * (float)(p + 1u) * 0.125f;
    obs[16] = (float)w->land[p];
    obs[17] = (float)w->total[p];
}

static uint32_t generals_richest(const generalsWorld *w, const uint32_t *rows, uint32_t side, uint32_t skip) {
    uint32_t best = GEN_CELLS;
    int32_t most = 1;
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = rows[y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            if (w->army[c] > most && c != skip) { most = w->army[c]; best = c; }
        }
    return best;
}

static uint32_t generals_toward(const generalsWorld *w, uint32_t side, uint32_t from, uint32_t to, uint32_t trail) {
    int dx = (int)(to & 31u) - (int)(from & 31u), dy = (int)(to >> 5) - (int)(from >> 5);
    uint32_t across = dx > 0 ? 3u : 2u, along = dy > 0 ? 1u : 0u, wide = abs(dx) >= abs(dy), back = 4u;
    uint32_t order[4] = { wide ? across : along, wide ? along : across, (wide ? along : across) ^ 1u, (wide ? across : along) ^ 1u };
    for (uint32_t i = 0u; i < 4u; i++) {
        uint32_t n = generals_neighbor(from, order[i], side);
        if (n >= GEN_CELLS || w->kind[n] == GEN_PEAK || (w->kind[n] == GEN_CITY && n != to && w->owner[n] == GEN_NOBODY)) continue;
        if (n != trail) return order[i];
        back = order[i];
    }
    return back;
}

static uint32_t generals_kill(const generalsKnobs *k, const generalsWorld *w, uint32_t b) {
    for (uint32_t q = 0u; q < k->players; q++) {
        if (q == b || !(w->alive >> q & 1u)) continue;
        uint32_t target = w->home[q];
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t from = generals_neighbor(target, d, k->side);
            if (from < GEN_CELLS && w->owner[from] == b && w->army[from] - 1 > w->army[target]) return generals_pack(from, d ^ 1u, 0u);
        }
    }
    return 0u;
}

typedef struct { uint32_t top, bottom, mask; } generalsWindow;

static generalsWindow generals_window(uint32_t cell, uint32_t radius, uint32_t side) {
    uint32_t x = cell & 31u, y = cell >> 5, low = x > radius ? x - radius : 0u, high = x + radius < side - 1u ? x + radius : side - 1u;
    return (generalsWindow){ y > radius ? y - radius + 1u : 1u, y + radius + 1u < side ? y + radius + 1u : side, ((2u << high) - 1u) & ~((1u << low) - 1u) };
}

static uint32_t generals_sees(const uint32_t *own, uint32_t cell) {
    uint32_t y = (cell >> 5) + 1u, r = own[y - 1u] | own[y] | own[y + 1u];
    return (r | r << 1 | r >> 1) >> (cell & 31u) & 1u;
}

static uint32_t generals_seize(const generalsWorld *w, uint32_t side, uint32_t b, uint32_t stack, uint32_t cities, uint32_t noise) {
    int32_t moved = w->army[stack] - 1;
    uint32_t home = w->home[b], dir = 4u, best = 0u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n = generals_neighbor(stack, d, side);
        if (n >= GEN_CELLS || w->kind[n] == GEN_PEAK || w->owner[n] == b || moved <= w->army[n] || (!cities && w->kind[n] == GEN_CITY && w->owner[n] == GEN_NOBODY)) continue;
        uint32_t rank = w->kind[n] == GEN_CAPITAL ? 4u : w->owner[n] != GEN_NOBODY ? 3u : w->kind[n] == GEN_CITY ? 1u : 2u;
        uint32_t score = rank << 24 | (generals_span(n, home) + 1u) << 16 | ((noise ^ (n * 0x9E3779B1u)) >> 16);
        if (score > best) { best = score; dir = d; }
    }
    return dir;
}

static uint32_t generals_nearest(const generalsWorld *w, const uint32_t *mine, uint32_t side, uint32_t stack, uint32_t radius) {
    int32_t moved = w->army[stack] - 1;
    uint32_t best = GEN_CELLS, reach = GEN_CELLS;
    generalsWindow v = generals_window(stack, radius, side);
    for (uint32_t y = v.top; y <= v.bottom; y++)
        for (uint32_t row = (mine[y - 1u] | mine[y + 1u] | mine[y] << 1 | mine[y] >> 1) & w->passable[y] & ~mine[y] & v.mask; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row), r = generals_span(stack, c) + (w->owner[c] == GEN_NOBODY ? GENERALS_HUNT : 0u);
            if (w->army[c] < moved && r < reach) { reach = r; best = c; }
        }
    return best;
}

static uint32_t generals_settler(const generalsKnobs *k, generalsWorld *w, uint32_t b, uint32_t seed, uint32_t game, uint32_t tick) {
    generalsMarshal *m = &w->marshal[b];
    uint32_t side = k->side, kill = generals_kill(k, w, b);
    if (kill) return kill;
    uint32_t stack = generals_richest(w, w->own[b], side, GEN_CELLS);
    if (stack == GEN_CELLS) return 0u;
    if (stack != m->stack) { m->trail = GEN_CELLS; m->aim = GEN_CELLS; }
    uint32_t dir = generals_seize(w, side, b, stack, 1u, philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_SETTLER, tick, 0u).x);
    if (dir > 3u) {
        if (m->aim == GEN_CELLS || w->owner[m->aim] == b || w->army[m->aim] >= w->army[stack] - 1) m->aim = generals_nearest(w, w->own[b], side, stack, GEN_SIDE);
        if (m->aim == GEN_CELLS) {
            uint32_t feeder = generals_richest(w, w->own[b], side, stack);
            if (feeder == GEN_CELLS) return 0u;
            dir = generals_toward(w, side, feeder, stack, GEN_CELLS);
            return dir > 3u ? 0u : generals_pack(feeder, dir, 0u);
        }
        dir = generals_toward(w, side, stack, m->aim, m->trail);
    }
    if (dir > 3u) return 0u;
    m->trail = stack;
    m->stack = generals_neighbor(stack, dir, side);
    return generals_pack(stack, dir, 0u);
}

static uint32_t generals_reserve(const generalsWorld *w, uint32_t b, uint32_t skip) {
    uint32_t best = GEN_CELLS;
    int32_t most = 1;
    for (uint32_t i = 0u; i < w->structures; i++) {
        uint32_t c = w->structure[i];
        if (w->owner[c] == b && c != skip && w->army[c] > most) { most = w->army[c]; best = c; }
    }
    return best;
}

static uint32_t generals_outward(const generalsWorld *w, uint32_t side, uint32_t b, uint32_t from, uint32_t trail) {
    uint32_t home = w->home[b], dir = 4u, far = 0u;
    int32_t moved = w->army[from] - 1;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n = generals_neighbor(from, d, side);
        if (n >= GEN_CELLS || n == trail || w->kind[n] == GEN_PEAK || (w->kind[n] == GEN_CITY && w->owner[n] != b) || (w->owner[n] != b && moved <= w->army[n])) continue;
        uint32_t reach = generals_span(n, home) + 1u;
        if (reach > far) { far = reach; dir = d; }
    }
    return dir;
}

static uint32_t generals_follow(const generalsKnobs *k, generalsWorld *w, uint32_t b) {
    generalsMarshal *m = &w->marshal[b];
    uint32_t side = k->side, home = w->home[b];
    if (m->mode == GEN_HOLD && w->owner[m->stack] == b) {
        uint32_t source = generals_reserve(w, b, m->stack);
        if (source != GEN_CELLS) { m->mode = GEN_FEED; m->aim = source; m->trail = GEN_CELLS; }
    }
    uint32_t mover = m->mode == GEN_FEED ? m->aim : m->stack, dir = 4u;
    if (m->mode == GEN_HOLD || w->owner[mover] != b || w->army[mover] < 2) { m->mode = GEN_HOLD; return 0u; }
    switch (m->mode) {
        case GEN_ASSAULT:
            if (w->owner[m->aim] == b) break;
            dir = generals_toward(w, side, mover, m->aim, m->trail);
            break;
        case GEN_RETREAT:
            if (mover == home) break;
            dir = generals_toward(w, side, mover, home, m->trail);
            break;
        case GEN_SIEGE:
            if (w->owner[m->aim] == b) break;
            dir = generals_toward(w, side, mover, m->aim, m->trail);
            break;
        case GEN_FEED:
            if (mover == m->stack) break;
            dir = generals_toward(w, side, mover, m->stack, m->trail);
            break;
        default:
            dir = generals_seize(w, side, b, mover, 0u, 0u);
            if (dir > 3u && m->aim < GEN_CELLS && w->owner[m->aim] != b && w->army[m->aim] < w->army[mover] - 1) dir = generals_toward(w, side, mover, m->aim, m->trail);
            if (dir > 3u) dir = generals_outward(w, side, b, mover, m->trail);
    }
    if (dir > 3u) { m->mode = GEN_HOLD; return 0u; }
    uint32_t next = generals_neighbor(mover, dir, side), split = m->split && mover == home;
    if (m->mode == GEN_FEED) m->aim = next;
    else m->stack = next;
    m->trail = mover;
    return generals_pack(mover, dir, split);
}

static uint32_t generals_replan(const generalsKnobs *k, generalsWorld *w, uint32_t b, uint32_t seed, uint32_t game, uint32_t tick) {
    generalsMarshal *m = &w->marshal[b];
    const float *theta = m->theta;
    uint32_t side = k->side, home = w->home[b];
    const uint32_t *mine = w->own[b];
    m->replans++;
    m->mode = GEN_HOLD;
    m->trail = GEN_CELLS;
    for (uint32_t q = 0u; q < k->players; q++) {
        if (q == b) continue;
        if (!(w->alive >> q & 1u)) { m->seen[q] = 0u; continue; }
        uint32_t target = w->home[q];
        if (generals_sees(mine, target)) {
            m->seen[q] = target + 1u;
            m->seenArmy[q] = w->army[target];
            m->seenTick[q] = tick;
            for (uint32_t d = 0u; d < 4u; d++) {
                uint32_t from = generals_neighbor(target, d, side);
                if (from < GEN_CELLS && w->owner[from] == b && w->army[from] - 1 > w->army[target]) return generals_pack(from, d ^ 1u, 0u);
            }
        } else if (m->seen[q] && generals_sees(mine, m->seen[q] - 1u)) m->seen[q] = 0u;
    }
    uint32_t stack = w->owner[m->stack] == b ? m->stack : home, second = GEN_CELLS;
    for (uint32_t i = 0u; i < w->structures; i++) {
        uint32_t c = w->structure[i];
        if (w->owner[c] != b || c == stack) continue;
        if (w->army[c] > w->army[stack]) { second = stack; stack = c; }
        else if (second == GEN_CELLS || w->army[c] > w->army[second]) second = c;
    }
    if (second != GEN_CELLS && w->army[second] < 2) second = GEN_CELLS;
    int32_t army = w->army[stack];
    if (army < 2) return 0u;
    m->stack = stack;
    m->split = stack == home && u01(philox4x32(seed, ENV_RNG_STREAM, game, GEN_TAG_MARSHAL + b, tick, 0u).x) < theta[GEN_SPLIT];
    generalsWindow guard = generals_window(home, (uint32_t)(theta[GEN_RADIUS] + 0.5f), side);
    float alarm = theta[GEN_ALARM] * (float)w->army[home];
    uint32_t threat = GEN_CELLS;
    int32_t worst = 0;
    for (uint32_t y = guard.top; y <= guard.bottom; y++) {
        uint32_t r = mine[y - 1u] | mine[y] | mine[y + 1u];
        for (uint32_t row = (r | r << 1 | r >> 1) & w->occupied[y] & ~mine[y] & guard.mask; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            int32_t danger = w->army[c] - (int32_t)generals_span(c, home);
            if ((float)danger > alarm && danger > worst) { worst = danger; threat = c; }
        }
    }
    if (threat != GEN_CELLS) {
        if (army - 1 - (int32_t)generals_span(stack, threat) > w->army[threat]) { m->mode = GEN_SIEGE; m->aim = threat; }
        else if (stack != home) m->mode = GEN_RETREAT;
        else if (second != GEN_CELLS) { m->mode = GEN_FEED; m->aim = second; }
        return 0u;
    }
    uint32_t target = GEN_PLAYERS;
    float targetScore = 0.0f;
    for (uint32_t q = 0u; q < k->players; q++) {
        if (q == b || !m->seen[q]) continue;
        float score = theta[GEN_TARGET] < 0.5f ? -(float)generals_span(stack, m->seen[q] - 1u) : -(float)w->total[q];
        if (target == GEN_PLAYERS || score > targetScore) { target = q; targetScore = score; }
    }
    if (target < GEN_PLAYERS) {
        float defense = (float)m->seenArmy[target] + (float)((tick - m->seenTick[target]) / GENERALS_TURN);
        if ((float)(army - 1) > theta[GEN_STRIKE] * defense + (float)generals_span(stack, m->seen[target] - 1u)) {
            m->mode = GEN_ASSAULT;
            m->aim = m->seen[target] - 1u;
            return 0u;
        }
    }
    uint32_t period = (uint32_t)(theta[GEN_GATHER] + 0.5f);
    if (second != GEN_CELLS && period && m->replans % period == 0u) {
        m->mode = GEN_FEED;
        m->aim = second;
        return 0u;
    }
    uint32_t city = GEN_CELLS, nearest = GEN_CELLS;
    for (uint32_t i = 0u; i < w->structures; i++) {
        uint32_t c = w->structure[i], reach = generals_span(stack, c);
        if (w->owner[c] != GEN_NOBODY || reach >= nearest || (float)(army - 1) <= theta[GEN_GREED] * (float)w->army[c] + (float)reach || !generals_sees(mine, c)) continue;
        city = c;
        nearest = reach;
    }
    if (city != GEN_CELLS) {
        m->mode = GEN_SIEGE;
        m->aim = city;
        return 0u;
    }
    if ((float)army >= theta[GEN_EXPAND]) {
        m->mode = GEN_SPREAD;
        m->aim = generals_nearest(w, mine, side, stack, GENERALS_SCOUT);
    } else if (second != GEN_CELLS) {
        m->mode = GEN_FEED;
        m->aim = second;
    }
    return 0u;
}

static uint32_t generals_marshal(const generalsKnobs *k, generalsWorld *w, uint32_t b, uint32_t seed, uint32_t game, uint32_t tick) {
    if (tick % GENERALS_REPLAN == b) {
        uint32_t kill = generals_replan(k, w, b, seed, game, tick);
        if (kill) return kill;
    }
    return generals_follow(k, w, b);
}

static void generals_think(const generalsKnobs *k, generalsWorld *w, const uint32_t *words, uint32_t seed) {
    uint32_t b = words[GW_PHASE];
    if (b >= k->players || b == words[GW_SEAT]) return;
    if (!(w->alive >> b & 1u)) { w->act[b] = 0u; return; }
    uint32_t game = words[GW_GAME], tick = words[GW_TICK];
    w->act[b] = k->opponent == GENERALS_SETTLER ? generals_settler(k, w, b, seed, game, tick) : generals_marshal(k, w, b, seed, game, tick);
}

static void generals_score(const generalsKnobs *k, generalsWorld *w, uint32_t seed, uint32_t game, uint32_t seat) {
    uint32_t agentAlive = w->alive >> seat & 1u;
    for (uint32_t p = 0u; p < k->players; p++) {
        if (p == seat) continue;
        float outcome = !(w->alive >> p & 1u) ? 1.0f : agentAlive ? 0.5f : 0.0f;
        leagueScore(w->league.winRate, w->marshal[p].member, outcome);
    }
    leagueRenew(w->league.winRate, w->league.theta, GEN_THETA, GENERALS_THETA_LOW, GENERALS_THETA_HIGH, seed, game);
}

static __attribute__((noinline)) float generals_tick(const generalsKnobs *k, generalsWorld *w, uint32_t *words, uint32_t seed, uint32_t *done) {
    uint32_t seat = words[GW_SEAT], time = words[GW_TICK], pending = words[GW_PENDING];
    w->act[seat] = pending;
    uint32_t moved = generals_resolve(k, w, time);
    time++;
    generals_growth(k, w, time);
    if (pending) {
        uint32_t dest = generals_neighbor(pending & (GEN_CELLS - 1u), pending >> 10 & 3u, k->side);
        if (dest < GEN_CELLS && w->owner[dest] == seat) words[GW_CURSOR] = dest;
    }
    uint32_t quiet = moved ? 0u : words[GW_QUIET] + 1u;
    words[GW_PENDING] = 0u;
    words[GW_PHASE] = 0u;
    words[GW_TICK] = time;
    words[GW_QUIET] = quiet;
    w->jumpReady = 0u;
    generals_look(w->own[seat], k->side, w->sight);
    uint32_t mine = 1u << seat;
    float reward = 0.0f;
    if (!(w->alive & mine)) { reward = -1.0f; *done = 1u; }
    else if (w->alive == mine) { reward = 1.0f; *done = 1u; }
    else if (time >= GENERALS_TICK_LIMIT || quiet >= GENERALS_AFK_TICKS) *done = 1u;
    if (!*done) return reward;
    uint32_t game = words[GW_GAME];
    if (k->league) generals_score(k, w, seed, game, seat);
    generals_begin(k, w, words, seed, game + 1u);
    return reward;
}

static uint32_t generals_action(const float *ctrl) {
    float a = ctrl[0];
    return a >= 0.0f && a < 6.0f ? (uint32_t)a : 0u;
}

static inline __attribute__((always_inline)) float generals_advance(const generalsKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    generalsWorld *w = generals_world(extra);
    generals_press(k, w, words, generals_action(ctrl));
    generals_think(k, w, words, seed);
    float reward = 0.0f;
    *done = 0u;
    if (words[GW_PHASE] == GENERALS_SUBSTEPS - 1u) reward = generals_tick(k, w, words, seed, done);
    else {
        if (w->survey.stage != GEN_SURVEY_DONE) generals_survey(k, &w->survey, seed);
        words[GW_PHASE]++;
    }
    words[GW_SWEEP] = words[GW_SWEEP] + 1u == k->players ? 0u : words[GW_SWEEP] + 1u;
    generals_observeWorld(k, w, words, obs);
    return reward;
}

void generals_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const generalsKnobs *k = &GENERALS_KNOBS[instance];
    generalsWorld *w = generals_world(extra);
    memset(w, 0, sizeof *w);
    w->survey.at = GEN_CELLS;
    if (k->league) {
        for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) w->league.winRate[i] = 0.5f;
        for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) {
            const float *style = GENERALS_STYLES + (GEN_STYLE_RUSHER + i % (GEN_STYLES - 1u)) * GEN_THETA;
            for (uint32_t d = 0u; d < GEN_THETA; d += 4u) {
                u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
                float4 n = normal4(r.x, r.y);
                for (uint32_t j = 0u; j < 4u; j++) {
                    float value = i < GEN_STYLES - 1u ? style[d + j] : style[d + j] + 0.1f * (GENERALS_THETA_HIGH[d + j] - GENERALS_THETA_LOW[d + j]) * n[j];
                    w->league.theta[i * GEN_THETA + d + j] = fminf(fmaxf(value, GENERALS_THETA_LOW[d + j]), GENERALS_THETA_HIGH[d + j]);
                }
            }
        }
    }
    while (w->survey.stage != GEN_SURVEY_DONE) generals_survey(k, &w->survey, seed);
    generals_begin(k, w, words, seed, 0u);
}

float generals_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return generals_advance(&GENERALS_KNOBS[0], seed, ctrl, words, extra, obs, done);
        case 1u: return generals_advance(&GENERALS_KNOBS[1], seed, ctrl, words, extra, obs, done);
        case 2u: return generals_advance(&GENERALS_KNOBS[2], seed, ctrl, words, extra, obs, done);
        case 3u: return generals_advance(&GENERALS_KNOBS[3], seed, ctrl, words, extra, obs, done);
        default: return generals_advance(&GENERALS_KNOBS[4], seed, ctrl, words, extra, obs, done);
    }
}

void generals_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    generals_observeWorld(&GENERALS_KNOBS[instance], generals_read(extra), words, obs);
}

static int32_t generals_growthBy(uint32_t tick, uint32_t span) {
    uint32_t end = tick + span;
    return (int32_t)(end / GENERALS_TURN - tick / GENERALS_TURN + end / GENERALS_ROUND - tick / GENERALS_ROUND);
}

static int32_t generals_carry(const generalsWorld *w, uint32_t side, uint32_t seat, const uint8_t *distance, uint32_t source, uint32_t split, uint32_t *steps, uint32_t *first) {
    uint32_t cell = source;
    int32_t moved = split ? w->army[source] / 2 : w->army[source] - 1;
    *first = 4u;
    for (uint32_t length = 1u; length < GEN_FAR; length++) {
        uint32_t dir = generals_descend(distance, cell, side);
        if (dir > 3u || moved < 1) return -1;
        if (length == 1u) *first = dir;
        cell = generals_neighbor(cell, dir, side);
        if (distance[cell] == 0u) { *steps = length; return moved; }
        if (w->owner[cell] == seat && w->army[cell] > w->army[source]) return -1;
        int32_t stack = w->owner[cell] == seat ? moved + w->army[cell] : moved - w->army[cell];
        if (w->owner[cell] != seat && stack < 1) return -1;
        moved = stack - 1;
    }
    return -1;
}

static int32_t generals_guard(const generalsWorld *w, uint32_t side, uint32_t q) {
    int32_t guard = 0;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t t = generals_neighbor(w->home[q], d, side);
        if (t < GEN_CELLS && w->owner[t] == q && w->army[t] - 1 > guard) guard = w->army[t] - 1;
    }
    return guard;
}

static uint32_t generals_route(uint32_t side, generalsOracle *o, const uint32_t *targets, const uint32_t *walk, uint32_t from) {
    generals_floodStart(&o->flood, o->distance, targets, side);
    while (o->distance[from] == GEN_FAR && generals_floodLayer(&o->flood, o->distance, walk, side)) {}
    return generals_descend(o->distance, from, side);
}

static uint32_t generals_advice(const generalsKnobs *k, const generalsWorld *w, const uint32_t *words, generalsOracle *o, uint32_t *moves) {
    uint32_t side = k->side, seat = words[GW_SEAT], home = w->home[seat], tick = words[GW_TICK], cursor = words[GW_CURSOR], n = 0u;
    const uint32_t *mine = w->own[seat];
    for (uint32_t q = 0u; q < k->players; q++) {
        if (q == seat || !(w->alive >> q & 1u)) continue;
        uint32_t g = w->home[q];
        int32_t guard = generals_guard(w, side, q);
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t t = generals_neighbor(g, d, side);
            if (t < GEN_CELLS && w->owner[t] == seat && w->army[t] - 1 > w->army[g] + guard && n < 4u) moves[n++] = generals_pack(t, d ^ 1u, 0u);
        }
    }
    uint32_t source[GEN_ROWS] = {0};
    generals_set(source, home);
    generals_floodStart(&o->flood, o->distance, source, side);
    for (uint32_t layer = 0u; layer < GENERALS_ORACLE_WATCH && generals_floodLayer(&o->flood, o->distance, w->passable, side); layer++) {}
    uint32_t threat = GEN_CELLS, reach = GEN_FAR;
    int32_t danger = 0;
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = o->flood.reached[y] & ~mine[y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row), d = o->distance[c], q = w->owner[c];
            if (q == GEN_NOBODY || (d > 4u && (k->opponent == GENERALS_SETTLER || w->marshal[q].seen[seat] != home + 1u))) continue;
            int32_t potential = w->army[c] - (int32_t)d - generals_growthBy(tick, d);
            if (potential > danger) { danger = potential; threat = c; reach = d; }
        }
    uint32_t warned = threat != GEN_CELLS;
    if (danger < w->army[home]) threat = GEN_CELLS;
    uint32_t tops[3] = { GEN_CELLS, GEN_CELLS, GEN_CELLS }, pool[GEN_ROWS];
    memcpy(pool, mine, sizeof pool);
    for (uint32_t i = 0u; i < 3u; i++) {
        tops[i] = generals_richest(w, pool, side, GEN_CELLS);
        if (tops[i] == GEN_CELLS) break;
        generals_clear(pool, tops[i]);
    }
    uint32_t strike = 0u, root = GEN_CELLS;
    int32_t margin = 0, closest = -0x7FFFFFFF;
    uint32_t marks[2] = { GEN_PLAYERS, GEN_PLAYERS };
    int32_t cost[2] = { 0, 0 };
    for (uint32_t q = 0u; q < k->players; q++) {
        if (q == seat || !(w->alive >> q & 1u) || tops[0] == GEN_CELLS) continue;
        int32_t price = w->army[w->home[q]] + generals_guard(w, side, q) + (int32_t)generals_span(tops[0], w->home[q]);
        if (marks[0] == GEN_PLAYERS || price < cost[0]) { marks[1] = marks[0]; cost[1] = cost[0]; marks[0] = q; cost[0] = price; }
        else if (marks[1] == GEN_PLAYERS || price < cost[1]) { marks[1] = q; cost[1] = price; }
    }
    uint32_t lanes[GEN_ROWS] = {0};
    for (uint32_t y = 1u; y <= side; y++) lanes[y] = w->passable[y] & ~(w->city[y] & ~w->occupied[y]);
    for (uint32_t t = 0u; t < 2u && marks[t] != GEN_PLAYERS; t++) {
        uint32_t q = marks[t], g = w->home[q], goal[GEN_ROWS] = {0};
        int32_t guard = generals_guard(w, side, q);
        generals_set(goal, g);
        generals_floodStart(&o->trail, o->route, goal, side);
        while ((o->route[tops[0]] == GEN_FAR || (tops[1] != GEN_CELLS && o->route[tops[1]] == GEN_FAR) || (tops[2] != GEN_CELLS && o->route[tops[2]] == GEN_FAR)) && generals_floodLayer(&o->trail, o->route, lanes, side)) {}
        for (uint32_t i = 0u; i < 3u && tops[i] != GEN_CELLS; i++)
            for (uint32_t split = 0u; split < 2u; split++) {
                uint32_t steps = 0u, first = 4u;
                int32_t carried = generals_carry(w, side, seat, o->route, tops[i], split, &steps, &first);
                if (carried < 0 || first > 3u) continue;
                int32_t edge = carried - w->army[g] - generals_growthBy(tick, steps) - guard - 1;
                int32_t kept = tops[i] == home ? w->army[home] - (split ? w->army[home] / 2 : w->army[home] - 1) : w->army[home];
                if (warned && steps >= reach && kept <= danger) continue;
                if (edge > closest) { closest = edge; root = tops[i]; }
                if (edge > margin) { margin = edge; strike = generals_pack(tops[i], first, split); }
            }
    }
    if (strike) moves[n++] = strike;
    uint32_t focus = !strike && root != GEN_CELLS && tick >= GENERALS_ROUND && -closest <= w->total[seat] - w->land[seat] - w->army[root];
    if (threat != GEN_CELLS) {
        for (uint32_t d = 0u; d < 4u && n < 20u; d++) {
            uint32_t t = generals_neighbor(threat, d, side);
            if (t < GEN_CELLS && w->owner[t] == seat && w->army[t] - 1 > w->army[threat]) moves[n++] = generals_pack(t, d ^ 1u, 0u);
        }
        uint32_t helpers[GEN_ROWS];
        memcpy(helpers, mine, sizeof helpers);
        generals_clear(helpers, home);
        for (uint32_t tries = 0u; tries < 4u && n < 20u; tries++) {
            uint32_t best = generals_richest(w, helpers, side, GEN_CELLS);
            if (best == GEN_CELLS) break;
            generals_clear(helpers, best);
            uint32_t dir = generals_descend(o->distance, best, side);
            if (dir < 4u && o->distance[best] <= reach + 1u) moves[n++] = generals_pack(best, dir, 0u);
        }
    }
    uint32_t stack = tops[0] == home && threat != GEN_CELLS ? tops[1] : tops[0];
    if (stack == GEN_CELLS) return n;
    if (focus) {
        generals_clear(source, home);
        generals_set(source, root);
        generals_floodStart(&o->flood, o->distance, source, side);
        while (generals_floodLayer(&o->flood, o->distance, mine, side)) {}
        int32_t heaviest = 1;
        uint32_t feed = 0u;
        for (uint32_t y = 1u; y <= side; y++)
            for (uint32_t row = mine[y]; row; row &= row - 1u) {
                uint32_t c = generals_cell(y, row);
                if (c == root || w->army[c] <= heaviest || o->distance[c] == GEN_FAR || (c == home && warned)) continue;
                uint32_t dir = generals_descend(o->distance, c, side);
                if (dir < 4u) { heaviest = w->army[c]; feed = generals_pack(c, dir, 0u); }
            }
        if (feed && n < 20u) moves[n++] = feed;
    }
    int32_t might = w->army[stack] - 1;
    uint32_t prey = GEN_CELLS;
    int32_t prize = might / 4 > 8 ? might / 4 : 8;
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = w->board[y] & ~mine[y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row), far = generals_span(stack, c);
            if (w->owner[c] == GEN_NOBODY || far > GENERALS_ORACLE_WATCH / 2u || w->army[c] <= prize || might - (int32_t)far <= w->army[c]) continue;
            prize = w->army[c];
            prey = c;
        }
    if (prey != GEN_CELLS && n < 20u) {
        uint32_t targets[GEN_ROWS] = {0};
        generals_set(targets, prey);
        uint32_t dir = generals_route(side, o, targets, w->passable, stack);
        if (dir < 4u) moves[n++] = generals_pack(stack, dir, 0u);
    }
    uint32_t picks[6] = {0}, value[6] = {0}, picked = 0u;
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = mine[y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            uint32_t half = c == home && warned;
            int32_t moved = half ? w->army[c] / 2 : w->army[c] - 1;
            if (moved < 1 || (half && w->army[c] - moved <= danger)) continue;
            for (uint32_t d = 0u; d < 4u; d++) {
                uint32_t t = generals_neighbor(c, d, side);
                if (t >= GEN_CELLS || w->kind[t] == GEN_PEAK || w->owner[t] == seat || moved <= w->army[t]) continue;
                uint32_t gain = w->kind[t] == GEN_CITY ? 3u : w->owner[t] == GEN_NOBODY ? 2u : 1u, far = generals_span(cursor, c);
                if (c == stack && gain < 3u) continue;
                uint32_t score = gain << 24 | (255u - (far < 255u ? far : 255u)) << 12 | (uint32_t)(moved < 4095 ? moved : 4095);
                uint32_t slot = picked < 6u ? picked++ : 5u;
                if (slot == 5u && picked == 6u && score <= value[5]) continue;
                while (slot > 0u && value[slot - 1u] < score) {
                    picks[slot] = picks[slot - 1u];
                    value[slot] = value[slot - 1u];
                    slot--;
                }
                picks[slot] = generals_pack(c, d, half);
                value[slot] = score;
            }
        }
    for (uint32_t i = 0u; i < picked && n < 24u; i++) moves[n++] = picks[i];
    uint32_t gather = 0u, hub = focus ? root : stack;
    int32_t most = 1;
    generals_clear(source, home);
    generals_set(source, hub);
    generals_floodStart(&o->flood, o->distance, source, side);
    while (generals_floodLayer(&o->flood, o->distance, mine, side)) {}
    for (uint32_t y = 1u; y <= side; y++)
        for (uint32_t row = mine[y]; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            if (c == hub || w->army[c] <= most || o->distance[c] == GEN_FAR || (c == home && warned)) continue;
            uint32_t dir = generals_descend(o->distance, c, side);
            if (dir < 4u) { most = w->army[c]; gather = generals_pack(c, dir, 0u); }
        }
    uint32_t targets[GEN_ROWS] = {0}, walk[GEN_ROWS] = {0};
    for (uint32_t y = 1u; y <= side; y++) {
        uint32_t rim = (mine[y - 1u] | mine[y + 1u] | mine[y] << 1 | mine[y] >> 1) & w->passable[y] & ~mine[y];
        for (uint32_t row = rim; row; row &= row - 1u) {
            uint32_t c = generals_cell(y, row);
            if (w->army[c] < might) targets[y] |= 1u << (c & 31u);
        }
        walk[y] = (mine[y] | targets[y]) & w->passable[y];
    }
    uint32_t dir = generals_route(side, o, targets, walk, stack);
    if (gather && most * 4 > might && n < 24u) moves[n++] = gather;
    if (dir < 4u && n < 24u) moves[n++] = generals_pack(stack, dir, 0u);
    if (gather && n < 24u) moves[n++] = gather;
    return n;
}

static uint32_t generals_steer(const generalsOracle *o, uint32_t cursor, uint32_t mode, uint32_t code, uint8_t *press, uint32_t budget) {
    uint32_t n = 0u, source = code & (GEN_CELLS - 1u), dir = code >> 10 & 3u, target = code & GEN_SPLIT_BIT ? GEN_HALF : GEN_ALL;
    if (cursor != source) {
        for (; mode != GEN_LOOK; mode = (mode + 1u) % 3u) press[n++] = 4u;
        uint32_t at = GENERALS_JUMPS, best = generals_span(cursor, source), via = GENERALS_JUMPS, hops = 0u;
        for (uint32_t i = 0u; i < o->jumps; i++)
            if (o->jump[i] == cursor) at = i;
        for (uint32_t i = 0u; i < o->jumps; i++) {
            if (i == at) continue;
            uint32_t cost = at < o->jumps ? (i + o->jumps - at) % o->jumps : i + 1u, total = cost + generals_span(o->jump[i], source);
            if (total < best) { best = total; via = i; hops = cost; }
        }
        if (n + best + 3u > budget) return budget + 1u;
        for (uint32_t c = 0u; c < hops; c++) press[n++] = 5u;
        if (via < GENERALS_JUMPS) cursor = o->jump[via];
        int dx = (int)(source & 31u) - (int)(cursor & 31u), dy = (int)(source >> 5) - (int)(cursor >> 5);
        for (; dx > 0; dx--) press[n++] = 3u;
        for (; dx < 0; dx++) press[n++] = 2u;
        for (; dy > 0; dy--) press[n++] = 1u;
        for (; dy < 0; dy++) press[n++] = 0u;
    }
    for (; mode != target; mode = (mode + 1u) % 3u) press[n++] = 4u;
    press[n++] = (uint8_t)dir;
    return n;
}

static void generals_plan(const generalsKnobs *k, const generalsWorld *w, const uint32_t *words, generalsOracle *o) {
    uint32_t moves[24], budget = GENERALS_SUBSTEPS - words[GW_PHASE];
    uint8_t press[GEN_ROWS * 2];
    o->jumps = generals_rank(k, w, words[GW_SEAT], o->jump);
    uint32_t count = generals_advice(k, w, words, o, moves);
    o->count = 0u;
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t n = generals_steer(o, words[GW_CURSOR], words[GW_MODE], moves[i], press, budget);
        if (n > budget) continue;
        memcpy(o->press, press, n);
        o->count = n;
        return;
    }
}

void generals_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const generalsKnobs *k = &GENERALS_KNOBS[instance];
    generalsWorld *w = generals_world(extra);
    generalsOracle *o = &w->oracle;
    if (!o->ready || o->game != words[GW_GAME] || o->tick != words[GW_TICK]) {
        generals_plan(k, w, words, o);
        o->game = words[GW_GAME];
        o->tick = words[GW_TICK];
        o->ready = 1u;
        o->start = words[GW_PHASE];
    }
    uint32_t at = words[GW_PHASE] - o->start;
    ctrl[0] = at < o->count ? (float)o->press[at] : (float)(at & 1u);
}

uint32_t generals_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    const generalsKnobs *k = &GENERALS_KNOBS[instance];
    if (!out) return k->side * k->side;
    const generalsWorld *w = generals_read(extra);
    for (uint32_t y = 0u; y < k->side; y++)
        for (uint32_t x = 0u; x < k->side; x++) {
            uint32_t cell = y << 5 | x, kind = w->kind[cell], terrain = kind == GEN_PEAK ? 1u : kind == GEN_CAPITAL ? 3u : kind == GEN_CITY ? 2u : 0u;
            uint32_t army = (uint32_t)w->army[cell];
            out[y * k->side + x] = (army & 0xFFFFu) | ((army >> 16 & 0x3FFu) | terrain << 10 | (uint32_t)w->owner[cell] << 12) << 16;
        }
    return k->side * k->side;
}
