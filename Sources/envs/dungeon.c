#include "../env.h"

typedef struct { uint32_t floors, radius, dark, shuffle, boss, guardian; } dungeonKnobs;

static const dungeonKnobs DUNGEON_KNOBS[4] = {
    { 6u, 2u, 0u, 0u, 14u, 0u },
    { 6u, 1u, 1u, 0u, 14u, 0u },
    { 6u, 1u, 1u, 1u, 14u, 0u },
    { 9u, 1u, 1u, 1u, 15u, 6u }
};

static const uint32_t DUNGEON_HP[16] = { 3u, 4u, 6u, 9u, 8u, 7u, 10u, 12u, 16u, 12u, 24u, 18u, 28u, 22u, 44u, 80u };
static const uint32_t DUNGEON_ATTACK[16] = { 2u, 2u, 3u, 4u, 4u, 3u, 5u, 5u, 6u, 5u, 7u, 7u, 8u, 7u, 8u, 10u };
static const uint32_t DUNGEON_DEFENCE[16] = { 0u, 1u, 1u, 0u, 1u, 2u, 2u, 2u, 3u, 3u, 4u, 3u, 4u, 4u, 4u, 5u };
static const uint32_t DUNGEON_SPEED[16] = { 12u, 18u, 12u, 12u, 15u, 12u, 15u, 12u, 12u, 15u, 12u, 12u, 12u, 12u, 12u, 12u };
static const uint32_t DUNGEON_DAMAGE[16] = { 0u, 0u, 0u, 3u, 0u, 0u, 1u, 2u, 0u, 3u, 0u, 1u, 2u, 4u, 0u, 1u };
static const uint32_t DUNGEON_BEHAVIOUR[16] = { 0u, 1u, 0u, 3u, 0u, 2u, 1u, 0u, 0u, 0u, 0u, 2u, 2u, 0u, 0u, 2u };
static const uint32_t DUNGEON_XP[16] = { 1u, 2u, 3u, 4u, 5u, 6u, 8u, 9u, 12u, 12u, 20u, 20u, 30u, 30u, 50u, 100u };
static const uint32_t DUNGEON_FLOOR[16] = { 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 6u, 9u };

enum { DUNGEON_SIDE = 32, DUNGEON_CELLS = DUNGEON_SIDE * DUNGEON_SIDE, DUNGEON_SLOTS = 32, DUNGEON_ALIVE = 24, DUNGEON_ITEMS = 16, DUNGEON_REGULAR = 14, DUNGEON_VIEW = 160 };
enum { DUNGEON_REACH = 3, DUNGEON_SPAN = 2 * DUNGEON_REACH + 1, DUNGEON_PATCH = DUNGEON_SPAN * DUNGEON_SPAN };
enum { DUNGEON_WARD = 300, DUNGEON_DAZE = 12, DUNGEON_DARK = 40, DUNGEON_ILL = 20 };
enum { DW_X, DW_Y, DW_HP, DW_LEVEL, DW_XP, DW_FOOD, DW_DEPTH, DW_TURN, DW_RUN, DW_SLOT, DW_RATIONS, DW_PACK, DW_PACK_HIGH, DW_POISON, DW_CONFUSED, DW_BLIND, DW_FIRE, DW_COLD,
       DW_EVENT, DW_MAGNITUDE, DW_GOLD, DW_SCORE, DW_PERM, DW_ROBIN, DW_MONSTERS, DW_ITEMS, DW_STAIRS, DW_LIT, DW_DEEPEST, DW_WORDS };
enum { DO_HEALTH = DUNGEON_PATCH, DO_FOOD, DO_LEVEL, DO_DEPTH, DO_MESSAGE, DO_SLOT, DO_RATIONS, DO_POTIONS, DO_COUNT = DO_POTIONS + 8 };
enum { DT_ROCK, DT_WALL, DT_FLOOR, DT_CORRIDOR, DT_DOOR, DT_STAIRS };
enum { DC_TERRAIN = 7, DC_ITEM = 15 << 3, DC_WHO = 31 << 7 };
enum { DI_GOLD = 1, DI_RATION, DI_POTION };
enum { DG_ITEM = DT_STAIRS, DG_MONSTER = DG_ITEM + DI_POTION + 8, DG_UNSEEN = DG_MONSTER + 16 };
enum { DK_SEEN = 1, DK_OPEN = 2, DK_ITEM = 4, DK_STILL = 8, DK_STAIRS = 16 };
_Static_assert(DO_COUNT <= OAK_ENV_OBS_MAX && DW_WORDS <= OAK_ENV_WORDS_MAX, "");

static const float DUNGEON_GLYPH[DG_UNSEEN + 1] = { -0.5f, -1.0f, 0.5f, 1.0f, 1.5f, 2.5f,
                                                    3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
                                                    -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f, -8.0f, -8.5f, -9.0f, -9.5f,
                                                    0.0f };
enum { DE_HEAL, DE_EXTRA, DE_CURE, DE_FIRE, DE_COLD, DE_POISON, DE_CONFUSION, DE_BLINDNESS };
enum { DB_CHASE, DB_ERRATIC, DB_RANGED, DB_STILL };
enum { DD_PHYSICAL, DD_FIRE, DD_COLD, DD_POISON, DD_DRAIN };
enum { DV_NONE, DV_HIT, DV_MISS, DV_KILL, DV_HURT, DV_GOLD, DV_RATION, DV_POTION, DV_EAT, DV_QUAFF, DV_DESCEND = DV_QUAFF + 8, DV_REST, DV_BUMP, DV_STARVE, DV_SICK, DV_RUN, DV_WIN };
enum { DM_AWAKE = 1, DM_GUARD = 2 };
enum { DP_ADJACENT = 1, DP_STILL = 2, DP_ERRATIC = 4, DP_AIM = 8, DP_RANGED = 16 };
enum { DUNGEON_ACT = 4, DUNGEON_NEXT = 5, DUNGEON_USE = 8 };

typedef struct { uint32_t word[16], next, fill, seed, pad; } dungeonDice;

typedef struct {
    uint8_t known[DUNGEON_CELLS], lore[8];
    uint32_t floor, run, turn, health, quaffed, boss, sick, fire, cold, dizzy;
} dungeonMind;

typedef struct {
    uint16_t cell[DUNGEON_CELLS];
    uint8_t mx[DUNGEON_SLOTS], my[DUNGEON_SLOTS], type[DUNGEON_SLOTS], hp[DUNGEON_SLOTS], energy[DUNGEON_SLOTS], speed[DUNGEON_SLOTS], flag[DUNGEON_SLOTS], kind[DUNGEON_SLOTS];
    uint8_t ix[DUNGEON_ITEMS], iy[DUNGEON_ITEMS], gold[DUNGEON_ITEMS];
    uint8_t rx0[16], ry0[16], rx1[16], ry1[16];
    dungeonDice dice;
    dungeonMind mind;
} dungeonWorld;

typedef struct { uint32_t seed, run, depth, block, at; uint16_t half[32]; } dungeonDraw;
typedef struct { uint8_t first[DUNGEON_SLOTS], second[DUNGEON_SLOTS], code[DUNGEON_SLOTS]; } dungeonPlan;
typedef struct { uint32_t x, y, radius, x0, y0, x1, y1, blind; } dungeonSight;

static dungeonWorld *dungeon_world(uint32_t *extra) { return (dungeonWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const dungeonWorld *dungeon_read(const uint32_t *extra) { return (const dungeonWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }

uint32_t dungeon_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)(sizeof(dungeonWorld) / sizeof(uint32_t)) + 4u;
}

static uint32_t dungeon_toward(uint32_t at, uint32_t direction) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    return at + (uint32_t)(dy * DUNGEON_SIDE + dx);
}

static uint32_t dungeon_pick(dungeonDraw *g, uint32_t n) {
    if (g->at == 32u) {
        u44 r = philox4(g->seed, ENV_RNG_STREAM, (lane4)g->run, (lane4)g->depth, (lane4)(g->block << 2 | 1u), (lane4){ 0u, 1u, 2u, 3u });
        __builtin_memcpy(g->half, &r, sizeof g->half);
        g->block++;
        g->at = 0u;
    }
    return (uint32_t)g->half[g->at++] * n >> 16;
}

static __attribute__((noinline)) void dungeon_refill(dungeonDice *d) {
    u44 r = philox4(d->seed, ENV_RNG_STREAM, (lane4)d->fill, (lane4){ 0u, 1u, 2u, 3u }, (lane4)3u, (lane4)0u);
    vst4q_u32(d->word, ((uint32x4x4_t){ { (uint32x4_t)r.x, (uint32x4_t)r.y, (uint32x4_t)r.z, (uint32x4_t)r.w } }));
    d->fill++;
    d->next = 0u;
}

static inline __attribute__((always_inline)) uint32_t dungeon_roll(dungeonDice *d) {
    if (d->next == 16u) dungeon_refill(d);
    return d->word[d->next++];
}

static uint32_t dungeon_hits(uint32_t roll, uint32_t attack, uint32_t defence) {
    uint32_t d20 = 1u + ((roll >> 16) * 20u >> 16);
    return d20 == 20u || (d20 != 1u && d20 + attack >= 11u + defence);
}

static uint32_t dungeon_damage(uint32_t roll, uint32_t attack) { return 1u + ((roll & 0xFFFFu) * attack >> 16); }

static uint32_t dungeon_most(uint32_t level) { return 8u + 4u * level; }

static uint32_t dungeon_potions(const uint32_t *words, uint32_t colour) { return words[DW_PACK + (colour >> 2)] >> (colour & 3u) * 8u & 255u; }

static void dungeon_note(uint32_t *words, uint32_t event, uint32_t magnitude) {
    if (words[DW_EVENT] != DV_NONE) return;
    words[DW_EVENT] = event;
    words[DW_MAGNITUDE] = magnitude;
}

static uint32_t dungeon_shuffle(uint32_t seed, uint32_t run) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, run, 0u, 2u, 0u);
    uint32_t draw[4] = { r.x, r.y, r.z, r.w }, perm = 0u;
    uint8_t colour[8] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
    for (uint32_t i = 7u; i > 0u; i--) {
        uint32_t k = 7u - i, half = draw[k >> 1] >> (k & 1u) * 16u & 0xFFFFu, j = half * (i + 1u) >> 16;
        uint8_t swap = colour[i];
        colour[i] = colour[j];
        colour[j] = swap;
    }
    for (uint32_t c = 0u; c < 8u; c++) perm |= (uint32_t)colour[c] << 3u * c;
    return perm;
}

static uint32_t dungeon_kinds(uint32_t depth, uint8_t *list) {
    uint32_t count = 0u;
    for (uint32_t t = 0u; t < DUNGEON_REGULAR; t++)
        if (DUNGEON_FLOOR[t] <= depth && DUNGEON_FLOOR[t] + 3u >= depth) list[count++] = (uint8_t)t;
    return count;
}

static void dungeon_spawn(dungeonWorld *w, uint32_t *words, uint32_t type, uint32_t at, uint32_t flag) {
    uint32_t n = words[DW_MONSTERS];
    w->mx[n] = (uint8_t)(at & 31u);
    w->my[n] = (uint8_t)(at >> 5);
    w->type[n] = (uint8_t)type;
    w->hp[n] = (uint8_t)DUNGEON_HP[type];
    w->energy[n] = 0u;
    w->speed[n] = (uint8_t)DUNGEON_SPEED[type];
    w->flag[n] = (uint8_t)flag;
    uint32_t behaviour = flag & DM_GUARD ? DB_STILL : DUNGEON_BEHAVIOUR[type];
    w->kind[n] = (uint8_t)(behaviour == DB_STILL ? DP_STILL : behaviour == DB_ERRATIC ? DP_ERRATIC : behaviour == DB_RANGED ? DP_RANGED : 0u);
    w->cell[at] = (uint16_t)(w->cell[at] | (n + 1u) << 7);
    words[DW_MONSTERS] = n + 1u;
}

static void dungeon_remove(dungeonWorld *w, uint32_t *words, uint32_t m) {
    uint32_t last = words[DW_MONSTERS] - 1u;
    w->cell[w->my[m] * DUNGEON_SIDE + w->mx[m]] &= (uint16_t)~DC_WHO;
    if (m != last) {
        w->mx[m] = w->mx[last]; w->my[m] = w->my[last]; w->type[m] = w->type[last]; w->hp[m] = w->hp[last];
        w->energy[m] = w->energy[last]; w->speed[m] = w->speed[last]; w->flag[m] = w->flag[last]; w->kind[m] = w->kind[last];
        uint32_t at = w->my[m] * DUNGEON_SIDE + w->mx[m];
        w->cell[at] = (uint16_t)((w->cell[at] & ~DC_WHO) | (m + 1u) << 7);
    }
    words[DW_MONSTERS] = last;
}

static void dungeon_join(dungeonWorld *w, dungeonDraw *g, uint32_t a, uint32_t b, uint32_t vertical) {
    const uint8_t *low = vertical ? w->rx0 : w->ry0, *high = vertical ? w->rx1 : w->ry1, *end = vertical ? w->ry1 : w->rx1, *start = vertical ? w->ry0 : w->rx0;
    uint32_t along = vertical ? DUNGEON_SIDE : 1u, across = vertical ? 1u : DUNGEON_SIDE;
    uint32_t sa = low[a] + dungeon_pick(g, high[a] - low[a] + 1u), sb = low[b] + dungeon_pick(g, high[b] - low[b] + 1u);
    uint32_t from = end[a] + 2u, to = start[b] - 2u, bend = from + dungeon_pick(g, to - from + 1u);
    uint16_t *doorA = &w->cell[(from - 1u) * along + sa * across], *doorB = &w->cell[(to + 1u) * along + sb * across];
    *doorA = (uint16_t)((*doorA & ~DC_TERRAIN) | DT_DOOR);
    *doorB = (uint16_t)((*doorB & ~DC_TERRAIN) | DT_DOOR);
    for (uint32_t t = from; t <= bend; t++) w->cell[t * along + sa * across] = DT_CORRIDOR;
    for (uint32_t s = sa < sb ? sa : sb; s <= (sa < sb ? sb : sa); s++) w->cell[bend * along + s * across] = DT_CORRIDOR;
    for (uint32_t t = bend; t <= to; t++) w->cell[t * along + sb * across] = DT_CORRIDOR;
}

static uint32_t dungeon_spot(const dungeonWorld *w, dungeonDraw *g, uint32_t room) {
    uint32_t x = w->rx0[room] + dungeon_pick(g, w->rx1[room] - w->rx0[room] + 1u), y = w->ry0[room] + dungeon_pick(g, w->ry1[room] - w->ry0[room] + 1u);
    return y * DUNGEON_SIDE + x;
}

static void dungeon_floor(const dungeonKnobs *k, uint32_t seed, uint32_t *words, dungeonWorld *w) {
    uint32_t depth = words[DW_DEPTH];
    dungeonDraw g = { seed, words[DW_RUN], depth, 0u, 32u, { 0u } };
    __builtin_memset(w->cell, 0, sizeof w->cell);
    for (uint32_t r = 0u; r < 9u; r++) {
        uint32_t x0 = 1u + 10u * (r % 3u), y0 = 1u + 10u * (r / 3u), width = 3u + dungeon_pick(&g, 5u), height = 3u + dungeon_pick(&g, 5u);
        uint32_t left = x0 + 1u + dungeon_pick(&g, 8u - width), top = y0 + 1u + dungeon_pick(&g, 8u - height);
        w->rx0[r] = (uint8_t)left; w->ry0[r] = (uint8_t)top; w->rx1[r] = (uint8_t)(left + width - 1u); w->ry1[r] = (uint8_t)(top + height - 1u);
        for (uint32_t y = top - 1u; y <= top + height; y++)
            for (uint32_t x = left - 1u; x <= left + width; x++) {
                uint32_t inside = x >= left && x < left + width && y >= top && y < top + height;
                w->cell[y * DUNGEON_SIDE + x] = (uint16_t)((inside ? DT_FLOOR : DT_WALL) | (r + 1u) << 12);
            }
    }
    uint8_t order[12] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u }, parent[9] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u };
    for (uint32_t i = 11u; i > 0u; i--) {
        uint32_t j = dungeon_pick(&g, i + 1u);
        uint8_t swap = order[i];
        order[i] = order[j];
        order[j] = swap;
    }
    for (uint32_t i = 0u, loops = 0u; i < 12u; i++) {
        uint32_t e = order[i], vertical = e >= 6u, a = vertical ? e - 6u : e / 2u * 3u + e % 2u, b = a + (vertical ? 3u : 1u), ra = a, rb = b;
        while (parent[ra] != ra) ra = parent[ra];
        while (parent[rb] != rb) rb = parent[rb];
        if (ra != rb) parent[ra] = (uint8_t)rb;
        else if (loops < 2u) loops++;
        else continue;
        dungeon_join(w, &g, a, b, vertical);
    }
    uint32_t arrive = dungeon_pick(&g, 9u), home = (arrive + 1u + dungeon_pick(&g, 8u)) % 9u, start = dungeon_spot(w, &g, arrive), goal = dungeon_spot(w, &g, home);
    uint32_t last = depth == k->floors;
    words[DW_X] = start & 31u;
    words[DW_Y] = start >> 5;
    words[DW_STAIRS] = last ? 0u : goal;
    if (!last) w->cell[goal] = (uint16_t)((w->cell[goal] & ~DC_TERRAIN) | DT_STAIRS);
    words[DW_LIT] = k->dark ? dungeon_pick(&g, 512u) : 511u;
    words[DW_MONSTERS] = 0u;
    words[DW_ITEMS] = 0u;
    words[DW_ROBIN] = 0u;
    if (last) dungeon_spawn(w, words, k->boss, goal, 0u);
    if (depth == k->guardian) dungeon_spawn(w, words, 14u, goal, DM_GUARD);
    uint8_t kinds[DUNGEON_REGULAR];
    uint32_t count = dungeon_kinds(depth, kinds);
    for (uint32_t i = 0u; i < 3u + depth; i++)
        for (uint32_t attempt = 0u; attempt < 4u; attempt++) {
            uint32_t at = dungeon_spot(w, &g, (arrive + 1u + dungeon_pick(&g, 8u)) % 9u);
            if ((w->cell[at] & DC_WHO) || at == goal) continue;
            dungeon_spawn(w, words, kinds[dungeon_pick(&g, count)], at, 0u);
            break;
        }
    for (uint32_t i = 0u, items = 6u + depth / 3u; i < items; i++)
        for (uint32_t attempt = 0u; attempt < 4u; attempt++) {
            uint32_t at = dungeon_spot(w, &g, dungeon_pick(&g, 9u));
            if ((w->cell[at] & DC_ITEM) || at == goal || at == start) continue;
            uint32_t n = words[DW_ITEMS], kind = i < 3u ? DI_GOLD : i == 3u ? DI_RATION : DI_POTION + dungeon_pick(&g, 8u);
            w->ix[n] = (uint8_t)(at & 31u);
            w->iy[n] = (uint8_t)(at >> 5);
            w->gold[n] = (uint8_t)(kind == DI_GOLD ? 1u + dungeon_pick(&g, 10u * depth) : 0u);
            w->cell[at] = (uint16_t)(w->cell[at] | kind << 3);
            words[DW_ITEMS] = n + 1u;
            break;
        }
}

static void dungeon_run(const dungeonKnobs *k, uint32_t seed, uint32_t *words, dungeonWorld *w, uint32_t event) {
    for (uint32_t i = DW_HP; i < DW_WORDS; i++)
        if (i != DW_RUN && i != DW_PERM && i != DW_DEEPEST) words[i] = 0u;
    words[DW_LEVEL] = 1u;
    words[DW_HP] = dungeon_most(1u);
    words[DW_FOOD] = 1000u;
    words[DW_DEPTH] = 1u;
    words[DW_EVENT] = event;
    if (k->shuffle) words[DW_PERM] = dungeon_shuffle(seed, words[DW_RUN]);
    dungeon_floor(k, seed, words, w);
}

static dungeonSight dungeon_sight(const dungeonKnobs *k, const uint32_t *words, const dungeonWorld *w) {
    uint32_t x = words[DW_X], y = words[DW_Y], room = (uint32_t)w->cell[y * DUNGEON_SIDE + x] >> 12;
    dungeonSight s = { x, y, k->radius, 255u, 255u, 0u, 0u, words[DW_BLIND] != 0u };
    if (room && (words[DW_LIT] >> (room - 1u) & 1u)) {
        s.x0 = w->rx0[room - 1u] - 1u; s.y0 = w->ry0[room - 1u] - 1u;
        s.x1 = w->rx1[room - 1u] + 1u; s.y1 = w->ry1[room - 1u] + 1u;
    }
    return s;
}

static uint32_t dungeon_lit(const dungeonSight *s, uint32_t x, uint32_t y) {
    uint32_t dx = x > s->x ? x - s->x : s->x - x, dy = y > s->y ? y - s->y : s->y - y;
    return (dx > dy ? dx : dy) <= s->radius || (x >= s->x0 && x <= s->x1 && y >= s->y0 && y <= s->y1);
}

static int dungeon_span(const dungeonSight *s, int dy, int *hi) {
    uint32_t ay = (uint32_t)(dy < 0 ? -dy : dy), y = s->y + (uint32_t)dy;
    int half = s->blind ? (ay == 0u) - (ay > 1u) : ay <= s->radius ? (int)s->radius : -1, lo = -half;
    *hi = half;
    if (!s->blind && y >= s->y0 && y <= s->y1) {
        int left = (int)s->x0 - (int)s->x, right = (int)s->x1 - (int)s->x;
        lo = left < lo ? left : lo;
        *hi = right > half ? right : half;
    }
    *hi = *hi < DUNGEON_REACH ? *hi : DUNGEON_REACH;
    return ay > DUNGEON_REACH ? DUNGEON_REACH + 1 : lo > -DUNGEON_REACH ? lo : -DUNGEON_REACH;
}

static uint32_t dungeon_visible(const dungeonSight *s, int dx, int dy) {
    int hi, lo = dungeon_span(s, dy, &hi);
    return dx >= lo && dx <= hi;
}

static inline uint8x16_t dungeon_lit16(const dungeonSight *s, uint8x16_t x, uint8x16_t y) {
    uint8x16_t dx = vabdq_u8(x, vdupq_n_u8((uint8_t)s->x)), dy = vabdq_u8(y, vdupq_n_u8((uint8_t)s->y));
    uint8x16_t across = vandq_u8(vcgeq_u8(x, vdupq_n_u8((uint8_t)s->x0)), vcleq_u8(x, vdupq_n_u8((uint8_t)s->x1)));
    uint8x16_t down = vandq_u8(vcgeq_u8(y, vdupq_n_u8((uint8_t)s->y0)), vcleq_u8(y, vdupq_n_u8((uint8_t)s->y1)));
    return vorrq_u8(vcleq_u8(vmaxq_u8(dx, dy), vdupq_n_u8((uint8_t)s->radius)), vandq_u8(across, down));
}

static inline uint8x16_t dungeon_valid(uint32_t half, uint32_t n) {
    static const uint8_t INDEX[32] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u };
    return vcltq_u8(vld1q_u8(INDEX + 16u * half), vdupq_n_u8((uint8_t)n));
}

static inline uint32_t dungeon_bits(uint8x16_t set) {
    static const uint8_t WEIGHT[16] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u };
    uint8x16_t m = vandq_u8(set, vld1q_u8(WEIGHT));
    return (uint32_t)vaddv_u8(vget_low_u8(m)) | (uint32_t)vaddv_u8(vget_high_u8(m)) << 8;
}

static void dungeon_rouse(const dungeonKnobs *k, const uint32_t *words, dungeonWorld *w) {
    dungeonSight s = dungeon_sight(k, words, w);
    uint32_t n = words[DW_MONSTERS];
    for (uint32_t h = 0u; h < (n > 16u ? 2u : 1u); h++) {
        uint8x16_t lit = vandq_u8(dungeon_lit16(&s, vld1q_u8(w->mx + 16u * h), vld1q_u8(w->my + 16u * h)), dungeon_valid(h, n));
        vst1q_u8(w->flag + 16u * h, vorrq_u8(vld1q_u8(w->flag + 16u * h), vandq_u8(lit, vdupq_n_u8(DM_AWAKE))));
    }
}

static uint32_t dungeon_count(const uint32_t *words, uint32_t slot) { return slot == 1u ? words[DW_RATIONS] : dungeon_potions(words, slot - 2u); }

static float dungeon_glyph(const dungeonWorld *w, uint32_t at) {
    uint32_t c = w->cell[at], who = c >> 7 & 31u, item = c >> 3 & 15u;
    return DUNGEON_GLYPH[who ? DG_MONSTER + w->type[who - 1u] : item ? DG_ITEM + item : c & DC_TERRAIN];
}

static void dungeon_obs(const dungeonKnobs *k, const uint32_t *words, const dungeonWorld *w, float *obs) {
    dungeonSight s = dungeon_sight(k, words, w);
    uint32_t level = words[DW_LEVEL];
    for (int dy = -DUNGEON_REACH; dy <= DUNGEON_REACH; dy++) {
        float *row = obs + (dy + DUNGEON_REACH) * DUNGEON_SPAN + DUNGEON_REACH;
        uint32_t centre = (s.y + (uint32_t)dy) * DUNGEON_SIDE + s.x;
        int hi, lo = dungeon_span(&s, dy, &hi);
        for (int dx = -DUNGEON_REACH; dx <= DUNGEON_REACH; dx++) row[dx] = DUNGEON_GLYPH[DG_UNSEEN];
        for (int dx = lo; dx <= hi; dx++) row[dx] = dungeon_glyph(w, centre + (uint32_t)dx);
    }
    obs[DO_HEALTH] = (float)(int32_t)words[DW_HP] / (float)dungeon_most(level);
    obs[DO_FOOD] = (float)words[DW_FOOD] * 0.001f;
    obs[DO_LEVEL] = (float)level * 0.1f;
    obs[DO_DEPTH] = (float)words[DW_DEPTH] * (1.0f / (float)k->floors);
    obs[DO_MESSAGE] = (float)words[DW_EVENT] * 0.25f;
    obs[DO_SLOT] = (float)words[DW_SLOT] * 0.25f;
    obs[DO_RATIONS] = (float)words[DW_RATIONS] * 0.25f;
    for (uint32_t c = 0u; c < 8u; c++) obs[DO_POTIONS + c] = (float)dungeon_potions(words, c) * 0.25f;
}

static uint32_t dungeon_player(const uint32_t *words) { return words[DW_Y] * DUNGEON_SIDE + words[DW_X]; }

static uint32_t dungeon_free(const dungeonWorld *w, uint32_t at, uint32_t player) {
    uint32_t c = w->cell[at];
    return (c & DC_TERRAIN) >= DT_FLOOR && !(c & DC_WHO) && at != player;
}

static uint32_t dungeon_go(dungeonWorld *w, uint32_t m, uint32_t from, uint32_t to, uint32_t player) {
    if (!dungeon_free(w, to, player)) return 0u;
    w->cell[to] = (uint16_t)(w->cell[to] | (w->cell[from] & DC_WHO));
    w->cell[from] &= (uint16_t)~DC_WHO;
    w->mx[m] = (uint8_t)(to & 31u);
    w->my[m] = (uint8_t)(to >> 5);
    return 1u;
}

static uint32_t dungeon_clear(const dungeonWorld *w, uint32_t from, uint32_t to, uint32_t stride) {
    for (uint32_t at = from + stride; at != to; at += stride) {
        uint32_t c = w->cell[at];
        if ((c & DC_TERRAIN) < DT_FLOOR || (c & DC_WHO)) return 0u;
    }
    return 1u;
}

static inline __attribute__((always_inline)) void dungeon_bite(dungeonDice *d, uint32_t *words, uint32_t type) {
    uint32_t level = words[DW_LEVEL], roll = dungeon_roll(d), attack = DUNGEON_ATTACK[type], kind = DUNGEON_DAMAGE[type];
    if (!dungeon_hits(roll, attack, level >> 1)) return;
    uint32_t damage = dungeon_damage(roll, attack);
    if ((kind == DD_FIRE && words[DW_FIRE]) || (kind == DD_COLD && words[DW_COLD])) damage /= 3u;
    words[DW_HP] -= damage;
    if (kind == DD_POISON && words[DW_POISON] < DUNGEON_ILL) words[DW_POISON] = DUNGEON_ILL;
    if (kind == DD_DRAIN && level > 1u && (roll >> 16) * 20u >> 16 >= 15u) {
        level--;
        words[DW_LEVEL] = level;
        words[DW_XP] = level > 1u ? 10u << (level - 2u) : 0u;
        if ((int32_t)words[DW_HP] > (int32_t)dungeon_most(level)) words[DW_HP] = dungeon_most(level);
    }
    dungeon_note(words, DV_HURT, damage);
}

static void dungeon_plan(const dungeonWorld *w, const uint32_t *words, uint32_t n, dungeonPlan *p) {
    uint8x16_t px = vdupq_n_u8((uint8_t)words[DW_X]), py = vdupq_n_u8((uint8_t)words[DW_Y]);
    for (uint32_t h = 0u; h < (n > 16u ? 2u : 1u); h++) {
        uint8x16_t x = vld1q_u8(w->mx + 16u * h), y = vld1q_u8(w->my + 16u * h), kind = vld1q_u8(w->kind + 16u * h);
        uint8x16_t dx = vabdq_u8(x, px), dy = vabdq_u8(y, py), sum = vaddq_u8(dx, dy), flat = vcgeq_u8(dx, dy);
        uint8x16_t across = vbslq_u8(vcgtq_u8(px, x), vdupq_n_u8(1u), vdupq_n_u8(255u)), down = vbslq_u8(vcgtq_u8(py, y), vdupq_n_u8(DUNGEON_SIDE), vdupq_n_u8(256u - DUNGEON_SIDE));
        uint8x16_t line = vandq_u8(vceqq_u8(vminq_u8(dx, dy), vdupq_n_u8(0u)), vcleq_u8(sum, vdupq_n_u8(5u)));
        uint8x16_t code = vorrq_u8(vandq_u8(vceqq_u8(sum, vdupq_n_u8(1u)), vdupq_n_u8(DP_ADJACENT)), vandq_u8(kind, vdupq_n_u8(DP_STILL | DP_ERRATIC)));
        vst1q_u8(p->first + 16u * h, vbslq_u8(flat, across, down));
        vst1q_u8(p->second + 16u * h, vbslq_u8(flat, vandq_u8(down, vtstq_u8(dy, dy)), vandq_u8(across, vtstq_u8(dx, dx))));
        vst1q_u8(p->code + 16u * h, vorrq_u8(code, vandq_u8(vandq_u8(vtstq_u8(kind, vdupq_n_u8(DP_RANGED)), line), vdupq_n_u8(DP_AIM))));
    }
}

static uint32_t dungeon_stride(uint8_t offset) { return (uint32_t)(int32_t)(int8_t)offset; }

static void dungeon_act(dungeonDice *d, uint32_t *words, dungeonWorld *w, uint32_t m, const dungeonPlan *p) {
    uint32_t code = p->code[m], from = w->my[m] * DUNGEON_SIDE + w->mx[m], player = dungeon_player(words);
    if (code & DP_ERRATIC) {
        uint32_t coin = dungeon_roll(d);
        if (coin >> 31) {
            dungeon_go(w, m, from, dungeon_toward(from, coin & 3u), player);
            return;
        }
    }
    if (code & DP_ADJACENT) { dungeon_bite(d, words, w->type[m]); return; }
    if (code & DP_STILL) return;
    uint32_t first = dungeon_stride(p->first[m]);
    if ((code & DP_AIM) && dungeon_clear(w, from, player, first)) { dungeon_bite(d, words, w->type[m]); return; }
    if (!dungeon_go(w, m, from, from + first, player) && p->second[m]) dungeon_go(w, m, from, from + dungeon_stride(p->second[m]), player);
}

static uint32_t dungeon_energize(dungeonWorld *w, uint32_t n, uint32_t feed) {
    uint32_t ready = 0u;
    for (uint32_t h = 0u; h < (n > 16u ? 2u : 1u); h++) {
        uint8x16_t awake = vandq_u8(vtstq_u8(vld1q_u8(w->flag + 16u * h), vdupq_n_u8(DM_AWAKE)), dungeon_valid(h, n)), energy = vld1q_u8(w->energy + 16u * h);
        if (feed) {
            energy = vminq_u8(vqaddq_u8(energy, vandq_u8(vld1q_u8(w->speed + 16u * h), awake)), vdupq_n_u8(24u));
            vst1q_u8(w->energy + 16u * h, energy);
        }
        ready |= dungeon_bits(vandq_u8(awake, vcgeq_u8(energy, vdupq_n_u8(12u)))) << 16u * h;
    }
    return ready;
}

static void dungeon_monsters(dungeonDice *d, uint32_t *words, dungeonWorld *w) {
    uint32_t n = words[DW_MONSTERS];
    if (!n) return;
    uint32_t start = words[DW_ROBIN] < n ? words[DW_ROBIN] : 0u, acted = 0u;
    for (uint32_t ready = dungeon_energize(w, n, 1u); ready; ready = dungeon_energize(w, n, 0u)) {
        dungeonPlan plan;
        dungeon_plan(w, words, n, &plan);
        for (uint32_t order = start ? ready >> start | ready << (32u - start) : ready; order; order &= order - 1u) {
            uint32_t m = ((uint32_t)__builtin_ctz(order) + start) & 31u;
            w->energy[m] = (uint8_t)(w->energy[m] - 12u);
            dungeon_act(d, words, w, m, &plan);
            if (++acted == 8u) { words[DW_ROBIN] = m + 1u; return; }
            if ((int32_t)words[DW_HP] <= 0) return;
        }
    }
}

static uint32_t dungeon_strike(const dungeonKnobs *k, dungeonDice *d, uint32_t *words, dungeonWorld *w, uint32_t m, uint32_t *won) {
    uint32_t level = words[DW_LEVEL], attack = 2u + 2u * level, type = w->type[m], roll = dungeon_roll(d);
    w->flag[m] |= DM_AWAKE;
    if (!dungeon_hits(roll, attack, DUNGEON_DEFENCE[type])) { dungeon_note(words, DV_MISS, 0u); return 0u; }
    uint32_t damage = dungeon_damage(roll, attack);
    if (damage < w->hp[m]) {
        w->hp[m] = (uint8_t)(w->hp[m] - damage);
        dungeon_note(words, DV_HIT, damage);
        return 0u;
    }
    uint32_t xp = DUNGEON_XP[type];
    words[DW_XP] += xp;
    while (level < 20u && words[DW_XP] >= 10u << (level - 1u)) {
        level++;
        words[DW_HP] += 4u;
    }
    words[DW_LEVEL] = level;
    dungeon_remove(w, words, m);
    dungeon_note(words, DV_KILL, xp);
    *won = type == k->boss;
    return 4u * xp + (type >= DUNGEON_REGULAR ? 1000u : 0u);
}

static uint32_t dungeon_pickup(dungeonWorld *w, uint32_t *words, uint32_t at, uint32_t item) {
    uint32_t n = words[DW_ITEMS] - 1u, i = 0u;
    while (w->iy[i] * DUNGEON_SIDE + w->ix[i] != at) i++;
    uint32_t gold = w->gold[i];
    w->ix[i] = w->ix[n]; w->iy[i] = w->iy[n]; w->gold[i] = w->gold[n];
    words[DW_ITEMS] = n;
    w->cell[at] &= (uint16_t)~DC_ITEM;
    if (item == DI_GOLD) {
        words[DW_GOLD] += gold;
        dungeon_note(words, DV_GOLD, gold);
        return gold;
    }
    if (item == DI_RATION) {
        words[DW_RATIONS]++;
        dungeon_note(words, DV_RATION, words[DW_RATIONS]);
        return 0u;
    }
    uint32_t colour = item - DI_POTION;
    words[DW_PACK + (colour >> 2)] += 1u << (colour & 3u) * 8u;
    dungeon_note(words, DV_POTION, colour + 1u);
    return 0u;
}

static uint32_t dungeon_move(const dungeonKnobs *k, dungeonDice *d, uint32_t direction, uint32_t *words, dungeonWorld *w, uint32_t *won) {
    if (words[DW_CONFUSED]) direction = dungeon_roll(d) & 3u;
    uint32_t at = dungeon_toward(words[DW_Y] * DUNGEON_SIDE + words[DW_X], direction), c = w->cell[at];
    if (c & DC_WHO) return dungeon_strike(k, d, words, w, (c >> 7 & 31u) - 1u, won);
    if ((c & DC_TERRAIN) < DT_FLOOR) { dungeon_note(words, DV_BUMP, 0u); return 0u; }
    words[DW_X] = at & 31u;
    words[DW_Y] = at >> 5;
    return c & DC_ITEM ? dungeon_pickup(w, words, at, c >> 3 & 15u) : 0u;
}

static void dungeon_quaff(dungeonDice *d, uint32_t *words, uint32_t effect) {
    int32_t hp = (int32_t)words[DW_HP], most = (int32_t)dungeon_most(words[DW_LEVEL]), before = hp;
    uint32_t magnitude = 0u;
    switch (effect) {
        case DE_HEAL: hp += most / 2 > 6 ? most / 2 : 6; hp = hp < most ? hp : most; words[DW_BLIND] = 0u; break;
        case DE_EXTRA: hp = most; words[DW_POISON] = 0u; words[DW_BLIND] = 0u; break;
        case DE_CURE: words[DW_POISON] = 0u; words[DW_CONFUSED] = 0u; words[DW_BLIND] = 0u; break;
        case DE_FIRE: words[DW_FIRE] = DUNGEON_WARD; magnitude = DUNGEON_WARD; break;
        case DE_COLD: words[DW_COLD] = DUNGEON_WARD; magnitude = DUNGEON_WARD; break;
        case DE_POISON:
            hp -= (int32_t)dungeon_damage(dungeon_roll(d), 4u);
            hp = hp > 1 ? hp : 1;
            if (words[DW_POISON] < DUNGEON_ILL) words[DW_POISON] = DUNGEON_ILL;
            break;
        case DE_CONFUSION: words[DW_CONFUSED] = DUNGEON_DAZE; magnitude = DUNGEON_DAZE; break;
        default: words[DW_BLIND] = DUNGEON_DARK; magnitude = DUNGEON_DARK; break;
    }
    words[DW_HP] = (uint32_t)hp;
    dungeon_note(words, DV_QUAFF + effect, hp != before ? (uint32_t)(hp > before ? hp - before : before - hp) : magnitude);
}

static uint32_t dungeon_use(const dungeonKnobs *k, uint32_t seed, dungeonDice *d, uint32_t *words, dungeonWorld *w, uint32_t *descended) {
    uint32_t slot = words[DW_SLOT];
    if (slot == 0u || !dungeon_count(words, slot)) {
        if (slot == 0u && (w->cell[words[DW_Y] * DUNGEON_SIDE + words[DW_X]] & DC_TERRAIN) == DT_STAIRS) {
            uint32_t depth = ++words[DW_DEPTH], record = depth > words[DW_DEEPEST];
            dungeon_floor(k, seed, words, w);
            dungeon_note(words, DV_DESCEND, depth);
            *descended = 1u;
            words[DW_DEEPEST] = record ? depth : words[DW_DEEPEST];
            return record ? 50u : 0u;
        }
        dungeon_note(words, DV_REST, 0u);
        return 0u;
    }
    if (slot == 1u) {
        uint32_t food = words[DW_FOOD], fed = food + 600u < 2000u ? food + 600u : 2000u;
        words[DW_FOOD] = fed;
        words[DW_RATIONS]--;
        dungeon_note(words, DV_EAT, fed - food);
        if (!words[DW_RATIONS]) words[DW_SLOT] = 0u;
        return 0u;
    }
    uint32_t colour = slot - 2u;
    words[DW_PACK + (colour >> 2)] -= 1u << (colour & 3u) * 8u;
    if (!dungeon_potions(words, colour)) words[DW_SLOT] = 0u;
    dungeon_quaff(d, words, words[DW_PERM] >> 3u * colour & 7u);
    return 0u;
}

static void dungeon_upkeep(const dungeonKnobs *k, dungeonDice *d, uint32_t *words, dungeonWorld *w) {
    uint32_t turn = words[DW_TURN], level = words[DW_LEVEL], food = words[DW_FOOD], sick = words[DW_POISON], quarter = !(turn & 3u);
    int32_t hp = (int32_t)words[DW_HP];
    food -= food != 0u;
    if (!food && quarter) { hp--; dungeon_note(words, DV_STARVE, 1u); }
    if (sick && quarter) { hp--; dungeon_note(words, DV_SICK, 1u); }
    if (food && !sick && hp < (int32_t)dungeon_most(level) && turn % (level < 9u ? 20u - 2u * level : 3u) == 0u) hp++;
    words[DW_FOOD] = food;
    words[DW_HP] = (uint32_t)hp;
    for (uint32_t i = DW_POISON; i <= DW_COLD; i++) words[i] -= words[i] != 0u;
    if (turn % 50u || words[DW_MONSTERS] >= DUNGEON_ALIVE) return;
    uint32_t coin = dungeon_roll(d);
    if (coin >= u01Bound(0.25f)) return;
    uint32_t place = dungeon_roll(d), room = (place >> 16) * 9u >> 16;
    uint32_t x = w->rx0[room] + ((place & 0xFFu) * (w->rx1[room] - w->rx0[room] + 1u) >> 8), y = w->ry0[room] + ((place >> 8 & 0xFFu) * (w->ry1[room] - w->ry0[room] + 1u) >> 8);
    uint32_t at = y * DUNGEON_SIDE + x;
    dungeonSight s = dungeon_sight(k, words, w);
    if (!dungeon_free(w, at, dungeon_player(words)) || at == words[DW_STAIRS] || dungeon_lit(&s, x, y)) return;
    uint8_t kinds[DUNGEON_REGULAR];
    uint32_t count = dungeon_kinds(words[DW_DEPTH], kinds);
    dungeon_spawn(w, words, kinds[(coin & 0xFFFFu) * count >> 16], at, 0u);
}

static uint32_t dungeon_next(const uint32_t *words) {
    uint32_t full = 1u | (uint32_t)(words[DW_RATIONS] != 0u) << 1, slot = words[DW_SLOT];
    for (uint32_t c = 0u; c < 8u; c++) full |= (uint32_t)(dungeon_potions(words, c) != 0u) << (2u + c);
    uint32_t above = full >> (slot + 1u) << (slot + 1u);
    return above ? (uint32_t)__builtin_ctz(above) : 0u;
}

static uint32_t dungeon_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static float dungeon_turn(const dungeonKnobs *k, uint32_t seed, uint32_t action, uint32_t *words, dungeonWorld *w, uint32_t *done) {
    dungeonDice *d = &w->dice;
    uint32_t won = 0u, descended = 0u, score;
    words[DW_EVENT] = DV_NONE;
    words[DW_MAGNITUDE] = 0u;
    words[DW_TURN]++;
    score = action < 4u ? dungeon_move(k, d, action, words, w, &won) : dungeon_use(k, seed, d, words, w, &descended);
    if (won) {
        words[DW_RUN]++;
        dungeon_run(k, seed, words, w, DV_WIN);
        *done = 1u;
        return (float)score;
    }
    if (!descended) dungeon_monsters(d, words, w);
    dungeon_upkeep(k, d, words, w);
    words[DW_SCORE] += score;
    if ((int32_t)words[DW_HP] <= 0) {
        words[DW_RUN]++;
        dungeon_run(k, seed, words, w, DV_RUN);
        *done = 1u;
        return (float)score;
    }
    return (float)score;
}

static inline __attribute__((always_inline)) float dungeon_play(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const dungeonKnobs *k = &DUNGEON_KNOBS[instance];
    dungeonWorld *w = dungeon_world(extra);
    uint32_t action = dungeon_action(ctrl);
    float reward = 0.0f;
    *done = 0u;
    if (action == DUNGEON_NEXT) words[DW_SLOT] = dungeon_next(words);
    else reward = dungeon_turn(k, seed, action, words, w, done);
    dungeon_rouse(k, words, w);
    dungeon_obs(k, words, w, obs);
    return reward;
}

float dungeon_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return dungeon_play(0u, seed, ctrl, words, extra, obs, done);
        case 1u: return dungeon_play(1u, seed, ctrl, words, extra, obs, done);
        case 2u: return dungeon_play(2u, seed, ctrl, words, extra, obs, done);
        case 3u: return dungeon_play(3u, seed, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void dungeon_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const dungeonKnobs *k = &DUNGEON_KNOBS[instance];
    dungeonWorld *w = dungeon_world(extra);
    __builtin_memset(w, 0, sizeof *w);
    w->dice.seed = seed;
    w->dice.next = 16u;
    for (uint32_t i = 0u; i < DW_WORDS; i++) words[i] = 0u;
    if (!k->shuffle) words[DW_PERM] = dungeon_shuffle(seed, 0xFFFFFFFFu);
    words[DW_DEEPEST] = 1u;
    dungeon_run(k, seed, words, w, DV_RUN);
}

void dungeon_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    dungeon_obs(&DUNGEON_KNOBS[instance], words, dungeon_read(extra), obs);
}

uint32_t dungeon_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)instance; (void)shared;
    if (!out) return DUNGEON_VIEW;
    const dungeonWorld *w = dungeon_read(extra);
    uint32_t lit = words[DW_LIT], items = words[DW_ITEMS], n = words[DW_MONSTERS];
    for (uint32_t i = 0u; i < DUNGEON_CELLS / 8u; i++) {
        uint32_t packed = 0u;
        for (uint32_t j = 0u; j < 8u; j++) {
            uint32_t c = w->cell[8u * i + j], room = c >> 12, bright = room && (lit >> (room - 1u) & 1u);
            packed |= ((c & DC_TERRAIN) | bright << 3) << 4u * j;
        }
        out[i] = packed;
    }
    for (uint32_t i = 0u; i < DUNGEON_ITEMS / 2u; i++) {
        uint32_t packed = 0u;
        for (uint32_t j = 0u; j < 2u; j++) {
            uint32_t s = 2u * i + j;
            if (s < items) packed |= (w->ix[s] | (uint32_t)w->iy[s] << 5 | (w->cell[w->iy[s] * DUNGEON_SIDE + w->ix[s]] >> 3 & 15u) << 10) << 16u * j;
        }
        out[DUNGEON_CELLS / 8u + i] = packed;
    }
    for (uint32_t m = 0u; m < DUNGEON_ALIVE; m++)
        out[DUNGEON_CELLS / 8u + DUNGEON_ITEMS / 2u + m] = m < n ? w->mx[m] | (uint32_t)w->my[m] << 5 | (uint32_t)w->type[m] << 10 | (uint32_t)(w->flag[m] & 3u) << 14 | (uint32_t)w->hp[m] << 16 : 0u;
    return DUNGEON_VIEW;
}

static void dungeon_recall(const dungeonKnobs *k, const dungeonWorld *w, dungeonMind *mind, uint32_t at) {
    uint32_t c = w->cell[at], who = c >> 7 & 31u, terrain = c & DC_TERRAIN;
    uint32_t still = who && (w->kind[who - 1u] & DP_STILL), boss = who && w->type[who - 1u] == k->boss;
    uint32_t under = who ? mind->known[at] & (DK_STAIRS | DK_ITEM) : (terrain == DT_STAIRS ? DK_STAIRS : 0u) | (c & DC_ITEM ? DK_ITEM : 0u);
    mind->known[at] = (uint8_t)(DK_SEEN | (terrain >= DT_FLOOR ? DK_OPEN : 0u) | under | (still ? DK_STILL : 0u));
    if (boss) mind->boss = at + 1u;
    else if (mind->boss == at + 1u) mind->boss = 0u;
}

static void dungeon_survey(const dungeonKnobs *k, const uint32_t *words, const dungeonWorld *w, dungeonMind *mind) {
    dungeonSight s = dungeon_sight(k, words, w);
    for (int dy = -DUNGEON_REACH; dy <= DUNGEON_REACH; dy++)
        for (int dx = -DUNGEON_REACH; dx <= DUNGEON_REACH; dx++)
            if (dungeon_visible(&s, dx, dy)) dungeon_recall(k, w, mind, (s.y + (uint32_t)dy) * DUNGEON_SIDE + s.x + (uint32_t)dx);
}

static uint32_t dungeon_venom(const dungeonWorld *w, uint32_t at) {
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t who = w->cell[dungeon_toward(at, d)] >> 7 & 31u;
        if (who && DUNGEON_DAMAGE[w->type[who - 1u]] == DD_POISON) return 1u;
    }
    return 0u;
}

static void dungeon_remember(const dungeonKnobs *k, const uint32_t *words, dungeonWorld *w) {
    dungeonMind *mind = &w->mind;
    uint32_t run = words[DW_RUN], turn = words[DW_TURN], floor = run * 16u + words[DW_DEPTH], event = words[DW_EVENT], effect = event - DV_QUAFF;
    if (mind->run != run) {
        if (k->shuffle) __builtin_memset(mind->lore, 0, sizeof mind->lore);
        mind->run = run;
        mind->sick = mind->fire = mind->cold = mind->dizzy = 0u;
    }
    if (mind->floor != floor) {
        __builtin_memset(mind->known, 0, sizeof mind->known);
        mind->floor = floor;
        mind->boss = 0u;
    }
    if (mind->turn != turn) {
        if (effect < 8u) {
            if (mind->quaffed) mind->lore[mind->quaffed - 1u] = (uint8_t)(effect + 1u);
            if (effect == DE_FIRE) mind->fire = turn + DUNGEON_WARD;
            if (effect == DE_COLD) mind->cold = turn + DUNGEON_WARD;
            if (effect == DE_CONFUSION) mind->dizzy = turn + DUNGEON_DAZE;
            if (effect == DE_POISON) mind->sick = turn + DUNGEON_ILL;
            if (effect == DE_EXTRA || effect == DE_CURE) mind->sick = 0u;
            if (effect == DE_CURE) mind->dizzy = 0u;
        }
        if ((int32_t)words[DW_HP] < (int32_t)mind->health && dungeon_venom(w, dungeon_player(words))) mind->sick = turn + DUNGEON_ILL;
        if (event == DV_SICK && mind->sick < turn + 4u) mind->sick = turn + 4u;
    }
    mind->turn = turn;
    mind->health = words[DW_HP];
    dungeon_survey(k, words, w, mind);
}

static void dungeon_flood(const dungeonMind *mind, const uint8_t *avoid, uint8_t *dist) {
    uint16_t queue[DUNGEON_CELLS];
    uint32_t head = 0u, tail = 0u;
    for (uint32_t at = 0u; at < DUNGEON_CELLS; at++) {
        if (dist[at]) continue;
        if (avoid && avoid[at]) dist[at] = 255u;
        else queue[tail++] = (uint16_t)at;
    }
    while (head < tail) {
        uint32_t at = queue[head++], next = dist[at] < 254u ? dist[at] + 1u : 254u;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t to = dungeon_toward(at, d);
            if (!(mind->known[to] & DK_OPEN) || dist[to] != 255u || (avoid && avoid[to])) continue;
            dist[to] = (uint8_t)next;
            queue[tail++] = (uint16_t)to;
        }
    }
}

static uint32_t dungeon_downhill(const uint8_t *dist, uint32_t at) {
    uint32_t best = dist[at], direction = 4u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t far = dist[dungeon_toward(at, d)];
        if (far < best) { best = far; direction = d; }
    }
    return direction;
}

static uint32_t dungeon_seek(const dungeonMind *mind, const uint8_t *avoid, const uint8_t *want, uint32_t at) {
    uint8_t dist[DUNGEON_CELLS];
    for (uint32_t c = 0u; c < DUNGEON_CELLS; c++) dist[c] = (uint8_t)(want[c] ? 0u : 255u);
    dungeon_flood(mind, avoid, dist);
    return dungeon_downhill(dist, at);
}

static uint32_t dungeon_head(const dungeonMind *mind, const uint8_t *avoid, const uint8_t *want, uint32_t at) {
    uint32_t direction = dungeon_seek(mind, avoid, want, at);
    return direction < 4u || want[at] ? direction : dungeon_seek(mind, 0, want, at);
}

static void dungeon_wary(const dungeonMind *mind, uint8_t *avoid) {
    __builtin_memset(avoid, 0, DUNGEON_CELLS);
    for (uint32_t at = 0u; at < DUNGEON_CELLS; at++) {
        if (!(mind->known[at] & DK_STILL)) continue;
        avoid[at] = 1u;
        for (uint32_t d = 0u; d < 4u; d++) avoid[dungeon_toward(at, d)] = 1u;
    }
}

static uint32_t dungeon_mark(const dungeonMind *mind, uint8_t *want, uint32_t flag) {
    uint32_t any = 0u;
    for (uint32_t at = 0u; at < DUNGEON_CELLS; at++) {
        want[at] = (uint8_t)((mind->known[at] & flag) != 0u);
        any |= want[at];
    }
    return any;
}

static uint32_t dungeon_frontier(const dungeonMind *mind, uint8_t *want) {
    uint32_t any = 0u;
    for (uint32_t at = 0u; at < DUNGEON_CELLS; at++) {
        uint32_t edge = 0u;
        if (mind->known[at] & DK_OPEN)
            for (uint32_t d = 0u; d < 4u; d++) edge |= !mind->known[dungeon_toward(at, d)];
        want[at] = (uint8_t)edge;
        any |= edge;
    }
    return any;
}

static uint32_t dungeon_goal(const dungeonMind *mind, uint32_t last, uint8_t *want) {
    if (!last) return dungeon_mark(mind, want, DK_STAIRS);
    __builtin_memset(want, 0, DUNGEON_CELLS);
    if (mind->boss) want[mind->boss - 1u] = 1u;
    return mind->boss != 0u;
}

static uint32_t dungeon_find(const dungeonMind *mind, const uint32_t *words, uint32_t lore) {
    for (uint32_t c = 0u; c < 8u; c++)
        if (mind->lore[c] == lore && dungeon_potions(words, c)) return 2u + c;
    return 0u;
}

static uint32_t dungeon_still(const dungeonWorld *w, uint32_t m) { return w->kind[m] & DP_STILL; }

static uint32_t dungeon_exposed(const dungeonWorld *w, const uint32_t *words, const dungeonSight *s, uint32_t at) {
    uint32_t x = at & 31u, y = at >> 5;
    for (uint32_t m = 0u; m < words[DW_MONSTERS]; m++) {
        if (!(w->kind[m] & DP_RANGED) || !dungeon_visible(s, (int)w->mx[m] - (int)s->x, (int)w->my[m] - (int)s->y)) continue;
        uint32_t dx = w->mx[m] > x ? w->mx[m] - x : x - w->mx[m], dy = w->my[m] > y ? w->my[m] - y : y - w->my[m];
        uint32_t from = w->my[m] * DUNGEON_SIDE + w->mx[m];
        if ((dx == 0u || dy == 0u) && dx + dy >= 2u && dx + dy <= 5u && dungeon_clear(w, from, at, dx ? (x > w->mx[m] ? 1u : 0u - 1u) : (y > w->my[m] ? (uint32_t)DUNGEON_SIDE : 0u - (uint32_t)DUNGEON_SIDE))) return 1u;
    }
    return 0u;
}

static uint32_t dungeon_decide(const dungeonKnobs *k, const uint32_t *words, dungeonWorld *w) {
    dungeon_remember(k, words, w);
    const dungeonMind *mind = &w->mind;
    dungeonSight s = dungeon_sight(k, words, w);
    uint32_t x = words[DW_X], y = words[DW_Y], at = y * DUNGEON_SIDE + x, most = dungeon_most(words[DW_LEVEL]), n = words[DW_MONSTERS], turn = words[DW_TURN];
    int32_t hp = (int32_t)words[DW_HP];
    uint32_t heal = dungeon_find(mind, words, DE_HEAL + 1u), full = dungeon_find(mind, words, DE_EXTRA + 1u), cure = dungeon_find(mind, words, DE_CURE + 1u);
    uint32_t warm = dungeon_find(mind, words, DE_FIRE + 1u), cool = dungeon_find(mind, words, DE_COLD + 1u), guess = dungeon_find(mind, words, 0u);
    uint32_t target = 4u, weakest = 256u, cornered = 0u, swift = 0u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t who = w->cell[dungeon_toward(at, d)] >> 7 & 31u;
        if (!who) continue;
        if (dungeon_still(w, who - 1u)) { cornered = 1u; continue; }
        uint32_t type = w->type[who - 1u];
        swift |= DUNGEON_SPEED[type] > 12u;
        if (DUNGEON_HP[type] < weakest) { weakest = DUNGEON_HP[type]; target = d; }
    }
    uint32_t near = 255u, hot = 0u, chill = 0u;
    for (uint32_t m = 0u; m < n; m++) {
        int dx = (int)w->mx[m] - (int)x, dy = (int)w->my[m] - (int)y;
        if (dungeon_still(w, m) || !dungeon_visible(&s, dx, dy)) continue;
        uint32_t dist = (uint32_t)((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
        near = dist < near ? dist : near;
        if (dist <= 6u) {
            hot |= DUNGEON_DAMAGE[w->type[m]] == DD_FIRE;
            chill |= DUNGEON_DAMAGE[w->type[m]] == DD_COLD;
        }
    }
    uint32_t last = words[DW_DEPTH] == k->floors, stairs = (w->cell[at] & DC_TERRAIN) == DT_STAIRS, danger = !last && hp * 100 < (int32_t)most * 35 && !heal && !full;
    uint32_t calm = near > 8u && !cornered && (words[DW_FOOD] > 200u || words[DW_RATIONS]);
    if (hp * 5 < (int32_t)most * 2) {
        if (heal) return DUNGEON_USE + heal;
        if (full) return DUNGEON_USE + full;
        if (guess && !calm) return DUNGEON_USE + guess;
    }
    if (mind->sick > turn && hp <= (int32_t)((mind->sick - turn) / 4u) + 2) {
        if (cure) return DUNGEON_USE + cure;
        if (full) return DUNGEON_USE + full;
        if (heal) return DUNGEON_USE + heal;
    }
    if (hot && mind->fire < turn + 5u && warm) return DUNGEON_USE + warm;
    if (chill && mind->cold < turn + 5u && cool) return DUNGEON_USE + cool;
    if (target < 4u && (!danger || swift)) return target;
    if (words[DW_FOOD] < 150u && words[DW_RATIONS]) return DUNGEON_USE + 1u;
    if (calm && guess && hp * 10 < (int32_t)most * 7 && hp * 5 >= (int32_t)most * 2) return DUNGEON_USE + guess;
    if (calm && !stairs && (hp * 10 < (int32_t)most * 7 || (last && !mind->boss && hp < (int32_t)most) || mind->dizzy > turn)) return DUNGEON_USE;
    if (dungeon_exposed(w, words, &s, at))
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t to = dungeon_toward(at, d);
            if (dungeon_free(w, to, at) && !dungeon_exposed(w, words, &s, to)) return d;
        }
    uint8_t avoid[DUNGEON_CELLS], want[DUNGEON_CELLS];
    dungeon_wary(mind, avoid);
    if (!danger && dungeon_mark(mind, want, DK_ITEM)) {
        uint32_t direction = dungeon_seek(mind, avoid, want, at);
        if (direction < 4u) return direction;
    }
    if (!(danger && dungeon_goal(mind, last, want)) && dungeon_frontier(mind, want)) {
        uint32_t direction = dungeon_head(mind, avoid, want, at);
        if (direction < 4u) return direction;
    }
    if (!dungeon_goal(mind, last, want) && (last || !dungeon_mark(mind, want, DK_STILL))) return DUNGEON_USE;
    uint32_t direction = dungeon_head(mind, avoid, want, at);
    return direction < 4u ? direction : DUNGEON_USE;
}

void dungeon_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    dungeonWorld *w = dungeon_world(extra);
    uint32_t choice = dungeon_decide(&DUNGEON_KNOBS[instance], words, w), slot = choice - DUNGEON_USE;
    uint32_t act = choice < DUNGEON_USE ? choice : slot == words[DW_SLOT] ? DUNGEON_ACT : DUNGEON_NEXT;
    w->mind.quaffed = act == DUNGEON_ACT && slot >= 2u ? slot - 1u : 0u;
    ctrl[0] = (float)act;
}
