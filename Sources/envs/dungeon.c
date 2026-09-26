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
enum { DW_X, DW_Y, DW_HP, DW_LEVEL, DW_XP, DW_FOOD, DW_DEPTH, DW_TURN, DW_RUN, DW_SLOT, DW_RATIONS, DW_PACK, DW_PACK_HIGH, DW_POISON, DW_CONFUSED, DW_BLIND, DW_FIRE, DW_COLD,
       DW_EVENT, DW_MAGNITUDE, DW_GOLD, DW_SCORE, DW_PERM, DW_ROBIN, DW_MONSTERS, DW_ITEMS, DW_STAIRS, DW_LIT, DW_DEEPEST, DW_WORDS };
enum { DT_ROCK, DT_WALL, DT_FLOOR, DT_CORRIDOR, DT_DOOR, DT_STAIRS };
enum { DC_TERRAIN = 7, DC_ITEM = 15 << 3, DC_WHO = 31 << 7 };
enum { DI_GOLD = 1, DI_RATION, DI_POTION };
enum { DE_HEAL, DE_EXTRA, DE_CURE, DE_FIRE, DE_COLD, DE_POISON, DE_CONFUSION, DE_BLINDNESS };
enum { DB_CHASE, DB_ERRATIC, DB_RANGED, DB_STILL };
enum { DD_PHYSICAL, DD_FIRE, DD_COLD, DD_POISON, DD_DRAIN };
enum { DV_NONE, DV_HIT, DV_MISS, DV_KILL, DV_HURT, DV_GOLD, DV_RATION, DV_POTION, DV_EAT, DV_QUAFF, DV_DESCEND = DV_QUAFF + 8, DV_REST, DV_BUMP, DV_STARVE, DV_SICK, DV_RUN, DV_WIN };
enum { DM_AWAKE = 1, DM_GUARD = 2 };
enum { DP_ADJACENT = 1, DP_STILL = 2, DP_ERRATIC = 4, DP_AIM = 8, DP_RANGED = 16 };
enum { DUNGEON_ACT = 4, DUNGEON_NEXT = 5, DUNGEON_USE = 8 };

typedef struct { uint32_t word[16], next, fill, seed, pad; } dungeonDice;

typedef struct {
    uint16_t cell[DUNGEON_CELLS];
    uint8_t mx[DUNGEON_SLOTS], my[DUNGEON_SLOTS], type[DUNGEON_SLOTS], hp[DUNGEON_SLOTS], energy[DUNGEON_SLOTS], speed[DUNGEON_SLOTS], flag[DUNGEON_SLOTS], kind[DUNGEON_SLOTS];
    uint8_t ix[DUNGEON_ITEMS], iy[DUNGEON_ITEMS], gold[DUNGEON_ITEMS];
    uint8_t rx0[16], ry0[16], rx1[16], ry1[16];
    dungeonDice dice;
    uint8_t goal[DUNGEON_CELLS], loot[DUNGEON_CELLS], avoid[DUNGEON_CELLS];
    uint64_t goalKey, lootKey, avoidKey;
} dungeonWorld;

typedef struct { uint32_t seed, run, depth, block, at; uint16_t half[32]; } dungeonDraw;
typedef struct { uint8_t first[DUNGEON_SLOTS], second[DUNGEON_SLOTS], code[DUNGEON_SLOTS]; } dungeonPlan;
typedef struct { uint32_t x, y, radius, x0, y0, x1, y1; } dungeonSight;

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
    dungeonSight s = { x, y, k->radius, 255u, 255u, 0u, 0u };
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

static inline uint8x16_t dungeon_lit16(const dungeonSight *s, uint8x16_t x, uint8x16_t y, uint8x16_t *far) {
    uint8x16_t dx = vabdq_u8(x, vdupq_n_u8((uint8_t)s->x)), dy = vabdq_u8(y, vdupq_n_u8((uint8_t)s->y));
    uint8x16_t across = vandq_u8(vcgeq_u8(x, vdupq_n_u8((uint8_t)s->x0)), vcleq_u8(x, vdupq_n_u8((uint8_t)s->x1)));
    uint8x16_t down = vandq_u8(vcgeq_u8(y, vdupq_n_u8((uint8_t)s->y0)), vcleq_u8(y, vdupq_n_u8((uint8_t)s->y1)));
    *far = vqaddq_u8(dx, dy);
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

static inline uint64_t dungeon_nibbles(uint8x16_t set) { return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(set), 4)), 0); }

static uint32_t dungeon_nearest(const dungeonSight *s, const uint8_t *xs, const uint8_t *ys, uint32_t n, uint8_t *wake, uint32_t blind) {
    uint8x16_t none = vdupq_n_u8(255u), far, low, high = none, lit = vandq_u8(dungeon_lit16(s, vld1q_u8(xs), vld1q_u8(ys), &far), dungeon_valid(0u, n));
    if (wake) vst1q_u8(wake, vorrq_u8(vld1q_u8(wake), vandq_u8(lit, vdupq_n_u8(DM_AWAKE))));
    low = vbslq_u8(lit, far, none);
    if (n > 16u) {
        lit = vandq_u8(dungeon_lit16(s, vld1q_u8(xs + 16u), vld1q_u8(ys + 16u), &far), dungeon_valid(1u, n));
        if (wake) vst1q_u8(wake + 16u, vorrq_u8(vld1q_u8(wake + 16u), vandq_u8(lit, vdupq_n_u8(DM_AWAKE))));
        high = vbslq_u8(lit, far, none);
    }
    if (blind) return DUNGEON_SLOTS;
    uint32_t best = vminvq_u8(vminq_u8(low, high));
    if (best == 255u) return DUNGEON_SLOTS;
    uint64_t first = dungeon_nibbles(vceqq_u8(low, vdupq_n_u8((uint8_t)best)));
    return first ? (uint32_t)__builtin_ctzll(first) >> 2 : 16u + ((uint32_t)__builtin_ctzll(dungeon_nibbles(vceqq_u8(high, vdupq_n_u8((uint8_t)best)))) >> 2);
}

static uint32_t dungeon_count(const uint32_t *words, const dungeonWorld *w, uint32_t slot) {
    if (slot == 0u) return (w->cell[words[DW_Y] * DUNGEON_SIDE + words[DW_X]] & DC_TERRAIN) == DT_STAIRS;
    if (slot == 1u) return words[DW_RATIONS];
    return dungeon_potions(words, slot - 2u);
}

static inline __attribute__((always_inline)) void dungeon_look(const dungeonKnobs *k, const uint32_t *words, const dungeonWorld *w, float *obs, uint8_t *wake) {
    static const float CODE[32] = { -1.0f, -1.0f, 0.0f, 0.0f, 0.25f, 1.0f, 0.5f, 0.625f, 0.75f, 0.78125f, 0.8125f, 0.84375f, 0.875f, 0.90625f, 0.9375f, 0.96875f,
                                    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f, -0.5f, -0.53125f, -0.5625f, -0.59375f, -0.625f, -0.65625f, -0.6875f, -0.71875f };
    uint32_t x = words[DW_X], y = words[DW_Y], at = y * DUNGEON_SIDE + x, level = words[DW_LEVEL], slot = words[DW_SLOT];
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t c = w->cell[dungeon_toward(at, d)], who = c >> 7 & 31u, item = c >> 3 & 15u;
        obs[d] = CODE[who ? 16u + w->type[who - 1u] : item ? 5u + item : c & DC_TERRAIN];
    }
    obs[4] = (float)(int32_t)words[DW_HP] / (float)dungeon_most(level);
    obs[5] = (float)words[DW_FOOD] * 0.001f;
    obs[6] = (float)level * 0.1f;
    obs[7] = (float)words[DW_DEPTH] * (1.0f / (float)k->floors);
    obs[8] = (float)slot * (1.0f / 9.0f);
    obs[9] = (float)dungeon_count(words, w, slot) * 0.125f;
    obs[10] = (float)words[DW_EVENT] * (1.0f / 24.0f);
    obs[11] = (float)words[DW_MAGNITUDE] * 0.0625f;
    for (uint32_t i = 12u; i < 18u; i++) obs[i] = 0.0f;
    dungeonSight s = dungeon_sight(k, words, w);
    uint32_t m = dungeon_nearest(&s, w->mx, w->my, words[DW_MONSTERS], wake, words[DW_BLIND]);
    if (words[DW_BLIND]) return;
    uint32_t stairs = words[DW_STAIRS], item = dungeon_nearest(&s, w->ix, w->iy, words[DW_ITEMS], 0, 0u);
    if (m < DUNGEON_SLOTS) {
        obs[12] = (float)((int)w->mx[m] - (int)x) * 0.125f;
        obs[13] = (float)((int)w->my[m] - (int)y) * 0.125f;
    }
    if (stairs && dungeon_lit(&s, stairs & 31u, stairs >> 5)) {
        obs[14] = (float)((int)(stairs & 31u) - (int)x) * 0.125f;
        obs[15] = (float)((int)(stairs >> 5) - (int)y) * 0.125f;
    }
    if (item < DUNGEON_SLOTS) {
        obs[16] = (float)((int)w->ix[item] - (int)x) * 0.125f;
        obs[17] = (float)((int)w->iy[item] - (int)y) * 0.125f;
    }
}

static void dungeon_obs(const dungeonKnobs *k, const uint32_t *words, const dungeonWorld *w, float *obs) { dungeon_look(k, words, w, obs, 0); }

static void dungeon_wakeful(const dungeonKnobs *k, const uint32_t *words, dungeonWorld *w, float *obs) { dungeon_look(k, words, w, obs, w->flag); }

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
    if (kind == DD_POISON && words[DW_POISON] < 20u) words[DW_POISON] = 20u;
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
        case DE_FIRE: words[DW_FIRE] = 300u; magnitude = 300u; break;
        case DE_COLD: words[DW_COLD] = 300u; magnitude = 300u; break;
        case DE_POISON:
            hp -= (int32_t)dungeon_damage(dungeon_roll(d), 4u);
            hp = hp > 1 ? hp : 1;
            if (words[DW_POISON] < 20u) words[DW_POISON] = 20u;
            break;
        case DE_CONFUSION: words[DW_CONFUSED] = 12u; magnitude = 12u; break;
        default: words[DW_BLIND] = 40u; magnitude = 40u; break;
    }
    words[DW_HP] = (uint32_t)hp;
    dungeon_note(words, DV_QUAFF + effect, hp != before ? (uint32_t)(hp > before ? hp - before : before - hp) : magnitude);
}

static uint32_t dungeon_use(const dungeonKnobs *k, uint32_t seed, dungeonDice *d, uint32_t *words, dungeonWorld *w, uint32_t *descended) {
    uint32_t slot = words[DW_SLOT];
    if (slot == 0u || !dungeon_count(words, w, slot)) {
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
    dungeon_wakeful(k, words, w, obs);
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
    w->goalKey = ~0ull;
    w->lootKey = ~0ull;
    w->avoidKey = ~0ull;
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

static void dungeon_flood(const dungeonWorld *w, uint32_t careful, uint8_t *dist, const uint16_t *source, uint32_t sources) {
    uint16_t queue[DUNGEON_CELLS];
    uint32_t head = 0u, tail = 0u;
    __builtin_memset(dist, 255, DUNGEON_CELLS);
    for (uint32_t s = 0u; s < sources; s++) {
        if (careful && w->avoid[source[s]]) continue;
        dist[source[s]] = 0u;
        queue[tail++] = source[s];
    }
    while (head < tail) {
        uint32_t at = queue[head++], next = dist[at] < 254u ? dist[at] + 1u : 254u;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t to = dungeon_toward(at, d);
            if ((w->cell[to] & DC_TERRAIN) < DT_FLOOR || dist[to] != 255u || (careful && w->avoid[to])) continue;
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

static uint32_t dungeon_find(const uint32_t *words, uint32_t effect) {
    for (uint32_t c = 0u; c < 8u; c++)
        if ((words[DW_PERM] >> 3u * c & 7u) == effect && dungeon_potions(words, c)) return 2u + c;
    return 0u;
}

static uint32_t dungeon_still(const dungeonWorld *w, uint32_t m) { return w->kind[m] & DP_STILL; }

static uint32_t dungeon_exposed(const dungeonWorld *w, const uint32_t *words, uint32_t at) {
    uint32_t x = at & 31u, y = at >> 5;
    for (uint32_t m = 0u; m < words[DW_MONSTERS]; m++) {
        if (!(w->flag[m] & DM_AWAKE) || !(w->kind[m] & DP_RANGED)) continue;
        uint32_t dx = w->mx[m] > x ? w->mx[m] - x : x - w->mx[m], dy = w->my[m] > y ? w->my[m] - y : y - w->my[m];
        uint32_t from = w->my[m] * DUNGEON_SIDE + w->mx[m];
        if ((dx == 0u || dy == 0u) && dx + dy >= 2u && dx + dy <= 5u && dungeon_clear(w, from, at, dx ? (x > w->mx[m] ? 1u : 0u - 1u) : (y > w->my[m] ? (uint32_t)DUNGEON_SIDE : 0u - (uint32_t)DUNGEON_SIDE))) return 1u;
    }
    return 0u;
}

static uint32_t dungeon_route(dungeonWorld *w, uint64_t key, uint32_t goal, uint32_t at) {
    uint16_t source = (uint16_t)goal;
    if ((w->goalKey & ~(1ull << 17)) != key) {
        dungeon_flood(w, 1u, w->goal, &source, 1u);
        w->goalKey = key;
    }
    uint32_t direction = dungeon_downhill(w->goal, at);
    if (direction == 4u && w->goal[at] != 0u && !(w->goalKey >> 17 & 1u)) {
        dungeon_flood(w, 0u, w->goal, &source, 1u);
        w->goalKey = key | 1ull << 17;
        direction = dungeon_downhill(w->goal, at);
    }
    return direction < 4u ? direction : DUNGEON_USE;
}

static uint32_t dungeon_decide(const dungeonKnobs *k, const uint32_t *words, dungeonWorld *w) {
    uint32_t x = words[DW_X], y = words[DW_Y], at = y * DUNGEON_SIDE + x, most = dungeon_most(words[DW_LEVEL]), n = words[DW_MONSTERS];
    int32_t hp = (int32_t)words[DW_HP];
    uint32_t heal = dungeon_find(words, DE_HEAL), full = dungeon_find(words, DE_EXTRA), cure = dungeon_find(words, DE_CURE);
    uint32_t warm = dungeon_find(words, DE_FIRE), cool = dungeon_find(words, DE_COLD), target = 4u, weakest = 256u, cornered = 0u, swift = 0u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t who = w->cell[dungeon_toward(at, d)] >> 7 & 31u;
        if (!who) continue;
        if (dungeon_still(w, who - 1u)) { cornered = 1u; continue; }
        swift |= w->speed[who - 1u] > 12u;
        if (w->hp[who - 1u] < weakest) { weakest = w->hp[who - 1u]; target = d; }
    }
    uint32_t near = 255u, hot = 0u, chill = 0u, bossAwake = 0u, bossAt = 0u, sign = 0u;
    for (uint32_t m = 0u; m < n; m++) {
        uint32_t dist = (w->mx[m] > x ? w->mx[m] - x : x - w->mx[m]) + (w->my[m] > y ? w->my[m] - y : y - w->my[m]);
        if (w->type[m] == k->boss) { bossAt = w->my[m] * DUNGEON_SIDE + w->mx[m]; bossAwake = w->flag[m] & DM_AWAKE; }
        if (dungeon_still(w, m)) { sign += 1u + ((uint32_t)w->my[m] * DUNGEON_SIDE + w->mx[m]) * 32u; continue; }
        if (!(w->flag[m] & DM_AWAKE)) continue;
        near = dist < near ? dist : near;
        if (dist <= 6u) {
            hot |= DUNGEON_DAMAGE[w->type[m]] == DD_FIRE;
            chill |= DUNGEON_DAMAGE[w->type[m]] == DD_COLD;
        }
    }
    uint32_t last = words[DW_DEPTH] == k->floors, stairs = (w->cell[at] & DC_TERRAIN) == DT_STAIRS, danger = !last && hp * 100 < (int32_t)most * 35 && !heal && !full;
    if (hp * 5 < (int32_t)most * 2) {
        if (heal) return DUNGEON_USE + heal;
        if (full) return DUNGEON_USE + full;
    }
    if (words[DW_POISON] && hp <= (int32_t)(words[DW_POISON] / 4u) + 2) {
        if (cure) return DUNGEON_USE + cure;
        if (full) return DUNGEON_USE + full;
        if (heal) return DUNGEON_USE + heal;
    }
    if (hot && words[DW_FIRE] < 5u && warm) return DUNGEON_USE + warm;
    if (chill && words[DW_COLD] < 5u && cool) return DUNGEON_USE + cool;
    if (target < 4u && (!danger || swift)) return target;
    if (words[DW_FOOD] < 150u && words[DW_RATIONS]) return DUNGEON_USE + 1u;
    uint32_t calm = near > 8u && !cornered && (words[DW_FOOD] > 200u || words[DW_RATIONS]);
    if (calm && !stairs && (hp * 10 < (int32_t)most * 7 || (last && !bossAwake && hp < (int32_t)most) || words[DW_CONFUSED])) return DUNGEON_USE;
    if (dungeon_exposed(w, words, at))
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t to = dungeon_toward(at, d);
            if (dungeon_free(w, to, at) && !dungeon_exposed(w, words, to)) return d;
        }
    uint64_t key = (uint64_t)words[DW_RUN] << 32 | (uint64_t)words[DW_DEPTH] << 28 | (sign & 0x1FFFFu);
    if (w->avoidKey != key) {
        __builtin_memset(w->avoid, 0, sizeof w->avoid);
        for (uint32_t m = 0u; m < n; m++) {
            if (!dungeon_still(w, m)) continue;
            uint32_t cell = w->my[m] * DUNGEON_SIDE + w->mx[m];
            w->avoid[cell] = 1u;
            for (uint32_t d = 0u; d < 4u; d++) w->avoid[dungeon_toward(cell, d)] = 1u;
        }
        w->avoidKey = key;
    }
    if (words[DW_ITEMS] && !danger) {
        uint64_t loot = key | (uint64_t)words[DW_ITEMS] << 18;
        if (w->lootKey != loot) {
            uint16_t source[DUNGEON_ITEMS];
            for (uint32_t i = 0u; i < words[DW_ITEMS]; i++) source[i] = (uint16_t)(w->iy[i] * DUNGEON_SIDE + w->ix[i]);
            dungeon_flood(w, 1u, w->loot, source, words[DW_ITEMS]);
            w->lootKey = loot;
        }
        uint32_t direction = dungeon_downhill(w->loot, at);
        if (direction < 4u) return direction;
    }
    uint32_t goal = last ? bossAt : words[DW_STAIRS];
    if (!goal) return DUNGEON_USE;
    return dungeon_route(w, key | (uint64_t)goal << 18, goal, at);
}

void dungeon_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    uint32_t choice = dungeon_decide(&DUNGEON_KNOBS[instance], words, dungeon_world(extra));
    ctrl[0] = (float)(choice < DUNGEON_USE ? choice : choice - DUNGEON_USE == words[DW_SLOT] ? DUNGEON_ACT : DUNGEON_NEXT);
}
