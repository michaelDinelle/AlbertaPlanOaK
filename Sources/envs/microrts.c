#include <string.h>
#include "../env.h"

typedef struct { uint32_t width, partial, opponent, substeps, maxCycles; } micrortsKnobs;

static const micrortsKnobs MICRORTS_KNOBS[5] = {
    { 8u, 0u, 0u, 12u, 3000u },
    { 8u, 0u, 1u, 12u, 3000u },
    { 16u, 0u, 2u, 16u, 5000u },
    { 16u, 1u, 3u, 16u, 5000u },
    { 16u, 1u, 4u, 16u, 5000u }
};

enum { MICRORTS_RESOURCE, MICRORTS_BASE, MICRORTS_BARRACKS, MICRORTS_WORKER, MICRORTS_LIGHT, MICRORTS_HEAVY, MICRORTS_RANGED };
enum { MICRORTS_NONE, MICRORTS_MOVE, MICRORTS_HARVEST, MICRORTS_RETURN, MICRORTS_PRODUCE, MICRORTS_ATTACK, MICRORTS_IDLE };
enum { MICRORTS_AIM_NONE, MICRORTS_AIM_MOVE, MICRORTS_AIM_ATTACK, MICRORTS_AIM_HARVEST, MICRORTS_AIM_BUILD, MICRORTS_AIM_TRAIN };
enum { MICRORTS_RANDOM_BIASED, MICRORTS_WORKER_RUSH, MICRORTS_LIGHT_RUSH, MICRORTS_HEAVY_RUSH, MICRORTS_RANGED_RUSH, MICRORTS_DOCTRINE };
enum { MICRORTS_SLOTS = 128, MICRORTS_EMPTY = 255, MICRORTS_NEUTRAL = 2, MICRORTS_WAIT = 10 };
enum { MICRORTS_WORLD_DRAWS = 0x52424149, MICRORTS_ORACLE_DRAWS = 0x4F52434C };

typedef struct {
    uint8_t cost, hp, damage, range, sight, move, attack, harvest, mobile, fighter, harvester, stockpile, makes;
    uint16_t build;
} micrortsType;

static const micrortsType MICRORTS_TYPES[7] = {
    { 1u, 1u, 1u, 1u, 0u, 10u, 10u, 10u, 0u, 0u, 0u, 0u, 0u, 10u },
    { 10u, 10u, 1u, 1u, 5u, 10u, 10u, 10u, 0u, 0u, 0u, 1u, 1u << MICRORTS_WORKER, 200u },
    { 5u, 4u, 1u, 1u, 3u, 10u, 10u, 10u, 0u, 0u, 0u, 0u, (1u << MICRORTS_LIGHT) | (1u << MICRORTS_HEAVY) | (1u << MICRORTS_RANGED), 100u },
    { 1u, 1u, 1u, 1u, 3u, 10u, 5u, 20u, 1u, 1u, 1u, 0u, (1u << MICRORTS_BASE) | (1u << MICRORTS_BARRACKS), 50u },
    { 2u, 4u, 2u, 1u, 2u, 8u, 5u, 10u, 1u, 1u, 0u, 0u, 0u, 80u },
    { 3u, 8u, 4u, 1u, 2u, 10u, 5u, 10u, 1u, 1u, 0u, 0u, 0u, 120u },
    { 2u, 1u, 1u, 3u, 3u, 10u, 5u, 10u, 1u, 1u, 0u, 0u, 0u, 100u }
};

typedef struct { uint8_t kind, owner, cell; uint8_t id, carry; } micrortsPlacement;

static const micrortsPlacement MICRORTS_MAP8[6] = {
    { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 0u, 4u, 20u }, { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 63u, 5u, 20u },
    { MICRORTS_BASE, 0u, 10u, 6u, 0u }, { MICRORTS_BASE, 1u, 53u, 7u, 0u },
    { MICRORTS_WORKER, 0u, 9u, 8u, 0u }, { MICRORTS_WORKER, 1u, 54u, 9u, 0u }
};

static const micrortsPlacement MICRORTS_MAP16[8] = {
    { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 0u, 16u, 25u }, { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 16u, 17u, 25u },
    { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 239u, 18u, 25u }, { MICRORTS_RESOURCE, MICRORTS_NEUTRAL, 255u, 19u, 25u },
    { MICRORTS_BASE, 0u, 34u, 20u, 0u }, { MICRORTS_BASE, 1u, 221u, 21u, 0u },
    { MICRORTS_WORKER, 0u, 17u, 22u, 0u }, { MICRORTS_WORKER, 1u, 238u, 23u, 0u }
};

static const uint32_t MICRORTS_DISC[6][11] = {
    { 0x10000u },
    { 0x10000u, 0x38000u, 0x10000u },
    { 0x10000u, 0x38000u, 0x7C000u, 0x38000u, 0x10000u },
    { 0x10000u, 0x7C000u, 0x7C000u, 0xFE000u, 0x7C000u, 0x7C000u, 0x10000u },
    { 0x10000u, 0x7C000u, 0xFE000u, 0xFE000u, 0x1FF000u, 0xFE000u, 0xFE000u, 0x7C000u, 0x10000u },
    { 0x10000u, 0xFE000u, 0x1FF000u, 0x1FF000u, 0x1FF000u, 0x3FF800u, 0x1FF000u, 0x1FF000u, 0x1FF000u, 0xFE000u, 0x10000u }
};

typedef struct { uint8_t slot, act, arg, make; int32_t eta; } micrortsOrder;

typedef struct { uint64_t bits[4]; } micrortsCells;

typedef struct {
    uint32_t time, nextId, units, pending, width, fog;
    int32_t gold[2];
    micrortsCells views[2];
    uint8_t eyes[2][256];
    uint8_t list[MICRORTS_SLOTS], queue[MICRORTS_SLOTS], at[256];
    uint32_t id[MICRORTS_SLOTS], start[MICRORTS_SLOTS];
    int32_t eta[MICRORTS_SLOTS], hp[MICRORTS_SLOTS], carry[MICRORTS_SLOTS];
    uint8_t kind[MICRORTS_SLOTS], owner[MICRORTS_SLOTS], cell[MICRORTS_SLOTS], act[MICRORTS_SLOTS], arg[MICRORTS_SLOTS], make[MICRORTS_SLOTS], live[MICRORTS_SLOTS];
} micrortsWorld;

typedef struct {
    uint32_t count;
    uint8_t order[MICRORTS_SLOTS], aim[MICRORTS_SLOTS], cell[MICRORTS_SLOTS], make[MICRORTS_SLOTS], target[MICRORTS_SLOTS], base[MICRORTS_SLOTS];
    uint32_t targetId[MICRORTS_SLOTS], baseId[MICRORTS_SLOTS];
} micrortsMind;

typedef struct {
    uint32_t player, script, fog, stage, index, count, head, spare, used, workers, orders, seed, game, draws, tag, army, threat;
    uint32_t troops[3];
    int32_t reservedGold, spent;
    uint32_t foes, mines, homes, forts, ranked, owned;
    micrortsCells view, reserved, path, taken, bodies, claimed, ordered;
    uint8_t roster[MICRORTS_SLOTS], rest[MICRORTS_SLOTS], foe[MICRORTS_SLOTS], mine[MICRORTS_SLOTS], home[MICRORTS_SLOTS];
    uint8_t fort[MICRORTS_SLOTS], ranks[MICRORTS_SLOTS], ours[MICRORTS_SLOTS];
    micrortsOrder out[MICRORTS_SLOTS];
    float theta[12];
} micrortsThought;

static int microrts_has(const micrortsCells *c, uint32_t cell) { return (int)((c->bits[cell >> 6] >> (cell & 63u)) & 1u); }
static void microrts_mark(micrortsCells *c, uint32_t cell) { c->bits[cell >> 6] |= 1ull << (cell & 63u); }
static void microrts_clear(micrortsCells *c, uint32_t cell) { c->bits[cell >> 6] &= ~(1ull << (cell & 63u)); }

static uint32_t microrts_x(uint32_t width, uint32_t cell) { return cell & (width - 1u); }
static uint32_t microrts_y(uint32_t width, uint32_t cell) { return cell >> __builtin_ctz(width); }

static void microrts_delta(uint32_t direction, int *dx, int *dy) { gridDelta((0x9Cu >> (2u * (direction & 3u))) & 3u, dx, dy); }

static int microrts_adjacent(uint32_t width, uint32_t cell, uint32_t direction, uint32_t *next) {
    int dx, dy;
    microrts_delta(direction, &dx, &dy);
    int x = (int)microrts_x(width, cell) + dx, y = (int)microrts_y(width, cell) + dy;
    if (x < 0 || y < 0 || x >= (int)width || y >= (int)width) return 0;
    *next = (uint32_t)x + (uint32_t)y * width;
    return 1;
}

static uint32_t microrts_manhattan(uint32_t width, uint32_t a, uint32_t b) {
    int dx = (int)microrts_x(width, a) - (int)microrts_x(width, b), dy = (int)microrts_y(width, a) - (int)microrts_y(width, b);
    return (uint32_t)(abs(dx) + abs(dy));
}

static uint32_t microrts_square(uint32_t width, uint32_t a, uint32_t b) {
    int dx = (int)microrts_x(width, a) - (int)microrts_x(width, b), dy = (int)microrts_y(width, a) - (int)microrts_y(width, b);
    return (uint32_t)(dx * dx + dy * dy);
}

static int32_t microrts_eta(const micrortsWorld *w, uint32_t slot, uint32_t act, uint32_t make, int32_t duration) {
    const micrortsType *t = &MICRORTS_TYPES[w->kind[slot]];
    switch (act) {
        case MICRORTS_MOVE: return t->move;
        case MICRORTS_HARVEST: return t->harvest;
        case MICRORTS_RETURN: return t->move;
        case MICRORTS_PRODUCE: return MICRORTS_TYPES[make].build;
        case MICRORTS_ATTACK: return t->attack;
        default: return duration;
    }
}

static micrortsOrder microrts_order(const micrortsWorld *w, uint32_t slot, uint32_t act, uint32_t arg, uint32_t make, int32_t duration) {
    micrortsOrder o = { (uint8_t)slot, (uint8_t)act, (uint8_t)arg, (uint8_t)make, 0 };
    o.eta = microrts_eta(w, slot, act, make, duration);
    return o;
}

static int microrts_reserves(uint32_t act) { return act == MICRORTS_MOVE || act == MICRORTS_PRODUCE; }

static uint32_t microrts_reserved(const micrortsWorld *w, uint32_t slot, uint32_t arg) {
    uint32_t next = 0u;
    microrts_adjacent(w->width, w->cell[slot], arg, &next);
    return next;
}

static int32_t microrts_committed(const micrortsWorld *w, uint32_t owner) {
    int32_t cost = 0;
    for (uint32_t q = 0u; q < w->pending; q++) {
        uint32_t e = w->queue[q];
        if (w->act[e] == MICRORTS_PRODUCE && w->owner[e] == owner) cost += MICRORTS_TYPES[w->make[e]].cost;
    }
    return cost;
}

static int microrts_claimed(const micrortsWorld *w, uint32_t owner, uint32_t cell) {
    for (uint32_t q = 0u; q < w->pending; q++) {
        uint32_t e = w->queue[q];
        if (w->owner[e] == owner && microrts_reserves(w->act[e]) && microrts_reserved(w, e, w->arg[e]) == cell) return 1;
    }
    return 0;
}

static int microrts_valid(const micrortsWorld *w, const micrortsOrder *o) {
    uint32_t slot = o->slot, next;
    const micrortsType *t = &MICRORTS_TYPES[w->kind[slot]];
    switch (o->act) {
        case MICRORTS_NONE: return 1;
        case MICRORTS_MOVE: return t->mobile && microrts_adjacent(w->width, w->cell[slot], o->arg, &next) && w->at[next] == MICRORTS_EMPTY && !microrts_claimed(w, w->owner[slot], next);
        case MICRORTS_HARVEST:
            if (!t->harvester || w->carry[slot] != 0 || !microrts_adjacent(w->width, w->cell[slot], o->arg, &next) || w->at[next] == MICRORTS_EMPTY) return 0;
            return w->kind[w->at[next]] == MICRORTS_RESOURCE;
        case MICRORTS_RETURN:
            if (w->carry[slot] <= 0 || !microrts_adjacent(w->width, w->cell[slot], o->arg, &next) || w->at[next] == MICRORTS_EMPTY) return 0;
            return MICRORTS_TYPES[w->kind[w->at[next]]].stockpile && w->owner[w->at[next]] == w->owner[slot];
        case MICRORTS_PRODUCE:
            if (!((t->makes >> o->make) & 1u) || !microrts_adjacent(w->width, w->cell[slot], o->arg, &next) || w->at[next] != MICRORTS_EMPTY || microrts_claimed(w, w->owner[slot], next)) return 0;
            return microrts_committed(w, w->owner[slot]) + (int32_t)MICRORTS_TYPES[o->make].cost <= w->gold[w->owner[slot]];
        default: {
            uint32_t target = o->arg, other = w->at[target];
            if (!t->fighter || microrts_square(w->width, w->cell[slot], target) > (uint32_t)t->range * t->range || other == MICRORTS_EMPTY) return 0;
            return w->owner[other] != MICRORTS_NEUTRAL && w->owner[other] != w->owner[slot];
        }
    }
}

static void microrts_enqueue(micrortsWorld *w, uint32_t slot, const micrortsOrder *o) {
    if (w->act[slot] == MICRORTS_IDLE) w->queue[w->pending++] = (uint8_t)slot;
    w->act[slot] = o->act;
    w->arg[slot] = o->arg;
    w->make[slot] = o->make;
    w->eta[slot] = o->eta;
    w->start[slot] = w->time;
}

static void microrts_issue(micrortsWorld *w, const micrortsOrder *orders, uint32_t count, uint8_t *rejected) {
    for (uint32_t i = 0u; i < count; i++) {
        micrortsOrder o = orders[i];
        if (!microrts_valid(w, &o)) {
            if (rejected) rejected[i] = 1u;
            continue;
        }
        if (microrts_reserves(o.act)) {
            uint32_t cell = microrts_reserved(w, o.slot, o.arg);
            for (uint32_t q = 0u; q < w->pending; q++) {
                uint32_t e = w->queue[q];
                if (e == o.slot || !microrts_reserves(w->act[e]) || microrts_reserved(w, e, w->arg[e]) != cell) continue;
                if (rejected) rejected[i] = 1u;
                if (w->start[e] == w->time) {
                    int32_t eta = o.eta < w->eta[e] ? o.eta : w->eta[e];
                    w->act[e] = MICRORTS_NONE;
                    w->eta[e] = eta;
                    o.eta = eta;
                } else o.eta = -1;
                o.act = MICRORTS_NONE;
            }
        }
        microrts_enqueue(w, o.slot, &o);
    }
}

static __attribute__((noinline)) void microrts_watch(micrortsWorld *w, uint32_t slot, int add) {
    uint32_t owner = w->owner[slot], width = w->width;
    if (owner > 1u) return;
    int sight = MICRORTS_TYPES[w->kind[slot]].sight, cx = (int)microrts_x(width, w->cell[slot]), cy = (int)microrts_y(width, w->cell[slot]);
    int y0 = cy - sight < 0 ? 0 : cy - sight, y1 = cy + sight >= (int)width ? (int)width - 1 : cy + sight;
    uint8_t *eyes = w->eyes[owner];
    micrortsCells *view = &w->views[owner];
    for (int y = y0; y <= y1; y++) {
        uint32_t row = (uint32_t)(((uint64_t)MICRORTS_DISC[sight][y - cy + sight] << cx) >> 16) & ((1u << width) - 1u);
        for (; row; row &= row - 1u) {
            uint32_t c = (uint32_t)y * width + (uint32_t)__builtin_ctz(row);
            if (add) { if (eyes[c]++ == 0u) microrts_mark(view, c); }
            else if (--eyes[c] == 0u) microrts_clear(view, c);
        }
    }
}

static uint32_t microrts_claim(micrortsWorld *w) {
    for (uint32_t s = 0u; s < MICRORTS_SLOTS; s++)
        if (!w->live[s]) return s;
    return MICRORTS_EMPTY;
}

static uint32_t microrts_spawn(micrortsWorld *w, uint32_t kind, uint32_t owner, uint32_t cell, uint32_t id, int32_t carry) {
    uint32_t s = microrts_claim(w);
    if (s == MICRORTS_EMPTY) return s;
    w->live[s] = 1u;
    w->id[s] = id;
    w->kind[s] = (uint8_t)kind;
    w->owner[s] = (uint8_t)owner;
    w->cell[s] = (uint8_t)cell;
    w->hp[s] = MICRORTS_TYPES[kind].hp;
    w->carry[s] = carry;
    w->act[s] = MICRORTS_IDLE;
    w->at[cell] = (uint8_t)s;
    w->list[w->units++] = (uint8_t)s;
    if (w->fog) microrts_watch(w, s, 1);
    return s;
}

static void microrts_forget(micrortsMind *m, uint32_t slot) {
    if (m->aim[slot] == MICRORTS_AIM_NONE) return;
    m->aim[slot] = MICRORTS_AIM_NONE;
    uint32_t j = 0u;
    for (uint32_t i = 0u; i < m->count; i++)
        if (m->order[i] != slot) m->order[j++] = m->order[i];
    m->count = j;
}

static void microrts_unqueue(micrortsWorld *w, uint32_t slot) {
    uint32_t j = 0u;
    for (uint32_t i = 0u; i < w->pending; i++)
        if (w->queue[i] != slot) w->queue[j++] = w->queue[i];
    w->pending = j;
    w->act[slot] = MICRORTS_IDLE;
}

static void microrts_remove(micrortsWorld *w, micrortsMind *minds, uint32_t slot) {
    if (w->fog) microrts_watch(w, slot, 0);
    if (w->act[slot] != MICRORTS_IDLE) microrts_unqueue(w, slot);
    w->live[slot] = 0u;
    if (w->at[w->cell[slot]] == slot) w->at[w->cell[slot]] = MICRORTS_EMPTY;
    uint32_t j = 0u;
    for (uint32_t i = 0u; i < w->units; i++)
        if (w->list[i] != slot) w->list[j++] = w->list[i];
    w->units = j;
    microrts_forget(&minds[0], slot);
    microrts_forget(&minds[1], slot);
}

typedef struct { uint32_t lost[2], produced[2], returned[2]; } micrortsTally;

static void microrts_perform(micrortsWorld *w, micrortsMind *minds, uint32_t slot, micrortsTally *tally) {
    uint32_t next = 0u, owner = w->owner[slot];
    switch (w->act[slot]) {
        case MICRORTS_MOVE:
            if (microrts_adjacent(w->width, w->cell[slot], w->arg[slot], &next) && w->at[next] == MICRORTS_EMPTY) {
                if (w->fog) microrts_watch(w, slot, 0);
                w->at[w->cell[slot]] = MICRORTS_EMPTY;
                w->cell[slot] = (uint8_t)next;
                w->at[next] = (uint8_t)slot;
                if (w->fog) microrts_watch(w, slot, 1);
            }
            return;
        case MICRORTS_HARVEST:
            if (microrts_adjacent(w->width, w->cell[slot], w->arg[slot], &next) && w->at[next] != MICRORTS_EMPTY && w->kind[w->at[next]] == MICRORTS_RESOURCE) {
                uint32_t r = w->at[next];
                w->carry[r] -= 1;
                w->carry[slot] += 1;
                if (w->carry[r] <= 0) microrts_remove(w, minds, r);
            }
            return;
        case MICRORTS_RETURN:
            if (microrts_adjacent(w->width, w->cell[slot], w->arg[slot], &next) && w->at[next] != MICRORTS_EMPTY) {
                uint32_t b = w->at[next];
                if (MICRORTS_TYPES[w->kind[b]].stockpile && w->owner[b] == owner) {
                    w->gold[owner] += w->carry[slot];
                    if (tally) tally->returned[owner & 1u] += (uint32_t)w->carry[slot];
                    w->carry[slot] = 0;
                }
            }
            return;
        case MICRORTS_PRODUCE: {
            uint32_t id = w->nextId++;
            if (microrts_adjacent(w->width, w->cell[slot], w->arg[slot], &next) && w->at[next] == MICRORTS_EMPTY && w->gold[owner] >= (int32_t)MICRORTS_TYPES[w->make[slot]].cost
                && microrts_spawn(w, w->make[slot], owner, next, id, 0) != MICRORTS_EMPTY) {
                w->gold[owner] -= MICRORTS_TYPES[w->make[slot]].cost;
                if (tally) tally->produced[owner & 1u]++;
            }
            return;
        }
        case MICRORTS_ATTACK: {
            uint32_t other = w->at[w->arg[slot]];
            if (other != MICRORTS_EMPTY) w->hp[other] -= MICRORTS_TYPES[w->kind[slot]].damage;
            return;
        }
        default: return;
    }
}

static void microrts_cycle(micrortsWorld *w, micrortsMind *minds, micrortsTally *tally) {
    uint8_t due[MICRORTS_SLOTS];
    uint32_t dueCount = 0u, j = 0u;
    w->time++;
    for (uint32_t i = 0u; i < w->pending; i++) {
        uint32_t s = w->queue[i];
        if ((int64_t)w->time >= (int64_t)w->start[s] + w->eta[s]) due[dueCount++] = (uint8_t)s;
        else w->queue[j++] = (uint8_t)s;
    }
    w->pending = j;
    uint32_t hits = 0u;
    for (uint32_t i = 0u; i < dueCount; i++) {
        uint32_t s = due[i];
        hits |= w->act[s] == MICRORTS_ATTACK;
        microrts_perform(w, minds, s, tally);
        w->act[s] = MICRORTS_IDLE;
    }
    for (uint32_t i = 0u; hits && i < w->units;) {
        uint32_t s = w->list[i];
        if (w->hp[s] <= 0 && w->kind[s] != MICRORTS_RESOURCE) {
            if (tally) tally->lost[w->owner[s] & 1u]++;
            microrts_remove(w, minds, s);
        } else i++;
    }
}

static int microrts_standing(const micrortsWorld *w, uint32_t player) {
    for (uint32_t i = 0u; i < w->units; i++)
        if (w->owner[w->list[i]] == player) return 1;
    return 0;
}

static const micrortsPlacement *microrts_map(uint32_t width, uint32_t *count) {
    *count = width == 8u ? 6u : 8u;
    return width == 8u ? MICRORTS_MAP8 : MICRORTS_MAP16;
}

static void microrts_load(micrortsWorld *w, uint32_t width, uint32_t fog) {
    memset(w, 0, sizeof *w);
    memset(w->at, MICRORTS_EMPTY, sizeof w->at);
    w->width = width;
    w->fog = fog;
    w->gold[0] = 5;
    w->gold[1] = 5;
    uint32_t count;
    const micrortsPlacement *map = microrts_map(width, &count);
    for (uint32_t i = 0u; i < count; i++) microrts_spawn(w, map[i].kind, map[i].owner, map[i].cell, map[i].id, map[i].carry);
    w->nextId = map[count - 1u].id + 1u;
}

static uint32_t microrts_row(const micrortsCells *c, uint32_t width, uint32_t y) {
    uint32_t at = y * width;
    return (uint32_t)(c->bits[at >> 6] >> (at & 63u)) & ((1u << width) - 1u);
}


static int microrts_visible(const micrortsWorld *w, const micrortsThought *t, uint32_t slot) {
    return !t->fog || w->owner[slot] == t->player || microrts_has(&t->view, w->cell[slot]);
}

static int microrts_search(const micrortsWorld *w, const micrortsCells *blocked, uint32_t start, uint32_t target, int32_t range) {
    uint32_t width = w->width;
    int tx = (int)microrts_x(width, target), ty = (int)microrts_y(width, target), sx = (int)microrts_x(width, start), sy = (int)microrts_y(width, start);
    uint16_t head[160], next[256];
    uint8_t first[256], cost[256];
    micrortsCells seen = { { 0u, 0u, 0u, 0u } };
    uint32_t base = (uint32_t)(abs(sx - tx) + abs(sy - ty)), low = 0u, high = 0u;
    head[0] = (uint16_t)start;
    next[start] = 0xFFFFu;
    cost[start] = 0u;
    first[start] = 0u;
    microrts_mark(&seen, start);
    for (;;) {
        while (low <= high && head[low] == 0xFFFFu) low++;
        if (low > high) return -1;
        uint32_t node = head[low];
        head[low] = next[node];
        int x = (int)microrts_x(width, node), y = (int)microrts_y(width, node);
        int goal = range < 0 ? node == target : (x - tx) * (x - tx) + (y - ty) * (y - ty) <= range * range;
        if (goal) return node == start ? -1 : (int)first[node];
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t nb;
            if (!microrts_adjacent(width, node, d, &nb) || microrts_has(blocked, nb) || microrts_has(&seen, nb)) continue;
            microrts_mark(&seen, nb);
            cost[nb] = (uint8_t)(cost[node] + 1u);
            first[nb] = node == start ? (uint8_t)d : first[node];
            int nx = (int)microrts_x(width, nb), ny = (int)microrts_y(width, nb);
            uint32_t bucket = (cost[nb] + (uint32_t)(abs(nx - tx) + abs(ny - ty)) - base) >> 1;
            while (high < bucket) head[++high] = 0xFFFFu;
            next[nb] = head[bucket];
            head[bucket] = (uint16_t)nb;
        }
    }
}

static void microrts_blocked(const micrortsThought *t, micrortsCells *blocked) {
    for (uint32_t i = 0u; i < 4u; i++) blocked->bits[i] = t->path.bits[i] | t->bodies.bits[i];
}


static void microrts_aim(micrortsMind *m, const micrortsWorld *w, uint32_t slot, uint32_t aim, uint32_t cell, uint32_t make, uint32_t target, uint32_t base) {
    if (m->aim[slot] == MICRORTS_AIM_NONE) m->order[m->count++] = (uint8_t)slot;
    m->aim[slot] = (uint8_t)aim;
    m->cell[slot] = (uint8_t)cell;
    m->make[slot] = (uint8_t)make;
    m->target[slot] = (uint8_t)target;
    m->base[slot] = (uint8_t)base;
    m->targetId[slot] = target == MICRORTS_EMPTY ? 0u : w->id[target];
    m->baseId[slot] = base == MICRORTS_EMPTY ? 0u : w->id[base];
}

static int microrts_present(const micrortsWorld *w, const micrortsThought *t, uint32_t slot, uint32_t id) {
    return slot != MICRORTS_EMPTY && w->live[slot] && w->id[slot] == id && microrts_visible(w, t, slot);
}

static int microrts_foe(const micrortsWorld *w, uint32_t player, uint32_t slot) { return w->owner[slot] != MICRORTS_NEUTRAL && w->owner[slot] != player; }

static uint32_t microrts_nearest(const micrortsWorld *w, const micrortsThought *t, uint32_t cell, uint32_t what) {
    const uint8_t *set = what == 0u ? t->foe : what == 1u ? t->mine : t->home;
    uint32_t n = what == 0u ? t->foes : what == 1u ? t->mines : t->homes, best = MICRORTS_EMPTY, far = 0xFFFFFFFFu;
    for (uint32_t i = 0u; i < n; i++) {
        uint32_t d = microrts_manhattan(w->width, cell, w->cell[set[i]]);
        if (d < far) { far = d; best = set[i]; }
    }
    return best;
}

static void microrts_explore(const micrortsWorld *w, micrortsMind *m, const micrortsThought *t, uint32_t slot) {
    uint32_t width = w->width, ux = microrts_x(width, w->cell[slot]), uy = microrts_y(width, w->cell[slot]), best = 0u, far = 0xFFFFFFFFu;
    for (uint32_t k = 0u; k < width && k * k <= far; k++)
        for (uint32_t side = 0u; side < (k ? 2u : 1u); side++) {
            uint32_t y = side ? uy + k : uy - k;
            if (y >= width) continue;
            uint32_t hidden = ~microrts_row(&t->view, width, y) & ((1u << width) - 1u);
            if (!hidden) continue;
            uint32_t left = hidden & ((2u << ux) - 1u), right = hidden >> ux, x;
            if (left && right) {
                uint32_t l = 31u - (uint32_t)__builtin_clz(left), r = ux + (uint32_t)__builtin_ctz(right);
                x = ux - l <= r - ux ? l : r;
            } else x = left ? 31u - (uint32_t)__builtin_clz(left) : ux + (uint32_t)__builtin_ctz(right);
            uint32_t dx = x > ux ? x - ux : ux - x, d = dx * dx + k * k, c = x + y * width;
            if (d < far || (d == far && c < best)) { far = d; best = c; }
        }
    if (far != 0xFFFFFFFFu) microrts_aim(m, w, slot, MICRORTS_AIM_MOVE, best, 0u, MICRORTS_EMPTY, MICRORTS_EMPTY);
}

static void microrts_melee(const micrortsWorld *w, micrortsMind *m, const micrortsThought *t, uint32_t slot) {
    uint32_t e = microrts_nearest(w, t, w->cell[slot], 0u);
    if (e != MICRORTS_EMPTY) microrts_aim(m, w, slot, MICRORTS_AIM_ATTACK, 0u, 0u, e, MICRORTS_EMPTY);
    else if (t->fog) microrts_explore(w, m, t, slot);
}

static uint32_t microrts_site(const micrortsWorld *w, const micrortsThought *t, uint32_t cell) {
    int width = (int)w->width, dx = (int)microrts_x(w->width, cell), dy = (int)microrts_y(w->width, cell);
    for (int l = 1; l <= width; l++)
        for (int side = 0; side < 4; side++)
            for (int i = -l; i <= l; i++) {
                int x = side == 1 ? dx + l : side == 3 ? dx - l : dx + i;
                int y = side == 0 ? dy - l : side == 2 ? dy + l : dy + i;
                if (x < 0 || y < 0 || x >= width || y >= width) continue;
                uint32_t c = (uint32_t)x + (uint32_t)y * w->width;
                if (!microrts_has(&t->bodies, c) && !microrts_has(&t->claimed, c)) return c;
            }
    return 0xFFFFu;
}

static void microrts_erect(const micrortsWorld *w, micrortsMind *m, micrortsThought *t, uint32_t slot, uint32_t kind) {
    if (m->aim[slot] == MICRORTS_AIM_BUILD && m->make[slot] == kind) return;
    uint32_t c = microrts_site(w, t, w->cell[slot]);
    if (c == 0xFFFFu) return;
    microrts_aim(m, w, slot, MICRORTS_AIM_BUILD, c, kind, MICRORTS_EMPTY, MICRORTS_EMPTY);
    microrts_mark(&t->claimed, c);
}

static int microrts_allowed(const micrortsWorld *w, const micrortsThought *t, uint32_t cell, uint32_t make) {
    if (microrts_has(&t->reserved, cell)) return 0;
    return !(t->reservedGold > 0 && t->reservedGold + (int32_t)MICRORTS_TYPES[make].cost > w->gold[t->player]);
}

static int32_t microrts_score(const micrortsWorld *w, const micrortsThought *t, uint32_t cell, uint32_t make) {
    uint32_t s = microrts_nearest(w, t, cell, make == MICRORTS_WORKER ? 1u : 0u);
    return s == MICRORTS_EMPTY ? 0 : -(int32_t)microrts_manhattan(w->width, cell, w->cell[s]);
}

static int microrts_route(const micrortsWorld *w, const micrortsThought *t, uint32_t slot, uint32_t goal, int32_t range, micrortsOrder *o) {
    micrortsCells blocked;
    microrts_blocked(t, &blocked);
    if (range < 0 && microrts_has(&blocked, goal)) return 0;
    int d = microrts_search(w, &blocked, w->cell[slot], goal, range);
    if (d < 0) return 0;
    *o = microrts_order(w, slot, MICRORTS_MOVE, (uint32_t)d, 0u, 0);
    return 1;
}

static uint32_t microrts_toward(uint32_t width, uint32_t from, uint32_t to) {
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n;
        if (microrts_adjacent(width, from, d, &n) && n == to) return d;
    }
    return 0u;
}

static int microrts_complete(const micrortsWorld *w, const micrortsMind *m, const micrortsThought *t, uint32_t slot) {
    switch (m->aim[slot]) {
        case MICRORTS_AIM_MOVE: return w->cell[slot] == m->cell[slot];
        case MICRORTS_AIM_ATTACK: return !microrts_present(w, t, m->target[slot], m->targetId[slot]);
        case MICRORTS_AIM_HARVEST: return w->carry[slot] == 0 ? !microrts_present(w, t, m->target[slot], m->targetId[slot]) : !microrts_present(w, t, m->base[slot], m->baseId[slot]);
        case MICRORTS_AIM_BUILD: return microrts_has(&t->bodies, m->cell[slot]);
        case MICRORTS_AIM_TRAIN: return m->cell[slot] != 0u;
        default: return 1;
    }
}

static __attribute__((noinline)) int microrts_execute(const micrortsWorld *w, micrortsMind *m, const micrortsThought *t, uint32_t slot, micrortsOrder *o) {
    uint32_t width = w->width, here = w->cell[slot];
    switch (m->aim[slot]) {
        case MICRORTS_AIM_MOVE: return microrts_route(w, t, slot, m->cell[slot], -1, o);
        case MICRORTS_AIM_ATTACK: {
            uint32_t there = w->cell[m->target[slot]], range = MICRORTS_TYPES[w->kind[slot]].range;
            if (microrts_square(width, here, there) <= range * range) { *o = microrts_order(w, slot, MICRORTS_ATTACK, there, 0u, 0); return 1; }
            return microrts_route(w, t, slot, there, (int32_t)range, o);
        }
        case MICRORTS_AIM_HARVEST: {
            uint32_t carrying = w->carry[slot] > 0, goal = w->cell[carrying ? m->base[slot] : m->target[slot]];
            if (microrts_manhattan(width, here, goal) == 1u) { *o = microrts_order(w, slot, carrying ? MICRORTS_RETURN : MICRORTS_HARVEST, microrts_toward(width, here, goal), 0u, 0); return 1; }
            return microrts_route(w, t, slot, goal, 1, o);
        }
        case MICRORTS_AIM_BUILD: {
            uint32_t goal = m->cell[slot];
            if (microrts_manhattan(width, here, goal) == 1u) {
                if (!microrts_allowed(w, t, goal, m->make[slot])) return 0;
                *o = microrts_order(w, slot, MICRORTS_PRODUCE, microrts_toward(width, here, goal), m->make[slot], 0);
                return 1;
            }
            return microrts_route(w, t, slot, goal, 1, o);
        }
        case MICRORTS_AIM_TRAIN: {
            int32_t best = 0;
            uint32_t pick = 4u;
            m->cell[slot] = 1u;
            for (uint32_t d = 0u; d < 4u; d++) {
                uint32_t n;
                if (!microrts_adjacent(width, here, d, &n) || microrts_has(&t->bodies, n) || !microrts_allowed(w, t, n, m->make[slot])) continue;
                int32_t s = microrts_score(w, t, n, m->make[slot]);
                if (pick == 4u || s > best) { best = s; pick = d; }
            }
            if (pick == 4u) return 0;
            *o = microrts_order(w, slot, MICRORTS_PRODUCE, pick, m->make[slot], 0);
            return 1;
        }
        default: return 0;
    }
}

static int microrts_fits(const micrortsWorld *w, const micrortsThought *t, const micrortsOrder *o) {
    if (microrts_reserves(o->act) && microrts_has(&t->taken, microrts_reserved(w, o->slot, o->arg))) return 0;
    if (o->act != MICRORTS_PRODUCE) return 1;
    return t->spent + (int32_t)MICRORTS_TYPES[o->make].cost <= w->gold[t->player];
}

static void microrts_accept(const micrortsWorld *w, micrortsThought *t, const micrortsOrder *o) {
    t->out[t->orders++] = *o;
    microrts_mark(&t->ordered, o->slot);
    if (microrts_reserves(o->act)) microrts_mark(&t->taken, microrts_reserved(w, o->slot, o->arg));
    if (o->act == MICRORTS_PRODUCE) t->spent += MICRORTS_TYPES[o->make].cost;
}

static void microrts_wait(const micrortsWorld *w, micrortsThought *t, uint32_t slot) {
    micrortsOrder o = microrts_order(w, slot, MICRORTS_NONE, 0u, 0u, MICRORTS_WAIT);
    t->out[t->orders++] = o;
    microrts_mark(&t->ordered, slot);
}

static uint32_t microrts_options(const micrortsWorld *w, uint32_t slot, micrortsOrder *options, uint32_t *weights) {
    uint32_t n = 0u, width = w->width, here = w->cell[slot], owner = w->owner[slot], next;
    const micrortsType *t = &MICRORTS_TYPES[w->kind[slot]];
    if (t->fighter && t->range == 1u)
        for (uint32_t d = 0u; d < 4u; d++) {
            if (!microrts_adjacent(width, here, d, &next) || w->at[next] == MICRORTS_EMPTY || !microrts_foe(w, owner, w->at[next])) continue;
            options[n] = microrts_order(w, slot, MICRORTS_ATTACK, next, 0u, 0);
            weights[n++] = 5u;
        }
    else if (t->fighter)
        for (uint32_t i = 0u; i < w->units; i++) {
            uint32_t s = w->list[i];
            if (!microrts_foe(w, owner, s) || microrts_square(width, here, w->cell[s]) > (uint32_t)t->range * t->range) continue;
            options[n] = microrts_order(w, slot, MICRORTS_ATTACK, w->cell[s], 0u, 0);
            weights[n++] = 5u;
        }
    if (t->harvester && w->carry[slot] == 0)
        for (uint32_t d = 0u; d < 4u; d++)
            if (microrts_adjacent(width, here, d, &next) && w->at[next] != MICRORTS_EMPTY && w->kind[w->at[next]] == MICRORTS_RESOURCE) {
                options[n] = microrts_order(w, slot, MICRORTS_HARVEST, d, 0u, 0);
                weights[n++] = 5u;
            }
    if (w->carry[slot] > 0)
        for (uint32_t d = 0u; d < 4u; d++)
            if (microrts_adjacent(width, here, d, &next) && w->at[next] != MICRORTS_EMPTY && MICRORTS_TYPES[w->kind[w->at[next]]].stockpile && w->owner[w->at[next]] == owner) {
                options[n] = microrts_order(w, slot, MICRORTS_RETURN, d, 0u, 0);
                weights[n++] = 5u;
            }
    for (uint32_t k = 0u; k < 7u; k++) {
        if (!((t->makes >> k) & 1u) || w->gold[owner] < (int32_t)MICRORTS_TYPES[k].cost) continue;
        for (uint32_t d = 0u; d < 4u; d++)
            if (microrts_adjacent(width, here, d, &next) && w->at[next] == MICRORTS_EMPTY) {
                options[n] = microrts_order(w, slot, MICRORTS_PRODUCE, d, k, 0);
                weights[n++] = 1u;
            }
    }
    if (t->mobile)
        for (uint32_t d = 0u; d < 4u; d++)
            if (microrts_adjacent(width, here, d, &next) && w->at[next] == MICRORTS_EMPTY) {
                options[n] = microrts_order(w, slot, MICRORTS_MOVE, d, 0u, 0);
                weights[n++] = 1u;
            }
    options[n] = microrts_order(w, slot, MICRORTS_NONE, 0u, 0u, MICRORTS_WAIT);
    weights[n++] = 1u;
    return n;
}

static __attribute__((noinline)) void microrts_gamble(const micrortsWorld *w, micrortsThought *t, uint32_t slot) {
    micrortsOrder options[MICRORTS_SLOTS + 64u];
    uint32_t weights[MICRORTS_SLOTS + 64u], n = microrts_options(w, slot, options, weights), total = 0u;
    for (uint32_t i = 0u; i < n; i++) total += weights[i];
    uint64_t draw = (uint64_t)(philox4x32(t->seed, ENV_RNG_STREAM, t->game, w->time, t->draws++, t->tag).x >> 8) * total;
    uint32_t pick = 0u;
    uint64_t sum = 0u;
    for (; pick + 1u < n; pick++) {
        sum += (uint64_t)weights[pick] << 24;
        if (draw <= sum) break;
    }
    if (microrts_fits(w, t, &options[pick])) microrts_accept(w, t, &options[pick]);
    else microrts_wait(w, t, slot);
}

static uint32_t microrts_round(float x) { return x > 0.0f ? (uint32_t)(x + 0.5f) : 0u; }

static uint32_t microrts_recruit(const micrortsThought *t) {
    uint32_t best = 0u;
    float top = 0.0f;
    for (uint32_t i = 0u; i < 3u; i++) {
        float v = t->theta[1u + i] / (float)(1u + t->troops[i]);
        if (v > top) { top = v; best = MICRORTS_LIGHT + i; }
    }
    return best;
}

static uint32_t microrts_prey(const micrortsWorld *w, const micrortsThought *t, uint32_t slot) {
    uint32_t best = MICRORTS_EMPTY;
    float top = 0.0f;
    for (uint32_t i = 0u; i < t->foes; i++) {
        uint32_t s = t->foe[i], kind = w->kind[s];
        float v = -t->theta[7] * (float)microrts_manhattan(w->width, w->cell[slot], w->cell[s]) - t->theta[8] * 10.0f * (float)w->hp[s] / (float)MICRORTS_TYPES[kind].hp
                  + t->theta[9] * 10.0f * (float)(kind == MICRORTS_WORKER) + t->theta[10] * 10.0f * (float)(kind == MICRORTS_BASE || kind == MICRORTS_BARRACKS);
        if (best == MICRORTS_EMPTY || v > top) { top = v; best = s; }
    }
    return best;
}

static void microrts_charge(const micrortsWorld *w, micrortsMind *m, const micrortsThought *t, uint32_t slot) {
    uint32_t e = microrts_prey(w, t, slot);
    if (e != MICRORTS_EMPTY) microrts_aim(m, w, slot, MICRORTS_AIM_ATTACK, 0u, 0u, e, MICRORTS_EMPTY);
    else if (t->fog) microrts_explore(w, m, t, slot);
}

static void microrts_rally(const micrortsWorld *w, micrortsMind *m, micrortsThought *t, uint32_t slot, uint32_t base, uint32_t reach) {
    if (microrts_manhattan(w->width, w->cell[slot], w->cell[base]) <= reach) return;
    uint32_t c = microrts_site(w, t, w->cell[base]);
    if (c != 0xFFFFu) microrts_aim(m, w, slot, MICRORTS_AIM_MOVE, c, 0u, MICRORTS_EMPTY, MICRORTS_EMPTY);
}

static void microrts_command(const micrortsWorld *w, micrortsMind *m, micrortsThought *t, uint32_t slot) {
    uint32_t base = microrts_nearest(w, t, w->cell[slot], 2u);
    if (base != MICRORTS_EMPTY && (float)w->hp[slot] <= t->theta[11] * (float)MICRORTS_TYPES[w->kind[slot]].hp) microrts_rally(w, m, t, slot, base, 2u);
    else if ((float)t->army >= t->theta[6] || t->threat || base == MICRORTS_EMPTY) microrts_charge(w, m, t, slot);
    else microrts_rally(w, m, t, slot, base, 3u);
}

enum { MICRORTS_THINK_SCAN, MICRORTS_THINK_BASES, MICRORTS_THINK_BARRACKS, MICRORTS_THINK_MELEE, MICRORTS_THINK_BUILD, MICRORTS_THINK_WORKERS, MICRORTS_THINK_REST, MICRORTS_THINK_TRANSLATE, MICRORTS_THINK_FILL, MICRORTS_THINK_RANDOM, MICRORTS_THINK_DONE };

static void microrts_begin(micrortsThought *t, uint32_t player, uint32_t script, uint32_t seed, uint32_t game, uint32_t tag, const micrortsCells *view) {
    t->player = player;
    t->script = script;
    t->fog = view != 0;
    if (view) t->view = *view;
    t->seed = seed;
    t->game = game;
    t->tag = tag;
    t->stage = MICRORTS_THINK_SCAN;
    t->index = 0u;
    t->orders = 0u;
}

static __attribute__((noinline)) void microrts_scan(const micrortsWorld *w, micrortsThought *t) {
    memset(&t->reserved, 0, sizeof t->reserved);
    memset(&t->bodies, 0, sizeof t->bodies);
    memset(&t->claimed, 0, sizeof t->claimed);
    memset(&t->ordered, 0, sizeof t->ordered);
    t->reservedGold = 0;
    for (uint32_t q = 0u; q < w->pending; q++) {
        uint32_t e = w->queue[q];
        if (!microrts_reserves(w->act[e]) || !microrts_visible(w, t, e)) continue;
        microrts_mark(&t->reserved, microrts_reserved(w, e, w->arg[e]));
        if (w->act[e] == MICRORTS_PRODUCE && w->owner[e] == t->player) t->reservedGold += MICRORTS_TYPES[w->make[e]].cost;
    }
    t->path = t->reserved;
    t->taken = t->reserved;
    t->spent = t->reservedGold;
    t->count = t->head = t->spare = t->used = t->workers = t->orders = t->draws = 0u;
    t->army = t->threat = t->troops[0] = t->troops[1] = t->troops[2] = t->foes = t->mines = t->homes = t->forts = t->ranked = t->owned = 0u;
    for (uint32_t i = 0u; i < w->units; i++) {
        uint32_t s = w->list[i], kind = w->kind[s];
        if (w->owner[s] == t->player) {
            microrts_mark(&t->bodies, w->cell[s]);
            t->ours[t->owned++] = (uint8_t)s;
            if (kind == MICRORTS_BASE) t->home[t->homes++] = (uint8_t)s;
            else if (kind == MICRORTS_BARRACKS) t->fort[t->forts++] = (uint8_t)s;
            else if (kind == MICRORTS_WORKER) { t->army++; t->workers++; t->roster[t->count++] = (uint8_t)s; }
            else { t->army += MICRORTS_TYPES[kind].cost; t->troops[kind - MICRORTS_LIGHT]++; t->ranks[t->ranked++] = (uint8_t)s; }
        } else if (microrts_visible(w, t, s)) {
            microrts_mark(&t->bodies, w->cell[s]);
            if (kind == MICRORTS_RESOURCE) t->mine[t->mines++] = (uint8_t)s;
            else t->foe[t->foes++] = (uint8_t)s;
        }
    }
    if (t->script == MICRORTS_DOCTRINE)
        for (uint32_t i = 0u; i < t->foes && !t->threat; i++) {
            uint32_t b = microrts_nearest(w, t, w->cell[t->foe[i]], 2u);
            if (b != MICRORTS_EMPTY && microrts_manhattan(w->width, w->cell[t->foe[i]], w->cell[b]) <= 5u) t->threat = 1u;
        }
}

static uint32_t microrts_army(uint32_t script) { return script == MICRORTS_HEAVY_RUSH ? MICRORTS_HEAVY : script == MICRORTS_RANGED_RUSH ? MICRORTS_RANGED : MICRORTS_LIGHT; }

static void microrts_harvester(const micrortsWorld *w, micrortsMind *m, micrortsThought *t, uint32_t slot) {
    uint32_t b = microrts_nearest(w, t, w->cell[slot], 2u), r = microrts_nearest(w, t, w->cell[slot], 1u);
    if (b != MICRORTS_EMPTY && (r != MICRORTS_EMPTY || w->carry[slot] > 0)) microrts_aim(m, w, slot, MICRORTS_AIM_HARVEST, 0u, 0u, w->carry[slot] > 0 ? MICRORTS_EMPTY : r, b);
    else t->rest[t->spare++] = (uint8_t)slot;
}

static void microrts_ponder(const micrortsWorld *w, micrortsMind *m, micrortsThought *t) {
    while (t->stage != MICRORTS_THINK_DONE) {
        switch (t->stage) {
            case MICRORTS_THINK_SCAN:
                microrts_scan(w, t);
                t->stage = t->script == MICRORTS_RANDOM_BIASED ? MICRORTS_THINK_RANDOM : MICRORTS_THINK_BASES;
                t->index = 0u;
                break;
            case MICRORTS_THINK_RANDOM: {
                if (t->index >= t->owned) { t->stage = MICRORTS_THINK_DONE; break; }
                uint32_t s = t->ours[t->index++];
                if (w->act[s] == MICRORTS_IDLE) microrts_gamble(w, t, s);
                break;
            }
            case MICRORTS_THINK_BASES: {
                if (t->index >= t->homes) { t->stage = t->script == MICRORTS_WORKER_RUSH ? MICRORTS_THINK_MELEE : MICRORTS_THINK_BARRACKS; t->index = 0u; break; }
                uint32_t s = t->home[t->index++];
                if (w->act[s] != MICRORTS_IDLE) break;
                uint32_t wanted = t->script == MICRORTS_DOCTRINE ? microrts_round(t->theta[4]) + microrts_round(8.0f * t->theta[0]) : t->script == MICRORTS_WORKER_RUSH ? 0xFFFFFFFFu : 1u;
                if (w->gold[t->player] >= 1 && t->workers < wanted) microrts_aim(m, w, s, MICRORTS_AIM_TRAIN, 0u, MICRORTS_WORKER, MICRORTS_EMPTY, MICRORTS_EMPTY);
                break;
            }
            case MICRORTS_THINK_BARRACKS: {
                if (t->index >= t->forts) { t->stage = MICRORTS_THINK_MELEE; t->index = 0u; break; }
                uint32_t s = t->fort[t->index++];
                if (w->act[s] != MICRORTS_IDLE) break;
                uint32_t kind = t->script == MICRORTS_DOCTRINE ? microrts_recruit(t) : microrts_army(t->script);
                if (kind == 0u) break;
                if (w->gold[t->player] >= (int32_t)MICRORTS_TYPES[kind].cost) microrts_aim(m, w, s, MICRORTS_AIM_TRAIN, 0u, kind, MICRORTS_EMPTY, MICRORTS_EMPTY);
                break;
            }
            case MICRORTS_THINK_MELEE: {
                if (t->index >= t->ranked) { t->stage = MICRORTS_THINK_BUILD; break; }
                uint32_t s = t->ranks[t->index++];
                if (w->act[s] != MICRORTS_IDLE) break;
                if (t->script == MICRORTS_DOCTRINE) microrts_command(w, m, t, s);
                else microrts_melee(w, m, t, s);
                break;
            }
            case MICRORTS_THINK_BUILD:
                if (t->homes == 0u && t->head < t->count && w->gold[t->player] >= 10 + (int32_t)t->used) {
                    microrts_erect(w, m, t, t->roster[t->head++], MICRORTS_BASE);
                    t->used += 10u;
                }
                if (t->script != MICRORTS_WORKER_RUSH && t->forts == 0u && t->head < t->count && w->gold[t->player] >= (t->script == MICRORTS_DOCTRINE ? (int32_t)microrts_round(t->theta[5]) : 5) + (int32_t)t->used
                    && (t->script != MICRORTS_DOCTRINE || t->theta[1] + t->theta[2] + t->theta[3] > 0.0f)) {
                    microrts_erect(w, m, t, t->roster[t->head++], MICRORTS_BARRACKS);
                    t->used += 5u;
                }
                t->stage = MICRORTS_THINK_WORKERS;
                t->index = t->head;
                t->spare = 0u;
                break;
            case MICRORTS_THINK_WORKERS:
                if (t->script == MICRORTS_WORKER_RUSH) {
                    if (t->head < t->count) {
                        uint32_t hw = t->roster[t->head++], b = microrts_nearest(w, t, w->cell[hw], 2u), r = microrts_nearest(w, t, w->cell[hw], 1u);
                        for (uint32_t i = t->head; i < t->count; i++) t->rest[t->spare++] = t->roster[i];
                        if (b != MICRORTS_EMPTY && (r != MICRORTS_EMPTY || w->carry[hw] > 0)) microrts_aim(m, w, hw, MICRORTS_AIM_HARVEST, 0u, 0u, w->carry[hw] > 0 ? MICRORTS_EMPTY : r, b);
                        else t->rest[t->spare++] = (uint8_t)hw;
                    }
                    t->stage = MICRORTS_THINK_REST;
                    t->index = 0u;
                    break;
                }
                if (t->index >= t->count) { t->stage = MICRORTS_THINK_REST; t->index = 0u; break; }
                if (t->script == MICRORTS_DOCTRINE && t->index - t->head >= microrts_round(t->theta[4])) t->rest[t->spare++] = t->roster[t->index++];
                else microrts_harvester(w, m, t, t->roster[t->index++]);
                break;
            case MICRORTS_THINK_REST:
                if (t->index >= t->spare) { t->stage = MICRORTS_THINK_TRANSLATE; t->index = 0u; break; }
                if (t->script == MICRORTS_DOCTRINE) microrts_command(w, m, t, t->rest[t->index++]);
                else microrts_melee(w, m, t, t->rest[t->index++]);
                break;
            case MICRORTS_THINK_TRANSLATE: {
                if (t->index >= m->count) { t->stage = MICRORTS_THINK_FILL; t->index = 0u; break; }
                uint32_t s = m->order[t->index];
                if (microrts_complete(w, m, t, s)) { microrts_forget(m, s); break; }
                t->index++;
                micrortsOrder o;
                if (w->act[s] != MICRORTS_IDLE || !microrts_execute(w, m, t, s, &o)) break;
                if (microrts_fits(w, t, &o)) microrts_accept(w, t, &o);
                if (microrts_reserves(o.act)) microrts_mark(&t->path, microrts_reserved(w, o.slot, o.arg));
                break;
            }
            case MICRORTS_THINK_FILL: {
                if (t->index >= t->owned) { t->stage = MICRORTS_THINK_DONE; break; }
                uint32_t s = t->ours[t->index++];
                if (w->act[s] == MICRORTS_IDLE && !microrts_has(&t->ordered, s)) microrts_wait(w, t, s);
                break;
            }
            default: t->stage = MICRORTS_THINK_DONE; break;
        }
    }
}

typedef struct { float theta[12]; } micrortsDoctrine;

static const micrortsDoctrine MICRORTS_PRESETS[4] = {
    { { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 12.0f, 0.0f, 1.0f, 0.0f, 0.3f, 0.3f, 0.0f } },
    { { 0.25f, 1.0f, 0.0f, 0.0f, 2.0f, 5.0f, 2.0f, 1.0f, 0.3f, 0.0f, 0.0f, 0.0f } },
    { { 0.25f, 0.0f, 1.0f, 0.0f, 2.0f, 5.0f, 3.0f, 1.0f, 0.3f, 0.0f, 0.0f, 0.0f } },
    { { 0.25f, 0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 2.0f, 1.0f, 0.5f, 0.0f, 0.0f, 0.2f } }
};

static const micrortsDoctrine MICRORTS_LOW = { { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };
static const micrortsDoctrine MICRORTS_HIGH = { { 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 12.0f, 12.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f } };

enum { MICRORTS_GAME, MICRORTS_SIDE, MICRORTS_PHASE, MICRORTS_LENGTH, MICRORTS_SELECTED, MICRORTS_MODE, MICRORTS_SWEEP, MICRORTS_CURSOR, MICRORTS_FIFO, MICRORTS_QUEUED, MICRORTS_EVENT, MICRORTS_FOE, MICRORTS_MEMBER, MICRORTS_WORDS };
enum { MICRORTS_LOST = 1, MICRORTS_KILLED, MICRORTS_PRODUCED, MICRORTS_REJECTED, MICRORTS_RETURNED };
enum { MICRORTS_REACH = 2, MICRORTS_SPAN = 2 * MICRORTS_REACH + 1, MICRORTS_PATCH = 5, MICRORTS_NEAREST = MICRORTS_PATCH + MICRORTS_SPAN * MICRORTS_SPAN, MICRORTS_CARD = MICRORTS_NEAREST + 2 };

typedef struct {
    micrortsWorld world;
    micrortsMind mind[2];
    micrortsThought thought;
    micrortsOrder plan[MICRORTS_SLOTS];
    uint32_t planned, idles, pick, foes;
    float card[MICRORTS_CARD];
    uint8_t addressed[MICRORTS_SLOTS], idle[MICRORTS_SLOTS], seen[MICRORTS_SLOTS], foe[MICRORTS_SLOTS];
    float winRate[LEAGUE_MEMBERS], theta[LEAGUE_MEMBERS * 12u];
} micrortsLane;

static micrortsLane *microrts_lane(uint32_t *extra) { return (micrortsLane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const micrortsLane *microrts_peek(const uint32_t *extra) { return (const micrortsLane *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }

static uint32_t microrts_seen(const micrortsLane *l, const micrortsKnobs *k, uint32_t side, uint32_t slot) {
    return !k->partial || l->world.owner[slot] == side || microrts_has(&l->world.views[side], l->world.cell[slot]);
}

static uint32_t microrts_next(micrortsLane *l) {
    uint32_t at = l->pick;
    for (uint32_t j = 0u; j < l->idles; j++) {
        at = at + 1u == l->idles ? 0u : at + 1u;
        if (!l->addressed[l->idle[at]]) { l->pick = at; return l->idle[at]; }
    }
    return MICRORTS_EMPTY;
}

static uint32_t microrts_targets(const micrortsWorld *w, uint32_t slot, uint8_t *out) {
    uint32_t n = 0u, range = MICRORTS_TYPES[w->kind[slot]].range;
    for (uint32_t i = 0u; i < w->units; i++) {
        uint32_t s = w->list[i];
        if (!microrts_foe(w, w->owner[slot], s) || microrts_square(w->width, w->cell[slot], w->cell[s]) > range * range) continue;
        uint32_t d = microrts_square(w->width, w->cell[slot], w->cell[s]), j = n++;
        while (j > 0u && microrts_square(w->width, w->cell[slot], w->cell[out[j - 1u]]) > d) { out[j] = out[j - 1u]; j--; }
        out[j] = (uint8_t)s;
    }
    return n;
}

static uint32_t microrts_modes(const micrortsWorld *w, uint32_t slot) {
    uint8_t targets[MICRORTS_SLOTS];
    switch (w->kind[slot]) {
        case MICRORTS_WORKER: case MICRORTS_BARRACKS: return 3u;
        case MICRORTS_RANGED: return 1u + microrts_targets(w, slot, targets);
        default: return 1u;
    }
}

static int microrts_smart(const micrortsWorld *w, uint32_t slot, uint32_t direction, uint32_t mode, micrortsOrder *o) {
    uint32_t kind = w->kind[slot], next;
    if (kind == MICRORTS_RANGED && mode > 0u) {
        uint8_t targets[MICRORTS_SLOTS];
        uint32_t n = microrts_targets(w, slot, targets);
        if (mode > n) return 0;
        *o = microrts_order(w, slot, MICRORTS_ATTACK, w->cell[targets[mode - 1u]], 0u, 0);
        return 1;
    }
    if (!microrts_adjacent(w->width, w->cell[slot], direction, &next)) return 0;
    uint32_t other = w->at[next];
    if (other == MICRORTS_EMPTY) {
        if (kind == MICRORTS_BASE) *o = microrts_order(w, slot, MICRORTS_PRODUCE, direction, MICRORTS_WORKER, 0);
        else if (kind == MICRORTS_BARRACKS) *o = microrts_order(w, slot, MICRORTS_PRODUCE, direction, MICRORTS_LIGHT + mode, 0);
        else if (kind == MICRORTS_WORKER && mode > 0u) *o = microrts_order(w, slot, MICRORTS_PRODUCE, direction, mode, 0);
        else if (MICRORTS_TYPES[kind].mobile) *o = microrts_order(w, slot, MICRORTS_MOVE, direction, 0u, 0);
        else return 0;
        return 1;
    }
    if (kind == MICRORTS_WORKER && mode == 0u && w->kind[other] == MICRORTS_RESOURCE && w->carry[slot] == 0) { *o = microrts_order(w, slot, MICRORTS_HARVEST, direction, 0u, 0); return 1; }
    if (kind == MICRORTS_WORKER && mode == 0u && w->kind[other] == MICRORTS_BASE && w->owner[other] == w->owner[slot] && w->carry[slot] > 0) { *o = microrts_order(w, slot, MICRORTS_RETURN, direction, 0u, 0); return 1; }
    if (MICRORTS_TYPES[kind].fighter && mode == 0u && microrts_foe(w, w->owner[slot], other)) { *o = microrts_order(w, slot, MICRORTS_ATTACK, next, 0u, 0); return 1; }
    return 0;
}

static void microrts_push(uint32_t *words, uint32_t code) {
    if (words[MICRORTS_QUEUED] >= 8u) return;
    words[MICRORTS_FIFO] |= code << (3u * words[MICRORTS_QUEUED]);
    words[MICRORTS_QUEUED]++;
}

static uint32_t microrts_code(const micrortsWorld *w, uint32_t slot) {
    return (0x5431760u >> (4u * w->kind[slot]) & 15u) + (w->kind[slot] == MICRORTS_WORKER && w->carry[slot] > 0);
}

static float microrts_badge(const micrortsWorld *w, uint32_t side, uint32_t slot) {
    return w->kind[slot] == MICRORTS_RESOURCE ? 1.0f : (w->owner[slot] == side ? 0.125f : -0.125f) * (float)microrts_code(w, slot);
}

static float microrts_glyph(const micrortsWorld *w, const micrortsKnobs *k, uint32_t side, int x, int y) {
    if (x < 0 || y < 0 || x >= (int)w->width || y >= (int)w->width) return 1.5f;
    uint32_t cell = (uint32_t)x + (uint32_t)y * w->width;
    if (k->partial && !microrts_has(&w->views[side], cell)) return 1.25f;
    return w->at[cell] == MICRORTS_EMPTY ? 0.0f : microrts_badge(w, side, w->at[cell]);
}

static void microrts_select(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t slot) {
    const micrortsWorld *w = &l->world;
    uint32_t side = words[MICRORTS_SIDE], far = 0xFFFFFFFFu;
    words[MICRORTS_SELECTED] = slot;
    words[MICRORTS_MODE] = 0u;
    memset(l->card, 0, sizeof l->card);
    if (slot == MICRORTS_EMPTY) return;
    uint32_t here = w->cell[slot];
    int x = (int)microrts_x(w->width, here), y = (int)microrts_y(w->width, here);
    words[MICRORTS_CURSOR] = here;
    l->card[0] = microrts_badge(w, side, slot);
    l->card[1] = (float)w->hp[slot] * 0.1f;
    l->card[3] = (float)x * 0.0625f;
    l->card[4] = (float)y * 0.0625f;
    for (int dy = -MICRORTS_REACH; dy <= MICRORTS_REACH; dy++)
        for (int dx = -MICRORTS_REACH; dx <= MICRORTS_REACH; dx++)
            l->card[MICRORTS_PATCH + (dy + MICRORTS_REACH) * MICRORTS_SPAN + dx + MICRORTS_REACH] = microrts_glyph(w, k, side, x + dx, y + dy);
    for (uint32_t i = l->foes; i-- > 0u;) {
        uint32_t s = l->foe[i], d = microrts_manhattan(w->width, here, w->cell[s]);
        if (d >= far) continue;
        far = d;
        l->card[MICRORTS_NEAREST] = (float)((int32_t)microrts_x(w->width, w->cell[s]) - x) * 0.0625f;
        l->card[MICRORTS_NEAREST + 1] = (float)((int32_t)microrts_y(w->width, w->cell[s]) - y) * 0.0625f;
    }
}

static void microrts_obey(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t action) {
    uint32_t sel = words[MICRORTS_SELECTED];
    if (sel == MICRORTS_EMPTY) return;
    if (action == 5u) {
        uint32_t modes = microrts_modes(&l->world, sel);
        if (modes == 1u) microrts_push(words, MICRORTS_REJECTED);
        words[MICRORTS_MODE] = (words[MICRORTS_MODE] + 1u) % modes;
        return;
    }
    if (action >= 1u) {
        micrortsOrder o;
        if (!microrts_smart(&l->world, sel, action - 1u, words[MICRORTS_MODE], &o)) { microrts_push(words, MICRORTS_REJECTED); return; }
        l->plan[l->planned++] = o;
    }
    l->addressed[sel] = 1u;
    microrts_select(l, words, k, microrts_next(l));
}

static uint32_t microrts_script(const micrortsKnobs *k, uint32_t seed, uint32_t game) {
    switch (k->opponent) {
        case 0u: return MICRORTS_RANDOM_BIASED;
        case 1u: return MICRORTS_WORKER_RUSH;
        case 2u: return MICRORTS_LIGHT_RUSH;
        case 3u: return MICRORTS_WORKER_RUSH + philox4x32(seed, ENV_RNG_STREAM, game, 0x464F4553u, 0u, 0u).x % 4u;
        default: return MICRORTS_DOCTRINE;
    }
}

static void microrts_open(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t seed) {
    micrortsWorld *w = &l->world;
    uint32_t side = words[MICRORTS_SIDE];
    memset(l->addressed, 0, sizeof l->addressed);
    l->planned = l->idles = 0u;
    uint32_t first = w->units;
    l->foes = 0u;
    for (uint32_t i = w->units; i-- > 0u;) {
        uint32_t s = w->list[i];
        l->seen[i] = (uint8_t)first;
        if (!microrts_seen(l, k, side, s)) continue;
        first = i;
        if (microrts_foe(w, side, s)) l->foe[l->foes++] = (uint8_t)s;
    }
    for (uint32_t i = w->units; i-- > 0u;) {
        if (l->seen[i] != w->units) break;
        l->seen[i] = (uint8_t)first;
    }
    for (uint32_t i = 0u; i < w->units; i++) {
        uint32_t s = w->list[i];
        if (w->owner[s] == side && w->act[s] == MICRORTS_IDLE) l->idle[l->idles++] = (uint8_t)s;
    }
    words[MICRORTS_PHASE] = 0u;
    words[MICRORTS_LENGTH] = k->substeps + l->idles;
    l->pick = l->idles ? l->idles - 1u : 0u;
    microrts_select(l, words, k, l->idles ? microrts_next(l) : MICRORTS_EMPTY);
    microrts_begin(&l->thought, 1u - side, words[MICRORTS_FOE], seed, words[MICRORTS_GAME], MICRORTS_WORLD_DRAWS, k->partial ? &w->views[1u - side] : 0);
    if (words[MICRORTS_FOE] == MICRORTS_DOCTRINE) memcpy(l->thought.theta, &l->theta[12u * words[MICRORTS_MEMBER]], sizeof l->thought.theta);
}

static void microrts_start(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t seed) {
    uint32_t game = words[MICRORTS_GAME];
    microrts_load(&l->world, k->width, k->partial);
    memset(l->mind, 0, sizeof l->mind);
    words[MICRORTS_SIDE] = game & 1u;
    words[MICRORTS_FOE] = microrts_script(k, seed, game);
    words[MICRORTS_MEMBER] = k->opponent == 4u ? leaguePick(l->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, game, 0x5049434Bu, 0u, 0u).x) : 0u;
    words[MICRORTS_SWEEP] = 0u;
    microrts_open(l, words, k, seed);
}

static void microrts_league(micrortsLane *l, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->winRate[i] = 0.5f;
    for (uint32_t i = 0u; i < 4u; i++) memcpy(&l->theta[12u * i], MICRORTS_PRESETS[i].theta, sizeof MICRORTS_PRESETS[i].theta);
    for (uint32_t i = 4u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < 12u; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = MICRORTS_PRESETS[i - 4u].theta[d + j] + 0.1f * (MICRORTS_HIGH.theta[d + j] - MICRORTS_LOW.theta[d + j]) * n[j];
                l->theta[12u * i + d + j] = fminf(fmaxf(value, MICRORTS_LOW.theta[d + j]), MICRORTS_HIGH.theta[d + j]);
            }
        }
}


static void microrts_obs(const micrortsLane *l, const micrortsKnobs *k, const uint32_t *words, float *obs) {
    const micrortsWorld *w = &l->world;
    uint32_t side = words[MICRORTS_SIDE], cursor = words[MICRORTS_CURSOR], at = words[MICRORTS_SWEEP];
    float *tail = obs + MICRORTS_CARD;
    memcpy(obs, l->card, sizeof l->card);
    obs[2] = (float)words[MICRORTS_MODE] * (1.0f / 3.0f);
    tail[0] = tail[1] = tail[2] = tail[3] = 0.0f;
    if (at < w->units && microrts_seen(l, k, side, w->list[at])) {
        uint32_t s = w->list[at];
        tail[0] = microrts_badge(w, side, s);
        tail[1] = (float)((int)microrts_x(w->width, w->cell[s]) - (int)microrts_x(w->width, cursor)) * 0.0625f;
        tail[2] = (float)((int)microrts_y(w->width, w->cell[s]) - (int)microrts_y(w->width, cursor)) * 0.0625f;
        tail[3] = (float)(w->kind[s] == MICRORTS_RESOURCE ? w->carry[s] : w->hp[s]) * 0.1f;
    }
    tail[4] = (float)w->gold[side] * 0.1f;
    tail[5] = (float)w->gold[1u - side] * 0.1f;
    tail[6] = (float)w->time / (float)k->maxCycles;
    tail[7] = (float)words[MICRORTS_PHASE] / (float)words[MICRORTS_LENGTH];
    tail[8] = (float)words[MICRORTS_EVENT] * 0.2f;
}

static void microrts_sweep(const micrortsLane *l, uint32_t *words) {
    uint32_t at = words[MICRORTS_SWEEP];
    words[MICRORTS_SWEEP] = at < l->world.units ? l->seen[at] : l->world.units ? l->seen[l->world.units - 1u] : 0u;
}

static float microrts_tick(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t *over) {
    micrortsWorld *w = &l->world;
    uint32_t side = words[MICRORTS_SIDE];
    micrortsTally tally;
    memset(&tally, 0, sizeof tally);
    uint8_t rejected[MICRORTS_SLOTS];
    memset(rejected, 0, sizeof rejected);
    for (uint32_t p = 0u; p < 2u; p++) {
        if (p == side) microrts_issue(w, l->plan, l->planned, rejected);
        else microrts_issue(w, l->thought.out, l->thought.orders, 0);
    }
    for (uint32_t i = 0u; i < l->planned; i++)
        if (rejected[i]) microrts_push(words, MICRORTS_REJECTED);
    microrts_cycle(w, l->mind, &tally);
    for (uint32_t i = 0u; i < tally.lost[side]; i++) microrts_push(words, MICRORTS_LOST);
    for (uint32_t i = 0u; i < tally.lost[1u - side]; i++) microrts_push(words, MICRORTS_KILLED);
    for (uint32_t i = 0u; i < tally.produced[side]; i++) microrts_push(words, MICRORTS_PRODUCED);
    if (tally.returned[side]) microrts_push(words, MICRORTS_RETURNED);
    uint32_t mine = microrts_standing(w, side), theirs = microrts_standing(w, 1u - side);
    *over = !mine || !theirs || w->time >= k->maxCycles;
    if (!*over) return 0.0f;
    return mine && !theirs ? 1.0f : theirs && !mine ? -1.0f : 0.0f;
}

static uint32_t microrts_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static void microrts_finish(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t seed, float reward) {
    if (k->opponent == 4u) {
        uint32_t member = words[MICRORTS_MEMBER];
        leagueScore(l->winRate, member, reward > 0.0f ? 1.0f : reward < 0.0f ? 0.0f : 0.5f);
        leagueRenew(l->winRate, l->theta, 12u, MICRORTS_LOW.theta, MICRORTS_HIGH.theta, seed, words[MICRORTS_GAME]);
    }
    words[MICRORTS_GAME]++;
    microrts_start(l, words, k, seed);
}

static __attribute__((noinline)) float microrts_turn(micrortsLane *l, uint32_t *words, const micrortsKnobs *k, uint32_t seed, uint32_t *done) {
    microrts_ponder(&l->world, &l->mind[1u - words[MICRORTS_SIDE]], &l->thought);
    uint32_t over;
    float reward = microrts_tick(l, words, k, &over);
    if (over) { *done = 1u; microrts_finish(l, words, k, seed, reward); }
    else microrts_open(l, words, k, seed);
    return reward;
}

float microrts_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    const micrortsKnobs *k = &MICRORTS_KNOBS[instance];
    micrortsLane *l = microrts_lane(extra);
    microrts_obey(l, words, k, microrts_action(ctrl));
    uint32_t phase = ++words[MICRORTS_PHASE];
    float reward = 0.0f;
    *done = 0u;
    if (words[MICRORTS_SELECTED] == MICRORTS_EMPTY || phase >= words[MICRORTS_LENGTH])
        do reward = microrts_turn(l, words, k, seed, done); while (!*done && l->idles == 0u);
    microrts_sweep(l, words);
    words[MICRORTS_EVENT] = words[MICRORTS_FIFO] & 7u;
    if (words[MICRORTS_QUEUED]) { words[MICRORTS_FIFO] >>= 3u; words[MICRORTS_QUEUED]--; }
    microrts_obs(l, k, words, obs);
    return reward;
}

void microrts_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    microrts_obs(microrts_peek(extra), &MICRORTS_KNOBS[instance], words, obs);
}

uint32_t microrts_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    uint32_t width = MICRORTS_KNOBS[instance].width, count = 2u + (width * width < MICRORTS_SLOTS ? width * width : MICRORTS_SLOTS);
    if (!out) return count;
    const micrortsWorld *w = &microrts_peek(extra)->world;
    memset(out, 0, count * sizeof(uint32_t));
    out[0] = w->time;
    out[1] = ((uint32_t)w->gold[0] & 0xFFFFu) | ((uint32_t)w->gold[1] & 0xFFFFu) << 16;
    for (uint32_t i = 0u; i < w->units && i + 2u < count; i++) {
        uint32_t s = w->list[i], arg = w->arg[s];
        if (w->act[s] == MICRORTS_ATTACK) arg = (uint32_t)((int32_t)microrts_x(width, arg) - (int32_t)microrts_x(width, w->cell[s]) + 3) | (uint32_t)((int32_t)microrts_y(width, arg) - (int32_t)microrts_y(width, w->cell[s]) + 3) << 3;
        else if (w->act[s] == MICRORTS_PRODUCE) arg |= (uint32_t)w->make[s] << 2;
        else if (w->act[s] == MICRORTS_NONE || w->act[s] == MICRORTS_IDLE) arg = 0u;
        out[2u + i] = w->cell[s] | (uint32_t)w->kind[s] << 8 | (uint32_t)w->owner[s] << 11 | (uint32_t)w->act[s] << 13
                      | ((uint32_t)w->hp[s] & 15u) << 16 | ((uint32_t)w->carry[s] & 31u) << 20 | (arg & 63u) << 25;
    }
    return count;
}

enum { MICRORTS_PRESSES = 3 * MICRORTS_SLOTS, MICRORTS_CANDIDATES = 9 };

static const micrortsDoctrine MICRORTS_GAMBITS[5] = {
    { { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 12.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f } },
    { { 0.25f, 1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.0f, 0.0f } },
    { { 0.25f, 1.0f, 0.0f, 0.0f, 2.0f, 5.0f, 6.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
    { { 0.25f, 0.0f, 1.0f, 0.0f, 2.0f, 5.0f, 6.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f } },
    { { 0.25f, 0.5f, 0.5f, 0.0f, 3.0f, 5.0f, 9.0f, 1.0f, 0.0f, 0.5f, 0.0f, 0.0f } }
};

typedef struct {
    uint32_t game, candidate, favorite, time, count, next, end, planned;
    float outcome;
    uint8_t presses[MICRORTS_PRESSES];
    micrortsOrder orders[MICRORTS_SLOTS];
    uint32_t words[MICRORTS_WORDS];
    micrortsThought mine;
    micrortsLane sim;
} micrortsOracle;

static micrortsOracle *microrts_oracleState(uint32_t *extra) {
    return (micrortsOracle *)(((uintptr_t)(extra + (sizeof(micrortsLane) + 15u) / sizeof(uint32_t)) + 15u) & ~(uintptr_t)15u);
}

static uint32_t microrts_press(const micrortsWorld *w, uint32_t slot, const micrortsOrder *o, uint32_t *direction) {
    *direction = o->arg;
    if (o->act == MICRORTS_PRODUCE) return w->kind[slot] == MICRORTS_BARRACKS ? o->make - MICRORTS_LIGHT : w->kind[slot] == MICRORTS_WORKER ? o->make : 0u;
    if (o->act != MICRORTS_ATTACK) return 0u;
    uint32_t n;
    for (uint32_t d = 0u; d < 4u; d++)
        if (microrts_adjacent(w->width, w->cell[slot], d, &n) && n == o->arg) { *direction = d; return 0u; }
    uint8_t targets[MICRORTS_SLOTS];
    uint32_t k = microrts_targets(w, slot, targets);
    *direction = 0u;
    for (uint32_t j = 0u; j < k; j++)
        if (w->cell[targets[j]] == o->arg) return j + 1u;
    return 0u;
}

static uint32_t microrts_realize(const micrortsWorld *w, const micrortsThought *t, uint32_t side, uint32_t budget, uint8_t *presses, micrortsOrder *orders, uint32_t *ordered) {
    const micrortsOrder *byUnit[MICRORTS_SLOTS];
    memset(byUnit, 0, sizeof byUnit);
    for (uint32_t i = 0u; i < t->orders; i++)
        if (t->out[i].act != MICRORTS_NONE) byUnit[t->out[i].slot] = &t->out[i];
    uint32_t count = 0u, pending = 0u;
    *ordered = 0u;
    for (uint32_t i = 0u; i < w->units; i++) {
        uint32_t s = w->list[i], direction;
        if (w->owner[s] != side || w->act[s] != MICRORTS_IDLE) continue;
        const micrortsOrder *o = byUnit[s];
        if (!o) { pending++; continue; }
        uint32_t mode = microrts_press(w, s, o, &direction);
        if (count + pending + mode + 1u > budget) break;
        for (; pending > 0u; pending--) presses[count++] = 0u;
        for (uint32_t j = 0u; j < mode; j++) presses[count++] = 5u;
        presses[count++] = (uint8_t)(1u + direction);
        orders[(*ordered)++] = *o;
    }
    return count;
}

static uint32_t microrts_idle(const micrortsWorld *w, uint32_t side) {
    uint32_t idle = 0u;
    for (uint32_t i = 0u; i < w->units; i++) idle += w->owner[w->list[i]] == side && w->act[w->list[i]] == MICRORTS_IDLE;
    return idle;
}

static void microrts_foresee(micrortsOracle *o, micrortsLane *l, const micrortsKnobs *k, uint32_t seed, uint32_t candidate, uint32_t *ordered) {
    micrortsWorld *w = &l->world;
    uint32_t side = o->words[MICRORTS_SIDE];
    microrts_begin(&o->mine, side, candidate < 4u ? MICRORTS_WORKER_RUSH + candidate : MICRORTS_DOCTRINE, seed, o->words[MICRORTS_GAME], MICRORTS_ORACLE_DRAWS, k->partial ? &w->views[side] : 0);
    if (candidate >= 4u) memcpy(o->mine.theta, MICRORTS_GAMBITS[candidate - 4u].theta, sizeof o->mine.theta);
    microrts_ponder(w, &l->mind[side], &o->mine);
    o->count = microrts_realize(w, &o->mine, side, k->substeps + microrts_idle(w, side), o->presses, o->orders, ordered);
}

static float microrts_rollout(micrortsOracle *o, const micrortsKnobs *k, uint32_t seed, uint32_t candidate, uint32_t *end) {
    micrortsLane *l = &o->sim;
    micrortsWorld *w = &l->world;
    uint32_t side = o->words[MICRORTS_SIDE], foe = o->words[MICRORTS_FOE];
    for (;;) {
        uint32_t ordered;
        microrts_foresee(o, l, k, seed, candidate, &ordered);
        microrts_begin(&l->thought, 1u - side, foe, seed, o->words[MICRORTS_GAME], MICRORTS_ORACLE_DRAWS, k->partial ? &w->views[1u - side] : 0);
        if (foe == MICRORTS_DOCTRINE) memcpy(l->thought.theta, &l->theta[12u * o->words[MICRORTS_MEMBER]], sizeof l->thought.theta);
        microrts_ponder(w, &l->mind[1u - side], &l->thought);
        for (uint32_t p = 0u; p < 2u; p++) {
            if (p == side) microrts_issue(w, o->orders, ordered, 0);
            else microrts_issue(w, l->thought.out, l->thought.orders, 0);
        }
        microrts_cycle(w, l->mind, 0);
        uint32_t ours = microrts_standing(w, side), theirs = microrts_standing(w, 1u - side);
        if (!ours || !theirs || w->time >= k->maxCycles) {
            *end = w->time;
            return ours && !theirs ? 1.0f : theirs && !ours ? -1.0f : 0.0f;
        }
    }
}

static void microrts_veil(micrortsLane *l, uint32_t side) {
    micrortsWorld *w = &l->world;
    uint32_t count, home = 256u;
    const micrortsPlacement *map = microrts_map(w->width, &count);
    for (uint32_t i = 0u; i < count; i++)
        if (map[i].kind == MICRORTS_BASE && map[i].owner == 1u - side) home = map[i].cell;
    for (uint32_t i = w->units; i-- > 0u;) {
        uint32_t s = w->list[i];
        if (w->owner[s] != 1u - side || microrts_has(&w->views[side], w->cell[s])) continue;
        if (w->kind[s] != MICRORTS_BASE || w->cell[s] != home) microrts_remove(w, l->mind, s);
        else if (w->act[s] != MICRORTS_IDLE) microrts_unqueue(w, s);
    }
    memset(&l->mind[1u - side], 0, sizeof l->mind[1u - side]);
}

static void microrts_plan(micrortsOracle *o, const micrortsLane *l, const micrortsKnobs *k, const uint32_t *words, uint32_t seed) {
    memcpy(o->words, words, sizeof o->words);
    uint32_t guess = philox4x32(seed, ENV_RNG_STREAM, words[MICRORTS_GAME], l->world.time, 0x464F4553u, MICRORTS_ORACLE_DRAWS).x;
    if (k->opponent == 3u) o->words[MICRORTS_FOE] = MICRORTS_WORKER_RUSH + guess % 4u;
    if (k->opponent == 4u) o->words[MICRORTS_MEMBER] = leaguePick(l->winRate, guess);
    float top = -3.0f;
    for (uint32_t i = 0u; i < MICRORTS_CANDIDATES; i++) {
        uint32_t candidate = (o->favorite + i) % MICRORTS_CANDIDATES, end;
        o->sim = *l;
        if (k->partial) microrts_veil(&o->sim, words[MICRORTS_SIDE]);
        float outcome = microrts_rollout(o, k, seed, candidate, &end);
        float value = outcome * 2.0f + (outcome < 0.0f ? (float)end / (float)k->maxCycles : 0.0f);
        if (value > top) { top = value; o->candidate = candidate; o->outcome = outcome; o->end = end; }
        if (outcome > 0.0f) break;
    }
    if (o->outcome > 0.0f) o->favorite = o->candidate;
    o->planned = l->world.time;
}

void microrts_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)obs; (void)shared;
    const micrortsKnobs *k = &MICRORTS_KNOBS[instance];
    micrortsLane *l = microrts_lane(extra);
    micrortsOracle *o = microrts_oracleState(extra);
    if (o->game != words[MICRORTS_GAME] + 1u) {
        microrts_plan(o, l, k, words, seed);
        o->game = words[MICRORTS_GAME] + 1u;
        o->time = 0xFFFFFFFFu;
    }
    if (o->time != l->world.time) {
        uint32_t ordered;
        if (o->outcome <= 0.0f && l->world.time >= o->planned + 32u) microrts_plan(o, l, k, words, seed);
        memcpy(o->words, words, sizeof o->words);
        microrts_foresee(o, l, k, seed, o->candidate, &ordered);
        o->next = 0u;
        o->time = l->world.time;
    }
    ctrl[0] = o->next < o->count ? (float)o->presses[o->next++] : 0.0f;
}

void microrts_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const micrortsKnobs *k = &MICRORTS_KNOBS[instance];
    micrortsLane *l = microrts_lane(extra);
    memset(words, 0, MICRORTS_WORDS * sizeof(uint32_t));
    memset(microrts_oracleState(extra), 0, sizeof(micrortsOracle));
    microrts_league(l, seed);
    microrts_start(l, words, k, seed);
}

uint32_t microrts_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)((sizeof(micrortsLane) + 15u + sizeof(micrortsOracle) + 15u) / sizeof(uint32_t)) + 4u;
}
