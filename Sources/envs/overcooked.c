#include "../env.h"

typedef struct { uint32_t terrain, width, height, x0, y0, x1, y1, horizon, cookTime, commisPatience; float soupReward; } overcookedKnobs;

static const overcookedKnobs OVERCOOKED_KNOBS[4] = {
    { 0u, 5u, 4u, 1u, 2u, 3u, 1u, 400u, 20u, 3u, 20.0f },
    { 20u, 9u, 5u, 6u, 2u, 1u, 3u, 400u, 20u, 3u, 20.0f },
    { 65u, 5u, 5u, 2u, 1u, 1u, 2u, 400u, 20u, 3u, 20.0f },
    { 90u, 8u, 5u, 3u, 3u, 3u, 1u, 400u, 20u, 3u, 20.0f }
};

static const uint32_t OVERCOOKED_TERRAIN[130] = {
    1, 1, 4, 1, 1,
    2, 0, 0, 0, 2,
    1, 0, 0, 0, 1,
    1, 3, 1, 5, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 0, 1, 5, 1, 2, 1, 0, 5,
    1, 0, 0, 0, 4, 0, 0, 0, 1,
    1, 0, 0, 0, 4, 0, 0, 0, 1,
    1, 1, 1, 3, 1, 3, 1, 1, 1,
    1, 1, 1, 4, 1,
    1, 0, 0, 0, 4,
    3, 0, 1, 0, 1,
    2, 0, 0, 0, 1,
    1, 2, 5, 1, 1,
    1, 1, 1, 4, 4, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 1,
    3, 0, 1, 1, 1, 1, 0, 5,
    1, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 2, 2, 1, 1, 1
};

enum { OVERCOOKED_FLOOR, OVERCOOKED_COUNTER, OVERCOOKED_ONIONS, OVERCOOKED_DISHES, OVERCOOKED_POT, OVERCOOKED_WINDOW, OVERCOOKED_VOID };
enum { OVERCOOKED_EMPTY, OVERCOOKED_ONION, OVERCOOKED_DISH, OVERCOOKED_SOUP };
enum { OVERCOOKED_STAY = 4, OVERCOOKED_INTERACT, OVERCOOKED_ACTIONS };
enum { OVERCOOKED_TO_ONIONS, OVERCOOKED_TO_DISHES, OVERCOOKED_TO_WINDOW, OVERCOOKED_TO_POT, OVERCOOKED_GOALS = OVERCOOKED_TO_POT + 2, OVERCOOKED_NO_GOAL = 7 };
enum { OVERCOOKED_AGENT, OVERCOOKED_PARTNER, OVERCOOKED_POTS, OVERCOOKED_COUNTERS, OVERCOOKED_CLOCK = OVERCOOKED_COUNTERS + 2, OVERCOOKED_GAME, OVERCOOKED_LAST, OVERCOOKED_WORDS };
enum { OVERCOOKED_CELLS = 80, OVERCOOKED_FULL = 3, OVERCOOKED_UNREACHABLE = 255 };

typedef struct {
    uint8_t tile[OVERCOOKED_CELLS], goal[OVERCOOKED_CELLS], route[OVERCOOKED_GOALS][OVERCOOKED_CELLS];
    uint32_t pots;
} overcookedMap;

enum { OVERCOOKED_MAP_WORDS = (sizeof(overcookedMap) + sizeof(float) - 1u) / sizeof(float) };

static const overcookedMap *overcooked_map(uint32_t instance, const float *shared) { return (const overcookedMap *)(shared + instance * OVERCOOKED_MAP_WORDS); }

static int overcooked_blocked(const void *grid, int x, int y) {
    const overcookedKnobs *k = (const overcookedKnobs *)grid;
    if (x < 0 || y < 0 || x >= (int)k->width || y >= (int)k->height) return 1;
    return OVERCOOKED_TERRAIN[k->terrain + (uint32_t)y * k->width + (uint32_t)x] != OVERCOOKED_FLOOR;
}

static uint32_t overcooked_goalOf(uint32_t code, uint32_t index) {
    switch (code) {
        case OVERCOOKED_ONIONS: return OVERCOOKED_TO_ONIONS;
        case OVERCOOKED_DISHES: return OVERCOOKED_TO_DISHES;
        case OVERCOOKED_WINDOW: return OVERCOOKED_TO_WINDOW;
        case OVERCOOKED_POT: return OVERCOOKED_TO_POT + index;
        default: return OVERCOOKED_NO_GOAL;
    }
}

static void overcooked_build(const overcookedKnobs *k, overcookedMap *m) {
    uint32_t counters = 0u, pots = 0u;
    for (uint32_t c = 0u; c < OVERCOOKED_CELLS; c++) {
        m->tile[c] = OVERCOOKED_VOID;
        m->goal[c] = OVERCOOKED_NO_GOAL;
    }
    for (uint32_t y = 0u; y < k->height; y++)
        for (uint32_t x = 0u; x < k->width; x++) {
            uint32_t code = OVERCOOKED_TERRAIN[k->terrain + y * k->width + x], index = 0u;
            if (code == OVERCOOKED_COUNTER) index = counters++;
            if (code == OVERCOOKED_POT) index = pots++;
            m->tile[x | y << 4] = (uint8_t)(code | index << 3);
            m->goal[x | y << 4] = (uint8_t)overcooked_goalOf(code, index);
        }
    m->pots = pots;
    for (uint32_t g = 0u; g < OVERCOOKED_GOALS; g++) {
        uint8_t nearest[256], field[256];
        for (uint32_t i = 0u; i < 256u; i++) nearest[i] = OVERCOOKED_UNREACHABLE;
        for (uint32_t c = 0u; c < OVERCOOKED_CELLS; c++) {
            if (m->goal[c] != g) continue;
            gridDistance(overcooked_blocked, k, (int)(c & 15u), (int)(c >> 4), field);
            for (uint32_t i = 0u; i < 256u; i++) nearest[i] = field[i] < nearest[i] ? field[i] : nearest[i];
        }
        for (uint32_t c = 0u; c < OVERCOOKED_CELLS; c++) {
            int x = (int)(c & 15u), y = (int)(c >> 4);
            uint32_t far = nearest[(x + 1) | (y + 1) << 4];
            m->route[g][c] = m->tile[c] != OVERCOOKED_FLOOR || far == OVERCOOKED_UNREACHABLE ? OVERCOOKED_UNREACHABLE : (uint8_t)((uint32_t)gridDescend(nearest, x, y) | (far - 1u) << 2);
        }
    }
}

uint64_t overcooked_shared(float *table) {
    uint32_t instances = sizeof OVERCOOKED_KNOBS / sizeof OVERCOOKED_KNOBS[0];
    if (table)
        for (uint32_t i = 0u; i < instances; i++) overcooked_build(&OVERCOOKED_KNOBS[i], (overcookedMap *)(table + i * OVERCOOKED_MAP_WORDS));
    return (uint64_t)instances * OVERCOOKED_MAP_WORDS;
}

static inline __attribute__((always_inline)) uint32_t overcooked_ahead(uint32_t cell, uint32_t direction) {
    int dx, dy;
    gridDelta(direction, &dx, &dy);
    return (uint32_t)((int)cell + dx + 16 * dy);
}

static uint32_t overcooked_ready(const overcookedKnobs *k) { return OVERCOOKED_FULL + k->cookTime; }

static inline __attribute__((always_inline)) uint32_t overcooked_rank(const overcookedKnobs *k, uint32_t pot) { return pot == overcooked_ready(k) ? 5u : pot >= OVERCOOKED_FULL ? 4u : pot + 1u; }

static inline __attribute__((always_inline)) uint32_t overcooked_nearer(const overcookedMap *m, uint32_t cell, uint32_t mask) {
    if (mask != 3u) return mask >> 1;
    return (m->route[OVERCOOKED_TO_POT + 1u][cell] >> 2) < (m->route[OVERCOOKED_TO_POT][cell] >> 2);
}

static inline __attribute__((always_inline)) uint32_t overcooked_aim(const overcookedKnobs *k, const overcookedMap *m, uint32_t self, uint32_t pots) {
    uint32_t cell = self & 255u, held = self >> 10 & 3u, two = m->pots > 1u, a = pots & 255u, b = pots >> 8 & 255u;
    if (held == OVERCOOKED_SOUP) return OVERCOOKED_TO_WINDOW;
    if (held == OVERCOOKED_ONION) {
        uint32_t open = (uint32_t)(a < OVERCOOKED_FULL) | (uint32_t)(two && b < OVERCOOKED_FULL) << 1;
        return OVERCOOKED_TO_POT + overcooked_nearer(m, cell, open ? open : 1u | two << 1);
    }
    if (held == OVERCOOKED_DISH) {
        uint32_t rankA = overcooked_rank(k, a), rankB = two ? overcooked_rank(k, b) : 0u;
        return OVERCOOKED_TO_POT + overcooked_nearer(m, cell, rankA == rankB ? 3u : rankA > rankB ? 1u : 2u);
    }
    return a >= OVERCOOKED_FULL || (two && b >= OVERCOOKED_FULL) ? OVERCOOKED_TO_DISHES : OVERCOOKED_TO_ONIONS;
}

static inline __attribute__((always_inline)) uint32_t overcooked_toward(const overcookedMap *m, uint32_t self, uint32_t goal) {
    uint32_t cell = self & 255u, route = m->route[goal][cell];
    if (route == OVERCOOKED_UNREACHABLE) return OVERCOOKED_STAY;
    if (route < 4u && m->goal[overcooked_ahead(cell, self >> 8 & 3u)] == goal) return OVERCOOKED_INTERACT;
    return route & 3u;
}

static inline __attribute__((always_inline)) uint32_t overcooked_commis(const overcookedKnobs *k, const overcookedMap *m, uint32_t seed, const uint32_t *words, uint32_t *chance) {
    uint32_t self = words[OVERCOOKED_PARTNER];
    if ((self >> 12 & 3u) >= k->commisPatience) {
        *chance = 1u;
        return (uint32_t)(((uint64_t)philox4x32(seed, ENV_RNG_STREAM, words[OVERCOOKED_GAME], words[OVERCOOKED_CLOCK] & 0xFFFFu, 1u, 0u).x * OVERCOOKED_ACTIONS) >> 32);
    }
    return overcooked_toward(m, self, overcooked_aim(k, m, self, words[OVERCOOKED_POTS]));
}

static inline __attribute__((always_inline)) uint32_t overcooked_use(const overcookedKnobs *k, const overcookedMap *m, uint32_t self, uint32_t *words, float *reward) {
    uint32_t held = self >> 10 & 3u, tile = m->tile[overcooked_ahead(self & 255u, self >> 8 & 3u)], index = tile >> 3;
    switch (tile & 7u) {
        case OVERCOOKED_COUNTER: {
            uint32_t *slot = &words[OVERCOOKED_COUNTERS + (index >> 4)], shift = (index & 15u) * 2u, item = *slot >> shift & 3u;
            if (held != OVERCOOKED_EMPTY && item == OVERCOOKED_EMPTY) {
                *slot |= held << shift;
                held = OVERCOOKED_EMPTY;
            } else if (held == OVERCOOKED_EMPTY && item != OVERCOOKED_EMPTY) {
                *slot &= ~(3u << shift);
                held = item;
            }
            break;
        }
        case OVERCOOKED_ONIONS:
            if (held == OVERCOOKED_EMPTY) held = OVERCOOKED_ONION;
            break;
        case OVERCOOKED_DISHES:
            if (held == OVERCOOKED_EMPTY) held = OVERCOOKED_DISH;
            break;
        case OVERCOOKED_POT: {
            uint32_t shift = index * 8u, pot = words[OVERCOOKED_POTS] >> shift & 255u;
            if (held == OVERCOOKED_ONION && pot < OVERCOOKED_FULL) {
                words[OVERCOOKED_POTS] += 1u << shift;
                held = OVERCOOKED_EMPTY;
            } else if (held == OVERCOOKED_DISH && pot == overcooked_ready(k)) {
                words[OVERCOOKED_POTS] &= ~(255u << shift);
                held = OVERCOOKED_SOUP;
            }
            break;
        }
        case OVERCOOKED_WINDOW:
            if (held == OVERCOOKED_SOUP) {
                held = OVERCOOKED_EMPTY;
                *reward += k->soupReward;
            }
            break;
        default: break;
    }
    return (self & ~(3u << 10)) | held << 10;
}

static void overcooked_start(const overcookedKnobs *k, uint32_t seed, uint32_t *words) {
    uint32_t seat = philox4x32(seed, ENV_RNG_STREAM, words[OVERCOOKED_GAME], 0u, 0u, 0u).x >> 31;
    uint32_t first = k->x0 | k->y0 << 4, second = k->x1 | k->y1 << 4;
    words[OVERCOOKED_AGENT] = seat ? second : first;
    words[OVERCOOKED_PARTNER] = seat ? first : second;
    words[OVERCOOKED_POTS] = 0u;
    words[OVERCOOKED_COUNTERS] = 0u;
    words[OVERCOOKED_COUNTERS + 1u] = 0u;
    words[OVERCOOKED_CLOCK] = seat << 16;
}

static float overcooked_third(uint32_t n) { return (float)n * (1.0f / 3.0f); }

static inline __attribute__((always_inline)) void overcooked_player(uint32_t self, float *obs) {
    int dx, dy;
    gridDelta(self >> 8 & 3u, &dx, &dy);
    obs[0] = (float)(self & 15u) * 0.125f;
    obs[1] = (float)(self >> 4 & 15u) * 0.25f;
    obs[2] = (float)dx;
    obs[3] = (float)dy;
    obs[4] = overcooked_third(self >> 10 & 3u);
}

static inline __attribute__((always_inline)) void overcooked_obs(const overcookedKnobs *k, const overcookedMap *m, const uint32_t *words, float *obs) {
    uint32_t self = words[OVERCOOKED_AGENT], tile = m->tile[overcooked_ahead(self & 255u, self >> 8 & 3u)], index = tile >> 3, pots = words[OVERCOOKED_POTS];
    overcooked_player(self, obs);
    overcooked_player(words[OVERCOOKED_PARTNER], obs + 5);
    obs[10] = (float)(tile & 7u) * 0.2f;
    obs[11] = (tile & 7u) == OVERCOOKED_COUNTER ? overcooked_third(words[OVERCOOKED_COUNTERS + (index >> 4)] >> (index & 15u) * 2u & 3u) : 0.0f;
    for (uint32_t p = 0u; p < 2u; p++) {
        uint32_t pot = pots >> 8u * p & 255u;
        obs[12 + 2 * p] = overcooked_third(pot < OVERCOOKED_FULL ? pot : OVERCOOKED_FULL);
        obs[13 + 2 * p] = pot > OVERCOOKED_FULL ? (float)(pot - OVERCOOKED_FULL) * (1.0f / (float)k->cookTime) : 0.0f;
    }
    obs[16] = (float)(words[OVERCOOKED_CLOCK] & 0xFFFFu) * (1.0f / (float)k->horizon);
}

static inline __attribute__((always_inline)) uint32_t overcooked_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)OVERCOOKED_ACTIONS ? (uint32_t)c : OVERCOOKED_STAY;
}

static inline __attribute__((always_inline)) uint32_t overcooked_target(const overcookedMap *m, uint32_t self, uint32_t act) {
    uint32_t cell = self & 255u, next = overcooked_ahead(cell, act);
    return act < OVERCOOKED_STAY && m->tile[next] == OVERCOOKED_FLOOR ? next : cell;
}

static inline __attribute__((always_inline)) uint32_t overcooked_turn(uint32_t self, uint32_t act) { return act < OVERCOOKED_STAY ? (self & ~(3u << 8)) | act << 8 : self; }

static inline __attribute__((always_inline)) float overcooked_play(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, float *obs, uint32_t *done, const float *shared) {
    const overcookedKnobs *k = &OVERCOOKED_KNOBS[instance];
    const overcookedMap *m = overcooked_map(instance, shared);
    uint32_t chance = 0u, mine = overcooked_action(ctrl), theirs = overcooked_commis(k, m, seed, words, &chance);
    uint32_t clock = words[OVERCOOKED_CLOCK], agent = words[OVERCOOKED_AGENT], partner = words[OVERCOOKED_PARTNER];
    float reward = 0.0f;
    if (clock >> 16) {
        if (theirs == OVERCOOKED_INTERACT) partner = overcooked_use(k, m, partner, words, &reward);
        if (mine == OVERCOOKED_INTERACT) agent = overcooked_use(k, m, agent, words, &reward);
    } else {
        if (mine == OVERCOOKED_INTERACT) agent = overcooked_use(k, m, agent, words, &reward);
        if (theirs == OVERCOOKED_INTERACT) partner = overcooked_use(k, m, partner, words, &reward);
    }
    uint32_t from = agent & 255u, to = overcooked_target(m, agent, mine), partnerFrom = partner & 255u, partnerTo = overcooked_target(m, partner, theirs);
    if (to == partnerTo || (to == partnerFrom && partnerTo == from)) {
        to = from;
        partnerTo = partnerFrom;
    }
    uint32_t still = partner >> 12 & 3u;
    still = partnerTo != partnerFrom ? 0u : still < k->commisPatience ? still + 1u : still;
    words[OVERCOOKED_AGENT] = (overcooked_turn(agent, mine) & ~255u) | to;
    words[OVERCOOKED_PARTNER] = (overcooked_turn(partner, theirs) & ~(255u | 3u << 12)) | partnerTo | still << 12;
    uint32_t pots = words[OVERCOOKED_POTS];
    words[OVERCOOKED_POTS] = pots + ((uint32_t)((pots & 255u) - OVERCOOKED_FULL < k->cookTime) | (uint32_t)((pots >> 8 & 255u) - OVERCOOKED_FULL < k->cookTime) << 8);
    words[OVERCOOKED_LAST] = mine | theirs << 3 | chance << 6;
    *done = 0u;
    if ((clock & 0xFFFFu) + 1u == k->horizon) {
        *done = 1u;
        words[OVERCOOKED_GAME]++;
        overcooked_start(k, seed, words);
    } else words[OVERCOOKED_CLOCK] = clock + 1u;
    overcooked_obs(k, m, words, obs);
    return reward;
}

void overcooked_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)extra; (void)steps;
    for (uint32_t i = 0u; i < OVERCOOKED_WORDS; i++) words[i] = 0u;
    overcooked_start(&OVERCOOKED_KNOBS[instance], seed, words);
}

float overcooked_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)extra;
    switch (instance) {
        case 0u: return overcooked_play(0u, seed, ctrl, words, obs, done, shared);
        case 1u: return overcooked_play(1u, seed, ctrl, words, obs, done, shared);
        case 2u: return overcooked_play(2u, seed, ctrl, words, obs, done, shared);
        default: return overcooked_play(3u, seed, ctrl, words, obs, done, shared);
    }
}

void overcooked_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step;
    overcooked_obs(&OVERCOOKED_KNOBS[instance], overcooked_map(instance, shared), words, obs);
}

typedef struct { const overcookedKnobs *k; const uint8_t *closed; } overcookedAvoid;

static int overcooked_avoid(const void *grid, int x, int y) {
    const overcookedAvoid *g = (const overcookedAvoid *)grid;
    return overcooked_blocked(g->k, x, y) || g->closed[(uint32_t)x | (uint32_t)y << 4];
}

static uint32_t overcooked_next(const overcookedMap *m, uint32_t self, uint32_t act) {
    if (act >= OVERCOOKED_STAY) return self & 255u;
    uint32_t next = overcooked_ahead(self & 255u, act);
    return m->tile[next] == OVERCOOKED_FLOOR ? next : self & 255u;
}

static uint32_t overcooked_moves(const overcookedMap *m, uint32_t cell, uint32_t goal) { return m->route[goal][cell] >> 2; }

static uint32_t overcooked_soupy(const overcookedMap *m, uint32_t pots) {
    uint32_t soups = 0u;
    for (uint32_t p = 0u; p < m->pots; p++) soups |= (uint32_t)((pots >> 8u * p & 255u) > OVERCOOKED_FULL) << p;
    return soups;
}

static uint32_t overcooked_pick(const overcookedMap *m, uint32_t cell, uint32_t pots, uint32_t allowed) {
    uint32_t best = OVERCOOKED_GOALS;
    for (uint32_t p = 0u; p < m->pots; p++) {
        if (!(allowed >> p & 1u)) continue;
        if (best == OVERCOOKED_GOALS) { best = p; continue; }
        uint32_t a = pots >> 8u * p & 255u, b = pots >> 8u * best & 255u;
        if (a > b || (a == b && overcooked_moves(m, cell, OVERCOOKED_TO_POT + p) < overcooked_moves(m, cell, OVERCOOKED_TO_POT + best))) best = p;
    }
    return best;
}

enum { OVERCOOKED_TO_COUNTER = OVERCOOKED_GOALS };

static uint32_t overcooked_plan(const overcookedKnobs *k, const overcookedMap *m, const uint32_t *words, uint32_t other) {
    uint32_t self = words[OVERCOOKED_AGENT], pots = words[OVERCOOKED_POTS], cell = self & 255u, held = self >> 10 & 3u;
    uint32_t partnerHeld = other >> 10 & 3u, soups = overcooked_soupy(m, pots), room = 0u, claimed = 0u, bound = OVERCOOKED_GOALS;
    if (partnerHeld == OVERCOOKED_ONION) bound = overcooked_aim(k, m, other, pots) - OVERCOOKED_TO_POT;
    for (uint32_t p = 0u; p < m->pots; p++) {
        uint32_t v = pots >> 8u * p & 255u, left = v < OVERCOOKED_FULL ? OVERCOOKED_FULL - v - (p == bound) : 0u;
        room |= (uint32_t)(v < OVERCOOKED_FULL && left + (p == bound) > (p == bound)) << p;
    }
    if (partnerHeld == OVERCOOKED_DISH || (partnerHeld == OVERCOOKED_EMPTY && soups)) {
        uint32_t aim = overcooked_aim(k, m, (other & ~(3u << 10)) | OVERCOOKED_DISH << 10, pots) - OVERCOOKED_TO_POT;
        claimed = soups >> aim & 1u ? 1u << aim : 0u;
    }
    uint32_t free = soups & ~claimed;
    if (held == OVERCOOKED_SOUP) return OVERCOOKED_TO_WINDOW;
    if (held == OVERCOOKED_DISH) return free ? OVERCOOKED_TO_POT + overcooked_pick(m, cell, pots, free) : OVERCOOKED_TO_COUNTER;
    if (held == OVERCOOKED_ONION) return OVERCOOKED_TO_POT + overcooked_pick(m, cell, pots, room ? room : (1u << m->pots) - 1u);
    return free ? OVERCOOKED_TO_DISHES : OVERCOOKED_TO_ONIONS;
}

static int overcooked_fits(const overcookedKnobs *k, const overcookedMap *m, const uint32_t *words, uint32_t other, uint32_t pot) {
    uint32_t v = words[OVERCOOKED_POTS] >> 8u * pot & 255u;
    uint32_t carried = (other >> 10 & 3u) == OVERCOOKED_ONION && overcooked_aim(k, m, other, words[OVERCOOKED_POTS]) == OVERCOOKED_TO_POT + pot;
    return v + 1u + carried <= OVERCOOKED_FULL;
}

static int overcooked_productive(const overcookedKnobs *k, uint32_t held, uint32_t goal, uint32_t pots) {
    if (goal >= OVERCOOKED_GOALS) return 1;
    if (goal < OVERCOOKED_TO_POT) return goal == OVERCOOKED_TO_WINDOW ? held == OVERCOOKED_SOUP : held == OVERCOOKED_EMPTY;
    uint32_t v = pots >> 8u * (goal - OVERCOOKED_TO_POT) & 255u;
    return held == OVERCOOKED_ONION ? v < OVERCOOKED_FULL : held == OVERCOOKED_DISH && v == overcooked_ready(k);
}

static int overcooked_wanted(const overcookedMap *m, uint32_t goal, const uint32_t *words, uint32_t tile) {
    if (goal < OVERCOOKED_GOALS) return m->goal[tile] == goal;
    uint32_t index = m->tile[tile] >> 3;
    return (m->tile[tile] & 7u) == OVERCOOKED_COUNTER && !(words[OVERCOOKED_COUNTERS + (index >> 4)] >> (index & 15u) * 2u & 3u);
}

static uint32_t overcooked_steer(const overcookedKnobs *k, const overcookedMap *m, const uint32_t *words, uint32_t goal, const uint8_t *closed, uint32_t self, uint32_t *reach) {
    overcookedAvoid avoid = { k, closed };
    uint8_t field[256];
    uint32_t cell = self & 255u, best = OVERCOOKED_UNREACHABLE, target = cell, tile = cell;
    gridDistance(overcooked_avoid, &avoid, (int)(cell & 15u), (int)(cell >> 4), field);
    for (uint32_t c = 0u; c < OVERCOOKED_CELLS; c++) {
        if (m->tile[c] == OVERCOOKED_FLOOR || m->tile[c] == OVERCOOKED_VOID || !overcooked_wanted(m, goal, words, c)) continue;
        for (uint32_t d = 0u; d < 4u; d++) {
            int dx, dy;
            gridDelta(d, &dx, &dy);
            int x = (int)(c & 15u) + dx, y = (int)(c >> 4) + dy;
            if (overcooked_blocked(k, x, y) || field[(x + 1) | (y + 1) << 4] >= best) continue;
            best = field[(x + 1) | (y + 1) << 4];
            target = (uint32_t)x | (uint32_t)y << 4;
            tile = c;
        }
    }
    *reach = best;
    if (best == OVERCOOKED_UNREACHABLE) return OVERCOOKED_STAY;
    if (best == 0u && overcooked_wanted(m, goal, words, overcooked_ahead(cell, self >> 8 & 3u))) return OVERCOOKED_INTERACT;
    if (best == 0u) target = tile;
    while (target != tile && field[((target & 15u) + 1u) | ((target >> 4) + 1u) << 4] > 1u) target = overcooked_ahead(target, (uint32_t)gridDescend(field, (int)(target & 15u), (int)(target >> 4)));
    for (uint32_t d = 0u; d < 4u; d++)
        if (overcooked_ahead(cell, d) == target) return d;
    return OVERCOOKED_STAY;
}

void overcooked_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)extra; (void)obs;
    const overcookedKnobs *k = &OVERCOOKED_KNOBS[instance];
    const overcookedMap *m = overcooked_map(instance, shared);
    uint32_t chance = 0u, self = words[OVERCOOKED_AGENT], other = words[OVERCOOKED_PARTNER], pots = words[OVERCOOKED_POTS];
    uint32_t cell = self & 255u, partnerCell = other & 255u, partnerAct = overcooked_commis(k, m, seed, words, &chance), partnerNext = overcooked_next(m, other, partnerAct);
    uint32_t soon = other;
    if (partnerAct == OVERCOOKED_INTERACT && (other >> 10 & 3u) == OVERCOOKED_EMPTY && m->goal[overcooked_ahead(partnerCell, other >> 8 & 3u)] == OVERCOOKED_TO_ONIONS) soon |= OVERCOOKED_ONION << 10;
    uint32_t goal = overcooked_plan(k, m, words, soon), held = self >> 10 & 3u;
    int productive = overcooked_productive(k, held, goal, pots) && !(held == OVERCOOKED_ONION && !overcooked_fits(k, m, words, soon, goal - OVERCOOKED_TO_POT));
    uint8_t closed[OVERCOOKED_CELLS] = {0}, trail[OVERCOOKED_CELLS] = {0};
    closed[partnerNext] = 1u;
    if (partnerNext == partnerCell || partnerNext == cell) closed[partnerCell] = 1u;
    uint32_t partnerGoal = overcooked_aim(k, m, other, pots), walk = partnerCell;
    trail[walk] = 1u;
    for (uint32_t i = 0u; i < 12u && m->route[partnerGoal][walk] >= 4u && m->route[partnerGoal][walk] != OVERCOOKED_UNREACHABLE; i++) {
        walk = overcooked_ahead(walk, m->route[partnerGoal][walk] & 3u);
        trail[walk] = 1u;
    }
    trail[partnerNext] = 1u;
    if (!productive)
        for (uint32_t c = 0u; c < OVERCOOKED_CELLS; c++) closed[c] |= trail[c] && c != cell;
    uint32_t reach, act = overcooked_steer(k, m, words, goal, closed, self, &reach);
    if (act == OVERCOOKED_INTERACT && !productive) act = OVERCOOKED_STAY;
    if (reach == OVERCOOKED_UNREACHABLE && productive && goal < OVERCOOKED_GOALS && m->route[goal][cell] >= 4u && !trail[overcooked_next(m, self, m->route[goal][cell] & 3u)]) act = m->route[goal][cell] & 3u;
    if (act < OVERCOOKED_STAY && closed[overcooked_next(m, self, act)]) act = OVERCOOKED_STAY;
    if (trail[cell] && !(act == OVERCOOKED_INTERACT && productive) && (act >= OVERCOOKED_STAY || trail[overcooked_next(m, self, act)])) {
        uint32_t best = 1024u;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t next = overcooked_ahead(cell, d);
            if (m->tile[next] != OVERCOOKED_FLOOR || next == partnerCell || next == partnerNext) continue;
            uint32_t there = trail[next] * 256u + (goal < OVERCOOKED_GOALS ? (uint32_t)m->route[goal][next] >> 2 : 0u);
            if (there < best) { best = there; act = d; }
        }
    }
    ctrl[0] = (float)act;
}
