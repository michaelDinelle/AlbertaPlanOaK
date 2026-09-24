#include "../env.h"

typedef struct { uint32_t trials, ticks, randomChemistry; } alchemyKnobs;

static const alchemyKnobs ALCHEMY_KNOBS[2] = {
    { 10u, 20u, 0u },
    { 10u, 20u, 1u }
};

static const float ALCHEMY_VALUE[8] = { -3.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 15.0f };

enum { ALCHEMY_STONES = 3, ALCHEMY_POTIONS = 12, ALCHEMY_CAULDRON = ALCHEMY_POTIONS, ALCHEMY_CONTAINERS = ALCHEMY_POTIONS + 1, ALCHEMY_VERTICES = 8, ALCHEMY_TYPES = 6 };
enum { ALCHEMY_STONE_BYTES, ALCHEMY_POTION_BYTES, ALCHEMY_LOOK_BYTES = ALCHEMY_POTION_BYTES + 3, ALCHEMY_EDGES = ALCHEMY_LOOK_BYTES + 2, ALCHEMY_TINTS, ALCHEMY_CURSOR, ALCHEMY_TICK, ALCHEMY_TRIAL, ALCHEMY_EPISODE };
enum { ALCHEMY_SPENT_STONE = 8, ALCHEMY_SPENT_POTION = 64, ALCHEMY_SKIP = 8 };
enum { ALCHEMY_HIGH_SIDES = 0xF0CCAA, ALCHEMY_ALL_EDGES = ALCHEMY_HIGH_SIDES ^ 0xFFFFFF, ALCHEMY_SINGLES = 12, ALCHEMY_PAIRS = 48, ALCHEMY_TRIPLES = 48 };
enum { ALCHEMY_ROUTES_FOR, ALCHEMY_NEEDS, ALCHEMY_FIRSTS = ALCHEMY_NEEDS + 2 * ALCHEMY_VERTICES * ALCHEMY_VERTICES, ALCHEMY_PLAN_FOR = ALCHEMY_FIRSTS + ALCHEMY_VERTICES * ALCHEMY_VERTICES / 4, ALCHEMY_PLAN = ALCHEMY_PLAN_FOR + 3, ALCHEMY_ORACLE_WORDS };

static const uint64_t ALCHEMY_ENOUGH = 0x0000808080808080ull;

static const uint16_t ALCHEMY_RULES[ALCHEMY_SINGLES + ALCHEMY_PAIRS + ALCHEMY_TRIPLES] = {
    0x001, 0x002, 0x003, 0x004, 0x005, 0x006, 0x007, 0x008, 0x009, 0x00A, 0x00B, 0x00C,
    0x031, 0x041, 0x051, 0x061, 0x091, 0x0A1, 0x0B1, 0x0C1, 0x032, 0x042, 0x052, 0x062, 0x092, 0x0A2, 0x0B2, 0x0C2,
    0x053, 0x063, 0x073, 0x083, 0x0B3, 0x0C3, 0x054, 0x064, 0x074, 0x084, 0x0B4, 0x0C4, 0x075, 0x085, 0x095, 0x0A5,
    0x076, 0x086, 0x096, 0x0A6, 0x097, 0x0A7, 0x0B7, 0x0C7, 0x098, 0x0A8, 0x0B8, 0x0C8, 0x0B9, 0x0C9, 0x0BA, 0x0CA,
    0x531, 0x631, 0xB31, 0xC31, 0x541, 0x641, 0xB41, 0xC41, 0xB91, 0xC91, 0xBA1, 0xCA1, 0x532, 0x632, 0xB32, 0xC32,
    0x542, 0x642, 0xB42, 0xC42, 0xB92, 0xC92, 0xBA2, 0xCA2, 0x753, 0x853, 0x763, 0x863, 0x754, 0x854, 0x764, 0x864,
    0x975, 0xA75, 0x985, 0xA85, 0x976, 0xA76, 0x986, 0xA86, 0xB97, 0xC97, 0xBA7, 0xCA7, 0xB98, 0xC98, 0xBA8, 0xCA8
};

uint32_t alchemy_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return ALCHEMY_ORACLE_WORDS;
}

static uint32_t alchemy_below(uint32_t draw, uint32_t n) { return (uint32_t)(((uint64_t)draw * n) >> 32); }

static uint32_t alchemy_precondition(uint32_t rule) {
    uint32_t axis = rule >> 2, other = (axis + 1u + (rule >> 1 & 1u)) % 3u, side = (ALCHEMY_HIGH_SIDES >> 8u * other & 0xFFu) ^ (rule & 1u ? 0u : 0xFFu);
    return ~(0xFFu << 8u * axis) | side << 8u * axis;
}

static uint32_t alchemy_graph(uint32_t preconditions, uint32_t draw) {
    if (preconditions == 0u) return ALCHEMY_ALL_EDGES;
    uint32_t first = preconditions == 1u ? 0u : preconditions == 2u ? ALCHEMY_SINGLES : ALCHEMY_SINGLES + ALCHEMY_PAIRS;
    uint32_t count = preconditions == 1u ? ALCHEMY_SINGLES : preconditions == 2u ? ALCHEMY_PAIRS : ALCHEMY_TRIPLES, edges = ALCHEMY_ALL_EDGES;
    for (uint32_t rules = ALCHEMY_RULES[first + alchemy_below(draw, count)]; rules; rules >>= 4) edges &= alchemy_precondition((rules & 15u) - 1u);
    return edges;
}

static uint32_t alchemy_look(uint32_t vertex, uint32_t mirror, uint32_t turn) {
    int c[3], f[3];
    for (uint32_t i = 0u; i < 3u; i++) c[i] = f[i] = (vertex ^ mirror) >> i & 1u ? 1 : -1;
    if (turn) {
        uint32_t a = turn % 3u, b = (turn + 1u) % 3u;
        f[a] = (c[a] - c[b]) / 2;
        f[b] = (c[a] + c[b]) / 2;
    }
    return (uint32_t)(f[0] + 1) | (uint32_t)(f[1] + 1) << 2 | (uint32_t)(f[2] + 1) << 4;
}

static void alchemy_chemistry(const alchemyKnobs *k, uint32_t seed, uint32_t *words) {
    u4 r = k->randomChemistry ? philox4x32(seed, ENV_RNG_STREAM, words[ALCHEMY_EPISODE], 0u, 0u, 1u) : (u4){ 0u, 0u, 0u, 0u };
    uint32_t mirror = r.x >> 2 & 7u, turn = r.x >> 5 & 3u, flip = r.x >> 7 & 7u, order = alchemy_below(r.z, 6u), lead = order >> 1, swap = order & 1u;
    uint32_t axisOf[3] = { lead, (lead + 1u + swap) % 3u, (lead + 2u - swap) % 3u }, tints = 0u;
    uint8_t look[ALCHEMY_VERTICES];
    for (uint32_t v = 0u; v < ALCHEMY_VERTICES; v++) look[v] = (uint8_t)alchemy_look(v, mirror, turn);
    for (uint32_t t = 0u; t < ALCHEMY_TYPES; t++) {
        uint32_t axis = axisOf[t >> 1];
        tints |= (2u * axis + ((t & 1u) ^ (flip >> axis & 1u))) << 3u * t;
    }
    __builtin_memcpy(words + ALCHEMY_LOOK_BYTES, look, sizeof look);
    words[ALCHEMY_EDGES] = alchemy_graph(r.x & 3u, r.y);
    words[ALCHEMY_TINTS] = tints;
}

static void alchemy_deal(uint32_t seed, uint32_t *words) {
    u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)words[ALCHEMY_EPISODE], (lane4)words[ALCHEMY_TRIAL], (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u);
    lane4 draws[3] = { r.y, r.z, r.w };
    uint32_t pick = r.x.x, tints = words[ALCHEMY_TINTS];
    uint8_t potion[ALCHEMY_POTIONS];
    for (uint32_t p = 0u; p < ALCHEMY_POTIONS; p++) {
        uint32_t type = alchemy_below(draws[p >> 2][p & 3u], ALCHEMY_TYPES);
        potion[p] = (uint8_t)(type | (tints >> 3u * type & 7u) << 3);
    }
    words[ALCHEMY_STONE_BYTES] = (pick & 7u) | (pick >> 3 & 7u) << 8 | (pick >> 6 & 7u) << 16;
    __builtin_memcpy(words + ALCHEMY_POTION_BYTES, potion, sizeof potion);
}

void alchemy_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    words[ALCHEMY_CURSOR] = 0u;
    words[ALCHEMY_TICK] = 0u;
    words[ALCHEMY_TRIAL] = 0u;
    words[ALCHEMY_EPISODE] = 0u;
    alchemy_chemistry(&ALCHEMY_KNOBS[instance], seed, words);
    alchemy_deal(seed, words);
    extra[ALCHEMY_ROUTES_FOR] = 0u;
    extra[ALCHEMY_PLAN_FOR] = 0u;
}

static void alchemy_obs(const alchemyKnobs *k, const uint32_t *words, float *obs) {
    const uint8_t *stone = (const uint8_t *)(words + ALCHEMY_STONE_BYTES), *look = (const uint8_t *)(words + ALCHEMY_LOOK_BYTES);
    for (uint32_t s = 0u; s < ALCHEMY_STONES; s++) {
        uint32_t held = stone[s], vertex = held & 7u;
        float4 seen = __builtin_convertvector((lane4)look[vertex] >> (lane4){ 0u, 2u, 4u, 6u } & 3u, float4) - 1.0f;
        seen.w = ALCHEMY_VALUE[vertex] / 15.0f;
        seen = select4((lane4)(held & ALCHEMY_SPENT_STONE ? 0u : ~0u), seen, splat4(0.0f));
        __builtin_memcpy(obs + 4u * s, &seen, sizeof seen);
    }
    uint32_t cursor = words[ALCHEMY_CURSOR], potion = cursor < ALCHEMY_POTIONS ? ((const uint8_t *)(words + ALCHEMY_POTION_BYTES))[cursor] : ALCHEMY_SPENT_POTION;
    uint32_t tint = potion >> 3 & 7u;
    float sign = potion & ALCHEMY_SPENT_POTION ? 0.0f : tint & 1u ? -1.0f : 1.0f;
    float4 colour = select4((lane4)((lane4)(tint >> 1) == (lane4){ 0u, 1u, 2u, 3u }), splat4(sign), splat4(0.0f));
    colour.w = (float)cursor / (float)ALCHEMY_CAULDRON;
    __builtin_memcpy(obs + 4u * ALCHEMY_STONES, &colour, sizeof colour);
    obs[4u * ALCHEMY_STONES + 4u] = (float)words[ALCHEMY_TICK] / (float)k->ticks;
    obs[4u * ALCHEMY_STONES + 5u] = (float)words[ALCHEMY_TRIAL] / (float)k->trials;
}

static uint32_t alchemy_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static float alchemy_put(uint32_t *words, uint32_t s, uint32_t cursor) {
    uint8_t *stone = (uint8_t *)(words + ALCHEMY_STONE_BYTES) + s;
    uint32_t held = *stone;
    if (held & ALCHEMY_SPENT_STONE) return 0.0f;
    if (cursor == ALCHEMY_CAULDRON) {
        *stone = (uint8_t)(held | ALCHEMY_SPENT_STONE);
        return ALCHEMY_VALUE[held];
    }
    uint8_t *potion = (uint8_t *)(words + ALCHEMY_POTION_BYTES) + cursor;
    uint32_t dose = *potion, axis = (dose & 7u) >> 1;
    if (dose & ALCHEMY_SPENT_POTION) return 0.0f;
    *potion = (uint8_t)(dose | ALCHEMY_SPENT_POTION);
    uint32_t moves = (held >> axis & 1u) == (dose & 1u) && (words[ALCHEMY_EDGES] >> (8u * axis + (held & ~(1u << axis))) & 1u);
    *stone = (uint8_t)(held ^ moves << axis);
    return 0.0f;
}

static __attribute__((noinline)) void alchemy_turn(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *done) {
    const alchemyKnobs *k = &ALCHEMY_KNOBS[instance];
    uint32_t trial = words[ALCHEMY_TRIAL] + 1u;
    words[ALCHEMY_TICK] = 0u;
    if (trial == k->trials) {
        trial = 0u;
        words[ALCHEMY_EPISODE]++;
        alchemy_chemistry(k, seed, words);
        *done = 1u;
    }
    words[ALCHEMY_TRIAL] = trial;
    alchemy_deal(seed, words);
}

static inline __attribute__((always_inline)) float alchemy_play(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    const alchemyKnobs *k = &ALCHEMY_KNOBS[instance];
    uint32_t action = alchemy_action(ctrl), cursor = words[ALCHEMY_CURSOR];
    float reward = 0.0f;
    *done = 0u;
    if (action == 0u) words[ALCHEMY_CURSOR] = cursor == ALCHEMY_CAULDRON ? 0u : cursor + 1u;
    else if (action == 1u) words[ALCHEMY_CURSOR] = cursor == 0u ? ALCHEMY_CAULDRON : cursor - 1u;
    else {
        if (action < 5u) reward = alchemy_put(words, action - 2u, cursor);
        if (++words[ALCHEMY_TICK] == k->ticks) alchemy_turn(instance, seed, words, done);
    }
    alchemy_obs(k, words, obs);
    return reward;
}

float alchemy_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)extra; (void)shared;
    switch (instance) {
        case 0u: return alchemy_play(0u, seed, ctrl, words, obs, done);
        case 1u: return alchemy_play(1u, seed, ctrl, words, obs, done);
        default: __builtin_trap();
    }
}

void alchemy_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    alchemy_obs(&ALCHEMY_KNOBS[instance], words, obs);
}

static void alchemy_route(uint32_t edges, uint32_t target, uint64_t *need, uint8_t *first) {
    uint32_t reached = 1u << target, layer = reached;
    need[target] = 0u;
    first[target] = 0u;
    while (layer) {
        uint32_t next = 0u;
        for (uint32_t axis = 0u; axis < 3u; axis++) {
            uint32_t lower = edges >> 8u * axis & 0xFFu, shift = 1u << axis;
            next |= (layer & lower) << shift | (layer >> shift & lower);
        }
        next &= ~reached;
        for (uint32_t rest = next; rest; rest &= rest - 1u) {
            uint32_t u = (uint32_t)__builtin_ctz(rest), types = 0u;
            for (uint32_t axis = 0u; axis < 3u; axis++) {
                uint32_t p = u ^ 1u << axis, type = 2u * axis + (u >> axis & 1u);
                if (!(layer >> p & 1u) || !(edges >> (8u * axis + (u & p)) & 1u)) continue;
                types |= 1u << type;
                need[u] = need[p] + (1ull << 8u * type);
            }
            first[u] = (uint8_t)types;
        }
        reached |= next;
        layer = next;
    }
}

static void alchemy_routes(uint32_t edges, uint32_t *extra) {
    uint64_t need[ALCHEMY_VERTICES][ALCHEMY_VERTICES];
    uint8_t first[ALCHEMY_VERTICES][ALCHEMY_VERTICES];
    for (uint32_t target = 0u; target < ALCHEMY_VERTICES; target++) alchemy_route(edges, target, need[target], first[target]);
    __builtin_memcpy(extra + ALCHEMY_NEEDS, need, sizeof need);
    __builtin_memcpy(extra + ALCHEMY_FIRSTS, first, sizeof first);
    extra[ALCHEMY_ROUTES_FOR] = edges | 1u << 24;
}

static uint32_t alchemy_plan(const uint32_t *words, const uint32_t *extra) {
    const uint8_t *stone = (const uint8_t *)(words + ALCHEMY_STONE_BYTES), *potion = (const uint8_t *)(words + ALCHEMY_POTION_BYTES);
    uint64_t have = 0u, cost[ALCHEMY_STONES][5];
    for (uint32_t p = 0u; p < ALCHEMY_POTIONS; p++) have += potion[p] & ALCHEMY_SPENT_POTION ? 0u : 1ull << 8u * (potion[p] & 7u);
    uint32_t target[ALCHEMY_STONES][5], counts[ALCHEMY_STONES];
    float gain[ALCHEMY_STONES][5];
    for (uint32_t s = 0u; s < ALCHEMY_STONES; s++) {
        target[s][0] = ALCHEMY_SKIP; cost[s][0] = 0u; gain[s][0] = 0.0f; counts[s] = 1u;
        for (uint32_t t = 0u; t < ALCHEMY_VERTICES && !(stone[s] & ALCHEMY_SPENT_STONE); t++) {
            uint64_t need;
            __builtin_memcpy(&need, extra + ALCHEMY_NEEDS + 2u * (t * ALCHEMY_VERTICES + (stone[s] & 7u)), sizeof need);
            if (ALCHEMY_VALUE[t] <= 0.0f || ((have | ALCHEMY_ENOUGH) - need & ALCHEMY_ENOUGH) != ALCHEMY_ENOUGH) continue;
            target[s][counts[s]] = t; cost[s][counts[s]] = need; gain[s][counts[s]++] = ALCHEMY_VALUE[t];
        }
    }
    float best = 0.0f;
    uint32_t fewest = 0u, plan = ALCHEMY_SKIP | ALCHEMY_SKIP << 4 | ALCHEMY_SKIP << 8;
    for (uint32_t a = 0u; a < counts[0]; a++)
        for (uint32_t b = 0u; b < counts[1]; b++)
            for (uint32_t c = 0u; c < counts[2]; c++) {
                uint64_t total = cost[0][a] + cost[1][b] + cost[2][c];
                float value = gain[0][a] + gain[1][b] + gain[2][c];
                uint32_t used = (uint32_t)((total * 0x0101010101010101ull) >> 56);
                if (((have | ALCHEMY_ENOUGH) - total & ALCHEMY_ENOUGH) != ALCHEMY_ENOUGH || value < best || (value == best && used >= fewest)) continue;
                best = value;
                fewest = used;
                plan = target[0][a] | target[1][b] << 4 | target[2][c] << 8;
            }
    return plan;
}

void alchemy_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)obs; (void)shared;
    if (extra[ALCHEMY_ROUTES_FOR] != (words[ALCHEMY_EDGES] | 1u << 24)) alchemy_routes(words[ALCHEMY_EDGES], extra);
    if (extra[ALCHEMY_PLAN_FOR] != words[ALCHEMY_EPISODE] + 1u || extra[ALCHEMY_PLAN_FOR + 1u] != words[ALCHEMY_TRIAL] || extra[ALCHEMY_PLAN_FOR + 2u] != words[ALCHEMY_TICK]) {
        extra[ALCHEMY_PLAN] = alchemy_plan(words, extra);
        extra[ALCHEMY_PLAN_FOR] = words[ALCHEMY_EPISODE] + 1u;
        extra[ALCHEMY_PLAN_FOR + 1u] = words[ALCHEMY_TRIAL];
        extra[ALCHEMY_PLAN_FOR + 2u] = words[ALCHEMY_TICK];
    }
    const uint8_t *stone = (const uint8_t *)(words + ALCHEMY_STONE_BYTES), *potion = (const uint8_t *)(words + ALCHEMY_POTION_BYTES), *first = (const uint8_t *)(extra + ALCHEMY_FIRSTS);
    uint32_t plan = extra[ALCHEMY_PLAN], wants[ALCHEMY_STONES], cursor = words[ALCHEMY_CURSOR];
    for (uint32_t s = 0u; s < ALCHEMY_STONES; s++) {
        uint32_t target = plan >> 4u * s & 15u, vertex = stone[s] & 7u;
        wants[s] = target == ALCHEMY_SKIP || stone[s] & ALCHEMY_SPENT_STONE ? 0u : target == vertex ? 1u << ALCHEMY_TYPES : first[target * ALCHEMY_VERTICES + vertex];
    }
    uint32_t any = wants[0] | wants[1] | wants[2];
    ctrl[0] = 5.0f;
    for (uint32_t distance = 0u; any && distance <= ALCHEMY_CONTAINERS / 2u; distance++)
        for (uint32_t back = 0u; back < 2u; back++) {
            uint32_t at = back ? (cursor + ALCHEMY_CONTAINERS - distance) % ALCHEMY_CONTAINERS : (cursor + distance) % ALCHEMY_CONTAINERS;
            uint32_t need = at == ALCHEMY_CAULDRON ? 1u << ALCHEMY_TYPES : potion[at] & ALCHEMY_SPENT_POTION ? 0u : 1u << (potion[at] & 7u);
            if (!(any & need)) continue;
            uint32_t s = wants[0] & need ? 0u : wants[1] & need ? 1u : 2u;
            ctrl[0] = distance == 0u ? (float)(2u + s) : back ? 1.0f : 0.0f;
            return;
        }
}
