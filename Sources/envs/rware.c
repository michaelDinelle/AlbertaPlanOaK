#include "../env.h"

typedef struct { uint32_t shelfRows, shelfColumns, columnHeight, agents, requests, maxSteps; } rwareKnobs;

static const rwareKnobs RWARE_KNOBS[3] = {
    { 1u, 3u, 8u, 2u, 2u, 500u },
    { 1u, 3u, 8u, 4u, 4u, 500u },
    { 2u, 5u, 8u, 4u, 2u, 500u }
};

enum { RWARE_NOOP, RWARE_FORWARD, RWARE_LEFT, RWARE_RIGHT, RWARE_TOGGLE, RWARE_ACTIONS };
enum { RWARE_CLOCK, RWARE_AGENT };
enum { RWARE_STRIDE = 16, RWARE_CELL = 511, RWARE_NONE = 511, RWARE_FACING = 9, RWARE_LOADED = 11, RWARE_AIM = 13, RWARE_STUCK = 22, RWARE_SCORE = 24 };
enum { RWARE_CARRIED = 0x8000, RWARE_HEADER = 64, RWARE_PATIENCE = 3, RWARE_ROOM = 4, RWARE_CROWDED = 8, RWARE_BUSY = 4, RWARE_WANTED = 2 };
enum { RWARE_DEAL, RWARE_REQUEST = 3, RWARE_WANDER = 5 };
enum { RWARE_TICK, RWARE_SELF, RWARE_PRESSED, RWARE_NOTES };
enum { RWARE_MAX_ROWS = 20, RWARE_MAX_CELLS = RWARE_MAX_ROWS * RWARE_STRIDE, RWARE_PITCH = RWARE_STRIDE + 2, RWARE_MARGIN = RWARE_PITCH + 1, RWARE_SLACK = 8, RWARE_VIEW = (RWARE_MAX_ROWS + 2) * RWARE_PITCH + RWARE_SLACK };

static const uint8_t RWARE_FACE[64] = { 0u, 0u, 0u, 0u, 1u, 3u, 3u, 2u, 3u, 1u, 2u, 3u, 1u, 1u, 2u, 2u, 2u, 3u, 1u, 3u, 1u, 3u, 1u, 2u, 2u, 1u, 1u, 3u, 1u, 1u, 1u, 2u,
                                        3u, 2u, 3u, 1u, 1u, 2u, 3u, 1u, 3u, 1u, 2u, 1u, 1u, 1u, 2u, 1u, 2u, 2u, 1u, 1u, 1u, 2u, 1u, 1u, 2u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
static const uint8_t RWARE_WANDERING[4] = { RWARE_FORWARD, RWARE_LEFT, RWARE_RIGHT, RWARE_FORWARD };
static const uint64_t RWARE_PRIME1 = 11400714785074694791ull, RWARE_PRIME2 = 14029467366897019727ull, RWARE_PRIME5 = 2870177450012600261ull;
static const uint64_t RWARE_LANES = 0x0001000100010001ull, RWARE_SIGNS = 0x8000800080008000ull;
static const uint32_t RWARE_SIDESTEPS = 0x01FF12EEu;

static uint32_t rware_width(const rwareKnobs *k) { return 3u * k->shelfColumns + 1u; }
static uint32_t rware_height(const rwareKnobs *k) { return (k->columnHeight + 1u) * k->shelfRows + 2u; }
static uint32_t rware_cells(const rwareKnobs *k) { return rware_height(k) * RWARE_STRIDE; }
static uint32_t rware_shelves(const rwareKnobs *k) { return 2u * k->columnHeight * (k->shelfRows * k->shelfColumns - 1u); }
static uint32_t rware_queue(const rwareKnobs *k) { return RWARE_AGENT + k->agents; }
static uint32_t rware_board(const rwareKnobs *k) { return rware_queue(k) + (k->requests + 1u) / 2u; }
static uint32_t rware_goal(const rwareKnobs *k) { return (rware_height(k) - 1u) * RWARE_STRIDE + rware_width(k) / 2u - 1u; }
static uint32_t rware_floorBytes(const rwareKnobs *k) { return (rware_height(k) + 2u) * RWARE_PITCH + RWARE_SLACK; }
static uint32_t rware_floorWords(const rwareKnobs *k) { return (rware_floorBytes(k) + 3u) / 4u; }
static uint32_t rware_boardWords(const rwareKnobs *k) { return (rware_height(k) + 1u) / 2u; }
static uint64_t rware_block(const rwareKnobs *k) { uint64_t cells = rware_cells(k); return RWARE_HEADER + 2u * cells + 2u * cells * cells; }
static const uint16_t *rware_parking(const uint8_t *chart) { return (const uint16_t *)(chart + RWARE_HEADER); }
static const uint8_t *rware_reach(const rwareKnobs *k, const uint8_t *chart) { return chart + RWARE_HEADER + 2u * rware_cells(k); }
static const uint8_t *rware_steer(const rwareKnobs *k, const uint8_t *chart) { return rware_reach(k, chart) + rware_cells(k) * rware_cells(k); }
static const uint8_t *rware_chartOf(uint32_t instance, const float *shared) { return (const uint8_t *)shared + (instance == 2u ? rware_block(&RWARE_KNOBS[0]) : 0u); }

static uint32_t rware_home(const rwareKnobs *k, uint32_t y) {
    uint32_t band = k->columnHeight + 1u, columns = 6u * (((1u << 3u * k->shelfColumns) - 1u) / 7u), middle = 3u << (rware_width(k) / 2u - 1u);
    if (y % band == 0u || y + 2u >= rware_height(k)) return 0u;
    return y > (k->shelfRows - 1u) * band ? columns & ~middle : columns;
}

static uint32_t rware_homeCell(const rwareKnobs *k, uint32_t cell) { return rware_home(k, cell >> 4) >> (cell & 15u) & 1u; }
static uint32_t rware_row(const uint32_t *board, uint32_t y) { return board[y >> 1] >> (y & 1u) * 16u & 0xFFFFu; }
static uint32_t rware_standing(const uint32_t *board, uint32_t cell) { return board[cell >> 5] >> (cell & 31u) & 1u; }
static uint32_t rware_entry(const uint32_t *queue, uint32_t slot) { return queue[slot >> 1] >> (slot & 1u) * 16u & 0xFFFFu; }
static uint32_t rware_below(uint32_t word, uint32_t n) { return (uint32_t)(((uint64_t)word * n) >> 32); }
static uint32_t rware_nth(uint32_t bits, uint32_t n) { while (n--) bits &= bits - 1u; return (uint32_t)__builtin_ctz(bits); }
static uint32_t rware_apart(uint32_t a, uint32_t b) { return (a & 15u) > (b & 15u) ? (a & 15u) - (b & 15u) : (b & 15u) - (a & 15u); }
static uint32_t rware_span(uint32_t a, uint32_t b) { return rware_apart(a, b) + (a >> 4 > b >> 4 ? (a >> 4) - (b >> 4) : (b >> 4) - (a >> 4)); }
static uint32_t rware_spot(uint32_t cell) { return cell + 2u * (cell >> 4) + RWARE_MARGIN; }
static uint32_t rware_sidestep(uint32_t facing) { return (uint32_t)(int32_t)(int8_t)(RWARE_SIDESTEPS >> 8u * facing); }
static uint32_t rware_busy(const uint8_t *floor, uint32_t cell) { return floor[rware_spot(cell)] >= RWARE_BUSY; }

static uint32_t rware_among(uint64_t lanes, uint32_t value) {
    uint64_t x = lanes ^ value * RWARE_LANES;
    return ((x - RWARE_LANES) & ~x & RWARE_SIGNS) != 0u;
}

static uint64_t rware_wanted(const rwareKnobs *k, const uint32_t *words) {
    const uint32_t *queue = words + rware_queue(k);
    return k->requests > 2u ? (uint64_t)queue[0] | (uint64_t)queue[1] << 32 : (uint64_t)queue[0] | 0xFFFFFFFF00000000ull;
}

static void rware_assign(uint32_t *queue, uint32_t slot, uint32_t value) {
    uint32_t shift = (slot & 1u) * 16u;
    queue[slot >> 1] = (queue[slot >> 1] & ~(0xFFFFu << shift)) | value << shift;
}

static void rware_relabel(const rwareKnobs *k, uint32_t *words, uint32_t from, uint32_t to) {
    uint32_t *queue = words + rware_queue(k);
    for (uint32_t s = 0u; s < k->requests; s++)
        if (rware_entry(queue, s) == from) rware_assign(queue, s, to);
}

static uint32_t rware_locate(const uint32_t *rows, uint32_t height, uint32_t n) {
    for (uint32_t y = 0u; y < height; y++) {
        uint32_t count = (uint32_t)__builtin_popcount(rows[y]);
        if (n < count) return y * RWARE_STRIDE + rware_nth(rows[y], n);
        n -= count;
    }
    return RWARE_NONE;
}

static uint32_t rware_skip(uint32_t *taken, uint32_t count, uint32_t pick) {
    for (uint32_t j = 0u; j < count; j++) pick += taken[j] <= pick;
    uint32_t at = count;
    while (at > 0u && taken[at - 1u] > pick) { taken[at] = taken[at - 1u]; at--; }
    taken[at] = pick;
    return pick;
}

static uint32_t rware_located(const uint32_t *words, uint32_t e) { return e & RWARE_CARRIED ? words[RWARE_AGENT + (e & 3u)] & RWARE_CELL : e; }

static __attribute__((noinline)) void rware_deal(const rwareKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint8_t *floor) {
    uint32_t draw[12], taken[RWARE_ROOM], chosen[RWARE_ROOM], rows[RWARE_MAX_ROWS], w = rware_width(k), h = rware_height(k), *board = words + rware_board(k);
    for (uint32_t b = 0u; b < 3u; b++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, RWARE_DEAL + b, 0u, 0u);
        draw[4u * b] = r.x; draw[4u * b + 1u] = r.y; draw[4u * b + 2u] = r.z; draw[4u * b + 3u] = r.w;
    }
    for (uint32_t i = 0u; i < rware_floorBytes(k); i++) floor[i] = 0u;
    for (uint32_t y = 0u; y < h; y++) {
        rows[y] = rware_home(k, y);
        for (uint32_t open = rows[y]; open; open &= open - 1u) floor[rware_spot(y * RWARE_STRIDE + (uint32_t)__builtin_ctz(open))] = 1u;
    }
    for (uint32_t y = 0u; y < h; y += 2u) board[y >> 1] = rows[y] | (y + 1u < h ? rows[y + 1u] << 16 : 0u);
    for (uint32_t i = 0u; i < k->agents; i++) {
        uint32_t spot = rware_skip(taken, i, rware_below(draw[i], w * h - i)), cell = (spot / w) * RWARE_STRIDE + spot % w, facing = draw[4u + i] & 3u;
        words[RWARE_AGENT + i] = cell | facing << RWARE_FACING | (uint32_t)RWARE_NONE << RWARE_AIM;
        floor[rware_spot(cell)] += (uint8_t)(RWARE_BUSY + RWARE_BUSY * facing);
    }
    for (uint32_t i = 0u; i < k->requests; i++) {
        uint32_t cell = rware_locate(rows, h, rware_skip(chosen, i, rware_below(draw[8u + i], rware_shelves(k) - i)));
        rware_assign(words + rware_queue(k), i, cell);
        floor[rware_spot(cell)] += RWARE_WANTED;
    }
    words[RWARE_CLOCK] = 0u;
}

static void rware_request(const rwareKnobs *k, uint32_t *words, uint8_t *floor, uint32_t slot, uint32_t draw) {
    uint32_t *queue = words + rware_queue(k), rows[RWARE_MAX_ROWS], h = rware_height(k), requested = 0u;
    uint32_t pick = rware_below(draw, rware_shelves(k) - k->requests);
    for (uint32_t y = 0u; y < h; y++) rows[y] = rware_row(words + rware_board(k), y);
    for (uint32_t s = 0u; s < k->requests; s++) {
        uint32_t e = rware_entry(queue, s);
        if (e & RWARE_CARRIED) requested |= 1u << (e & 3u);
        else rows[e >> 4] &= ~(1u << (e & 15u));
    }
    uint32_t chosen = rware_locate(rows, h, pick);
    if (chosen == RWARE_NONE) {
        for (uint32_t y = 0u; y < h; y++) pick -= (uint32_t)__builtin_popcount(rows[y]);
        for (uint32_t i = 0u; i < k->agents && chosen == RWARE_NONE; i++) {
            if (!(words[RWARE_AGENT + i] >> RWARE_LOADED & 1u) || (requested >> i & 1u)) continue;
            if (pick == 0u) chosen = RWARE_CARRIED | i;
            else pick--;
        }
        if (chosen == RWARE_NONE) __builtin_trap();
    }
    floor[rware_spot(rware_located(words, rware_entry(queue, slot)))] -= RWARE_WANTED;
    floor[rware_spot(rware_located(words, chosen))] += RWARE_WANTED;
    rware_assign(queue, slot, chosen);
}

static __attribute__((noinline)) float rware_handover(const rwareKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint8_t *floor) {
    float reward = 0.0f;
    for (uint32_t side = 0u; side < 2u; side++)
        for (uint32_t i = 0u; i < k->agents; i++) {
            uint32_t a = words[RWARE_AGENT + i], *queue = words + rware_queue(k);
            if ((a & RWARE_CELL) != rware_goal(k) + side || !(a >> RWARE_LOADED & 1u)) continue;
            for (uint32_t s = 0u; s < k->requests; s++) {
                if (rware_entry(queue, s) != (RWARE_CARRIED | i)) continue;
                rware_request(k, words, floor, s, philox4x32(seed, ENV_RNG_STREAM, step, RWARE_REQUEST + side, 0u, 0u).x);
                words[RWARE_AGENT + i] = a + (1u << RWARE_SCORE);
                reward += i == 0u ? 1.0f : 0.0f;
                break;
            }
        }
    return reward;
}

static uint64_t rware_hash(uint32_t cell) {
    uint64_t acc = RWARE_PRIME5;
    acc += (uint64_t)(cell & 15u) * RWARE_PRIME2;
    acc = (acc << 31 | acc >> 33) * RWARE_PRIME1;
    acc += (uint64_t)(cell >> 4) * RWARE_PRIME2;
    acc = (acc << 31 | acc >> 33) * RWARE_PRIME1;
    acc += 2u ^ (RWARE_PRIME5 ^ 3527539u);
    return acc == ~0ull ? 1546275796u : acc;
}

static void rware_scatter(const uint32_t *cells, uint32_t count, uint32_t *order) {
    uint32_t slot[8], used = 0u, n = 0u;
    for (uint32_t i = 0u; i < count; i++) {
        uint64_t hash = rware_hash(cells[i]), perturb = hash;
        uint32_t at = (uint32_t)hash & 7u;
        while (used >> at & 1u) {
            perturb >>= 5;
            at = (at * 5u + 1u + (uint32_t)perturb) & 7u;
        }
        used |= 1u << at;
        slot[at] = cells[i];
    }
    for (uint32_t at = 0u; at < 8u; at++)
        if (used >> at & 1u) order[n++] = slot[at];
}

typedef struct { uint32_t node[8], owner[8], label[8], src[RWARE_ROOM], dst[RWARE_ROOM], nodes, agents; } rwareGraph;

static void rware_scramble(const rwareGraph *g, uint32_t root, uint32_t *rank) {
    uint32_t visit[8], cells[8], once[8], twice[8], count = 1u, seen = 1u << root;
    visit[0] = root;
    for (uint32_t head = 0u; head < count; head++) {
        uint32_t v = visit[head], next = g->owner[v] < RWARE_ROOM ? g->dst[g->owner[v]] : v;
        if (!(seen >> next & 1u)) { seen |= 1u << next; visit[count++] = next; }
        for (uint32_t i = 0u; i < g->agents; i++)
            if (g->dst[i] == v && !(seen >> g->src[i] & 1u)) { seen |= 1u << g->src[i]; visit[count++] = g->src[i]; }
    }
    for (uint32_t i = 0u; i < count; i++) cells[i] = g->node[visit[i]];
    rware_scatter(cells, count, once);
    rware_scatter(once, count, twice);
    for (uint32_t i = 0u; i < count; i++)
        for (uint32_t j = 0u; j < g->nodes; j++)
            if (g->node[j] == twice[i]) rank[j] = i;
}

static __attribute__((noinline)) void rware_settle(uint32_t agents, const uint32_t *from, uint32_t *to) {
    rwareGraph g = { .nodes = 0u, .agents = agents };
    uint32_t go = 0u;
    for (uint32_t i = 0u; i < agents; i++)
        for (uint32_t end = 0u; end < 2u; end++) {
            uint32_t cell = end ? to[i] : from[i], j = 0u;
            while (j < g.nodes && g.node[j] != cell) j++;
            if (j == g.nodes) { g.node[j] = cell; g.owner[j] = RWARE_ROOM; g.label[j] = j; g.nodes++; }
            if (end) g.dst[i] = j;
            else { g.src[i] = j; g.owner[j] = i; }
        }
    for (uint32_t i = 0u; i < agents; i++) {
        uint32_t a = g.label[g.src[i]], b = g.label[g.dst[i]], lo = a < b ? a : b, hi = a ^ b ^ lo;
        for (uint32_t j = 0u; j < g.nodes; j++) g.label[j] = g.label[j] == hi ? lo : g.label[j];
    }
    for (uint32_t root = 0u; root < g.nodes; root++) {
        if (g.label[root] != root) continue;
        uint32_t size = 0u, sink = 8u, height[8], rank[8];
        for (uint32_t j = 0u; j < g.nodes; j++) {
            height[j] = 0u;
            rank[j] = j;
            if (g.label[j] != root) continue;
            size++;
            if (g.owner[j] == RWARE_ROOM) sink = j;
        }
        if (sink == 8u) {
            uint32_t at = root, length = 0u, crew = 0u;
            for (uint32_t r = 0u; r < g.nodes; r++) at = g.dst[g.owner[at]];
            uint32_t walk = at;
            do {
                crew |= 1u << g.owner[walk];
                walk = g.dst[g.owner[walk]];
                length++;
            } while (walk != at);
            go |= length == 2u ? 0u : crew;
            continue;
        }
        for (uint32_t pass = 0u; pass < agents; pass++)
            for (uint32_t i = 0u; i < agents; i++)
                if (g.label[g.src[i]] == root && height[g.dst[i]] < height[g.src[i]] + 1u) height[g.dst[i]] = height[g.src[i]] + 1u;
        if (2u * size < g.nodes) rware_scramble(&g, root, rank);
        for (uint32_t at = sink;;) {
            uint32_t best = RWARE_ROOM;
            for (uint32_t i = 0u; i < agents; i++)
                if (g.dst[i] == at && height[g.src[i]] + 1u == height[at] && (best == RWARE_ROOM || rank[g.src[i]] < rank[g.src[best]])) best = i;
            if (best == RWARE_ROOM) break;
            go |= 1u << best;
            at = g.src[best];
        }
    }
    for (uint32_t i = 0u; i < agents; i++)
        if (!(go >> i & 1u)) to[i] = from[i];
}

static __attribute__((noinline)) void rware_follow(uint32_t agents, const uint32_t *from, uint32_t *to) {
    uint32_t stay = 0u;
    for (uint32_t i = 0u; i < agents; i++) {
        if (to[i] == from[i]) continue;
        uint32_t at = i, length = 0u, blocked = 1u;
        for (uint32_t hop = 0u; hop <= agents; hop++) {
            uint32_t next = RWARE_ROOM;
            for (uint32_t j = 0u; j < agents; j++) next = from[j] == to[at] ? j : next;
            length++;
            if (next == RWARE_ROOM) { blocked = 0u; break; }
            if (next == i) { blocked = length == 2u; break; }
            if (to[next] == from[next]) break;
            at = next;
        }
        stay |= blocked << i;
    }
    for (uint32_t i = 0u; i < agents; i++)
        if (stay >> i & 1u) to[i] = from[i];
}

static inline __attribute__((always_inline)) void rware_resolve(uint32_t agents, const uint32_t *from, uint32_t *to) {
    uint32x4_t f = vld1q_u32(from), t = vld1q_u32(to), moving = vmvnq_u32(vceqq_u32(t, f));
    uint32x4_t hit = vorrq_u32(vorrq_u32(vceqq_u32(t, vextq_u32(f, f, 1)), vceqq_u32(t, vextq_u32(f, f, 2))), vceqq_u32(t, vextq_u32(f, f, 3)));
    uint32x4_t tie = vorrq_u32(vorrq_u32(vandq_u32(vceqq_u32(t, vextq_u32(t, t, 1)), vextq_u32(moving, moving, 1)), vandq_u32(vceqq_u32(t, vextq_u32(t, t, 2)), vextq_u32(moving, moving, 2))),
                               vandq_u32(vceqq_u32(t, vextq_u32(t, t, 3)), vextq_u32(moving, moving, 3)));
    tie = vandq_u32(tie, moving);
    if (vmaxvq_u32(tie)) rware_settle(agents, from, to);
    else if (vmaxvq_u32(hit)) rware_follow(agents, from, to);
}

static uint32_t rware_toggle(const rwareKnobs *k, uint32_t *words, uint32_t i, uint32_t cell, uint32_t a) {
    uint32_t *board = words + rware_board(k);
    if (!(a >> RWARE_LOADED & 1u)) {
        if (!rware_standing(board, cell)) return a;
        board[cell >> 5] &= ~(1u << (cell & 31u));
        rware_relabel(k, words, cell, RWARE_CARRIED | i);
        return a | 1u << RWARE_LOADED;
    }
    if (!rware_homeCell(k, cell)) return a;
    board[cell >> 5] |= 1u << (cell & 31u);
    rware_relabel(k, words, RWARE_CARRIED | i, cell);
    return a & ~(1u << RWARE_LOADED);
}

static inline __attribute__((always_inline)) uint32x4_t rware_lanes(const rwareKnobs *k, const uint32_t *words) {
    return k->agents == 4u ? vld1q_u32(words + RWARE_AGENT) : vcombine_u32(vld1_u32(words + RWARE_AGENT), vdup_n_u32(0u));
}

static inline __attribute__((always_inline)) void rware_move(const rwareKnobs *k, uint32_t *words, uint8_t *floor, const uint32_t *act) {
    const uint32x4_t lane = { 0u, 1u, 2u, 3u }, one = vdupq_n_u32(1u);
    uint32x4_t a = rware_lanes(k, words), cell = vandq_u32(a, vdupq_n_u32(RWARE_CELL)), facing = vandq_u32(vshrq_n_u32(a, RWARE_FACING), vdupq_n_u32(3u));
    uint32x4_t x = vandq_u32(cell, vdupq_n_u32(15u)), y = vshrq_n_u32(cell, 4), valid = vcltq_u32(lane, vdupq_n_u32(k->agents)), order = vld1q_u32(act);
    uint32x4_t up = vceqq_u32(facing, vdupq_n_u32(0u)), down = vceqq_u32(facing, one), left = vceqq_u32(facing, vdupq_n_u32(2u)), right = vceqq_u32(facing, vdupq_n_u32(3u));
    uint32x4_t inside = vorrq_u32(vorrq_u32(vandq_u32(up, vcgtq_u32(y, vdupq_n_u32(0u))), vandq_u32(down, vcltq_u32(vaddq_u32(y, one), vdupq_n_u32(rware_height(k))))),
                                  vorrq_u32(vandq_u32(left, vcgtq_u32(x, vdupq_n_u32(0u))), vandq_u32(right, vcltq_u32(vaddq_u32(x, one), vdupq_n_u32(rware_width(k))))));
    int32x4_t step = vbslq_s32(up, vdupq_n_s32(-RWARE_STRIDE), vbslq_s32(down, vdupq_n_s32(RWARE_STRIDE), vbslq_s32(left, vdupq_n_s32(-1), vdupq_n_s32(1))));
    uint32x4_t next = vaddq_u32(cell, vreinterpretq_u32_s32(step)), probe = vbslq_u32(inside, next, cell);
    const uint32_t *board = words + rware_board(k);
    uint32_t spots[RWARE_ROOM], stop[RWARE_ROOM] = { 0u }, from[RWARE_ROOM], to[RWARE_ROOM], carry[RWARE_ROOM], turn[RWARE_ROOM];
    vst1q_u32(spots, probe);
    for (uint32_t i = 0u; i < k->agents; i++) stop[i] = 0u - (words[RWARE_AGENT + i] >> RWARE_LOADED & rware_standing(board, spots[i]));
    uint32x4_t go = vandq_u32(vandq_u32(vceqq_u32(order, one), inside), vbicq_u32(valid, vld1q_u32(stop)));
    vst1q_u32(from, vbslq_u32(valid, cell, vaddq_u32(lane, vdupq_n_u32(0x1000u))));
    vst1q_u32(to, vbslq_u32(valid, vbslq_u32(go, next, cell), vaddq_u32(lane, vdupq_n_u32(0x2000u))));
    rware_resolve(k->agents, from, to);
    uint32x4_t land = vld1q_u32(to), moved = vmvnq_u32(vceqq_u32(land, cell));
    uint32x4_t turning = vandq_u32(vcgeq_u32(order, vdupq_n_u32(RWARE_LEFT)), vcleq_u32(order, vdupq_n_u32(RWARE_RIGHT)));
    uint32x4_t turned = vbslq_u32(turning, veorq_u32(veorq_u32(facing, vshrq_n_u32(facing, 1)), vorrq_u32(vdupq_n_u32(2u), vandq_u32(vceqq_u32(order, vdupq_n_u32(RWARE_RIGHT)), one))), facing);
    uint32x4_t stuck = vandq_u32(vshrq_n_u32(a, RWARE_STUCK), vdupq_n_u32(3u));
    stuck = vbicq_u32(vsubq_u32(stuck, vandq_u32(vceqq_u32(order, one), vcltq_u32(stuck, vdupq_n_u32(RWARE_PATIENCE)))), moved);
    stuck = vandq_u32(stuck, vcgtq_u32(lane, vdupq_n_u32(0u)));
    vst1q_u32(turn, vshlq_n_u32(vsubq_u32(turned, facing), 2));
    for (uint32_t i = 0u; i < k->agents; i++) {
        uint32_t w = words[RWARE_AGENT + i], at = rware_spot(from[i]);
        carry[i] = RWARE_BUSY + RWARE_BUSY * (w >> RWARE_FACING & 3u) + (w >> RWARE_LOADED & 1u ? floor[at] & 3u : 0u);
        floor[at] -= (uint8_t)carry[i];
    }
    for (uint32_t i = 0u; i < k->agents; i++) floor[rware_spot(to[i])] += (uint8_t)(carry[i] + turn[i]);
    uint32x4_t keep = vdupq_n_u32(~((uint32_t)RWARE_CELL | 3u << RWARE_FACING | 3u << RWARE_STUCK));
    a = vorrq_u32(vorrq_u32(vandq_u32(a, keep), land), vorrq_u32(vshlq_n_u32(turned, RWARE_FACING), vshlq_n_u32(stuck, RWARE_STUCK)));
    if (k->agents == 4u) vst1q_u32(words + RWARE_AGENT, a);
    else vst1_u32(words + RWARE_AGENT, vget_low_u32(a));
    if (vmaxvq_u32(vandq_u32(vceqq_u32(order, vdupq_n_u32(RWARE_TOGGLE)), valid)))
        for (uint32_t i = 0u; i < k->agents; i++)
            if (act[i] == RWARE_TOGGLE) words[RWARE_AGENT + i] = rware_toggle(k, words, i, words[RWARE_AGENT + i] & RWARE_CELL, words[RWARE_AGENT + i]);
}

static inline __attribute__((always_inline)) float rware_deliver(const rwareKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint8_t *floor) {
    uint32_t goal = rware_spot(rware_goal(k));
    return (floor[goal] | floor[goal + 1u]) & RWARE_WANTED ? rware_handover(k, seed, step, words, floor) : 0.0f;
}

static uint32_t rware_heading(uint32_t cell, uint32_t target) {
    uint32_t x = cell & 15u, y = cell >> 4, tx = target & 15u, ty = target >> 4;
    return (uint32_t)(ty < y) | (uint32_t)(ty > y) << 1 | (uint32_t)(tx < x) << 2 | (uint32_t)(tx > x) << 3;
}

static uint32_t rware_clear(uint32_t good, uint32_t cell, uint32_t facing, const uint8_t *floor) {
    uint32_t ahead = 1u << facing;
    return (good & ahead) && (good & ~ahead) && floor[rware_spot(cell) + rware_sidestep(facing)] >= RWARE_BUSY ? good & ~ahead : good;
}

static uint32_t rware_effort(uint32_t cell, uint32_t target, uint32_t facing) {
    uint32_t good = rware_heading(cell, target);
    return rware_span(cell, target) + (good && !(good >> facing & 1u)) + (good && !(good & ~(1u << (facing ^ 1u))));
}

static uint32_t rware_taken(const uint32_t *words, uint32_t spot, uint32_t self, uint32_t upto) {
    uint32_t taken = 0u;
    for (uint32_t m = 1u; m < upto; m++) {
        uint32_t a = words[RWARE_AGENT + m];
        taken |= m != self && !(a >> RWARE_LOADED & 1u) && (a >> RWARE_AIM & RWARE_NONE) == spot;
    }
    return taken;
}

static __attribute__((noinline)) uint32_t rware_claim(const rwareKnobs *k, uint32_t *words, uint32_t j) {
    uint32_t a = words[RWARE_AGENT + j], cell = a & RWARE_CELL, facing = a >> RWARE_FACING & 3u, best = RWARE_NONE, far = ~0u;
    for (uint32_t s = 0u; s < k->requests; s++) {
        uint32_t spot = rware_entry(words + rware_queue(k), s);
        if (spot & RWARE_CARRIED || rware_taken(words, spot, j, k->agents)) continue;
        uint32_t effort = rware_effort(cell, spot, facing);
        if (effort < far) { far = effort; best = spot; }
    }
    words[RWARE_AGENT + j] = (a & ~((uint32_t)RWARE_NONE << RWARE_AIM)) | best << RWARE_AIM;
    return best;
}

static __attribute__((noinline)) uint32_t rware_vacancy(const rwareKnobs *k, const uint8_t *chart, const uint32_t *board, const uint8_t *floor, uint32_t cell) {
    const uint8_t *reach = rware_reach(k, chart);
    uint32_t best = RWARE_NONE, far = ~0u;
    for (uint32_t y = 0u; y < rware_height(k); y++)
        for (uint32_t open = rware_home(k, y) & ~rware_row(board, y); open; open &= open - 1u) {
            uint32_t spot = y * RWARE_STRIDE + (uint32_t)__builtin_ctz(open);
            uint32_t cost = reach[(size_t)spot * rware_cells(k) + cell] + RWARE_CROWDED * rware_busy(floor, spot);
            if (cost < far) { far = cost; best = spot; }
        }
    return best;
}

static uint32_t rware_dock(const rwareKnobs *k, const uint8_t *chart, const uint8_t *floor, uint32_t cell) {
    const uint8_t *reach = rware_reach(k, chart);
    uint32_t left = rware_goal(k), right = left + 1u;
    if (cell - left < 2u) return cell;
    uint32_t near = reach[(size_t)left * rware_cells(k) + cell] + RWARE_CROWDED * rware_busy(floor, left);
    uint32_t far = reach[(size_t)right * rware_cells(k) + cell] + RWARE_CROWDED * rware_busy(floor, right);
    return far < near ? right : left;
}

static inline __attribute__((always_inline)) uint32_t rware_robot(const rwareKnobs *k, const uint8_t *chart, uint32_t *words, const uint8_t *floor, uint32_t j, uint64_t wanted, uint32_t wander) {
    uint32_t a = words[RWARE_AGENT + j], cell = a & RWARE_CELL, facing = a >> RWARE_FACING & 3u, aim = a >> RWARE_AIM & RWARE_NONE, good;
    if ((a >> RWARE_STUCK & 3u) >= RWARE_PATIENCE) return RWARE_WANDERING[wander & 3u];
    if (a >> RWARE_LOADED & 1u) {
        uint32_t target = aim;
        if (rware_among(wanted, RWARE_CARRIED | j)) target = rware_dock(k, chart, floor, cell);
        else {
            if (rware_standing(words + rware_board(k), aim)) {
                target = rware_vacancy(k, chart, words + rware_board(k), floor, cell);
                words[RWARE_AGENT + j] = (a & ~((uint32_t)RWARE_NONE << RWARE_AIM)) | target << RWARE_AIM;
            }
            if (cell == target) return RWARE_TOGGLE;
        }
        good = rware_steer(k, chart)[(size_t)target * rware_cells(k) + cell];
    } else {
        if (!rware_among(wanted, aim) || rware_taken(words, aim, j, j)) aim = rware_claim(k, words, j);
        if (aim == cell) return RWARE_TOGGLE;
        good = rware_heading(cell, aim == RWARE_NONE ? rware_parking(chart)[cell] : aim);
    }
    return RWARE_FACE[rware_clear(good, cell, facing, floor) << 2 | facing];
}

static uint32_t rware_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)RWARE_ACTIONS ? (uint32_t)c : RWARE_NOOP;
}

static uint32_t rware_load32(const uint8_t *at) {
    uint32_t v;
    __builtin_memcpy(&v, at, sizeof v);
    return v;
}

static inline __attribute__((always_inline)) void rware_obs(const rwareKnobs *k, const uint32_t *words, const uint8_t *floor, float *obs) {
    uint32_t a = words[RWARE_AGENT], cell = a & RWARE_CELL;
    const uint8_t *corner = floor + rware_spot(cell) - RWARE_MARGIN;
    uint32_t top = rware_load32(corner), middle = rware_load32(corner + RWARE_PITCH), bottom = rware_load32(corner + 2u * RWARE_PITCH);
    uint16x8_t codes = vmovl_u8(vcreate_u8((uint64_t)(top & 0xFFFFFFu) | (uint64_t)(middle & 0xFFFFFFu) << 24 | (uint64_t)bottom << 48));
    obs[0] = (float)(cell & 15u);
    obs[1] = (float)(cell >> 4);
    obs[2] = (float)(a >> RWARE_LOADED & 1u);
    obs[3] = (float)(a >> RWARE_FACING & 3u);
    obs[4] = (float)(1u - rware_homeCell(k, cell));
    vst1q_f32(obs + 5, vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes))));
    vst1q_f32(obs + 9, vcvtq_f32_u32(vmovl_u16(vget_high_u16(codes))));
    obs[13] = (float)(bottom >> 16 & 0xFFu);
    obs[14] = 1.0f;
}

static inline __attribute__((always_inline)) float rware_play(const rwareKnobs *k, const uint8_t *chart, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint8_t *floor, float *obs, uint32_t *done) {
    uint32_t act[RWARE_ROOM] = { rware_action(ctrl), RWARE_NOOP, RWARE_NOOP, RWARE_NOOP }, restless = 0u, wander = 0u;
    uint64_t wanted = rware_wanted(k, words);
    for (uint32_t j = 1u; j < k->agents; j++) restless |= (words[RWARE_AGENT + j] >> RWARE_STUCK & 3u) >= RWARE_PATIENCE;
    if (restless) wander = philox4x32(seed, ENV_RNG_STREAM, step, RWARE_WANDER, 0u, 0u).x;
    for (uint32_t j = 1u; j < k->agents; j++) act[j] = rware_robot(k, chart, words, floor, j, wanted, wander >> 2u * j);
    rware_move(k, words, floor, act);
    float reward = rware_deliver(k, seed, step, words, floor);
    uint32_t clock = words[RWARE_CLOCK] + 1u;
    *done = clock >= k->maxSteps;
    if (*done) rware_deal(k, seed, step, words, floor);
    else words[RWARE_CLOCK] = clock;
    rware_obs(k, words, floor, obs);
    return reward;
}

static uint32_t rware_narrow(const rwareKnobs *k, uint32_t cell) {
    uint32_t x = cell & 15u;
    return x % 3u == 0u && !rware_homeCell(k, cell) && (x == 0u || rware_homeCell(k, cell - 1u)) && (x + 1u == rware_width(k) || rware_homeCell(k, cell + 1u));
}

static uint32_t rware_lawful(const rwareKnobs *k, uint32_t cell, uint32_t d) {
    uint32_t x = cell & 15u, y = cell >> 4, band = k->columnHeight + 1u;
    if (d < 2u) {
        uint32_t next = d ? cell + RWARE_STRIDE : cell - RWARE_STRIDE;
        return d == ((x / 3u) & 1u) || !(rware_narrow(k, cell) || (next < rware_cells(k) && rware_narrow(k, next)));
    }
    return y % band != 0u || y + 2u >= rware_height(k) || d == 3u - ((y / band) & 1u);
}

static void rware_chart(const rwareKnobs *k, uint8_t *chart) {
    uint32_t w = rware_width(k), h = rware_height(k), cells = rware_cells(k), centre = RWARE_NONE, least = ~0u;
    uint16_t *parking = (uint16_t *)(chart + RWARE_HEADER), queue[RWARE_MAX_CELLS];
    uint8_t *reach = chart + RWARE_HEADER + 2u * cells, *steer = reach + (size_t)cells * cells;
    for (uint32_t target = 0u; target < cells; target++) {
        uint8_t *row = reach + (size_t)target * cells, *moves = steer + (size_t)target * cells;
        for (uint32_t cell = 0u; cell < cells; cell++) { row[cell] = 0xFFu; moves[cell] = 0u; }
        if ((target & 15u) >= w) continue;
        uint32_t head = 0u, tail = 0u;
        row[target] = 0u;
        queue[tail++] = (uint16_t)target;
        while (head < tail) {
            uint32_t cell = queue[head++];
            if (cell != target && rware_homeCell(k, cell)) continue;
            for (uint32_t d = 0u; d < 4u; d++) {
                int dx, dy;
                gridDelta(d, &dx, &dy);
                uint32_t x = (cell & 15u) - (uint32_t)dx, y = (cell >> 4) - (uint32_t)dy, prior = y * RWARE_STRIDE + x;
                if (x >= w || y >= h || row[prior] != 0xFFu || !rware_lawful(k, prior, d)) continue;
                row[prior] = (uint8_t)(row[cell] + 1u);
                queue[tail++] = (uint16_t)prior;
            }
        }
        for (uint32_t cell = 0u; cell < cells; cell++) {
            if ((cell & 15u) >= w || cell == target) continue;
            for (uint32_t d = 0u; d < 4u; d++) {
                int dx, dy;
                gridDelta(d, &dx, &dy);
                uint32_t x = (cell & 15u) + (uint32_t)dx, y = (cell >> 4) + (uint32_t)dy, next = y * RWARE_STRIDE + x;
                if (x < w && y < h && (next == target || !rware_homeCell(k, next)) && rware_lawful(k, cell, d) && row[next] + 1u == row[cell]) moves[cell] |= (uint8_t)(1u << d);
            }
        }
    }
    for (uint32_t spot = 0u; spot < cells; spot++) {
        if ((spot & 15u) >= w || !rware_homeCell(k, spot)) continue;
        uint32_t sum = 0u;
        for (uint32_t other = 0u; other < cells; other++)
            if ((other & 15u) < w && rware_homeCell(k, other)) sum += rware_span(spot, other);
        if (sum < least) { least = sum; centre = spot; }
    }
    ((uint16_t *)chart)[0] = (uint16_t)centre;
    for (uint32_t cell = 0u; cell < cells; cell++) {
        uint32_t best = RWARE_NONE, far = ~0u;
        for (uint32_t spot = 0u; spot < cells; spot++) {
            if ((spot & 15u) >= w || !rware_homeCell(k, spot) || rware_span(cell, spot) >= far) continue;
            far = rware_span(cell, spot);
            best = spot;
        }
        parking[cell] = (uint16_t)best;
    }
}

uint64_t rware_shared(float *table) {
    uint64_t tiny = rware_block(&RWARE_KNOBS[0]), size = tiny + rware_block(&RWARE_KNOBS[2]);
    if (table) {
        rware_chart(&RWARE_KNOBS[0], (uint8_t *)table);
        rware_chart(&RWARE_KNOBS[2], (uint8_t *)table + tiny);
    }
    return size / 4u;
}

uint32_t rware_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return rware_floorWords(&RWARE_KNOBS[instance]) + RWARE_NOTES + 3u * rware_boardWords(&RWARE_KNOBS[instance]) + rware_cells(&RWARE_KNOBS[instance]);
}

void rware_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    rware_deal(&RWARE_KNOBS[instance], seed, 0u, words, (uint8_t *)extra);
}

float rware_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    switch (instance) {
        case 0u: return rware_play(&RWARE_KNOBS[0], rware_chartOf(0u, shared), seed, step, ctrl, words, (uint8_t *)extra, obs, done);
        case 1u: return rware_play(&RWARE_KNOBS[1], rware_chartOf(1u, shared), seed, step, ctrl, words, (uint8_t *)extra, obs, done);
        case 2u: return rware_play(&RWARE_KNOBS[2], rware_chartOf(2u, shared), seed, step, ctrl, words, (uint8_t *)extra, obs, done);
        default: __builtin_trap();
    }
}

void rware_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    rware_obs(&RWARE_KNOBS[instance], words, (const uint8_t *)extra, obs);
}

static void rware_mark(uint32_t *board, uint32_t cell, uint32_t bit) {
    board[cell >> 5] = (board[cell >> 5] & ~(1u << (cell & 31u))) | bit << (cell & 31u);
}

static uint32_t rware_follows(uint32_t before, uint32_t pressed, uint32_t after) {
    uint32_t cell = before & RWARE_CELL, facing = before >> RWARE_FACING & 3u, now = after & RWARE_CELL, turned = after >> RWARE_FACING & 3u;
    uint32_t same = !((before ^ after) & ((uint32_t)RWARE_CELL | 3u << RWARE_FACING | 1u << RWARE_LOADED));
    int dx, dy;
    gridDelta(facing, &dx, &dy);
    switch (pressed) {
        case RWARE_NOOP + 1u: return same;
        case RWARE_FORWARD + 1u: return same || (turned == facing && !((before ^ after) >> RWARE_LOADED & 1u) && now == cell + (uint32_t)(dy * RWARE_STRIDE + dx));
        case RWARE_LEFT + 1u:
        case RWARE_RIGHT + 1u: return now == cell && !((before ^ after) >> RWARE_LOADED & 1u) && turned == ((facing ^ facing >> 1) ^ (2u | (pressed == RWARE_RIGHT + 1u)));
        case RWARE_TOGGLE + 1u: return now == cell && turned == facing;
        default: return 0u;
    }
}

static void rware_forget(const rwareKnobs *k, uint32_t *memory) {
    uint32_t *standing = memory + RWARE_NOTES;
    for (uint32_t i = 0u; i < 3u * rware_boardWords(k) + rware_cells(k); i++) standing[i] = 0u;
    for (uint32_t y = 0u; y < rware_height(k); y++) rware_assign(standing, y, rware_home(k, y));
}

static void rware_glance(const rwareKnobs *k, uint32_t *memory, uint32_t cell, uint32_t loaded, const uint8_t *view) {
    uint32_t boards = rware_boardWords(k), *standing = memory + RWARE_NOTES, *wanted = standing + boards, *avoid = wanted + boards, *stamp = avoid + boards;
    for (uint32_t v = 0u; v < 9u; v++) {
        uint32_t x = (cell & 15u) + v % 3u - 1u, y = (cell >> 4) + v / 3u - 1u, at = y * RWARE_STRIDE + x;
        if (x >= rware_width(k) || y >= rware_height(k)) continue;
        uint32_t code = view[rware_spot(at)], other = (uint32_t)(at != cell), shelf = code & 1u & (other | (loaded ^ 1u));
        if (!other || code < RWARE_BUSY) rware_mark(avoid, at, 0u);
        if (!rware_homeCell(k, at)) continue;
        stamp[at] = memory[RWARE_TICK];
        if (other && code >= RWARE_BUSY && shelf) continue;
        rware_mark(standing, at, shelf);
        rware_mark(wanted, at, shelf & code >> 1);
    }
}

static void rware_recall(const rwareKnobs *k, uint32_t *memory, uint32_t cell, uint32_t loaded, const uint8_t *view) {
    uint32_t boards = rware_boardWords(k), *standing = memory + RWARE_NOTES, *wanted = standing + boards, gaps = 0u;
    rware_glance(k, memory, cell, loaded, view);
    for (uint32_t y = 0u; y < rware_height(k); y++) gaps |= rware_home(k, y) & ~rware_row(standing, y);
    if (gaps || !loaded) return;
    for (uint32_t i = 0u; i < boards; i++) standing[i] = wanted[i];
    rware_glance(k, memory, cell, loaded, view);
}

static uint32_t rware_fetch(const rwareKnobs *k, const uint8_t *chart, const uint32_t *memory, uint32_t cell) {
    const uint8_t *reach = rware_reach(k, chart);
    uint32_t boards = rware_boardWords(k), cells = rware_cells(k), left = rware_goal(k), best = RWARE_NONE, fresh = 0u, far = ~0u;
    const uint32_t *wanted = memory + RWARE_NOTES + boards, *avoid = wanted + boards, *stamp = avoid + boards;
    for (uint32_t y = 0u; y < rware_height(k); y++)
        for (uint32_t open = rware_home(k, y) & rware_row(wanted, y) & ~rware_row(avoid, y); open; open &= open - 1u) {
            uint32_t spot = y * RWARE_STRIDE + (uint32_t)__builtin_ctz(open), trip = reach[(size_t)left * cells + spot], other = reach[(size_t)(left + 1u) * cells + spot];
            uint32_t cost = rware_span(cell, spot) + 2u * (trip < other ? trip : other);
            if (stamp[spot] > fresh || (stamp[spot] == fresh && cost < far)) { fresh = stamp[spot]; far = cost; best = spot; }
        }
    return best;
}

static uint32_t rware_sweep(const rwareKnobs *k, const uint32_t *memory, uint32_t cell) {
    uint32_t boards = rware_boardWords(k), best = RWARE_NONE, stale = ~0u, far = ~0u;
    const uint32_t *avoid = memory + RWARE_NOTES + 2u * boards, *stamp = avoid + boards;
    for (uint32_t y = 0u; y < rware_height(k); y++)
        for (uint32_t open = rware_home(k, y) & ~rware_row(avoid, y); open; open &= open - 1u) {
            uint32_t spot = y * RWARE_STRIDE + (uint32_t)__builtin_ctz(open), cost = rware_span(cell, spot);
            if (stamp[spot] < stale || (stamp[spot] == stale && cost < far)) { stale = stamp[spot]; far = cost; best = spot; }
        }
    return best;
}

static uint32_t rware_haul(const rwareKnobs *k, const uint8_t *chart, const uint32_t *memory, const uint8_t *view, uint32_t cell) {
    uint32_t boards = rware_boardWords(k), shut[RWARE_MAX_ROWS / 2u];
    const uint32_t *standing = memory + RWARE_NOTES, *avoid = standing + 2u * boards;
    if (view[rware_spot(cell)] & RWARE_WANTED) {
        uint32_t left = rware_goal(k), target = rware_dock(k, chart, view, cell), other = 2u * left + 1u - target;
        return target - left < 2u && rware_standing(avoid, target) && !rware_standing(avoid, other) ? other : target;
    }
    for (uint32_t i = 0u; i < boards; i++) shut[i] = standing[i] | avoid[i];
    uint32_t target = rware_vacancy(k, chart, shut, view, cell);
    if (target == RWARE_NONE) target = rware_sweep(k, memory, cell);
    return target == RWARE_NONE ? rware_vacancy(k, chart, standing, view, cell) : target;
}

static uint32_t rware_aside(const rwareKnobs *k, const uint8_t *view, uint32_t cell, uint32_t loaded, uint32_t facing) {
    uint32_t best = RWARE_NONE, far = ~0u;
    for (uint32_t d = 0u; d < 4u; d++) {
        int dx, dy;
        gridDelta(d, &dx, &dy);
        uint32_t x = (cell & 15u) + (uint32_t)dx, y = (cell >> 4) + (uint32_t)dy, spot = y * RWARE_STRIDE + x, turns = (d != facing) + (d == (facing ^ 1u));
        if (x >= rware_width(k) || y >= rware_height(k) || view[rware_spot(spot)] >= RWARE_BUSY || (loaded & view[rware_spot(spot)]) || turns >= far) continue;
        far = turns;
        best = spot;
    }
    return best;
}

static inline __attribute__((always_inline)) uint32_t rware_guide(const rwareKnobs *k, const uint8_t *chart, const float *obs, uint32_t *memory) {
    uint32_t cell = (uint32_t)obs[1] * RWARE_STRIDE + (uint32_t)obs[0], loaded = (uint32_t)obs[2], facing = (uint32_t)obs[3], boards = rware_boardWords(k);
    uint32_t last = memory[RWARE_SELF], held = last >> RWARE_STUCK & 3u, self = cell | facing << RWARE_FACING | loaded << RWARE_LOADED, *avoid = memory + RWARE_NOTES + 2u * boards, target, act, side;
    uint32_t stall = (last & RWARE_CELL) != cell ? 0u : held + (held < RWARE_PATIENCE && memory[RWARE_PRESSED] != RWARE_TOGGLE + 1u);
    uint8_t view[RWARE_VIEW] = { 0u };
    for (uint32_t v = 0u; v < 9u; v++) view[rware_spot(cell) - RWARE_MARGIN + v / 3u * RWARE_PITCH + v % 3u] = (uint8_t)obs[5u + v];
    if (!rware_follows(last, memory[RWARE_PRESSED], self)) { rware_forget(k, memory); stall = 0u; }
    else if ((last ^ self) >> RWARE_LOADED & 1u)
        for (uint32_t i = 0u; i < boards; i++) avoid[i] = 0u;
    if (stall > held && stall == RWARE_PATIENCE) rware_mark(avoid, last >> RWARE_AIM & RWARE_NONE, 1u);
    memory[RWARE_TICK]++;
    rware_recall(k, memory, cell, loaded, view);
    if (loaded && rware_homeCell(k, cell) && !(view[rware_spot(cell)] & RWARE_WANTED)) { target = cell; act = RWARE_TOGGLE; }
    else if (loaded) {
        target = rware_haul(k, chart, memory, view, cell);
        act = RWARE_FACE[rware_clear(rware_steer(k, chart)[(size_t)target * rware_cells(k) + cell], cell, facing, view) << 2 | facing];
    } else {
        target = rware_fetch(k, chart, memory, cell);
        if (target == RWARE_NONE) target = rware_sweep(k, memory, cell);
        if (target == RWARE_NONE) target = ((const uint16_t *)chart)[0];
        act = target == cell ? RWARE_TOGGLE : RWARE_FACE[rware_clear(rware_heading(cell, target), cell, facing, view) << 2 | facing];
    }
    side = stall == RWARE_PATIENCE ? rware_aside(k, view, cell, loaded, facing) : RWARE_NONE;
    if (side != RWARE_NONE) { target = side; act = RWARE_FACE[rware_heading(cell, side) << 2 | facing]; }
    else if (stall == RWARE_PATIENCE && loaded && rware_homeCell(k, cell) && rware_aside(k, view, cell, 0u, facing) != RWARE_NONE) { target = cell; act = RWARE_TOGGLE; }
    memory[RWARE_SELF] = self | target << RWARE_AIM | stall << RWARE_STUCK;
    memory[RWARE_PRESSED] = act + 1u;
    return act;
}

void rware_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)words;
    switch (instance) {
        case 0u: ctrl[0] = (float)rware_guide(&RWARE_KNOBS[0], rware_chartOf(0u, shared), obs, extra + rware_floorWords(&RWARE_KNOBS[0])); return;
        case 1u: ctrl[0] = (float)rware_guide(&RWARE_KNOBS[1], rware_chartOf(1u, shared), obs, extra + rware_floorWords(&RWARE_KNOBS[1])); return;
        case 2u: ctrl[0] = (float)rware_guide(&RWARE_KNOBS[2], rware_chartOf(2u, shared), obs, extra + rware_floorWords(&RWARE_KNOBS[2])); return;
        default: __builtin_trap();
    }
}
