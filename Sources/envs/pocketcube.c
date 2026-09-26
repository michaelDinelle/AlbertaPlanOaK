#include "../env.h"

typedef struct { uint32_t shell, budget, randomState; } pocketcubeKnobs;

static const pocketcubeKnobs POCKETCUBE_KNOBS[2] = {
    { 3u, 14u, 0u },
    { 0u, 0u, 1u }
};

enum {
    POCKETCUBE_MOVES = 6,
    POCKETCUBE_CUBIES = 7,
    POCKETCUBE_PERMS = 5040,
    POCKETCUBE_TWISTS = 729,
    POCKETCUBE_STATES = POCKETCUBE_PERMS * POCKETCUBE_TWISTS,
    POCKETCUBE_WCA_MOVES = 4,
    POCKETCUBE_SHELL_STATES = 120,
    POCKETCUBE_NEAR_STATES = 385,
    POCKETCUBE_PERM_MOVES = 0,
    POCKETCUBE_TWIST_MOVES = POCKETCUBE_PERM_MOVES + POCKETCUBE_PERMS * POCKETCUBE_MOVES / 2,
    POCKETCUBE_DISTANCE = POCKETCUBE_TWIST_MOVES + POCKETCUBE_TWISTS * POCKETCUBE_MOVES / 2,
    POCKETCUBE_SHARED_WORDS = POCKETCUBE_DISTANCE + POCKETCUBE_STATES / 16
};

static const uint64_t POCKETCUBE_SOLVED = 0x0706050403020100ull;

static const uint8_t POCKETCUBE_TURNS[POCKETCUBE_MOVES][16] = {
    { 3, 0, 1, 2, 4, 5, 6, 7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 1, 2, 3, 0, 4, 5, 6, 7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 4, 1, 2, 0, 6, 5, 3, 7, 0x20, 0x00, 0x00, 0x10, 0x10, 0x00, 0x20, 0x00 },
    { 3, 1, 2, 6, 0, 5, 4, 7, 0x20, 0x00, 0x00, 0x10, 0x10, 0x00, 0x20, 0x00 },
    { 1, 5, 2, 3, 0, 4, 6, 7, 0x10, 0x20, 0x00, 0x00, 0x20, 0x10, 0x00, 0x00 },
    { 4, 0, 2, 3, 5, 1, 6, 7, 0x10, 0x20, 0x00, 0x00, 0x20, 0x10, 0x00, 0x00 }
};

static uint32_t pocketcube_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)POCKETCUBE_MOVES ? (uint32_t)c : 0u;
}

static uint64_t pocketcube_state(const uint32_t *words) {
    uint64_t state;
    __builtin_memcpy(&state, words, sizeof state);
    return state;
}

static void pocketcube_keep(uint32_t *words, uint64_t state, uint32_t moves) {
    __builtin_memcpy(words, &state, sizeof state);
    words[2] = moves;
}

static uint64_t pocketcube_turn(uint64_t state, uint32_t move) {
    uint8x16_t turn = vld1q_u8(POCKETCUBE_TURNS[move]);
    uint8x8_t moved = vadd_u8(vtbl1_u8(vcreate_u8(state), vget_low_u8(turn)), vget_high_u8(turn));
    return vget_lane_u64(vreinterpret_u64_u8(vmin_u8(moved, vsub_u8(moved, vdup_n_u8(0x30u)))), 0);
}

static void pocketcube_obs(uint64_t state, float *obs) {
    uint16x8_t bytes = vmovl_u8(vcreate_u8(state));
    lane4 front = (lane4)vmovl_u16(vget_low_u16(bytes)), back = (lane4)vmovl_high_u16(bytes);
    float4 frontId = __builtin_convertvector(front & 15u, float4) / (float)(POCKETCUBE_CUBIES - 1), frontTwist = __builtin_convertvector(front >> 4u, float4) / 2.0f;
    float4 backId = __builtin_convertvector(back & 15u, float4) / (float)(POCKETCUBE_CUBIES - 1), backTwist = __builtin_convertvector(back >> 4u, float4) / 2.0f;
    float4 rows[4] = { __builtin_shufflevector(frontId, frontTwist, 0, 4, 1, 5), __builtin_shufflevector(frontId, frontTwist, 2, 6, 3, 7),
                       __builtin_shufflevector(backId, backTwist, 0, 4, 1, 5), __builtin_shufflevector(backId, backTwist, 2, 6, 3, 7) };
    __builtin_memcpy(obs, rows, 2u * POCKETCUBE_CUBIES * sizeof(float));
    obs[2 * POCKETCUBE_CUBIES] = 1.0f;
}

static uint32_t pocketcube_index(uint64_t state) {
    uint32_t perm = 0u, twist = 0u, seen = 0u;
    for (uint32_t slot = 0u; slot < POCKETCUBE_CUBIES; slot++) {
        uint32_t cubie = (uint32_t)(state >> (8u * slot)) & 15u;
        perm = perm * (POCKETCUBE_CUBIES - slot) + cubie - (uint32_t)oak_popcount(seen & ((1u << cubie) - 1u));
        seen |= 1u << cubie;
        twist = slot + 1u < POCKETCUBE_CUBIES ? twist * 3u + ((uint32_t)(state >> (8u * slot + 4u)) & 15u) : twist;
    }
    return perm * POCKETCUBE_TWISTS + twist;
}

static uint64_t pocketcube_decode(uint32_t index) {
    uint32_t perm = index / POCKETCUBE_TWISTS, twist = index % POCKETCUBE_TWISTS, digit[POCKETCUBE_CUBIES], unused = 0x6543210u, sum = 0u;
    uint64_t state = (uint64_t)POCKETCUBE_CUBIES << (8u * POCKETCUBE_CUBIES);
    for (uint32_t slot = POCKETCUBE_CUBIES; slot-- > 0u;) {
        digit[slot] = perm % (POCKETCUBE_CUBIES - slot);
        perm /= POCKETCUBE_CUBIES - slot;
    }
    for (uint32_t slot = 0u; slot < POCKETCUBE_CUBIES; slot++) {
        uint32_t shift = 4u * digit[slot];
        state |= (uint64_t)(unused >> shift & 15u) << (8u * slot);
        unused = (unused & ((1u << shift) - 1u)) | (unused >> (shift + 4u) << shift);
    }
    for (uint32_t slot = POCKETCUBE_CUBIES - 1u; slot-- > 0u;) {
        sum += twist % 3u;
        state |= (uint64_t)(twist % 3u) << (8u * slot + 4u);
        twist /= 3u;
    }
    return state | (uint64_t)((3u - sum % 3u) % 3u) << (8u * (POCKETCUBE_CUBIES - 1u) + 4u);
}

static uint32_t pocketcube_ball(uint32_t radius, uint32_t halfTurns, uint64_t *ball, uint32_t capacity, uint32_t *layer) {
    uint32_t count = 1u;
    ball[0] = POCKETCUBE_SOLVED;
    *layer = 0u;
    for (uint32_t depth = 0u; depth < radius; depth++) {
        uint32_t from = *layer, to = count;
        for (uint32_t i = from; i < to; i++)
            for (uint32_t move = 0u; move < POCKETCUBE_MOVES; move++) {
                uint64_t next = ball[i];
                for (uint32_t turns = 0u; turns <= halfTurns; turns++) {
                    next = pocketcube_turn(next, move);
                    uint32_t known = 0u;
                    for (uint32_t j = 0u; j < count; j++) known |= ball[j] == next;
                    if (known) continue;
                    if (count == capacity) __builtin_trap();
                    ball[count++] = next;
                }
            }
        *layer = to;
    }
    return count;
}

static uint32_t pocketcube_near(const uint32_t *near, uint32_t index) {
    uint32_t low = 0u, high = POCKETCUBE_NEAR_STATES;
    while (low < high) {
        uint32_t middle = (low + high) / 2u;
        if (near[middle] < index) low = middle + 1u;
        else high = middle;
    }
    return low < POCKETCUBE_NEAR_STATES && near[low] == index;
}

static uint64_t pocketcube_scramble(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *extra) {
    uint32_t random = POCKETCUBE_KNOBS[instance].randomState, bound = random ? POCKETCUBE_STATES : POCKETCUBE_SHELL_STATES, floor = (0u - bound) % bound;
    for (uint32_t block = 0u;; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, block, 0u, 0u);
        uint32_t draws[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t i = 0u; i < 4u; i++) {
            uint64_t scaled = (uint64_t)draws[i] * bound;
            uint32_t pick = (uint32_t)(scaled >> 32);
            if ((uint32_t)scaled < floor) continue;
            if (!random) return pocketcube_state(extra + 2u * pick);
            if (!pocketcube_near(extra, pick)) return pocketcube_decode(pick);
        }
    }
}

uint32_t pocketcube_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return POCKETCUBE_KNOBS[instance].randomState ? POCKETCUBE_NEAR_STATES : 2u * POCKETCUBE_SHELL_STATES;
}

void pocketcube_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const pocketcubeKnobs *k = &POCKETCUBE_KNOBS[instance];
    uint64_t ball[POCKETCUBE_NEAR_STATES];
    uint32_t layer, count = pocketcube_ball(k->randomState ? POCKETCUBE_WCA_MOVES - 1u : k->shell, k->randomState, ball, POCKETCUBE_NEAR_STATES, &layer);
    if (k->randomState) {
        if (count != POCKETCUBE_NEAR_STATES) __builtin_trap();
        for (uint32_t i = 0u; i < count; i++) {
            uint32_t index = pocketcube_index(ball[i]), j = i;
            for (; j > 0u && extra[j - 1u] > index; j--) extra[j] = extra[j - 1u];
            extra[j] = index;
        }
    } else {
        if (count - layer != POCKETCUBE_SHELL_STATES) __builtin_trap();
        __builtin_memcpy(extra, ball + layer, POCKETCUBE_SHELL_STATES * sizeof ball[0]);
    }
    pocketcube_keep(words, pocketcube_scramble(instance, seed, 0u, extra), 0u);
}

static __attribute__((noinline)) float pocketcube_finish(uint32_t instance, uint32_t seed, uint32_t step, uint64_t state, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    uint64_t next = pocketcube_scramble(instance, seed, step, extra);
    pocketcube_keep(words, next, 0u);
    *done = 1u;
    pocketcube_obs(next, obs);
    return state == POCKETCUBE_SOLVED ? 1.0f : 0.0f;
}

static inline __attribute__((always_inline)) float pocketcube_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    const pocketcubeKnobs *k = &POCKETCUBE_KNOBS[instance];
    uint64_t state = pocketcube_turn(pocketcube_state(words), pocketcube_action(ctrl));
    uint32_t moves = words[2] + 1u;
    if (state == POCKETCUBE_SOLVED || (k->budget != 0u && moves == k->budget)) return pocketcube_finish(instance, seed, step, state, words, extra, obs, done);
    pocketcube_keep(words, state, moves);
    *done = 0u;
    pocketcube_obs(state, obs);
    return 0.0f;
}

float pocketcube_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return pocketcube_play(0u, seed, step, ctrl, words, extra, obs, done);
        default: return pocketcube_play(1u, seed, step, ctrl, words, extra, obs, done);
    }
}

void pocketcube_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    pocketcube_obs(pocketcube_state(words), obs);
}

static uint32_t pocketcube_mark(const uint8_t *distance, uint32_t index) { return distance[index >> 2] >> (2u * (index & 3u)) & 3u; }

static uint32_t pocketcube_neighbour(const uint8_t *table, uint32_t perm, uint32_t twist, uint32_t move) {
    uint16_t to[2];
    __builtin_memcpy(&to[0], table + 4u * POCKETCUBE_PERM_MOVES + 2u * (perm * POCKETCUBE_MOVES + move), sizeof to[0]);
    __builtin_memcpy(&to[1], table + 4u * POCKETCUBE_TWIST_MOVES + 2u * (twist * POCKETCUBE_MOVES + move), sizeof to[1]);
    return to[0] * (uint32_t)POCKETCUBE_TWISTS + to[1];
}

uint64_t pocketcube_shared(float *table) {
    if (!table) return POCKETCUBE_SHARED_WORDS;
    uint8_t *bytes = (uint8_t *)table, *distance = bytes + 4u * POCKETCUBE_DISTANCE;
    for (uint32_t move = 0u; move < POCKETCUBE_MOVES; move++) {
        for (uint32_t perm = 0u; perm < POCKETCUBE_PERMS; perm++) {
            uint16_t to = (uint16_t)(pocketcube_index(pocketcube_turn(pocketcube_decode(perm * POCKETCUBE_TWISTS), move)) / POCKETCUBE_TWISTS);
            __builtin_memcpy(bytes + 4u * POCKETCUBE_PERM_MOVES + 2u * (perm * POCKETCUBE_MOVES + move), &to, sizeof to);
        }
        for (uint32_t twist = 0u; twist < POCKETCUBE_TWISTS; twist++) {
            uint16_t to = (uint16_t)(pocketcube_index(pocketcube_turn(pocketcube_decode(twist), move)) % POCKETCUBE_TWISTS);
            __builtin_memcpy(bytes + 4u * POCKETCUBE_TWIST_MOVES + 2u * (twist * POCKETCUBE_MOVES + move), &to, sizeof to);
        }
    }
    __builtin_memset(distance, 0xFF, POCKETCUBE_STATES / 4u);
    distance[0] = 0xFCu;
    uint32_t reached = 1u;
    for (uint32_t depth = 0u, added = 1u; added != 0u; depth++, reached += added) {
        added = 0u;
        for (uint32_t perm = 0u; perm < POCKETCUBE_PERMS; perm++)
            for (uint32_t twist = 0u; twist < POCKETCUBE_TWISTS; twist++) {
                if (pocketcube_mark(distance, perm * POCKETCUBE_TWISTS + twist) != depth % 3u) continue;
                for (uint32_t move = 0u; move < POCKETCUBE_MOVES; move++) {
                    uint32_t next = pocketcube_neighbour(bytes, perm, twist, move), mark = pocketcube_mark(distance, next);
                    if (mark == depth % 3u) __builtin_trap();
                    if (mark != 3u) continue;
                    distance[next >> 2] ^= (uint8_t)((3u ^ (depth + 1u) % 3u) << (2u * (next & 3u)));
                    added++;
                }
            }
    }
    if (reached != POCKETCUBE_STATES) __builtin_trap();
    return POCKETCUBE_SHARED_WORDS;
}

void pocketcube_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    const uint8_t *table = (const uint8_t *)shared, *distance = table + 4u * POCKETCUBE_DISTANCE;
    uint32_t index = pocketcube_index(pocketcube_state(words)), perm = index / POCKETCUBE_TWISTS, twist = index % POCKETCUBE_TWISTS;
    uint32_t closer = (pocketcube_mark(distance, index) + 2u) % 3u, found = 0u;
    for (uint32_t move = 0u; move < POCKETCUBE_MOVES; move++) found |= (uint32_t)(pocketcube_mark(distance, pocketcube_neighbour(table, perm, twist, move)) == closer) << move;
    ctrl[0] = (float)(found ? (uint32_t)__builtin_ctz(found) : 0u);
}
