#include "../env.h"

typedef struct { uint32_t disks; float remapRate; } hanoiKnobs;

static const hanoiKnobs HANOI_KNOBS[3] = {
    { 4u, 0.0f },
    { 8u, 0.0f },
    { 8u, 0.00001f }
};

enum { HANOI_TARGET = 3, HANOI_CODES = 4, HANOI_MOVES = 6, HANOI_ARRANGEMENTS = 720, HANOI_IDENTITY = 0x543210 };

uint32_t hanoi_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return HANOI_KNOBS[instance].remapRate > 0.0f ? 1u + ENV_BATCH : 0u;
}

static uint32_t hanoi_arrange(uint32_t draw) {
    uint32_t codes = HANOI_IDENTITY, rank = draw % HANOI_ARRANGEMENTS;
    for (uint32_t i = HANOI_MOVES - 1u; i > 0u; i--) {
        uint32_t j = rank % (i + 1u), swap = ((codes >> 4u * i) ^ (codes >> 4u * j)) & 15u;
        codes ^= swap << 4u * i | swap << 4u * j;
        rank /= i + 1u;
    }
    return codes;
}

static uint32_t hanoi_codes(const hanoiKnobs *k, const uint32_t *words) { return k->remapRate > 0.0f ? words[HANOI_CODES] : HANOI_IDENTITY; }

static void hanoi_obs(const hanoiKnobs *k, const uint32_t *words, float *obs) {
    uint32_t middle = words[1], right = words[2], target = words[HANOI_TARGET];
    for (uint32_t d = 0u; d < k->disks; d += 4u) {
        lane4 disk = (lane4){ 1u, 2u, 4u, 8u } << d;
        float4 peg = 0.5f * flag4(middle & disk) + flag4(right & disk);
        __builtin_memcpy(obs + d, &peg, sizeof peg);
    }
    obs[k->disks] = 0.5f * (float)target;
    obs[k->disks + 1u] = 1.0f;
}

static void hanoi_draw(const hanoiKnobs *k, uint32_t seed, uint32_t step, uint32_t *flips) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 flip = (lane4)(r.x < u01Bound(k->remapRate)) & 1u;
    __builtin_memcpy(flips, &flip, sizeof flip);
}

static __attribute__((noinline)) float hanoi_recode(uint32_t seed, uint32_t step, uint32_t *words, float reward) {
    words[HANOI_CODES] = hanoi_arrange(philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u).x);
    return reward;
}

void hanoi_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const hanoiKnobs *k = &HANOI_KNOBS[instance];
    words[0] = (1u << k->disks) - 1u; words[1] = 0u; words[2] = 0u; words[HANOI_TARGET] = 2u;
    if (k->remapRate > 0.0f) {
        words[HANOI_CODES] = hanoi_arrange(philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x);
        batchReset(extra);
    }
}

static inline __attribute__((always_inline)) float hanoi_move(const hanoiKnobs *k, uint32_t flip, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    float pressed = ctrl[0];
    uint32_t code = pressed >= 0.0f && pressed < (float)HANOI_MOVES ? (uint32_t)pressed : 0u;
    uint32_t move = (hanoi_codes(k, words) >> 4u * code) & 15u, from = move >> 1u, side = move & 1u, to = side + (side >= from);
    uint32_t source = words[from], sink = words[to], top = source & -source, under = sink & -sink;
    uint32_t moved = top - 1u < under - 1u ? top : 0u, target = words[HANOI_TARGET];
    uint32_t solved = to == target && (sink | moved) == (1u << k->disks) - 1u;
    words[from] = source ^ moved;
    words[to] = sink | moved;
    words[HANOI_TARGET] = solved ? (target == 2u ? 0u : target + 1u) : target;
    *done = solved;
    hanoi_obs(k, words, obs);
    return flip ? hanoi_recode(seed, step, words, solved ? 1.0f : 0.0f) : solved ? 1.0f : 0.0f;
}

static inline __attribute__((always_inline)) float hanoi_redraw(const hanoiKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    hanoi_draw(k, seed, step, extra + 1);
    return hanoi_move(k, extra[1], seed, step, ctrl, words, obs, done);
}

static __attribute__((noinline)) float hanoi_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: return hanoi_redraw(&HANOI_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
        case 1u: return hanoi_redraw(&HANOI_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
        case 2u: return hanoi_redraw(&HANOI_KNOBS[2], seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float hanoi_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const hanoiKnobs *k = &HANOI_KNOBS[instance];
    if (k->remapRate > 0.0f) {
        uint32_t slot = batchSlot(extra, step);
        if (slot == ENV_BATCH) return hanoi_refill(instance, seed, step, ctrl, words, extra, obs, done);
        return hanoi_move(k, extra[1u + slot], seed, step, ctrl, words, obs, done);
    }
    return hanoi_move(k, 0u, seed, step, ctrl, words, obs, done);
}

float hanoi_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return hanoi_play(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return hanoi_play(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return hanoi_play(2u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void hanoi_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    hanoi_obs(&HANOI_KNOBS[instance], words, obs);
}

void hanoi_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    const hanoiKnobs *k = &HANOI_KNOBS[instance];
    uint32_t goal = words[HANOI_TARGET], from = 0u, to = 0u;
    for (uint32_t d = k->disks; d-- > 0u;) {
        uint32_t peg = ((words[1] >> d) & 1u) | ((words[2] >> d) & 1u) << 1u;
        if (peg == goal) continue;
        from = peg;
        to = goal;
        goal = 3u - peg - goal;
    }
    uint32_t move = 2u * from + to - (to > from), codes = hanoi_codes(k, words), code = 0u;
    for (uint32_t c = 0u; c < HANOI_MOVES; c++) code = ((codes >> 4u * c) & 15u) == move ? c : code;
    ctrl[0] = (float)code;
}
