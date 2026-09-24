#include "../env.h"

#define WALKBET_CELLS 20u
#define WALKBET_PERIOD 200000u

typedef struct { uint32_t drift; } walkbetKnobs;

static const walkbetKnobs WALKBET_KNOBS[2] = {
    { 0u },
    { 1u }
};


enum { WALKBET_FIXED_RULE, WALKBET_DRIFT_RULES, WALKBET_SHARED_WORDS = WALKBET_DRIFT_RULES + WALKBET_PERIOD };

static float walkbet_rightProbability(const walkbetKnobs *k, uint32_t step) {
    if (!k->drift) return 0.5f;
    float s, c;
    oak_sincos(2.0f * OAK_PI * (float)(step % WALKBET_PERIOD) / (float)WALKBET_PERIOD, &s, &c);
    return 0.5f + 0.2f * s;
}

static float walkbet_absorbRight(float p, uint32_t position) {
    if (fabsf(p - 0.5f) < 1e-6f) return (float)position / (float)WALKBET_CELLS;
    float ratio = (1.0f - p) / p;
    float num = 1.0f - oak_exp((float)position * oak_log(ratio));
    float den = 1.0f - oak_exp((float)WALKBET_CELLS * oak_log(ratio));
    return num / den;
}

static uint32_t walkbet_rule(const walkbetKnobs *k, uint32_t step) {
    float p = walkbet_rightProbability(k, step);
    uint32_t first = WALKBET_CELLS + 1u;
    for (uint32_t position = 0u; position <= WALKBET_CELLS; position++) {
        uint32_t right = walkbet_absorbRight(p, position) > 0.5f;
        if (right && first > WALKBET_CELLS) first = position;
        if (!right && first <= WALKBET_CELLS) __builtin_trap();
    }
    return u01Bound(p) | first;
}

uint64_t walkbet_shared(float *table) {
    if (table) {
        table[WALKBET_FIXED_RULE] = wordFloat(walkbet_rule(&WALKBET_KNOBS[0], 0u));
        for (uint32_t step = 0u; step < WALKBET_PERIOD; step++) table[WALKBET_DRIFT_RULES + step] = wordFloat(walkbet_rule(&WALKBET_KNOBS[1], step));
    }
    return WALKBET_SHARED_WORDS;
}

uint32_t walkbet_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static uint32_t walkbet_ruleAt(const walkbetKnobs *k, const float *shared, uint32_t step) { return floatWord(shared[k->drift ? WALKBET_DRIFT_RULES + step % WALKBET_PERIOD : WALKBET_FIXED_RULE]); }

static void walkbet_draw(const walkbetKnobs *k, const float *shared, uint32_t seed, uint32_t step, uint32_t *moves) {
    lane4 rules = { walkbet_ruleAt(k, shared, step), walkbet_ruleAt(k, shared, step + 1u), walkbet_ruleAt(k, shared, step + 2u), walkbet_ruleAt(k, shared, step + 3u) };
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 move = ((lane4)(r.x < (rules & ~255u)) & 128u) | (rules & 127u);
    __builtin_memcpy(moves, &move, sizeof move);
}

static void walkbet_obs(uint32_t position, float *obs) { obs[0] = (float)position / (float)WALKBET_CELLS; obs[1] = 1.0f; }

static inline __attribute__((always_inline)) float walkbet_play(const float *ctrl, uint32_t *words, uint32_t move, float *obs, uint32_t *done) {
    uint32_t position = words[0];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == (position >= (move & 127u) ? 1u : 0u) ? 1.0f : 0.0f;
    position += move & 128u ? 1u : (uint32_t)-1;
    if (position == 0u || position >= WALKBET_CELLS) { position = WALKBET_CELLS / 2u; *done = 1u; }
    words[0] = position;
    walkbet_obs(position, obs);
    return reward;
}

static __attribute__((noinline)) float walkbet_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    walkbet_draw(&WALKBET_KNOBS[instance], shared, seed, step, extra + 1);
    return walkbet_play(ctrl, words, extra[1], obs, done);
}

void walkbet_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)instance; (void)seed;
    words[0] = WALKBET_CELLS / 2u;
    batchReset(&extra[0]);
}

float walkbet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return walkbet_refill(instance, seed, step, ctrl, words, extra, obs, done, shared);
    return walkbet_play(ctrl, words, extra[1u + slot], obs, done);
}

void walkbet_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    walkbet_obs(words[0], obs);
}

void walkbet_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)extra; (void)obs;
    ctrl[0] = words[0] >= (walkbet_ruleAt(&WALKBET_KNOBS[instance], shared, step + 1u) & 255u) ? 1.0f : 0.0f;
}
