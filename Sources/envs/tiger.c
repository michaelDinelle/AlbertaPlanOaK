#include "../env.h"

#define TIGER_TREASURE 1.0f
#define TIGER_MAULING -10.0f
#define TIGER_LISTEN_COST -0.1f
#define TIGER_TALLY_MAX 32u

typedef struct { float accuracy; } tigerKnobs;

static const tigerKnobs TIGER_KNOBS[2] = {
    { 0.85f },
    { 0.6f }
};


uint32_t tiger_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 2u + ENV_BATCH; }

static void tiger_obs(uint32_t heard, uint32_t hint, float *obs) {
    obs[0] = (float)heard;
    obs[1] = heard ? (float)hint : 0.0f;
    obs[2] = 1.0f;
}

static uint32_t tiger_threshold(float accuracy) {
    double a = (double)accuracy, ratio = (1.0 - a) / a, bestValue = (double)TIGER_LISTEN_COST;
    uint32_t best = 0u;
    for (uint32_t k = 1u; k <= TIGER_TALLY_MAX; k++) {
        double power = pow(ratio, (double)k), correct = 1.0 / (1.0 + power), listens = (double)k / (2.0 * a - 1.0) * (1.0 - power) / (1.0 + power);
        double value = (correct * (double)TIGER_TREASURE + (1.0 - correct) * (double)TIGER_MAULING + (double)TIGER_LISTEN_COST * listens) / (listens + 1.0);
        if (value > bestValue) { bestValue = value; best = k; }
    }
    return best;
}

static void tiger_draw(const tigerKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 drawn = ((lane4)(r.x < u01Bound(k->accuracy)) & 1u) | ((r.y & 1u) << 1u);
    __builtin_memcpy(draws, &drawn, sizeof drawn);
}

static inline __attribute__((always_inline)) float tiger_play(const float *ctrl, uint32_t *words, uint32_t drawn, float *obs, uint32_t *done) {
    uint32_t tiger = words[0], action = (uint32_t)ctrl[0] % 3u, listen = action == 2u;
    uint32_t hint = listen ? tiger ^ (~drawn & 1u) : 0u;
    int32_t tally = (int32_t)words[3];
    float reward = listen ? TIGER_LISTEN_COST : action == tiger ? TIGER_MAULING : TIGER_TREASURE;
    tally = listen ? tally + (hint ? 1 : -1) : 0;
    *done = !listen;
    words[0] = listen ? tiger : (drawn >> 1u) & 1u; words[1] = listen; words[2] = hint; words[3] = (uint32_t)tally;
    tiger_obs(listen, hint, obs);
    return reward;
}

static __attribute__((noinline)) float tiger_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    tiger_draw(&TIGER_KNOBS[instance], seed, step, extra + 1);
    return tiger_play(ctrl, words, extra[1], obs, done);
}

void tiger_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
    batchReset(&extra[0]);
    extra[1u + ENV_BATCH] = tiger_threshold(TIGER_KNOBS[instance].accuracy);
}

float tiger_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return tiger_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return tiger_play(ctrl, words, extra[1u + slot], obs, done);
}

void tiger_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    tiger_obs(words[1], words[2], obs);
}

void tiger_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    int32_t threshold = (int32_t)extra[1u + ENV_BATCH], tally = (int32_t)words[3];
    if (threshold > 0 && tally >= threshold) ctrl[0] = 0.0f;
    else if (threshold > 0 && -tally >= threshold) ctrl[0] = 1.0f;
    else ctrl[0] = 2.0f;
}
