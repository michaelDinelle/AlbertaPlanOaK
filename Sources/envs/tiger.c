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

void tiger_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
}

float tiger_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const tigerKnobs *k = &TIGER_KNOBS[instance];
    uint32_t tiger = words[0], action = (uint32_t)ctrl[0] % 3u, heard = 0u, hint = 0u;
    int32_t tally = (int32_t)words[3];
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float reward;
    *done = 0u;
    if (action == 2u) {
        reward = TIGER_LISTEN_COST;
        heard = 1u;
        hint = u01(r.x) < k->accuracy ? tiger : tiger ^ 1u;
        tally += hint ? 1 : -1;
    } else {
        reward = action == tiger ? TIGER_MAULING : TIGER_TREASURE;
        tiger = r.y & 1u;
        tally = 0;
        *done = 1u;
    }
    words[0] = tiger; words[1] = heard; words[2] = hint; words[3] = (uint32_t)tally;
    tiger_obs(heard, hint, obs);
    return reward;
}

void tiger_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    tiger_obs(words[1], words[2], obs);
}

void tiger_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    int32_t threshold = (int32_t)tiger_threshold(TIGER_KNOBS[instance].accuracy), tally = (int32_t)words[3];
    if (threshold > 0 && tally >= threshold) ctrl[0] = 0.0f;
    else if (threshold > 0 && -tally >= threshold) ctrl[0] = 1.0f;
    else ctrl[0] = 2.0f;
}
