#include "../env.h"

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
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float reward;
    *done = 0u;
    if (action == 2u) {
        reward = -0.1f;
        heard = 1u;
        hint = u01(r.x) < k->accuracy ? tiger : tiger ^ 1u;
    } else {
        reward = action == tiger ? -10.0f : 1.0f;
        tiger = r.y & 1u;
        *done = 1u;
    }
    words[0] = tiger; words[1] = heard; words[2] = hint;
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
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)(words[0] ^ 1u);
}
