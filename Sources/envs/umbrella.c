#include "../env.h"

typedef struct { uint32_t length, distractors; } umbrellaKnobs;

static const umbrellaKnobs UMBRELLA_KNOBS[6] = {
    { 4u, 0u },
    { 16u, 0u },
    { 64u, 0u },
    { 4u, 12u },
    { 16u, 12u },
    { 64u, 12u }
};


static void umbrella_obs(const umbrellaKnobs *k, uint32_t seed, uint32_t step, uint32_t phase, uint32_t need, uint32_t have, float *obs) {
    obs[0] = phase == 0u ? (float)need : 0.0f;
    obs[1] = (float)have;
    obs[2] = (float)phase / (float)k->length;
    obs[3] = 1.0f;
    for (uint32_t d = 0u; d < k->distractors; d++)
        obs[4u + d] = (float)(philox4x32(seed, ENV_RNG_STREAM, step, 16u + (d >> 2u), 0u, 0u).x >> (8u * (d & 3u)) & 1u);
}

void umbrella_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = 0u;
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
    words[2] = 0u;
}

float umbrella_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const umbrellaKnobs *k = &UMBRELLA_KNOBS[instance];
    uint32_t phase = words[0], need = words[1], have = words[2];
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    float reward;
    *done = 0u;
    if (phase == 0u) have = (uint32_t)ctrl[0] & 1u;
    if (phase + 1u >= k->length) {
        reward = have == need ? 1.0f : -1.0f;
        phase = 0u;
        need = r.y & 1u;
        have = 0u;
        *done = 1u;
    } else {
        reward = (r.x & 1u) ? 1.0f : -1.0f;
        phase += 1u;
    }
    words[0] = phase; words[1] = need; words[2] = have;
    umbrella_obs(k, seed, step, phase, need, have, obs);
    return reward;
}

void umbrella_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    umbrella_obs(&UMBRELLA_KNOBS[instance], seed, step, words[0], words[1], words[2], obs);
}

void umbrella_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 0u ? (float)words[1] : 0.0f;
}
