#include "../env.h"

typedef struct { uint32_t states, bits; } riverswimKnobs;

static const riverswimKnobs RIVERSWIM_KNOBS[2] = {
    { 6u, 3u },
    { 24u, 5u }
};


static void riverswim_obs(const riverswimKnobs *k, uint32_t position, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b++) obs[b] = (float)((position >> b) & 1u);
    obs[k->bits] = 1.0f;
}

void riverswim_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float riverswim_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const riverswimKnobs *k = &RIVERSWIM_KNOBS[instance];
    *done = 0u;
    uint32_t position = words[0], last = k->states - 1u;
    float reward = 0.0f;
    if (((uint32_t)ctrl[0] & 1u) == 0u) {
        if (position == 0u) reward = 0.005f;
        else position -= 1u;
    } else {
        float roll = u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x);
        if (position == last) {
            if (roll < 0.3f) reward = 1.0f; else if (roll < 0.4f) position -= 1u;
        } else if (roll < 0.3f) position += 1u;
        else if (roll < 0.4f && position > 0u) position -= 1u;
    }
    words[0] = position;
    riverswim_obs(k, position, obs);
    return reward;
}

void riverswim_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    riverswim_obs(&RIVERSWIM_KNOBS[instance], words[0], obs);
}

void riverswim_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
