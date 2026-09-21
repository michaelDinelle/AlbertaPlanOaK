#include "../env.h"

#define WALKBET_CELLS 20u
#define WALKBET_PERIOD 200000u

typedef struct { uint32_t drift; } walkbetKnobs;

static const walkbetKnobs WALKBET_KNOBS[2] = {
    { 0u },
    { 1u }
};


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

static void walkbet_obs(uint32_t position, float *obs) { obs[0] = (float)position / (float)WALKBET_CELLS; obs[1] = 1.0f; }

void walkbet_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)extra; words[0] = WALKBET_CELLS / 2u; }

float walkbet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const walkbetKnobs *k = &WALKBET_KNOBS[instance];
    float p = walkbet_rightProbability(k, step);
    uint32_t position = words[0];
    *done = 0u;
    float reward = ((uint32_t)ctrl[0] & 1u) == (walkbet_absorbRight(p, position) > 0.5f ? 1u : 0u) ? 1.0f : 0.0f;
    position += u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) < p ? 1u : (uint32_t)-1;
    if (position == 0u || position >= WALKBET_CELLS) { position = WALKBET_CELLS / 2u; *done = 1u; }
    words[0] = position;
    walkbet_obs(position, obs);
    return reward;
}

void walkbet_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    walkbet_obs(words[0], obs);
}

void walkbet_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)obs;
    ctrl[0] = walkbet_absorbRight(walkbet_rightProbability(&WALKBET_KNOBS[instance], step + 1u), words[0]) > 0.5f ? 1.0f : 0.0f;
}
