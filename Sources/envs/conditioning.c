#include "../env.h"

#define CONDITIONING_BLINK_REWARD 1.0f
#define CONDITIONING_FALSE_BLINK -0.05f

typedef struct { uint32_t interval, pattern; } conditioningKnobs;

static const conditioningKnobs CONDITIONING_KNOBS[3] = {
    { 10u, 0u },
    { 100u, 0u },
    { 10u, 1u }
};


uint32_t conditioning_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static uint32_t conditioning_predicts(const conditioningKnobs *k, uint32_t cues) { return k->pattern ? (cues & 3u) == 3u && (cues & 4u) == 0u : cues != 0u; }

static void conditioning_obs(const conditioningKnobs *k, uint32_t cues, uint32_t stimulus, float *obs) {
    lane4 shown = k->pattern ? (lane4){ 1u, 2u, 4u, 8u } : (lane4){ 1u, 0u, 0u, 8u };
    float4 row = flag4((cues | stimulus << 3u) & shown);
    __builtin_memcpy(obs, &row, sizeof row);
    obs[4] = 1.0f;
}

static inline __attribute__((always_inline)) void conditioning_draw(const conditioningKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 cues = 1u;
    if (k->pattern) cues += r.y % 7u;
    cues &= (lane4)(r.x < u01Bound(1.0f / (float)(2u * k->interval)));
    __builtin_memcpy(draws, &cues, sizeof cues);
}

static inline __attribute__((always_inline)) float conditioning_advance(const conditioningKnobs *k, const float *ctrl, uint32_t *words, uint32_t drawn, float *obs, uint32_t *done) {
    uint32_t countdown = words[0];
    *done = 0u;
    uint32_t blink = (uint32_t)ctrl[0] & 1u, stimulus = countdown == 1u;
    float reward = stimulus ? (blink ? CONDITIONING_BLINK_REWARD : 0.0f) : (blink ? CONDITIONING_FALSE_BLINK : 0.0f);
    if (countdown > 0u) countdown -= 1u;
    uint32_t cues = countdown == 0u ? drawn : 0u;
    if (conditioning_predicts(k, cues)) countdown = k->interval + 1u;
    words[0] = countdown; words[1] = cues; words[2] = stimulus;
    conditioning_obs(k, cues, stimulus, obs);
    return reward;
}

static inline __attribute__((always_inline)) float conditioning_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t drawn, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: return conditioning_advance(&CONDITIONING_KNOBS[0], ctrl, words, drawn, obs, done);
        case 1u: return conditioning_advance(&CONDITIONING_KNOBS[1], ctrl, words, drawn, obs, done);
        default: return conditioning_advance(&CONDITIONING_KNOBS[2], ctrl, words, drawn, obs, done);
    }
}

static __attribute__((noinline)) float conditioning_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    conditioning_draw(&CONDITIONING_KNOBS[instance], seed, step, extra + 1);
    return conditioning_play(instance, ctrl, words, extra[1], obs, done);
}

void conditioning_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(&extra[0]); }

float conditioning_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return conditioning_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return conditioning_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void conditioning_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    conditioning_obs(&CONDITIONING_KNOBS[instance], words[1], words[2], obs);
}

void conditioning_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 1u ? 1.0f : 0.0f;
}
