#include "../env.h"

typedef struct { float cueRate; } operantKnobs;

static const operantKnobs OPERANT_KNOBS[2] = {
    { 1.0f / 16.0f },
    { 1.0f / 64.0f }
};


uint32_t operant_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static void operant_draw(const operantKnobs *k, uint32_t seed, uint32_t step, uint32_t *cues) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 cue = (lane4)(r.x < u01Bound(k->cueRate)) & 1u;
    __builtin_memcpy(cues, &cue, sizeof cue);
}

static inline __attribute__((always_inline)) float operant_play(const float *ctrl, uint32_t *words, uint32_t cue, float *obs, uint32_t *done) {
    *done = 0u;
    uint32_t press = (uint32_t)ctrl[0] & 1u;
    float reward = words[0] != 0u ? (press ? 1.0f : 0.0f) : (press ? -0.1f : 0.0f);
    words[0] = cue;
    obs[0] = (float)words[0];
    obs[1] = 1.0f;
    return reward;
}

static __attribute__((noinline)) float operant_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    operant_draw(&OPERANT_KNOBS[instance], seed, step, extra + 1);
    return operant_play(ctrl, words, extra[1], obs, done);
}

void operant_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(&extra[0]); }

float operant_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return operant_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return operant_play(ctrl, words, extra[1u + slot], obs, done);
}

void operant_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    obs[0] = (float)words[0];
    obs[1] = 1.0f;
}

void operant_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)words[0];
}
