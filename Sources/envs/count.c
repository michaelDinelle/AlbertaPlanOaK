#include "../env.h"

#define COUNT_MODULUS 4u

typedef struct { uint32_t period; } countKnobs;

static const countKnobs COUNT_KNOBS[2] = {
    { 8u },
    { 32u }
};


uint32_t count_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static void count_draw(uint32_t seed, uint32_t step, uint32_t *events) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 event = (lane4)(r.x < u01Bound(0.3f)) & 1u;
    __builtin_memcpy(events, &event, sizeof event);
}

static void count_obs(uint32_t event, uint32_t query, float *obs) {
    obs[0] = (float)event;
    obs[1] = (float)query;
    obs[2] = 1.0f;
}

static inline __attribute__((always_inline)) float count_advance(const countKnobs *k, const float *ctrl, uint32_t *words, uint32_t event, float *obs) {
    uint32_t count = words[0], phase = words[1], query = words[3];
    float reward = query && (uint32_t)ctrl[0] % COUNT_MODULUS == count % COUNT_MODULUS ? 1.0f : 0.0f;
    count = (query ? 0u : count) + words[2];
    phase = (phase + 1u) % k->period;
    query = phase == 0u ? 1u : 0u;
    words[0] = count; words[1] = phase; words[2] = event; words[3] = query;
    count_obs(event, query, obs);
    return reward;
}

static inline __attribute__((always_inline)) float count_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t event, float *obs, uint32_t *done) {
    *done = 0u;
    return instance == 0u ? count_advance(&COUNT_KNOBS[0], ctrl, words, event, obs) : count_advance(&COUNT_KNOBS[1], ctrl, words, event, obs);
}

static __attribute__((noinline)) float count_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    count_draw(seed, step, extra + 1);
    return count_play(instance, ctrl, words, extra[1], obs, done);
}

void count_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(&extra[0]); }

float count_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return count_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return count_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void count_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    count_obs(words[2], words[3], obs);
}

void count_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)(words[0] % COUNT_MODULUS);
}
