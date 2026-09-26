#include "../env.h"

typedef struct { uint32_t delay; } parityKnobs;

static const parityKnobs PARITY_KNOBS[2] = {
    { 4u },
    { 64u }
};


uint32_t parity_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static uint32_t parity_bitAt(const uint32_t *words, uint32_t age) { return (words[age >> 5u] >> (age & 31u)) & 1u; }

static void parity_push(uint32_t *words, uint32_t bit) {
    for (uint32_t w = 2u; w > 0u; w--) words[w] = (words[w] << 1u) | (words[w - 1u] >> 31u);
    words[0] = (words[0] << 1u) | bit;
}

static uint32_t parity_target(const parityKnobs *k, const uint32_t *words) { return parity_bitAt(words, k->delay) ^ parity_bitAt(words, k->delay + 1u); }

static void parity_draw(uint32_t seed, uint32_t step, uint32_t *bits) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 bit = r.x & 1u;
    __builtin_memcpy(bits, &bit, sizeof bit);
}

static inline __attribute__((always_inline)) float parity_advance(const parityKnobs *k, const float *ctrl, uint32_t *words, uint32_t bit, float *obs) {
    float reward = ((uint32_t)ctrl[0] & 1u) == parity_target(k, words) ? 1.0f : 0.0f;
    parity_push(words, bit);
    obs[0] = (float)parity_bitAt(words, 0u);
    obs[1] = 1.0f;
    return reward;
}

static inline __attribute__((always_inline)) float parity_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t bit, float *obs, uint32_t *done) {
    *done = 0u;
    return instance == 0u ? parity_advance(&PARITY_KNOBS[0], ctrl, words, bit, obs) : parity_advance(&PARITY_KNOBS[1], ctrl, words, bit, obs);
}

static __attribute__((noinline)) float parity_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    parity_draw(seed, step, extra + 1);
    return parity_play(instance, ctrl, words, extra[1], obs, done);
}

void parity_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    u4 r = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    words[0] = r.x; words[1] = r.y; words[2] = r.z;
    batchReset(&extra[0]);
}

float parity_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return parity_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return parity_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void parity_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    obs[0] = (float)parity_bitAt(words, 0u);
    obs[1] = 1.0f;
}

void parity_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)parity_target(&PARITY_KNOBS[instance], words);
}
