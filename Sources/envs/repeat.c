#include "../env.h"

#define REPEAT_SYMBOLS 4u

typedef struct { uint32_t lag; } repeatKnobs;

static const repeatKnobs REPEAT_KNOBS[2] = {
    { 4u },
    { 32u }
};


uint32_t repeat_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static uint32_t repeat_symbolAt(const uint32_t *words, uint32_t age) { return (words[age >> 4u] >> (2u * (age & 15u))) & 3u; }

static void repeat_push(uint32_t *words, uint32_t symbol) {
    for (uint32_t w = 2u; w > 0u; w--) words[w] = (words[w] << 2u) | (words[w - 1u] >> 30u);
    words[0] = (words[0] << 2u) | symbol;
}

static void repeat_obs(uint32_t symbol, float *obs) {
    float4 onehot = flag4((lane4)(symbol == (lane4){ 0u, 1u, 2u, 3u }));
    __builtin_memcpy(obs, &onehot, sizeof onehot);
    obs[REPEAT_SYMBOLS] = 1.0f;
}

static void repeat_draw(uint32_t seed, uint32_t step, uint32_t *symbols) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 symbol = r.x & 3u;
    __builtin_memcpy(symbols, &symbol, sizeof symbol);
}

static inline __attribute__((always_inline)) float repeat_advance(const repeatKnobs *k, const float *ctrl, uint32_t *words, uint32_t symbol, float *obs) {
    float reward = (uint32_t)ctrl[0] % REPEAT_SYMBOLS == repeat_symbolAt(words, k->lag) ? 1.0f : 0.0f;
    repeat_push(words, symbol);
    repeat_obs(symbol, obs);
    return reward;
}

static inline __attribute__((always_inline)) float repeat_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t symbol, float *obs, uint32_t *done) {
    *done = 0u;
    return instance == 0u ? repeat_advance(&REPEAT_KNOBS[0], ctrl, words, symbol, obs) : repeat_advance(&REPEAT_KNOBS[1], ctrl, words, symbol, obs);
}

static __attribute__((noinline)) float repeat_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    repeat_draw(seed, step, extra + 1);
    return repeat_play(instance, ctrl, words, extra[1], obs, done);
}

void repeat_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    u4 r = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    words[0] = r.x; words[1] = r.y; words[2] = r.z;
    batchReset(&extra[0]);
}

float repeat_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return repeat_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return repeat_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void repeat_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    repeat_obs(repeat_symbolAt(words, 0u), obs);
}

void repeat_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)repeat_symbolAt(words, (&REPEAT_KNOBS[instance])->lag);
}
