#include "../env.h"

#define REPEAT_SYMBOLS 4u

typedef struct { uint32_t lag; } repeatKnobs;

static const repeatKnobs REPEAT_KNOBS[2] = {
    { 4u },
    { 32u }
};


static uint32_t repeat_symbolAt(const uint32_t *words, uint32_t age) { return (words[age >> 4u] >> (2u * (age & 15u))) & 3u; }

static void repeat_push(uint32_t *words, uint32_t symbol) {
    for (uint32_t w = 2u; w > 0u; w--) words[w] = (words[w] << 2u) | (words[w - 1u] >> 30u);
    words[0] = (words[0] << 2u) | symbol;
}

static void repeat_obs(uint32_t symbol, float *obs) {
    for (uint32_t s = 0u; s < REPEAT_SYMBOLS; s++) obs[s] = s == symbol ? 1.0f : 0.0f;
    obs[REPEAT_SYMBOLS] = 1.0f;
}

void repeat_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    u4 r = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    words[0] = r.x; words[1] = r.y; words[2] = r.z;
}

float repeat_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const repeatKnobs *k = &REPEAT_KNOBS[instance];
    *done = 0u;
    float reward = (uint32_t)ctrl[0] % REPEAT_SYMBOLS == repeat_symbolAt(words, k->lag) ? 1.0f : 0.0f;
    repeat_push(words, philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x & 3u);
    repeat_obs(repeat_symbolAt(words, 0u), obs);
    return reward;
}

void repeat_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    repeat_obs(repeat_symbolAt(words, 0u), obs);
}

void repeat_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)repeat_symbolAt(words, (&REPEAT_KNOBS[instance])->lag);
}
