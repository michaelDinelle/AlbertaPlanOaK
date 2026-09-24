#include "../env.h"

typedef struct { uint32_t states, bits; } riverswimKnobs;

static const riverswimKnobs RIVERSWIM_KNOBS[2] = {
    { 6u, 3u },
    { 24u, 5u }
};


uint32_t riverswim_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 1u + ENV_BATCH; }

static void riverswim_obs(const riverswimKnobs *k, uint32_t position, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b += 4u) {
        float4 row = flag4(position & ((lane4){ 1u, 2u, 4u, 8u } << b));
        __builtin_memcpy(obs + b, &row, sizeof row);
    }
    obs[k->bits] = 1.0f;
}

static void riverswim_draw(uint32_t seed, uint32_t step, uint32_t *drifts) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 drift = ((lane4)(r.x < u01Bound(0.3f)) & 2u) - ((lane4)(r.x < u01Bound(0.4f)) & 1u);
    __builtin_memcpy(drifts, &drift, sizeof drift);
}

static inline __attribute__((always_inline)) float riverswim_advance(const riverswimKnobs *k, const float *ctrl, uint32_t *words, uint32_t drift, float *obs) {
    uint32_t position = words[0], right = (uint32_t)ctrl[0] & 1u, last = k->states - 1u, moved = position + (right ? drift : (uint32_t)-1);
    float reward = right ? (position == last && drift == 1u ? 1.0f : 0.0f) : (position == 0u ? 0.005f : 0.0f);
    position = moved > last ? position : moved;
    words[0] = position;
    riverswim_obs(k, position, obs);
    return reward;
}

static inline __attribute__((always_inline)) float riverswim_play(uint32_t instance, const float *ctrl, uint32_t *words, uint32_t drift, float *obs, uint32_t *done) {
    *done = 0u;
    return instance == 0u ? riverswim_advance(&RIVERSWIM_KNOBS[0], ctrl, words, drift, obs) : riverswim_advance(&RIVERSWIM_KNOBS[1], ctrl, words, drift, obs);
}

static __attribute__((noinline)) float riverswim_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    riverswim_draw(seed, step, extra + 1);
    return riverswim_play(instance, ctrl, words, extra[1], obs, done);
}

void riverswim_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(&extra[0]); }

float riverswim_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return riverswim_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return riverswim_play(instance, ctrl, words, extra[1u + slot], obs, done);
}

void riverswim_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    riverswim_obs(&RIVERSWIM_KNOBS[instance], words[0], obs);
}

void riverswim_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
