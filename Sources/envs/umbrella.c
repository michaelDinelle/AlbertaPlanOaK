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


uint32_t umbrella_extraWords(uint32_t instance, uint32_t steps) { (void)steps; return 1u + ENV_BATCH * (1u + UMBRELLA_KNOBS[instance].distractors / 4u); }

static inline __attribute__((always_inline)) void umbrella_draw(const umbrellaKnobs *k, uint32_t seed, uint32_t step, uint32_t *draws) {
    lane4 steps = step + (lane4){ 0u, 1u, 2u, 3u };
    u44 r = philox4(seed, ENV_RNG_STREAM, steps, 0u, 0u, 0u);
    lane4 drawn = (r.x & 1u) | (r.y & 1u) << 1u;
    __builtin_memcpy(draws, &drawn, sizeof drawn);
    for (uint32_t part = 1u; part <= k->distractors / 4u; part++) {
        r = philox4(seed, ENV_RNG_STREAM, steps, 15u + part, 0u, 0u);
        __builtin_memcpy(draws + ENV_BATCH * part, &r.x, sizeof r.x);
    }
}

static void umbrella_obs(const umbrellaKnobs *k, const uint32_t *drawn, uint32_t phase, uint32_t need, uint32_t have, float *obs) {
    obs[0] = phase == 0u ? (float)need : 0.0f;
    obs[1] = (float)have;
    obs[2] = (float)phase / (float)k->length;
    obs[3] = 1.0f;
    for (uint32_t d = 0u; d < k->distractors; d += 4u) {
        float4 row = flag4(drawn[ENV_BATCH * (1u + d / 4u)] & (lane4){ 1u, 1u << 8u, 1u << 16u, 1u << 24u });
        __builtin_memcpy(obs + 4u + d, &row, sizeof row);
    }
}

static inline __attribute__((always_inline)) float umbrella_advance(const umbrellaKnobs *k, const float *ctrl, uint32_t *words, const uint32_t *drawn, float *obs, uint32_t *done) {
    uint32_t phase = words[0], need = words[1], have = phase == 0u ? (uint32_t)ctrl[0] & 1u : words[2], last = phase + 1u >= k->length;
    float settle = have == need ? 1.0f : -1.0f, coin = drawn[0] & 1u ? 1.0f : -1.0f, reward = last ? settle : coin;
    phase = last ? 0u : phase + 1u;
    need = last ? (drawn[0] >> 1u) & 1u : need;
    have = last ? 0u : have;
    *done = last;
    words[0] = phase; words[1] = need; words[2] = have;
    umbrella_obs(k, drawn, phase, need, have, obs);
    return reward;
}

static inline __attribute__((always_inline)) float umbrella_play(uint32_t instance, const float *ctrl, uint32_t *words, const uint32_t *drawn, float *obs, uint32_t *done) {
    switch (instance) {
        case 0u: return umbrella_advance(&UMBRELLA_KNOBS[0], ctrl, words, drawn, obs, done);
        case 1u: return umbrella_advance(&UMBRELLA_KNOBS[1], ctrl, words, drawn, obs, done);
        case 2u: return umbrella_advance(&UMBRELLA_KNOBS[2], ctrl, words, drawn, obs, done);
        case 3u: return umbrella_advance(&UMBRELLA_KNOBS[3], ctrl, words, drawn, obs, done);
        case 4u: return umbrella_advance(&UMBRELLA_KNOBS[4], ctrl, words, drawn, obs, done);
        default: return umbrella_advance(&UMBRELLA_KNOBS[5], ctrl, words, drawn, obs, done);
    }
}

static __attribute__((noinline)) float umbrella_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    umbrella_draw(&UMBRELLA_KNOBS[instance], seed, step, extra + 1);
    return umbrella_play(instance, ctrl, words, extra + 1, obs, done);
}

void umbrella_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    words[0] = 0u;
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x & 1u;
    words[2] = 0u;
    batchReset(&extra[0]);
}

float umbrella_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    uint32_t slot = batchSlot(&extra[0], step);
    if (slot == ENV_BATCH) return umbrella_refill(instance, seed, step, ctrl, words, extra, obs, done);
    return umbrella_play(instance, ctrl, words, extra + 1u + slot, obs, done);
}

void umbrella_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    const umbrellaKnobs *k = &UMBRELLA_KNOBS[instance];
    uint32_t drawn[ENV_BATCH * OAK_ENV_OBS_MAX / 4u];
    umbrella_draw(k, seed, step, drawn);
    umbrella_obs(k, drawn, words[0], words[1], words[2], obs);
}

void umbrella_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 0u ? (float)words[1] : 0.0f;
}
