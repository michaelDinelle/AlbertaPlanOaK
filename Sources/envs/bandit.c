#include "../env.h"

typedef struct { uint32_t arms; float drift; } banditKnobs;

static const banditKnobs BANDIT_KNOBS[4] = {
    { 3u, 0.0f },
    { 6u, 0.0f },
    { 3u, 0.001f },
    { 6u, 0.001f }
};

static uint32_t bandit_blocks(const banditKnobs *k) { return k->drift != 0.0f ? k->arms + 1u : 1u; }
static uint32_t bandit_draws(const banditKnobs *k) { return 2u * ENV_BATCH * bandit_blocks(k); }
static uint32_t bandit_tag(const banditKnobs *k) { return 3u * ENV_BATCH * bandit_blocks(k); }

static inline __attribute__((always_inline)) void bandit_block(uint32_t seed, uint32_t first, uint32_t b, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, b, 0u, 0u);
    __builtin_memcpy(blocks + 2u * ENV_BATCH * b, &r.x, sizeof r.x);
    __builtin_memcpy(blocks + 2u * ENV_BATCH * b + ENV_BATCH, &r.y, sizeof r.y);
}

static inline __attribute__((always_inline)) void bandit_draw(const banditKnobs *k, uint32_t b, const uint32_t *blocks, uint32_t *draws) {
    lane4 w0, w1;
    __builtin_memcpy(&w0, blocks + 2u * ENV_BATCH * b, sizeof w0);
    __builtin_memcpy(&w1, blocks + 2u * ENV_BATCH * b + ENV_BATCH, sizeof w1);
    float4 d = (b == 0u ? 0.5f : k->drift) * normal4(w0, w1);
    __builtin_memcpy(draws + ENV_BATCH * b, &d, sizeof d);
}

static __attribute__((noinline, cold)) float bandit_settle(const banditKnobs *k, uint32_t seed, uint32_t step, uint32_t arm, uint32_t *words, float noise) {
    float reward = wordFloat(words[arm]) + noise;
    for (uint32_t i = 0u; i < k->arms; i++)
        if (words[i] == floatWord(-0.0f)) {
            u4 d = philox4x32(seed, ENV_RNG_STREAM, step, 1u + i, 0u, 0u);
            words[i] = floatWord(wordFloat(words[i]) + k->drift * normalFrom(d.x, d.y));
        }
    return reward;
}

static inline __attribute__((always_inline)) float bandit_advance(const banditKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, const uint32_t *draws, float *obs) {
    uint32_t arm = (uint32_t)ctrl[0] % k->arms;
    obs[0] = 1.0f;
    if (k->drift != 0.0f) {
        float reward = wordFloat(words[arm]) + wordFloat(draws[0]);
        for (uint32_t i = 0u; i < k->arms; i++) words[i] = floatWord(wordFloat(words[i]) + wordFloat(draws[ENV_BATCH * (1u + i)]));
        return reward;
    }
    uint32_t negativeZero = 0u;
    for (uint32_t i = 0u; i < k->arms; i++) negativeZero |= words[i] == floatWord(-0.0f);
    if (negativeZero) return bandit_settle(k, seed, step, arm, words, wordFloat(draws[0]));
    return wordFloat(words[arm]) + wordFloat(draws[0]);
}

static inline __attribute__((always_inline)) float bandit_restock(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    for (uint32_t b = 0u; b < bandit_blocks(k); b++) {
        bandit_draw(k, b, extra, extra + bandit_draws(k));
        bandit_block(seed, step + ENV_BATCH, b, extra);
    }
    return bandit_advance(k, seed, step, ctrl, words, extra + bandit_draws(k), obs);
}

static __attribute__((noinline)) float bandit_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    switch (instance) {
        case 0u: return bandit_restock(0u, seed, step, ctrl, words, extra, obs);
        case 1u: return bandit_restock(1u, seed, step, ctrl, words, extra, obs);
        case 2u: return bandit_restock(2u, seed, step, ctrl, words, extra, obs);
        default: return bandit_restock(3u, seed, step, ctrl, words, extra, obs);
    }
}

static inline __attribute__((always_inline)) float bandit_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    uint32_t slot = batchSlot(&extra[bandit_tag(k)], step);
    if (slot == ENV_BATCH) return bandit_refill(instance, seed, step, ctrl, words, extra, obs);
    return bandit_advance(k, seed, step, ctrl, words, extra + bandit_draws(k) + slot, obs);
}

uint32_t bandit_extraWords(uint32_t instance, uint32_t steps) { (void)steps; return bandit_tag(&BANDIT_KNOBS[instance]) + 1u; }

void bandit_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    for (uint32_t i = 0u; i < k->arms; i++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, i, 0u, 0u, 0u);
        words[i] = floatWord(normalFrom(r.x, r.y));
    }
    batchReset(&extra[bandit_tag(k)]);
    for (uint32_t b = 0u; b < bandit_blocks(k); b++) bandit_block(seed, extra[bandit_tag(k)] + ENV_BATCH, b, extra);
}

float bandit_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    switch (instance) {
        case 0u: return bandit_play(0u, seed, step, ctrl, words, extra, obs);
        case 1u: return bandit_play(1u, seed, step, ctrl, words, extra, obs);
        case 2u: return bandit_play(2u, seed, step, ctrl, words, extra, obs);
        default: return bandit_play(3u, seed, step, ctrl, words, extra, obs);
    }
}

void bandit_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)extra; (void)step;
    obs[0] = 1.0f;
}

void bandit_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    const banditKnobs *k = &BANDIT_KNOBS[instance];
    uint32_t best = 0u;
    for (uint32_t i = 1u; i < k->arms; i++) if (wordFloat(words[i]) > wordFloat(words[best])) best = i;
    ctrl[0] = (float)best;
}
