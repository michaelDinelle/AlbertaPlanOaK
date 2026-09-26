#include "../env.h"

#define NEXTING_PERIOD 100u

typedef struct { uint32_t horizon; } nextingKnobs;

static const nextingKnobs NEXTING_KNOBS[2] = {
    { 1u },
    { 10u }
};

enum { NEXTING_BLOCKS = 0, NEXTING_DRAWS = NEXTING_BLOCKS + 4 * ENV_BATCH, NEXTING_WAVE = NEXTING_DRAWS + 2 * ENV_BATCH, NEXTING_TAG = NEXTING_WAVE + 2 * NEXTING_PERIOD, NEXTING_EXTRA = NEXTING_TAG + 1 };

static uint32_t nexting_pulseWithin(const nextingKnobs *k, uint32_t phase) { return phase + k->horizon >= NEXTING_PERIOD; }

static void nexting_block(uint32_t seed, uint32_t first, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    __builtin_memcpy(blocks, &r, sizeof r);
}

static void nexting_draw(const uint32_t *blocks, uint32_t *draws) {
    u44 r;
    __builtin_memcpy(&r, blocks, sizeof r);
    float4 s = 0.3f * normal4(r.x, r.y), c = 0.3f * normal4(r.z, r.w);
    float4 early = __builtin_shufflevector(s, c, 0, 4, 1, 5), late = __builtin_shufflevector(s, c, 2, 6, 3, 7);
    __builtin_memcpy(draws, &early, sizeof early);
    __builtin_memcpy(draws + ENV_BATCH, &late, sizeof late);
}

static void nexting_obs(uint32_t phase, const uint32_t *extra, const uint32_t *draws, float *obs) {
    const uint32_t *wave = extra + NEXTING_WAVE + (size_t)phase * 2u;
    obs[0] = wordFloat(wave[0]) + wordFloat(draws[0]);
    obs[1] = wordFloat(wave[1]) + wordFloat(draws[1]);
    obs[2] = phase == 0u ? 1.0f : 0.0f;
    obs[3] = 1.0f;
}

static float nexting_advance(const nextingKnobs *k, const float *ctrl, uint32_t *words, const uint32_t *extra, const uint32_t *draws, float *obs) {
    float reward = ((uint32_t)ctrl[0] & 1u) == nexting_pulseWithin(k, words[0]) ? 1.0f : 0.0f;
    words[0] = words[0] + 1u == NEXTING_PERIOD ? 0u : words[0] + 1u;
    nexting_obs(words[0], extra, draws, obs);
    return reward;
}

static __attribute__((noinline)) float nexting_refill(const nextingKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    nexting_draw(extra + NEXTING_BLOCKS, extra + NEXTING_DRAWS);
    nexting_block(seed, step + ENV_BATCH, extra + NEXTING_BLOCKS);
    return nexting_advance(k, ctrl, words, extra, extra + NEXTING_DRAWS, obs);
}

static inline __attribute__((always_inline)) float nexting_play(const nextingKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    uint32_t slot = batchSlot(&extra[NEXTING_TAG], step);
    if (slot == ENV_BATCH) return nexting_refill(k, seed, step, ctrl, words, extra, obs);
    return nexting_advance(k, ctrl, words, extra, extra + NEXTING_DRAWS + 2u * slot, obs);
}

uint32_t nexting_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return NEXTING_EXTRA; }

void nexting_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % NEXTING_PERIOD;
    for (uint32_t phase = 0u; phase < NEXTING_PERIOD; phase++) {
        float s, c;
        oak_sincos(2.0f * OAK_PI * (float)phase / (float)NEXTING_PERIOD, &s, &c);
        extra[NEXTING_WAVE + 2u * phase] = floatWord(s);
        extra[NEXTING_WAVE + 2u * phase + 1u] = floatWord(c);
    }
    batchReset(&extra[NEXTING_TAG]);
    nexting_block(seed, extra[NEXTING_TAG] + ENV_BATCH, extra + NEXTING_BLOCKS);
}

float nexting_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    if (instance == 0u) return nexting_play(&NEXTING_KNOBS[0], seed, step, ctrl, words, extra, obs);
    return nexting_play(&NEXTING_KNOBS[1], seed, step, ctrl, words, extra, obs);
}

void nexting_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance;
    uint32_t blocks[4 * ENV_BATCH], draws[2 * ENV_BATCH];
    nexting_block(seed, step, blocks);
    nexting_draw(blocks, draws);
    nexting_obs(words[0], extra, draws, obs);
}

void nexting_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)nexting_pulseWithin(&NEXTING_KNOBS[instance], words[0]);
}
