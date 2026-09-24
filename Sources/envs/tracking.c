#include "../env.h"

typedef struct { float eta, innovation; } trackingKnobs;

static const trackingKnobs TRACKING_KNOBS[2] = {
    { 0.99f, 0.14106736f },
    { 0.999f, 0.044710178f }
};

enum { TRACKING_BLOCKS = 0, TRACKING_DRAWS = TRACKING_BLOCKS + 4 * ENV_BATCH, TRACKING_TAG = TRACKING_DRAWS + 2 * ENV_BATCH, TRACKING_EXTRA = TRACKING_TAG + 1 };

static void tracking_block(uint32_t seed, uint32_t first, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    __builtin_memcpy(blocks, &r, sizeof r);
}

static void tracking_draw(const trackingKnobs *k, const uint32_t *blocks, uint32_t *draws) {
    u44 r;
    __builtin_memcpy(&r, blocks, sizeof r);
    float4 drive = k->innovation * normal4(r.x, r.y), noise = 0.5f * normal4(r.z, r.w);
    __builtin_memcpy(draws, &drive, sizeof drive);
    __builtin_memcpy(draws + ENV_BATCH, &noise, sizeof noise);
}

static float tracking_advance(const trackingKnobs *k, const float *ctrl, uint32_t *words, const uint32_t *draws, float *obs) {
    float next = k->eta * wordFloat(words[0]) + wordFloat(draws[0]);
    float miss = 3.0f * ctrl[0] - next;
    words[0] = floatWord(next);
    obs[0] = next + wordFloat(draws[ENV_BATCH]);
    obs[1] = 1.0f;
    return -miss * miss;
}

static __attribute__((noinline)) float tracking_refill(const trackingKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    tracking_draw(k, extra + TRACKING_BLOCKS, extra + TRACKING_DRAWS);
    tracking_block(seed, step + ENV_BATCH, extra + TRACKING_BLOCKS);
    return tracking_advance(k, ctrl, words, extra + TRACKING_DRAWS, obs);
}

uint32_t tracking_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return TRACKING_EXTRA; }

void tracking_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)instance; (void)words;
    batchReset(&extra[TRACKING_TAG]);
    tracking_block(seed, extra[TRACKING_TAG] + ENV_BATCH, extra + TRACKING_BLOCKS);
}

float tracking_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const trackingKnobs *k = &TRACKING_KNOBS[instance];
    *done = 0u;
    uint32_t slot = batchSlot(&extra[TRACKING_TAG], step);
    if (slot == ENV_BATCH) return tracking_refill(k, seed, step, ctrl, words, extra, obs);
    return tracking_advance(k, ctrl, words, extra + TRACKING_DRAWS + slot, obs);
}

void tracking_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    uint32_t blocks[4 * ENV_BATCH], draws[2 * ENV_BATCH];
    tracking_block(seed, step, blocks);
    tracking_draw(&TRACKING_KNOBS[instance], blocks, draws);
    obs[0] = wordFloat(words[0]) + wordFloat(draws[ENV_BATCH]);
    obs[1] = 1.0f;
}

void tracking_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = oak_clamp((&TRACKING_KNOBS[instance])->eta * wordFloat(words[0]) / 3.0f, -1.0f, 1.0f);
}
