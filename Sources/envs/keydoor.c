#include "../env.h"

typedef struct { uint32_t delay; } keydoorKnobs;

static const keydoorKnobs KEYDOOR_KNOBS[3] = {
    { 8u },
    { 64u },
    { 512u }
};


static void keydoor_obs(const keydoorKnobs *k, uint32_t phase, uint32_t hasKey, float *obs) {
    obs[0] = phase == 0u ? 1.0f : 0.0f;
    obs[1] = (float)hasKey;
    obs[2] = (float)phase / (float)k->delay;
    obs[3] = 1.0f;
}

static float4 keydoor_draw(uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    float4 noise = 0.1f * (float4)vcvtq_n_f32_s32((int32x4_t)((int4)(r.x >> 8u) - 0x800000), 24);
    __builtin_memcpy(draws + 1, &noise, sizeof noise);
    return noise;
}

uint32_t keydoor_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 1u + ENV_BATCH;
}

void keydoor_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(extra); }

static inline __attribute__((always_inline)) float keydoor_move(const keydoorKnobs *k, float noise, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    uint32_t phase = words[0], opening = phase == 0u;
    uint32_t hasKey = (words[1] & (opening - 1u)) | ((uint32_t)ctrl[0] & opening);
    uint32_t door = phase + 1u >= k->delay;
    float reward = door ? (hasKey ? 1.0f : 0.0f) : noise;
    phase = door ? 0u : phase + 1u;
    hasKey = door ? 0u : hasKey;
    *done = door;
    words[0] = phase; words[1] = hasKey;
    keydoor_obs(k, phase, hasKey, obs);
    return reward;
}

static __attribute__((noinline)) float keydoor_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    float noise = keydoor_draw(seed, step, extra)[0];
    switch (instance) {
        case 0u: return keydoor_move(&KEYDOOR_KNOBS[0], noise, ctrl, words, obs, done);
        case 1u: return keydoor_move(&KEYDOOR_KNOBS[1], noise, ctrl, words, obs, done);
        case 2u: return keydoor_move(&KEYDOOR_KNOBS[2], noise, ctrl, words, obs, done);
        default: __builtin_trap();
    }
}

static inline __attribute__((always_inline)) float keydoor_wait(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t slot = batchSlot(extra, step);
    if (slot == ENV_BATCH) return keydoor_refill(instance, seed, step, ctrl, words, extra, obs, done);
    float noise;
    __builtin_memcpy(&noise, &extra[1u + slot], sizeof noise);
    return keydoor_move(&KEYDOOR_KNOBS[instance], noise, ctrl, words, obs, done);
}

float keydoor_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return keydoor_wait(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return keydoor_wait(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return keydoor_wait(2u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void keydoor_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    keydoor_obs(&KEYDOOR_KNOBS[instance], words[0], words[1], obs);
}

void keydoor_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == 0u ? 1.0f : 0.0f;
}
