#include "../env.h"

typedef struct { float slip; uint32_t positionOnly; } mountaincarKnobs;

static const mountaincarKnobs MOUNTAINCAR_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};

enum { MOUNTAINCAR_BATCH, MOUNTAINCAR_SLIPS, MOUNTAINCAR_NOISE = MOUNTAINCAR_SLIPS + ENV_BATCH, MOUNTAINCAR_EXTRA = MOUNTAINCAR_NOISE + ENV_BATCH };

uint32_t mountainCar_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return MOUNTAINCAR_EXTRA;
}

static inline __attribute__((always_inline)) float4 mountainCar_noise(uint32_t seed, uint32_t step) {
    lane4 zero = 0u;
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero + 1u, zero, zero);
    return 0.01f * normal4(r.x, r.y);
}

static void mountainCar_draw(const mountaincarKnobs *k, uint32_t seed, uint32_t step, uint32_t *extra) {
    if (k->slip > 0.0f) {
        lane4 zero = 0u;
        u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero, zero, zero);
        lane4 slipped = (lane4)(r.x < u01Bound(k->slip)), drawn = (slipped & (r.y % 3u)) | (~slipped & 3u);
        __builtin_memcpy(extra + MOUNTAINCAR_SLIPS, &drawn, sizeof drawn);
    }
    if (k->positionOnly) {
        float4 noise = mountainCar_noise(seed, step);
        __builtin_memcpy(extra + MOUNTAINCAR_NOISE, &noise, sizeof noise);
    }
}

static float mountainCar_slope(float pos) {
    int below = pos < -0x1.921fb6p-1f, low = pos < -0x1.0c1524p-2f, high = pos >= 0x1.0c1524p-2f;
    float left = below ? -2.0f : -1.0f, right = high ? 1.0f : 0.0f, s, c;
    oak_sincosQuarter(3.0f * pos, low ? left : right, &s, &c);
    return 0.0025f * c;
}

static void mountainCar_obs(const mountaincarKnobs *k, float pos, float vel, float noise, float *obs) {
    obs[0] = k->positionOnly ? pos + noise : pos; obs[1] = k->positionOnly ? 0.0f : vel; obs[2] = 1.0f;
}

void mountainCar_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed;
    words[0] = floatWord(-0.5f);
    batchReset(extra + MOUNTAINCAR_BATCH);
}

static inline __attribute__((always_inline)) float mountainCar_advance(const mountaincarKnobs *k, uint32_t slot, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *doneOut) {
    static const float FORCE[3] = { -0.001f, 0.0f, 0.001f };
    uint32_t action = (uint32_t)ctrl[0] % 3u;
    if (k->slip > 0.0f && extra[MOUNTAINCAR_SLIPS + slot] < 3u) action = extra[MOUNTAINCAR_SLIPS + slot];
    float pos = wordFloat(words[0]), vel = wordFloat(words[1]);
    vel += FORCE[action] - mountainCar_slope(pos);
    vel = oak_clamp(vel, -0.07f, 0.07f);
    pos += vel;
    pos = oak_clamp(pos, -1.2f, 0.6f);
    if (pos == -1.2f && vel < 0.0f) vel = 0.0f;
    float reward = 0.0f;
    uint32_t done = 0u;
    if (__builtin_expect(pos >= 0.5f, 0)) {
        reward = 1.0f;
        done = 1u;
        pos = -0.5f; vel = 0.0f;
    }
    words[0] = floatWord(pos); words[1] = floatWord(vel);
    mountainCar_obs(k, pos, vel, wordFloat(extra[MOUNTAINCAR_NOISE + slot]), obs);
    *doneOut = done;
    return reward;
}

static __attribute__((noinline)) float mountainCar_refill(const mountaincarKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut) {
    mountainCar_draw(k, seed, step, extra);
    return mountainCar_advance(k, 0u, ctrl, words, extra, obs, doneOut);
}

float mountainCar_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut, const float *shared) {
    (void)shared;
    const mountaincarKnobs *k = &MOUNTAINCAR_KNOBS[instance];
    uint32_t slot = 0u;
    if (k->slip > 0.0f || k->positionOnly) {
        slot = batchSlot(extra + MOUNTAINCAR_BATCH, step);
        if (slot == ENV_BATCH) return mountainCar_refill(k, seed, step, ctrl, words, extra, obs, doneOut);
    }
    return mountainCar_advance(k, slot, ctrl, words, extra, obs, doneOut);
}

void mountainCar_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared; (void)extra;
    const mountaincarKnobs *k = &MOUNTAINCAR_KNOBS[instance];
    mountainCar_obs(k, wordFloat(words[0]), wordFloat(words[1]), k->positionOnly ? mountainCar_noise(seed, step).x : 0.0f, obs);
}

void mountainCar_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = wordFloat(words[1]) < 0.0f ? 0.0f : 2.0f;
}
