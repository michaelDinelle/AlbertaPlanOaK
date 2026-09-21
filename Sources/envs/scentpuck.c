#include "../env.h"

typedef struct { uint32_t competing; } scentpuckKnobs;

static const scentpuckKnobs SCENTPUCK_KNOBS[2] = {
    { 0u },
    { 1u }
};


static float scentpuck_scent(float x, float y, float sx, float sy) {
    float dx = x - sx, dy = y - sy;
    return oak_exp(-4.0f * (dx * dx + dy * dy));
}

static void scentpuck_place(uint32_t seed, uint32_t index, uint32_t slot, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 8u + slot, 0u, 0u);
    words[2u + 2u * slot] = floatWord(uniformRange(r.x, -1.0f, 1.0f));
    words[3u + 2u * slot] = floatWord(uniformRange(r.y, -1.0f, 1.0f));
}

static void scentpuck_obs(const scentpuckKnobs *k, uint32_t seed, uint32_t step, const uint32_t *words, float *obs) {
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u);
    obs[0] = scentpuck_scent(x, y, wordFloat(words[2]), wordFloat(words[3])) + 0.02f * normalFrom(r.x, r.y);
    obs[1] = k->competing ? scentpuck_scent(x, y, wordFloat(words[4]), wordFloat(words[5])) + 0.02f * normalFrom(r.z, r.w) : 0.0f;
    obs[2] = x;
    obs[3] = y;
    obs[4] = 1.0f;
}

void scentpuck_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const scentpuckKnobs *k = &SCENTPUCK_KNOBS[instance];
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f);
    scentpuck_place(seed, 0u, 0u, words);
    if (k->competing) scentpuck_place(seed, 0u, 1u, words);
}

float scentpuck_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const scentpuckKnobs *k = &SCENTPUCK_KNOBS[instance];
    *done = 0u;
    uint32_t action = (uint32_t)ctrl[0] % 5u;
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    if (action < 4u) {
        int dx, dy;
        gridDelta(action, &dx, &dy);
        x = oak_clamp(x + 0.05f * (float)dx, -1.0f, 1.0f);
        y = oak_clamp(y + 0.05f * (float)dy, -1.0f, 1.0f);
    }
    float gx = wordFloat(words[2]), gy = wordFloat(words[3]);
    float dx = x - gx, dy = y - gy;
    float reward = -sqrtf(dx * dx + dy * dy);
    if (k->competing) {
        float bx = x - wordFloat(words[4]), by = y - wordFloat(words[5]);
        reward -= oak_exp(-16.0f * (bx * bx + by * by));
    }
    if (dx * dx + dy * dy < 0.0025f) {
        reward += 1.0f;
        scentpuck_place(seed, step, 0u, words);
        if (k->competing) scentpuck_place(seed, step, 1u, words);
        *done = 1u;
    }
    words[0] = floatWord(x); words[1] = floatWord(y);
    scentpuck_obs(k, seed, step, words, obs);
    return reward;
}

void scentpuck_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    scentpuck_obs(&SCENTPUCK_KNOBS[instance], seed, step, words, obs);
}

void scentpuck_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    float dx = wordFloat(words[2]) - wordFloat(words[0]), dy = wordFloat(words[3]) - wordFloat(words[1]);
    if (fabsf(dx) < 0.025f && fabsf(dy) < 0.025f) { ctrl[0] = 4.0f; return; }
    ctrl[0] = fabsf(dx) > fabsf(dy) ? (dx > 0.0f ? 3.0f : 2.0f) : (dy > 0.0f ? 1.0f : 0.0f);
}
