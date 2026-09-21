#include "../env.h"

static float puddle_depth(float x, float y) {
    float d1 = fabsf(y - 0.75f), d2 = fabsf(x - 0.45f);
    float in1 = (x >= 0.1f && x <= 0.45f) ? d1 : (x < 0.1f ? sqrtf((x - 0.1f) * (x - 0.1f) + d1 * d1) : sqrtf((x - 0.45f) * (x - 0.45f) + d1 * d1));
    float in2 = (y >= 0.4f && y <= 0.8f) ? d2 : (y < 0.4f ? sqrtf(d2 * d2 + (y - 0.4f) * (y - 0.4f)) : sqrtf(d2 * d2 + (y - 0.8f) * (y - 0.8f)));
    float depth = 0.0f;
    if (in1 < 0.1f) depth += 0.1f - in1;
    if (in2 < 0.1f) depth += 0.1f - in2;
    return depth;
}

static float puddle_reward(float x, float y) { return -1.0f - 400.0f * puddle_depth(x, y); }

static void puddle_obs(const uint32_t *words, float *obs) { obs[0] = wordFloat(words[0]); obs[1] = wordFloat(words[1]); obs[2] = 1.0f; }

static void puddle_start(uint32_t seed, uint32_t index, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 9u, 0u, 0u);
    words[0] = floatWord(u01(r.x)); words[1] = floatWord(u01(r.y));
}

void puddle_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)extra; puddle_start(seed, 0u, words); }

static void puddle_move(uint32_t seed, uint32_t step, uint32_t action, float *x, float *y) {
    int dx, dy;
    gridDelta(action & 3u, &dx, &dy);
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    *x = oak_clamp(*x + 0.05f * (float)dx + 0.01f * normalFrom(r.x, r.y), 0.0f, 1.0f);
    *y = oak_clamp(*y - 0.05f * (float)dy + 0.01f * normalFrom(r.z, r.w), 0.0f, 1.0f);
}

float puddle_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    puddle_move(seed, step, (uint32_t)ctrl[0], &x, &y);
    *done = 0u;
    float reward = puddle_reward(x, y);
    if (x >= 0.95f && y >= 0.95f) { reward = 0.0f; puddle_start(seed, step, words); *done = 1u; }
    else { words[0] = floatWord(x); words[1] = floatWord(y); }
    puddle_obs(words, obs);
    return reward;
}

void puddle_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    puddle_obs(words, obs);
}

void puddle_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    uint32_t best = 0u;
    float bestScore = -FLT_MAX;
    for (uint32_t a = 0u; a < 4u; a++) {
        int dx, dy;
        gridDelta(a, &dx, &dy);
        float nx = oak_clamp(x + 0.05f * (float)dx, 0.0f, 1.0f), ny = oak_clamp(y - 0.05f * (float)dy, 0.0f, 1.0f);
        float score = -((1.0f - nx) + (1.0f - ny)) * 20.0f - 400.0f * puddle_depth(nx, ny);
        if (score > bestScore) { bestScore = score; best = a; }
    }
    ctrl[0] = (float)best;
}
