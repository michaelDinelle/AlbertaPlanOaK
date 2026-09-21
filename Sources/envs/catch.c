#include "../env.h"

typedef struct { uint32_t width, height; float noise, scale; } catchKnobs;

static const catchKnobs CATCH_KNOBS[4] = {
    { 5u, 10u, 0.0f, 1.0f },
    { 10u, 20u, 0.0f, 1.0f },
    { 5u, 10u, 0.5f, 1.0f },
    { 5u, 10u, 0.0f, 100.0f }
};


static void catch_obs(const catchKnobs *k, uint32_t ballColumn, uint32_t ballRow, uint32_t paddle, float *obs) {
    obs[0] = (float)ballColumn / (float)k->width;
    obs[1] = (float)ballRow / (float)k->height;
    obs[2] = (float)paddle / (float)k->width;
    obs[3] = 1.0f;
}

static void catch_launch(const catchKnobs *k, uint32_t seed, uint32_t index, uint32_t *words) {
    words[0] = philox4x32(seed, ENV_RNG_STREAM, index, 1u, 0u, 0u).x % k->width;
    words[1] = 0u;
}

void catch_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    const catchKnobs *k = &CATCH_KNOBS[instance];
    catch_launch(k, seed, 0u, words);
    words[2] = k->width / 2u;
}

float catch_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const catchKnobs *k = &CATCH_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0] % 3u, paddle = words[2];
    if (action == 0u && paddle > 0u) paddle -= 1u;
    if (action == 2u && paddle + 1u < k->width) paddle += 1u;
    float reward = 0.0f;
    *done = 0u;
    words[1] += 1u;
    if (words[1] + 1u >= k->height) {
        reward = words[0] == paddle ? 1.0f : -1.0f;
        catch_launch(k, seed, step, words);
        *done = 1u;
    }
    if (k->noise != 0.0f) reward += k->noise * normalFrom(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x, philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).y);
    words[2] = paddle;
    catch_obs(k, words[0], words[1], paddle, obs);
    return reward * k->scale;
}

void catch_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    catch_obs(&CATCH_KNOBS[instance], words[0], words[1], words[2], obs);
}

void catch_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] < words[2] ? 0.0f : words[0] > words[2] ? 2.0f : 1.0f;
}
