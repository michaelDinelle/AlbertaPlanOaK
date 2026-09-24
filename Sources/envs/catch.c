#include "../env.h"

typedef struct { uint32_t width, height; float noise, scale; } catchKnobs;

static const catchKnobs CATCH_KNOBS[4] = {
    { 5u, 10u, 0.0f, 1.0f },
    { 10u, 20u, 0.0f, 1.0f },
    { 5u, 10u, 0.5f, 1.0f },
    { 5u, 10u, 0.0f, 100.0f }
};

enum { CATCH_BLOCKS = 0, CATCH_DRAWS = CATCH_BLOCKS + 2 * ENV_BATCH, CATCH_TAG = CATCH_DRAWS + ENV_BATCH, CATCH_COLUMNS = CATCH_TAG + 1 };

static uint32_t catch_rows(const catchKnobs *k) { return CATCH_COLUMNS + k->width; }

static void catch_obs(const catchKnobs *k, const uint32_t *extra, uint32_t ballColumn, uint32_t ballRow, uint32_t paddle, float *obs) {
    obs[0] = wordFloat(extra[CATCH_COLUMNS + ballColumn]);
    obs[1] = wordFloat(extra[catch_rows(k) + ballRow]);
    obs[2] = wordFloat(extra[CATCH_COLUMNS + paddle]);
    obs[3] = 1.0f;
}

static void catch_launch(const catchKnobs *k, uint32_t seed, uint32_t index, uint32_t *words) {
    words[0] = philox4x32(seed, ENV_RNG_STREAM, index, 1u, 0u, 0u).x % k->width;
    words[1] = 0u;
}

static void catch_block(uint32_t seed, uint32_t first, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    __builtin_memcpy(blocks, &r.x, sizeof r.x);
    __builtin_memcpy(blocks + ENV_BATCH, &r.y, sizeof r.y);
}

static void catch_draw(const catchKnobs *k, const uint32_t *blocks, uint32_t *draws) {
    lane4 w0, w1;
    __builtin_memcpy(&w0, blocks, sizeof w0);
    __builtin_memcpy(&w1, blocks + ENV_BATCH, sizeof w1);
    float4 n = k->noise * normal4(w0, w1);
    __builtin_memcpy(draws, &n, sizeof n);
}

static inline __attribute__((always_inline)) float catch_finish(const catchKnobs *k, uint32_t *words, const uint32_t *extra, uint32_t paddle, float reward, float noise, float *obs) {
    if (k->noise != 0.0f) reward += noise;
    words[2] = paddle;
    catch_obs(k, extra, words[0], words[1], paddle, obs);
    return reward * k->scale;
}

static __attribute__((noinline)) float catch_land(const catchKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, const uint32_t *extra, uint32_t paddle, float noise, float *obs, uint32_t *done) {
    float reward = words[0] == paddle ? 1.0f : -1.0f;
    catch_launch(k, seed, step, words);
    *done = 1u;
    return catch_finish(k, words, extra, paddle, reward, noise, obs);
}

static inline __attribute__((always_inline)) float catch_advance(const catchKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, const uint32_t *extra, float noise, float *obs, uint32_t *done) {
    uint32_t action = (uint32_t)ctrl[0] % 3u, paddle = words[2];
    paddle -= action == 0u && paddle > 0u;
    paddle += action == 2u && paddle + 1u < k->width;
    *done = 0u;
    words[1] += 1u;
    if (words[1] + 1u >= k->height) return catch_land(k, seed, step, words, extra, paddle, noise, obs, done);
    return catch_finish(k, words, extra, paddle, 0.0f, noise, obs);
}

static __attribute__((noinline)) float catch_refill(const catchKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    catch_draw(k, extra + CATCH_BLOCKS, extra + CATCH_DRAWS);
    catch_block(seed, step + ENV_BATCH, extra + CATCH_BLOCKS);
    return catch_advance(k, seed, step, ctrl, words, extra, wordFloat(extra[CATCH_DRAWS]), obs, done);
}

static inline __attribute__((always_inline)) float catch_play(const catchKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    if (k->noise == 0.0f) return catch_advance(k, seed, step, ctrl, words, extra, 0.0f, obs, done);
    uint32_t slot = batchSlot(&extra[CATCH_TAG], step);
    if (slot == ENV_BATCH) return catch_refill(k, seed, step, ctrl, words, extra, obs, done);
    return catch_advance(k, seed, step, ctrl, words, extra, wordFloat(extra[CATCH_DRAWS + slot]), obs, done);
}

uint32_t catch_extraWords(uint32_t instance, uint32_t steps) { (void)steps; return catch_rows(&CATCH_KNOBS[instance]) + CATCH_KNOBS[instance].height; }

void catch_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const catchKnobs *k = &CATCH_KNOBS[instance];
    catch_launch(k, seed, 0u, words);
    words[2] = k->width / 2u;
    for (uint32_t i = 0u; i < k->width; i++) extra[CATCH_COLUMNS + i] = floatWord((float)i / (float)k->width);
    for (uint32_t i = 0u; i < k->height; i++) extra[catch_rows(k) + i] = floatWord((float)i / (float)k->height);
    if (k->noise == 0.0f) return;
    batchReset(&extra[CATCH_TAG]);
    catch_block(seed, extra[CATCH_TAG] + ENV_BATCH, extra + CATCH_BLOCKS);
}

float catch_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return catch_play(&CATCH_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
        case 1u: return catch_play(&CATCH_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
        case 2u: return catch_play(&CATCH_KNOBS[2], seed, step, ctrl, words, extra, obs, done);
        default: return catch_play(&CATCH_KNOBS[3], seed, step, ctrl, words, extra, obs, done);
    }
}

void catch_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)step;
    catch_obs(&CATCH_KNOBS[instance], extra, words[0], words[1], words[2], obs);
}

void catch_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] < words[2] ? 0.0f : words[0] > words[2] ? 2.0f : 1.0f;
}
