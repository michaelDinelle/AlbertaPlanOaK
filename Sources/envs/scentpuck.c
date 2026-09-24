#include "../env.h"

#define SCENTPUCK_GOAL_REWARD 100.0f

typedef struct { uint32_t competing; } scentpuckKnobs;

static const scentpuckKnobs SCENTPUCK_KNOBS[2] = {
    { 0u },
    { 1u }
};

enum { SCENTPUCK_BLOCKS = 0, SCENTPUCK_DRAWS = SCENTPUCK_BLOCKS + 4 * ENV_BATCH, SCENTPUCK_TAG = SCENTPUCK_DRAWS + 2 * ENV_BATCH, SCENTPUCK_EXTRA = SCENTPUCK_TAG + 1 };

static float4 scentpuck_exp(float4 x) {
    static const float TERMS[12] = { 1.44269504f, 0.693145751953125f, 1.428606765330187e-06f, 1.0f / 5040.0f, 1.0f / 720.0f, 1.0f / 120.0f,
                                     1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f, -87.33654f, 0.0f };
    float4 a, b, c;
    __builtin_memcpy(&a, opaque(TERMS), sizeof a);
    __builtin_memcpy(&b, (const float *)opaque(TERMS) + 4, sizeof b);
    __builtin_memcpy(&c, (const float *)opaque(TERMS) + 8, sizeof c);
    float4 n = __builtin_elementwise_floor(__builtin_elementwise_fma(x, a.xxxx, c.xxxx));
    float4 f = __builtin_elementwise_fma(-n, a.yyyy, x);
    f = __builtin_elementwise_fma(-n, a.zzzz, f);
    float4 p = a.wwww;
    p = __builtin_elementwise_fma(p, f, b.xxxx);
    p = __builtin_elementwise_fma(p, f, b.yyyy);
    p = __builtin_elementwise_fma(p, f, b.zzzz);
    p = __builtin_elementwise_fma(p, f, b.wwww);
    p = __builtin_elementwise_fma(p, f, c.xxxx);
    p = __builtin_elementwise_fma(p, f, c.yyyy);
    p = __builtin_elementwise_fma(p, f, c.yyyy);
    float4 scale = (float4)((lane4)(__builtin_convertvector(n, int4) + 127) << 23u);
    return select4((lane4)(x < c.zzzz), splat4(0.0f), p * scale);
}

static float scentpuck_spread(float x, float y, float sx, float sy) {
    float dx = x - sx, dy = y - sy;
    return dx * dx + dy * dy;
}

static void scentpuck_place(uint32_t seed, uint32_t index, uint32_t slot, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 8u + slot, 0u, 0u);
    words[2u + 2u * slot] = floatWord(uniformRange(r.x, -1.0f, 1.0f));
    words[3u + 2u * slot] = floatWord(uniformRange(r.y, -1.0f, 1.0f));
}

static void scentpuck_block(uint32_t seed, uint32_t first, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, 1u, 0u, 0u);
    __builtin_memcpy(blocks, &r, sizeof r);
}

static inline __attribute__((always_inline)) void scentpuck_draw(const scentpuckKnobs *k, const uint32_t *blocks, uint32_t *draws) {
    u44 r;
    __builtin_memcpy(&r, blocks, sizeof r);
    float4 near = 0.02f * normal4(r.x, r.y);
    if (!k->competing) {
        __builtin_memcpy(draws, &near, sizeof near);
        return;
    }
    float4 far = 0.02f * normal4(r.z, r.w);
    float4 early = __builtin_shufflevector(near, far, 0, 4, 1, 5), late = __builtin_shufflevector(near, far, 2, 6, 3, 7);
    __builtin_memcpy(draws, &early, sizeof early);
    __builtin_memcpy(draws + ENV_BATCH, &late, sizeof late);
}

static float4 scentpuck_obs(const scentpuckKnobs *k, const uint32_t *words, const uint32_t *draws, float crowd, float *obs) {
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    float4 e = scentpuck_exp((float4){ -4.0f * scentpuck_spread(x, y, wordFloat(words[2]), wordFloat(words[3])),
                                       k->competing ? -4.0f * scentpuck_spread(x, y, wordFloat(words[4]), wordFloat(words[5])) : 0.0f, crowd, 0.0f });
    obs[0] = e[0] + wordFloat(draws[0]);
    obs[1] = k->competing ? e[1] + wordFloat(draws[1]) : 0.0f;
    obs[2] = x;
    obs[3] = y;
    obs[4] = 1.0f;
    return e;
}

static inline __attribute__((always_inline)) float scentpuck_finish(const scentpuckKnobs *k, float x, float y, float reach, float crowd, uint32_t goal, uint32_t *words, const uint32_t *draws, float *obs) {
    words[0] = floatWord(x); words[1] = floatWord(y);
    float4 e = scentpuck_obs(k, words, draws, crowd, obs);
    float reward = -sqrtf(reach);
    if (k->competing) reward -= e[2];
    if (goal) reward += SCENTPUCK_GOAL_REWARD;
    return reward;
}

static __attribute__((noinline)) float scentpuck_score(const scentpuckKnobs *k, uint32_t seed, uint32_t step, float x, float y, float reach, float crowd, uint32_t *words, const uint32_t *draws, float *obs, uint32_t *done) {
    scentpuck_place(seed, step, 0u, words);
    if (k->competing) scentpuck_place(seed, step, 1u, words);
    *done = 1u;
    return scentpuck_finish(k, x, y, reach, crowd, 1u, words, draws, obs);
}

static inline __attribute__((always_inline)) float scentpuck_advance(const scentpuckKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, const uint32_t *draws, float *obs, uint32_t *done) {
    static const float STRIDE[10] = { 0.0f, -0.05f, 0.0f, 0.05f, -0.05f, 0.0f, 0.05f, 0.0f, 0.0f, 0.0f };
    *done = 0u;
    const float *stride = STRIDE + (size_t)((uint32_t)ctrl[0] % 5u) * 2u;
    float x = oak_clamp(wordFloat(words[0]) + stride[0], -1.0f, 1.0f), y = oak_clamp(wordFloat(words[1]) + stride[1], -1.0f, 1.0f);
    float reach = scentpuck_spread(x, y, wordFloat(words[2]), wordFloat(words[3]));
    float crowd = k->competing ? -16.0f * scentpuck_spread(x, y, wordFloat(words[4]), wordFloat(words[5])) : 0.0f;
    if (reach < 0.0025f) return scentpuck_score(k, seed, step, x, y, reach, crowd, words, draws, obs, done);
    return scentpuck_finish(k, x, y, reach, crowd, 0u, words, draws, obs);
}

static __attribute__((noinline)) float scentpuck_refill(const scentpuckKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    scentpuck_draw(k, extra + SCENTPUCK_BLOCKS, extra + SCENTPUCK_DRAWS);
    scentpuck_block(seed, step + ENV_BATCH, extra + SCENTPUCK_BLOCKS);
    return scentpuck_advance(k, seed, step, ctrl, words, extra + SCENTPUCK_DRAWS, obs, done);
}

static inline __attribute__((always_inline)) float scentpuck_play(const scentpuckKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t slot = batchSlot(&extra[SCENTPUCK_TAG], step);
    if (slot == ENV_BATCH) return scentpuck_refill(k, seed, step, ctrl, words, extra, obs, done);
    return scentpuck_advance(k, seed, step, ctrl, words, extra + SCENTPUCK_DRAWS + (1u + k->competing) * slot, obs, done);
}

uint32_t scentpuck_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return SCENTPUCK_EXTRA; }

void scentpuck_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const scentpuckKnobs *k = &SCENTPUCK_KNOBS[instance];
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f);
    scentpuck_place(seed, 0u, 0u, words);
    if (k->competing) scentpuck_place(seed, 0u, 1u, words);
    batchReset(&extra[SCENTPUCK_TAG]);
    scentpuck_block(seed, extra[SCENTPUCK_TAG] + ENV_BATCH, extra + SCENTPUCK_BLOCKS);
}

float scentpuck_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    if (instance == 0u) return scentpuck_play(&SCENTPUCK_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
    return scentpuck_play(&SCENTPUCK_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
}

void scentpuck_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    const scentpuckKnobs *k = &SCENTPUCK_KNOBS[instance];
    uint32_t blocks[4 * ENV_BATCH], draws[2 * ENV_BATCH];
    scentpuck_block(seed, step, blocks);
    scentpuck_draw(k, blocks, draws);
    scentpuck_obs(k, words, draws, 0.0f, obs);
}

void scentpuck_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    float dx = wordFloat(words[2]) - wordFloat(words[0]), dy = wordFloat(words[3]) - wordFloat(words[1]);
    if (fabsf(dx) < 0.025f && fabsf(dy) < 0.025f) { ctrl[0] = 4.0f; return; }
    ctrl[0] = fabsf(dx) > fabsf(dy) ? (dx > 0.0f ? 3.0f : 2.0f) : (dy > 0.0f ? 1.0f : 0.0f);
}
