#include "../env.h"

#define REGRESSION_BINS 5u

typedef struct { uint32_t inputs; float drift; } regressionKnobs;

static const regressionKnobs REGRESSION_KNOBS[2] = {
    { 8u, 0.001f },
    { 16u, 0.001f }
};

static const float REGRESSION_EDGE[REGRESSION_BINS - 1u] = { -1.0f, -0.3f, 0.3f, 1.0f };

enum { REGRESSION_NEXT = 0, REGRESSION_TARGET = 1, REGRESSION_HELD = 2, REGRESSION_BLOCKS = 4, REGRESSION_EVALS = 8 };

static inline __attribute__((always_inline)) uint32_t regression_counter(const regressionKnobs *k, uint32_t block, uint32_t lane) {
    return (8u * block < k->inputs ? 256u + 8u * block : 1u + 8u * block - k->inputs) + 2u * lane;
}

static inline __attribute__((always_inline)) u44 regression_block(const regressionKnobs *k, uint32_t seed, uint32_t index, uint32_t block) {
    lane4 counter = { regression_counter(k, block, 0u), regression_counter(k, block, 1u), regression_counter(k, block, 2u), regression_counter(k, block, 3u) };
    return philox4(seed, ENV_RNG_STREAM, index, counter, 0u, 0u);
}

static inline __attribute__((always_inline)) u4 regression_word(const regressionKnobs *k, uint32_t seed, uint32_t index, uint32_t eval) {
    return philox4x32(seed, ENV_RNG_STREAM, index, regression_counter(k, eval / 4u, eval % 4u), 0u, 0u);
}

static inline __attribute__((always_inline)) void regression_hold(uint32_t *held, uint32_t eval, u4 r) {
    uint32_t *at = held + 16u * (eval / 4u) + eval % 4u;
    at[0] = r.x;
    at[4] = r.y;
    at[8] = r.z;
    at[12] = r.w;
}

static inline __attribute__((always_inline)) float32x4x2_t regression_normals(u44 r) {
    return (float32x4x2_t){ { normal4(r.x, r.y), normal4(r.z, r.w) } };
}

static inline __attribute__((always_inline)) void regression_draw(const regressionKnobs *k, uint32_t seed, uint32_t index, uint32_t drifting, uint32_t *held, float32x4x2_t *n) {
    uint32_t groups = k->inputs / 8u, blocks = (drifting ? 2u : 1u) * groups, per = 4u * groups / blocks;
    u44 r[REGRESSION_BLOCKS];
    u4 next[REGRESSION_EVALS];
    __builtin_memcpy(r, held, groups * sizeof r[0]);
    for (uint32_t b = groups; b < blocks; b++) r[b] = regression_block(k, seed, index, b);
    for (uint32_t b = 0u; b < blocks; b++) {
        n[b] = regression_normals(r[b]);
        for (uint32_t e = per * b; e < per * (b + 1u); e++) next[e] = regression_word(k, seed, index + 1u, e);
    }
    for (uint32_t e = 0u; e < 4u * groups; e++) regression_hold(held, e, next[e]);
}

static inline __attribute__((always_inline)) uint32_t regression_evolve(const regressionKnobs *k, uint32_t seed, uint32_t index, uint32_t drifting, uint32_t *held, uint32_t *words, float *obs) {
    float32x4x2_t n[REGRESSION_BLOCKS];
    float y = 0.0f;
    regression_draw(k, seed, index, drifting, held, n);
    for (uint32_t j = 0u; j < k->inputs; j += 8u) {
        float32x4x2_t w = vld2q_f32((const float *)words + j), x = n[j / 8u];
        if (drifting) {
            float32x4x2_t d = n[(k->inputs + j) / 8u];
            w.val[0] += k->drift * d.val[0];
            w.val[1] += k->drift * d.val[1];
            vst2q_f32((float *)words + j, w);
        }
        vst2q_f32(obs + j, x);
        for (uint32_t i = 0u; i < 4u; i++) {
            y = fmaf(w.val[0][i], x.val[0][i], y);
            y = fmaf(w.val[1][i], x.val[1][i], y);
        }
    }
    obs[k->inputs] = 1.0f;
    float4 edge;
    __builtin_memcpy(&edge, REGRESSION_EDGE, sizeof edge);
    return 0u - vaddvq_u32((uint32x4_t)(splat4(y) > edge));
}

uint32_t regression_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return REGRESSION_HELD + 2u * REGRESSION_KNOBS[instance].inputs;
}

void regression_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const regressionKnobs *k = &REGRESSION_KNOBS[instance];
    float scale = oak_rsqrt((float)k->inputs), obs[OAK_ENV_OBS_MAX];
    for (uint32_t j = 0u; j < k->inputs; j += 4u) {
        u44 r = philox4(seed, RNG_STREAM_RESET, j + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
        float4 w = scale * normal4(r.x, r.y);
        __builtin_memcpy(words + j, &w, sizeof w);
    }
    for (uint32_t e = 0u; e < k->inputs / 2u; e++) regression_hold(extra + REGRESSION_HELD, e, regression_word(k, seed, 0u, e));
    extra[REGRESSION_NEXT] = 1u;
    extra[REGRESSION_TARGET] = regression_evolve(k, seed, 0u, 0u, extra + REGRESSION_HELD, words, obs);
}

static inline __attribute__((always_inline)) float regression_advance(const regressionKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    if (extra[REGRESSION_NEXT] != step) __builtin_trap();
    float reward = (uint32_t)ctrl[0] % REGRESSION_BINS == extra[REGRESSION_TARGET] ? 1.0f : 0.0f;
    extra[REGRESSION_NEXT] = step + 1u;
    extra[REGRESSION_TARGET] = regression_evolve(k, seed, step, 1u, extra + REGRESSION_HELD, words, obs);
    return reward;
}

float regression_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    if (instance == 0u) return regression_advance(&REGRESSION_KNOBS[0], seed, step, ctrl, words, extra, obs);
    return regression_advance(&REGRESSION_KNOBS[1], seed, step, ctrl, words, extra, obs);
}

void regression_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)words;
    const regressionKnobs *k = &REGRESSION_KNOBS[instance];
    if (extra[REGRESSION_NEXT] != step + 1u) __builtin_trap();
    float32x4x2_t n[REGRESSION_BLOCKS];
    uint32_t held[4u * REGRESSION_EVALS];
    for (uint32_t e = 0u; e < k->inputs / 2u; e++) regression_hold(held, e, regression_word(k, seed, step, e));
    regression_draw(k, seed, step, 0u, held, n);
    for (uint32_t j = 0u; j < k->inputs; j += 8u) vst2q_f32(obs + j, n[j / 8u]);
    obs[k->inputs] = 1.0f;
}

void regression_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)words; (void)obs;
    if (extra[REGRESSION_NEXT] != step + 1u) __builtin_trap();
    ctrl[0] = (float)extra[REGRESSION_TARGET];
}
