#include "../env.h"

#define CONTEXTUAL_ARMS 4u

typedef struct { uint32_t inputs; float drift; } contextualKnobs;

static const contextualKnobs CONTEXTUAL_KNOBS[4] = {
    { 8u, 0.0f },
    { 16u, 0.0f },
    { 8u, 0.001f },
    { 16u, 0.001f }
};

enum { CONTEXTUAL_VALUE = 0, CONTEXTUAL_NOISE = 4, CONTEXTUAL_NEXT = 8, CONTEXTUAL_TAG = 9, CONTEXTUAL_HELD = 12, CONTEXTUAL_GROUPS = 10, CONTEXTUAL_ROWS = 16, CONTEXTUAL_EVALS = 8 };

static inline float4 contextual_load(const uint32_t *at) { float4 v; __builtin_memcpy(&v, at, sizeof v); return v; }
static inline void contextual_store(uint32_t *at, float4 v) { __builtin_memcpy(at, &v, sizeof v); }

static inline __attribute__((always_inline)) uint32_t contextual_held(const contextualKnobs *k) { return k->drift != 0.0f ? k->inputs / 8u : 1u; }

static inline __attribute__((always_inline)) uint32_t contextual_counter(const contextualKnobs *k, uint32_t group, uint32_t lane) {
    uint32_t contexts = k->inputs / 8u;
    return group < contexts ? 256u + 8u * group + 2u * lane : 1u + 2u * (group - contexts) + k->inputs * lane;
}

static inline __attribute__((always_inline)) u44 contextual_block(const contextualKnobs *k, uint32_t seed, uint32_t step, uint32_t group) {
    lane4 counter = { contextual_counter(k, group, 0u), contextual_counter(k, group, 1u), contextual_counter(k, group, 2u), contextual_counter(k, group, 3u) };
    return philox4(seed, ENV_RNG_STREAM, step, counter, 0u, 0u);
}

static inline __attribute__((always_inline)) u4 contextual_word(const contextualKnobs *k, uint32_t seed, uint32_t step, uint32_t eval) {
    return philox4x32(seed, ENV_RNG_STREAM, step, contextual_counter(k, eval / 4u, eval % 4u), 0u, 0u);
}

static inline __attribute__((always_inline)) void contextual_hold(uint32_t *held, uint32_t eval, u4 r) {
    uint32_t *at = held + 16u * (eval / 4u) + eval % 4u;
    at[0] = r.x;
    at[4] = r.y;
    at[8] = r.z;
    at[12] = r.w;
}

static inline __attribute__((always_inline)) float32x4x2_t contextual_normals(u44 r) {
    return (float32x4x2_t){ { normal4(r.x, r.y), normal4(r.z, r.w) } };
}

static inline __attribute__((always_inline)) void contextual_draw(const contextualKnobs *k, uint32_t seed, uint32_t step, uint32_t groups, uint32_t *held, float32x4x2_t *n) {
    uint32_t evals = 4u * contextual_held(k), per = (evals + groups - 1u) / groups;
    u44 r[CONTEXTUAL_GROUPS];
    u4 next[CONTEXTUAL_EVALS];
    __builtin_memcpy(r, held, contextual_held(k) * sizeof r[0]);
    for (uint32_t g = contextual_held(k); g < groups; g++) r[g] = contextual_block(k, seed, step, g);
    for (uint32_t g = 0u; g < groups; g++) {
        n[g] = contextual_normals(r[g]);
        for (uint32_t e = per * g; e < per * (g + 1u) && e < evals; e++) next[e] = contextual_word(k, seed, step + 1u, e);
    }
    for (uint32_t e = 0u; e < evals; e++) contextual_hold(held, e, next[e]);
}

static inline __attribute__((always_inline)) void contextual_evolve(const contextualKnobs *k, uint32_t seed, uint32_t step, uint32_t drifting, uint32_t *extra, float *obs) {
    uint32_t contexts = k->inputs / 8u;
    float32x4x2_t n[CONTEXTUAL_GROUPS];
    float4 value = 0.0f, weight[CONTEXTUAL_ROWS];
    contextual_draw(k, seed, step, contexts + (drifting ? k->inputs / 2u : 0u), extra + CONTEXTUAL_ARMS * k->inputs + CONTEXTUAL_HELD, n);
    for (uint32_t g = 0u; g < contexts; g++) {
        for (uint32_t i = 0u; i < 4u; i++) {
            uint32_t *row = extra + CONTEXTUAL_ARMS * (8u * g + 2u * i);
            float4 first = contextual_load(row), second = contextual_load(row + CONTEXTUAL_ARMS);
            if (drifting) {
                first += k->drift * n[contexts + 4u * g + i].val[0];
                second += k->drift * n[contexts + 4u * g + i].val[1];
            }
            weight[8u * g + 2u * i] = first;
            weight[8u * g + 2u * i + 1u] = second;
            value = __builtin_elementwise_fma(first, splat4(n[g].val[0][i]), value);
            value = __builtin_elementwise_fma(second, splat4(n[g].val[1][i]), value);
        }
    }
    for (uint32_t g = 0u; g < contexts; g++) vst2q_f32(obs + 8u * g, n[g]);
    if (drifting)
        for (uint32_t j = 0u; j < k->inputs; j++) contextual_store(extra + CONTEXTUAL_ARMS * j, weight[j]);
    obs[k->inputs] = 1.0f;
    contextual_store(extra + CONTEXTUAL_ARMS * k->inputs + CONTEXTUAL_VALUE, value);
}

uint32_t contextual_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    return CONTEXTUAL_ARMS * k->inputs + CONTEXTUAL_HELD + 16u * contextual_held(k);
}

void contextual_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)words;
    const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    uint32_t *cache = extra + CONTEXTUAL_ARMS * k->inputs;
    float scale = oak_rsqrt((float)k->inputs), obs[OAK_ENV_OBS_MAX];
    for (uint32_t j = 0u; j < k->inputs; j++) {
        u44 r = philox4(seed, RNG_STREAM_RESET, j + k->inputs * (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
        contextual_store(extra + CONTEXTUAL_ARMS * j, scale * normal4(r.x, r.y));
    }
    for (uint32_t e = 0u; e < 4u * contextual_held(k); e++) contextual_hold(cache + CONTEXTUAL_HELD, e, contextual_word(k, seed, 0u, e));
    contextual_evolve(k, seed, 0u, 0u, extra, obs);
    cache[CONTEXTUAL_NEXT] = 1u;
    batchReset(cache + CONTEXTUAL_TAG);
}

static inline __attribute__((always_inline)) float contextual_advance(const contextualKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *extra, float *obs) {
    uint32_t *cache = extra + CONTEXTUAL_ARMS * k->inputs;
    uint32_t arm = (uint32_t)ctrl[0] % CONTEXTUAL_ARMS;
    if (cache[CONTEXTUAL_NEXT] != step) __builtin_trap();
    cache[CONTEXTUAL_NEXT] = step + 1u;
    uint32_t slot = batchSlot(cache + CONTEXTUAL_TAG, step);
    if (slot == ENV_BATCH) {
        u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
        contextual_store(cache + CONTEXTUAL_NOISE, 0.1f * normal4(r.x, r.y));
        slot = 0u;
    }
    float reward = wordFloat(cache[CONTEXTUAL_VALUE + arm]) + wordFloat(cache[CONTEXTUAL_NOISE + slot]);
    contextual_evolve(k, seed, step, k->drift != 0.0f, extra, obs);
    return reward;
}

float contextual_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)words;
    *done = 0u;
    switch (instance) {
        case 0u: return contextual_advance(&CONTEXTUAL_KNOBS[0], seed, step, ctrl, extra, obs);
        case 1u: return contextual_advance(&CONTEXTUAL_KNOBS[1], seed, step, ctrl, extra, obs);
        case 2u: return contextual_advance(&CONTEXTUAL_KNOBS[2], seed, step, ctrl, extra, obs);
        default: return contextual_advance(&CONTEXTUAL_KNOBS[3], seed, step, ctrl, extra, obs);
    }
}

void contextual_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)words;
    const contextualKnobs *k = &CONTEXTUAL_KNOBS[instance];
    if (extra[CONTEXTUAL_ARMS * k->inputs + CONTEXTUAL_NEXT] != step + 1u) __builtin_trap();
    float32x4x2_t n[CONTEXTUAL_GROUPS];
    uint32_t held[4u * CONTEXTUAL_EVALS];
    for (uint32_t e = 0u; e < 4u * contextual_held(k); e++) contextual_hold(held, e, contextual_word(k, seed, step, e));
    contextual_draw(k, seed, step, k->inputs / 8u, held, n);
    for (uint32_t g = 0u; g < k->inputs / 8u; g++) vst2q_f32(obs + 8u * g, n[g]);
    obs[k->inputs] = 1.0f;
}

void contextual_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)words; (void)obs;
    const uint32_t *cache = extra + CONTEXTUAL_ARMS * CONTEXTUAL_KNOBS[instance].inputs;
    if (cache[CONTEXTUAL_NEXT] != step + 1u) __builtin_trap();
    uint32_t best = 0u;
    float bestValue = -FLT_MAX;
    for (uint32_t arm = 0u; arm < CONTEXTUAL_ARMS; arm++) {
        float value = wordFloat(cache[CONTEXTUAL_VALUE + arm]);
        if (value > bestValue) { bestValue = value; best = arm; }
    }
    ctrl[0] = (float)best;
}
