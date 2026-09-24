#include "../env.h"

#define PUDDLE_STEP 0.05f
#define PUDDLE_NOISE 0.01f
#define PUDDLE_PENALTY 400.0f
#define PUDDLE_GOAL 0.95f
#define PUDDLE_GOAL_REWARD 100.0f
#define PUDDLE_PLAN_SIDE 101u
#define PUDDLE_PLAN_SWEEPS 512u

typedef float float2 __attribute__((ext_vector_type(2)));

static float puddle_depth(float x, float y) {
    float2 p = { x, y }, e = p - __builtin_elementwise_min(__builtin_elementwise_max(p, (float2){ 0.1f, 0.4f }), (float2){ 0.45f, 0.8f }), d = p.yx - (float2){ 0.75f, 0.45f };
    float2 depth = __builtin_elementwise_max(0.1f - __builtin_elementwise_sqrt(e * e + d * d), (float2)0.0f);
    return depth.x + depth.y;
}

static int puddle_atGoal(float x, float y) { return x >= PUDDLE_GOAL && y >= PUDDLE_GOAL; }

static float puddle_cost(float x, float y) { return 1.0f + PUDDLE_PENALTY * puddle_depth(x, y); }

static void puddle_obs(const uint32_t *words, float *obs) { obs[0] = wordFloat(words[0]); obs[1] = wordFloat(words[1]); obs[2] = 1.0f; }

static void puddle_start(uint32_t seed, uint32_t index, uint32_t *words) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, index, 9u, 0u, 0u);
    words[0] = floatWord(u01(r.x)); words[1] = floatWord(u01(r.y));
}

static void puddle_drift(uint32_t action, float *x, float *y) {
    int dx, dy;
    gridDelta(action & 3u, &dx, &dy);
    *x = oak_clamp(*x + PUDDLE_STEP * (float)dx, 0.0f, 1.0f);
    *y = oak_clamp(*y - PUDDLE_STEP * (float)dy, 0.0f, 1.0f);
}

static float2 puddle_move(uint32_t action, float2 p, float2 jitter) {
    int dx, dy;
    gridDelta(action & 3u, &dx, &dy);
    return __builtin_elementwise_min(__builtin_elementwise_max(p + (float2){ PUDDLE_STEP * (float)dx, -(PUDDLE_STEP * (float)dy) } + jitter, (float2)0.0f), (float2)1.0f);
}

static float puddle_planCoordinate(uint32_t index) { return (float)index / (float)(PUDDLE_PLAN_SIDE - 1u); }

static uint32_t puddle_planNode(float x, float y) {
    uint32_t i = (uint32_t)(x * (float)(PUDDLE_PLAN_SIDE - 1u) + 0.5f), j = (uint32_t)(y * (float)(PUDDLE_PLAN_SIDE - 1u) + 0.5f);
    return j * PUDDLE_PLAN_SIDE + i;
}

static float puddle_planScore(const float *plan, float x, float y) {
    return puddle_atGoal(x, y) ? 0.0f : plan[puddle_planNode(x, y)] + puddle_cost(x, y);
}

enum { PUDDLE_READY = PUDDLE_PLAN_SIDE * PUDDLE_PLAN_SIDE, PUDDLE_BATCH, PUDDLE_JITTER, PUDDLE_EXTRA = PUDDLE_JITTER + 2 * ENV_BATCH };

uint32_t puddle_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return PUDDLE_EXTRA;
}

static void puddle_draw(uint32_t seed, uint32_t step, uint32_t *extra) {
    lane4 zero = 0u;
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero, zero, zero);
    float4 x = PUDDLE_NOISE * normal4(r.x, r.y), y = PUDDLE_NOISE * normal4(r.z, r.w);
    float4 early = __builtin_shufflevector(x, y, 0, 4, 1, 5), late = __builtin_shufflevector(x, y, 2, 6, 3, 7);
    __builtin_memcpy(extra + PUDDLE_JITTER, &early, sizeof early);
    __builtin_memcpy(extra + PUDDLE_JITTER + 4, &late, sizeof late);
}

static void puddle_plan(float *plan) {
    for (uint32_t node = 0u; node < PUDDLE_PLAN_SIDE * PUDDLE_PLAN_SIDE; node++) {
        float x = puddle_planCoordinate(node % PUDDLE_PLAN_SIDE), y = puddle_planCoordinate(node / PUDDLE_PLAN_SIDE);
        plan[node] = puddle_atGoal(x, y) ? 0.0f : FLT_MAX;
    }
    for (uint32_t sweep = 0u; sweep < PUDDLE_PLAN_SWEEPS; sweep++) {
        int changed = 0;
        for (uint32_t node = 0u; node < PUDDLE_PLAN_SIDE * PUDDLE_PLAN_SIDE; node++) {
            float x = puddle_planCoordinate(node % PUDDLE_PLAN_SIDE), y = puddle_planCoordinate(node / PUDDLE_PLAN_SIDE);
            if (puddle_atGoal(x, y)) continue;
            float best = plan[node];
            for (uint32_t action = 0u; action < 4u; action++) {
                float nx = x, ny = y;
                puddle_drift(action, &nx, &ny);
                float score = puddle_planScore(plan, nx, ny);
                if (score < best) best = score;
            }
            if (best < plan[node]) { plan[node] = best; changed = 1; }
        }
        if (!changed) break;
    }
}

void puddle_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)instance;
    puddle_start(seed, 0u, words);
    batchReset(extra + PUDDLE_BATCH);
}

static inline __attribute__((always_inline)) float puddle_advance(uint32_t seed, uint32_t step, uint32_t slot, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    float2 jitter;
    __builtin_memcpy(&jitter, extra + PUDDLE_JITTER + 2u * slot, sizeof jitter);
    float2 p = puddle_move((uint32_t)ctrl[0], (float2){ wordFloat(words[0]), wordFloat(words[1]) }, jitter);
    *done = 0u;
    float reward = -puddle_cost(p.x, p.y);
    if (__builtin_expect(puddle_atGoal(p.x, p.y), 0)) { reward = PUDDLE_GOAL_REWARD; puddle_start(seed, step, words); *done = 1u; }
    else { words[0] = floatWord(p.x); words[1] = floatWord(p.y); }
    puddle_obs(words, obs);
    return reward;
}

static __attribute__((noinline)) float puddle_refill(uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    puddle_draw(seed, step, extra);
    return puddle_advance(seed, step, 0u, ctrl, words, extra, obs, done);
}

float puddle_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance;
    uint32_t slot = batchSlot(extra + PUDDLE_BATCH, step);
    if (slot == ENV_BATCH) return puddle_refill(seed, step, ctrl, words, extra, obs, done);
    return puddle_advance(seed, step, slot, ctrl, words, extra, obs, done);
}

void puddle_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    puddle_obs(words, obs);
}

void puddle_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    float *plan = (float *)extra;
    if (!extra[PUDDLE_READY]) { puddle_plan(plan); extra[PUDDLE_READY] = 1u; }
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    uint32_t best = 0u;
    float bestScore = FLT_MAX;
    for (uint32_t action = 0u; action < 4u; action++) {
        float nx = x, ny = y;
        puddle_drift(action, &nx, &ny);
        float score = puddle_planScore(plan, nx, ny);
        if (score < bestScore) { bestScore = score; best = action; }
    }
    ctrl[0] = (float)best;
}
