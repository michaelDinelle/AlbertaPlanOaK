#include "../env.h"

#define PUDDLE_STEP 0.05f
#define PUDDLE_NOISE 0.01f
#define PUDDLE_PENALTY 400.0f
#define PUDDLE_GOAL 0.95f
#define PUDDLE_GOAL_REWARD 100.0f
#define PUDDLE_PLAN_SIDE 101u
#define PUDDLE_PLAN_SWEEPS 512u

static float puddle_depth(float x, float y) {
    float d1 = fabsf(y - 0.75f), d2 = fabsf(x - 0.45f);
    float in1 = (x >= 0.1f && x <= 0.45f) ? d1 : (x < 0.1f ? sqrtf((x - 0.1f) * (x - 0.1f) + d1 * d1) : sqrtf((x - 0.45f) * (x - 0.45f) + d1 * d1));
    float in2 = (y >= 0.4f && y <= 0.8f) ? d2 : (y < 0.4f ? sqrtf(d2 * d2 + (y - 0.4f) * (y - 0.4f)) : sqrtf(d2 * d2 + (y - 0.8f) * (y - 0.8f)));
    float depth = 0.0f;
    if (in1 < 0.1f) depth += 0.1f - in1;
    if (in2 < 0.1f) depth += 0.1f - in2;
    return depth;
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

static void puddle_move(uint32_t seed, uint32_t step, uint32_t action, float *x, float *y) {
    int dx, dy;
    gridDelta(action & 3u, &dx, &dy);
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    *x = oak_clamp(*x + PUDDLE_STEP * (float)dx + PUDDLE_NOISE * normalFrom(r.x, r.y), 0.0f, 1.0f);
    *y = oak_clamp(*y - PUDDLE_STEP * (float)dy + PUDDLE_NOISE * normalFrom(r.z, r.w), 0.0f, 1.0f);
}

static float puddle_planCoordinate(uint32_t index) { return (float)index / (float)(PUDDLE_PLAN_SIDE - 1u); }

static uint32_t puddle_planNode(float x, float y) {
    uint32_t i = (uint32_t)(x * (float)(PUDDLE_PLAN_SIDE - 1u) + 0.5f), j = (uint32_t)(y * (float)(PUDDLE_PLAN_SIDE - 1u) + 0.5f);
    return j * PUDDLE_PLAN_SIDE + i;
}

static float puddle_planScore(const float *plan, float x, float y) {
    return puddle_atGoal(x, y) ? 0.0f : plan[puddle_planNode(x, y)] + puddle_cost(x, y);
}

uint32_t puddle_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return PUDDLE_PLAN_SIDE * PUDDLE_PLAN_SIDE;
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
    puddle_plan((float *)extra);
}

float puddle_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)instance; (void)extra;
    float x = wordFloat(words[0]), y = wordFloat(words[1]);
    puddle_move(seed, step, (uint32_t)ctrl[0], &x, &y);
    *done = 0u;
    float reward = -puddle_cost(x, y);
    if (puddle_atGoal(x, y)) { reward = PUDDLE_GOAL_REWARD; puddle_start(seed, step, words); *done = 1u; }
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
    (void)instance; (void)seed; (void)step; (void)obs;
    const float *plan = (const float *)extra;
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
