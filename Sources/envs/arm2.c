#include "../env.h"

#define ARM2_DT 0.02f
#define ARM2_LINK1 1.0f
#define ARM2_LINK2 1.0f
#define ARM2_MASS1 1.0f
#define ARM2_MASS2 1.0f
#define ARM2_GRAVITY 9.81f
#define ARM2_DAMPING 5.0f
#define ARM2_TORQUE1 60.0f
#define ARM2_TORQUE2 30.0f
#define ARM2_SPEED 12.566371f
#define ARM2_ARRIVAL 0.1f
#define ARM2_SPEED_SCALE 0.1f
#define ARM2_ORACLE_STIFFNESS 100.0f
#define ARM2_ORACLE_DAMPING 14.0f

typedef struct { float switchRate; uint32_t regimes; } arm2Knobs;

static const arm2Knobs ARM2_KNOBS[2] = {
    { 0.0f, 1u },
    { 0.00002f, 4u }
};

static const float ARM2_PAYLOAD[4] = { 0.0f, 0.25f, 0.5f, 1.0f };

static const float ARM2_INERTIA1 = ARM2_MASS1 * ARM2_LINK1 * ARM2_LINK1 / 3.0f + ARM2_MASS2 * ARM2_LINK1 * ARM2_LINK1;
static const float ARM2_INERTIA2 = ARM2_MASS2 * ARM2_LINK2 * ARM2_LINK2 / 3.0f;
static const float ARM2_COUPLING = ARM2_MASS2 * ARM2_LINK2 / 2.0f * ARM2_LINK1;
static const float ARM2_LIFT1 = (ARM2_MASS1 * ARM2_LINK1 / 2.0f + ARM2_MASS2 * ARM2_LINK1) * ARM2_GRAVITY;
static const float ARM2_LIFT2 = ARM2_MASS2 * ARM2_LINK2 / 2.0f * ARM2_GRAVITY;
static const float ARM2_DRAG = ARM2_DT * ARM2_DAMPING;
static const float ARM2_REACH = ARM2_LINK1 + ARM2_LINK2;

enum { ARM2_Q1, ARM2_Q2, ARM2_W1, ARM2_W2, ARM2_TARGET_X, ARM2_TARGET_Y, ARM2_DISTANCE, ARM2_REGIME };
enum { ARM2_SIN1, ARM2_COS1, ARM2_SIN2, ARM2_COS2, ARM2_TAG1, ARM2_TAG2, ARM2_BATCH, ARM2_SHIFTS, ARM2_EXTRA = ARM2_SHIFTS + ENV_BATCH };

typedef struct { float s1, c1, s2, c2; } arm2Pose;
typedef struct { float a11, a12, a22, bias1, bias2; } arm2Load;

uint32_t arm2_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return ARM2_KNOBS[instance].regimes > 1u ? ARM2_EXTRA : ARM2_BATCH;
}

static inline __attribute__((always_inline)) arm2Load arm2_load(float payload, arm2Pose p, float w1, float w2) {
    float inertia1 = ARM2_INERTIA1 + payload * (ARM2_LINK1 * ARM2_LINK1), inertia2 = ARM2_INERTIA2 + payload * (ARM2_LINK2 * ARM2_LINK2);
    float coupling = ARM2_COUPLING + payload * (ARM2_LINK1 * ARM2_LINK2), lift1 = ARM2_LIFT1 + payload * (ARM2_LINK1 * ARM2_GRAVITY), lift2 = ARM2_LIFT2 + payload * (ARM2_LINK2 * ARM2_GRAVITY);
    float bend = coupling * p.c2, spin = coupling * p.s2, gravity2 = lift2 * (p.s1 * p.c2 + p.c1 * p.s2);
    return (arm2Load){ inertia1 + inertia2 + ARM2_DRAG + 2.0f * bend, inertia2 + bend, inertia2 + ARM2_DRAG,
                       ARM2_DAMPING * w1 + lift1 * p.s1 + gravity2 - spin * w2 * (2.0f * w1 + w2), ARM2_DAMPING * w2 + gravity2 + spin * w1 * w1 };
}

static inline __attribute__((always_inline)) arm2Pose arm2_recall(const uint32_t *words, const uint32_t *extra) {
    if (extra[ARM2_TAG1] != words[ARM2_Q1] || extra[ARM2_TAG2] != words[ARM2_Q2]) __builtin_trap();
    return (arm2Pose){ wordFloat(extra[ARM2_SIN1]), wordFloat(extra[ARM2_COS1]), wordFloat(extra[ARM2_SIN2]), wordFloat(extra[ARM2_COS2]) };
}

static inline __attribute__((always_inline)) arm2Pose arm2_pose(float q1, float q2, uint32_t *extra) {
    arm2Pose p;
    oak_sincos(q1, &p.s1, &p.c1);
    oak_sincos(q2, &p.s2, &p.c2);
    extra[ARM2_SIN1] = floatWord(p.s1); extra[ARM2_COS1] = floatWord(p.c1); extra[ARM2_SIN2] = floatWord(p.s2); extra[ARM2_COS2] = floatWord(p.c2);
    extra[ARM2_TAG1] = floatWord(q1); extra[ARM2_TAG2] = floatWord(q2);
    return p;
}

static inline __attribute__((always_inline)) void arm2_tip(arm2Pose p, float *x, float *y) {
    *x = ARM2_LINK1 * p.s1 + ARM2_LINK2 * (p.s1 * p.c2 + p.c1 * p.s2);
    *y = -(ARM2_LINK1 * p.c1 + ARM2_LINK2 * (p.c1 * p.c2 - p.s1 * p.s2));
}

static void arm2_target(u4 r, arm2Pose p, uint32_t *words) {
    float radius = ARM2_REACH * sqrtf(u01(r.x)), s, c, x, y;
    oak_sincos(2.0f * OAK_PI * u01(r.y), &s, &c);
    float tx = radius * c, ty = radius * s;
    arm2_tip(p, &x, &y);
    float distance = sqrtf((tx - x) * (tx - x) + (ty - y) * (ty - y));
    words[ARM2_TARGET_X] = floatWord(tx); words[ARM2_TARGET_Y] = floatWord(ty); words[ARM2_DISTANCE] = floatWord(distance);
}

static inline __attribute__((always_inline)) void arm2_obs(arm2Pose p, const uint32_t *words, float *obs) {
    float x, y, tx = wordFloat(words[ARM2_TARGET_X]), ty = wordFloat(words[ARM2_TARGET_Y]);
    arm2_tip(p, &x, &y);
    obs[0] = p.c1; obs[1] = p.s1; obs[2] = p.c2; obs[3] = p.s2;
    obs[4] = ARM2_SPEED_SCALE * wordFloat(words[ARM2_W1]); obs[5] = ARM2_SPEED_SCALE * wordFloat(words[ARM2_W2]);
    obs[6] = tx; obs[7] = ty; obs[8] = tx - x; obs[9] = ty - y; obs[10] = 1.0f;
}

void arm2_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const arm2Knobs *k = &ARM2_KNOBS[instance];
    u4 r = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    arm2Pose p = arm2_pose(0.0f, 0.0f, extra);
    words[ARM2_Q1] = floatWord(0.0f); words[ARM2_Q2] = floatWord(0.0f); words[ARM2_W1] = floatWord(0.0f); words[ARM2_W2] = floatWord(0.0f);
    words[ARM2_REGIME] = r.z % k->regimes;
    arm2_target(r, p, words);
    if (k->regimes > 1u) batchReset(extra + ARM2_BATCH);
}

static void arm2_draw(const arm2Knobs *k, uint32_t seed, uint32_t step, uint32_t *shifts) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    lane4 flip = (lane4)(r.x < u01Bound(k->switchRate));
    lane4 shift = flip & (1u + r.y % (k->regimes - 1u));
    __builtin_memcpy(shifts, &shift, sizeof shift);
}

static __attribute__((noinline)) float arm2_arrive(uint32_t seed, uint32_t step, arm2Pose p, uint32_t regime, float before, float distance, uint32_t *words, float *obs, uint32_t *done) {
    arm2_target(philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u), p, words);
    words[ARM2_REGIME] = regime;
    arm2_obs(p, words, obs);
    *done = 1u;
    return before - distance;
}

static inline __attribute__((always_inline)) float arm2_move(const arm2Knobs *k, uint32_t seed, uint32_t step, uint32_t shift, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    arm2Pose p = arm2_recall(words, extra);
    uint32_t regime = words[ARM2_REGIME];
    float q1 = wordFloat(words[ARM2_Q1]), q2 = wordFloat(words[ARM2_Q2]), w1 = wordFloat(words[ARM2_W1]), w2 = wordFloat(words[ARM2_W2]);
    arm2Load m = arm2_load(ARM2_PAYLOAD[k->regimes > 1u ? regime : 0u], p, w1, w2);
    float f1 = ARM2_TORQUE1 * oak_clamp(ctrl[0], -1.0f, 1.0f) - m.bias1, f2 = ARM2_TORQUE2 * oak_clamp(ctrl[1], -1.0f, 1.0f) - m.bias2;
    float scale = ARM2_DT / (m.a11 * m.a22 - m.a12 * m.a12);
    w1 = oak_clamp(fmaf(scale, m.a22 * f1 - m.a12 * f2, w1), -ARM2_SPEED, ARM2_SPEED);
    w2 = oak_clamp(fmaf(scale, m.a11 * f2 - m.a12 * f1, w2), -ARM2_SPEED, ARM2_SPEED);
    q1 = fmaf(ARM2_DT, w1, q1);
    q2 = fmaf(ARM2_DT, w2, q2);
    q1 = q1 >= OAK_PI ? q1 - 2.0f * OAK_PI : q1 < -OAK_PI ? q1 + 2.0f * OAK_PI : q1;
    q2 = q2 >= OAK_PI ? q2 - 2.0f * OAK_PI : q2 < -OAK_PI ? q2 + 2.0f * OAK_PI : q2;
    words[ARM2_Q1] = floatWord(q1); words[ARM2_Q2] = floatWord(q2); words[ARM2_W1] = floatWord(w1); words[ARM2_W2] = floatWord(w2);
    p = arm2_pose(q1, q2, extra);
    float x, y, tx = wordFloat(words[ARM2_TARGET_X]), ty = wordFloat(words[ARM2_TARGET_Y]), before = wordFloat(words[ARM2_DISTANCE]);
    arm2_tip(p, &x, &y);
    float distance = sqrtf((tx - x) * (tx - x) + (ty - y) * (ty - y));
    regime += shift;
    regime = regime >= k->regimes ? regime - k->regimes : regime;
    if (__builtin_expect(distance < ARM2_ARRIVAL, 0)) return arm2_arrive(seed, step, p, regime, before, distance, words, obs, done);
    words[ARM2_DISTANCE] = floatWord(distance);
    words[ARM2_REGIME] = regime;
    arm2_obs(p, words, obs);
    *done = 0u;
    return before - distance;
}

static __attribute__((noinline)) float arm2_refill(const arm2Knobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    arm2_draw(k, seed, step, extra + ARM2_SHIFTS);
    return arm2_move(k, seed, step, extra[ARM2_SHIFTS], ctrl, words, extra, obs, done);
}

float arm2_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return arm2_move(&ARM2_KNOBS[0], seed, step, 0u, ctrl, words, extra, obs, done);
        case 1u: {
            uint32_t slot = batchSlot(extra + ARM2_BATCH, step);
            if (slot == ENV_BATCH) return arm2_refill(&ARM2_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
            return arm2_move(&ARM2_KNOBS[1], seed, step, extra[ARM2_SHIFTS + slot], ctrl, words, extra, obs, done);
        }
        default: __builtin_trap();
    }
}

void arm2_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)shared;
    arm2_obs(arm2_recall(words, extra), words, obs);
}

static float arm2_chord(float sine, float cosine) { return copysignf(sqrtf(fmaxf(2.0f - 2.0f * cosine, 0.0f)), sine); }

void arm2_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)obs; (void)shared;
    arm2Pose p = arm2_recall(words, extra);
    float w1 = wordFloat(words[ARM2_W1]), w2 = wordFloat(words[ARM2_W2]), tx = wordFloat(words[ARM2_TARGET_X]), ty = wordFloat(words[ARM2_TARGET_Y]);
    float reach2 = tx * tx + ty * ty, inverse = reach2 > 0.0f ? 1.0f / reach2 : 0.0f;
    float elbowCos = oak_clamp((reach2 - ARM2_LINK1 * ARM2_LINK1 - ARM2_LINK2 * ARM2_LINK2) / (2.0f * ARM2_LINK1 * ARM2_LINK2), -1.0f, 1.0f), elbowSin = sqrtf(1.0f - elbowCos * elbowCos);
    float along = ARM2_LINK1 + ARM2_LINK2 * elbowCos, across = ARM2_LINK2 * elbowSin;
    float shoulderCos = (tx * across - ty * along) * inverse, shoulderSin = (tx * along + ty * across) * inverse;
    float e1 = arm2_chord(shoulderSin * p.c1 - shoulderCos * p.s1, shoulderCos * p.c1 + shoulderSin * p.s1);
    float e2 = arm2_chord(elbowSin * p.c2 - elbowCos * p.s2, elbowCos * p.c2 + elbowSin * p.s2);
    float v1 = ARM2_ORACLE_STIFFNESS * e1 - ARM2_ORACLE_DAMPING * w1, v2 = ARM2_ORACLE_STIFFNESS * e2 - ARM2_ORACLE_DAMPING * w2;
    arm2Load m = arm2_load(ARM2_PAYLOAD[words[ARM2_REGIME]], p, w1, w2);
    ctrl[0] = oak_clamp((m.a11 * v1 + m.a12 * v2 + m.bias1) / ARM2_TORQUE1, -1.0f, 1.0f);
    ctrl[1] = oak_clamp((m.a12 * v1 + m.a22 * v2 + m.bias2) / ARM2_TORQUE2, -1.0f, 1.0f);
}
