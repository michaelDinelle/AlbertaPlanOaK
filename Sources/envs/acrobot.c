#include "../env.h"

#define ACROBOT_DT 0.2f
#define ACROBOT_OMEGA1_MAX 12.566371f
#define ACROBOT_OMEGA2_MAX 28.274334f
#define ACROBOT_ENERGY_TOP 19.6f
#define ACROBOT_GOAL_REWARD 1.0f

typedef struct { float slip; uint32_t positionOnly; } acrobotKnobs;

static const acrobotKnobs ACROBOT_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};

typedef struct { float sinTheta2, cosTheta2, weight1, gravity2; } acrobotTerms;

enum { ACROBOT_SLIP_TAG = sizeof(acrobotTerms) / sizeof(uint32_t), ACROBOT_SLIPS, ACROBOT_SEEN = ACROBOT_SLIPS + ENV_BATCH, ACROBOT_EXTRA = ACROBOT_SEEN + 2 };

static inline float acrobot_finite(float x) { __builtin_assume(__builtin_isfinite(x)); return x; }

static __attribute__((noinline)) float acrobot_wrapWide(float angle) { return wrapPi(angle); }

static inline __attribute__((always_inline)) float acrobot_wrap(float angle) {
    if (fabsf(angle) < 3.0f * OAK_PI) return angle - (angle >= OAK_PI ? 2.0f * OAK_PI : angle < -OAK_PI ? -2.0f * OAK_PI : 0.0f);
    return acrobot_wrapWide(angle);
}

static void acrobot_obs(const acrobotKnobs *k, float sinTheta1, float cosTheta1, float sinTheta2, float cosTheta2, float omega1, float omega2, float *obs) {
    obs[0] = cosTheta1; obs[1] = sinTheta1; obs[2] = cosTheta2; obs[3] = sinTheta2;
    obs[4] = k->positionOnly ? 0.0f : omega1; obs[5] = k->positionOnly ? 0.0f : omega2; obs[6] = 1.0f;
}

static inline __attribute__((always_inline)) acrobotTerms acrobot_terms(float theta1, float theta2) {
    acrobotTerms t;
    oak_sincos(acrobot_finite(theta2), &t.sinTheta2, &t.cosTheta2);
    t.weight1 = 14.7f * oak_cos(acrobot_finite(theta1 - OAK_PI / 2.0f));
    t.gravity2 = 4.9f * oak_cos(acrobot_finite(theta1 + theta2 - OAK_PI / 2.0f));
    return t;
}

static void acrobot_remember(uint32_t *extra, acrobotTerms t) {
    extra[0] = floatWord(t.sinTheta2); extra[1] = floatWord(t.cosTheta2); extra[2] = floatWord(t.weight1); extra[3] = floatWord(t.gravity2);
}

static inline __attribute__((always_inline)) acrobotTerms acrobot_recall(const uint32_t *extra) {
    return (acrobotTerms){ wordFloat(extra[0]), wordFloat(extra[1]), wordFloat(extra[2]), wordFloat(extra[3]) };
}

static inline __attribute__((always_inline)) void acrobot_dynamics(float omega1, float omega2, acrobotTerms t, float tau, float *derivative) {
    float inertia1 = 3.5f + t.cosTheta2;
    float inertia2 = fmaf(0.5f, t.cosTheta2, 1.25f);
    float gravity1 = -0.5f * omega2 * omega2 * t.sinTheta2 - omega2 * omega1 * t.sinTheta2 + t.weight1 + t.gravity2;
    float alpha2 = (tau + inertia2 / inertia1 * gravity1 - 0.5f * omega1 * omega1 * t.sinTheta2 - t.gravity2) / (1.25f - inertia2 * inertia2 / inertia1);
    float alpha1 = (inertia2 * alpha2 + gravity1) / -inertia1;
    derivative[0] = omega1; derivative[1] = omega2; derivative[2] = alpha1; derivative[3] = alpha2;
}

static void acrobot_draw(const acrobotKnobs *k, uint32_t seed, uint32_t step, uint32_t *slips) {
    lane4 zero = 0u;
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero, zero, zero);
    lane4 slipped = (lane4)(r.x < u01Bound(k->slip));
    lane4 drawn = (slipped & (r.y % 3u)) | (~slipped & 3u);
    __builtin_memcpy(slips, &drawn, sizeof drawn);
}

static __attribute__((noinline)) float acrobot_restart(const acrobotKnobs *k, float reward, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut) {
    acrobotTerms t = acrobot_terms(0.0f, 0.0f);
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f); words[2] = floatWord(0.0f); words[3] = floatWord(0.0f);
    acrobot_remember(extra, t);
    acrobot_obs(k, t.sinTheta2, t.cosTheta2, t.sinTheta2, t.cosTheta2, 0.0f, 0.0f, obs);
    *doneOut = 1u;
    return reward;
}

uint32_t acrobot_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return ACROBOT_EXTRA;
}

void acrobot_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)seed; (void)steps;
    acrobot_remember(extra, acrobot_terms(wordFloat(words[0]), wordFloat(words[1])));
    batchReset(&extra[ACROBOT_SLIP_TAG]);
    extra[ACROBOT_SEEN] = words[0]; extra[ACROBOT_SEEN + 1] = words[1];
}

float acrobot_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut, const float *shared) {
    (void)shared;
    static const float torque[3] = { -1.0f, 0.0f, 1.0f };
    const acrobotKnobs *k = &ACROBOT_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0] % 3u;
    if (k->slip > 0.0f) {
        uint32_t slot = batchSlot(&extra[ACROBOT_SLIP_TAG], step);
        if (slot == ENV_BATCH) { acrobot_draw(k, seed, step, extra + ACROBOT_SLIPS); slot = 0u; }
        if (extra[ACROBOT_SLIPS + slot] < 3u) action = extra[ACROBOT_SLIPS + slot];
    }
    float theta1 = wordFloat(words[0]), theta2 = wordFloat(words[1]);
    float omega1 = wordFloat(words[2]), omega2 = wordFloat(words[3]);
    float tau = torque[action];
    float k1[4], k2[4], k3[4], k4[4];
    acrobot_dynamics(omega1, omega2, acrobot_recall(extra), tau, k1);
    float halfDt = ACROBOT_DT / 2.0f;
    acrobot_dynamics(omega1 + halfDt * k1[2], omega2 + halfDt * k1[3], acrobot_terms(theta1 + halfDt * k1[0], theta2 + halfDt * k1[1]), tau, k2);
    acrobot_dynamics(omega1 + halfDt * k2[2], omega2 + halfDt * k2[3], acrobot_terms(theta1 + halfDt * k2[0], theta2 + halfDt * k2[1]), tau, k3);
    acrobot_dynamics(omega1 + ACROBOT_DT * k3[2], omega2 + ACROBOT_DT * k3[3], acrobot_terms(theta1 + ACROBOT_DT * k3[0], theta2 + ACROBOT_DT * k3[1]), tau, k4);
    float sixthDt = ACROBOT_DT / 6.0f;
    theta1 += sixthDt * (fmaf(2.0f, k3[0], fmaf(2.0f, k2[0], k1[0])) + k4[0]);
    theta2 += sixthDt * (fmaf(2.0f, k3[1], fmaf(2.0f, k2[1], k1[1])) + k4[1]);
    omega1 += sixthDt * (fmaf(2.0f, k3[2], fmaf(2.0f, k2[2], k1[2])) + k4[2]);
    omega2 += sixthDt * (fmaf(2.0f, k3[3], fmaf(2.0f, k2[3], k1[3])) + k4[3]);
    theta1 = acrobot_wrap(theta1);
    theta2 = acrobot_wrap(theta2);
    acrobotTerms t = acrobot_terms(theta1, theta2);
    float s1, c1;
    oak_sincos(acrobot_finite(theta1), &s1, &c1);
    if (fabsf(omega1) > ACROBOT_OMEGA1_MAX || fabsf(omega2) > ACROBOT_OMEGA2_MAX) return acrobot_restart(k, 0.0f, words, extra, obs, doneOut);
    if (-c1 - oak_cos(acrobot_finite(theta1 + theta2)) > 1.0f) return acrobot_restart(k, ACROBOT_GOAL_REWARD, words, extra, obs, doneOut);
    words[0] = floatWord(theta1); words[1] = floatWord(theta2); words[2] = floatWord(omega1); words[3] = floatWord(omega2);
    acrobot_remember(extra, t);
    acrobot_obs(k, s1, c1, t.sinTheta2, t.cosTheta2, omega1, omega2, obs);
    *doneOut = 0u;
    return 0.0f;
}

void acrobot_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    float s1, c1, s2, c2;
    oak_sincos(acrobot_finite(wordFloat(words[0])), &s1, &c1);
    oak_sincos(acrobot_finite(wordFloat(words[1])), &s2, &c2);
    acrobot_obs(&ACROBOT_KNOBS[instance], s1, c1, s2, c2, wordFloat(words[2]), wordFloat(words[3]), obs);
}

static float acrobot_energy(float theta1, float theta2, float omega1, float omega2) {
    float theta12 = acrobot_finite(theta1 + theta2), cosTheta2 = oak_cos(acrobot_finite(theta2));
    float inertia1 = 3.5f + cosTheta2, inertia2 = 1.25f + 0.5f * cosTheta2, inertia3 = 1.25f;
    float kinetic = 0.5f * (inertia1 * omega1 * omega1 + 2.0f * inertia2 * omega1 * omega2 + inertia3 * omega2 * omega2);
    return kinetic - 14.7f * oak_cos(acrobot_finite(theta1)) - 4.9f * oak_cos(theta12);
}

static float acrobot_angle(float sine, float cosine) {
    float ay = fabsf(sine), ax = fabsf(cosine);
    float r = fminf(ax, ay) / fmaxf(ax, ay), r2 = r * r;
    float a = r * (0.99997726f + r2 * (-0.33262347f + r2 * (0.19354346f + r2 * (-0.11643287f + r2 * (0.05265332f + r2 * -0.01172120f)))));
    a = ay > ax ? OAK_PI / 2.0f - a : a;
    a = cosine < 0.0f ? OAK_PI - a : a;
    return sine < 0.0f ? -a : a;
}

static float acrobot_recalled(uint32_t *seen, float angle) {
    float swept = acrobot_wrap(angle - wordFloat(*seen));
    *seen = floatWord(angle);
    return swept / ACROBOT_DT;
}

void acrobot_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step;
    float theta1, theta2, omega1, omega2;
    if (ACROBOT_KNOBS[instance].positionOnly) {
        theta1 = acrobot_angle(obs[1], obs[0]); theta2 = acrobot_angle(obs[3], obs[2]);
        omega1 = acrobot_recalled(&extra[ACROBOT_SEEN], theta1); omega2 = acrobot_recalled(&extra[ACROBOT_SEEN + 1], theta2);
    } else {
        theta1 = wordFloat(words[0]); theta2 = wordFloat(words[1]); omega1 = wordFloat(words[2]); omega2 = wordFloat(words[3]);
    }
    if (acrobot_energy(theta1, theta2, omega1, omega2) > ACROBOT_ENERGY_TOP) ctrl[0] = omega1 > 0.0f ? 0.0f : 2.0f;
    else ctrl[0] = omega2 < 0.0f ? 0.0f : 2.0f;
}
