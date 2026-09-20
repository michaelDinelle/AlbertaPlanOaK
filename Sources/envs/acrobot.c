#include "../env.h"

#define ACROBOT_DT 0.2f
#define ACROBOT_OMEGA1_MAX 12.566371f
#define ACROBOT_OMEGA2_MAX 28.274334f

typedef struct { float slip; uint32_t positionOnly; } acrobotKnobs;

OAK_CONSTANT acrobotKnobs ACROBOT_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};


static void acrobot_obs(constant const acrobotKnobs *k, float theta1, float theta2, float omega1, float omega2, device float *obs) {
    float s1, c1, s2, c2;
    oak_sincos(theta1, &s1, &c1);
    oak_sincos(theta2, &s2, &c2);
    obs[0] = c1; obs[1] = s1; obs[2] = c2; obs[3] = s2;
    obs[4] = k->positionOnly ? 0.0f : omega1; obs[5] = k->positionOnly ? 0.0f : omega2; obs[6] = 1.0f;
}

static void acrobot_dynamics(float theta1, float theta2, float omega1, float omega2, float tau, thread float *derivative) {
    float sinTheta2, cosTheta2;
    oak_sincos(theta2, &sinTheta2, &cosTheta2);
    float inertia1 = 3.5f + cosTheta2;
    float inertia2 = 1.25f + 0.5f * cosTheta2;
    float gravity2 = 4.9f * oak_cos(theta1 + theta2 - OAK_PI / 2.0f);
    float gravity1 = -0.5f * omega2 * omega2 * sinTheta2 - omega2 * omega1 * sinTheta2 + 14.7f * oak_cos(theta1 - OAK_PI / 2.0f) + gravity2;
    float alpha2 = (tau + inertia2 / inertia1 * gravity1 - 0.5f * omega1 * omega1 * sinTheta2 - gravity2) / (inertia2 - inertia2 * inertia2 / inertia1);
    float alpha1 = -(inertia2 * alpha2 + gravity1) / inertia1;
    derivative[0] = omega1; derivative[1] = omega2; derivative[2] = alpha1; derivative[3] = alpha2;
}

void acrobot_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)extra; (void)instance; (void)seed; (void)words; }

float acrobot_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *doneOut, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const acrobotKnobs *k = &ACROBOT_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0];
    if (k->slip > 0.0f) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
        if (u01(r.x) < k->slip) action = r.y % 3u;
    }
    float theta1 = wordFloat(words[0]), theta2 = wordFloat(words[1]);
    float omega1 = wordFloat(words[2]), omega2 = wordFloat(words[3]);
    float tau = (float)((int)(action % 3u)) - 1.0f;
    float k1[4], k2[4], k3[4], k4[4];
    acrobot_dynamics(theta1, theta2, omega1, omega2, tau, k1);
    float halfDt = ACROBOT_DT / 2.0f;
    acrobot_dynamics(theta1 + halfDt * k1[0], theta2 + halfDt * k1[1], omega1 + halfDt * k1[2], omega2 + halfDt * k1[3], tau, k2);
    acrobot_dynamics(theta1 + halfDt * k2[0], theta2 + halfDt * k2[1], omega1 + halfDt * k2[2], omega2 + halfDt * k2[3], tau, k3);
    acrobot_dynamics(theta1 + ACROBOT_DT * k3[0], theta2 + ACROBOT_DT * k3[1], omega1 + ACROBOT_DT * k3[2], omega2 + ACROBOT_DT * k3[3], tau, k4);
    float sixthDt = ACROBOT_DT / 6.0f;
    theta1 += sixthDt * (k1[0] + 2.0f * k2[0] + 2.0f * k3[0] + k4[0]);
    theta2 += sixthDt * (k1[1] + 2.0f * k2[1] + 2.0f * k3[1] + k4[1]);
    omega1 += sixthDt * (k1[2] + 2.0f * k2[2] + 2.0f * k3[2] + k4[2]);
    omega2 += sixthDt * (k1[3] + 2.0f * k2[3] + 2.0f * k3[3] + k4[3]);
    omega1 = oak_clamp(omega1, -ACROBOT_OMEGA1_MAX, ACROBOT_OMEGA1_MAX);
    omega2 = oak_clamp(omega2, -ACROBOT_OMEGA2_MAX, ACROBOT_OMEGA2_MAX);
    theta1 = wrapPi(theta1);
    theta2 = wrapPi(theta2);
    float reward = 0.0f;
    uint32_t done = 0u;
    if (-oak_cos(theta1) - oak_cos(theta1 + theta2) > 1.0f) {
        reward = 1.0f;
        done = 1u;
        theta1 = 0.0f; theta2 = 0.0f; omega1 = 0.0f; omega2 = 0.0f;
    }
    words[0] = floatWord(theta1); words[1] = floatWord(theta2); words[2] = floatWord(omega1); words[3] = floatWord(omega2);
    acrobot_obs(k, theta1, theta2, omega1, omega2, obs);
    *doneOut = done;
    return reward;
}

void acrobot_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    acrobot_obs(&ACROBOT_KNOBS[instance], wordFloat(words[0]), wordFloat(words[1]), wordFloat(words[2]), wordFloat(words[3]), obs);
}

void acrobot_pose(uint32_t instance, device const uint32_t *words, thread const float *ctrl, thread float *pose) {
    (void)instance;
    poseZero(pose);
    for (int i = 0; i < 4; i++) pose[i] = wordFloat(words[i]);
    pose[4] = ctrl[0];
}

void acrobot_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance;
    (void)seed; (void)step; (void)extra; (void)obs;
    float omega2 = wordFloat(words[3]);
    ctrl[0] = omega2 < 0.0f ? 0.0f : 2.0f;
}
