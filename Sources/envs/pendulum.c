#include "../env.h"

typedef struct { float slip; uint32_t positionOnly; } pendulumKnobs;

static const pendulumKnobs PENDULUM_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};


static void pendulum_obs(const pendulumKnobs *k, float theta, float thetaDot, float *obs) {
    float s, c;
    oak_sincos(theta, &s, &c);
    obs[0] = c; obs[1] = s; obs[2] = k->positionOnly ? 0.0f : thetaDot; obs[3] = 1.0f;
}

void pendulum_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    (void)instance;
    (void)seed;
    words[0] = floatWord(OAK_PI);
}

float pendulum_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const pendulumKnobs *k = &PENDULUM_KNOBS[instance];
    *done = 0u;
    float theta = wordFloat(words[0]), thetaDot = wordFloat(words[1]);
    float command = ctrl[0];
    if (k->slip > 0.0f) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
        if (u01(r.x) < k->slip) command = uniformRange(r.y, -1.0f, 1.0f);
    }
    float torque = oak_clamp(2.0f * command, -2.0f, 2.0f);
    float thetaWrapped = wrapPi(theta);
    float cost = thetaWrapped * thetaWrapped + 0.1f * thetaDot * thetaDot + 0.001f * torque * torque;
    float nextThetaDot = thetaDot + (15.0f * oak_sin(theta + OAK_PI) + 3.0f * torque) * 0.05f;
    theta = theta + nextThetaDot * 0.05f;
    nextThetaDot = oak_clamp(nextThetaDot, -8.0f, 8.0f);
    theta = wrapPi(theta);
    words[0] = floatWord(theta); words[1] = floatWord(nextThetaDot);
    pendulum_obs(k, theta, nextThetaDot, obs);
    return -cost;
}

void pendulum_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    pendulum_obs(&PENDULUM_KNOBS[instance], wordFloat(words[0]), wordFloat(words[1]), obs);
}

void pendulum_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)seed; (void)step; (void)extra; (void)obs;
    float theta = wrapPi(wordFloat(words[0])), thetaDot = wordFloat(words[1]);
    ctrl[0] = oak_clamp(-(2.0f * theta + 1.0f * thetaDot) * 0.5f, -1.0f, 1.0f);
}
