#include "../env.h"

#define CARTPOLE_DT 0.02f
#define CARTPOLE_GRAVITY 9.8f
#define CARTPOLE_CART 1.0f
#define CARTPOLE_POLE 0.1f
#define CARTPOLE_HALF 0.5f
#define CARTPOLE_FORCE 10.0f

typedef struct { float slip; } cartpoleKnobs;

static const cartpoleKnobs CARTPOLE_KNOBS[2] = {
    { 0.0f },
    { 0.2f }
};


static void cartpole_obs(float x, float xDot, float theta, float thetaDot, float *obs) {
    float s, c;
    oak_sincos(theta, &s, &c);
    obs[0] = x / 2.4f; obs[1] = xDot / 5.0f; obs[2] = c; obs[3] = s; obs[4] = thetaDot / 10.0f; obs[5] = 1.0f;
}

void cartpole_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed; (void)extra;
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f); words[2] = floatWord(OAK_PI); words[3] = floatWord(0.0f);
}

static void cartpole_integrate(uint32_t action, float *x, float *xDot, float *theta, float *thetaDot) {
    float force = ((float)action - 1.0f) * CARTPOLE_FORCE;
    float s, c;
    oak_sincos(*theta, &s, &c);
    float total = CARTPOLE_CART + CARTPOLE_POLE;
    float temp = (force + CARTPOLE_POLE * CARTPOLE_HALF * *thetaDot * *thetaDot * s) / total;
    float thetaAcc = (CARTPOLE_GRAVITY * s - c * temp) / (CARTPOLE_HALF * (4.0f / 3.0f - CARTPOLE_POLE * c * c / total));
    float xAcc = temp - CARTPOLE_POLE * CARTPOLE_HALF * thetaAcc * c / total;
    *x += CARTPOLE_DT * *xDot;
    *xDot += CARTPOLE_DT * xAcc;
    *theta = wrapPi(*theta + CARTPOLE_DT * *thetaDot);
    *thetaDot += CARTPOLE_DT * thetaAcc;
}

float cartpole_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const cartpoleKnobs *k = &CARTPOLE_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0] % 3u;
    if (k->slip > 0.0f) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
        if (u01(r.x) < k->slip) action = r.y % 3u;
    }
    float x = wordFloat(words[0]), xDot = wordFloat(words[1]), theta = wordFloat(words[2]), thetaDot = wordFloat(words[3]);
    cartpole_integrate(action, &x, &xDot, &theta, &thetaDot);
    *done = 0u;
    float reward = oak_cos(theta) - 0.01f * x * x;
    if (fabsf(x) > 2.4f) { x = 0.0f; xDot = 0.0f; theta = OAK_PI; thetaDot = 0.0f; reward = -1.0f; *done = 1u; }
    words[0] = floatWord(x); words[1] = floatWord(xDot); words[2] = floatWord(theta); words[3] = floatWord(thetaDot);
    cartpole_obs(x, xDot, theta, thetaDot, obs);
    return reward;
}

void cartpole_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    cartpole_obs(wordFloat(words[0]), wordFloat(words[1]), wordFloat(words[2]), wordFloat(words[3]), obs);
}

void cartpole_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t best = 1u;
    float bestCost = FLT_MAX;
    for (uint32_t first = 0u; first < 3u; first++) {
        float x = wordFloat(words[0]), xDot = wordFloat(words[1]), theta = wordFloat(words[2]), thetaDot = wordFloat(words[3]);
        float cost = 0.0f;
        for (uint32_t t = 0u; t < 30u; t++) {
            cartpole_integrate(t < 6u ? first : 1u, &x, &xDot, &theta, &thetaDot);
            cost += 1.0f - oak_cos(theta) + 0.01f * x * x + (fabsf(x) > 2.4f ? 100.0f : 0.0f);
        }
        if (cost < bestCost) { bestCost = cost; best = first; }
    }
    ctrl[0] = (float)best;
}
