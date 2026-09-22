#include "../env.h"

#define CARTPOLE_DT 0.02f
#define CARTPOLE_GRAVITY 9.8f
#define CARTPOLE_CART 1.0f
#define CARTPOLE_POLE 0.1f
#define CARTPOLE_HALF 0.5f
#define CARTPOLE_FORCE 10.0f
#define CARTPOLE_TRACK 2.4f
#define CARTPOLE_FALL -1.0f
#define CARTPOLE_ORACLE_EFFORT 1e-4
#define CARTPOLE_ORACLE_PUMP 15.0f
#define CARTPOLE_ORACLE_CENTER 1.0f
#define CARTPOLE_ORACLE_DAMP 2.0f
#define CARTPOLE_ORACLE_BAND 0.6f
#define CARTPOLE_ORACLE_ITERATIONS 200000u
#define CARTPOLE_ORACLE_TOLERANCE 1e-13

typedef struct { float slip; } cartpoleKnobs;

static const cartpoleKnobs CARTPOLE_KNOBS[2] = {
    { 0.0f },
    { 0.2f }
};


static void cartpole_obs(float x, float xDot, float theta, float thetaDot, float *obs) {
    float s, c;
    oak_sincos(theta, &s, &c);
    obs[0] = x / CARTPOLE_TRACK; obs[1] = xDot / 5.0f; obs[2] = c; obs[3] = s; obs[4] = thetaDot / 10.0f; obs[5] = 1.0f;
}

uint32_t cartpole_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 4u;
}

static void cartpole_gains(float *gains) {
    double total = (double)CARTPOLE_CART + (double)CARTPOLE_POLE, share = 4.0 / 3.0 - (double)CARTPOLE_POLE / total, dt = (double)CARTPOLE_DT;
    double swing = (double)CARTPOLE_GRAVITY / ((double)CARTPOLE_HALF * share), push = -1.0 / (total * (double)CARTPOLE_HALF * share);
    double a[16] = { 1.0, dt, 0.0, 0.0, 0.0, 1.0, -dt * (double)CARTPOLE_POLE * (double)CARTPOLE_HALF * swing / total, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, dt * swing, 1.0 };
    double b[4] = { 0.0, dt * (1.0 / total - (double)CARTPOLE_POLE * (double)CARTPOLE_HALF * push / total), 0.0, dt * push };
    double q[4] = { 0.01, 0.0, 0.5, 0.0 };
    double p[16], k[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (uint32_t i = 0u; i < 16u; i++) p[i] = (i % 5u == 0u) ? q[i / 5u] : 0.0;
    for (uint32_t iteration = 0u; iteration < CARTPOLE_ORACLE_ITERATIONS; iteration++) {
        double pb[4], bpb = 0.0, pa[16], next[16], closed[16], scale = 0.0, delta = 0.0;
        for (uint32_t i = 0u; i < 4u; i++) { pb[i] = 0.0; for (uint32_t j = 0u; j < 4u; j++) pb[i] += p[i * 4u + j] * b[j]; bpb += b[i] * pb[i]; }
        for (uint32_t i = 0u; i < 4u; i++) for (uint32_t j = 0u; j < 4u; j++) { pa[i * 4u + j] = 0.0; for (uint32_t m = 0u; m < 4u; m++) pa[i * 4u + j] += p[i * 4u + m] * a[m * 4u + j]; }
        for (uint32_t j = 0u; j < 4u; j++) { double bpa = 0.0; for (uint32_t i = 0u; i < 4u; i++) bpa += b[i] * pa[i * 4u + j]; k[j] = bpa / (CARTPOLE_ORACLE_EFFORT + bpb); }
        for (uint32_t i = 0u; i < 4u; i++) for (uint32_t j = 0u; j < 4u; j++) closed[i * 4u + j] = a[i * 4u + j] - b[i] * k[j];
        for (uint32_t i = 0u; i < 4u; i++) for (uint32_t j = 0u; j < 4u; j++) {
            double sum = (i == j) ? q[i] : 0.0;
            for (uint32_t m = 0u; m < 4u; m++) { double pm = 0.0; for (uint32_t n = 0u; n < 4u; n++) pm += p[m * 4u + n] * closed[n * 4u + j]; sum += a[m * 4u + i] * pm; }
            next[i * 4u + j] = sum;
        }
        for (uint32_t i = 0u; i < 16u; i++) { double change = fabs(next[i] - p[i]); if (change > delta) delta = change; if (fabs(p[i]) > scale) scale = fabs(p[i]); p[i] = next[i]; }
        if (delta < CARTPOLE_ORACLE_TOLERANCE * (1.0 + scale)) break;
    }
    for (uint32_t j = 0u; j < 4u; j++) gains[j] = (float)k[j];
}

void cartpole_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed;
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f); words[2] = floatWord(OAK_PI); words[3] = floatWord(0.0f);
    cartpole_gains((float *)extra);
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
    if (fabsf(x) > CARTPOLE_TRACK) { x = 0.0f; xDot = 0.0f; theta = OAK_PI; thetaDot = 0.0f; reward = CARTPOLE_FALL; *done = 1u; }
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
    (void)instance; (void)seed; (void)step; (void)obs;
    const float *gains = (const float *)extra;
    float x = wordFloat(words[0]), xDot = wordFloat(words[1]), theta = wordFloat(words[2]), thetaDot = wordFloat(words[3]);
    float s, c;
    oak_sincos(theta, &s, &c);
    float drive;
    if (c > CARTPOLE_ORACLE_BAND) drive = -(gains[0] * x + gains[1] * xDot + gains[2] * theta + gains[3] * thetaDot);
    else {
        float top = CARTPOLE_POLE * CARTPOLE_GRAVITY * CARTPOLE_HALF;
        float energy = 0.5f * (4.0f / 3.0f) * CARTPOLE_POLE * CARTPOLE_HALF * CARTPOLE_HALF * thetaDot * thetaDot + top * c;
        drive = CARTPOLE_ORACLE_PUMP * (energy - top) * thetaDot * c - CARTPOLE_ORACLE_CENTER * x - CARTPOLE_ORACLE_DAMP * xDot;
    }
    ctrl[0] = drive > 0.0f ? 2.0f : drive < 0.0f ? 0.0f : 1.0f;
}
