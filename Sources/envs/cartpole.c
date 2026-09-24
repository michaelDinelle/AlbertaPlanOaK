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

enum { CARTPOLE_GAINS, CARTPOLE_SIN = CARTPOLE_GAINS + 4, CARTPOLE_COS, CARTPOLE_READY, CARTPOLE_THETA, CARTPOLE_BATCH, CARTPOLE_SLIPS, CARTPOLE_EXTRA = CARTPOLE_SLIPS + ENV_BATCH };

static void cartpole_obs(float x, float xDot, float s, float c, float thetaDot, float *obs) {
    obs[0] = x / CARTPOLE_TRACK; obs[1] = xDot / 5.0f; obs[2] = c; obs[3] = s; obs[4] = thetaDot / 10.0f; obs[5] = 1.0f;
}

uint32_t cartpole_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return CARTPOLE_EXTRA;
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

static void cartpole_draw(const cartpoleKnobs *k, uint32_t seed, uint32_t step, uint32_t *extra) {
    lane4 zero = 0u;
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero, zero, zero);
    lane4 slipped = (lane4)(r.x < u01Bound(k->slip)), drawn = (slipped & (r.y % 3u)) | (~slipped & 3u);
    __builtin_memcpy(extra + CARTPOLE_SLIPS, &drawn, sizeof drawn);
}

static void cartpole_settle(float theta, uint32_t *extra) {
    float s, c;
    oak_sincos(theta, &s, &c);
    extra[CARTPOLE_THETA] = floatWord(theta); extra[CARTPOLE_SIN] = floatWord(s); extra[CARTPOLE_COS] = floatWord(c);
}

void cartpole_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed;
    words[0] = floatWord(0.0f); words[1] = floatWord(0.0f); words[2] = floatWord(OAK_PI); words[3] = floatWord(0.0f);
    cartpole_settle(OAK_PI, extra);
    batchReset(extra + CARTPOLE_BATCH);
}

static void cartpole_integrate(float force, float s, float c, float *x, float *xDot, float *theta, float *thetaDot) {
    float total = CARTPOLE_CART + CARTPOLE_POLE;
    float temp = (force + CARTPOLE_POLE * CARTPOLE_HALF * *thetaDot * *thetaDot * s) / total;
    float thetaAcc = (CARTPOLE_GRAVITY * s - c * temp) / (CARTPOLE_HALF * (4.0f / 3.0f) - CARTPOLE_POLE * c * c / (total / CARTPOLE_HALF));
    float xAcc = temp - CARTPOLE_POLE * CARTPOLE_HALF * thetaAcc * c / total;
    *x += CARTPOLE_DT * *xDot;
    *xDot += CARTPOLE_DT * xAcc;
    *theta += CARTPOLE_DT * *thetaDot;
    *thetaDot += CARTPOLE_DT * thetaAcc;
}

static float cartpole_angle(float angle, float *s, float *c) {
    int below = angle < -OAK_PI, above = angle >= OAK_PI;
    int minus3 = angle < -0x1.2d97c8p+1f, minus1 = angle < -0x1.921fb6p-1f, plus1 = angle < 0x1.921fb6p-1f, plus3 = angle < 0x1.2d97c8p+1f;
    float kept = above ? angle - 2.0f * OAK_PI : angle, theta = below ? angle + 2.0f * OAK_PI : kept;
    float lowEnd = below ? 2.0f : -2.0f, low = minus3 ? lowEnd : -1.0f;
    float highMid = plus1 ? 0.0f : 1.0f, highEnd = above ? -2.0f : 2.0f, high = plus3 ? highMid : highEnd;
    oak_sincosQuarter(theta, minus1 ? low : high, s, c);
    return theta;
}

static float cartpole_finish(float x, float xDot, float theta, float thetaDot, float s, float c, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    *done = 0u;
    float reward = c - 0.01f * x * x;
    if (__builtin_expect(fabsf(x) > CARTPOLE_TRACK, 0)) { x = 0.0f; xDot = 0.0f; theta = OAK_PI; thetaDot = 0.0f; reward = CARTPOLE_FALL; *done = 1u; oak_sincos(theta, &s, &c); }
    words[0] = floatWord(x); words[1] = floatWord(xDot); words[2] = floatWord(theta); words[3] = floatWord(thetaDot);
    extra[CARTPOLE_THETA] = words[2]; extra[CARTPOLE_SIN] = floatWord(s); extra[CARTPOLE_COS] = floatWord(c);
    cartpole_obs(x, xDot, s, c, thetaDot, obs);
    return reward;
}

static __attribute__((noinline)) float cartpole_far(float x, float xDot, float angle, float thetaDot, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    float theta = wrapPi(angle), s, c;
    oak_sincos(theta, &s, &c);
    return cartpole_finish(x, xDot, theta, thetaDot, s, c, words, extra, obs, done);
}

static inline __attribute__((always_inline)) float cartpole_advance(const cartpoleKnobs *k, uint32_t slot, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    static const float FORCE[3] = { -CARTPOLE_FORCE, 0.0f, CARTPOLE_FORCE };
    uint32_t action = (uint32_t)ctrl[0] % 3u;
    if (k->slip > 0.0f && extra[CARTPOLE_SLIPS + slot] < 3u) action = extra[CARTPOLE_SLIPS + slot];
    if (extra[CARTPOLE_THETA] != words[2]) __builtin_trap();
    float x = wordFloat(words[0]), xDot = wordFloat(words[1]), theta = wordFloat(words[2]), thetaDot = wordFloat(words[3]);
    cartpole_integrate(FORCE[action], wordFloat(extra[CARTPOLE_SIN]), wordFloat(extra[CARTPOLE_COS]), &x, &xDot, &theta, &thetaDot);
    if (__builtin_expect(!(fabsf(theta) < 0x1.f6a7a4p+1f), 0)) return cartpole_far(x, xDot, theta, thetaDot, words, extra, obs, done);
    float s, c;
    theta = cartpole_angle(theta, &s, &c);
    return cartpole_finish(x, xDot, theta, thetaDot, s, c, words, extra, obs, done);
}

static __attribute__((noinline)) float cartpole_refill(const cartpoleKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    cartpole_draw(k, seed, step, extra);
    return cartpole_advance(k, 0u, ctrl, words, extra, obs, done);
}

float cartpole_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const cartpoleKnobs *k = &CARTPOLE_KNOBS[instance];
    uint32_t slot = 0u;
    if (k->slip > 0.0f) {
        slot = batchSlot(extra + CARTPOLE_BATCH, step);
        if (slot == ENV_BATCH) return cartpole_refill(k, seed, step, ctrl, words, extra, obs, done);
    }
    return cartpole_advance(k, slot, ctrl, words, extra, obs, done);
}

void cartpole_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step;
    if (extra[CARTPOLE_THETA] != words[2]) __builtin_trap();
    cartpole_obs(wordFloat(words[0]), wordFloat(words[1]), wordFloat(extra[CARTPOLE_SIN]), wordFloat(extra[CARTPOLE_COS]), wordFloat(words[3]), obs);
}

void cartpole_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    float *gains = (float *)(extra + CARTPOLE_GAINS);
    if (!extra[CARTPOLE_READY]) { cartpole_gains(gains); extra[CARTPOLE_READY] = 1u; }
    if (extra[CARTPOLE_THETA] != words[2]) __builtin_trap();
    float x = wordFloat(words[0]), xDot = wordFloat(words[1]), theta = wordFloat(words[2]), thetaDot = wordFloat(words[3]);
    float c = wordFloat(extra[CARTPOLE_COS]);
    float drive;
    if (c > CARTPOLE_ORACLE_BAND) drive = -(gains[0] * x + gains[1] * xDot + gains[2] * theta + gains[3] * thetaDot);
    else {
        float top = CARTPOLE_POLE * CARTPOLE_GRAVITY * CARTPOLE_HALF;
        float energy = 0.5f * (4.0f / 3.0f) * CARTPOLE_POLE * CARTPOLE_HALF * CARTPOLE_HALF * thetaDot * thetaDot + top * c;
        drive = CARTPOLE_ORACLE_PUMP * (energy - top) * thetaDot * c - CARTPOLE_ORACLE_CENTER * x - CARTPOLE_ORACLE_DAMP * xDot;
    }
    ctrl[0] = drive > 0.0f ? 2.0f : drive < 0.0f ? 0.0f : 1.0f;
}
