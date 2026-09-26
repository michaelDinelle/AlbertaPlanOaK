#include "../env.h"

typedef struct { float slip; uint32_t positionOnly; } pendulumKnobs;

static const pendulumKnobs PENDULUM_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};

enum { PENDULUM_THETA, PENDULUM_GRAVITY, PENDULUM_BATCH, PENDULUM_SLIPS, PENDULUM_EXTRA = PENDULUM_SLIPS + ENV_BATCH };
static const uint32_t PENDULUM_KEPT = 0xFFFFFFFFu;

uint32_t pendulum_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return PENDULUM_EXTRA;
}

static void pendulum_draw(const pendulumKnobs *k, uint32_t seed, uint32_t step, uint32_t *extra) {
    lane4 zero = 0u;
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, zero, zero, zero);
    lane4 slipped = (lane4)(r.x < u01Bound(k->slip));
    lane4 command = (lane4)(-1.0f + (1.0f - -1.0f) * (__builtin_convertvector(r.y >> 8u, float4) * 0x1p-24f));
    lane4 drawn = (slipped & command) | (~slipped & PENDULUM_KEPT);
    __builtin_memcpy(extra + PENDULUM_SLIPS, &drawn, sizeof drawn);
}

static float pendulum_wrapped(float theta) { return theta >= OAK_PI ? theta - 2.0f * OAK_PI : theta; }

static float pendulum_angle(float angle, float *s, float *c, float *gravity) {
    int below = angle < -OAK_PI, above = angle >= OAK_PI;
    int minus3 = angle < -0x1.2d97c8p+1f, minus1 = angle < -0x1.921fb6p-1f, plus1 = angle < 0x1.921fb6p-1f, plus3 = angle < 0x1.2d97c8p+1f;
    int swingMinus1 = angle < -0x1.921fbcp-1f, swingPlus1 = angle < 0x1.921fb4p-1f;
    float kept = above ? angle - 2.0f * OAK_PI : angle, theta = below ? angle + 2.0f * OAK_PI : kept;
    float lowEnd = below ? 2.0f : -2.0f, low = minus3 ? lowEnd : -1.0f;
    float highMid = plus1 ? 0.0f : 1.0f, highEnd = above ? -2.0f : 2.0f, high = plus3 ? highMid : highEnd;
    oak_sincosQuarter(theta, minus1 ? low : high, s, c);
    float swingLowEnd = below ? 4.0f : 0.0f, swingLow = minus3 ? swingLowEnd : 1.0f;
    float swingHighMid = swingPlus1 ? 2.0f : 3.0f, swingHighEnd = above ? 0.0f : 4.0f, swingHigh = plus3 ? swingHighMid : swingHighEnd;
    float gs, gc;
    oak_sincosQuarter(theta + OAK_PI, swingMinus1 ? swingLow : swingHigh, &gs, &gc);
    *gravity = 15.0f * gs;
    return theta;
}

static void pendulum_obs(const pendulumKnobs *k, float s, float c, float thetaDot, float *obs) {
    obs[0] = c; obs[1] = s; obs[2] = k->positionOnly ? 0.0f : thetaDot; obs[3] = 1.0f;
}

void pendulum_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)seed;
    float theta = OAK_PI;
    words[0] = floatWord(theta);
    extra[PENDULUM_THETA] = words[0];
    extra[PENDULUM_GRAVITY] = floatWord(15.0f * oak_sin(theta + OAK_PI));
    batchReset(extra + PENDULUM_BATCH);
}

static inline __attribute__((always_inline)) float pendulum_advance(const pendulumKnobs *k, uint32_t slot, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    float command = ctrl[0];
    if (k->slip > 0.0f && extra[PENDULUM_SLIPS + slot] != PENDULUM_KEPT) command = wordFloat(extra[PENDULUM_SLIPS + slot]);
    *done = 0u;
    if (extra[PENDULUM_THETA] != words[0]) __builtin_trap();
    float theta = wordFloat(words[0]), thetaDot = wordFloat(words[1]);
    float torque = oak_clamp(2.0f * command, -2.0f, 2.0f);
    float thetaWrapped = pendulum_wrapped(theta);
    float cost = thetaWrapped * thetaWrapped + 0.1f * thetaDot * thetaDot + 0.001f * torque * torque;
    float nextThetaDot = thetaDot + (wordFloat(extra[PENDULUM_GRAVITY]) + 3.0f * torque) * 0.05f;
    theta = theta + nextThetaDot * 0.05f;
    nextThetaDot = oak_clamp(nextThetaDot, -8.0f, 8.0f);
    float s, c, gravity;
    theta = pendulum_angle(theta, &s, &c, &gravity);
    words[0] = floatWord(theta); words[1] = floatWord(nextThetaDot);
    extra[PENDULUM_THETA] = words[0]; extra[PENDULUM_GRAVITY] = floatWord(gravity);
    pendulum_obs(k, s, c, nextThetaDot, obs);
    return -cost;
}

static __attribute__((noinline)) float pendulum_refill(const pendulumKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    pendulum_draw(k, seed, step, extra);
    return pendulum_advance(k, 0u, ctrl, words, extra, obs, done);
}

float pendulum_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const pendulumKnobs *k = &PENDULUM_KNOBS[instance];
    uint32_t slot = 0u;
    if (k->slip > 0.0f) {
        slot = batchSlot(extra + PENDULUM_BATCH, step);
        if (slot == ENV_BATCH) return pendulum_refill(k, seed, step, ctrl, words, extra, obs, done);
    }
    return pendulum_advance(k, slot, ctrl, words, extra, obs, done);
}

void pendulum_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    float s, c;
    oak_sincos(wordFloat(words[0]), &s, &c);
    pendulum_obs(&PENDULUM_KNOBS[instance], s, c, wordFloat(words[1]), obs);
}

void pendulum_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)seed; (void)step; (void)extra; (void)obs;
    float theta = pendulum_wrapped(wordFloat(words[0])), thetaDot = wordFloat(words[1]);
    ctrl[0] = oak_clamp(-(2.0f * theta + 1.0f * thetaDot) * 0.5f, -1.0f, 1.0f);
}
