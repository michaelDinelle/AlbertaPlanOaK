#include "../env.h"

#define QUADROTOR_MASS 0.486f
#define QUADROTOR_ARM 0.25f
#define QUADROTOR_INERTIA 0.00383f
#define QUADROTOR_GRAVITY 9.81f
#define QUADROTOR_SUBSTEPS 2
#define QUADROTOR_SUBSTEP 0.01f
#define QUADROTOR_WALL 4.0f
#define QUADROTOR_CEILING 5.0f
#define QUADROTOR_LAUNCH 1.0f
#define QUADROTOR_REACH 3.0f
#define QUADROTOR_LOW 1.0f
#define QUADROTOR_HIGH 4.0f
#define QUADROTOR_CRASH -500.0f
#define QUADROTOR_POSITION_SCALE 0.25f
#define QUADROTOR_SPIN_SCALE 0.0625f
#define QUADROTOR_HOVER (QUADROTOR_MASS * QUADROTOR_GRAVITY / 2.0f)
#define QUADROTOR_LIFT (QUADROTOR_HOVER / QUADROTOR_MASS)
#define QUADROTOR_TORQUE (QUADROTOR_ARM * QUADROTOR_HOVER / QUADROTOR_INERTIA)
#define QUADROTOR_ORACLE_POSITION 1.0
#define QUADROTOR_ORACLE_ATTITUDE 0.1
#define QUADROTOR_ORACLE_VELOCITY 0.03
#define QUADROTOR_ORACLE_SPIN 0.001
#define QUADROTOR_ORACLE_EFFORT 0.5
#define QUADROTOR_ORACLE_LEASH 2.0f
#define QUADROTOR_ORACLE_BANK 0.5f
#define QUADROTOR_ORACLE_BRAKE 0.9f
#define QUADROTOR_ORACLE_UPRIGHT 0.5f
#define QUADROTOR_ORACLE_ITERATIONS 100000u
#define QUADROTOR_ORACLE_TOLERANCE 1e-12

typedef struct { float jumpPeriod, gustPeriod, gustX, gustY; } quadrotorKnobs;

static const quadrotorKnobs QUADROTOR_KNOBS[2] = {
    { 500.0f, 0.0f, 0.0f, 0.0f },
    { 500.0f, 1000.0f, 4.905f, 2.4525f }
};

enum { QUADROTOR_X, QUADROTOR_Y, QUADROTOR_ANGLE, QUADROTOR_VX, QUADROTOR_VY, QUADROTOR_SPIN, QUADROTOR_GOAL_X, QUADROTOR_GOAL_Y, QUADROTOR_WIND_X, QUADROTOR_WIND_Y };
enum { QUADROTOR_SIN, QUADROTOR_COS, QUADROTOR_TAG, QUADROTOR_JUMP, QUADROTOR_GUST, QUADROTOR_EXTRA };
enum { QUADROTOR_STATE = 6, QUADROTOR_INPUTS = 2, QUADROTOR_GAINS = QUADROTOR_STATE * QUADROTOR_INPUTS };

uint32_t quadrotor_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return QUADROTOR_EXTRA;
}

static void quadrotor_model(double *a, double *b) {
    double h = (double)QUADROTOR_SUBSTEP, lift = (double)QUADROTOR_LIFT, torque = (double)QUADROTOR_TORQUE, fall = 2.0 * lift;
    double m[QUADROTOR_STATE * QUADROTOR_STATE] = { 0.0 }, n[QUADROTOR_GAINS] = { 0.0 };
    for (uint32_t i = 0u; i < QUADROTOR_STATE; i++) m[i * (QUADROTOR_STATE + 1u)] = 1.0;
    m[0 * QUADROTOR_STATE + 2] = -h * h * fall; m[0 * QUADROTOR_STATE + 3] = h; m[3 * QUADROTOR_STATE + 2] = -h * fall;
    m[1 * QUADROTOR_STATE + 4] = h; m[2 * QUADROTOR_STATE + 5] = h;
    n[1 * QUADROTOR_INPUTS + 0] = h * h * lift; n[1 * QUADROTOR_INPUTS + 1] = h * h * lift;
    n[4 * QUADROTOR_INPUTS + 0] = h * lift; n[4 * QUADROTOR_INPUTS + 1] = h * lift;
    n[2 * QUADROTOR_INPUTS + 0] = 0.5 * h * h * torque; n[2 * QUADROTOR_INPUTS + 1] = -0.5 * h * h * torque;
    n[5 * QUADROTOR_INPUTS + 0] = h * torque; n[5 * QUADROTOR_INPUTS + 1] = -h * torque;
    for (uint32_t i = 0u; i < QUADROTOR_STATE * QUADROTOR_STATE; i++) a[i] = i % (QUADROTOR_STATE + 1u) == 0u ? 1.0 : 0.0;
    for (uint32_t i = 0u; i < QUADROTOR_GAINS; i++) b[i] = 0.0;
    for (uint32_t substep = 0u; substep < QUADROTOR_SUBSTEPS; substep++) {
        double nextA[QUADROTOR_STATE * QUADROTOR_STATE], nextB[QUADROTOR_GAINS];
        for (uint32_t i = 0u; i < QUADROTOR_STATE; i++) {
            for (uint32_t j = 0u; j < QUADROTOR_STATE; j++) {
                double sum = 0.0;
                for (uint32_t k = 0u; k < QUADROTOR_STATE; k++) sum += m[i * QUADROTOR_STATE + k] * a[k * QUADROTOR_STATE + j];
                nextA[i * QUADROTOR_STATE + j] = sum;
            }
            for (uint32_t j = 0u; j < QUADROTOR_INPUTS; j++) {
                double sum = n[i * QUADROTOR_INPUTS + j];
                for (uint32_t k = 0u; k < QUADROTOR_STATE; k++) sum += m[i * QUADROTOR_STATE + k] * b[k * QUADROTOR_INPUTS + j];
                nextB[i * QUADROTOR_INPUTS + j] = sum;
            }
        }
        for (uint32_t i = 0u; i < QUADROTOR_STATE * QUADROTOR_STATE; i++) a[i] = nextA[i];
        for (uint32_t i = 0u; i < QUADROTOR_GAINS; i++) b[i] = nextB[i];
    }
}

uint64_t quadrotor_shared(float *table) {
    if (!table) return QUADROTOR_GAINS;
    double a[QUADROTOR_STATE * QUADROTOR_STATE], b[QUADROTOR_GAINS], p[QUADROTOR_STATE * QUADROTOR_STATE] = { 0.0 }, k[QUADROTOR_GAINS] = { 0.0 };
    double q[QUADROTOR_STATE] = { QUADROTOR_ORACLE_POSITION, QUADROTOR_ORACLE_POSITION, QUADROTOR_ORACLE_ATTITUDE, QUADROTOR_ORACLE_VELOCITY, QUADROTOR_ORACLE_VELOCITY, QUADROTOR_ORACLE_SPIN };
    quadrotor_model(a, b);
    for (uint32_t i = 0u; i < QUADROTOR_STATE; i++) p[i * (QUADROTOR_STATE + 1u)] = q[i];
    for (uint32_t iteration = 0u; iteration < QUADROTOR_ORACLE_ITERATIONS; iteration++) {
        double pa[QUADROTOR_STATE * QUADROTOR_STATE], pb[QUADROTOR_GAINS], bpb[QUADROTOR_INPUTS * QUADROTOR_INPUTS], bpa[QUADROTOR_GAINS], next[QUADROTOR_STATE * QUADROTOR_STATE];
        double delta = 0.0, scale = 0.0;
        for (uint32_t i = 0u; i < QUADROTOR_STATE; i++) {
            for (uint32_t j = 0u; j < QUADROTOR_STATE; j++) {
                double sum = 0.0;
                for (uint32_t m = 0u; m < QUADROTOR_STATE; m++) sum += p[i * QUADROTOR_STATE + m] * a[m * QUADROTOR_STATE + j];
                pa[i * QUADROTOR_STATE + j] = sum;
            }
            for (uint32_t j = 0u; j < QUADROTOR_INPUTS; j++) {
                double sum = 0.0;
                for (uint32_t m = 0u; m < QUADROTOR_STATE; m++) sum += p[i * QUADROTOR_STATE + m] * b[m * QUADROTOR_INPUTS + j];
                pb[i * QUADROTOR_INPUTS + j] = sum;
            }
        }
        for (uint32_t i = 0u; i < QUADROTOR_INPUTS; i++) {
            for (uint32_t j = 0u; j < QUADROTOR_INPUTS; j++) {
                double sum = i == j ? QUADROTOR_ORACLE_EFFORT : 0.0;
                for (uint32_t m = 0u; m < QUADROTOR_STATE; m++) sum += b[m * QUADROTOR_INPUTS + i] * pb[m * QUADROTOR_INPUTS + j];
                bpb[i * QUADROTOR_INPUTS + j] = sum;
            }
            for (uint32_t j = 0u; j < QUADROTOR_STATE; j++) {
                double sum = 0.0;
                for (uint32_t m = 0u; m < QUADROTOR_STATE; m++) sum += b[m * QUADROTOR_INPUTS + i] * pa[m * QUADROTOR_STATE + j];
                bpa[i * QUADROTOR_STATE + j] = sum;
            }
        }
        double det = bpb[0] * bpb[3] - bpb[1] * bpb[2];
        for (uint32_t j = 0u; j < QUADROTOR_STATE; j++) {
            k[j] = (bpb[3] * bpa[j] - bpb[1] * bpa[QUADROTOR_STATE + j]) / det;
            k[QUADROTOR_STATE + j] = (bpb[0] * bpa[QUADROTOR_STATE + j] - bpb[2] * bpa[j]) / det;
        }
        for (uint32_t i = 0u; i < QUADROTOR_STATE; i++)
            for (uint32_t j = 0u; j < QUADROTOR_STATE; j++) {
                double sum = i == j ? q[i] : 0.0;
                for (uint32_t m = 0u; m < QUADROTOR_STATE; m++) sum += a[m * QUADROTOR_STATE + i] * pa[m * QUADROTOR_STATE + j];
                for (uint32_t m = 0u; m < QUADROTOR_INPUTS; m++) sum -= bpa[m * QUADROTOR_STATE + i] * k[m * QUADROTOR_STATE + j];
                next[i * QUADROTOR_STATE + j] = sum;
            }
        for (uint32_t i = 0u; i < QUADROTOR_STATE * QUADROTOR_STATE; i++) {
            double change = fabs(next[i] - p[i]);
            if (change > delta) delta = change;
            if (fabs(p[i]) > scale) scale = fabs(p[i]);
            p[i] = next[i];
        }
        if (delta < QUADROTOR_ORACLE_TOLERANCE * (1.0 + scale)) break;
    }
    for (uint32_t i = 0u; i < QUADROTOR_GAINS; i++) table[i] = (float)k[i];
    return QUADROTOR_GAINS;
}

static inline float quadrotor_finite(float x) { __builtin_assume(__builtin_isfinite(x)); return x; }

static inline __attribute__((always_inline)) float quadrotor_wrap(float angle) {
    if (__builtin_expect(fabsf(angle) >= OAK_PI, 0)) return wrapPi(angle);
    return angle;
}

static inline __attribute__((always_inline)) uint32_t quadrotor_touches(float x, float y, float s, float c) {
    float rise = QUADROTOR_ARM * fabsf(s), reach = fmaf(QUADROTOR_ARM, fabsf(c), fabsf(x));
    return (y - rise <= 0.0f) | (y + rise >= QUADROTOR_CEILING) | (reach >= QUADROTOR_WALL);
}

static uint32_t quadrotor_interval(uint32_t draw, float period) { return 1u + (uint32_t)(-period * oak_log(u01open(draw))); }

static void quadrotor_jump(const quadrotorKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint32_t *extra) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    words[QUADROTOR_GOAL_X] = floatWord(uniformRange(r.x, -QUADROTOR_REACH, QUADROTOR_REACH));
    words[QUADROTOR_GOAL_Y] = floatWord(uniformRange(r.y, QUADROTOR_LOW, QUADROTOR_HIGH));
    extra[QUADROTOR_JUMP] = step + quadrotor_interval(r.z, k->jumpPeriod);
}

static void quadrotor_gust(const quadrotorKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint32_t *extra) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u);
    words[QUADROTOR_WIND_X] = floatWord(uniformRange(r.x, -k->gustX, k->gustX));
    words[QUADROTOR_WIND_Y] = floatWord(uniformRange(r.y, -k->gustY, k->gustY));
    extra[QUADROTOR_GUST] = step + quadrotor_interval(r.z, k->gustPeriod);
}

static void quadrotor_obs(const uint32_t *words, float s, float c, float *obs) {
    float x = wordFloat(words[QUADROTOR_X]), y = wordFloat(words[QUADROTOR_Y]);
    obs[0] = QUADROTOR_POSITION_SCALE * x;
    obs[1] = QUADROTOR_POSITION_SCALE * (y - 0.5f * QUADROTOR_CEILING);
    obs[2] = QUADROTOR_POSITION_SCALE * (wordFloat(words[QUADROTOR_GOAL_X]) - x);
    obs[3] = QUADROTOR_POSITION_SCALE * (wordFloat(words[QUADROTOR_GOAL_Y]) - y);
    obs[4] = s;
    obs[5] = c;
    obs[6] = QUADROTOR_POSITION_SCALE * wordFloat(words[QUADROTOR_VX]);
    obs[7] = QUADROTOR_POSITION_SCALE * wordFloat(words[QUADROTOR_VY]);
    obs[8] = QUADROTOR_SPIN_SCALE * wordFloat(words[QUADROTOR_SPIN]);
    obs[9] = 1.0f;
}

static inline __attribute__((always_inline)) void quadrotor_settle(float x, float y, float angle, float vx, float vy, float spin, float s, float c, uint32_t *words, uint32_t *extra) {
    words[QUADROTOR_X] = floatWord(x); words[QUADROTOR_Y] = floatWord(y); words[QUADROTOR_ANGLE] = floatWord(angle);
    words[QUADROTOR_VX] = floatWord(vx); words[QUADROTOR_VY] = floatWord(vy); words[QUADROTOR_SPIN] = floatWord(spin);
    extra[QUADROTOR_SIN] = floatWord(s); extra[QUADROTOR_COS] = floatWord(c); extra[QUADROTOR_TAG] = words[QUADROTOR_ANGLE];
}

static inline __attribute__((always_inline)) void quadrotor_launch(uint32_t *words, uint32_t *extra) {
    float s, c;
    oak_sincos(0.0f, &s, &c);
    quadrotor_settle(0.0f, QUADROTOR_LAUNCH, 0.0f, 0.0f, 0.0f, 0.0f, s, c, words, extra);
}

void quadrotor_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const quadrotorKnobs *k = &QUADROTOR_KNOBS[instance];
    quadrotor_launch(words, extra);
    quadrotor_jump(k, seed, 0u, words, extra);
    if (k->gustPeriod > 0.0f) quadrotor_gust(k, seed, 0u, words, extra);
}

static __attribute__((noinline)) float quadrotor_events(const quadrotorKnobs *k, uint32_t seed, uint32_t step, uint32_t *words, uint32_t *extra, float *obs, float reward) {
    if (step >= extra[QUADROTOR_JUMP]) quadrotor_jump(k, seed, step, words, extra);
    if (k->gustPeriod > 0.0f && step >= extra[QUADROTOR_GUST]) quadrotor_gust(k, seed, step, words, extra);
    quadrotor_obs(words, wordFloat(extra[QUADROTOR_SIN]), wordFloat(extra[QUADROTOR_COS]), obs);
    return reward;
}

static inline __attribute__((always_inline)) void quadrotor_glide(float lift, float windX, float fall, float s, float c, float *x, float *y, float *vx, float *vy) {
    *vx = fmaf(QUADROTOR_SUBSTEP, fmaf(-lift, s, windX), *vx);
    *vy = fmaf(QUADROTOR_SUBSTEP, fmaf(lift, c, fall), *vy);
    *x = fmaf(QUADROTOR_SUBSTEP, *vx, *x);
    *y = fmaf(QUADROTOR_SUBSTEP, *vy, *y);
}

static inline __attribute__((always_inline)) float quadrotor_fly(const quadrotorKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    if (extra[QUADROTOR_TAG] != words[QUADROTOR_ANGLE]) __builtin_trap();
    float first = oak_clamp(ctrl[0], -1.0f, 1.0f), second = oak_clamp(ctrl[1], -1.0f, 1.0f);
    float lift = QUADROTOR_LIFT * (2.0f + (first + second)), turn = QUADROTOR_TORQUE * (first - second);
    float windX = k->gustPeriod > 0.0f ? wordFloat(words[QUADROTOR_WIND_X]) : 0.0f;
    float fall = k->gustPeriod > 0.0f ? wordFloat(words[QUADROTOR_WIND_Y]) - QUADROTOR_GRAVITY : -QUADROTOR_GRAVITY;
    float x = wordFloat(words[QUADROTOR_X]), y = wordFloat(words[QUADROTOR_Y]), vx = wordFloat(words[QUADROTOR_VX]), vy = wordFloat(words[QUADROTOR_VY]);
    float spin = wordFloat(words[QUADROTOR_SPIN]), angle = wordFloat(words[QUADROTOR_ANGLE]);
    float angles[QUADROTOR_SUBSTEPS], sines[QUADROTOR_SUBSTEPS + 1], cosines[QUADROTOR_SUBSTEPS + 1];
    for (uint32_t i = 0u; i < QUADROTOR_SUBSTEPS; i++) {
        angle = fmaf(QUADROTOR_SUBSTEP, fmaf(0.5f * QUADROTOR_SUBSTEP, turn, spin), angle);
        spin = fmaf(QUADROTOR_SUBSTEP, turn, spin);
        angles[i] = angle;
    }
    angle = angles[QUADROTOR_SUBSTEPS - 1] = quadrotor_wrap(angle);
    sines[0] = wordFloat(extra[QUADROTOR_SIN]); cosines[0] = wordFloat(extra[QUADROTOR_COS]);
    for (uint32_t i = 0u; i < QUADROTOR_SUBSTEPS; i++) oak_sincos(quadrotor_finite(angles[i]), &sines[i + 1], &cosines[i + 1]);
    uint32_t crashed = 0u;
    for (uint32_t i = 0u; i < QUADROTOR_SUBSTEPS; i++) {
        quadrotor_glide(lift, windX, fall, sines[i], cosines[i], &x, &y, &vx, &vy);
        crashed |= quadrotor_touches(x, y, sines[i + 1], cosines[i + 1]);
    }
    float reward = QUADROTOR_CRASH;
    *done = crashed;
    if (__builtin_expect(crashed, 0)) quadrotor_launch(words, extra);
    else {
        float dx = x - wordFloat(words[QUADROTOR_GOAL_X]), dy = y - wordFloat(words[QUADROTOR_GOAL_Y]);
        reward = -sqrtf(fmaf(dx, dx, dy * dy));
        quadrotor_settle(x, y, angle, vx, vy, spin, sines[QUADROTOR_SUBSTEPS], cosines[QUADROTOR_SUBSTEPS], words, extra);
    }
    if (__builtin_expect(step >= extra[QUADROTOR_JUMP] || (k->gustPeriod > 0.0f && step >= extra[QUADROTOR_GUST]), 0)) return quadrotor_events(k, seed, step, words, extra, obs, reward);
    quadrotor_obs(words, wordFloat(extra[QUADROTOR_SIN]), wordFloat(extra[QUADROTOR_COS]), obs);
    return reward;
}

float quadrotor_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return quadrotor_fly(&QUADROTOR_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
        case 1u: return quadrotor_fly(&QUADROTOR_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void quadrotor_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)shared;
    if (extra[QUADROTOR_TAG] != words[QUADROTOR_ANGLE]) __builtin_trap();
    quadrotor_obs(words, wordFloat(extra[QUADROTOR_SIN]), wordFloat(extra[QUADROTOR_COS]), obs);
}

static float quadrotor_heading(float sine, float cosine) {
    float angle = sine;
    for (uint32_t i = 0u; i < 3u; i++) {
        float s, c;
        oak_sincos(angle, &s, &c);
        angle -= (s * cosine - c * sine) / (c * cosine + s * sine);
    }
    return angle;
}

void quadrotor_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs;
    const quadrotorKnobs *k = &QUADROTOR_KNOBS[instance];
    if (extra[QUADROTOR_TAG] != words[QUADROTOR_ANGLE]) __builtin_trap();
    float windX = k->gustPeriod > 0.0f ? wordFloat(words[QUADROTOR_WIND_X]) : 0.0f;
    float lean = k->gustPeriod > 0.0f ? QUADROTOR_GRAVITY - wordFloat(words[QUADROTOR_WIND_Y]) : QUADROTOR_GRAVITY;
    float lift = sqrtf(fmaf(windX, windX, lean * lean)), across = windX / lift, along = lean / lift;
    float ex = wordFloat(words[QUADROTOR_X]) - wordFloat(words[QUADROTOR_GOAL_X]), ey = wordFloat(words[QUADROTOR_Y]) - wordFloat(words[QUADROTOR_GOAL_Y]);
    float vx = wordFloat(words[QUADROTOR_VX]), vy = wordFloat(words[QUADROTOR_VY]), spin = wordFloat(words[QUADROTOR_SPIN]);
    float px = along * ex + across * ey, py = along * ey - across * ex, qx = along * vx + across * vy, qy = along * vy - across * vx;
    float span = sqrtf(px * px + py * py);
    if (span > QUADROTOR_ORACLE_LEASH) { px *= QUADROTOR_ORACLE_LEASH / span; py *= QUADROTOR_ORACLE_LEASH / span; }
    const float *gain = shared;
    float hold = lift / (2.0f * QUADROTOR_LIFT) - (gain[1] * py + gain[4] * qy);
    float margin = oak_clamp(0.5f * hold, QUADROTOR_ORACLE_BANK, QUADROTOR_ORACLE_BRAKE), limit = quadrotor_heading(sqrtf(1.0f - margin * margin), margin);
    float bank = quadrotor_wrap(wordFloat(words[QUADROTOR_ANGLE]) - quadrotor_heading(across, along));
    float reference = oak_clamp(-(gain[0] * px + gain[3] * qx) / gain[2], -limit, limit);
    float turn = oak_clamp(-(gain[2] * (bank - reference) + gain[5] * spin), -1.0f, 1.0f), room = 1.0f - fabsf(turn);
    float upright = wordFloat(extra[QUADROTOR_COS]) * along + wordFloat(extra[QUADROTOR_SIN]) * across;
    float collective = oak_clamp(hold / fmaxf(upright, QUADROTOR_ORACLE_UPRIGHT) - 1.0f, -room, room);
    ctrl[0] = collective + turn;
    ctrl[1] = collective - turn;
}
