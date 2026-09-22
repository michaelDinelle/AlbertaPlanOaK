#include "../env.h"

#define ACCESS_SERVERS 10u
#define ACCESS_PRIORITIES 4u
#define ACCESS_DRIFT_PERIOD 200000u
#define ACCESS_POLICY_PERIOD (ACCESS_DRIFT_PERIOD / 256u)
#define ACCESS_SOLVE_ITERATIONS 4096u
#define ACCESS_SOLVE_TOLERANCE 1e-12

typedef struct { uint32_t drift; } accessKnobs;

static const accessKnobs ACCESS_KNOBS[2] = {
    { 0u },
    { 1u }
};

static const float ACCESS_REWARD[ACCESS_PRIORITIES] = { 1.0f, 2.0f, 4.0f, 8.0f };

static float access_freeRate(const accessKnobs *k, uint32_t step) {
    if (!k->drift) return 0.06f;
    float s, c;
    oak_sincos(2.0f * OAK_PI * (float)(step % ACCESS_DRIFT_PERIOD) / (float)ACCESS_DRIFT_PERIOD, &s, &c);
    return 0.04f + 0.02f * s;
}

static void access_obs(uint32_t free, uint32_t priority, float *obs) {
    obs[0] = (float)free / (float)ACCESS_SERVERS;
    for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++) obs[1u + p] = p == priority ? 1.0f : 0.0f;
    obs[1u + ACCESS_PRIORITIES] = 1.0f;
}

uint32_t access_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 1u + ACCESS_PRIORITIES * (ACCESS_SERVERS + 1u);
}

static double access_binomial(uint32_t trials, uint32_t successes, double rate) {
    double combinations = 1.0;
    for (uint32_t i = 1u; i <= successes; i++) combinations = combinations * (double)(trials - successes + i) / (double)i;
    return combinations * pow(rate, (double)successes) * pow(1.0 - rate, (double)(trials - successes));
}

static double access_expected(const double *value, uint32_t free, double rate) {
    double expected = 0.0;
    uint32_t busy = ACCESS_SERVERS - free;
    for (uint32_t freed = 0u; freed <= busy; freed++) expected += access_binomial(busy, freed, rate) * value[free + freed];
    return expected;
}

static void access_solve(double rate, uint32_t *policy) {
    double value[ACCESS_SERVERS + 1u], next[ACCESS_SERVERS + 1u];
    for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) value[free] = 0.0;
    for (uint32_t iteration = 0u; iteration < ACCESS_SOLVE_ITERATIONS; iteration++) {
        for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) {
            double reject = access_expected(value, free, rate), total = 0.0;
            for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++) {
                double accept = free > 0u ? (double)ACCESS_REWARD[p] + access_expected(value, free - 1u, rate) : reject;
                total += (accept > reject ? accept : reject) / (double)ACCESS_PRIORITIES;
            }
            next[free] = total;
        }
        double reference = next[0], delta = 0.0;
        for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) {
            double updated = next[free] - reference, change = fabs(updated - value[free]);
            if (change > delta) delta = change;
            value[free] = updated;
        }
        if (delta < ACCESS_SOLVE_TOLERANCE) break;
    }
    for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++)
        for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++)
            policy[p * (ACCESS_SERVERS + 1u) + free] = free > 0u && (double)ACCESS_REWARD[p] + access_expected(value, free - 1u, rate) > access_expected(value, free, rate);
}

static void access_refresh(const accessKnobs *k, uint32_t step, uint32_t *extra) {
    uint32_t bucket = k->drift ? 1u + step / ACCESS_POLICY_PERIOD : 1u;
    if (extra[0] == bucket) return;
    access_solve((double)access_freeRate(k, step), extra + 1);
    extra[0] = bucket;
}

void access_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    words[0] = ACCESS_SERVERS;
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % ACCESS_PRIORITIES;
    extra[0] = 0u;
    access_refresh(&ACCESS_KNOBS[instance], 0u, extra);
}

float access_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const accessKnobs *k = &ACCESS_KNOBS[instance];
    uint32_t free = words[0], priority = words[1];
    *done = 0u;
    float reward = 0.0f;
    if (((uint32_t)ctrl[0] & 1u) != 0u && free > 0u) { reward = ACCESS_REWARD[priority]; free -= 1u; }
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    uint32_t first[4] = { r.x, r.y, r.z, r.w };
    float freeRate = access_freeRate(k, step);
    uint32_t busy = ACCESS_SERVERS - free;
    for (uint32_t s = 0u; s < busy; s++) {
        uint32_t word = s < 4u ? first[s] : philox4x32(seed, ENV_RNG_STREAM, step, 1u + s, 0u, 0u).x;
        if (u01(word) < freeRate) free += 1u;
    }
    priority = philox4x32(seed, ENV_RNG_STREAM, step, 32u, 0u, 0u).x % ACCESS_PRIORITIES;
    words[0] = free; words[1] = priority;
    access_refresh(k, step, extra);
    access_obs(free, priority, obs);
    return reward;
}

void access_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    access_obs(words[0], words[1], obs);
}

void access_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)obs;
    ctrl[0] = (float)extra[1u + words[1] * (ACCESS_SERVERS + 1u) + words[0]];
}
