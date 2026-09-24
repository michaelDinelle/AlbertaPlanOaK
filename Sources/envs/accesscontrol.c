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

enum { ACCESS_DRAW, ACCESS_TAG = ACCESS_DRAW + 2 * ENV_BATCH, ACCESS_POLICY_BUCKET, ACCESS_POLICY };

static float access_freeRate(const accessKnobs *k, uint32_t step) {
    if (!k->drift) return 0.06f;
    float s, c, angle = 2.0f * OAK_PI * (float)(step % ACCESS_DRIFT_PERIOD) / (float)ACCESS_DRIFT_PERIOD;
    __builtin_assume(angle >= 0.0f && angle < 2.0f * OAK_PI);
    oak_sincos(angle, &s, &c);
    return 0.04f + 0.02f * s;
}

static void access_obs(uint32_t free, uint32_t priority, float *obs) {
    float4 hot = flag4((lane4)((lane4)priority == (lane4){ 0u, 1u, 2u, 3u }));
    obs[0] = (float)free / (float)ACCESS_SERVERS;
    __builtin_memcpy(obs + 1, &hot, sizeof hot);
    obs[1u + ACCESS_PRIORITIES] = 1.0f;
}

uint32_t access_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return ACCESS_POLICY + ACCESS_PRIORITIES * (ACCESS_SERVERS + 1u);
}

static double access_binomial(uint32_t trials, uint32_t successes, double rate) {
    double combinations = 1.0;
    for (uint32_t i = 1u; i <= successes; i++) combinations = combinations * (double)(trials - successes + i) / (double)i;
    return combinations * pow(rate, (double)successes) * pow(1.0 - rate, (double)(trials - successes));
}

static void access_expected(const double chance[][ACCESS_SERVERS + 1u], const double *value, double *expected) {
    for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) {
        expected[free] = 0.0;
        for (uint32_t freed = 0u; freed <= ACCESS_SERVERS - free; freed++) expected[free] += chance[free][freed] * value[free + freed];
    }
}

static void access_solve(double rate, uint32_t *policy) {
    double chance[ACCESS_SERVERS + 1u][ACCESS_SERVERS + 1u], value[ACCESS_SERVERS + 1u], next[ACCESS_SERVERS + 1u], expected[ACCESS_SERVERS + 1u];
    for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++)
        for (uint32_t freed = 0u; freed <= ACCESS_SERVERS - free; freed++) chance[free][freed] = access_binomial(ACCESS_SERVERS - free, freed, rate);
    for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) value[free] = 0.0;
    for (uint32_t iteration = 0u; iteration < ACCESS_SOLVE_ITERATIONS; iteration++) {
        access_expected(chance, value, expected);
        for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++) {
            double reject = expected[free], total = 0.0;
            for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++) {
                double accept = free > 0u ? (double)ACCESS_REWARD[p] + expected[free - 1u] : reject;
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
    access_expected(chance, value, expected);
    for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++)
        for (uint32_t free = 0u; free <= ACCESS_SERVERS; free++)
            policy[p * (ACCESS_SERVERS + 1u) + free] = free > 0u && (double)ACCESS_REWARD[p] + expected[free - 1u] > expected[free];
}

static uint32_t access_freed(uint64_t draw, uint32_t free) { return (uint32_t)(draw >> (4u * free)) & 15u; }

static inline __attribute__((always_inline)) lane4 access_word(uint32_t seed, lane4 steps, uint32_t block) {
    return philox4(seed, ENV_RNG_STREAM, steps, (lane4)block, (lane4)0u, (lane4)0u).x;
}

static inline __attribute__((always_inline)) void access_count(lane4 word, lane4 below, lane4 *count, lane4 *packed) {
    *count -= (lane4)(word < below);
    *packed = *packed << 4u | *count;
}

static inline __attribute__((always_inline)) void access_draw(uint32_t seed, uint32_t step, lane4 below, uint32_t *extra) {
    lane4 steps = step + (lane4){ 0u, 1u, 2u, 3u };
    u44 first = philox4(seed, ENV_RNG_STREAM, steps, (lane4)0u, (lane4)0u, (lane4)0u);
    lane4 count = (lane4)0u, low = (lane4)0u, high = (access_word(seed, steps, 32u) % ACCESS_PRIORITIES) << 8u;
    access_count(first.x, below, &count, &high);
    access_count(first.y, below, &count, &high);
    access_count(first.z, below, &count, &low);
    access_count(first.w, below, &count, &low);
    for (uint32_t s = 4u; s < ACCESS_SERVERS; s++) access_count(access_word(seed, steps, 1u + s), below, &count, &low);
    lane4 draw[2] = { __builtin_shufflevector(low, high, 0, 4, 1, 5), __builtin_shufflevector(low, high, 2, 6, 3, 7) };
    __builtin_memcpy(extra + ACCESS_DRAW, draw, sizeof draw);
}

static inline __attribute__((always_inline)) float access_advance(const float *ctrl, uint32_t *words, const uint32_t *draws, float *obs) {
    uint32_t free = words[0], priority = words[1];
    uint32_t accept = ((uint32_t)ctrl[0] & 1u) != 0u && free > 0u;
    float gain = ACCESS_REWARD[priority], reward = accept ? gain : 0.0f;
    free -= accept;
    uint64_t draw;
    __builtin_memcpy(&draw, draws, sizeof draw);
    free += access_freed(draw, free);
    priority = (uint32_t)(draw >> 48);
    words[0] = free; words[1] = priority;
    access_obs(free, priority, obs);
    return reward;
}

static __attribute__((noinline)) float access_refill(const accessKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    lane4 below = { u01Bound(access_freeRate(k, step)), u01Bound(access_freeRate(k, step + 1u)), u01Bound(access_freeRate(k, step + 2u)), u01Bound(access_freeRate(k, step + 3u)) };
    access_draw(seed, step, below, extra);
    return access_advance(ctrl, words, extra + ACCESS_DRAW, obs);
}

void access_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)steps;
    words[0] = ACCESS_SERVERS;
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % ACCESS_PRIORITIES;
    batchReset(&extra[ACCESS_TAG]);
}

float access_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    uint32_t slot = batchSlot(&extra[ACCESS_TAG], step);
    if (slot == ENV_BATCH) return access_refill(&ACCESS_KNOBS[instance], seed, step, ctrl, words, extra, obs);
    return access_advance(ctrl, words, extra + ACCESS_DRAW + 2u * slot, obs);
}

void access_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    access_obs(words[0], words[1], obs);
}

void access_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)obs;
    const accessKnobs *k = &ACCESS_KNOBS[instance];
    uint32_t bucket = k->drift ? 1u + step / ACCESS_POLICY_PERIOD : 1u;
    if (extra[ACCESS_POLICY_BUCKET] != bucket) {
        access_solve((double)access_freeRate(k, (bucket - 1u) * ACCESS_POLICY_PERIOD), extra + ACCESS_POLICY);
        extra[ACCESS_POLICY_BUCKET] = bucket;
    }
    ctrl[0] = (float)extra[ACCESS_POLICY + words[1] * (ACCESS_SERVERS + 1u) + words[0]];
}
