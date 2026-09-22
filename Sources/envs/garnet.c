#include "../env.h"

#define GARNET_PERIOD 100000u
#define GARNET_DISCOUNT 0.9999
#define GARNET_STATES_MAX 50u
#define GARNET_IMPROVEMENTS 200u
#define GARNET_IMPROVEMENT_MARGIN 1e-9

typedef struct { uint32_t states, actions, branching, bits, resample; } garnetKnobs;

static const garnetKnobs GARNET_KNOBS[4] = {
    { 10u, 2u, 2u, 4u, 0u },
    { 50u, 4u, 3u, 6u, 0u },
    { 10u, 2u, 2u, 4u, 1u },
    { 50u, 4u, 3u, 6u, 1u }
};


typedef struct { uint32_t *next; float *prob; float *reward; float *value; uint32_t *greedy; uint32_t epoch; } garnetModel;

static garnetModel garnet_model(const garnetKnobs *k, uint32_t *extra) {
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    garnetModel m;
    uint32_t *base = extra;
    m.next = base + 1u;
    m.prob = (float *)(m.next + edges);
    m.reward = m.prob + edges;
    m.value = m.reward + pairs;
    m.greedy = (uint32_t *)(m.value + k->states);
    m.epoch = base[0];
    return m;
}

uint32_t garnet_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    return 1u + 2u * edges + pairs + k->states + k->states;
}

static double garnet_q(const garnetKnobs *k, const garnetModel *m, const double *value, uint32_t state, uint32_t action) {
    uint32_t pair = state * k->actions + action;
    double q = (double)m->reward[pair];
    for (uint32_t b = 0u; b < k->branching; b++) q += GARNET_DISCOUNT * (double)m->prob[pair * k->branching + b] * value[m->next[pair * k->branching + b]];
    return q;
}

static void garnet_evaluate(const garnetKnobs *k, const garnetModel *m, double *value) {
    uint32_t n = k->states;
    double a[GARNET_STATES_MAX * GARNET_STATES_MAX], b[GARNET_STATES_MAX];
    for (uint32_t i = 0u; i < n * n; i++) a[i] = 0.0;
    for (uint32_t s = 0u; s < n; s++) {
        uint32_t pair = s * k->actions + m->greedy[s];
        a[s * n + s] += 1.0;
        for (uint32_t br = 0u; br < k->branching; br++) a[s * n + m->next[pair * k->branching + br]] -= GARNET_DISCOUNT * (double)m->prob[pair * k->branching + br];
        b[s] = (double)m->reward[pair];
    }
    for (uint32_t col = 0u; col < n; col++) {
        uint32_t pivot = col;
        for (uint32_t row = col + 1u; row < n; row++) if (fabs(a[row * n + col]) > fabs(a[pivot * n + col])) pivot = row;
        if (pivot != col) {
            for (uint32_t c = 0u; c < n; c++) { double t = a[col * n + c]; a[col * n + c] = a[pivot * n + c]; a[pivot * n + c] = t; }
            double t = b[col]; b[col] = b[pivot]; b[pivot] = t;
        }
        for (uint32_t row = col + 1u; row < n; row++) {
            double factor = a[row * n + col] / a[col * n + col];
            if (factor == 0.0) continue;
            for (uint32_t c = col; c < n; c++) a[row * n + c] -= factor * a[col * n + c];
            b[row] -= factor * b[col];
        }
    }
    for (uint32_t back = 0u; back < n; back++) {
        uint32_t row = n - 1u - back;
        double sum = b[row];
        for (uint32_t c = row + 1u; c < n; c++) sum -= a[row * n + c] * value[c];
        value[row] = sum / a[row * n + row];
    }
}

static void garnet_solve(const garnetKnobs *k, garnetModel m) {
    double value[GARNET_STATES_MAX];
    for (uint32_t s = 0u; s < k->states; s++) {
        uint32_t best = 0u;
        for (uint32_t action = 1u; action < k->actions; action++) if (m.reward[s * k->actions + action] > m.reward[s * k->actions + best]) best = action;
        m.greedy[s] = best;
    }
    for (uint32_t improvement = 0u; improvement < GARNET_IMPROVEMENTS; improvement++) {
        garnet_evaluate(k, &m, value);
        uint32_t changed = 0u;
        for (uint32_t s = 0u; s < k->states; s++) {
            uint32_t best = m.greedy[s];
            double bestQ = garnet_q(k, &m, value, s, best);
            for (uint32_t action = 0u; action < k->actions; action++) {
                double q = garnet_q(k, &m, value, s, action);
                if (q > bestQ + GARNET_IMPROVEMENT_MARGIN) { bestQ = q; best = action; }
            }
            if (best != m.greedy[s]) { m.greedy[s] = best; changed = 1u; }
        }
        for (uint32_t s = 0u; s < k->states; s++) m.value[s] = (float)value[s];
        if (!changed) break;
    }
}

static void garnet_build(const garnetKnobs *k, uint32_t seed, uint32_t epoch, uint32_t *extra) {
    extra[0] = epoch;
    garnetModel m = garnet_model(k, extra);
    for (uint32_t s = 0u; s < k->states; s++)
        for (uint32_t a = 0u; a < k->actions; a++) {
            uint32_t pair = s * k->actions + a;
            u4 r = philox4x32(seed, RNG_STREAM_RESET, epoch, pair, 0u, 0u);
            float total = 0.0f;
            for (uint32_t b = 0u; b < k->branching; b++) {
                u4 e = philox4x32(seed, RNG_STREAM_RESET, epoch, pair, 1u + b, 0u);
                m.next[pair * k->branching + b] = e.x % k->states;
                m.prob[pair * k->branching + b] = u01open(e.y);
                total += m.prob[pair * k->branching + b];
            }
            for (uint32_t b = 0u; b < k->branching; b++) m.prob[pair * k->branching + b] /= total;
            m.reward[pair] = normalFrom(r.y, r.z);
        }
    garnet_solve(k, m);
}

static void garnet_obs(const garnetKnobs *k, uint32_t state, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b++) obs[b] = (float)((state >> b) & 1u);
    obs[k->bits] = 1.0f;
}

void garnet_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    garnet_build(k, seed, 0u, extra);
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0xffffu, 0u, 0u, 0u).x % k->states;
}

float garnet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    *done = 0u;
    if (k->resample && step / GARNET_PERIOD != extra[0]) garnet_build(k, seed, step / GARNET_PERIOD, extra);
    garnetModel m = garnet_model(k, extra);
    uint32_t pair = words[0] * k->actions + (uint32_t)ctrl[0] % k->actions;
    float reward = m.reward[pair];
    float roll = u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x), cumulative = 0.0f;
    uint32_t next = m.next[pair * k->branching + k->branching - 1u];
    for (uint32_t b = 0u; b < k->branching; b++) {
        cumulative += m.prob[pair * k->branching + b];
        if (roll < cumulative) { next = m.next[pair * k->branching + b]; break; }
    }
    words[0] = next;
    garnet_obs(k, next, obs);
    return reward;
}

void garnet_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    garnet_obs(&GARNET_KNOBS[instance], words[0], obs);
}

void garnet_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)obs;
    garnetModel m = garnet_model(&GARNET_KNOBS[instance], (uint32_t *)extra);
    ctrl[0] = (float)m.greedy[words[0]];
}
