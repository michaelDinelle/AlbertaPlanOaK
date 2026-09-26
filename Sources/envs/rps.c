#include "../env.h"

typedef struct { uint32_t window; } rpsKnobs;

static const rpsKnobs RPS_KNOBS[2] = {
    { 10u },
    { 100u }
};


static uint32_t rps_counter(uint32_t move) { return move == 2u ? 0u : move + 1u; }

static uint32_t rps_prediction(const uint32_t *words) {
    uint32_t best = words[2] > words[1] ? 1u : 0u, most = words[2] > words[1] ? words[2] : words[1];
    return words[3] > most ? 2u : best;
}

static uint32_t rps_historyAt(const uint32_t *words, uint32_t index) { return (words[4u + (index >> 4u)] >> (2u * (index & 15u))) & 3u; }

static void rps_historySet(uint32_t *words, uint32_t index, uint32_t move) {
    uint32_t *w = &words[4u + (index >> 4u)];
    *w = (*w & ~(3u << (2u * (index & 15u)))) | (move << (2u * (index & 15u)));
}

static void rps_obs(uint32_t lastOpponent, float *obs) {
    float4 row = flag4((lane4)(lastOpponent == (lane4){ 0u, 1u, 2u, 3u }) | (lane4){ 0u, 0u, 0u, ~0u });
    __builtin_memcpy(obs, &row, sizeof row);
}

static inline __attribute__((always_inline)) float rps_advance(const rpsKnobs *k, const float *ctrl, uint32_t *words, float *obs) {
    uint32_t move = (uint32_t)ctrl[0] % 3u, filled = words[11], cursor = words[12];
    uint32_t opponent = filled == 0u ? 0u : rps_counter(rps_prediction(words));
    float beat = move == rps_counter(opponent) ? 1.0f : -1.0f, reward = move == opponent ? 0.0f : beat;
    uint32_t full = filled >= k->window, old = full ? rps_historyAt(words, cursor) : 3u;
    for (uint32_t m = 0u; m < 3u; m++) words[1u + m] += (uint32_t)(move == m) - (uint32_t)(old == m);
    rps_historySet(words, cursor, move);
    words[11] = filled + !full; words[12] = cursor + 1u == k->window ? 0u : cursor + 1u; words[0] = opponent;
    rps_obs(opponent, obs);
    return reward;
}

void rps_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float rps_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    *done = 0u;
    return instance == 0u ? rps_advance(&RPS_KNOBS[0], ctrl, words, obs) : rps_advance(&RPS_KNOBS[1], ctrl, words, obs);
}

void rps_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    rps_obs(words[0], obs);
}

void rps_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t opponent = words[11] == 0u ? 0u : rps_counter(rps_prediction(words));
    ctrl[0] = (float)rps_counter(opponent);
}
