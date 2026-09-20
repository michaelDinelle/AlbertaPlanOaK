#include "../env.h"

typedef struct { uint32_t window; } rpsKnobs;

OAK_CONSTANT rpsKnobs RPS_KNOBS[2] = {
    { 10u },
    { 100u }
};


static uint32_t rps_counter(uint32_t move) { return (move + 1u) % 3u; }

static uint32_t rps_prediction(const device uint32_t *words) {
    uint32_t best = 0u;
    for (uint32_t m = 1u; m < 3u; m++) if (words[1u + m] > words[1u + best]) best = m;
    return best;
}

static uint32_t rps_historyAt(const device uint32_t *words, uint32_t index) { return (words[4u + (index >> 4u)] >> (2u * (index & 15u))) & 3u; }

static void rps_historySet(device uint32_t *words, uint32_t index, uint32_t move) {
    device uint32_t *w = &words[4u + (index >> 4u)];
    *w = (*w & ~(3u << (2u * (index & 15u)))) | (move << (2u * (index & 15u)));
}

static void rps_obs(uint32_t lastOpponent, device float *obs) {
    for (uint32_t m = 0u; m < 3u; m++) obs[m] = lastOpponent == m ? 1.0f : 0.0f;
    obs[3] = 1.0f;
}

void rps_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float rps_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    constant const rpsKnobs *k = &RPS_KNOBS[instance];
    *done = 0u;
    uint32_t move = (uint32_t)ctrl[0] % 3u, filled = words[11], cursor = words[12];
    uint32_t opponent = filled == 0u ? 0u : rps_counter(rps_prediction(words));
    float reward = move == opponent ? 0.0f : move == rps_counter(opponent) ? 1.0f : -1.0f;
    if (filled >= k->window) words[1u + rps_historyAt(words, cursor)] -= 1u; else filled += 1u;
    rps_historySet(words, cursor, move);
    words[1u + move] += 1u;
    words[11] = filled; words[12] = (cursor + 1u) % k->window; words[0] = opponent;
    rps_obs(opponent, obs);
    return reward;
}

void rps_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    rps_obs(words[0], obs);
}

void rps_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint32_t opponent = words[11] == 0u ? 0u : rps_counter(rps_prediction(words));
    ctrl[0] = (float)rps_counter(opponent);
}
