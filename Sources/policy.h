#ifndef OAK_POLICY_H
#define OAK_POLICY_H
#include "env.h"

enum { POLICY_OAK, POLICY_PPO, POLICY_RANDOM, POLICY_ORACLE, POLICY_COUNT };

uint32_t policy_stateWords(uint32_t policy);
void policy_init(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
uint32_t oak_stateWords(void);
void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void oak_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
uint32_t ppo_stateWords(void);
void ppo_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void ppo_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
void random_act(uint32_t env, uint32_t seed, uint32_t step, float *ctrl);

static inline __attribute__((always_inline)) void policy_act(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step,
                                                             const float *obs, float reward, uint32_t done, const uint32_t *words, uint32_t *extra, float *ctrl,
                                                             const float *shared) {
    switch (policy) {
        case POLICY_OAK: oak_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_PPO: ppo_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_RANDOM: random_act(env, seed, step, ctrl); return;
        case POLICY_ORACLE: env_oracle(env, seed, step, words, extra, obs, ctrl, shared); return;
        default: return;
    }
}

#endif
