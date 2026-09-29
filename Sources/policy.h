#ifndef OAK_POLICY_H
#define OAK_POLICY_H
#include "env.h"

enum { POLICY_OAK, POLICY_PPO, POLICY_RANDOM, POLICY_ORACLE, POLICY_CONSTANT, POLICY_STREAM, POLICY_OAK_PERMUTED, POLICY_OAK_RESCALED, POLICY_COUNT };

uint32_t policy_stateWords(uint32_t policy);
void policy_init(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
uint32_t oak_stateWords(void);
void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void oak_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
uint32_t ppo_stateWords(void);
void ppo_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void ppo_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
void random_act(uint32_t env, uint32_t seed, uint32_t step, float *ctrl);
void constant_act(uint32_t env, float *ctrl);
uint32_t stream_stateWords(void);
void stream_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void stream_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
uint32_t invariance_stateWords(void);
void invariance_init(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs);
void invariance_act(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl);
uint32_t oak_nonfinite(const uint32_t *state);

static inline __attribute__((always_inline)) void policy_act(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step,
                                                             const float *obs, float reward, uint32_t done, const uint32_t *words, uint32_t *extra, float *ctrl,
                                                             const float *shared) {
    switch (policy) {
        case POLICY_OAK: oak_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_PPO: ppo_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_RANDOM: random_act(env, seed, step, ctrl); return;
        case POLICY_ORACLE: env_oracle(env, seed, step, words, extra, obs, ctrl, shared); return;
        case POLICY_CONSTANT: constant_act(env, ctrl); return;
        case POLICY_STREAM: stream_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_OAK_PERMUTED: invariance_act(POLICY_OAK_PERMUTED, state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_OAK_RESCALED: invariance_act(POLICY_OAK_RESCALED, state, lane, env, seed, step, obs, reward, done, ctrl); return;
        default: return;
    }
}

#endif
