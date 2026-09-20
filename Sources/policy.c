#include "policy.h"

uint32_t policy_stateWords(uint32_t policy) {
    switch (policy) {
        case POLICY_OAK: return oak_stateWords();
        case POLICY_PPO: return ppo_stateWords();
        default: return 0u;
    }
}

void policy_init(uint32_t policy, device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, device const float *obs) {
    switch (policy) {
        case POLICY_OAK: oak_init(state, lane, env, seed, obs); return;
        case POLICY_PPO: ppo_init(state, lane, env, seed, obs); return;
        default: return;
    }
}

void policy_act(uint32_t policy, device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, device const float *obs, float reward, uint32_t done,
                device const uint32_t *words, device const uint32_t *extra, thread float *ctrl, constant const float *shared) {
    switch (policy) {
        case POLICY_OAK: oak_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_PPO: ppo_act(state, lane, env, seed, step, obs, reward, done, ctrl); return;
        case POLICY_RANDOM: random_act(env, seed, step, ctrl); return;
        case POLICY_ORACLE: env_oracle(env, seed, step, words, extra, obs, ctrl, shared); return;
        default: return;
    }
}
