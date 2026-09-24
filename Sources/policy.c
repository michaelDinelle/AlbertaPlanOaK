#include "policy.h"

uint32_t policy_stateWords(uint32_t policy) {
    switch (policy) {
        case POLICY_OAK: return oak_stateWords();
        case POLICY_PPO: return ppo_stateWords();
        default: return 0u;
    }
}

void policy_init(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    switch (policy) {
        case POLICY_OAK: oak_init(state, lane, env, seed, obs); return;
        case POLICY_PPO: ppo_init(state, lane, env, seed, obs); return;
        default: return;
    }
}
