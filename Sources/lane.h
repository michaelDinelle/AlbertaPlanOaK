#ifndef OAK_LANE_H
#define OAK_LANE_H
#include "policy.h"

#define OAK_WINDOW 1024u
#define OAK_HASH_BASIS 0xcbf29ce484222325ull
#define OAK_HASH_PRIME 0x100000001b3ull
#define OAK_HEALTH_ALIVE 0xffffffffu

typedef struct {
    uint64_t hash;
    uint32_t env, seed, step, done, health, extraOffset;
    float reward, sum, compensation;
    uint32_t words[OAK_ENV_WORDS_MAX];
    float obs[OAK_ENV_OBS_MAX];
} Lane;

static inline void lane_init(Lane *l, uint32_t lane, uint32_t steps, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy) {
    l->hash = OAK_HASH_BASIS;
    l->step = 0u;
    l->done = 0u;
    l->health = OAK_HEALTH_ALIVE;
    l->reward = 0.0f;
    l->sum = 0.0f;
    l->compensation = 0.0f;
    for (uint32_t i = 0u; i < OAK_ENV_WORDS_MAX; i++) l->words[i] = 0u;
    for (uint32_t i = 0u; i < OAK_ENV_OBS_MAX; i++) l->obs[i] = 0.0f;
    env_reset(l->env, l->seed, l->words, extra, steps);
    env_observe(l->env, l->seed, l->words, extra, 0u, l->obs, shared);
    policy_init(policy, state, lane, l->env, l->seed, l->obs);
}

static inline void lane_step(Lane *l, uint32_t lane, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy, float *ctrl) {
    for (uint32_t d = 0u; d < OAK_ENV_ACT_MAX; d++) ctrl[d] = 0.0f;
    policy_act(policy, state, lane, l->env, l->seed, l->step, l->obs, l->reward, l->done, l->words, extra, ctrl, shared);
    uint32_t done = 0u;
    float reward = env_step(l->env, l->seed, l->step + 1u, ctrl, l->words, extra, l->obs, &done, shared);
    l->done = done;
    l->reward = reward;
    l->hash = (l->hash ^ (uint64_t)floatWord(reward)) * OAK_HASH_PRIME;
    if (isfinite(reward)) {
        float adjusted = reward - l->compensation;
        float next = l->sum + adjusted;
        l->compensation = (next - l->sum) - adjusted;
        l->sum = next;
    } else if (l->health == OAK_HEALTH_ALIVE) {
        l->health = l->step + 1u;
    }
    uint32_t obsCount = env_obs(l->env);
    for (uint32_t i = 0u; i < obsCount; i++)
        if (!isfinite(l->obs[i]) && l->health == OAK_HEALTH_ALIVE) l->health = l->step + 1u;
    l->step += 1u;
}

static inline void lane_window(Lane *l, uint32_t lane, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy, uint32_t end) {
    float ctrl[OAK_ENV_ACT_MAX];
    while (l->step < end) lane_step(l, lane, state, extra, shared, policy, ctrl);
}

#endif
