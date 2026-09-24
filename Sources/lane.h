#ifndef OAK_LANE_H
#define OAK_LANE_H
#include <string.h>
#include "policy.h"

#define OAK_WINDOW 1024u
#define OAK_HASH_BASIS 0xcbf29ce484222325ull
#define OAK_HASH_PRIME 0x100000001b3ull
#define OAK_HEALTH_ALIVE 0xffffffffu

typedef struct {
    uint64_t hash;
    uint32_t step, done, health;
    float reward, sum, compensation;
} LaneTally;

typedef struct {
    LaneTally tally;
    uint32_t env, seed, extraOffset;
    _Alignas(16) uint32_t words[OAK_ENV_WORDS_MAX];
    _Alignas(16) float obs[OAK_ENV_OBS_MAX];
} Lane;

static inline void lane_init(Lane *l, uint32_t lane, uint32_t steps, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy) {
    l->tally = (LaneTally){ .hash = OAK_HASH_BASIS, .health = OAK_HEALTH_ALIVE };
    for (uint32_t i = 0u; i < OAK_ENV_WORDS_MAX; i++) l->words[i] = 0u;
    for (uint32_t i = 0u; i < OAK_ENV_OBS_MAX; i++) l->obs[i] = 0.0f;
    env_reset(l->env, l->seed, l->words, extra, steps);
    env_observe(l->env, l->seed, l->words, extra, 0u, l->obs, shared);
    policy_init(policy, state, lane, l->env, l->seed, l->obs);
}

static inline int lane_finite(const float *obs, uint32_t count) {
    lane4 bad = 0u;
    for (uint32_t i = 0u; i < count; i += 4u) {
        lane4 w, at = i + (lane4){ 0u, 1u, 2u, 3u };
        memcpy(&w, obs + i, sizeof w);
        bad |= ((w & 0x7fffffffu) >= 0x7f800000u) & (at < count);
    }
    return !(bad.x | bad.y | bad.z | bad.w);
}

static inline __attribute__((always_inline)) void lane_run(Lane *l, uint32_t lane, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy,
                                                           uint32_t end, float *ctrl) {
    LaneTally t = l->tally;
    uint32_t env = l->env, seed = l->seed, obsCount = env_obs(env);
    while (t.step < end) {
        for (uint32_t d = 0u; d < OAK_ENV_ACT_MAX; d++) ctrl[d] = 0.0f;
        policy_act(policy, state, lane, env, seed, t.step, l->obs, t.reward, t.done, l->words, extra, ctrl, shared);
        uint32_t done = 0u;
        float reward = env_step(env, seed, t.step + 1u, ctrl, l->words, extra, l->obs, &done, shared);
        t.done = done;
        t.reward = reward;
        t.hash = (t.hash ^ (uint64_t)floatWord(reward)) * OAK_HASH_PRIME;
        if (isfinite(reward)) {
            float adjusted = reward - t.compensation;
            float next = t.sum + adjusted;
            t.compensation = (next - t.sum) - adjusted;
            t.sum = next;
        }
        if (t.health == OAK_HEALTH_ALIVE && (!isfinite(reward) || !lane_finite(l->obs, obsCount))) t.health = t.step + 1u;
        t.step += 1u;
    }
    l->tally = t;
}

static inline void lane_window(Lane *l, uint32_t lane, uint32_t *state, uint32_t *extra, const float *shared, uint32_t policy, uint32_t end, float *ctrl) {
    switch (policy) {
        case POLICY_OAK: lane_run(l, lane, state, extra, shared, POLICY_OAK, end, ctrl); return;
        case POLICY_PPO: lane_run(l, lane, state, extra, shared, POLICY_PPO, end, ctrl); return;
        case POLICY_RANDOM: lane_run(l, lane, state, extra, shared, POLICY_RANDOM, end, ctrl); return;
        default: lane_run(l, lane, state, extra, shared, POLICY_ORACLE, end, ctrl); return;
    }
}

#endif
