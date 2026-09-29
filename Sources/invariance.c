#include "policy.h"

#define INVARIANCE_STREAM 0x1A5Eu
#define INVARIANCE_LN10 2.302585093f

typedef struct {
    uint32_t perm[OAK_ENV_OBS_MAX], relabel[OAK_ENV_ACT_MAX];
    float scale[OAK_ENV_OBS_MAX], offset[OAK_ENV_OBS_MAX];
} Invariance;

static Invariance *invariance_of(uint32_t *state) { return (Invariance *)(state + oak_stateWords()); }

uint32_t invariance_stateWords(void) { return oak_stateWords() + (uint32_t)((sizeof(Invariance) + sizeof(uint32_t) - 1u) / sizeof(uint32_t)); }

static void invariance_shuffle(uint32_t *order, uint32_t count, uint32_t seed, uint32_t tag) {
    for (uint32_t i = 0u; i < count; i++) order[i] = i;
    for (uint32_t i = count; i > 1u; i--) {
        uint32_t j = (uint32_t)(((uint64_t)philox4x32(seed, INVARIANCE_STREAM, tag, i, 0u, 0u).x * i) >> 32);
        uint32_t held = order[i - 1u];
        order[i - 1u] = order[j];
        order[j] = held;
    }
}

static void invariance_view(uint32_t policy, const Invariance *v, uint32_t count, const float *obs, float *view) {
    for (uint32_t i = 0u; i < count; i++) view[i] = policy == POLICY_OAK_PERMUTED ? obs[v->perm[i]] : v->scale[i] * obs[i] + v->offset[i];
}

void invariance_init(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    uint32_t count = env_obs(env);
    Invariance *v = invariance_of(state);
    invariance_shuffle(v->perm, count, seed, 1u);
    invariance_shuffle(v->relabel, env_actions(env), seed, 2u);
    for (uint32_t i = 0u; i < count; i++) {
        u4 r = philox4x32(seed, INVARIANCE_STREAM, 3u, i, 0u, 0u);
        v->scale[i] = oak_exp(INVARIANCE_LN10 * (4.0f * u01(r.x) - 2.0f));
        v->offset[i] = 20.0f * u01(r.y) - 10.0f;
    }
    float view[OAK_ENV_OBS_MAX] = { 0.0f };
    invariance_view(policy, v, count, obs, view);
    oak_init(state, lane, env, seed, view);
}

void invariance_act(uint32_t policy, uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl) {
    const Invariance *v = invariance_of(state);
    uint32_t count = env_obs(env), actions = env_actions(env);
    float view[OAK_ENV_OBS_MAX] = { 0.0f }, chosen[OAK_ENV_ACT_MAX] = { 0.0f };
    invariance_view(policy, v, count, obs, view);
    oak_act(state, lane, env, seed, step, view, reward, done, chosen);
    if (policy == POLICY_OAK_RESCALED) {
        for (uint32_t d = 0u; d < actions; d++) ctrl[d] = chosen[d];
        return;
    }
    if (env_box(env)) {
        for (uint32_t d = 0u; d < actions; d++) ctrl[v->relabel[d]] = chosen[d];
        return;
    }
    uint32_t a = (uint32_t)chosen[0];
    ctrl[0] = (float)v->relabel[a < actions ? a : 0u];
    for (uint32_t d = 1u; d < actions; d++) ctrl[d] = 0.0f;
}
