#include "../env.h"

typedef struct { uint32_t limit; float scale; uint32_t lastActionObs; } walkKnobs;

OAK_CONSTANT walkKnobs WALK_KNOBS[1] = {
    { 15u, 16.0f, 0u }
};


static void walk_obs(constant const walkKnobs *k, uint32_t pos, uint32_t lastAction, device float *obs) {
    obs[0] = (float)pos / k->scale;
    if (k->lastActionObs) { obs[1] = (float)lastAction; obs[2] = 1.0f; }
    else obs[1] = 1.0f;
}

void walk_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) { (void)steps; (void)extra; (void)instance; (void)seed; (void)words; }

float walk_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    constant const walkKnobs *k = &WALK_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0];
    uint32_t pos = words[0];
    float reward = 0.0f;
    *done = 0u;
    if ((action & 1u) != 0u) pos = pos + 1u;
    else pos = (pos > 0u) ? pos - 1u : 0u;
    if ((action & 1u) != 0u && pos >= k->limit) { reward = 1.0f; pos = 0u; *done = 1u; }
    words[0] = pos;
    words[1] = action & 1u;
    walk_obs(k, pos, words[1], obs);
    return reward;
}

void walk_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    walk_obs(&WALK_KNOBS[instance], words[0], words[1], obs);
}

void walk_pose(uint32_t instance, device const uint32_t *words, thread const float *ctrl, thread float *pose) {
    (void)ctrl;
    constant const walkKnobs *k = &WALK_KNOBS[instance];
    poseZero(pose);
    pose[0] = (float)words[0];
    pose[1] = (float)k->limit;
    pose[2] = k->lastActionObs ? (float)words[1] : 0.0f;
}

void walk_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
