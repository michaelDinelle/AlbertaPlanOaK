#include "../env.h"

typedef struct { float slip; uint32_t positionOnly; } mountaincarKnobs;

static const mountaincarKnobs MOUNTAINCAR_KNOBS[3] = {
    { 0.0f, 0u },
    { 0.2f, 0u },
    { 0.0f, 1u }
};


static void mountainCar_obs(const mountaincarKnobs *k, uint32_t seed, uint32_t step, float pos, float vel, float *obs) {
    if (k->positionOnly) { obs[0] = pos + 0.01f * normalFrom(philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u).x, philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u).y); obs[1] = 0.0f; }
    else { obs[0] = pos; obs[1] = vel; }
    obs[2] = 1.0f;
}

void mountainCar_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    (void)instance;
    (void)seed;
    words[0] = floatWord(-0.5f);
}

float mountainCar_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut, const float *shared) {
    (void)shared;
    (void)extra;
    const mountaincarKnobs *k = &MOUNTAINCAR_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0];
    if (k->slip > 0.0f) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
        if (u01(r.x) < k->slip) action = r.y % 3u;
    }
    float pos = wordFloat(words[0]), vel = wordFloat(words[1]);
    float force = (float)((int)(action % 3u)) - 1.0f;
    vel += force * 0.001f - 0.0025f * oak_cos(3.0f * pos);
    vel = oak_clamp(vel, -0.07f, 0.07f);
    pos += vel;
    pos = oak_clamp(pos, -1.2f, 0.6f);
    if (pos == -1.2f && vel < 0.0f) vel = 0.0f;
    float reward = 0.0f;
    uint32_t done = 0u;
    if (pos >= 0.5f) {
        reward = 1.0f;
        done = 1u;
        pos = -0.5f; vel = 0.0f;
    }
    words[0] = floatWord(pos); words[1] = floatWord(vel);
    mountainCar_obs(k, seed, step, pos, vel, obs);
    *doneOut = done;
    return reward;
}

void mountainCar_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)extra;
    mountainCar_obs(&MOUNTAINCAR_KNOBS[instance], seed, step, wordFloat(words[0]), wordFloat(words[1]), obs);
}

void mountainCar_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose) {
    (void)instance;
    poseZero(pose);
    pose[0] = wordFloat(words[0]); pose[1] = wordFloat(words[1]); pose[2] = 0.5f; pose[3] = ctrl[0];
}

void mountainCar_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = wordFloat(words[1]) < 0.0f ? 0.0f : 2.0f;
}
