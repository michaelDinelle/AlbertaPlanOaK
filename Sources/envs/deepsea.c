#include "../env.h"

typedef struct { int extent; uint32_t mapWords; float moveCost, slip; } deepseaKnobs;

OAK_CONSTANT deepseaKnobs DEEPSEA_KNOBS[4] = {
    { 7, 2u, -0.00125f, 0.0f },
    { 15, 8u, -0.000625f, 0.0f },
    { 31, 32u, -0.0003125f, 0.0f },
    { 15, 8u, -0.000625f, 0.05f }
};


static void deepsea_obs(constant const deepseaKnobs *k, uint32_t row, uint32_t column, uint32_t lastAction, device float *obs) {
    float gridExtent = (float)k->extent;
    obs[0] = (float)row / gridExtent;
    obs[1] = (float)column / gridExtent;
    obs[2] = (float)(k->extent - (int)column) / gridExtent;
    obs[3] = (float)(k->extent - (int)row) / gridExtent;
    obs[4] = (column == (uint32_t)k->extent) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = (float)lastAction;
    obs[7] = 1.0f;
}

static uint32_t deepsea_rightward(constant const deepseaKnobs *k, const device uint32_t *words) {
    uint32_t mapBitIndex = words[0] * (uint32_t)(k->extent + 1) + words[1];
    return (words[3u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) & 1u;
}

void deepsea_reset(uint32_t instance, uint32_t seed, device uint32_t *words, device uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)extra;
    constant const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    words[0] = 0u; words[1] = 0u; words[2] = 0u;
    for (uint32_t block = 0u; block * 4u < k->mapWords; block++) {
        u4 rm = philox4x32(seed, RNG_STREAM_RESET, block, 0u, 0u, 0u);
        uint32_t w[4] = { rm.x, rm.y, rm.z, rm.w };
        for (uint32_t i = 0u; i < 4u && block * 4u + i < k->mapWords; i++) words[3u + block * 4u + i] = w[i];
    }
}

float deepsea_step(uint32_t instance, uint32_t seed, uint32_t step, thread const float *ctrl, device uint32_t *words, device uint32_t *extra, device float *obs, thread uint32_t *done, constant const float *shared) {
    (void)shared;
    (void)extra;
    constant const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    *done = 0u;
    uint32_t action = (uint32_t)ctrl[0];
    if (k->slip > 0.0f && u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x) < k->slip) action ^= 1u;
    uint32_t row = words[0], column = words[1];
    uint32_t rightwardAction = deepsea_rightward(k, words);
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = (int)column + columnDelta;
    if (nextColumn < 0) nextColumn = 0;
    if (nextColumn > k->extent) nextColumn = k->extent;
    uint32_t nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? k->moveCost : 0.0f;
    uint32_t nextLastAction = action & 1u;
    if (nextRow >= (uint32_t)k->extent) {
        if (nextColumn == k->extent) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    words[0] = nextRow; words[1] = (uint32_t)nextColumn; words[2] = nextLastAction;
    deepsea_obs(k, nextRow, (uint32_t)nextColumn, nextLastAction, obs);
    return reward;
}

void deepsea_observe(uint32_t instance, uint32_t seed, device const uint32_t *words, device const uint32_t *extra, uint32_t step, device float *obs, constant const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    deepsea_obs(&DEEPSEA_KNOBS[instance], words[0], words[1], words[2], obs);
}

void deepsea_pose(uint32_t instance, device const uint32_t *words, thread const float *ctrl, thread float *pose) {
    constant const deepseaKnobs *k = &DEEPSEA_KNOBS[instance];
    poseZero(pose);
    pose[0] = (float)words[0]; pose[1] = (float)words[1]; pose[2] = (float)k->extent; pose[3] = (float)k->extent; pose[4] = ctrl[0];
}

void deepsea_oracle(uint32_t instance, uint32_t seed, uint32_t step, device const uint32_t *words, device const uint32_t *extra, device const float *obs, thread float *ctrl, constant const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)deepsea_rightward(&DEEPSEA_KNOBS[instance], words);
}
