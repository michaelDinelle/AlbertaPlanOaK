#ifndef OAK_ENV_METAL_INCLUDED
#define OAK_ENV_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)

constant uint OAK_ENV_CHEETAH     = 0u;
constant uint OAK_ENV_FOURROOMS   = 1u;
constant uint OAK_ENV_DEEPSEA     = 2u;
constant uint OAK_ENV_KEYCORRIDOR = 3u;
constant uint OAK_ENV_ACROBOT     = 4u;
constant uint OAK_ENV_MOUNTAINCAR = 5u;
constant uint OAK_ENV_CARTPOLE    = 6u;
constant uint OAK_ENV_PENDULUM    = 7u;
constant uint OAK_ENV_CHAIN       = 8u;
constant uint OAK_ENV_LARGEDEEPSEA = 9u;
constant uint OAK_ENV_MULTIKEYCORRIDOR = 10u;
constant uint OAK_ENV_MEMORYCHAIN = 11u;
constant uint OAK_ENV_PROCEDURALMAZE = 12u;
constant uint OAK_ENV_HIGHDIMMAZE = 13u;
constant uint OAK_ENV_POMAZE      = 14u;
constant uint OAK_ENV_DELAYEDCHAIN = 15u;
constant uint OAK_ENV_COUNT       = 16u;

struct OakEnvSpec { uint nameHash; uint obsDimNatural; uint actionKind; uint actionCount; };
constant OakEnvSpec OAK_ENV_SPECS[OAK_ENV_COUNT] = {
    { 0x9d7fd44bu, 17u, 0u, 6u },
    { 0x8014a187u, 17u, 1u, 4u },
    { 0x56af5bf2u,  8u, 1u, 2u },
    { 0xdd537ebau, 14u, 1u, 4u },
    { 0x93531833u,  7u, 1u, 3u },
    { 0xf5b3ddc8u,  3u, 1u, 3u },
    { 0x41f9937du,  5u, 1u, 2u },
    { 0x69bbbb73u,  4u, 0u, 1u },
    { 0x7b267422u,  2u, 1u, 2u },
    { 0xb96ddc3du,  8u, 1u, 2u },
    { 0x4c3c2d3fu, 17u, 1u, 4u },
    { 0xcf293fd7u,  6u, 1u, 2u },
    { 0xd75dd5f9u, 17u, 1u, 4u },
    { 0xbfeb2b7au, 16u, 1u, 6u },
    { 0x0582e383u, 15u, 1u, 4u },
    { 0x56184b96u,  3u, 1u, 2u },
};

inline void env_reset(uint envKind, uint seed,
                      thread uint *words, thread HalfCheetahState &cheetah,
                      thread uint &elapsed, thread uint &resets) {
    for (int i = 0; i < 10; i++) words[i] = 0u;
    if (envKind != OAK_ENV_CHEETAH) halfCheetah_clear(cheetah);
    if (envKind == OAK_ENV_FOURROOMS) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
        words[4] = 0u; words[5] = 0u;
    } else if (envKind == OAK_ENV_DEEPSEA) {
        uint4 rm = hc_philox4x32(uint2(seed, HC_RNG_STREAM_RESET),
                                 uint4(0u, 0u, 0u, 0u));
        words[0] = 0u; words[1] = 0u; words[2] = 0u;
        words[3] = rm.x; words[4] = rm.y; words[5] = 0u;
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 3u; words[3] = 5u;
        words[4] = 7u; words[5] = 1u;
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        words[0] = as_type<uint>(-0.5f);
    } else if (envKind == OAK_ENV_PENDULUM) {
        words[0] = as_type<uint>(M_PI_F);
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        uint4 rm0 = hc_philox4x32(uint2(seed, HC_RNG_STREAM_RESET),
                                  uint4(0u, 0u, 0u, 0u));
        uint4 rm1 = hc_philox4x32(uint2(seed, HC_RNG_STREAM_RESET),
                                  uint4(1u, 0u, 0u, 0u));
        words[0] = 0u;
        words[1] = rm0.x; words[2] = rm0.y;
        words[3] = rm0.z; words[4] = rm0.w;
        words[5] = rm1.x; words[6] = rm1.y;
        words[7] = rm1.z; words[8] = rm1.w;
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 3u | (5u << 4u);
        words[3] = 7u | (5u << 4u);
        words[4] = 11u | (5u << 4u);
        words[5] = 15u; words[6] = 1u;
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        uint mazeWalls[3];
        proceduralMaze_generate(seed, mazeWalls);
        words[6] = mazeWalls[0]; words[7] = mazeWalls[1]; words[8] = mazeWalls[2];
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
    } else if (envKind == OAK_ENV_HIGHDIMMAZE || envKind == OAK_ENV_POMAZE) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
        words[6] = 1u;
    } else if (envKind == OAK_ENV_CHEETAH) {
        halfCheetah_reset(cheetah, seed, elapsed, resets);
    }
}

inline void env_observe(uint envKind, thread const uint *words, thread const HalfCheetahState &cheetah,
                        thread float *obs) {
    if (envKind == OAK_ENV_FOURROOMS) {
        fourRooms_obs(1u, 1u, 9u, 9u, 0u, obs);
    } else if (envKind == OAK_ENV_DEEPSEA) {
        deepSea_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        keyCorridor_obs(1u, 1u, 3u, 5u, 7u, 1u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_ACROBOT) {
        acrobot_obs(0.0f, 0.0f, 0.0f, 0.0f, obs);
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        mountainCar_obs(-0.5f, 0.0f, obs);
    } else if (envKind == OAK_ENV_CARTPOLE) {
        cartPole_obs(0.0f, 0.0f, 0.0f, 0.0f, obs);
    } else if (envKind == OAK_ENV_PENDULUM) {
        pendulum_obs(M_PI_F, 0.0f, obs);
    } else if (envKind == OAK_ENV_CHAIN) {
        obs[0] = 0.0f; obs[1] = 1.0f;
        for (int i = 2; i < 17; i++) obs[i] = 0.0f;
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        largeDeepSea_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        multiKeyCorridor_obs(1u, 1u, 3u | (5u << 4u), 7u | (5u << 4u), 11u | (5u << 4u),
                   15u, 1u, 0u, obs);
    } else if (envKind == OAK_ENV_MEMORYCHAIN) {
        memoryChain_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        proceduralMaze_obs(1u, 1u, 9u, 9u, 0u, words[6], words[7], words[8], obs);
    } else if (envKind == OAK_ENV_HIGHDIMMAZE) {
        highDimMaze_obs(1u, 1u, 9u, 9u, obs);
    } else if (envKind == OAK_ENV_POMAZE) {
        poMaze_obs(1u, 1u, 9u, 9u, obs);
    } else if (envKind == OAK_ENV_DELAYEDCHAIN) {
        obs[0] = 0.0f; obs[1] = 0.0f; obs[2] = 1.0f;
        for (int i = 3; i < 17; i++) obs[i] = 0.0f;
    } else {
        for (uint i = 0u; i + 1u < 9u; ++i) obs[i] = cheetah.qpos[i + 1u];
        for (uint i = 0u; i < 9u; ++i) obs[8u + i] = cheetah.qvel[i];
    }
}

inline float env_step(uint envKind, uint seed, thread const float *ctrl,
                      thread uint *words, thread HalfCheetahState &cheetah,
                      thread uint &elapsed, thread uint &resets,
                      uint episodeLen,
                      thread float *obs, thread uint &done) {
    done = 0u;
    if (envKind == OAK_ENV_CHEETAH) {
        float reward;
        if (episodeLen != 0u && elapsed >= episodeLen) {
            halfCheetah_reset(cheetah, seed, elapsed, resets);
            reward = 0.0f;
            done = 0u;
        } else {
            float xBefore = cheetah.qpos[0];
            for (int u = 0; u < 6; u++) cheetah.ctrl[u] = ctrl[u];
            for (int k = 0; k < HC_FRAME_SKIP; k++) halfCheetah_step(cheetah);
            float forwardVelocity = (cheetah.qpos[0] - xBefore) / (float(HC_FRAME_SKIP) * HC_TIMESTEP);
            float ctrlCost = 0.0f;
            for (int u = 0; u < 6; u++) ctrlCost += 0.1f * ctrl[u] * ctrl[u];
            reward = forwardVelocity * 1.0f - ctrlCost;
            elapsed++;
            done = (episodeLen != 0u && elapsed >= episodeLen) ? 1u : 0u;
        }
        for (int i = 1; i < 9; i++) obs[i - 1] = cheetah.qpos[i];
        for (int i = 0; i < 9; i++) obs[8 + i] = cheetah.qvel[i];
        return reward;
    }
    uint discreteAction = uint(ctrl[0]);
    if (envKind == OAK_ENV_FOURROOMS) return fourRooms_step(words, seed, discreteAction, obs);
    if (envKind == OAK_ENV_DEEPSEA) return deepSea_step(words, discreteAction, obs);
    if (envKind == OAK_ENV_KEYCORRIDOR) return keyCorridor_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_CHAIN) return chain_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_LARGEDEEPSEA) return largeDeepSea_step(words, discreteAction, obs);
    if (envKind == OAK_ENV_MULTIKEYCORRIDOR) return multiKeyCorridor_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_MEMORYCHAIN) return memoryChain_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_PROCEDURALMAZE) return proceduralMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_HIGHDIMMAZE) return highDimMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_POMAZE) return poMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_DELAYEDCHAIN) return delayedChain_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_ACROBOT) return acrobot_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_MOUNTAINCAR) return mountainCar_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_CARTPOLE) return cartPole_step(words, discreteAction, obs, &done);
    return pendulum_step(words, ctrl[0], obs);
}

inline void env_pose(uint envKind, thread const uint *words, thread const HalfCheetahState &cheetah,
                     thread const float *ctrl, thread float *pose) {
    if (envKind == OAK_ENV_FOURROOMS) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_DEEPSEA) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = 7.0f; pose[3] = 7.0f;
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[4]); pose[3] = float(words[5]);
        pose[4] = float(words[2]); pose[5] = float(words[3]);
        pose[6] = float(words[6] & 1u); pose[7] = float((words[6] >> 1u) & 1u);
        pose[8] = 0.0f;
    } else if (envKind == OAK_ENV_CHAIN) {
        pose[0] = float(words[0]); pose[1] = 15.0f;
        for (int i = 2; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        pose[0] = float(words[0] & 15u); pose[1] = float((words[0] >> 4u) & 15u);
        pose[2] = 15.0f; pose[3] = 15.0f;
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[5]); pose[3] = float(words[6]);
        pose[4] = float(words[7] & 7u);
        pose[5] = float((words[7] >> 3u) & 7u);
        for (int i = 6; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MEMORYCHAIN) {
        pose[0] = float(words[0]); pose[1] = 31.0f;
        pose[2] = float(words[1]);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_HIGHDIMMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        pose[4] = float(words[6]);
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_POMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_DELAYEDCHAIN) {
        pose[0] = float(words[0]); pose[1] = 31.0f;
        pose[2] = float(words[1]);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_ACROBOT) {
        float theta1 = as_type<float>(words[0]), theta2 = as_type<float>(words[1]);
        pose[0] = theta1; pose[1] = theta2;
        pose[2] = as_type<float>(words[2]); pose[3] = as_type<float>(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        pose[0] = as_type<float>(words[0]); pose[1] = as_type<float>(words[1]);
        pose[2] = 0.5f;
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_CARTPOLE) {
        pose[0] = as_type<float>(words[0]); pose[1] = as_type<float>(words[1]);
        pose[2] = as_type<float>(words[2]); pose[3] = as_type<float>(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_PENDULUM) {
        float theta = as_type<float>(words[0]), thetaDot = as_type<float>(words[1]);
        pose[0] = theta; pose[1] = thetaDot; pose[2] = clamp(2.0f * ctrl[0], -2.0f, 2.0f);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else {
        for (int i = 0; i < 9; i++) pose[i] = cheetah.qpos[i];
    }
}

#endif
