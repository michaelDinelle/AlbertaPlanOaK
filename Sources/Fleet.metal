#pragma METAL fp math_mode(safe)

constant uint OAK_DEATH_NONE = 0xffffffffu;
constant uint OAK_DET_ENV_REWARD = 1u;
constant uint OAK_DET_ENV_OBS = 3u;

struct OakFleetEnvCfg {
    uint laneCount;
    uint recordCapacity;
    uint persistentMask;
    float invObsFloor;
};

struct OakFleetLane {
    uint elapsed, resets;
    uint steps;
    float rewardSum, rewardSumCompensation, averageReward;
    uint deathStep, deathDetector;

    uint envKind;
};

inline device uint *laneEnvWords(device uint *envState, uint envK) {
    return envState + OAK_ENV_SPECS[envK].actionCount;
}

inline device float *laneEnvObs(device uint *envState, uint envK) {
    return (device float *)(envState + OAK_ENV_SPECS[envK].actionCount + OAK_ENV_SPECS[envK].envStateWords);
}

inline device uint *lanePersistent(device uint *envState, uint envK) {
    uint words = OAK_ENV_SPECS[envK].actionCount + OAK_ENV_SPECS[envK].envStateWords +
                 OAK_ENV_SPECS[envK].obsDimNatural;
    return envState + ((words + 3u) & ~3u);
}

inline bool laneIsMj(uint envK) {
    return OAK_ENV_SPECS[envK].mjIndex != 0xffffffffu;
}

struct OakLaneSnapshot {
    uint paramNonFinite;
    float rewardSum, rho;
    uint deathStep, deathDetector;
};

inline float finiteOrZero(float x) {
    return isfinite(x) ? x : 0.0f;
}

constant uint OAK_MEGA_STAGE_MASK [[function_constant(4)]];
constant uint OAK_MEGA_STAGES =
    is_function_constant_defined(OAK_MEGA_STAGE_MASK) ? OAK_MEGA_STAGE_MASK : 0xffffffffu;

inline void laneFence() {
    simdgroup_barrier(mem_flags::mem_device);
}

inline void resetLane(device OakFleetLane *lane, device uint *envState, constant uint *mjModels, uint kind,
                      uint seed) {
    lane->envKind = kind;
    lane->deathStep = OAK_DEATH_NONE;
    uint elapsed = 0u, resets = 0u;
    device uint *words = laneEnvWords(envState, kind);
    if (laneIsMj(kind))
        mj_reset(kind, mjModels, words, seed, elapsed, resets, 0u);
    else if (kind != OAK_ENV_JELLYBEANWORLD)
        env_reset(kind, seed, words, elapsed, resets);
    lane->elapsed = elapsed;
    lane->resets = resets;
}

inline void observeLane(uint envK, uint laneSeed, constant uint *mjModels, device const float *envConstants,
                        constant OakFleetEnvCfg &ecfg, device uint *envState, device float *obsOut) {
    if (envK == OAK_ENV_JELLYBEANWORLD) {
        jbw_observe(laneSeed, laneEnvWords(envState, envK), lanePersistent(envState, envK),
                    ecfg.persistentMask, envConstants, 0u, 0u, 1u, obsOut);
        return;
    }
    if (laneIsMj(envK)) {
        mj_observe(envK, mjModels, laneEnvWords(envState, envK), obsOut, 0u);
        return;
    }
    float obs[OAK_ENV_OBS_MAX];
    env_observe(envK, laneEnvWords(envState, envK), obs);
    for (uint i = 0u; i < OAK_ENV_SPECS[envK].obsDimNatural; i++) obsOut[i] = obs[i];
}

inline float advanceEnvironment(device OakFleetLane *lane, device uint *envState, constant uint *mjModels,
                                device float *mjScratch, device const float *envConstants, uint envK,
                                uint laneSeed, constant OakFleetEnvCfg &ecfg, thread float *ctrl,
                                device float *obsOut, thread uint &done, uint slane, bool wide) {
    uint actionCount = OAK_ENV_SPECS[envK].actionCount;
    device const float *pending = ((device float *)envState);
    for (uint u = 0u; u < OAK_ENV_ACT_MAX; u++) ctrl[u] = (u < actionCount) ? pending[u] : 0.0f;
    device uint *stateWords = laneEnvWords(envState, envK);
    if (envK == OAK_ENV_JELLYBEANWORLD) {
        done = 0u;
        return jbw_step(laneSeed, uint(ctrl[0]), stateWords, lanePersistent(envState, envK),
                        ecfg.persistentMask, envConstants, lane->steps, slane, wide ? 32u : 1u, obsOut);
    }
    if (laneIsMj(envK)) {
        uint elapsed = lane->elapsed, resets = lane->resets;
        done = 0u;
        float mjReward = mj_step_env(envK, mjModels, stateWords, mjScratch, laneSeed, ctrl, elapsed, resets,
                                     obsOut, done, slane);
        if (slane == 0u) {
            lane->elapsed = elapsed;
            lane->resets = resets;
        }
        return mjReward;
    }
    float reward = 0.0f;
    if (slane == 0u) {
        float obs[OAK_ENV_OBS_MAX];
        reward = env_step(envK, laneSeed, ctrl, stateWords, obs, done);
        for (uint i = 0u; i < OAK_ENV_SPECS[envK].obsDimNatural; i++) obsOut[i] = obs[i];
    }
    if (wide) {
        reward = simd_broadcast_first(reward);
        done = simd_broadcast_first(done);
    }
    return reward;
}

inline void accumulateReward(device OakFleetLane *lane, float reward, uint stepNum,
                             constant OakFleetEnvCfg &ecfg) {
    if (isfinite(reward)) {
        float adjusted = reward - lane->rewardSumCompensation;
        float sum = lane->rewardSum + adjusted;
        lane->rewardSumCompensation = (sum - lane->rewardSum) - adjusted;
        lane->rewardSum = sum;
        float rate = max(1.0f / float(stepNum), ecfg.invObsFloor);
        lane->averageReward += rate * (reward - lane->averageReward);
    } else if (lane->deathStep == OAK_DEATH_NONE) {
        lane->deathStep = stepNum;
        lane->deathDetector = OAK_DET_ENV_REWARD;
    }
}

inline void recordPose(uint kind, constant uint *models, device uint *state, thread const float *control,
                       uint step, uint done, float reward, constant OakFleetEnvCfg &config,
                       device float *records, uint lane) {
    float frame[OAK_RECORD_FLOATS] = {float(done)};
    thread float *pose = frame + 1;
    if (kind == OAK_ENV_JELLYBEANWORLD) {
        device const uint *words = laneEnvWords(state, kind);
        pose[0] = float(as_type<int>(words[0]));
        pose[1] = float(as_type<int>(words[1]));
        pose[2] = float(words[2]);
        for (uint i = 0u; i < 4u; i++) pose[3u + i] = float(words[4u + i]);
    } else if (laneIsMj(kind)) {
        mj_pose(kind, models, laneEnvWords(state, kind), control, pose, 0u);
    } else {
        env_pose(kind, laneEnvWords(state, kind), control, pose);
    }
    frame[OAK_RECORD_FLOATS - 1u] = reward;
    ulong offset =
        ((step - 1u) % config.recordCapacity) * (OAK_RECORD_FLOATS * ulong(config.laneCount)) + lane;
    for (uint field = 0; field < OAK_RECORD_FLOATS; field++) {
        records[offset + field * ulong(config.laneCount)] = frame[field];
    }
}

inline void writeLaneSnapshot(device OakLaneSnapshot *snapshot, device const OakFleetLane *lane,
                              uint nonFinite) {
    *snapshot = {nonFinite, lane->rewardSum, lane->averageReward, lane->deathStep, lane->deathDetector};
}

inline uint baselineWord(uint seed, uint stream, uint stepIdx, uint i) {
    uint4 r = philox4x32(uint2(seed, stream), uint4(stepIdx, i >> 2u, 0u, 0u));
    return r[i & 3u];
}

kernel void fleet_snapshot(device OakFleetLane *lanes [[buffer(6)]],
                                  constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                                  device OakLaneSnapshot *snaps [[buffer(9)]],
                                  constant uint &snapBase [[buffer(10)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device OakLaneSnapshot *snap = &snaps[snapBase + gid];
    writeLaneSnapshot(snap, lane, 0u);
}
