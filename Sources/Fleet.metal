#pragma METAL fp math_mode(safe)

constant uint OAK_DEATH_NONE       = 0xffffffffu;
constant uint OAK_DET_ENV_REWARD   = 1u;
constant uint OAK_DET_POLICY_MEAN  = 2u;
constant uint OAK_DET_ENV_OBS      = 3u;

struct KernelScalars {
    uint stepCount;
    uint stateSlot;
    float averageReward;
};

struct OakFleetEnvCfg {
    uint laneCount;
    uint episodeLen;
    uint recordCapacity;
    float invObsFloor;
};

struct OakFleetLane {
    uint elapsed, resets;
    uint steps;
    float rewardSum, rewardSumCompensation, averageReward;
    uint deathStep, deathDetector;

    uint envKind;
    uint envObservationCount, envActionCount;
};
static_assert(sizeof(OakFleetLane) == 44, "OakFleetLane layout drifted from the Swift mirror");

inline device uint *laneEnvState(device uint *envArena, device const uint *envOff, uint gid) {
    return envArena + envOff[gid];
}

inline device float *lanePendingAction(device uint *envState) {
    return (device float *)envState;
}

inline device uint *laneEnvWords(device uint *envState, uint envK) {
    return envState + OAK_ENV_SPECS[envK].actionCount;
}

inline device float *laneEnvObs(device uint *envState, uint envK) {
    return (device float *)(envState + OAK_ENV_SPECS[envK].actionCount
                            + OAK_ENV_SPECS[envK].envStateWords);
}

inline bool laneIsMj(uint envK) { return OAK_ENV_SPECS[envK].mjIndex != 0xffffffffu; }

inline bool laneIsMjEngine(uint envK) {
    return OAK_ENV_SPECS[envK].mjIndex != 0xffffffffu && envK != OAK_ENV_CHEETAH;
}

struct OakLaneSnapshot {
    KernelScalars scalars;
    float paramAbsMax;
    uint paramNonFinite;
    float rewardSum, rho;
    uint deathStep, deathDetector;
};
static_assert(sizeof(OakLaneSnapshot) == 36, "OakLaneSnapshot layout drifted from the Swift mirror");

constant uint OAK_DIAG_K = 64u;
constant uint OAK_DIAG_GRAM = OAK_DIAG_K * OAK_DIAG_K;
constant uint OAK_DIAG_PPO_PARAMS =
    ((2u * OAK_ENV_OBS_MAX * OAK_DIAG_K + 2u * OAK_DIAG_K + OAK_DIAG_K * OAK_ENV_ACT_MAX
      + 2u * OAK_ENV_ACT_MAX + OAK_DIAG_K + 1u) + 3u) & ~3u;

inline uint oakDiagFloats(uint rank) { return 5u * rank + 3u + OAK_DIAG_GRAM; }

struct OakDiagAccum {
    float gram[OAK_DIAG_GRAM];
    float sumAbsHiddenActor[OAK_DIAG_K];
    float sumAbsHiddenCritic[OAK_DIAG_K];
};
static_assert(sizeof(OakDiagAccum) == 16896, "OakDiagAccum layout drifted from the Swift mirror");

struct PpoDiagRecord {
    float sumAbsHiddenActor[OAK_DIAG_K];
    float sumAbsHiddenCritic[OAK_DIAG_K];
    float weightAbsMean;
    float adamT;
    float adamV[OAK_DIAG_PPO_PARAMS];
    float gram[OAK_DIAG_GRAM];
};

constant uint OAK_DIAG_ACCUM_FLOATS = sizeof(OakDiagAccum) / 4u;
constant uint OAK_DIAG_OUT_FLOATS = sizeof(PpoDiagRecord) / 4u;

inline float finiteOrZero(float x) { return isfinite(x) ? x : 0.0f; }

inline void kahanAdd(device float *sum, device float *comp, float x) {
    float y = x - *comp;
    float t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

inline void resetLane(device OakFleetLane *lane, device uint *envState,
                      constant uint *mjModels,
                      uint storedEnvKind, uint laneSeed) {
    lane->envKind = storedEnvKind;
    lane->envObservationCount = OAK_ENV_SPECS[storedEnvKind].obsDimNatural;
    lane->envActionCount = OAK_ENV_SPECS[storedEnvKind].actionCount;
    uint elapsed = 0u, resets = 0u;
    uint actionCount = OAK_ENV_SPECS[storedEnvKind].actionCount;
    device float *pending = lanePendingAction(envState);
    for (uint u = 0u; u < actionCount; u++) pending[u] = 0.0f;
    device uint *stateWords = laneEnvWords(envState, storedEnvKind);
    if (laneIsMj(storedEnvKind)) {
        mj_reset(storedEnvKind, mjModels, stateWords, laneSeed, elapsed, resets, 0u);
    } else {
        uint envWords[OAK_ENV_WORDS_MAX];
        env_reset(storedEnvKind, laneSeed, envWords, elapsed, resets);
        for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) stateWords[i] = envWords[i];
    }
    lane->elapsed = elapsed; lane->resets = resets;
    lane->steps = 0u;
    lane->rewardSum = 0.0f; lane->rewardSumCompensation = 0.0f; lane->averageReward = 0.0f;
    lane->deathStep = OAK_DEATH_NONE; lane->deathDetector = 0u;
}

inline void observeLane(uint envK, constant uint *mjModels, device uint *envState,
                        device float *obsOut) {
    if (laneIsMj(envK)) {
        mj_observe(envK, mjModels, laneEnvWords(envState, envK), obsOut, 0u);
        return;
    }
    uint envWords[OAK_ENV_WORDS_MAX];
    device const uint *stateWords = laneEnvWords(envState, envK);
    for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) envWords[i] = stateWords[i];
    float obs[OAK_ENV_CLASSIC_OBS_MAX];
    env_observe(envK, envWords, obs);
    for (uint i = 0u; i < OAK_ENV_SPECS[envK].obsDimNatural; i++) obsOut[i] = obs[i];
}

inline float advanceEnvironment(device OakFleetLane *lane, device uint *envState,
                                constant uint *mjModels, device float *mjScratch,
                                uint envK, uint laneSeed,
                                constant OakFleetEnvCfg &ecfg,
                                thread float *ctrl, device float *obsOut,
                                thread uint &done, uint slane, bool wide) {
    uint actionCount = OAK_ENV_SPECS[envK].actionCount;
    device const float *pending = lanePendingAction(envState);
    for (uint u = 0u; u < OAK_ENV_ACT_MAX; u++) ctrl[u] = (u < actionCount) ? pending[u] : 0.0f;
    device uint *stateWords = laneEnvWords(envState, envK);
    if (laneIsMjEngine(envK)) {
        uint elapsed = lane->elapsed, resets = lane->resets;
        done = 0u;
        float reward = mj_step_env(envK, mjModels, stateWords, mjScratch, laneSeed, ctrl,
                                   elapsed, resets, ecfg.episodeLen, obsOut, done, slane);
        if (slane == 0u) {
            lane->elapsed = elapsed;
            lane->resets = resets;
        }
        return reward;
    }
    float reward = 0.0f;
    if (slane == 0u) {
        if (laneIsMj(envK)) {
            uint elapsed = lane->elapsed, resets = lane->resets;
            done = 0u;
            reward = mj_step_env(envK, mjModels, stateWords, mjScratch, laneSeed, ctrl,
                                 elapsed, resets, ecfg.episodeLen, obsOut, done, 0u);
            lane->elapsed = elapsed;
            lane->resets = resets;
        } else {
            uint envWords[OAK_ENV_WORDS_MAX];
            for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) envWords[i] = stateWords[i];
            float obs[OAK_ENV_CLASSIC_OBS_MAX];
            reward = env_step(envK, laneSeed, ctrl, envWords, obs, done);
            for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) stateWords[i] = envWords[i];
            for (uint i = 0u; i < OAK_ENV_SPECS[envK].obsDimNatural; i++) obsOut[i] = obs[i];
        }
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
        kahanAdd(&lane->rewardSum, &lane->rewardSumCompensation, reward);
        float rate = max(1.0f / float(stepNum), ecfg.invObsFloor);
        lane->averageReward += rate * (reward - lane->averageReward);
    } else if (lane->deathStep == OAK_DEATH_NONE) {
        lane->deathStep = stepNum;
        lane->deathDetector = OAK_DET_ENV_REWARD;
    }
}

inline void recordPose(uint envK, constant uint *mjModels, device uint *envState,
                       thread const float *ctrl, uint stepNum, uint done,
                       constant OakFleetEnvCfg &ecfg,
                       device const int *recordSlot, device float *recordBuf, uint gid) {
    if (ecfg.recordCapacity != 0u) {
        int slot = recordSlot[gid];
        if (slot >= 0) {
            float pose[OAK_POSE_LEN];
            if (laneIsMj(envK)) {
                mj_pose(envK, mjModels, laneEnvWords(envState, envK), ctrl, pose, 0u);
            } else {
                uint envWords[OAK_ENV_WORDS_MAX];
                device const uint *stateWords = laneEnvWords(envState, envK);
                for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) envWords[i] = stateWords[i];
                env_pose(envK, envWords, ctrl, pose);
            }
            ulong lanes = (ulong)ecfg.laneCount;
            device float *dst = recordBuf
                + ((stepNum - 1u) % ecfg.recordCapacity) * (OAK_RECORD_FLOATS * lanes)
                + (ulong)slot;
            dst[0] = float(done);
            for (uint i = 0u; i < OAK_POSE_LEN; i++) dst[(i + 1u) * lanes] = pose[i];
        }
    }
}
