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
    HalfCheetahState halfCheetah;
    uint elapsed, resets;
    uint steps;
    float pendingAction[6];
    float rewardSum, rewardSumCompensation, averageReward;
    uint deathStep, deathDetector;

    uint envKind;
    uint envWords[10];
    uint envObservationCount, envActionCount;
};
static_assert(sizeof(OakFleetLane) == 240, "OakFleetLane layout drifted from the Swift mirror");

struct OakLaneSnapshot {
    KernelScalars scalars;
    float paramAbsMax;
    uint paramNonFinite;
    float rewardSum, rho;
    uint deathStep, deathDetector;
};
static_assert(sizeof(OakLaneSnapshot) == 36, "OakLaneSnapshot layout drifted from the Swift mirror");

constant uint OAK_DIAG_RANK = 480u;
constant uint OAK_DIAG_K = 64u;
constant uint OAK_DIAG_GRAM = OAK_DIAG_K * OAK_DIAG_K;
constant uint OAK_DIAG_PPO_PARAMS = 2768u;

struct OakDiagAccum {
    float gram[OAK_DIAG_GRAM];
    float sumAbsHiddenActor[OAK_DIAG_K];
    float sumAbsHiddenCritic[OAK_DIAG_K];
};
static_assert(sizeof(OakDiagAccum) == 16896, "OakDiagAccum layout drifted from the Swift mirror");

struct OakDiagRecord {
    float featureAge[OAK_DIAG_RANK];
    float featureUtility[OAK_DIAG_RANK];
    float valueAlpha[OAK_DIAG_RANK];
    float optionValueAlpha[OAK_DIAG_RANK];
    float modelAlpha[OAK_DIAG_RANK];
    float featureBirths;
    float weightAbsMean;
    float weightAbsMeanValue;
    float gram[OAK_DIAG_GRAM];
};
static_assert(sizeof(OakDiagRecord) == 25996, "OakDiagRecord layout drifted from the Swift mirror");

struct PpoDiagRecord {
    float sumAbsHiddenActor[OAK_DIAG_K];
    float sumAbsHiddenCritic[OAK_DIAG_K];
    float weightAbsMean;
    float adamT;
    float adamV[OAK_DIAG_PPO_PARAMS];
    float gram[OAK_DIAG_GRAM];
};
static_assert(sizeof(PpoDiagRecord) == 27976, "PpoDiagRecord layout drifted from the Swift mirror");
static_assert(sizeof(OakDiagRecord) <= sizeof(PpoDiagRecord), "diagOut stride is sized by PpoDiagRecord");

constant uint OAK_DIAG_ACCUM_FLOATS = sizeof(OakDiagAccum) / 4u;
constant uint OAK_DIAG_OUT_FLOATS = sizeof(PpoDiagRecord) / 4u;

inline float finiteOrZero(float x) { return isfinite(x) ? x : 0.0f; }

inline void kahanAdd(device float *sum, device float *comp, float x) {
    float y = x - *comp;
    float t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

inline void resetLane(device OakFleetLane *lane, uint storedEnvKind, uint laneSeed,
                      thread uint *envWords, thread HalfCheetahState &halfCheetah) {
    lane->envKind = storedEnvKind;
    lane->envObservationCount = OAK_ENV_SPECS[storedEnvKind].obsDimNatural;
    lane->envActionCount = OAK_ENV_SPECS[storedEnvKind].actionCount;
    uint elapsed = 0u, resets = 0u;
    env_reset(storedEnvKind, laneSeed, envWords, halfCheetah, elapsed, resets);
    for (int i = 0; i < 10; i++) lane->envWords[i] = envWords[i];
    lane->halfCheetah = halfCheetah;
    lane->elapsed = elapsed; lane->resets = resets;
    lane->steps = 0u;
    for (int u = 0; u < 6; u++) lane->pendingAction[u] = 0.0f;
    lane->rewardSum = 0.0f; lane->rewardSumCompensation = 0.0f; lane->averageReward = 0.0f;
    lane->deathStep = OAK_DEATH_NONE; lane->deathDetector = 0u;
}

inline float advanceEnvironment(device OakFleetLane *lane, uint envK, uint laneSeed,
                                constant OakFleetEnvCfg &ecfg,
                                thread float *ctrl, thread float *obs,
                                thread uint *envWords, thread HalfCheetahState &halfCheetah,
                                thread uint &done) {
    for (int u = 0; u < 6; u++) ctrl[u] = lane->pendingAction[u];
    for (int i = 0; i < 10; i++) envWords[i] = lane->envWords[i];
    bool isCheetah = envK == OAK_ENV_CHEETAH;
    if (isCheetah) halfCheetah = lane->halfCheetah;
    uint elapsed = lane->elapsed, resets = lane->resets;
    float reward = env_step(envK, laneSeed, ctrl, envWords, halfCheetah, elapsed, resets,
                            ecfg.episodeLen, obs, done);
    for (int i = 0; i < 10; i++) lane->envWords[i] = envWords[i];
    if (isCheetah) lane->halfCheetah = halfCheetah;
    lane->elapsed = elapsed;
    lane->resets = resets;
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

inline void recordPose(uint envK, thread const uint *envWords,
                       thread const HalfCheetahState &halfCheetah,
                       thread const float *ctrl, uint stepNum, uint done,
                       constant OakFleetEnvCfg &ecfg,
                       device const int *recordSlot, device float *recordBuf, uint gid) {
    if (ecfg.recordCapacity != 0u) {
        int slot = recordSlot[gid];
        if (slot >= 0) {
            float pose[9];
            env_pose(envK, envWords, halfCheetah, ctrl, pose);
            ulong lanes = (ulong)ecfg.laneCount;
            device float *dst = recordBuf
                + ((stepNum - 1u) % ecfg.recordCapacity) * (OAK_RECORD_FLOATS * lanes)
                + (ulong)slot;
            dst[0] = float(done);
            for (uint i = 0u; i < OAK_POSE_LEN; i++) dst[(i + 1u) * lanes] = pose[i];
        }
    }
}
