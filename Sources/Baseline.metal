#pragma METAL fp math_mode(safe)

kernel void baseline_fleet_init(device OakFleetLane *lanes [[buffer(6)]],
                                device const uint *laneSeeds [[buffer(7)]],
                                constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                                device const uint *laneEnvKinds [[buffer(9)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    HalfCheetahState halfCheetah;
    uint envWords[10];
    resetLane(&lanes[gid], laneEnvKinds[gid], laneSeeds[gid], envWords, halfCheetah);
}

kernel void baseline_fleet_snapshot(device OakFleetLane *lanes [[buffer(6)]],
                                    constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                                    device OakLaneSnapshot *snaps [[buffer(9)]],
                                    constant uint &snapBase [[buffer(10)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device OakLaneSnapshot *snap = &snaps[snapBase + gid];
    snap->scalars.stepCount = lane->steps;
    snap->scalars.stateSlot = 0u;
    snap->scalars.averageReward = lane->averageReward;
    snap->paramAbsMax = -1.0f;
    snap->paramNonFinite = 0u;
    snap->rewardSum = lane->rewardSum;
    snap->rho = lane->averageReward;
    snap->deathStep = lane->deathStep;
    snap->deathDetector = lane->deathDetector;
}

kernel void random_megastep(device OakFleetLane *lanes [[buffer(6)]],
                            device const uint *laneSeeds [[buffer(7)]],
                            constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                            device const int *recordSlot [[buffer(9)]],
                            device float *recordBuf [[buffer(10)]],
                            constant uint3 &dispatchSpan [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    uint envK = lane->envKind;
    uint seed = laneSeeds[gid];
    uint actionKind = OAK_ENV_SPECS[envK].actionKind;
    uint actionCount = OAK_ENV_SPECS[envK].actionCount;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        uint stepIdx = lane->steps;
        uint4 rw = philox4x32(uint2(seed, RNG_STREAM_RANDOM_ACT), uint4(stepIdx, 0u, 0u, 0u));
        uint4 rw1 = philox4x32(uint2(seed, RNG_STREAM_RANDOM_ACT), uint4(stepIdx, 1u, 0u, 0u));
        uint words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
        if (actionKind == 1u) {
            uint a = min(uint(u01(words[0]) * float(actionCount)), actionCount - 1u);
            lane->pendingAction[0] = float(a);
            for (int d = 1; d < 6; d++) lane->pendingAction[d] = 0.0f;
        } else {
            for (uint d = 0u; d < 6u; d++)
                lane->pendingAction[d] = (d < actionCount) ? (2.0f * u01(words[d]) - 1.0f) : 0.0f;
        }
        lane->steps = stepIdx + 1u;
        float ctrl[6];
        float obs[17];
        uint envWords[10];
        uint done = 0u;
        HalfCheetahState halfCheetah;
        float reward = advanceEnvironment(lane, envK, seed, ecfg, ctrl, obs,
                                          envWords, halfCheetah, done);
        uint stepNum = lane->steps;
        accumulateReward(lane, reward, stepNum, ecfg);
        recordPose(envK, envWords, halfCheetah, ctrl, stepNum, done, ecfg, recordSlot, recordBuf, gid);
    }
}
