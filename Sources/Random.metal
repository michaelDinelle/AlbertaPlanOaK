#pragma METAL fp math_mode(safe)

kernel void random_fleet_init(device OakFleetLane *lanes [[buffer(6)]],
                              device const uint *laneSeeds [[buffer(7)]],
                              constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                              device const uint *laneEnvKinds [[buffer(9)]],
                              device uint *envArena [[buffer(14)]], device const uint *envOff [[buffer(15)]],
                              constant uint *mjModels [[buffer(16)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    resetLane(&lanes[gid], (envArena + envOff[gid]), mjModels, laneEnvKinds[gid], laneSeeds[gid]);
}

kernel void
random_megastep(device OakFleetLane *lanes [[buffer(6)]], device const uint *laneSeeds [[buffer(7)]],
                constant OakFleetEnvCfg &ecfg [[buffer(8)]], device float *recordBuf [[buffer(10)]],
                constant uint3 &dispatchSpan [[buffer(11)]], device uint *envArena [[buffer(14)]],
                device const uint *envOff [[buffer(15)]], constant uint *mjModels [[buffer(16)]],
                device float *mjScratch [[buffer(17)]], device const uint *mjScratchOff [[buffer(18)]],
                device const float *envConstants [[buffer(19)]], uint gid [[thread_position_in_grid]],
                uint slane [[thread_index_in_simdgroup]]) {
    uint laneIdx = gid / 32u;
    if (laneIdx >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[laneIdx];
    device uint *envState = (envArena + envOff[laneIdx]);
    device float *pending = ((device float *)envState);
    uint envK = lane->envKind;
    uint seed = laneSeeds[laneIdx];
    uint actionKind = OAK_ENV_SPECS[envK].actionKind;
    uint actionCount = OAK_ENV_SPECS[envK].actionCount;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        uint stepIdx = lane->steps;
        if (slane == 0u) {
            if (actionKind == 1u) {
                uint w0 = baselineWord(seed, RNG_STREAM_RANDOM_ACT, stepIdx, 0u);
                uint a = min(uint(u01(w0) * float(actionCount)), actionCount - 1u);
                pending[0] = float(a);
                for (uint d = 1u; d < actionCount; d++) pending[d] = 0.0f;
            } else {
                for (uint d = 0u; d < actionCount; d++)
                    pending[d] = 2.0f * u01(baselineWord(seed, RNG_STREAM_RANDOM_ACT, stepIdx, d)) - 1.0f;
            }
            lane->steps = stepIdx + 1u;
        }
        simdgroup_barrier(mem_flags::mem_device);
        float ctrl[OAK_ENV_ACT_MAX];
        uint done = 0u;
        float reward =
            advanceEnvironment(lane, envState, mjModels, mjScratch + mjScratchOff[laneIdx], envConstants,
                               envK, seed, ecfg, ctrl, laneEnvObs(envState, envK), done, slane, true);
        if (slane == 0u) {
            uint stepNum = lane->steps;
            accumulateReward(lane, reward, stepNum, ecfg);
            recordPose(envK, mjModels, envState, ctrl, stepNum, done, reward, ecfg, recordBuf, laneIdx);
        }
        simdgroup_barrier(mem_flags::mem_device);
    }
}
