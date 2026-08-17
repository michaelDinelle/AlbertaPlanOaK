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

constant uint PPO_H = 64u;
constant uint PPO_T = 128u;
constant uint PPO_EPOCHS = 4u;
constant uint PPO_MB = 32u;
constant uint PPO_OMAX = 17u;
constant uint PPO_AMAX = 6u;
constant uint PPO_PARAM_STRIDE = 2768u;
constant uint PPO_ROLL_STRIDE = 3744u;
constant uint PPO_SCRATCH_STRIDE = 12064u;
constant uint PPO_STATE_STRIDE = 48u;
constant float PPO_LR = 3e-4f;
constant float PPO_GAMMA = 0.99f;
constant float PPO_LAMBDA = 0.95f;
constant float PPO_CLIP = 0.2f;
constant float PPO_VF_COEF = 0.5f;
constant float PPO_ENT_COEF_DISCRETE = 0.01f;
constant float PPO_MAX_GRAD_NORM = 0.5f;
constant float PPO_ADAM_B1 = 0.9f;
constant float PPO_ADAM_B2 = 0.999f;
constant float PPO_ADAM_EPS = 1e-5f;
constant float PPO_NORM_EPS = 1e-8f;
constant float PPO_NORM_CLIP = 10.0f;

constant uint PPO_ROLL_OBS = 0u;
constant uint PPO_ROLL_ACT = 2193u;
constant uint PPO_ROLL_LOGP = 2961u;
constant uint PPO_ROLL_VAL = 3089u;
constant uint PPO_ROLL_REW = 3217u;
constant uint PPO_ROLL_DONE = 3345u;
constant uint PPO_ROLL_ADV = 3473u;
constant uint PPO_ROLL_RET = 3601u;

constant uint PPO_SCR_OBSN = 0u;
constant uint PPO_SCR_HA = 544u;
constant uint PPO_SCR_HC = 2592u;
constant uint PPO_SCR_DHA = 4640u;
constant uint PPO_SCR_DHC = 6688u;
constant uint PPO_SCR_DLG = 8736u;
constant uint PPO_SCR_DLS = 8928u;
constant uint PPO_SCR_DV = 9120u;
constant uint PPO_SCR_IDX = 9152u;
constant uint PPO_SCR_GRAD = 9280u;

constant uint PPO_ST_OBSMEAN = 0u;
constant uint PPO_ST_OBSM2 = 17u;
constant uint PPO_ST_OBSCOUNT = 34u;
constant uint PPO_ST_RETTRACE = 35u;
constant uint PPO_ST_RETMEAN = 36u;
constant uint PPO_ST_RETM2 = 37u;
constant uint PPO_ST_RETCOUNT = 38u;
constant uint PPO_ST_ADAMT = 39u;

struct PpoLayout {
    uint O, A;
    bool box;
    uint w1a, b1a, w2a, b2a, logstd, w1c, b1c, w2c, b2c, count;
};

inline PpoLayout ppoLayout(uint envK) {
    PpoLayout L;
    L.O = OAK_ENV_SPECS[envK].obsDimNatural;
    L.A = OAK_ENV_SPECS[envK].actionCount;
    L.box = OAK_ENV_SPECS[envK].actionKind == 0u;
    L.w1a = 0u;
    L.b1a = L.w1a + L.O * PPO_H;
    L.w2a = L.b1a + PPO_H;
    L.b2a = L.w2a + PPO_H * L.A;
    L.logstd = L.b2a + L.A;
    L.w1c = L.logstd + (L.box ? L.A : 0u);
    L.b1c = L.w1c + L.O * PPO_H;
    L.w2c = L.b1c + PPO_H;
    L.b2c = L.w2c + PPO_H;
    L.count = L.b2c + 1u;
    return L;
}

inline float ppoGauss(uint seed, uint stream, uint c0, uint c1) {
    uint4 r = philox4x32(uint2(seed, stream), uint4(c0, c1, 0u, 0u));
    float u1 = u01open(r.x);
    float u2 = u01(r.y);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

inline void ppoNormObs(device const float *S, uint O, thread const float *raw, thread float *outN) {
    float count = max(S[PPO_ST_OBSCOUNT], 1.0f);
    for (uint d = 0u; d < O; ++d) {
        float variance = S[PPO_ST_OBSM2 + d] / count;
        float n = (raw[d] - S[PPO_ST_OBSMEAN + d]) / sqrt(variance + PPO_NORM_EPS);
        outN[d] = clamp(n, -PPO_NORM_CLIP, PPO_NORM_CLIP);
    }
}

inline void ppoWelfordObs(device float *S, uint O, thread const float *raw) {
    float count = S[PPO_ST_OBSCOUNT] + 1.0f;
    S[PPO_ST_OBSCOUNT] = count;
    for (uint d = 0u; d < O; ++d) {
        float delta = raw[d] - S[PPO_ST_OBSMEAN + d];
        S[PPO_ST_OBSMEAN + d] += delta / count;
        S[PPO_ST_OBSM2 + d] += delta * (raw[d] - S[PPO_ST_OBSMEAN + d]);
    }
}

inline float ppoScaleReward(device float *S, float reward, uint done) {
    float trace = S[PPO_ST_RETTRACE] * PPO_GAMMA * (done != 0u ? 0.0f : 1.0f) + reward;
    S[PPO_ST_RETTRACE] = trace;
    float count = S[PPO_ST_RETCOUNT] + 1.0f;
    S[PPO_ST_RETCOUNT] = count;
    float delta = trace - S[PPO_ST_RETMEAN];
    S[PPO_ST_RETMEAN] += delta / count;
    S[PPO_ST_RETM2] += delta * (trace - S[PPO_ST_RETMEAN]);
    float variance = S[PPO_ST_RETM2] / count;
    return clamp(reward / sqrt(variance + PPO_NORM_EPS), -PPO_NORM_CLIP, PPO_NORM_CLIP);
}

inline void ppoForwardWide(device const float *P, PpoLayout L, thread const float *obsn, uint tid,
                           thread float *logits, thread float &value) {
    float hh[2];
    for (uint k = 0u; k < 2u; ++k) {
        uint hu = tid + 32u * k;
        float acc = P[L.b1a + hu];
        for (uint o = 0u; o < L.O; ++o) acc += P[L.w1a + o * PPO_H + hu] * obsn[o];
        hh[k] = tanh(acc);
    }
    for (uint a = 0u; a < L.A; ++a) {
        float part = hh[0] * P[L.w2a + tid * L.A + a] + hh[1] * P[L.w2a + (tid + 32u) * L.A + a];
        logits[a] = simd_sum(part) + P[L.b2a + a];
    }
    float hc[2];
    for (uint k = 0u; k < 2u; ++k) {
        uint hu = tid + 32u * k;
        float acc = P[L.b1c + hu];
        for (uint o = 0u; o < L.O; ++o) acc += P[L.w1c + o * PPO_H + hu] * obsn[o];
        hc[k] = tanh(acc);
    }
    float partV = hc[0] * P[L.w2c + tid] + hc[1] * P[L.w2c + tid + 32u];
    value = simd_sum(partV) + P[L.b2c];
}

inline float ppoForwardSample(device const float *P, PpoLayout L, thread const float *obsn,
                              device float *haOut, device float *hcOut, thread float *logits) {
    for (uint hu = 0u; hu < PPO_H; ++hu) {
        float acc = P[L.b1a + hu];
        for (uint o = 0u; o < L.O; ++o) acc += P[L.w1a + o * PPO_H + hu] * obsn[o];
        haOut[hu] = tanh(acc);
    }
    for (uint a = 0u; a < L.A; ++a) {
        float acc = P[L.b2a + a];
        for (uint hu = 0u; hu < PPO_H; ++hu) acc += P[L.w2a + hu * L.A + a] * haOut[hu];
        logits[a] = acc;
    }
    float v = P[L.b2c];
    for (uint hu = 0u; hu < PPO_H; ++hu) {
        float acc = P[L.b1c + hu];
        for (uint o = 0u; o < L.O; ++o) acc += P[L.w1c + o * PPO_H + hu] * obsn[o];
        hcOut[hu] = tanh(acc);
        v += P[L.w2c + hu] * hcOut[hu];
    }
    return v;
}

kernel void ppo_fleet_init(device float *params [[buffer(0)]],
                           device float *ppoState [[buffer(4)]],
                           device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           device const uint *laneEnvKinds [[buffer(9)]],
                           device float *rollout [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint tid [[thread_index_in_simdgroup]]) {
    uint laneIdx = gid / 32u;
    if (laneIdx >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[laneIdx];
    uint seed = laneSeeds[laneIdx];
    PpoLayout L = ppoLayout(laneEnvKinds[laneIdx]);
    device float *P = params + laneIdx * PPO_PARAM_STRIDE;
    float stdIn = sqrt(2.0f / float(L.O + PPO_H));
    float stdOutA = 0.01f * sqrt(2.0f / float(PPO_H + L.A));
    float stdOutC = sqrt(2.0f / float(PPO_H + 1u));
    for (uint p = tid; p < L.count; p += 32u) {
        float std = 0.0f;
        if (p < L.b1a) std = stdIn;
        else if (p >= L.w2a && p < L.b2a) std = stdOutA;
        else if (p >= L.w1c && p < L.b1c) std = stdIn;
        else if (p >= L.w2c && p < L.b2c) std = stdOutC;
        P[p] = std == 0.0f ? 0.0f : std * ppoGauss(seed, RNG_STREAM_PPO_INIT, p, 0u);
    }
    device float *S = ppoState + laneIdx * PPO_STATE_STRIDE;
    for (uint i = tid; i < PPO_STATE_STRIDE; i += 32u) S[i] = 0.0f;
    if (tid == 0u) {
        HalfCheetahState halfCheetah;
        uint envWords[10];
        resetLane(lane, laneEnvKinds[laneIdx], seed, envWords, halfCheetah);
        float obs[17];
        env_observe(lane->envKind, envWords, halfCheetah, obs);
        device float *R = rollout + laneIdx * PPO_ROLL_STRIDE;
        for (uint d = 0u; d < L.O; ++d) R[PPO_ROLL_OBS + d] = obs[d];
        ppoWelfordObs(S, L.O, obs);
    }
}

kernel void ppo_fleet_snapshot(device const float *params [[buffer(0)]],
                               device OakFleetLane *lanes [[buffer(6)]],
                               constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                               device OakLaneSnapshot *snaps [[buffer(9)]],
                               constant uint &snapBase [[buffer(10)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device OakLaneSnapshot *snap = &snaps[snapBase + gid];
    PpoLayout L = ppoLayout(lane->envKind);
    device const float *P = params + gid * PPO_PARAM_STRIDE;
    float absMax = 0.0f;
    uint nonFinite = 0u;
    for (uint p = 0u; p < L.count; ++p) {
        float v = P[p];
        if (!isfinite(v)) nonFinite += 1u;
        else absMax = max(absMax, fabs(v));
    }
    snap->scalars.stepCount = lane->steps;
    snap->scalars.stateSlot = 0u;
    snap->scalars.averageReward = lane->averageReward;
    snap->paramAbsMax = absMax;
    snap->paramNonFinite = nonFinite;
    snap->rewardSum = lane->rewardSum;
    snap->rho = lane->averageReward;
    snap->deathStep = lane->deathStep;
    snap->deathDetector = lane->deathDetector;
}

kernel void ppo_megastep(device float *params [[buffer(0)]],
                         device float *adam [[buffer(1)]],
                         device float *rollout [[buffer(2)]],
                         device float *scratch [[buffer(3)]],
                         device float *ppoState [[buffer(4)]],
                         device OakFleetLane *lanes [[buffer(6)]],
                         device const uint *laneSeeds [[buffer(7)]],
                         constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                         device const int *recordSlot [[buffer(9)]],
                         device float *recordBuf [[buffer(10)]],
                         constant uint3 &dispatchSpan [[buffer(11)]],
                         uint gid [[thread_position_in_grid]],
                         uint tid [[thread_index_in_simdgroup]]) {
    uint laneIdx = gid / 32u;
    if (laneIdx >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[laneIdx];
    uint envK = lane->envKind;
    uint seed = laneSeeds[laneIdx];
    PpoLayout L = ppoLayout(envK);
    device float *P = params + laneIdx * PPO_PARAM_STRIDE;
    device float *M = adam + laneIdx * 2u * PPO_PARAM_STRIDE;
    device float *V = M + PPO_PARAM_STRIDE;
    device float *R = rollout + laneIdx * PPO_ROLL_STRIDE;
    device float *W = scratch + laneIdx * PPO_SCRATCH_STRIDE;
    device float *S = ppoState + laneIdx * PPO_STATE_STRIDE;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        uint stepIdx = lane->steps;
        uint t = stepIdx % PPO_T;
        float obsRaw[17];
        for (uint d = 0u; d < L.O; ++d) obsRaw[d] = R[PPO_ROLL_OBS + t * PPO_OMAX + d];
        float obsn[17];
        ppoNormObs(S, L.O, obsRaw, obsn);
        float logits[6];
        float value;
        ppoForwardWide(P, L, obsn, tid, logits, value);
        if (tid == 0u) {
            float logp;
            if (L.box) {
                uint4 r0 = philox4x32(uint2(seed, RNG_STREAM_PPO_ACT), uint4(stepIdx, 0u, 0u, 0u));
                uint4 r1 = philox4x32(uint2(seed, RNG_STREAM_PPO_ACT), uint4(stepIdx, 1u, 0u, 0u));
                uint words[8] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w };
                logp = 0.0f;
                for (uint d = 0u; d < L.A; ++d) {
                    float u1 = u01open(words[d]);
                    float u2 = u01(words[(d + L.A) & 7u]);
                    float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
                    float ls = P[L.logstd + d];
                    float sample = logits[d] + exp(ls) * z;
                    R[PPO_ROLL_ACT + t * PPO_AMAX + d] = sample;
                    lane->pendingAction[d] = clamp(sample, -1.0f, 1.0f);
                    logp += -0.5f * z * z - ls - 0.9189385332f;
                }
                for (uint d = L.A; d < 6u; ++d) lane->pendingAction[d] = 0.0f;
            } else {
                float mx = logits[0];
                for (uint a = 1u; a < L.A; ++a) mx = max(mx, logits[a]);
                float denom = 0.0f;
                for (uint a = 0u; a < L.A; ++a) denom += exp(logits[a] - mx);
                uint4 r0 = philox4x32(uint2(seed, RNG_STREAM_PPO_ACT), uint4(stepIdx, 0u, 0u, 0u));
                float u = u01(r0.x);
                float cdf = 0.0f;
                uint act = L.A - 1u;
                for (uint a = 0u; a < L.A; ++a) {
                    cdf += exp(logits[a] - mx) / denom;
                    if (u < cdf) { act = a; break; }
                }
                logp = logits[act] - mx - log(denom);
                R[PPO_ROLL_ACT + t * PPO_AMAX] = float(act);
                lane->pendingAction[0] = float(act);
                for (int d = 1; d < 6; d++) lane->pendingAction[d] = 0.0f;
            }
            R[PPO_ROLL_LOGP + t] = logp;
            R[PPO_ROLL_VAL + t] = value;
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
            recordPose(envK, envWords, halfCheetah, ctrl, stepNum, done, ecfg,
                       recordSlot, recordBuf, laneIdx);
            float safeReward = isfinite(reward) ? reward : 0.0f;
            R[PPO_ROLL_REW + t] = ppoScaleReward(S, safeReward, done);
            R[PPO_ROLL_DONE + t] = float(done);
            for (uint d = 0u; d < L.O; ++d) R[PPO_ROLL_OBS + (t + 1u) * PPO_OMAX + d] = obs[d];
            ppoWelfordObs(S, L.O, obs);
        }
        laneFence();
        if ((lane->steps % PPO_T) != 0u) continue;
        uint updateIdx = lane->steps / PPO_T;
        {
            float bootObsRaw[17];
            for (uint d = 0u; d < L.O; ++d) bootObsRaw[d] = R[PPO_ROLL_OBS + PPO_T * PPO_OMAX + d];
            float bootObsn[17];
            ppoNormObs(S, L.O, bootObsRaw, bootObsn);
            float bootLogits[6];
            float bootV;
            ppoForwardWide(P, L, bootObsn, tid, bootLogits, bootV);
            if (tid == 0u) {
                float lastGae = 0.0f;
                for (int ti = int(PPO_T) - 1; ti >= 0; --ti) {
                    float nextV = (ti == int(PPO_T) - 1) ? bootV : R[PPO_ROLL_VAL + ti + 1];
                    float nonTerm = 1.0f - R[PPO_ROLL_DONE + ti];
                    float delta = R[PPO_ROLL_REW + ti] + PPO_GAMMA * nextV * nonTerm - R[PPO_ROLL_VAL + ti];
                    lastGae = delta + PPO_GAMMA * PPO_LAMBDA * nonTerm * lastGae;
                    R[PPO_ROLL_ADV + ti] = lastGae;
                    R[PPO_ROLL_RET + ti] = lastGae + R[PPO_ROLL_VAL + ti];
                }
            }
        }
        laneFence();
        for (uint epoch = 0u; epoch < PPO_EPOCHS; ++epoch) {
            if (tid == 0u) {
                for (uint i = 0u; i < PPO_T; ++i) W[PPO_SCR_IDX + i] = float(i);
                uint block = 0u;
                uint wordIdx = 4u;
                uint4 rb = uint4(0u);
                for (uint i = PPO_T - 1u; i > 0u; --i) {
                    if (wordIdx == 4u) {
                        rb = philox4x32(uint2(seed, RNG_STREAM_PPO_SHUFFLE),
                                        uint4(updateIdx, epoch, block, 0u));
                        block += 1u;
                        wordIdx = 0u;
                    }
                    uint words[4] = { rb.x, rb.y, rb.z, rb.w };
                    uint j = words[wordIdx] % (i + 1u);
                    wordIdx += 1u;
                    float tmp = W[PPO_SCR_IDX + i];
                    W[PPO_SCR_IDX + i] = W[PPO_SCR_IDX + j];
                    W[PPO_SCR_IDX + j] = tmp;
                }
            }
            laneFence();
            for (uint mb = 0u; mb < PPO_T / PPO_MB; ++mb) {
                uint s = uint(W[PPO_SCR_IDX + mb * PPO_MB + tid]);
                float advS = R[PPO_ROLL_ADV + s];
                float mbMean = simd_sum(advS) / float(PPO_MB);
                float dev = advS - mbMean;
                float mbStd = sqrt(simd_sum(dev * dev) / float(PPO_MB));
                float an = dev / (mbStd + PPO_NORM_EPS);
                float sObsRaw[17];
                for (uint d = 0u; d < L.O; ++d) sObsRaw[d] = R[PPO_ROLL_OBS + s * PPO_OMAX + d];
                float sObsn[17];
                ppoNormObs(S, L.O, sObsRaw, sObsn);
                device float *obsnRow = W + PPO_SCR_OBSN + tid * PPO_OMAX;
                for (uint d = 0u; d < L.O; ++d) obsnRow[d] = sObsn[d];
                device float *haRow = W + PPO_SCR_HA + tid * PPO_H;
                device float *hcRow = W + PPO_SCR_HC + tid * PPO_H;
                float sLogits[6];
                float v = ppoForwardSample(P, L, sObsn, haRow, hcRow, sLogits);
                float oldLogp = R[PPO_ROLL_LOGP + s];
                float newLogp;
                float dlg[6];
                float dls[6];
                if (L.box) {
                    newLogp = 0.0f;
                    for (uint d = 0u; d < L.A; ++d) {
                        float ls = P[L.logstd + d];
                        float sigma = exp(ls);
                        float x = R[PPO_ROLL_ACT + s * PPO_AMAX + d];
                        float zd = (x - sLogits[d]) / sigma;
                        newLogp += -0.5f * zd * zd - ls - 0.9189385332f;
                        dlg[d] = zd / sigma;
                        dls[d] = zd * zd - 1.0f;
                    }
                } else {
                    float mx = sLogits[0];
                    for (uint a = 1u; a < L.A; ++a) mx = max(mx, sLogits[a]);
                    float denom = 0.0f;
                    for (uint a = 0u; a < L.A; ++a) denom += exp(sLogits[a] - mx);
                    uint act = uint(R[PPO_ROLL_ACT + s * PPO_AMAX]);
                    newLogp = sLogits[act] - mx - log(denom);
                    float entropy = 0.0f;
                    for (uint a = 0u; a < L.A; ++a) {
                        float pa = exp(sLogits[a] - mx) / denom;
                        if (pa > 0.0f) entropy -= pa * log(pa);
                    }
                    float ratio = exp(newLogp - oldLogp);
                    bool clippedOut = (an >= 0.0f && ratio > 1.0f + PPO_CLIP)
                        || (an <= 0.0f && ratio < 1.0f - PPO_CLIP);
                    float dNewLogp = clippedOut ? 0.0f : -an * ratio;
                    for (uint a = 0u; a < L.A; ++a) {
                        float pa = exp(sLogits[a] - mx) / denom;
                        float dEnt = PPO_ENT_COEF_DISCRETE * pa * (log(max(pa, 1e-30f)) + entropy);
                        dlg[a] = dNewLogp * ((a == act ? 1.0f : 0.0f) - pa) + dEnt;
                        dls[a] = 0.0f;
                    }
                }
                if (L.box) {
                    float ratio = exp(newLogp - oldLogp);
                    bool clippedOut = (an >= 0.0f && ratio > 1.0f + PPO_CLIP)
                        || (an <= 0.0f && ratio < 1.0f - PPO_CLIP);
                    float dNewLogp = clippedOut ? 0.0f : -an * ratio;
                    for (uint d = 0u; d < L.A; ++d) {
                        dlg[d] *= dNewLogp;
                        dls[d] *= dNewLogp;
                    }
                }
                float retS = R[PPO_ROLL_RET + s];
                float oldV = R[PPO_ROLL_VAL + s];
                float vClipped = oldV + clamp(v - oldV, -PPO_CLIP, PPO_CLIP);
                float l1 = (v - retS) * (v - retS);
                float l2 = (vClipped - retS) * (vClipped - retS);
                float dV;
                if (l1 >= l2) dV = PPO_VF_COEF * (v - retS);
                else dV = fabs(v - oldV) < PPO_CLIP ? PPO_VF_COEF * (vClipped - retS) : 0.0f;
                device float *dlgRow = W + PPO_SCR_DLG + tid * PPO_AMAX;
                device float *dlsRow = W + PPO_SCR_DLS + tid * PPO_AMAX;
                for (uint a = 0u; a < L.A; ++a) { dlgRow[a] = dlg[a]; dlsRow[a] = dls[a]; }
                W[PPO_SCR_DV + tid] = dV;
                device float *dhaRow = W + PPO_SCR_DHA + tid * PPO_H;
                device float *dhcRow = W + PPO_SCR_DHC + tid * PPO_H;
                for (uint hu = 0u; hu < PPO_H; ++hu) {
                    float accA = 0.0f;
                    for (uint a = 0u; a < L.A; ++a) accA += dlg[a] * P[L.w2a + hu * L.A + a];
                    float ha = haRow[hu];
                    dhaRow[hu] = accA * (1.0f - ha * ha);
                    float hc = hcRow[hu];
                    dhcRow[hu] = dV * P[L.w2c + hu] * (1.0f - hc * hc);
                }
                laneFence();
                float sumSq = 0.0f;
                for (uint p = tid; p < L.count; p += 32u) {
                    float g = 0.0f;
                    if (p < L.b1a) {
                        uint o = p / PPO_H;
                        uint hu = p % PPO_H;
                        for (uint ss = 0u; ss < PPO_MB; ++ss)
                            g += W[PPO_SCR_DHA + ss * PPO_H + hu] * W[PPO_SCR_OBSN + ss * PPO_OMAX + o];
                    } else if (p < L.w2a) {
                        uint hu = p - L.b1a;
                        for (uint ss = 0u; ss < PPO_MB; ++ss) g += W[PPO_SCR_DHA + ss * PPO_H + hu];
                    } else if (p < L.b2a) {
                        uint rel = p - L.w2a;
                        uint hu = rel / L.A;
                        uint a = rel % L.A;
                        for (uint ss = 0u; ss < PPO_MB; ++ss)
                            g += W[PPO_SCR_DLG + ss * PPO_AMAX + a] * W[PPO_SCR_HA + ss * PPO_H + hu];
                    } else if (p < L.logstd) {
                        uint a = p - L.b2a;
                        for (uint ss = 0u; ss < PPO_MB; ++ss) g += W[PPO_SCR_DLG + ss * PPO_AMAX + a];
                    } else if (p < L.w1c) {
                        uint a = p - L.logstd;
                        for (uint ss = 0u; ss < PPO_MB; ++ss) g += W[PPO_SCR_DLS + ss * PPO_AMAX + a];
                    } else if (p < L.b1c) {
                        uint rel = p - L.w1c;
                        uint o = rel / PPO_H;
                        uint hu = rel % PPO_H;
                        for (uint ss = 0u; ss < PPO_MB; ++ss)
                            g += W[PPO_SCR_DHC + ss * PPO_H + hu] * W[PPO_SCR_OBSN + ss * PPO_OMAX + o];
                    } else if (p < L.w2c) {
                        uint hu = p - L.b1c;
                        for (uint ss = 0u; ss < PPO_MB; ++ss) g += W[PPO_SCR_DHC + ss * PPO_H + hu];
                    } else if (p < L.b2c) {
                        uint hu = p - L.w2c;
                        for (uint ss = 0u; ss < PPO_MB; ++ss)
                            g += W[PPO_SCR_DV + ss] * W[PPO_SCR_HC + ss * PPO_H + hu];
                    } else {
                        for (uint ss = 0u; ss < PPO_MB; ++ss) g += W[PPO_SCR_DV + ss];
                    }
                    g /= float(PPO_MB);
                    W[PPO_SCR_GRAD + p] = g;
                    sumSq += g * g;
                }
                float gNorm = sqrt(simd_sum(sumSq));
                float scale = gNorm > PPO_MAX_GRAD_NORM ? PPO_MAX_GRAD_NORM / gNorm : 1.0f;
                float adamT = S[PPO_ST_ADAMT] + 1.0f;
                float bc1 = 1.0f - powr(PPO_ADAM_B1, adamT);
                float bc2 = 1.0f - powr(PPO_ADAM_B2, adamT);
                for (uint p = tid; p < L.count; p += 32u) {
                    float g = W[PPO_SCR_GRAD + p] * scale;
                    float m = PPO_ADAM_B1 * M[p] + (1.0f - PPO_ADAM_B1) * g;
                    float vv = PPO_ADAM_B2 * V[p] + (1.0f - PPO_ADAM_B2) * g * g;
                    M[p] = m;
                    V[p] = vv;
                    P[p] -= PPO_LR * (m / bc1) / (sqrt(vv / bc2) + PPO_ADAM_EPS);
                }
                laneFence();
                if (tid == 0u) S[PPO_ST_ADAMT] = adamT;
                laneFence();
            }
        }
        if (tid == 0u) {
            for (uint d = 0u; d < L.O; ++d)
                R[PPO_ROLL_OBS + d] = R[PPO_ROLL_OBS + PPO_T * PPO_OMAX + d];
        }
        laneFence();
    }
}
