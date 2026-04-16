#include <metal_stdlib>
#include <float.h>
using namespace metal;

/// Per-term magnitude bound so `termCount` summed squares stay in `float` range: each term ≤ `FLT_MAX/n` ⇒ sum ≤ `FLT_MAX`.
/// **Not** reward shaping or display scaling (Alberta invariant 7): IEEE overflow hygiene for diagnostic aggregates only;
/// learning uses unsquared `sanitize` paths elsewhere. `FLT_MAX` from `<float.h>`; `sqrt` from Metal.
inline float stats_abs_bound_for_sum_squares(uint termCount) {
    float n = float(max(termCount, 1u));
    return sqrt(FLT_MAX / n);
}

// Oak / Alberta Plan (12 steps): continual phases every tick (no train/eval). Packed `params` + `state` + `uint` arenas
// keep per-kernel buffer bindings ≤ 8 (split pipelines share the same layout).
// Shared latent `state` is read-only within a phase; consumers share one representation (AGENTS.md).

struct KernelConfig {
    uint obsDim;
    uint rank;
    uint actDim;
    uint batchSize;
    uint actionIsDiscrete;
    uint optionCount;
    uint primitivePackedSize;
    uint optionForwardStride;
    uint planK;
    uint baseObsDim;
    uint pEncoder;
    uint pPolicy;
    uint pValue;
    uint pReward;
    uint pStateWeights;
    uint pOptPolicy;
    uint pOptValues;
    uint pOptCritic;
    uint pFeatSecondMoment;
    uint pFeatResidual;
    uint sObservation;
    uint sObsMean;
    uint sPrevState;
    uint sPrevAction;
    uint sReward;
    uint sState;
    uint sAction;
    uint sLatentError;
    uint sRewardPred;
    uint sTd;
    uint sActionGrad;
    uint sRollout;
    uint sLatentPower;
    uint uDone;
    uint uActive;
    uint uStomp;
    uint uCounters;
    uint floatArenaAutostepBase;
    uint floatArenaTdOptionBase;
    uint floatArenaPlanDeltaBase;
    uint floatArenaGvfWeightsBase;
    uint floatArenaGvfCumulantBase;
    uint floatArenaGvfTdBase;
    uint floatArenaGvfRhoBase;
    uint floatArenaKeyboardBase;
    uint floatArenaKeyboardPredBase;
    uint floatArenaUtilityBase;
    uint floatArenaFeatureGenBase;
    uint floatArenaSubtaskRhoBase;
    uint floatArenaOptionBetaBase;
    uint floatArenaDeviceScratchBase;
    uint genDirectionCount;
    uint floatArenaOffsetInLatentPower;
    uint phase;
    uint gammaSubtaskMilli;
    uint betaLearnRateMilli;
};
struct KernelScalars { uint stepCount, rewardCount, activeCount, admissionCursor; float rewardRate; };
struct KernelStats {
    float rewardRate, latentPredictionMSE, rewardPredictionMSE, latentStateMeanSquare, activeRank, actionDeltaL2Mean, planningActionChangeL2Mean;
    float rawNonFiniteCount, sanitizedNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
    /// Mean squared (sanitized) error between `state` and `rolloutState` for rows where dynamics matched `batchDynOpt` (STOMP option-model quality).
    float optionModelPredictionMSE;
    /// Batch mean fraction of planning rows using a **non-default** option dynamics slice (`po > 0`; slice 0 is shared one-step dynamics).
    float planningUsedOptionDynamicsMean;
    /// Count of option slices zeroed this bookkeeping pass (continual replacement / capacity discipline).
    float oakOptionReinitCount;
    float utilityFeatureMean, utilitySubtaskMean, utilityOptionMean, utilityPlanningMean;
};

inline float sanitize(float v, thread uint& raw, thread uint& san) {
    if (isfinite(v)) return v;
    raw++; san++;
    return 0.0f;
}

inline float masked_dot(device const float *w, uint wOff, device const float *v, uint vOff, device const uint *active, uint count, thread uint& r, thread uint& s) {
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        uint idx = active[i];
        sum += w[wOff + idx] * sanitize(v[vOff + idx], r, s);
    }
    return sanitize(sum, r, s);
}

inline float dot_active(device const float *w, uint wOff, device const float *v, uint vOff, device const uint *activeList, uint activeCount, thread uint& raw, thread uint& san) {
    return masked_dot(w, wOff, v, vOff, activeList, activeCount, raw, san);
}

/// Ridge from representation capacity (rank × batch): avoids free ε floors (NLMS / autostep family).
inline float capacity_ridge(uint rank, uint batchSize) {
    float cap = float(max(rank * max(batchSize, 1u), 1u));
    return 1.0f / (cap + 1.0f);
}

/// Same `cap` as in `capacity_ridge` (i.e. `1/capacity_ridge − 1`): per-coordinate clamp magnitude for shared
/// representations so continual forward passes do not diverge toward float overflow (Alberta: temporal uniformity).
inline float structural_activation_cap(uint rank, uint batchSize) {
    return float(max(rank * max(batchSize, 1u), 1u));
}

/// Online-average weight 1/(t+1) uses step index after this transition (stepCount is steps so far).
inline float online_inv_time(device const KernelScalars *scalars) {
    return 1.0f / float(max(scalars->stepCount + 1u, 1u));
}

/// Mahmood–Sutton–White Autostep meta-step η (structural; not env-specific).
inline float autostep_eta(uint rank) {
    return 1e-2f / float(max(rank, 1u));
}

inline float plan_merge_lambda(uint rank, uint actDim) {
    return 1.0f / sqrt(float(max(rank + actDim, 1u)));
}

inline device float *arena_ptr(device float *arena, uint base) {
    return arena + base;
}

inline uint gvf_bank_size(uint rank) { return rank * rank; }

inline float gvf_weight(device const float *packed, uint g, uint i, uint rank) {
    return packed[g * rank + i];
}

inline float forward_ws(device const float *packed, uint rank, uint actDim, uint outRow, uint inIdx) {
    uint base = gvf_bank_size(rank);
    return packed[base + outRow * (rank + actDim) + inIdx];
}

inline float forward_wa(device const float *packed, uint rank, uint actDim, uint outRow, uint aIdx) {
    uint base = gvf_bank_size(rank) + outRow * (rank + actDim) + rank;
    return packed[base + aIdx];
}

/// Option dynamics block `opt`: `base = primitivePacked + opt * optionForwardStride` (`primitivePacked` = GVF size).
inline float opt_forward_ws(device const float *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint inIdx) {
    return packed[blockBase + outRow * (rank + actDim) + inIdx];
}
inline float opt_forward_wa(device const float *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint aIdx) {
    return packed[blockBase + outRow * (rank + actDim) + rank + aIdx];
}

/// `stompState` layout: indices `[0, 4*batch)` phase/elapsed per row (4 uints/row); `[4*batch, 5*batch)` batch dynamics option index (STOMP).
inline bool use_option_rollout(device const uint *stompState, uint row, uint rank) {
    uint phase = stompState[row * 4u];
    uint elapsed = stompState[row * 4u + 1u];
    return phase == 1u && elapsed < rank;
}

inline uint batch_dynamics_opt(device const uint *stompState, uint batchSize, uint row) {
    return stompState[4u * batchSize + row];
}

inline uint pick_option_slice(device const uint *stompState, device const float *latentPower, uint row, uint rank, float prevOptF, uint optionCount) {
    if (use_option_rollout(stompState, row, rank)) {
        uint po = uint(floor(prevOptF + 0.5f));
        if (po >= optionCount) po = optionCount - 1u;
        return po;
    }
    return 0u;
}

/// Start of the packed **forward** segment: after GVF bank (same layout for primitive and option slices).
inline uint forward_segment_base(uint rank) { return gvf_bank_size(rank); }

inline float dyn_forward_ws(device const float *packed, uint forwardBase, uint rank, uint actDim, uint outRow, uint inIdx) {
    return packed[forwardBase + outRow * (rank + actDim) + inIdx];
}
inline float dyn_forward_wa(device const float *packed, uint forwardBase, uint rank, uint actDim, uint outRow, uint aIdx) {
    return packed[forwardBase + outRow * (rank + actDim) + rank + aIdx];
}

/// One-step lookahead: primary value + **subtask** value head (`optionValues` row) on predicted features (planning→representation loop).
inline float lookahead_value_thread(device const float *stateWeights, uint forwardBase, device const float *valueWeights, device const float *optionSubValues,
    uint optIdx, device const float *lat, uint sOff, uint actDim, uint rank, device const uint *activeList, uint activeCount,     thread uint& raw, thread uint& san,
    device const float *actionVec) {
    float vCand = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint idx = activeList[li];
        float ps = 0.0f;
        for (uint lj = 0u; lj < activeCount; ++lj) {
            uint jdx = activeList[lj];
            ps += dyn_forward_ws(stateWeights, forwardBase, rank, actDim, idx, jdx) * sanitize(lat[sOff + jdx], raw, san);
        }
        float pa = 0.0f;
        for (uint a = 0u; a < actDim; ++a) {
            pa += dyn_forward_wa(stateWeights, forwardBase, rank, actDim, idx, a) * sanitize(actionVec[a], raw, san);
        }
        float predNext = sanitize(ps + pa, raw, san);
        float pri = valueWeights[idx] * predNext;
        float sec = optionSubValues[optIdx * rank + idx] * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

inline float discrete_lookahead1(device const float *stateWeights, uint forwardBase, device const float *valueWeights,
    device const float *optionSubValues, uint optIdx,
    device const float *lat, uint sOff, uint actDim, uint rank, device const uint *activeList, uint activeCount,
    uint cand, thread uint& raw, thread uint& san) {
    float vCand = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint idx = activeList[li];
        float ps = 0.0f;
        for (uint lj = 0u; lj < activeCount; ++lj) {
            uint jdx = activeList[lj];
            ps += dyn_forward_ws(stateWeights, forwardBase, rank, actDim, idx, jdx) * sanitize(lat[sOff + jdx], raw, san);
        }
        float pa = dyn_forward_wa(stateWeights, forwardBase, rank, actDim, idx, cand);
        float predNext = sanitize(ps + pa, raw, san);
        float pri = valueWeights[idx] * predNext;
        float sec = optionSubValues[optIdx * rank + idx] * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

inline float kstep_plan_discrete(device const float *stateWeights, uint forwardBase, device const float *valueWeights,
    device const float *optionSubValues, uint optIdx,
    device const KernelScalars *scalars, device const float *lat, uint sOff, uint actDim, uint rank,
    device const uint *activeList, uint activeCount, uint planK, thread uint& raw, thread uint& san,
    device float *scratch, uint maxRank) {
    for (uint i = 0u; i < rank && i < maxRank; ++i) scratch[i] = sanitize(lat[sOff + i], raw, san);
    float rho = scalars->rewardRate;
    float vk = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint i = activeList[li];
        float x = scratch[i];
        vk += valueWeights[i] * x + optionSubValues[optIdx * rank + i] * x;
    }
    float acc = 0.0f;
    for (uint step = 0u; step < planK; ++step) {
        uint bestA = 0u;
        float bestV = -1e10f;
        for (uint cand = 0u; cand < actDim; ++cand) {
            float vCand = 0.0f;
            for (uint li = 0u; li < activeCount; ++li) {
                uint idx = activeList[li];
                float ps = 0.0f;
                for (uint lj = 0u; lj < activeCount; ++lj) {
                    uint jdx = activeList[lj];
                    ps += dyn_forward_ws(stateWeights, forwardBase, rank, actDim, idx, jdx) * scratch[jdx];
                }
                float pa = dyn_forward_wa(stateWeights, forwardBase, rank, actDim, idx, cand);
                float predNext = sanitize(ps + pa, raw, san);
                vCand += valueWeights[idx] * predNext + optionSubValues[optIdx * rank + idx] * predNext;
            }
            if (cand == 0u || vCand > bestV) {
                bestV = vCand;
                bestA = cand;
            }
        }
        for (uint feat = 0u; feat < rank && feat < maxRank; ++feat) {
            float ps = 0.0f;
            for (uint i = 0u; i < activeCount; ++i) {
                uint idx = activeList[i];
                ps += dyn_forward_ws(stateWeights, forwardBase, rank, actDim, feat, idx) * scratch[idx];
            }
            float pa = dyn_forward_wa(stateWeights, forwardBase, rank, actDim, feat, bestA);
            scratch[feat] = sanitize(ps + pa, raw, san);
        }
        float vNext = 0.0f;
        for (uint li = 0u; li < activeCount; ++li) {
            uint i = activeList[li];
            float x = scratch[i];
            vNext += valueWeights[i] * x + optionSubValues[optIdx * rank + i] * x;
        }
        acc += (vNext - vk) - rho;
        vk = vNext;
    }
    return acc;
}


// Device scratch layout (float indices from `arena + floatArenaDeviceScratchBase`; must match Swift `floatArenaDeviceScratchSize`).
inline uint oak_ds_pred_row(uint gid, uint rank) { return gid * 2u * rank; }
inline uint oak_ds_plan_kstep(uint batchSize, uint rank, uint gid) { return 2u * batchSize * rank + gid * rank; }
inline uint oak_ds_plan_probe(uint batchSize, uint rank, uint gid) { return 3u * batchSize * rank + gid * 128u; }
inline uint oak_ds_learn_base(uint batchSize, uint rank) { return 3u * batchSize * rank + 128u * batchSize; }

struct OakPhaseContext {
    uint obsDim, rank, actDim, batchSize, optionCount, primitivePacked, optionForwardStride, planK, baseObsDim;
    uint actionIsDiscrete;
    float gammaSub, betaLR, ridge, invT;
    uint activeCount;
    device KernelScalars *scalars;
    device float *encoderWeights;
    device float *policyWeights;
    device float *valueWeights;
    device float *rewardWeights;
    device float *stateWeights;
    device float *optionPolicyWeights;
    device float *optionValues;
    device float *optionCriticWeights;
    device float *featureSecondMoment;
    device float *featureResidual;
    device float *observation;
    device float *observationMean;
    device float *previousState;
    device float *previousAction;
    device float *reward;
    device float *state;
    device float *action;
    device float *latentError;
    device float *rewardPrediction;
    device float *td;
    device float *actionGrad;
    device float *rolloutState;
    device float *latentPower;
    device uint *done;
    device uint *activeMask;
    device uint *activeList;
    device uint *stompState;
    device uint *counters;
    device float *tdOptionArena;
    device float *gvfTdArena;
    device float *gvfHordeW;
    device float *gvfHordeU;
    device float *gvfHordeRho;
    device float *subtaskRhoArena;
    device float *optionBetaArena;
    device float *planDeltaArena;
    device float *keyboardArena;
    device float *keyboardPredArena;
    device float *utilityArena;
    device float *featureGenArena;
    device float *arena;
    device float *ds;
    uint floatArenaAutostepBase;
};

inline OakPhaseContext oak_make_phase_context(
    device float *params,
    device float *stateArena,
    device uint *uintArena,
    device KernelScalars *scalars,
    constant KernelConfig &config
) {
    const uint obsDim = config.obsDim;
    const uint rank = config.rank;
    const uint actDim = config.actDim;
    const uint batchSize = config.batchSize;
    const uint optionCount = max(config.optionCount, 1u);
    OakPhaseContext c;
    c.obsDim = obsDim;
    c.rank = rank;
    c.actDim = actDim;
    c.batchSize = batchSize;
    c.optionCount = optionCount;
    c.primitivePacked = config.primitivePackedSize;
    c.optionForwardStride = config.optionForwardStride;
    c.planK = max(1u, config.planK);
    c.baseObsDim = config.baseObsDim;
    c.actionIsDiscrete = config.actionIsDiscrete;
    c.gammaSub = float(config.gammaSubtaskMilli) / 1000000.0f;
    c.betaLR = float(config.betaLearnRateMilli) / 1000000.0f;
    c.scalars = scalars;
    c.ridge = capacity_ridge(rank, batchSize);
    c.invT = online_inv_time(scalars);
    c.activeCount = scalars->activeCount;

    c.encoderWeights = params + config.pEncoder;
    c.policyWeights = params + config.pPolicy;
    c.valueWeights = params + config.pValue;
    c.rewardWeights = params + config.pReward;
    c.stateWeights = params + config.pStateWeights;
    c.optionPolicyWeights = params + config.pOptPolicy;
    c.optionValues = params + config.pOptValues;
    c.optionCriticWeights = params + config.pOptCritic;
    c.featureSecondMoment = params + config.pFeatSecondMoment;
    c.featureResidual = params + config.pFeatResidual;

    c.observation = stateArena + config.sObservation;
    c.observationMean = stateArena + config.sObsMean;
    c.previousState = stateArena + config.sPrevState;
    c.previousAction = stateArena + config.sPrevAction;
    c.reward = stateArena + config.sReward;
    c.state = stateArena + config.sState;
    c.action = stateArena + config.sAction;
    c.latentError = stateArena + config.sLatentError;
    c.rewardPrediction = stateArena + config.sRewardPred;
    c.td = stateArena + config.sTd;
    c.actionGrad = stateArena + config.sActionGrad;
    c.rolloutState = stateArena + config.sRollout;
    c.latentPower = stateArena + config.sLatentPower;

    c.done = uintArena + config.uDone;
    device uint *activeState = uintArena + config.uActive;
    c.stompState = uintArena + config.uStomp;
    c.counters = uintArena + config.uCounters;
    c.activeMask = activeState;
    c.activeList = activeState + rank;

    device float *floatArena = stateArena + config.sLatentPower + config.floatArenaOffsetInLatentPower;
    c.arena = floatArena;
    c.ds = floatArena + config.floatArenaDeviceScratchBase;
    c.tdOptionArena = arena_ptr(floatArena, config.floatArenaTdOptionBase);
    c.gvfTdArena = arena_ptr(floatArena, config.floatArenaGvfTdBase);
    c.gvfHordeW = arena_ptr(floatArena, config.floatArenaGvfWeightsBase);
    c.gvfHordeU = arena_ptr(floatArena, config.floatArenaGvfCumulantBase);
    c.gvfHordeRho = arena_ptr(floatArena, config.floatArenaGvfRhoBase);
    c.subtaskRhoArena = arena_ptr(floatArena, config.floatArenaSubtaskRhoBase);
    c.optionBetaArena = arena_ptr(floatArena, config.floatArenaOptionBetaBase);
    c.planDeltaArena = arena_ptr(floatArena, config.floatArenaPlanDeltaBase);
    c.keyboardArena = arena_ptr(floatArena, config.floatArenaKeyboardBase);
    c.keyboardPredArena = arena_ptr(floatArena, config.floatArenaKeyboardPredBase);
    c.utilityArena = arena_ptr(floatArena, config.floatArenaUtilityBase);
    c.featureGenArena = arena_ptr(floatArena, config.floatArenaFeatureGenBase);
    c.floatArenaAutostepBase = config.floatArenaAutostepBase;
    return c;
}

inline void oak_phase_encode_impl(OakPhaseContext ctx, uint gid) {
    uint row = gid / ctx.rank;
    uint feat = gid % ctx.rank;
    uint raw = 0u, san = 0u;
    if (row < ctx.batchSize) {
        float cap = structural_activation_cap(ctx.rank, ctx.batchSize);
        float centered = sanitize(ctx.observation[row * ctx.obsDim + feat] - ctx.observationMean[feat], raw, san);
        float z = sanitize(centered * ctx.encoderWeights[feat], raw, san);
        ctx.state[row * ctx.rank + feat] = clamp(z, -cap, cap);
    }
}

inline void oak_phase_prediction_impl(OakPhaseContext ctx, uint gid) {
    if (gid >= ctx.batchSize) return;
    uint raw = 0u, san = 0u;
    float cap = structural_activation_cap(ctx.rank, ctx.batchSize);
    uint sOff = gid * ctx.rank;
    device float *dsp = ctx.ds + oak_ds_pred_row(gid, ctx.rank);
    device const uint *activeList = ctx.activeList;
    uint activeCount = ctx.activeCount;
    for (uint i = 0u; i < activeCount; ++i) {
        uint idx = activeList[i];
        dsp[i] = sanitize(ctx.previousState[sOff + idx], raw, san);
        dsp[ctx.rank + i] = sanitize(ctx.state[sOff + idx], raw, san);
    }
    float rPred = 0.0f;
    float vPrev = 0.0f;
    float vNextBoot = 0.0f;
    for (uint i = 0u; i < activeCount; ++i) {
        uint idx = activeList[i];
        float sp = dsp[i];
        float ss = dsp[ctx.rank + i];
        rPred += ctx.rewardWeights[idx] * sp;
        vPrev += ctx.valueWeights[idx] * sp;
        vNextBoot += ctx.valueWeights[idx] * ss;
    }
    ctx.rewardPrediction[gid] = clamp(sanitize(rPred, raw, san), -cap, cap);
    vPrev = clamp(sanitize(vPrev, raw, san), -cap, cap);
    vNextBoot = clamp(sanitize(vNextBoot, raw, san), -cap, cap);
    float terminalMask = ctx.done[gid] ? 0.0f : 1.0f;
    float vNow = terminalMask * vNextBoot;
    float delta = ctx.reward[gid] - ctx.scalars->rewardRate + vNow - vPrev;
    float prevNorm2 = 0.0f;
    for (uint i = 0u; i < activeCount; ++i) {
        float x = dsp[i];
        prevNorm2 += x * x;
    }
    float denom = 1.0f + prevNorm2 / float(max(activeCount, 1u));
    ctx.td[gid] = clamp(sanitize(delta / denom, raw, san), -cap, cap);
    float prevOptF = ctx.latentPower[gid];
    uint po = uint(floor(prevOptF + 0.5f));
    if (po >= ctx.optionCount) po = ctx.optionCount - 1u;
    float vSubPrev = dot_active(ctx.optionValues, po * ctx.rank, ctx.previousState, sOff, activeList, activeCount, raw, san);
    float vSubNext = dot_active(ctx.optionValues, po * ctx.rank, ctx.state, sOff, activeList, activeCount, raw, san);
    float vSubNextMasked = terminalMask * vSubNext;
    float cNext = (po < ctx.rank) ? sanitize(ctx.state[sOff + po], raw, san) : 0.0f;
    float rhoK = ctx.subtaskRhoArena[po];
    float deltaOpt = cNext - rhoK + ctx.gammaSub * vSubNextMasked - vSubPrev;
    ctx.tdOptionArena[gid] = clamp(sanitize(deltaOpt / denom, raw, san), -cap, cap);
    float predHorde = masked_dot(ctx.gvfHordeW, 0, ctx.previousState, sOff, activeList, activeCount, raw, san);
    float cNextHorde = masked_dot(ctx.gvfHordeU, 0, ctx.state, sOff, activeList, activeCount, raw, san);
    float vGnext = masked_dot(ctx.gvfHordeW, 0, ctx.state, sOff, activeList, activeCount, raw, san) * terminalMask;
    float rhoH = ctx.gvfHordeRho[0];
    float dHorde = cNextHorde - rhoH + ctx.gammaSub * vGnext - predHorde;
    ctx.gvfTdArena[gid] = clamp(sanitize(dHorde, raw, san), -cap, cap);
    float rowErrSum = 0.0f;
    for (uint g = 0u; g < ctx.rank; ++g) {
        float acc = 0.0f;
        for (uint i = 0u; i < activeCount; ++i) {
            uint idx = activeList[i];
            acc += gvf_weight(ctx.stateWeights, g, idx, ctx.rank) * dsp[i];
        }
        ctx.latentError[gid * ctx.rank + g] = clamp(sanitize(acc, raw, san), -cap, cap);
        rowErrSum += ctx.state[sOff + g] - ctx.latentError[gid * ctx.rank + g];
    }
    ctx.latentPower[ctx.batchSize + gid] = rowErrSum;
}

inline void oak_phase_forward_model_impl(OakPhaseContext ctx, uint gid) {
    uint row = gid / ctx.rank;
    uint feat = gid % ctx.rank;
    uint raw = 0u, san = 0u;
    if (row >= ctx.batchSize) return;
    uint sOff = row * ctx.rank;
    uint aOff = row * ctx.actDim;
    float prevOptF = ctx.latentPower[row];
    uint dynOpt = pick_option_slice(ctx.stompState, ctx.latentPower, row, ctx.rank, prevOptF, ctx.optionCount);
    uint blockBase = ctx.primitivePacked + dynOpt * ctx.optionForwardStride;
    if (feat == 0u) {
        ctx.stompState[4u * ctx.batchSize + row] = dynOpt;
    }
    float ps = 0.0f;
    for (uint i = 0u; i < ctx.activeCount; ++i) {
        uint idx = ctx.activeList[i];
        ps += opt_forward_ws(ctx.stateWeights, blockBase, ctx.rank, ctx.actDim, feat, idx) * sanitize(ctx.previousState[sOff + idx], raw, san);
    }
    float pa = 0.0f;
    for (uint a = 0u; a < ctx.actDim; ++a) {
        pa += opt_forward_wa(ctx.stateWeights, blockBase, ctx.rank, ctx.actDim, feat, a) * sanitize(ctx.previousAction[aOff + a], raw, san);
    }
    ctx.rolloutState[row * ctx.rank + feat] = sanitize(ps + pa, raw, san);
}

inline void oak_phase_learn_impl(OakPhaseContext ctx, uint gid) {
    const uint policyBase = 3u * ctx.rank;
    const uint gvfHordeLearnBase = 4u * ctx.rank;
    uint raw = 0u, san = 0u;
    const float invT = ctx.invT;
    const float ridge = ctx.ridge;
    device float *learnBase = ctx.ds + oak_ds_learn_base(ctx.batchSize, ctx.rank);

    if (gid < ctx.rank) {
        uint feat = gid;
        float weightUpdate = 0.0f;
        float utility = 0.0f;
        float smAcc = 0.0f;
        float gvfUtil = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float sVal = ctx.state[row * ctx.rank + feat];
            float tdVal = ctx.td[row];
            weightUpdate += tdVal * sVal;
            utility += fabs(tdVal * sVal);
            smAcc += sVal * sVal;
            for (uint g = 0u; g < ctx.rank; ++g) {
                float pred = ctx.latentError[row * ctx.rank + g];
                float tgt = ctx.state[row * ctx.rank + g];
                float err = tgt - pred;
                gvfUtil += fabs(err * sVal);
            }
        }
        float mBatch = smAcc / float(ctx.batchSize);
        float gFeat = weightUpdate / float(ctx.batchSize);
        device float *asBase = arena_ptr(ctx.arena, ctx.floatArenaAutostepBase);
        float vTr = asBase[feat * 2u];
        float al = asBase[feat * 2u + 1u];
        float eta = autostep_eta(ctx.rank);
        vTr += eta * (gFeat - vTr);
        float dTilde = gFeat - vTr;
        al *= exp(eta * gFeat * dTilde);
        al = clamp(al, 1e-8f, 1e2f);
        asBase[feat * 2u] = vTr;
        asBase[feat * 2u + 1u] = al;
        ctx.valueWeights[feat] += al * gFeat;
        float uBatch = utility / float(ctx.batchSize);
        float gvfBatch = gvfUtil / float(max(ctx.batchSize * ctx.rank, 1u));
        ctx.featureResidual[feat] += ((uBatch + gvfBatch) - ctx.featureResidual[feat]) * invT;
        ctx.featureSecondMoment[feat] += (mBatch - ctx.featureSecondMoment[feat]) * invT;
    }
    if (gid >= ctx.rank && gid < ctx.rank + ctx.rank) {
        uint g = gid - ctx.rank;
        uint gvfBase = g * ctx.rank;
        device float *gradAcc = learnBase + g * ctx.rank;
        for (uint i = 0u; i < ctx.rank; ++i) { gradAcc[i] = 0.0f; }
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            float err = sanitize(ctx.state[row * ctx.rank + g] - ctx.latentError[row * ctx.rank + g], raw, san);
            for (uint i = 0u; i < ctx.rank; ++i) {
                gradAcc[i] += err * sanitize(ctx.previousState[sOff + i], raw, san);
            }
        }
        for (uint i = 0u; i < ctx.rank; ++i) {
            float step = autostep_eta(ctx.rank) / (ctx.featureSecondMoment[i] + ridge);
            ctx.stateWeights[gvfBase + i] += (gradAcc[i] / float(ctx.batchSize)) * step;
        }
    }
    if (gid >= 2u * ctx.rank && gid < 3u * ctx.rank) {
        uint outRow = gid - 2u * ctx.rank;
        uint slot = ctx.rank * ctx.rank + outRow * (ctx.rank + ctx.actDim);
        device float *gradAcc = learnBase + slot;
        device float *gradAccA = gradAcc + ctx.rank;
        for (uint i = 0u; i < ctx.rank; ++i) { gradAcc[i] = 0.0f; }
        for (uint a = 0u; a < ctx.actDim; ++a) { gradAccA[a] = 0.0f; }
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            uint aOff = row * ctx.actDim;
            float err = sanitize(ctx.state[row * ctx.rank + outRow] - ctx.rolloutState[row * ctx.rank + outRow], raw, san);
            for (uint i = 0u; i < ctx.rank; ++i) {
                gradAcc[i] += err * sanitize(ctx.previousState[sOff + i], raw, san);
            }
            for (uint a = 0u; a < ctx.actDim; ++a) {
                gradAccA[a] += err * sanitize(ctx.previousAction[aOff + a], raw, san);
            }
        }
        float invB = 1.0f / float(ctx.batchSize);
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float wMask = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                if (batch_dynamics_opt(ctx.stompState, ctx.batchSize, row) == opt) wMask += 1.0f;
            }
            wMask *= invB;
            if (wMask < 1e-8f) continue;
            uint blockBase = ctx.primitivePacked + opt * ctx.optionForwardStride;
            for (uint i = 0u; i < ctx.rank; ++i) {
                float step = autostep_eta(ctx.rank) / (ctx.featureSecondMoment[i] + ridge);
                ctx.stateWeights[blockBase + outRow * (ctx.rank + ctx.actDim) + i] += gradAcc[i] * invB * step * wMask;
            }
            for (uint a = 0u; a < ctx.actDim; ++a) {
                float stepA = autostep_eta(ctx.rank) / (ctx.featureSecondMoment[outRow] + ridge);
                ctx.stateWeights[blockBase + outRow * (ctx.rank + ctx.actDim) + ctx.rank + a] += gradAccA[a] * invB * stepA * wMask;
            }
        }
    }

    if (gid >= policyBase && gid < policyBase + ctx.rank) {
        uint feat = gid - policyBase;
        float muF = 1.0f / (ctx.featureSecondMoment[feat] + ridge);
        float stepPol = autostep_eta(ctx.rank) * muF;
        float gradP = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            gradP += ctx.td[row] * sanitize(ctx.previousState[sOff + feat], raw, san);
        }
        float dPol = (gradP / float(ctx.batchSize)) * stepPol;
        for (uint actIdx = 0u; actIdx < ctx.actDim; ++actIdx) {
            ctx.policyWeights[actIdx * ctx.rank + feat] += dPol;
        }
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            uint aOff = row * ctx.actDim;
            float prevOptF = ctx.latentPower[row];
            uint po = uint(floor(prevOptF + 0.5f));
            if (po >= ctx.optionCount) po = ctx.optionCount - 1u;
            float sv = ctx.tdOptionArena[row] * sanitize(ctx.previousState[sOff + feat], raw, san);
            ctx.optionValues[po * ctx.rank + feat] += (sv / float(ctx.batchSize)) * stepPol;
            uint base = po * ctx.rank * ctx.actDim;
            for (uint actIdx = 0u; actIdx < ctx.actDim; ++actIdx) {
                float actScale = sanitize(ctx.previousAction[aOff + actIdx], raw, san);
                ctx.optionPolicyWeights[base + actIdx * ctx.rank + feat] += ((sv * actScale) / float(ctx.batchSize)) * stepPol;
            }
        }
        if (feat < ctx.baseObsDim) {
            float gradE = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                float centered = ctx.observation[row * ctx.obsDim + feat] - ctx.observationMean[feat];
                gradE += ctx.latentPower[ctx.batchSize + row] * centered;
            }
            ctx.encoderWeights[feat] += (gradE / float(ctx.batchSize)) * stepPol;
        }
        if (feat < ctx.rank) {
            float gradR = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                uint sOff = row * ctx.rank;
                float rErr = ctx.reward[row] - ctx.rewardPrediction[row];
                gradR += rErr * sanitize(ctx.previousState[sOff + feat], raw, san);
            }
            ctx.rewardWeights[feat] += (gradR / float(ctx.batchSize)) * stepPol;
        }
    }
    if (gid >= gvfHordeLearnBase && gid < gvfHordeLearnBase + ctx.rank) {
        uint feat = gid - gvfHordeLearnBase;
        float gradH = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            gradH += ctx.gvfTdArena[row] * sanitize(ctx.previousState[sOff + feat], raw, san);
        }
        float stepH = autostep_eta(ctx.rank) / (ctx.featureSecondMoment[feat] + ridge);
        ctx.gvfHordeW[feat] += (gradH / float(ctx.batchSize)) * stepH;
    }
}

inline void oak_phase_meta_policy_impl(OakPhaseContext ctx, device const float *actionBounds, uint gid) {
    uint raw = 0u, san = 0u;
        uint row = gid / ctx.actDim;
        uint actIdx = gid % ctx.actDim;
        if (row < ctx.batchSize) {
            uint sOff = row * ctx.rank;
            float prevOptF = ctx.latentPower[row];
            uint poPrev = uint(floor(prevOptF + 0.5f));
            if (poPrev >= ctx.optionCount) poPrev = ctx.optionCount - 1u;
            uint phase = ctx.stompState[row * 4u];
            uint elapsed = ctx.stompState[row * 4u + 1u];
            uint activeOption = 0u;
            float pickScore = 0.0f;
            if (phase == 1u && elapsed < ctx.rank) {
                activeOption = poPrev;
                for (uint i = 0u; i < ctx.activeCount; ++i) {
                    uint idx = ctx.activeList[i];
                    pickScore += ctx.optionValues[activeOption * ctx.rank + idx] * sanitize(ctx.state[sOff + idx], raw, san);
                }
            } else {
                // Option keyboard (Alberta Plan): Design B — one subtask/option episode selects a **single** terminal shaping
                // vector (`ctx.optionValues` row); chords are not a separate blend kernel (bounded `ctx.optionCount` slots).
                float kbSum = 0.0f;
                for (uint k = 0u; k < ctx.optionCount; ++k) kbSum += max(ctx.keyboardArena[row * ctx.optionCount + k], 1e-6f);
                float invKb = 1.0f / kbSum;
                float bestScore = -1e10f;
                for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
                    float score = 0.0f;
                    uint off = opt * ctx.rank;
                    for (uint i = 0u; i < ctx.activeCount; ++i) {
                        uint idx = ctx.activeList[i];
                        float sv = sanitize(ctx.state[sOff + idx], raw, san);
                        score += ctx.optionValues[off + idx] * sv;
                    }
                    // STOMP / reward-respecting tie: terminal feature emphasis for subtask `opt` (generic latent index).
                    if (opt < ctx.rank) {
                        score += sanitize(ctx.state[sOff + opt], raw, san) * (1.0f / float(max(ctx.rank, 1u)));
                    }
                    float w = ctx.keyboardArena[row * ctx.optionCount + opt] * invKb * float(ctx.optionCount);
                    float weighted = score * w;
                    if (opt == 0u || weighted > bestScore) {
                        bestScore = weighted;
                        activeOption = opt;
                    }
                }
                pickScore = bestScore;
                if (actIdx == 0u) {
                    ctx.stompState[row * 4u] = 1u;
                    ctx.stompState[row * 4u + 1u] = 0u;
                    uint ocR = ctx.optionCount * ctx.rank;
                    ctx.optionCriticWeights[ocR + activeOption] += (pickScore - ctx.optionCriticWeights[ocR + activeOption]) * ctx.invT;
                    const uint kbd0 = 2u * ctx.batchSize;
                    float ssum = 0.0f;
                    for (uint k = 0u; k < ctx.optionCount; ++k) {
                        float sk = 0.0f;
                        uint off = k * ctx.rank;
                        for (uint i = 0u; i < ctx.activeCount; ++i) {
                            uint idx = ctx.activeList[i];
                            sk += ctx.optionValues[off + idx] * sanitize(ctx.state[sOff + idx], raw, san);
                        }
                        if (k < ctx.rank) sk += sanitize(ctx.state[sOff + k], raw, san) * (1.0f / float(max(ctx.rank, 1u)));
                        ssum += max(sk, 0.0f);
                    }
                    float invSS = 1.0f / (ssum + 1e-5f);
                    for (uint k = 0u; k < ctx.optionCount; ++k) {
                        float sk = 0.0f;
                        uint off = k * ctx.rank;
                        for (uint i = 0u; i < ctx.activeCount; ++i) {
                            uint idx = ctx.activeList[i];
                            sk += ctx.optionValues[off + idx] * sanitize(ctx.state[sOff + idx], raw, san);
                        }
                        if (k < ctx.rank) sk += sanitize(ctx.state[sOff + k], raw, san) * (1.0f / float(max(ctx.rank, 1u)));
                        float tgt = max(sk, 0.0f) * invSS;
                        uint kix = kbd0 + row * ctx.optionCount + k;
                        float oldv = ctx.keyboardArena[row * ctx.optionCount + k];
                        ctx.keyboardArena[row * ctx.optionCount + k] = oldv + (tgt - oldv) * ctx.invT;
                        ctx.latentPower[kix] = ctx.keyboardArena[row * ctx.optionCount + k];
                    }
                }
            }
            uint base = activeOption * ctx.rank * ctx.actDim;
            float baseLogit = masked_dot(ctx.policyWeights, actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            float optLogit = masked_dot(ctx.optionPolicyWeights, base + actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            uint aOff = row * ctx.actDim;
            float lamPlan = plan_merge_lambda(ctx.rank, ctx.actDim);
            float planBump = ctx.planDeltaArena[aOff + actIdx];
            float logit = sanitize(baseLogit + optLogit + lamPlan * planBump, raw, san);
            float lo = actionBounds[actIdx];
            float hi = actionBounds[ctx.actDim + actIdx];
            float bounded = clamp(logit, lo, hi);
            ctx.action[aOff + actIdx] = bounded;
            ctx.actionGrad[aOff + actIdx] = bounded;
            ctx.latentPower[ctx.batchSize + row] = float(activeOption);
        }
        return;
}
inline void oak_phase_planning_impl(OakPhaseContext ctx, device const float *actionBounds, uint gid) {
    uint raw = 0u, san = 0u;
        if (gid < ctx.batchSize) {
            uint aOff = gid * ctx.actDim;
            uint sOff = gid * ctx.rank;
            float prevOptF = ctx.latentPower[gid];
            uint po = uint(floor(prevOptF + 0.5f));
            if (po >= ctx.optionCount) po = ctx.optionCount - 1u;
            uint forwardPlanBase = ctx.primitivePacked + po * ctx.optionForwardStride;
            float vNow = masked_dot(ctx.valueWeights, 0, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            float vSubNow = dot_active(ctx.optionValues, po * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            vNow += vSubNow;

            if ((ctx.actionIsDiscrete != 0u)) {
                uint bestA = 0u;
                float bestV = -1e10f;
                for (uint cand = 0u; cand < ctx.actDim; ++cand) {
                    float vCand = discrete_lookahead1(ctx.stateWeights, forwardPlanBase, ctx.valueWeights, ctx.optionValues, po,
                        ctx.state, sOff, ctx.actDim, ctx.rank, ctx.activeList, ctx.activeCount, cand, raw, san);
                    if (cand == 0u || vCand > bestV) {
                        bestV = vCand;
                        bestA = cand;
                    }
                }
                float adv1 = bestV - vNow;
                float adv = adv1;
                if (ctx.planK > 1u) {
                    device float *kscratch = ctx.ds + oak_ds_plan_kstep(ctx.batchSize, ctx.rank, gid);
                    const uint maxR = 384u;
                    uint capRank = min(ctx.rank, maxR);
                    float advK = kstep_plan_discrete(ctx.stateWeights, forwardPlanBase, ctx.valueWeights, ctx.optionValues, po,
                        ctx.scalars, ctx.state, sOff, ctx.actDim, ctx.rank, ctx.activeList, ctx.activeCount, ctx.planK, raw, san, kscratch, capRank);
                    adv = max(adv1, advK);
                }
                if (adv > 0.0f) {
                    float scale = adv / (1.0f + fabs(vNow));
                    for (uint a = 0u; a < ctx.actDim; ++a) {
                        ctx.planDeltaArena[aOff + a] = (a == bestA) ? scale : 0.0f;
                    }
                } else {
                    for (uint a = 0u; a < ctx.actDim; ++a) ctx.planDeltaArena[aOff + a] = 0.0f;
                }
            } else {
                float bestV = vNow;
                device float *probeBase = ctx.ds + oak_ds_plan_probe(ctx.batchSize, ctx.rank, gid);
                device float *probeAct = probeBase;
                device float *bestVec = probeBase + 64u;
                uint adCap = min(ctx.actDim, 64u);
                uint acP = min(ctx.activeCount, ctx.rank);
                for (uint a = 0u; a < adCap; ++a) bestVec[a] = ctx.previousAction[aOff + a];
                // Search control: probe action coordinates in an order derived from the active latent list (no full-state sweep).
                for (uint pi = 0u; pi < adCap; ++pi) {
                    uint dim = pi;
                    if (acP > 0u) {
                        dim = (ctx.activeList[pi % acP] + pi) % adCap;
                    }
                    float span = actionBounds[ctx.actDim + dim] - actionBounds[dim];
                    float eps = span / float(max(ctx.rank + ctx.actDim, 1u));
                    for (uint sp = 0u; sp < 2u; ++sp) {
                        float sgn = (sp == 0u) ? -1.0f : 1.0f;
                        for (uint a = 0u; a < adCap; ++a) probeAct[a] = ctx.previousAction[aOff + a];
                        float trial = probeAct[dim] + sgn * eps;
                        probeAct[dim] = clamp(trial, actionBounds[dim], actionBounds[ctx.actDim + dim]);
                        float vCand = lookahead_value_thread(ctx.stateWeights, forwardPlanBase, ctx.valueWeights, ctx.optionValues, po, ctx.state, sOff, ctx.actDim, ctx.rank, ctx.activeList, ctx.activeCount, raw, san, probeAct);
                        if (vCand > bestV) {
                            bestV = vCand;
                            for (uint a = 0u; a < adCap; ++a) bestVec[a] = probeAct[a];
                        }
                    }
                }
                float adv = bestV - vNow;
                if (adv > 0.0f) {
                    float scale = adv / (1.0f + fabs(vNow));
                    for (uint a = 0u; a < adCap; ++a) {
                        float base = ctx.previousAction[aOff + a];
                        ctx.planDeltaArena[aOff + a] = scale * (bestVec[a] - base);
                    }
                    for (uint a = adCap; a < ctx.actDim; ++a) ctx.planDeltaArena[aOff + a] = 0.0f;
                } else {
                    for (uint a = 0u; a < ctx.actDim; ++a) ctx.planDeltaArena[aOff + a] = 0.0f;
                }
            }
        }
        return;
}
inline void oak_phase_bookkeeping_impl(OakPhaseContext ctx, device KernelStats *statsOut, constant KernelConfig &config, uint gid) {
    if (gid != 0u) return;
        uint rawB = 0u, sanB = 0u;
        float invTUfeat = online_inv_time(ctx.scalars);
        for (uint i = 0u; i < ctx.rank; ++i) {
            float up = ctx.utilityArena[ctx.rank + 2u * ctx.optionCount + i];
            ctx.featureResidual[i] += invTUfeat * up;
        }
        float totalUtility = 0.0f;
        for (uint i = 0u; i < ctx.rank; ++i) totalUtility += ctx.featureResidual[i];
        float threshold = totalUtility / float(max(ctx.rank * ctx.rank, 1u));
        uint count = 0u;
        for (uint i = 0u; i < ctx.rank; ++i) {
            if (ctx.featureResidual[i] > threshold) {
                ctx.activeMask[i] = 1u;
                ctx.activeList[count++] = i;
            } else {
                ctx.activeMask[i] = 0u;
            }
        }
        if (count == 0u) {
            ctx.activeMask[0] = 1u;
            ctx.activeList[0] = 0u;
            count = 1u;
        }
        ctx.scalars->activeCount = count;
        const uint ac = max(ctx.scalars->activeCount, 1u);
        // Oak / FC-STOMP: swap one low-utility active feature for the strongest inactive candidate (single swap, bounded work).
        if (count > 1u && count < ctx.rank) {
            uint worstIdx = ctx.activeList[0];
            float worstRes = ctx.featureResidual[worstIdx];
            for (uint j = 1u; j < count; ++j) {
                uint ix = ctx.activeList[j];
                float r = ctx.featureResidual[ix];
                if (r < worstRes) { worstRes = r; worstIdx = ix; }
            }
            uint bestIn = 0u;
            float bestR = 0.0f;
            bool found = false;
            for (uint i = 0u; i < ctx.rank; ++i) {
                if (ctx.activeMask[i] != 0u) continue;
                float r = ctx.featureResidual[i];
                if (!found || r > bestR) { bestR = r; bestIn = i; found = true; }
            }
            if (found && bestR > worstRes * 1.01f) {
                ctx.activeMask[worstIdx] = 0u;
                ctx.activeMask[bestIn] = 1u;
                for (uint j = 0u; j < count; ++j) {
                    if (ctx.activeList[j] == worstIdx) { ctx.activeList[j] = bestIn; break; }
                }
                uint gdir = max(config.genDirectionCount, 1u);
                float bestScore = -1e10f;
                uint bestM = 0u;
                for (uint mm = 0u; mm < gdir; ++mm) {
                    float score = 0.0f;
                    for (uint j = 0u; j < ctx.rank; ++j) {
                        score += ctx.utilityArena[j] * ctx.featureGenArena[mm * ctx.rank + j];
                    }
                    if (mm == 0u || score > bestScore) {
                        bestScore = score;
                        bestM = mm;
                    }
                }
                ctx.scalars->admissionCursor = (ctx.scalars->admissionCursor + 1u) % gdir;
                uint m = bestM;
                float mix = 0.0f;
                for (uint j = 0u; j < ctx.rank; ++j) {
                    mix += ctx.featureGenArena[m * ctx.rank + j] * ctx.encoderWeights[j];
                }
                ctx.encoderWeights[worstIdx] = sanitize(mix / float(max(ctx.rank, 1u)), rawB, sanB);
            }
        }
        // STOMP option rollout: reward-respecting stop (zStop vs subtask value) + horizon cap (average-reward Oak adaptation).
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint p = ctx.stompState[row * 4u];
            if (p != 1u) continue;
            uint elapsed = ctx.stompState[row * 4u + 1u];
            uint sOff = row * ctx.rank;
            float prevOptF = ctx.latentPower[row];
            uint po = uint(floor(prevOptF + 0.5f));
            if (po >= ctx.optionCount) po = ctx.optionCount - 1u;
            float vSub = 0.0f;
            float zStop = 0.0f;
            for (uint li = 0u; li < ac; ++li) {
                uint idx = ctx.activeList[li];
                float sv = ctx.state[sOff + idx];
                if (!isfinite(sv)) sv = 0.0f;
                vSub += ctx.optionValues[po * ctx.rank + idx] * sv;
                zStop += ctx.valueWeights[idx] * sv;
            }
            if (po < ctx.rank) {
                float t = ctx.state[sOff + po];
                if (!isfinite(t)) t = 0.0f;
                zStop += t * (1.0f / float(max(ctx.rank, 1u)));
            }
            bool horizon = (elapsed + 1u >= ctx.rank);
            bool valueStop = (zStop >= vSub);
            bool betaTerm = (float(elapsed) >= ctx.optionBetaArena[po] * float(max(ctx.rank, 1u)));
            if (horizon || valueStop || betaTerm) {
                float kbSumT = 0.0f;
                for (uint k = 0u; k < ctx.optionCount; ++k) kbSumT += max(ctx.keyboardArena[row * ctx.optionCount + k], 1e-6f);
                float invKbT = 1.0f / kbSumT;
                float ckbd = 0.0f;
                for (uint k = 0u; k < ctx.optionCount; ++k) {
                    float wk = ctx.keyboardArena[row * ctx.optionCount + k] * invKbT * float(ctx.optionCount);
                    float zk = (k < ctx.rank) ? ctx.state[sOff + k] : 0.0f;
                    ckbd += wk * zk;
                }
                float prevKb = ctx.keyboardPredArena[po];
                float invL = online_inv_time(ctx.scalars);
                ctx.keyboardPredArena[po] = prevKb + invL * (ckbd - prevKb);
                ctx.stompState[row * 4u] = 0u;
                ctx.stompState[row * 4u + 1u] = 0u;
            } else {
                ctx.stompState[row * 4u + 1u] = elapsed + 1u;
            }
        }

        float batchR = 0.0f;
        for (uint i = 0u; i < ctx.batchSize; ++i) batchR += ctx.reward[i];
        float meanR = batchR / float(ctx.batchSize);
        float t = float(max(ctx.scalars->stepCount + 1u, 1u));
        ctx.scalars->rewardRate += (meanR - ctx.scalars->rewardRate) / t;
        ctx.scalars->stepCount++;
        float invLRho = online_inv_time(ctx.scalars);
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float cacc = 0.0f;
            uint cc = 0u;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                float prevOptF = ctx.latentPower[row];
                uint po = uint(floor(prevOptF + 0.5f));
                if (po >= ctx.optionCount) po = ctx.optionCount - 1u;
                if (po != opt) continue;
                if (opt < ctx.rank) {
                    float sv = ctx.state[row * ctx.rank + opt];
                    if (!isfinite(sv)) sv = 0.0f;
                    cacc += sv;
                    cc++;
                }
            }
            float cmean = cc > 0u ? cacc / float(cc) : 0.0f;
            ctx.subtaskRhoArena[opt] += invLRho * (cmean - ctx.subtaskRhoArena[opt]);
        }
        float cHsum = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff2 = row * ctx.rank;
            cHsum += masked_dot(ctx.gvfHordeU, 0, ctx.state, sOff2, ctx.activeList, ac, rawB, sanB);
        }
        float cHmean = cHsum / float(max(ctx.batchSize, 1u));
        ctx.gvfHordeRho[0] += invLRho * (cHmean - ctx.gvfHordeRho[0]);

        float lmse = 0.0f;
        float rmse = 0.0f;
        float vtrace = 0.0f;
        float planL2 = 0.0f;
        float actDelta = 0.0f;
        float maxLat = 0.0f;
        float maxRP = 0.0f;
        float maxPol = 0.0f;
        uint nraw = 0u;
        uint nsan = 0u;
        uint nEnc = 0u;
        uint nLearn = 0u;
        uint nPlan = 0u;
        float denom = float(max(ctx.batchSize * ctx.rank, 1u));
        uint stRaw = 0u;
        uint stSan = 0u;
        const uint nLatTerms = max(ctx.batchSize * ctx.rank, 1u);
        const uint nRewTerms = max(ctx.batchSize, 1u);
        const uint nActTerms = max(ctx.batchSize * ctx.actDim, 1u);
        const float boundLat = stats_abs_bound_for_sum_squares(nLatTerms);
        const float boundRew = stats_abs_bound_for_sum_squares(nRewTerms);
        const float boundAct = stats_abs_bound_for_sum_squares(nActTerms);
        // Scale-invariant telemetry: each squared term is divided by bound² so per-term contribution ∈ [0,1] and reported
        // means stay O(1) instead of pinning to FLT_MAX-derived ceilings when raw errors are large.
        const float invBoundLat = 1.0f / boundLat;
        const float invBoundRew = 1.0f / boundRew;
        const float invBoundAct = 1.0f / boundAct;
        // One pass over rows: fused ctx.rank stats + obs sums for online mean (ctx.rank/ctx.obsDim ≤ 384).
        device float *sumObsByF = ctx.ds;
        for (uint f = 0u; f < ctx.obsDim; ++f) { sumObsByF[f] = 0.0f; }

        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            uint aOff = row * ctx.actDim;
            for (uint k = 0u; k < ctx.rank; ++k) {
                float pred = ctx.latentError[row * ctx.rank + k];
                float tgt = ctx.state[row * ctx.rank + k];
                float ds = sanitize(tgt - pred, stRaw, stSan);
                float dc = clamp(ds, -boundLat, boundLat);
                float nlc = dc * invBoundLat;
                lmse = fma(nlc, nlc, lmse);
                maxLat = max(maxLat, fabs(ds));
                float sv = ctx.state[sOff + k];
                if (!isfinite(sv)) nLearn++;
                float sc = sanitize(sv, stRaw, stSan);
                float vc = clamp(sc, -boundLat, boundLat);
                float nvc = vc * invBoundLat;
                vtrace = fma(nvc, nvc, vtrace);
            }
            float rds = sanitize(ctx.reward[row] - ctx.rewardPrediction[row], stRaw, stSan);
            float rc = clamp(rds, -boundRew, boundRew);
            float nrc = rc * invBoundRew;
            rmse = fma(nrc, nrc, rmse);
            maxRP = max(maxRP, fabs(rds));
            float pa = 0.0f;
            float pb = 0.0f;
            for (uint a = 0u; a < ctx.actDim; ++a) {
                float x = ctx.action[aOff + a];
                float yg = ctx.actionGrad[aOff + a];
                float ps = sanitize(x - yg, stRaw, stSan);
                float pc = clamp(ps, -boundAct, boundAct);
                float npc = pc * invBoundAct;
                pa = fma(npc, npc, pa);
                maxPol = max(maxPol, fabs(x));
                if (!isfinite(x)) nPlan++;
                if (!isfinite(yg)) nPlan++;
                float py = ctx.previousAction[aOff + a];
                float pbs = sanitize(x - py, stRaw, stSan);
                float pbc = clamp(pbs, -boundAct, boundAct);
                float npb = pbc * invBoundAct;
                pb = fma(npb, npb, pb);
            }
            planL2 += pa;
            actDelta += pb;
            for (uint f = 0u; f < ctx.obsDim; ++f) {
                float o = ctx.observation[row * ctx.obsDim + f];
                if (!isfinite(o)) nEnc++;
                sumObsByF[f] += o;
            }
        }
        nraw = nEnc + nLearn + nPlan + stRaw;
        nsan = stSan;
        float invBatch = 1.0f / float(max(ctx.batchSize, 1u));
        float invStep = 1.0f / float(max(ctx.scalars->stepCount, 1u));
        for (uint f = 0u; f < ctx.obsDim; ++f) {
            float batchMean = sumObsByF[f] * invBatch;
            ctx.observationMean[f] += invStep * (batchMean - ctx.observationMean[f]);
        }
        float latentPredMean = lmse / denom;
        const uint kbd0bk = 2u * ctx.batchSize;
        const uint ofb0 = kbd0bk + ctx.batchSize * ctx.optionCount;
        // Match latentPredictionMSE: sanitize, clamp to stats_abs_bound, divide by bound² so telemetry stays O(1) and finite.
        float optPredAcc = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            for (uint k = 0u; k < ctx.rank; ++k) {
                float ds = sanitize(ctx.state[sOff + k] - ctx.rolloutState[row * ctx.rank + k], stRaw, stSan);
                float dc = clamp(ds, -boundLat, boundLat);
                float nlc = dc * invBoundLat;
                optPredAcc = fma(nlc, nlc, optPredAcc);
            }
        }
        statsOut->optionModelPredictionMSE = optPredAcc / denom;
        float planNonDefault = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float prevPo = ctx.latentPower[row];
            uint ip = uint(floor(prevPo + 0.5f));
            if (ip > 0u && ip < ctx.optionCount) planNonDefault += 1.0f;
        }
        statsOut->planningUsedOptionDynamicsMean = planNonDefault * invBatch;
        float invLife = 1.0f / float(max(ctx.scalars->stepCount, 1u));
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float sumErr = 0.0f;
            uint cntTerms = 0u;
            uint cntRows = 0u;
            uint cntPlanPick = 0u;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                float prevPo = ctx.latentPower[row];
                uint poP = uint(floor(prevPo + 0.5f));
                if (poP >= ctx.optionCount) poP = ctx.optionCount - 1u;
                if (poP == opt) cntPlanPick++;
                if (batch_dynamics_opt(ctx.stompState, ctx.batchSize, row) != opt) continue;
                cntRows++;
                uint sOff = row * ctx.rank;
                for (uint k = 0u; k < ctx.rank; ++k) {
                    float ds = sanitize(ctx.state[sOff + k] - ctx.rolloutState[row * ctx.rank + k], stRaw, stSan);
                    float dc = clamp(ds, -boundLat, boundLat);
                    float nlc = dc * invBoundLat;
                    sumErr = fma(nlc, nlc, sumErr);
                }
                cntTerms += ctx.rank;
            }
            float meanErr = cntTerms > 0u ? sumErr / float(cntTerms) : 0.0f;
            float prevE = ctx.latentPower[ofb0 + opt];
            ctx.latentPower[ofb0 + opt] = prevE + (meanErr - prevE) * invLife;
            float useFrac = float(cntRows) / float(max(ctx.batchSize, 1u));
            uint uix = ofb0 + ctx.optionCount + opt;
            float prevU = ctx.latentPower[uix];
            ctx.latentPower[uix] = prevU + (useFrac - prevU) * invLife;
            float planFrac = float(cntPlanPick) * invBatch;
            uint pix = ofb0 + 2u * ctx.optionCount + opt;
            float prevP = ctx.latentPower[pix];
            ctx.latentPower[pix] = prevP + (planFrac - prevP) * invLife;
            ctx.optionBetaArena[opt] += ctx.betaLR * (meanErr - ctx.optionBetaArena[opt]) * invLife;
        }
        float reinits = 0.0f;
        float invKey = 1.0f / float(max(ctx.optionCount, 1u));
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float oe = ctx.latentPower[ofb0 + opt];
            float ou = ctx.latentPower[ofb0 + ctx.optionCount + opt];
            float opPick = ctx.latentPower[ofb0 + 2u * ctx.optionCount + opt];
            if (oe > latentPredMean * 4.0f && ou < 0.05f && opPick < 0.05f) {
                uint bb = ctx.primitivePacked + opt * ctx.optionForwardStride;
                for (uint i = 0u; i < ctx.optionForwardStride; ++i) {
                    ctx.stateWeights[bb + i] = 0.0f;
                }
                uint polBase = opt * ctx.rank * ctx.actDim;
                for (uint i = 0u; i < ctx.rank * ctx.actDim; ++i) {
                    ctx.optionPolicyWeights[polBase + i] = 0.0f;
                }
                uint mN = ctx.scalars->admissionCursor % max(config.genDirectionCount, 1u);
                for (uint i = 0u; i < ctx.rank; ++i) {
                    ctx.optionValues[opt * ctx.rank + i] = ctx.featureGenArena[mN * ctx.rank + i];
                }
                uint ocR = ctx.optionCount * ctx.rank;
                for (uint i = 0u; i < ctx.rank; ++i) {
                    ctx.optionCriticWeights[opt * ctx.rank + i] = 0.0f;
                }
                ctx.optionCriticWeights[ocR + opt] = 0.0f;
                for (uint row = 0u; row < ctx.batchSize; ++row) {
                    ctx.latentPower[kbd0bk + row * ctx.optionCount + opt] = invKey;
                    ctx.keyboardArena[row * ctx.optionCount + opt] = invKey;
                }
                ctx.latentPower[ofb0 + opt] = 0.0f;
                ctx.latentPower[ofb0 + ctx.optionCount + opt] = 0.0f;
                ctx.latentPower[ofb0 + 2u * ctx.optionCount + opt] = 0.0f;
                reinits += 1.0f;
            }
        }
        statsOut->oakOptionReinitCount = reinits;
        float invBU = 1.0f / float(max(ctx.batchSize, 1u));
        float invTU = online_inv_time(ctx.scalars);
        float uFeatSum = 0.0f;
        for (uint i = 0u; i < ctx.rank; ++i) {
            float sacc = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                sacc += fabs(ctx.td[row] * ctx.state[row * ctx.rank + i]);
            }
            float nv = sacc * invBU;
            float prev = ctx.utilityArena[i];
            ctx.utilityArena[i] = prev + invTU * (nv - prev);
            uFeatSum += ctx.utilityArena[i];
        }
        float uSubSum = 0.0f;
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float sacc = 0.0f;
            uint cnt = 0u;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                float prevOptF = ctx.latentPower[row];
                uint poR = uint(floor(prevOptF + 0.5f));
                if (poR >= ctx.optionCount) poR = ctx.optionCount - 1u;
                if (poR != opt) continue;
                sacc += fabs(ctx.tdOptionArena[row]);
                cnt++;
            }
            float nv = cnt > 0u ? sacc / float(cnt) : 0.0f;
            float prev = ctx.utilityArena[ctx.rank + opt];
            ctx.utilityArena[ctx.rank + opt] = prev + invTU * (nv - prev);
            uSubSum += ctx.utilityArena[ctx.rank + opt];
        }
        float uOptSum = 0.0f;
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float prev = ctx.utilityArena[ctx.rank + ctx.optionCount + opt];
            float nv = ctx.latentPower[ofb0 + opt];
            ctx.utilityArena[ctx.rank + ctx.optionCount + opt] = prev + invTU * (nv - prev);
            uOptSum += ctx.utilityArena[ctx.rank + ctx.optionCount + opt];
        }
        float pdMag = 0.0f;
        for (uint a = 0u; a < ctx.actDim; ++a) pdMag += fabs(ctx.planDeltaArena[a]);
        float uPlanSum = 0.0f;
        for (uint i = 0u; i < ctx.rank; ++i) {
            float nv = pdMag * fabs(ctx.state[i]);
            float prev = ctx.utilityArena[ctx.rank + 2u * ctx.optionCount + i];
            ctx.utilityArena[ctx.rank + 2u * ctx.optionCount + i] = prev + invTU * (nv - prev);
            uPlanSum += ctx.utilityArena[ctx.rank + 2u * ctx.optionCount + i];
        }
        statsOut->utilityFeatureMean = uFeatSum / float(max(ctx.rank, 1u));
        statsOut->utilitySubtaskMean = uSubSum / float(max(ctx.optionCount, 1u));
        statsOut->utilityOptionMean = uOptSum / float(max(ctx.optionCount, 1u));
        statsOut->utilityPlanningMean = uPlanSum / float(max(ctx.rank, 1u));
        statsOut->rewardRate = ctx.scalars->rewardRate;
        statsOut->latentPredictionMSE = lmse / denom;
        statsOut->rewardPredictionMSE = rmse / float(max(ctx.batchSize, 1u));
        statsOut->latentStateMeanSquare = vtrace / float(max(ctx.batchSize * ctx.rank, 1u));
        statsOut->activeRank = float(ctx.scalars->activeCount);
        statsOut->planningActionChangeL2Mean = planL2 / float(max(ctx.batchSize * ctx.actDim, 1u));
        statsOut->actionDeltaL2Mean = actDelta / float(max(ctx.batchSize * ctx.actDim, 1u));
        statsOut->rawNonFiniteCount = float(nraw);
        statsOut->sanitizedNonFiniteCount = float(nsan);
        statsOut->encodeNonFiniteCount = float(nEnc);
        statsOut->learnNonFiniteCount = float(nLearn);
        statsOut->planNonFiniteCount = float(nPlan);
        statsOut->maxAbsLatentPrediction = maxLat;
        statsOut->maxAbsRewardPrediction = maxRP;
        statsOut->maxAbsPolicyOutput = maxPol;
        ctx.counters[0] = nEnc;
        ctx.counters[1] = nLearn;
        ctx.counters[2] = nPlan;
}

#define OAK_KERNEL_SIG \
    device float *params [[buffer(0)]], \
    device float *stateArena [[buffer(1)]], \
    device uint *uintArena [[buffer(2)]], \
    device KernelScalars *scalars [[buffer(3)]], \
    device KernelStats *statsOut [[buffer(4)]], \
    device const float *actionBounds [[buffer(5)]], \
    constant KernelConfig &config [[buffer(6)]], \
    uint gid [[thread_position_in_grid]]

kernel void oak_phase_encode(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_encode_impl(ctx, gid);
}
kernel void oak_phase_prediction(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_prediction_impl(ctx, gid);
}
kernel void oak_phase_forward_model(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_forward_model_impl(ctx, gid);
}
kernel void oak_phase_learn(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_learn_impl(ctx, gid);
}
kernel void oak_phase_planning(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_planning_impl(ctx, actionBounds, gid);
}
kernel void oak_phase_meta_policy(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_meta_policy_impl(ctx, actionBounds, gid);
}
kernel void oak_phase_bookkeeping(OAK_KERNEL_SIG) {
    OakPhaseContext ctx = oak_make_phase_context(params, stateArena, uintArena, scalars, config);
    oak_phase_bookkeeping_impl(ctx, statsOut, config, gid);
}
#undef OAK_KERNEL_SIG
