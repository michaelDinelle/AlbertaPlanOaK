#include <metal_stdlib>
#include <float.h>
using namespace metal;

/// Specialized at pipeline creation (must match Swift `MTLFunctionConstantValues` index 0).
[[function_constant(0)]] constant bool fc_action_is_discrete;

/// Per-term magnitude bound so `termCount` summed squares stay in `float` range: each term ≤ `FLT_MAX/n` ⇒ sum ≤ `FLT_MAX`.
/// **Not** reward shaping or display scaling (Alberta invariant 7): IEEE overflow hygiene for diagnostic aggregates only;
/// learning uses unsquared `sanitize` paths elsewhere. `FLT_MAX` from `<float.h>`; `sqrt` from Metal.
inline float stats_abs_bound_for_sum_squares(uint termCount) {
    float n = float(max(termCount, 1u));
    return sqrt(FLT_MAX / n);
}

// Future: optional `simdgroup_matrix` GVF paths should gate on Apple GPU Family 9+ with a scalar half/float fallback
// (same learning semantics). Threadgroup `activeList` tiling stays trace-driven (small activeCount often loses vs device reads).

// Oak / Alberta Plan (12 steps): continual phases every tick (no train/eval). GVF/planning: single Metal path (no duplicate
// MPS matmul); Swift `structuralMaxGvfHeads` caps future batched GenTD width. Packed `params` + `state` + `uint` arenas
// keep per-kernel buffer bindings ≤ 8 (split pipelines share the same layout).
// Shared latent `state` is read-only within a phase; consumers share one representation (AGENTS.md).

// GenTD batched horde: Xu et al., "GenTD: A Generalized Algorithm for TD Learning" (NeurIPS 2022); scale reference:
// Martin et al. "The Deep RL Bootcamp: 4000 GVFs" (tutorial) — literature anchors for multi-demon TD in one kernel.
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
    uint floatArenaGvfTdBase;
    uint floatArenaGvfRhoBase;
    uint floatArenaGvfGenAuxBase;
    uint floatArenaPolicyTraceBase;
    uint floatArenaMgscEtaBase;
    uint floatArenaKeyboardBase;
    uint floatArenaKeyboardPredBase;
    uint floatArenaUtilityBase;
    uint floatArenaSubtaskRhoBase;
    uint floatArenaOptionBetaBase;
    uint floatArenaDeviceScratchBase;
    uint floatArenaOffsetInLatentPower;
    uint phase;
    uint gammaSubtaskMilli;
    uint betaLearnRateMilli;
    /// Batched GVF demon count: `min(4096, 16*rank)`; structural (Swift).
    uint nGvfHeads;
    float gammaPhi0;
    float gammaPhi1;
    float gammaPhi2;
    /// Precomputed in Swift (same formulas as former inline helpers): capacity ridge, structural cap, NLMS scale, telemetry bounds.
    float capacityRidge;
    float structuralCap;
    float nlmsStepScale;
    float sqrtFltEpsilon;
    float boundLat;
    float boundRew;
    float boundAct;
    float invBoundLat;
    float invBoundRew;
    float invBoundAct;
    /// Equals `1/sqrt(rank+actDim)` — multiply instead of `plan_merge_lambda` per thread.
    float rsqrtPlanLambda;
};
struct KernelScalars { uint stepCount, rewardCount, activeCount, admissionCursor; float rewardRate; float pEff; };
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

/// Float arena weights (e.g. GVF stack); `w` is fp32 storage.
inline float masked_dot_arena(device const float *w, uint wOff, device const float *v, uint vOff, device const uint *active, uint count, thread uint& r, thread uint& s) {
    float sum = 0.0f;
    uint pairs = 0u;
    int ic = int(count);
    int ii = 0;
    for (; ii + 3 < ic; ii += 4) {
        uint i0 = active[ii];
        uint i1 = active[ii + 1];
        uint i2 = active[ii + 2];
        uint i3 = active[ii + 3];
        float vx0 = v[vOff + i0];
        float vx1 = v[vOff + i1];
        float vx2 = v[vOff + i2];
        float vx3 = v[vOff + i3];
        pairs += select(1u, 0u, isfinite(vx0));
        pairs += select(1u, 0u, isfinite(vx1));
        pairs += select(1u, 0u, isfinite(vx2));
        pairs += select(1u, 0u, isfinite(vx3));
        float vv0 = select(0.0f, vx0, isfinite(vx0));
        float vv1 = select(0.0f, vx1, isfinite(vx1));
        float vv2 = select(0.0f, vx2, isfinite(vx2));
        float vv3 = select(0.0f, vx3, isfinite(vx3));
        float4 ww = float4(w[wOff + i0], w[wOff + i1], w[wOff + i2], w[wOff + i3]);
        float4 vv = float4(vv0, vv1, vv2, vv3);
        sum += dot(ww, vv);
    }
    for (; ii < ic; ++ii) {
        uint idx = active[ii];
        float vx = v[vOff + idx];
        pairs += select(1u, 0u, isfinite(vx));
        float vv = select(0.0f, vx, isfinite(vx));
        sum = fma(w[wOff + idx], vv, sum);
    }
    pairs += select(1u, 0u, isfinite(sum));
    sum = select(0.0f, sum, isfinite(sum));
    r += pairs;
    s += pairs;
    return sum;
}

/// Param arena (`params` buffer) stored as **half**; accumulators stay float.
inline float masked_dot_param(device const half *w, uint wOff, device const float *v, uint vOff, device const uint *active, uint count, thread uint& r, thread uint& s) {
    float sum = 0.0f;
    uint pairs = 0u;
    int ic = int(count);
    int ii = 0;
    for (; ii + 3 < ic; ii += 4) {
        uint i0 = active[ii];
        uint i1 = active[ii + 1];
        uint i2 = active[ii + 2];
        uint i3 = active[ii + 3];
        float vx0 = v[vOff + i0];
        float vx1 = v[vOff + i1];
        float vx2 = v[vOff + i2];
        float vx3 = v[vOff + i3];
        pairs += select(1u, 0u, isfinite(vx0));
        pairs += select(1u, 0u, isfinite(vx1));
        pairs += select(1u, 0u, isfinite(vx2));
        pairs += select(1u, 0u, isfinite(vx3));
        float vv0 = select(0.0f, vx0, isfinite(vx0));
        float vv1 = select(0.0f, vx1, isfinite(vx1));
        float vv2 = select(0.0f, vx2, isfinite(vx2));
        float vv3 = select(0.0f, vx3, isfinite(vx3));
        float4 ww = float4(float(w[wOff + i0]), float(w[wOff + i1]), float(w[wOff + i2]), float(w[wOff + i3]));
        float4 vv = float4(vv0, vv1, vv2, vv3);
        sum += dot(ww, vv);
    }
    for (; ii < ic; ++ii) {
        uint idx = active[ii];
        float vx = v[vOff + idx];
        pairs += select(1u, 0u, isfinite(vx));
        float vv = select(0.0f, vx, isfinite(vx));
        sum = fma(float(w[wOff + idx]), vv, sum);
    }
    pairs += select(1u, 0u, isfinite(sum));
    sum = select(0.0f, sum, isfinite(sum));
    r += pairs;
    s += pairs;
    return sum;
}

inline float dot_active_arena(device const float *w, uint wOff, device const float *v, uint vOff, device const uint *activeList, uint activeCount, thread uint& raw, thread uint& san) {
    return masked_dot_arena(w, wOff, v, vOff, activeList, activeCount, raw, san);
}

inline float dot_active_param(device const half *w, uint wOff, device const float *v, uint vOff, device const uint *activeList, uint activeCount, thread uint& raw, thread uint& san) {
    return masked_dot_param(w, wOff, v, vOff, activeList, activeCount, raw, san);
}

/// Sparse dot `sum_i w[wOff+list[i]] * val[i]` where `val[i]` are **already** sanitized (e.g. prediction scratch).
inline float dot_sparse_weights_preval(device const half *w, uint wOff, device const float *val, device const uint *list, uint n) {
    float sum = 0.0f;
    uint i = 0u;
    for (; i + 3u < n; i += 4u) {
        uint i0 = list[i];
        uint i1 = list[i + 1u];
        uint i2 = list[i + 2u];
        uint i3 = list[i + 3u];
        float4 ww = float4(float(w[wOff + i0]), float(w[wOff + i1]), float(w[wOff + i2]), float(w[wOff + i3]));
        float4 vv = float4(val[i], val[i + 1u], val[i + 2u], val[i + 3u]);
        sum += dot(ww, vv);
    }
    for (; i < n; ++i) {
        sum = fma(float(w[wOff + list[i]]), val[i], sum);
    }
    return sum;
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

/// NLMS / policy step scale: 1/(rank×batch) capacity inverse (structural; aligns with streaming AC step-size practice, e.g. arXiv:2410.14606, without env-specific tuning).
inline float nlms_step_scale(uint rank, uint batchSize) {
    return 1.0f / float(max(rank * max(batchSize, 1u), 1u));
}

/// Running max scale from Autostep-style feature second moments (`featureSecondMoment` EMA of batch mean \(x^2\)).
inline float feature_sigma_max(device const half *featureSecondMoment, uint rank) {
    float m = 0.0f;
    for (uint i = 0u; i < rank; ++i) m = max(m, float(featureSecondMoment[i]));
    return sqrt(max(m, FLT_MIN));
}

/// Structural numerical floor (continual ridge / machine epsilon): replaces ad-hoc `1e-6`/`1e-8` in softmax-style ops.
/// `p_eff`/`N`/`C` from rank, online step index, and capacity volume (same `cap` family as `capacity_ridge`).
inline float autostep_eps_cap(uint rank, uint batchSize, device const KernelScalars *scalars, device const half *featureSecondMoment, float sqrtFltEpsilon) {
    float sigma = feature_sigma_max(featureSecondMoment, rank);
    float capVol = float(max(rank * max(batchSize, 1u), 1u));
    float N = float(max(scalars->stepCount, 1u));
    float peff = max(scalars->pEff, 1.0f);
    float ratio = peff / (N + capVol);
    return sqrtFltEpsilon * sigma * sqrt(max(ratio, FLT_MIN));
}

inline float plan_merge_lambda(uint rank, uint actDim) {
    return 1.0f / sqrt(float(max(rank + actDim, 1u)));
}

inline device float *arena_ptr(device float *arena, uint base) {
    return arena + base;
}

inline uint gvf_bank_size(uint rank) { return rank * rank; }

inline float gvf_weight(device const half *packed, uint g, uint i, uint rank) {
    return float(packed[g * rank + i]);
}

inline float forward_ws(device const half *packed, uint rank, uint actDim, uint outRow, uint inIdx) {
    uint base = gvf_bank_size(rank);
    return float(packed[base + outRow * (rank + actDim) + inIdx]);
}

inline float forward_wa(device const half *packed, uint rank, uint actDim, uint outRow, uint aIdx) {
    uint base = gvf_bank_size(rank) + outRow * (rank + actDim) + rank;
    return float(packed[base + aIdx]);
}

/// Option dynamics block `opt`: `base = primitivePacked + opt * optionForwardStride` (`primitivePacked` = GVF size).
inline float opt_forward_ws(device const half *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint inIdx) {
    return float(packed[blockBase + outRow * (rank + actDim) + inIdx]);
}
inline float opt_forward_wa(device const half *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint aIdx) {
    return float(packed[blockBase + outRow * (rank + actDim) + rank + aIdx]);
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

inline float dyn_forward_ws(device const half *packed, uint forwardBase, uint rank, uint actDim, uint outRow, uint inIdx) {
    return float(packed[forwardBase + outRow * (rank + actDim) + inIdx]);
}
inline float dyn_forward_wa(device const half *packed, uint forwardBase, uint rank, uint actDim, uint outRow, uint aIdx) {
    return float(packed[forwardBase + outRow * (rank + actDim) + rank + aIdx]);
}

/// One-step lookahead: primary value + **subtask** value head (`optionValues` row) on predicted features (planning→representation loop).
inline float lookahead_value_thread(device const half *stateWeights, uint forwardBase, device const half *valueWeights, device const half *optionSubValues,
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
        float pri = float(valueWeights[idx]) * predNext;
        float sec = float(optionSubValues[optIdx * rank + idx]) * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

inline float discrete_lookahead1(device const half *stateWeights, uint forwardBase, device const half *valueWeights,
    device const half *optionSubValues, uint optIdx,
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
        float pri = float(valueWeights[idx]) * predNext;
        float sec = float(optionSubValues[optIdx * rank + idx]) * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

inline float kstep_plan_discrete(device const half *stateWeights, uint forwardBase, device const half *valueWeights,
    device const half *optionSubValues, uint optIdx,
    device const KernelScalars *scalars, device const float *lat, uint sOff, uint actDim, uint rank,
    device const uint *activeList, uint activeCount, uint planK, thread uint& raw, thread uint& san,
    device float *scratch, uint maxRank) {
    for (uint i = 0u; i < rank && i < maxRank; ++i) scratch[i] = sanitize(lat[sOff + i], raw, san);
    float rho = scalars->rewardRate;
    float vk = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint i = activeList[li];
        float x = scratch[i];
        vk += float(valueWeights[i]) * x + float(optionSubValues[optIdx * rank + i]) * x;
    }
    float acc = 0.0f;
    for (uint step = 0u; step < planK; ++step) {
        uint bestA = 0u;
        float bestV = -FLT_MAX;
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
                vCand += float(valueWeights[idx]) * predNext + float(optionSubValues[optIdx * rank + idx]) * predNext;
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
            vNext += float(valueWeights[i]) * x + float(optionSubValues[optIdx * rank + i]) * x;
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
    float gammaPhi0, gammaPhi1, gammaPhi2;
    float structuralCap;
    float nlmsStepScale;
    float sqrtFltEpsilon;
    float rsqrtPlanLambda;
    float boundLat, boundRew, boundAct;
    float invBoundLat, invBoundRew, invBoundAct;
    uint activeCount;
    uint nGvfHeads;
    device KernelScalars *scalars;
    device half *encoderWeights;
    device half *policyWeights;
    device half *valueWeights;
    device half *rewardWeights;
    device half *stateWeights;
    device half *optionPolicyWeights;
    device half *optionValues;
    device half *optionCriticWeights;
    device half *featureSecondMoment;
    device half *featureResidual;
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
    device float *gvfStackW;
    device float *gvfStackH;
    device float *gvfStackU;
    device float *gvfHordeRho;
    device float *subtaskRhoArena;
    device float *optionBetaArena;
    device float *planDeltaArena;
    device float *keyboardArena;
    device float *keyboardPredArena;
    device float *utilityArena;
    device float *policyTraceArena;
    device float *gvfGenAuxArena;
    device float *mgscEtaArena;
    device float *arena;
    device float *ds;
    uint floatArenaAutostepBase;
};

inline OakPhaseContext oak_make_phase_context(
    device half *params,
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
    c.gammaPhi0 = config.gammaPhi0;
    c.gammaPhi1 = config.gammaPhi1;
    c.gammaPhi2 = config.gammaPhi2;
    c.scalars = scalars;
    c.ridge = config.capacityRidge;
    c.structuralCap = config.structuralCap;
    c.nlmsStepScale = config.nlmsStepScale;
    c.sqrtFltEpsilon = config.sqrtFltEpsilon;
    c.rsqrtPlanLambda = config.rsqrtPlanLambda;
    c.boundLat = config.boundLat;
    c.boundRew = config.boundRew;
    c.boundAct = config.boundAct;
    c.invBoundLat = config.invBoundLat;
    c.invBoundRew = config.invBoundRew;
    c.invBoundAct = config.invBoundAct;
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
    uint nGvf = max(config.nGvfHeads, 1u);
    uint gvfStride = nGvf * rank;
    device float *gvfBase = arena_ptr(floatArena, config.floatArenaGvfWeightsBase);
    c.nGvfHeads = nGvf;
    c.gvfStackW = gvfBase;
    c.gvfStackH = gvfBase + gvfStride;
    c.gvfStackU = gvfBase + 2u * gvfStride;
    c.gvfHordeRho = arena_ptr(floatArena, config.floatArenaGvfRhoBase);
    c.subtaskRhoArena = arena_ptr(floatArena, config.floatArenaSubtaskRhoBase);
    c.optionBetaArena = arena_ptr(floatArena, config.floatArenaOptionBetaBase);
    c.planDeltaArena = arena_ptr(floatArena, config.floatArenaPlanDeltaBase);
    c.keyboardArena = arena_ptr(floatArena, config.floatArenaKeyboardBase);
    c.keyboardPredArena = arena_ptr(floatArena, config.floatArenaKeyboardPredBase);
    c.utilityArena = arena_ptr(floatArena, config.floatArenaUtilityBase);
    c.policyTraceArena = arena_ptr(floatArena, config.floatArenaPolicyTraceBase);
    c.gvfGenAuxArena = arena_ptr(floatArena, config.floatArenaGvfGenAuxBase);
    c.mgscEtaArena = arena_ptr(floatArena, config.floatArenaMgscEtaBase);
    c.floatArenaAutostepBase = config.floatArenaAutostepBase;
    return c;
}

inline void oak_phase_encode_impl(OakPhaseContext ctx, uint gid) {
    uint row = gid / ctx.rank;
    uint feat = gid % ctx.rank;
    uint raw = 0u, san = 0u;
    if (row < ctx.batchSize) {
        float cap = ctx.structuralCap;
        float centered = sanitize(ctx.observation[row * ctx.obsDim + feat] - ctx.observationMean[feat], raw, san);
        float z = sanitize(centered * float(ctx.encoderWeights[feat]), raw, san);
        ctx.state[row * ctx.rank + feat] = clamp(z, -cap, cap);
    }
}

inline void oak_phase_prediction_impl(OakPhaseContext ctx, uint gid) {
    if (gid >= ctx.batchSize) return;
    uint raw = 0u, san = 0u;
    float cap = ctx.structuralCap;
    uint sOff = gid * ctx.rank;
    device float *dsp = ctx.ds + oak_ds_pred_row(gid, ctx.rank);
    device const uint *activeList = ctx.activeList;
    uint activeCount = ctx.activeCount;
    for (uint i = 0u; i < activeCount; ++i) {
        uint idx = activeList[i];
        dsp[i] = sanitize(ctx.previousState[sOff + idx], raw, san);
        dsp[ctx.rank + i] = sanitize(ctx.state[sOff + idx], raw, san);
    }
    float rPred = dot_sparse_weights_preval(ctx.rewardWeights, 0, dsp, activeList, activeCount);
    float vPrev = dot_sparse_weights_preval(ctx.valueWeights, 0, dsp, activeList, activeCount);
    float vNextBoot = dot_sparse_weights_preval(ctx.valueWeights, 0, dsp + ctx.rank, activeList, activeCount);
    ctx.rewardPrediction[gid] = clamp(sanitize(rPred, raw, san), -cap, cap);
    vPrev = clamp(sanitize(vPrev, raw, san), -cap, cap);
    vNextBoot = clamp(sanitize(vNextBoot, raw, san), -cap, cap);
    float terminalMask = ctx.done[gid] ? 0.0f : 1.0f;
    float vNow = terminalMask * vNextBoot;
    // Differential TD / reward centering (Naik et al.): use r̄ from **previous** step; bookkeeping updates r̄ once per step after this phase.
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
    float vSubPrev = dot_active_param(ctx.optionValues, po * ctx.rank, ctx.previousState, sOff, activeList, activeCount, raw, san);
    float vSubNext = dot_active_param(ctx.optionValues, po * ctx.rank, ctx.state, sOff, activeList, activeCount, raw, san);
    float vSubNextMasked = terminalMask * vSubNext;
    float cNext = (po < ctx.rank) ? sanitize(ctx.state[sOff + po], raw, san) : 0.0f;
    float rhoK = ctx.subtaskRhoArena[po];
    float deltaOpt = cNext - rhoK + ctx.gammaSub * vSubNextMasked - vSubPrev;
    ctx.tdOptionArena[gid] = clamp(sanitize(deltaOpt / denom, raw, san), -cap, cap);
    float rhoH = ctx.gvfHordeRho[0];
    uint n = ctx.nGvfHeads;
    for (uint j = 0u; j < n; ++j) {
        float predj = 0.0f;
        float cj = 0.0f;
        float vjpAcc = 0.0f;
        int ac = int(activeCount);
        for (int ii = 0; ii < ac; ++ii) {
            uint idx = activeList[ii];
            float ps = ctx.previousState[sOff + idx];
            float ss = ctx.state[sOff + idx];
            uint jOff = j * ctx.rank + idx;
            predj = fma(ctx.gvfStackW[jOff], sanitize(ps, raw, san), predj);
            cj = fma(ctx.gvfStackU[jOff], sanitize(ss, raw, san), cj);
            vjpAcc = fma(ctx.gvfStackW[jOff], sanitize(ss, raw, san), vjpAcc);
        }
        predj = sanitize(predj, raw, san);
        cj = sanitize(cj, raw, san);
        vjpAcc = sanitize(vjpAcc, raw, san);
        if (j == 0u && n > 1u) {
            cj += ctx.gammaPhi0 * masked_dot_arena(ctx.gvfStackU, ctx.rank, ctx.state, sOff, activeList, activeCount, raw, san);
            cj += fma(ctx.gammaPhi1, 0.0f, ctx.gammaPhi2 * ctx.ridge);
        }
        float vjp = vjpAcc * terminalMask;
        float dj = cj - rhoH + ctx.gammaPhi0 * vjp - predj;
        ctx.gvfTdArena[gid * n + j] = clamp(sanitize(dj, raw, san), -cap, cap);
    }
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
        float wS = opt_forward_ws(ctx.stateWeights, blockBase, ctx.rank, ctx.actDim, feat, idx);
        float vS = sanitize(ctx.previousState[sOff + idx], raw, san);
        ps = fma(wS, vS, ps);
    }
    float pa = 0.0f;
    for (uint a = 0u; a < ctx.actDim; ++a) {
        float wA = opt_forward_wa(ctx.stateWeights, blockBase, ctx.rank, ctx.actDim, feat, a);
        float vA = sanitize(ctx.previousAction[aOff + a], raw, san);
        pa = fma(wA, vA, pa);
    }
    ctx.rolloutState[row * ctx.rank + feat] = sanitize(ps + pa, raw, san);
}

inline void oak_phase_learn_impl(OakPhaseContext ctx, uint gid) {
    const uint policyBase = 3u * ctx.rank;
    const uint gvfHordeLearnBase = 4u * ctx.rank;
    const uint n = ctx.nGvfHeads;
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
        float emaAbsG = asBase[feat];
        emaAbsG += (fabs(gFeat) - emaAbsG) * invT;
        asBase[feat] = emaAbsG;
        float cap = ctx.structuralCap;
        float kappa = 2.0f * sqrt(cap);
        float stepV = min(1.0f / (emaAbsG + ridge), kappa / (fabs(gFeat) + ridge));
        ctx.valueWeights[feat] += stepV * gFeat;
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
            float step = ctx.nlmsStepScale / (ctx.featureSecondMoment[i] + ridge);
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
            float epsDyn = autostep_eps_cap(ctx.rank, ctx.batchSize, ctx.scalars, ctx.featureSecondMoment, ctx.sqrtFltEpsilon);
            if (wMask < epsDyn) continue;
            uint blockBase = ctx.primitivePacked + opt * ctx.optionForwardStride;
            for (uint i = 0u; i < ctx.rank; ++i) {
                float step = ctx.nlmsStepScale / (ctx.featureSecondMoment[i] + ridge);
                ctx.stateWeights[blockBase + outRow * (ctx.rank + ctx.actDim) + i] += gradAcc[i] * invB * step * wMask;
            }
            for (uint a = 0u; a < ctx.actDim; ++a) {
                float stepA = ctx.nlmsStepScale / (ctx.featureSecondMoment[outRow] + ridge);
                ctx.stateWeights[blockBase + outRow * (ctx.rank + ctx.actDim) + ctx.rank + a] += gradAccA[a] * invB * stepA * wMask;
            }
        }
    }

    if (gid >= policyBase && gid < policyBase + ctx.rank) {
        uint feat = gid - policyBase;
        float muF = 1.0f / (ctx.featureSecondMoment[feat] + ridge);
        float stepPol = ctx.nlmsStepScale * muF;
        float lamTr = 1.0f - ctx.nlmsStepScale;
        for (uint actIdx = 0u; actIdx < ctx.actDim; ++actIdx) {
            float gradP = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                uint sOff = row * ctx.rank;
                uint aOff = row * ctx.actDim;
                gradP += ctx.td[row] * sanitize(ctx.previousState[sOff + feat], raw, san) * sanitize(ctx.previousAction[aOff + actIdx], raw, san);
            }
            float gInst = gradP / float(ctx.batchSize);
            uint trOff = actIdx * ctx.rank + feat;
            float trPrev = ctx.policyTraceArena[trOff];
            float trNew = lamTr * trPrev + gInst;
            ctx.policyTraceArena[trOff] = trNew;
            float dPol = stepPol * trNew;
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
    if (gid >= gvfHordeLearnBase && gid < gvfHordeLearnBase + ctx.rank * n) {
        uint lin = gid - gvfHordeLearnBase;
        uint feat = lin % ctx.rank;
        uint j = lin / ctx.rank;
        float stepH = ctx.nlmsStepScale / (ctx.featureSecondMoment[feat] + ridge);
        float bGen = ctx.nlmsStepScale;
        if (j == 1u && n > 1u) {
            float gradU1 = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                uint sOff = row * ctx.rank;
                gradU1 += ctx.gvfTdArena[row * n + 0u] * sanitize(ctx.state[sOff + feat], raw, san);
            }
            ctx.gvfStackU[ctx.rank + feat] += stepH * (gradU1 / float(ctx.batchSize));
        } else if (j == 0u) {
            float gradH = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                uint sOff = row * ctx.rank;
                gradH += ctx.gvfTdArena[row * n + 0u] * sanitize(ctx.previousState[sOff + feat], raw, san);
            }
            float gAvg = gradH / float(ctx.batchSize);
            float rAux = ctx.gvfGenAuxArena[feat];
            ctx.gvfStackW[feat] += stepH * (gAvg - rAux);
            rAux += bGen * (gAvg - rAux);
            ctx.gvfGenAuxArena[feat] = rAux;
            ctx.gvfStackH[feat] += stepH * gAvg;
        } else {
            float gradH = 0.0f;
            float gradU = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                uint sOff = row * ctx.rank;
                float ph = sanitize(ctx.previousState[sOff + feat], raw, san);
                float tdv = ctx.gvfTdArena[row * n + j];
                gradH += tdv * ph;
                gradU += tdv * sanitize(ctx.state[sOff + feat], raw, san);
            }
            float gAvg = gradH / float(ctx.batchSize);
            uint auxOff = feat + j * ctx.rank;
            float rAux = ctx.gvfGenAuxArena[auxOff];
            ctx.gvfStackW[feat + j * ctx.rank] += stepH * (gAvg - rAux);
            rAux += bGen * (gAvg - rAux);
            ctx.gvfGenAuxArena[auxOff] = rAux;
            ctx.gvfStackH[feat + j * ctx.rank] += stepH * gAvg;
            ctx.gvfStackU[feat + j * ctx.rank] += stepH * (gradU / float(ctx.batchSize));
        }
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
            float epsK = autostep_eps_cap(ctx.rank, ctx.batchSize, ctx.scalars, ctx.featureSecondMoment, ctx.sqrtFltEpsilon);
            if (phase == 1u && elapsed < ctx.rank) {
                activeOption = poPrev;
                for (uint i = 0u; i < ctx.activeCount; ++i) {
                    uint idx = ctx.activeList[i];
                    pickScore += ctx.optionValues[activeOption * ctx.rank + idx] * sanitize(ctx.state[sOff + idx], raw, san);
                }
            } else {
                // Option keyboard — Barreto-style **unnormalized** weights `w_k`; mixture of option-policy logits (no argmax routing).
                float pickAgg = 0.0f;
                for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
                    float psi = 0.0f;
                    uint off = opt * ctx.rank;
                    for (uint i = 0u; i < ctx.activeCount; ++i) {
                        uint idx = ctx.activeList[i];
                        float sv = sanitize(ctx.state[sOff + idx], raw, san);
                        psi += ctx.optionValues[off + idx] * sv;
                    }
                    if (opt < ctx.rank) {
                        psi += sanitize(ctx.state[sOff + opt], raw, san) * (1.0f / float(max(ctx.rank, 1u)));
                    }
                    float wk = max(ctx.keyboardArena[row * ctx.optionCount + opt], epsK);
                    pickAgg += wk * psi;
                }
                pickScore = pickAgg;
                activeOption = poPrev;
                if (actIdx == 0u) {
                    ctx.stompState[row * 4u] = 1u;
                    ctx.stompState[row * 4u + 1u] = 0u;
                    uint ocR = ctx.optionCount * ctx.rank;
                    ctx.optionCriticWeights[ocR + activeOption] += (pickScore - ctx.optionCriticWeights[ocR + activeOption]) * ctx.invT;
                    const uint kbd0 = 2u * ctx.batchSize;
                    float epsMeta = max(epsK, ctx.ridge);
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
                    float invSS = 1.0f / (ssum + epsMeta);
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
            float baseLogit = masked_dot_param(ctx.policyWeights, actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            float lamMix = ctx.rsqrtPlanLambda;
            float optMix = 0.0f;
            if (phase == 1u && elapsed < ctx.rank) {
                uint k = poPrev;
                optMix = masked_dot_param(ctx.optionPolicyWeights, k * ctx.rank * ctx.actDim + actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            } else {
                for (uint k = 0u; k < ctx.optionCount; ++k) {
                    float wk = max(ctx.keyboardArena[row * ctx.optionCount + k], epsK);
                    float lk = masked_dot_param(ctx.optionPolicyWeights, k * ctx.rank * ctx.actDim + actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
                    optMix += lamMix * wk * lk;
                }
            }
            uint aOff = row * ctx.actDim;
            float lamPlan = ctx.rsqrtPlanLambda;
            float planBump = ctx.planDeltaArena[aOff + actIdx];
            float logit = sanitize(baseLogit + optMix + lamPlan * planBump, raw, san);
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
            float vNow = masked_dot_param(ctx.valueWeights, 0, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            float vSubNow = dot_active_param(ctx.optionValues, po * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount, raw, san);
            vNow += vSubNow;

            if (fc_action_is_discrete) {
                uint bestA = 0u;
                float bestV = -FLT_MAX;
                float lamEta = ctx.rsqrtPlanLambda;
                for (uint cand = 0u; cand < ctx.actDim; ++cand) {
                    float vCand = discrete_lookahead1(ctx.stateWeights, forwardPlanBase, ctx.valueWeights, ctx.optionValues, po,
                        ctx.state, sOff, ctx.actDim, ctx.rank, ctx.activeList, ctx.activeCount, cand, raw, san);
                    vCand += ctx.mgscEtaArena[cand] * lamEta;
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
inline void oak_phase_bookkeeping_impl(OakPhaseContext ctx, device KernelStats *statsOut, uint gid) {
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
        // MGSC-style search control (Burega et al.): η with capacity-derived cadence + Autostep-like momentum/variance (no fixed Adam β).
        if (ctx.actionIsDiscrete != 0u) {
            uint Kmeta = max(1u, ctx.rank / 64u);
            float doMeta = ((ctx.scalars->stepCount % Kmeta) == 0u) ? 1.0f : 0.0f;
            float metaStep = ctx.nlmsStepScale;
            float invB = 1.0f / float(max(ctx.batchSize, 1u));
            float tMeta = float(max(ctx.scalars->stepCount, 1u));
            float beta1 = 1.0f / tMeta;
            float beta2 = ctx.nlmsStepScale;
            device float *mEta = ctx.mgscEtaArena + ctx.actDim;
            device float *vEta = mEta + ctx.actDim;
            for (uint cand = 0u; cand < ctx.actDim; ++cand) {
                float gAcc = 0.0f;
                for (uint row = 0u; row < ctx.batchSize; ++row) {
                    uint aOff = row * ctx.actDim;
                    uint bestA = 0u;
                    float bestD = ctx.planDeltaArena[aOff];
                    for (uint a = 1u; a < ctx.actDim; ++a) {
                        float v = ctx.planDeltaArena[aOff + a];
                        if (v > bestD) {
                            bestD = v;
                            bestA = a;
                        }
                    }
                    float mx = ctx.mgscEtaArena[0];
                    for (uint a = 1u; a < ctx.actDim; ++a) mx = max(mx, ctx.mgscEtaArena[a]);
                    float sumE = 0.0f;
                    for (uint a = 0u; a < ctx.actDim; ++a) sumE += exp(ctx.mgscEtaArena[a] - mx);
                    float p = exp(ctx.mgscEtaArena[cand] - mx) / max(sumE, FLT_MIN);
                    float y = (bestA == cand) ? 1.0f : 0.0f;
                    gAcc += (y - p);
                }
                float g = gAcc * invB;
                mEta[cand] = (1.0f - beta1) * mEta[cand] + g;
                vEta[cand] = (1.0f - beta2) * vEta[cand] + g * g;
                ctx.mgscEtaArena[cand] += doMeta * metaStep * mEta[cand] / sqrt(vEta[cand] + ctx.ridge);
            }
        }
        // CBP-style plasticity: if lowest active utility is below mean − std (structural band), reinit that latent’s encoder with deterministic jitter (no GPU RNG).
        if (ac >= 1u) {
            float mean_u = totalUtility / float(max(ctx.rank, 1u));
            float var_u = 0.0f;
            for (uint i = 0u; i < ctx.rank; ++i) {
                float d = ctx.featureResidual[i] - mean_u;
                var_u += d * d;
            }
            float std_u = sqrt(var_u / float(max(ctx.rank, 1u)));
            uint worstIdx = ctx.activeList[0];
            float worstRes = ctx.featureResidual[worstIdx];
            for (uint j = 1u; j < ac; ++j) {
                uint ix = ctx.activeList[j];
                float r = ctx.featureResidual[ix];
                if (r < worstRes) {
                    worstRes = r;
                    worstIdx = ix;
                }
            }
            if (worstRes < mean_u - std_u) {
                if (worstIdx < ctx.baseObsDim) {
                    float rjv = ctx.ridge;
                    // `2246822519u` = odd constant in splitmix32-style mixing (not an ad-hoc learning rate).
                    uint h0 = ctx.scalars->stepCount ^ (worstIdx * 2246822519u);
                    uint i0 = 0u;
                    uint i1 = 0u;
                    float f0 = ctx.featureResidual[0];
                    float f1 = ctx.featureResidual[0];
                    if (ctx.rank > 1u) {
                        i1 = 1u;
                        f1 = ctx.featureResidual[1];
                        if (f1 > f0) {
                            i0 = 1u;
                            i1 = 0u;
                            float tmpf = f0;
                            f0 = f1;
                            f1 = tmpf;
                        }
                        for (uint j = 2u; j < ctx.rank; ++j) {
                            float fj = ctx.featureResidual[j];
                            if (fj > f0) {
                                f1 = f0;
                                i1 = i0;
                                f0 = fj;
                                i0 = j;
                            } else if (fj > f1) {
                                f1 = fj;
                                i1 = j;
                            }
                        }
                    }
                    float alpha = float(h0 & 0xffu) / 255.0f;
                    float z = alpha * ctx.featureResidual[i0] + (1.0f - alpha) * ctx.featureResidual[i1];
                    float ang = z * rjv * sqrt(float(max(ctx.rank, 1u)));
                    // Deep Fourier–style feature from mixed parents (2410.20634); deterministic phase, no GPU RNG.
                    ctx.encoderWeights[worstIdx] = sanitize(sin(ang) + cos(ang), rawB, sanB);
                }
                ctx.featureResidual[worstIdx] = mean_u;
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
                float epsKb = autostep_eps_cap(ctx.rank, ctx.batchSize, ctx.scalars, ctx.featureSecondMoment, ctx.sqrtFltEpsilon);
                float kbSumT = 0.0f;
                for (uint k = 0u; k < ctx.optionCount; ++k) kbSumT += max(ctx.keyboardArena[row * ctx.optionCount + k], epsKb);
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
        // Single average-reward estimate writer: prediction reads `rewardRate` from the previous step; this batch EMA updates it once per step (Naik-style centering compatible).
        ctx.scalars->rewardRate += (meanR - ctx.scalars->rewardRate) / t;
        uint Tpe = max(1000u, ctx.scalars->stepCount / max(ctx.rank / 8u, 1u));
        if (Tpe > 0u && (ctx.scalars->stepCount % Tpe) == 0u) {
            float tr = 0.0f;
            float mx = float(ctx.featureSecondMoment[0]);
            for (uint i = 0u; i < ctx.rank; ++i) {
                tr += float(ctx.featureSecondMoment[i]);
                mx = max(mx, float(ctx.featureSecondMoment[i]));
            }
            ctx.scalars->pEff = tr / max(mx, FLT_MIN);
        }
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
            cHsum += masked_dot_arena(ctx.gvfStackU, 0, ctx.state, sOff2, ctx.activeList, ac, rawB, sanB);
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
        // Bounds / invBounds precomputed in Swift (`KernelConfig`), same formulas as `stats_abs_bound_for_sum_squares`.
        const float boundLat = ctx.boundLat;
        const float boundRew = ctx.boundRew;
        const float boundAct = ctx.boundAct;
        const float invBoundLat = ctx.invBoundLat;
        const float invBoundRew = ctx.invBoundRew;
        const float invBoundAct = ctx.invBoundAct;
        // One pass over rows: fused ctx.rank stats + obs sums for online mean (ctx.rank/ctx.obsDim ≤ 384).
        device float *sumObsByF = ctx.ds;
        for (uint f = 0u; f < ctx.obsDim; ++f) { sumObsByF[f] = 0.0f; }

        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            uint aOff = row * ctx.actDim;
            for (uint k = 0u; k < ctx.rank; ++k) {
                float pred = ctx.latentError[row * ctx.rank + k];
                float tgt = ctx.state[row * ctx.rank + k];
                float diff = tgt - pred;
                uint badDiff = select(1u, 0u, isfinite(diff));
                stRaw += badDiff;
                stSan += badDiff;
                float ds = select(0.0f, diff, isfinite(diff));
                float dc = clamp(ds, -boundLat, boundLat);
                float nlc = dc * invBoundLat;
                lmse = fma(nlc, nlc, lmse);
                maxLat = max(maxLat, fabs(ds));
                float sv = ctx.state[sOff + k];
                uint badSv = select(1u, 0u, isfinite(sv));
                nLearn += badSv;
                stRaw += badSv;
                stSan += badSv;
                float sc = select(0.0f, sv, isfinite(sv));
                float vc = clamp(sc, -boundLat, boundLat);
                float nvc = vc * invBoundLat;
                vtrace = fma(nvc, nvc, vtrace);
            }
            float rdiff = ctx.reward[row] - ctx.rewardPrediction[row];
            uint badRew = select(1u, 0u, isfinite(rdiff));
            stRaw += badRew;
            stSan += badRew;
            float rds = select(0.0f, rdiff, isfinite(rdiff));
            float rc = clamp(rds, -boundRew, boundRew);
            float nrc = rc * invBoundRew;
            rmse = fma(nrc, nrc, rmse);
            maxRP = max(maxRP, fabs(rds));
            float pa = 0.0f;
            float pb = 0.0f;
            for (uint a = 0u; a < ctx.actDim; ++a) {
                float x = ctx.action[aOff + a];
                float yg = ctx.actionGrad[aOff + a];
                float psRaw = x - yg;
                uint badPs = select(1u, 0u, isfinite(psRaw));
                stRaw += badPs;
                stSan += badPs;
                float ps = select(0.0f, psRaw, isfinite(psRaw));
                float pc = clamp(ps, -boundAct, boundAct);
                float npc = pc * invBoundAct;
                pa = fma(npc, npc, pa);
                maxPol = max(maxPol, fabs(x));
                nPlan += select(1u, 0u, isfinite(x));
                nPlan += select(1u, 0u, isfinite(yg));
                float py = ctx.previousAction[aOff + a];
                float pbRaw = x - py;
                uint badPb = select(1u, 0u, isfinite(pbRaw));
                stRaw += badPb;
                stSan += badPb;
                float pbs = select(0.0f, pbRaw, isfinite(pbRaw));
                float pbc = clamp(pbs, -boundAct, boundAct);
                float npb = pbc * invBoundAct;
                pb = fma(npb, npb, pb);
            }
            planL2 += pa;
            actDelta += pb;
            for (uint f = 0u; f < ctx.obsDim; ++f) {
                float o = ctx.observation[row * ctx.obsDim + f];
                uint badO = select(1u, 0u, isfinite(o));
                nEnc += badO;
                sumObsByF[f] += select(0.0f, o, isfinite(o));
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
        const uint kbd0bk = 2u * ctx.batchSize;
        const uint ofb0 = kbd0bk + ctx.batchSize * ctx.optionCount;
        // Match latentPredictionMSE: sanitize, clamp to stats_abs_bound, divide by bound² so telemetry stays O(1) and finite.
        float optPredAcc = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            uint sOff = row * ctx.rank;
            for (uint k = 0u; k < ctx.rank; ++k) {
                float rdx = ctx.state[sOff + k] - ctx.rolloutState[row * ctx.rank + k];
                uint badRd = select(1u, 0u, isfinite(rdx));
                stRaw += badRd;
                stSan += badRd;
                float ds = select(0.0f, rdx, isfinite(rdx));
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
                    float rdx = ctx.state[sOff + k] - ctx.rolloutState[row * ctx.rank + k];
                    uint badRd = select(1u, 0u, isfinite(rdx));
                    stRaw += badRd;
                    stSan += badRd;
                    float ds = select(0.0f, rdx, isfinite(rdx));
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
        float meanFeat = totalUtility / float(max(ctx.rank, 1u));
        float varFeat = 0.0f;
        for (uint i = 0u; i < ctx.rank; ++i) {
            float d = ctx.featureResidual[i] - meanFeat;
            varFeat += d * d;
        }
        float stdFeat = sqrt(varFeat / float(max(ctx.rank, 1u)));
        float reinits = 0.0f;
        float invKey = 1.0f / float(max(ctx.optionCount, 1u));
        for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
            float oe = ctx.latentPower[ofb0 + opt];
            float ou = ctx.latentPower[ofb0 + ctx.optionCount + opt];
            float opPick = ctx.latentPower[ofb0 + 2u * ctx.optionCount + opt];
            float errHigh = meanFeat + stdFeat;
            float massFloor = 1.0f / float(20u * max(ctx.rank, 1u));
            if (oe > errHigh && ou < massFloor && opPick < massFloor) {
                uint bb = ctx.primitivePacked + opt * ctx.optionForwardStride;
                for (uint i = 0u; i < ctx.optionForwardStride; ++i) {
                    ctx.stateWeights[bb + i] = 0.0f;
                }
                uint polBase = opt * ctx.rank * ctx.actDim;
                for (uint i = 0u; i < ctx.rank * ctx.actDim; ++i) {
                    ctx.optionPolicyWeights[polBase + i] = 0.0f;
                }
                for (uint i = 0u; i < ctx.rank; ++i) {
                    ctx.optionValues[opt * ctx.rank + i] = 0.0f;
                }
                if (opt < ctx.rank) {
                    ctx.optionValues[opt * ctx.rank + opt] = 1.0f;
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
    device half *params [[buffer(0)]], \
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
    oak_phase_bookkeeping_impl(ctx, statsOut, gid);
}
#undef OAK_KERNEL_SIG
