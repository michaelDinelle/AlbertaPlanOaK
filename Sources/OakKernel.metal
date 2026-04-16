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

// Oak / Alberta Plan (12 steps): one continual `oak_step` loop — same phases every tick (no train/eval).
// Shared latent `state` is read-only within a phase; consumers (encoder, GVFs, critic, forward/option
// dynamics, options, planning) all use the same representation in one step (AGENTS: one shared state).

// Phase ids (dispatch order is fixed by data dependencies, not hyperparameters).
static constant uint kPhaseEncode = 0u;
static constant uint kPhasePrediction = 1u;
static constant uint kPhaseLearn = 2u;
static constant uint kPhaseMetaPolicy = 3u;
static constant uint kPhasePlanning = 4u;
static constant uint kPhaseForwardModel = 6u;
/// Structural continual maintenance: STOMP / active-mask pruning, online average-reward estimate ρ, observation
/// running mean, plus **telemetry** into `KernelStats` (does not write gradients; same buffers as the rest of `oak_step`).
static constant uint kPhaseBookkeeping = 7u;

struct KernelConfig {
    uint phase;
    uint obsDim;
    uint rank;
    uint actDim;
    uint batchSize;
    uint actionIsDiscrete;
    uint optionCount;
    /// Float count of the primitive packed block: GVF bank + one forward transition (must match Swift `packedDynamicsParameterCount`).
    uint primitivePackedSize;
    /// Floats per option dynamics slice (= `rank * (rank + actDim)`), appended after the primitive block.
    uint optionForwardStride;
    /// Structural planning horizon: `min(4, max(rank,1))` (no free integer; set in host from rank only).
    uint planK;
    /// Raw env observation width before optional aux concat (encoder NLMS only for `feat < baseObsDim`).
    uint baseObsDim;
};
struct KernelScalars { uint stepCount, rewardCount, activeCount; float rewardRate; };
struct KernelStats {
    float rewardRate, latentPredictionMSE, rewardPredictionMSE, latentStateMeanSquare, activeRank, actionDeltaL2Mean, planningActionChangeL2Mean;
    float rawNonFiniteCount, sanitizedNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
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

/// Online-average weight 1/(t+1) uses step index after this transition (stepCount is steps so far).
inline float online_inv_time(device const KernelScalars *scalars) {
    return 1.0f / float(max(scalars->stepCount + 1u, 1u));
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

/// Option-only forward block (no GVF): base = primitivePacked + opt * optionForwardStride.
inline float opt_forward_ws(device const float *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint inIdx) {
    return packed[blockBase + outRow * (rank + actDim) + inIdx];
}
inline float opt_forward_wa(device const float *packed, uint blockBase, uint rank, uint actDim, uint outRow, uint aIdx) {
    return packed[blockBase + outRow * (rank + actDim) + rank + aIdx];
}

inline bool use_option_rollout(device const uint *stompState, uint row, uint rank) {
    uint phase = stompState[row * 2u];
    uint elapsed = stompState[row * 2u + 1u];
    return phase == 1u && elapsed < rank;
}

inline uint pick_option_slice(device const uint *stompState, device const float *latentPower, uint row, uint rank, float prevOptF, uint optionCount) {
    if (use_option_rollout(stompState, row, rank)) {
        uint po = uint(floor(prevOptF + 0.5f));
        if (po >= optionCount) po = optionCount - 1u;
        return po;
    }
    return 0u;
}

/// One-step lookahead V(f̂(s,a')) using primitive forward weights (continuous probes); secondary critic on predicted features.
inline float lookahead_value_thread(device const float *stateWeights, device const float *valueWeights, device const float *optionCriticWeights,
    uint criticOpt, device const float *lat, uint sOff, uint actDim, uint rank, device const uint *activeList, uint activeCount, thread uint& raw, thread uint& san,
    thread const float *actionVec) {
    float vCand = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint idx = activeList[li];
        float ps = 0.0f;
        for (uint lj = 0u; lj < activeCount; ++lj) {
            uint jdx = activeList[lj];
            ps += forward_ws(stateWeights, rank, actDim, idx, jdx) * sanitize(lat[sOff + jdx], raw, san);
        }
        float pa = 0.0f;
        for (uint a = 0u; a < actDim; ++a) {
            pa += forward_wa(stateWeights, rank, actDim, idx, a) * sanitize(actionVec[a], raw, san);
        }
        float predNext = sanitize(ps + pa, raw, san);
        float pri = valueWeights[idx] * predNext;
        float sec = optionCriticWeights[criticOpt * rank + idx] * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

/// Discrete: one-step value after primitive transition, primary + optional secondary critic head.
inline float discrete_lookahead1(device const float *stateWeights, device const float *valueWeights,
    device const float *optionCriticWeights, uint criticOpt,
    device const float *lat, uint sOff, uint actDim, uint rank, device const uint *activeList, uint activeCount,
    uint cand, thread uint& raw, thread uint& san) {
    float vCand = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint idx = activeList[li];
        float ps = 0.0f;
        for (uint lj = 0u; lj < activeCount; ++lj) {
            uint jdx = activeList[lj];
            ps += forward_ws(stateWeights, rank, actDim, idx, jdx) * sanitize(lat[sOff + jdx], raw, san);
        }
        float pa = forward_wa(stateWeights, rank, actDim, idx, cand);
        float predNext = sanitize(ps + pa, raw, san);
        float pri = valueWeights[idx] * predNext;
        float sec = optionCriticWeights[criticOpt * rank + idx] * predNext;
        vCand += pri + sec;
    }
    return vCand;
}

/// Roll greedy discrete lookaheads on primitive dynamics; return accumulated sum_k [(V_{k+1}-V_k) - rho].
inline float kstep_plan_discrete(device const float *stateWeights, device const float *valueWeights,
    device const float *optionCriticWeights, uint criticOpt,
    device const KernelScalars *scalars, device const float *lat, uint sOff, uint actDim, uint rank,
    device const uint *activeList, uint activeCount, uint planK, thread uint& raw, thread uint& san,
    thread float *scratch, uint maxRank) {
    for (uint i = 0u; i < rank && i < maxRank; ++i) scratch[i] = sanitize(lat[sOff + i], raw, san);
    float rho = scalars->rewardRate;
    float vk = 0.0f;
    for (uint li = 0u; li < activeCount; ++li) {
        uint i = activeList[li];
        float x = scratch[i];
        vk += valueWeights[i] * x + optionCriticWeights[criticOpt * rank + i] * x;
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
                    ps += forward_ws(stateWeights, rank, actDim, idx, jdx) * scratch[jdx];
                }
                float pa = forward_wa(stateWeights, rank, actDim, idx, cand);
                float predNext = sanitize(ps + pa, raw, san);
                vCand += valueWeights[idx] * predNext + optionCriticWeights[criticOpt * rank + idx] * predNext;
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
                ps += forward_ws(stateWeights, rank, actDim, feat, idx) * scratch[idx];
            }
            float pa = forward_wa(stateWeights, rank, actDim, feat, bestA);
            scratch[feat] = sanitize(ps + pa, raw, san);
        }
        float vNext = 0.0f;
        for (uint li = 0u; li < activeCount; ++li) {
            uint i = activeList[li];
            float x = scratch[i];
            vNext += valueWeights[i] * x + optionCriticWeights[criticOpt * rank + i] * x;
        }
        acc += (vNext - vk) - rho;
        vk = vNext;
    }
    return acc;
}

kernel void oak_step(
    device const float *observation [[buffer(0)]],
    device float *observationMean [[buffer(1)]],
    device float *encoderWeights [[buffer(2)]],
    device float *policyWeights [[buffer(3)]],
    device float *valueWeights [[buffer(4)]],
    device float *rewardWeights [[buffer(5)]],
    device float *stateWeights [[buffer(6)]],
    device float *optionPolicyWeights [[buffer(7)]],
    device float *optionValues [[buffer(8)]],
    device float *previousState [[buffer(9)]],
    device float *previousAction [[buffer(10)]],
    device const float *reward [[buffer(11)]],
    device uint *done [[buffer(12)]],
    device float *state [[buffer(13)]],
    device float *action [[buffer(14)]],
    device float *latentError [[buffer(15)]],
    device float *rewardPrediction [[buffer(16)]],
    device float *td [[buffer(17)]],
    device float *actionGrad [[buffer(18)]],
    device KernelScalars *scalars [[buffer(19)]],
    device KernelStats *statsOut [[buffer(20)]],
    device float *featureSecondMoment [[buffer(21)]],
    device float *featureResidual [[buffer(22)]],
    device uint *activeState [[buffer(23)]],
    device float *rolloutState [[buffer(24)]],
    device float *latentPower [[buffer(25)]],
    device const float *actionBounds [[buffer(26)]],
    device uint *counters [[buffer(27)]],
    device float *optionCriticWeights [[buffer(28)]],
    device uint *stompState [[buffer(29)]],
    constant KernelConfig &config [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint obsDim = config.obsDim;
    const uint rank = config.rank;
    const uint actDim = config.actDim;
    const uint batchSize = config.batchSize;
    const uint optionCount = max(config.optionCount, 1u);
    const uint primitivePacked = config.primitivePackedSize;
    const uint optionForwardStride = config.optionForwardStride;
    const uint planK = max(1u, config.planK);
    const uint baseObsDim = config.baseObsDim;
    const bool isDisc = config.actionIsDiscrete != 0u;
    device uint *activeMask = activeState;
    device uint *activeList = activeState + rank;
    const uint activeCount = scalars->activeCount;
    const float ridge = capacity_ridge(rank, batchSize);
    const float invT = online_inv_time(scalars);
    uint raw = 0, san = 0;

    if (config.phase == kPhaseEncode) {
        uint row = gid / rank;
        uint feat = gid % rank;
        if (row < batchSize) {
            float centered = sanitize(observation[row * obsDim + feat] - observationMean[feat], raw, san);
            state[row * rank + feat] = sanitize(centered * encoderWeights[feat], raw, san);
        }
        return;
    }

    if (config.phase == kPhasePrediction) {
        if (gid < batchSize) {
            uint sOff = gid * rank;
            // One sanitize/load per active slot; reuse for reward, value, TD norm, and GVF bank (same math as masked_dot).
            thread float scratchPrev[384];
            thread float scratchState[384];
            for (uint i = 0u; i < activeCount; ++i) {
                uint idx = activeList[i];
                scratchPrev[i] = sanitize(previousState[sOff + idx], raw, san);
                scratchState[i] = sanitize(state[sOff + idx], raw, san);
            }
            float rPred = 0.0f;
            float vPrev = 0.0f;
            float vNextBoot = 0.0f;
            for (uint i = 0u; i < activeCount; ++i) {
                uint idx = activeList[i];
                float sp = scratchPrev[i];
                float ss = scratchState[i];
                rPred += rewardWeights[idx] * sp;
                vPrev += valueWeights[idx] * sp;
                vNextBoot += valueWeights[idx] * ss;
            }
            rewardPrediction[gid] = sanitize(rPred, raw, san);
            vPrev = sanitize(vPrev, raw, san);
            vNextBoot = sanitize(vNextBoot, raw, san);
            // Average-reward TD: δ = r - ρ + V(s') - V(s). At env termination, only the *bootstrap target* is masked:
            // V(s') is replaced by 0 (AGENTS: no episodic weight/trace reset — target gating at termination only).
            float terminalMask = done[gid] ? 0.0f : 1.0f;
            float vNow = terminalMask * vNextBoot;
            float delta = reward[gid] - scalars->rewardRate + vNow - vPrev;
            float prevNorm2 = 0.0f;
            for (uint i = 0u; i < activeCount; ++i) {
                float x = scratchPrev[i];
                prevNorm2 += x * x;
            }
            float denom = 1.0f + prevNorm2 / float(max(activeCount, 1u));
            td[gid] = sanitize(delta / denom, raw, san);
            // Prediction I: rank GVFs — one linear head per latent dimension (cumulant = next state component).
            float rowErrSum = 0.0f;
            for (uint g = 0u; g < rank; ++g) {
                float acc = 0.0f;
                for (uint i = 0u; i < activeCount; ++i) {
                    uint idx = activeList[i];
                    acc += gvf_weight(stateWeights, g, idx, rank) * scratchPrev[i];
                }
                latentError[gid * rank + g] = sanitize(acc, raw, san);
                rowErrSum += state[sOff + g] - latentError[gid * rank + g];
            }
            // `latentPower[batchSize+row]`: encoder NLMS row sum (overwritten by metaPolicy after learn; avoids extra Metal buffer slot).
            latentPower[batchSize + gid] = rowErrSum;
        }
        return;
    }

    if (config.phase == kPhaseForwardModel) {
        uint row = gid / rank;
        uint feat = gid % rank;
        if (row < batchSize) {
            uint sOff = row * rank;
            uint aOff = row * actDim;
            float prevOptF = latentPower[row];
            uint dynOpt = pick_option_slice(stompState, latentPower, row, rank, prevOptF, optionCount);
            uint blockBase = primitivePacked + dynOpt * optionForwardStride;
            float ps = 0.0f;
            for (uint i = 0u; i < activeCount; ++i) {
                uint idx = activeList[i];
                ps += opt_forward_ws(stateWeights, blockBase, rank, actDim, feat, idx) * sanitize(previousState[sOff + idx], raw, san);
            }
            float pa = 0.0f;
            for (uint a = 0u; a < actDim; ++a) {
                pa += opt_forward_wa(stateWeights, blockBase, rank, actDim, feat, a) * sanitize(previousAction[aOff + a], raw, san);
            }
            rolloutState[row * rank + feat] = sanitize(ps + pa, raw, san);
        }
        return;
    }

    if (config.phase == kPhaseLearn) {
        // Fused option-dynamics NLMS: one thread per outRow loops over option slices (same FLOPs, fewer threads vs oc×rank separate threads).
        const uint policyBase = 4u * rank;
        const uint criticBase = 5u * rank;

        if (gid < rank) {
            uint feat = gid;
            float weightUpdate = 0.0f;
            float utility = 0.0f;
            float smAcc = 0.0f;
            float gvfUtil = 0.0f;
            for (uint row = 0u; row < batchSize; ++row) {
                float sVal = state[row * rank + feat];
                float tdVal = td[row];
                weightUpdate += tdVal * sVal;
                utility += fabs(tdVal * sVal);
                smAcc += sVal * sVal;
                for (uint g = 0u; g < rank; ++g) {
                    float pred = latentError[row * rank + g];
                    float tgt = state[row * rank + g];
                    float err = tgt - pred;
                    gvfUtil += fabs(err * sVal);
                }
            }
            float mBatch = smAcc / float(batchSize);
            float mu = 1.0f / (featureSecondMoment[feat] + ridge);
            // Control I: `td` is the advantage estimate; value head supplies the baseline for policy / option logits.
            valueWeights[feat] += (weightUpdate / float(batchSize)) * mu;
            float uBatch = utility / float(batchSize);
            float gvfBatch = gvfUtil / float(max(batchSize * rank, 1u));
            featureResidual[feat] += ((uBatch + gvfBatch) - featureResidual[feat]) * invT;
            featureSecondMoment[feat] += (mBatch - featureSecondMoment[feat]) * invT;
        }
        if (gid >= rank && gid < rank + rank) {
            uint g = gid - rank;
            uint gvfBase = g * rank;
            thread float gradAcc[384];
            for (uint i = 0u; i < rank; ++i) { gradAcc[i] = 0.0f; }
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                float err = sanitize(state[row * rank + g] - latentError[row * rank + g], raw, san);
                for (uint i = 0u; i < rank; ++i) {
                    gradAcc[i] += err * sanitize(previousState[sOff + i], raw, san);
                }
            }
            for (uint i = 0u; i < rank; ++i) {
                float step = 1.0f / (featureSecondMoment[i] + ridge);
                stateWeights[gvfBase + i] += (gradAcc[i] / float(batchSize)) * step;
            }
        }
        if (gid >= rank + rank && gid < rank + rank + rank) {
            uint outRow = gid - rank - rank;
            uint fwdBase = gvf_bank_size(rank) + outRow * (rank + actDim);
            // One err per (row, outRow); reuse across all i and a (same math, fewer loads / subs).
            thread float gradAcc[384];
            for (uint i = 0u; i < rank; ++i) { gradAcc[i] = 0.0f; }
            thread float gradAccA[384];
            for (uint a = 0u; a < actDim; ++a) { gradAccA[a] = 0.0f; }
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                uint aOff = row * actDim;
                float err = sanitize(state[row * rank + outRow] - rolloutState[row * rank + outRow], raw, san);
                for (uint i = 0u; i < rank; ++i) {
                    gradAcc[i] += err * sanitize(previousState[sOff + i], raw, san);
                }
                for (uint a = 0u; a < actDim; ++a) {
                    gradAccA[a] += err * sanitize(previousAction[aOff + a], raw, san);
                }
            }
            for (uint i = 0u; i < rank; ++i) {
                float step = 1.0f / (featureSecondMoment[i] + ridge);
                stateWeights[fwdBase + i] += (gradAcc[i] / float(batchSize)) * step;
            }
            for (uint a = 0u; a < actDim; ++a) {
                float stepA = 1.0f / (featureSecondMoment[outRow] + ridge);
                stateWeights[fwdBase + rank + a] += (gradAccA[a] / float(batchSize)) * stepA;
            }
        }
        if (gid >= 3u * rank && gid < 4u * rank) {
            uint outRow = gid - 3u * rank;
            // err and NLMS grads do not depend on option index; compute once, apply to every option slice (same math as prior per-opt recompute).
            thread float gradAcc[384];
            thread float gradAccA[384];
            for (uint i = 0u; i < rank; ++i) { gradAcc[i] = 0.0f; }
            for (uint a = 0u; a < actDim; ++a) { gradAccA[a] = 0.0f; }
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                uint aOff = row * actDim;
                float err = sanitize(state[row * rank + outRow] - rolloutState[row * rank + outRow], raw, san);
                for (uint i = 0u; i < rank; ++i) {
                    gradAcc[i] += err * sanitize(previousState[sOff + i], raw, san);
                }
                for (uint a = 0u; a < actDim; ++a) {
                    gradAccA[a] += err * sanitize(previousAction[aOff + a], raw, san);
                }
            }
            float invB = 1.0f / float(batchSize);
            for (uint opt = 0u; opt < optionCount; ++opt) {
                uint blockBase = primitivePacked + opt * optionForwardStride;
                for (uint i = 0u; i < rank; ++i) {
                    float step = 1.0f / (featureSecondMoment[i] + ridge);
                    stateWeights[blockBase + outRow * (rank + actDim) + i] += gradAcc[i] * invB * step;
                }
                for (uint a = 0u; a < actDim; ++a) {
                    float stepA = 1.0f / (featureSecondMoment[outRow] + ridge);
                    stateWeights[blockBase + outRow * (rank + actDim) + rank + a] += gradAccA[a] * invB * stepA;
                }
            }
        }

        if (gid >= policyBase && gid < policyBase + rank) {
            uint feat = gid - policyBase;
            float muF = 1.0f / (featureSecondMoment[feat] + ridge);
            // gradP does not depend on actIdx (same td × state feature for every action row).
            float gradP = 0.0f;
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                gradP += td[row] * sanitize(previousState[sOff + feat], raw, san);
            }
            float dPol = (gradP / float(batchSize)) * muF;
            for (uint actIdx = 0u; actIdx < actDim; ++actIdx) {
                policyWeights[actIdx * rank + feat] += dPol;
            }
            // Row-outer: each row belongs to one option index po (no scan over all opts).
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                uint aOff = row * actDim;
                float prevOptF = latentPower[row];
                uint po = uint(floor(prevOptF + 0.5f));
                if (po >= optionCount) po = optionCount - 1u;
                float sv = td[row] * sanitize(previousState[sOff + feat], raw, san);
                optionValues[po * rank + feat] += (sv / float(batchSize)) * muF;
                uint base = po * rank * actDim;
                for (uint actIdx = 0u; actIdx < actDim; ++actIdx) {
                    float actScale = sanitize(previousAction[aOff + actIdx], raw, san);
                    optionPolicyWeights[base + actIdx * rank + feat] += ((sv * actScale) / float(batchSize)) * muF;
                }
            }
            if (feat < baseObsDim) {
                float gradE = 0.0f;
                for (uint row = 0u; row < batchSize; ++row) {
                    float centered = observation[row * obsDim + feat] - observationMean[feat];
                    gradE += latentPower[batchSize + row] * centered;
                }
                encoderWeights[feat] += (gradE / float(batchSize)) * muF;
            }
            if (feat < rank) {
                float gradR = 0.0f;
                for (uint row = 0u; row < batchSize; ++row) {
                    uint sOff = row * rank;
                    float rErr = reward[row] - rewardPrediction[row];
                    gradR += rErr * sanitize(previousState[sOff + feat], raw, san);
                }
                rewardWeights[feat] += (gradR / float(batchSize)) * muF;
            }
        }

        if (gid >= criticBase && gid < criticBase + optionCount * rank) {
            uint lin = gid - criticBase;
            uint opt = lin / rank;
            uint feat = lin % rank;
            float muF = 1.0f / (featureSecondMoment[feat] + ridge);
            float gradC = 0.0f;
            for (uint row = 0u; row < batchSize; ++row) {
                uint sOff = row * rank;
                float prevOptF = latentPower[row];
                uint po = uint(floor(prevOptF + 0.5f));
                if (po >= optionCount) po = optionCount - 1u;
                if (po != opt) continue;
                gradC += td[row] * sanitize(previousState[sOff + feat], raw, san);
            }
            optionCriticWeights[opt * rank + feat] += (gradC / float(batchSize)) * muF;
        }
        return;
    }

    if (config.phase == kPhaseMetaPolicy) {
        uint row = gid / actDim;
        uint actIdx = gid % actDim;
        if (row < batchSize) {
            uint sOff = row * rank;
            float prevOptF = latentPower[row];
            uint poPrev = uint(floor(prevOptF + 0.5f));
            if (poPrev >= optionCount) poPrev = optionCount - 1u;
            uint phase = stompState[row * 2u];
            uint elapsed = stompState[row * 2u + 1u];
            uint activeOption = 0u;
            float pickScore = 0.0f;
            if (phase == 1u && elapsed < rank) {
                activeOption = poPrev;
                for (uint i = 0u; i < activeCount; ++i) {
                    uint idx = activeList[i];
                    pickScore += optionValues[activeOption * rank + idx] * sanitize(state[sOff + idx], raw, san);
                }
            } else {
                float bestScore = -1e10f;
                for (uint opt = 0u; opt < optionCount; ++opt) {
                    float score = 0.0f;
                    uint off = opt * rank;
                    for (uint i = 0u; i < activeCount; ++i) {
                        uint idx = activeList[i];
                        float sv = sanitize(state[sOff + idx], raw, san);
                        score += optionValues[off + idx] * sv;
                    }
                    if (opt == 0u || score > bestScore) {
                        bestScore = score;
                        activeOption = opt;
                    }
                }
                pickScore = bestScore;
                if (actIdx == 0u) {
                    stompState[row * 2u] = 1u;
                    stompState[row * 2u + 1u] = 0u;
                    uint ocR = optionCount * rank;
                    optionCriticWeights[ocR + activeOption] += (pickScore - optionCriticWeights[ocR + activeOption]) * invT;
                }
            }
            uint base = activeOption * rank * actDim;
            float baseLogit = masked_dot(policyWeights, actIdx * rank, state, sOff, activeList, activeCount, raw, san);
            float optLogit = masked_dot(optionPolicyWeights, base + actIdx * rank, state, sOff, activeList, activeCount, raw, san);
            float logit = sanitize(baseLogit + optLogit, raw, san);
            uint aOff = row * actDim;
            float lo = actionBounds[actIdx];
            float hi = actionBounds[actDim + actIdx];
            float bounded = clamp(logit, lo, hi);
            action[aOff + actIdx] = bounded;
            actionGrad[aOff + actIdx] = bounded;
            latentPower[batchSize + row] = float(activeOption);
        }
        return;
    }

    // Planning: average-reward lookahead — K-step greedy roll (structural K) + primary/secondary critics; objective-only (Invariant 8).
    if (config.phase == kPhasePlanning) {
        if (gid < batchSize) {
            uint aOff = gid * actDim;
            uint sOff = gid * rank;
            float prevOptF = latentPower[gid];
            uint po = uint(floor(prevOptF + 0.5f));
            if (po >= optionCount) po = optionCount - 1u;
            float vNow = masked_dot(valueWeights, 0, state, sOff, activeList, activeCount, raw, san);
            float vSecNow = dot_active(optionCriticWeights, po * rank, state, sOff, activeList, activeCount, raw, san);
            vNow += vSecNow;

            if (isDisc) {
                uint bestA = 0u;
                float bestV = -1e10f;
                for (uint cand = 0u; cand < actDim; ++cand) {
                    float vCand = discrete_lookahead1(stateWeights, valueWeights, optionCriticWeights, po,
                        state, sOff, actDim, rank, activeList, activeCount, cand, raw, san);
                    if (cand == 0u || vCand > bestV) {
                        bestV = vCand;
                        bestA = cand;
                    }
                }
                float adv1 = bestV - vNow;
                float adv = adv1;
                if (planK > 1u) {
                    thread float scratch[384];
                    const uint maxR = 384u;
                    uint capRank = min(rank, maxR);
                    float advK = kstep_plan_discrete(stateWeights, valueWeights, optionCriticWeights, po,
                        scalars, state, sOff, actDim, rank, activeList, activeCount, planK, raw, san, scratch, capRank);
                    adv = max(adv1, advK);
                }
                if (adv > 0.0f) {
                    float scale = adv / (1.0f + fabs(vNow));
                    for (uint a = 0u; a < actDim; ++a) {
                        float base = action[aOff + a];
                        float plan = (a == bestA) ? (base + scale) : base;
                        action[aOff + a] = sanitize(plan, raw, san);
                    }
                }
                for (uint a = 0u; a < actDim; ++a) {
                    action[aOff + a] = clamp(action[aOff + a], actionBounds[a], actionBounds[actDim + a]);
                }
            } else {
                float bestV = vNow;
                thread float probeAct[64];
                thread float bestVec[64];
                uint adCap = min(actDim, 64u);
                for (uint a = 0u; a < adCap; ++a) bestVec[a] = action[aOff + a];
                for (uint dim = 0u; dim < adCap; ++dim) {
                    float span = actionBounds[actDim + dim] - actionBounds[dim];
                    float eps = span / float(max(rank + actDim, 1u));
                    for (uint sp = 0u; sp < 2u; ++sp) {
                        float sgn = (sp == 0u) ? -1.0f : 1.0f;
                        for (uint a = 0u; a < adCap; ++a) probeAct[a] = action[aOff + a];
                        float trial = probeAct[dim] + sgn * eps;
                        probeAct[dim] = clamp(trial, actionBounds[dim], actionBounds[actDim + dim]);
                        float vCand = lookahead_value_thread(stateWeights, valueWeights, optionCriticWeights, po, state, sOff, actDim, rank, activeList, activeCount, raw, san, probeAct);
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
                        float base = action[aOff + a];
                        float plan = base + scale * (bestVec[a] - base);
                        action[aOff + a] = sanitize(plan, raw, san);
                    }
                }
                for (uint a = 0u; a < actDim; ++a) {
                    action[aOff + a] = clamp(action[aOff + a], actionBounds[a], actionBounds[actDim + a]);
                }
            }
        }
        return;
    }

    // Fused online maintenance + health telemetry. Raw errors are clamped to `stats_abs_bound_*` so float sums stay finite;
    // squared terms are then divided by bound² so reported means are O(1) (fraction of worst-case telemetry scale).
    // TD/learn phases use `sanitize` on raw errors without this bound—use `maxAbs*` in stats when tails matter.
    if (config.phase == kPhaseBookkeeping && gid == 0u) {
        for (uint row = 0u; row < batchSize; ++row) {
            uint p = stompState[row * 2u];
            if (p == 1u) {
                stompState[row * 2u + 1u] += 1u;
                if (stompState[row * 2u + 1u] >= rank) {
                    stompState[row * 2u] = 0u;
                    stompState[row * 2u + 1u] = 0u;
                }
            }
        }
        float totalUtility = 0.0f;
        for (uint i = 0u; i < rank; ++i) totalUtility += featureResidual[i];
        float threshold = totalUtility / float(max(rank * rank, 1u));
        uint count = 0u;
        for (uint i = 0u; i < rank; ++i) {
            if (featureResidual[i] > threshold) {
                activeMask[i] = 1u;
                activeList[count++] = i;
            } else {
                activeMask[i] = 0u;
            }
        }
        scalars->activeCount = count > 0u ? count : 1u;

        float batchR = 0.0f;
        for (uint i = 0u; i < batchSize; ++i) batchR += reward[i];
        float meanR = batchR / float(batchSize);
        float t = float(max(scalars->stepCount + 1u, 1u));
        scalars->rewardRate += (meanR - scalars->rewardRate) / t;
        scalars->stepCount++;

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
        float denom = float(max(batchSize * rank, 1u));
        uint stRaw = 0u;
        uint stSan = 0u;
        const uint nLatTerms = max(batchSize * rank, 1u);
        const uint nRewTerms = max(batchSize, 1u);
        const uint nActTerms = max(batchSize * actDim, 1u);
        const float boundLat = stats_abs_bound_for_sum_squares(nLatTerms);
        const float boundRew = stats_abs_bound_for_sum_squares(nRewTerms);
        const float boundAct = stats_abs_bound_for_sum_squares(nActTerms);
        // Scale-invariant telemetry: each squared term is divided by bound² so per-term contribution ∈ [0,1] and reported
        // means stay O(1) instead of pinning to FLT_MAX-derived ceilings when raw errors are large.
        const float invBoundLat = 1.0f / boundLat;
        const float invBoundRew = 1.0f / boundRew;
        const float invBoundAct = 1.0f / boundAct;
        // One pass over rows: fused rank stats + obs sums for online mean (rank/obsDim ≤ 384).
        thread float sumObsByF[384];
        for (uint f = 0u; f < obsDim; ++f) { sumObsByF[f] = 0.0f; }

        for (uint row = 0u; row < batchSize; ++row) {
            uint sOff = row * rank;
            uint aOff = row * actDim;
            for (uint k = 0u; k < rank; ++k) {
                float pred = latentError[row * rank + k];
                float tgt = state[row * rank + k];
                float ds = sanitize(tgt - pred, stRaw, stSan);
                float dc = clamp(ds, -boundLat, boundLat);
                float nlc = dc * invBoundLat;
                lmse = fma(nlc, nlc, lmse);
                maxLat = max(maxLat, fabs(ds));
                float sv = state[sOff + k];
                if (!isfinite(sv)) nLearn++;
                float sc = sanitize(sv, stRaw, stSan);
                float vc = clamp(sc, -boundLat, boundLat);
                float nvc = vc * invBoundLat;
                vtrace = fma(nvc, nvc, vtrace);
            }
            float rds = sanitize(reward[row] - rewardPrediction[row], stRaw, stSan);
            float rc = clamp(rds, -boundRew, boundRew);
            float nrc = rc * invBoundRew;
            rmse = fma(nrc, nrc, rmse);
            maxRP = max(maxRP, fabs(rds));
            float pa = 0.0f;
            float pb = 0.0f;
            for (uint a = 0u; a < actDim; ++a) {
                float x = action[aOff + a];
                float yg = actionGrad[aOff + a];
                float ps = sanitize(x - yg, stRaw, stSan);
                float pc = clamp(ps, -boundAct, boundAct);
                float npc = pc * invBoundAct;
                pa = fma(npc, npc, pa);
                maxPol = max(maxPol, fabs(x));
                if (!isfinite(x)) nPlan++;
                if (!isfinite(yg)) nPlan++;
                float py = previousAction[aOff + a];
                float pbs = sanitize(x - py, stRaw, stSan);
                float pbc = clamp(pbs, -boundAct, boundAct);
                float npb = pbc * invBoundAct;
                pb = fma(npb, npb, pb);
            }
            planL2 += pa;
            actDelta += pb;
            for (uint f = 0u; f < obsDim; ++f) {
                float o = observation[row * obsDim + f];
                if (!isfinite(o)) nEnc++;
                sumObsByF[f] += o;
            }
        }
        nraw = nEnc + nLearn + nPlan + stRaw;
        nsan = stSan;
        float invBatch = 1.0f / float(max(batchSize, 1u));
        float invStep = 1.0f / float(max(scalars->stepCount, 1u));
        for (uint f = 0u; f < obsDim; ++f) {
            float batchMean = sumObsByF[f] * invBatch;
            observationMean[f] += invStep * (batchMean - observationMean[f]);
        }
        statsOut->rewardRate = scalars->rewardRate;
        statsOut->latentPredictionMSE = lmse / denom;
        statsOut->rewardPredictionMSE = rmse / float(max(batchSize, 1u));
        statsOut->latentStateMeanSquare = vtrace / float(max(batchSize * rank, 1u));
        statsOut->activeRank = float(scalars->activeCount);
        statsOut->planningActionChangeL2Mean = planL2 / float(max(batchSize * actDim, 1u));
        statsOut->actionDeltaL2Mean = actDelta / float(max(batchSize * actDim, 1u));
        statsOut->rawNonFiniteCount = float(nraw);
        statsOut->sanitizedNonFiniteCount = float(nsan);
        statsOut->encodeNonFiniteCount = float(nEnc);
        statsOut->learnNonFiniteCount = float(nLearn);
        statsOut->planNonFiniteCount = float(nPlan);
        statsOut->maxAbsLatentPrediction = maxLat;
        statsOut->maxAbsRewardPrediction = maxRP;
        statsOut->maxAbsPolicyOutput = maxPol;
        counters[0] = nEnc;
        counters[1] = nLearn;
        counters[2] = nPlan;
    }
}
