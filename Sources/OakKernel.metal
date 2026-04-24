// OakKernel.metal - Step 1-11 OaK rebuild.
//
// This kernel file follows the Plan order rather than the old L3-L7 fused scaffold:
// normalized fixed anchors -> constructed features -> GVFs -> actor-critic ->
// average reward -> option models/planning -> search control -> STOMP/Oak feedback.
//
// The agent still receives only the continuing (observation, previous action descriptor,
// previous reward) stream. The wrapper remains the only action-space-aware code.
#include <metal_stdlib>
#include <float.h>
using namespace metal;

constant uint OAK_CNT_OBS_NF        = 0u;
constant uint OAK_CNT_LEARN_NF      = 1u;
constant uint OAK_CNT_PLAN_NF       = 2u;
constant uint OAK_CNT_REINIT_OPT    = 3u;
constant uint OAK_CNT_REINIT_FEAT   = 4u;
constant uint OAK_CNT_TDPSI_NF      = 5u;
constant uint OAK_CNT_IMPORTANCE_NF = 6u;

constant uint OAK_MAX_RANK = 256u;
constant uint OAK_SEARCH_SLOT_COUNT = 3u;  // high-priority, stalest, diagnostic/coverage
constant uint OAK_SEARCH_SLOT_HIGH = 0u;
constant uint OAK_SEARCH_SLOT_STALE = 1u;
constant uint OAK_GVF_KIND_COUNT = 4u;     // state, action, reward-rate, model-residual

struct KernelConfig {
    uint obsDim, rank, optionCount, baseObsDim, actionParamCount;
    uint baseObsTraceLevels;
    uint linearAnchorCount;
    uint optionPageWidth;
    float invRank, invOptionCount;
    float invOneInputWidth, metaEta;
    float betaCap, betaFloor;
    uint pValue, pPsi;
    uint pOptPolicy, pOptReward, pOptTransition, pOptDuration;
    uint pOptValue, pPsiCross;
    uint pOneReward, pOneTransition;
    uint pGvfRate, pGvfValue;
    uint sObservation, sObsMean, sObsSM;
    uint sState;
    uint sPolicyMean, sActionSignal, sScoreGrad;
    uint sOptionPolicyMean;
    uint sImportance, sImportanceLog;
    uint sReward, sTd, sTdPsi;
    uint sChord, sChordPrev;
    uint sUtilFeat, sUtilOpt;
    uint sBonusW;
    uint sFeatureMean, sFeatureSM;
    uint sPlanAdv;
    uint sAlpha, sBeta, sH;
    uint sNTdMat;
    uint sOptRewardTd;
    uint sOptionDiffTd;
    uint sOptValueTd, sOptValueDenom, sCrossTd, sOneTransitionTd;
    uint sOneRewardTd, sOneStateTd, sOneValueWeightedError, sOneStepDenom;
    uint sDurationTd, sDurationHat;
    uint sQOption;
    uint sObsTraceMulti;
    uint sPrioOpt;
    uint sContinuation;
    uint sRhoOpt;
    uint sLambda, sKeyboardScore, sBestModelNext;
    uint sGvfTd, sGvfUtility, sFeatureTraceDenom;
    uint sReplacementAgeMean;
    uint uCounters;
    uint uOptFeature;
    uint uFeatureAge;
    uint uOptAge;
    uint uFeatureEverUsed;
    uint uFeatureParentA, uFeatureParentB, uFeatureKind;
    uint uFeatureBirthCount, uParentUseCount;
    uint uSelectedOption, uHighModelOption, uStaleModelOption;
    uint uKeyboardAge;
    uint uModelAge;
};

struct KernelScalars {
    uint stepCount;
    uint stateSlot;
    float rho;
    float alphaPi;
    float meanPrio;
};

struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    float chordEffectiveK;
    float rrBonusMean, psiStopGapMean, psiStopGapAbsMean, psiAbsMean;
    float rrStopAbsMean, valueAbsMean, terminationRate, optionDurationMean;
    float neverUsedFeatureCount, replacementAgeMean, stateEffectiveRank;
    float eligibleOptionCount, eligibleFeatureCount, replacementCommitRate;
    float modelUpdateRate, modelPriorityEntropy, optionDiffTdMean;
    float optionCount, optionPageCount;
    float linearAnchorCount, constructedFeatureCount;
    float featureCorrMean, featureMaxCorr, featureEffectiveRank;
    float candidateSurvivalMean, parentReuseEntropy, utilityTransferMean;
    float importanceLogMean, importanceLogVar, importanceEss;
    float optValueTdMean, rhoValueTdMean, crossPsiDiagRatio;
    float keyboardGpiMargin, keyboardSelectedOption, keyboardDesign;
    float oneStepRewardTdMean, oneStepStateTdMean;
    float oneStepValueWeightedError;
    float modelQueueCoverage, modelUncertaintyMean, stalePageAgeMean;
    float gvfCount, gvfRateMean, gvfTdMean;
    float gvfDifferentialValueMean, gvfUtilityMean;
    float lambdaEffectiveRank, lambdaPlanMass;
    float lambdaRrMass, lambdaUtilityMass;
    float keyboardCacheCoverage, keyboardScoreStaleness;
};

inline float safe_w(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

inline float hash_uniform(uint a, uint b, uint c) {
    uint z = a ^ (b * 0x7feb352du) ^ (c * 0x846ca68bu);
    z = (z ^ (z >> 16)) * 0x9e3779b9u;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z ^= z >> 16;
    return float(z >> 8) * 5.9604645e-8f;
}

struct Ctx {
    uint O, R, K, baseO, D, traceLevels, anchorCount, pageWidth;
    uint stepCount, stateSlot;
    float rho, alphaPi, meanPrio;
    float invR, invK, invOneInputWidth, metaEta, betaCap, betaFloor;
    device float *valueW, *psi, *optPolicy, *optReward, *optTransition, *optDuration;
    device float *optValue, *psiCross, *oneReward, *oneTransition;
    device float *gvfRate, *gvfValue;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *actionSignal, *scoreGrad, *optionPolicyMean;
    device float *importance, *importanceLog;
    device float *reward, *td, *tdPsi;
    device float *chord, *chordPrev;
    device float *utilFeat, *utilOpt, *bonusW, *featureMean, *featureSM;
    device float *planAdv, *alpha, *beta, *hMeta;
    device float *nTdMat, *optRewardTd, *optionDiffTd, *optValueTd, *optValueDenom, *crossTd, *oneTransitionTd;
    device float *oneRewardTd, *oneStateTd, *oneValueWeightedError, *oneStepDenom;
    device float *durationTd, *durationHat, *qOption, *obsTraceMulti, *prioOpt, *continuation, *rhoOpt;
    device float *lambda, *keyboardScore, *bestModelNext, *gvfTd, *gvfUtility, *featureTraceDenom;
    device float *replacementAgeMean;
    device uint *counters, *optFeature, *featureAge, *optAge, *featureEverUsed;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount, *parentUseCount;
    device uint *selectedOption, *highModelOption, *staleModelOption, *keyboardAge, *modelAge;
    device KernelScalars *scalars;
    device KernelStats *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount; c.baseO = cfg.baseObsDim;
    c.D = cfg.actionParamCount; c.traceLevels = cfg.baseObsTraceLevels;
    c.anchorCount = cfg.linearAnchorCount; c.pageWidth = cfg.optionPageWidth;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho; c.alphaPi = scalars->alphaPi; c.meanPrio = scalars->meanPrio;
    c.invR = cfg.invRank; c.invK = cfg.invOptionCount;
    c.invOneInputWidth = cfg.invOneInputWidth; c.metaEta = cfg.metaEta;
    c.betaCap = cfg.betaCap; c.betaFloor = cfg.betaFloor;
    c.valueW = params + cfg.pValue; c.psi = params + cfg.pPsi;
    c.optPolicy = params + cfg.pOptPolicy; c.optReward = params + cfg.pOptReward;
    c.optTransition = params + cfg.pOptTransition; c.optDuration = params + cfg.pOptDuration;
    c.optValue = params + cfg.pOptValue; c.psiCross = params + cfg.pPsiCross;
    c.oneReward = params + cfg.pOneReward; c.oneTransition = params + cfg.pOneTransition;
    c.gvfRate = params + cfg.pGvfRate; c.gvfValue = params + cfg.pGvfValue;
    c.observation = stateArena + cfg.sObservation; c.observationMean = stateArena + cfg.sObsMean;
    c.obsSM = stateArena + cfg.sObsSM; c.state = stateArena + cfg.sState;
    c.policyMean = stateArena + cfg.sPolicyMean; c.actionSignal = stateArena + cfg.sActionSignal;
    c.scoreGrad = stateArena + cfg.sScoreGrad; c.optionPolicyMean = stateArena + cfg.sOptionPolicyMean;
    c.importance = stateArena + cfg.sImportance; c.importanceLog = stateArena + cfg.sImportanceLog;
    c.reward = stateArena + cfg.sReward; c.td = stateArena + cfg.sTd; c.tdPsi = stateArena + cfg.sTdPsi;
    c.chord = stateArena + cfg.sChord; c.chordPrev = stateArena + cfg.sChordPrev;
    c.utilFeat = stateArena + cfg.sUtilFeat; c.utilOpt = stateArena + cfg.sUtilOpt;
    c.bonusW = stateArena + cfg.sBonusW; c.featureMean = stateArena + cfg.sFeatureMean;
    c.featureSM = stateArena + cfg.sFeatureSM; c.planAdv = stateArena + cfg.sPlanAdv;
    c.alpha = stateArena + cfg.sAlpha; c.beta = stateArena + cfg.sBeta; c.hMeta = stateArena + cfg.sH;
    c.nTdMat = stateArena + cfg.sNTdMat; c.optRewardTd = stateArena + cfg.sOptRewardTd;
    c.optionDiffTd = stateArena + cfg.sOptionDiffTd; c.optValueTd = stateArena + cfg.sOptValueTd;
    c.optValueDenom = stateArena + cfg.sOptValueDenom; c.crossTd = stateArena + cfg.sCrossTd;
    c.oneTransitionTd = stateArena + cfg.sOneTransitionTd; c.oneRewardTd = stateArena + cfg.sOneRewardTd;
    c.oneStateTd = stateArena + cfg.sOneStateTd; c.oneValueWeightedError = stateArena + cfg.sOneValueWeightedError;
    c.oneStepDenom = stateArena + cfg.sOneStepDenom;
    c.durationTd = stateArena + cfg.sDurationTd; c.durationHat = stateArena + cfg.sDurationHat;
    c.qOption = stateArena + cfg.sQOption; c.obsTraceMulti = stateArena + cfg.sObsTraceMulti;
    c.prioOpt = stateArena + cfg.sPrioOpt; c.continuation = stateArena + cfg.sContinuation;
    c.rhoOpt = stateArena + cfg.sRhoOpt; c.lambda = stateArena + cfg.sLambda;
    c.keyboardScore = stateArena + cfg.sKeyboardScore; c.bestModelNext = stateArena + cfg.sBestModelNext;
    c.gvfTd = stateArena + cfg.sGvfTd; c.gvfUtility = stateArena + cfg.sGvfUtility;
    c.featureTraceDenom = stateArena + cfg.sFeatureTraceDenom;
    c.replacementAgeMean = stateArena + cfg.sReplacementAgeMean;
    c.counters = uintArena + cfg.uCounters; c.optFeature = uintArena + cfg.uOptFeature;
    c.featureAge = uintArena + cfg.uFeatureAge; c.optAge = uintArena + cfg.uOptAge;
    c.featureEverUsed = uintArena + cfg.uFeatureEverUsed;
    c.featureParentA = uintArena + cfg.uFeatureParentA; c.featureParentB = uintArena + cfg.uFeatureParentB;
    c.featureKind = uintArena + cfg.uFeatureKind; c.featureBirthCount = uintArena + cfg.uFeatureBirthCount;
    c.parentUseCount = uintArena + cfg.uParentUseCount; c.selectedOption = uintArena + cfg.uSelectedOption;
    c.highModelOption = uintArena + cfg.uHighModelOption; c.staleModelOption = uintArena + cfg.uStaleModelOption;
    c.keyboardAge = uintArena + cfg.uKeyboardAge; c.modelAge = uintArena + cfg.uModelAge;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline uint linear_input_width(Ctx ctx) { return (ctx.traceLevels + 1u) * ctx.baseO + ctx.D + 1u; }

inline float standardized_obs(Ctx ctx, uint o) {
    float mean = ctx.observationMean[o];
    float sm = ctx.obsSM[o];
    float var = max(sm - mean * mean, 0.0f);
    return (ctx.observation[o] - mean) * rsqrt(var + 1.0f / float(max(ctx.baseO, 1u)));
}

inline float linear_input(Ctx ctx, uint idx) {
    uint obsSpan = (ctx.traceLevels + 1u) * ctx.baseO;
    if (idx < ctx.baseO) return standardized_obs(ctx, idx);
    if (idx < obsSpan) {
        uint t = idx - ctx.baseO;
        return ctx.obsTraceMulti[t];
    }
    uint tail = idx - obsSpan;
    if (tail < ctx.D) return ctx.actionSignal[tail];
    return ctx.reward[0];
}

inline float one_input(Ctx ctx, device const float *prev, uint idx) {
    if (idx < ctx.R) return prev[idx];
    return ctx.actionSignal[idx - ctx.R];
}

inline float gvf_cumulant(Ctx ctx, device const float *curr, uint g) {
    uint kind = g % OAK_GVF_KIND_COUNT;
    if (kind == 0u) return curr[g % ctx.R];
    if (kind == 1u) {
        if (ctx.D == 0u) return 0.0f;
        return ctx.actionSignal[g % ctx.D];
    }
    if (kind == 2u) return ctx.reward[0] - ctx.scalars->rho;
    float e = ctx.oneValueWeightedError[0];
    return isfinite(e) ? fabs(e) : 0.0f;
}

inline void oak_keyboard_predict_body(Ctx ctx, uint tid,
                                      threadgroup float *tgQNext,
                                      threadgroup float *tgG,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgStopGap,
                                      threadgroup float *tgTerm,
                                      threadgroup float *tgOptTdAbs,
                                      threadgroup float *tgLambda,
                                      threadgroup float *tgScalars) {
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float vPrev = 0.0f, vCurr = 0.0f;
    if (tid == 0u) {
        for (uint i = 0u; i < ctx.R; ++i) {
            vPrev = fma(ctx.valueW[i], prev[i], vPrev);
            vCurr = fma(ctx.valueW[i], curr[i], vCurr);
        }
        tgScalars[0] = vPrev;
        tgScalars[1] = vCurr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vPrev = tgScalars[0];
    vCurr = tgScalars[1];

    if (tid < ctx.K) {
        float q = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) q = fma(ctx.optValue[tid * ctx.R + i], curr[i], q);
        tgQNext[tid] = isfinite(q) ? q : -FLT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float maxQNext = -FLT_MAX;
        for (uint k = 0u; k < ctx.K; ++k) maxQNext = max(maxQNext, tgQNext[k]);
        if (!(maxQNext > -FLT_MAX)) maxQNext = vCurr;
        tgScalars[2] = maxQNext;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxQNext = tgScalars[2];

    float rhoOld = ctx.scalars->rho;
    float rewardC = ctx.reward[0] - rhoOld;
    if (tid == 0u) {
        float delta = rewardC + vCurr - vPrev;
        if (!isfinite(delta)) { delta = 0.0f; ctx.counters[OAK_CNT_LEARN_NF] += 1u; }
        ctx.td[0] = delta;
        float rhoNew = rhoOld + ctx.invR * delta;
        ctx.scalars->rho = isfinite(rhoNew) ? rhoNew : rhoOld;
        ctx.stats->rhoValueTdMean = delta;
    }
    if (tid < ctx.anchorCount) {
        float hPrev = 0.0f, hNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.gvfValue[tid * ctx.R + i];
            hPrev = fma(w, prev[i], hPrev);
            hNext = fma(w, curr[i], hNext);
        }
        float d = gvf_cumulant(ctx, curr, tid) - ctx.gvfRate[tid] + hNext - hPrev;
        ctx.gvfTd[tid] = isfinite(d) ? d : 0.0f;
    }
    if (tid == 0u) {
        float alphaTraceNorm = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float a = ctx.alpha[i];
            float x = prev[i];
            if (isfinite(a) && isfinite(x)) alphaTraceNorm = fma(a, x * x, alphaTraceNorm);
        }
        tgScalars[7] = (isfinite(alphaTraceNorm) && alphaTraceNorm > 0.0f) ? alphaTraceNorm : 0.0f;
        ctx.featureTraceDenom[0] = tgScalars[7] + 1.0f;
        float oneTraceNorm = tgScalars[7];
        float actionAlpha = ctx.invOneInputWidth;
        for (uint d = 0u; d < ctx.D; ++d) {
            float x = ctx.actionSignal[d];
            if (isfinite(x)) oneTraceNorm = fma(actionAlpha, x * x, oneTraceNorm);
        }
        float oneDenom = oneTraceNorm + 1.0f;
        ctx.oneStepDenom[0] = (isfinite(oneDenom) && oneDenom > 0.0f) ? oneDenom : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        uint feat = ctx.optFeature[tid] % ctx.R;
        uint base = tid * ctx.R;
        float psiPrev = 0.0f, psiNext = 0.0f, qPrev = 0.0f, qNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            psiPrev = fma(ctx.psi[base + i], prev[i], psiPrev);
            psiNext = fma(ctx.psi[base + i], curr[i], psiNext);
            qPrev = fma(ctx.optValue[base + i], prev[i], qPrev);
            qNext = fma(ctx.optValue[base + i], curr[i], qNext);
        }
        float zStop = vCurr - ctx.valueW[feat] * curr[feat] + ctx.bonusW[feat] * curr[feat];
        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop >= psiNext) ? 1.0f : 0.0f;
        float cont = 1.0f - beta;
        ctx.continuation[tid] = cont;
        float dPsi = rewardC + cont * psiNext + beta * zStop - psiPrev;
        if (!isfinite(dPsi)) dPsi = 0.0f;
        ctx.tdPsi[tid] = dPsi;
        float u = cont * qNext + beta * maxQNext;
        float dQ = rewardC + u - qPrev;
        if (!isfinite(dQ)) dQ = 0.0f;
        ctx.optValueTd[tid] = dQ;
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = fma(rho, tgScalars[7], 1.0f);
        ctx.optValueDenom[tid] = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        tgCont[tid] = cont;
        tgStopGap[tid] = isfinite(psiNext - zStop) ? (psiNext - zStop) : 0.0f;
        tgTerm[tid] = beta;
        tgOptTdAbs[tid] = fabs(dQ);
        tgG[tid] = isfinite(psiNext) ? fabs(psiNext) : 0.0f;
        tgLambda[tid] = isfinite(zStop) ? fabs(zStop) : 0.0f;
    }
    if (tid < ctx.K) {
        uint target = ctx.stepCount % max(ctx.K, 1u);
        uint base = (tid * ctx.K + target) * ctx.R;
        float crossPrev = 0.0f, crossNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.psiCross[base + i];
            crossPrev = fma(w, prev[i], crossPrev);
            crossNext = fma(w, curr[i], crossNext);
        }
        float z = vCurr - ctx.valueW[target] * curr[target] + ctx.bonusW[target] * curr[target];
        float beta = (z >= crossNext) ? 1.0f : 0.0f;
        float td = (ctx.reward[0] - ctx.scalars->rho) + (1.0f - beta) * crossNext + beta * z - crossPrev;
        ctx.crossTd[tid] = isfinite(td) ? td : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gapAbsSum = 0.0f, psiAbsSum = 0.0f, zAbsSum = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            gapAbsSum += fabs(tgStopGap[k]);
            psiAbsSum += tgG[k];
            zAbsSum += tgLambda[k];
        }
        float invK = ctx.invK;
        ctx.stats->psiStopGapAbsMean = gapAbsSum * invK;
        ctx.stats->psiAbsMean = psiAbsSum * invK;
        ctx.stats->rrStopAbsMean = zAbsSum * invK;
        ctx.stats->valueAbsMean = fabs(vCurr);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        float plan = ctx.valueW[tid] * (ctx.bestModelNext[tid] - curr[tid]);
        float rr = (ctx.bonusW[tid] - ctx.valueW[tid]) * curr[tid];
        float util = ctx.utilFeat[tid];
        tgQNext[tid] = (isfinite(plan) && plan > 0.0f) ? plan : 0.0f;
        tgG[tid] = (isfinite(rr) && rr > 0.0f) ? rr : 0.0f;
        tgLambda[tid] = (isfinite(util) && util > 0.0f) ? util : 0.0f;
        if (ctx.keyboardAge[tid] < 0xffffffffu) ctx.keyboardAge[tid] += 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float planSum = 0.0f, rrSum = 0.0f, utilSum = 0.0f;
        for (uint i = 0u; i < ctx.K; ++i) {
            planSum += tgQNext[i];
            rrSum += tgG[i];
            utilSum += tgLambda[i];
        }
        uint live = (planSum > 0.0f ? 1u : 0u) + (rrSum > 0.0f ? 1u : 0u)
                  + (utilSum > 0.0f ? 1u : 0u);
        float invLive = (live > 0u) ? (1.0f / float(live)) : 0.0f;
        float invK = ctx.invK;
        float l2 = 0.0f;
        for (uint i = 0u; i < ctx.K; ++i) {
            float lam = (live == 0u) ? invK : 0.0f;
            if (planSum > 0.0f) lam += invLive * tgQNext[i] / planSum;
            if (rrSum > 0.0f) lam += invLive * tgG[i] / rrSum;
            if (utilSum > 0.0f) lam += invLive * tgLambda[i] / utilSum;
            if (!isfinite(lam) || lam < 0.0f) lam = invK;
            ctx.lambda[i] = lam;
            tgLambda[i] = lam;
            l2 += lam * lam;
        }
        tgScalars[3] = (l2 > 0.0f) ? (1.0f / l2) : float(ctx.K);
        tgScalars[4] = (planSum > 0.0f) ? invLive : 0.0f;
        tgScalars[5] = (rrSum > 0.0f) ? invLive : 0.0f;
        tgScalars[6] = (utilSum > 0.0f) ? invLive : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pageCount = (ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
    uint page = ctx.stepCount % max(pageCount, 1u);
    uint cand = page * ctx.pageWidth + tid;
    if (tid < ctx.pageWidth && cand < ctx.K) {
        float G = 0.0f;
        for (uint key = 0u; key < ctx.K; ++key) {
            float lam = ctx.lambda[key];
            if (lam <= 0.0f) continue;
            uint base = (cand * ctx.K + key) * ctx.R;
            float p = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) p = fma(ctx.psiCross[base + i], curr[i], p);
            G = fma(lam, p, G);
        }
        ctx.keyboardScore[cand] = isfinite(G) ? G : -FLT_MAX;
        ctx.keyboardAge[cand] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint best = 0u;
        float bestG = -FLT_MAX, secondG = -FLT_MAX;
        float stopGapSum = 0.0f, termSum = 0.0f, optTdAbs = 0.0f;
        float diagAbs = 0.0f, scoreAbs = 0.0f, ageSum = 0.0f, covered = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            ctx.chordPrev[k] = ctx.chord[k];
            stopGapSum += tgStopGap[k];
            termSum += tgTerm[k];
            optTdAbs += tgOptTdAbs[k];
            uint diagBase = (k * ctx.K + k) * ctx.R;
            float pDiag = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) pDiag = fma(ctx.psiCross[diagBase + i], curr[i], pDiag);
            diagAbs += fabs(pDiag);
            float G = ctx.keyboardScore[k];
            scoreAbs += fabs(G);
            if (G > bestG) { secondG = bestG; bestG = G; best = k; }
            else if (G > secondG) secondG = G;
            uint age = ctx.keyboardAge[k];
            ageSum += float(age);
            if (age <= pageCount) covered += 1.0f;
        }
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        uint selected = (ctx.continuation[prevSel] > 0.0f) ? prevSel : best;
        ctx.selectedOption[0] = selected;
        for (uint k = 0u; k < ctx.K; ++k) ctx.chord[k] = ctx.lambda[k];

        float invK = ctx.invK;
        ctx.stats->chordEffectiveK = tgScalars[3];
        ctx.stats->psiStopGapMean = stopGapSum * invK;
        ctx.stats->terminationRate = termSum * invK;
        ctx.stats->optValueTdMean = optTdAbs * invK;
        ctx.stats->crossPsiDiagRatio = (diagAbs + scoreAbs > 0.0f) ? (diagAbs / (diagAbs + scoreAbs)) : 0.0f;
        ctx.stats->keyboardGpiMargin = isfinite(bestG - secondG) ? (bestG - secondG) : 0.0f;
        ctx.stats->keyboardSelectedOption = float(selected);
        ctx.stats->keyboardDesign = 0.0f;
        ctx.stats->lambdaEffectiveRank = tgScalars[3];
        ctx.stats->lambdaPlanMass = tgScalars[4];
        ctx.stats->lambdaRrMass = tgScalars[5];
        ctx.stats->lambdaUtilityMass = tgScalars[6];
        ctx.stats->keyboardCacheCoverage = covered * invK;
        ctx.stats->keyboardScoreStaleness = ageSum * invK;
        ctx.stats->optionCount = float(ctx.K);
        ctx.stats->optionPageCount = float((ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u));
        ctx.stats->linearAnchorCount = float(ctx.anchorCount);
        ctx.stats->constructedFeatureCount = float(ctx.R - min(ctx.anchorCount, ctx.R));

        float ss = 0.0f, gg = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) ss = fma(curr[i], curr[i], ss);
        for (uint d = 0u; d < ctx.D; ++d) gg = fma(ctx.scoreGrad[d], ctx.scoreGrad[d], gg);
        float tdEnergy = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float r = ctx.importance[k];
            float d = ctx.tdPsi[k];
            if (isfinite(r) && isfinite(d)) tdEnergy = fma(r * d, r * d, tdEnergy);
        }
        uint policyVolume = max(ctx.K * (ctx.R + ctx.D), 1u);
        ctx.scalars->alphaPi = 1.0f / (float(policyVolume) + ss * gg * max(tdEnergy, 1.0f));
    }
}

kernel void oak_encode(device float *params [[buffer(0)]],
                       device float *stateArena [[buffer(1)]],
                       device uint *uintArena [[buffer(2)]],
                       device KernelScalars *scalars [[buffer(3)]],
                       device KernelStats *stats [[buffer(4)]],
                       constant KernelConfig &cfg [[buffer(5)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgW [[threads_per_threadgroup]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    threadgroup float tgQNext[OAK_MAX_RANK];
    threadgroup float tgG[OAK_MAX_RANK];
    threadgroup float tgCont[OAK_MAX_RANK];
    threadgroup float tgStopGap[OAK_MAX_RANK];
    threadgroup float tgTerm[OAK_MAX_RANK];
    threadgroup float tgOptTdAbs[OAK_MAX_RANK];
    threadgroup float tgLambda[OAK_MAX_RANK];
    threadgroup float tgScalars[16];

    uint nf = 0u;
    float obsRate = track_rate(ctx.stepCount, ctx.baseO);
    for (uint o = tid; o < ctx.baseO; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nf += 1u; continue; }
        ctx.observationMean[o] += obsRate * (x - ctx.observationMean[o]);
        ctx.obsSM[o] += obsRate * (x * x - ctx.obsSM[o]);
        float s = standardized_obs(ctx, o);
        for (uint m = 1u; m <= ctx.traceLevels; ++m) {
            uint slot = (m - 1u) * ctx.baseO + o;
            float rate = 1.0f / float(1u << m);
            ctx.obsTraceMulti[slot] += rate * (s - ctx.obsTraceMulti[slot]);
        }
    }
    if (nf > 0u) ctx.counters[OAK_CNT_OBS_NF] += nf;
    threadgroup_barrier(mem_flags::mem_device);

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float raw = 0.0f;
        if (i < ctx.anchorCount) {
            uint width = max(linear_input_width(ctx), 1u);
            if (width <= ctx.anchorCount) {
                raw = linear_input(ctx, i % width);
            } else {
                for (uint p = 0u; p < width; ++p) {
                    float sign = (hash_uniform(i, p, 0x515701u) >= 0.5f) ? 1.0f : -1.0f;
                    raw += sign * linear_input(ctx, p);
                }
                raw *= rsqrt(float(width));
            }
        } else {
            uint pa = ctx.featureParentA[i] % ctx.R;
            uint pb = ctx.featureParentB[i] % ctx.R;
            float a = prev[pa], b = prev[pb];
            uint kind = ctx.featureKind[i] % 3u;
            if (kind == 0u) raw = a * b;
            else if (kind == 1u) raw = (a + b) * 0.7071067811865475f;
            else raw = a * (b - a);
        }
        uint age = ctx.featureAge[i];
        float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
        ctx.featureMean[i] += rate * (raw - ctx.featureMean[i]);
        ctx.featureSM[i] += rate * (raw * raw - ctx.featureSM[i]);
        float var = max(ctx.featureSM[i] - ctx.featureMean[i] * ctx.featureMean[i], 0.0f);
        float ridge = 1.0f / (float(age) + 1.0f);
        float x = (raw - ctx.featureMean[i]) * rsqrt(var + ridge);
        if (!isfinite(x)) {
            x = prev[i];
            ctx.counters[OAK_CNT_OBS_NF] += 1u;
        }
        curr[i] = x;
    }
    threadgroup_barrier(mem_flags::mem_device);
    oak_keyboard_predict_body(ctx, tid, tgQNext, tgG, tgCont, tgStopGap, tgTerm,
                              tgOptTdAbs, tgLambda, tgScalars);
}

kernel void oak_plan(device float *params [[buffer(0)]],
                     device float *stateArena [[buffer(1)]],
                     device uint *uintArena [[buffer(2)]],
                     device KernelScalars *scalars [[buffer(3)]],
                     device KernelStats *stats [[buffer(4)]],
                     constant KernelConfig &cfg [[buffer(5)]],
                     uint j [[thread_position_in_threadgroup]],
                     uint k [[threadgroup_position_in_grid]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint slot = k;
    if (ctx.K <= ctx.pageWidth) {
        k = slot;
    } else if (slot == OAK_SEARCH_SLOT_HIGH) k = ctx.highModelOption[0] % ctx.K;
    else if (slot == OAK_SEARCH_SLOT_STALE) k = ctx.staleModelOption[0] % ctx.K;
    else k = ctx.stepCount % max(ctx.K, 1u);
    if (k >= ctx.K) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    bool active = j < ctx.R;
    float cont = ctx.continuation[k];

    threadgroup float tgNHat[OAK_MAX_RANK];
    threadgroup float tgModelErr[OAK_MAX_RANK];
    threadgroup float tgOneAbs[OAK_MAX_RANK];
    threadgroup float tgOneVal[OAK_MAX_RANK];

    float nHat = 0.0f, nTd = 0.0f;
    if (active) {
        uint base = k * ctx.R * ctx.R + j * ctx.R;
        float nPrev = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.optTransition[base + i];
            nHat = fma(w, curr[i], nHat);
            nPrev = fma(w, prev[i], nPrev);
        }
        float target = (1.0f - cont) * curr[j] + cont * nHat;
        nTd = target - nPrev;
        if (!isfinite(nTd)) nTd = 0.0f;
        ctx.nTdMat[k * ctx.R + j] = nTd;
        float vRel = isfinite(ctx.valueW[j]) ? ctx.valueW[j] * ctx.valueW[j] : 0.0f;
        tgNHat[j] = nHat;
        tgModelErr[j] = vRel * nTd * nTd;

        if (k == 0u) {
            uint oneW = ctx.R + ctx.D;
            float pred = 0.0f;
            for (uint i = 0u; i < oneW; ++i)
                pred = fma(ctx.oneTransition[j * oneW + i], one_input(ctx, prev, i), pred);
            float e = curr[j] - pred;
            if (!isfinite(e)) e = 0.0f;
            ctx.oneTransitionTd[j] = e;
            tgOneAbs[j] = fabs(e);
            tgOneVal[j] = fabs(e * ctx.valueW[j]);
        } else {
            tgOneAbs[j] = 0.0f;
            tgOneVal[j] = 0.0f;
        }
    } else {
        tgNHat[j] = 0.0f; tgModelErr[j] = 0.0f; tgOneAbs[j] = 0.0f; tgOneVal[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j == 0u) {
        float maxQ = -FLT_MAX;
        for (uint kk = 0u; kk < ctx.K; ++kk) {
            float q = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) q = fma(ctx.optValue[kk * ctx.R + i], tgNHat[i], q);
            if (isfinite(q) && q > maxQ) maxQ = q;
        }
        if (!(maxQ > -FLT_MAX)) maxQ = 0.0f;
        float nErr = 0.0f, oneAbs = 0.0f, oneVal = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            nErr += tgModelErr[i];
            oneAbs += tgOneAbs[i];
            oneVal += tgOneVal[i];
        }
        float rHat = 0.0f, rPrev = 0.0f, lRaw = 0.0f, lPrev = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            uint idx = k * ctx.R + i;
            rHat = fma(ctx.optReward[idx], curr[i], rHat);
            rPrev = fma(ctx.optReward[idx], prev[i], rPrev);
            lRaw = fma(ctx.optDuration[idx], curr[i], lRaw);
            lPrev = fma(ctx.optDuration[idx], prev[i], lPrev);
        }
        float lHat = decode_duration(lRaw);
        float lPrevHat = decode_duration(lPrev);
        float rTd = ctx.reward[0] + cont * rHat - rPrev;
        float dTd = 1.0f + cont * lHat - lPrevHat;
        if (!isfinite(rTd)) rTd = 0.0f;
        if (!isfinite(dTd)) dTd = 0.0f;
        ctx.optRewardTd[k] = rTd;
        ctx.durationTd[k] = dTd;
        ctx.durationHat[k] = lHat;
        ctx.optionDiffTd[k] = rTd - ctx.scalars->rho * dTd;
        float Q = rHat - ctx.scalars->rho * lHat + maxQ;
        if (!isfinite(Q)) { Q = -FLT_MAX; ctx.counters[OAK_CNT_PLAN_NF] += 1u; }
        ctx.qOption[k] = Q;
        float modelErr = nErr + rTd * rTd + (ctx.scalars->rho * dTd) * (ctx.scalars->rho * dTd);
        ctx.prioOpt[k] = isfinite(modelErr) ? sqrt(max(modelErr, 0.0f)) * (1.0f + float(ctx.modelAge[k]) / float(max(ctx.K, 1u))) : 0.0f;

        if (k == 0u) {
            uint oneW = ctx.R + ctx.D;
            float rPred = 0.0f;
            for (uint i = 0u; i < oneW; ++i) rPred = fma(ctx.oneReward[i], one_input(ctx, prev, i), rPred);
            float rE = ctx.reward[0] - rPred;
            ctx.oneRewardTd[0] = isfinite(rE) ? rE : 0.0f;
            ctx.oneStateTd[0] = oneAbs * ctx.invR;
            ctx.oneValueWeightedError[0] = oneVal * ctx.invR;
        }
    }
}

kernel void oak_plan_reduce(device float *params [[buffer(0)]],
                            device float *stateArena [[buffer(1)]],
                            device uint *uintArena [[buffer(2)]],
                            device KernelScalars *scalars [[buffer(3)]],
                            device KernelStats *stats [[buffer(4)]],
                            constant KernelConfig &cfg [[buffer(5)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    if (tid != 0u) return;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float vCurr = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);

    float bestQ = -FLT_MAX, prioSum = 0.0f, diffSum = 0.0f, durSum = 0.0f;
    float maxPrio = -FLT_MAX;
    uint high = 0u, stale = 0u, bestK = 0u, maxAge = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float q = ctx.qOption[k];
        if (isfinite(q) && q > bestQ) { bestQ = q; bestK = k; }
        float p = ctx.prioOpt[k];
        if (!isfinite(p)) p = 0.0f;
        prioSum += p;
        if (p > maxPrio) { maxPrio = p; high = k; }
        if (ctx.modelAge[k] >= maxAge) { maxAge = ctx.modelAge[k]; stale = k; }
        diffSum += fabs(ctx.optionDiffTd[k]);
        durSum += ctx.durationHat[k];
    }
    float advRaw = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
    ctx.planAdv[0] = isfinite(advRaw) ? (advRaw / (1.0f + fabs(advRaw))) : 0.0f;
    ctx.highModelOption[0] = high;
    ctx.staleModelOption[0] = stale;
    for (uint j = 0u; j < ctx.R; ++j) {
        float n = 0.0f;
        uint base = bestK * ctx.R * ctx.R + j * ctx.R;
        for (uint i = 0u; i < ctx.R; ++i) n = fma(ctx.optTransition[base + i], curr[i], n);
        ctx.bestModelNext[j] = isfinite(n) ? n : curr[j];
    }
    float invK = ctx.invK;
    ctx.scalars->meanPrio = prioSum * invK;
    ctx.stats->optionDurationMean = durSum * invK;
    ctx.stats->optionDiffTdMean = diffSum * invK;
    float entropy = 0.0f;
    if (prioSum > 0.0f) {
        for (uint k = 0u; k < ctx.K; ++k) {
            float p = max(ctx.prioOpt[k], 0.0f) / prioSum;
            if (p > 0.0f) entropy -= p * log(p);
        }
    }
    ctx.stats->modelPriorityEntropy = (ctx.K > 1u) ? entropy / log(float(ctx.K)) : 0.0f;
    ctx.stats->modelUpdateRate = (ctx.K <= ctx.pageWidth)
        ? 1.0f : min(1.0f, float(OAK_SEARCH_SLOT_COUNT) * invK);
    ctx.stats->stalePageAgeMean = float(maxAge);
    ctx.stats->modelQueueCoverage = (maxAge <= ctx.K) ? 1.0f : (float(ctx.K) / float(maxAge));
}

kernel void oak_learn_heads(device float *params [[buffer(0)]],
                            device float *stateArena [[buffer(1)]],
                            device uint *uintArena [[buffer(2)]],
                            device KernelScalars *scalars [[buffer(3)]],
                            device KernelStats *stats [[buffer(4)]],
                            constant KernelConfig &cfg [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    uint nVal = ctx.R;
    uint nGRate = ctx.anchorCount;
    uint nGValue = ctx.anchorCount * ctx.R;
    uint nOV = ctx.K * ctx.R;
    uint nPsi = ctx.K * ctx.R;
    uint nCross = ctx.K * ctx.R;
    uint nPi = ctx.K * ctx.D * ctx.R;
    uint nOR = ctx.K * ctx.R;
    uint nOT = ctx.K * ctx.R * ctx.R;
    uint nOD = ctx.K * ctx.R;
    uint oneW = ctx.R + ctx.D;
    uint nOneR = oneW;
    uint nOneT = ctx.R * oneW;
    uint total = nVal + nGRate + nGValue + nOV + nPsi + nCross + nPi + nOR + nOT
               + nOD + nOneR + nOneT;
    if (gid >= total) return;

    if (gid < nVal) {
        uint i = gid;
        float w = ctx.valueW[i];
        float delta = ctx.td[0] * prev[i] + ctx.planAdv[0] * curr[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta);
        return;
    }
    gid -= nVal;
    if (gid < nGRate) {
        uint g = gid;
        float d = ctx.gvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        float rate = ctx.gvfRate[g] + ctx.invR * d;
        ctx.gvfRate[g] = safe_w(ctx.gvfRate[g], rate);
        float u = ctx.gvfUtility[g] + track_rate(ctx.stepCount, ctx.R) * (fabs(d) - ctx.gvfUtility[g]);
        ctx.gvfUtility[g] = safe_w(ctx.gvfUtility[g], u);
        return;
    }
    gid -= nGRate;
    if (gid < nGValue) {
        uint g = gid / ctx.R, i = gid % ctx.R;
        float d = ctx.gvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        float w = ctx.gvfValue[g * ctx.R + i];
        float denom = ctx.featureTraceDenom[0];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.gvfValue[g * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * d * prev[i] * scale);
        return;
    }
    gid -= nGValue;
    if (gid < nOV) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) { rho = 0.0f; ctx.counters[OAK_CNT_IMPORTANCE_NF] += 1u; }
        float w = ctx.optValue[k * ctx.R + i];
        float denom = ctx.optValueDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.optValue[k * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * rho * ctx.optValueTd[k] * prev[i] * scale);
        return;
    }
    gid -= nOV;
    if (gid < nPsi) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float w = ctx.psi[k * ctx.R + i];
        float denom = ctx.optValueDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.psi[k * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * rho * ctx.tdPsi[k] * prev[i] * scale);
        return;
    }
    gid -= nPsi;
    if (gid < nCross) {
        uint target = ctx.stepCount % max(ctx.K, 1u);
        uint k = gid / ctx.R, i = gid % ctx.R;
        uint base = (k * ctx.K + target) * ctx.R;
        float td = ctx.crossTd[k];
        if (!isfinite(td)) td = 0.0f;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float w = ctx.psiCross[base + i];
        ctx.psiCross[base + i] = safe_w(w, w + ctx.alpha[i] * rho * td * prev[i]);
        return;
    }
    gid -= nCross;
    if (gid < nPi) {
        uint k = gid / (ctx.D * ctx.R);
        uint rem = gid - k * ctx.D * ctx.R;
        uint d = rem / ctx.R, i = rem % ctx.R;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float relevance = ctx.alpha[i] * float(max(ctx.R, 1u));
        float w = ctx.optPolicy[(k * ctx.D + d) * ctx.R + i];
        float grad = rho * ctx.scoreGrad[d] * prev[i];
        ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] =
            safe_w(w, w + ctx.alphaPi * relevance * ctx.tdPsi[k] * grad);
        return;
    }
    gid -= nPi;
    if (gid < nOR) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.optValueDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optReward[k * ctx.R + i];
        ctx.optReward[k * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * rho * ctx.optRewardTd[k] * prev[i] * scale);
        return;
    }
    gid -= nOR;
    if (gid < nOT) {
        uint k = gid / (ctx.R * ctx.R);
        uint rem = gid - k * ctx.R * ctx.R;
        uint j = rem / ctx.R, i = rem % ctx.R;
        uint diag = ctx.stepCount % max(ctx.K, 1u);
        if (!(ctx.K <= ctx.pageWidth || k == ctx.highModelOption[0] || k == ctx.staleModelOption[0] || k == diag)) return;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.optValueDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i];
        ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] =
            safe_w(w, w + ctx.alpha[i] * rho * ctx.nTdMat[k * ctx.R + j] * prev[i] * scale);
        return;
    }
    gid -= nOT;
    if (gid < nOD) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.optValueDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optDuration[k * ctx.R + i];
        ctx.optDuration[k * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * rho * ctx.durationTd[k] * prev[i] * scale);
        return;
    }
    gid -= nOD;
    if (gid < nOneR) {
        uint i = gid;
        float x = one_input(ctx, prev, i);
        float a = (i < ctx.R) ? ctx.alpha[i] : ctx.invOneInputWidth;
        float w = ctx.oneReward[i];
        float denom = ctx.oneStepDenom[0];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.oneReward[i] = safe_w(w, w + a * ctx.oneRewardTd[0] * x * scale);
        return;
    }
    gid -= nOneR;
    if (gid < nOneT) {
        uint j = gid / oneW, i = gid % oneW;
        float x = one_input(ctx, prev, i);
        float a = (i < ctx.R) ? ctx.alpha[i] : ctx.invOneInputWidth;
        float w = ctx.oneTransition[j * oneW + i];
        float denom = ctx.oneStepDenom[0];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.oneTransition[j * oneW + i] = safe_w(w, w + a * ctx.oneTransitionTd[j] * x * scale);
    }
}

kernel void oak_action_emit(device float *params [[buffer(0)]],
                            device float *stateArena [[buffer(1)]],
                            device uint *uintArena [[buffer(2)]],
                            device KernelScalars *scalars [[buffer(3)]],
                            device KernelStats *stats [[buffer(4)]],
                            constant KernelConfig &cfg [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    if (gid >= ctx.K * ctx.D) return;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    uint k = gid / ctx.D, d = gid % ctx.D;
    float mu = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) mu = fma(ctx.optPolicy[(k * ctx.D + d) * ctx.R + i], curr[i], mu);
    if (!isfinite(mu)) mu = ctx.optionPolicyMean[k * ctx.D + d];
        ctx.optionPolicyMean[k * ctx.D + d] = mu;
    if (k == (ctx.selectedOption[0] % ctx.K)) ctx.policyMean[d] = mu;
}

inline void emit_action(Ctx ctx, device const float *curr, uint tid, uint tgW) {
    uint selected = ctx.selectedOption[0] % ctx.K;
    for (uint gid = tid; gid < ctx.K * ctx.D; gid += tgW) {
        uint k = gid / ctx.D, d = gid % ctx.D;
        float mu = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            mu = fma(ctx.optPolicy[(k * ctx.D + d) * ctx.R + i], curr[i], mu);
        if (!isfinite(mu)) mu = ctx.optionPolicyMean[k * ctx.D + d];
        ctx.optionPolicyMean[k * ctx.D + d] = mu;
        if (k == selected) ctx.policyMean[d] = mu;
    }
}

kernel void oak_replace(device float *params [[buffer(0)]],
                        device float *stateArena [[buffer(1)]],
                        device uint *uintArena [[buffer(2)]],
                        device KernelScalars *scalars [[buffer(3)]],
                        device KernelStats *stats [[buffer(4)]],
                        constant KernelConfig &cfg [[buffer(5)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgW [[threads_per_threadgroup]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    emit_action(ctx, curr, tid, tgW);
    threadgroup_barrier(mem_flags::mem_device);
    float invT = track_rate(ctx.stepCount, ctx.R);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float g = fabs(ctx.td[0] * prev[i]);
        if (i < ctx.K) g += fabs(ctx.tdPsi[i] * prev[i]) + fabs(ctx.optValueTd[i] * prev[i]);
        ctx.utilFeat[i] += invT * (g - ctx.utilFeat[i]);
        float aOld = ctx.alpha[i];
        float hOld = ctx.hMeta[i];
        float dw = aOld * ctx.td[0] * prev[i];
        float nb = ctx.beta[i] + ctx.metaEta * ctx.td[0] * prev[i] * hOld;
        if (!isfinite(nb)) nb = ctx.beta[i];
        nb = clamp(nb, ctx.betaFloor, ctx.betaCap);
        ctx.hMeta[i] = (1.0f - ctx.invR) * hOld + dw;
        ctx.beta[i] = nb;
        ctx.alpha[i] = exp(nb);
        if (isfinite(ctx.valueW[i]) && ctx.valueW[i] > ctx.bonusW[i]) ctx.bonusW[i] = ctx.valueW[i];
    }
    for (uint k = tid; k < ctx.K; k += tgW) {
        float u = ctx.tdPsi[k] * ctx.tdPsi[k] + ctx.optValueTd[k] * ctx.optValueTd[k] + ctx.prioOpt[k];
        ctx.utilOpt[k] += invT * (u - ctx.utilOpt[k]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid != 0u) return;
    float sum2 = 0.0f, sum4 = 0.0f, corrSum = 0.0f, corrMax = 0.0f;
    uint never = 0u;
    for (uint i = 0u; i < ctx.R; ++i) {
        float x = curr[i];
        float x2 = x * x;
        sum2 += x2; sum4 += x2 * x2;
        if (ctx.featureEverUsed[i] == 0u) never += 1u;
        uint pa = ctx.featureParentA[i] % ctx.R;
        float c = fabs(x * curr[pa]);
        corrSum += c; corrMax = max(corrMax, c);
    }
    float effRank = (sum4 > 0.0f) ? (sum2 * sum2 / sum4) : 0.0f;
    ctx.stats->stateEffectiveRank = effRank;
    ctx.stats->featureEffectiveRank = effRank;
    ctx.stats->featureCorrMean = corrSum / float(max(ctx.R, 1u));
    ctx.stats->featureMaxCorr = corrMax;
    ctx.stats->neverUsedFeatureCount = float(never);

    float logSum = 0.0f, logSq = 0.0f, wSum = 0.0f, wSq = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float lr = ctx.importanceLog[k];
        float w = ctx.importance[k];
        if (!isfinite(lr) || !isfinite(w)) { lr = 0.0f; w = 0.0f; ctx.counters[OAK_CNT_IMPORTANCE_NF] += 1u; }
        logSum += lr; logSq += lr * lr; wSum += w; wSq += w * w;
    }
    float invK = ctx.invK;
    float lm = logSum * invK;
    ctx.stats->importanceLogMean = lm;
    ctx.stats->importanceLogVar = max(0.0f, logSq * invK - lm * lm);
    ctx.stats->importanceEss = (wSq > 0.0f) ? ((wSum * wSum) / (float(ctx.K) * wSq)) : 0.0f;

    uint eligible = 0u, worst = ctx.anchorCount;
    float worstScore = FLT_MAX;
    float worstUtil = 0.0f, worstRed = 0.0f, worstDownstream = 0.0f;
    float eligibleUtilSum = 0.0f, eligibleRedSum = 0.0f, eligibleDownstreamSum = 0.0f;
    float ageSum = 0.0f, ageN = 0.0f;
    for (uint i = ctx.anchorCount; i < ctx.R; ++i) {
        if (ctx.featureAge[i] >= ctx.R) {
            eligible += 1u;
            uint pa = ctx.featureParentA[i] % ctx.R;
            float util = ctx.utilFeat[i];
            if (!isfinite(util)) util = 0.0f;
            float red = fabs(curr[i] * curr[pa]);
            if (!isfinite(red)) red = 0.0f;
            float modelUse = (i < ctx.K) ? (ctx.utilOpt[i] + fabs(ctx.qOption[i]) + ctx.prioOpt[i]) : 0.0f;
            if (!isfinite(modelUse)) modelUse = 0.0f;
            float gvfUse = 0.0f;
            for (uint g = 0u; g < max(ctx.anchorCount, 1u); ++g) {
                float w = ctx.gvfValue[g * ctx.R + i];
                if (isfinite(w)) gvfUse += fabs(w);
            }
            gvfUse *= 1.0f / float(max(ctx.anchorCount, 1u));
            float downstreamUse = modelUse + gvfUse + fabs(ctx.valueW[i]);
            if (!isfinite(downstreamUse)) downstreamUse = 0.0f;
            eligibleUtilSum += util;
            eligibleRedSum += red;
            eligibleDownstreamSum += downstreamUse;
            float score = util + downstreamUse - red;
            if (!isfinite(score)) score = -FLT_MAX;
            if (score < worstScore) {
                worstScore = score; worst = i;
                worstUtil = util; worstRed = red; worstDownstream = downstreamUse;
            }
        }
        ageSum += float(ctx.featureAge[i]); ageN += 1.0f;
    }
    ctx.stats->eligibleFeatureCount = float(eligible);
    ctx.stats->eligibleOptionCount = float(ctx.K);
    ctx.stats->candidateSurvivalMean = (ageN > 0.0f) ? ageSum / ageN : 0.0f;

    float parentTotal = 0.0f, parentEntropy = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) parentTotal += float(ctx.parentUseCount[i]);
    if (parentTotal > 0.0f) {
        for (uint i = 0u; i < ctx.R; ++i) {
            float p = float(ctx.parentUseCount[i]) / parentTotal;
            if (p > 0.0f) parentEntropy -= p * log(p);
        }
        parentEntropy = parentEntropy / log(float(ctx.R));
    }
    ctx.stats->parentReuseEntropy = parentEntropy;

    float gvfRateAbs = 0.0f, gvfTdAbs = 0.0f, gvfValueAbs = 0.0f, gvfUtility = 0.0f;
    uint G = max(ctx.anchorCount, 1u);
    for (uint g = 0u; g < G; ++g) {
        gvfRateAbs += fabs(ctx.gvfRate[g]);
        gvfTdAbs += fabs(ctx.gvfTd[g]);
        gvfUtility += ctx.gvfUtility[g];
        float h = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) h = fma(ctx.gvfValue[g * ctx.R + i], curr[i], h);
        gvfValueAbs += fabs(h);
    }
    float invG = 1.0f / float(G);
    ctx.stats->gvfCount = float(G);
    ctx.stats->gvfRateMean = gvfRateAbs * invG;
    ctx.stats->gvfTdMean = gvfTdAbs * invG;
    ctx.stats->gvfDifferentialValueMean = gvfValueAbs * invG;
    ctx.stats->gvfUtilityMean = gvfUtility * invG;

    uint high = ctx.highModelOption[0] % ctx.K;
    uint stale = ctx.staleModelOption[0] % ctx.K;
    uint diag = ctx.stepCount % max(ctx.K, 1u);
    uint maxAge = 0u;
    float ageModelSum = 0.0f, prioSum = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        if (ctx.modelAge[k] < 0xffffffffu) ctx.modelAge[k] += 1u;
        if (k == high || k == stale || k == diag) ctx.modelAge[k] = 0u;
        maxAge = max(maxAge, ctx.modelAge[k]);
        ageModelSum += float(ctx.modelAge[k]);
        prioSum += max(ctx.prioOpt[k], 0.0f);
        if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;
    }
    for (uint i = 0u; i < ctx.R; ++i) if (ctx.featureAge[i] < 0xffffffffu) ctx.featureAge[i] += 1u;
    ctx.stats->modelUncertaintyMean = prioSum * invK;
    ctx.stats->stalePageAgeMean = ageModelSum * invK;
    ctx.stats->modelQueueCoverage = (maxAge <= ctx.K) ? 1.0f : float(ctx.K) / float(maxAge);

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRed = eligibleRedSum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtil < meanEligibleUtil;
    bool highRedundancy = worstRed > meanEligibleRed;
    bool lowDownstreamUse = worstDownstream < meanEligibleDownstream;
    bool unusedCandidate = (worst < ctx.R) && (ctx.featureEverUsed[worst] == 0u);
    bool structurallyReplaceable = (eligible > 1u) &&
        ((lowUtility && highRedundancy && lowDownstreamUse) ||
         (unusedCandidate && (lowUtility || highRedundancy)));
    bool doReplace = structurallyReplaceable && ((ctx.stepCount % max(ctx.R, 1u)) == 0u);
    if (doReplace && worst < ctx.R) {
        uint pa = 0u, pb = (ctx.R > 1u) ? 1u : 0u;
        float massA = 0.0f, massB = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst) continue;
            float u = max(ctx.utilFeat[i], 0.0f);
            if (isfinite(u)) massA += u;
        }
        float drawA = hash_uniform(worst, ctx.featureBirthCount[worst], ctx.stepCount) * massA;
        float accA = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst) continue;
            float u = max(ctx.utilFeat[i], 0.0f);
            if (!isfinite(u)) u = 0.0f;
            accA += u;
            if (accA >= drawA) { pa = i; break; }
        }
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst || i == pa) continue;
            float decor = 1.0f / (1.0f + fabs(curr[i] * curr[pa]));
            float u = max(ctx.utilFeat[i], 0.0f) * decor;
            if (isfinite(u)) massB += u;
        }
        float drawB = hash_uniform(worst, ctx.featureBirthCount[worst], ctx.stepCount ^ 0x9e3779b9u) * massB;
        float accB = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst || i == pa) continue;
            float decor = 1.0f / (1.0f + fabs(curr[i] * curr[pa]));
            float u = max(ctx.utilFeat[i], 0.0f) * decor;
            if (!isfinite(u)) u = 0.0f;
            accB += u;
            if (accB >= drawB) { pb = i; break; }
        }
        uint birth = ctx.featureBirthCount[worst] + 1u;
        ctx.featureBirthCount[worst] = birth;
        ctx.featureParentA[worst] = pa;
        ctx.featureParentB[worst] = pb;
        ctx.featureKind[worst] = birth % 3u;
        ctx.parentUseCount[pa] += 1u;
        ctx.parentUseCount[pb] += 1u;
        float transfer = 0.5f * (ctx.utilFeat[pa] + ctx.utilFeat[pb]);
        ctx.stats->utilityTransferMean = isfinite(transfer) ? transfer : 0.0f;
        ctx.featureAge[worst] = 0u;
        ctx.featureEverUsed[worst] = 1u;
        ctx.featureMean[worst] = 0.0f; ctx.featureSM[worst] = 0.0f;
        ctx.utilFeat[worst] = 0.0f; ctx.valueW[worst] = 0.0f; ctx.bonusW[worst] = 0.0f;
        if (worst < ctx.anchorCount) {
            ctx.gvfRate[worst] = 0.0f; ctx.gvfTd[worst] = 0.0f; ctx.gvfUtility[worst] = 0.0f;
        }
        ctx.alpha[worst] = ctx.invR;
        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f;
        uint k = worst % ctx.K;
        ctx.optAge[k] = 0u; ctx.optFeature[k] = worst; ctx.utilOpt[k] = float(ctx.R);
        for (uint i = 0u; i < ctx.R; ++i) {
            ctx.psi[k * ctx.R + i] = 0.0f;
            ctx.optValue[k * ctx.R + i] = 0.0f;
            ctx.optReward[k * ctx.R + i] = 0.0f;
            ctx.optDuration[k * ctx.R + i] = 0.0f;
            if (worst < ctx.anchorCount) ctx.gvfValue[worst * ctx.R + i] = 0.0f;
            if (i < ctx.anchorCount) ctx.gvfValue[i * ctx.R + worst] = 0.0f;
            for (uint d = 0u; d < ctx.D; ++d) ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] = 0.0f;
            for (uint j = 0u; j < ctx.R; ++j) ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] = 0.0f;
            for (uint ci = 0u; ci < ctx.K; ++ci) {
                ctx.psiCross[(k * ctx.K + ci) * ctx.R + i] = 0.0f;
                ctx.psiCross[(ci * ctx.K + k) * ctx.R + i] = 0.0f;
            }
        }
        float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
        ctx.psi[k * ctx.R + worst] = invSqrtR;
        ctx.optValue[k * ctx.R + worst] = invSqrtR;
        ctx.psiCross[(k * ctx.K + worst) * ctx.R + worst] = invSqrtR;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
        ctx.replacementAgeMean[0] += invT * (float(ctx.R) - ctx.replacementAgeMean[0]);
    }

    float bonusGap = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) bonusGap += fabs(ctx.bonusW[i] - ctx.valueW[i]);
    ctx.stats->rrBonusMean = bonusGap / float(max(ctx.R, 1u));
    ctx.stats->replacementAgeMean = ctx.replacementAgeMean[0];
    ctx.stats->replacementCommitRate =
        float(ctx.counters[OAK_CNT_REINIT_FEAT] + ctx.counters[OAK_CNT_REINIT_OPT]) /
        max(1.0f, 2.0f * float(ctx.stepCount + 1u));
    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount = float(ctx.counters[OAK_CNT_TDPSI_NF] + ctx.counters[OAK_CNT_IMPORTANCE_NF]);
    ctx.stats->oakOptionReinitCount = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);
    ctx.stats->oneStepRewardTdMean = fabs(ctx.oneRewardTd[0]);
    ctx.stats->oneStepStateTdMean = ctx.oneStateTd[0];
    ctx.stats->oneStepValueWeightedError = ctx.oneValueWeightedError[0];
    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
