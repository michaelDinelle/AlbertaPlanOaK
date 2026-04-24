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
    uint pAuxGvfRate, pAuxGvfValue;
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
    uint sShadowRewardScore, sShadowTaskScore;
    uint sGvfTd, sGvfUtility, sFeatureTraceDenom;
    uint sAuxGvfTd, sAuxGvfUtility;
    uint sRhoQTd, sModelExpectedImprovement;
    uint sReplacementAgeMean, sReplacementWouldCommitRate;
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
    uint uGvfAge, uBadFeatureAge;
    uint uAuxGvfAge;
    uint deepDiagnostics;
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
    float optionControlRateMean, optionControlRateVar, optionControlRateTdMean, optionControlRateAgreement;
    float actorUpdateNormMean, actorUpdateNormVar, actorTdMean, actorImportanceWeightedTdMean;
    float actorSupportEntropy;
    float shadowRewardSelectedOption, shadowTaskSelectedOption;
    float shadowRewardAgreement, shadowTaskAgreement;
    float shadowRewardGpiMargin, shadowTaskGpiMargin;
    float gvfPageCoverage, gvfRateVar, gvfTerminationTdMean, gvfRewardTdMean, gvfUtilityRankMean;
    float modelPageCoverage, modelPageMaxAge, modelPriorityMax, modelExpectedImprovementMean;
    float replacementWouldCommitRate, badFeatureAgeMean, badFeatureAgeMax, replacementSurvivalMean;
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
    uint deepDiagnostics;
    float rho, alphaPi, meanPrio;
    float invR, invK, invOneInputWidth, metaEta, betaCap, betaFloor;
    device float *valueW, *psi, *optPolicy, *optReward, *optTransition, *optDuration;
    device float *optValue, *psiCross, *oneReward, *oneTransition;
    device float *gvfRate, *gvfValue, *auxGvfRate, *auxGvfValue;
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
    device float *lambda, *keyboardScore, *bestModelNext, *shadowRewardScore, *shadowTaskScore;
    device float *gvfTd, *gvfUtility, *featureTraceDenom, *auxGvfTd, *auxGvfUtility;
    device float *rhoQTd, *modelExpectedImprovement;
    device float *replacementAgeMean, *replacementWouldCommitRate;
    device uint *counters, *optFeature, *featureAge, *optAge, *featureEverUsed;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount, *parentUseCount;
    device uint *selectedOption, *highModelOption, *staleModelOption, *keyboardAge, *modelAge;
    device uint *gvfAge, *badFeatureAge, *auxGvfAge;
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
    c.deepDiagnostics = cfg.deepDiagnostics;
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
    c.auxGvfRate = params + cfg.pAuxGvfRate; c.auxGvfValue = params + cfg.pAuxGvfValue;
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
    c.shadowRewardScore = stateArena + cfg.sShadowRewardScore;
    c.shadowTaskScore = stateArena + cfg.sShadowTaskScore;
    c.gvfTd = stateArena + cfg.sGvfTd; c.gvfUtility = stateArena + cfg.sGvfUtility;
    c.featureTraceDenom = stateArena + cfg.sFeatureTraceDenom;
    c.auxGvfTd = stateArena + cfg.sAuxGvfTd; c.auxGvfUtility = stateArena + cfg.sAuxGvfUtility;
    c.rhoQTd = stateArena + cfg.sRhoQTd;
    c.modelExpectedImprovement = stateArena + cfg.sModelExpectedImprovement;
    c.replacementAgeMean = stateArena + cfg.sReplacementAgeMean;
    c.replacementWouldCommitRate = stateArena + cfg.sReplacementWouldCommitRate;
    c.counters = uintArena + cfg.uCounters; c.optFeature = uintArena + cfg.uOptFeature;
    c.featureAge = uintArena + cfg.uFeatureAge; c.optAge = uintArena + cfg.uOptAge;
    c.featureEverUsed = uintArena + cfg.uFeatureEverUsed;
    c.featureParentA = uintArena + cfg.uFeatureParentA; c.featureParentB = uintArena + cfg.uFeatureParentB;
    c.featureKind = uintArena + cfg.uFeatureKind; c.featureBirthCount = uintArena + cfg.uFeatureBirthCount;
    c.parentUseCount = uintArena + cfg.uParentUseCount; c.selectedOption = uintArena + cfg.uSelectedOption;
    c.highModelOption = uintArena + cfg.uHighModelOption; c.staleModelOption = uintArena + cfg.uStaleModelOption;
    c.keyboardAge = uintArena + cfg.uKeyboardAge; c.modelAge = uintArena + cfg.uModelAge;
    c.gvfAge = uintArena + cfg.uGvfAge; c.badFeatureAge = uintArena + cfg.uBadFeatureAge;
    c.auxGvfAge = uintArena + cfg.uAuxGvfAge;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline uint linear_input_width(Ctx ctx) { return (ctx.traceLevels + 1u) * ctx.baseO + ctx.D + 1u; }
inline uint gvf_count(Ctx ctx) { return max(ctx.anchorCount, 1u); }
inline uint aux_gvf_count(Ctx ctx) { return ctx.R + ctx.K + 3u; }
inline uint aux_gvf_page_count(Ctx ctx) {
    return (aux_gvf_count(ctx) + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
}
inline uint aux_gvf_page_base(Ctx ctx) {
    return (ctx.stepCount % max(aux_gvf_page_count(ctx), 1u)) * ctx.pageWidth;
}
inline uint gvf_page_count(Ctx ctx) {
    return (gvf_count(ctx) + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
}
inline uint gvf_page_base(Ctx ctx) {
    return (ctx.stepCount % max(gvf_page_count(ctx), 1u)) * ctx.pageWidth;
}

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
    if (gvf_count(ctx) <= ctx.anchorCount) {
        uint kind = g % 4u;
        if (kind == 0u) return curr[g % ctx.R];
        if (kind == 1u) {
            if (ctx.D == 0u) return 0.0f;
            return ctx.actionSignal[g % ctx.D];
        }
        if (kind == 2u) return ctx.reward[0] - ctx.scalars->rho;
        float e = ctx.oneValueWeightedError[0];
        return isfinite(e) ? fabs(e) : 0.0f;
    }
    if (g < ctx.R) return curr[g];
    uint r = g - ctx.R;
    if (r == 0u) return ctx.reward[0] - ctx.scalars->rho;
    if (r == 1u) {
        float e = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) e = fma(ctx.actionSignal[d], ctx.actionSignal[d], e);
        return e / float(max(ctx.D, 1u));
    }
    r -= 2u;
    if (r < ctx.K) return 1.0f - ctx.continuation[r];
    float m = ctx.oneValueWeightedError[0];
    return isfinite(m) ? fabs(m) : 0.0f;
}

inline float gvf_cumulant_tg(Ctx ctx, threadgroup const float *curr, uint g) {
    if (gvf_count(ctx) <= ctx.anchorCount) {
        uint kind = g % 4u;
        if (kind == 0u) return curr[g % ctx.R];
        if (kind == 1u) {
            if (ctx.D == 0u) return 0.0f;
            return ctx.actionSignal[g % ctx.D];
        }
        if (kind == 2u) return ctx.reward[0] - ctx.scalars->rho;
        float e = ctx.oneValueWeightedError[0];
        return isfinite(e) ? fabs(e) : 0.0f;
    }
    if (g < ctx.R) return curr[g];
    uint r = g - ctx.R;
    if (r == 0u) return ctx.reward[0] - ctx.scalars->rho;
    if (r == 1u) {
        float e = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) e = fma(ctx.actionSignal[d], ctx.actionSignal[d], e);
        return e / float(max(ctx.D, 1u));
    }
    r -= 2u;
    if (r < ctx.K) return 1.0f - ctx.continuation[r];
    float m = ctx.oneValueWeightedError[0];
    return isfinite(m) ? fabs(m) : 0.0f;
}

inline float aux_gvf_cumulant(Ctx ctx, device const float *curr, uint g) {
    if (g < ctx.R) return curr[g];
    uint r = g - ctx.R;
    if (r == 0u) return ctx.reward[0] - ctx.scalars->rho;
    if (r == 1u) {
        float e = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) e = fma(ctx.actionSignal[d], ctx.actionSignal[d], e);
        return e / float(max(ctx.D, 1u));
    }
    r -= 2u;
    if (r < ctx.K) return 1.0f - ctx.continuation[r];
    float m = ctx.oneValueWeightedError[0];
    return isfinite(m) ? fabs(m) : 0.0f;
}

inline void oak_keyboard_predict_body(Ctx ctx, uint tid,
                                      threadgroup float *tgPrev,
                                      threadgroup float *tgCurr,
                                      threadgroup float *tgQNext,
                                      threadgroup float *tgG,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgStopGap,
                                      threadgroup float *tgTerm,
                                      threadgroup float *tgOptTdAbs,
                                      threadgroup float *tgLambda,
                                      threadgroup float *tgScalars) {
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float vPrev = 0.0f, vCurr = 0.0f;
    if (tid == 0u) {
        for (uint i = 0u; i < ctx.R; ++i) {
            vPrev = fma(ctx.valueW[i], tgPrev[i], vPrev);
            vCurr = fma(ctx.valueW[i], tgCurr[i], vCurr);
        }
        tgScalars[0] = vPrev;
        tgScalars[1] = vCurr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vPrev = tgScalars[0];
    vCurr = tgScalars[1];

    if (tid < ctx.K) {
        float q = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) q = fma(ctx.optValue[tid * ctx.R + i], tgCurr[i], q);
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
    if (tid == 0u) {
        float alphaTraceNorm = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float a = ctx.alpha[i];
            float x = tgPrev[i];
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
            psiPrev = fma(ctx.psi[base + i], tgPrev[i], psiPrev);
            psiNext = fma(ctx.psi[base + i], tgCurr[i], psiNext);
            qPrev = fma(ctx.optValue[base + i], tgPrev[i], qPrev);
            qNext = fma(ctx.optValue[base + i], tgCurr[i], qNext);
        }
        float zStop = vCurr - ctx.valueW[feat] * tgCurr[feat] + ctx.bonusW[feat] * tgCurr[feat];
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
        if (ctx.deepDiagnostics != 0u) {
            float dQOption = (ctx.reward[0] - ctx.rhoOpt[tid]) + u - qPrev;
            if (!isfinite(dQOption)) dQOption = 0.0f;
            ctx.rhoQTd[tid] = dQOption;
            float rhoRate = 1.0f / float(max(ctx.optAge[tid] + 1u, ctx.R));
            float rhoQNew = ctx.rhoOpt[tid] + rho * dQOption * rhoRate;
            ctx.rhoOpt[tid] = safe_w(ctx.rhoOpt[tid], rhoQNew);
        }
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
            crossPrev = fma(w, tgPrev[i], crossPrev);
            crossNext = fma(w, tgCurr[i], crossNext);
        }
        float z = vCurr - ctx.valueW[target] * tgCurr[target] + ctx.bonusW[target] * tgCurr[target];
        float beta = (z >= crossNext) ? 1.0f : 0.0f;
        float td = (ctx.reward[0] - ctx.scalars->rho) + (1.0f - beta) * crossNext + beta * z - crossPrev;
        ctx.crossTd[tid] = isfinite(td) ? td : 0.0f;
    }
    if (tid < ctx.pageWidth) {
        uint g = gvf_page_base(ctx) + tid;
        if (g < gvf_count(ctx)) {
            float hPrev = 0.0f, hNext = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) {
                float w = ctx.gvfValue[g * ctx.R + i];
                hPrev = fma(w, tgPrev[i], hPrev);
                hNext = fma(w, tgCurr[i], hNext);
            }
            float d = gvf_cumulant_tg(ctx, tgCurr, g) - ctx.gvfRate[g] + hNext - hPrev;
            ctx.gvfTd[g] = isfinite(d) ? d : 0.0f;
        }
        uint ag = aux_gvf_page_base(ctx) + tid;
        if (ctx.deepDiagnostics != 0u && ag < aux_gvf_count(ctx)) {
            float hPrev = 0.0f, hNext = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) {
                float w = ctx.auxGvfValue[ag * ctx.R + i];
	                hPrev = fma(w, tgPrev[i], hPrev);
	                hNext = fma(w, tgCurr[i], hNext);
            }
            float d = aux_gvf_cumulant(ctx, curr, ag) - ctx.auxGvfRate[ag] + hNext - hPrev;
            ctx.auxGvfTd[ag] = isfinite(d) ? d : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gapAbsSum = 0.0f, psiAbsSum = 0.0f, zAbsSum = 0.0f;
        float rateSum = 0.0f, rateSq = 0.0f, rateTdAbs = 0.0f, rateAgree = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            gapAbsSum += fabs(tgStopGap[k]);
            psiAbsSum += tgG[k];
            zAbsSum += tgLambda[k];
            float rq = ctx.rhoOpt[k];
            if (!isfinite(rq)) rq = 0.0f;
            rateSum += rq;
            rateSq += rq * rq;
            if (ctx.deepDiagnostics != 0u) {
                float rg = ctx.optValueTd[k], ro = ctx.rhoQTd[k];
                if (!isfinite(rg)) rg = 0.0f;
                if (!isfinite(ro)) ro = 0.0f;
                rateTdAbs += fabs(ro);
                rateAgree += (rg * ro >= 0.0f) ? 1.0f : 0.0f;
            }
        }
        float invK = ctx.invK;
        ctx.stats->psiStopGapAbsMean = gapAbsSum * invK;
        ctx.stats->psiAbsMean = psiAbsSum * invK;
        ctx.stats->rrStopAbsMean = zAbsSum * invK;
        ctx.stats->valueAbsMean = fabs(vCurr);
        float rateMean = rateSum * invK;
        ctx.stats->optionControlRateMean = (ctx.deepDiagnostics != 0u) ? rateMean : 0.0f;
        ctx.stats->optionControlRateVar = (ctx.deepDiagnostics != 0u)
            ? max(0.0f, rateSq * invK - rateMean * rateMean) : 0.0f;
        ctx.stats->optionControlRateTdMean = (ctx.deepDiagnostics != 0u) ? (rateTdAbs * invK) : 0.0f;
        ctx.stats->optionControlRateAgreement = (ctx.deepDiagnostics != 0u) ? (rateAgree * invK) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        float plan = ctx.valueW[tid] * (ctx.bestModelNext[tid] - tgCurr[tid]);
        float rr = (ctx.bonusW[tid] - ctx.valueW[tid]) * tgCurr[tid];
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
        if (ctx.deepDiagnostics != 0u) {
            float rewardL1 = 0.0f, taskL1 = 0.0f;
            for (uint i = 0u; i < ctx.K; ++i) {
                float wr = (i < ctx.R) ? ctx.oneReward[i] : 0.0f;
                float wt = (i < ctx.R) ? (ctx.bonusW[i] - ctx.valueW[i]) : 0.0f;
                if (!isfinite(wr)) wr = 0.0f;
                if (!isfinite(wt)) wt = 0.0f;
                tgQNext[i] = wr;
                tgG[i] = wt;
                rewardL1 += fabs(wr);
                taskL1 += fabs(wt);
            }
            for (uint i = 0u; i < ctx.K; ++i) {
                tgQNext[i] = (rewardL1 > 0.0f) ? (tgQNext[i] / rewardL1) : invK;
                tgG[i] = (taskL1 > 0.0f) ? (tgG[i] / taskL1) : invK;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pageCount = (ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
    uint page = ctx.stepCount % max(pageCount, 1u);
    uint cand = page * ctx.pageWidth + tid;
    if (tid < ctx.pageWidth && cand < ctx.K) {
        float G = 0.0f, GR = 0.0f, GT = 0.0f;
        for (uint key = 0u; key < ctx.K; ++key) {
            float lam = tgLambda[key];
            uint base = (cand * ctx.K + key) * ctx.R;
            float p = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) p = fma(ctx.psiCross[base + i], tgCurr[i], p);
            if (lam > 0.0f) G = fma(lam, p, G);
            if (ctx.deepDiagnostics != 0u) {
                GR = fma(tgQNext[key], p, GR);
                GT = fma(tgG[key], p, GT);
            }
        }
        ctx.keyboardScore[cand] = isfinite(G) ? G : -FLT_MAX;
        if (ctx.deepDiagnostics != 0u) {
            ctx.shadowRewardScore[cand] = isfinite(GR) ? GR : -FLT_MAX;
            ctx.shadowTaskScore[cand] = isfinite(GT) ? GT : -FLT_MAX;
        }
        ctx.keyboardAge[cand] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint best = 0u, bestReward = 0u, bestTask = 0u;
        float bestG = -FLT_MAX, secondG = -FLT_MAX;
        float bestRewardG = -FLT_MAX, secondRewardG = -FLT_MAX;
        float bestTaskG = -FLT_MAX, secondTaskG = -FLT_MAX;
        float stopGapSum = 0.0f, termSum = 0.0f, optTdAbs = 0.0f;
        float diagAbs = 0.0f, scoreAbs = 0.0f, ageSum = 0.0f, covered = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            ctx.chordPrev[k] = ctx.chord[k];
            stopGapSum += tgStopGap[k];
            termSum += tgTerm[k];
            optTdAbs += tgOptTdAbs[k];
            float G = ctx.keyboardScore[k];
            scoreAbs += fabs(G);
            if (G > bestG) { secondG = bestG; bestG = G; best = k; }
            else if (G > secondG) secondG = G;
            if (ctx.deepDiagnostics != 0u) {
                uint diagBase = (k * ctx.K + k) * ctx.R;
                float pDiag = 0.0f;
                for (uint i = 0u; i < ctx.R; ++i) pDiag = fma(ctx.psiCross[diagBase + i], tgCurr[i], pDiag);
                diagAbs += fabs(pDiag);
                float rG = ctx.shadowRewardScore[k];
                if (rG > bestRewardG) { secondRewardG = bestRewardG; bestRewardG = rG; bestReward = k; }
                else if (rG > secondRewardG) secondRewardG = rG;
                float tG = ctx.shadowTaskScore[k];
                if (tG > bestTaskG) { secondTaskG = bestTaskG; bestTaskG = tG; bestTask = k; }
                else if (tG > secondTaskG) secondTaskG = tG;
            }
            uint age = ctx.keyboardAge[k];
            ageSum += float(age);
            if (age <= pageCount) covered += 1.0f;
        }
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        uint selected = (ctx.continuation[prevSel] > 0.0f) ? prevSel : best;
        ctx.selectedOption[0] = selected;
        for (uint k = 0u; k < ctx.K; ++k) ctx.chord[k] = tgLambda[k];

        float invK = ctx.invK;
        ctx.stats->chordEffectiveK = tgScalars[3];
        ctx.stats->psiStopGapMean = stopGapSum * invK;
        ctx.stats->terminationRate = termSum * invK;
        ctx.stats->optValueTdMean = optTdAbs * invK;
        ctx.stats->crossPsiDiagRatio = (ctx.deepDiagnostics != 0u && diagAbs + scoreAbs > 0.0f)
            ? (diagAbs / (diagAbs + scoreAbs)) : 0.0f;
        ctx.stats->keyboardGpiMargin = isfinite(bestG - secondG) ? (bestG - secondG) : 0.0f;
        ctx.stats->keyboardSelectedOption = float(selected);
        ctx.stats->keyboardDesign = 0.0f;
        ctx.stats->lambdaEffectiveRank = tgScalars[3];
        ctx.stats->lambdaPlanMass = tgScalars[4];
        ctx.stats->lambdaRrMass = tgScalars[5];
        ctx.stats->lambdaUtilityMass = tgScalars[6];
        ctx.stats->keyboardCacheCoverage = covered * invK;
        ctx.stats->keyboardScoreStaleness = ageSum * invK;
        ctx.stats->shadowRewardSelectedOption = (ctx.deepDiagnostics != 0u) ? float(bestReward) : 0.0f;
        ctx.stats->shadowTaskSelectedOption = (ctx.deepDiagnostics != 0u) ? float(bestTask) : 0.0f;
        ctx.stats->shadowRewardAgreement = (ctx.deepDiagnostics != 0u && bestReward == selected) ? 1.0f : 0.0f;
        ctx.stats->shadowTaskAgreement = (ctx.deepDiagnostics != 0u && bestTask == selected) ? 1.0f : 0.0f;
        ctx.stats->shadowRewardGpiMargin = (ctx.deepDiagnostics != 0u && isfinite(bestRewardG - secondRewardG))
            ? (bestRewardG - secondRewardG) : 0.0f;
        ctx.stats->shadowTaskGpiMargin = (ctx.deepDiagnostics != 0u && isfinite(bestTaskG - secondTaskG))
            ? (bestTaskG - secondTaskG) : 0.0f;
        ctx.stats->optionCount = float(ctx.K);
        ctx.stats->optionPageCount = float((ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u));
        ctx.stats->linearAnchorCount = float(ctx.anchorCount);
        ctx.stats->constructedFeatureCount = float(ctx.R - min(ctx.anchorCount, ctx.R));

        float ss = 0.0f, gg = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) ss = fma(tgCurr[i], tgCurr[i], ss);
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
    threadgroup float tgPrevState[OAK_MAX_RANK];
    threadgroup float tgCurrState[OAK_MAX_RANK];

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
    for (uint i = tid; i < ctx.R; i += tgW) {
        tgPrevState[i] = prev[i];
        tgCurrState[i] = curr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    oak_keyboard_predict_body(ctx, tid, tgPrevState, tgCurrState, tgQNext, tgG, tgCont, tgStopGap, tgTerm,
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
    } else {
        uint pageBase = (ctx.highModelOption[0] / max(ctx.pageWidth, 1u)) * ctx.pageWidth;
        k = pageBase + slot;
    }
    if (k >= ctx.K) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    bool active = j < ctx.R;
    float cont = ctx.continuation[k];

    threadgroup float tgNHat[OAK_MAX_RANK];
    threadgroup float tgModelErr[OAK_MAX_RANK];
    threadgroup float tgOneAbs[OAK_MAX_RANK];
    threadgroup float tgOneVal[OAK_MAX_RANK];
    threadgroup float tgQVals[OAK_MAX_RANK];

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

        if (ctx.deepDiagnostics != 0u && k == 0u) {
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

    if (j < ctx.K) {
        float q = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) q = fma(ctx.optValue[j * ctx.R + i], tgNHat[i], q);
        tgQVals[j] = isfinite(q) ? q : -FLT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j == 0u) {
        float maxQ = -FLT_MAX;
        for (uint kk = 0u; kk < ctx.K; ++kk) maxQ = max(maxQ, tgQVals[kk]);
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
        float expected = fabs(rTd) + fabs(dTd) + sqrt(max(nErr, 0.0f));
        ctx.modelExpectedImprovement[k] = isfinite(expected) ? expected : 0.0f;

        if (ctx.deepDiagnostics != 0u && k == 0u) {
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
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    threadgroup uint tgBestK[1];
    if (tid == 0u) {
        float vCurr = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);

        float bestQ = -FLT_MAX, prioSum = 0.0f, diffSum = 0.0f, durSum = 0.0f;
        float maxPrio = -FLT_MAX, expectedSum = 0.0f;
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
            float e = ctx.modelExpectedImprovement[k];
            if (!isfinite(e)) e = 0.0f;
            expectedSum += e;
        }
        uint pageCount = (ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
        uint highPage = 0u, stalePage = 0u;
        float highPagePriority = -FLT_MAX;
        uint stalePageAge = 0u;
        for (uint p = 0u; p < pageCount; ++p) {
            float pp = -FLT_MAX;
            uint pa = 0u;
            uint start = p * ctx.pageWidth;
            uint end = min(start + ctx.pageWidth, ctx.K);
            for (uint k = start; k < end; ++k) {
                pp = max(pp, ctx.modelExpectedImprovement[k]);
                pa = max(pa, ctx.modelAge[k]);
            }
            if (pp > highPagePriority) { highPagePriority = pp; highPage = p; }
            if (pa >= stalePageAge) { stalePageAge = pa; stalePage = p; }
        }
        uint selectedPage = (stalePageAge > pageCount) ? stalePage : highPage;
        float advRaw = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
        ctx.planAdv[0] = isfinite(advRaw) ? (advRaw / (1.0f + fabs(advRaw))) : 0.0f;
        ctx.highModelOption[0] = (ctx.K <= ctx.pageWidth) ? high : min(selectedPage * ctx.pageWidth, ctx.K - 1u);
        ctx.staleModelOption[0] = (ctx.K <= ctx.pageWidth) ? stale : min(stalePage * ctx.pageWidth, ctx.K - 1u);
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
            ? 1.0f : min(1.0f, float(ctx.pageWidth) * invK);
        ctx.stats->stalePageAgeMean = float(maxAge);
        ctx.stats->modelQueueCoverage = (maxAge <= ctx.K) ? 1.0f : (float(ctx.K) / float(maxAge));
        ctx.stats->modelPageCoverage = (ctx.K <= ctx.pageWidth)
            ? 1.0f
            : ((stalePageAge <= pageCount) ? 1.0f : (float(pageCount) / float(stalePageAge)));
        ctx.stats->modelPageMaxAge = (ctx.K <= ctx.pageWidth) ? 0.0f : float(stalePageAge);
        ctx.stats->modelPriorityMax = maxPrio > -FLT_MAX ? maxPrio : 0.0f;
        ctx.stats->modelExpectedImprovementMean = expectedSum * invK;
        tgBestK[0] = bestK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bestK = tgBestK[0];
    for (uint j = tid; j < ctx.R; j += ctx.pageWidth) {
        float n = 0.0f;
        uint base = bestK * ctx.R * ctx.R + j * ctx.R;
        for (uint i = 0u; i < ctx.R; ++i) n = fma(ctx.optTransition[base + i], curr[i], n);
        ctx.bestModelNext[j] = isfinite(n) ? n : curr[j];
    }
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
    uint G = gvf_count(ctx);
    uint gvfBase = gvf_page_base(ctx);
    uint nGRate = min(ctx.pageWidth, G);
    uint nGValue = nGRate * ctx.R;
    uint AG = aux_gvf_count(ctx);
    uint auxGvfBase = aux_gvf_page_base(ctx);
    uint nAuxGRate = (ctx.deepDiagnostics != 0u) ? min(ctx.pageWidth, AG) : 0u;
    uint nAuxGValue = nAuxGRate * ctx.R;
    uint nOV = ctx.K * ctx.R;
    uint nPsi = ctx.K * ctx.R;
    uint nCross = ctx.K * ctx.R;
    uint nPi = ctx.K * ctx.D * ctx.R;
    uint nOR = ctx.K * ctx.R;
    uint nOT = ctx.K * ctx.R * ctx.R;
    uint nOD = ctx.K * ctx.R;
    uint oneW = ctx.R + ctx.D;
    uint nOneR = (ctx.deepDiagnostics != 0u) ? oneW : 0u;
    uint nOneT = (ctx.deepDiagnostics != 0u) ? (ctx.R * oneW) : 0u;
    uint total = nVal + nGRate + nGValue + nOV + nPsi + nCross + nPi + nOR + nOT
               + nOD + nOneR + nOneT + nAuxGRate + nAuxGValue;
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
        uint g = gvfBase + gid;
        if (g >= G) return;
        float d = ctx.gvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        float alphaRate = 1.0f / float(max(ctx.gvfAge[g] + 1u, ctx.R));
        float rate = ctx.gvfRate[g] + alphaRate * d;
        ctx.gvfRate[g] = safe_w(ctx.gvfRate[g], rate);
        float u = ctx.gvfUtility[g] + alphaRate * (fabs(d) - ctx.gvfUtility[g]);
        ctx.gvfUtility[g] = safe_w(ctx.gvfUtility[g], u);
        return;
    }
    gid -= nGRate;
    if (gid < nGValue) {
        uint g = gvfBase + gid / ctx.R, i = gid % ctx.R;
        if (g >= G) return;
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
        uint selectedPage = ctx.highModelOption[0] / max(ctx.pageWidth, 1u);
        if (!(ctx.K <= ctx.pageWidth || (k / max(ctx.pageWidth, 1u)) == selectedPage)) return;
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
        return;
    }
    gid -= nOneT;
    if (gid < nAuxGRate) {
        uint g = auxGvfBase + gid;
        if (g >= AG) return;
        float d = ctx.auxGvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        float alphaRate = 1.0f / float(max(ctx.auxGvfAge[g] + 1u, ctx.R));
        float rate = ctx.auxGvfRate[g] + alphaRate * d;
        ctx.auxGvfRate[g] = safe_w(ctx.auxGvfRate[g], rate);
        float u = ctx.auxGvfUtility[g] + alphaRate * (fabs(d) - ctx.auxGvfUtility[g]);
        ctx.auxGvfUtility[g] = safe_w(ctx.auxGvfUtility[g], u);
        return;
    }
    gid -= nAuxGRate;
    if (gid < nAuxGValue) {
        uint g = auxGvfBase + gid / ctx.R, i = gid % ctx.R;
        if (g >= AG) return;
        float d = ctx.auxGvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        float w = ctx.auxGvfValue[g * ctx.R + i];
        float denom = ctx.featureTraceDenom[0];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.auxGvfValue[g * ctx.R + i] = safe_w(w, w + ctx.alpha[i] * d * prev[i] * scale);
    }
}

inline void emit_action_grid(Ctx ctx, device const float *curr, uint group, uint tid, uint tgW) {
    uint gid = group * tgW + tid;
    if (gid >= ctx.K * ctx.D) return;
    uint selected = ctx.selectedOption[0] % ctx.K;
    uint k = gid / ctx.D, d = gid % ctx.D;
    float mu = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i)
        mu = fma(ctx.optPolicy[(k * ctx.D + d) * ctx.R + i], curr[i], mu);
    if (!isfinite(mu)) mu = ctx.optionPolicyMean[k * ctx.D + d];
    ctx.optionPolicyMean[k * ctx.D + d] = mu;
    if (k == selected) ctx.policyMean[d] = mu;
}

kernel void oak_replace(device float *params [[buffer(0)]],
                        device float *stateArena [[buffer(1)]],
                        device uint *uintArena [[buffer(2)]],
                        device KernelScalars *scalars [[buffer(3)]],
                        device KernelStats *stats [[buffer(4)]],
                        constant KernelConfig &cfg [[buffer(5)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgW [[threads_per_threadgroup]]) {
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    emit_action_grid(ctx, curr, group, tid, tgW);
    if (group != 0u) return;
    threadgroup float tgCurrState[OAK_MAX_RANK];
    for (uint i = tid; i < ctx.R; i += tgW) tgCurrState[i] = curr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
        float x = tgCurrState[i];
        float x2 = x * x;
        sum2 += x2; sum4 += x2 * x2;
        if (ctx.featureEverUsed[i] == 0u) never += 1u;
        uint pa = ctx.featureParentA[i] % ctx.R;
        float c = fabs(x * tgCurrState[pa]);
        corrSum += c; corrMax = max(corrMax, c);
    }
    float effRank = (sum4 > 0.0f) ? (sum2 * sum2 / sum4) : 0.0f;
    ctx.stats->stateEffectiveRank = effRank;
    ctx.stats->featureEffectiveRank = effRank;
    ctx.stats->featureCorrMean = corrSum / float(max(ctx.R, 1u));
    ctx.stats->featureMaxCorr = corrMax;
    ctx.stats->neverUsedFeatureCount = float(never);

	    float logSum = 0.0f, logSq = 0.0f, wSum = 0.0f, wSq = 0.0f;
	    float actorTdSum = 0.0f, actorIwTdSum = 0.0f;
	    for (uint k = 0u; k < ctx.K; ++k) {
	        float lr = ctx.importanceLog[k];
	        float w = ctx.importance[k];
	        if (!isfinite(lr) || !isfinite(w)) { lr = 0.0f; w = 0.0f; ctx.counters[OAK_CNT_IMPORTANCE_NF] += 1u; }
	        logSum += lr; logSq += lr * lr; wSum += w; wSq += w * w;
	        float d = ctx.tdPsi[k];
	        if (!isfinite(d)) d = 0.0f;
	        actorTdSum += d;
	        actorIwTdSum += w * d;
	    }
	    float invK = ctx.invK;
	    float lm = logSum * invK;
	    ctx.stats->importanceLogMean = lm;
	    ctx.stats->importanceLogVar = max(0.0f, logSq * invK - lm * lm);
	    ctx.stats->importanceEss = (wSq > 0.0f) ? ((wSum * wSum) / (float(ctx.K) * wSq)) : 0.0f;
	    ctx.stats->actorTdMean = actorTdSum * invK;
	    ctx.stats->actorImportanceWeightedTdMean = actorIwTdSum * invK;
	    float actorEntropy = 0.0f;
	    if (wSum > 0.0f) {
	        for (uint k = 0u; k < ctx.K; ++k) {
	            float p = max(ctx.importance[k], 0.0f) / wSum;
	            if (p > 0.0f) actorEntropy -= p * log(p);
	        }
		    }
		    ctx.stats->actorSupportEntropy = (ctx.K > 1u) ? actorEntropy / log(float(ctx.K)) : 0.0f;
		    if (ctx.deepDiagnostics != 0u) {
		        float sgNorm = 0.0f, relNorm = 0.0f;
		        for (uint d = 0u; d < ctx.D; ++d) {
		            float g = ctx.scoreGrad[d];
		            if (isfinite(g)) sgNorm = fma(g, g, sgNorm);
		        }
		        for (uint i = 0u; i < ctx.R; ++i) {
		            float rel = ctx.alpha[i] * float(ctx.R) * prev[i];
		            if (isfinite(rel)) relNorm = fma(rel, rel, relNorm);
		        }
		        float actorUpdateSum = 0.0f, actorUpdateSq = 0.0f;
		        float actorBase = ctx.alphaPi * sqrt(max(sgNorm * relNorm, 0.0f));
		        for (uint k = 0u; k < ctx.K; ++k) {
		            float u = fabs(actorBase * ctx.importance[k] * ctx.tdPsi[k]);
		            if (!isfinite(u)) u = 0.0f;
		            actorUpdateSum += u;
		            actorUpdateSq += u * u;
		        }
		        float actorUpdateMean = actorUpdateSum * invK;
		        ctx.stats->actorUpdateNormMean = actorUpdateMean;
		        ctx.stats->actorUpdateNormVar = max(0.0f, actorUpdateSq * invK - actorUpdateMean * actorUpdateMean);
		    } else {
		        ctx.stats->actorUpdateNormMean = 0.0f;
		        ctx.stats->actorUpdateNormVar = 0.0f;
		    }

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
	            float red = fabs(tgCurrState[i] * tgCurrState[pa]);
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

		    if (ctx.deepDiagnostics != 0u) {
		        float gvfRateAbs = 0.0f, gvfRateSum = 0.0f, gvfRateSq = 0.0f, gvfTdAbs = 0.0f;
		        float gvfValueAbs = 0.0f, gvfUtility = 0.0f;
		        float gvfTermTd = 0.0f, gvfRewardTd = 0.0f;
		        uint gvfAgeMax = 0u;
		        uint G = max(aux_gvf_count(ctx), 1u);
		        uint gBaseStats = aux_gvf_page_base(ctx);
		        float valueN = 0.0f;
		        for (uint g = 0u; g < G; ++g) {
		            float rate = ctx.auxGvfRate[g];
		            if (!isfinite(rate)) rate = 0.0f;
		            gvfRateAbs += fabs(rate);
		            gvfRateSum += rate;
		            gvfRateSq += rate * rate;
		            gvfTdAbs += fabs(ctx.auxGvfTd[g]);
		            gvfUtility += ctx.auxGvfUtility[g];
		            if (g == gBaseStats) {
		                float h = 0.0f;
		            for (uint i = 0u; i < ctx.R; ++i) h = fma(ctx.auxGvfValue[g * ctx.R + i], tgCurrState[i], h);
		                gvfValueAbs += fabs(h);
		                valueN += 1.0f;
		            }
		            gvfAgeMax = max(gvfAgeMax, ctx.auxGvfAge[g]);
		        }
		        if (ctx.R < G) gvfRewardTd = fabs(ctx.auxGvfTd[ctx.R]);
		        uint termStart = ctx.R + 2u;
		        for (uint k = 0u; k < ctx.K && termStart + k < G; ++k) {
		            gvfTermTd += fabs(ctx.auxGvfTd[termStart + k]);
		        }
		        float invG = 1.0f / float(G);
		        ctx.stats->gvfCount = float(G);
		        ctx.stats->gvfRateMean = gvfRateAbs * invG;
		        float signedRateMean = gvfRateSum * invG;
		        ctx.stats->gvfRateVar = max(0.0f, gvfRateSq * invG - signedRateMean * signedRateMean);
		        ctx.stats->gvfTdMean = gvfTdAbs * invG;
		        ctx.stats->gvfDifferentialValueMean = gvfValueAbs / max(valueN, 1.0f);
		        ctx.stats->gvfUtilityMean = gvfUtility * invG;
		        ctx.stats->gvfRewardTdMean = gvfRewardTd;
		        ctx.stats->gvfTerminationTdMean = gvfTermTd / float(max(ctx.K, 1u));
		        ctx.stats->gvfUtilityRankMean = ctx.stats->gvfUtilityMean;
		        uint gPages = max(aux_gvf_page_count(ctx), 1u);
		        ctx.stats->gvfPageCoverage = (gvfAgeMax <= gPages) ? 1.0f : (float(gPages) / float(gvfAgeMax));
		    } else {
		        ctx.stats->gvfCount = float(gvf_count(ctx));
		        ctx.stats->gvfRateVar = 0.0f;
		        ctx.stats->gvfRewardTdMean = 0.0f;
		        ctx.stats->gvfTerminationTdMean = 0.0f;
		        ctx.stats->gvfUtilityRankMean = ctx.stats->gvfUtilityMean;
		        ctx.stats->gvfPageCoverage = 0.0f;
		    }

	    uint maxAge = 0u;
	    float ageModelSum = 0.0f, prioSum = 0.0f;
	    uint high = ctx.highModelOption[0] % ctx.K;
	    uint stale = ctx.staleModelOption[0] % ctx.K;
	    uint diag = ctx.stepCount % max(ctx.K, 1u);
	    uint selectedPage = ctx.highModelOption[0] / max(ctx.pageWidth, 1u);
	    for (uint k = 0u; k < ctx.K; ++k) {
	        if (ctx.modelAge[k] < 0xffffffffu) ctx.modelAge[k] += 1u;
	        bool planned = (ctx.K <= ctx.pageWidth)
	            ? (k == high || k == stale || k == diag)
	            : ((k / max(ctx.pageWidth, 1u)) == selectedPage);
	        if (planned) ctx.modelAge[k] = 0u;
	        maxAge = max(maxAge, ctx.modelAge[k]);
	        ageModelSum += float(ctx.modelAge[k]);
	        prioSum += max(ctx.prioOpt[k], 0.0f);
	        if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;
	    }
	    for (uint i = 0u; i < ctx.R; ++i) if (ctx.featureAge[i] < 0xffffffffu) ctx.featureAge[i] += 1u;
	    uint gBase = gvf_page_base(ctx);
	    uint GAll = gvf_count(ctx);
	    for (uint g = 0u; g < GAll; ++g) {
	        if (ctx.gvfAge[g] < 0xffffffffu) ctx.gvfAge[g] += 1u;
	        if (g >= gBase && g < min(gBase + ctx.pageWidth, GAll)) ctx.gvfAge[g] = 0u;
	    }
		    if (ctx.deepDiagnostics != 0u) {
		        uint agBase = aux_gvf_page_base(ctx);
		        uint AGAll = aux_gvf_count(ctx);
		        for (uint g = 0u; g < AGAll; ++g) {
		            if (ctx.auxGvfAge[g] < 0xffffffffu) ctx.auxGvfAge[g] += 1u;
		            if (g >= agBase && g < min(agBase + ctx.pageWidth, AGAll)) ctx.auxGvfAge[g] = 0u;
		        }
		    }
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
		    if (ctx.deepDiagnostics != 0u) {
		        float badAgeSum = 0.0f, badAgeMax = 0.0f, survivalSum = 0.0f, survivalN = 0.0f;
		        uint wouldCommit = 0u;
		        for (uint i = ctx.anchorCount; i < ctx.R; ++i) {
		            bool bad = false;
		            if (ctx.featureAge[i] >= ctx.R && eligible > 1u) {
		                uint pa = ctx.featureParentA[i] % ctx.R;
		                float util = ctx.utilFeat[i];
		                if (!isfinite(util)) util = 0.0f;
		                float red = fabs(tgCurrState[i] * tgCurrState[pa]);
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
		                bad = (util < meanEligibleUtil) && (red > meanEligibleRed) && (downstreamUse < meanEligibleDownstream);
		            }
		            if (bad) {
		                if (ctx.badFeatureAge[i] < 0xffffffffu) ctx.badFeatureAge[i] += 1u;
		            } else {
		                ctx.badFeatureAge[i] = 0u;
		            }
		            badAgeSum += float(ctx.badFeatureAge[i]);
		            badAgeMax = max(badAgeMax, float(ctx.badFeatureAge[i]));
		            survivalSum += float(ctx.featureAge[i]);
		            survivalN += 1.0f;
		        }
		        if (worst < ctx.R && ctx.badFeatureAge[worst] >= ctx.R) wouldCommit = 1u;
		        ctx.replacementWouldCommitRate[0] += invT * (float(wouldCommit) - ctx.replacementWouldCommitRate[0]);
		        ctx.stats->replacementWouldCommitRate = ctx.replacementWouldCommitRate[0];
		        ctx.stats->badFeatureAgeMean = (survivalN > 0.0f) ? (badAgeSum / survivalN) : 0.0f;
		        ctx.stats->badFeatureAgeMax = badAgeMax;
		        ctx.stats->replacementSurvivalMean = (survivalN > 0.0f) ? (survivalSum / survivalN) : 0.0f;
		    } else {
		        ctx.stats->replacementWouldCommitRate = 0.0f;
		        ctx.stats->badFeatureAgeMean = 0.0f;
		        ctx.stats->badFeatureAgeMax = 0.0f;
		        ctx.stats->replacementSurvivalMean = ctx.stats->candidateSurvivalMean;
		    }
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
	            float decor = 1.0f / (1.0f + fabs(tgCurrState[i] * tgCurrState[pa]));
            float u = max(ctx.utilFeat[i], 0.0f) * decor;
            if (isfinite(u)) massB += u;
        }
        float drawB = hash_uniform(worst, ctx.featureBirthCount[worst], ctx.stepCount ^ 0x9e3779b9u) * massB;
        float accB = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst || i == pa) continue;
	            float decor = 1.0f / (1.0f + fabs(tgCurrState[i] * tgCurrState[pa]));
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
	        ctx.badFeatureAge[worst] = 0u;
	        ctx.featureEverUsed[worst] = 1u;
	        ctx.featureMean[worst] = 0.0f; ctx.featureSM[worst] = 0.0f;
	        ctx.utilFeat[worst] = 0.0f; ctx.valueW[worst] = 0.0f; ctx.bonusW[worst] = 0.0f;
	        uint GAll = gvf_count(ctx);
	        if (worst < GAll) {
	            ctx.gvfRate[worst] = 0.0f; ctx.gvfTd[worst] = 0.0f; ctx.gvfUtility[worst] = 0.0f;
	            ctx.gvfAge[worst] = 0u;
	        }
		        uint AGAll = (ctx.deepDiagnostics != 0u) ? aux_gvf_count(ctx) : 0u;
		        if (worst < AGAll) {
		            ctx.auxGvfRate[worst] = 0.0f; ctx.auxGvfTd[worst] = 0.0f; ctx.auxGvfUtility[worst] = 0.0f;
		            ctx.auxGvfAge[worst] = 0u;
		        }
	        ctx.alpha[worst] = ctx.invR;
	        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f;
	        uint k = worst % ctx.K;
	        ctx.optAge[k] = 0u; ctx.optFeature[k] = worst; ctx.utilOpt[k] = float(ctx.R);
	        ctx.rhoOpt[k] = 0.0f; ctx.rhoQTd[k] = 0.0f;
	        uint termG = ctx.R + 2u + k;
	        if (termG < GAll) {
	            ctx.gvfRate[termG] = 0.0f; ctx.gvfTd[termG] = 0.0f; ctx.gvfUtility[termG] = 0.0f;
	            ctx.gvfAge[termG] = 0u;
	        }
	        if (termG < AGAll) {
	            ctx.auxGvfRate[termG] = 0.0f; ctx.auxGvfTd[termG] = 0.0f; ctx.auxGvfUtility[termG] = 0.0f;
	            ctx.auxGvfAge[termG] = 0u;
	        }
	        for (uint i = 0u; i < ctx.R; ++i) {
	            ctx.psi[k * ctx.R + i] = 0.0f;
	            ctx.optValue[k * ctx.R + i] = 0.0f;
	            ctx.optReward[k * ctx.R + i] = 0.0f;
	            ctx.optDuration[k * ctx.R + i] = 0.0f;
	            if (worst < GAll) ctx.gvfValue[worst * ctx.R + i] = 0.0f;
	            if (termG < GAll) ctx.gvfValue[termG * ctx.R + i] = 0.0f;
	            if (worst < AGAll) ctx.auxGvfValue[worst * ctx.R + i] = 0.0f;
	            if (termG < AGAll) ctx.auxGvfValue[termG * ctx.R + i] = 0.0f;
	            for (uint d = 0u; d < ctx.D; ++d) ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] = 0.0f;
	            for (uint j = 0u; j < ctx.R; ++j) ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] = 0.0f;
	            for (uint ci = 0u; ci < ctx.K; ++ci) {
	                ctx.psiCross[(k * ctx.K + ci) * ctx.R + i] = 0.0f;
	                ctx.psiCross[(ci * ctx.K + k) * ctx.R + i] = 0.0f;
	            }
	        }
	        for (uint g = 0u; g < GAll; ++g) ctx.gvfValue[g * ctx.R + worst] = 0.0f;
	        for (uint g = 0u; g < AGAll; ++g) ctx.auxGvfValue[g * ctx.R + worst] = 0.0f;
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
