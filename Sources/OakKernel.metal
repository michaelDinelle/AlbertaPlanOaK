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
    uint obsDim, rank, optionCount, actionParamCount;
    uint obsTraceLevels;
    uint linearAnchorCount;
    uint optionPageWidth;
    float invRank, invOptionCount;
    float invOneInputWidth, metaEta;
    uint pValue, pPsi;
    uint pOptPolicy, pOptReward, pOptTransition, pOptDuration;
    uint pPsiCross;
    uint pOneReward;
    uint pGvfRate, pGvfValue;
    uint sObservation, sObsMean, sObsSM;
    uint sState;
    uint sPolicyMean, sActionSignal, sScoreGrad;
    uint sOptionScoreGrad;
    uint sOptionPolicyMean;
    uint sImportance;
    uint sReward, sTd, sTdPsi;
    uint sUtilFeat, sUtilOpt;
    uint sBonusW;
    uint sFeatureMean, sFeatureSM;
    uint sAlpha, sBeta, sH;
    uint sGradEnergy;
    uint sAlphaPsi, sBetaPsi, sHPsi, sGradEnergyPsi;
    uint sAlphaModel, sBetaModel, sHModel, sGradEnergyModel;
    uint sAlphaGvf, sBetaGvf, sHGvf, sGradEnergyGvf;
    uint sActorAdv;
    uint sNTdMat;
    uint sOptRewardTd;
    uint sHeadDenom, sCrossTd;
    uint sOneRewardTd, sOneStepDenom;
    uint sDurationTd, sDurationHat;
    uint sObsTraceMulti;
    uint sPrioOpt;
    uint sContinuation;
    uint sLambda, sKeyboardScore;
    uint sGvfTd, sFeatureTraceDenom;
    uint sModelDenom;
    uint sOptionPlanningUse;
    uint sValueGauge;
    uint uCounters;
    uint uOptFeature;
    uint uFeatureAge;
    uint uOptAge;
    uint uFeatureParentA, uFeatureParentB, uFeatureKind;
    uint uFeatureBirthCount;
    uint uSelectedOption, uHighModelOption;
    uint uModelAge;
    uint uGvfAge, uBadFeatureAge;
    uint uPrimitiveChannel;
};

struct KernelScalars {
    uint stepCount;
    uint stateSlot;
    float rho;
};

struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    float rrBonusMean, psiStopGapMean, psiStopGapAbsMean, psiAbsMean;
    float rrStopAbsMean, valueAbsMean, terminationRate, optionDurationMean;
    float stateEffectiveRank;
    float importanceEss;
    float gvfTdMean;
    float lambdaEffectiveRank;
    float actorUpdateNormMean, actorTdMean;
    float actorSupportEntropy;
    float modelPriorityMax;
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
    uint O, R, K, D, traceLevels, anchorCount, pageWidth;
    uint stepCount, stateSlot;
    float rho;
    float invR, invK, invOneInputWidth, metaEta;
    device float *valueW, *psi, *optPolicy, *optReward, *optTransition, *optDuration;
    device float *psiCross, *oneReward;
    device float *gvfRate, *gvfValue;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *actionSignal, *scoreGrad, *optionScoreGrad, *optionPolicyMean;
    device float *importance;
    device float *reward, *td, *tdPsi;
    device float *utilFeat, *utilOpt, *bonusW, *featureMean, *featureSM;
    device float *alpha, *beta, *hMeta;
    device float *gradEnergy;
    device float *alphaPsi, *betaPsi, *hPsi, *gradEnergyPsi;
    device float *alphaModel, *betaModel, *hModel, *gradEnergyModel;
    device float *alphaGvf, *betaGvf, *hGvf, *gradEnergyGvf;
    device float *actorAdv;
    device float *nTdMat, *optRewardTd, *headDenom, *crossTd;
    device float *oneRewardTd, *oneStepDenom;
    device float *durationTd, *durationHat, *obsTraceMulti, *prioOpt, *continuation;
    device float *lambda, *keyboardScore;
    device float *gvfTd, *featureTraceDenom;
    device float *modelDenom;
    device float *optionPlanningUse;
    device float *valueGauge;
    device uint *counters, *optFeature, *featureAge, *optAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption, *highModelOption, *modelAge;
    device uint *gvfAge, *badFeatureAge;
    device uint *primitiveChannel;
    device KernelScalars *scalars;
    device KernelStats *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount;
    c.D = cfg.actionParamCount; c.traceLevels = cfg.obsTraceLevels;
    c.anchorCount = cfg.linearAnchorCount; c.pageWidth = cfg.optionPageWidth;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho;
    c.invR = cfg.invRank; c.invK = cfg.invOptionCount;
    c.invOneInputWidth = cfg.invOneInputWidth; c.metaEta = cfg.metaEta;
    c.valueW = params + cfg.pValue; c.psi = params + cfg.pPsi;
    c.optPolicy = params + cfg.pOptPolicy; c.optReward = params + cfg.pOptReward;
    c.optTransition = params + cfg.pOptTransition; c.optDuration = params + cfg.pOptDuration;
    c.psiCross = params + cfg.pPsiCross;
    c.oneReward = params + cfg.pOneReward;
    c.gvfRate = params + cfg.pGvfRate; c.gvfValue = params + cfg.pGvfValue;
    c.observation = stateArena + cfg.sObservation; c.observationMean = stateArena + cfg.sObsMean;
    c.obsSM = stateArena + cfg.sObsSM; c.state = stateArena + cfg.sState;
    c.policyMean = stateArena + cfg.sPolicyMean; c.actionSignal = stateArena + cfg.sActionSignal;
    c.scoreGrad = stateArena + cfg.sScoreGrad;
    c.optionScoreGrad = stateArena + cfg.sOptionScoreGrad;
    c.optionPolicyMean = stateArena + cfg.sOptionPolicyMean;
    c.importance = stateArena + cfg.sImportance;
    c.reward = stateArena + cfg.sReward; c.td = stateArena + cfg.sTd; c.tdPsi = stateArena + cfg.sTdPsi;
    c.utilFeat = stateArena + cfg.sUtilFeat; c.utilOpt = stateArena + cfg.sUtilOpt;
    c.bonusW = stateArena + cfg.sBonusW; c.featureMean = stateArena + cfg.sFeatureMean;
    c.featureSM = stateArena + cfg.sFeatureSM;
    c.alpha = stateArena + cfg.sAlpha; c.beta = stateArena + cfg.sBeta; c.hMeta = stateArena + cfg.sH;
    c.gradEnergy = stateArena + cfg.sGradEnergy;
    c.alphaPsi = stateArena + cfg.sAlphaPsi; c.betaPsi = stateArena + cfg.sBetaPsi;
    c.hPsi = stateArena + cfg.sHPsi; c.gradEnergyPsi = stateArena + cfg.sGradEnergyPsi;
    c.alphaModel = stateArena + cfg.sAlphaModel; c.betaModel = stateArena + cfg.sBetaModel;
    c.hModel = stateArena + cfg.sHModel; c.gradEnergyModel = stateArena + cfg.sGradEnergyModel;
    c.alphaGvf = stateArena + cfg.sAlphaGvf; c.betaGvf = stateArena + cfg.sBetaGvf;
    c.hGvf = stateArena + cfg.sHGvf; c.gradEnergyGvf = stateArena + cfg.sGradEnergyGvf;
    c.actorAdv = stateArena + cfg.sActorAdv;
    c.nTdMat = stateArena + cfg.sNTdMat; c.optRewardTd = stateArena + cfg.sOptRewardTd;
    c.headDenom = stateArena + cfg.sHeadDenom; c.crossTd = stateArena + cfg.sCrossTd;
    c.oneRewardTd = stateArena + cfg.sOneRewardTd;
    c.oneStepDenom = stateArena + cfg.sOneStepDenom;
    c.durationTd = stateArena + cfg.sDurationTd; c.durationHat = stateArena + cfg.sDurationHat;
    c.obsTraceMulti = stateArena + cfg.sObsTraceMulti;
    c.prioOpt = stateArena + cfg.sPrioOpt; c.continuation = stateArena + cfg.sContinuation;
    c.lambda = stateArena + cfg.sLambda;
    c.keyboardScore = stateArena + cfg.sKeyboardScore;
    c.gvfTd = stateArena + cfg.sGvfTd;
    c.featureTraceDenom = stateArena + cfg.sFeatureTraceDenom;
    c.modelDenom = stateArena + cfg.sModelDenom;
    c.optionPlanningUse = stateArena + cfg.sOptionPlanningUse;
    c.valueGauge = stateArena + cfg.sValueGauge;
    c.counters = uintArena + cfg.uCounters; c.optFeature = uintArena + cfg.uOptFeature;
    c.featureAge = uintArena + cfg.uFeatureAge; c.optAge = uintArena + cfg.uOptAge;
    c.featureParentA = uintArena + cfg.uFeatureParentA; c.featureParentB = uintArena + cfg.uFeatureParentB;
    c.featureKind = uintArena + cfg.uFeatureKind; c.featureBirthCount = uintArena + cfg.uFeatureBirthCount;
    c.selectedOption = uintArena + cfg.uSelectedOption;
    c.highModelOption = uintArena + cfg.uHighModelOption;
    c.modelAge = uintArena + cfg.uModelAge;
    c.gvfAge = uintArena + cfg.uGvfAge; c.badFeatureAge = uintArena + cfg.uBadFeatureAge;
    c.primitiveChannel = uintArena + cfg.uPrimitiveChannel;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline uint linear_input_width(Ctx ctx) { return (ctx.traceLevels + 1u) * ctx.O + ctx.D + 2u; }
inline uint gvf_count(Ctx ctx) { return ctx.R + ctx.K + 2u; }
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
    return (ctx.observation[o] - mean) * rsqrt(var + 1.0f / float(max(ctx.O, 1u)));
}

inline float linear_input(Ctx ctx, uint idx) {
    uint obsSpan = (ctx.traceLevels + 1u) * ctx.O;
    if (idx < ctx.O) return standardized_obs(ctx, idx);
    if (idx < obsSpan) {
        uint t = idx - ctx.O;
        return ctx.obsTraceMulti[t];
    }
    uint tail = idx - obsSpan;
    if (tail < ctx.D) return ctx.actionSignal[tail];
    if (tail == ctx.D) return ctx.reward[0];
    return 1.0f;
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
        return 0.0f;
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
    return 0.0f;
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
        return 0.0f;
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
    return 0.0f;
}

inline void metaopt_update_slot(device float *alpha,
                                device float *beta,
                                device float *h,
                                device float *energy,
                                uint i,
                                float signal,
                                float x,
                                uint age,
                                float metaEta,
                                float decay) {
    if (!isfinite(signal) || !isfinite(x)) return;
    float aOld = alpha[i];
    float hOld = h[i];
    float grad = signal * x;
    float metaGrad = grad * hOld;
    float geRate = 1.0f / (float(age) + 1.0f);
    float geOld = energy[i];
    float geNew = geOld + geRate * (metaGrad * metaGrad - geOld);
    energy[i] = safe_w(geOld, geNew);
    float ridge = 1.0f / (float(age) + 1.0f);
    float normGrad = metaGrad * rsqrt(max(energy[i], 0.0f) + ridge);
    float bOld = beta[i];
    float bNew = bOld + metaEta * normGrad;
    beta[i] = safe_w(bOld, bNew);
    alpha[i] = safe_w(aOld, exp(beta[i]));
    float curvature = x * x;
    float traceDecay = (1.0f - decay) * (1.0f - aOld * curvature);
    h[i] = safe_w(hOld, traceDecay * hOld + aOld * grad);
}

inline void oak_keyboard_predict_body(Ctx ctx, uint tid,
                                      threadgroup float *tgPrev,
                                      threadgroup float *tgCurr,
                                      threadgroup float *tgQNext,
                                      threadgroup float *tgG,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgStopGap,
                                      threadgroup float *tgTerm,
                                      threadgroup float *tgLambda,
                                      threadgroup float *tgTdPsi,
                                      threadgroup float *tgKeyboardScore,
                                      threadgroup float *tgScalars) {
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
    float gauge = isfinite(ctx.valueGauge[0]) ? ctx.valueGauge[0] : 0.0f;
    float vPrevG = vPrev - gauge;
    float vCurrG = vCurr - gauge;

    float rhoOld = ctx.scalars->rho;
    float rewardC = ctx.reward[0] - rhoOld;
    if (tid == 0u) {
        float delta = rewardC + vCurrG - vPrevG;
        if (!isfinite(delta)) { delta = 0.0f; ctx.counters[OAK_CNT_LEARN_NF] += 1u; }
        ctx.td[0] = delta;
        float rhoNew = rhoOld + ctx.invR * delta;
        ctx.scalars->rho = isfinite(rhoNew) ? rhoNew : rhoOld;
    }
    if (tid == 0u) {
        float alphaTraceNorm = 0.0f;
        float modelTraceNorm = 0.0f;
        float gvfTraceNorm = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float x = tgPrev[i];
            if (!isfinite(x)) continue;
            float xx = x * x;
            float av = ctx.alpha[i];
            float am = ctx.alphaModel[i];
            float ag = ctx.alphaGvf[i];
            if (isfinite(av) && av > 0.0f) alphaTraceNorm = fma(av, xx, alphaTraceNorm);
            if (isfinite(am) && am > 0.0f) modelTraceNorm = fma(am, xx, modelTraceNorm);
            if (isfinite(ag) && ag > 0.0f) gvfTraceNorm = fma(ag, xx, gvfTraceNorm);
        }
        tgScalars[7] = (isfinite(alphaTraceNorm) && alphaTraceNorm > 0.0f) ? alphaTraceNorm : 0.0f;
        tgScalars[10] = (isfinite(modelTraceNorm) && modelTraceNorm > 0.0f) ? modelTraceNorm : 0.0f;
        ctx.featureTraceDenom[0] = ((isfinite(gvfTraceNorm) && gvfTraceNorm > 0.0f) ? gvfTraceNorm : 0.0f) + 1.0f;
        float oneTraceNorm = tgScalars[10];
        float actionAlpha = ctx.invOneInputWidth;
        for (uint d = 0u; d < ctx.D; ++d) {
            float x = ctx.actionSignal[d];
            if (isfinite(x)) oneTraceNorm = fma(actionAlpha, x * x, oneTraceNorm);
        }
        float oneDenom = oneTraceNorm + 1.0f;
        ctx.oneStepDenom[0] = (isfinite(oneDenom) && oneDenom > 0.0f) ? oneDenom : 0.0f;
        float scoreNorm = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) {
            float g = ctx.scoreGrad[d];
            if (isfinite(g)) scoreNorm = fma(g, g, scoreNorm);
        }
        tgScalars[9] = scoreNorm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        uint feat = ctx.optFeature[tid] % ctx.R;
        uint base = tid * ctx.R;
        float psiPrev = 0.0f, psiNext = 0.0f, psiXX = 0.0f, psiXY = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            psiPrev = fma(ctx.psi[base + i], tgPrev[i], psiPrev);
            psiNext = fma(ctx.psi[base + i], tgCurr[i], psiNext);
            float a = ctx.alphaPsi[i];
            if (isfinite(a) && a > 0.0f) {
                psiXX = fma(a * tgPrev[i], tgPrev[i], psiXX);
                psiXY = fma(a * tgPrev[i], tgCurr[i], psiXY);
            }
        }
        float zStop = vCurrG - ctx.valueW[feat] * tgCurr[feat] + ctx.bonusW[feat] * tgCurr[feat];
        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop >= psiNext) ? 1.0f : 0.0f;
        float cont = 1.0f - beta;
        ctx.continuation[tid] = cont;
        float dPsi = rewardC + cont * psiNext + beta * zStop - psiPrev;
        if (!isfinite(dPsi)) {
            dPsi = 0.0f;
            ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        }
        ctx.tdPsi[tid] = dPsi;
        tgTdPsi[tid] = dPsi;
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float implicitTrace = psiXX - cont * psiXY;
        float denom = fma(rho, implicitTrace, 1.0f);
        ctx.headDenom[tid] = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        float modelDenom = fma(rho, tgScalars[10], 1.0f);
        ctx.modelDenom[tid] = (isfinite(modelDenom) && modelDenom > 0.0f) ? modelDenom : 0.0f;
        tgCont[tid] = cont;
        tgStopGap[tid] = isfinite(psiNext - zStop) ? (psiNext - zStop) : 0.0f;
        tgTerm[tid] = beta;
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
        float z = vCurrG - ctx.valueW[target] * tgCurr[target] + ctx.bonusW[target] * tgCurr[target];
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
            if (ctx.R > 0u && isfinite(hNext)) {
                float w0 = ctx.gvfValue[g * ctx.R];
                ctx.gvfValue[g * ctx.R] = safe_w(w0, w0 - hNext);
            }
        }
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
        float gaugeRate = track_rate(ctx.stepCount, ctx.R);
        float gaugeNew = ctx.valueGauge[0] + gaugeRate * (vCurr - ctx.valueGauge[0]);
        ctx.valueGauge[0] = safe_w(ctx.valueGauge[0], gaugeNew);
        float actorBaseline = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) actorBaseline += tgTdPsi[k];
        actorBaseline *= invK;
        for (uint k = 0u; k < ctx.K; ++k) {
            float adv = tgTdPsi[k] - actorBaseline;
            if (!isfinite(adv)) adv = 0.0f;
            tgTdPsi[k] = adv;
            ctx.actorAdv[k] = adv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float rewardL1 = 0.0f, valueL1 = 0.0f;
        for (uint i = 0u; i < ctx.K; ++i) {
            float wr = (i < ctx.R) ? ctx.oneReward[i] : 0.0f;
            float wv = (i < ctx.R) ? ctx.valueW[i] : 0.0f;
            if (!isfinite(wr)) wr = 0.0f;
            if (!isfinite(wv)) wv = 0.0f;
            tgQNext[i] = wr;
            tgG[i] = wv;
            rewardL1 += fabs(wr);
            valueL1 += fabs(wv);
        }
        float invK = ctx.invK;
        float l2 = 0.0f, zLambda = 0.0f;
        for (uint i = 0u; i < ctx.K; ++i) {
            float lam = invK;
            if (rewardL1 > 0.0f) lam = tgQNext[i] / rewardL1;
            else if (valueL1 > 0.0f) lam = tgG[i] / valueL1;
            if (!isfinite(lam)) lam = invK;
            ctx.lambda[i] = lam;
            tgLambda[i] = lam;
            l2 += lam * lam;
            if (i < ctx.R) {
                float z = vCurrG - ctx.valueW[i] * tgCurr[i] + ctx.bonusW[i] * tgCurr[i];
                if (isfinite(z)) zLambda = fma(lam, z, zLambda);
            }
        }
        tgScalars[3] = (l2 > 0.0f) ? (1.0f / l2) : float(ctx.K);
        tgScalars[8] = isfinite(zLambda) ? zLambda : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) tgKeyboardScore[tid] = ctx.keyboardScore[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pageCount = (ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
    uint page = ctx.stepCount % max(pageCount, 1u);
    uint cand = page * ctx.pageWidth + tid;
    if (tid < ctx.pageWidth && cand < ctx.K) {
        float G = 0.0f;
        for (uint key = 0u; key < ctx.K; ++key) {
            float lam = tgLambda[key];
            uint base = (cand * ctx.K + key) * ctx.R;
            float p = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) p = fma(ctx.psiCross[base + i], tgCurr[i], p);
            G = fma(lam, p, G);
        }
        float liveG = isfinite(G) ? G : -FLT_MAX;
        ctx.keyboardScore[cand] = liveG;
        tgKeyboardScore[cand] = liveG;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint best = 0u;
        float bestG = -FLT_MAX;
        float stopGapSum = 0.0f, termSum = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            stopGapSum += tgStopGap[k];
            termSum += tgTerm[k];
            float G = tgKeyboardScore[k];
            if (G > bestG) { bestG = G; best = k; }
        }
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        uint selected = (tgCont[prevSel] > 0.0f && tgKeyboardScore[prevSel] >= tgScalars[8]) ? prevSel : best;
        ctx.selectedOption[0] = selected;
        float invK = ctx.invK;
        ctx.stats->psiStopGapMean = stopGapSum * invK;
        ctx.stats->terminationRate = termSum * invK;
        ctx.stats->lambdaEffectiveRank = tgScalars[3];

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
    threadgroup float tgLambda[OAK_MAX_RANK];
    threadgroup float tgTdPsi[OAK_MAX_RANK];
    threadgroup float tgKeyboardScore[OAK_MAX_RANK];
    threadgroup float tgScalars[16];
    threadgroup float tgPrevState[OAK_MAX_RANK];
    threadgroup float tgCurrState[OAK_MAX_RANK];

    uint nf = 0u;
    float obsRate = track_rate(ctx.stepCount, ctx.O);
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nf += 1u; continue; }
        ctx.observationMean[o] += obsRate * (x - ctx.observationMean[o]);
        ctx.obsSM[o] += obsRate * (x * x - ctx.obsSM[o]);
        float s = standardized_obs(ctx, o);
        for (uint m = 1u; m <= ctx.traceLevels; ++m) {
            uint slot = (m - 1u) * ctx.O + o;
            float rate = 1.0f / float(1u << m);
            ctx.obsTraceMulti[slot] += rate * (s - ctx.obsTraceMulti[slot]);
        }
    }
    if (nf > 0u) ctx.counters[OAK_CNT_OBS_NF] += nf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float raw = 0.0f;
        bool directBias = false;
        if (i < ctx.anchorCount) {
            uint width = max(linear_input_width(ctx), 1u);
            uint channel = ctx.primitiveChannel[i] % width;
            raw = linear_input(ctx, channel);
            directBias = (channel + 1u == width);
        } else {
            uint pa = ctx.featureParentA[i] % ctx.R;
            uint pb = ctx.featureParentB[i] % ctx.R;
            float a = prev[pa], b = prev[pb];
            uint kind = ctx.featureKind[i] % 3u;
            if (kind == 0u) raw = a * b;
            else if (kind == 1u) raw = (a + b) * 0.7071067811865475f;
            else raw = a * (b - a);
        }
        if (directBias) {
            curr[i] = 1.0f;
            continue;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ctx.R; i += tgW) {
        tgPrevState[i] = prev[i];
        tgCurrState[i] = curr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    oak_keyboard_predict_body(ctx, tid, tgPrevState, tgCurrState, tgQNext, tgG, tgCont, tgStopGap, tgTerm,
                              tgLambda, tgTdPsi, tgKeyboardScore, tgScalars);
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

    threadgroup float tgModelErr[OAK_MAX_RANK];

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
        tgModelErr[j] = vRel * nTd * nTd;
    } else {
        tgModelErr[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j == 0u) {
        float nErr = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            nErr += tgModelErr[i];
        }
        float rHat = 0.0f, rPrev = 0.0f, lRaw = 0.0f, lPrev = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            uint idx = k * ctx.R + i;
            rHat = fma(ctx.optReward[idx], curr[i], rHat);
            rPrev = fma(ctx.optReward[idx], prev[i], rPrev);
            lRaw = fma(ctx.optDuration[idx], curr[i], lRaw);
            lPrev = fma(ctx.optDuration[idx], prev[i], lPrev);
        }
        float rTd = ctx.reward[0] + cont * rHat - rPrev;
        float dTd = 1.0f + cont * lRaw - lPrev;
        if (!isfinite(rTd)) rTd = 0.0f;
        if (!isfinite(dTd)) dTd = 0.0f;
        ctx.optRewardTd[k] = rTd;
        ctx.durationTd[k] = dTd;
        ctx.durationHat[k] = decode_duration(lRaw);
        float modelErr = nErr + rTd * rTd + (ctx.scalars->rho * dTd) * (ctx.scalars->rho * dTd);
        ctx.prioOpt[k] = isfinite(modelErr) ? sqrt(max(modelErr, 0.0f)) : 0.0f;
        float useNew = ctx.optionPlanningUse[k] + ctx.invR * (ctx.prioOpt[k] - ctx.optionPlanningUse[k]);
        ctx.optionPlanningUse[k] = safe_w(ctx.optionPlanningUse[k], useNew);

        if (k == 0u) {
            float rPred = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) rPred = fma(ctx.oneReward[i], prev[i], rPred);
            float rE = ctx.reward[0] - rPred;
            ctx.oneRewardTd[0] = isfinite(rE) ? rE : 0.0f;
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
    threadgroup float tgPrio[OAK_MAX_RANK];
    threadgroup float tgDur[OAK_MAX_RANK];
    threadgroup uint tgModelAge[OAK_MAX_RANK];
    if (tid < ctx.K) {
        float p = ctx.prioOpt[tid];
        tgPrio[tid] = isfinite(p) ? p : 0.0f;
        tgDur[tid] = ctx.durationHat[tid];
        tgModelAge[tid] = ctx.modelAge[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float invK = ctx.invK;
        float durSum = 0.0f;
        float maxPrio = -FLT_MAX;
        uint high = 0u;
        for (uint k = 0u; k < ctx.K; ++k) {
            float p = tgPrio[k];
            if (p > maxPrio) { maxPrio = p; high = k; }
            durSum += tgDur[k];
        }
        uint pageCount = (ctx.K + ctx.pageWidth - 1u) / max(ctx.pageWidth, 1u);
        uint selectedPage = 0u, stalePage = 0u, stalePageAge = 0u;
        if (ctx.K > ctx.pageWidth) {
            uint highPage = 0u;
            float highPagePriority = -FLT_MAX;
            for (uint p = 0u; p < pageCount; ++p) {
                float pp = -FLT_MAX;
                uint pa = 0u;
                uint start = p * ctx.pageWidth;
                uint end = min(start + ctx.pageWidth, ctx.K);
                for (uint k = start; k < end; ++k) {
                    pp = max(pp, tgPrio[k]);
                    pa = max(pa, tgModelAge[k]);
                }
                if (pp > highPagePriority) { highPagePriority = pp; highPage = p; }
                if (pa >= stalePageAge) { stalePageAge = pa; stalePage = p; }
            }
            selectedPage = (stalePageAge > pageCount) ? stalePage : highPage;
        }
        ctx.highModelOption[0] = (ctx.K <= ctx.pageWidth) ? high : min(selectedPage * ctx.pageWidth, ctx.K - 1u);
        ctx.stats->optionDurationMean = durSum * invK;
        ctx.stats->modelPriorityMax = maxPrio > -FLT_MAX ? maxPrio : 0.0f;
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
    uint nPsi = ctx.K * ctx.R;
    uint nCross = ctx.K * ctx.R;
    uint nPi = ctx.K * ctx.D * ctx.R;
    uint nOR = ctx.K * ctx.R;
    uint nOT = ctx.K * ctx.R * ctx.R;
    uint nOD = ctx.K * ctx.R;
    uint nOneR = ctx.R;
    uint total = nVal + nGRate + nGValue + nPsi + nCross + nPi + nOR + nOT
               + nOD + nOneR;
    if (gid >= total) return;

    if (gid < nVal) {
        uint i = gid;
        float w = ctx.valueW[i];
        float delta = ctx.td[0] * prev[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta);
        return;
    }
    gid -= nVal;
    if (gid < nGRate) {
        uint g = gvfBase + gid;
        if (g >= G) return;
        float d = ctx.gvfTd[g];
        if (!isfinite(d)) d = 0.0f;
        uint last = min(ctx.gvfAge[g], ctx.stepCount);
        uint age = ctx.stepCount - last;
        float alphaRate = 1.0f / float(max(age + 1u, ctx.R));
        float c = gvf_cumulant(ctx, curr, g);
        if (!isfinite(c)) c = 0.0f;
        float rateErr = c - ctx.gvfRate[g];
        if (!isfinite(rateErr)) rateErr = 0.0f;
        float rate = ctx.gvfRate[g] + alphaRate * rateErr;
        ctx.gvfRate[g] = safe_w(ctx.gvfRate[g], rate);
        ctx.gvfAge[g] = ctx.stepCount;
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
        ctx.gvfValue[g * ctx.R + i] = safe_w(w, w + ctx.alphaGvf[i] * d * prev[i] * scale);
        return;
    }
    gid -= nGValue;
    if (gid < nPsi) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float w = ctx.psi[k * ctx.R + i];
        float denom = ctx.headDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.psi[k * ctx.R + i] = safe_w(w, w + ctx.alphaPsi[i] * rho * ctx.tdPsi[k] * prev[i] * scale);
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
        float denom = ctx.headDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.psiCross[base + i] = safe_w(w, w + ctx.alphaPsi[i] * rho * td * prev[i] * scale);
        return;
    }
    gid -= nCross;
    if (gid < nPi) {
        uint k = gid / (ctx.D * ctx.R);
        uint rem = gid - k * ctx.D * ctx.R;
        uint d = rem / ctx.R, i = rem % ctx.R;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float w = ctx.optPolicy[(k * ctx.D + d) * ctx.R + i];
        float targetGrad = ctx.optionScoreGrad[k * ctx.D + d];
        if (!isfinite(targetGrad)) targetGrad = 0.0f;
        float grad = rho * targetGrad * prev[i];
        float denom = ctx.headDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float actorAlpha = 1.0f / float(max(ctx.K * ctx.D * ctx.R, 1u));
        ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] =
            safe_w(w, w + actorAlpha * ctx.actorAdv[k] * grad * scale);
        return;
    }
    gid -= nPi;
    if (gid < nOR) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.modelDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optReward[k * ctx.R + i];
        ctx.optReward[k * ctx.R + i] = safe_w(w, w + ctx.alphaModel[i] * rho * ctx.optRewardTd[k] * prev[i] * scale);
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
        float denom = ctx.modelDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i];
        ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] =
            safe_w(w, w + ctx.alphaModel[i] * rho * ctx.nTdMat[k * ctx.R + j] * prev[i] * scale);
        return;
    }
    gid -= nOT;
    if (gid < nOD) {
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.modelDenom[k];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optDuration[k * ctx.R + i];
        ctx.optDuration[k * ctx.R + i] = safe_w(w, w + ctx.alphaModel[i] * rho * ctx.durationTd[k] * prev[i] * scale);
        return;
    }
    gid -= nOD;
    if (gid < nOneR) {
        uint i = gid;
        float x = one_input(ctx, prev, i);
        float a = (i < ctx.R) ? ctx.alphaModel[i] : ctx.invOneInputWidth;
        float w = ctx.oneReward[i];
        float denom = ctx.oneStepDenom[0];
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.oneReward[i] = safe_w(w, w + a * ctx.oneRewardTd[0] * x * scale);
        return;
    }
    gid -= nOneR;
}

inline void emit_action_grid(Ctx ctx, threadgroup const float *curr, uint group, uint tid, uint tgW) {
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
    threadgroup float tgCurrState[OAK_MAX_RANK];
    for (uint i = tid; i < ctx.R; i += tgW) tgCurrState[i] = curr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    emit_action_grid(ctx, tgCurrState, group, tid, tgW);
    if (group != 0u) return;
    float invT = track_rate(ctx.stepCount, ctx.R);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float psiSignal = (i < ctx.K) ? ctx.tdPsi[i] : 0.0f;
        float modelSignal = (i < ctx.K) ? ctx.prioOpt[i] : 0.0f;
        float g = fabs(ctx.td[0] * prev[i]) + fabs(psiSignal * prev[i])
                + fabs(modelSignal * prev[i]);
        ctx.utilFeat[i] += invT * (g - ctx.utilFeat[i]);
        uint age = ctx.featureAge[i];
        metaopt_update_slot(ctx.alpha, ctx.beta, ctx.hMeta, ctx.gradEnergy,
                            i, ctx.td[0], prev[i], age, ctx.metaEta, ctx.invR);
        uint metaSlot = ctx.stepCount % max(ctx.R, 1u);
        if (i == metaSlot) {
            metaopt_update_slot(ctx.alphaPsi, ctx.betaPsi, ctx.hPsi, ctx.gradEnergyPsi,
                                i, psiSignal, prev[i], age, ctx.metaEta, ctx.invR);
            metaopt_update_slot(ctx.alphaModel, ctx.betaModel, ctx.hModel, ctx.gradEnergyModel,
                                i, modelSignal, prev[i], age, ctx.metaEta, ctx.invR);
        }
        if (isfinite(ctx.valueW[i]) && ctx.valueW[i] > ctx.bonusW[i]) ctx.bonusW[i] = ctx.valueW[i];
    }
    for (uint k = tid; k < ctx.K; k += tgW) {
        float u = ctx.tdPsi[k] * ctx.tdPsi[k] + ctx.prioOpt[k];
        ctx.utilOpt[k] += invT * (u - ctx.utilOpt[k]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid != 0u) return;
    float sum2 = 0.0f, sum4 = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) {
        float x = tgCurrState[i];
        float x2 = x * x;
        sum2 += x2; sum4 += x2 * x2;
    }
    float effRank = (sum4 > 0.0f) ? (sum2 * sum2 / sum4) : 0.0f;
    ctx.stats->stateEffectiveRank = effRank;

	    float wSum = 0.0f, wSq = 0.0f;
	    float actorTdSum = 0.0f;
	    for (uint k = 0u; k < ctx.K; ++k) {
	        float w = ctx.importance[k];
	        if (!isfinite(w)) { w = 0.0f; ctx.counters[OAK_CNT_IMPORTANCE_NF] += 1u; }
	        wSum += w; wSq += w * w;
	        float d = ctx.tdPsi[k];
	        if (!isfinite(d)) d = 0.0f;
	        actorTdSum += d;
	    }
	    float invK = ctx.invK;
	    ctx.stats->importanceEss = (wSq > 0.0f) ? ((wSum * wSum) / (float(ctx.K) * wSq)) : 0.0f;
	    ctx.stats->actorTdMean = actorTdSum * invK;
	    float actorEntropy = 0.0f;
	    if (wSum > 0.0f) {
	        for (uint k = 0u; k < ctx.K; ++k) {
	            float p = max(ctx.importance[k], 0.0f) / wSum;
	            if (p > 0.0f) actorEntropy -= p * log(p);
	        }
		    }
		    ctx.stats->actorSupportEntropy = (ctx.K > 1u) ? actorEntropy / log(float(ctx.K)) : 0.0f;
				    float actorUpdateSum = 0.0f;
				    float actorAlpha = 1.0f / float(max(ctx.K * ctx.D * ctx.R, 1u));
				    for (uint k = 0u; k < ctx.K; ++k) {
				        float u = fabs(actorAlpha * ctx.importance[k] * ctx.actorAdv[k]);
				        if (!isfinite(u)) u = 0.0f;
				        actorUpdateSum += u;
				    }
				    ctx.stats->actorUpdateNormMean = actorUpdateSum * invK;

    uint candidateStart = ctx.anchorCount;
    uint eligible = 0u, worst = candidateStart;
    bool haveWorst = false;
    float worstUtil = 0.0f, worstRed = 0.0f, worstDownstream = 0.0f;
    uint worstAge = 0u;
    float eligibleUtilSum = 0.0f, eligibleRedSum = 0.0f, eligibleDownstreamSum = 0.0f;
    for (uint i = candidateStart; i < ctx.R; ++i) {
        if (ctx.featureAge[i] >= ctx.R) {
            eligible += 1u;
            uint pa = ctx.featureParentA[i] % ctx.R;
            float util = ctx.utilFeat[i];
            if (!isfinite(util)) util = 0.0f;
	            float red = fabs(tgCurrState[i] * tgCurrState[pa]);
            if (!isfinite(red)) red = 0.0f;
	            float modelUse = (i < ctx.K)
	                ? (ctx.utilOpt[i] + ctx.prioOpt[i] + ctx.optionPlanningUse[i])
	                : 0.0f;
	            if (!isfinite(modelUse)) modelUse = 0.0f;
	            float downstreamUse = modelUse + fabs(ctx.valueW[i]);
            if (!isfinite(downstreamUse)) downstreamUse = 0.0f;
            eligibleUtilSum += util;
            eligibleRedSum += red;
            eligibleDownstreamSum += downstreamUse;
            bool worse = !haveWorst
                || downstreamUse < worstDownstream
                || (downstreamUse == worstDownstream && util < worstUtil)
                || (downstreamUse == worstDownstream && util == worstUtil && red > worstRed)
                || (downstreamUse == worstDownstream && util == worstUtil && red == worstRed && ctx.featureAge[i] > worstAge);
            if (worse) {
                haveWorst = true; worst = i; worstAge = ctx.featureAge[i];
                worstUtil = util; worstRed = red; worstDownstream = downstreamUse;
            }
        }
    }
		    float gvfTdAbs = 0.0f;
		    uint G = max(gvf_count(ctx), 1u);
		    uint gBaseStats = gvf_page_base(ctx);
		    uint gEndStats = min(gBaseStats + ctx.pageWidth, G);
		    float gStatsN = float(max(gEndStats - gBaseStats, 1u));
		    for (uint g = gBaseStats; g < gEndStats; ++g) {
		        gvfTdAbs += fabs(ctx.gvfTd[g]);
		    }
		    float invG = 1.0f / gStatsN;
			    ctx.stats->gvfTdMean = gvfTdAbs * invG;

	    uint selectedPage = ctx.highModelOption[0] / max(ctx.pageWidth, 1u);
	    for (uint k = 0u; k < ctx.K; ++k) {
	        if (ctx.modelAge[k] < 0xffffffffu) ctx.modelAge[k] += 1u;
	        bool planned = (ctx.K <= ctx.pageWidth)
	            ? true
	            : ((k / max(ctx.pageWidth, 1u)) == selectedPage);
	        if (planned) ctx.modelAge[k] = 0u;
	        if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;
	    }
	    for (uint i = 0u; i < ctx.R; ++i) if (ctx.featureAge[i] < 0xffffffffu) ctx.featureAge[i] += 1u;

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRed = eligibleRedSum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtil < meanEligibleUtil;
    bool highRedundancy = worstRed > meanEligibleRed;
    bool lowDownstreamUse = worstDownstream < meanEligibleDownstream;
	    bool structurallyReplaceable = (eligible > 1u) &&
	        (lowUtility && highRedundancy && lowDownstreamUse);
		    uint wouldCommit = 0u;
		    for (uint i = candidateStart; i < ctx.R; ++i) {
		            bool bad = false;
		            if (ctx.featureAge[i] >= ctx.R && eligible > 1u) {
		                uint pa = ctx.featureParentA[i] % ctx.R;
		                float util = ctx.utilFeat[i];
		                if (!isfinite(util)) util = 0.0f;
		                float red = fabs(tgCurrState[i] * tgCurrState[pa]);
		                if (!isfinite(red)) red = 0.0f;
			                float modelUse = (i < ctx.K)
		                    ? (ctx.utilOpt[i] + ctx.prioOpt[i] + ctx.optionPlanningUse[i])
				                    : 0.0f;
			                if (!isfinite(modelUse)) modelUse = 0.0f;
			                float downstreamUse = modelUse + fabs(ctx.valueW[i]);
		                if (!isfinite(downstreamUse)) downstreamUse = 0.0f;
		                bad = (util < meanEligibleUtil) && (red > meanEligibleRed) && (downstreamUse < meanEligibleDownstream);
		            }
		            if (bad) {
		                if (ctx.badFeatureAge[i] < 0xffffffffu) ctx.badFeatureAge[i] += 1u;
		            } else {
		                ctx.badFeatureAge[i] = 0u;
		            }
		    }
		    if (worst < ctx.R && ctx.badFeatureAge[worst] >= ctx.R) wouldCommit = 1u;
	    bool doReplace = structurallyReplaceable && (wouldCommit != 0u);
    if (doReplace && worst < ctx.R) {
        uint pa = 0u, pb = (ctx.R > 1u) ? 1u : 0u;
        float bestA = -FLT_MAX, bestB = -FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst) continue;
            float u = ctx.utilFeat[i];
            if (!isfinite(u)) u = -FLT_MAX;
            if (u > bestA) { bestA = u; pa = i; }
        }
        for (uint i = 0u; i < ctx.R; ++i) {
            if (i == worst || i == pa) continue;
	        float decor = 1.0f / (1.0f + fabs(tgCurrState[i] * tgCurrState[pa]));
	        float u = ctx.utilFeat[i] * decor;
            if (!isfinite(u)) u = -FLT_MAX;
            if (u > bestB) { bestB = u; pb = i; }
        }
        uint birth = ctx.featureBirthCount[worst] + 1u;
        ctx.featureBirthCount[worst] = birth;
        ctx.featureParentA[worst] = pa;
        ctx.featureParentB[worst] = pb;
        ctx.featureKind[worst] = birth % 3u;
	        ctx.featureAge[worst] = 0u;
	        ctx.badFeatureAge[worst] = 0u;
	        ctx.featureMean[worst] = 0.0f; ctx.featureSM[worst] = 0.0f;
	        ctx.utilFeat[worst] = 0.0f; ctx.valueW[worst] = 0.0f; ctx.bonusW[worst] = 0.0f;
	        uint GAll = gvf_count(ctx);
	        if (worst < GAll) {
	            ctx.gvfRate[worst] = 0.0f; ctx.gvfTd[worst] = 0.0f;
	            ctx.gvfAge[worst] = 0u;
	        }
	        ctx.alpha[worst] = ctx.invR;
	        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f; ctx.gradEnergy[worst] = 0.0f;
	        ctx.alphaPsi[worst] = ctx.invR;
	        ctx.betaPsi[worst] = log(ctx.alphaPsi[worst]); ctx.hPsi[worst] = 0.0f; ctx.gradEnergyPsi[worst] = 0.0f;
		        ctx.alphaModel[worst] = ctx.invR;
	        ctx.betaModel[worst] = log(ctx.alphaModel[worst]); ctx.hModel[worst] = 0.0f; ctx.gradEnergyModel[worst] = 0.0f;
	        ctx.alphaGvf[worst] = ctx.invR;
	        ctx.betaGvf[worst] = log(ctx.alphaGvf[worst]); ctx.hGvf[worst] = 0.0f; ctx.gradEnergyGvf[worst] = 0.0f;
	        uint k = 0u;
	        float worstBasis = FLT_MAX;
	        for (uint ok = 0u; ok < ctx.K; ++ok) {
	            float use = ctx.optionPlanningUse[ok] + ctx.utilOpt[ok];
	            if (!isfinite(use)) use = FLT_MAX;
	            if (ctx.optFeature[ok] == worst) { k = ok; worstBasis = -FLT_MAX; break; }
	            if (use < worstBasis) { worstBasis = use; k = ok; }
	        }
		        ctx.optAge[k] = 0u; ctx.optFeature[k] = worst; ctx.utilOpt[k] = float(ctx.R);
	        ctx.actorAdv[k] = 0.0f;
        ctx.optionPlanningUse[k] = 0.0f;
	        uint termG = ctx.R + 2u + k;
	        if (termG < GAll) {
	            ctx.gvfRate[termG] = 0.0f; ctx.gvfTd[termG] = 0.0f;
	            ctx.gvfAge[termG] = 0u;
	        }
		        for (uint i = 0u; i < ctx.R; ++i) {
		            ctx.psi[k * ctx.R + i] = 0.0f;
		            ctx.optReward[k * ctx.R + i] = 0.0f;
	            ctx.optDuration[k * ctx.R + i] = 0.0f;
	            if (worst < GAll) ctx.gvfValue[worst * ctx.R + i] = 0.0f;
	            if (termG < GAll) ctx.gvfValue[termG * ctx.R + i] = 0.0f;
	            for (uint d = 0u; d < ctx.D; ++d) ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] = 0.0f;
	            for (uint j = 0u; j < ctx.R; ++j) ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] = 0.0f;
	            for (uint ci = 0u; ci < ctx.K; ++ci) {
	                ctx.psiCross[(k * ctx.K + ci) * ctx.R + i] = 0.0f;
	                ctx.psiCross[(ci * ctx.K + k) * ctx.R + i] = 0.0f;
	            }
        }
	        for (uint g = 0u; g < GAll; ++g) ctx.gvfValue[g * ctx.R + worst] = 0.0f;
	        float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
	        ctx.psi[k * ctx.R + worst] = invSqrtR;
	        ctx.psiCross[(k * ctx.K + worst) * ctx.R + worst] = invSqrtR;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
    }

    float bonusGap = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) bonusGap += fabs(ctx.bonusW[i] - ctx.valueW[i]);
    ctx.stats->rrBonusMean = bonusGap / float(max(ctx.R, 1u));
    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount = float(ctx.counters[OAK_CNT_TDPSI_NF] + ctx.counters[OAK_CNT_IMPORTANCE_NF]);
    ctx.stats->oakOptionReinitCount = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);
    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
