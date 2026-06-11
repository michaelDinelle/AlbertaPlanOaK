#include <metal_stdlib>
#include <float.h>
#include "OakPhysics.metal"
using namespace metal;

constant uint OAK_CNT_OBS_NF        = 0u;
constant uint OAK_CNT_LEARN_NF      = 1u;
constant uint OAK_CNT_PLAN_NF       = 2u;
constant uint OAK_CNT_REINIT_OPT    = 3u;
constant uint OAK_CNT_REINIT_FEAT   = 4u;
constant uint OAK_CNT_TDPSI_NF      = 5u;
constant uint OAK_CNT_IMPORTANCE_NF = 6u;

constant bool OAK_ABLATE_FEATURE_CONSTRUCTION [[function_constant(0)]];
constant bool OAK_ABLATE_GVF_PREDICTION [[function_constant(1)]];
constant bool OAK_ABLATE_RR_SUBTASKS [[function_constant(2)]];
constant bool OAK_ABLATE_OPTION_POLICIES [[function_constant(3)]];
constant bool OAK_ABLATE_OPTION_MODELS [[function_constant(4)]];
constant bool OAK_ABLATE_PLANNING [[function_constant(5)]];
constant bool OAK_ABLATE_OPTION_KEYBOARD [[function_constant(6)]];
constant bool OAK_ABLATE_MODEL_TO_KEYBOARD [[function_constant(7)]];
constant bool OAK_ABLATE_KEYBOARD_TO_POLICY [[function_constant(8)]];
constant bool OAK_ABLATE_SEARCH_CONTROL [[function_constant(9)]];
constant bool OAK_ABLATE_SEARCH_TO_GENERATE_TEST [[function_constant(10)]];
constant bool OAK_ABLATE_GENERATE_AND_TEST [[function_constant(11)]];

struct KernelConfig {
    uint obsDim, rank, optionCount, actionParamCount;
    uint obsTraceLevels;
    uint linearAnchorCount;
    float invRank, invOptionCount;
    float metaEta;
    uint agentCount, paramStride, stateStride, uintStride;
};

struct KernelScalars {
    uint stepCount;
    uint stateSlot;
    float rho;
};

struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    float terminationRate, optionDurationMean;
    float importanceEss;
    float actorUpdateNormMean;
    float actorSupportEntropy;
    float modelPriorityMax;
    float selectedOptionAge, selectedOptionDuration, selectedOptionPriority;
    float keyboardL1, actorAdvantageL1;
    float selectedOptionIndex, keyboardOrigin;
    float selectedOptionFeature, selectedContinuation, selectedKeyboardWeight;
    float mainTd, planTdValue;
    float keyboardSketchDim, keyboardEffectiveL1;
    float searchControlUseMean, generateTestEligibleCount;
    float learningMature;  // mean over options of (1 - fast/peak) of |tdPsi[k]|
};

inline float safe_w(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct Ctx {
    uint O, R, K, D, traceLevels, anchorCount;
    uint stepCount, stateSlot;
    float rho;
    float invR, invK, metaEta;
    device float *valueW, *psi, *optPolicy, *optReward, *optTransition, *optDuration;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *actionSignal, *optionActorCredit, *optionPolicyMean;
    device float *importance;
    device float *reward, *td, *planTd, *tdPsi;
    device float *utilFeat, *utilOpt, *bonusW, *featureMean, *featureSM;
    device float *alpha, *beta, *hMeta;
    device float *gradEnergy;
    device float *alphaPsi, *betaPsi, *hPsi, *gradEnergyPsi;
    device float *alphaModel, *betaModel, *hModel, *gradEnergyModel;
    device float *actorAdv;
    device float *nTdMat, *optRewardTd, *headDenom;
    device float *durationTd, *durationHat, *obsTraceMulti, *prioOpt, *continuation;
    device float *keyboard, *modelDenom;
    device float *valueGauge;
    device float *tdPsiFast, *tdPsiPeak;  // per-option learning-progress EMAs
    device uint *counters, *optFeature, *featureAge, *optAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption, *keyboardOrigin;
    device uint *badFeatureAge;
    device KernelScalars *scalars;
    device KernelStats *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg, uint agentID) {
    params += agentID * cfg.paramStride;
    stateArena += agentID * cfg.stateStride;
    uintArena += agentID * cfg.uintStride;
    scalars += agentID;
    stats += agentID;
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount;
    c.D = cfg.actionParamCount; c.traceLevels = cfg.obsTraceLevels;
    c.anchorCount = cfg.linearAnchorCount;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho;
    c.invR = cfg.invRank; c.invK = cfg.invOptionCount; c.metaEta = cfg.metaEta;
    uint p = 0u;
    c.valueW = params + p; p += c.R;
    c.psi = params + p; p += c.K * c.R;
    c.optPolicy = params + p; p += c.K * c.D * c.R;
    c.optReward = params + p; p += c.K * c.R;
    c.optTransition = params + p; p += c.K * c.R * c.R;
    c.optDuration = params + p;

    uint s = 0u;
    c.observation = stateArena + s; s += c.O;
    c.observationMean = stateArena + s; s += c.O;
    c.obsSM = stateArena + s; s += c.O;
    c.state = stateArena + s; s += 2u * c.R;
    c.policyMean = stateArena + s; s += c.D;
    c.actionSignal = stateArena + s; s += c.D;
    c.optionActorCredit = stateArena + s; s += c.K * c.D;
    c.optionPolicyMean = stateArena + s; s += c.K * c.D;
    c.importance = stateArena + s; s += c.K;
    c.reward = stateArena + s; s += 1u;
    c.td = stateArena + s; s += 1u;
    c.planTd = stateArena + s; s += 1u;
    c.tdPsi = stateArena + s; s += c.K;
    c.utilFeat = stateArena + s; s += c.R;
    c.utilOpt = stateArena + s; s += c.K;
    c.bonusW = stateArena + s; s += c.R;
    c.featureMean = stateArena + s; s += c.R;
    c.featureSM = stateArena + s; s += c.R;
    c.alpha = stateArena + s; s += c.R;
    c.beta = stateArena + s; s += c.R;
    c.hMeta = stateArena + s; s += c.R;
    c.gradEnergy = stateArena + s; s += c.R;
    c.alphaPsi = stateArena + s; s += c.R;
    c.betaPsi = stateArena + s; s += c.R;
    c.hPsi = stateArena + s; s += c.R;
    c.gradEnergyPsi = stateArena + s; s += c.R;
    c.alphaModel = stateArena + s; s += c.R;
    c.betaModel = stateArena + s; s += c.R;
    c.hModel = stateArena + s; s += c.R;
    c.gradEnergyModel = stateArena + s; s += c.R;
    c.actorAdv = stateArena + s; s += c.K;
    c.nTdMat = stateArena + s; s += c.K * c.R;
    c.optRewardTd = stateArena + s; s += c.K;
    c.headDenom = stateArena + s; s += c.K;
    c.durationTd = stateArena + s; s += c.K;
    c.durationHat = stateArena + s; s += c.K;
    c.obsTraceMulti = stateArena + s; s += c.traceLevels * c.O;
    c.prioOpt = stateArena + s; s += c.K;
    c.continuation = stateArena + s; s += c.K;
    c.keyboard = stateArena + s; s += c.K;
    c.modelDenom = stateArena + s; s += c.K;
    c.valueGauge = stateArena + s; s += 1u;
    c.tdPsiFast = stateArena + s; s += c.K;
    c.tdPsiPeak = stateArena + s; s += c.K;

    uint u = 0u;
    c.counters = uintArena + u; u += 7u;
    c.optFeature = uintArena + u; u += c.K;
    c.featureAge = uintArena + u; u += c.R;
    c.optAge = uintArena + u; u += c.K;
    c.featureParentA = uintArena + u; u += c.R;
    c.featureParentB = uintArena + u; u += c.R;
    c.featureKind = uintArena + u; u += c.R;
    c.featureBirthCount = uintArena + u; u += c.R;
    c.selectedOption = uintArena + u; u += 1u;
    c.keyboardOrigin = uintArena + u; u += 1u;
    c.badFeatureAge = uintArena + u;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline bool usable_priority(float p) { return isfinite(p) && p >= 0.0f; }
inline bool usable_duration(Ctx ctx, float d) {
    return isfinite(d) && d > 0.0f;
}
inline uint linear_input_width(Ctx ctx) { return (ctx.traceLevels + 1u) * ctx.O + ctx.D + ctx.K + 2u; }

inline uint keyboard_sketch_dim(Ctx ctx) {
    return max(1u, min(ctx.D, ctx.K));
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
    tail -= ctx.D;
    if (tail < ctx.K) {
        float w = ctx.keyboard[tail];
        if (!isfinite(w)) return 0.0f;
        if ((ctx.keyboardOrigin[0] % 3u) == 2u) {
            float norm = 0.0f;
            for (uint k = 0u; k < ctx.K; ++k) {
                float wk = ctx.keyboard[k];
                if (isfinite(wk)) norm += fabs(wk);
            }
            if (norm > float(ctx.K * ctx.K)) return w / norm;
        }
        return w;
    }
    if (tail == ctx.K) return ctx.reward[0];
    return 1.0f;
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

inline void fcstomp_record_keyboard_stats(Ctx ctx,
                                          uint selected,
                                          float termSum,
                                          float keyL1,
                                          float advL1,
                                          float effectiveL1) {
    uint sOpt = selected % max(ctx.K, 1u);
    float sDur = ctx.durationHat[sOpt];
    float sPrio = ctx.prioOpt[sOpt];
    ctx.stats->terminationRate = termSum * ctx.invK;
    ctx.stats->selectedOptionAge = float(ctx.optAge[sOpt]);
    ctx.stats->selectedOptionDuration = isfinite(sDur) ? sDur : 0.0f;
    ctx.stats->selectedOptionPriority = isfinite(sPrio) ? sPrio : 0.0f;
    ctx.stats->keyboardL1 = keyL1;
    ctx.stats->keyboardEffectiveL1 = effectiveL1;
    ctx.stats->actorAdvantageL1 = advL1;
    ctx.stats->selectedOptionIndex = float(sOpt);
    ctx.stats->keyboardOrigin = float(ctx.keyboardOrigin[0] % 3u);
    ctx.stats->selectedOptionFeature = float(ctx.optFeature[sOpt] % ctx.R);
    ctx.stats->selectedContinuation = ctx.continuation[sOpt];
    ctx.stats->selectedKeyboardWeight = isfinite(ctx.keyboard[sOpt]) ? ctx.keyboard[sOpt] : 0.0f;
    ctx.stats->mainTd = ctx.td[0];
    ctx.stats->planTdValue = ctx.planTd[0];
    ctx.stats->keyboardSketchDim = float(keyboard_sketch_dim(ctx));
}

inline void fcstomp_prediction_control_loop(Ctx ctx, uint tid,
                                      threadgroup float *tgPrev,
                                      threadgroup float *tgCurr,
                                      threadgroup float *tgG,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgTerm,
                                      threadgroup float *tgTdPsi,
                                      threadgroup float *tgDir,
                                      threadgroup float *tgEffort,
                                      threadgroup float *tgScalars) {
    // GVFPrediction: reward-centered differential value and shared trace scales.
    float vPrev = 0.0f, vCurr = 0.0f;
    if (tid == 0u) {
        for (uint i = 0u; i < ctx.R; ++i) {
            vPrev = fma(ctx.valueW[i], tgPrev[i], vPrev);
            vCurr = fma(ctx.valueW[i], tgCurr[i], vCurr);
        }
        if (OAK_ABLATE_GVF_PREDICTION) {
            vPrev = 0.0f;
            vCurr = 0.0f;
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
        float delta = OAK_ABLATE_GVF_PREDICTION ? 0.0f : (rewardC + vCurrG - vPrevG);
        if (!isfinite(delta)) { delta = 0.0f; ctx.counters[OAK_CNT_LEARN_NF] += 1u; }
        ctx.td[0] = delta;
        float rhoNew = rhoOld + ctx.invR * delta;
        float rhoUse = (!OAK_ABLATE_GVF_PREDICTION && isfinite(rhoNew)) ? rhoNew : rhoOld;
        ctx.scalars->rho = rhoUse;
        tgScalars[11] = rhoUse;
    }
    if (tid == 0u) {
        float alphaTraceNorm = 0.0f;
        float modelTraceNorm = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float x = tgPrev[i];
            if (!isfinite(x)) continue;
            float xx = x * x;
            float av = ctx.alpha[i];
            float am = ctx.alphaModel[i];
            if (isfinite(av) && av > 0.0f) alphaTraceNorm = fma(av, xx, alphaTraceNorm);
            if (isfinite(am) && am > 0.0f) modelTraceNorm = fma(am, xx, modelTraceNorm);
        }
        tgScalars[7] = (isfinite(alphaTraceNorm) && alphaTraceNorm > 0.0f) ? alphaTraceNorm : 0.0f;
        tgScalars[10] = (isfinite(modelTraceNorm) && modelTraceNorm > 0.0f) ? modelTraceNorm : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RewardRespectingSubtasks + OptionModels: RR continuation and model-backed option query.
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
        bool rrFinite = isfinite(psiPrev) && isfinite(psiNext) && isfinite(zStop);
        if (!rrFinite) ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop >= psiNext) ? 1.0f : 0.0f;
        if (OAK_ABLATE_RR_SUBTASKS) {
            rrFinite = true;
            beta = 1.0f;
        }
        float cont = 1.0f - beta;
        ctx.continuation[tid] = cont;
        float dPsi = (!OAK_ABLATE_RR_SUBTASKS && rrFinite)
            ? (rewardC + cont * psiNext + beta * zStop - psiPrev) : 0.0f;
        if (!isfinite(dPsi)) {
            dPsi = 0.0f;
            ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        }
        ctx.tdPsi[tid] = dPsi;
        tgTdPsi[tid] = dPsi;
        // Per-option learning-progress tracker (Structural Addition A).
        // Fast EMA of |dPsi| with R-step memory; peak as pure all-time max.
        // Maturity predicate (fast < peak) means recent error has improved
        // past the historical high-water mark — i.e., learning has happened.
        // Both quantities scale-free; only intrinsic constant is invR.
        float absD = isfinite(dPsi) ? fabs(dPsi) : 0.0f;
        float fastOld = ctx.tdPsiFast[tid];
        float peakOld = ctx.tdPsiPeak[tid];
        ctx.tdPsiFast[tid] = (1.0f - ctx.invR) * (isfinite(fastOld) ? fastOld : 0.0f) + ctx.invR * absD;
        ctx.tdPsiPeak[tid] = max(isfinite(peakOld) ? peakOld : 0.0f, absD);
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float implicitTrace = psiXX - cont * psiXY;
        float denom = fma(rho, implicitTrace, 1.0f);
        ctx.headDenom[tid] = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        float modelDenom = fma(rho, tgScalars[10], 1.0f);
        ctx.modelDenom[tid] = (isfinite(modelDenom) && modelDenom > 0.0f) ? modelDenom : 0.0f;
        tgCont[tid] = cont;
        tgTerm[tid] = beta;
        float rHat = 0.0f, lRaw = 0.0f, vModel = 0.0f, direction = 0.0f;
        if (!OAK_ABLATE_OPTION_MODELS) {
            for (uint j = 0u; j < ctx.R; ++j) {
                uint idx = tid * ctx.R + j;
                rHat = fma(ctx.optReward[idx], tgCurr[j], rHat);
                lRaw = fma(ctx.optDuration[idx], tgCurr[j], lRaw);
                float nHat = 0.0f;
                uint nBase = tid * ctx.R * ctx.R + j * ctx.R;
                for (uint i = 0u; i < ctx.R; ++i) nHat = fma(ctx.optTransition[nBase + i], tgCurr[i], nHat);
                vModel = fma(ctx.valueW[j], nHat, vModel);
                direction = fma(nHat - tgCurr[j], tgCurr[j] - tgPrev[j], direction);
            }
        }
        bool modelFinite = isfinite(rHat) && isfinite(lRaw) && isfinite(vModel);
        float dur = decode_duration(lRaw);
        bool durationTrusted = usable_duration(ctx, dur);
        bool modelTrusted = OAK_ABLATE_SEARCH_CONTROL ? modelFinite : usable_priority(ctx.prioOpt[tid]);
        float q = rHat - tgScalars[11] * dur + vModel;
        float qAdv = q - vCurr;
        bool queryTrusted = !OAK_ABLATE_OPTION_MODELS
            && !OAK_ABLATE_PLANNING
            && rrFinite && modelTrusted && modelFinite && durationTrusted && isfinite(qAdv);
        ctx.durationHat[tid] = dur;
        tgG[tid] = (queryTrusted && !OAK_ABLATE_MODEL_TO_KEYBOARD) ? qAdv : 0.0f;
        tgDir[tid] = (queryTrusted && isfinite(direction) && !OAK_ABLATE_MODEL_TO_KEYBOARD)
            ? direction : 0.0f;
        float effort = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) {
            float mu = ctx.optionPolicyMean[tid * ctx.D + d];
            if (isfinite(mu)) effort = fma(mu, mu, effort);
        }
        tgEffort[tid] = effort * (1.0f / float(max(ctx.D, 1u)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // OptionPolicies: centered RR advantages drive the option actor heads.
    if (tid == 0u) {
        float invK = ctx.invK;
        float gaugeRate = track_rate(ctx.stepCount, ctx.R);
        float gaugeNew = ctx.valueGauge[0] + gaugeRate * (vCurr - ctx.valueGauge[0]);
        ctx.valueGauge[0] = safe_w(ctx.valueGauge[0], gaugeNew);
        float actorBaseline = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            actorBaseline += tgTdPsi[k];
        }
        actorBaseline *= invK;
        for (uint k = 0u; k < ctx.K; ++k) {
            float adv = OAK_ABLATE_OPTION_POLICIES ? 0.0f : (tgTdPsi[k] - actorBaseline);
            if (!isfinite(adv)) adv = 0.0f;
            tgTdPsi[k] = adv;
            ctx.actorAdv[k] = adv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // OptionKeyboard + Planning: GPI-style live-option selection and value pressure.
    if (tid == 0u) {
        uint best = 0u;
        float bestScore = -FLT_MAX, bestQ = 0.0f;
        float meanQ = 0.0f, maxScorePrio = 0.0f;
        float termSum = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            termSum += tgTerm[k];
            float qAdv = tgG[k];
            if (!isfinite(qAdv)) qAdv = 0.0f;
            float dur = ctx.durationHat[k];
            if (!isfinite(dur) || dur <= 0.0f) dur = 1.0f;
            float modelErr = OAK_ABLATE_SEARCH_CONTROL ? 0.0f : ctx.prioOpt[k];
            bool usableModel = !OAK_ABLATE_MODEL_TO_KEYBOARD
                && (OAK_ABLATE_SEARCH_CONTROL || usable_priority(modelErr))
                && usable_duration(ctx, dur);
            float score = usableModel ? ((qAdv - modelErr) / dur) : -FLT_MAX;
            meanQ += (usableModel && isfinite(score)) ? score : 0.0f;
            if (usableModel) maxScorePrio = max(maxScorePrio, modelErr);
            if (score > bestScore) { bestScore = score; bestQ = qAdv; best = k; }
        }
        uint selected = best;
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        bool continuing = tgCont[prevSel] > 0.0f;
        meanQ *= ctx.invK;
        if (continuing) selected = prevSel;
        float prevMag = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            if (isfinite(w)) prevMag += fabs(w);
        }
        bool hasKeyboardCommitment = prevMag > 0.0f;
        float actorMass = 0.0f;
        float maxDur = 0.0f;
        float impSum = 0.0f, impSq = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float a = ctx.actorAdv[k];
            if (isfinite(a)) actorMass += fabs(a);
            float imp = ctx.importance[k];
            if (isfinite(imp) && imp > 0.0f) {
                impSum += imp;
                impSq = fma(imp, imp, impSq);
            }
            float d = ctx.durationHat[k];
            if (isfinite(d)) maxDur = max(maxDur, d);
        }
        ctx.stats->importanceEss = (impSq > 0.0f) ? ((impSum * impSum) / (float(ctx.K) * impSq)) : 0.0f;
        bool durationGrounded = maxDur <= float(ctx.stepCount + 1u);
        bool deadCreditRefresh = termSum <= 0.0f && actorMass <= 0.0f && durationGrounded;
        float prevDur = ctx.durationHat[prevSel];
        if (!isfinite(prevDur) || prevDur <= 0.0f) prevDur = 1.0f;
        bool negativeRate = ctx.scalars->rho < -ctx.invR;
        bool surprising = maxScorePrio > float(ctx.K * ctx.K);
        bool broadTermination = termSum * termSum > float(ctx.K);
        bool activeShortKeyboard = negativeRate && surprising && broadTermination
            && prevMag < float(ctx.K) && prevDur < float(ctx.K);
        uint origin = ctx.keyboardOrigin[0] % 3u;
        bool lowMassKeyboard = prevMag < float(ctx.K);
        bool finiteQuietBlowup = hasKeyboardCommitment
            && prevMag > float(ctx.K * ctx.K * ctx.K)
            && maxScorePrio > prevMag
            && prevDur > float(ctx.K)
            && termSum <= 1.0f
            && ctx.scalars->rho < 0.0f;
        bool selectedStale = ctx.scalars->rho < 0.0f
            && termSum <= 0.0f
            && actorMass <= 0.0f
            && hasKeyboardCommitment
            && origin == 2u;
        bool selectedInvalid = selectedStale
            || (!OAK_ABLATE_SEARCH_CONTROL && !usable_priority(ctx.prioOpt[prevSel]));
        bool invalidQuietRescue = selectedInvalid && continuing && hasKeyboardCommitment
            && termSum <= 1.0f && actorMass <= 0.0f;
        bool staleQuietRescue = selectedStale && continuing && hasKeyboardCommitment
            && termSum <= 1.0f;
        bool ungroundedInvalidCommitment = selectedInvalid && continuing && hasKeyboardCommitment
            && prevMag > float(ctx.stepCount + 1u);
        bool quietPath = (finiteQuietBlowup || invalidQuietRescue || staleQuietRescue
            || ungroundedInvalidCommitment) && !activeShortKeyboard;
        bool relativeSupportLocked = origin == 2u
            && negativeRate
            && (ctx.td[0] > 0.0f || broadTermination)
            && ctx.stats->importanceEss <= 2.0f * ctx.invK
            && hasKeyboardCommitment
            && !quietPath
            && !activeShortKeyboard;
        if (relativeSupportLocked) selected = prevSel;
        float keyL1Pre = 0.0f, advL1Pre = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            float a = ctx.actorAdv[k];
            if (isfinite(w)) keyL1Pre += fabs(w);
            if (isfinite(a)) advL1Pre += fabs(a);
        }
        if (OAK_ABLATE_OPTION_KEYBOARD) {
            for (uint k = 0u; k < ctx.K; ++k) ctx.keyboard[k] = 0.0f;
            ctx.keyboardOrigin[0] = 0u;
            ctx.planTd[0] = 0.0f;
            ctx.selectedOption[0] = selected;
            fcstomp_record_keyboard_stats(ctx, selected, termSum, 0.0f, advL1Pre, 0.0f);
            return;
        }
        bool quietRescue = (lowMassKeyboard && maxScorePrio > float(ctx.K * ctx.K) && termSum <= 1.0f)
            || quietPath;
        bool refreshKeyboard = !continuing || prevMag <= 0.0f || deadCreditRefresh
            || (continuing && activeShortKeyboard) || quietPath;
        if (relativeSupportLocked) refreshKeyboard = false;
        bool rawEligible = refreshKeyboard && hasKeyboardCommitment
            && (activeShortKeyboard || (!quietPath && origin == 1u && !lowMassKeyboard));
        bool relativeEligible = refreshKeyboard && !rawEligible
            && hasKeyboardCommitment && (quietRescue || origin == 2u);
        float bankEffort = 0.0f;
        float rawMean = 0.0f, relativeMean = 0.0f;
        float rawAbs = 0.0f, relativeAbs = 0.0f;
        if (rawEligible) {
            for (uint k = 0u; k < ctx.K; ++k) {
                float effort = isfinite(tgEffort[k]) ? tgEffort[k] : 0.0f;
                float rawScore = tgDir[k] / max(effort + ctx.invR, ctx.invR);
                rawMean += isfinite(rawScore) ? rawScore : 0.0f;
            }
            rawMean *= ctx.invK;
            for (uint k = 0u; k < ctx.K; ++k) {
                float effort = isfinite(tgEffort[k]) ? tgEffort[k] : 0.0f;
                float rawScore = tgDir[k] / max(effort + ctx.invR, ctx.invR);
                rawAbs += isfinite(rawScore) ? fabs(rawScore - rawMean) : 0.0f;
            }
            rawAbs *= ctx.invK;
        } else if (relativeEligible) {
            for (uint k = 0u; k < ctx.K; ++k) {
                float effort = tgEffort[k];
                if (isfinite(effort)) bankEffort += effort;
            }
            bankEffort *= ctx.invK;
            for (uint k = 0u; k < ctx.K; ++k) {
                float effort = isfinite(tgEffort[k]) ? tgEffort[k] : 0.0f;
                float relScore = tgDir[k] / max(bankEffort + effort + ctx.invR, ctx.invR);
                relativeMean += isfinite(relScore) ? relScore : 0.0f;
            }
            relativeMean *= ctx.invK;
            for (uint k = 0u; k < ctx.K; ++k) {
                float effort = isfinite(tgEffort[k]) ? tgEffort[k] : 0.0f;
                float relScore = tgDir[k] / max(bankEffort + effort + ctx.invR, ctx.invR);
                relativeAbs += isfinite(relScore) ? fabs(relScore - relativeMean) : 0.0f;
            }
            relativeAbs *= ctx.invK;
        }
        bool useRaw = rawEligible && rawAbs > ctx.invR;
        bool useRelative = relativeEligible && relativeAbs > ctx.invR;
        float mag = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float qAdv = tgG[k];
            if (!isfinite(qAdv)) qAdv = 0.0f;
            float modelErr = OAK_ABLATE_SEARCH_CONTROL ? 0.0f : ctx.prioOpt[k];
            float dur = ctx.durationHat[k];
            if (!isfinite(dur) || dur <= 0.0f) dur = 1.0f;
            bool queryTrusted = !OAK_ABLATE_MODEL_TO_KEYBOARD
                && (OAK_ABLATE_SEARCH_CONTROL || usable_priority(modelErr))
                && usable_duration(ctx, ctx.durationHat[k]);
            float lam = (queryTrusted && !quietPath) ? ((qAdv - modelErr) / dur - meanQ) : 0.0f;
            if (useRaw || useRelative) {
                float effort = isfinite(tgEffort[k]) ? tgEffort[k] : 0.0f;
                float denom = useRaw ? max(effort + ctx.invR, ctx.invR)
                                     : max(bankEffort + effort + ctx.invR, ctx.invR);
                float dScore = tgDir[k] / denom;
                if (!isfinite(dScore)) dScore = 0.0f;
                lam = useRaw ? (dScore - rawMean) : (dScore - relativeMean);
            }
            if (!isfinite(lam)) lam = 0.0f;
            if (!queryTrusted) {
                float incumbent = ctx.keyboard[k];
                lam = (refreshKeyboard && hasKeyboardCommitment && !quietPath && isfinite(incumbent))
                    ? incumbent : 0.0f;
            }
            if (refreshKeyboard) ctx.keyboard[k] = lam;
            mag += fabs(lam);
        }
        if (refreshKeyboard) {
            uint nextOrigin = useRaw ? 1u : (useRelative ? 2u : 0u);
            ctx.keyboardOrigin[0] = nextOrigin;
        }
        uint probe = ctx.stepCount % max(ctx.K, 1u);
        float floor = ctx.invK;
        if (refreshKeyboard && mag <= 0.0f) {
            for (uint k = 0u; k < ctx.K; ++k) {
                float probeSign = (k == probe) ? 1.0f : -ctx.invK;
                ctx.keyboard[k] = probeSign * floor;
            }
            ctx.keyboardOrigin[0] = 0u;
        }
        float planNum = 0.0f, planNorm = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            float qAdv = tgG[k];
            if (!isfinite(w)) w = 0.0f;
            if (!isfinite(qAdv)) qAdv = 0.0f;
            planNum = fma(w, qAdv, planNum);
            planNorm += fabs(w);
        }
        float planDelta = (!OAK_ABLATE_PLANNING && planNorm > 0.0f) ? (planNum / planNorm) : bestQ;
        if (OAK_ABLATE_PLANNING) planDelta = 0.0f;
        ctx.planTd[0] = isfinite(planDelta) ? planDelta : 0.0f;
        ctx.selectedOption[0] = selected;
        float keyL1 = 0.0f, advL1 = 0.0f, effectiveL1 = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            float a = ctx.actorAdv[k];
            if (isfinite(w)) keyL1 += fabs(w);
            if (isfinite(a)) advL1 += fabs(a);
        }
        effectiveL1 = keyL1 > 0.0f ? 1.0f : 0.0f;
        fcstomp_record_keyboard_stats(ctx, selected, termSum, keyL1, advL1, effectiveL1);

    }
}

inline void FeatureConstruction(Ctx ctx,
                                uint tid,
                                uint tgW,
                                threadgroup float *tgPrevState,
                                threadgroup float *tgCurrState) {
    uint nf = 0u;
    float obsRate = track_rate(ctx.stepCount, ctx.O);
    if (!OAK_ABLATE_FEATURE_CONSTRUCTION) {
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
    }
    if (nf > 0u) ctx.counters[OAK_CNT_OBS_NF] += nf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float raw = 0.0f;
        bool directBias = false;
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            directBias = (i == 0u);
        } else if (i < ctx.anchorCount) {
            uint width = max(linear_input_width(ctx), 1u);
            uint channel = (i == 0u)
                ? width - 1u
                : ((i * max(width - 1u, 1u)) / max(ctx.anchorCount, 1u)) % max(width - 1u, 1u);
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
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            curr[i] = 0.0f;
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
}

kernel void oak_encode(device float *params [[buffer(0)]],
                       device float *stateArena [[buffer(1)]],
                       device uint *uintArena [[buffer(2)]],
                       device KernelScalars *scalars [[buffer(3)]],
                       device KernelStats *stats [[buffer(4)]],
                       constant KernelConfig &cfg [[buffer(5)]],
                       threadgroup float *tgG [[threadgroup(0)]],
                       threadgroup float *tgCont [[threadgroup(1)]],
                       threadgroup float *tgTerm [[threadgroup(2)]],
                       threadgroup float *tgTdPsi [[threadgroup(3)]],
                       threadgroup float *tgScalars [[threadgroup(4)]],
                       threadgroup float *tgPrevState [[threadgroup(5)]],
                       threadgroup float *tgCurrState [[threadgroup(6)]],
                       threadgroup float *tgDir [[threadgroup(7)]],
                       threadgroup float *tgEffort [[threadgroup(8)]],
                       uint2 tpos [[thread_position_in_threadgroup]],
                       uint2 tdim [[threads_per_threadgroup]],
                       uint2 tgPos [[threadgroup_position_in_grid]]) {
    uint tid = tpos.x, tgW = tdim.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    FeatureConstruction(ctx, tid, tgW, tgPrevState, tgCurrState);
    fcstomp_prediction_control_loop(ctx, tid, tgPrevState, tgCurrState, tgG, tgCont, tgTerm,
                                    tgTdPsi, tgDir, tgEffort, tgScalars);
}

kernel void oak_plan(device float *params [[buffer(0)]],
                     device float *stateArena [[buffer(1)]],
                     device uint *uintArena [[buffer(2)]],
                     device KernelScalars *scalars [[buffer(3)]],
                     device KernelStats *stats [[buffer(4)]],
                     constant KernelConfig &cfg [[buffer(5)]],
                     threadgroup float *tgModelErr [[threadgroup(0)]],
                     uint2 jpos [[thread_position_in_threadgroup]],
                     uint2 tgPos [[threadgroup_position_in_grid]]) {
    uint j = jpos.x, k = tgPos.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    if (k >= ctx.K) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    bool active = j < ctx.R;
    float cont = ctx.continuation[k];
    // OptionModels + SearchControl: learn option reward/transition/duration and priority debt.
    if (OAK_ABLATE_OPTION_MODELS) {
        if (active) ctx.nTdMat[k * ctx.R + j] = 0.0f;
        if (j == 0u) {
            ctx.optRewardTd[k] = 0.0f;
            ctx.durationTd[k] = 0.0f;
            ctx.durationHat[k] = 1.0f;
            ctx.prioOpt[k] = 0.0f;
        }
        return;
    }

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
        if (!isfinite(nTd)) {
            nTd = 0.0f;
            ctx.counters[OAK_CNT_PLAN_NF] += 1u;
        }
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
        bool modelFinite = isfinite(nErr) && isfinite(rTd) && isfinite(dTd) && isfinite(lRaw);
        if (!isfinite(rTd)) rTd = 0.0f;
        if (!isfinite(dTd)) dTd = 0.0f;
        ctx.optRewardTd[k] = rTd;
        ctx.durationTd[k] = dTd;
        float dur = decode_duration(lRaw);
        ctx.durationHat[k] = dur;
        float modelErr = nErr + rTd * rTd + (ctx.scalars->rho * dTd) * (ctx.scalars->rho * dTd);
        if (OAK_ABLATE_SEARCH_CONTROL) {
            ctx.prioOpt[k] = 0.0f;
        } else if (modelFinite && isfinite(modelErr)) {
            ctx.prioOpt[k] = sqrt(max(modelErr, 0.0f));
        } else {
            float debt = fabs(ctx.prioOpt[k]);
            if (!isfinite(debt) || debt <= 0.0f) debt = 1.0f;
            ctx.prioOpt[k] = -debt;
            ctx.counters[OAK_CNT_PLAN_NF] += 1u;
        }
    }
}

kernel void oak_learn_heads(device float *params [[buffer(0)]],
                            device float *stateArena [[buffer(1)]],
                            device uint *uintArena [[buffer(2)]],
                            device KernelScalars *scalars [[buffer(3)]],
                            device KernelStats *stats [[buffer(4)]],
                            constant KernelConfig &cfg [[buffer(5)]],
                            uint2 gpos [[thread_position_in_grid]]) {
    uint gid = gpos.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gpos.y);
    // Uniform learning: value, GVFs, option policies, and option models share one stream.
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    uint nVal = ctx.R;
    uint nPsi = ctx.K * ctx.R;
    uint nPi = ctx.K * ctx.D * ctx.R;
    uint nOR = ctx.K * ctx.R;
    uint nOT = ctx.K * ctx.R * ctx.R;
    uint nOD = ctx.K * ctx.R;
    uint total = nVal + nPsi + nPi + nOR + nOT + nOD;
    if (gid >= total) return;

    if (gid < nVal) {
        if (OAK_ABLATE_GVF_PREDICTION) return;
        uint i = gid;
        float w = ctx.valueW[i];
        float delta = ctx.td[0] * prev[i] + ctx.planTd[0] * curr[i];
        float denom = 1.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            float a = ctx.alpha[j];
            if (isfinite(a) && a > 0.0f) denom += a * (prev[j] * prev[j] + curr[j] * curr[j]);
        }
        float updateEnergy = ctx.td[0] * ctx.td[0] + ctx.planTd[0] * ctx.planTd[0];
        if (isfinite(updateEnergy)) denom += ctx.invR * updateEnergy;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta * scale);
        return;
    }
    gid -= nVal;
    if (gid < nPsi) {
        if (OAK_ABLATE_RR_SUBTASKS) return;
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
    if (gid < nPi) {
        if (OAK_ABLATE_OPTION_POLICIES) return;
        uint k = gid / (ctx.D * ctx.R);
        uint rem = gid - k * ctx.D * ctx.R;
        uint d = rem / ctx.R, i = rem % ctx.R;
        bool relativeSupportCollapsed = (ctx.keyboardOrigin[0] % 3u) == 2u
            && ctx.stats->importanceEss <= ctx.invK;
        if (relativeSupportCollapsed) return;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float w = ctx.optPolicy[(k * ctx.D + d) * ctx.R + i];
        float targetGrad = ctx.optionActorCredit[k * ctx.D + d];
        if (!isfinite(targetGrad)) targetGrad = 0.0f;
        float actorAlpha = 1.0f / float(max(ctx.D * ctx.R, 1u));
        float gradScale = rho * targetGrad;
        float xNorm = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) xNorm = fma(prev[j], prev[j], xNorm);
        float denom = 1.0f + actorAlpha * ctx.actorAdv[k] * ctx.actorAdv[k] * gradScale * gradScale * xNorm;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] =
            safe_w(w, w + actorAlpha * ctx.actorAdv[k] * gradScale * prev[i] * scale);
        return;
    }
    gid -= nPi;
    if (gid < nOR) {
        if (OAK_ABLATE_OPTION_MODELS) return;
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
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / (ctx.R * ctx.R);
        uint rem = gid - k * ctx.R * ctx.R;
        uint j = rem / ctx.R, i = rem % ctx.R;
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
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / ctx.R, i = gid % ctx.R;
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.modelDenom[k];
        float dEnergy = ctx.durationTd[k] * ctx.durationTd[k];
        if (isfinite(dEnergy)) denom += ctx.invR * dEnergy;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optDuration[k * ctx.R + i];
        ctx.optDuration[k * ctx.R + i] = safe_w(w, w + ctx.alphaModel[i] * rho * ctx.durationTd[k] * prev[i] * scale);
        return;
    }
}

inline void emit_action_grid(Ctx ctx, threadgroup const float *curr, uint group, uint tid, uint tgW) {
    // ActionEmission: expose option policy means plus the wrapper-facing behavior descriptor.
    uint gid = group * tgW + tid;
    if (gid >= ctx.K * ctx.D) return;
    uint selected = ctx.selectedOption[0] % ctx.K;
    uint k = gid / ctx.D, d = gid % ctx.D;
    if (OAK_ABLATE_OPTION_POLICIES) {
        ctx.optionPolicyMean[k * ctx.D + d] = 0.0f;
        if (gid < ctx.D) ctx.policyMean[d] = 0.0f;
        return;
    }
    float mu = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i)
        mu = fma(ctx.optPolicy[(k * ctx.D + d) * ctx.R + i], curr[i], mu);
    if (!isfinite(mu)) mu = ctx.optionPolicyMean[k * ctx.D + d];
    ctx.optionPolicyMean[k * ctx.D + d] = mu;
    if (gid < ctx.D) {
        float blend = 0.0f, wNorm = 0.0f, fallback = 0.0f;
        for (uint ok = 0u; ok < ctx.K; ++ok) {
            float om = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i)
                om = fma(ctx.optPolicy[(ok * ctx.D + d) * ctx.R + i], curr[i], om);
            if (!isfinite(om)) om = 0.0f;
            if (ok == selected) fallback = om;
            float w = ctx.keyboard[ok];
            if (!isfinite(w)) w = 0.0f;
            blend = fma(w, om, blend);
            wNorm += fabs(w);
        }
        ctx.policyMean[d] = (wNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
            ? blend / wNorm : fallback;
    }
}

kernel void oak_replace(device float *params [[buffer(0)]],
                        device float *stateArena [[buffer(1)]],
                        device uint *uintArena [[buffer(2)]],
                        device KernelScalars *scalars [[buffer(3)]],
                        device KernelStats *stats [[buffer(4)]],
                        constant KernelConfig &cfg [[buffer(5)]],
                        threadgroup float *tgCurrState [[threadgroup(0)]],
                        uint2 tpos [[thread_position_in_threadgroup]],
                        uint2 tdim [[threads_per_threadgroup]],
                        uint2 tgPos [[threadgroup_position_in_grid]]) {
    uint tid = tpos.x, tgW = tdim.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    // GenerateAndTest: bundle utility, replacement, and FC-STOMP health diagnostics.
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    for (uint i = tid; i < ctx.R; i += tgW) tgCurrState[i] = curr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invT = track_rate(ctx.stepCount, ctx.R);
    for (uint i = tid; i < ctx.R; i += tgW) {
        float psiSignal = 0.0f, modelSignal = 0.0f, keyboardSignal = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            if ((ctx.optFeature[k] % ctx.R) != i) continue;
            float psiUse = fabs(ctx.tdPsi[k]);
            float keyMass = fabs(ctx.keyboard[k]);
            float modelUse = OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : keyMass * fabs(ctx.planTd[0]);
            float keyUse = OAK_ABLATE_SEARCH_TO_GENERATE_TEST
                ? 0.0f : keyMass * (fabs(ctx.actorAdv[k]) + max(ctx.importance[k], 0.0f));
            if (isfinite(psiUse)) psiSignal += psiUse;
            if (isfinite(modelUse)) modelSignal += modelUse;
            if (isfinite(keyUse)) keyboardSignal += keyUse;
        }
        float planningSignal = fabs(ctx.td[0]) + fabs(ctx.planTd[0]);
        float g = fabs(planningSignal * prev[i])
                + fabs((psiSignal + modelSignal + keyboardSignal) * prev[i]);
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
        float err = ctx.prioOpt[k];
        if (!isfinite(err) || OAK_ABLATE_SEARCH_CONTROL) err = 0.0f;
        float keyMass = OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(ctx.keyboard[k]);
        float u = ctx.tdPsi[k] * ctx.tdPsi[k]
                + keyMass * (fabs(ctx.planTd[0]) + fabs(ctx.actorAdv[k]))
                - err;
        ctx.utilOpt[k] += invT * (u - ctx.utilOpt[k]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid != 0u) return;
	    float wSum = 0.0f, wSq = 0.0f;
	    for (uint k = 0u; k < ctx.K; ++k) {
	        float w = ctx.importance[k];
	        if (!isfinite(w)) { w = 0.0f; ctx.counters[OAK_CNT_IMPORTANCE_NF] += 1u; }
	        wSum += w; wSq += w * w;
	    }
	    float invK = ctx.invK;
	    ctx.stats->importanceEss = (wSq > 0.0f) ? ((wSum * wSum) / (float(ctx.K) * wSq)) : 0.0f;
	    float actorEntropy = 0.0f;
	    if (wSum > 0.0f) {
	        for (uint k = 0u; k < ctx.K; ++k) {
	            float p = max(ctx.importance[k], 0.0f) / wSum;
	            if (p > 0.0f) actorEntropy -= p * log(p);
	        }
		    }
		    ctx.stats->actorSupportEntropy = (ctx.K > 1u) ? actorEntropy / log(float(ctx.K)) : 0.0f;
				    float actorUpdateSum = 0.0f;
						    float actorAlpha = 1.0f / float(max(ctx.D * ctx.R, 1u));
				    for (uint k = 0u; k < ctx.K; ++k) {
				        float u = fabs(actorAlpha * ctx.importance[k] * ctx.actorAdv[k]);
				        if (!isfinite(u)) u = 0.0f;
				        actorUpdateSum += u;
				    }
				    ctx.stats->actorUpdateNormMean = actorUpdateSum * invK;
    float durSum = 0.0f;
    float maxPrio = 0.0f;
    float searchUse = 0.0f;
    float keyNorm = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float w = ctx.keyboard[k];
        if (isfinite(w)) keyNorm += fabs(w);
    }
    for (uint k = 0u; k < ctx.K; ++k) {
        float d = ctx.durationHat[k];
        float p = ctx.prioOpt[k];
        float w = ctx.keyboard[k];
        if (!isfinite(d)) d = 0.0f;
        if (!isfinite(p)) p = 0.0f;
        if (!isfinite(w)) w = 0.0f;
        durSum += d;
        maxPrio = max(maxPrio, p);
        if (keyNorm > 0.0f) searchUse += max(p, 0.0f) * fabs(w) / keyNorm;
    }
    ctx.stats->optionDurationMean = durSum * invK;
    ctx.stats->modelPriorityMax = maxPrio;
    ctx.stats->searchControlUseMean = searchUse * invK;

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
            float optionUse = 0.0f;
            for (uint k = 0u; k < ctx.K; ++k) {
                if ((ctx.optFeature[k] % ctx.R) != i) continue;
                float use = ctx.utilOpt[k]
                    + (OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(ctx.keyboard[k]));
                if (isfinite(use)) optionUse += use;
            }
		            float downstreamUse = optionUse + fabs(ctx.valueW[i]) + fabs(ctx.bonusW[i]);
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
	    for (uint k = 0u; k < ctx.K; ++k) {
	        if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;
	    }
	    for (uint i = 0u; i < ctx.R; ++i) if (ctx.featureAge[i] < 0xffffffffu) ctx.featureAge[i] += 1u;

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRed = eligibleRedSum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtil < meanEligibleUtil;
    bool highRedundancy = worstRed > meanEligibleRed;
    bool lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
			    bool structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);
    ctx.stats->generateTestEligibleCount = float(eligible);
    for (uint i = candidateStart; i < ctx.R; ++i) {
        if (structurallyReplaceable && i == worst) {
            if (ctx.badFeatureAge[i] < 0xffffffffu) ctx.badFeatureAge[i] += 1u;
        } else {
            ctx.badFeatureAge[i] = 0u;
        }
    }
		    uint replaceAge = ctx.R * max(eligible, 1u);
		    bool doReplace = !OAK_ABLATE_GENERATE_AND_TEST
            && structurallyReplaceable && worst < ctx.R && ctx.badFeatureAge[worst] >= replaceAge;
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
	        ctx.alpha[worst] = ctx.invR;
	        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f; ctx.gradEnergy[worst] = 0.0f;
	        ctx.alphaPsi[worst] = ctx.invR;
	        ctx.betaPsi[worst] = log(ctx.alphaPsi[worst]); ctx.hPsi[worst] = 0.0f; ctx.gradEnergyPsi[worst] = 0.0f;
		        ctx.alphaModel[worst] = ctx.invR;
	        ctx.betaModel[worst] = log(ctx.alphaModel[worst]); ctx.hModel[worst] = 0.0f; ctx.gradEnergyModel[worst] = 0.0f;
        uint k = 0u;
        float worstBasis = FLT_MAX;
        for (uint ok = 0u; ok < ctx.K; ++ok) {
            float use = ctx.utilOpt[ok] + ctx.prioOpt[ok];
            if (!isfinite(use)) use = FLT_MAX;
            if (ctx.optFeature[ok] == worst) { k = ok; worstBasis = -FLT_MAX; break; }
            if (use < worstBasis) { worstBasis = use; k = ok; }
        }
				        ctx.optAge[k] = 0u; ctx.optFeature[k] = worst; ctx.utilOpt[k] = float(ctx.R);
		        ctx.actorAdv[k] = 0.0f;
			        for (uint i = 0u; i < ctx.R; ++i) {
			            ctx.psi[k * ctx.R + i] = 0.0f;
			            ctx.optReward[k * ctx.R + i] = 0.0f;
		            ctx.optDuration[k * ctx.R + i] = 0.0f;
		            for (uint d = 0u; d < ctx.D; ++d) ctx.optPolicy[(k * ctx.D + d) * ctx.R + i] = 0.0f;
		            for (uint j = 0u; j < ctx.R; ++j) ctx.optTransition[k * ctx.R * ctx.R + j * ctx.R + i] = 0.0f;
	        }
			        if (ctx.D > 0u) {
			            uint slot = k % ctx.D;
			            float sign = ((k / ctx.D) % 2u == 0u) ? 1.0f : -1.0f;
			            ctx.optPolicy[(k * ctx.D + slot) * ctx.R] = sign / float(ctx.D);
			        }
		        float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
		        ctx.psi[k * ctx.R + worst] = invSqrtR;
	        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
	        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
	    }
    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount = float(ctx.counters[OAK_CNT_TDPSI_NF] + ctx.counters[OAK_CNT_IMPORTANCE_NF]);
    ctx.stats->oakOptionReinitCount = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);
    // Aggregate learning-progress diagnostic: mean over options of
    //   1 - fast[k] / peak[k]
    // where fast is the R-step EMA of |dPsi[k]| and peak is its all-time max.
    // 0 means current error == peak (no progress); 1 means current error == 0
    // (fully converged). Continuous, no threshold.
    float progressSum = 0.0f;
    uint progressCount = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float peak = ctx.tdPsiPeak[k];
        float fast = ctx.tdPsiFast[k];
        if (isfinite(peak) && peak > 0.0f && isfinite(fast)) {
            progressSum += 1.0f - min(fast / peak, 1.0f);
            progressCount += 1u;
        }
    }
    ctx.stats->learningMature = (progressCount > 0u) ? progressSum / float(progressCount) : 0.0f;
    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}

kernel void oak_action_emit(device float *params [[buffer(0)]], device float *stateArena [[buffer(1)]], device uint *uintArena [[buffer(2)]], device KernelScalars *scalars [[buffer(3)]], device KernelStats *stats [[buffer(4)]], constant KernelConfig &cfg [[buffer(5)]], threadgroup float *tgCurrState [[threadgroup(0)]], uint2 tgPos [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]], uint2 tdim [[threads_per_threadgroup]]) {
    uint group = tgPos.x, tid = tpos.x, tgW = tdim.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    device const float *curr = state_at(ctx, ctx.stateSlot);
    for (uint i = tid; i < ctx.R; i += tgW) tgCurrState[i] = curr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    emit_action_grid(ctx, tgCurrState, group, tid, tgW);
}

// ===========================================================================
// Phase F3: GPU-resident fleet env layer.
//   oak_fleet_init   — per-lane env reset + arena seeding (one thread/lane)
//   oak_sample_act   — exact Metal port of HCEnv::step_oak/step_zero/
//                      step_random sampling (setup.sh oak_env_bridge.cc),
//                      Philox stream 2 for the policy normals
//   oak_fleet_env    — hc_env_step body + envInfo/reward accumulators +
//                      death-detector latches + action recording
//   oak_fleet_snapshot — heartbeat copy of per-lane health into a
//                      double-buffered snapshot the host writer drains
// The agent kernels above and the physics in OakPhysics.metal are untouched.
// ===========================================================================

constant uint HC_RNG_STREAM_ACT = 2u;  // 0 = reset noise, 1 = random ctrl

// Death detector codes (host maps to the deaths.jsonl detector strings).
constant uint OAK_DEATH_NONE       = 0xffffffffu;
constant uint OAK_DET_ENV_REWARD   = 1u;
constant uint OAK_DET_POLICY_MEAN  = 2u;
constant uint OAK_DET_ENV_OBS      = 3u;

struct OakFleetEnvCfg {
    uint laneCount;
    uint episodeLen;       // 0 = continuing (CPU-era bridge semantics), 1000 = gym
    uint policyMode;       // 0 = oak, 1 = zero, 2 = random
    uint recordCapacity;   // ring records per recorded lane (0 = no recording)
    float invObsFloor;     // Run floor = 1/max(obsDim,1) for the host-rho EMA
};

struct OakFleetInfoAcc { uint count, finiteCount; float sum, sumC, first, last; };

// Per-lane persistent fleet state (env + stream accumulators). All members
// 4-byte scalars: layout is mirrored field-for-field on the Swift side.
struct OakFleetLane {
    HCState s;                            // qpos[9] qvel[9] ctrl[6] warmstart[9]
    uint elapsed, resets;
    uint steps, doneCount;
    float ctrlPending[6];                 // clamped action from oak_sample_act
    float lastDesc[12];                   // HCEnv::last_policy_ (d_policy = 12)
    float deltaSum, deltaC, deltaPeak;    // Kahan delta_sum_ + running-mean peak
    float rewardSum, rewardC, rho;        // Kahan rewardSum + host-EMA reward rate
    uint deathStep, deathDetector;        // first-nonfinite latch
    float deathRewardSum, deathRho;       // values at the latch step
    OakFleetInfoAcc info[4];              // reward_run, reward_ctrl, x_position, x_velocity
};
static_assert(sizeof(OakFleetLane) == 356, "OakFleetLane layout drifted from the Swift mirror");

// Heartbeat snapshot row (one per lane, double-buffered host-visible).
struct OakLaneSnapshot {
    KernelScalars scalars;
    KernelStats stats;
    float paramAbsMax;
    uint paramNonFinite;
    uint steps, doneCount, resets;
    float rewardSum, rho;
    float deltaSum, deltaPeak;
    uint deathStep, deathDetector;
    float deathRewardSum, deathRho;
    OakFleetInfoAcc info[4];
};
static_assert(sizeof(OakLaneSnapshot) == 276, "OakLaneSnapshot layout drifted from the Swift mirror");

inline float oak_safe(float x) { return isfinite(x) ? x : 0.0f; }

// 6 standard normals via Philox stream 2 + Box-Muller (same construction as
// hc_rng_reset_noise). PORT-NOTE: the CPU bridge uses mt19937 +
// std::normal_distribution; only the law N(0,1) is shared, per the F2 RNG
// contract for hc_rng_ctrl.
inline void oak_rng_normal6(uint laneSeed, uint stepIdx, thread float *z) {
    uint2 key = uint2(laneSeed, HC_RNG_STREAM_ACT);
    uint4 r0 = hc_philox4x32(key, uint4(stepIdx, 0u, 0u, 0u));
    uint4 r1 = hc_philox4x32(key, uint4(stepIdx, 1u, 0u, 0u));
    uint raw[6] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y };
    for (int p = 0; p < 3; p++) {
        float u1 = hc_u01open(raw[2*p]);
        float u2 = hc_u01(raw[2*p + 1]);
        float rad = sqrt(-2.0f * log(u1));
        float ang = 2.0f * M_PI_F * u2;
        z[2*p] = rad * cos(ang);
        z[2*p + 1] = rad * sin(ang);
    }
}

inline void oak_kahan_add(device float *sum, device float *comp, float x) {
    float y = x - *comp;
    float t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

// One thread per lane: env reset + lane-state zeroing + arena seeding,
// replicating the CPU fleet prologue (env.prepare(rank); env.reset();) —
// obs = reset obs, reward = 0, actionSignal/credit = 0, importance = 1.
kernel void oak_fleet_init(device float *params [[buffer(0)]],
                           device float *stateArena [[buffer(1)]],
                           device uint *uintArena [[buffer(2)]],
                           device KernelScalars *scalars [[buffer(3)]],
                           device KernelStats *stats [[buffer(4)]],
                           constant KernelConfig &cfg [[buffer(5)]],
                           device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    HCState s;
    uint elapsed = 0u, resets = 0u;
    hc_env_reset(s, laneSeeds[gid], elapsed, resets);
    lane->s = s;
    lane->elapsed = elapsed; lane->resets = resets;
    lane->steps = 0u; lane->doneCount = 0u;
    for (int u = 0; u < 6; u++) lane->ctrlPending[u] = 0.0f;
    for (int i = 0; i < 12; i++) lane->lastDesc[i] = 0.0f;
    lane->deltaSum = 0.0f; lane->deltaC = 0.0f; lane->deltaPeak = 0.0f;
    lane->rewardSum = 0.0f; lane->rewardC = 0.0f; lane->rho = 0.0f;
    lane->deathStep = OAK_DEATH_NONE; lane->deathDetector = 0u;
    lane->deathRewardSum = 0.0f; lane->deathRho = 0.0f;
    for (int j = 0; j < 4; j++) {
        lane->info[j].count = 0u; lane->info[j].finiteCount = 0u;
        lane->info[j].sum = 0.0f; lane->info[j].sumC = 0.0f;
        lane->info[j].first = 0.0f; lane->info[j].last = 0.0f;
    }
    if (ecfg.policyMode == 0u) {
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid);
        for (uint i = 0u; i + 1u < 9u; ++i) ctx.observation[i] = s.qpos[i + 1u];
        for (uint i = 0u; i < 9u; ++i) ctx.observation[8u + i] = s.qvel[i];
        ctx.reward[0] = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) ctx.actionSignal[d] = 0.0f;
        for (uint i = 0u; i < ctx.K * ctx.D; ++i) ctx.optionActorCredit[i] = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) ctx.importance[k] = 1.0f;
    }
}

// One thread per lane: action sampling. Exact port of HCEnv::step_oak /
// step_zero / step_random up to (but excluding) send():
//   descriptor[i] = safe(policyMean[i]); sd = 1/d_policy;
//   per action dim d: sample = safe(mu + sd*z); ctrl = clamp(sample, -1, 1);
//   actionSignal[d] = ctrl, actionSignal[A+d] = 0;
//   behavior += log_sigma - z^2/2;
//   per option o: oz = (sample - option_mu)/sd (UNclamped sample, as on CPU);
//   logp[o] += log_sigma - oz^2/2; credit[o*D+d] = sample - option_mu;
//   credit[o*D+A+d] = 0; importance[o] = exp(logp[o] - behavior).
// Plus the actionDelta tracker: delta_sum += sum((desc-last)^2)/D, last = desc,
// steps++ — all before the env step, exactly like the bridge.
// PORT-NOTE: the bridge accumulates delta_sum_ in double; here fp32 Kahan.
kernel void oak_sample_act(device float *params [[buffer(0)]],
                           device float *stateArena [[buffer(1)]],
                           device uint *uintArena [[buffer(2)]],
                           device KernelScalars *scalars [[buffer(3)]],
                           device KernelStats *stats [[buffer(4)]],
                           constant KernelConfig &cfg [[buffer(5)]],
                           device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    uint stepIdx = lane->steps;  // 0-based index of the step being taken
    float desc[12];
    float ctrl[6];
    if (ecfg.policyMode == 1u) {  // step_zero
        for (int u = 0; u < 6; u++) ctrl[u] = 0.0f;
        for (int i = 0; i < 12; i++) desc[i] = 0.0f;
    } else if (ecfg.policyMode == 2u) {  // step_random
        hc_rng_ctrl(laneSeeds[gid], stepIdx, ctrl);
        for (int d = 0; d < 6; d++) { desc[d] = ctrl[d]; desc[6 + d] = 0.0f; }
    } else {  // step_oak
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid);
        uint A = ctx.D / 2u;  // act_dim (Box exposes 2A policy coordinates)
        float sd = 1.0f / float(max(ctx.D, 1u));
        float log_sigma = -log(sd);
        float behavior = 0.0f;
        for (uint i = 0u; i < ctx.D; ++i) desc[i] = oak_safe(ctx.policyMean[i]);
        float z[6];
        oak_rng_normal6(laneSeeds[gid], stepIdx, z);
        float logp[64];
        for (uint o = 0u; o < ctx.K; ++o) logp[o] = 0.0f;
        for (uint d = 0u; d < A; ++d) {
            float mu = desc[d];
            float sample = oak_safe(mu + sd * z[d]);
            float a = max(-1.0f, min(1.0f, sample));
            ctrl[d] = a;
            ctx.actionSignal[d] = a;
            ctx.actionSignal[A + d] = 0.0f;
            behavior += log_sigma - 0.5f * z[d] * z[d];
            for (uint o = 0u; o < ctx.K; ++o) {
                uint off = o * ctx.D;
                float om = oak_safe(ctx.optionPolicyMean[off + d]);
                float oz = (sample - om) / sd;
                logp[o] += log_sigma - 0.5f * oz * oz;
                ctx.optionActorCredit[off + d] = sample - om;
                ctx.optionActorCredit[off + A + d] = 0.0f;
            }
        }
        for (uint o = 0u; o < ctx.K; ++o) ctx.importance[o] = exp(logp[o] - behavior);
    }
    // delta(descriptor_); ++steps_;
    float d2 = 0.0f;
    for (int i = 0; i < 12; i++) {
        float v = desc[i] - lane->lastDesc[i];
        d2 += v * v;
        lane->lastDesc[i] = desc[i];
    }
    oak_kahan_add(&lane->deltaSum, &lane->deltaC, d2 / 12.0f);
    lane->steps = stepIdx + 1u;
    float adm = lane->deltaSum / float(lane->steps);
    if (isfinite(adm) && adm > lane->deltaPeak) lane->deltaPeak = adm;
    for (int u = 0; u < 6; u++) lane->ctrlPending[u] = ctrl[u];
}

// One thread per lane: the env step (HCEnv::send semantics — exactly the
// hc_env_step control flow, inlined to also expose the four envInfo values),
// arena obs/reward write-back, stream accumulators, death latches, recording.
kernel void oak_fleet_env(device float *params [[buffer(0)]],
                          device float *stateArena [[buffer(1)]],
                          device uint *uintArena [[buffer(2)]],
                          device KernelScalars *scalars [[buffer(3)]],
                          device KernelStats *stats [[buffer(4)]],
                          constant KernelConfig &cfg [[buffer(5)]],
                          device OakFleetLane *lanes [[buffer(6)]],
                          device const uint *laneSeeds [[buffer(7)]],
                          constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                          device const int *recordSlot [[buffer(9)]],
                          device float *recordBuf [[buffer(10)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    HCState s = lane->s;
    uint elapsed = lane->elapsed, resets = lane->resets;
    float ctrl[6];
    for (int u = 0; u < 6; u++) ctrl[u] = lane->ctrlPending[u];
    float reward;
    uint done;
    float info[4];
    if (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) {
        // HCRaw::ResetRaw: action discarded, reward 0, done flag 0, info zeroed.
        hc_env_reset(s, laneSeeds[gid], elapsed, resets);
        reward = 0.0f;
        done = 0u;
        info[0] = 0.0f; info[1] = -0.0f; info[2] = 0.0f; info[3] = 0.0f;
    } else {
        float xBefore = s.qpos[0];
        for (int u = 0; u < 6; u++) s.ctrl[u] = ctrl[u];
        for (int k = 0; k < HC_FRAME_SKIP; k++) hc_mj_step(s);
        float xv = (s.qpos[0] - xBefore) / (float(HC_FRAME_SKIP) * HC_TIMESTEP);
        float ctrlCost = 0.0f;
        for (int u = 0; u < 6; u++) ctrlCost += 0.1f * ctrl[u] * ctrl[u];
        reward = xv * 1.0f - ctrlCost;
        elapsed++;
        done = (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) ? 1u : 0u;
        info[0] = xv; info[1] = -ctrlCost; info[2] = s.qpos[0]; info[3] = xv;
    }
    float obs[17];
    for (int i = 1; i < 9; i++) obs[i - 1] = s.qpos[i];
    for (int i = 0; i < 9; i++) obs[8 + i] = s.qvel[i];
    lane->s = s;
    lane->elapsed = elapsed;
    lane->resets = resets;
    lane->doneCount += done;
    uint stepNum = lane->steps;  // incremented by oak_sample_act

    // Run.note: only finite rewards accumulate; non-finite latches envReward.
    if (isfinite(reward)) {
        oak_kahan_add(&lane->rewardSum, &lane->rewardC, reward);
        float rate = max(1.0f / float(stepNum), ecfg.invObsFloor);
        lane->rho += rate * (reward - lane->rho);
    } else if (lane->deathStep == OAK_DEATH_NONE) {
        lane->deathStep = stepNum;
        lane->deathDetector = OAK_DET_ENV_REWARD;
        lane->deathRewardSum = lane->rewardSum;
        lane->deathRho = lane->rho;
    }
    // CPU fleet order: envReward first, then policyMean (engine lanes only).
    if (ecfg.policyMode == 0u) {
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid);
        if (lane->deathStep == OAK_DEATH_NONE) {
            for (uint i = 0u; i < ctx.D; ++i) {
                if (!isfinite(ctx.policyMean[i])) {
                    lane->deathStep = stepNum;
                    lane->deathDetector = OAK_DET_POLICY_MEAN;
                    lane->deathRewardSum = lane->rewardSum;
                    lane->deathRho = lane->rho;
                    break;
                }
            }
        }
        for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = obs[i];
        ctx.reward[0] = reward;
    }
    // envObs is GPU-mode-only: the CPU loop never finite-checked obs (qpos/qvel
    // auto-reset on bad values inside mj_step), so this latch is belt-and-braces.
    if (lane->deathStep == OAK_DEATH_NONE) {
        bool obsBad = false;
        for (int i = 0; i < 17; i++) obsBad = obsBad || !isfinite(obs[i]);
        if (obsBad) {
            lane->deathStep = stepNum;
            lane->deathDetector = OAK_DET_ENV_OBS;
            lane->deathRewardSum = lane->rewardSum;
            lane->deathRho = lane->rho;
        }
    }
    // envInfo accumulators (InfoStat.note semantics).
    for (int j = 0; j < 4; j++) {
        device OakFleetInfoAcc *a = &lane->info[j];
        a->count += 1u;
        if (isfinite(info[j])) {
            if (a->finiteCount == 0u) a->first = info[j];
            a->finiteCount += 1u;
            oak_kahan_add(&a->sum, &a->sumC, info[j]);
            a->last = info[j];
        }
    }
    // ActionRecorder.record: actionSignal[0..actDim) (= clamped action) + done.
    if (ecfg.recordCapacity != 0u) {
        int slot = recordSlot[gid];
        if (slot >= 0) {
            ulong base = ((ulong)slot * ecfg.recordCapacity
                          + ((stepNum - 1u) % ecfg.recordCapacity)) * 7ul;
            for (int u = 0; u < 6; u++) recordBuf[base + u] = ctrl[u];
            recordBuf[base + 6] = float(done);
        }
    }
}

// One thread per lane: heartbeat snapshot into snap[snapBase + lane].
// paramAbsMax loops the lane's full param slice (CPU heartbeat parity:
// non-finite short-circuits and is flagged; otherwise the running max).
kernel void oak_fleet_snapshot(device float *params [[buffer(0)]],
                               device float *stateArena [[buffer(1)]],
                               device uint *uintArena [[buffer(2)]],
                               device KernelScalars *scalars [[buffer(3)]],
                               device KernelStats *stats [[buffer(4)]],
                               constant KernelConfig &cfg [[buffer(5)]],
                               device OakFleetLane *lanes [[buffer(6)]],
                               device const uint *laneSeeds [[buffer(7)]],
                               constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                               device OakLaneSnapshot *snap [[buffer(9)]],
                               constant uint &snapBase [[buffer(10)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device OakLaneSnapshot *o = &snap[snapBase + gid];
    if (ecfg.policyMode == 0u) {
        o->scalars = scalars[gid];
        o->stats = stats[gid];
        device const float *lp = params + gid * cfg.paramStride;
        float mx = 0.0f;
        uint nonFinite = 0u;
        for (uint i = 0u; i < cfg.paramStride; ++i) {
            float v = lp[i];
            if (!isfinite(v)) { nonFinite = 1u; break; }
            if (fabs(v) > mx) mx = fabs(v);
        }
        o->paramAbsMax = mx;
        o->paramNonFinite = nonFinite;
    } else {
        o->scalars = KernelScalars{0u, 0u, 0.0f};
        KernelStats zs;
        thread float *zp = (thread float *)&zs;
        for (uint i = 0u; i < sizeof(KernelStats) / sizeof(float); ++i) zp[i] = 0.0f;
        o->stats = zs;
        o->paramAbsMax = 0.0f;
        o->paramNonFinite = 0u;
    }
    o->steps = lane->steps;
    o->doneCount = lane->doneCount;
    o->resets = lane->resets;
    o->rewardSum = lane->rewardSum;
    o->rho = lane->rho;
    o->deltaSum = lane->deltaSum;
    o->deltaPeak = lane->deltaPeak;
    o->deathStep = lane->deathStep;
    o->deathDetector = lane->deathDetector;
    o->deathRewardSum = lane->deathRewardSum;
    o->deathRho = lane->deathRho;
    for (int j = 0; j < 4; j++) o->info[j] = lane->info[j];
}
