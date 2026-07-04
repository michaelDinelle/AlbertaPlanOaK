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

constant uint OAK_MEGA_G [[function_constant(12)]];

constant uint OAK_SPEC_OBS_DIM [[function_constant(13)]];
constant uint OAK_SPEC_RANK [[function_constant(14)]];
constant uint OAK_SPEC_OPTION_COUNT [[function_constant(15)]];
constant uint OAK_SPEC_ACTION_PARAMS [[function_constant(16)]];
constant uint OAK_SPEC_TRACE_LEVELS [[function_constant(17)]];
constant uint OAK_SPEC_ANCHOR_COUNT [[function_constant(18)]];
constant uint OAK_SPEC_PARAM_STRIDE [[function_constant(19)]];
constant uint OAK_SPEC_STATE_STRIDE [[function_constant(20)]];
constant uint OAK_SPEC_UINT_STRIDE [[function_constant(21)]];
constant float OAK_SPEC_INV_RANK [[function_constant(22)]];
constant float OAK_SPEC_INV_OPTION_COUNT [[function_constant(23)]];
constant float OAK_SPEC_META_ETA [[function_constant(24)]];

constant uint OAK_SPEC_NURSERY [[function_constant(28)]];

// Timing probe (trajectories are garbage): replace active-index gathers with
// fixed contiguous indices to separate memory-scatter cost from issue cost.
constant bool OAK_PROBE_FIXEDIDX_FC [[function_constant(30)]];
constant bool OAK_PROBE_FIXEDIDX = is_function_constant_defined(OAK_PROBE_FIXEDIDX_FC)
    ? OAK_PROBE_FIXEDIDX_FC : false;
inline uint act_of(device const uint *act, uint a) {
    return OAK_PROBE_FIXEDIDX ? a : act[a];
}

// Timing probe (trajectories are garbage): flush sub-normal magnitudes on learn
// writes to convict/acquit fp32 denormal execution cost as the SPS settling
// transient (weights/aux decay toward underflow by construction).
constant bool OAK_PROBE_FTZ_FC [[function_constant(31)]];
constant bool OAK_PROBE_FTZ = is_function_constant_defined(OAK_PROBE_FTZ_FC)
    ? OAK_PROBE_FTZ_FC : false;
inline float ftz_probe(float w) {
    return (OAK_PROBE_FTZ && fabs(w) < 1.1754944e-38f) ? 0.0f : w;
}

constant uint OAK_MEGA_STAGE_MASK [[function_constant(25)]];
constant uint OAK_MEGA_STAGES = is_function_constant_defined(OAK_MEGA_STAGE_MASK)
    ? OAK_MEGA_STAGE_MASK : 0xffffffffu;

constant bool OAK_MEGA_WIDE2X_FC [[function_constant(26)]];
constant bool OAK_MEGA_WIDE2X = is_function_constant_defined(OAK_MEGA_WIDE2X_FC)
    ? OAK_MEGA_WIDE2X_FC : false;

constant bool OAK_MEGA_ENV_COOP_FC [[function_constant(27)]];
constant bool OAK_MEGA_ENV_COOP = is_function_constant_defined(OAK_MEGA_ENV_COOP_FC)
    ? OAK_MEGA_ENV_COOP_FC : true;

struct KernelConfig {
    uint obsDim, rank, optionCount, actionParamCount;
    uint obsTraceLevels;
    uint linearAnchorCount;
    float invRank, invOptionCount;
    float metaEta;
    uint agentCount, paramStride, stateStride, uintStride;
    uint nurseryCount;
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
    float learningMature;
};

inline float safe_w_raw(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }
#define safe_w(o, n) ftz_probe(safe_w_raw((o), (n)))

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct Ctx {
    uint O, R, K, D, traceLevels, anchorCount, Rn;
    uint stepCount, stateSlot;
    float rho;
    float invR, invK, metaEta;
    device float *valueW, *psi, *optPolicy, *optReward, *optDuration;
    device float *psiAux, *vStop, *vsAux;
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
    device float *optRewardTd, *headDenom;
    device float *durationTd, *durationHat, *obsTraceMulti, *prioOpt, *continuation;
    device float *keyboard, *modelDenom;
    device float *valueGauge;
    device float *tdPsiFast, *tdPsiPeak;
    device float *headScale, *modelScale, *valueScale, *xNormPrev;
    device float *hDotPsi, *psiAuxScale, *psiRho;
    device float *vsTd, *vsDot, *vsScale, *vsAuxScale;
    device float *nurStats;
    device float *muRaw, *polCoef, *optUseByFeat, *lcbScr;
    device uint *counters, *optFeature, *featureAge, *optAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption, *keyboardOrigin;
    device uint *explosionStep, *explosionElem;
    device uint *lastStopAge;
    device uint *nurParentA, *nurParentB, *nurKind, *nurAge;
    device uint *actIdx;
    device KernelScalars *scalars;
    device KernelStats *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg, uint agentID) {
    uint paramStride = is_function_constant_defined(OAK_SPEC_PARAM_STRIDE)
        ? OAK_SPEC_PARAM_STRIDE : cfg.paramStride;
    uint stateStride = is_function_constant_defined(OAK_SPEC_STATE_STRIDE)
        ? OAK_SPEC_STATE_STRIDE : cfg.stateStride;
    uint uintStride = is_function_constant_defined(OAK_SPEC_UINT_STRIDE)
        ? OAK_SPEC_UINT_STRIDE : cfg.uintStride;
    params += agentID * paramStride;
    stateArena += agentID * stateStride;
    uintArena += agentID * uintStride;
    scalars += agentID;
    stats += agentID;
    Ctx c;
    c.O = is_function_constant_defined(OAK_SPEC_OBS_DIM) ? OAK_SPEC_OBS_DIM : cfg.obsDim;
    c.R = is_function_constant_defined(OAK_SPEC_RANK) ? OAK_SPEC_RANK : cfg.rank;
    c.K = is_function_constant_defined(OAK_SPEC_OPTION_COUNT)
        ? OAK_SPEC_OPTION_COUNT : cfg.optionCount;
    c.D = is_function_constant_defined(OAK_SPEC_ACTION_PARAMS)
        ? OAK_SPEC_ACTION_PARAMS : cfg.actionParamCount;
    c.traceLevels = is_function_constant_defined(OAK_SPEC_TRACE_LEVELS)
        ? OAK_SPEC_TRACE_LEVELS : cfg.obsTraceLevels;
    c.anchorCount = is_function_constant_defined(OAK_SPEC_ANCHOR_COUNT)
        ? OAK_SPEC_ANCHOR_COUNT : cfg.linearAnchorCount;
    c.Rn = is_function_constant_defined(OAK_SPEC_NURSERY) ? OAK_SPEC_NURSERY : cfg.nurseryCount;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho;
    c.invR = is_function_constant_defined(OAK_SPEC_INV_RANK) ? OAK_SPEC_INV_RANK : cfg.invRank;
    c.invK = is_function_constant_defined(OAK_SPEC_INV_OPTION_COUNT)
        ? OAK_SPEC_INV_OPTION_COUNT : cfg.invOptionCount;
    c.metaEta = is_function_constant_defined(OAK_SPEC_META_ETA) ? OAK_SPEC_META_ETA : cfg.metaEta;
    uint p = 0u;
    c.valueW = params + p; p += c.R;
    c.psi = params + p; p += c.K * c.R;
    c.optPolicy = params + p; p += c.K * c.D * c.R;
    c.optReward = params + p; p += c.K * c.R;
    c.optDuration = params + p; p += c.K * c.R;
    c.psiAux = params + p; p += c.K * c.R;
    c.vStop = params + p; p += c.K * c.R;
    c.vsAux = params + p;

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
    c.headScale = stateArena + s; s += c.K;
    c.modelScale = stateArena + s; s += c.K;
    c.valueScale = stateArena + s; s += 1u;
    c.xNormPrev = stateArena + s; s += 1u;
    c.hDotPsi = stateArena + s; s += c.K;
    c.psiAuxScale = stateArena + s; s += c.K;
    c.psiRho = stateArena + s; s += c.K;
    c.vsTd = stateArena + s; s += c.K;
    c.vsDot = stateArena + s; s += c.K;
    c.vsScale = stateArena + s; s += c.K;
    c.vsAuxScale = stateArena + s; s += c.K;
    s += 3u;  // 16B-align nurStats/muRaw/polCoef (float4 access)
    // Interleaved candidate stats {mean, sm, util, utilSM}: one line per
    // candidate; access is sequential over n so coalescing is preserved.
    c.nurStats = stateArena + s; s += 4u * c.Rn;
    c.muRaw = stateArena + s; s += c.K * c.D;
    c.polCoef = stateArena + s; s += c.K * c.D;
    c.optUseByFeat = stateArena + s; s += c.R;
    c.lcbScr = stateArena + s; s += c.Rn;

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
    c.explosionStep = uintArena + u; u += 1u;
    c.explosionElem = uintArena + u; u += 1u;
    c.lastStopAge = uintArena + u; u += c.K;
    c.nurParentA = uintArena + u; u += c.Rn;
    c.nurParentB = uintArena + u; u += c.Rn;
    c.nurKind = uintArena + u; u += c.Rn;
    c.nurAge = uintArena + u; u += c.Rn;
    c.actIdx = uintArena + u;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
// Active-set indices for a state slot (32 = execution width, banked WTA).
inline device uint *act_at(Ctx ctx, uint slot) { return ctx.actIdx + slot * 32u; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline bool usable_priority(float p) { return isfinite(p) && p >= 0.0f; }
inline bool usable_duration(Ctx ctx, float d) {
    return isfinite(d) && d > 0.0f;
}
inline uint linear_input_width(Ctx ctx) { return (ctx.traceLevels + 1u) * ctx.O + ctx.D + 2u; }

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
    if (tail == 0u) return ctx.reward[0];
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
    // Structural bounds: alpha in [1/cap^2, 1] (capacity-derived, no knob).
    beta[i] = safe_w(bOld, clamp(bNew, 2.0f * log(decay), 0.0f));
    alpha[i] = safe_w(aOld, exp(beta[i]));
    float curvature = x * x;
    // IDBD trace: h <- h * max(0, 1 - alpha x^2) + alpha delta x; the max(0,·)
    // keeps the meta-gradient from flipping sign when alpha x^2 > 1.
    float traceDecay = (1.0f - decay) * max(0.0f, 1.0f - aOld * curvature);
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

// Precompute the learn-stage scale factors that are identical for every learned
// element this step (Inv-10 bit-identical hoist: same expressions, same fma and
// divide order as the per-element code they replace).
inline void fcstomp_store_learn_scales(Ctx ctx,
                                       threadgroup const float *prev,
                                       threadgroup const float *curr) {
    // prev/curr are ACTIVE-value caches (32 wide); indices come from actIdx.
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    float denom = 1.0f;
    for (uint a = 0u; a < 32u; ++a) {
        float ap = ctx.alpha[actP[a]];
        if (isfinite(ap) && ap > 0.0f) denom += ap * (prev[a] * prev[a]);
        float ac = ctx.alpha[actC[a]];
        if (isfinite(ac) && ac > 0.0f) denom += ac * (curr[a] * curr[a]);
    }
    float updateEnergy = ctx.td[0] * ctx.td[0] + ctx.planTd[0] * ctx.planTd[0];
    if (isfinite(updateEnergy)) denom += ctx.invR * updateEnergy;
    ctx.valueScale[0] = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
    float xNorm = 0.0f;
    for (uint a = 0u; a < 32u; ++a) xNorm = fma(prev[a], prev[a], xNorm);
    ctx.xNormPrev[0] = xNorm;
}

inline void fcstomp_prediction_control_loop(Ctx ctx, uint tid,
                                      threadgroup float *tgPrev,
                                      threadgroup float *tgCurr,
                                      threadgroup float *tgG,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgTerm,
                                      threadgroup float *tgTdPsi,
                                      threadgroup float *tgScalars) {
    device const float *dPrev = state_at(ctx, ctx.stateSlot);
    device const float *dCurr = state_at(ctx, 1u - ctx.stateSlot);
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    float vPrev = 0.0f, vCurr = 0.0f;
    if (tid == 0u) {
        for (uint a = 0u; a < 32u; ++a) {
            vPrev = fma(ctx.valueW[act_of(actP, a)], tgPrev[a], vPrev);
            vCurr = fma(ctx.valueW[act_of(actC, a)], tgCurr[a], vCurr);
        }
        if (OAK_ABLATE_GVF_PREDICTION) {
            vPrev = 0.0f;
            vCurr = 0.0f;
        }
        tgScalars[0] = vPrev;
        tgScalars[1] = vCurr;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
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
        float modelTraceNorm = 0.0f;
        float xNormRaw = 0.0f;
        for (uint a = 0u; a < 32u; ++a) {
            float x = tgPrev[a];
            if (!isfinite(x)) continue;
            float xx = x * x;
            xNormRaw += xx;
            float am = ctx.alphaModel[act_of(actP, a)];
            if (isfinite(am) && am > 0.0f) modelTraceNorm = fma(am, xx, modelTraceNorm);
        }
        tgScalars[10] = (isfinite(modelTraceNorm) && modelTraceNorm > 0.0f) ? modelTraceNorm : 0.0f;
        tgScalars[7] = (isfinite(xNormRaw) && xNormRaw > 0.0f) ? xNormRaw : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        uint feat = ctx.optFeature[tid] % ctx.R;
        uint base = tid * ctx.R;
        float psiPrev = 0.0f, psiNext = 0.0f, psiXX = 0.0f, psiXY = 0.0f, psiYY = 0.0f;
        for (uint a = 0u; a < 32u; ++a) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            psiPrev = fma(ctx.psi[(ip) * ctx.K + tid], tgPrev[a], psiPrev);
            psiNext = fma(ctx.psi[(ic) * ctx.K + tid], tgCurr[a], psiNext);
            float ap = ctx.alphaPsi[ip];
            if (isfinite(ap) && ap > 0.0f) {
                psiXX = fma(ap * tgPrev[a], tgPrev[a], psiXX);
                // exact: dCurr is dense with zeros off the active set
                psiXY = fma(ap * tgPrev[a], dCurr[ip], psiXY);
            }
            float ac = ctx.alphaPsi[ic];
            if (isfinite(ac) && ac > 0.0f) psiYY = fma(ac * tgCurr[a], tgCurr[a], psiYY);
        }
        // RR eq. 4 stopping value with a genuine unit bonus in standardized
        // feature units (bonus weight = valueW[feat] + 1): stopping is worth
        // v(s') + x_f(s'), so attaining the feature stays permanently valuable
        // and the bonus can never track itself away.
        float zStop = vCurrG + dCurr[feat];
        bool rrFinite = isfinite(psiPrev) && isfinite(psiNext) && isfinite(zStop);
        if (!rrFinite) ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        // Strict inequality: ties (e.g. zero init) break toward continuation.
        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop > psiNext) ? 1.0f : 0.0f;
        if (OAK_ABLATE_RR_SUBTASKS) {
            rrFinite = true;
            beta = 1.0f;
        }
        float cont = 1.0f - beta;
        ctx.continuation[tid] = cont;
        if (beta > 0.0f) ctx.lastStopAge[tid] = 0u;
        // Differential (average-reward) GVF, Plan Step 5 / Reward Centering
        // applied per head: rewardC is centered for the MAIN policy; each
        // option's own average differs, and with cont ~ 1 (a never-stopping
        // head) an uncentered cumulant makes the fixed point unbounded.
        float psiRhoOld = ctx.psiRho[tid];
        if (!isfinite(psiRhoOld)) psiRhoOld = 0.0f;
        float dPsi = (!OAK_ABLATE_RR_SUBTASKS && rrFinite)
            ? (rewardC - psiRhoOld + cont * psiNext + beta * zStop - psiPrev) : 0.0f;
        if (!isfinite(dPsi)) {
            dPsi = 0.0f;
            ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        }
        ctx.tdPsi[tid] = dPsi;
        tgTdPsi[tid] = dPsi;
        float psiRhoNew = psiRhoOld + ctx.invR * dPsi;
        ctx.psiRho[tid] = safe_w(psiRhoOld, psiRhoNew);
        float absD = isfinite(dPsi) ? fabs(dPsi) : 0.0f;
        float fastOld = ctx.tdPsiFast[tid];
        float peakOld = ctx.tdPsiPeak[tid];
        ctx.tdPsiFast[tid] = (1.0f - ctx.invR) * (isfinite(fastOld) ? fastOld : 0.0f) + ctx.invR * absD;
        ctx.tdPsiPeak[tid] = max(isfinite(peakOld) ? peakOld : 0.0f, absD);
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        // GTD2 implicit normalizer: alpha-weighted norm of the bootstrap
        // difference vector (x - cont*x'), nonnegative by construction.
        float implicitTrace = psiXX - 2.0f * cont * psiXY + cont * cont * psiYY;
        float denom = fma(rho, implicitTrace, 1.0f);
        ctx.headDenom[tid] = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        float modelDenom = fma(rho, tgScalars[10], 1.0f);
        ctx.modelDenom[tid] = (isfinite(modelDenom) && modelDenom > 0.0f) ? modelDenom : 0.0f;
        {
            // Per-head reciprocal, hoisted from the learn stage (identical divide).
            float hd = ctx.headDenom[tid];
            ctx.headScale[tid] = (isfinite(hd) && hd > 0.0f) ? (1.0f / hd) : 0.0f;
            float md = ctx.modelDenom[tid];
            ctx.modelScale[tid] = (isfinite(md) && md > 0.0f) ? (1.0f / md) : 0.0f;
        }
        {
            // TDC gradient correction (Sutton et al. 2009): hDot = h_k^T x for the
            // off-policy bootstrapped psi head; at h = 0 the corrected update
            // reduces exactly to plain TD. auxDenom is the implicit-update
            // normalizer of h's LMS regression of delta onto x.
            float hDot = 0.0f;
            for (uint a = 0u; a < 32u; ++a)
                hDot = fma(ctx.psiAux[act_of(actP, a) * ctx.K + tid], tgPrev[a], hDot);
            ctx.hDotPsi[tid] = isfinite(hDot) ? hDot : 0.0f;
            // NLMS: h is a pure regression of delta onto x, so it runs at the
            // fastest stable rate (unit effective step), parameter-free. A fast,
            // accurate h is what GTD2 needs to stop costing learning speed.
            float auxDenom = fma(rho, psiXX, 1.0f);
            ctx.psiAuxScale[tid] = (isfinite(auxDenom) && auxDenom > 0.0f)
                ? (1.0f / auxDenom) : 0.0f;
        }
        tgCont[tid] = cont;
        tgTerm[tid] = beta;
        float rHat = 0.0f, lRaw = 0.0f, vModel = 0.0f;
        if (!OAK_ABLATE_OPTION_MODELS) {
            for (uint a = 0u; a < 32u; ++a) {
                uint idx = act_of(actC, a) * ctx.K + tid;
                rHat = fma(ctx.optReward[idx], tgCurr[a], rHat);
                lRaw = fma(ctx.optDuration[idx], tgCurr[a], lRaw);
                // vStop: scalar GVF predicting v at option stop (replaces the
                // K*R*R transition tensor: v(n-hat) was its only live consumer).
                vModel = fma(ctx.vStop[idx], tgCurr[a], vModel);
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
        // NaN marks an untrusted option; a finite tgG[k] is a usable model-backed
        // advantage (queryTrusted implies qAdv is finite).
        tgG[tid] = (queryTrusted && !OAK_ABLATE_MODEL_TO_KEYBOARD) ? qAdv : NAN;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
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
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint best = 0u;
        float bestScore = -FLT_MAX;
        float termSum = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            termSum += tgTerm[k];
            bool trusted = isfinite(tgG[k]);
            float qAdv = trusted ? tgG[k] : 0.0f;
            float dur = ctx.durationHat[k];
            if (!isfinite(dur) || dur <= 0.0f) dur = 1.0f;
            float modelErr = OAK_ABLATE_SEARCH_CONTROL ? 0.0f : max(ctx.prioOpt[k], 0.0f);
            if (!isfinite(modelErr)) modelErr = 0.0f;
            float score = trusted ? ((qAdv - modelErr) / dur) : -FLT_MAX;
            if (score > bestScore) { bestScore = score; best = k; }
        }
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        bool continuing = tgCont[prevSel] > 0.0f;
        uint selected = continuing ? prevSel : best;
        float advL1 = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float a = ctx.actorAdv[k];
            if (isfinite(a)) advL1 += fabs(a);
        }
        if (OAK_ABLATE_OPTION_KEYBOARD) {
            for (uint k = 0u; k < ctx.K; ++k) ctx.keyboard[k] = 0.0f;
            ctx.keyboardOrigin[0] = 0u;
            ctx.planTd[0] = 0.0f;
            ctx.selectedOption[0] = selected;
            fcstomp_store_learn_scales(ctx, tgPrev, tgCurr);
            fcstomp_record_keyboard_stats(ctx, selected, termSum, 0.0f, advL1, 0.0f);
            return;
        }
        // Plain GPI keyboard: nonnegative softmax weights over the model-backed
        // scores of trusted options; commitment is governed by RR continuation
        // and nothing else.
        float prevMag = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            if (isfinite(w) && w > 0.0f) prevMag += w;
        }
        bool refreshKeyboard = !continuing || prevMag <= 0.0f;
        if (refreshKeyboard) {
            float mag = 0.0f;
            bool haveBest = bestScore > -FLT_MAX;
            for (uint k = 0u; k < ctx.K; ++k) {
                bool trusted = isfinite(tgG[k]) && haveBest;
                float qAdv = trusted ? tgG[k] : 0.0f;
                float dur = ctx.durationHat[k];
                if (!isfinite(dur) || dur <= 0.0f) dur = 1.0f;
                float modelErr = OAK_ABLATE_SEARCH_CONTROL ? 0.0f : max(ctx.prioOpt[k], 0.0f);
                if (!isfinite(modelErr)) modelErr = 0.0f;
                float score = (qAdv - modelErr) / dur;
                float lam = trusted ? exp(score - bestScore) : 0.0f;
                if (!isfinite(lam)) lam = 0.0f;
                ctx.keyboard[k] = lam;
                mag += lam;
            }
            if (mag <= 0.0f) {
                uint probe = ctx.stepCount % max(ctx.K, 1u);
                for (uint k = 0u; k < ctx.K; ++k)
                    ctx.keyboard[k] = (k == probe) ? ctx.invK : 0.0f;
            }
            ctx.keyboardOrigin[0] = 0u;
        }
        // Signed one-step planning backup at the current state: best model-backed
        // duration-normalized, model-error-pessimistic advantage over v(s). Never
        // clamped nonnegative — max(·,0) here is a value-inflation pump.
        float planDelta = (OAK_ABLATE_PLANNING || bestScore <= -FLT_MAX) ? 0.0f : bestScore;
        ctx.planTd[0] = isfinite(planDelta) ? planDelta : 0.0f;
        ctx.selectedOption[0] = selected;
        fcstomp_store_learn_scales(ctx, tgPrev, tgCurr);
        float keyL1 = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float w = ctx.keyboard[k];
            if (isfinite(w)) keyL1 += fabs(w);
        }
        fcstomp_record_keyboard_stats(ctx, selected, termSum, keyL1, advL1,
                                      keyL1 > 0.0f ? 1.0f : 0.0f);
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
            float sd = standardized_obs(ctx, o);
            for (uint m = 1u; m <= ctx.traceLevels; ++m) {
                uint slot = (m - 1u) * ctx.O + o;
                float rate = 1.0f / float(1u << m);
                ctx.obsTraceMulti[slot] += rate * (sd - ctx.obsTraceMulti[slot]);
            }
        }
    }
    if (nf > 0u) ctx.counters[OAK_CNT_OBS_NF] += nf;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    // Banked winner-take-all: thread tid owns bank tid (R/32 slots). Each bank
    // contributes exactly one active feature per step (its max-|x| slot); all
    // other slots output EXACTLY 0, so sparse dots over the active set are
    // exact. Statistics update densely (cheap); learning sees active steps.
    uint bank = ctx.R / 32u;
    uint lo = tid * bank, hi = lo + bank;
    uint winner = lo;
    float winAbs = -1.0f, winX = 0.0f;
    for (uint i = lo; i < hi; ++i) {
        float x = 0.0f;
        bool directBias = false;
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            directBias = (i == 0u);
        } else if (i < ctx.anchorCount) {
            uint width = max(linear_input_width(ctx), 1u);
            uint channel = min(i, width - 1u);
            float raw = linear_input(ctx, channel);
            directBias = (channel + 1u == width);
            if (!directBias) {
                uint age = ctx.featureAge[i];
                float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
                ctx.featureMean[i] += rate * (raw - ctx.featureMean[i]);
                ctx.featureSM[i] += rate * (raw * raw - ctx.featureSM[i]);
                float var = max(ctx.featureSM[i] - ctx.featureMean[i] * ctx.featureMean[i], 0.0f);
                float ridge = 1.0f / (float(age) + 1.0f);
                x = (raw - ctx.featureMean[i]) * rsqrt(var + ridge);
                if (!isfinite(x)) { x = 0.0f; ctx.counters[OAK_CNT_OBS_NF] += 1u; }
            }
        } else {
            uint pa = ctx.featureParentA[i] % ctx.R;
            uint pb = ctx.featureParentB[i] % ctx.R;
            float a = prev[pa], b = prev[pb];
            uint kind = ctx.featureKind[i] % 3u;
            float raw = (kind == 0u) ? a * b
                      : (kind == 1u) ? (a + b) * 0.7071067811865475f
                                     : a * (b - a);
            uint age = ctx.featureAge[i];
            float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
            ctx.featureMean[i] += rate * (raw - ctx.featureMean[i]);
            ctx.featureSM[i] += rate * (raw * raw - ctx.featureSM[i]);
            float var = max(ctx.featureSM[i] - ctx.featureMean[i] * ctx.featureMean[i], 0.0f);
            float ridge = 1.0f / (float(age) + 1.0f);
            x = (raw - ctx.featureMean[i]) * rsqrt(var + ridge);
            if (!isfinite(x)) { x = 0.0f; ctx.counters[OAK_CNT_OBS_NF] += 1u; }
        }
        if (directBias) x = 1.0f;
        curr[i] = 0.0f;
        float ax = fabs(x);
        if (ax > winAbs) { winAbs = ax; winner = i; winX = x; }
    }
    curr[winner] = winX;
    actC[tid] = winner;
    if (ctx.featureAge[winner] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
        ctx.featureAge[winner] += 1u;
    simdgroup_barrier(mem_flags::mem_device);
    // tg caches hold the ACTIVE values of prev/curr (32 each), indexed by lane.
    {
        device const uint *actP = act_at(ctx, ctx.stateSlot);
        tgPrevState[tid] = prev[actP[tid]];
        tgCurrState[tid] = winX;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void oak_stage_encode(Ctx ctx, uint tid, uint tgW,
                                    threadgroup float *tgG,
                                    threadgroup float *tgCont,
                                    threadgroup float *tgTerm,
                                    threadgroup float *tgTdPsi,
                                    threadgroup float *tgScalars,
                                    threadgroup float *tgPrevState,
                                    threadgroup float *tgCurrState) {
    FeatureConstruction(ctx, tid, tgW, tgPrevState, tgCurrState);
    fcstomp_prediction_control_loop(ctx, tid, tgPrevState, tgCurrState, tgG, tgCont, tgTerm,
                                    tgTdPsi, tgScalars);
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
                       uint2 tpos [[thread_position_in_threadgroup]],
                       uint2 tdim [[threads_per_threadgroup]],
                       uint2 tgPos [[threadgroup_position_in_grid]]) {
    uint tid = tpos.x, tgW = tdim.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    oak_stage_encode(ctx, tid, tgW, tgG, tgCont, tgTerm, tgTdPsi, tgScalars,
                     tgPrevState, tgCurrState);
}

static inline void oak_plan_one(Ctx ctx, uint k,
                                device const float *prev,
                                device const float *curr) {
    float cont = ctx.continuation[k];
    if (OAK_ABLATE_OPTION_MODELS) {
        ctx.optRewardTd[k] = 0.0f;
        ctx.durationTd[k] = 0.0f;
        ctx.durationHat[k] = 1.0f;
        ctx.prioOpt[k] = 0.0f;
        ctx.vsTd[k] = 0.0f;
        ctx.vsDot[k] = 0.0f;
        ctx.vsScale[k] = 0.0f;
        ctx.vsAuxScale[k] = 0.0f;
        return;
    }

        device const uint *actP = act_at(ctx, ctx.stateSlot);
        device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
        float rHat = 0.0f, rPrev = 0.0f, lRaw = 0.0f, lPrev = 0.0f;
        float vsPrev = 0.0f, vsNext = 0.0f, vX = 0.0f, vsDot = 0.0f;
        float bootNorm = 0.0f, xNormA = 0.0f;
        for (uint a = 0u; a < 32u; ++a) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            uint idxP = ip * ctx.K + k, idxC = ic * ctx.K + k;
            float xp = prev[ip], xc = curr[ic];
            rPrev = fma(ctx.optReward[idxP], xp, rPrev);
            rHat = fma(ctx.optReward[idxC], xc, rHat);
            lPrev = fma(ctx.optDuration[idxP], xp, lPrev);
            lRaw = fma(ctx.optDuration[idxC], xc, lRaw);
            vsPrev = fma(ctx.vStop[idxP], xp, vsPrev);
            vsNext = fma(ctx.vStop[idxC], xc, vsNext);
            vX = fma(ctx.valueW[ic], xc, vX);
            vsDot = fma(ctx.vsAux[idxP], xp, vsDot);
            // bootNorm = sum alpha_i (prev_i - cont*curr_i)^2, decomposed over
            // the two active sets (exact: dense states are zero off-set).
            float ap = ctx.alphaModel[ip];
            if (isfinite(ap) && ap > 0.0f) {
                bootNorm += ap * (xp * xp - 2.0f * cont * xp * curr[ip]);
                xNormA = fma(ap * xp, xp, xNormA);
            }
            float ac = ctx.alphaModel[ic];
            if (isfinite(ac) && ac > 0.0f) bootNorm += ac * cont * cont * xc * xc;
        }
        bootNorm = max(bootNorm, 0.0f);
        float rTd = ctx.reward[0] + cont * rHat - rPrev;
        float dTd = 1.0f + cont * lRaw - lPrev;
        // vStop GVF: cumulant 0, continuation cont, payoff v(x') at stopping.
        float vsTd = cont * vsNext + (1.0f - cont) * vX - vsPrev;
        bool modelFinite = isfinite(rTd) && isfinite(dTd) && isfinite(lRaw) && isfinite(vsTd);
        if (!isfinite(rTd)) rTd = 0.0f;
        if (!isfinite(dTd)) dTd = 0.0f;
        if (!isfinite(vsTd)) vsTd = 0.0f;
        ctx.optRewardTd[k] = rTd;
        ctx.durationTd[k] = dTd;
        ctx.vsTd[k] = vsTd;
        ctx.vsDot[k] = isfinite(vsDot) ? vsDot : 0.0f;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float sDen = fma(rho, bootNorm, 1.0f);
        ctx.vsScale[k] = (isfinite(sDen) && sDen > 0.0f) ? (1.0f / sDen) : 0.0f;
        float aDen = fma(rho, xNormA, 1.0f);
        ctx.vsAuxScale[k] = (isfinite(aDen) && aDen > 0.0f) ? (1.0f / aDen) : 0.0f;
        float dur = decode_duration(lRaw);
        ctx.durationHat[k] = dur;
        float modelErr = vsTd * vsTd + rTd * rTd
            + (ctx.scalars->rho * dTd) * (ctx.scalars->rho * dTd);
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

// All options planned in parallel: thread t owns option t (pure thread remap
// of the former serial k-loop; the j threads only idled — bit-identical class).
static inline void oak_stage_plan_all(Ctx ctx, uint tid) {
    if (tid >= ctx.K) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    oak_plan_one(ctx, tid, prev, curr);
    // Per-(k,d) actor coefficient, hoisted out of the K*D*A learn elements
    // (they were 60% of learn traffic at ~6 device accesses each).
    {
        uint k = tid;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float adv = ctx.actorAdv[k];
        if (!isfinite(adv)) adv = 0.0f;
        float actorAlpha = 1.0f / float(max(ctx.D * 32u, 1u));
        float xNorm = ctx.xNormPrev[0];
        for (uint d = 0u; d < ctx.D; ++d) {
            float targetGrad = ctx.optionActorCredit[k * ctx.D + d];
            if (!isfinite(targetGrad)) targetGrad = 0.0f;
            float gradScale = rho * targetGrad;
            float denom = 1.0f + actorAlpha * adv * adv * gradScale * gradScale * xNorm;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float c = actorAlpha * adv * gradScale * scale;
            ctx.polCoef[k * ctx.D + d] = isfinite(c) ? c : 0.0f;
        }
    }
}

static inline void oak_stage_plan_option(Ctx ctx, uint k, uint j,
                                         threadgroup float *tgModelErr) {
    if (j == 0u) {
        device const float *prev = state_at(ctx, ctx.stateSlot);
        device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
        oak_plan_one(ctx, k, prev, curr);
    }
    (void)tgModelErr;
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
    oak_stage_plan_option(ctx, k, j, tgModelErr);
}

inline uint oak_learn_total(Ctx ctx) {
    // Active-column banks (A = 32): value P+C, psi P+C, vStop P+C, policy P,
    // optReward P, optDuration P, psiAux P, vsAux P.
    return 2u * 32u + 2u * ctx.K * 32u + 2u * ctx.K * 32u
        + ctx.K * ctx.D * 32u + 4u * ctx.K * 32u;
}

static inline void oak_learn_element(Ctx ctx, device const float *prev,
                                     device const float *curr, uint gid) {
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    const uint A = 32u;
    uint nValP = A, nValC = A;
    uint nPsiP = ctx.K * A, nPsiC = ctx.K * A;
    uint nVSP = ctx.K * A, nVSC = ctx.K * A;
    uint nPi = ctx.K * ctx.D * A;
    uint nOR = ctx.K * A;

    if (gid < nValP) {
        if (OAK_ABLATE_GVF_PREDICTION) return;
        // P-bank owns i fully (both terms; curr[i] is a dense-exact read) so the
        // C-bank never touches the same weight — race-free by ownership.
        uint i = act_of(actP, gid);
        float w = ctx.valueW[i];
        float delta = ctx.td[0] * prev[i] + ctx.planTd[0] * curr[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta * ctx.valueScale[0]);
        return;
    }
    gid -= nValP;
    if (gid < nValC) {
        if (OAK_ABLATE_GVF_PREDICTION) return;
        uint i = act_of(actC, gid);
        if (actP[i / (ctx.R / 32u)] == i) return;  // index-exact: owned by the P-bank
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * ctx.planTd[0] * curr[i] * ctx.valueScale[0]);
        return;
    }
    gid -= nValC;
    if (gid < nPsiP) {
        if (OAK_ABLATE_RR_SUBTASKS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float cont = ctx.continuation[k];
        if (!isfinite(cont)) cont = 0.0f;
        float w = ctx.psi[(i) * ctx.K + k];
        // P-bank owns i fully: the GTD2 grad including the -cont*curr correction.
        float g = (cont > 0.0f) ? (ctx.hDotPsi[k] * (prev[i] - cont * curr[i]))
                                : (ctx.tdPsi[k] * prev[i]);
        ctx.psi[(i) * ctx.K + k] = safe_w(w, w + ctx.alphaPsi[i] * rho * g * ctx.headScale[k]);
        return;
    }
    gid -= nPsiP;
    if (gid < nPsiC) {
        if (OAK_ABLATE_RR_SUBTASKS) return;
        uint k = gid / A, i = act_of(actC, gid % A);
        if (actP[i / (ctx.R / 32u)] == i) return;  // index-exact: owned by the P-bank
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float cont = ctx.continuation[k];
        if (!isfinite(cont) || cont <= 0.0f) return;
        float w = ctx.psi[(i) * ctx.K + k];
        float g = -ctx.hDotPsi[k] * cont * curr[i];
        ctx.psi[(i) * ctx.K + k] = safe_w(w, w + ctx.alphaPsi[i] * rho * g * ctx.headScale[k]);
        return;
    }
    gid -= nPsiC;
    if (gid < nVSP) {
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float cont = ctx.continuation[k];
        if (!isfinite(cont)) cont = 0.0f;
        float w = ctx.vStop[(i) * ctx.K + k];
        float g = (cont > 0.0f) ? (ctx.vsDot[k] * (prev[i] - cont * curr[i]))
                                : (ctx.vsTd[k] * prev[i]);
        ctx.vStop[(i) * ctx.K + k] = safe_w(w, w + ctx.alphaModel[i] * rho * g * ctx.vsScale[k]);
        return;
    }
    gid -= nVSP;
    if (gid < nVSC) {
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / A, i = act_of(actC, gid % A);
        if (actP[i / (ctx.R / 32u)] == i) return;  // index-exact: owned by the P-bank
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float cont = ctx.continuation[k];
        if (!isfinite(cont) || cont <= 0.0f) return;
        float w = ctx.vStop[(i) * ctx.K + k];
        float g = -ctx.vsDot[k] * cont * curr[i];
        ctx.vStop[(i) * ctx.K + k] = safe_w(w, w + ctx.alphaModel[i] * rho * g * ctx.vsScale[k]);
        return;
    }
    gid -= nVSC;
    if (gid < nPi) {
        if (OAK_ABLATE_OPTION_POLICIES) return;
        uint k = gid / (ctx.D * A);
        uint rem = gid - k * ctx.D * A;
        uint d = rem / A, i = act_of(actP, rem % A);
        float w = ctx.optPolicy[(i) * ctx.K * ctx.D + k * ctx.D + d];
        ctx.optPolicy[(i) * ctx.K * ctx.D + k * ctx.D + d] =
            safe_w(w, w + ctx.polCoef[k * ctx.D + d] * prev[i]);
        return;
    }
    gid -= nPi;
    if (gid < nOR) {
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float w = ctx.optReward[(i) * ctx.K + k];
        ctx.optReward[(i) * ctx.K + k] =
            safe_w(w, w + ctx.alphaModel[i] * rho * ctx.optRewardTd[k] * prev[i] * ctx.modelScale[k]);
        return;
    }
    gid -= nOR;
    if (gid < nOR) {
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k]; if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float denom = ctx.modelDenom[k];
        float dEnergy = ctx.durationTd[k] * ctx.durationTd[k];
        if (isfinite(dEnergy)) denom += ctx.invR * dEnergy;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float w = ctx.optDuration[(i) * ctx.K + k];
        ctx.optDuration[(i) * ctx.K + k] =
            safe_w(w, w + ctx.alphaModel[i] * rho * ctx.durationTd[k] * prev[i] * scale);
        return;
    }
    gid -= nOR;
    if (gid < nOR) {
        if (OAK_ABLATE_RR_SUBTASKS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float h = ctx.psiAux[(i) * ctx.K + k];
        float err = ctx.tdPsi[k] - ctx.hDotPsi[k];
        ctx.psiAux[(i) * ctx.K + k] =
            safe_w(h, h + ctx.alphaPsi[i] * rho * err * prev[i] * ctx.psiAuxScale[k]);
        return;
    }
    gid -= nOR;
    {
        if (OAK_ABLATE_OPTION_MODELS) return;
        uint k = gid / A, i = act_of(actP, gid % A);
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float h = ctx.vsAux[(i) * ctx.K + k];
        float err = ctx.vsTd[k] - ctx.vsDot[k];
        ctx.vsAux[(i) * ctx.K + k] =
            safe_w(h, h + ctx.alphaModel[i] * rho * err * prev[i] * ctx.vsAuxScale[k]);
        return;
    }
}


// Quad-vectorized learn (megastep path): feature-major layout makes 4
// consecutive heads' weights one aligned float4. Each thread owns stationary
// head-quads (per-quad scalars loaded once into registers) and iterates the
// active set with one vector RMW per quad — cutting the dominant instruction
// stream ~4x. Per-component operand order matches the scalar path exactly
// (bit-identical class).
static inline void oak_learn_quads(Ctx ctx, device const float *prev,
                                   device const float *curr, uint tid, uint tw) {
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    const uint A = 32u;
    uint bankSize = ctx.R / 32u;
    uint K4 = ctx.K / 4u;
    uint KD4 = (ctx.K * ctx.D) / 4u;
    uint total = KD4 + 8u * K4;
    for (uint q = tid; q < total; q += tw) {
        if (q < KD4) {
            if (OAK_ABLATE_OPTION_POLICIES) continue;
            uint kd0 = q * 4u;
            float4 c = *(device const float4 *)&ctx.polCoef[kd0];
            for (uint a = 0u; a < A; ++a) {
                uint i = act_of(actP, a);
                float x = prev[i];
                device float4 *w4 = (device float4 *)&ctx.optPolicy[i * ctx.K * ctx.D + kd0];
                float4 w = *w4;
                w.x = safe_w(w.x, w.x + c.x * x);
                w.y = safe_w(w.y, w.y + c.y * x);
                w.z = safe_w(w.z, w.z + c.z * x);
                w.w = safe_w(w.w, w.w + c.w * x);
                *w4 = w;
            }
            continue;
        }
        uint qq = q - KD4;
        uint bank = qq / K4;
        uint k0 = (qq % K4) * 4u;
        float4 rho, cont;
        for (uint c4 = 0u; c4 < 4u; ++c4) {
            float r = ctx.importance[k0 + c4];
            if (!isfinite(r) || r < 0.0f) r = 0.0f;
            rho[c4] = r;
            float cn = ctx.continuation[k0 + c4];
            cont[c4] = isfinite(cn) ? cn : 0.0f;
        }
        if (bank == 0u || bank == 1u) {
            // psi P / psi C (GTD2)
            if (OAK_ABLATE_RR_SUBTASKS) continue;
            float4 hD, sc, td;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                hD[c4] = ctx.hDotPsi[k0 + c4];
                sc[c4] = ctx.headScale[k0 + c4];
                td[c4] = ctx.tdPsi[k0 + c4];
            }
            if (bank == 0u) {
                for (uint a = 0u; a < A; ++a) {
                    uint i = act_of(actP, a);
                    float xp = prev[i], xc = curr[i];
                    float aP = ctx.alphaPsi[i];
                    device float4 *w4 = (device float4 *)&ctx.psi[i * ctx.K + k0];
                    float4 w = *w4;
                    for (uint c4 = 0u; c4 < 4u; ++c4) {
                        float g = (cont[c4] > 0.0f) ? (hD[c4] * (xp - cont[c4] * xc))
                                                    : (td[c4] * xp);
                        w[c4] = safe_w(w[c4], w[c4] + aP * rho[c4] * g * sc[c4]);
                    }
                    *w4 = w;
                }
            } else {
                for (uint a = 0u; a < A; ++a) {
                    uint i = act_of(actC, a);
                    if (actP[i / bankSize] == i) continue;
                    float xc = curr[i];
                    float aP = ctx.alphaPsi[i];
                    device float4 *w4 = (device float4 *)&ctx.psi[i * ctx.K + k0];
                    float4 w = *w4;
                    for (uint c4 = 0u; c4 < 4u; ++c4) {
                        if (cont[c4] <= 0.0f) continue;
                        float g = -hD[c4] * cont[c4] * xc;
                        w[c4] = safe_w(w[c4], w[c4] + aP * rho[c4] * g * sc[c4]);
                    }
                    *w4 = w;
                }
            }
            continue;
        }
        if (bank == 2u || bank == 3u) {
            // vStop P / C (GTD2)
            if (OAK_ABLATE_OPTION_MODELS) continue;
            float4 vD, vsc, vtd;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                vD[c4] = ctx.vsDot[k0 + c4];
                vsc[c4] = ctx.vsScale[k0 + c4];
                vtd[c4] = ctx.vsTd[k0 + c4];
            }
            if (bank == 2u) {
                for (uint a = 0u; a < A; ++a) {
                    uint i = act_of(actP, a);
                    float xp = prev[i], xc = curr[i];
                    float aM = ctx.alphaModel[i];
                    device float4 *w4 = (device float4 *)&ctx.vStop[i * ctx.K + k0];
                    float4 w = *w4;
                    for (uint c4 = 0u; c4 < 4u; ++c4) {
                        float g = (cont[c4] > 0.0f) ? (vD[c4] * (xp - cont[c4] * xc))
                                                    : (vtd[c4] * xp);
                        w[c4] = safe_w(w[c4], w[c4] + aM * rho[c4] * g * vsc[c4]);
                    }
                    *w4 = w;
                }
            } else {
                for (uint a = 0u; a < A; ++a) {
                    uint i = act_of(actC, a);
                    if (actP[i / bankSize] == i) continue;
                    float xc = curr[i];
                    float aM = ctx.alphaModel[i];
                    device float4 *w4 = (device float4 *)&ctx.vStop[i * ctx.K + k0];
                    float4 w = *w4;
                    for (uint c4 = 0u; c4 < 4u; ++c4) {
                        if (cont[c4] <= 0.0f) continue;
                        float g = -vD[c4] * cont[c4] * xc;
                        w[c4] = safe_w(w[c4], w[c4] + aM * rho[c4] * g * vsc[c4]);
                    }
                    *w4 = w;
                }
            }
            continue;
        }
        if (bank == 4u) {
            if (OAK_ABLATE_OPTION_MODELS) continue;
            float4 rtd, msc;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                rtd[c4] = ctx.optRewardTd[k0 + c4];
                msc[c4] = ctx.modelScale[k0 + c4];
            }
            for (uint a = 0u; a < A; ++a) {
                uint i = act_of(actP, a);
                float xp = prev[i];
                float aM = ctx.alphaModel[i];
                device float4 *w4 = (device float4 *)&ctx.optReward[i * ctx.K + k0];
                float4 w = *w4;
                for (uint c4 = 0u; c4 < 4u; ++c4)
                    w[c4] = safe_w(w[c4], w[c4] + aM * rho[c4] * rtd[c4] * xp * msc[c4]);
                *w4 = w;
            }
            continue;
        }
        if (bank == 5u) {
            if (OAK_ABLATE_OPTION_MODELS) continue;
            float4 dtd, dsc;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                float dTd = ctx.durationTd[k0 + c4];
                dtd[c4] = dTd;
                float denom = ctx.modelDenom[k0 + c4];
                float dE = dTd * dTd;
                if (isfinite(dE)) denom += ctx.invR * dE;
                dsc[c4] = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            }
            for (uint a = 0u; a < A; ++a) {
                uint i = act_of(actP, a);
                float xp = prev[i];
                float aM = ctx.alphaModel[i];
                device float4 *w4 = (device float4 *)&ctx.optDuration[i * ctx.K + k0];
                float4 w = *w4;
                for (uint c4 = 0u; c4 < 4u; ++c4)
                    w[c4] = safe_w(w[c4], w[c4] + aM * rho[c4] * dtd[c4] * xp * dsc[c4]);
                *w4 = w;
            }
            continue;
        }
        if (bank == 6u) {
            if (OAK_ABLATE_RR_SUBTASKS) continue;
            float4 err, asc;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                err[c4] = ctx.tdPsi[k0 + c4] - ctx.hDotPsi[k0 + c4];
                asc[c4] = ctx.psiAuxScale[k0 + c4];
            }
            for (uint a = 0u; a < A; ++a) {
                uint i = act_of(actP, a);
                float xp = prev[i];
                float aP = ctx.alphaPsi[i];
                device float4 *w4 = (device float4 *)&ctx.psiAux[i * ctx.K + k0];
                float4 w = *w4;
                for (uint c4 = 0u; c4 < 4u; ++c4)
                    w[c4] = safe_w(w[c4], w[c4] + aP * rho[c4] * err[c4] * xp * asc[c4]);
                *w4 = w;
            }
            continue;
        }
        {
            if (OAK_ABLATE_OPTION_MODELS) continue;
            float4 err, asc;
            for (uint c4 = 0u; c4 < 4u; ++c4) {
                err[c4] = ctx.vsTd[k0 + c4] - ctx.vsDot[k0 + c4];
                asc[c4] = ctx.vsAuxScale[k0 + c4];
            }
            for (uint a = 0u; a < A; ++a) {
                uint i = act_of(actP, a);
                float xp = prev[i];
                float aM = ctx.alphaModel[i];
                device float4 *w4 = (device float4 *)&ctx.vsAux[i * ctx.K + k0];
                float4 w = *w4;
                for (uint c4 = 0u; c4 < 4u; ++c4)
                    w[c4] = safe_w(w[c4], w[c4] + aM * rho[c4] * err[c4] * xp * asc[c4]);
                *w4 = w;
            }
        }
    }
    // value banks (feature-indexed, scalar)
    if (!OAK_ABLATE_GVF_PREDICTION) {
        for (uint e = tid; e < 2u * A; e += tw) {
            if (e < A) {
                uint i = act_of(actP, e);
                float w = ctx.valueW[i];
                float delta = ctx.td[0] * prev[i] + ctx.planTd[0] * curr[i];
                ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta * ctx.valueScale[0]);
            } else {
                uint i = act_of(actC, e - A);
                if (actP[i / bankSize] == i) continue;
                float w = ctx.valueW[i];
                ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * ctx.planTd[0] * curr[i] * ctx.valueScale[0]);
            }
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
    if (gid >= oak_learn_total(ctx)) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    oak_learn_element(ctx, prev, curr, gid);
}

inline void emit_action_grid(Ctx ctx, threadgroup const float *curr, uint group, uint tid, uint tgW) {
    uint gid = group * tgW + tid;
    if (gid >= ctx.K * ctx.D) return;
    uint selected = ctx.selectedOption[0] % ctx.K;
    uint k = gid / ctx.D, d = gid % ctx.D;
    device const uint *act = act_at(ctx, ctx.stateSlot);
    if (OAK_ABLATE_OPTION_POLICIES) {
        ctx.optionPolicyMean[k * ctx.D + d] = 0.0f;
        ctx.muRaw[k * ctx.D + d] = 0.0f;
        if (gid < ctx.D) ctx.policyMean[d] = 0.0f;
        return;
    }
    float mu = 0.0f;
    for (uint a = 0u; a < 32u; ++a)
        mu = fma(ctx.optPolicy[act_of(act, a) * ctx.K * ctx.D + k * ctx.D + d], curr[a], mu);
    // Raw mean (pre-fallback) stashed for the blend pass — kills the K*32
    // re-gather per action dim.
    ctx.muRaw[k * ctx.D + d] = mu;
    if (!isfinite(mu)) mu = ctx.optionPolicyMean[k * ctx.D + d];
    ctx.optionPolicyMean[k * ctx.D + d] = mu;
}

inline void emit_blend_grid(Ctx ctx, uint tid) {
    if (tid >= ctx.D) return;
    uint d = tid;
    uint selected = ctx.selectedOption[0] % ctx.K;
    float blend = 0.0f, wNorm = 0.0f, fallback = 0.0f;
    for (uint ok = 0u; ok < ctx.K; ++ok) {
        float om = ctx.muRaw[ok * ctx.D + d];
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

static inline void oak_stage_replace(Ctx ctx, uint tid, uint tgW,
                                     threadgroup float *tgCurrState) {
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    if (tid < 32u) tgCurrState[tid] = curr[act_at(ctx, 1u - ctx.stateSlot)[tid]];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float invT = track_rate(ctx.stepCount, ctx.R);
    // Utility / meta-learning advance on ACTIVE features only (age = data seen).
    if (tid < 32u) {
        uint i = actP[tid];
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
        // Behavioral load (GraMa lineage): value + action-path weight magnitudes
        // persist when learning goes quiet, so the curator does not knife the
        // features carrying a converged policy (the self-dismantling path).
        uint sel = ctx.selectedOption[0] % ctx.K;
        float actLoad = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d)
            actLoad += fabs(ctx.optPolicy[i * ctx.K * ctx.D + sel * ctx.D + d]);
        float load = fabs(prev[i]) * (fabs(ctx.valueW[i]) + actLoad);
        if (!isfinite(load)) load = 0.0f;
        float g = fabs(planningSignal * prev[i])
                + fabs((psiSignal + modelSignal + keyboardSignal) * prev[i])
                + load;
        ctx.utilFeat[i] += invT * (g - ctx.utilFeat[i]);
        uint age = ctx.featureAge[i];
        metaopt_update_slot(ctx.alpha, ctx.beta, ctx.hMeta, ctx.gradEnergy,
                            i, ctx.td[0], prev[i], age, ctx.metaEta, ctx.invR);
        uint metaSlot = ctx.stepCount % 32u;
        if (tid == metaSlot) {
            // Signed per-head TDs (round-robin over heads): IDBD's mechanism is
            // the sign correlation of successive gradients; a magnitude signal
            // self-correlates and ratchets beta monotonically.
            uint kSlot = ctx.stepCount % max(ctx.K, 1u);
            float psiSigned = ctx.tdPsi[kSlot];
            float modelSigned = ctx.optRewardTd[kSlot];
            metaopt_update_slot(ctx.alphaPsi, ctx.betaPsi, ctx.hPsi, ctx.gradEnergyPsi,
                                i, psiSigned, prev[i], age, ctx.metaEta, ctx.invR);
            metaopt_update_slot(ctx.alphaModel, ctx.betaModel, ctx.hModel, ctx.gradEnergyModel,
                                i, modelSigned, prev[i], age, ctx.metaEta, ctx.invR);
        }
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
    if (!OAK_ABLATE_GENERATE_AND_TEST) {
        // Nursery (candidate axis): shadow feature candidates learn their
        // standardization and a predictive utility from the stream every step,
        // in parallel, without touching behavior. Promotion is evidence-based.
        for (uint n = tid; n < ctx.Rn; n += tgW) {
            uint pa = ctx.nurParentA[n] % ctx.R;
            uint pb = ctx.nurParentB[n] % ctx.R;
            float a = prev[pa], b = prev[pb];
            uint kind = ctx.nurKind[n] % 3u;
            float raw = (kind == 0u) ? a * b
                      : (kind == 1u) ? (a + b) * 0.7071067811865475f
                                     : a * (b - a);
            uint age = ctx.nurAge[n];
            float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
            // One aligned float4 RMW per candidate ({mean, sm, util, utilSM}).
            device float4 *st4 = (device float4 *)&ctx.nurStats[(n) * 4u];
            float4 st = *st4;
            st.x += rate * (raw - st.x);
            st.y += rate * (raw * raw - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);
            float ridge = 1.0f / (float(age) + 1.0f);
            float xt = (raw - st.x) * rsqrt(var + ridge);
            float sig = fabs(ctx.td[0] * xt) + fabs(ctx.planTd[0] * xt);
            if (isfinite(sig)) {
                st.z += invT * (sig - st.z);
                st.w += invT * (sig * sig - st.w);
            }
            *st4 = st;
            if (age < 0xffffffffu) ctx.nurAge[n] = age + 1u;
        }
    }
    if (!OAK_ABLATE_GENERATE_AND_TEST) {
        // Parallel precompute for the (previously serial) curation scans:
        // per-candidate LCB and zeroed per-feature option-use scatter targets.
        for (uint n = tid; n < ctx.Rn; n += tgW) {
            float uN = ctx.nurStats[(n) * 4u + 2u];
            float lcb = -FLT_MAX;
            if (ctx.nurAge[n] >= ctx.R && isfinite(uN)) {
                float varU = max(ctx.nurStats[(n) * 4u + 3u] - uN * uN, 0.0f);
                float rateU = max(1.0f / (float(ctx.nurAge[n]) + 1.0f), ctx.invR);
                float v = uN - sqrt(varU * rateU);
                if (isfinite(v)) lcb = v;
            }
            ctx.lcbScr[n] = lcb;
        }
        for (uint i = tid; i < ctx.R; i += tgW) ctx.optUseByFeat[i] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_device);
    {
        // Rotating-stripe explosion scan: covers the whole param arena every 32
        // steps, so death certificates are dated in-kernel (within 32 steps)
        // instead of at heartbeat cadence. valueW is the params base pointer.
        uint total = ctx.R + 7u * ctx.K * ctx.R + ctx.K * ctx.D * ctx.R;
        uint stripe = ctx.stepCount % ctx.R;
        uint badElem = 0xffffffffu;
        for (uint e = stripe + ctx.R * tid; e < total; e += ctx.R * tgW) {
            float v = ctx.valueW[e];
            if (!isfinite(v) || fabs(v) > 1e8f) { badElem = e; break; }
        }
        if (badElem != 0xffffffffu && ctx.explosionStep[0] == 0xffffffffu) {
            ctx.explosionStep[0] = ctx.stepCount;
            ctx.explosionElem[0] = badElem;
        }
    }
    simdgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
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
    // Scatter per-option use onto features once (K ops) — replaces the 157x32
    // register scan inside the serial eligibility loop.
    for (uint k = 0u; k < ctx.K; ++k) {
        float use = ctx.utilOpt[k]
            + (OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(ctx.keyboard[k]));
        if (isfinite(use)) ctx.optUseByFeat[ctx.optFeature[k] % ctx.R] += use;
    }
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
	            float red = fabs(curr[i] * curr[pa]);
            if (!isfinite(red)) red = 0.0f;
            float optionUse = ctx.optUseByFeat[i];
		            float downstreamUse = optionUse + fabs(ctx.valueW[i]);
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
	        if (ctx.lastStopAge[k] < 0xffffffffu) ctx.lastStopAge[k] += 1u;
	    }
    // Subtask viability (generate-and-test, not interruption): an option that has
    // not stopped once in R*K steps is a degenerate subtask — its stopping-based
    // quantities (psi, n-hat, l-hat) are ill-defined and grow without bound.
    // Recycle the HEAD (fresh learner, same feature); behavior is untouched.
    if (!OAK_ABLATE_GENERATE_AND_TEST) {
        // R*K*K: catches true never-stoppers (observed SR explosions need >28k
        // stale steps) without churning heads that stop on option timescales.
        uint staleLimit = ctx.R * ctx.K * ctx.K;
        for (uint k = 0u; k < ctx.K; ++k) {
            if (ctx.lastStopAge[k] < staleLimit) continue;
            ctx.optAge[k] = 0u;
            ctx.lastStopAge[k] = 0u;
            ctx.utilOpt[k] = float(ctx.R);
            ctx.actorAdv[k] = 0.0f;
            ctx.psiRho[k] = 0.0f;
            uint feat = ctx.optFeature[k] % ctx.R;
            for (uint i = 0u; i < ctx.R; ++i) {
                ctx.psi[(i) * ctx.K + k] = 0.0f;
                ctx.psiAux[(i) * ctx.K + k] = 0.0f;
                ctx.optReward[(i) * ctx.K + k] = 0.0f;
                ctx.optDuration[(i) * ctx.K + k] = 0.0f;
                ctx.vStop[(i) * ctx.K + k] = 0.0f;
                ctx.vsAux[(i) * ctx.K + k] = 0.0f;
            }
            float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
            ctx.psi[(feat) * ctx.K + k] = invSqrtR;
            ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
            break;
        }
    }

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRed = eligibleRedSum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtil < meanEligibleUtil;
    bool highRedundancy = worstRed > meanEligibleRed;
    bool lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
			    bool structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);
    ctx.stats->generateTestEligibleCount = float(eligible);
		    // Nursery promotion (replace only on demonstrated better): the best
    // mature shadow candidate must beat the worst incumbent's TOTAL utility
    // (learning signal + behavioral load + downstream use). Maturity = R steps
    // of standardization/utility evidence.
    uint bestN = 0xffffffffu;
    float bestNUtil = -FLT_MAX;
    // LCB per candidate precomputed in parallel; serial pass is exact argmax.
    for (uint n = 0u; n < ctx.Rn; ++n) {
        float lcb = ctx.lcbScr[n];
        if (lcb > bestNUtil) { bestNUtil = lcb; bestN = n; }
    }
    // Promotion bar vs the MEAN eligible total utility, not the graveyard's
    // worst: with 157 incumbents the worst is always a near-zero slot, so any
    // mature candidate cleared the old bar — a promotion every ~70 steps whose
    // bank-wide resets ground knowledge faster than learning rebuilt it (the
    // median-decline driver) AND ramped serial curation work (the SPS decay:
    // 376K->260K windowed; G&T-ablated probe: flat). The fabric tear is only
    // worth an expected top-half contributor.
    bool doReplace = !OAK_ABLATE_GENERATE_AND_TEST
        && structurallyReplaceable && worst < ctx.R
        && bestN != 0xffffffffu
        && bestNUtil > meanEligibleUtil + meanEligibleDownstream;
    if (doReplace && worst < ctx.R) {
        uint pa = ctx.nurParentA[bestN] % ctx.R;
        uint pb = ctx.nurParentB[bestN] % ctx.R;
uint birth = ctx.featureBirthCount[worst] + 1u;
        ctx.featureBirthCount[worst] = birth;
        ctx.featureParentA[worst] = pa;
        ctx.featureParentB[worst] = pb;
        ctx.featureKind[worst] = ctx.nurKind[bestN] % 3u;
        // Warm start: the graduate keeps the statistics and age it earned in the
        // nursery — no cold-normalization shock on promotion.
        ctx.featureAge[worst] = ctx.nurAge[bestN];
        ctx.featureMean[worst] = ctx.nurStats[(bestN) * 4u];
        ctx.featureSM[worst] = ctx.nurStats[(bestN) * 4u + 1u];
        ctx.utilFeat[worst] = ctx.nurStats[(bestN) * 4u + 2u];
        ctx.valueW[worst] = 0.0f; ctx.bonusW[worst] = 0.0f;
        // Reseed the nursery slot: utility-guided parents, rotated kind.
        {
            uint spa = 0u, spb = (ctx.R > 1u) ? 1u : 0u;
            float bestA = -FLT_MAX, bestB = -FLT_MAX;
            for (uint i = 0u; i < ctx.R; ++i) {
                if (i == worst) continue;
                float uu = ctx.utilFeat[i];
                if (!isfinite(uu)) uu = -FLT_MAX;
                if (uu > bestA) { bestA = uu; spa = i; }
            }
            for (uint i = 0u; i < ctx.R; ++i) {
                if (i == worst || i == spa) continue;
                float decor = 1.0f / (1.0f + fabs(curr[i] * curr[spa]));
                float uu = ctx.utilFeat[i] * decor;
                if (!isfinite(uu)) uu = -FLT_MAX;
                if (uu > bestB) { bestB = uu; spb = i; }
            }
            ctx.nurParentA[bestN] = spa;
            ctx.nurParentB[bestN] = spb;
            ctx.nurKind[bestN] = (ctx.nurKind[bestN] + 1u) % 3u;
            ctx.nurAge[bestN] = 0u;
            ctx.nurStats[(bestN) * 4u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 1u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 2u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 3u] = 0.0f;
        }
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
			            ctx.psi[(i) * ctx.K + k] = 0.0f;
			            ctx.psiAux[(i) * ctx.K + k] = 0.0f;
			            ctx.optReward[(i) * ctx.K + k] = 0.0f;
		            ctx.optDuration[(i) * ctx.K + k] = 0.0f;
		            ctx.vStop[(i) * ctx.K + k] = 0.0f;
		            ctx.vsAux[(i) * ctx.K + k] = 0.0f;
		            for (uint d = 0u; d < ctx.D; ++d) ctx.optPolicy[(i) * ctx.K * ctx.D + k * ctx.D + d] = 0.0f;
	        }
			        if (ctx.D > 0u) {
			            uint slot = k % ctx.D;
			            float sign = ((k / ctx.D) % 2u == 0u) ? 1.0f : -1.0f;
			            ctx.optPolicy[k * ctx.D + slot] = sign / float(ctx.D);
			        }
		        // Feature `worst` is now a DIFFERENT signal: every head's weight on that
		        // index (and every transition row predicting it) is stale knowledge about
		        // the old feature. Leaving them in place tears the shared fabric: on a
		        // bootstrapped substrate the mismatch compounds into runaway model error
		        // (observed: lane-wide 1e9 blow-ups driven by curation churn). Clear the
		        // column bank-wide.
		        for (uint ok = 0u; ok < ctx.K; ++ok) {
		            ctx.psi[worst * ctx.K + ok] = 0.0f;
		            ctx.psiAux[worst * ctx.K + ok] = 0.0f;
		            ctx.optReward[worst * ctx.K + ok] = 0.0f;
		            ctx.optDuration[worst * ctx.K + ok] = 0.0f;
		            for (uint d = 0u; d < ctx.D; ++d)
		                ctx.optPolicy[worst * ctx.K * ctx.D + ok * ctx.D + d] = 0.0f;
		            ctx.vStop[worst * ctx.K + ok] = 0.0f;
		            ctx.vsAux[worst * ctx.K + ok] = 0.0f;
		        }
		        float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
		        ctx.psi[(worst) * ctx.K + k] = invSqrtR;
	        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
	        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
	    }
    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount = float(ctx.counters[OAK_CNT_TDPSI_NF] + ctx.counters[OAK_CNT_IMPORTANCE_NF]);
    ctx.stats->oakOptionReinitCount = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);
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
    oak_stage_replace(ctx, tid, tgW, tgCurrState);
}

static inline void oak_stage_action_emit(Ctx ctx, uint group, uint tid, uint tgW,
                                         threadgroup float *tgCurrState) {
    device const float *curr = state_at(ctx, ctx.stateSlot);
    device const uint *act = act_at(ctx, ctx.stateSlot);
    if (tid < 32u) tgCurrState[tid] = curr[act[tid]];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    emit_action_grid(ctx, tgCurrState, group, tid, tgW);
    simdgroup_barrier(mem_flags::mem_device);
    if (group == 0u) emit_blend_grid(ctx, tid);
}

kernel void oak_action_emit(device float *params [[buffer(0)]], device float *stateArena [[buffer(1)]], device uint *uintArena [[buffer(2)]], device KernelScalars *scalars [[buffer(3)]], device KernelStats *stats [[buffer(4)]], constant KernelConfig &cfg [[buffer(5)]], threadgroup float *tgCurrState [[threadgroup(0)]], uint2 tgPos [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]], uint2 tdim [[threads_per_threadgroup]]) {
    uint group = tgPos.x, tid = tpos.x, tgW = tdim.x;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, tgPos.y);
    oak_stage_action_emit(ctx, group, tid, tgW, tgCurrState);
}

constant uint HC_RNG_STREAM_ACT = 2u;

constant uint OAK_DEATH_NONE       = 0xffffffffu;
constant uint OAK_DET_ENV_REWARD   = 1u;
constant uint OAK_DET_POLICY_MEAN  = 2u;
constant uint OAK_DET_ENV_OBS      = 3u;

struct OakFleetEnvCfg {
    uint laneCount;
    uint episodeLen;
    uint policyMode;
    uint recordCapacity;
    float invObsFloor;
};

struct OakFleetInfoAcc { uint count, finiteCount; float sum, sumC, first, last; };

struct OakFleetLane {
    HCState s;
    uint elapsed, resets;
    uint steps, doneCount;
    float ctrlPending[6];
    float lastDesc[12];
    float deltaSum, deltaC, deltaPeak;
    float rewardSum, rewardC, rho;
    uint deathStep, deathDetector;
    float deathRewardSum, deathRho;
    OakFleetInfoAcc info[4];
};
static_assert(sizeof(OakFleetLane) == 356, "OakFleetLane layout drifted from the Swift mirror");

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

static inline void oak_stage_sample_act(device float *params,
                                        device float *stateArena,
                                        device uint *uintArena,
                                        device KernelScalars *scalars,
                                        device KernelStats *stats,
                                        constant KernelConfig &cfg,
                                        device OakFleetLane *lanes,
                                        device const uint *laneSeeds,
                                        constant OakFleetEnvCfg &ecfg,
                                        uint gid) {
    device OakFleetLane *lane = &lanes[gid];
    uint stepIdx = lane->steps;
    float desc[12];
    float ctrl[6];
    if (ecfg.policyMode == 1u) {
        for (int u = 0; u < 6; u++) ctrl[u] = 0.0f;
        for (int i = 0; i < 12; i++) desc[i] = 0.0f;
    } else if (ecfg.policyMode == 2u) {
        hc_rng_ctrl(laneSeeds[gid], stepIdx, ctrl);
        for (int d = 0; d < 6; d++) { desc[d] = ctrl[d]; desc[6 + d] = 0.0f; }
    } else {
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid);
        uint A = ctx.D / 2u;
        float sd = 1.0f / float(max(ctx.D, 1u));
        for (uint i = 0u; i < ctx.D; ++i) desc[i] = oak_safe(ctx.policyMean[i]);
        float z[6];
        oak_rng_normal6(laneSeeds[gid], stepIdx, z);
        for (uint o = 0u; o < ctx.K; ++o) ctx.importance[o] = 0.0f;
        for (uint d = 0u; d < A; ++d) {
            float mu = desc[d];
            float sample = oak_safe(mu + sd * z[d]);
            float a = max(-1.0f, min(1.0f, sample));
            ctrl[d] = a;
            ctx.actionSignal[d] = a;
            ctx.actionSignal[A + d] = 0.0f;
            float diffB = sample - mu;
            for (uint o = 0u; o < ctx.K; ++o) {
                uint off = o * ctx.D;
                float om = oak_safe(ctx.optionPolicyMean[off + d]);
                ctx.optionActorCredit[off + d] = sample - om;
                ctx.optionActorCredit[off + A + d] = 0.0f;
                float diffO = sample - om;
                ctx.importance[o] += diffB * diffB - diffO * diffO;
            }
        }
        // Exact one-step Gaussian ratio pi_k(a|s)/b(a|s), shared sigma; capped
        // at K (bank capacity) to bound a single step's off-policy influence.
        float inv2s2 = 1.0f / (2.0f * sd * sd);
        for (uint o = 0u; o < ctx.K; ++o) {
            float ratio = exp(ctx.importance[o] * inv2s2);
            ctx.importance[o] = isfinite(ratio) ? min(ratio, float(ctx.K)) : float(ctx.K);
        }
    }
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

static inline void oak_stage_sample_act_wide(Ctx ctx,
                                             device OakFleetLane *lanes,
                                             device const uint *laneSeeds,
                                             uint gid, uint tid,
                                             threadgroup float *scratch) {
    device OakFleetLane *lane = &lanes[gid];
    uint A = ctx.D / 2u;
    float sd = 1.0f / float(max(ctx.D, 1u));
    uint stepIdx = lane->steps;
    // Sampling spread across action dims (bit-identical: each thread computes
    // its own dim's Box-Muller pair redundantly — same values, parallel latency).
    if (tid < A) {
        uint d = tid;
        uint2 key = uint2(laneSeeds[gid], HC_RNG_STREAM_ACT);
        uint pair = d >> 1u;
        uint4 r = hc_philox4x32(key, uint4(stepIdx, pair >> 1u, 0u, 0u));
        // words for pair p within the two philox blocks (layout matches
        // oak_rng_normal6: raw = {r0.x,r0.y,r0.z,r0.w,r1.x,r1.y})
        uint w0, w1;
        if (pair == 0u) { w0 = r.x; w1 = r.y; }
        else if (pair == 1u) { w0 = r.z; w1 = r.w; }
        else { w0 = r.x; w1 = r.y; }
        float u1 = hc_u01open(w0);
        float u2 = hc_u01(w1);
        float rad = sqrt(-2.0f * log(u1));
        float ang = 2.0f * M_PI_F * u2;
        float z = ((d & 1u) == 0u) ? (rad * cos(ang)) : (rad * sin(ang));
        float mu = oak_safe(ctx.policyMean[d]);
        float sample = oak_safe(mu + sd * z);
        float a = max(-1.0f, min(1.0f, sample));
        lane->ctrlPending[d] = a;
        ctx.actionSignal[d] = a;
        ctx.actionSignal[A + d] = 0.0f;
        scratch[20u + d] = sample;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float desc[12];
        for (uint i = 0u; i < ctx.D; ++i) desc[i] = oak_safe(ctx.policyMean[i]);
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
    }
    if (tid < ctx.K) {
        uint off = tid * ctx.D;
        float acc = 0.0f;
        for (uint d = 0u; d < A; ++d) {
            float sample = scratch[20u + d];
            float om = oak_safe(ctx.optionPolicyMean[off + d]);
            ctx.optionActorCredit[off + d] = sample - om;
            ctx.optionActorCredit[off + A + d] = 0.0f;
            float diffB = sample - oak_safe(ctx.policyMean[d]);
            float diffO = sample - om;
            acc += diffB * diffB - diffO * diffO;
        }
        // Exact one-step Gaussian ratio pi_k(a|s)/b(a|s), shared sigma; capped
        // at K (bank capacity) to bound a single step's off-policy influence.
        float inv2s2 = 1.0f / (2.0f * sd * sd);
        float ratio = exp(acc * inv2s2);
        ctx.importance[tid] = isfinite(ratio) ? min(ratio, float(ctx.K)) : float(ctx.K);
    }
}

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
    oak_stage_sample_act(params, stateArena, uintArena, scalars, stats, cfg,
                         lanes, laneSeeds, ecfg, gid);
}

static inline void oak_stage_fleet_env(device float *params,
                                       device float *stateArena,
                                       device uint *uintArena,
                                       device KernelScalars *scalars,
                                       device KernelStats *stats,
                                       constant KernelConfig &cfg,
                                       device OakFleetLane *lanes,
                                       device const uint *laneSeeds,
                                       constant OakFleetEnvCfg &ecfg,
                                       device const int *recordSlot,
                                       device float *recordBuf,
                                       uint gid) {
    device OakFleetLane *lane = &lanes[gid];
    HCState s = lane->s;
    uint elapsed = lane->elapsed, resets = lane->resets;
    float ctrl[6];
    for (int u = 0; u < 6; u++) ctrl[u] = lane->ctrlPending[u];
    float reward;
    uint done;
    float info[4];
    if (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) {
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
    uint stepNum = lane->steps;

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

static inline void oak_stage_fleet_env_simd(device float *params,
                                            device float *stateArena,
                                            device uint *uintArena,
                                            device KernelScalars *scalars,
                                            device KernelStats *stats,
                                            constant KernelConfig &cfg,
                                            device OakFleetLane *lanes,
                                            device const uint *laneSeeds,
                                            constant OakFleetEnvCfg &ecfg,
                                            device const int *recordSlot,
                                            device float *recordBuf,
                                            device HCSimdRows *rows,
                                            uint gid, uint sgTid,
                                            threadgroup float *shr) {
    device OakFleetLane *lane = &lanes[gid];
    HCState s = lane->s;
    uint elapsed = lane->elapsed, resets = lane->resets;
    float ctrl[6];
    for (int u = 0; u < 6; u++) ctrl[u] = lane->ctrlPending[u];
    float reward;
    uint done;
    float info[4];
    if (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) {
        hc_env_reset(s, laneSeeds[gid], elapsed, resets);
        reward = 0.0f;
        done = 0u;
        info[0] = 0.0f; info[1] = -0.0f; info[2] = 0.0f; info[3] = 0.0f;
    } else {
        float xBefore = s.qpos[0];
        for (int u = 0; u < 6; u++) s.ctrl[u] = ctrl[u];
        for (int k = 0; k < HC_FRAME_SKIP; k++) hc_mj_step_simd(s, rows, sgTid, shr);
        float xv = (s.qpos[0] - xBefore) / (float(HC_FRAME_SKIP) * HC_TIMESTEP);
        float ctrlCost = 0.0f;
        for (int u = 0; u < 6; u++) ctrlCost += 0.1f * ctrl[u] * ctrl[u];
        reward = xv * 1.0f - ctrlCost;
        elapsed++;
        done = (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) ? 1u : 0u;
        info[0] = xv; info[1] = -ctrlCost; info[2] = s.qpos[0]; info[3] = xv;
    }
    if (sgTid != 0u) return;
    float obs[17];
    for (int i = 1; i < 9; i++) obs[i - 1] = s.qpos[i];
    for (int i = 0; i < 9; i++) obs[8 + i] = s.qvel[i];
    lane->s = s;
    lane->elapsed = elapsed;
    lane->resets = resets;
    lane->doneCount += done;
    uint stepNum = lane->steps;

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
    oak_stage_fleet_env(params, stateArena, uintArena, scalars, stats, cfg,
                        lanes, laneSeeds, ecfg, recordSlot, recordBuf, gid);
}

// Fence between agent stages of the SAME lane. All agent-stage data flow is
// lane-local (fleet lane-disjointness invariant) and one lane = one simdgroup,
// so a simdgroup-scope device fence suffices — except under WIDE2X, where the
// learn stage spans two simdgroups per lane. The env stage still crosses
// simdgroups (packed on simdgroup 0) and keeps full threadgroup barriers.
static inline void oak_lane_fence() {
    if (OAK_MEGA_WIDE2X) threadgroup_barrier(mem_flags::mem_device);
    else simdgroup_barrier(mem_flags::mem_device);
}

kernel void oak_megastep(device float *params [[buffer(0)]],
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
                         constant uint &stepsThisDispatch [[buffer(11)]],
                         device HCSimdRows *physRows [[buffer(12)]],
                         threadgroup float *tgG [[threadgroup(0)]],
                         threadgroup float *tgCont [[threadgroup(1)]],
                         threadgroup float *tgTerm [[threadgroup(2)]],
                         threadgroup float *tgTdPsi [[threadgroup(3)]],
                         threadgroup float *tgScalars [[threadgroup(4)]],
                         threadgroup float *tgPrevState [[threadgroup(5)]],
                         threadgroup float *tgCurrState [[threadgroup(6)]],
                         threadgroup float *tgPhys [[threadgroup(7)]],
                         uint tgIdx [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint slane [[thread_index_in_simdgroup]],
                         uint sgrp [[simdgroup_index_in_threadgroup]]) {
    uint laneBase = tgIdx * OAK_MEGA_G;
    if (laneBase >= ecfg.laneCount) return;
    uint lanesThisTg = min(OAK_MEGA_G, ecfg.laneCount - laneBase);
    uint laneSg = OAK_MEGA_WIDE2X ? (sgrp >> 1) : sgrp;
    bool laneLead = OAK_MEGA_WIDE2X ? ((sgrp & 1u) == 0u) : true;
    bool wideActive = laneLead && laneSg < lanesThisTg;
    bool learnActive = laneSg < lanesThisTg;
    uint learnTid = OAK_MEGA_WIDE2X ? (slane + 32u * (sgrp & 1u)) : slane;
    uint learnW = OAK_MEGA_WIDE2X ? 64u : 32u;
    uint wideLane = laneBase + laneSg;
    bool packedActive = tid < lanesThisTg;
    uint packedLane = laneBase + tid;
    uint off = laneSg * 32u;
    threadgroup float *lG = tgG + off, *lCont = tgCont + off, *lTerm = tgTerm + off,
                      *lTdPsi = tgTdPsi + off, *lScalars = tgScalars + off,
                      *lPrev = tgPrevState + off, *lCurr = tgCurrState + off;
    for (uint it = 0u; it < stepsThisDispatch; ++it) {
        if (ecfg.policyMode == 0u) {
            if (wideActive && (OAK_MEGA_STAGES & 1u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_encode(ctx, slane, 32u, lG, lCont, lTerm, lTdPsi, lScalars,
                                 lPrev, lCurr);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 2u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_plan_all(ctx, slane);
            }
            oak_lane_fence();
            if (learnActive && (OAK_MEGA_STAGES & 4u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                device const float *prev = state_at(ctx, ctx.stateSlot);
                device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
                oak_learn_quads(ctx, prev, curr, learnTid, learnW);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 8u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_replace(ctx, slane, 32u, lCurr);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 16u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                device const float *curr = state_at(ctx, ctx.stateSlot);
                device const uint *act = act_at(ctx, ctx.stateSlot);
                lCurr[slane] = curr[act[slane]];
                simdgroup_barrier(mem_flags::mem_threadgroup);
                uint emitGroups = (ctx.K * ctx.D + 31u) / 32u;
                for (uint g = 0u; g < emitGroups; ++g)
                    emit_action_grid(ctx, lCurr, g, slane, 32u);
                simdgroup_barrier(mem_flags::mem_device);
                emit_blend_grid(ctx, slane);
            }
            oak_lane_fence();
        }
        if (ecfg.policyMode == 0u) {
            if (wideActive && (OAK_MEGA_STAGES & 32u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_sample_act_wide(ctx, lanes, laneSeeds, wideLane, slane, lScalars);
            }
        } else if (packedActive && (OAK_MEGA_STAGES & 32u) != 0u) {
            oak_stage_sample_act(params, stateArena, uintArena, scalars, stats, cfg,
                                 lanes, laneSeeds, ecfg, packedLane);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (OAK_MEGA_ENV_COOP) {
                if (wideActive)
                    oak_stage_fleet_env_simd(params, stateArena, uintArena, scalars, stats,
                                             cfg, lanes, laneSeeds, ecfg, recordSlot,
                                             recordBuf, &physRows[wideLane], wideLane, slane,
                                             tgPhys + laneSg * uint(HC_SIMD_SCRATCH));
            } else if (packedActive) {
                oak_stage_fleet_env(params, stateArena, uintArena, scalars, stats, cfg,
                                    lanes, laneSeeds, ecfg, recordSlot, recordBuf, packedLane);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

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
