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
    float dynaBackupMean;
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
    device float *hDotPsi, *psiAuxScale, *psiRho, *optRho, *optFit;
    device float *alphaAct, *betaAct, *hAct, *energyAct;
    device float *impStage, *creditStage, *sampleScr;
    device float *rHatC, *lRawC, *vsNextC;
    device float *vsTd, *vsDot, *vsScale, *vsAuxScale;
    device float *planScale;
    device float *nurStats;
    device float *muRaw, *polCoef, *optUseByFeat, *lcbScr, *eligScr;
    device float *fcG, *fcS;
    device float *ring, *dynaTd, *dynaD, *dynaAcc, *dynaC, *planScr;
    device float *vStop2;
    device uint *counters, *optFeature, *featureAge, *optAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption, *keyboardOrigin;
    device uint *explosionStep, *explosionElem;
    device uint *lastStopAge;
    device uint *nurParentA, *nurParentB, *nurKind, *nurAge;
    device uint *actIdx;
    device uint *dynaSlot;
    device uint *ringIdx;
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
    c.optPolicy = params + p; p += c.K * max(c.D / 2u, 1u) * c.R;
    c.optReward = params + p; p += c.K * c.R;
    c.optDuration = params + p; p += c.K * c.R;
    c.psiAux = params + p; p += c.K * c.R;
    c.vStop = params + p; p += c.K * c.R;
    c.vsAux = params + p; p += c.K * c.R;
    c.vStop2 = params + p;

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
    c.optRho = stateArena + s; s += c.K;
    c.optFit = stateArena + s; s += c.K;
    c.alphaAct = stateArena + s; s += c.K;
    c.betaAct = stateArena + s; s += c.K;
    c.hAct = stateArena + s; s += c.K;
    c.energyAct = stateArena + s; s += c.K;
    c.impStage = stateArena + s; s += c.K;
    c.creditStage = stateArena + s; s += c.K * c.D;
    c.sampleScr = stateArena + s; s += 8u;
    c.rHatC = stateArena + s; s += c.K;
    c.lRawC = stateArena + s; s += c.K;
    c.vsNextC = stateArena + s; s += c.K;
    c.vsTd = stateArena + s; s += c.K;
    c.vsDot = stateArena + s; s += c.K;
    c.vsScale = stateArena + s; s += c.K;
    c.vsAuxScale = stateArena + s; s += c.K;
    c.planScale = stateArena + s; s += 1u;
    s = (s + 3u) & ~3u;  // 16B-align nurStats/muRaw/polCoef (float4 access)
    // Interleaved candidate stats {mean, sm, util, utilSM}: one line per
    // candidate; access is sequential over n so coalescing is preserved.
    c.nurStats = stateArena + s; s += 4u * c.Rn;
    c.muRaw = stateArena + s; s += c.K * c.D;
    c.polCoef = stateArena + s; s += c.K * c.D;
    c.optUseByFeat = stateArena + s; s += c.R;
    c.lcbScr = stateArena + s; s += c.Rn;
    s += 3u;  // 16B-align eligScr (float4 per slot)
    s &= ~3u;
    c.eligScr = stateArena + s; s += 4u * c.R;
    c.fcG = stateArena + s; s += c.K;
    c.fcS = stateArena + s; s += 16u;
    // Prioritized-sweeping ring: remembered states (R slots x 64 active tuples:
    // 32 core values + 32 gated winner values), per-slot backup priorities,
    // per-thread backup deltas, and the dense per-feature backup accumulator.
    c.ring = stateArena + s; s += 64u * c.R;
    c.dynaTd = stateArena + s; s += c.R;
    c.dynaD = stateArena + s; s += 32u;
    c.dynaAcc = stateArena + s; s += c.R;
    c.dynaC = stateArena + s; s += c.K;
    c.planScr = stateArena + s; s += 4u;

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
    c.actIdx = uintArena + u; u += 2u * 64u;
    c.dynaSlot = uintArena + u; u += 32u;
    c.ringIdx = uintArena + u;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
// Geometry-gated sparse representation (Mark III law: sparsity is a stationary
// function of state; ALL raw inputs always visible):
//   core   slots 0..31   — 17 std obs, 12 prev-action, prev reward, bias, spare;
//                          unconditionally active every step.
//   gated  slots 32..255 — 32 banks x 7 slots; exactly one slot per bank is
//                          active, selected by a fixed tile of a designated
//                          core channel (tile-coding geometry, never magnitude
//                          competition). Contents (traces / constructed
//                          products) are generate-and-test replaceable.
// Active width A = 64: actIdx entries 0..31 are the core identity, 32..63 the
// gated bank winners.
constant uint OAK_CORE = 32u;
constant uint OAK_ACT = 64u;
constant uint OAK_GATE_BANK = 7u;
inline device uint *act_at(Ctx ctx, uint slot) { return ctx.actIdx + slot * OAK_ACT; }
// Live action dims: the wrapper reserves the second half of the descriptor
// (inert; zero score/credit). Only the live half is STORED for optPolicy.
inline uint oak_dl(Ctx ctx) { return max(ctx.D / 2u, 1u); }
// Which core channel gates bank b (cycle over obs+action+reward; skip bias/spare).
inline uint gate_channel(Ctx ctx, uint b) { return b % (ctx.O + ctx.D + 1u); }
// Stationary tile of a core-channel value: 7 unit-width bins over [-3.5, 3.5).
inline uint gate_tile(float z) {
    if (!isfinite(z)) return 3u;
    return uint(clamp(floor(z + 3.5f), 0.0f, 6.0f));
}
// True iff feature i is in the P-side active set (core is always in both sets;
// a gated slot is in P iff it is its bank's recorded P winner).
inline bool oak_in_prev_active(device const uint *actP, uint i) {
    return i < OAK_CORE || actP[OAK_CORE + (i - OAK_CORE) / OAK_GATE_BANK] == i;
}
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline bool usable_priority(float p) { return isfinite(p) && p >= 0.0f; }
inline bool usable_duration(Ctx ctx, float d) {
    return isfinite(d) && d > 0.0f;
}
inline uint keyboard_sketch_dim(Ctx ctx) {
    return max(1u, min(ctx.D, ctx.K));
}

inline float standardized_obs(Ctx ctx, uint o) {
    float mean = ctx.observationMean[o];
    float sm = ctx.obsSM[o];
    float var = max(sm - mean * mean, 0.0f);
    return (ctx.observation[o] - mean) * rsqrt(var + 1.0f / float(max(ctx.O, 1u)));
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
    // Mark V forensics: an unbounded beta exploded valueW through the
    // prioritized-sweep apply (alpha[i]*acc has no alpha normalizer) —
    // lane 857 of mk4-tierL, valueW[bias] -> 3.5e34. Never-diverge outranks
    // no-caps purism; this bound is capacity-derived, not a tunable.
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
// element this step. The reward-TD and planning-TD updates carry DECOUPLED
// implicit normalizers (pure NLMS form, 1 + sum alpha x^2): a large planning
// residual must not throttle the reward TD step, and there is no TD-energy
// term — that was a hidden soft cap on exactly the walking-scale TDs.
inline void fcstomp_store_learn_scales(Ctx ctx,
                                       threadgroup const float *prev,
                                       threadgroup const float *curr) {
    // prev/curr are ACTIVE-value caches (OAK_ACT wide); indices from actIdx.
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    float denomTd = 1.0f, denomPlan = 1.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) {
        float ap = ctx.alpha[actP[a]];
        if (isfinite(ap) && ap > 0.0f) denomTd += ap * (prev[a] * prev[a]);
        float ac = ctx.alpha[actC[a]];
        if (isfinite(ac) && ac > 0.0f) denomPlan += ac * (curr[a] * curr[a]);
    }
    ctx.valueScale[0] = (isfinite(denomTd) && denomTd > 0.0f) ? (1.0f / denomTd) : 0.0f;
    ctx.planScale[0] = (isfinite(denomPlan) && denomPlan > 0.0f) ? (1.0f / denomPlan) : 0.0f;
    float xNorm = 0.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) xNorm = fma(prev[a], prev[a], xNorm);
    ctx.xNormPrev[0] = xNorm;
}

inline void fcstomp_prediction_control_loop(Ctx ctx, uint tid,
                                      threadgroup float *tgPrev,
                                      threadgroup float *tgCurr,
                                      threadgroup float *tgCont) {
    // tg diet (2 tgs/core): tgG -> ctx.fcG, tgScalars -> ctx.fcS, tgTdPsi ->
    // ctx.tdPsi (device, same values), tgTerm derived as (1 - tgCont).
    device float *tgG = ctx.fcG;
    device float *tgScalars = ctx.fcS;
    device const float *dCurr = state_at(ctx, 1u - ctx.stateSlot);
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    float vPrev = 0.0f, vCurr = 0.0f;
    {
        // simd-parallel value dots: each thread covers active entries tid and
        // tid+32; simd_sum broadcasts the totals to the whole lane warp.
        float pP = 0.0f, pC = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            pP = fma(ctx.valueW[act_of(actP, a)], tgPrev[a], pP);
            pC = fma(ctx.valueW[act_of(actC, a)], tgCurr[a], pC);
        }
        vPrev = simd_sum(pP);
        vCurr = simd_sum(pC);
        if (OAK_ABLATE_GVF_PREDICTION) {
            vPrev = 0.0f;
            vCurr = 0.0f;
        }
        if (tid == 0u) {
            tgScalars[0] = vPrev;
            tgScalars[1] = vCurr;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
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
        tgScalars[4] = rhoUse;
    }
    if (tid == 0u) {
        float modelTraceNorm = 0.0f;
        float xNormRaw = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            float x = tgPrev[a];
            if (!isfinite(x)) continue;
            float xx = x * x;
            xNormRaw += xx;
            float am = ctx.alphaModel[act_of(actP, a)];
            if (isfinite(am) && am > 0.0f) modelTraceNorm = fma(am, xx, modelTraceNorm);
        }
        tgScalars[3] = (isfinite(modelTraceNorm) && modelTraceNorm > 0.0f) ? modelTraceNorm : 0.0f;
        tgScalars[2] = (isfinite(xNormRaw) && xNormRaw > 0.0f) ? xNormRaw : 0.0f;
    }
    // Option-INDEPENDENT norm terms, computed once instead of 32x per step:
    // psi GTD2 trace decomposes as T1 - 2*cont*T2 + cont^2*T3 per option, and
    // plan's bootstrap norm as M1 - 2*cont*M2 + cont^2*M3 (alphaModel family).
    {
        float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f, mA = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            float xp = tgPrev[a], xc = tgCurr[a];
            float ap = ctx.alphaPsi[ip];
            if (isfinite(ap) && ap > 0.0f) {
                t1 = fma(ap * xp, xp, t1);
                t2 = fma(ap * xp, dCurr[ip], t2);
            }
            float acps = ctx.alphaPsi[ic];
            if (isfinite(acps) && acps > 0.0f) t3 = fma(acps * xc, xc, t3);
            float am = ctx.alphaModel[ip];
            if (isfinite(am) && am > 0.0f) {
                m1 = fma(am * xp, xp, m1);
                m2 = fma(am * xp, dCurr[ip], m2);
                mA = fma(am * xp, xp, mA);
            }
            float amc = ctx.alphaModel[ic];
            if (isfinite(amc) && amc > 0.0f) m3 = fma(amc * xc, xc, m3);
        }
        tgScalars[5] = simd_sum(t1);
        tgScalars[6] = simd_sum(t2);
        tgScalars[7] = simd_sum(t3);
        ctx.fcS[8] = simd_sum(m1);
        ctx.fcS[9] = simd_sum(m2);
        ctx.fcS[10] = simd_sum(m3);
        (void)mA;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K) {
        uint feat = ctx.optFeature[tid] % ctx.R;
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            psiPrev = fma(ctx.psi[(ip) * ctx.K + tid], tgPrev[a], psiPrev);
            psiNext = fma(ctx.psi[(ic) * ctx.K + tid], tgCurr[a], psiNext);
        }
        // Option-independent GTD2 trace terms (computed once above).
        float psiXX = tgScalars[5], psiXY = tgScalars[6], psiYY = tgScalars[7];
        // RR eq. 4 stopping value: z = v(s') + (wBar_f - w_f + 1) x_f(s').
        // wBar is the HIGH-WATER MARK of valueW[f] (RR-Subtasks section 6's
        // optimism dial: remembered-good regimes stay permanently attractive),
        // plus the structural unit bonus so attainment never tracks itself
        // away. wBar >= w by construction (monotone max from a shared init).
        float wBonus = ctx.bonusW[feat] - ctx.valueW[feat];
        if (!isfinite(wBonus) || wBonus < 0.0f) wBonus = 0.0f;
        float zStop = vCurrG + (1.0f + wBonus) * dCurr[feat];
        bool rrFinite = isfinite(psiPrev) && isfinite(psiNext) && isfinite(zStop);
        if (!rrFinite) ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        // RR eq. 9: stop when the stopping value matches or beats continuing.
        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop >= psiNext) ? 1.0f : 0.0f;
        // Option 0 is the MAIN-TASK key (RR p.12 subtask 0): cumulant = reward,
        // beta = 0, never stops — every lane carries a policy that directly
        // maximizes the reward stream.
        if (tid == 0u) beta = 0.0f;
        if (OAK_ABLATE_RR_SUBTASKS) {
            rrFinite = true;
            beta = (tid == 0u) ? 0.0f : 1.0f;
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
        // Main key: advantage = main TD, DIFFERENTIALLY CENTERED (Mark III
        // law (d)). rho-bar absorbs the planning TD, so E[td] = -E[planTd] < 0
        // at steady state — uncentered, the actor persistently un-learns the
        // gait (measured: 40% positive @30k collapsing to ~17% @100k across
        // every other treatment).
        if (tid == 0u) dPsi = ctx.td[0] - psiRhoOld;
        ctx.tdPsi[tid] = dPsi;
        {
            float psiRhoNew = psiRhoOld + ctx.invR * dPsi;
            ctx.psiRho[tid] = safe_w(psiRhoOld, psiRhoNew);
        }
        float absD = isfinite(dPsi) ? fabs(dPsi) : 0.0f;
        float fastOld = ctx.tdPsiFast[tid];
        float peakOld = ctx.tdPsiPeak[tid];
        ctx.tdPsiFast[tid] = (1.0f - ctx.invR) * (isfinite(fastOld) ? fastOld : 0.0f) + ctx.invR * absD;
        ctx.tdPsiPeak[tid] = max(isfinite(peakOld) ? peakOld : 0.0f, absD);
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        {
            // ARS fitness (Mania et al. 2018): match-weighted running mean of
            // RAW reward while this option's action was the one played (exact
            // one-step ratio as the match weight). NOT rho*(r - rho-bar): the
            // planning-biased rho-bar made that uniformly negative, so higher
            // match meant lower fitness and the consensus pushed theta_0 AWAY
            // from executed behavior (the measured median eroder). Raw-reward
            // means are bias-free after the bank-centering at read.
            float fitOld = ctx.optFit[tid];
            if (!isfinite(fitOld)) fitOld = 0.0f;
            float rRaw = ctx.reward[0];
            if (!isfinite(rRaw)) rRaw = 0.0f;
            ctx.optFit[tid] = safe_w(fitOld, fitOld + ctx.invR * rho * (rRaw - fitOld));
        }
        // GTD2 implicit normalizer: alpha-weighted norm of the bootstrap
        // difference vector (x - cont*x'), nonnegative by construction.
        float implicitTrace = psiXX - 2.0f * cont * psiXY + cont * cont * psiYY;
        float denom = fma(rho, implicitTrace, 1.0f);
        ctx.headDenom[tid] = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        float modelDenom = fma(rho, tgScalars[3], 1.0f);
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
            for (uint a = 0u; a < OAK_ACT; ++a)
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
        float rHat = 0.0f, lRaw = 0.0f, vModel = 0.0f;
        if (!OAK_ABLATE_OPTION_MODELS) {
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint idx = act_of(actC, a) * ctx.K + tid;
                rHat = fma(ctx.optReward[idx], tgCurr[a], rHat);
                lRaw = fma(ctx.optDuration[idx], tgCurr[a], lRaw);
                // vStop: scalar GVF predicting v at option stop (replaces the
                // K*R*R transition tensor: v(n-hat) was its only live consumer).
                vModel = fma(ctx.vStop[idx], tgCurr[a], vModel);
            }
        }
        // Cache the curr-side model dots for the plan stage (same state, same
        // weights — plan runs before any learn write this step).
        ctx.rHatC[tid] = rHat;
        ctx.lRawC[tid] = lRaw;
        ctx.vsNextC[tid] = vModel;
        // Main-task key never stops: its tail value is v(s) itself, so its
        // model-backed advantage is the (centered) differential reward model —
        // a structurally ~0 "continue as usual" floor for GPI selection.
        if (tid == 0u) vModel = vCurr;
        bool modelFinite = isfinite(rHat) && isfinite(lRaw) && isfinite(vModel);
        float dur = decode_duration(lRaw);
        bool durationTrusted = usable_duration(ctx, dur);
        bool modelTrusted = OAK_ABLATE_SEARCH_CONTROL ? modelFinite : usable_priority(ctx.prioOpt[tid]);
        // Orthodox differential option value (Wan-Naik-Sutton / RR eq. 19 form):
        // r-hat is the CENTERED reward-to-stop model, so q = r-hat + v(stop).
        // No duration division and no model-error shift — uncertainty gates
        // trust (queryTrusted), it never biases the target.
        float q = rHat + vModel;
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
        float gaugeRate = track_rate(ctx.stepCount, ctx.R);
        float gaugeNew = ctx.valueGauge[0] + gaugeRate * (vCurr - ctx.valueGauge[0]);
        ctx.valueGauge[0] = safe_w(ctx.valueGauge[0], gaugeNew);
    }
    if (tid < ctx.K) {
        // RR eq. 10: each option's actor climbs ITS OWN subtask TD. The old
        // cross-option mean baseline coupled the actors and actively un-learned
        // any option progressing less than the bank average.
        float adv = OAK_ABLATE_OPTION_POLICIES ? 0.0f : ctx.tdPsi[tid];
        if (!isfinite(adv)) adv = 0.0f;
        ctx.actorAdv[tid] = adv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        // GPI selection in differential-value space: argmax over the trusted
        // option bank (main-task key included) of the model-backed advantage.
        // No duration division, no model-error subtraction — trust is a gate
        // (untrusted options simply don't compete), never a target shift.
        uint best = 0u;
        float bestScore = -FLT_MAX;
        float termSum = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            termSum += 1.0f - tgCont[k];
            bool trusted = isfinite(tgG[k]);
            float score = trusted ? tgG[k] : -FLT_MAX;
            if (score > bestScore) { bestScore = score; best = k; }
        }
        uint prevSel = ctx.selectedOption[0] % ctx.K;
        // The main-task key is a PRIMITIVE (one-step) option for selection —
        // committing to its beta = 0 learning continuation would freeze GPI
        // forever. Subtask options commit until their RR stopping fires.
        bool continuing = (prevSel != 0u) && tgCont[prevSel] > 0.0f;
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
        // Re-blend only when LEADERSHIP changes (or the chord is empty):
        // per-step refresh made the chord teleport between distant option
        // means whenever the one-step main key held selection — measured as
        // the 8x action-thrash separating crashed lanes (actDelta 0.82,
        // keyL1 1.0) from ignited ones (0.10, 1.9).
        bool refreshKeyboard = (!continuing && best != prevSel) || prevMag <= 0.0f;
        if (refreshKeyboard) {
            float mag = 0.0f;
            bool haveBest = bestScore > -FLT_MAX;
            for (uint k = 0u; k < ctx.K; ++k) {
                bool trusted = isfinite(tgG[k]) && haveBest;
                float lam = trusted ? exp(tgG[k] - bestScore) : 0.0f;
                if (!isfinite(lam)) lam = 0.0f;
                // Unproven young directions are evaluated by SELECTION, not
                // smeared into every blend (the measured median eroder).
                if (ctx.optAge[k] < ctx.R && k != best && k != 0u) lam = 0.0f;
                ctx.keyboard[k] = lam;
                mag += lam;
            }
            if (mag <= 0.0f) {
                // Unit-weight probe (anchor parity): at invK the probed
                // direction got ~3% execution share — fitness evaluation
                // starved and the consensus harvested nothing (900k
                // forensics: chord 97% anchor at the plateau).
                uint probe = ctx.stepCount % max(ctx.K, 1u);
                for (uint k = 0u; k < ctx.K; ++k)
                    ctx.keyboard[k] = (k == probe) ? 1.0f : 0.0f;
            }
            ctx.keyboardOrigin[0] = 0u;
        }
        // Signed one-step eq.-19 planning backup at the current state:
        // max_o q(s,o) - v(s) over the trusted bank (the main-task key's ~0
        // advantage floors it, so there is no structural negative drip). Never
        // clamped nonnegative — max(·,0) here is a value-inflation pump.
        // Model error GATES the backup magnitude (trust -> 0 as the winning
        // option's prediction error grows); it never shifts the target.
        // RAW eq.-19 max (the gated-9 0.708-ceiling form). No trust multiplier,
        // no /dur, no -modelErr — suppression of the backup target is the
        // measured ceiling-killer (iter-10). Untrusted options are excluded
        // from the max upstream (queryTrusted); stability comes from the
        // decoupled planScale contraction, the differential-planning rho-bar
        // coupling below, and the capacity-derived alpha bound.
        float planDelta = (OAK_ABLATE_PLANNING || bestScore <= -FLT_MAX) ? 0.0f : bestScore;
        {
            // Model-confidence GATE (never a target shift): scales the backup
            // by the winning option's prediction reliability. Removing this
            // measured 13 rho-bar runaways to +539 by 200k (planTd -> rho-bar
            // positive feedback); with it, mk4 had zero such deaths and a
            // growing ceiling.
            float err = ctx.prioOpt[best];
            float trust = (isfinite(err) && err >= 0.0f) ? (1.0f / (1.0f + err)) : 0.0f;
            planDelta *= trust;
        }
        // Differential planning (Wan-Naik-Sutton): the average-reward estimate
        // is updated by the PLANNING TD as well as the experience TD. Without
        // this, every planning boost to v(s_{t+1}) reads as a negative reward
        // bias one step later and rho-bar diverges downward (measured: rho-bar
        // ran to -planTd within 10k steps).
        if (!OAK_ABLATE_GVF_PREDICTION && isfinite(planDelta)) {
            float r2 = ctx.scalars->rho + ctx.invR * planDelta;
            if (isfinite(r2)) ctx.scalars->rho = r2;
        }
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
    // Publish the staged sample outputs (written by last step's sample stage,
    // which runs before the learning window): everything downstream — this
    // encode, and the window's plan/learn/dyna — pairs the transition
    // (s_{t-1} -> s_t) with a_{t-1}'s ratios and score gradients.
    if (tid < ctx.K) ctx.importance[tid] = ctx.impStage[tid];
    for (uint j = tid; j < ctx.K * ctx.D; j += tgW)
        ctx.optionActorCredit[j] = ctx.creditStage[j];
    simdgroup_barrier(mem_flags::mem_device);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    // CORE (always active, Mark III law): thread tid owns core slot tid.
    {
        uint i = tid;
        float x;
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            x = (i == 0u) ? 1.0f : 0.0f;
        } else if (i < ctx.O) {
            x = standardized_obs(ctx, i);
        } else if (i < ctx.O + ctx.D) {
            x = ctx.actionSignal[i - ctx.O];
        } else if (i == ctx.O + ctx.D) {
            x = ctx.reward[0];
        } else if (i == ctx.O + ctx.D + 1u) {
            x = 1.0f;  // bias/affine channel — always visible
        } else {
            x = 0.0f;  // spare
        }
        if (!isfinite(x)) { x = 0.0f; ctx.counters[OAK_CNT_OBS_NF] += 1u; }
        curr[i] = x;
        actC[tid] = i;
        if (ctx.featureAge[i] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
            ctx.featureAge[i] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
    // GATED region: thread tid owns bank tid (OAK_GATE_BANK slots). Exactly one
    // slot per bank is active, selected by a STATIONARY tile of the bank's
    // designated core channel (tile-coding geometry) — never by magnitude
    // competition, so the same physical state always maps to the same active
    // set and replacement damage stays regional. Statistics update densely.
    uint gateCh = gate_channel(ctx, tid);
    uint tile = gate_tile(curr[gateCh]);
    uint lo = OAK_CORE + tid * OAK_GATE_BANK, hi = lo + OAK_GATE_BANK;
    uint winner = lo + tile;
    float winX = 0.0f;
    for (uint i = lo; i < hi; ++i) {
        float x = 0.0f;
        if (!OAK_ABLATE_FEATURE_CONSTRUCTION) {
            uint pa = ctx.featureParentA[i] % ctx.R;
            uint pb = ctx.featureParentB[i] % ctx.R;
            uint kind = ctx.featureKind[i] % 4u;
            float raw;
            if (kind == 3u) {
                // trace feature: a multi-timescale obs trace channel
                raw = ctx.obsTraceMulti[ctx.featureParentA[i] % (ctx.traceLevels * ctx.O)];
            } else {
                float a = prev[pa], b = prev[pb];
                raw = (kind == 0u) ? a * b
                    : (kind == 1u) ? (a + b) * 0.7071067811865475f
                                   : a * (b - a);
            }
            uint age = ctx.featureAge[i];
            float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
            ctx.featureMean[i] += rate * (raw - ctx.featureMean[i]);
            ctx.featureSM[i] += rate * (raw * raw - ctx.featureSM[i]);
            float var = max(ctx.featureSM[i] - ctx.featureMean[i] * ctx.featureMean[i], 0.0f);
            // Stationary ridge floor (1/O pattern): an age-vanishing ridge lets
            // a variance-collapsed feature become a spike amplifier.
            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(max(ctx.O, 1u)));
            x = (raw - ctx.featureMean[i]) * rsqrt(var + ridge);
            if (!isfinite(x)) { x = 0.0f; ctx.counters[OAK_CNT_OBS_NF] += 1u; }
        }
        curr[i] = 0.0f;
        if (i == winner) winX = x;
    }
    curr[winner] = winX;
    actC[OAK_CORE + tid] = winner;
    if (ctx.featureAge[winner] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
        ctx.featureAge[winner] += 1u;
    simdgroup_barrier(mem_flags::mem_device);
    // tg caches hold the ACTIVE values of prev/curr (OAK_ACT each) per lane.
    {
        device const uint *actP = act_at(ctx, ctx.stateSlot);
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            tgPrevState[a] = prev[actP[a]];
            tgCurrState[a] = curr[actC[a]];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void oak_stage_encode(Ctx ctx, uint tid, uint tgW,
                                    threadgroup float *tgCont,
                                    threadgroup float *tgPrevState,
                                    threadgroup float *tgCurrState) {
    FeatureConstruction(ctx, tid, tgW, tgPrevState, tgCurrState);
    fcstomp_prediction_control_loop(ctx, tid, tgPrevState, tgCurrState, tgCont);
}

static inline void oak_plan_one(Ctx ctx, uint k,
                                device const float *prev,
                                device const float *curr, uint pSlot) {
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

        device const uint *actP = act_at(ctx, pSlot);
        // Curr-side dots (rHat, lRaw, vsNext) and v(curr) come from the encode
        // cache — same state, same weights (plan runs in the env window of the
        // same step; learn follows it on the same warp).
        float rHat = ctx.rHatC[k], lRaw = ctx.lRawC[k], vsNext = ctx.vsNextC[k];
        float vX = ctx.fcS[1];
        float vsPrev = 0.0f, vsDot = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = act_of(actP, a);
            uint idxP = ip * ctx.K + k;
            float xp = prev[ip];
            vsPrev = fma(ctx.vStop[idxP], xp, vsPrev);
            vsDot = fma(ctx.vsAux[idxP], xp, vsDot);
        }
        // Option-independent bootstrap-norm terms from the encode cache:
        // bootNorm = M1 - 2*cont*M2 + cont^2*M3 (alphaModel family).
        float xNormA = ctx.fcS[8];
        float bootNorm = ctx.fcS[8] - 2.0f * cont * ctx.fcS[9]
                       + cont * cont * ctx.fcS[10];
        bootNorm = max(bootNorm, 0.0f);
        if (!isfinite(bootNorm)) bootNorm = 0.0f;
        if (!isfinite(xNormA) || xNormA < 0.0f) xNormA = 0.0f;
        // CENTERED reward-to-stop cumulant (Mark III law (d): every option-model
        // head carries per-head differential centering — the psiRho pattern).
        // rewardC = r - rho_main; optRho[k] tracks the residual per-head bias,
        // so slow-stopping options have a bounded fixed point.
        // CONTINUAL learning restored (no frozen organs — Inv-7): the death
        // channel was optReward's plain ||x||^2 implicit scale, which
        // under-damps as cont -> 1. These heads now use the BOOTSTRAP-NORM
        // scale (vsScale, alpha-weighted ||x - cont*x'||^2) — the correct
        // implicit-TD contraction for bootstrapped heads; vStop uses it and
        // never appeared in a death certificate.
        float rPrev = 0.0f, lPrev = 0.0f, rHat2 = 0.0f, lRaw2 = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = act_of(actP, a);
            uint idxP = ip * ctx.K + k;
            float xp = prev[ip];
            rPrev = fma(ctx.optReward[idxP], xp, rPrev);
            lPrev = fma(ctx.optDuration[idxP], xp, lPrev);
        }
        rHat2 = rHat; lRaw2 = lRaw;
        float optRhoOld = ctx.optRho[k];
        if (!isfinite(optRhoOld)) optRhoOld = 0.0f;
        float rTd = (ctx.planScr[0] - ctx.scalars->rho) - optRhoOld + cont * rHat2 - rPrev;
        if (isfinite(rTd)) ctx.optRho[k] = optRhoOld + ctx.invR * rTd;
        float dTd = (k == 0u) ? 0.0f : (1.0f + cont * lRaw2 - lPrev);
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
        float modelErr = vsTd * vsTd + rTd * rTd;
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
static inline void oak_stage_plan_all(Ctx ctx, uint tid, uint pSlot) {
    if (tid >= ctx.K) return;
    device const float *prev = state_at(ctx, pSlot);
    device const float *curr = state_at(ctx, 1u - pSlot);
    oak_plan_one(ctx, tid, prev, curr, pSlot);
    // Per-(k,d) actor coefficient, hoisted out of the K*D*A learn elements
    // (they were 60% of learn traffic at ~6 device accesses each).
    {
        uint k = tid;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float adv = ctx.actorAdv[k];
        if (!isfinite(adv)) adv = 0.0f;
        // Actor IDBD (MetaOptimize on the actor, per option): step-size grows
        // under correlated advantage signal and collapses under noise — the
        // reconciliation of fast ignition (gradient = the engine; freezing it
        // measured 1.5% positivity) with zero late-phase diffusion (fixed-alpha
        // gradient measured exponential positivity decay once adv ~ noise).
        metaopt_update_slot(ctx.alphaAct, ctx.betaAct, ctx.hAct, ctx.energyAct,
                            k, adv, 1.0f, ctx.optAge[k], ctx.metaEta, ctx.invR);
        float actorAlpha = ctx.alphaAct[k] / float(max(ctx.D * 32u, 1u));
        float xNorm = ctx.xNormPrev[0];
        for (uint d = 0u; d < oak_dl(ctx); ++d) {
            float targetGrad = ctx.optionActorCredit[k * ctx.D + d];
            if (!isfinite(targetGrad)) targetGrad = 0.0f;
            float gradScale = rho * targetGrad;
            // Full implicit-update denominator (adv^2 restored): with the main
            // key climbing the raw main TD, advantages are no longer small and
            // the un-normalized actor overshoots (measured: positivity 40.7%
            // at 30k crashing to 16.8% by 100k, live median -0.22).
            float denom = 1.0f + actorAlpha * adv * adv * gradScale * gradScale * xNorm;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float c = actorAlpha * adv * gradScale * scale;
            ctx.polCoef[k * oak_dl(ctx) + d] = isfinite(c) ? c : 0.0f;
        }
    }
}

// Row-major learn: the lane walks the active rows once; within a row,
// thread t owns head column k = t (one consecutive float per K-wide head
// family) and the contiguous polCoef window [t*W, t*W+W) of the K*D policy
// row — every read-modify-write is a coalesced row segment instead of a
// stride-K/K*D gather. Each weight receives exactly the same single update
// as the head-major form (no accumulation-order change per weight).
static inline void oak_learn_rows(Ctx ctx, device const float *prev,
                                  device const float *curr, uint tid, uint tw,
                                  uint pSlot) {
    device const uint *actP = act_at(ctx, pSlot);
    device const uint *actC = act_at(ctx, 1u - pSlot);
    const uint A = OAK_ACT;
    uint K = ctx.K;
    uint DL = oak_dl(ctx);
    uint KDL = K * DL;
    // Per-thread head-k scalars (k = tid), loaded once.
    uint k = tid;
    float rho = ctx.importance[k];
    if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
    float cont = ctx.continuation[k];
    if (!isfinite(cont)) cont = 0.0f;
    float hD = ctx.hDotPsi[k], sc = ctx.headScale[k], td = ctx.tdPsi[k];
    float vD = ctx.vsDot[k], vsc = ctx.vsScale[k], vtd = ctx.vsTd[k];
    float psiErr = td - hD, psc = ctx.psiAuxScale[k];
    float vErr = vtd - vD, vasc = ctx.vsAuxScale[k];
    float cj[16];
    for (uint j = 0u; j < DL; ++j) cj[j] = ctx.polCoef[tid * DL + j];
    for (uint a = 0u; a < A; ++a) {
        uint i = act_of(actP, a);
        float xp = prev[i], xc = curr[i];
        float aP = ctx.alphaPsi[i], aM = ctx.alphaModel[i];
        if (!OAK_ABLATE_OPTION_POLICIES) {
            device float *prow = &ctx.optPolicy[i * KDL + tid * DL];
            for (uint j = 0u; j < DL; ++j)
                prow[j] = safe_w(prow[j], prow[j] + cj[j] * xp);
        }
        if (!OAK_ABLATE_RR_SUBTASKS && k != 0u) {
            float g = (cont > 0.0f) ? (hD * (xp - cont * xc)) : (td * xp);
            device float *w = &ctx.psi[i * K + k];
            *w = safe_w(*w, *w + aP * rho * g * sc);
            device float *h = &ctx.psiAux[i * K + k];
            *h = safe_w(*h, *h + aP * rho * psiErr * xp * psc);
        }
        if (!OAK_ABLATE_OPTION_MODELS) {
            float gv = (cont > 0.0f) ? (vD * (xp - cont * xc)) : (vtd * xp);
            device float *w = &ctx.vStop[i * K + k];
            *w = safe_w(*w, *w + aM * rho * gv * vsc);
            device float *h = &ctx.vsAux[i * K + k];
            *h = safe_w(*h, *h + aM * rho * vErr * xp * vasc);
        }
    }
    // C-side rows: only gated entries can be C-only; psi/vStop GTD2 correction.
    if (cont > 0.0f && rho > 0.0f) {
        for (uint a = OAK_CORE; a < A; ++a) {
            uint i = act_of(actC, a);
            if (oak_in_prev_active(actP, i)) continue;
            float xc = curr[i];
            float aP = ctx.alphaPsi[i], aM = ctx.alphaModel[i];
            if (!OAK_ABLATE_RR_SUBTASKS && k != 0u) {
                device float *w = &ctx.psi[i * K + k];
                *w = safe_w(*w, *w + aP * rho * (-hD * cont * xc) * sc);
            }
            if (!OAK_ABLATE_OPTION_MODELS) {
                device float *w = &ctx.vStop[i * K + k];
                *w = safe_w(*w, *w + aM * rho * (-vD * cont * xc) * vsc);
            }
        }
    }
    // value banks (feature-indexed, scalar): one writer per slot via i % tw.
    if (!OAK_ABLATE_GVF_PREDICTION) {
        for (uint e = 0u; e < 2u * A; ++e) {
            uint i = (e < A) ? act_of(actP, e) : act_of(actC, e - A);
            if (i % tw != tid) continue;
            if (e >= A && oak_in_prev_active(actP, i)) continue;
            float w = ctx.valueW[i];
            float delta = (e < A)
                ? (ctx.td[0] * prev[i] * ctx.valueScale[0]
                   + ctx.planTd[0] * curr[i] * ctx.planScale[0])
                : (ctx.planTd[0] * curr[i] * ctx.planScale[0]);
            ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta);
        }
    }
    // Eq.-19 planning apply (prioritized-sweeping accumulator, same ownership).
    if (!OAK_ABLATE_PLANNING && !OAK_ABLATE_GVF_PREDICTION) {
        for (uint i = tid; i < ctx.R; i += tw) {
            float acc = ctx.dynaAcc[i];
            if (acc == 0.0f || !isfinite(acc)) continue;
            float w = ctx.valueW[i];
            ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * acc);
            ctx.dynaAcc[i] = 0.0f;
        }
    }
}


// Row-major transposed emit: thread t owns the contiguous kd-window
// [t*W, t*W+W) of the K*D outputs (W = K*D/32 = D at K=32, a compile-time
// shape constant), and the lane walks the A active rows once — every
// optPolicy read is a coalesced row segment instead of a stride-K*D gather.
// Same fma count as the per-(k,d) grid, ~none of its scatter traffic.
inline void emit_all_options(Ctx ctx, threadgroup const float *curr, uint tid) {
    // Thread t owns option t. optPolicy stores only the LIVE half of the
    // descriptor (K x D/2 x R); the interface arrays (muRaw/optionPolicyMean)
    // keep the full K x D stride with the reserved half pinned to 0.
    uint DL = oak_dl(ctx);
    uint KDL = ctx.K * DL;
    device const uint *act = act_at(ctx, ctx.stateSlot);
    if (OAK_ABLATE_OPTION_POLICIES) {
        for (uint j = 0u; j < ctx.D; ++j) {
            ctx.optionPolicyMean[tid * ctx.D + j] = 0.0f;
            ctx.muRaw[tid * ctx.D + j] = 0.0f;
        }
        if (tid < ctx.D) ctx.policyMean[tid] = 0.0f;
        return;
    }
    float acc[16];
    for (uint j = 0u; j < DL; ++j) acc[j] = 0.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) {
        uint i = act_of(act, a);
        float x = curr[a];
        device const float *row = &ctx.optPolicy[i * KDL + tid * DL];
        for (uint j = 0u; j < DL; ++j) acc[j] = fma(row[j], x, acc[j]);
    }
    for (uint j = 0u; j < ctx.D; ++j) {
        uint kd = tid * ctx.D + j;
        // BOUNDED action means (Mark VIII): tanh tells the actor the actuator
        // saturates. Forensics: every zenith collapse and death traced to
        // unbounded mean growth past the env clamp (weights grow with zero
        // behavioral gradient until the head stack detonates). tanh bounds
        // the mean by construction and its score gradient (1 - m^2) self-
        // extinguishes weight growth at saturation — the missing pullback.
        float mu = (j < DL) ? tanh(acc[j]) : 0.0f;
        ctx.muRaw[kd] = mu;
        if (!isfinite(mu)) mu = ctx.optionPolicyMean[kd];
        ctx.optionPolicyMean[kd] = mu;
    }
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
    // Consensus anchor: the main key always carries unit weight in the chord.
    // Burn-loop lanes (perpetually-young banks -> degenerate one-option chords
    // -> thrash, the deep-negative tail) get a calm recovery path; igniter
    // lanes are barely affected (their leader dominates the blend).
    {
        float m0 = ctx.muRaw[d];
        if (isfinite(m0)) { blend += m0; wNorm += 1.0f; }
    }
    ctx.policyMean[d] = (wNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
        ? blend / wNorm : fallback;
}

// COMMIT (reactive path): the minimal, ordering-critical slice of the old
// replace stage — ring write + backup priority, the reward snapshot for the
// concurrent env window, curation decisions applied from LAST window's
// analytics (eligScr/lcb/util EMAs are one step stale — they are EMAs), and
// the step/slot advance. Everything heavy moved to the env window.
static inline void oak_replace_commit(Ctx ctx, uint tid) {
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    if (!OAK_ABLATE_PLANNING) {
        // Dyna ring: remember this step's state as sparse active tuples
        // (pre-curation values — the state learning actually saw).
        uint slot = ctx.stepCount % ctx.R;
        device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
        if (tid < 32u) {
            // 64 tuples per slot: core values at 0..31, gated winner values at
            // 32..63 (indices stored only for the gated half — core is identity).
            ctx.ring[slot * 64u + tid] = curr[actC[tid]];
            uint gi = actC[OAK_CORE + tid];
            ctx.ringIdx[slot * 32u + tid] = gi;
            ctx.ring[slot * 64u + OAK_CORE + tid] = curr[gi];
        }
        // Fresh slot enters the sweep at "surprise" priority (Moore-Atkeson):
        // the magnitude of this step's learning signal.
        if (tid == 0u) {
            float p0 = fabs(ctx.td[0]) + fabs(ctx.planTd[0]);
            ctx.dynaTd[slot] = isfinite(p0) ? p0 : 0.0f;
        }
    }

    if (tid == 0u) ctx.planScr[0] = ctx.reward[0];
    simdgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
    uint candidateStart = ctx.anchorCount;
    uint eligible = 0u, worst = candidateStart;
    bool haveWorst = false;
    float worstUtil = 0.0f, worstRed = 0.0f, worstDownstream = 0.0f;
    uint worstAge = 0u;
    float eligibleUtilSum = 0.0f, eligibleRedSum = 0.0f, eligibleDownstreamSum = 0.0f;
    for (uint i = candidateStart; i < ctx.R; ++i) {
        float4 e = *(device const float4 *)&ctx.eligScr[i * 4u];
        if (e.w > 0.0f) {
            eligible += 1u;
            float util = e.x, red = e.y, downstreamUse = e.z;
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
        // Option-timescale staleness (R*K): a subtask that has not stopped in
        // R*K steps is degenerate. Option 0 is the main-task key (beta = 0 by
        // design) and is exempt.
        uint staleLimit = ctx.R * ctx.K;
        for (uint k = 1u; k < ctx.K; ++k) {
            if (ctx.lastStopAge[k] < staleLimit) continue;
            ctx.optAge[k] = 0u;
            ctx.lastStopAge[k] = 0u;
            ctx.utilOpt[k] = float(ctx.R);
            ctx.actorAdv[k] = 0.0f;
            ctx.psiRho[k] = 0.0f;
            ctx.optRho[k] = 0.0f;
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
    // Mean-based promotion bar (the pre-authorized median fallback): measured
    // at horizon, worst-bar feature churn erodes the non-igniting majority
    // (recentP50 -0.10 -> -0.24 between 200k and 400k) even with regional
    // geometry-gated damage. Ignition no longer depends on feature churn —
    // Mark V's option-perturbation search + consensus recombination is the
    // ignition engine (it did not exist in the D1 era).
    bool doReplace = !OAK_ABLATE_GENERATE_AND_TEST
        && structurallyReplaceable && worst < ctx.R
        && bestN != 0xffffffffu
        && bestNUtil > worstUtil + worstDownstream;
    if (doReplace && worst < ctx.R) {
        uint pa = ctx.nurParentA[bestN] % ctx.R;
        uint pb = ctx.nurParentB[bestN] % ctx.R;
uint birth = ctx.featureBirthCount[worst] + 1u;
        ctx.featureBirthCount[worst] = birth;
        ctx.featureParentA[worst] = pa;
        ctx.featureParentB[worst] = pb;
        ctx.featureKind[worst] = ctx.nurKind[bestN] % 4u;
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
            ctx.nurKind[bestN] = (ctx.nurKind[bestN] + 1u) % 4u;
            ctx.nurAge[bestN] = 0u;
            ctx.nurStats[(bestN) * 4u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 1u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 2u] = 0.0f;
            ctx.nurStats[(bestN) * 4u + 3u] = 0.0f;
        }
        // Graduates learn at anchor speed: alpha resets to alpha0 = 1/O, the
        // same init as the core anchors — not 1/R, which handicapped every
        // promotion 15x against the incumbents it must beat.
        float alpha0 = 1.0f / float(max(ctx.O, 1u));
        ctx.alpha[worst] = alpha0;
        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f; ctx.gradEnergy[worst] = 0.0f;
        ctx.alphaPsi[worst] = alpha0;
        ctx.betaPsi[worst] = log(ctx.alphaPsi[worst]); ctx.hPsi[worst] = 0.0f; ctx.gradEnergyPsi[worst] = 0.0f;
        ctx.alphaModel[worst] = alpha0;
        ctx.betaModel[worst] = log(ctx.alphaModel[worst]); ctx.hModel[worst] = 0.0f; ctx.gradEnergyModel[worst] = 0.0f;
        // Rebind an option to the new feature — never option 0 (main-task key).
        uint k = 1u;
        float worstBasis = FLT_MAX;
        for (uint ok = 1u; ok < ctx.K; ++ok) {
            float use = ctx.utilOpt[ok] + ctx.prioOpt[ok];
            if (!isfinite(use)) use = FLT_MAX;
            if (ctx.optFeature[ok] == worst) { k = ok; worstBasis = -FLT_MAX; break; }
            if (use < worstBasis) { worstBasis = use; k = ok; }
        }
        ctx.optAge[k] = 0u; ctx.optFeature[k] = worst; ctx.utilOpt[k] = float(ctx.R);
        ctx.actorAdv[k] = 0.0f;
        ctx.psiRho[k] = 0.0f;
        ctx.optRho[k] = 0.0f;
        ctx.optFit[k] = 0.0f;
        ctx.lastStopAge[k] = 0u;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
        // Broadcast the decision; the heavy column work runs warp-parallel
        // below (dynaSlot is free scratch at commit time — the window's dyna
        // rewrites it later this step).
        ctx.dynaSlot[0] = 1u; ctx.dynaSlot[1] = worst;
        ctx.dynaSlot[2] = k; ctx.dynaSlot[3] = birth;
    } else {
        ctx.dynaSlot[0] = 0u;
    }
    if (ctx.dynaSlot[0] == 0u && !OAK_ABLATE_GENERATE_AND_TEST
        && (ctx.stepCount % max(ctx.K, 1u)) == 0u) {
        // Fitness-triggered direction reseed (decoupled from feature
        // promotion — coupling them measured ignition 12.4% -> 0.9%).
        float fitMean = 0.0f;
        for (uint k2 = 0u; k2 < ctx.K; ++k2) {
            float f = ctx.optFit[k2];
            fitMean += isfinite(f) ? f : 0.0f;
        }
        fitMean *= ctx.invK;
        uint kR = 0u; float fWorst = FLT_MAX;
        for (uint k2 = 1u; k2 < ctx.K; ++k2) {
            if (ctx.optAge[k2] < ctx.R) continue;
            float f = ctx.optFit[k2];
            if (!isfinite(f)) f = 0.0f;
            if (f < fWorst) { fWorst = f; kR = k2; }
        }
        if (kR != 0u && fWorst < 0.0f && fWorst < fitMean) {
            ctx.optAge[kR] = 0u; ctx.lastStopAge[kR] = 0u;
            ctx.optFit[kR] = 0.0f; ctx.psiRho[kR] = 0.0f; ctx.optRho[kR] = 0.0f;
            ctx.actorAdv[kR] = 0.0f;
            ctx.dynaSlot[0] = 2u; ctx.dynaSlot[2] = kR;
            ctx.dynaSlot[3] = ctx.stepCount;
        }
    }
    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
    }
    // ---- warp-parallel promotion apply (was ~50k serial ops on one thread
    // every ~50 steps — a measured critical-path drag once the pre-D1
    // replacement rate was restored) ----
    simdgroup_barrier(mem_flags::mem_device);
    if (ctx.dynaSlot[0] != 0u) {
        uint worst = ctx.dynaSlot[1], k = ctx.dynaSlot[2], birth = ctx.dynaSlot[3];
        float basis = 1.0f / float(max(ctx.D, 1u));
        uint DLc = oak_dl(ctx);
        uint KDL = ctx.K * DLc;
        // Phase 1: fresh learner for the rebound option k; its policy is a
        // seeded perturbation OF THE CONSENSUS theta_0 (ARS direction refresh).
        for (uint i = tid; i < ctx.R; i += 32u) {
            ctx.psi[(i) * ctx.K + k] = 0.0f;
            ctx.psiAux[(i) * ctx.K + k] = 0.0f;
            ctx.optReward[(i) * ctx.K + k] = 0.0f;
            ctx.optDuration[(i) * ctx.K + k] = 0.0f;
            ctx.vStop[(i) * ctx.K + k] = 0.0f;
            ctx.vsAux[(i) * ctx.K + k] = 0.0f;
            for (uint d = 0u; d < DLc; ++d) {
                uint h = (i * 2654435761u) ^ (d * 40503u)
                       ^ (k * 2246822519u) ^ (birth * 3266489917u)
                       ^ ctx.stepCount;
                h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
                float z = float(h & 0xffffu) / 32768.0f - 1.0f;  // [-1, 1)
                // Perturbation scale: basis/sqrt(A) so the EMITTED action
                // noise of a fresh direction matches the one-hot init scale
                // (dense +-basis across A active rows emitted ~sqrt(A)x hotter
                // — measured flooding the mk5 population with violent actions).
                float w0 = ctx.optPolicy[i * KDL + d];
                if (!isfinite(w0)) w0 = 0.0f;
                ctx.optPolicy[i * KDL + k * DLc + d] =
                    w0 + z * basis * rsqrt(float(OAK_ACT));
            }
        }
        simdgroup_barrier(mem_flags::mem_device);
        // Phase 2 (promotions only): feature `worst` is now a DIFFERENT signal — every head's
        // weight on that index is stale knowledge about the old feature;
        // leaving it tears the shared fabric (observed 1e9 blow-ups). Clear
        // the column bank-wide (one option per thread).
        if (tid < ctx.K && ctx.dynaSlot[0] == 1u) {
            uint ok = tid;
            ctx.psi[worst * ctx.K + ok] = 0.0f;
            ctx.psiAux[worst * ctx.K + ok] = 0.0f;
            ctx.optReward[worst * ctx.K + ok] = 0.0f;
            ctx.optDuration[worst * ctx.K + ok] = 0.0f;
            ctx.vStop[worst * ctx.K + ok] = 0.0f;
            ctx.vsAux[worst * ctx.K + ok] = 0.0f;
            ctx.vStop2[worst * ctx.K + ok] = 0.0f;
            for (uint d = 0u; d < DLc; ++d)
                ctx.optPolicy[worst * KDL + ok * DLc + d] = 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {
            float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
            uint fi = (ctx.dynaSlot[0] == 1u) ? worst : (ctx.optFeature[k] % ctx.R);
            ctx.psi[fi * ctx.K + k] = invSqrtR;
        }
    }
}

// WINDOW ANALYTICS (rides the env window on the idle warps; post-flip ctx):
// utility/meta/bonusW EMAs, option utility, ARS consensus recombination,
// nursery, curation precomputes (optUse/lcb/eligScr), explosion scan, stats.
// All writes are lane-local and this lane's whole window job runs on ONE
// warp, so per-lane bitwise determinism is preserved.
static inline void oak_window_analytics(Ctx ctx, uint tid) {
    const uint tgW = 32u;
    uint tPrev = ctx.stepCount - 1u;
    device const float *prev = state_at(ctx, 1u - ctx.stateSlot);
    device const float *curr = state_at(ctx, ctx.stateSlot);
    device const uint *actP = act_at(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(tPrev, ctx.R);
    // Utility / meta-learning advance on ACTIVE features only (age = data seen).
    for (uint aIdx = tid; aIdx < OAK_ACT; aIdx += 32u) {
        uint i = actP[aIdx];
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
        for (uint d = 0u; d < oak_dl(ctx); ++d)
            actLoad += fabs(ctx.optPolicy[i * ctx.K * oak_dl(ctx) + sel * oak_dl(ctx) + d]);
        float load = fabs(prev[i]) * (fabs(ctx.valueW[i]) + actLoad);
        if (!isfinite(load)) load = 0.0f;
        float g = fabs(planningSignal * prev[i])
                + fabs((psiSignal + modelSignal + keyboardSignal) * prev[i])
                + load;
        ctx.utilFeat[i] += invT * (g - ctx.utilFeat[i]);
        // RR section-6 optimism: bonusW is the high-water mark of valueW.
        {
            float hw = ctx.bonusW[i];
            if (!isfinite(hw)) hw = 0.0f;
            float w = ctx.valueW[i];
            if (isfinite(w) && w > hw) ctx.bonusW[i] = w;
        }
        uint age = ctx.featureAge[i];
        metaopt_update_slot(ctx.alpha, ctx.beta, ctx.hMeta, ctx.gradEnergy,
                            i, ctx.td[0], prev[i], age, ctx.metaEta, ctx.invR);
        uint metaSlot = tPrev % 32u;
        if (aIdx == metaSlot) {
            // Signed per-head TDs (round-robin over heads): IDBD's mechanism is
            // the sign correlation of successive gradients; a magnitude signal
            // self-correlates and ratchets beta monotonically.
            uint kSlot = tPrev % max(ctx.K, 1u);
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
    // Consensus recombination (ARS, Mania et al. 2018 — the published
    // all-seeds-positive mechanism for this env at linear capacity): the
    // main-task key theta_0 absorbs the bank's fitness-weighted perturbation
    // directions every step, on the active feature rows. Fitness is centered
    // across the bank (ARS's across-directions baseline — a POPULATION
    // baseline over options, distinct from the per-option TD baseline that
    // corrupted subtask learning). Thread ownership: (active-row, dim) pairs;
    // active rows are distinct, so theta_0 writes are conflict-free. The
    // learn stage's writes are behind the stage fence; the curation block
    // below is behind this stage's own device barrier.
    if (!OAK_ABLATE_OPTION_POLICIES && !OAK_ABLATE_GENERATE_AND_TEST) {
        // FUNCTION-SPACE form: the fitness-weighted direction is computed over
        // the option action MEANS (K·D) and applied to theta_0's active rows
        // as one NLMS rank-1 update (A·D) — same organ as the weight-space
        // recombination at ~1/20 the FLOPs. Means are the previous emit's
        // (one-step lag, EMA dynamics).
        float fitMean = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float f = ctx.optFit[k];
            fitMean += isfinite(f) ? f : 0.0f;
        }
        fitMean *= ctx.invK;
        float xNorm = ctx.xNormPrev[0];
        if (!isfinite(xNorm) || xNorm < 0.0f) xNorm = 0.0f;
        float rate = ctx.invR * ctx.invK / (1.0f + xNorm);
        for (uint d = tid; d < max(ctx.D / 2u, 1u); d += tgW) {
            float m0 = ctx.optionPolicyMean[d];
            if (!isfinite(m0)) m0 = 0.0f;
            float dir = 0.0f;
            for (uint k = 1u; k < ctx.K; ++k) {
                float f = ctx.optFit[k];
                float cf = (isfinite(f) ? f : 0.0f) - fitMean;
                float mk = ctx.optionPolicyMean[k * ctx.D + d];
                dir = fma(cf, (isfinite(mk) ? mk : 0.0f) - m0, dir);
            }
            // Contractive direction: dir/(1+|dir|) bounds the per-step
            // consensus pull (the mu <-> theta_0 loop was the one unbounded
            // multiplicative channel — measured pMax 3e38 explosions once the
            // fitness fix made consensus effective).
            dir = isfinite(dir) ? (dir / (1.0f + fabs(dir))) : 0.0f;
            // rank-1 apply over the active rows (theta_0 column d)
            uint KDL = ctx.K * oak_dl(ctx);
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint i = actP[a];
                float w0 = ctx.optPolicy[i * KDL + d];
                ctx.optPolicy[i * KDL + d] = safe_w(w0, w0 + rate * dir * prev[i]);
            }
        }
    }
    if (!OAK_ABLATE_GENERATE_AND_TEST) {
        // Nursery (candidate axis): shadow feature candidates learn their
        // standardization and a predictive utility from the stream every step,
        // in parallel, without touching behavior. Promotion is evidence-based.
        for (uint n = tid + (((tPrev >> 1u) & 1u) != 0u ? tgW : 0u); n < ctx.Rn; n += 2u * tgW) {
            uint pa = ctx.nurParentA[n] % ctx.R;
            uint pb = ctx.nurParentB[n] % ctx.R;
            uint kind = ctx.nurKind[n] % 4u;
            float raw;
            if (kind == 3u) {
                raw = ctx.obsTraceMulti[ctx.nurParentA[n] % (ctx.traceLevels * ctx.O)];
            } else {
                float a = prev[pa], b = prev[pb];
                raw = (kind == 0u) ? a * b
                    : (kind == 1u) ? (a + b) * 0.7071067811865475f
                                   : a * (b - a);
            }
            uint age = ctx.nurAge[n];
            float rate = max(1.0f / (float(age) + 1.0f), ctx.invR);
            // One aligned float4 RMW per candidate ({mean, sm, util, utilSM}).
            device float4 *st4 = (device float4 *)&ctx.nurStats[(n) * 4u];
            float4 st = *st4;
            st.x += rate * (raw - st.x);
            st.y += rate * (raw * raw - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);
            // Stationary ridge floor (matches FeatureConstruction).
            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(max(ctx.O, 1u)));
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
        if (tid == 0u) {
            // K-scatter of option use (serial: options may share a feature).
            for (uint i = 0u; i < ctx.R; ++i) ctx.optUseByFeat[i] = 0.0f;
            for (uint k = 0u; k < ctx.K; ++k) {
                float use = ctx.utilOpt[k]
                    + (OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(ctx.keyboard[k]));
                if (isfinite(use)) ctx.optUseByFeat[ctx.optFeature[k] % ctx.R] += use;
            }
        }
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
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (!OAK_ABLATE_GENERATE_AND_TEST) {
        // Per-slot eligibility tuple {util, red, downstreamUse, eligible}: the
        // serial pass becomes a constant-size compare/accumulate (no growth —
        // this loop body was the residual SPS settling transient).
        device const float *currS = curr;
        for (uint i = ctx.anchorCount + tid + (((tPrev >> 1u) & 1u) != 0u ? tgW : 0u);
             i < ctx.R; i += 2u * tgW) {
            device float4 *e4 = (device float4 *)&ctx.eligScr[i * 4u];
            if (ctx.featureAge[i] < ctx.R) { *e4 = float4(0.0f, 0.0f, 0.0f, 0.0f); continue; }
            uint pa = ctx.featureParentA[i] % ctx.R;
            float util = ctx.utilFeat[i];
            if (!isfinite(util)) util = 0.0f;
            float red = fabs(currS[i] * currS[pa]);
            if (!isfinite(red)) red = 0.0f;
            float downstreamUse = ctx.optUseByFeat[i] + fabs(ctx.valueW[i]);
            if (!isfinite(downstreamUse)) downstreamUse = 0.0f;
            *e4 = float4(util, red, downstreamUse, 1.0f);
        }
    }
    simdgroup_barrier(mem_flags::mem_device);
    {
        // Rotating-stripe explosion scan: covers the whole param arena every 32
        // steps, so death certificates are dated in-kernel (within 32 steps)
        // instead of at heartbeat cadence. valueW is the params base pointer.
        uint total = ctx.R + 7u * ctx.K * ctx.R + ctx.K * oak_dl(ctx) * ctx.R;
        uint stripe = tPrev % ctx.R;
        uint badElem = 0xffffffffu;
        for (uint e = stripe + ctx.R * tid; e < total; e += ctx.R * tgW) {
            float v = ctx.valueW[e];
            if (!isfinite(v) || fabs(v) > 1e16f) { badElem = e; break; }
        }
        if (badElem != 0xffffffffu && ctx.explosionStep[0] == 0xffffffffu) {
            ctx.explosionStep[0] = ctx.stepCount;
            ctx.explosionElem[0] = badElem;
        }
    }
    
    simdgroup_barrier(mem_flags::mem_device);
    // Diagnostics cadence: this block only feeds heartbeat snapshots; the
    // log()-heavy entropy/ESS section runs every 4th step.
    if (tid == 0u && ((tPrev >> 1u) & 1u) == 0u) {
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
    ctx.stats->dynaBackupMean = ctx.planScr[1];

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
    
    }
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
static_assert(sizeof(OakLaneSnapshot) == 280, "OakLaneSnapshot layout drifted from the Swift mirror");

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
        for (uint k = 0u; k < ctx.K; ++k) {
            ctx.importance[k] = 1.0f;
            ctx.impStage[k] = 1.0f;
        }
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
        for (uint o = 0u; o < ctx.K; ++o) ctx.impStage[o] = 0.0f;
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
                ctx.creditStage[off + d] = (sample - om) * (1.0f - om * om);
                ctx.creditStage[off + A + d] = 0.0f;
                float diffO = sample - om;
                ctx.impStage[o] += diffB * diffB - diffO * diffO;
            }
        }
        // Exact one-step Gaussian ratio pi_k(a|s)/b(a|s), shared sigma; bounded
        // at K (bank capacity — structural). Mark V forensics: unbounded ratios
        // collapsed actor-support entropy to 0 and exploded psiAux (lane 950 of
        // mk4-tierL). Overflow/non-finite -> 0 (~zero-probability option),
        // never the maximum (the old code's overflow->K was backwards).
        float inv2s2 = 1.0f / (2.0f * sd * sd);
        for (uint o = 0u; o < ctx.K; ++o) {
            float ratio = exp(ctx.impStage[o] * inv2s2);
            ctx.impStage[o] = isfinite(ratio) ? min(ratio, float(ctx.K)) : 0.0f;
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
                                             uint gid, uint tid) {
    device float *scratch = ctx.sampleScr;
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
        scratch[d] = sample;
    }
    simdgroup_barrier(mem_flags::mem_device);
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
            float sample = scratch[d];
            float om = oak_safe(ctx.optionPolicyMean[off + d]);
            // d log pi / d mu_raw = (a - m) * (1 - m^2) / sigma^2
            ctx.creditStage[off + d] = (sample - om) * (1.0f - om * om);
            ctx.creditStage[off + A + d] = 0.0f;
            float diffB = sample - oak_safe(ctx.policyMean[d]);
            float diffO = sample - om;
            acc += diffB * diffB - diffO * diffO;
        }
        // Exact one-step Gaussian ratio pi_k(a|s)/b(a|s), shared sigma; bounded
        // at K (bank capacity — structural; see the scalar-path note). Overflow
        // or non-finite -> 0, never the maximum.
        float inv2s2 = 1.0f / (2.0f * sd * sd);
        float ratio = exp(acc * inv2s2);
        ctx.impStage[tid] = isfinite(ratio) ? min(ratio, float(ctx.K)) : 0.0f;
    }
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

// Fence between agent stages of the SAME lane. All agent-stage data flow is
// lane-local (fleet lane-disjointness invariant) and one lane = one simdgroup,
// so a simdgroup-scope device fence suffices — except under WIDE2X, where the
// learn stage spans two simdgroups per lane. The env stage still crosses
// simdgroups (packed on simdgroup 0) and keeps full threadgroup barriers.
static inline void oak_lane_fence() {
    simdgroup_barrier(mem_flags::mem_device);
}

// ---- Wall-free eq.-19 planning with PRIORITIZED SWEEPING (Plan Step 9;
// Moore-Atkeson 1993). Runs in the env barrier interval on the simdgroups
// that idle while one warp steps the physics chain. Each thread scans its
// partition of the remembered-state ring, backs up the highest-priority
// slot with the full-bank eq.-19 target max_o[r-hat + v-stop] (main-task
// key floors the max at ~v(x), so no negative drip), and accumulates the
// resulting per-feature deltas into dynaAcc, which the NEXT learn stage
// applies to the behavior value function. Twin-ensemble disagreement gates
// trust (never shifts the target); each backup is contraction-normalized
// and the step's total planning mass is one experience-step equivalent.
// Region contract (no intra-interval sync): env writes {lanes[],
// observation[], reward[0], recordBuf}; this code reads params/state/ring
// (stable since learn/replace) and writes only {dynaTd priorities, dynaD /
// dynaSlot scratch, dynaAcc (partition-owned), dynaC, copy-2 vStop2 rows}.
static inline void oak_dyna_backups(Ctx ctx, uint tid) {
    if (OAK_ABLATE_PLANNING || tid >= 32u) return;
    uint M = ctx.R;
    // NOTE: replace already advanced stepCount, so `written` counts committed
    // ring slots exactly.
    uint written = min(ctx.stepCount, M);
    uint part = M / 32u;
    uint lo = tid * part;
    // 1) Prioritized selection within this thread's partition.
    uint bestSlot = 0xffffffffu;
    float bestPrio = -1.0f;
    for (uint s = lo; s < lo + part; ++s) {
        if (s >= written) continue;
        float p = ctx.dynaTd[s];
        if (!isfinite(p)) p = 0.0f;
        if (p > bestPrio) { bestPrio = p; bestSlot = s; }
    }
    // 2) Full-bank backup at the selected remembered state.
    float d = 0.0f;
    if (bestSlot != 0xffffffffu) {
        device const uint *ri = ctx.ringIdx + bestSlot * 32u;
        device const float *rv = ctx.ring + bestSlot * 64u;
        float vX = 0.0f, xmNorm = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint i = (a < OAK_CORE) ? a : ri[a - OAK_CORE];
            float xv = rv[a];
            vX = fma(ctx.valueW[i], xv, vX);
            float ai = ctx.alpha[i];
            if (isfinite(ai) && ai > 0.0f) xmNorm = fma(ai * xv, xv, xmNorm);
        }
        // Budgeted backup (Mark V — the full-bank sweep was ~200K fma/lane-step
        // and dominated the step, SPS 489K->156K): evaluate 4 structurally
        // named candidates — the main key, the executing option, the keyboard
        // argmax, and the rotating probe. Selection in encode still scores the
        // full bank every step.
        uint kbBest = 0u;
        {
            float kbw = -FLT_MAX;
            for (uint k = 0u; k < ctx.K; ++k) {
                float w = ctx.keyboard[k];
                if (isfinite(w) && w > kbw) { kbw = w; kbBest = k; }
            }
        }
        const uint cand[4] = { 0u, ctx.selectedOption[0] % ctx.K, kbBest,
                               (ctx.stepCount + tid) % ctx.K };
        float bq = -FLT_MAX, bdis = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            uint k = cand[c];
            float r1 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint i = (a < OAK_CORE) ? a : ri[a - OAK_CORE];
                float xv = rv[a];
                uint idx = i * ctx.K + k;
                r1 = fma(ctx.optReward[idx], xv, r1);
                s1 = fma(ctx.vStop[idx], xv, s1);
                s2 = fma(ctx.vStop2[idx], xv, s2);
            }
            // Main-task key never stops: its tail is v(x) itself.
            float tail = (k == 0u) ? vX : 0.5f * (s1 + s2);
            float q = r1 + tail;
            float dis = (k == 0u) ? 0.0f : fabs(s1 - s2);
            if (isfinite(q) && q > bq) { bq = q; bdis = isfinite(dis) ? dis : 1e6f; }
        }
        if (bq > -FLT_MAX && isfinite(vX)) {
            // Disagreement gates TRUST, never the target (measured law: the
            // min-pessimism target compounded to Tier-L IQM -0.341).
            float trust = 1.0f / (1.0f + bdis);
            d = (bq - vX) * trust;
            // Per-backup contractive normalization (the shipped never-diverge
            // sponge form: implicit norm + delta energy + disagreement).
            float den = 1.0f + xmNorm + ctx.invR * d * d + bdis;
            d = (isfinite(d) && den > 0.0f) ? (d / den) : 0.0f;
            // Mass normalization: 32 backups/step total one experience-step
            // equivalent — planning must never out-mass the real TD stream.
            d *= ctx.invK;
            // Priority update: remaining surprise at this slot.
            ctx.dynaTd[bestSlot] = fabs(d) * float(ctx.K);
        }
    }
    ctx.dynaD[tid] = d;
    ctx.dynaSlot[tid] = bestSlot;
    simdgroup_barrier(mem_flags::mem_device);
    // 3) Conflict-free accumulation into dynaAcc: thread a owns core feature a
    // and ring tuple position a of the gated half (a feature can appear only
    // at its own position: core at a, a bank's winner at OAK_CORE + bank).
    {
        ctx.dynaAcc[tid] = 0.0f;
        for (uint i = OAK_CORE + tid * OAK_GATE_BANK;
             i < OAK_CORE + (tid + 1u) * OAK_GATE_BANK; ++i)
            ctx.dynaAcc[i] = 0.0f;
        simdgroup_barrier(mem_flags::mem_device);
        float coreAcc = 0.0f;
        float dSum = 0.0f;
        for (uint t = 0u; t < 32u; ++t) {
            float dt = ctx.dynaD[t];
            dSum += fabs(dt);
            uint slot = ctx.dynaSlot[t];
            if (dt == 0.0f || slot == 0xffffffffu) continue;
            coreAcc = fma(dt, ctx.ring[slot * 64u + tid], coreAcc);
            uint gi = ctx.ringIdx[slot * 32u + tid];
            ctx.dynaAcc[gi] += dt * ctx.ring[slot * 64u + OAK_CORE + tid];
        }
        ctx.dynaAcc[tid] += coreAcc;
        if (tid == 0u) {
            // Search-control diagnostic: EMA of applied backup mass.
            float m = ctx.planScr[1];
            if (!isfinite(m)) m = 0.0f;
            ctx.planScr[1] = m + ctx.invR * (dSum - m);
        }
    }
    // 4) Copy-2 model update on the live transition (s_{t-1} -> s_t), keeping
    // the twin ensemble decorrelated through nonstationarity. Thread tid's
    // option column coefficient, then ROW-ownership apply (2 rows per thread
    // at OAK_ACT = 64).
    {
        uint k = tid;
        device const float *prevS = state_at(ctx, 1u - ctx.stateSlot);
        device const float *currS = state_at(ctx, ctx.stateSlot);
        device const uint *actP = act_at(ctx, 1u - ctx.stateSlot);
        device const uint *actC = act_at(ctx, ctx.stateSlot);
        float cont = ctx.continuation[k];
        if (!isfinite(cont)) cont = 0.0f;
        float imp = ctx.importance[k];
        if (!isfinite(imp) || imp < 0.0f) imp = 0.0f;
        float vsPrev = 0.0f, vsNext = 0.0f, vX = 0.0f, xNormP = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            float xp = prevS[ip], xc = currS[ic];
            vsPrev = fma(ctx.vStop2[ip * ctx.K + k], xp, vsPrev);
            vsNext = fma(ctx.vStop2[ic * ctx.K + k], xc, vsNext);
            vX = fma(ctx.valueW[ic], xc, vX);
            float am = ctx.alphaModel[ip];
            if (isfinite(am) && am > 0.0f) xNormP = fma(am * xp, xp, xNormP);
        }
        float vsTd = cont * vsNext + (1.0f - cont) * vX - vsPrev;
        if (!isfinite(vsTd)) vsTd = 0.0f;
        float scS = fma(imp, xNormP, 1.0f);
        scS = (isfinite(scS) && scS > 0.0f) ? (1.0f / scS) : 0.0f;
        ctx.dynaC[k] = imp * vsTd * scS;
        simdgroup_barrier(mem_flags::mem_device);
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            uint i = act_of(actP, a);
            float aM = ctx.alphaModel[i] * prevS[i];
            uint row = i * ctx.K;
            for (uint kk = 0u; kk < ctx.K; ++kk) {
                float w = ctx.vStop2[row + kk];
                ctx.vStop2[row + kk] = safe_w(w, w + aM * ctx.dynaC[kk]);
            }
        }
    }
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
                         threadgroup float *tgCont [[threadgroup(0)]],
                         threadgroup float *tgPrevState [[threadgroup(1)]],
                         threadgroup float *tgCurrState [[threadgroup(2)]],
                         threadgroup float *tgPhys [[threadgroup(3)]],
                         uint tgIdx [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint slane [[thread_index_in_simdgroup]],
                         uint sgrp [[simdgroup_index_in_threadgroup]]) {
    uint laneBase = tgIdx * OAK_MEGA_G;
    if (laneBase >= ecfg.laneCount) return;
    uint lanesThisTg = min(OAK_MEGA_G, ecfg.laneCount - laneBase);
    uint laneSg = sgrp;
    bool wideActive = laneSg < lanesThisTg;
    uint wideLane = laneBase + laneSg;
    bool packedActive = tid < lanesThisTg;
    uint packedLane = laneBase + tid;
    uint off = laneSg * OAK_ACT;
    threadgroup float *lCont = tgCont + off,
                      *lPrev = tgPrevState + off, *lCurr = tgCurrState + off;
    for (uint it = 0u; it < stepsThisDispatch; ++it) {
        // ---- REACTIVE PATH (Alberta Plan: reactive action prioritized) ----
        if (ecfg.policyMode == 0u) {
            if (wideActive && (OAK_MEGA_STAGES & 1u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_encode(ctx, slane, 32u, lCont, lPrev, lCurr);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 8u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_replace_commit(ctx, slane);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 16u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                device const float *curr = state_at(ctx, ctx.stateSlot);
                device const uint *act = act_at(ctx, ctx.stateSlot);
                for (uint a = slane; a < OAK_ACT; a += 32u) lCurr[a] = curr[act[a]];
                simdgroup_barrier(mem_flags::mem_threadgroup);
                emit_all_options(ctx, lCurr, slane);
                simdgroup_barrier(mem_flags::mem_device);
                emit_blend_grid(ctx, slane);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 32u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane);
                oak_stage_sample_act_wide(ctx, lanes, laneSeeds, wideLane, slane);
            }
        } else if (packedActive && (OAK_MEGA_STAGES & 32u) != 0u) {
            oak_stage_sample_act(params, stateArena, uintArena, scalars, stats, cfg,
                                 lanes, laneSeeds, ecfg, packedLane);
        }
        threadgroup_barrier(mem_flags::mem_device);
        // ---- ENV WINDOW: warp 0 steps the physics for all lanes; the other
        // warps run one whole lane's learning window each (plan -> learn ->
        // prioritized-sweeping backups -> curation analytics) — the entire
        // job for a lane stays on ONE warp, so per-lane bitwise determinism
        // is preserved. With G lanes and G-1 free warps, one rotating lane
        // per step skips its window (a deterministic ~1/G learning-rate
        // haircut, the round-robin precedent of the metaSlot meta-updates).
        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (packedActive && sgrp == 0u)
                oak_stage_fleet_env(params, stateArena, uintArena, scalars, stats, cfg,
                                    lanes, laneSeeds, ecfg, recordSlot, recordBuf, packedLane);
        }
        if (ecfg.policyMode == 0u && sgrp > 0u && lanesThisTg > 0u) {
            uint idx = sgrp - 1u;
            if (lanesThisTg > OAK_MEGA_G - 1u) {
                // stepCount is post-commit and uniform across lanes of the TG.
                uint T = scalars[laneBase].stepCount;
                uint skip = T % lanesThisTg;
                if (idx >= skip) idx += 1u;
            }
            if (idx < lanesThisTg) {
                uint wLane = laneBase + idx;
                Ctx pctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wLane);
                uint pSlot = 1u - pctx.stateSlot;
                if ((OAK_MEGA_STAGES & 2u) != 0u)
                    oak_stage_plan_all(pctx, slane, pSlot);
                simdgroup_barrier(mem_flags::mem_device);
                if ((OAK_MEGA_STAGES & 4u) != 0u) {
                    device const float *prevS = state_at(pctx, pSlot);
                    device const float *currS = state_at(pctx, 1u - pSlot);
                    oak_learn_rows(pctx, prevS, currS, slane, 32u, pSlot);
                }
                simdgroup_barrier(mem_flags::mem_device);
                if ((OAK_MEGA_STAGES & 2u) != 0u)
                    oak_dyna_backups(pctx, slane);
                simdgroup_barrier(mem_flags::mem_device);
                if ((OAK_MEGA_STAGES & 8u) != 0u)
                    oak_window_analytics(pctx, slane);
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
