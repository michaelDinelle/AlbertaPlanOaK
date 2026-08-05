#include <metal_stdlib>
#include <float.h>
#include "OakPhysics.metal"
#include "Gridworld.metal"
#include "ClassicControl.metal"
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
constant bool OAK_ABLATE_KEYBOARD_TO_POLICY [[function_constant(8)]];
constant bool OAK_ABLATE_SEARCH_CONTROL [[function_constant(9)]];
constant bool OAK_ABLATE_SEARCH_TO_GENERATE_TEST [[function_constant(10)]];
constant bool OAK_ABLATE_GENERATE_AND_TEST [[function_constant(11)]];

// Phase-7 representation-geometry diagnostic organs (representation-generality
// design §5.3 step 4 / §7). Each gates ONLY the !legacy (non-cheetah) path, so
// cheetah stays bitwise-identical under any combination. Indices 26/27 are
// the free slots: 12-25/28-31 are OAK_MEGA_G, OAK_SPEC_*, the stage mask,
// OAK_SPEC_NURSERY, and the probes. (Index 29 is free again: the Phase-7 §3
// generalized-G&T organ was deleted outright — the KeyCorridor population
// bisect convicted the enumerated pair sweep and the two extra construct
// kinds of the feature-explosion divergence. Greenfield removal, no flag.)
constant bool OAK_ABLATE_DERIVED_CORE [[function_constant(26)]];
constant bool OAK_ABLATE_JOINT_GEOMETRY [[function_constant(27)]];

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

constant bool OAK_PROBE_FIXEDIDX_FC [[function_constant(30)]];
constant bool OAK_PROBE_FIXEDIDX = is_function_constant_defined(OAK_PROBE_FIXEDIDX_FC)
    ? OAK_PROBE_FIXEDIDX_FC : false;
inline uint act_of(device const uint *act, uint a) {
    return OAK_PROBE_FIXEDIDX ? a : act[a];
}

constant bool OAK_PROBE_FTZ_FC [[function_constant(31)]];
constant bool OAK_PROBE_FTZ = is_function_constant_defined(OAK_PROBE_FTZ_FC)
    ? OAK_PROBE_FTZ_FC : false;
inline float ftz_probe(float w) {
    return (OAK_PROBE_FTZ && fabs(w) < 1.1754944e-38f) ? 0.0f : w;
}

constant uint OAK_MEGA_STAGE_MASK [[function_constant(25)]];
constant uint OAK_MEGA_STAGES = is_function_constant_defined(OAK_MEGA_STAGE_MASK)
    ? OAK_MEGA_STAGE_MASK : 0xffffffffu;

struct KernelConfig {
    uint obsDim, rank, optionCount, actionParamCount;
    uint obsTraceLevels;
    uint linearAnchorCount;
    float invRank, invOptionCount;
    float metaEta;
    uint agentCount, paramStride, stateStride, uintStride;
    uint nurseryCount;
    float sampleSigma;
    float sampleInv2S2;
    float actorAlphaDenom;
};

constant uint OAK_PLANSCR_INVSQRTR = 2u;

struct KernelScalars {
    uint stepCount;
    uint stateSlot;
    float rho;
};

struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    float terminationRate;
    float importanceEss;
    float actorUpdateNormMean;
    float actorSupportEntropy;
    float modelPriorityMax;
    float selectedOptionAge, selectedOptionPriority;
    float keyboardL1, actorAdvantageL1;
    float selectedOptionIndex;
    float selectedContinuation, selectedKeyboardWeight;
    float mainTd, planTdValue;
    float keyboardEffectiveL1;
    float searchControlUseMean, generateTestEligibleCount;
};

inline float safe_w_raw(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }
#define safe_w(o, n) ftz_probe(safe_w_raw((o), (n)))

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct Ctx {
    uint O, R, K, D, traceLevels, anchorCount, Rn;
    // Stream shape (natural obs width / natural action count), from
    // OakFleetLane via make_ctx; drives the §1-§2 representation geometry.
    uint naturalO, naturalA;
    uint stepCount, stateSlot;
    float rho;
    float invR, invK, metaEta;
    float sampleSigma, sampleInv2S2, actorAlphaDenom;
    device float *valueW, *psi, *optPolicy, *optReward, *optDuration;
    device float *psiAux, *vStop, *vsAux;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *policyMeanRaw, *actionSignal, *optionActorCredit, *optionPolicyMean;
    device float *importance;
    device float *reward, *td, *planTd, *tdPsi;
    device float *utilFeat, *utilOpt, *bonusW, *featureMean, *featureSM;
    device float *alpha, *beta, *hMeta;
    device float *gradEnergy;
    device float *alphaPsi, *betaPsi, *hPsi, *gradEnergyPsi;
    device float *alphaModel, *betaModel, *hModel, *gradEnergyModel;
    device float *actorAdv;
    device float *optRewardTd, *optDurationTd;
    device float *obsTraceMulti, *prioOpt, *continuation;
    device float *keyboard;
    device float *valueGauge;
    device float *headScale, *valueScale, *xNormPrev;
    device float *hDotPsi, *psiAuxScale, *psiRho, *optRho, *optFit;
    device float *alphaAct, *betaAct, *hAct, *energyAct;
    device float *impStage, *creditStage, *sampleScr;
    device float *rHatC, *lRawC, *vsNextC;
    device float *twinDis;
    device float *vsTd, *vsDot, *vsScale, *vsAuxScale;
    device float *planScale;
    device float *nurStats;
    device float *muRaw, *polCoef, *optUseByFeat, *lcbScr, *eligScr;
    device float *fcG, *fcS;
    device float *ring, *dynaTd, *dynaD, *planU, *dynaC, *planScr;
    device float *ringReward;
    device float *vStop2;
    device uint *counters, *optFeature, *featureAge, *optAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption;
    device uint *explosionStep, *explosionElem;
    device uint *lastStopAge;
    device uint *nurParentA, *nurParentB, *nurKind, *nurAge;
    device uint *actIdx;
    device uint *dynaSlot;
    device uint *ringIdx;
    device uint *planTree;
    device uint *parked;
    device KernelScalars *scalars;
    device KernelStats *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg, uint agentID,
                    uint naturalO, uint naturalA) {
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
    // Shape default: a caller without lane shape must land on legacy geometry
    // (O, D/2) — never (0,0), which would collapse the §1.3 source cycle.
    c.naturalO = (naturalO != 0u) ? naturalO : c.O;
    c.naturalA = (naturalA != 0u) ? naturalA : max(c.D / 2u, 1u);
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho;
    c.invR = is_function_constant_defined(OAK_SPEC_INV_RANK) ? OAK_SPEC_INV_RANK : cfg.invRank;
    c.invK = is_function_constant_defined(OAK_SPEC_INV_OPTION_COUNT)
        ? OAK_SPEC_INV_OPTION_COUNT : cfg.invOptionCount;
    c.metaEta = is_function_constant_defined(OAK_SPEC_META_ETA) ? OAK_SPEC_META_ETA : cfg.metaEta;
    c.sampleSigma = cfg.sampleSigma;
    c.sampleInv2S2 = cfg.sampleInv2S2;
    c.actorAlphaDenom = cfg.actorAlphaDenom;
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
    c.policyMeanRaw = stateArena + s; s += c.D;
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

    s += c.K;  // headDenom (reserved)
    c.optDurationTd = stateArena + s; s += c.K;  // revived from reserved (durationTd)
    s += c.K;  // durationHat (reserved)
    c.obsTraceMulti = stateArena + s; s += c.traceLevels * c.O;
    c.prioOpt = stateArena + s; s += c.K;
    c.continuation = stateArena + s; s += c.K;
    c.keyboard = stateArena + s; s += c.K;
    s += c.K;
    c.valueGauge = stateArena + s; s += 1u;
    s += c.K;
    s += c.K;
    c.headScale = stateArena + s; s += c.K;
    s += c.K;
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
    c.twinDis = stateArena + s; s += c.K;
    c.vsTd = stateArena + s; s += c.K;
    c.vsDot = stateArena + s; s += c.K;
    c.vsScale = stateArena + s; s += c.K;
    c.vsAuxScale = stateArena + s; s += c.K;
    c.planScale = stateArena + s; s += 1u;
    s = (s + 3u) & ~3u;

    c.nurStats = stateArena + s; s += 4u * c.Rn;
    c.muRaw = stateArena + s; s += c.K * c.D;
    c.polCoef = stateArena + s; s += c.K * c.D;
    c.optUseByFeat = stateArena + s; s += c.R;
    c.lcbScr = stateArena + s; s += c.Rn;
    s += 3u;
    s &= ~3u;
    c.eligScr = stateArena + s; s += 4u * c.R;
    c.fcG = stateArena + s; s += c.K;
    c.fcS = stateArena + s; s += 16u;

    c.ring = stateArena + s; s += 64u * c.R;
    c.dynaTd = stateArena + s; s += c.R;
    c.dynaD = stateArena + s; s += 32u;
    // planU reclaims the dynaAcc arena (the dynaAcc->valueW poison was deleted
    // in Phase 0, so the accumulator is dead). Planner-owned value table over
    // the ring, sealed from the learner (HVH).
    c.planU = stateArena + s; s += c.R;
    c.dynaC = stateArena + s; s += c.K;
    c.planScr = stateArena + s; s += 4u;
    // ringReward: the reward received transitioning OUT of each ring slot,
    // for the trajectory relay backup U(x_i) <- R_{i+1} - rho-bar + U(x_{i+1}).
    c.ringReward = stateArena + s; s += c.R;

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
    u += 1u;
    c.explosionStep = uintArena + u; u += 1u;
    c.explosionElem = uintArena + u; u += 1u;
    c.lastStopAge = uintArena + u; u += c.K;
    c.nurParentA = uintArena + u; u += c.Rn;
    c.nurParentB = uintArena + u; u += c.Rn;
    c.nurKind = uintArena + u; u += c.Rn;
    c.nurAge = uintArena + u; u += c.Rn;
    c.actIdx = uintArena + u; u += 2u * 64u;
    c.dynaSlot = uintArena + u; u += 32u;
    c.ringIdx = uintArena + u; u += 32u * c.R;
    // planTree (Phase 9.1 priority index): binary tournament tree over the
    // ring's dynaTd priorities, one aligned subtree per sweep partition;
    // internal nodes 8..M-1 hold winner slots (leaves are dynaTd itself).
    // Appended last so every legacy uint offset is unchanged.
    c.planTree = uintArena + u; u += c.R;
    // parked (Phase 9.3 K-parking): per-option hibernation flags (0 = live).
    // Appended after planTree so every earlier uint offset is unchanged.
    c.parked = uintArena + u; u += c.K;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

constant uint OAK_CORE = 32u;
constant uint OAK_ACT = 64u;
constant uint OAK_GATE_BANK = 7u;
inline device uint *act_at(Ctx ctx, uint slot) { return ctx.actIdx + slot * OAK_ACT; }

inline uint oak_dl(Ctx ctx) { return max(ctx.D / 2u, 1u); }

inline uint gate_channel(Ctx ctx, uint b) { return b % (ctx.O + ctx.D + 1u); }

inline uint gate_tile(float z) {
    if (!isfinite(z)) return 3u;
    return uint(clamp(floor(z + 3.5f), 0.0f, 6.0f));
}

// §1.2 legacy predicate (the bitwise firewall): the (17, 6) stream shape keeps
// today's exact geometry; everything in §1.3/§2 lives behind !legacy. Deliber-
// ately shape-based, not identity-based (Inv-1).
inline bool oak_legacy_geometry(Ctx ctx) {
    return (ctx.naturalO == ctx.O) && (2u * ctx.naturalA == ctx.D);
}

// §1.3 informative-source cycle: S = O_n + A_n + 1 gate sources per occurrence
// round; bank b -> (src = b % S, occ = b / S). src selects a core channel
// among obs (O_n) -> action (A_n) -> reward (1); occ picks the re-coverage
// rule (0: single tile; >=1 even b: zoom; >=1 odd b: joint bank).
inline uint gate_source_channel(Ctx ctx, uint src) {
    return (src < ctx.naturalO) ? src
         : (src < ctx.naturalO + ctx.naturalA) ? ctx.O + (src - ctx.naturalO)
                                              : ctx.O + ctx.D;
}

// Zoom tile (occ >= 1, even b): bins of width 2^-z_l sigma over
// +/-3.5*2^-z_l, edges saturate (extreme detector); z_l = min(occ, 2) since
// finer than T/4-wide bins adds no tile capacity. z_l = 0 is gate_tile.
inline uint gate_tile_zoom(float z, uint zl) {
    if (!isfinite(z)) return 3u;
    float zoom = float(1u << min(zl, 2u));
    return uint(clamp(floor(z * zoom + 3.5f), 0.0f, 6.0f));
}

// Joint cell (occ >= 1, odd b): balanced 3x2 tiling of two channels plus
// overflow cell 6 — the maximal balanced 2-D tiling of one T=7 bank (T prime).
// occ half-bin staggers the grid; cell 6 mirrors gate_tile's clamp/NF
// conventions. Used by the §1.3 gate.
inline uint joint_cell(float va, float vb, uint occ) {
    if (!isfinite(va) || !isfinite(vb) || fabs(va) >= 3.5f || fabs(vb) >= 3.5f) return 6u;
    va += ((occ & 1u) == 0u) ? (7.0f / 6.0f) : 0.0f;
    uint ta = uint(clamp(floor((va + 3.5f) * (3.0f / 7.0f)), 0.0f, 2.0f));
    uint tb = uint(clamp(floor((vb + 3.5f) * (2.0f / 7.0f)), 0.0f, 1.0f));
    return ta * 2u + tb;
}

inline bool oak_in_prev_active(device const uint *actP, uint i) {
    return i < OAK_CORE || actP[OAK_CORE + (i - OAK_CORE) / OAK_GATE_BANK] == i;
}
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline bool usable_priority(float p) { return isfinite(p) && p >= 0.0f; }
inline bool usable_duration(Ctx ctx, float d) {
    return isfinite(d) && d > 0.0f;
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

    beta[i] = safe_w(bOld, clamp(bNew, 2.0f * log(decay), 0.0f));
    alpha[i] = safe_w(aOld, exp(beta[i]));
    float curvature = x * x;

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
    float sPrio = ctx.prioOpt[sOpt];
    ctx.stats->terminationRate = termSum * ctx.invK;
    ctx.stats->selectedOptionAge = float(ctx.optAge[sOpt]);
    ctx.stats->selectedOptionPriority = isfinite(sPrio) ? sPrio : 0.0f;
    ctx.stats->keyboardL1 = keyL1;
    ctx.stats->keyboardEffectiveL1 = effectiveL1;
    ctx.stats->actorAdvantageL1 = advL1;
    ctx.stats->selectedOptionIndex = float(sOpt);
    ctx.stats->selectedContinuation = ctx.continuation[sOpt];
    ctx.stats->selectedKeyboardWeight = isfinite(ctx.keyboard[sOpt]) ? ctx.keyboard[sOpt] : 0.0f;
    ctx.stats->mainTd = ctx.td[0];
    ctx.stats->planTdValue = ctx.planTd[0];
}

inline void fcstomp_store_learn_scales(Ctx ctx, uint tid) {

    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    device const float *prevS = state_at(ctx, ctx.stateSlot);
    device const float *currS = state_at(ctx, 1u - ctx.stateSlot);
    // Warp-cooperative gather + ordered replay: lanes load their strided
    // rows in parallel, then every lane replays the original ascending-a
    // accumulation from simd_shuffle'd RAW values, re-evaluating the exact
    // baseline expressions on the receiving lanes (Inv-10 lane-ordered
    // reduction). The full expression MUST be re-evaluated after the
    // shuffle: precomputing a product term per lane and adding it after the
    // shuffle breaks the compiler's mul+add -> fma contraction of the
    // baseline expression and diverges by 1 ulp (lab-verified on-device).
    float denomTd = 1.0f, denomPlan = 1.0f, xNorm = 0.0f;
    for (uint base = 0u; base < OAK_ACT; base += 32u) {
        uint a = base + tid;
        uint ip = actP[a], ic = actC[a];
        float xp = prevS[ip], xc = currS[ic];
        float ap = ctx.alpha[ip];
        float ac = ctx.alpha[ic];
        for (uint t = 0u; t < 32u; ++t) {
            float x = simd_shuffle(xp, (ushort)t);
            float aps = simd_shuffle(ap, (ushort)t);
            if (isfinite(aps) && aps > 0.0f) denomTd += aps * (x * x);
            float xs = simd_shuffle(xc, (ushort)t);
            float acs = simd_shuffle(ac, (ushort)t);
            if (isfinite(acs) && acs > 0.0f) denomPlan += acs * (xs * xs);
            xNorm = fma(x, x, xNorm);
        }
    }
    if (tid == 0u) {
        ctx.valueScale[0] = (isfinite(denomTd) && denomTd > 0.0f) ? (1.0f / denomTd) : 0.0f;
        ctx.planScale[0] = (isfinite(denomPlan) && denomPlan > 0.0f) ? (1.0f / denomPlan) : 0.0f;
        ctx.xNormPrev[0] = xNorm;
    }
}

inline void fcstomp_prediction_control_loop(Ctx ctx, uint tid,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgStash) {

    device float *tgG = ctx.fcG;
    device float *tgScalars = ctx.fcS;
    device const float *dPrev = state_at(ctx, ctx.stateSlot);
    device const float *dCurr = state_at(ctx, 1u - ctx.stateSlot);
    device const uint *actP = act_at(ctx, ctx.stateSlot);
    device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
    float vPrev = 0.0f, vCurr = 0.0f;
    {

        float pP = 0.0f, pC = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            pP = fma(ctx.valueW[ip], dPrev[ip], pP);
            pC = fma(ctx.valueW[ic], dCurr[ic], pC);
        }
        vPrev = simd_sum(pP);
        vCurr = simd_sum(pC);
        if (OAK_ABLATE_GVF_PREDICTION) {
            vPrev = 0.0f;
            vCurr = 0.0f;
        }
        if (tid == 0u) {
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
    }

    {
        float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            float xp = dPrev[ip], xc = dCurr[ic];
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
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < ctx.K && ctx.parked[tid] != 0u) {
        // Phase 9.3 K-parking: a parked option hibernates — its parameter
        // rows are KEPT but frozen; the 11-dot A-gather, RR TDs, and model
        // trust computation are skipped. The outputs written here are the
        // neutral elements of the full-width K scans below (stopped
        // continuation, zero tdPsi/keyboard, untrusted q -> never GPI-
        // selectable), so selection/keyboard/GPI keep their exact shapes
        // over the LIVE bank. twinDis/lRawC deliberately keep their last
        // live values: the vacuum probe reads them across the full bank and
        // may select a parked option — the probe-revive path (the window
        // analytics assessment unparks it the same step).
        ctx.continuation[tid] = 0.0f;
        ctx.tdPsi[tid] = 0.0f;
        ctx.keyboard[tid] = 0.0f;
        tgCont[tid] = 0.0f;
        tgG[tid] = NAN;
    } else if (tid < ctx.K) {
        uint feat = ctx.optFeature[tid] % ctx.R;
        // One-pass gather over the active rows: psi/hDot/model dots plus the
        // plan-stage prev-dots, each an independent accumulator chain keeping
        // its original ascending-a fma order (the chains are interleaved for
        // ILP, never reordered).
        float psiPrev = 0.0f, psiNext = 0.0f, hDot = 0.0f;
        float rHat = 0.0f, lRaw = 0.0f, vModel = 0.0f, vModel2 = 0.0f;
        float pdVs = 0.0f, pdVsA = 0.0f, pdR = 0.0f, pdL = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = act_of(actP, a), ic = act_of(actC, a);
            float xp = dPrev[ip], xc = dCurr[ic];
            uint idxP = ip * ctx.K + tid, idxC = ic * ctx.K + tid;
            psiPrev = fma(ctx.psi[idxP], xp, psiPrev);
            psiNext = fma(ctx.psi[idxC], xc, psiNext);
            hDot = fma(ctx.psiAux[idxP], xp, hDot);
            if (!OAK_ABLATE_OPTION_MODELS) {
                rHat = fma(ctx.optReward[idxC], xc, rHat);
                lRaw = fma(ctx.optDuration[idxC], xc, lRaw);
                vModel = fma(ctx.vStop[idxC], xc, vModel);
                vModel2 = fma(ctx.vStop2[idxC], xc, vModel2);
            }
            pdVs = fma(ctx.vStop[idxP], xp, pdVs);
            pdVsA = fma(ctx.vsAux[idxP], xp, pdVsA);
            pdR = fma(ctx.optReward[idxP], xp, pdR);
            pdL = fma(ctx.optDuration[idxP], xp, pdL);
        }
        // Stash the prev-dots for oak_plan_one (threadgroup memory, visible
        // after the stage threadgroup_barrier). oak_replace_commit zeroes an
        // option's stash when it reinitializes that option's rows, matching
        // what a post-replace gather would produce (rows zeroed, finite state
        // -> +0.0f).
        tgStash[tid * 4u + 0u] = pdVs;
        tgStash[tid * 4u + 1u] = pdVsA;
        tgStash[tid * 4u + 2u] = pdR;
        tgStash[tid * 4u + 3u] = pdL;

        float psiXX = tgScalars[5], psiXY = tgScalars[6], psiYY = tgScalars[7];

        float wBonus = ctx.bonusW[feat] - ctx.valueW[feat];
        if (!isfinite(wBonus) || wBonus < 0.0f) wBonus = 0.0f;
        float zStop = vCurrG + (1.0f + wBonus) * dCurr[feat];
        bool rrFinite = isfinite(psiPrev) && isfinite(psiNext) && isfinite(zStop);
        if (!rrFinite) ctx.counters[OAK_CNT_TDPSI_NF] += 1u;

        float beta = (isfinite(psiNext) && isfinite(zStop) && zStop >= psiNext) ? 1.0f : 0.0f;

        if (tid == 0u) beta = 0.0f;
        if (OAK_ABLATE_RR_SUBTASKS) {
            rrFinite = true;
            beta = (tid == 0u) ? 0.0f : 1.0f;
        }
        float cont = 1.0f - beta;
        ctx.continuation[tid] = cont;
        if (beta > 0.0f) ctx.lastStopAge[tid] = 0u;

        float psiRhoOld = ctx.psiRho[tid];
        if (!isfinite(psiRhoOld)) psiRhoOld = 0.0f;
        float dPsi = (!OAK_ABLATE_RR_SUBTASKS && rrFinite)
            ? (rewardC - psiRhoOld + cont * psiNext + beta * zStop - psiPrev) : 0.0f;
        if (!isfinite(dPsi)) {
            dPsi = 0.0f;
            ctx.counters[OAK_CNT_TDPSI_NF] += 1u;
        }

        if (tid == 0u) dPsi = ctx.td[0] - psiRhoOld;
        ctx.tdPsi[tid] = dPsi;
        {
            float psiRhoNew = psiRhoOld + ctx.invR * dPsi;
            ctx.psiRho[tid] = safe_w(psiRhoOld, psiRhoNew);
        }
        float rho = ctx.importance[tid];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        {

            float fitOld = ctx.optFit[tid];
            if (!isfinite(fitOld)) fitOld = 0.0f;
            float rRaw = ctx.reward[0];
            if (!isfinite(rRaw)) rRaw = 0.0f;
            ctx.optFit[tid] = safe_w(fitOld, fitOld + ctx.invR * rho * (rRaw - fitOld));
        }

        float implicitTrace = psiXX - 2.0f * cont * psiXY + cont * cont * psiYY;
        float denom = fma(rho, implicitTrace, 1.0f);

        float hd = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        ctx.headScale[tid] = (isfinite(hd) && hd > 0.0f) ? (1.0f / hd) : 0.0f;
        {
            ctx.hDotPsi[tid] = isfinite(hDot) ? hDot : 0.0f;

            float auxDenom = fma(rho, psiXX, 1.0f);
            ctx.psiAuxScale[tid] = (isfinite(auxDenom) && auxDenom > 0.0f)
                ? (1.0f / auxDenom) : 0.0f;
        }
        tgCont[tid] = cont;
        ctx.rHatC[tid] = rHat;
        ctx.lRawC[tid] = lRaw;
        ctx.vsNextC[tid] = vModel;
        // Twin disagreement (Phase 3): epistemic uncertainty at the current
        // state, per option — feeds the vacuum probe (disagreement·duration
        // argmax). With the data-split twins this is now a real signal.
        float tdis = fabs(vModel - vModel2);
        ctx.twinDis[tid] = isfinite(tdis) ? tdis : 0.0f;

        if (tid == 0u) vModel = vCurr;
        // Toxic greedy-Q-selection organ EXCISED (Phase 7.5): the model-backed
        // qAdv -> gpiBest selection path was CONVICTED (Phase 6: enabling it
        // halved IQM 0.107 vs 0.206, crushed igniters — the never-dies-never-
        // walks trap). It is permanently OFF (tgG = NAN), so the keeper's
        // model-to-keyboard behavior is the DEFAULT — no ablation flag, `none`
        // is healthy. gpiBest/keyboard-lam/planDelta degenerate to the keeper's
        // frozen-chord behavior exactly as under the old ablation.
        tgG[tid] = NAN;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gaugeRate = track_rate(ctx.stepCount, ctx.R);
        float gaugeNew = ctx.valueGauge[0] + gaugeRate * (vCurr - ctx.valueGauge[0]);
        ctx.valueGauge[0] = safe_w(ctx.valueGauge[0], gaugeNew);
    }
    if (tid < ctx.K) {

        float adv = OAK_ABLATE_OPTION_POLICIES ? 0.0f : ctx.tdPsi[tid];
        if (!isfinite(adv)) adv = 0.0f;
        ctx.actorAdv[tid] = adv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint gpiBest;
    float gpiBestScore;
    {
        float g = (tid < ctx.K) ? tgG[tid] : -FLT_MAX;
        float score = isfinite(g) ? g : -FLT_MAX;
        gpiBestScore = simd_max(score);
        gpiBest = simd_min((score == gpiBestScore) ? tid : 0xffffffffu);
    }
    // Learn scales are gathered warp-cooperatively (ordered replay, exact
    // serial add/fma sequence); the value is identical at both former call
    // sites — nothing between them writes alpha/act/prev/curr.
    fcstomp_store_learn_scales(ctx, tid);
    // Warp-uniform keyboard refresh: the per-k raw values are gathered
    // one-per-lane and the original expressions re-evaluated in ascending-k
    // order from simd_shuffle (skips replicated exactly) — the identical
    // add sequence as the former lane-0 serial sums (Inv-10 lane-ordered
    // reduction).
    uint prevSel = ctx.selectedOption[0] % ctx.K;
    bool continuing = (prevSel != 0u) && tgCont[prevSel] > 0.0f;
    uint selected = continuing ? prevSel : gpiBest;
    float contK = 0.0f, advK = 0.0f, prevK = 0.0f;
    float twinDisK = 0.0f, lRawK = 0.0f;
    if (tid < ctx.K) {
        contK = tgCont[tid];
        advK = ctx.actorAdv[tid];
        prevK = ctx.keyboard[tid];
        twinDisK = ctx.twinDis[tid];
        lRawK = ctx.lRawC[tid];
    }
    float termSum = 0.0f, advL1 = 0.0f, prevMag = 0.0f;
    for (uint t = 0u; t < ctx.K; ++t) {
        float c = simd_shuffle(contK, (ushort)t);
        termSum += 1.0f - c;
        float a = simd_shuffle(advK, (ushort)t);
        if (isfinite(a)) advL1 += fabs(a);
        float w = simd_shuffle(prevK, (ushort)t);
        if (isfinite(w) && w > 0.0f) prevMag += w;
    }
    if (OAK_ABLATE_OPTION_KEYBOARD) {
        if (tid < ctx.K) ctx.keyboard[tid] = 0.0f;
        if (tid == 0u) {
            ctx.planTd[0] = 0.0f;
            ctx.selectedOption[0] = selected;
            fcstomp_record_keyboard_stats(ctx, selected, termSum, 0.0f, advL1, 0.0f);
        }
        return;
    }

    // Softmax refresh: each lane computes and writes its own chord weight
    // (same exp inputs as the serial loop); the mag replay keeps the
    // ascending-k add order.
    float lamK = 0.0f;
    bool refreshKeyboard = (!continuing && gpiBest != prevSel) || prevMag <= 0.0f;
    if (refreshKeyboard) {
        bool haveBest = gpiBestScore > -FLT_MAX;
        if (tid < ctx.K) {
            bool trusted = isfinite(tgG[tid]) && haveBest;
            float lam = trusted ? exp(tgG[tid] - gpiBestScore) : 0.0f;
            if (!isfinite(lam)) lam = 0.0f;

            if (ctx.optAge[tid] < ctx.R && tid != gpiBest && tid != 0u) lam = 0.0f;
            ctx.keyboard[tid] = lam;
            lamK = lam;
        }
        float mag = 0.0f;
        for (uint t = 0u; t < ctx.K; ++t) mag += simd_shuffle(lamK, (ushort)t);
        if (mag <= 0.0f) {
            // Vacuum probe (Phase 3, the chicken-and-egg breaker): probe the
            // option with the highest twin-disagreement × duration — epistemic
            // uncertainty at the current state, weighted toward temporally-
            // extended commitments. Fires ONLY in commitment vacuum (a rocket
            // lane is always continuing, so it never sees this). Replaces the
            // blind round-robin probe. Collects exactly the transition data
            // that maximally reduces model error (M2AC's mask inverted into a
            // directed data-collection rule). Early on l-hat≈1 so it's pure
            // disagreement; anneals toward long-horizon probing as duration
            // models learn (stationary rule, annealing behavior, no schedule).
            float bestScore = -FLT_MAX;
            uint probe = 0u;
            for (uint t = 0u; t < ctx.K; ++t) {
                float dis = simd_shuffle(twinDisK, (ushort)t);
                float dur = decode_duration(simd_shuffle(lRawK, (ushort)t));
                float sc = dis * dur;
                if (isfinite(sc) && sc > bestScore) { bestScore = sc; probe = t; }
            }
            lamK = (tid == probe) ? 1.0f : 0.0f;
            if (tid < ctx.K) ctx.keyboard[tid] = lamK;
        }
    }

    float planDelta = (OAK_ABLATE_PLANNING || gpiBestScore <= -FLT_MAX) ? 0.0f : gpiBestScore;
    {

        float err = ctx.prioOpt[gpiBest];
        float trust = (isfinite(err) && err >= 0.0f) ? (1.0f / (1.0f + err)) : 0.0f;
        planDelta *= trust;
    }

    // Final chord weights: what this lane just wrote, or the untouched
    // device value on a no-refresh step (identical bits either way). The
    // keyL1 replay re-evaluates the original skip expression on the
    // shuffled raw weights.
    float keyK = 0.0f;
    if (tid < ctx.K) keyK = refreshKeyboard ? lamK : ctx.keyboard[tid];
    float keyL1 = 0.0f;
    for (uint t = 0u; t < ctx.K; ++t) {
        float w = simd_shuffle(keyK, (ushort)t);
        if (isfinite(w)) keyL1 += fabs(w);
    }
    if (tid == 0u) {
        // Poison deleted (Phase 0): planTd does NOT feed rho-bar. rho-bar is
        // real-stream-only (Wan-Naik-Sutton 2006.16318); model estimates must
        // never write into rho-bar (the self-referential estimator — the 13
        // measured runaways to +539). planTd is a diagnostic-only signal now.
        ctx.planTd[0] = isfinite(planDelta) ? planDelta : 0.0f;
        ctx.selectedOption[0] = selected;
        fcstomp_record_keyboard_stats(ctx, selected, termSum, keyL1, advL1,
                                      keyL1 > 0.0f ? 1.0f : 0.0f);
    }
}

inline void FeatureConstruction(Ctx ctx,
                                uint tid,
                                uint tgW) {
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

    if (tid < ctx.K) ctx.importance[tid] = ctx.impStage[tid];
    for (uint j = tid; j < ctx.K * ctx.D; j += tgW)
        ctx.optionActorCredit[j] = ctx.creditStage[j];
    // No fence here: the staged copies above are consumed either same-thread
    // (fcstomp reads importance[tid]) or in later stages behind the step's
    // threadgroup_barrier (plan reads optionActorCredit). The cross-thread
    // RAWs below (curr[gateCh], kind-3 obsTraceMulti) are covered by the
    // mem_device barrier after the core-slot writes.

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device float *curr = state_at(ctx, 1u - ctx.stateSlot);
    device uint *actC = act_at(ctx, 1u - ctx.stateSlot);

    const bool legacy = oak_legacy_geometry(ctx);
    // derived-core organ (§2 content off): the padded-width slots [naturalO,O)
    // and slot 31 fall back to the legacy writes (standardized_obs / 0.0f)
    // while §1 geometry stays live. The trace barrier below
    // is value-neutral under the organ, so it stays conditioned on !legacy.
    const bool derivedCore = !legacy && !OAK_ABLATE_DERIVED_CORE;
    if (!legacy) {
        // §2's derived core slots below read obsTraceMulti slots updated by
        // OTHER threads of this warp in the trace loop above; order the
        // cross-thread RAW (the same discipline as the mem_device barrier
        // after the core-slot writes). Uniform across the lane's simdgroup;
        // the legacy path reads no other-thread trace state here and skips it.
        simdgroup_barrier(mem_flags::mem_device);
    }

    {
        uint i = tid;
        float x;
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            x = (i == 0u) ? 1.0f : 0.0f;
        } else if (i < ctx.O) {
            if (derivedCore && i >= ctx.naturalO) {
                // §2 derived core channel: the slots the padded width leaves
                // unused hold multi-timescale obs traces, row-major
                // (channel, level): free-slot j = level*O_n + channel.
                uint j = i - ctx.naturalO;
                uint lvl = j / ctx.naturalO;
                x = (lvl < ctx.traceLevels)
                    ? ctx.obsTraceMulti[lvl * ctx.O + (j - lvl * ctx.naturalO)]
                    : 0.0f;
            } else {
                x = standardized_obs(ctx, i);
            }
        } else if (i < ctx.O + ctx.D) {
            x = ctx.actionSignal[i - ctx.O];
        } else if (i == ctx.O + ctx.D) {
            x = ctx.reward[0];
        } else if (i == ctx.O + ctx.D + 1u) {
            x = 1.0f;
        } else if (derivedCore) {
            // §2 slot 31: the last free slot (j = O - O_n), same trace fill.
            uint j = ctx.O - ctx.naturalO;
            uint lvl = j / ctx.naturalO;
            x = (lvl < ctx.traceLevels)
                ? ctx.obsTraceMulti[lvl * ctx.O + (j - lvl * ctx.naturalO)]
                : 0.0f;
        } else {
            x = 0.0f;
        }
        if (!isfinite(x)) { x = 0.0f; ctx.counters[OAK_CNT_OBS_NF] += 1u; }
        curr[i] = x;
        actC[tid] = i;
        if (ctx.featureAge[i] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
            ctx.featureAge[i] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    uint tile;
    if (legacy) {
        uint gateCh = gate_channel(ctx, tid);
        tile = gate_tile(curr[gateCh]);
    } else {
        // §1.3 geometry from stream shape: S = O_n + A_n + 1 informative gate
        // sources per occurrence round. occ 0 = single tile; occ >= 1 even
        // bank = zoom tile; occ >= 1 odd bank = joint 3x2+overflow cell over
        // (ch(src), obs (src+occ) % O_n). Exactly one winner per bank either
        // way (Mark III law (b): stationary geometry gating, no magnitude
        // competition).
        uint S = ctx.naturalO + ctx.naturalA + 1u;
        uint src = tid % S, occ = tid / S;
        // joint-geometry organ (§1.3 zoom/joint off): every bank keeps the
        // informative-source cycle ch(src) but gates on the occ-0 single tile
        // — single-channel tiles, still shape-derived; §2 stays live.
        if (occ == 0u || OAK_ABLATE_JOINT_GEOMETRY) {
            tile = gate_tile(curr[gate_source_channel(ctx, src)]);
        } else if ((tid & 1u) == 0u) {
            tile = gate_tile_zoom(curr[gate_source_channel(ctx, src)], occ);
        } else {
            uint ca = gate_source_channel(ctx, src);
            uint cb = (src + occ) % ctx.naturalO;
            tile = joint_cell(curr[ca], curr[cb], occ);
        }
    }
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
}

static inline void oak_stage_encode(Ctx ctx, uint tid, uint tgW,
                                    threadgroup float *tgCont,
                                    threadgroup float *tgStash) {
    FeatureConstruction(ctx, tid, tgW);
    fcstomp_prediction_control_loop(ctx, tid, tgCont, tgStash);
}

static inline void oak_plan_one(Ctx ctx, uint k,
                                threadgroup const float *stash,
                                bool stashDirty, uint pSlot) {
    float cont = ctx.continuation[k];
    if (OAK_ABLATE_OPTION_MODELS) {
        ctx.optRewardTd[k] = 0.0f;
        ctx.prioOpt[k] = 0.0f;
        ctx.vsTd[k] = 0.0f;
        ctx.vsDot[k] = 0.0f;
        ctx.vsScale[k] = 0.0f;
        ctx.vsAuxScale[k] = 0.0f;
        return;
    }

        // Prev-dots were gathered by encode (same actP/prev rows, same fma
        // order) and stashed in threadgroup memory; replace zeroes the stash
        // of any option whose rows it reinitialized this step. A feature
        // replacement (dynaSlot type 1) additionally zeroes one feature ROW
        // of every option's model rows after the stash was written; that
        // (rare) step sets the lane's dirty flag and we re-gather here —
        // identical chains to the stash path, read post-replace.
        float vsPrev, vsDot, rPrev, lPrev;
        if (!stashDirty) {
            vsPrev = stash[k * 4u + 0u];
            vsDot = stash[k * 4u + 1u];
            rPrev = stash[k * 4u + 2u];
            lPrev = stash[k * 4u + 3u];
        } else {
            device const uint *actP = act_at(ctx, pSlot);
            device const float *prev = state_at(ctx, pSlot);
            vsPrev = 0.0f; vsDot = 0.0f; rPrev = 0.0f; lPrev = 0.0f;
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint ip = act_of(actP, a);
                uint idxP = ip * ctx.K + k;
                float xp = prev[ip];
                vsPrev = fma(ctx.vStop[idxP], xp, vsPrev);
                vsDot = fma(ctx.vsAux[idxP], xp, vsDot);
                rPrev = fma(ctx.optReward[idxP], xp, rPrev);
                lPrev = fma(ctx.optDuration[idxP], xp, lPrev);
            }
        }

        float rHat = ctx.rHatC[k], lRaw = ctx.lRawC[k], vsNext = ctx.vsNextC[k];
        float vX = ctx.fcS[1];

        float xNormA = ctx.fcS[8];
        float bootNorm = ctx.fcS[8] - 2.0f * cont * ctx.fcS[9]
                       + cont * cont * ctx.fcS[10];
        bootNorm = max(bootNorm, 0.0f);
        if (!isfinite(bootNorm)) bootNorm = 0.0f;
        if (!isfinite(xNormA) || xNormA < 0.0f) xNormA = 0.0f;

        float rHat2 = rHat, lRaw2 = lRaw;
        float optRhoOld = ctx.optRho[k];
        if (!isfinite(optRhoOld)) optRhoOld = 0.0f;
        // optRho as a simple importance-weighted EMA of the reward (NOT the
        // optReward bootstrap TD — the optReward model's unanchored bootstrap
        // explodes over long horizons and is REMOVED from the VI backup). This
        // tracks the option's average reward rate directly, stably, for the
        // de-bias (optRho_k - rho-bar)·l-hat.
        float rTd = (ctx.planScr[0] - optRhoOld) + cont * rHat2 - rPrev;
        float rhoImp = ctx.importance[k];
        if (!isfinite(rhoImp) || rhoImp < 0.0f) rhoImp = 0.0f;
        if (isfinite(ctx.planScr[0]))
            ctx.optRho[k] = optRhoOld + track_rate(ctx.stepCount, ctx.R) * rhoImp * (ctx.planScr[0] - optRhoOld);
        float dTd = (k == 0u) ? 0.0f : (1.0f + cont * lRaw2 - lPrev);

        float vsTd = cont * vsNext + (1.0f - cont) * vX - vsPrev;
        bool modelFinite = isfinite(rTd) && isfinite(dTd) && isfinite(lRaw) && isfinite(vsTd);
        if (!isfinite(rTd)) rTd = 0.0f;
        if (!isfinite(dTd)) dTd = 0.0f;
        if (!isfinite(vsTd)) vsTd = 0.0f;
        ctx.optRewardTd[k] = rTd;
        ctx.optDurationTd[k] = dTd;
        ctx.vsTd[k] = vsTd;
        ctx.vsDot[k] = isfinite(vsDot) ? vsDot : 0.0f;
        float rho = ctx.importance[k];
        if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
        float sDen = fma(rho, bootNorm, 1.0f);
        ctx.vsScale[k] = (isfinite(sDen) && sDen > 0.0f) ? (1.0f / sDen) : 0.0f;
        float aDen = fma(rho, xNormA, 1.0f);
        ctx.vsAuxScale[k] = (isfinite(aDen) && aDen > 0.0f) ? (1.0f / aDen) : 0.0f;
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

static inline void oak_stage_plan_all(Ctx ctx, uint tid,
                                      threadgroup const float *stash,
                                      bool stashDirty, uint pSlot) {
    if (tid >= ctx.K) return;
    if (ctx.parked[tid] != 0u) {
        // Phase 9.3: a parked option's model rows are frozen — skip the model
        // TDs, polCoef, and the actor meta-step, and publish the neutral
        // outputs (the same zeros the option-models ablation path writes).
        // prioOpt = 0 keeps the full-width utility/priority scans measuring
        // the LIVE bank only.
        ctx.optRewardTd[tid] = 0.0f;
        ctx.optDurationTd[tid] = 0.0f;
        ctx.vsTd[tid] = 0.0f;
        ctx.vsDot[tid] = 0.0f;
        ctx.vsScale[tid] = 0.0f;
        ctx.vsAuxScale[tid] = 0.0f;
        ctx.prioOpt[tid] = 0.0f;
        return;
    }
    oak_plan_one(ctx, tid, stash, stashDirty, pSlot);

    {
        uint k = tid;
        float rho = ctx.importance[k];
        if (!isfinite(rho)) rho = 0.0f;
        float adv = ctx.actorAdv[k];
        if (!isfinite(adv)) adv = 0.0f;

        metaopt_update_slot(ctx.alphaAct, ctx.betaAct, ctx.hAct, ctx.energyAct,
                            k, adv, 1.0f, ctx.optAge[k], ctx.metaEta, ctx.invR);
        float actorAlpha = ctx.alphaAct[k] / ctx.actorAlphaDenom;
        float xNorm = ctx.xNormPrev[0];
        for (uint d = 0u; d < oak_dl(ctx); ++d) {
            float targetGrad = ctx.optionActorCredit[k * ctx.D + d];
            if (!isfinite(targetGrad)) targetGrad = 0.0f;
            float gradScale = rho * targetGrad;

            float denom = 1.0f + actorAlpha * adv * adv * gradScale * gradScale * xNorm;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float c = actorAlpha * adv * gradScale * scale;
            ctx.polCoef[k * oak_dl(ctx) + d] = isfinite(c) ? c : 0.0f;
        }
    }
}

static inline void oak_learn_rows(Ctx ctx, device const float *prev,
                                  device const float *curr, uint tid, uint tw,
                                  uint pSlot) {
    device const uint *actP = act_at(ctx, pSlot);
    device const uint *actC = act_at(ctx, 1u - pSlot);
    const uint A = OAK_ACT;
    uint K = ctx.K;
    uint DL = oak_dl(ctx);
    uint KDL = K * DL;

    uint k = tid;
    // Phase 9.3: a parked option's rows are frozen — skip its column RMWs
    // (the A×DL policy apply and the A-priced psi/model applies). The lane
    // still loads and broadcasts its transpose rows below (every lane's row
    // slice is needed by all 32 option columns) and still applies its owned
    // valueW rows at the end: the value bank is feature-indexed, never
    // option-indexed.
    const bool optLive = (ctx.parked[k] == 0u);
    float rho = ctx.importance[k];
    if (!isfinite(rho) || rho < 0.0f) rho = 0.0f;
    float cont = ctx.continuation[k];
    if (!isfinite(cont)) cont = 0.0f;
    float hD = ctx.hDotPsi[k], sc = ctx.headScale[k], td = ctx.tdPsi[k];
    float vD = ctx.vsDot[k], vsc = ctx.vsScale[k], vtd = ctx.vsTd[k];
    float rtd = ctx.optRewardTd[k], dtd = ctx.optDurationTd[k];
    float psiErr = td - hD, psc = ctx.psiAuxScale[k];
    float vErr = vtd - vD, vasc = ctx.vsAuxScale[k];
    float cj[16];
    for (uint j = 0u; j < DL; ++j) cj[j] = ctx.polCoef[tid * DL + j];
    // Row-data transpose (Inv-10 bit-identical load reduction): the five
    // i-indexed row values (actP entry, prev/curr, alphaPsi/alphaModel) are
    // lane-uniform — every lane re-loads the same address per row. Lane tid
    // holds the cb+tid row slice as RAW loads and each broadcast round
    // re-evaluates the exact baseline RMW bodies from simd_shuffle'd raw
    // values (never precomputed products — same expression tree, same fma
    // contraction), rows ascending per lane. No skips; the RMW bodies are
    // untouched.
    for (uint cb = 0u; cb < A; cb += 32u) {
        uint aH = cb + tid;
        uint iH = act_of(actP, aH);
        float xpH = prev[iH], xcH = curr[iH];
        float aPH = ctx.alphaPsi[iH], aMH = ctx.alphaModel[iH];
        uint lEnd = min(32u, A - cb);
        for (uint l = 0u; l < lEnd; ++l) {
            uint i = simd_shuffle(iH, (ushort)l);
            float xp = simd_shuffle(xpH, (ushort)l);
            float xc = simd_shuffle(xcH, (ushort)l);
            float aP = simd_shuffle(aPH, (ushort)l);
            float aM = simd_shuffle(aMH, (ushort)l);
            if (!optLive) continue;  // parked: frozen rows (Phase 9.3)
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
                // Patch A (Phase 1): the option reward/duration models actually
                // optReward model's unanchored bootstrap explodes over long
                // horizons (11 deaths @200k) and is REMOVED from the VI backup
                // — the reward-to-go comes from the trajectory relay (Holland
                // 2018), not a bootstrap reward model. optDuration at 0.05 rate
                // (its bootstrap cont*lRaw2 also runs away over long horizons,
                // 3 deaths @300k) — conservative rate for the de-bias duration.
                device float *lw = &ctx.optDuration[i * K + k];
                *lw = safe_w(*lw, *lw + aM * rho * dtd * xp * vsc * 0.05f);
            }
        }
    }

    // Same transpose for the curr-only rows (OAK_CORE..A is exactly one
    // simdgroup chunk). The membership flag and the four row values are RAW
    // loads broadcast by uniform shuffles; the baseline's whole-loop gate
    // (cont > 0 && rho > 0) is lane-constant, so applying it per row is
    // value-identical — rows are independent RMWs on disjoint (i,k) slots,
    // ascending order preserved.
    {
        uint i2H = act_of(actC, OAK_CORE + tid);
        uint in2H = oak_in_prev_active(actP, i2H) ? 1u : 0u;
        float xc2H = curr[i2H];
        float aP2H = ctx.alphaPsi[i2H], aM2H = ctx.alphaModel[i2H];
        for (uint l = 0u; l < 32u; ++l) {
            uint i = simd_shuffle(i2H, (ushort)l);
            uint inP = simd_shuffle(in2H, (ushort)l);
            float xc = simd_shuffle(xc2H, (ushort)l);
            float aP = simd_shuffle(aP2H, (ushort)l);
            float aM = simd_shuffle(aM2H, (ushort)l);
            if (!optLive || inP != 0u) continue;
            if (cont > 0.0f && rho > 0.0f) {
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
    }

    if (!OAK_ABLATE_GVF_PREDICTION) {
        // Same transpose for the active-set walk: lane tid holds the
        // e = cb+tid active-row index as a RAW load, broadcast by uniform
        // shuffles; the i % tw ownership filter and the RMW body are
        // unchanged and stay per-lane. The TD scalar is a loop-invariant RAW
        // load hoisted with no expression-tree change.
        float td0 = ctx.td[0];
        float vScale0 = ctx.valueScale[0];
        // Poison deleted (Phase 0): planTd and dynaAcc do NOT write into
        // valueW (model estimates must never write into learner weights — the
        // measured population anti-learner). Only the real experience TD moves
        // the value bank; the e>=A (planTd-only) half is deleted entirely.
        for (uint cb = 0u; cb < A; cb += 32u) {
            uint eH = cb + tid;
            uint iH = act_of(actP, eH);
            for (uint l = 0u; l < 32u; ++l) {
                uint e = cb + l;
                uint i = simd_shuffle(iH, (ushort)l);
                if (i % tw != tid) continue;
                float w = ctx.valueW[i];
                float delta = td0 * prev[i] * vScale0;
                ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * delta);
            }
        }
    }
}

inline void emit_all_options(Ctx ctx, uint tid) {

    uint DL = oak_dl(ctx);
    uint KDL = ctx.K * DL;
    device const uint *act = act_at(ctx, ctx.stateSlot);
    device const float *currS = state_at(ctx, ctx.stateSlot);
    if (OAK_ABLATE_OPTION_POLICIES) {
        for (uint j = 0u; j < ctx.D; ++j) {
            ctx.optionPolicyMean[tid * ctx.D + j] = 0.0f;
            ctx.muRaw[tid * ctx.D + j] = 0.0f;
        }
        if (tid < ctx.D) ctx.policyMean[tid] = 0.0f;
        return;
    }
    if (ctx.parked[tid] != 0u) {
        // Phase 9.3: a parked option emits exact zeros and skips the A×DL
        // gather — its keyboard weight is zero (or it is being probe-revived
        // this step), and its rows stay frozen, never zeroed.
        for (uint j = 0u; j < ctx.D; ++j) {
            ctx.optionPolicyMean[tid * ctx.D + j] = 0.0f;
            ctx.muRaw[tid * ctx.D + j] = 0.0f;
        }
        return;
    }
    if (DL == 6u) {
        // Shape-specialized emit gather (DL = 6 under OAK_SPEC_ACTION_PARAMS,
        // Inv-10 shape-specialized unrolling): six scalar accumulators keep
        // the per-dim fma chains in registers — the generic acc[16] array
        // risks a thread-memory spill. Pure gather; each chain keeps its
        // original ascending-a fma order, interleaved for ILP only.
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
        float acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint i = act_of(act, a);
            float x = currS[i];
            device const float *row = &ctx.optPolicy[i * KDL + tid * 6u];
            acc0 = fma(row[0], x, acc0);
            acc1 = fma(row[1], x, acc1);
            acc2 = fma(row[2], x, acc2);
            acc3 = fma(row[3], x, acc3);
            acc4 = fma(row[4], x, acc4);
            acc5 = fma(row[5], x, acc5);
        }
        float accv[6] = {acc0, acc1, acc2, acc3, acc4, acc5};
        for (uint j = 0u; j < ctx.D; ++j) {
            uint kd = tid * ctx.D + j;

            // Zenith law scoped to Box (Phase 2): tanh stays on optionPolicyMean
            // (cheetah actuator bounds, bit-identical); muRaw now holds the RAW
            // acc — the unbounded discrete logits (commitment on sparse envs).
            float mu = (j < 6u) ? tanh(accv[j]) : 0.0f;
            ctx.muRaw[kd] = (j < 6u) ? accv[j] : 0.0f;
            if (!isfinite(mu)) mu = ctx.optionPolicyMean[kd];
            ctx.optionPolicyMean[kd] = mu;
        }
        return;
    }
    float acc[16];
    for (uint j = 0u; j < DL; ++j) acc[j] = 0.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) {
        uint i = act_of(act, a);
        float x = currS[i];
        device const float *row = &ctx.optPolicy[i * KDL + tid * DL];
        for (uint j = 0u; j < DL; ++j) acc[j] = fma(row[j], x, acc[j]);
    }
    for (uint j = 0u; j < ctx.D; ++j) {
        uint kd = tid * ctx.D + j;

        float mu = (j < DL) ? tanh(acc[j]) : 0.0f;
        ctx.muRaw[kd] = (j < DL) ? acc[j] : 0.0f;
        if (!isfinite(mu)) mu = ctx.optionPolicyMean[kd];
        ctx.optionPolicyMean[kd] = mu;
    }
}

inline void emit_blend_grid(Ctx ctx, uint tid) {
    if (tid >= ctx.D) return;
    uint d = tid;
    uint selected = ctx.selectedOption[0] % ctx.K;
    // Cheetah behavior blend over tanh means (unchanged); discrete raw blend
    // over raw-acc means (unbounded logits). Two separate accumulators.
    float blend = 0.0f, wNorm = 0.0f, fallback = 0.0f;
    float blendR = 0.0f, fallbackR = 0.0f;
    for (uint ok = 0u; ok < ctx.K; ++ok) {
        float om = ctx.optionPolicyMean[ok * ctx.D + d];
        if (!isfinite(om)) om = 0.0f;
        if (ok == selected) fallback = om;
        float omR = ctx.muRaw[ok * ctx.D + d];
        if (!isfinite(omR)) omR = 0.0f;
        if (ok == selected) fallbackR = omR;
        float w = ctx.keyboard[ok];
        if (!isfinite(w)) w = 0.0f;
        blend = fma(w, om, blend);
        blendR = fma(w, omR, blendR);
        wNorm += fabs(w);
    }

    {
        float m0 = ctx.optionPolicyMean[d];
        if (isfinite(m0)) { blend += m0; wNorm += 1.0f; }
        float m0R = ctx.muRaw[d];
        if (isfinite(m0R)) blendR += m0R;
    }
    ctx.policyMean[d] = (wNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
        ? blend / wNorm : fallback;
    ctx.policyMeanRaw[d] = (wNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
        ? blendR / wNorm : fallbackR;
}

// Phase 9.1 planning priority index (Inv-10 bit-identical class): a binary
// tournament tree over the ring's dynaTd priorities, one aligned subtree per
// sweep partition (the P=8 budget; partitions are pow2-aligned, so partition
// p's slots are exactly the subtree rooted at node 8+p). Internal node i
// (8 <= i < M) stores the winner slot of its range under the sweep scan's
// exact ordering — higher priority wins, ties go to the lower slot — so the
// sweep's per-partition argmax is the single read planTree[8+p] instead of an
// M/8-slot scan (this is what keeps a larger ring M scan-free). dynaTd has
// exactly two write sites (the ring-write seed in oak_replace_commit and the
// backup's own write in oak_dyna_backups); both call oak_plan_index_update,
// so the tree equals a fresh full scan at every read. Comparison-only: no FP
// arithmetic, no value or selection change.
static inline float oak_plan_prio(Ctx ctx, uint slot) {
    float p = ctx.dynaTd[slot];
    return isfinite(p) ? p : 0.0f;  // the scan's own defensive map
}

static inline uint oak_plan_winner(Ctx ctx, uint node, uint M) {
    // Winner slot of a tree position: leaves (node >= M) win their own slot.
    return (node >= M) ? (node - M) : ctx.planTree[node];
}

static inline void oak_plan_index_update(Ctx ctx, uint slot) {
    uint M = ctx.R;
    uint proot = 8u + slot / (M / 8u);  // this slot's partition root
    for (uint i = (M + slot) >> 1; i >= proot; i >>= 1) {
        uint wa = oak_plan_winner(ctx, 2u * i, M);
        uint wb = oak_plan_winner(ctx, 2u * i + 1u, M);
        // Scan tie-break (strictly-greater keeps the first, lowest slot): the
        // left child's winner is always the lower slot, so the right child
        // must strictly exceed it to take the node.
        ctx.planTree[i] =
            (oak_plan_prio(ctx, wb) > oak_plan_prio(ctx, wa)) ? wb : wa;
    }
}

static inline void oak_replace_commit(Ctx ctx, uint tid, threadgroup float *tgStash,
                                      threadgroup float *tgDirty) {
    // Per-step stash validity flag for plan (cleared here, set if a feature
    // replacement zeroes a model row out from under encode's stashed dots).
    if (tid == 0u) tgDirty[0] = 0.0f;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    if (!OAK_ABLATE_PLANNING) {

        uint slot = ctx.stepCount % ctx.R;
        device const uint *actC = act_at(ctx, 1u - ctx.stateSlot);
        if (tid < 32u) {

            ctx.ring[slot * 64u + tid] = curr[actC[tid]];
            uint gi = actC[OAK_CORE + tid];
            ctx.ringIdx[slot * 32u + tid] = gi;
            ctx.ring[slot * 64u + OAK_CORE + tid] = curr[gi];
        }

        if (tid == 0u) {
            // Phase 4: seal the previous slot's outbound reward, then init this
            // slot's planner value to the learner's v(x) at ring-write.
            if (ctx.stepCount > 0u) {
                uint prevSlot = (ctx.stepCount - 1u) % ctx.R;
                ctx.ringReward[prevSlot] = ctx.reward[0];
            }
            float vX = 0.0f;
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint gi = (a < OAK_CORE) ? a : ctx.ringIdx[slot * 32u + (a - OAK_CORE)];
                vX = fma(ctx.valueW[gi], ctx.ring[slot * 64u + a], vX);
            }
            ctx.planU[slot] = isfinite(vX) ? vX : 0.0f;
            float p0 = fabs(ctx.td[0]) + fabs(ctx.planTd[0]);
            ctx.dynaTd[slot] = isfinite(p0) ? p0 : 0.0f;
            oak_plan_index_update(ctx, slot);
        }
    }

    if (tid == 0u) ctx.planScr[0] = ctx.reward[0];
    simdgroup_barrier(mem_flags::mem_device);

    uint bestN;
    float bestNUtil;
    {
        float bv = -FLT_MAX;
        uint bn = 0xffffffffu;
        for (uint n = tid; n < ctx.Rn; n += 32u) {
            float lcb = ctx.lcbScr[n];
            if (lcb > bv) { bv = lcb; bn = n; }
        }
        bestNUtil = simd_max(bv);
        bestN = simd_min((bv == bestNUtil) ? bn : 0xffffffffu);
    }

    uint haveWorstW = 0u, worstW = 0u, worstAgeW = 0u;
    float worstUtilW = 0.0f, worstRedW = 0.0f, worstDownW = 0.0f;
    for (uint i = ctx.anchorCount + tid; i < ctx.R; i += 32u) {
        float4 e = *(device const float4 *)&ctx.eligScr[i * 4u];
        if (e.w <= 0.0f) continue;
        float util = e.x, red = e.y, dn = e.z;
        uint age = ctx.featureAge[i];
        bool worse = (haveWorstW == 0u)
            || dn < worstDownW
            || (dn == worstDownW && util < worstUtilW)
            || (dn == worstDownW && util == worstUtilW && red > worstRedW)
            || (dn == worstDownW && util == worstUtilW && red == worstRedW && age > worstAgeW)
            || (dn == worstDownW && util == worstUtilW && red == worstRedW
                && age == worstAgeW && i < worstW);
        if (worse) {
            haveWorstW = 1u; worstW = i; worstAgeW = age;
            worstUtilW = util; worstRedW = red; worstDownW = dn;
        }
    }

    for (uint t = 1u; t < 32u; ++t) {
        uint oHave = simd_shuffle(haveWorstW, (ushort)t);
        if (oHave == 0u) continue;
        uint oWorst = simd_shuffle(worstW, (ushort)t);
        uint oAge = simd_shuffle(worstAgeW, (ushort)t);
        float oUtil = simd_shuffle(worstUtilW, (ushort)t);
        float oRed = simd_shuffle(worstRedW, (ushort)t);
        float oDown = simd_shuffle(worstDownW, (ushort)t);
        bool worse = (haveWorstW == 0u)
            || oDown < worstDownW
            || (oDown == worstDownW && oUtil < worstUtilW)
            || (oDown == worstDownW && oUtil == worstUtilW && oRed > worstRedW)
            || (oDown == worstDownW && oUtil == worstUtilW && oRed == worstRedW && oAge > worstAgeW)
            || (oDown == worstDownW && oUtil == worstUtilW && oRed == worstRedW
                && oAge == worstAgeW && oWorst < worstW);
        if (worse) {
            haveWorstW = 1u; worstW = oWorst; worstAgeW = oAge;
            worstUtilW = oUtil; worstRedW = oRed; worstDownW = oDown;
        }
    }

    for (uint k = tid; k < ctx.K; k += 32u) {
        if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;
        if (ctx.lastStopAge[k] < 0xffffffffu) ctx.lastStopAge[k] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
    uint candidateStart = ctx.anchorCount;
    uint eligible = 0u;
    float eligibleUtilSum = 0.0f, eligibleRedSum = 0.0f, eligibleDownstreamSum = 0.0f;
    for (uint i = candidateStart; i < ctx.R; ++i) {
        float4 e = *(device const float4 *)&ctx.eligScr[i * 4u];
        if (e.w > 0.0f) {
            eligible += 1u;
            eligibleUtilSum += e.x;
            eligibleRedSum += e.y;
            eligibleDownstreamSum += e.z;
        }
    }
    uint worst = (haveWorstW != 0u) ? worstW : candidateStart;
    float worstUtil = worstUtilW, worstRed = worstRedW, worstDownstream = worstDownW;

    if (!OAK_ABLATE_GENERATE_AND_TEST) {

        uint staleLimit = ctx.R * ctx.K;
        // Phase 9.3: parked options are the first victims of the staleStop
        // recycle (hibernate-don't-delete — the recycle zeroes the rows and
        // clears the parked flag below, doubling as hibernation's revive
        // path). The stale limit still gates parked victims, so parking
        // never schedules an immediate deletion.
        uint kStale = 0u;
        for (uint k2 = 1u; k2 < ctx.K; ++k2) {
            if (ctx.parked[k2] == 0u || ctx.lastStopAge[k2] < staleLimit) continue;
            kStale = k2; break;
        }
        if (kStale == 0u) {
            for (uint k2 = 1u; k2 < ctx.K; ++k2) {
                if (ctx.parked[k2] != 0u || ctx.lastStopAge[k2] < staleLimit) continue;
                kStale = k2; break;
            }
        }
        if (kStale != 0u) {
            uint k = kStale;
            ctx.optAge[k] = 0u;
            ctx.lastStopAge[k] = 0u;
            ctx.utilOpt[k] = float(ctx.R);
            ctx.actorAdv[k] = 0.0f;
            ctx.psiRho[k] = 0.0f;
            ctx.optRho[k] = 0.0f;
            ctx.parked[k] = 0u;
            uint feat = ctx.optFeature[k] % ctx.R;
            for (uint i = 0u; i < ctx.R; ++i) {
                ctx.psi[(i) * ctx.K + k] = 0.0f;
                ctx.psiAux[(i) * ctx.K + k] = 0.0f;
                ctx.optReward[(i) * ctx.K + k] = 0.0f;
                ctx.optDuration[(i) * ctx.K + k] = 0.0f;
                ctx.vStop[(i) * ctx.K + k] = 0.0f;
                ctx.vsAux[(i) * ctx.K + k] = 0.0f;
            }
            ctx.psi[(feat) * ctx.K + k] = ctx.planScr[OAK_PLANSCR_INVSQRTR];
            // Rows zeroed above -> encode's stashed prev-dots for k no longer
            // match a post-replace gather; zero the stash (finite state ->
            // the gather would produce +0.0f).
            tgStash[k * 4u + 0u] = 0.0f;
            tgStash[k * 4u + 1u] = 0.0f;
            tgStash[k * 4u + 2u] = 0.0f;
            tgStash[k * 4u + 3u] = 0.0f;
            ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
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

    bool doReplace = !OAK_ABLATE_GENERATE_AND_TEST
        && structurallyReplaceable && worst < ctx.R
        && bestN != 0xffffffffu
        && bestNUtil > worstUtil + worstDownstream;
    if (doReplace && worst < ctx.R) {
        const bool legacy = oak_legacy_geometry(ctx);
        uint pa = ctx.nurParentA[bestN] % ctx.R;
        uint pb = ctx.nurParentB[bestN] % ctx.R;
        uint kind = ctx.nurKind[bestN] % 4u;
        uint birth = ctx.featureBirthCount[worst] + 1u;
        ctx.featureBirthCount[worst] = birth;
        ctx.featureParentA[worst] = pa;
        ctx.featureParentB[worst] = pb;
        ctx.featureKind[worst] = kind;

        ctx.featureAge[worst] = ctx.nurAge[bestN];
        ctx.featureMean[worst] = ctx.nurStats[(bestN) * 4u];
        ctx.featureSM[worst] = ctx.nurStats[(bestN) * 4u + 1u];
        ctx.utilFeat[worst] = ctx.nurStats[(bestN) * 4u + 2u];
        ctx.valueW[worst] = 0.0f; ctx.bonusW[worst] = 0.0f;

        uint spa = 0u, spb = (ctx.R > 1u) ? 1u : 0u;
        float bestA = -FLT_MAX, bestB = -FLT_MAX;
        // Parents are total functions of state — the utility argmax is
        // core-only on the !legacy path (gated slots are partial: 0 whenever
        // not the bank winner), matching the host seeding invariant.
        uint pLim = legacy ? ctx.R : uint(OAK_CORE);
        for (uint i = 0u; i < pLim; ++i) {
            if (i == worst) continue;
            float uu = ctx.utilFeat[i];
            if (!isfinite(uu)) uu = -FLT_MAX;
            if (uu > bestA) { bestA = uu; spa = i; }
        }
        for (uint i = 0u; i < pLim; ++i) {
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

        float alpha0 = 1.0f / float(max(ctx.O, 1u));
        ctx.alpha[worst] = alpha0;
        ctx.beta[worst] = log(ctx.alpha[worst]); ctx.hMeta[worst] = 0.0f; ctx.gradEnergy[worst] = 0.0f;
        ctx.alphaPsi[worst] = alpha0;
        ctx.betaPsi[worst] = log(ctx.alphaPsi[worst]); ctx.hPsi[worst] = 0.0f; ctx.gradEnergyPsi[worst] = 0.0f;
        ctx.alphaModel[worst] = alpha0;
        ctx.betaModel[worst] = log(ctx.alphaModel[worst]); ctx.hModel[worst] = 0.0f; ctx.gradEnergyModel[worst] = 0.0f;

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

        ctx.dynaSlot[0] = 1u; ctx.dynaSlot[1] = worst;
        ctx.dynaSlot[2] = k; ctx.dynaSlot[3] = birth;
    } else {
        ctx.dynaSlot[0] = 0u;
    }
    if (ctx.dynaSlot[0] == 0u && !OAK_ABLATE_GENERATE_AND_TEST
        && (ctx.stepCount % max(ctx.K, 1u)) == 0u) {

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

    simdgroup_barrier(mem_flags::mem_device);
    if (ctx.dynaSlot[0] != 0u) {
        uint worst = ctx.dynaSlot[1], k = ctx.dynaSlot[2], birth = ctx.dynaSlot[3];
        float basis = 1.0f / float(max(ctx.D, 1u));
        uint DLc = oak_dl(ctx);
        uint KDL = ctx.K * DLc;

        if (tid == 0u) {
            // Option k's rows are zeroed below; keep encode's stashed
            // prev-dots in step with a post-replace gather (+0.0f).
            tgStash[k * 4u + 0u] = 0.0f;
            tgStash[k * 4u + 1u] = 0.0f;
            tgStash[k * 4u + 2u] = 0.0f;
            tgStash[k * 4u + 3u] = 0.0f;
            // Phase 9.3: a recycled option is reborn live (fresh zeroed
            // rows, age 0 probation) — clear its hibernation flag.
            ctx.parked[k] = 0u;
            // Type 1 additionally zeroes feature row `worst` of EVERY
            // option's model rows; the encode-time stash cannot be patched
            // for that (fp order), so plan re-gathers this step.
            if (ctx.dynaSlot[0] == 1u) tgDirty[0] = 1.0f;
        }

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
                float z = float(h & 0xffffu) / 32768.0f - 1.0f;

                float w0 = ctx.optPolicy[i * KDL + d];
                if (!isfinite(w0)) w0 = 0.0f;
                ctx.optPolicy[i * KDL + k * DLc + d] =
                    w0 + z * basis * rsqrt(float(OAK_ACT));
            }
        }
        simdgroup_barrier(mem_flags::mem_device);

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
            uint fi = (ctx.dynaSlot[0] == 1u) ? worst : (ctx.optFeature[k] % ctx.R);
            ctx.psi[fi * ctx.K + k] = ctx.planScr[OAK_PLANSCR_INVSQRTR];
        }
    }
}

static inline void oak_window_analytics(Ctx ctx, uint tid) {
    const uint tgW = 32u;
    uint tPrev = ctx.stepCount - 1u;
    device const float *prev = state_at(ctx, 1u - ctx.stateSlot);
    device const float *curr = state_at(ctx, ctx.stateSlot);
    device const uint *actP = act_at(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(tPrev, ctx.R);

    // Transposed K-scan (Inv-10 bit-identical load reduction): lane tid
    // holds the k = m+tid slice of the five K-indexed arrays as RAW loads;
    // each broadcast round re-evaluates the exact baseline expressions from
    // simd_shuffle'd raw values (never precomputed products — the fma
    // contraction must see the same expression tree as the baseline), in
    // ascending-k accumulation order per row. Same per-row add chains, same
    // isfinite gates; only the redundant device loads are deduplicated.
    const uint nSlot = OAK_ACT / 32u;
    uint iS[nSlot];
    float psiSigS[nSlot], modSigS[nSlot], keySigS[nSlot];
    for (uint s = 0u; s < nSlot; ++s) {
        iS[s] = actP[tid + s * 32u];
        psiSigS[s] = 0.0f; modSigS[s] = 0.0f; keySigS[s] = 0.0f;
    }
    float planTd0 = ctx.planTd[0];
    float td0 = ctx.td[0];
    for (uint m = 0u; m < ctx.K; m += 32u) {
        uint kH = m + tid;
        bool kLive = kH < ctx.K;
        uint   ofH = kLive ? ctx.optFeature[kH] : 0u;
        float tdH = kLive ? ctx.tdPsi[kH] : 0.0f;
        float kbH = kLive ? ctx.keyboard[kH] : 0.0f;
        float aaH = kLive ? ctx.actorAdv[kH] : 0.0f;
        float imH = kLive ? ctx.importance[kH] : 0.0f;
        uint kEnd = min(32u, ctx.K - m);
        for (uint l = 0u; l < kEnd; ++l) {
            uint   ofV = simd_shuffle(ofH, (ushort)l);
            float tdV = simd_shuffle(tdH, (ushort)l);
            float kbV = simd_shuffle(kbH, (ushort)l);
            float aaV = simd_shuffle(aaH, (ushort)l);
            float imV = simd_shuffle(imH, (ushort)l);
            for (uint s = 0u; s < nSlot; ++s) {
                if ((ofV % ctx.R) != iS[s]) continue;
                float psiUse = fabs(tdV);
                float keyMass = fabs(kbV);
                float modelUse = OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : keyMass * fabs(planTd0);
                float keyUse = OAK_ABLATE_SEARCH_TO_GENERATE_TEST
                    ? 0.0f : keyMass * (fabs(aaV) + max(imV, 0.0f));
                if (isfinite(psiUse)) psiSigS[s] += psiUse;
                if (isfinite(modelUse)) modSigS[s] += modelUse;
                if (isfinite(keyUse)) keySigS[s] += keyUse;
            }
        }
    }
    uint sel = ctx.selectedOption[0] % ctx.K;
    for (uint s = 0u; s < nSlot; ++s) {
        uint aIdx = tid + s * 32u;
        uint i = iS[s];
        float psiSignal = psiSigS[s], modelSignal = modSigS[s], keyboardSignal = keySigS[s];
        float planningSignal = fabs(td0) + fabs(planTd0);

        float actLoad = 0.0f;
        for (uint d = 0u; d < oak_dl(ctx); ++d)
            actLoad += fabs(ctx.optPolicy[i * ctx.K * oak_dl(ctx) + sel * oak_dl(ctx) + d]);
        float load = fabs(prev[i]) * (fabs(ctx.valueW[i]) + actLoad);
        if (!isfinite(load)) load = 0.0f;
        float g = fabs(planningSignal * prev[i])
                + fabs((psiSignal + modelSignal + keyboardSignal) * prev[i])
                + load;
        ctx.utilFeat[i] += invT * (g - ctx.utilFeat[i]);

        {
            float hw = ctx.bonusW[i];
            if (!isfinite(hw)) hw = 0.0f;
            float w = ctx.valueW[i];
            if (isfinite(w) && w > hw) ctx.bonusW[i] = w;
        }
        uint age = ctx.featureAge[i];
        metaopt_update_slot(ctx.alpha, ctx.beta, ctx.hMeta, ctx.gradEnergy,
                            i, td0, prev[i], age, ctx.metaEta, ctx.invR);
        uint metaSlot = tPrev % 32u;
        if (aIdx == metaSlot) {

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

    // Phase 9.3 K-parking (per-option hibernation, allocation design §4a):
    // the park/unpark assessment runs EVERY step (Inv-7) and commits at most
    // one park and one unpark per step (the structural single-commit rate,
    // mirroring the G&T single-reinit break — no step-count gates). The
    // threshold is the mean utility of the LIVE matured bank (optAge >= R —
    // the same maturation horizon as keyboard eligibility and nursery LCB;
    // Inv-9 structural statistic, no new constants). Park exemptions: option
    // 0 (main-task key), the selected option, and any chord/probe member
    // (keyboard mass — parking a blend contributor would silently alter the
    // emitted action). Unpark = revive, never a freeze (Mark IV law (d)):
    // selected, chord/probe mass (the vacuum probe can pick a parked option
    // on its kept stale twin-disagreement), or utility back at/above the
    // live matured mean. Rows are KEPT while parked; the staleStop recycle
    // (replace stage) prefers parked victims and clears the flag.
    if (tid == 0u) {
        float utilSum = 0.0f;
        uint maturedLive = 0u;
        for (uint k = 0u; k < ctx.K; ++k) {
            if (ctx.parked[k] != 0u || ctx.optAge[k] < ctx.R) continue;
            float u = ctx.utilOpt[k];
            utilSum += isfinite(u) ? u : 0.0f;
            maturedLive += 1u;
        }
        float meanLiveUtil = (maturedLive > 0u) ? utilSum / float(maturedLive) : 0.0f;
        uint selOpt = ctx.selectedOption[0] % ctx.K;
        for (uint k = 1u; k < ctx.K; ++k) {
            if (ctx.parked[k] == 0u) continue;
            float u = ctx.utilOpt[k];
            bool revived = (k == selOpt) || (ctx.keyboard[k] > 0.0f)
                || (isfinite(u) && u >= meanLiveUtil);
            if (revived) { ctx.parked[k] = 0u; break; }
        }
        uint parkK = 0xffffffffu;
        float parkUtil = 0.0f;
        for (uint k = 1u; k < ctx.K; ++k) {
            if (ctx.parked[k] != 0u || ctx.optAge[k] < ctx.R || k == selOpt) continue;
            float kb = ctx.keyboard[k];
            if (!isfinite(kb) || kb > 0.0f) continue;
            float u = ctx.utilOpt[k];
            if (!isfinite(u)) u = 0.0f;
            if (u < meanLiveUtil && (parkK == 0xffffffffu || u < parkUtil)) {
                parkK = k; parkUtil = u;
            }
        }
        if (parkK != 0xffffffffu) ctx.parked[parkK] = 1u;
    }

    if (!OAK_ABLATE_OPTION_POLICIES && !OAK_ABLATE_GENERATE_AND_TEST) {

        // Lane tid holds optFit[tid] as a RAW load; the full-warp sequential
        // sums below re-read it via uniform simd_shuffle in ascending-k
        // order — identical add/fma chains, one device load per lane instead
        // of one per (lane, k).
        float fH = (tid < ctx.K) ? ctx.optFit[tid] : 0.0f;
        float fitMean = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float f = simd_shuffle(fH, (ushort)k);
            fitMean += isfinite(f) ? f : 0.0f;
        }
        fitMean *= ctx.invK;
        float xNorm = ctx.xNormPrev[0];
        if (!isfinite(xNorm) || xNorm < 0.0f) xNorm = 0.0f;
        float rate = ctx.invR * ctx.invK / (1.0f + xNorm);

        const uint DLc = max(ctx.D / 2u, 1u);
        const uint KDL = ctx.K * oak_dl(ctx);
        // Consensus dir hoist: dir depends only on the action dim d, so
        // compute it once per d (lanes 0..DLc-1) instead of once per
        // (d, feature) item — identical fma order per d, broadcast by shuffle.
        // The optFit re-reads ride the same uniform fH broadcast; the
        // per-lane chain (and its optionPolicyMean loads) stays predicated
        // on tid < DLc exactly as the baseline's early-exit form.
        float dirD = 0.0f;
        float m0 = 0.0f;
        if (tid < DLc) {
            m0 = ctx.optionPolicyMean[tid];
            if (!isfinite(m0)) m0 = 0.0f;
        }
        float dir = 0.0f;
        for (uint k = 1u; k < ctx.K; ++k) {
            float f = simd_shuffle(fH, (ushort)k);
            // Phase 9.3: the consensus direction is a property of the LIVE
            // bank (parked options emit zeros and their rows are frozen).
            if (ctx.parked[k] != 0u) continue;
            if (tid < DLc) {
                float cf = (isfinite(f) ? f : 0.0f) - fitMean;
                float mk = ctx.optionPolicyMean[k * ctx.D + tid];
                dir = fma(cf, (isfinite(mk) ? mk : 0.0f) - m0, dir);
            }
        }
        if (tid < DLc)
            dirD = isfinite(dir) ? (dir / (1.0f + fabs(dir))) : 0.0f;
        for (uint witem = tid; witem < DLc * OAK_ACT; witem += tgW) {
            uint d = witem / OAK_ACT;
            uint a = witem % OAK_ACT;
            float dir = simd_shuffle(dirD, (ushort)d);

            uint i = actP[a];
            float w0 = ctx.optPolicy[i * KDL + d];
            ctx.optPolicy[i * KDL + d] = safe_w(w0, w0 + rate * dir * prev[i]);
        }
    }
    if (!OAK_ABLATE_GENERATE_AND_TEST) {

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

            device float4 *st4 = (device float4 *)&ctx.nurStats[(n) * 4u];
            float4 st = *st4;
            st.x += rate * (raw - st.x);
            st.y += rate * (raw * raw - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);

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

        for (uint i = tid; i < ctx.R; i += tgW) ctx.optUseByFeat[i] = 0.0f;
        simdgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {

            for (uint k = 0u; k < ctx.K; ++k) {
                float use = ctx.utilOpt[k]
                    + (OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(ctx.keyboard[k]));
                if (isfinite(use)) ctx.optUseByFeat[ctx.optFeature[k] % ctx.R] += use;
            }
        }

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

    if (tid == 0u && (tPrev & 3u) == 0u) {
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
    float maxPrio = 0.0f;
    float searchUse = 0.0f;
    float keyNorm = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float w = ctx.keyboard[k];
        if (isfinite(w)) keyNorm += fabs(w);
    }
    for (uint k = 0u; k < ctx.K; ++k) {
        float p = ctx.prioOpt[k];
        float w = ctx.keyboard[k];
        if (!isfinite(p)) p = 0.0f;
        if (!isfinite(w)) w = 0.0f;
        maxPrio = max(maxPrio, p);
        if (keyNorm > 0.0f) searchUse += max(p, 0.0f) * fabs(w) / keyNorm;
    }
    ctx.stats->modelPriorityMax = maxPrio;
    ctx.stats->searchControlUseMean = searchUse * invK;

    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount = float(ctx.counters[OAK_CNT_TDPSI_NF] + ctx.counters[OAK_CNT_IMPORTANCE_NF]);
    ctx.stats->oakOptionReinitCount = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

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

    uint envKind;
    uint gw[10];
    // Stream SHAPE (Inv-1): written by oak_fleet_init from OAK_ENV_SPECS; the
    // agent reads only these widths, never envKind.
    uint naturalO, naturalA;
};
static_assert(sizeof(OakFleetLane) == 312, "OakFleetLane layout drifted from the Swift mirror");

struct OakLaneSnapshot {
    KernelScalars scalars;
    KernelStats stats;
    float paramAbsMax;
    uint paramNonFinite;
    uint steps, doneCount;
    float rewardSum, rho;
    float deltaSum, deltaPeak;
    uint deathStep, deathDetector;
    float deathRewardSum, deathRho;
};
static_assert(sizeof(OakLaneSnapshot) == 152, "OakLaneSnapshot layout drifted from the Swift mirror");

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
                           device const uint *laneEnvKinds [[buffer(9)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    uint ek = laneEnvKinds[gid];
    lane->envKind = ek;
    // Sanctioned OAK_ENV_SPECS read (fleet init): deliver the lane's stream
    // shape to the agent (Inv-1 — shape, never identity).
    lane->naturalO = OAK_ENV_SPECS[ek].obsDimNatural;
    lane->naturalA = OAK_ENV_SPECS[ek].actionCount;
    HCState s;
    uint elapsed = 0u, resets = 0u;
    for (int i = 0; i < 10; i++) lane->gw[i] = 0u;
    if (ek == OAK_ENV_FOURROOMS) {

        hc_resetData(s);
        lane->gw[0] = 1u; lane->gw[1] = 1u;
        lane->gw[2] = 9u; lane->gw[3] = 9u;
        lane->gw[4] = 0u; lane->gw[5] = 0u;
    } else if (ek == OAK_ENV_DEEPSEA) {

        hc_resetData(s);
        uint4 rm = hc_philox4x32(uint2(laneSeeds[gid], HC_RNG_STREAM_RESET),
                                 uint4(0u, 0u, 0u, 0u));
        lane->gw[0] = 0u; lane->gw[1] = 0u; lane->gw[2] = 0u;
        lane->gw[3] = rm.x; lane->gw[4] = rm.y; lane->gw[5] = 0u;
    } else if (ek == OAK_ENV_KEYCORRIDOR) {

        hc_resetData(s);
        lane->gw[0] = 1u; lane->gw[1] = 1u;   // agent, left room
        lane->gw[2] = 3u; lane->gw[3] = 5u;   // key, left room
        lane->gw[4] = 7u; lane->gw[5] = 1u;   // goal, right room
    } else if (ek == OAK_ENV_MOUNTAINCAR) {

        hc_resetData(s);
        lane->gw[0] = as_type<uint>(-0.5f);   // valley
    } else if (ek == OAK_ENV_PENDULUM) {

        hc_resetData(s);
        lane->gw[0] = as_type<uint>(M_PI_F);  // hang down
    } else if (ek == OAK_ENV_LARGEDEEPSEA) {

        hc_resetData(s);
        uint4 rm0 = hc_philox4x32(uint2(laneSeeds[gid], HC_RNG_STREAM_RESET),
                                  uint4(0u, 0u, 0u, 0u));
        uint4 rm1 = hc_philox4x32(uint2(laneSeeds[gid], HC_RNG_STREAM_RESET),
                                  uint4(1u, 0u, 0u, 0u));
        lane->gw[0] = 0u;                       // r=0, c=0, lastA=0 (packed)
        lane->gw[1] = rm0.x; lane->gw[2] = rm0.y;
        lane->gw[3] = rm0.z; lane->gw[4] = rm0.w;
        lane->gw[5] = rm1.x; lane->gw[6] = rm1.y;
        lane->gw[7] = rm1.z; lane->gw[8] = rm1.w;
    } else if (ek == OAK_ENV_MULTIKEYCORRIDOR) {

        hc_resetData(s);
        lane->gw[0] = 1u; lane->gw[1] = 1u;            // agent, room 0
        lane->gw[2] = 3u | (5u << 4u);                 // key1, room 0
        lane->gw[3] = 7u | (5u << 4u);                 // key2, room 1
        lane->gw[4] = 11u | (5u << 4u);                // key3, room 2
        lane->gw[5] = 15u; lane->gw[6] = 1u;           // goal, room 3
    } else if (ek == OAK_ENV_PROCEDURALMAZE) {

        hc_resetData(s);
        uint pmw[3];
        gw_pmaze_gen(laneSeeds[gid], pmw);             // per-lane wall layout
        lane->gw[6] = pmw[0]; lane->gw[7] = pmw[1]; lane->gw[8] = pmw[2];
        lane->gw[0] = 1u; lane->gw[1] = 1u;            // agent (path start)
        lane->gw[2] = 9u; lane->gw[3] = 9u;            // goal (path end)
    } else if (ek == OAK_ENV_HIGHDIMMAZE || ek == OAK_ENV_POMAZE) {

        hc_resetData(s);
        lane->gw[0] = 1u; lane->gw[1] = 1u;            // agent, corridor end
        lane->gw[2] = 9u; lane->gw[3] = 9u;            // goal, far end
        lane->gw[6] = 1u;                              // lastDir = down (hdm)
    } else if (ek == OAK_ENV_ACROBOT || ek == OAK_ENV_CARTPOLE ||
               ek == OAK_ENV_CHAIN || ek == OAK_ENV_MEMORYCHAIN ||
               ek == OAK_ENV_DELAYEDCHAIN) {

        hc_resetData(s);   // deterministic zero state (hang / upright / pos 0)
    } else {
        hc_env_reset(s, laneSeeds[gid], elapsed, resets);
    }
    lane->s = s;
    lane->elapsed = elapsed; lane->resets = resets;
    lane->steps = 0u; lane->doneCount = 0u;
    for (int u = 0; u < 6; u++) lane->ctrlPending[u] = 0.0f;
    for (int i = 0; i < 12; i++) lane->lastDesc[i] = 0.0f;
    lane->deltaSum = 0.0f; lane->deltaC = 0.0f; lane->deltaPeak = 0.0f;
    lane->rewardSum = 0.0f; lane->rewardC = 0.0f; lane->rho = 0.0f;
    lane->deathStep = OAK_DEATH_NONE; lane->deathDetector = 0u;
    lane->deathRewardSum = 0.0f; lane->deathRho = 0.0f;
    if (ecfg.policyMode == 0u) {
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid,
                           lane->naturalO, lane->naturalA);
        if (ek == OAK_ENV_FOURROOMS) {
            float o[17];
            gw_four_obs(1u, 1u, 9u, 9u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_DEEPSEA) {
            float o[17];
            gw_deep_obs(0u, 0u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_KEYCORRIDOR) {
            float o[17];
            gw_kc_obs(1u, 1u, 3u, 5u, 7u, 1u, 0u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_ACROBOT) {
            float o[17];
            gw_acro_obs(0.0f, 0.0f, 0.0f, 0.0f, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_MOUNTAINCAR) {
            float o[17];
            gw_mcar_obs(-0.5f, 0.0f, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_CARTPOLE) {
            float o[17];
            gw_pole_obs(0.0f, 0.0f, 0.0f, 0.0f, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_PENDULUM) {
            float o[17];
            gw_pend_obs(M_PI_F, 0.0f, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_CHAIN) {
            float o[17];
            o[0] = 0.0f; o[1] = 1.0f;
            for (int i = 2; i < 17; i++) o[i] = 0.0f;
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_LARGEDEEPSEA) {
            float o[17];
            gw_ldeep_obs(0u, 0u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_MULTIKEYCORRIDOR) {
            float o[17];
            gw_mkc_obs(1u, 1u, 3u | (5u << 4u), 7u | (5u << 4u), 11u | (5u << 4u),
                       15u, 1u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_MEMORYCHAIN) {
            float o[17];
            gw_mchain_obs(0u, 0u, 0u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_PROCEDURALMAZE) {
            float o[17];
            gw_pmaze_obs(1u, 1u, 9u, 9u, 0u,
                         lane->gw[6], lane->gw[7], lane->gw[8], o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_HIGHDIMMAZE) {
            float o[17];
            gw_hdm_obs(1u, 1u, 9u, 9u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_POMAZE) {
            float o[17];
            gw_po_obs(1u, 1u, 9u, 9u, o);
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else if (ek == OAK_ENV_DELAYEDCHAIN) {
            float o[17];
            o[0] = 0.0f; o[1] = 0.0f; o[2] = 1.0f;
            for (int i = 3; i < 17; i++) o[i] = 0.0f;
            for (uint i = 0u; i < ctx.O && i < 17u; ++i) ctx.observation[i] = o[i];
        } else {
            for (uint i = 0u; i + 1u < 9u; ++i) ctx.observation[i] = s.qpos[i + 1u];
            for (uint i = 0u; i < 9u; ++i) ctx.observation[8u + i] = s.qvel[i];
        }
        ctx.reward[0] = 0.0f;
        for (uint d = 0u; d < ctx.D; ++d) ctx.actionSignal[d] = 0.0f;
        for (uint i = 0u; i < ctx.K * ctx.D; ++i) ctx.optionActorCredit[i] = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            ctx.importance[k] = 1.0f;
            ctx.impStage[k] = 1.0f;
        }

        ctx.planScr[OAK_PLANSCR_INVSQRTR] = rsqrt(float(max(ctx.R, 1u)));
        // Phase 4 planner state: planU starts at zero (first ring-write seals
        // each slot to v(x)); ringReward starts at zero (no outbound reward
        // until the first transition out of a slot).
        for (uint i = 0u; i < ctx.R; ++i) {
            ctx.planU[i] = 0.0f;
            ctx.ringReward[i] = 0.0f;
        }
        // Phase 9.1 priority index: with dynaTd all-zero every internal node's
        // winner is its range's lowest slot (ties go left); descending order
        // computes children before their parents.
        for (uint i = ctx.R - 1u; i >= 8u; --i)
            ctx.planTree[i] = oak_plan_winner(ctx, 2u * i, ctx.R);
        // Phase 9.3 K-parking: every option starts live.
        for (uint k = 0u; k < ctx.K; ++k) ctx.parked[k] = 0u;
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
    if (ecfg.policyMode != 0u && OAK_ENV_SPECS[lane->envKind].actionKind == 1u) {

        uint N = OAK_ENV_SPECS[lane->envKind].actionCount;
        uint4 rw = hc_philox4x32(uint2(laneSeeds[gid], HC_RNG_STREAM_ACT),
                                 uint4(stepIdx, 0u, 0u, 0u));
        uint a = uint(hc_u01(rw.x) * float(N));
        if (a >= N) a = N - 1u;
        for (int u = 0; u < 6; u++) ctrl[u] = 0.0f;
        ctrl[0] = float(a);
        for (int i = 0; i < 12; i++) desc[i] = 0.0f;
        if (ecfg.policyMode == 2u) desc[a] = 1.0f;
    } else if (ecfg.policyMode == 1u) {
        for (int u = 0; u < 6; u++) ctrl[u] = 0.0f;
        for (int i = 0; i < 12; i++) desc[i] = 0.0f;
    } else if (ecfg.policyMode == 2u) {
        hc_rng_ctrl(laneSeeds[gid], stepIdx, ctrl);
        for (int d = 0; d < 6; d++) { desc[d] = ctrl[d]; desc[6 + d] = 0.0f; }
    } else {
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid,
                           lane->naturalO, lane->naturalA);
        uint A = ctx.D / 2u;
        float sd = ctx.sampleSigma;
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

        float inv2s2 = ctx.sampleInv2S2;
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

static inline void oak_stage_sample_act_discrete(Ctx ctx,
                                                 device OakFleetLane *lane,
                                                 device const uint *laneSeeds,
                                                 uint gid, uint tid) {
    uint N = OAK_ENV_SPECS[lane->envKind].actionCount;
    uint stepIdx = lane->steps;
    uint4 rw = hc_philox4x32(uint2(laneSeeds[gid], HC_RNG_STREAM_ACT),
                             uint4(stepIdx, 0u, 0u, 0u));
    uint words[4] = { rw.x, rw.y, rw.z, rw.w };

    // Discrete sampler on RAW (unbounded) logits (Phase 2 — the sparse-env
    // unlock): policyMeanRaw for the behavior logits, muRaw for the target
    // option logits. Max-subtracted softmax for numerical safety (unbounded
    // logits can overflow exp; softmax is shift-invariant, argmax unchanged).
    float bMax = -FLT_MAX;
    for (uint d = 0u; d < N; ++d) bMax = max(bMax, oak_safe(ctx.policyMeanRaw[d]));
    float bSum = 0.0f;
    for (uint d = 0u; d < N; ++d) bSum += exp(oak_safe(ctx.policyMeanRaw[d]) - bMax);
    uint act = 0u;
    float bestG = -FLT_MAX;
    for (uint d = 0u; d < N; ++d) {
        float mu = oak_safe(ctx.policyMeanRaw[d]);

        float u = min(hc_u01open(words[d]), 0x1.fffffep-1f);
        float g = mu - log(-log(u));
        if (g > bestG) { bestG = g; act = d; }
    }

    simdgroup_barrier(mem_flags::mem_device);
    if (tid < 6u) lane->ctrlPending[tid] = (tid == 0u) ? float(act) : 0.0f;
    if (tid < ctx.D) {
        float s = 0.0f;
        if (tid < N)
            s = ((tid == act) ? 1.0f : 0.0f) - exp(oak_safe(ctx.policyMeanRaw[tid]) - bMax) / bSum;
        ctx.actionSignal[tid] = s;
    }
    if (tid == 0u) {
        float desc[12];
        for (uint i = 0u; i < ctx.D; ++i) desc[i] = oak_safe(ctx.policyMeanRaw[i]);
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
        float oMax = -FLT_MAX;
        for (uint d = 0u; d < N; ++d) oMax = max(oMax, oak_safe(ctx.muRaw[off + d]));
        float oSum = 0.0f;
        for (uint d = 0u; d < N; ++d)
            oSum += exp(oak_safe(ctx.muRaw[off + d]) - oMax);
        for (uint d = 0u; d < ctx.D; ++d) {
            float cr = 0.0f;
            if (d < N)
                cr = ((d == act) ? 1.0f : 0.0f)
                   - exp(oak_safe(ctx.muRaw[off + d]) - oMax) / oSum;
            ctx.creditStage[off + d] = cr;
        }

        float pbA = exp(oak_safe(ctx.policyMeanRaw[act]) - bMax) / bSum;
        float poA = exp(oak_safe(ctx.muRaw[off + act]) - oMax) / oSum;
        float ratio = poA / pbA;
        ctx.impStage[tid] = isfinite(ratio) ? min(ratio, float(ctx.K)) : 0.0f;
    }
}

static inline void oak_stage_sample_act_wide(Ctx ctx,
                                             device OakFleetLane *lanes,
                                             device const uint *laneSeeds,
                                             uint gid, uint tid) {
    device OakFleetLane *lane = &lanes[gid];
    if (OAK_ENV_SPECS[lane->envKind].actionKind == 1u) {
        oak_stage_sample_act_discrete(ctx, lane, laneSeeds, gid, tid);
        return;
    }
    device float *scratch = ctx.sampleScr;
    uint A = OAK_ENV_SPECS[lane->envKind].actionCount;
    float sd = ctx.sampleSigma;
    uint stepIdx = lane->steps;

    if (tid < A) {
        uint d = tid;
        uint2 key = uint2(laneSeeds[gid], HC_RNG_STREAM_ACT);
        uint pair = d >> 1u;
        uint4 r = hc_philox4x32(key, uint4(stepIdx, pair >> 1u, 0u, 0u));

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

            ctx.creditStage[off + d] = (sample - om) * (1.0f - om * om);
            ctx.creditStage[off + A + d] = 0.0f;
            float diffB = sample - oak_safe(ctx.policyMean[d]);
            float diffO = sample - om;
            acc += diffB * diffB - diffO * diffO;
        }

        float ratio = exp(acc * ctx.sampleInv2S2);
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
    float ctrl[6];
    for (int u = 0; u < 6; u++) ctrl[u] = lane->ctrlPending[u];
    float reward;
    uint done = 0u;
    float obs[17];
    float pose[9];
    if (lane->envKind == OAK_ENV_FOURROOMS) {

        uint gws[6];
        for (int i = 0; i < 6; i++) gws[i] = lane->gw[i];
        reward = gw_four_step(gws, laneSeeds[gid], uint(ctrl[0]), obs);
        for (int i = 0; i < 6; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[2]); pose[3] = float(lane->gw[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_DEEPSEA) {
        uint gws[6];
        for (int i = 0; i < 6; i++) gws[i] = lane->gw[i];
        reward = gw_deep_step(gws, uint(ctrl[0]), obs);
        for (int i = 0; i < 6; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = 7.0f; pose[3] = 7.0f;
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_KEYCORRIDOR) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_kc_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[4]); pose[3] = float(lane->gw[5]);
        pose[4] = float(lane->gw[2]); pose[5] = float(lane->gw[3]);
        pose[6] = float(lane->gw[6] & 1u); pose[7] = float((lane->gw[6] >> 1u) & 1u);
        pose[8] = 0.0f;
    } else if (lane->envKind == OAK_ENV_CHAIN) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_chain_step(gws, uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = 15.0f;
        for (int i = 2; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_LARGEDEEPSEA) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        reward = gw_ldeep_step(gws, uint(ctrl[0]), obs);
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0] & 15u); pose[1] = float((lane->gw[0] >> 4u) & 15u);
        pose[2] = 15.0f; pose[3] = 15.0f;
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_MULTIKEYCORRIDOR) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_mkc_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[5]); pose[3] = float(lane->gw[6]);
        pose[4] = float(lane->gw[7] & 7u);
        pose[5] = float((lane->gw[7] >> 3u) & 7u);
        for (int i = 6; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_MEMORYCHAIN) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_mchain_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = 31.0f;
        pose[2] = float(lane->gw[1]);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_PROCEDURALMAZE) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_pmaze_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[2]); pose[3] = float(lane->gw[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_HIGHDIMMAZE) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_hdm_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[2]); pose[3] = float(lane->gw[3]);
        pose[4] = float(lane->gw[6]);
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_POMAZE) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_po_step(gws, laneSeeds[gid], uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = float(lane->gw[1]);
        pose[2] = float(lane->gw[2]); pose[3] = float(lane->gw[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_DELAYEDCHAIN) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_dchain_step(gws, uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = float(lane->gw[0]); pose[1] = 31.0f;
        pose[2] = float(lane->gw[1]);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_ACROBOT) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_acro_step(gws, uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        float th1 = as_type<float>(lane->gw[0]), th2 = as_type<float>(lane->gw[1]);
        pose[0] = th1; pose[1] = th2;
        pose[2] = as_type<float>(lane->gw[2]); pose[3] = as_type<float>(lane->gw[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_MOUNTAINCAR) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_mcar_step(gws, uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = as_type<float>(lane->gw[0]); pose[1] = as_type<float>(lane->gw[1]);
        pose[2] = 0.5f;
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_CARTPOLE) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        uint dn = 0u;
        reward = gw_pole_step(gws, uint(ctrl[0]), obs, &dn);
        done = dn;
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        pose[0] = as_type<float>(lane->gw[0]); pose[1] = as_type<float>(lane->gw[1]);
        pose[2] = as_type<float>(lane->gw[2]); pose[3] = as_type<float>(lane->gw[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (lane->envKind == OAK_ENV_PENDULUM) {

        uint gws[10];
        for (int i = 0; i < 10; i++) gws[i] = lane->gw[i];
        reward = gw_pend_step(gws, ctrl[0], obs);
        for (int i = 0; i < 10; i++) lane->gw[i] = gws[i];
        float th = as_type<float>(lane->gw[0]), thd = as_type<float>(lane->gw[1]);
        pose[0] = th; pose[1] = thd; pose[2] = clamp(2.0f * ctrl[0], -2.0f, 2.0f);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else {
        HCState s = lane->s;
        uint elapsed = lane->elapsed, resets = lane->resets;
        if (ecfg.episodeLen != 0u && elapsed >= ecfg.episodeLen) {
            hc_env_reset(s, laneSeeds[gid], elapsed, resets);
            reward = 0.0f;
            done = 0u;
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
        }
        for (int i = 1; i < 9; i++) obs[i - 1] = s.qpos[i];
        for (int i = 0; i < 9; i++) obs[8 + i] = s.qvel[i];
        lane->s = s;
        lane->elapsed = elapsed;
        lane->resets = resets;
        for (int i = 0; i < 9; i++) pose[i] = s.qpos[i];
    }
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
        Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, gid,
                           lane->naturalO, lane->naturalA);
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
    if (ecfg.recordCapacity != 0u) {
        int slot = recordSlot[gid];
        if (slot >= 0) {

            ulong base = ((ulong)slot * ecfg.recordCapacity
                          + ((stepNum - 1u) % ecfg.recordCapacity)) * 16ul;
            for (int u = 0; u < 6; u++) recordBuf[base + u] = ctrl[u];
            recordBuf[base + 6] = float(done);
            for (int i = 0; i < 9; i++) recordBuf[base + 7 + i] = pose[i];
        }
    }
}

static inline void oak_lane_fence() {
    simdgroup_barrier(mem_flags::mem_device);
}

static inline void oak_dyna_backups(Ctx ctx, uint tid) {
    // TRUST-SWEEP budget (Phase 4): P=8 backups/step (the dossier's structural
    // budget) — threads 0..7 each sweep one ring partition; the full-bank
    // 32-thread sweep was ~3x SPS cost (262K fma/step). Threads 8..31 skip the
    // sweep but still run the copy-2 twin update below.
    if (OAK_ABLATE_PLANNING) return;
    uint M = ctx.R;
    uint written = min(ctx.stepCount, M);
    uint bestSlot = 0xffffffffu;
    float d = 0.0f;
    if (tid < 8u) {
        uint part = M / 8u;
        uint lo = tid * part;

    // Phase 9.1: the per-partition argmax is one priority-index read. The
    // tree winner is provably the scan's leftmost-max over the eligible
    // slots: unwritten slots hold zero and lose every tie to the partition's
    // lowest slot, which is eligible whenever any slot is; the guard covers
    // the scan's all-ineligible -> no-selection case.
    if (lo < written) bestSlot = ctx.planTree[8u + tid];

    if (bestSlot != 0xffffffffu) {
        device const uint *ri = ctx.ringIdx + bestSlot * 32u;
        device const float *rv = ctx.ring + bestSlot * 64u;
        float vX = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint i = (a < OAK_CORE) ? a : ri[a - OAK_CORE];
            vX = fma(ctx.valueW[i], rv[a], vX);
        }

        // TRUST-SWEEP backup (Phase 4, the arbitrary-horizon organ). The
        // planner owns planU; nothing here writes valueW/psi/rho/policies.
        // Two backup kinds: a BUDGETED masked option-model VI (4 structurally
        // named candidates — the full-bank sweep made the idle warps the new
        // bottleneck) and the trajectory relay (model-free, HVH-safe, the
        // PRIMARY credit mechanism — Holland 2018: value is in long credit
        // chains, not 1-step fan-out). Horizon is not a parameter.
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
        float modelU = -FLT_MAX;
        for (uint c = 0u; c < 4u; ++c) {
            uint k = cand[c];
            if (ctx.parked[k] != 0u) continue;  // Phase 9.3: frozen rows, stale model dots
            float s1 = 0.0f, s2 = 0.0f, l1 = 0.0f;
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint i = (a < OAK_CORE) ? a : ri[a - OAK_CORE];
                float xv = rv[a];
                uint idx = i * ctx.K + k;
                l1 = fma(ctx.optDuration[idx], xv, l1);
                s1 = fma(ctx.vStop[idx], xv, s1);
                s2 = fma(ctx.vStop2[idx], xv, s2);
            }
            // Twin MEAN tail (never twin-min — TD3 target-side pessimism).
            // optReward is REMOVED from the VI backup: its unanchored bootstrap
            // explodes over long horizons (11 deaths @200k). The option value
            // here is the twin-mean stop value plus the centering de-bias
            // (optRho_k - rho-bar)·l-hat (Wan-Naik-Sutton 2110.13855); the
            // reward-to-go comes from the trajectory relay (the PRIMARY credit
            // mechanism, Holland 2018), not a bootstrap reward model.
            float tail = (k == 0u) ? vX : 0.5f * (s1 + s2);
            float dur = decode_duration(l1);
            float rDeb = (k == 0u) ? 0.0f : (ctx.optRho[k] - ctx.scalars->rho) * dur;
            if (!isfinite(rDeb)) rDeb = 0.0f;
            float dis = (k == 0u) ? 0.0f : fabs(s1 - s2);
            // Twin-disagreement trust gate (structural epsilon): skip options
            // whose twins disagree past the gate — model USE, never a target.
            if (isfinite(dis) && dis <= 1.0f && isfinite(rDeb + tail) && (rDeb + tail) > modelU)
                modelU = rDeb + tail;
        }
        // Trajectory relay (model-free, HVH-safe): real reward + next slot's U.
        float relayU = -FLT_MAX;
        uint nextSlot = (bestSlot + 1u) % M;
        if (bestSlot != (ctx.stepCount % M) && nextSlot < written) {
            float rOut = ctx.ringReward[bestSlot];
            if (isfinite(rOut)) relayU = rOut - ctx.scalars->rho + ctx.planU[nextSlot];
        }
        // Fused target: prefer the model when trusted, else the relay.
        float target = (modelU > -FLT_MAX) ? modelU : relayU;
        if (target > -FLT_MAX && isfinite(target)) {
            float oldU = ctx.planU[bestSlot];
            d = target - oldU;
            if (isfinite(d)) {
                ctx.planU[bestSlot] = oldU + d;
                // Priority: remaining surprise at this slot (Moore-Atkeson).
                ctx.dynaTd[bestSlot] = fabs(d);
                oak_plan_index_update(ctx, bestSlot);
            } else {
                d = 0.0f;
            }
        }
    }
    }  // end if (tid < 8u) budgeted sweep
    ctx.dynaD[tid] = d;
    ctx.dynaSlot[tid] = bestSlot;
    simdgroup_barrier(mem_flags::mem_device);
    // Poison deleted (Phase 0): the dynaAcc accumulation into valueW is gone,
    // so the accumulator fill is dead work — removed. The dynaAcc arena is
    // reserved for the Phase-4 planner-owned U table.

    {
        uint k = tid;
        device const float *prevS = state_at(ctx, 1u - ctx.stateSlot);
        device const uint *actP = act_at(ctx, 1u - ctx.stateSlot);
        if (ctx.parked[k] != 0u) {
            // Phase 9.3: parked twin row frozen — neutral update coefficient,
            // no dots; the apply below skips every parked column.
            ctx.dynaC[k] = 0.0f;
        } else {
        device const float *currS = state_at(ctx, ctx.stateSlot);
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
        // vStop2 at 0.1 rate — it lacks vStop's GTD2 anchor (vStop uses the
        // aux-corrected gv gradient; vStop2 uses the raw dynaC form) and was
        // exploding over long horizons (7 deaths @100k, 5 @200k, 10 @400k).
        ctx.dynaC[k] = imp * vsTd * scS * 0.1f;
        }
        simdgroup_barrier(mem_flags::mem_device);
        for (uint a = tid; a < OAK_ACT; a += 32u) {
            uint i = act_of(actP, a);
            float aM = ctx.alphaModel[i] * prevS[i];
            uint row = i * ctx.K;
            for (uint kk = 0u; kk < ctx.K; ++kk) {
                if (ctx.parked[kk] != 0u) continue;  // Phase 9.3: frozen row
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
                         threadgroup float *tgCont [[threadgroup(0)]],
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
    // tg diet: prev/curr active rows are gathered from device state at the
    // point of use (same values, L2-resident 1KB row). Threadgroup memory
    // holds, per lane: the K-wide continuation vector, then a 4K prev-dot
    // stash written by encode and consumed gather-free by oak_plan_one.
    uint tgK = is_function_constant_defined(OAK_SPEC_OPTION_COUNT)
        ? OAK_SPEC_OPTION_COUNT : cfg.optionCount;
    threadgroup float *lCont = tgCont + laneSg * tgK;
    threadgroup float *lStash = tgCont + OAK_MEGA_G * tgK + laneSg * 4u * tgK;
    threadgroup float *lDirty = tgCont + 5u * OAK_MEGA_G * tgK + laneSg;
    for (uint it = 0u; it < stepsThisDispatch; ++it) {

        if (ecfg.policyMode == 0u) {
            if (wideActive && (OAK_MEGA_STAGES & 1u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane,
                                   lanes[wideLane].naturalO, lanes[wideLane].naturalA);
                oak_stage_encode(ctx, slane, 32u, lCont, lStash);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 8u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane,
                                   lanes[wideLane].naturalO, lanes[wideLane].naturalA);
                oak_replace_commit(ctx, slane, lStash, lDirty);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 16u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane,
                                   lanes[wideLane].naturalO, lanes[wideLane].naturalA);
                emit_all_options(ctx, slane);
                simdgroup_barrier(mem_flags::mem_device);
                emit_blend_grid(ctx, slane);
            }
            oak_lane_fence();
            if (wideActive && (OAK_MEGA_STAGES & 32u) != 0u) {
                Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wideLane,
                                   lanes[wideLane].naturalO, lanes[wideLane].naturalA);
                oak_stage_sample_act_wide(ctx, lanes, laneSeeds, wideLane, slane);
            }
        } else if (packedActive && (OAK_MEGA_STAGES & 32u) != 0u) {
            oak_stage_sample_act(params, stateArena, uintArena, scalars, stats, cfg,
                                 lanes, laneSeeds, ecfg, packedLane);
        }
        // mem_threadgroup ordering added for the prev-dot stash (encode /
        // replace write it, plan reads it on a different simdgroup).
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (packedActive && sgrp == 0u)
                oak_stage_fleet_env(params, stateArena, uintArena, scalars, stats, cfg,
                                    lanes, laneSeeds, ecfg, recordSlot, recordBuf, packedLane);
        }
        if (ecfg.policyMode == 0u && sgrp > 0u && lanesThisTg > 0u) {
            uint idx = sgrp - 1u;
            if (lanesThisTg > OAK_MEGA_G - 1u) {

                uint T = scalars[laneBase].stepCount;
                uint skip = T % lanesThisTg;
                if (idx >= skip) idx += 1u;
            }
            if (idx < lanesThisTg) {
                uint wLane = laneBase + idx;
                Ctx pctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg, wLane,
                                    lanes[wLane].naturalO, lanes[wLane].naturalA);
                uint pSlot = 1u - pctx.stateSlot;
                threadgroup const float *pStash = tgCont + OAK_MEGA_G * tgK + idx * 4u * tgK;
                bool stashDirty = tgCont[5u * OAK_MEGA_G * tgK + idx] != 0.0f;
                if ((OAK_MEGA_STAGES & 2u) != 0u)
                    oak_stage_plan_all(pctx, slane, pStash, stashDirty, pSlot);
                // No fence plan->learn: they communicate only through
                // column-k slots (vsTd/vsDot/vsScale/vsAuxScale/optRewardTd/
                // prioOpt/polCoef) produced and consumed by the same thread
                // k = tid, and shared arrays are learn-write/plan-read-only
                // in the same column. Everything wider is behind the step's
                // threadgroup_barrier.
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
                               constant uint &fullScan [[buffer(11)]],
                               uint slane [[thread_index_in_simdgroup]],
                               uint sgid [[threadgroup_position_in_grid]]) {

    if (sgid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[sgid];
    device OakLaneSnapshot *o = &snap[snapBase + sgid];
    if (ecfg.policyMode == 0u) {
        if (slane == 0u) {
            o->scalars = scalars[sgid];
            o->stats = stats[sgid];
        }
        // The full-parameter scan is O(paramStride) reads per lane — heartbeat
        // cadence only. Per-CB tape snapshots skip it (paramAbsMax = -1
        // sentinel: "not scanned"); the per-step rotating-stripe explosion
        // detector covers deaths continuously between heartbeats.
        if (fullScan != 0u) {
            device const float *lp = params + sgid * cfg.paramStride;

            uint firstNF = 0xffffffffu;
            for (uint i = slane; i < cfg.paramStride; i += 32u) {
                if (!isfinite(lp[i])) { firstNF = i; break; }
            }
            firstNF = simd_min(firstNF);
            uint lim = min(firstNF, cfg.paramStride);
            float mx = 0.0f;
            for (uint i = slane; i < lim; i += 32u) mx = max(mx, fabs(lp[i]));
            mx = simd_max(mx);
            if (slane == 0u) {
                o->paramAbsMax = mx;
                o->paramNonFinite = (firstNF == 0xffffffffu) ? 0u : 1u;
            }
        } else if (slane == 0u) {
            o->paramAbsMax = -1.0f;
            o->paramNonFinite = 0u;
        }
    } else {
        if (slane == 0u) {
            o->scalars = KernelScalars{0u, 0u, 0.0f};
            KernelStats zs;
            thread float *zp = (thread float *)&zs;
            for (uint i = 0u; i < sizeof(KernelStats) / sizeof(float); ++i) zp[i] = 0.0f;
            o->stats = zs;
            o->paramAbsMax = 0.0f;
            o->paramNonFinite = 0u;
        }
    }
    if (slane != 0u) return;
    o->steps = lane->steps;
    o->doneCount = lane->doneCount;
    o->rewardSum = lane->rewardSum;
    o->rho = lane->rho;
    o->deltaSum = lane->deltaSum;
    o->deltaPeak = lane->deltaPeak;
    o->deathStep = lane->deathStep;
    o->deathDetector = lane->deathDetector;
    o->deathRewardSum = lane->deathRewardSum;
    o->deathRho = lane->deathRho;
}
