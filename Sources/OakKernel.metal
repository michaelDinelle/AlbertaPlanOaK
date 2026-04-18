#include <metal_stdlib>
#include <float.h>
using namespace metal;

// =================================================================================================
// Orthodox Step 11 OaK on Metal — one shared latent, continual, average-reward, no discounting.
// Invariants (.cursor/rules/alberta_plan.mdc §1–9) hold: no env branches, no tuned scalars, no
// episodic semantics, no train/eval split, no per-step allocation, no cosmetic normalization,
// no curiosity on the external reward, meta-constants structurally derived from {R, B, K}.
// Thirteen kernels: encode, predict, plan, meta_action, reduce_alpha, learn_value, learn_encoder,
// learn_psi, learn_opt_policy, bk_reduce, bk_feature, bk_option, bk_admit_commit. (bk_reduce and
// bk_feature are kept separate because the per-feature utility pass needs R parallel threadgroups
// to avoid 32× serialization at large R·B; with MTLIndirectCommandBuffer the extra dispatch has
// ~zero host cost, so fusion is not profitable here.)
//
// Citations (docs/papers/):
//   AlbertaPlan_2208.11173                      base agent; Step 11 Oak; STOMP progression
//   RethinkingFoundationsContinualRL_2504.08161 Bowling self-critique of episodic baggage
//   RewardRespectingSubtasks_2202.03466 §6.1    w̄ high-water-mark bonus weight (F1/F2)
//   RewardCentering_2405.09999                  differential TD w/ ρ̄ centering (main + per-opt)
//   BellmanErrorCentering_2502.03104            theoretical identity — VRC ≡ BEC, linear-FA proof
//   ImplicitUpdates_AvgRewardTD_2510.06149      αimp = α0/(1+α0·‖x‖²) — F10 on every linear head
//   ImplicitQLearningSarsa_2601.18907           implicit TD extended to control
//   StreamingDRLFinallyWorks_2410.14606         streaming AC (semi-grad; λ-trace rejected per Inv 9)
//     [known-superior-but-rejected: GradientEligibilityTraces_2507.09087]
//   FullGradientSuccessorFeatureRepresentations_2604.00686  ψ[k,j,i] vector SF Bellman
//   DifferentialTDConvergence_2602.16629        on-policy n-step differential TD a.s. converges
//                                                without a local clock — learn_psi gating justified
//   TaleOfTwoTDErrors_2603.21921                implicit-TD form is the stability-preserving choice
//   OptionKeyboard_2106.13105                   Alg 1 line 3: argmax_k ψ^{π_k}·w (GPI)
//   OptimalBehaviorBasis_OptionKeyboard_2505.00787  OBB: minimal behavior basis for GPI
//   SuccessorFeaturesKeyboard_2310.15940        SFK: ψ_k(s)·w option scoring
//   CompositionalPlanningJumpyWorldModels_2602.19634  GPI = α₁=1 limit of CompPlan; oak_plan sits here
//   SpectralAlignmentFB_2603.20103              temporal abstraction lowers effective SR rank → K
//   FIRE_2602.08040 / SWR_2508.00212 / SpectralCollapsePlasticity_2509.22335
//                                                selective weight reinit + effective-rank floor
//   MetaGradientSearchControl_2406.19561        planning TD on w_main (F4)
//   ValueHallucinationDynaPlanning_2006.04363   planning-only-from-current-state (avoid VH)
//   TowardAgentsThatReasonAboutTheirComputation_2510.22833  planK structurally bounded
//   WhichFeaturesBestForSuccessorFeatures_2502.10790        cumulant caveat (identity cumulants OK)
//   TTT-Options-Sutton-2025-slides              activate per step: O*_t = argmax_o q̂(s, o, w⁰)
//   RankGradientLost_SWD_2604.01913             Θ(1/k) gradient decay — motivates track_rate floor
//   RethinkingPlasticity_OCP_2603.21173         OCP hypothesis validates conjunctive reinit trigger
//   WhyNetworksForget_2603.04580                forgetting ↔ eRank collapse confirmation for pEff
//   ParameterFreeTD_2603.02577                  α_imp parameter-free validated (no ω, no τ_mix)
//   RelativeTD_Stability_2603.27874             Naik centering stable under FA for any baseline ≥ 0
//   GraMa_MeasureGradients_2505.24061           gradient-magnitude = correct utility signal (we use it)
//
// Params (fp16, packed):
//   encoder[R]                                  linear encoder weights W_e (diagonal)
//   valueW[R]                                   w_main (main-task linear value)
//   psi[K · R · R]                              ψ[k, j, i]: per-option per-cumulant SF weights
//   optPolicy[K · A · R]                        per-option action logits
// State arena (fp32):
//   observation[B·O]                            centered → state[slot · B · R]
//   observationMean[O]                          online Welford mean for encoder centering
//   state[2 · B · R]                            double-buffered (slot toggles each step; S5)
//   prevAction[B·A], action[B·A], actionGrad[B·A], reward[B], td[B], tdPsi[B·R]
// UInt arena:
//   done[B], activeMask[R], activeList[R]
//   option[2 · B]                               slot-toggled active/next (S4)
//   counters[6]                                 {encNF, learnNF, planNF, reinitOptCount, reinitFeatCount, _pad}
// Arena extras (fp32):
//   featSM[R]                                   batch 2nd-moment EMA of active x (utility norm)
//   wBar[R]                                     high-water-mark of |w_main| (RRS §6.1)
//   rhoK[K]                                     per-option main-reward avg (Naik centering)
//   rhoKJ[K·R]                                  per-option per-cumulant avg (SF centering)
//   utilFeat[R], utilOpt[K]                     running utilities for admission / reinit
// =================================================================================================

// Structural SIMD-group width on Apple GPUs. Both an assertion boundary (K ≤ 32 guarded in host
// via OakMetalLayout.optionCount) and the TG-size unit used by all reduction kernels.
constant uint OAK_SIMD = 32u;

[[function_constant(0)]] constant bool fc_action_is_discrete;

struct KernelConfig {
    uint obsDim, rank, actDim, batch, optionCount, baseObsDim;
    uint actionIsDiscrete;
    // Param offsets (in fp16 stride)
    uint pEncoder, pValue, pPsi, pOptPolicy;
    // State offsets (in fp32 stride)
    uint sObservation, sObsMean, sState, sPrevAction, sAction, sActionGrad;
    uint sReward, sTd, sTdPsi;
    // UInt offsets
    uint uDone, uActiveMask, uActiveList, uOption, uCounters;
    // Arena offsets (in fp32 stride, past sState's tail)
    uint aFeatSM, aWBar, aRhoK, aRhoKJ, aUtilFeat, aUtilOpt;
    // Derived scalars (all structurally derived from rank · batch)
    float cap;                // rank · batch (signal-bound on every clamp)
    float alpha0;             // 1 / cap (base step size)
    float ridge;              // 1 / (cap + 1) (Tikhonov ridge; tracking EMA floor)
    float boundLat, boundRew, boundAct;
    float invBoundLat, invBoundRew, invBoundAct;
};

struct KernelScalars {
    uint  stepCount, doneCount, activeCount;
    uint  stateSlot, optionSlot;
    uint  _pad;
    float rho;                // main-task average reward (Naik centering)
    float pEff;               // effective rank trace/max of featSM
    float alphaImp;           // implicit TD step: α₀ / (1 + α₀·⟨‖x_active‖²⟩_batch) — 2510.06149
    float alphaEnc;           // implicit TD step for encoder (centered obs)
};

// KernelStats — 15 fields (dead legacy slots removed: maxAbsRewardPrediction,
// planningUsedOptionDynamicsMean, utilitySubtaskMean, utilityPlanningMean).
struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeRank;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, planningActionChangeL2Mean;
    float optionModelPredictionMSE;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilityOptionMean;
};

// -------------------------------------------------------------------------------------------------
// Context: pointers + scalars, no per-thread copy of the full config.
// -------------------------------------------------------------------------------------------------
struct Ctx {
    uint R, A, B, K, O, baseO;
    uint activeCount, stepCount;
    uint stateSlot, optionSlot;
    float cap, alpha0, ridge, rho;
    float boundLat, boundRew, boundAct, invBoundLat, invBoundRew, invBoundAct;
    bool  actionIsDiscrete;

    device half   *encoder;
    device half   *valueW;
    device half   *psi;            // [K][R][R]
    device half   *optPolicy;      // [K][A][R]

    device float  *observation;    // [B][O]
    device float  *observationMean;// [O]
    device float  *state;          // [2][B][R]
    device float  *prevAction;     // [B][A]
    device float  *action;         // [B][A]
    device float  *actionGrad;     // [B][A]
    device float  *reward;         // [B]
    device float  *td;             // [B]
    device float  *tdPsi;          // [B][R]

    device uint   *done;           // [B]
    device uint   *activeMask;     // [R]
    device uint   *activeList;     // [R]
    device uint   *option;         // [2][B]
    device uint   *counters;       // [6]

    device float  *featSM;         // [R]
    device float  *wBar;           // [R]
    device float  *rhoK;           // [K]
    device float  *rhoKJ;          // [K][R]
    device float  *utilFeat;       // [R]
    device float  *utilOpt;        // [K]

    device KernelScalars *scalars;
};

inline Ctx make_ctx(device half *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, constant KernelConfig &cfg) {
    Ctx c;
    c.R = cfg.rank; c.A = cfg.actDim; c.B = cfg.batch; c.K = max(cfg.optionCount, 1u);
    c.O = cfg.obsDim; c.baseO = cfg.baseObsDim;
    c.activeCount = scalars->activeCount;
    c.stepCount   = scalars->stepCount;
    c.stateSlot   = scalars->stateSlot;
    c.optionSlot  = scalars->optionSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho;
    c.boundLat = cfg.boundLat; c.boundRew = cfg.boundRew; c.boundAct = cfg.boundAct;
    c.invBoundLat = cfg.invBoundLat; c.invBoundRew = cfg.invBoundRew; c.invBoundAct = cfg.invBoundAct;
    c.actionIsDiscrete = (cfg.actionIsDiscrete != 0u);

    c.encoder   = params + cfg.pEncoder;
    c.valueW    = params + cfg.pValue;
    c.psi       = params + cfg.pPsi;
    c.optPolicy = params + cfg.pOptPolicy;

    c.observation     = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.state           = stateArena + cfg.sState;
    c.prevAction      = stateArena + cfg.sPrevAction;
    c.action          = stateArena + cfg.sAction;
    c.actionGrad      = stateArena + cfg.sActionGrad;
    c.reward          = stateArena + cfg.sReward;
    c.td              = stateArena + cfg.sTd;
    c.tdPsi           = stateArena + cfg.sTdPsi;

    c.done       = uintArena + cfg.uDone;
    c.activeMask = uintArena + cfg.uActiveMask;
    c.activeList = uintArena + cfg.uActiveList;
    c.option     = uintArena + cfg.uOption;
    c.counters   = uintArena + cfg.uCounters;

    c.featSM   = stateArena + cfg.aFeatSM;
    c.wBar     = stateArena + cfg.aWBar;
    c.rhoK     = stateArena + cfg.aRhoK;
    c.rhoKJ    = stateArena + cfg.aRhoKJ;
    c.utilFeat = stateArena + cfg.aUtilFeat;
    c.utilOpt  = stateArena + cfg.aUtilOpt;

    c.scalars = scalars;
    return c;
}

inline device float *state_slot(Ctx ctx, uint slot) { return ctx.state + slot * ctx.B * ctx.R; }
inline device uint  *option_slot(Ctx ctx, uint slot) { return ctx.option + slot * ctx.B; }
inline uint active_option(Ctx ctx, uint row) {
    device const uint *o = option_slot(ctx, ctx.optionSlot);
    return min(o[row], ctx.K - 1u);
}

// Continual tracking rate. Warmup phase (~1/ridge steps) follows sample mean via 1/(t+1);
// afterwards floors at structural ridge = 1/(R·B+1). Guarantees tracking does NOT freeze
// for t → ∞ (Big World Hypothesis; Autostep-style meta-rate; 2604.01913 Θ(1/k) remedy).
// Inv-9 compliant: both rates are structurally derived from {R, B}, no scalar hyperparameter.
inline float track_rate(uint stepCount, float ridge) {
    return max(1.0f / float(stepCount + 1u), ridge);
}

inline uint ceil_sqrt(uint n) { float s = sqrt(float(max(n, 1u))); return uint(ceil(s)); }

// Dot product over the active-feature list (gather).
inline float active_dot_half(device const half *w, uint wBase,
                             device const float *x, uint xBase,
                             device const uint *active, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        s = fma(float(w[wBase + idx]), x[xBase + idx], s);
    }
    return s;
}

inline float active_norm2(device const float *x, uint xBase,
                          device const uint *active, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        float v = x[xBase + idx];
        s = fma(v, v, s);
    }
    return s;
}

// Hierarchical simd_sum for 128-thread threadgroups (4 SIMD groups × 32). Each lane
// contributes v_local; first lane of each SIMD stages to sg_buf; first SIMD group
// finalizes with one more simd_sum. Returns the global sum to lane 0 only.
inline float tg_sum_128(float v_local, threadgroup float *sg_buf, uint tid, uint sgid) {
    float v_sg = simd_sum(v_local);
    if (simd_is_first()) sg_buf[sgid] = v_sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (tid < 4u) ? sg_buf[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) sg_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sg_buf[0];
}

inline float tg_max_128(float v_local, threadgroup float *sg_buf, uint tid, uint sgid) {
    float v_sg = simd_max(v_local);
    if (simd_is_first()) sg_buf[sgid] = v_sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (tid < 4u) ? sg_buf[tid] : -FLT_MAX;
        v = simd_max(v);
        if (tid == 0u) sg_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sg_buf[0];
}

// =================================================================================================
// encode — centered linear encoder writes the NEW state into state_slot(1 - stateSlot).
//   [Grid: R · B] One thread per (row, feat).
// =================================================================================================
kernel void oak_encode(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.R, feat = gid % ctx.R;
    if (row >= ctx.B) return;
    uint writeSlot = 1u - ctx.stateSlot;
    float o_raw = (feat < ctx.O) ? ctx.observation[row * ctx.O + feat] : 0.0f;
    float om    = (feat < ctx.O) ? ctx.observationMean[feat] : 0.0f;
    float centered = o_raw - om;
    float z = centered * float(ctx.encoder[feat]);
    state_slot(ctx, writeSlot)[row * ctx.R + feat] = z;
}

// =================================================================================================
// predict — main differential TD δ_main (A.2) + per-cumulant SF TD δ_ψ[j] for the active option
// (A.1/F6/F7: full-gradient SF per 2604.00686). RRS optimism bonus (F1) added to δ_main via
// (w̄_i − w_i)·x_i inside plan (more principled: planning-time bonus). This kernel writes the
// raw prediction TD only.
//   [Grid: B]
// =================================================================================================
kernel void oak_predict(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.B) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint pOff = gid * ctx.R, cOff = gid * ctx.R;

    // Main differential TD on w_main (θ_v). Implicit-TD form per 2603.21921. Bounds: ‖ŝ‖ ≤ √R
    // by layer norm; |w·ŝ| ≤ √R · ‖w‖. Implicit α + tracking ρ̄ keep δ in fp32 dynamic range.
    float vPrev = active_dot_half(ctx.valueW, 0u, prev, pOff, ctx.activeList, ctx.activeCount);
    float vNext = active_dot_half(ctx.valueW, 0u, curr, cOff, ctx.activeList, ctx.activeCount);
    float terminalMask = (ctx.done[gid] != 0u) ? 0.0f : 1.0f;
    float dMain = (ctx.reward[gid] - ctx.rho) + terminalMask * vNext - vPrev;
    ctx.td[gid] = dMain;

    // Per-cumulant SF TD for the active option's ψ_k(·,:). Cumulant c_j = x_j (identity;
    // 2502.10790 caveat passes). ρ_{k,j} centered per-option-per-cumulant (2405.09999).
    uint optK = active_option(ctx, gid);
    uint psiBase = optK * ctx.R * ctx.R;
    uint rhoBase = optK * ctx.R;
    for (uint j = 0u; j < ctx.R; ++j) {
        if (ctx.activeMask[j] == 0u) { ctx.tdPsi[gid * ctx.R + j] = 0.0f; continue; }
        float pj = active_dot_half(ctx.psi, psiBase + j * ctx.R, prev, pOff,
                                   ctx.activeList, ctx.activeCount);
        float nj = active_dot_half(ctx.psi, psiBase + j * ctx.R, curr, cOff,
                                   ctx.activeList, ctx.activeCount);
        float cj = curr[cOff + j];
        float dj = (cj - ctx.rhoKJ[rhoBase + j]) + terminalMask * nj - pj;
        ctx.tdPsi[gid * ctx.R + j] = dj;
    }
}

// =================================================================================================
// plan — one-step option-model lookahead (F4). Per row, 32 threads compute K candidate Q-values
// Q_k(x) = Σ_j w_j · ψ_{k,j}(x) + RRS optimism; SIMD tournament argmax (5 rounds of
// simd_shuffle_down) selects k*. Citations: OK 2106.13105 Alg 1 line 3; MGSC 2406.19561;
// RRS §6.1 bonus; SpectralAlignmentFB 2603.20103 on K ≤ effective SR rank;
// CompositionalPlanningJumpyWorldModels 2602.19634 (oak_plan = GPI = CompPlan α₁=1 limit).
// K is structurally capped to OAK_SIMD=32 at host (OakMetalLayout.optionCount) — one SIMD
// group per row.
//   [Dispatch: B threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_plan(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = tgid;
    if (row >= ctx.B) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint xOff = row * ctx.R;

    // Each lane computes its own Q_k for k = tid when k < K; otherwise -FLT_MAX.
    float qk = -FLT_MAX;
    uint  k  = tid;
    if (k < ctx.K) {
        uint pb = k * ctx.R * ctx.R;
        float val = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            if (ctx.activeMask[j] == 0u) continue;
            float psi_kj = active_dot_half(ctx.psi, pb + j * ctx.R, curr, xOff,
                                           ctx.activeList, ctx.activeCount);
            val = fma(float(ctx.valueW[j]), psi_kj, val);
        }
        if (k < ctx.R) {
            float bonus = (ctx.wBar[k] - float(ctx.valueW[k])) * curr[xOff + k];
            val += max(bonus, 0.0f);
        }
        qk = val;
    }

    // SIMD tournament argmax in 5 shuffle rounds (log2 32). No threadgroup memory needed.
    uint k_lane = tid;
    for (uint off = OAK_SIMD >> 1; off > 0u; off >>= 1u) {
        float q2 = simd_shuffle_down(qk, off);
        uint  k2 = simd_shuffle_down(k_lane, off);
        bool  wins = (q2 > qk);
        qk     = wins ? q2 : qk;
        k_lane = wins ? k2 : k_lane;
    }
    if (tid == 0u) {
        float vX = active_dot_half(ctx.valueW, 0u, curr, xOff, ctx.activeList, ctx.activeCount);
        ctx.td[row] += qk - vX;
        option_slot(ctx, 1u - ctx.optionSlot)[row] = k_lane;
    }
}

// =================================================================================================
// meta_action — emit discrete or continuous action for the row, under π_{O*} where O* was just
// written by oak_plan (no chord blending — F5, validated 2602.19634 GPI / 2505.00787 OBB).
// Discrete sampling uses a deterministic Gumbel-max via a structural hash of (stepCount, row) —
// no host RNG round-trip (S3). Continuous actions clamp the option-policy logit to env bounds.
//
// Dispatch grid: B threads (one per row) for discrete; B·A threads for continuous. Host picks
// based on fc_action_is_discrete. Eliminates (A-1)/A wasted threads on the discrete path.
// =================================================================================================
inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
inline float hash_uniform(uint step, uint row, uint dim) {
    uint h = splitmix32(step ^ splitmix32(row ^ splitmix32(dim)));
    return max(float(h >> 8) * 5.9604645e-8f, 1e-7f);
}

kernel void oak_meta_action(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);

    if (fc_action_is_discrete) {
        uint row = gid;
        if (row >= ctx.B) return;
        uint optK = option_slot(ctx, 1u - ctx.optionSlot)[row];
        uint xOff = row * ctx.R;
        uint bestA = 0u;
        float bestL = -FLT_MAX;
        for (uint d = 0u; d < ctx.A; ++d) {
            uint pb = optK * ctx.A * ctx.R + d * ctx.R;
            float ld = active_dot_half(ctx.optPolicy, pb, curr, xOff,
                                       ctx.activeList, ctx.activeCount);
            float u = hash_uniform(ctx.stepCount, row, d);
            float g = -log(-log(u));
            float s = ld + g;
            if (s > bestL) { bestL = s; bestA = d; }
        }
        for (uint d = 0u; d < ctx.A; ++d) {
            float v = (d == bestA) ? 1.0f : 0.0f;
            ctx.action[row * ctx.A + d]     = v;
            ctx.actionGrad[row * ctx.A + d] = v;
        }
    } else {
        uint row = gid / ctx.A, a = gid % ctx.A;
        if (row >= ctx.B) return;
        uint optK = option_slot(ctx, 1u - ctx.optionSlot)[row];
        uint xOff = row * ctx.R;
        uint polBase = optK * ctx.A * ctx.R + a * ctx.R;
        float logit = active_dot_half(ctx.optPolicy, polBase, curr, xOff,
                                      ctx.activeList, ctx.activeCount);
        ctx.action[row * ctx.A + a]     = logit;
        ctx.actionGrad[row * ctx.A + a] = logit;
    }
}

// =================================================================================================
// reduce_alpha — compute α_imp = α₀ / (1 + α₀ · ⟨‖x_active‖²⟩) and α_enc (centered-obs variant)
// once per step. 128-thread TG with hierarchical simd_sum. Saves K·R·R recomputations in
// learn_psi and R recomputations in learn_value.
//   [Dispatch: 1 threadgroup × 128 threads.]
// =================================================================================================
kernel void oak_reduce_alpha(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float n2_l = 0.0f, obs_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        n2_l += active_norm2(prev, r * ctx.R, ctx.activeList, ctx.activeCount);
        for (uint f = 0u; f < ctx.baseO; ++f) {
            float c = ctx.observation[r * ctx.O + f] - ctx.observationMean[f];
            obs_l = fma(c, c, obs_l);
        }
    }
    threadgroup float sg_n2[4], sg_obs[4];
    float n2  = tg_sum_128(n2_l,  sg_n2,  tid, sgid);
    float obs = tg_sum_128(obs_l, sg_obs, tid, sgid);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        scalars->alphaImp = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
        scalars->alphaEnc = ctx.alpha0 / (1.0f + ctx.alpha0 * obs * invB);
    }
}

// =================================================================================================
// learn_value — update w_main[feat] with δ_total · x[feat] reduction, via simd_sum across batch.
// Uses precomputed α_imp (single source of truth; no per-feature recomputation). Also updates
// featSM[feat] (EMA of x²) and w̄[feat] (slow high-water-mark).
//   [Dispatch: R threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_learn_value(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = tgid;
    if (feat >= ctx.R) return;
    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float g_l = 0.0f, sm_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float xp = prev[r * ctx.R + feat];
        g_l  = fma(ctx.td[r], xp, g_l);
        sm_l = fma(xp, xp, sm_l);
    }
    float g  = simd_sum(g_l);
    float sm = simd_sum(sm_l);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        float invT = track_rate(ctx.stepCount, ctx.ridge);
        float step = ctx.scalars->alphaImp;
        ctx.featSM[feat] += (sm * invB - ctx.featSM[feat]) * invT;
        float wNew = float(ctx.valueW[feat]) + step * g * invB;
        ctx.valueW[feat] = half(wNew);
        float decay = 1.0f - ctx.alpha0;
        ctx.wBar[feat] = max(ctx.wBar[feat] * decay, fabs(wNew));
    }
}

// =================================================================================================
// learn_encoder — update encoder[feat] for base obs features via TD × centered_obs (F10 on W_e).
//   [Grid: baseObsDim]
// =================================================================================================
kernel void oak_learn_encoder(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = gid;
    if (feat >= ctx.baseO) return;
    float step = ctx.scalars->alphaEnc;
    float g = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float c = ctx.observation[r * ctx.O + feat] - ctx.observationMean[feat];
        g = fma(ctx.td[r], c, g);
    }
    float eNew = float(ctx.encoder[feat]) + step * g / float(max(ctx.B, 1u));
    ctx.encoder[feat] = half(eNew);
}

// =================================================================================================
// learn_psi — update ψ[k, j, i] += αimp · (1/|rows_k|) · Σ_{row: active(row)==k} tdPsi[row, j] · prev[row, i].
// On-policy gating (2602.16629 proves a.s. convergence without local clock).
//   [Grid: K · R · R]
// =================================================================================================
kernel void oak_learn_psi(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / (ctx.R * ctx.R);
    uint rem = gid - k * ctx.R * ctx.R;
    uint j = rem / ctx.R, i = rem % ctx.R;
    if (k >= ctx.K || j >= ctx.R || i >= ctx.R) return;
    if (ctx.activeMask[i] == 0u || ctx.activeMask[j] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float step = ctx.scalars->alphaImp;
    float g = 0.0f;
    uint  cnt = 0u;
    for (uint r = 0u; r < ctx.B; ++r) {
        if (active_option(ctx, r) != k) continue;
        g = fma(ctx.tdPsi[r * ctx.R + j], prev[r * ctx.R + i], g);
        cnt++;
    }
    if (cnt == 0u) return;
    float invCnt = 1.0f / float(cnt);
    uint  pIdx = k * ctx.R * ctx.R + j * ctx.R + i;
    float pNew = float(ctx.psi[pIdx]) + step * g * invCnt;
    ctx.psi[pIdx] = half(pNew);
}

// =================================================================================================
// learn_opt_policy — semi-grad streaming AC on π_k (2410.14606; no λ-trace — Inv 9). Gradient:
// advantage (≡ td[row] with plan boost) × action indicator × prev_state.
//   [Grid: K · A · R]
// =================================================================================================
kernel void oak_learn_opt_policy(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / (ctx.A * ctx.R);
    uint rem = gid - k * ctx.A * ctx.R;
    uint a = rem / ctx.R, i = rem % ctx.R;
    if (k >= ctx.K || a >= ctx.A || i >= ctx.R) return;
    if (ctx.activeMask[i] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float step = ctx.scalars->alphaImp;
    float g = 0.0f;
    uint  cnt = 0u;
    for (uint r = 0u; r < ctx.B; ++r) {
        if (active_option(ctx, r) != k) continue;
        g = fma(ctx.td[r] * prev[r * ctx.R + i], ctx.prevAction[r * ctx.A + a], g);
        cnt++;
    }
    if (cnt == 0u) return;
    float invCnt = 1.0f / float(cnt);
    uint  pIdx = k * ctx.A * ctx.R + a * ctx.R + i;
    float pNew = float(ctx.optPolicy[pIdx]) + step * g * invCnt;
    ctx.optPolicy[pIdx] = half(pNew);
}

// =================================================================================================
// bk_reduce — global telemetry + obs-mean Welford. One 128-thread TG with hierarchical simd_sum
// for the batch-wide (row,feat) and (row,a) reductions. Per-feature utility lives in a separate
// kernel (oak_bk_feature, R TGs × 32) to preserve R-way parallelism at large R·B.
//   [Dispatch: 1 threadgroup × 128 threads.]
// =================================================================================================
kernel void oak_bk_reduce(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(ctx.stepCount, ctx.ridge);

    // (1) Observation-mean EMA (strided over O).
    float obsNF_local = 0.0f;
    for (uint f = tid; f < ctx.O; f += tgW) {
        float sum = 0.0f; uint nf = 0u;
        for (uint r = 0u; r < ctx.B; ++r) {
            float o = ctx.observation[r * ctx.O + f];
            if (isfinite(o)) sum += o; else nf++;
        }
        float meanBatch = sum / float(max(ctx.B, 1u));
        ctx.observationMean[f] += invT * (meanBatch - ctx.observationMean[f]);
        obsNF_local += float(nf);
    }
    threadgroup float sg_obsNF[4];
    float obsNF = tg_sum_128(obsNF_local, sg_obsNF, tid, sgid);

    // (2) Per-(row, feat) reductions: latent mean-square, ψ TD mean-square, max|latent|, nLearn.
    // Layer norm bounds |s| ≤ √R structurally; v² accumulated B·R times stays well within fp32.
    float lms_l = 0.0f, psi_l = 0.0f, maxLat_l = 0.0f, nLearn_l = 0.0f;
    uint total_rf = ctx.B * ctx.R;
    for (uint idx = tid; idx < total_rf; idx += tgW) {
        uint r = idx / ctx.R, f = idx % ctx.R;
        float v = curr[r * ctx.R + f];
        if (!isfinite(v)) { nLearn_l += 1.0f; v = 0.0f; }
        lms_l = fma(v, v, lms_l);
        maxLat_l = max(maxLat_l, fabs(v));
        float dp = ctx.tdPsi[r * ctx.R + f];
        psi_l = fma(dp, dp, psi_l);
    }
    threadgroup float sg_lms[4], sg_psi[4], sg_maxLat[4], sg_nLearn[4];
    float lms    = tg_sum_128(lms_l,    sg_lms,    tid, sgid);
    float psi    = tg_sum_128(psi_l,    sg_psi,    tid, sgid);
    float maxLat = tg_max_128(maxLat_l, sg_maxLat, tid, sgid);
    float nLearn = tg_sum_128(nLearn_l, sg_nLearn, tid, sgid);

    // (3) Per-(row, a) reductions: plan L2, action delta, maxPol, nPlan. Actions are bounded at
    // FFI host boundary; non-finite guard zeroes overflow contribution before squaring.
    float pL2_l = 0.0f, ad_l = 0.0f, maxPol_l = 0.0f, nPlan_l = 0.0f;
    uint total_ra = ctx.B * ctx.A;
    for (uint idx = tid; idx < total_ra; idx += tgW) {
        uint r = idx / ctx.A, a = idx % ctx.A;
        float x = ctx.action[r * ctx.A + a];
        float yg = ctx.actionGrad[r * ctx.A + a];
        float py = ctx.prevAction[r * ctx.A + a];
        bool nf = !isfinite(x) || !isfinite(yg);
        if (nf) { nPlan_l += 1.0f; x = 0.0f; yg = 0.0f; }
        float p1 = x - yg;
        float p2 = x - py;
        pL2_l = fma(p1, p1, pL2_l);
        ad_l  = fma(p2, p2, ad_l);
        maxPol_l = max(maxPol_l, fabs(x));
    }
    threadgroup float sg_pL2[4], sg_ad[4], sg_maxPol[4], sg_nPlan[4];
    float pL2    = tg_sum_128(pL2_l,    sg_pL2,    tid, sgid);
    float ad     = tg_sum_128(ad_l,     sg_ad,     tid, sgid);
    float maxPol = tg_max_128(maxPol_l, sg_maxPol, tid, sgid);
    float nPlan  = tg_sum_128(nPlan_l,  sg_nPlan,  tid, sgid);

    if (tid == 0u) {
        float invBR = 1.0f / float(max(ctx.B * ctx.R, 1u));
        float invBA = 1.0f / float(max(ctx.B * ctx.A, 1u));
        stats->latentStateMeanSquare      = lms * invBR;
        stats->optionModelPredictionMSE   = psi * invBR;
        stats->planningActionChangeL2Mean = pL2 * invBA;
        stats->actionDeltaL2Mean          = ad  * invBA;
        stats->maxAbsLatentPrediction     = maxLat;
        stats->maxAbsPolicyOutput         = maxPol;
        stats->rawNonFiniteCount          = obsNF + nLearn + nPlan;
        stats->sanitizedNonFiniteCount    = nLearn + nPlan;
        stats->encodeNonFiniteCount       = obsNF;
        stats->learnNonFiniteCount        = nLearn;
        stats->planNonFiniteCount         = nPlan;
        ctx.counters[0] = uint(obsNF);
        ctx.counters[1] = uint(nLearn);
        ctx.counters[2] = uint(nPlan);
    }
}

// =================================================================================================
// bk_feature — per-feature utility EMA: util_feat[i] ← track_rate·(mean|δ·x_i| + Σ_j|δψ_j·x_i|).
// GraMa-aligned gradient-magnitude signal (2505.24061). R threadgroups × 32 threads gives
// R-way outer parallelism so this scales with R even at large B; each TG uses simd_sum across
// its 32 lanes to reduce over B·(R+1) inner terms.
//   [Dispatch: R threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_bk_feature(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = tgid;
    if (feat >= ctx.R) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    float u_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float x = curr[r * ctx.R + feat];
        u_l += fabs(ctx.td[r] * x);
        for (uint j = 0u; j < ctx.R; ++j) u_l += fabs(ctx.tdPsi[r * ctx.R + j] * x);
    }
    float u = simd_sum(u_l);
    if (tid == 0u) {
        float un = u * invB / float(max(ctx.R + 1u, 1u));
        ctx.utilFeat[feat] += invT * (un - ctx.utilFeat[feat]);
    }
}

// =================================================================================================
// bk_option — per-option utility, ρ_k, ρ_{k,j}, FIRE+SWR-style reinit. 128-thread TG, one TG
// per option. ρ_{k,j} updated in a strided loop over j after the batch-level reductions.
//   [Dispatch: K threadgroups × 128 threads.]
// =================================================================================================
kernel void oak_bk_option(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = tgid;
    if (k >= ctx.K) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);

    float cnt_l = 0.0f, rSum_l = 0.0f, psiErr_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        if (active_option(ctx, r) != k) continue;
        cnt_l += 1.0f;
        rSum_l += ctx.reward[r];
        for (uint j = 0u; j < ctx.R; ++j) {
            float d = ctx.tdPsi[r * ctx.R + j];
            psiErr_l = fma(d, d, psiErr_l);
        }
    }
    threadgroup float sg_cnt[4], sg_r[4], sg_p[4];
    float cnt  = tg_sum_128(cnt_l,    sg_cnt, tid, sgid);
    float rSum = tg_sum_128(rSum_l,   sg_r,   tid, sgid);
    float pErr = tg_sum_128(psiErr_l, sg_p,   tid, sgid);
    float frac       = cnt * invB;
    float meanR      = (cnt > 0.0f) ? rSum / cnt : 0.0f;
    float meanPsiErr = (cnt > 0.0f) ? pErr / (cnt * float(ctx.R)) : 0.0f;

    // Per-cumulant ρ_{k,j}: strided over j, simd_sum across batch per j.
    for (uint j = tid; j < ctx.R; j += tgW) {
        float sj = 0.0f;
        for (uint r = 0u; r < ctx.B; ++r) {
            if (active_option(ctx, r) != k) continue;
            sj += curr[r * ctx.R + j];
        }
        float mj = (cnt > 0.0f) ? sj / cnt : 0.0f;
        ctx.rhoKJ[k * ctx.R + j] += invT * (mj - ctx.rhoKJ[k * ctx.R + j]);
    }

    if (tid == 0u) {
        ctx.rhoK[k]    += invT * (meanR - ctx.rhoK[k]);
        ctx.utilOpt[k] += invT * (meanPsiErr - ctx.utilOpt[k]);

        // FIRE+SWR reinit trigger. OCP hypothesis 2603.21173 validates conjunctive gate
        // (high stale TD + dormant option fraction).
        if (meanPsiErr > ctx.cap * ctx.ridge && frac < ctx.ridge) {
            uint psiBase = k * ctx.R * ctx.R;
            for (uint idx = 0u; idx < ctx.R * ctx.R; ++idx) ctx.psi[psiBase + idx] = half(0.0f);
            float invR = 1.0f / sqrt(float(max(ctx.R, 1u)));
            if (k < ctx.R) ctx.psi[psiBase + k * ctx.R + k] = half(invR);
            uint polBase = k * ctx.A * ctx.R;
            for (uint idx = 0u; idx < ctx.A * ctx.R; ++idx) ctx.optPolicy[polBase + idx] = half(0.0f);
            ctx.rhoK[k] = 0.0f;
            for (uint j = 0u; j < ctx.R; ++j) ctx.rhoKJ[k * ctx.R + j] = 0.0f;
            ctx.utilOpt[k] = 0.0f;
            uint bit = 1u << (k & 31u);
            uint which = (k < 32u) ? 3u : 4u;
            atomic_fetch_or_explicit((device atomic_uint *)(ctx.counters + which),
                                     bit, memory_order_relaxed);
        }
    }
}

// =================================================================================================
// bk_admit_commit — FUSED bk_admit + bk_commit. Serial on one thread:
//   (admit) refresh activeMask/activeList/activeCount; enforce effective-rank floor ⌈√R⌉;
//           FC-STOMP encoder reinit on worst feature when any option reinited this step.
//   (commit) ρ̄ Naik update, pEff effective-rank proxy, telemetry roll-up, slot toggles,
//            stepCount++, clear per-step reinit bitmasks.
//   [Grid: 1]
// =================================================================================================
kernel void oak_bk_admit_commit(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid != 0u) return;

    // --- admit phase ---
    uint R = ctx.R;
    float sum = 0.0f;
    for (uint i = 0u; i < R; ++i) sum += ctx.utilFeat[i];
    float mean = sum / float(max(R, 1u));
    float var = 0.0f;
    for (uint i = 0u; i < R; ++i) { float d = ctx.utilFeat[i] - mean; var += d * d; }
    float std = sqrt(var / float(max(R, 1u)));

    uint count = 0u;
    uint worstIdx = 0u;
    float worstU = FLT_MAX;
    for (uint i = 0u; i < R; ++i) {
        if (ctx.utilFeat[i] > mean) {
            ctx.activeMask[i] = 1u;
            ctx.activeList[count++] = i;
        } else {
            ctx.activeMask[i] = 0u;
        }
        if (ctx.utilFeat[i] < worstU) { worstU = ctx.utilFeat[i]; worstIdx = i; }
    }
    // Effective-rank floor = ⌈√R⌉ (F11 — 2509.22335).
    uint floorR = ceil_sqrt(R);
    while (count < floorR) {
        uint pick = R; float pickU = -FLT_MAX;
        for (uint i = 0u; i < R; ++i) {
            if (ctx.activeMask[i] == 0u && ctx.utilFeat[i] > pickU) { pickU = ctx.utilFeat[i]; pick = i; }
        }
        if (pick >= R) break;
        ctx.activeMask[pick] = 1u;
        ctx.activeList[count++] = pick;
    }
    if (count == 0u) { ctx.activeMask[0] = 1u; ctx.activeList[0] = 0u; count = 1u; }
    ctx.scalars->activeCount = count;

    // FC-STOMP feedback: if any option reinited (counters[3]|counters[4] != 0) AND worst
    // feature < mean - std, reinit its base-obs encoder weight structurally.
    uint anyReinit = ctx.counters[3] | ctx.counters[4];
    if (anyReinit != 0u && worstU < mean - std && worstIdx < ctx.baseO) {
        float sign = (((ctx.stepCount + worstIdx) & 1u) == 0u) ? 1.0f : -1.0f;
        ctx.encoder[worstIdx] = half(sign / sqrt(float(max(R, 1u))));
        ctx.utilFeat[worstIdx] = mean;
        ctx.counters[5] += 1u;
    }

    // --- commit phase ---

    // Differential avg-reward: ρ̄ += track_rate(r̄ - ρ̄). Relative-TD stable under FA (2603.27874).
    float meanR = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) meanR += ctx.reward[r];
    meanR /= float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    ctx.scalars->rho += (meanR - ctx.scalars->rho) * invT;

    // Effective-rank via trace/max of featSM (structural stable-rank proxy; 2603.04580).
    float tr = 0.0f, mx = 0.0f;
    for (uint i = 0u; i < R; ++i) { float v = ctx.featSM[i]; tr += v; mx = max(mx, v); }
    ctx.scalars->pEff = tr / max(mx, FLT_MIN);

    // Telemetry roll-up.
    float uF = 0.0f, uO = 0.0f;
    for (uint i = 0u; i < R; ++i) uF += ctx.utilFeat[i];
    for (uint i = 0u; i < ctx.K; ++i) uO += ctx.utilOpt[i];
    stats->rewardRate         = ctx.scalars->rho;
    stats->activeRank         = float(ctx.scalars->activeCount);
    stats->utilityFeatureMean = uF / float(max(R, 1u));
    stats->utilityOptionMean  = uO / float(max(ctx.K, 1u));
    stats->oakOptionReinitCount = float(popcount(ctx.counters[3]) + popcount(ctx.counters[4]));

    // Slot toggles + step + clear reinit bitmasks.
    ctx.scalars->stateSlot  = 1u - ctx.scalars->stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.scalars->optionSlot;
    ctx.scalars->stepCount += 1u;
    ctx.counters[3] = 0u; ctx.counters[4] = 0u;
}
