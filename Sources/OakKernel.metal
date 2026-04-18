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
//   psi[K · M]                                  W_ψ[k] · concat[s, λ]: shared chord-conditional
//                                               SF (Design 2; OBB §2.4 Eq.4). M = R + K.
//   optPolicy[K · 2·A · R]                      per-option Gaussian action params [μ_d, log_σ_d]
//                                               for d ∈ [0, A). Sampled in oak_meta_action with
//                                               canonical Gaussian score function ∇log π written
//                                               to scoreGrad. Pass 10: ALWAYS Gaussian — no
//                                               discrete branch in agent (env-wrapper interprets).
// State arena (fp32):
//   observation[B·O]                            centered → state[slot · B · R]
//   observationMean[O]                          online Welford mean for encoder centering
//   state[2 · B · R]                            double-buffered (slot toggles each step; S5)
//   prevAction[B·A]                             previous step's action (sent to env at step N-1)
//   action[B·A]                                 current step's sampled real-valued action vector
//   scoreGrad[B · 2·A]                          Gaussian score [(z/σ)_d, ((z²−1)/2)_d] for each d
//   reward[B], td[B]
//   tdPsi[B · K]                                K cumulant TDs per row (Design 2 collapse)
// UInt arena:
//   done[B], activeMask[R], activeList[R]
//   option[2 · B]                               slot-toggled active/next option index (S4)
//   counters[6]                                 {encNF, learnNF, planNF, reinitOptCount, reinitFeatCount, _pad}
// Arena extras (fp32):
//   featSM[R]                                   batch 2nd-moment EMA of active x (utility norm)
//   wBar[R]                                     high-water-mark of |w_main| (RRS §6.1)
//   featMean[R]                                 per-feature running mean (cumulant centering ρ_k)
//   utilFeat[R], utilOpt[K]                     running utilities for admission / reinit
// =================================================================================================

// Structural SIMD-group width on Apple GPUs. Both an assertion boundary (K ≤ 32 guarded in host
// via OakMetalLayout.optionCount) and the TG-size unit used by all reduction kernels.
constant uint OAK_SIMD = 32u;

// Named offsets into counters[6]. Self-documenting replacement for hardcoded indices 0..5.
// All telemetry counters live here; layout is fixed in OakMetalLayout.uintLayout.
constant uint OAK_CNT_OBS_NF        = 0u;  // bk_reduce: non-finite obs count
constant uint OAK_CNT_LEARN_NF      = 1u;  // bk_reduce: non-finite latent / TD-target count
constant uint OAK_CNT_PLAN_NF       = 2u;  // bk_reduce: non-finite action / scoreGrad count
constant uint OAK_CNT_REINIT_OPT    = 3u;  // bk_option: bitmask of options reinited this step
                                            //   (K ≤ OAK_SIMD = 32, so one uint suffices)
constant uint OAK_CNT_REINIT_FEAT   = 4u;  // bk_admit_commit: encoder-feature reinit count
                                            //   (single counter, atomic increment)
// Index 5 is reserved (was second half of option-reinit bitmask for K > 32; dead under
// the K ≤ OAK_SIMD constraint).

// No function constants. Pass 10 unification: the agent has zero a-priori knowledge of action-
// space type. optPolicy is always K · 2·A · R (Gaussian μ_d, log_σ_d per action dim); the env-
// wrapper layer (EnvPoolEnv.step in Main.swift) is the SOLE site that interprets the agent's
// continuous-vector output (argmax for discrete envs, bound-clamp for continuous envs).
// Inv-1: no env-conditioned branches in agent; Inv-3: no env-type-conditioned widths.

struct KernelConfig {
    uint obsDim, rank, actDim, batch, optionCount, baseObsDim;
    // Param offsets (in fp16 stride). ψ is laid out as [K rows × M cols] where M = R + K
    // (Alberta Plan Design 2: shared chord-conditional ψ_k(s, λ) = W_ψ[k] · concat[s, λ]).
    uint pEncoder, pValue, pPsi, pOptPolicy;
    // State offsets (in fp32 stride). scoreGrad[B · D_params] holds ∇log π(a|s) for the
    // sampled action — written by oak_meta_action, consumed by oak_learn_pi at the *next*
    // step (when the action's TD error δ becomes available). vCurr[B] caches w_main · curr
    // computed once in oak_predict and read by oak_plan (Tier 11 fused dot).
    uint sObservation, sObsMean, sState, sPrevAction, sAction, sScoreGrad;
    uint sReward, sTd, sTdPsi, sVCurr;
    // UInt offsets. subtaskFeatureOf[K] = state-feature index that cumulant k tracks
    // (top-K by per-feature utility, re-ranked by oak_bk_admit_commit each step; Tier 7 RED).
    uint uDone, uActiveMask, uActiveList, uOption, uCounters, uSubtaskFeatureOf;
    // Arena offsets (in fp32 stride, past sState's tail). featMean[R] is the running EMA of
    // current-step feature values used as cumulant centering ρ_k = featMean[sf[k]] (RED).
    uint aFeatSM, aWBar, aFeatMean, aUtilFeat, aUtilOpt;
    // Derived scalars (all structurally derived from rank · batch).
    float cap;                // rank · batch (capacity volume; ridge / α₀ derived from it)
    float alpha0;             // 1 / cap (base step size)
    float ridge;              // 1 / (cap + 1) (Tikhonov ridge; tracking EMA floor)
};

struct KernelScalars {
    uint  stepCount, activeCount;
    uint  stateSlot, optionSlot;
    float rho;                // main-task average reward (Naik centering)
    float pEff;               // effective rank trace/max of featSM
    float alphaImp;           // implicit TD step: α₀ / (1 + α₀·⟨‖x_active‖²⟩_batch) — 2510.06149
    float alphaEnc;           // implicit TD step for encoder (centered obs)
};

// KernelStats — 16 Float fields. Layout must match Swift KernelStats byte-for-byte.
struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeRank;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, scoreGradL2Mean;
    float optionModelPredictionMSE;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilityOptionMean;
};

// -------------------------------------------------------------------------------------------------
// Context: pointers + scalars, no per-thread copy of the full config.
// M = R + K is the concat dim for ψ's chord-conditional input.
// -------------------------------------------------------------------------------------------------
struct Ctx {
    uint R, A, B, K, O, baseO, M, D;
    uint activeCount, stepCount;
    uint stateSlot, optionSlot;
    float cap, alpha0, ridge, rho;

    device half   *encoder;
    device half   *valueW;
    device half   *psi;            // [K][M] : W_ψ[k] = (state weights, chord weights)
    device half   *optPolicy;      // [K][D][R] : per-option action params (μ_d, log_σ_d, or logit)

    device float  *observation;    // [B][O]
    device float  *observationMean;// [O]
    device float  *state;          // [2][B][R]
    device float  *prevAction;     // [B][A] : action sent to env on previous step
    device float  *action;         // [B][A] : action being sent to env this step (sampled)
    device float  *scoreGrad;      // [B][D] : ∇log π(a|s) score function for sampled action
    device float  *reward;         // [B]
    device float  *td;             // [B]
    device float  *tdPsi;          // [B][K]  -- one cumulant TD per row per option
    device float  *vCurr;          // [B]     -- w_main · curr_state, cached by predict, read by plan

    device uint   *done;           // [B]
    device uint   *activeMask;     // [R]
    device uint   *activeList;     // [R]
    device uint   *option;         // [2][B]
    device uint   *counters;       // [6]
    device uint   *sf;             // [K] subtaskFeatureOf — dynamic top-K feature ranking

    device float  *featSM;         // [R]
    device float  *wBar;           // [R]
    device float  *featMean;       // [R]  -- per-feature running mean (cumulant centering ρ_k)
    device float  *utilFeat;       // [R]
    device float  *utilOpt;        // [K]

    device KernelScalars *scalars;
};

inline Ctx make_ctx(device half *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, constant KernelConfig &cfg) {
    Ctx c;
    c.R = cfg.rank; c.A = cfg.actDim; c.B = cfg.batch; c.K = max(cfg.optionCount, 1u);
    c.O = cfg.obsDim; c.baseO = cfg.baseObsDim;
    c.M = c.R + c.K;
    c.D = 2u * c.A;  // structural: Gaussian μ + log_σ per action dim (Pass 10 unification)
    c.activeCount = scalars->activeCount;
    c.stepCount   = scalars->stepCount;
    c.stateSlot   = scalars->stateSlot;
    c.optionSlot  = scalars->optionSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho;

    c.encoder   = params + cfg.pEncoder;
    c.valueW    = params + cfg.pValue;
    c.psi       = params + cfg.pPsi;
    c.optPolicy = params + cfg.pOptPolicy;

    c.observation     = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.state           = stateArena + cfg.sState;
    c.prevAction      = stateArena + cfg.sPrevAction;
    c.action          = stateArena + cfg.sAction;
    c.scoreGrad       = stateArena + cfg.sScoreGrad;
    c.reward          = stateArena + cfg.sReward;
    c.td              = stateArena + cfg.sTd;
    c.tdPsi           = stateArena + cfg.sTdPsi;
    c.vCurr           = stateArena + cfg.sVCurr;

    c.done       = uintArena + cfg.uDone;
    c.activeMask = uintArena + cfg.uActiveMask;
    c.activeList = uintArena + cfg.uActiveList;
    c.option     = uintArena + cfg.uOption;
    c.counters   = uintArena + cfg.uCounters;
    c.sf         = uintArena + cfg.uSubtaskFeatureOf;

    c.featSM   = stateArena + cfg.aFeatSM;
    c.wBar     = stateArena + cfg.aWBar;
    c.featMean = stateArena + cfg.aFeatMean;
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
    constant KernelConfig &cfg    [[buffer(5)]],
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
// predict — main differential TD δ_main (A.2) + K per-cumulant SF TDs δ_ψ[k] under the row's
// active option e_{k*} (Alberta Plan Design 2: shared ψ_k(s, λ) = W_ψ[k] · concat[s, λ]).
// ψ_k is a SCALAR SF for cumulant k under chord λ — not an R-vector per option. Subtask feature
// sf[k] = k for Tier 3; Tier 7 makes sf[] dynamic top-K via bk_admit_commit re-rank. Cumulant
// centering ρ_k = featMean[sf[k]] tracks the per-feature running mean (2405.09999 differential
// TD; 2603.27874 stable under FA for any baseline).
//   [Grid: B]
// =================================================================================================
kernel void oak_predict(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.B) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint pOff = gid * ctx.R, cOff = gid * ctx.R;
    uint M = ctx.M;

    // Main differential TD on w_main (θ_v). Implicit-TD form per 2603.21921. Implicit α
    // (oak_reduce_alpha) + ρ̄ centering keep δ in fp32 dynamic range; no explicit projection.
    float vPrev = active_dot_half(ctx.valueW, 0u, prev, pOff, ctx.activeList, ctx.activeCount);
    float vNext = active_dot_half(ctx.valueW, 0u, curr, cOff, ctx.activeList, ctx.activeCount);
    ctx.vCurr[gid] = vNext;  // cached for oak_plan to read instead of recomputing (Tier 11)
    float terminalMask = (ctx.done[gid] != 0u) ? 0.0f : 1.0f;
    float dMain = (ctx.reward[gid] - ctx.rho) + terminalMask * vNext - vPrev;
    ctx.td[gid] = dMain;

    // Per-cumulant SF TD for the row's active option k* (one-hot chord λ = e_{k*}).
    //   ψ_k(s, e_{k*}) = Σ_i W_ψ[k][i] · s[i] + W_ψ[k][R + k*]
    // RED cumulant (Rojas-Lee 2410.10578 Theorem 5.1):
    //   c_k(s', r) = (r − ρ̄) + (state[sf[k]] − featMean[sf[k]])
    // Both terms are mean-zero by construction (ρ̄ centers reward; featMean centers feature).
    // ρ_k vanishes — no per-cumulant centering needed beyond the per-feature running mean.
    //   δ_k = c_k + terminalMask · ψ_k(s', e_{k*}) − ψ_k(s, e_{k*})
    uint optK = active_option(ctx, gid);
    float reward_centered = ctx.reward[gid] - ctx.rho;
    for (uint k = 0u; k < ctx.K; ++k) {
        uint psiBase = k * M;
        float chord_part = float(ctx.psi[psiBase + ctx.R + optK]);
        float psi_prev = active_dot_half(ctx.psi, psiBase, prev, pOff,
                                         ctx.activeList, ctx.activeCount) + chord_part;
        float psi_next = active_dot_half(ctx.psi, psiBase, curr, cOff,
                                         ctx.activeList, ctx.activeCount) + chord_part;
        uint sf_k = ctx.sf[k];  // dynamic top-K feature ranking (re-ranked in bk_admit_commit)
        float feat_centered = (sf_k < ctx.R) ? (curr[cOff + sf_k] - ctx.featMean[sf_k]) : 0.0f;
        float c_RED = reward_centered + feat_centered;
        ctx.tdPsi[gid * ctx.K + k] = c_RED + terminalMask * psi_next - psi_prev;
    }
}

// =================================================================================================
// plan — one-step option-model lookahead (F4) under Design 2 shared chord-conditional ψ.
//   Q_k(s) = w_main[sf[k]] · ψ_k(s, e_k) + max(0, (w̄[sf[k]] − w_main[sf[k]]) · s[sf[k]])
// where ψ_k(s, e_k) = Σ_i W_ψ[k][i] · s[i] + W_ψ[k][R + k] (one-hot chord λ = e_k).
// SIMD tournament argmax across K lanes (one TG = one SIMD group; K ≤ OAK_SIMD = 32 by host
// OakMetalLayout.optionCount). RRS optimism (§6.1) on the cumulant's own feature is the
// planning-time bonus that drives feature attainment. Citations: OBB 2505.00787 §2.4 Eq.4
// (chord-conditional Q); OK 2106.13105 Alg 1 line 3 (GPI argmax); RRS 2202.03466 §6.1.
//   [Dispatch: B threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_plan(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = tgid;
    if (row >= ctx.B) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint xOff = row * ctx.R;
    uint M = ctx.M;

    float qk = -FLT_MAX;
    uint  k  = tid;
    if (k < ctx.K) {
        uint psiBase = k * M;
        // ψ_k(s, e_k) under one-hot chord at index k.
        float psi_k = active_dot_half(ctx.psi, psiBase, curr, xOff,
                                      ctx.activeList, ctx.activeCount)
                    + float(ctx.psi[psiBase + ctx.R + k]);
        uint sf_k = ctx.sf[k];  // dynamic feature ranking (Tier 7)
        float wk  = (sf_k < ctx.R) ? float(ctx.valueW[sf_k]) : 0.0f;
        float val = wk * psi_k;
        if (sf_k < ctx.R) {
            float bonus = (ctx.wBar[sf_k] - wk) * curr[xOff + sf_k];
            val += max(bonus, 0.0f);
        }
        qk = val;
    }

    // SIMD tournament argmax in log2(OAK_SIMD) shuffle rounds. No threadgroup memory needed.
    uint k_lane = tid;
    for (uint off = OAK_SIMD >> 1; off > 0u; off >>= 1u) {
        float q2 = simd_shuffle_down(qk, off);
        uint  k2 = simd_shuffle_down(k_lane, off);
        bool  wins = (q2 > qk);
        qk     = wins ? q2 : qk;
        k_lane = wins ? k2 : k_lane;
    }
    if (tid == 0u) {
        // Reuse w_main · curr from oak_predict (Tier 11 fused dot — saves B · activeCount FMAs).
        ctx.td[row] += qk - ctx.vCurr[row];  // planning-time advantage on main TD
        option_slot(ctx, 1u - ctx.optionSlot)[row] = k_lane;
    }
}

// =================================================================================================
// meta_action — emit a real-valued action vector by sampling from the active option's per-dim
// Gaussian policy and compute the canonical Gaussian score function for the sample. Pass 10
// unification: ONE code path. The agent has zero a-priori knowledge of action-space type — it
// always emits actDim continuous values; the env-wrapper layer (EnvPoolEnv.step) interprets
// (argmax → discrete index | bound-clamp → continuous control). Inv-1: no env-conditioned
// branches; Bitter Lesson: no human-engineered "discrete vs continuous" taxonomy in the agent.
//
// Per (row, action-dim a):
//   μ_a     = W_π[k*][2a]   · s_curr     (state-conditional mean)
//   log_σ_a = W_π[k*][2a+1] · s_curr     (state-conditional log-std; σ = exp(log_σ) > 0 always)
//   z_a ~ N(0, 1) via Box-Muller from splitmix32(step, row, a) (pair of uniforms)
//   action[a] = μ_a + σ_a · z_a
//   scoreGrad[2a]   = z_a / σ_a                  (= (a−μ)/σ²; ∇_μ log p)
//   scoreGrad[2a+1] = (z_a² − 1) / 2             (= ((a−μ)²/σ² − 1)/2; ∇_logσ log p)
//
// No clamping anywhere — env clamps continuous-control action at the FFI boundary; implicit α +
// reward centering keep the parameter updates bounded under the Big World Hypothesis.
//   [Grid: B · A — one thread per (row, action-dim).]
// =================================================================================================
inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
inline float hash_uniform(uint step, uint row, uint dim) {
    uint h = splitmix32(step ^ splitmix32(row ^ splitmix32(dim)));
    // Convert 24-bit mantissa to (0, 1). FLT_MIN floor avoids log(0) in Box-Muller.
    return max(float(h >> 8) * 5.9604645e-8f, FLT_MIN);
}

kernel void oak_meta_action(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.A, a = gid % ctx.A;
    if (row >= ctx.B) return;

    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint optK    = option_slot(ctx, 1u - ctx.optionSlot)[row];
    uint xOff    = row * ctx.R;
    uint optBase = optK * ctx.D * ctx.R;

    // Per-dim Gaussian parameters from the (2a, 2a+1) rows of optPolicy[k*].
    float mu     = active_dot_half(ctx.optPolicy, optBase + (2u * a     ) * ctx.R,
                                   curr, xOff, ctx.activeList, ctx.activeCount);
    float log_sg = active_dot_half(ctx.optPolicy, optBase + (2u * a + 1u) * ctx.R,
                                   curr, xOff, ctx.activeList, ctx.activeCount);
    float sigma = exp(log_sg);  // strictly positive without clamp

    // Box-Muller standard-normal sample. Two structural splitmix32 hashes → two uniforms → one z.
    float u1 = hash_uniform(ctx.stepCount, row, 2u * a);
    float u2 = hash_uniform(ctx.stepCount, row, 2u * a + 1u);
    float z  = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);

    ctx.action[row * ctx.A + a] = mu + sigma * z;
    // Canonical Gaussian score function:
    //   ∇_μ     log p(a | μ, σ) = (a − μ) / σ²       = z / σ
    //   ∇_logσ  log p(a | μ, σ) = ((a−μ)²/σ² − 1)/2  = (z² − 1) / 2
    ctx.scoreGrad[row * ctx.D + 2u * a     ] = z / max(sigma, FLT_MIN);
    ctx.scoreGrad[row * ctx.D + 2u * a + 1u] = (z * z - 1.0f) * 0.5f;
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
    constant KernelConfig &cfg    [[buffer(5)]],
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
    constant KernelConfig &cfg    [[buffer(5)]],
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
    constant KernelConfig &cfg    [[buffer(5)]],
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
// learn_psi — update shared ψ[k, m] for m ∈ [0, M=R+K). Cumulant k's gradient w.r.t. weight m:
//   ∂L/∂W_ψ[k][m] = − (1/B) Σ_r δ_k[r] · concat[s_r, λ_r][m]
// where concat[r][m] = prev[r][m] for m < R, else (m == R + active_option(r) ? 1 : 0).
// Every row contributes to every cumulant's parameters via its own δ_k[r] (chord-conditional
// formulation; OBB §2.4). Implicit α from oak_reduce_alpha provides per-step contraction
// independent of input scale (Kim 2510.06149); no clamp needed.
//   [Grid: K · M]
// =================================================================================================
kernel void oak_learn_psi(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint M = ctx.M;
    uint k = gid / M, m = gid % M;
    if (k >= ctx.K) return;
    if (m < ctx.R && ctx.activeMask[m] == 0u) return;  // skip inactive state weights

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float step = ctx.scalars->alphaImp;
    float g = 0.0f;
    if (m < ctx.R) {
        // State weight: every row contributes via its own δ_k.
        for (uint r = 0u; r < ctx.B; ++r) {
            g = fma(ctx.tdPsi[r * ctx.K + k], prev[r * ctx.R + m], g);
        }
    } else {
        // Chord weight W_ψ[k][R + cidx]: contributes only from rows whose active option == cidx.
        uint cidx = m - ctx.R;
        for (uint r = 0u; r < ctx.B; ++r) {
            if (active_option(ctx, r) != cidx) continue;
            g += ctx.tdPsi[r * ctx.K + k];
        }
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    uint  pIdx = k * M + m;
    ctx.psi[pIdx] = half(float(ctx.psi[pIdx]) + step * g * invB);
}

// =================================================================================================
// learn_pi — REAL REINFORCE on the per-dim Gaussian policy. Gradient for parameter (k, d, i):
//   g[k,d,i] = (1 / |rows_k|) Σ_{r: opt(r)==k} δ_r · scoreGrad[r,d] · prev_state[r,i]
// where δ_r = td[r] is the main differential TD (with planning advantage from oak_plan), and
// scoreGrad[r,d] is the canonical Gaussian ∇_θ_d log π(a_r | s_r) computed by oak_meta_action
// at the PREVIOUS step.
//
// Per-update implicit step (Kim 2510.06149 generalized): the global `α_imp` from oak_reduce_alpha
// adapts only to feature norm, NOT to scoreGrad. With unbounded log_σ the score `(z²−1)/2` can
// reach O(z² ≈ 100), accumulating into log_σ updates of O(α₀ · 100 · prev) per step that drive
// log_σ → ∞ in tens of steps and overflow exp(). The fix is structural: compute the local
// gradient norm `||g_local||²` from the actual (prev[i] · scoreGrad[d])² products this thread
// is summing, then use `α_eff = α₀ / (1 + α₀ · ||g_local||²/cnt)`. When scoreGrad blows up,
// `α_eff` shrinks proportionally; the per-step parameter delta stays bounded regardless of σ
// dynamics. This is the SAME tracking-vs-imposing-priors discipline as oak_reduce_alpha — no
// clamp, no entropy regularizer, no σ floor. BWH-orthodox.
//   [Grid: K · D · R]
// =================================================================================================
kernel void oak_learn_pi(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / (ctx.D * ctx.R);
    uint rem = gid - k * ctx.D * ctx.R;
    uint d = rem / ctx.R, i = rem % ctx.R;
    if (k >= ctx.K || d >= ctx.D || i >= ctx.R) return;
    if (ctx.activeMask[i] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float g = 0.0f;
    float n2 = 0.0f;  // ||per-(k,d,i) gradient terms||²; per-update implicit-α denominator
    uint  cnt = 0u;
    for (uint r = 0u; r < ctx.B; ++r) {
        if (active_option(ctx, r) != k) continue;
        float pi_term = prev[r * ctx.R + i] * ctx.scoreGrad[r * ctx.D + d];
        g = fma(ctx.td[r], pi_term, g);
        n2 = fma(pi_term, pi_term, n2);
        cnt++;
    }
    if (cnt == 0u) return;
    float invCnt = 1.0f / float(cnt);
    float alpha_eff = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invCnt);
    uint  pIdx = k * ctx.D * ctx.R + d * ctx.R + i;
    ctx.optPolicy[pIdx] = half(float(ctx.optPolicy[pIdx]) + alpha_eff * g * invCnt);
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
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(ctx.stepCount, ctx.ridge);

    // (1) Observation-mean EMA (strided over O). Welford-style running mean — TRACKS the
    // world's actual statistics (BWH-orthodox), no fixed assumption about obs scale.
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

    // (1b) Per-feature mean EMA on the encoded state (running ρ_k tracking for cumulant
    // centering in oak_predict; replaces per-option-per-cumulant rhoKJ).
    for (uint f = tid; f < ctx.R; f += tgW) {
        float sum = 0.0f;
        for (uint r = 0u; r < ctx.B; ++r) sum += curr[r * ctx.R + f];
        float meanBatch = sum / float(max(ctx.B, 1u));
        ctx.featMean[f] += invT * (meanBatch - ctx.featMean[f]);
    }

    // (2) Latent telemetry: mean-square + max|latent| + non-finite count (B·R cells).
    float lms_l = 0.0f, maxLat_l = 0.0f, nLearn_l = 0.0f;
    uint total_rf = ctx.B * ctx.R;
    for (uint idx = tid; idx < total_rf; idx += tgW) {
        uint r = idx / ctx.R, f = idx % ctx.R;
        float v = curr[r * ctx.R + f];
        if (!isfinite(v)) { nLearn_l += 1.0f; v = 0.0f; }
        lms_l = fma(v, v, lms_l);
        maxLat_l = max(maxLat_l, fabs(v));
    }
    // (2b) ψ TD telemetry over B·K cells (Design 2: K cumulants per row, not R).
    float psi_l = 0.0f;
    uint total_rk = ctx.B * ctx.K;
    for (uint idx = tid; idx < total_rk; idx += tgW) {
        float dp = ctx.tdPsi[idx];
        psi_l = fma(dp, dp, psi_l);
    }
    threadgroup float sg_lms[4], sg_psi[4], sg_maxLat[4], sg_nLearn[4];
    float lms    = tg_sum_128(lms_l,    sg_lms,    tid, sgid);
    float psi    = tg_sum_128(psi_l,    sg_psi,    tid, sgid);
    float maxLat = tg_max_128(maxLat_l, sg_maxLat, tid, sgid);
    float nLearn = tg_sum_128(nLearn_l, sg_nLearn, tid, sgid);

    // (3) Per-(row, a) reductions: action variance, action delta vs prev, max|action|, non-finite.
    // Action delta vs prev (was `actionDeltaL2Mean`) is the actuation change signal. The old
    // `planningActionChangeL2Mean` used actionGrad which was identically equal to action under
    // pre-pass-9 — meaningless after the policy-grad refactor and removed.
    float ad_l = 0.0f, sgL2_l = 0.0f, maxPol_l = 0.0f, nPlan_l = 0.0f;
    uint total_ra = ctx.B * ctx.A;
    for (uint idx = tid; idx < total_ra; idx += tgW) {
        uint r = idx / ctx.A, a = idx % ctx.A;
        float x = ctx.action[r * ctx.A + a];
        float py = ctx.prevAction[r * ctx.A + a];
        bool nf = !isfinite(x);
        if (nf) { nPlan_l += 1.0f; x = 0.0f; }
        float p2 = x - py;
        ad_l = fma(p2, p2, ad_l);
        maxPol_l = max(maxPol_l, fabs(x));
    }
    // (3b) scoreGrad L² over B·D — non-zero variance is the probe-side signal that the policy
    // is producing meaningful score functions (Tier 6 verification).
    uint total_rd = ctx.B * ctx.D;
    for (uint idx = tid; idx < total_rd; idx += tgW) {
        float sg = ctx.scoreGrad[idx];
        if (!isfinite(sg)) sg = 0.0f;
        sgL2_l = fma(sg, sg, sgL2_l);
    }
    threadgroup float sg_ad[4], sg_sgL2[4], sg_maxPol[4], sg_nPlan[4];
    float ad     = tg_sum_128(ad_l,     sg_ad,     tid, sgid);
    float sgL2   = tg_sum_128(sgL2_l,   sg_sgL2,   tid, sgid);
    float maxPol = tg_max_128(maxPol_l, sg_maxPol, tid, sgid);
    float nPlan  = tg_sum_128(nPlan_l,  sg_nPlan,  tid, sgid);

    if (tid == 0u) {
        float invBR = 1.0f / float(max(ctx.B * ctx.R, 1u));
        float invBK = 1.0f / float(max(ctx.B * ctx.K, 1u));
        float invBA = 1.0f / float(max(ctx.B * ctx.A, 1u));
        float invBD = 1.0f / float(max(ctx.B * ctx.D, 1u));
        stats->latentStateMeanSquare      = lms * invBR;
        stats->optionModelPredictionMSE   = psi * invBK;
        stats->scoreGradL2Mean            = sgL2 * invBD;  // mean ‖∇log π‖² across rows × params
        stats->actionDeltaL2Mean          = ad  * invBA;
        stats->maxAbsLatentPrediction     = maxLat;
        stats->maxAbsPolicyOutput         = maxPol;
        stats->rawNonFiniteCount          = obsNF + nLearn + nPlan;
        stats->sanitizedNonFiniteCount    = nLearn + nPlan;
        stats->encodeNonFiniteCount       = obsNF;
        stats->learnNonFiniteCount        = nLearn;
        stats->planNonFiniteCount         = nPlan;
        ctx.counters[OAK_CNT_OBS_NF]   = uint(obsNF);
        ctx.counters[OAK_CNT_LEARN_NF] = uint(nLearn);
        ctx.counters[OAK_CNT_PLAN_NF]  = uint(nPlan);
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
    constant KernelConfig &cfg    [[buffer(5)]],
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
    // Per-feature utility = mean over batch of |δ_main · x| + Σ_k |δ_ψ_k · x|. 1+K terms per
    // row. K cumulants now (was R features) under Design 2 — divisor is structural 1+K.
    float u_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float x = curr[r * ctx.R + feat];
        u_l += fabs(ctx.td[r] * x);
        for (uint k = 0u; k < ctx.K; ++k) u_l += fabs(ctx.tdPsi[r * ctx.K + k] * x);
    }
    float u = simd_sum(u_l);
    if (tid == 0u) {
        float un = u * invB / float(1u + ctx.K);
        ctx.utilFeat[feat] += invT * (un - ctx.utilFeat[feat]);
    }
}

// =================================================================================================
// bk_option — per-option utility tracking + FIRE+SWR-style reinit (Design 2 scalar cumulants).
// Cumulant centering ρ_k now lives in featMean[sf[k]] (tracked in bk_reduce); per-option
// reward-mean rhoK was telemetry-only and has been deleted. utilOpt[k] tracks |δ_k|² over rows
// where option k is active. OCP hypothesis (2603.21173) validates the conjunctive reinit gate.
//   [Dispatch: K threadgroups × 128 threads.]
// =================================================================================================
kernel void oak_bk_option(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = tgid;
    if (k >= ctx.K) return;
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);

    // Per-option counts and squared-cumulant-error (option k's own δ_k summed over its rows).
    float cnt_l = 0.0f, psiErr_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        if (active_option(ctx, r) != k) continue;
        cnt_l += 1.0f;
        float d = ctx.tdPsi[r * ctx.K + k];
        psiErr_l = fma(d, d, psiErr_l);
    }
    threadgroup float sg_cnt[4], sg_p[4];
    float cnt  = tg_sum_128(cnt_l,    sg_cnt, tid, sgid);
    float pErr = tg_sum_128(psiErr_l, sg_p,   tid, sgid);
    float frac       = cnt * invB;
    float meanPsiErr = (cnt > 0.0f) ? pErr / cnt : 0.0f;

    if (tid == 0u) {
        ctx.utilOpt[k] += invT * (meanPsiErr - ctx.utilOpt[k]);

        // FIRE+SWR reinit trigger.
        if (meanPsiErr > ctx.cap * ctx.ridge && frac < ctx.ridge) {
            uint psiBase = k * ctx.M;
            for (uint m = 0u; m < ctx.M; ++m) ctx.psi[psiBase + m] = half(0.0f);
            float invR = 1.0f / sqrt(float(max(ctx.R, 1u)));
            // Diagonal seed: cumulant k's W_ψ[k][sf[k]] = +1/√R (RRS feature-attainment).
            if (k < ctx.R) ctx.psi[psiBase + k] = half(invR);
            uint polBase = k * ctx.D * ctx.R;
            for (uint idx = 0u; idx < ctx.D * ctx.R; ++idx) ctx.optPolicy[polBase + idx] = half(0.0f);
            ctx.utilOpt[k] = 0.0f;
            // K ≤ OAK_SIMD by host-side cap (OakMetalLayout.optionCount), so one uint
            // bitmask covers all options. Atomic OR for race-free per-option signaling
            // when multiple TGs trigger reinit in the same step.
            atomic_fetch_or_explicit((device atomic_uint *)(ctx.counters + OAK_CNT_REINIT_OPT),
                                     1u << k, memory_order_relaxed);
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
    constant KernelConfig &cfg    [[buffer(5)]],
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

    // Tier 7: dynamic top-K subtask-feature ranking. Cumulant k tracks the feature with the
    // k-th highest utilFeat. BWH-orthodox structural meta-learning: the agent picks its own
    // most-informative cumulants, no prior knowledge of which features matter.
    // O(K · R) selection-sort top-K (K is hardware-capped at 32, so this is small).
    for (uint k = 0u; k < ctx.K; ++k) {
        uint best = R; float bestU = -FLT_MAX;
        for (uint i = 0u; i < R; ++i) {
            bool taken = false;
            for (uint k2 = 0u; k2 < k; ++k2) if (ctx.sf[k2] == i) { taken = true; break; }
            if (taken) continue;
            if (ctx.utilFeat[i] > bestU) { bestU = ctx.utilFeat[i]; best = i; }
        }
        ctx.sf[k] = (best < R) ? best : min(k, R - 1u);
    }

    // FC-STOMP feedback: if any option reinited this step (set by oak_bk_option) AND the
    // worst feature is more than 1σ below mean utility, reinit its base-obs encoder weight
    // with a structural ±1/√R unit (OCP hypothesis 2603.21173 conjunctive-gate validation).
    if (ctx.counters[OAK_CNT_REINIT_OPT] != 0u && worstU < mean - std && worstIdx < ctx.baseO) {
        float sign = (((ctx.stepCount + worstIdx) & 1u) == 0u) ? 1.0f : -1.0f;
        ctx.encoder[worstIdx] = half(sign / sqrt(float(max(R, 1u))));
        ctx.utilFeat[worstIdx] = mean;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
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
    stats->oakOptionReinitCount = float(popcount(ctx.counters[OAK_CNT_REINIT_OPT]));

    // Slot toggles + step + clear option-reinit bitmask for next step.
    ctx.scalars->stateSlot  = 1u - ctx.scalars->stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.scalars->optionSlot;
    ctx.scalars->stepCount += 1u;
    ctx.counters[OAK_CNT_REINIT_OPT] = 0u;
}
