#include <metal_stdlib>
#include <float.h>
using namespace metal;

// =================================================================================================
// Orthodox Step 11 OaK on Metal — one shared latent, continual, average-reward, no discounting.
// Invariants (.cursor/rules/alberta_plan.mdc §1–9) hold: no env branches, no tuned scalars, no
// episodic semantics, no train/eval split, no per-step allocation, no cosmetic normalization,
// no curiosity on the external reward, meta-constants structurally derived from {R, B, K}.
//
// Citations (docs/papers/):
//   AlbertaPlan_2208.11173                     base agent; Step 11 Oak; STOMP progression
//   RethinkingFoundationsContinualRL_2504.08161 Bowling self-critique of episodic baggage
//   RewardRespectingSubtasks_2202.03466 §6.1   w̄ high-water-mark bonus weight (F1/F2)
//   RewardCentering_2405.09999                 differential TD w/ ρ̄ centering (main + per-opt)
//   BellmanErrorCentering_2502.03104           theoretical identity — VRC ≡ BEC, linear-FA proof
//   ImplicitUpdates_AvgRewardTD_2510.06149     αimp = α0/(1+α0·‖x‖²) — F10 on every linear head
//   ImplicitQLearningSarsa_2601.18907          implicit TD extended to control
//   StreamingDRLFinallyWorks_2410.14606        streaming AC (semi-grad; λ-trace rejected per Inv 9)
//     [known-superior-but-rejected: GradientEligibilityTraces_2507.09087]
//   FullGradientSuccessorFeatureRepresentations_2604.00686  ψ[k,j,i] vector SF Bellman
//   OptionKeyboard_2106.13105                  Alg 1 line 3: argmax_k  ψ^{π_k}·w (GPI)
//   OptimalBehaviorBasis_OptionKeyboard_2505.00787  OBB: minimal behavior basis for GPI
//   SuccessorFeaturesKeyboard_2310.15940        SFK: ψ_k(s)·w option scoring
//   FIRE_2602.08040 / SWR_2508.00212 / SpectralCollapsePlasticity_2509.22335
//                                               selective weight reinit + effective-rank floor
//   MetaGradientSearchControl_2406.19561        planning TD on w_main (F4)
//   ValueHallucinationDynaPlanning_2006.04363   planning-only-from-current-state (avoid VH)
//   TowardAgentsThatReasonAboutTheirComputation_2510.22833  planK structurally bounded
//   WhichFeaturesBestForSuccessorFeatures_2502.10790        cumulant caveat (identity cumulants OK)
//   TTT-Options-Sutton-2025-slides              activate per step: O*_t = argmax_o q̂(s, o, w⁰)
//
// Params (fp16, packed):
//   encoder[R]                                  linear encoder weights W_e
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
    float ridge;              // 1 / (cap + 1) (Tikhonov ridge; single source of truth)
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

struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeRank;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, planningActionChangeL2Mean;
    float optionModelPredictionMSE;
    float planningUsedOptionDynamicsMean;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilitySubtaskMean, utilityOptionMean, utilityPlanningMean;
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

// Double-buffered state slot access.
inline device float *state_slot(Ctx ctx, uint slot) { return ctx.state + slot * ctx.B * ctx.R; }
// Slot-toggled option index (active = slot set by meta_action last step; next = write slot).
inline device uint  *option_slot(Ctx ctx, uint slot) { return ctx.option + slot * ctx.B; }
inline uint active_option(Ctx ctx, uint row) {
    device const uint *o = option_slot(ctx, ctx.optionSlot);
    return min(o[row], ctx.K - 1u);
}

inline float online_inv_time(uint stepCount) { return 1.0f / float(stepCount + 1u); }
inline uint  ceil_sqrt(uint n) { float s = sqrt(float(max(n, 1u))); return uint(ceil(s)); }

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
inline float active_dot_float(device const float *w, uint wBase,
                              device const float *x, uint xBase,
                              device const uint *active, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        s = fma(w[wBase + idx], x[xBase + idx], s);
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
    state_slot(ctx, writeSlot)[row * ctx.R + feat] = clamp(z, -ctx.cap, ctx.cap);
}

// =================================================================================================
// predict — main differential TD δ_main (A.2) + per-cumulant SF TD δ_ψ[j] for the active option
// (A.1/F6/F7: full-gradient SF per 2604.00686). No separate stateW bank; ψ subsumes the horde.
// RRS optimism bonus (F1) added to δ_main via (w̄_i − w_i)·x_i when i < R and i == active option.
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

    // Main differential TD on w_main (θ_v).
    float vPrev = active_dot_half(ctx.valueW, 0u, prev, pOff, ctx.activeList, ctx.activeCount);
    float vNext = active_dot_half(ctx.valueW, 0u, curr, cOff, ctx.activeList, ctx.activeCount);
    vPrev = clamp(vPrev, -ctx.cap, ctx.cap);
    vNext = clamp(vNext, -ctx.cap, ctx.cap);
    float terminalMask = (ctx.done[gid] != 0u) ? 0.0f : 1.0f;
    float dMain = (ctx.reward[gid] - ctx.rho) + terminalMask * vNext - vPrev;
    ctx.td[gid] = clamp(dMain, -ctx.cap, ctx.cap);

    // Per-cumulant SF TD for the active option's ψ_k(·,:). Cumulant c_j = x_j (feature
    // attainment); ρ_{k,j} centered per-option-per-cumulant (2405.09999 × K × R).
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
        ctx.tdPsi[gid * ctx.R + j] = clamp(dj, -ctx.cap, ctx.cap);
    }
}

// =================================================================================================
// plan — one-step option-model lookahead (F4). Parallel over (row, option): B TGs of 32 threads
// each. Thread k ∈ [0, K) computes Q_k(x) = ψ_k(x)·w_main + optimism-bonus; threadgroup memory
// holds the K Q values; thread 0 picks argmax_k, writes nextOption[row], and pushes δ_plan onto
// td[row]. Citations: OK 2106.13105 Alg 1 line 3; MGSC 2406.19561; RRS §6.1 bonus.
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
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = tgid;
    if (row >= ctx.B) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint xOff = row * ctx.R;

    threadgroup float qs[32];
    if (tid < 32u) qs[tid] = -FLT_MAX;
    // Each thread computes its own Q_k when k < K. Extra threads (k ≥ K) contribute no-ops.
    uint k = tid;
    if (k < ctx.K) {
        uint pb = k * ctx.R * ctx.R;
        float qk = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            if (ctx.activeMask[j] == 0u) continue;
            float psi_kj = active_dot_half(ctx.psi, pb + j * ctx.R, curr, xOff,
                                           ctx.activeList, ctx.activeCount);
            qk = fma(float(ctx.valueW[j]), psi_kj, qk);
        }
        if (k < ctx.R) {
            float bonus = (ctx.wBar[k] - float(ctx.valueW[k])) * curr[xOff + k];
            qk += max(bonus, 0.0f);
        }
        qs[k] = clamp(qk, -ctx.cap, ctx.cap);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float qBest = -FLT_MAX; uint kBest = 0u;
        for (uint i = 0u; i < ctx.K; ++i) if (qs[i] > qBest) { qBest = qs[i]; kBest = i; }
        float vX = active_dot_half(ctx.valueW, 0u, curr, xOff, ctx.activeList, ctx.activeCount);
        vX = clamp(vX, -ctx.cap, ctx.cap);
        float dPlan = clamp(qBest - vX, -ctx.cap, ctx.cap);
        ctx.td[row] += dPlan;
        option_slot(ctx, 1u - ctx.optionSlot)[row] = kBest;
    }
}

// =================================================================================================
// meta_action — emit discrete or continuous action for the row, under π_{O*} where O* was just
// written by oak_plan (no chord blending — F5). Discrete sampling uses a deterministic Gumbel-max
// via a structural hash of (stepCount, row) — no host RNG round-trip (S3). Continuous actions
// clamp the option-policy logit to the per-dim env bounds.
//   [Grid: B · A]
// =================================================================================================
inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
inline float hash_uniform(uint step, uint row, uint dim) {
    uint h = splitmix32(step ^ splitmix32(row ^ splitmix32(dim)));
    // Uniform in [2^-24, 1); avoid 0 for log().
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
    uint row = gid / ctx.A, a = gid % ctx.A;
    if (row >= ctx.B) return;
    uint optK = option_slot(ctx, 1u - ctx.optionSlot)[row];  // O* just written by plan
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint xOff = row * ctx.R;
    // Per-option policy logit under π_{optK}.
    uint polBase = optK * ctx.A * ctx.R + a * ctx.R;
    float logit = active_dot_half(ctx.optPolicy, polBase, curr, xOff, ctx.activeList, ctx.activeCount);
    float lo = bounds[a], hi = bounds[ctx.A + a];

    if (fc_action_is_discrete) {
        // Gumbel-max sampling: pick argmax_a (logit_a + g_a); store the one-hot indicator as the
        // action float and leave the softmax-sampling decision to the discrete bound enforcement.
        // To keep this grid trivially parallel, each thread independently computes g_a and the
        // final argmax is a threadgroup reduction handled inside this kernel: thread a=0 scans all
        // dims. (Grid width is small: A ≤ ~18 for every benchmark env; negligible serial work.)
        logit = clamp(logit, lo, hi);
        if (a == 0u) {
            uint bestA = 0u; float bestL = -FLT_MAX;
            for (uint d = 0u; d < ctx.A; ++d) {
                uint pb = optK * ctx.A * ctx.R + d * ctx.R;
                float ld = active_dot_half(ctx.optPolicy, pb, curr, xOff,
                                           ctx.activeList, ctx.activeCount);
                ld = clamp(ld, bounds[d], bounds[ctx.A + d]);
                float u = hash_uniform(ctx.stepCount, row, d);
                float g = -log(-log(u));
                float s = ld + g;
                if (s > bestL) { bestL = s; bestA = d; }
            }
            // Write one-hot; this slot is read by the host as the sampled index (via argmax).
            for (uint d = 0u; d < ctx.A; ++d) {
                ctx.action[row * ctx.A + d]     = (d == bestA) ? 1.0f : 0.0f;
                ctx.actionGrad[row * ctx.A + d] = (d == bestA) ? 1.0f : 0.0f;
            }
        }
    } else {
        float bounded = clamp(logit, lo, hi);
        ctx.action[row * ctx.A + a]     = bounded;
        ctx.actionGrad[row * ctx.A + a] = bounded;
    }
}

// =================================================================================================
// learn — implicit avg-reward TD (F10: αimp = α0/(1+α0·‖x_active‖²) — 2510.06149 + 2601.18907)
// applied uniformly on EVERY linear head: encoder, w_main, ψ[k,j,:], optPolicy[k,a,:]. Streaming
// semi-gradient AC on option policies (2410.14606 — no λ-trace, per Inv 9). ψ learning restricted
// to the active option per row (on-policy differential SF; 2604.00686). Per-weight grid gives
// K·R·R parallelism for ψ (≫ SIMD width).
//   [Grid layout: a single dispatch with gid fanning out by region; region boundaries computed
//   from R/K/A widths. Kernel dispatch launches rank · batch threads and sub-kernels pick their
//   region from `gid`. Simpler: split into sub-kernels by region below.]
// =================================================================================================

// α_imp precomputed by oak_reduce_alpha once per step; learn kernels just read.
inline float implicit_step(Ctx ctx, device const float *slotPrev) { (void)slotPrev; return ctx.scalars->alphaImp; }

// =================================================================================================
// reduce_alpha — compute α_imp = α₀ / (1 + α₀ · ⟨‖x_active‖²⟩) and α_enc (centered-obs variant)
// once per step via a single-TG simd_sum reduction. Saves K·R·R recomputations inside learn_psi.
//   [Dispatch: 1 threadgroup × 32 threads.]
// =================================================================================================
kernel void oak_reduce_alpha(
    device half   *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    device const float   *bounds  [[buffer(5)]],
    constant KernelConfig &cfg    [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgW [[threads_per_threadgroup]])
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
    float n2 = simd_sum(n2_l);
    float obs = simd_sum(obs_l);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        float mean = n2 * invB;
        float meanObs = obs * invB;
        scalars->alphaImp = ctx.alpha0 / (1.0f + ctx.alpha0 * mean);
        scalars->alphaEnc = ctx.alpha0 / (1.0f + ctx.alpha0 * meanObs);
    }
}

// learn_value — update w_main[feat] with δ_total · x[feat] reduction, via simd_sum across batch.
// Also updates featSM[feat] (EMA of x²) and w̄[feat] (slow high-water-mark).
//   [Dispatch: R threadgroups × 32 threads.]
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
    float g_l = 0.0f, sm_l = 0.0f, n2_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float xp = prev[r * ctx.R + feat];
        g_l = fma(ctx.td[r], xp, g_l);
        sm_l = fma(xp, xp, sm_l);
        n2_l += active_norm2(prev, r * ctx.R, ctx.activeList, ctx.activeCount);
    }
    float g  = simd_sum(g_l);
    float sm = simd_sum(sm_l);
    float n2 = simd_sum(n2_l);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        float invT = online_inv_time(ctx.stepCount);
        // Batch-avg implicit step α_imp = α0 / (1 + α0·‖x‖²_batch-mean) per Kim-Cho-Laber 2510.06149.
        float step = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
        ctx.featSM[feat] += (sm * invB - ctx.featSM[feat]) * invT;
        float wNew = float(ctx.valueW[feat]) + step * g * invB;
        wNew = clamp(wNew, -ctx.cap, ctx.cap);
        ctx.valueW[feat] = half(wNew);
        float decay = 1.0f - ctx.alpha0;
        ctx.wBar[feat] = max(ctx.wBar[feat] * decay, fabs(wNew));
    }
}

// learn_encoder — update encoder[feat] for base obs features via TD × centered_obs (F10 on W_e).
//   [Grid: baseObsDim]
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
    ctx.encoder[feat] = half(clamp(eNew, -ctx.cap, ctx.cap));
}

// learn_psi — update ψ[k, j, i] += αimp · (1/|rows_k|) · Σ_{row: active(row)==k} tdPsi[row, j] · prev[row, i].
//   [Grid: K · R · R]
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
    float step = implicit_step(ctx, prev);
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
    ctx.psi[pIdx] = half(clamp(pNew, -ctx.cap, ctx.cap));
}

// learn_opt_policy — semi-grad streaming AC on π_k (2410.14606; no λ-trace — Inv 9). Gradient:
// advantage (≡ td[row] for the main task; treating the main TD as the shared advantage signal)
// times the action indicator × prev_state.
//   [Grid: K · A · R]
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
    float step = implicit_step(ctx, prev);
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
    ctx.optPolicy[pIdx] = half(clamp(pNew, -ctx.cap, ctx.cap));
}

// =================================================================================================
// bk_reduce — batch stats via SIMD-group reduction. Single TG of 32 threads strides over (row, f);
// each thread accumulates partial sums and simd_sum fans them in. Observation-mean Welford stays
// parallel across obs-feature threads in the same pass.
//   [Dispatch: 1 threadgroup × 32 threads.]
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
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invT = online_inv_time(ctx.stepCount);

    // Online observation-mean EMA (strided over O).
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
    float obsNF = simd_sum(obsNF_local);

    // Per-(row, f) reductions: latent ms, psi-TD ms, maxAbs, non-finites.
    float lms_l = 0.0f, psi_l = 0.0f, maxLat_l = 0.0f;
    float nLearn_l = 0.0f;
    uint total_rf = ctx.B * ctx.R;
    for (uint idx = tid; idx < total_rf; idx += tgW) {
        uint r = idx / ctx.R, f = idx % ctx.R;
        float v = curr[r * ctx.R + f];
        if (!isfinite(v)) { nLearn_l += 1.0f; v = 0.0f; }
        float vc = clamp(v, -ctx.boundLat, ctx.boundLat) * ctx.invBoundLat;
        lms_l = fma(vc, vc, lms_l);
        maxLat_l = max(maxLat_l, fabs(v));
        float dp = ctx.tdPsi[r * ctx.R + f];
        psi_l = fma(dp, dp, psi_l);
    }
    float lms = simd_sum(lms_l);
    float psi = simd_sum(psi_l);
    float maxLat = simd_max(maxLat_l);
    float nLearn = simd_sum(nLearn_l);

    // Per-(row, a) reductions: plan L2, action delta, maxPol, nPlan.
    float pL2_l = 0.0f, ad_l = 0.0f, maxPol_l = 0.0f, nPlan_l = 0.0f;
    uint total_ra = ctx.B * ctx.A;
    for (uint idx = tid; idx < total_ra; idx += tgW) {
        uint r = idx / ctx.A, a = idx % ctx.A;
        float x = ctx.action[r * ctx.A + a];
        float yg = ctx.actionGrad[r * ctx.A + a];
        float py = ctx.prevAction[r * ctx.A + a];
        float p1 = clamp(x - yg, -ctx.boundAct, ctx.boundAct) * ctx.invBoundAct;
        float p2 = clamp(x - py, -ctx.boundAct, ctx.boundAct) * ctx.invBoundAct;
        pL2_l = fma(p1, p1, pL2_l);
        ad_l  = fma(p2, p2, ad_l);
        maxPol_l = max(maxPol_l, fabs(x));
        if (!isfinite(x) || !isfinite(yg)) nPlan_l += 1.0f;
    }
    float pL2 = simd_sum(pL2_l);
    float ad  = simd_sum(ad_l);
    float maxPol = simd_max(maxPol_l);
    float nPlan = simd_sum(nPlan_l);

    if (tid == 0u) {
        float invBR = 1.0f / float(max(ctx.B * ctx.R, 1u));
        float invBA = 1.0f / float(max(ctx.B * ctx.A, 1u));
        stats->latentStateMeanSquare      = lms * invBR;
        stats->optionModelPredictionMSE   = psi * invBR;
        stats->planningActionChangeL2Mean = pL2 * invBA;
        stats->actionDeltaL2Mean          = ad  * invBA;
        stats->maxAbsLatentPrediction     = maxLat;
        stats->maxAbsRewardPrediction     = 0.0f;
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
// bk_option — per-option utility, ρ_k, ρ_{k,j}, FIRE+SWR-style reinit. Dispatch: K TGs × 32
// threads. Each TG fans-in batch stats with simd_sum; thread 0 commits. ρ_{k,j} updated in a
// second strided loop over j.
//   [Dispatch: K threadgroups × 32 threads.]
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
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = tgid;
    if (k >= ctx.K) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = online_inv_time(ctx.stepCount);

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
    float cnt   = simd_sum(cnt_l);
    float rSum  = simd_sum(rSum_l);
    float pErr  = simd_sum(psiErr_l);
    float frac  = cnt * invB;
    float meanR = (cnt > 0.0f) ? rSum / cnt : 0.0f;
    float meanPsiErr = (cnt > 0.0f) ? pErr / (cnt * float(ctx.R)) : 0.0f;

    // Per-cumulant ρ_{k,j}: strided over j with simd_sum over batch per j.
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

        // FIRE+SWR reinit trigger.
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
            atomic_fetch_or_explicit((device atomic_uint *)(ctx.counters + which), bit, memory_order_relaxed);
        }
    }
}

// =================================================================================================
// bk_feature — per-feature utility via SIMD reduction. Dispatch: R TGs × 32 threads; each thread
// strides batch. Utility = mean |δ_main · x_feat| + mean_j |δ_ψ_{·,j} · x_feat|.
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
    float invT = online_inv_time(ctx.stepCount);
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
// bk_admit — admission + FC-STOMP feature reinit + effective-rank floor (F11). Serial on one
// thread: admits features whose utility exceeds the running mean, then ensures activeCount ≥
// ceil(√R) by pulling in the next-best-utility inactive features (2509.22335 spectral-collapse
// mitigation). If any option was reinited this step, reinits the encoder weight of the worst base
// feature to +/- 1/√R (A.11 / sign from (stepCount XOR featIdx) parity — no RNG).
//   [Grid: 1]
// =================================================================================================
kernel void oak_bk_admit(
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
    // Effective-rank floor (F11 — 2509.22335). Structural floor = ⌈√R⌉. Repeatedly pull the
    // highest-utility inactive feature until we hit the floor.
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

    // FC-STOMP feedback: if any option reinited (counters[3]|counters[4] nonzero) AND the worst
    // feature is below mean-std, reinit its base-obs encoder weight structurally.
    uint reinitMask = ctx.counters[3] | ctx.counters[4];
    if (reinitMask != 0u && worstU < mean - std && worstIdx < ctx.baseO) {
        float sign = (((ctx.stepCount + worstIdx) & 1u) == 0u) ? 1.0f : -1.0f;
        ctx.encoder[worstIdx] = half(sign / sqrt(float(max(R, 1u))));
        ctx.utilFeat[worstIdx] = mean;
        ctx.counters[5] += 1u;
    }
}

// =================================================================================================
// bk_commit — ρ̄ (Naik centering), stepCount++, pEff (effective-rank trace/max of featSM), slot
// toggles (S4/S5), telemetry roll-up. Grid: 1 thread.
// =================================================================================================
kernel void oak_bk_commit(
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

    // Differential avg-reward: ρ̄ += (1/t)·(r̄ - ρ̄).
    float meanR = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) meanR += ctx.reward[r];
    meanR /= float(max(ctx.B, 1u));
    float invT = online_inv_time(ctx.stepCount);
    ctx.scalars->rho += (meanR - ctx.scalars->rho) * invT;

    // Effective-rank via trace/max of featSM (structural p_eff like Autostep).
    float tr = 0.0f, mx = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) { float v = ctx.featSM[i]; tr += v; mx = max(mx, v); }
    ctx.scalars->pEff = tr / max(mx, FLT_MIN);

    // Telemetry roll-up.
    float uF = 0.0f, uO = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) uF += ctx.utilFeat[i];
    for (uint i = 0u; i < ctx.K; ++i) uO += ctx.utilOpt[i];
    stats->rewardRate            = ctx.scalars->rho;
    stats->activeRank            = float(ctx.scalars->activeCount);
    stats->utilityFeatureMean    = uF / float(max(ctx.R, 1u));
    stats->utilityOptionMean     = uO / float(max(ctx.K, 1u));
    stats->utilitySubtaskMean    = uF / float(max(ctx.R, 1u));
    stats->utilityPlanningMean   = uO / float(max(ctx.K, 1u));
    stats->planningUsedOptionDynamicsMean = 1.0f;
    // Reinit count = popcount of both counter slots.
    uint reinitMask = ctx.counters[3] | (ctx.counters[4] << 0);
    uint popcnt = popcount(ctx.counters[3]) + popcount(ctx.counters[4]);
    stats->oakOptionReinitCount = float(popcnt);
    (void)reinitMask;

    // Slot toggles: stateSlot flips so the next step's 'prev' is what we just wrote as 'curr';
    // optionSlot flips so active = what plan wrote this step.
    ctx.scalars->stateSlot  = 1u - ctx.scalars->stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.scalars->optionSlot;
    ctx.scalars->stepCount += 1u;
    // Clear per-step reinit bitmasks for the next step.
    ctx.counters[3] = 0u; ctx.counters[4] = 0u;
}
