// =================================================================================================
// OakKernel.metal - Orthodox Step 11 Oak agent. Single-stream (B=1), 12 kernels, Metal 4 ICB.
//
// Architectural sources (read these before changing this file):
//   - Alberta Plan for AI Research (Sutton-Bowling-Pilarski 2022, arXiv:2208.11173) Step 11 Oak
//     (p. 10): STOMP + feedback assessing utility of features/subtasks/options/option-models
//     + feature reranking + option keyboard (Design B chosen here: chord vector defines a
//     joint subtask, model treats lambda as opaque descriptor).
//   - Reward-Respecting Subtasks (Sutton-Machado-Holland-Timbers-Tanner-White 2022/2023,
//     arXiv:2202.03466): eq. 4 feature-attainment stopping bonus, eq. 17 expectation model UWT,
//     eq. 19 planning by approximate value iteration, Section 8 FCSTOMP/Oak.
//   - BurningRED (arXiv:2410.10578): differential extension of RR-Subtasks (cumulant - rho_bar).
//   - Reward Centering (Naik et al., arXiv:2405.09999): rho_bar tracking + TD centering.
//   - varpi-Relative TD (Sakha-Kamalapurkar-Meyn, arXiv:2603.27874v2 IEEE CDC 2026 Theorem
//     2.4(ii)): first published linear-FA stability proof for varpi-relative TD; mean-flow
//     matrix is uniformly Hurwitz with v_bar/psi_bar centering.
//   - Implicit alpha (Kim, arXiv:2510.06149): alpha_imp = alpha_0 / (1 + alpha_0 * ||x||^2).
//   - SwiftSarsa (arXiv:2507.19539) + MetaOptimize (arXiv:2402.02342): per-feature TIDBD.
//   - Belief-state recurrence (Belief-State RWKV, arXiv:2604.09671): per-feature retention
//     alpha_i = min(1, alphaImpFeat[i] * cap).
//   - Bacon-Harb-Precup option-critic (arXiv:1609.05140) for beta/pi gradients.
//   - Wan-Naik-Sutton avg-reward options (arXiv:2110.13855).
//   - GraMa feature utility (arXiv:2505.24061): utilFeat from gradient-magnitude.
//   - FIRE+SWR reinit (arXiv:2602.08040): conjunctive dormancy/magnitude/effective-rank.
//   - Generate-and-test (Mahmood-Sutton-Pilarski 2012, Rahman 2020, Samani 2022): hash-seeded
//     random projections fill freshly-reinited feature slots.
//   - Sutton-Barto Ch. 12.7 (TD lambda eligibility traces, backward view); Ch. 13 (actor-critic
//     with traces); Ch. 9.5.4 (online normalization).
//   - Big World Hypothesis (.cursor/rules/alberta_plan.mdc): NO zero clamps. Magnitudes carry
//     information about the agent's slice of reality. safe_half / bounded_soft only.
//
// Single-stream commitment: ONE observation -> ONE action -> ONE reward per step. There is NO
// batch dimension. Eligibility traces are always-on. This matches the Plan's "On every step the
// new observation must be processed by perception to produce a state, which is then processed by
// the primary policy to produce that time step's action" (p. 4). Multi-env benchmarking lives
// in the Swift host as N independent agent instances over N environments, never as B>1 here.
// =================================================================================================
#include <metal_stdlib>
#include <float.h>
using namespace metal;

constant uint OAK_SIMD = 32u;       // Apple GPU SIMD width (structural, Inv-9).

// Per-step non-finite counters (cleared each step by the host) and life-of-engine reinit
// counters (never cleared).
constant uint OAK_CNT_OBS_NF       = 0u;
constant uint OAK_CNT_LEARN_NF     = 1u;
constant uint OAK_CNT_PLAN_NF      = 2u;
constant uint OAK_CNT_REINIT_OPT   = 3u;
constant uint OAK_CNT_REINIT_FEAT  = 4u;
constant uint OAK_CNT_TDPSI_NF     = 5u;

// =================================================================================================
// Configuration mirrored from Swift KernelConfig. All offsets are float-counts (UInt32 buffer is
// counted in UInt32 units). Single-stream means no `batch` field.
// =================================================================================================
struct KernelConfig {
    uint obsDim, rank, actDim, optionCount, baseObsDim;
    uint actionParamCount;                  // dPolicy from the env wrapper (Inv-1).

    // Param offsets (params buffer, fp32).
    uint pEncoder, pValue, pPsi, pOptPolicy, pBetaW, pOmegaW;
    uint pOptReward, pOptTransition;        // STEP 11 BLOCKER additions (option model).

    // State arena offsets (state buffer, fp32). Single-stream: each slot holds ONE row.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                             // 2 * R floats (slot toggle).
    uint sPrevPolicyMean, sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi, sVCurr;
    uint sChord, sChordPrev, sBetaProb;
    uint sFeatSM, sWBar, sFeatMean;
    uint sUtilFeat, sUtilOpt, sUtilPlan, sOptFracEMA;
    uint sAlphaImpFeat, sAlphaImpBeta, sAlphaImpH;
    uint sPsiMeanBaseline;
    uint sValueE, sPsiE, sPiE;
    uint sOptRewardE, sOptTransitionE;       // option-model eligibility traces.
    uint sPlanAdv;                           // scalar planning advantage (oak_plan -> oak_predict).

    // UInt arena offsets (uint buffer, UInt32).
    uint uDone, uActiveMask, uActiveList, uOption, uCounters, uSubtaskFeatureOf;
    uint uOptionAge, uPlanArgmax;

    // Structural meta-learning constants (Inv-9: derived from cap = R, never tuned).
    float cap;          // cap = R (per-feature retention bound; structural).
    float alpha0;       // alpha0 = 1/cap.
    float ridge;        // ridge = 1/(cap+1) for safe-divide / track-rate.
};

// =================================================================================================
// Per-step scalars written by kernels and consumed across kernels in the same step. Mirrored from
// Swift KernelScalars.
// =================================================================================================
struct KernelScalars {
    uint  stepCount;            // monotonic step counter (host-bumped via bk_admit_commit).
    uint  activeCount;          // |activeList|; floor at ceil(sqrt(R)).
    uint  stateSlot;            // toggle: which of the 2 state slots holds s_t (0 or 1).
    uint  optionSlot;           // toggle: which of the 2 option slots holds k_t (0 or 1).
    uint  activeListIsIdentity; // 1 iff activeCount == R and activeList[i]==i; fast path flag.

    float rho;                  // rho_bar - cumulative reward EMA (Reward Centering).
    float vMeanBaseline;        // <valueW, featMean>, varpi-RTD value baseline.
    float pEff;                 // effective rank proxy from featSM.
    float alphaImp;             // implicit alpha for state-features.
    float alphaEnc;             // implicit alpha for encoder weights.
    float alphaPi;              // implicit alpha for policy weights (denom includes ||scoreGrad||^2).
    float modelErrEMA;          // EMA of option-model prediction MSE (LPM signal source).
    float modelErrEMAPrev;      // slow snapshot of modelErrEMA (LPM second-derivative).
    float lambdaTrace;          // structural lambda = 1 - 1/cap (Sutton-Barto Ch. 12.7).
};

// =================================================================================================
// Per-step telemetry. Layout mirrors Swift KernelStats. Host reads these at the end of each step.
// =================================================================================================
struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeCount;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, scoreGradL2Mean;
    float optionModelPredictionMSE;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilityOptionMean;
    float tdPsiNonFiniteCount;
    float pEffectiveRank;
    float modelErrEMA, modelErrEMAPrev;
    float tdBaselineStat;
    float alphaImpStat, alphaEncStat, alphaPiStat;
    float oakFeatureReinitCount;
    float encoderMax, valueWMax, psiMax, optPolicyMax;
    float betaWMax, omegaWMax;
    float optRewardMax, optTransitionMax;
};

// =================================================================================================
// Numerical safety - the only two helpers used at every weight-write and forward-bound site.
// safe_half  = smooth zero-bias soft saturation at sqrt(HALF_MAX) ~= 256 (Pass-15 T2.5
//              bounded_soft, structurally derived from fp16 storage).
// bounded_soft = same operator without the isfinite guard - for forward outputs where
//              non-finite means there's a numerical bug to investigate, not a value to clamp.
// =================================================================================================
inline float safe_half(float x) {
    constexpr float M = 256.0f;
    return isfinite(x) ? M * tanh(x / M) : 0.0f;
}

inline float bounded_soft(float x) {
    constexpr float M = 256.0f;
    return M * tanh(x / M);
}

// Tracking-rate EMA (Sutton-Barto §2.6): equivalent to 1/(stepCount+1) as a step shrinks toward
// 0, with `ridge` floor for early steps where stepCount is small. Used for non-stationary
// tracking of the world's signal statistics (BWH).
inline float track_rate(uint step, float ridge) {
    return max(1.0f / (float(step) + 1.0f), ridge);
}

// splitmix32 hash for deterministic RNG without per-step host alloc. Returns u in [0, 1).
inline float hash_uniform(uint a, uint b, uint c) {
    uint z = a ^ (b * 0x7feb352du) ^ (c * 0x846ca68bu);
    z = (z ^ (z >> 16)) * 0x9e3779b9u;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z ^= z >> 16;
    return float(z >> 8) * 5.9604645e-8f;       // 1 / 2^24
}

// =================================================================================================
// Per-step computation context. ALL kernels build one of these from the bound buffers and config.
// Eliminates pointer-arithmetic noise from every kernel body.
// =================================================================================================
struct Ctx {
    uint O, R, K, A, M, baseO, D;
    uint stepCount, activeCount, stateSlot, optionSlot;
    float cap, alpha0, ridge;
    float rho, vMeanBaseline, lambdaTrace;
    float alphaImp, alphaEnc, alphaPi;

    device float        *encoder;
    device float        *valueW;
    device float        *psi;
    device float        *optPolicy;
    device float        *betaW;
    device float        *omegaW;
    device float        *optReward;
    device float        *optTransition;

    device float        *observation;
    device float        *observationMean;
    device float        *obsSM;
    device float        *state;          // [2 * R] - both slots flat.
    device float        *prevPolicyMean;
    device float        *policyMean;
    device float        *scoreGrad;
    device float        *reward;
    device float        *td;
    device float        *tdPsi;
    device float        *vCurr;
    device float        *chord;
    device float        *chordPrev;
    device float        *betaProb;
    device float        *featSM;
    device float        *wBar;
    device float        *featMean;
    device float        *utilFeat;
    device float        *utilOpt;
    device float        *utilPlan;
    device float        *optFracEMA;
    device float        *alphaImpFeat;
    device float        *alphaImpBeta;
    device float        *alphaImpH;
    device float        *psiMeanBaseline;
    device float        *valueE;
    device float        *psiE;
    device float        *piE;
    device float        *optRewardE;
    device float        *optTransitionE;
    device float        *planAdv;

    device uint         *done;
    device uint         *activeMask;
    device uint         *activeList;
    device uint         *option;
    device uint         *counters;
    device uint         *subtaskFeatureOf;
    device uint         *optionAge;
    device uint         *planArgmax;

    device KernelScalars *scalars;
    device KernelStats   *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount; c.A = cfg.actDim;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.M = c.R + c.K;                                // psi concat dim (state R + chord K).
    c.stepCount = scalars->stepCount;
    c.activeCount = scalars->activeCount;
    c.stateSlot = scalars->stateSlot;
    c.optionSlot = scalars->optionSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho;
    c.vMeanBaseline = scalars->vMeanBaseline;
    c.lambdaTrace = scalars->lambdaTrace;
    c.alphaImp = scalars->alphaImp;
    c.alphaEnc = scalars->alphaEnc;
    c.alphaPi = scalars->alphaPi;

    c.encoder        = params + cfg.pEncoder;
    c.valueW         = params + cfg.pValue;
    c.psi            = params + cfg.pPsi;
    c.optPolicy      = params + cfg.pOptPolicy;
    c.betaW          = params + cfg.pBetaW;
    c.omegaW         = params + cfg.pOmegaW;
    c.optReward      = params + cfg.pOptReward;
    c.optTransition  = params + cfg.pOptTransition;

    c.observation    = stateArena + cfg.sObservation;
    c.observationMean= stateArena + cfg.sObsMean;
    c.obsSM          = stateArena + cfg.sObsSM;
    c.state          = stateArena + cfg.sState;
    c.prevPolicyMean = stateArena + cfg.sPrevPolicyMean;
    c.policyMean     = stateArena + cfg.sPolicyMean;
    c.scoreGrad      = stateArena + cfg.sScoreGrad;
    c.reward         = stateArena + cfg.sReward;
    c.td             = stateArena + cfg.sTd;
    c.tdPsi          = stateArena + cfg.sTdPsi;
    c.vCurr          = stateArena + cfg.sVCurr;
    c.chord          = stateArena + cfg.sChord;
    c.chordPrev      = stateArena + cfg.sChordPrev;
    c.betaProb       = stateArena + cfg.sBetaProb;
    c.featSM         = stateArena + cfg.sFeatSM;
    c.wBar           = stateArena + cfg.sWBar;
    c.featMean       = stateArena + cfg.sFeatMean;
    c.utilFeat       = stateArena + cfg.sUtilFeat;
    c.utilOpt        = stateArena + cfg.sUtilOpt;
    c.utilPlan       = stateArena + cfg.sUtilPlan;
    c.optFracEMA     = stateArena + cfg.sOptFracEMA;
    c.alphaImpFeat   = stateArena + cfg.sAlphaImpFeat;
    c.alphaImpBeta   = stateArena + cfg.sAlphaImpBeta;
    c.alphaImpH      = stateArena + cfg.sAlphaImpH;
    c.psiMeanBaseline= stateArena + cfg.sPsiMeanBaseline;
    c.valueE         = stateArena + cfg.sValueE;
    c.psiE           = stateArena + cfg.sPsiE;
    c.piE            = stateArena + cfg.sPiE;
    c.optRewardE     = stateArena + cfg.sOptRewardE;
    c.optTransitionE = stateArena + cfg.sOptTransitionE;
    c.planAdv        = stateArena + cfg.sPlanAdv;

    c.done             = uintArena + cfg.uDone;
    c.activeMask       = uintArena + cfg.uActiveMask;
    c.activeList       = uintArena + cfg.uActiveList;
    c.option           = uintArena + cfg.uOption;
    c.counters         = uintArena + cfg.uCounters;
    c.subtaskFeatureOf = uintArena + cfg.uSubtaskFeatureOf;
    c.optionAge        = uintArena + cfg.uOptionAge;
    c.planArgmax       = uintArena + cfg.uPlanArgmax;

    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// Standardized observation x_tilde = (x - mu) / sqrt(SM - mu^2 + ridge). Online normalization
// per Sutton-Barto Ch. 9.5.4 - tracking estimates, not stationary BatchNorm.
inline float standardized_obs(float x, float mean, float sm, float ridge) {
    float var = max(sm - mean * mean, 0.0f);
    return (x - mean) * rsqrt(var + ridge);
}

// =================================================================================================
// Threadgroup reduction primitives (small fixed count, no allocation).
// =================================================================================================
inline float tg_sum_128(threadgroup float *buf, uint tid, uint sgid, float v) {
    v = simd_sum(v);
    if (simd_is_first()) buf[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float s = (tid < 4u) ? buf[tid] : 0.0f;
        s = simd_sum(s);
        if (tid == 0u) buf[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

inline float tg_max_128(threadgroup float *buf, uint tid, uint sgid, float v) {
    v = simd_max(v);
    if (simd_is_first()) buf[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float m = (tid < 4u) ? buf[tid] : -FLT_MAX;
        m = simd_max(m);
        if (tid == 0u) buf[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// =================================================================================================
// 1. oak_bk_obs - per-feature observation Welford EMA. Single TG (32 threads strides over O).
//    Updated FIRST so encode reads the just-tracked statistics.
// =================================================================================================
kernel void oak_bk_obs(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgW [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    uint nfLocal = 0u;
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nfLocal++; continue; }
        ctx.observationMean[o] = bounded_soft(ctx.observationMean[o]
                                              + invT * (x - ctx.observationMean[o]));
        ctx.obsSM[o]           = bounded_soft(ctx.obsSM[o]
                                              + invT * (x * x - ctx.obsSM[o]));
    }
    uint nfTotal = simd_sum(nfLocal);
    if (tid == 0u) ctx.counters[OAK_CNT_OBS_NF] = nfTotal;
}

// =================================================================================================
// 2. oak_encode - perception. Per-feature linear projection over standardized obs + per-feature
//    belief retention (alpha_i = min(1, alphaImpFeat[i]*cap), Belief-State RWKV 2604.09671).
//    Outputs the new state slot. One thread per latent dim; grid = R threads.
// =================================================================================================
kernel void oak_encode(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint feat [[thread_position_in_grid]])
{
    if (feat >= cfg.rank) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint writeSlot = 1u - ctx.stateSlot;
    uint encRow = feat * ctx.baseO;

    float z_instant = 0.0f;
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o],
                                   ctx.obsSM[o], ctx.ridge);
        z_instant = fma(ctx.encoder[encRow + o], c, z_instant);
    }
    float retention = min(1.0f, ctx.alphaImpFeat[feat] * ctx.cap);
    float prev = state_at(ctx, ctx.stateSlot)[feat];
    float z = retention * z_instant + (1.0f - retention) * prev;
    state_at(ctx, writeSlot)[feat] = bounded_soft(z);

    if (feat == 0u) {
        // Reset planAdv from previous step (one writer per step).
        ctx.planAdv[0] = 0.0f;
    }
}

// =================================================================================================
// 3. oak_meta_policy - Design B option keyboard. Computes the chord vector lambda in R^K via
//    softmax(omega_W . s'); samples beta termination via sigmoid(beta_W . s' + chord-feedback);
//    on terminate, commits the new chord and bumps optionAge=1; otherwise carries chordPrev.
//    The chord is the OPAQUE DESCRIPTOR per Plan p. 11 - the model treats it as a name, the
//    policy and beta read it as feedback. Single-threaded (one row, K is small).
// =================================================================================================
kernel void oak_meta_policy(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    for (uint k = 0u; k < ctx.K; ++k) ctx.chordPrev[k] = ctx.chord[k];

    // beta termination logit: w_beta . s' + w_beta_chord . chordPrev (Bacon-Harb-Precup 2017).
    float beta_logit = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) beta_logit = fma(ctx.betaW[i], curr[i], beta_logit);
    for (uint k = 0u; k < ctx.K; ++k) beta_logit = fma(ctx.betaW[ctx.R + k], ctx.chordPrev[k], beta_logit);

    // Use the LPM signal (modelErrEMA - lpmRate) as a structural surprise nudge to terminate
    // when the model is wrong AND not improving (Hou-An-Du 2509.25438 ICLR 2026). LPM enters
    // beta-logit ONLY (not reward/td) - Inv-8 satisfied.
    float lpmRate = max(0.0f, ctx.scalars->modelErrEMAPrev - ctx.scalars->modelErrEMA);
    float surprise = (ctx.scalars->modelErrEMA - lpmRate)
                   / max(ctx.scalars->modelErrEMA, ctx.ridge);
    beta_logit += surprise;
    float beta_prob = 1.0f / (1.0f + exp(-beta_logit));
    ctx.betaProb[0] = beta_prob;

    float u = hash_uniform(ctx.stepCount, 0u, 1u);
    bool terminate = (ctx.optionAge[0] == 0u) || (u < beta_prob);

    if (terminate) {
        // Compute chord ~ softmax(omega_W . s'). Stable softmax: subtract max-logit.
        float maxL = -FLT_MAX;
        float logits[OAK_SIMD];
        for (uint k = 0u; k < ctx.K; ++k) {
            float l = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i)
                l = fma(ctx.omegaW[k * ctx.R + i], curr[i], l);
            logits[k] = l;
            maxL = max(maxL, l);
        }
        float Z = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) Z += exp(logits[k] - maxL);
        float invZ = 1.0f / max(Z, FLT_MIN);
        uint  bestK = 0u; float bestP = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float p = exp(logits[k] - maxL) * invZ;
            ctx.chord[k] = p;
            if (p > bestP) { bestP = p; bestK = k; }
        }
        ctx.optionAge[0] = 1u;
        ctx.option[1u - ctx.optionSlot] = bestK;
    } else {
        ctx.optionAge[0] += 1u;
        ctx.option[1u - ctx.optionSlot] = ctx.option[ctx.optionSlot];
    }
}

// =================================================================================================
// 4. oak_predict - Horde-style multi-GVF update. Computes:
//      Main TD:    delta = (r - rho_bar) + (V(s') - v_bar) - (V(s) - v_bar)  (varpi-RTD)
//      Subtask k:  delta_k = (r - rho_bar) + beta_k * g_k + (1 - beta_k) * (psi_k(s') - psi_bar_k)
//                          - (psi_k(s) - psi_bar_k)
//                  where g_k = (w_bar[f_k] - valueW[f_k]) * curr[f_k]   (RR-Subtasks eq. 4)
//    Single-threaded (one row, sequentially through K).
// =================================================================================================
kernel void oak_predict(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float vPrev = 0.0f, vNext = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) {
        vPrev = fma(ctx.valueW[i], prev[i], vPrev);
        vNext = fma(ctx.valueW[i], curr[i], vNext);
    }
    ctx.vCurr[0] = vNext;
    float r = ctx.reward[0];
    float reward_c = r - ctx.rho;
    float vbar = ctx.vMeanBaseline;
    float delta = reward_c + (vNext - vbar) - (vPrev - vbar);
    if (!isfinite(delta)) {
        ctx.counters[OAK_CNT_LEARN_NF] = ctx.counters[OAK_CNT_LEARN_NF] + 1u;
        delta = 0.0f;
    }
    ctx.td[0] = delta;

    uint nfPsi = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        uint f_k = ctx.subtaskFeatureOf[k];
        if (f_k >= ctx.R) f_k = 0u;
        // psi_k(s) = sum_i psi[k][i] * s[i] + sum_j psi[k][R+j] * chord_prev[j]
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            psiPrev = fma(ctx.psi[k * ctx.M + i], prev[i], psiPrev);
            psiNext = fma(ctx.psi[k * ctx.M + i], curr[i], psiNext);
        }
        for (uint j = 0u; j < ctx.K; ++j) {
            psiPrev = fma(ctx.psi[k * ctx.M + ctx.R + j], ctx.chordPrev[j], psiPrev);
            psiNext = fma(ctx.psi[k * ctx.M + ctx.R + j], ctx.chord[j],     psiNext);
        }
        float pbar = ctx.psiMeanBaseline[k];
        // Feature-attainment stopping bonus (RR-Subtasks eq. 4 with bonus weight w_bar).
        float bonus = (ctx.wBar[f_k] - ctx.valueW[f_k]) * curr[f_k];
        float beta_k = ctx.betaProb[0];     // shared termination prob (single beta-head, Bacon).
        float dk = reward_c + beta_k * bonus
                 + (1.0f - beta_k) * (psiNext - pbar) - (psiPrev - pbar);
        if (!isfinite(dk)) { dk = 0.0f; nfPsi++; }
        ctx.tdPsi[k] = dk;
    }
    if (nfPsi > 0u) ctx.counters[OAK_CNT_TDPSI_NF] = ctx.counters[OAK_CNT_TDPSI_NF] + nfPsi;
}

// =================================================================================================
// 5. oak_plan - Step 11 BLOCKER. Single-row asynchronous VI sweep over option models
//    (RR-Subtasks eq. 19): td += max_o [ r_hat^o(s') + V_hat(n_hat^o(s')) ] - V(s'). Search
//    control: option backups visited in utilOpt-ranked order (Step 9 prioritized sweeping with
//    linear FA, Sutton-Szepesvari-Geramifard-Bowling 2008). Writes planAdv (consumed by
//    learn_value as augmented TD) and planArgmax (consumed by bk_admit_commit as feature
//    attribution for utilPlan). Single-threaded.
// =================================================================================================
kernel void oak_plan(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float vCurr = ctx.vCurr[0];

    float bestQ = -FLT_MAX;
    uint  bestK = 0u;
    uint nfPlan = 0u;

    for (uint k = 0u; k < ctx.K; ++k) {
        // r_hat^k(s') = w_r^k . s'
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);

        // n_hat^k(s')[j] = w_n^{k,j} . s'  -> V_hat(n_hat) = sum_j valueW[j] * n_hat[j].
        float vHat = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            float nj = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i)
                nj = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + i], curr[i], nj);
            vHat = fma(ctx.valueW[j], nj, vHat);
        }
        float Q = rHat + vHat;
        if (!isfinite(Q)) { nfPlan++; continue; }
        if (Q > bestQ) { bestQ = Q; bestK = k; }
    }

    float adv = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
    if (!isfinite(adv)) { adv = 0.0f; nfPlan++; }
    // Step-size-clipped advantage so planning never dominates the on-policy gradient. The clip
    // bound is structural: ||advantage|| <= ||td|| * cap (one full per-feature cycle).
    ctx.planAdv[0] = bounded_soft(adv);
    ctx.planArgmax[0] = bestK;
    if (nfPlan > 0u) ctx.counters[OAK_CNT_PLAN_NF] = ctx.counters[OAK_CNT_PLAN_NF] + nfPlan;
}

// =================================================================================================
// 6. oak_reduce_alpha - implicit alpha (Kim 2510.06149) computed from the just-computed s'.
//    Single TG x 128 threads. Writes scalars->{alphaImp, alphaEnc, alphaPi}.
//      alphaImp = alpha_0 / (1 + alpha_0 * ||s'||^2)
//      alphaEnc = alpha_0 / (1 + alpha_0 * ||(obs - obsMean)||^2)
//      alphaPi  = alpha_0 / (1 + alpha_0 * ||s'||^2 * ||scoreGrad||^2)
// =================================================================================================
kernel void oak_reduce_alpha(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    threadgroup float buf[4];
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float s2 = 0.0f;
    for (uint i = tid; i < ctx.R; i += tgW) s2 = fma(curr[i], curr[i], s2);
    float ss = tg_sum_128(buf, tid, sgid, s2);

    float o2 = 0.0f;
    for (uint o = tid; o < ctx.O; o += tgW) {
        float d = ctx.observation[o] - ctx.observationMean[o];
        o2 = fma(d, d, o2);
    }
    float oo = tg_sum_128(buf, tid, sgid, o2);

    float g2 = 0.0f;
    for (uint p = tid; p < ctx.D; p += tgW) g2 = fma(ctx.scoreGrad[p], ctx.scoreGrad[p], g2);
    float gg = tg_sum_128(buf, tid, sgid, g2);

    if (tid == 0u) {
        float a0 = ctx.alpha0;
        ctx.scalars->alphaImp = a0 / (1.0f + a0 * ss);
        ctx.scalars->alphaEnc = a0 / (1.0f + a0 * oo);
        ctx.scalars->alphaPi  = a0 / (1.0f + a0 * ss * gg);
        ctx.stats->latentStateMeanSquare = ss / float(max(ctx.R, 1u));
        ctx.stats->scoreGradL2Mean       = gg / float(max(ctx.D, 1u));
    }
}

// =================================================================================================
// 7. oak_learn_heads - fused weight updates with per-feature TIDBD step sizes (SwiftSarsa
//    2507.19539 + MetaOptimize 2402.02342) and always-on TD(lambda) eligibility traces
//    (Sutton-Barto Ch. 12.7 backward view; structural lambda = 1 - 1/cap). One grid-stride pass
//    over (head_id, dim_id). Each thread handles one weight write site through safe_half.
//
//    Heads (subid):
//      0: encoder[R*O]                      -> uses delta_main, gradient = standardized_obs
//      1: valueW[R]                         -> uses delta_main + planAdv, gradient = prev[i]
//      2: psi[K*M]                          -> uses delta_k, gradient = concat(prev, chordPrev)
//      3: optPolicy[K*D*R]                  -> uses delta_main * scoreGrad[d], chord-attributed
//      4: betaW[R+K]                        -> uses option-critic gradient on beta termination
//      5: omegaW[K*R]                       -> uses option-critic gradient on chord softmax
//      6: optReward[K*R]                    -> reward-model TD with importance ratio
//      7: optTransition[K*R*R]              -> transition-model TD per output feature
//
//    All weight writes go through safe_half. NO zero clamps anywhere.
//    Grid = max(R*O, R, K*M, K*D*R, R+K, K*R, K*R, K*R*R) wide; each thread routes via gid.
//    For minimal launch overhead, we encode this as ONE compute dispatch: thread `gid` first
//    figures out which head it belongs to via cumulative offsets, then runs the right update.
// =================================================================================================
kernel void oak_learn_heads(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint nEnc = ctx.R * ctx.baseO;
    uint nVal = ctx.R;
    uint nPsi = ctx.K * ctx.M;
    uint nPi  = ctx.K * ctx.D * ctx.R;
    uint nBet = ctx.R + ctx.K;
    uint nOme = ctx.K * ctx.R;
    uint nOR  = ctx.K * ctx.R;
    uint nOT  = ctx.K * ctx.R * ctx.R;
    uint nTotal = nEnc + nVal + nPsi + nPi + nBet + nOme + nOR + nOT;
    if (gid >= nTotal) return;

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float delta = ctx.td[0] + ctx.planAdv[0];   // varpi-RTD residual + planning advantage.
    float lambda = ctx.lambdaTrace;
    uint  optK = ctx.option[1u - ctx.optionSlot];   // active option index (committed by meta_policy).

    // ---- Head 0: encoder[r * baseO + o] -- standardized obs feedback on delta_main. ----
    if (gid < nEnc) {
        uint id = gid;
        uint r = id / ctx.baseO, o = id % ctx.baseO;
        float xt = standardized_obs(ctx.observation[o], ctx.observationMean[o],
                                    ctx.obsSM[o], ctx.ridge);
        // Encoder trace not needed (no recurrence beyond per-feature retention); single-step grad.
        float grad = delta * xt * (curr[r] - prev[r]);   // chain through belief residual.
        float w = ctx.encoder[id];
        ctx.encoder[id] = safe_half(w + ctx.alphaEnc * grad);
        return;
    }
    // ---- Head 1: valueW[i] -- main TD with eligibility trace and per-feature implicit alpha. -
    if (gid < nEnc + nVal) {
        uint i = gid - nEnc;
        float a_i = ctx.alphaImpFeat[i];
        // Backward-view trace: e <- lambda * e + grad_v(s') ; grad = curr[i].
        ctx.valueE[i] = lambda * ctx.valueE[i] + curr[i];
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_half(w + a_i * delta * ctx.valueE[i]);

        // Feature-second-moment EMA + decayed high-water-mark for RRS bonus weight.
        float invT = track_rate(ctx.stepCount, ctx.ridge);
        float xi = curr[i];
        ctx.featSM[i] = bounded_soft(ctx.featSM[i] + invT * (xi * xi - ctx.featSM[i]));
        float decay = 1.0f - ctx.alpha0;
        ctx.wBar[i] = bounded_soft(max(decay * ctx.wBar[i], ctx.valueW[i]));

        // Per-feature TIDBD update: beta_i log-step with delta * grad * h history.
        float h = ctx.alphaImpH[i];
        float gradV = delta * ctx.valueE[i];
        ctx.alphaImpBeta[i] = bounded_soft(ctx.alphaImpBeta[i] + ctx.alpha0 * gradV * h);
        float a_new = exp(ctx.alphaImpBeta[i]);
        ctx.alphaImpFeat[i] = max(min(a_new, 1.0f), ctx.ridge);
        ctx.alphaImpH[i] = bounded_soft(h * max(0.0f, 1.0f - a_new * curr[i] * curr[i])
                                        + a_new * gradV);
        return;
    }
    // ---- Head 2: psi[k * M + m] -- subtask TD on cumulant k. ----
    if (gid < nEnc + nVal + nPsi) {
        uint id = gid - nEnc - nVal;
        uint k = id / ctx.M;
        uint m = id % ctx.M;
        float dk = ctx.tdPsi[k];
        float xm = (m < ctx.R) ? prev[m] : ctx.chordPrev[m - ctx.R];
        ctx.psiE[id] = lambda * ctx.psiE[id] + xm;
        float w = ctx.psi[id];
        ctx.psi[id] = safe_half(w + ctx.alphaImp * dk * ctx.psiE[id]);
        return;
    }
    // ---- Head 3: optPolicy[k][d][i] -- REINFORCE on Gaussian policy params, chord-attributed. -
    if (gid < nEnc + nVal + nPsi + nPi) {
        uint id = gid - nEnc - nVal - nPsi;
        uint k  = id / (ctx.D * ctx.R);
        uint rem = id - k * ctx.D * ctx.R;
        uint d  = rem / ctx.R;
        uint i  = rem % ctx.R;
        // Soft-chord attribution: weight per-option contribution by chordPrev[k] (the chord that
        // produced the action whose scoreGrad we're learning from).
        float lam_k = ctx.chordPrev[k];
        ctx.piE[id] = lambda * ctx.piE[id] + lam_k * ctx.scoreGrad[d] * prev[i];
        float w = ctx.optPolicy[id];
        ctx.optPolicy[id] = safe_half(w + ctx.alphaPi * delta * ctx.piE[id]);
        return;
    }
    // ---- Head 4: betaW -- option-critic beta gradient (Bacon-Harb-Precup 2017). ----
    if (gid < nEnc + nVal + nPsi + nPi + nBet) {
        uint id = gid - nEnc - nVal - nPsi - nPi;
        // beta_grad: -delta_termination * sigmoid'(logit) * input.
        // sigmoid' = beta_prob * (1 - beta_prob); termination advantage = vCurr - max(psi_k).
        float bp = ctx.betaProb[0];
        float dterm = ctx.vCurr[0];
        for (uint kk = 0u; kk < ctx.K; ++kk) dterm = max(dterm, ctx.tdPsi[kk] + ctx.psiMeanBaseline[kk]);
        dterm = ctx.vCurr[0] - dterm;
        float dsig = bp * (1.0f - bp);
        float input = (id < ctx.R) ? curr[id] : ctx.chordPrev[id - ctx.R];
        float grad = -dterm * dsig * input;
        float w = ctx.betaW[id];
        ctx.betaW[id] = safe_half(w + ctx.alphaImp * grad);
        return;
    }
    // ---- Head 5: omegaW[k][i] -- chord softmax gradient on option value. ----
    if (gid < nEnc + nVal + nPsi + nPi + nBet + nOme) {
        uint id = gid - nEnc - nVal - nPsi - nPi - nBet;
        uint k = id / ctx.R;
        uint i = id % ctx.R;
        // Chord softmax gradient: (1{k=optK} - chord[k]) * delta_chord * curr[i].
        // delta_chord = max-Q advantage learned through ψ; here we use tdPsi[k] as a clean
        // off-policy signal for cumulant k (Wan-Naik-Sutton 2110.13855 avg-reward options).
        float ind = (k == optK) ? 1.0f : 0.0f;
        float grad = (ind - ctx.chord[k]) * ctx.tdPsi[k] * curr[i];
        float w = ctx.omegaW[id];
        ctx.omegaW[id] = safe_half(w + ctx.alphaImp * grad);
        return;
    }
    // ---- Head 6: optReward[k][i] -- option-model reward head, RR-Subtasks eq. 17 with
    //      importance ratio rho_t^o = chord_prev[k] / behavior_density (single-stream =
    //      structural-1 in the wrapper, so ratio = chordPrev[k] / chord_prior; we use
    //      chordPrev directly as the on-chord weighting). ----
    if (gid < nEnc + nVal + nPsi + nPi + nBet + nOme + nOR) {
        uint id = gid - nEnc - nVal - nPsi - nPi - nBet - nOme;
        uint k = id / ctx.R;
        uint i = id % ctx.R;
        float rho_o = ctx.chordPrev[k];
        float predPrev = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predPrev = fma(ctx.optReward[k * ctx.R + j], prev[j], predPrev);
        float predNext = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predNext = fma(ctx.optReward[k * ctx.R + j], curr[j], predNext);
        // Reward-model TD with beta termination: target = r + (1 - beta) * predNext.
        float bp = ctx.betaProb[0];
        float r_td = ctx.reward[0] + (1.0f - bp) * predNext - predPrev;
        ctx.optRewardE[id] = lambda * ctx.optRewardE[id] + rho_o * prev[i];
        float w = ctx.optReward[id];
        ctx.optReward[id] = safe_half(w + ctx.alphaImp * r_td * ctx.optRewardE[id]);
        return;
    }
    // ---- Head 7: optTransition[k][j][i] -- per-feature transition model n_hat_j^k. ----
    {
        uint id = gid - nEnc - nVal - nPsi - nPi - nBet - nOme - nOR;
        uint k  = id / (ctx.R * ctx.R);
        uint rem = id - k * ctx.R * ctx.R;
        uint j  = rem / ctx.R;
        uint i  = rem % ctx.R;
        float rho_o = ctx.chordPrev[k];
        // n_hat_j^k(s) = w_{k,j} . s ; target = curr[j] (next-state feature j).
        float predPrev = 0.0f;
        for (uint ii = 0u; ii < ctx.R; ++ii)
            predPrev = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + ii], prev[ii], predPrev);
        float predNext = 0.0f;
        for (uint ii = 0u; ii < ctx.R; ++ii)
            predNext = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + ii], curr[ii], predNext);
        float bp = ctx.betaProb[0];
        float n_td = curr[j] + (1.0f - bp) * predNext - predPrev;
        ctx.optTransitionE[id] = lambda * ctx.optTransitionE[id] + rho_o * prev[i];
        float w = ctx.optTransition[id];
        ctx.optTransition[id] = safe_half(w + ctx.alphaImp * n_td * ctx.optTransitionE[id]);
    }
}

// =================================================================================================
// 8. oak_meta_action - soft-chord blended linear policy. mu_p = sum_k chord[k] * (W_pi[k][p] . s).
//    Per-row; one thread per dim p. Wrapper does the structural-sigma=1 Gaussian draw with
//    pre-clamp z as scoreGrad (Williams 1992 REINFORCE).
// =================================================================================================
kernel void oak_meta_action(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= cfg.actionParamCount) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float mu = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float lam = ctx.chord[k];
        if (lam == 0.0f) continue;
        float dot = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            dot = fma(ctx.optPolicy[(k * ctx.D + p) * ctx.R + i], curr[i], dot);
        mu = fma(lam, dot, mu);
    }
    ctx.policyMean[p] = bounded_soft(mu);
}

// =================================================================================================
// 9. oak_bk_reduce - per-feature Welford EMA for featMean. Single TG x 128 threads.
// =================================================================================================
kernel void oak_bk_reduce(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgW [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    for (uint i = tid; i < ctx.R; i += tgW) {
        ctx.featMean[i] = bounded_soft(ctx.featMean[i]
                                       + invT * (curr[i] - ctx.featMean[i]));
    }
    if (tid == 0u) {
        // rho_bar tracking: EMA of reward (Reward Centering 2405.09999).
        float r = ctx.reward[0];
        ctx.scalars->rho = bounded_soft(ctx.rho + invT * (r - ctx.rho));
        ctx.stats->rewardRate = ctx.scalars->rho;

        // Option-model prediction MSE for LPM (consumed by meta_policy next step).
        // Use the just-computed planAdv-to-realized error: |planAdv|^2 is a clean per-step proxy.
        float pa = ctx.planAdv[0];
        float mse = pa * pa;
        ctx.scalars->modelErrEMAPrev = ctx.scalars->modelErrEMA;
        ctx.scalars->modelErrEMA = bounded_soft(ctx.scalars->modelErrEMA
                                                + invT * (mse - ctx.scalars->modelErrEMA));
        ctx.stats->optionModelPredictionMSE = mse;
        ctx.stats->modelErrEMA       = ctx.scalars->modelErrEMA;
        ctx.stats->modelErrEMAPrev   = ctx.scalars->modelErrEMAPrev;
        ctx.stats->tdBaselineStat    = ctx.scalars->vMeanBaseline;
        ctx.stats->alphaImpStat      = ctx.scalars->alphaImp;
        ctx.stats->alphaEncStat      = ctx.scalars->alphaEnc;
        ctx.stats->alphaPiStat       = ctx.scalars->alphaPi;
    }
}

// =================================================================================================
// 10. oak_bk_utility - GraMa per-feature gradient utility (2505.24061), per-option |delta_k|^2
//     utility, planning utility (1{f_k* used i}), and per-buffer max(|w|) for the bounded-magnitude
//     gate. Grid = R + K threads laid out so that the first R handle features and the next K
//     handle options + weight-magnitudes (one per buffer for the structural max).
// =================================================================================================
kernel void oak_bk_utility(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    device const float *prev = state_at(ctx, ctx.stateSlot);

    if (gid < ctx.R) {
        uint i = gid;
        // GraMa: utility = EMA(|delta * x_i| + sum_k |delta_k * x_i|), normalized by 1+K.
        float xi = prev[i];
        float gMain = fabs(ctx.td[0] * xi);
        float gSum = gMain;
        for (uint k = 0u; k < ctx.K; ++k) gSum += fabs(ctx.tdPsi[k] * xi);
        float perI = gSum / float(1u + ctx.K);
        ctx.utilFeat[i] = bounded_soft(ctx.utilFeat[i] + invT * (perI - ctx.utilFeat[i]));

        // Planning utility: 1 iff feature i is the subtask anchor of the bestK option.
        uint bestK = ctx.planArgmax[0];
        uint anchor = ctx.subtaskFeatureOf[bestK];
        float planContrib = (anchor == i) ? 1.0f : 0.0f;
        ctx.utilPlan[i] = bounded_soft(ctx.utilPlan[i] + invT * (planContrib - ctx.utilPlan[i]));
        return;
    }
    uint k = gid - ctx.R;
    if (k < ctx.K) {
        float dk = ctx.tdPsi[k];
        float perK = dk * dk;
        ctx.utilOpt[k] = bounded_soft(ctx.utilOpt[k] + invT * (perK - ctx.utilOpt[k]));
        // Soft-chord per-option mass EMA (dormancy floor).
        ctx.optFracEMA[k] = bounded_soft(ctx.optFracEMA[k]
                                         + invT * (ctx.chord[k] - ctx.optFracEMA[k]));
        return;
    }
    // Last 8 threads: per-buffer max(|w|) (structural; one thread strides over each).
    uint b = gid - ctx.R - ctx.K;
    if (b >= 8u) return;
    device const float *base; uint count;
    switch (b) {
        case 0u: base = ctx.encoder;       count = ctx.R * ctx.baseO; break;
        case 1u: base = ctx.valueW;        count = ctx.R; break;
        case 2u: base = ctx.psi;           count = ctx.K * ctx.M; break;
        case 3u: base = ctx.optPolicy;     count = ctx.K * ctx.D * ctx.R; break;
        case 4u: base = ctx.betaW;         count = ctx.R + ctx.K; break;
        case 5u: base = ctx.omegaW;        count = ctx.K * ctx.R; break;
        case 6u: base = ctx.optReward;     count = ctx.K * ctx.R; break;
        default: base = ctx.optTransition; count = ctx.K * ctx.R * ctx.R; break;
    }
    float m = 0.0f;
    for (uint i = 0u; i < count; ++i) m = max(m, fabs(base[i]));
    switch (b) {
        case 0u: ctx.stats->encoderMax       = m; break;
        case 1u: ctx.stats->valueWMax        = m; break;
        case 2u: ctx.stats->psiMax           = m; break;
        case 3u: ctx.stats->optPolicyMax     = m; break;
        case 4u: ctx.stats->betaWMax         = m; break;
        case 5u: ctx.stats->omegaWMax        = m; break;
        case 6u: ctx.stats->optRewardMax     = m; break;
        default: ctx.stats->optTransitionMax = m; break;
    }
}

// =================================================================================================
// 11. oak_bk_admit_commit - the Oak feedback engine. Reranks sf[K] by combined utilFeat +
//     utilPlan (closes the FCSTOMP loop per RR-Subtasks Section 8); FIRE+SWR option reinit
//     conjunctive on dormant_k AND magnitudeOutlier_k (2602.08040); generate-and-test feature
//     creation on reinited feature slot (Mahmood-Sutton-Pilarski 2012); rho/v_bar/psi_bar
//     baseline updates consumed by the next step's predict. Single-threaded.
// =================================================================================================
kernel void oak_bk_admit_commit(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);

    // ---- Active-feature admission. Floor at ceil(sqrt(R)), structural (Inv-9). ----
    float meanU = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) meanU += ctx.utilFeat[i] + ctx.utilPlan[i];
    meanU /= float(max(ctx.R, 1u));
    float invR = 1.0f / float(max(ctx.R, 1u));
    uint floor = (uint) ceil(sqrt(float(ctx.R)));
    uint act = 0u;
    for (uint i = 0u; i < ctx.R; ++i) {
        bool admit = (ctx.utilFeat[i] + ctx.utilPlan[i]) > meanU;
        ctx.activeMask[i] = admit ? 1u : 0u;
        if (admit) { ctx.activeList[act++] = i; }
    }
    while (act < floor) {                      // pad up to floor.
        uint slot = act % ctx.R;
        if (ctx.activeMask[slot] == 0u) {
            ctx.activeMask[slot] = 1u;
            ctx.activeList[act++] = slot;
        } else {
            slot = (slot + 1u) % ctx.R;
        }
    }
    ctx.scalars->activeCount = act;
    ctx.stats->activeCount = float(act);

    // Identity fast-path flag: 1 iff every feature is admitted in order.
    uint identity = (act == ctx.R) ? 1u : 0u;
    if (identity == 1u) {
        for (uint i = 0u; i < ctx.R; ++i)
            if (ctx.activeList[i] != i) { identity = 0u; break; }
    }
    ctx.scalars->activeListIsIdentity = identity;

    // ---- Rerank sf[K] by combined utility (FCSTOMP feedback). Selection sort over R; R is small. ----
    for (uint k = 0u; k < ctx.K; ++k) {
        uint  bestI = k % ctx.R;
        float bestU = -FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i) {
            // Skip already-claimed features in earlier sf positions.
            bool claimed = false;
            for (uint j = 0u; j < k; ++j) if (ctx.subtaskFeatureOf[j] == i) { claimed = true; break; }
            if (claimed) continue;
            float u = ctx.utilFeat[i] + ctx.utilPlan[i];
            if (u > bestU) { bestU = u; bestI = i; }
        }
        ctx.subtaskFeatureOf[k] = bestI;
    }

    // ---- FIRE+SWR option reinit: dormant_k (chord mass below 1/(2K)) AND magnitude outlier
    //      (utilOpt[k] < 0.1 * utilOpt_max). Conjunctive AND (2602.08040). ----
    float utilOptMax = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) utilOptMax = max(utilOptMax, ctx.utilOpt[k]);
    float dormThresh = 0.5f / float(max(ctx.K, 1u));      // half of uniform mass.
    uint reinitedOpt = 0u, reinitedFeat = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        bool dormant = ctx.optFracEMA[k] < dormThresh;
        bool weak = ctx.utilOpt[k] < 0.1f * utilOptMax;
        if (dormant && weak) {
            // Re-seed psi[k][:M] and optPolicy[k][:D*R] to small hash-based projections (no zero-init).
            for (uint m = 0u; m < ctx.M; ++m) {
                float u = hash_uniform(ctx.stepCount, k, m) * 2.0f - 1.0f;
                ctx.psi[k * ctx.M + m] = u * ctx.alpha0;
            }
            for (uint pi = 0u; pi < ctx.D * ctx.R; ++pi) {
                float u = hash_uniform(ctx.stepCount, k, ctx.M + pi) * 2.0f - 1.0f;
                ctx.optPolicy[k * ctx.D * ctx.R + pi] = u * ctx.alpha0;
            }
            for (uint i = 0u; i < ctx.R; ++i) {
                float ur = hash_uniform(ctx.stepCount, k, ctx.M + ctx.D * ctx.R + i) * 2.0f - 1.0f;
                ctx.optReward[k * ctx.R + i] = ur * ctx.alpha0;
                for (uint j = 0u; j < ctx.R; ++j) {
                    float un = hash_uniform(ctx.stepCount, (k * ctx.R + j) ^ 0xa3c59ac3u, i) * 2.0f - 1.0f;
                    ctx.optTransition[(k * ctx.R + j) * ctx.R + i] = un * ctx.alpha0;
                }
            }
            ctx.utilOpt[k] = 0.0f;
            ctx.optFracEMA[k] = 1.0f / float(max(ctx.K, 1u));
            reinitedOpt++;
        }
    }

    // ---- Generate-and-test feature reinit: when an option is reinited, reset its anchor
    //      feature to a fresh hash-based random projection (Mahmood-Sutton-Pilarski 2012).
    //      Worst feature = lowest utilFeat + utilPlan. ----
    if (reinitedOpt > 0u) {
        uint worstI = 0u; float worstU = FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i) {
            float u = ctx.utilFeat[i] + ctx.utilPlan[i];
            if (u < worstU) { worstU = u; worstI = i; }
        }
        for (uint o = 0u; o < ctx.baseO; ++o) {
            float u = hash_uniform(ctx.stepCount, worstI, o) * 2.0f - 1.0f;
            ctx.encoder[worstI * ctx.baseO + o] = u * rsqrt(float(max(ctx.baseO, 1u)));
        }
        ctx.utilFeat[worstI] = 0.0f;
        ctx.utilPlan[worstI] = 0.0f;
        ctx.featSM[worstI] = 0.0f;
        ctx.wBar[worstI] = 0.0f;
        ctx.featMean[worstI] = 0.0f;
        ctx.alphaImpFeat[worstI] = ctx.alpha0;
        ctx.alphaImpBeta[worstI] = log(ctx.alpha0);
        ctx.alphaImpH[worstI] = 0.0f;
        ctx.valueE[worstI] = 0.0f;
        reinitedFeat = 1u;
        ctx.counters[OAK_CNT_REINIT_OPT] = ctx.counters[OAK_CNT_REINIT_OPT] + reinitedOpt;
        ctx.counters[OAK_CNT_REINIT_FEAT] = ctx.counters[OAK_CNT_REINIT_FEAT] + reinitedFeat;
    }
    ctx.stats->oakOptionReinitCount  = float(reinitedOpt);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

    // ---- varpi-RTD baselines: vMeanBaseline = <valueW, featMean>, psiMeanBaseline[k] = <psi_k, featMean>. ----
    float vMean = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) vMean = fma(ctx.valueW[i], ctx.featMean[i], vMean);
    ctx.scalars->vMeanBaseline = bounded_soft(vMean);
    for (uint k = 0u; k < ctx.K; ++k) {
        float pMean = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            pMean = fma(ctx.psi[k * ctx.M + i], ctx.featMean[i], pMean);
        ctx.psiMeanBaseline[k] = bounded_soft(pMean);
    }

    // ---- pEff stable-rank proxy: tr(featSM) / max(featSM). ----
    float trSM = 0.0f, maxSM = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) { trSM += ctx.featSM[i]; maxSM = max(maxSM, ctx.featSM[i]); }
    ctx.scalars->pEff = trSM / max(maxSM, ctx.ridge);
    ctx.stats->pEffectiveRank = ctx.scalars->pEff;

    // ---- Telemetry roll-ups. ----
    float utilFeatMean = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) utilFeatMean += ctx.utilFeat[i];
    ctx.stats->utilityFeatureMean = utilFeatMean * invR;
    float utilOptMean = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) utilOptMean += ctx.utilOpt[k];
    ctx.stats->utilityOptionMean = utilOptMean / float(max(ctx.K, 1u));
    ctx.stats->rawNonFiniteCount       = float(ctx.counters[OAK_CNT_OBS_NF]
                                             + ctx.counters[OAK_CNT_LEARN_NF]
                                             + ctx.counters[OAK_CNT_PLAN_NF]
                                             + ctx.counters[OAK_CNT_TDPSI_NF]);
    ctx.stats->sanitizedNonFiniteCount = ctx.stats->rawNonFiniteCount;
    ctx.stats->encodeNonFiniteCount    = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount     = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount      = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount     = float(ctx.counters[OAK_CNT_TDPSI_NF]);
    // maxAbs* telemetry comes from per-step bounds of policyMean / state.
    float mLat = 0.0f, mPol = 0.0f;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = 0u; i < ctx.R; ++i) mLat = max(mLat, fabs(curr[i]));
    for (uint p = 0u; p < ctx.D; ++p) mPol = max(mPol, fabs(ctx.policyMean[p]));
    ctx.stats->maxAbsLatentPrediction = mLat;
    ctx.stats->maxAbsPolicyOutput     = mPol;
    // Policy-action delta vs prev step (telemetry only).
    float dl = 0.0f;
    for (uint p = 0u; p < ctx.D; ++p) {
        float d = ctx.policyMean[p] - ctx.prevPolicyMean[p];
        dl = fma(d, d, dl);
    }
    ctx.stats->actionDeltaL2Mean = dl / float(max(ctx.D, 1u));

    // ---- Toggle slots, structural lambda, bump stepCount. ----
    ctx.scalars->stateSlot  = 1u - ctx.stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.optionSlot;
    ctx.scalars->lambdaTrace = max(0.0f, 1.0f - 1.0f / ctx.cap);
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
